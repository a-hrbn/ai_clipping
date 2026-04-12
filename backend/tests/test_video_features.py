import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import video_features
from services.timeline_types import Candidate
from services.transcriber import RichSegment


class _FakeCap:
    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == video_features.cv2.CAP_PROP_FPS:
            return 1.0
        if prop == video_features.cv2.CAP_PROP_FRAME_COUNT:
            return 5.0
        return 0.0

    def release(self) -> None:
        return None


class _FakeCV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    COLOR_BGR2GRAY = 3
    COLOR_BGR2RGB = 5
    HISTCMP_BHATTACHARYYA = 4

    @staticmethod
    def VideoCapture(_path: str) -> _FakeCap:
        return _FakeCap()

    @staticmethod
    def compareHist(_left, _right, _mode) -> float:
        return 0.0

    @staticmethod
    def resize(frame, _size, fx=1.0, fy=1.0):
        return frame

    @staticmethod
    def cvtColor(frame, _code):
        return frame[:, :, 0]

    @staticmethod
    def calcOpticalFlowFarneback(prev, curr, *_args, **_kwargs):
        return np.zeros((prev.shape[0], prev.shape[1], 2), dtype=np.float32)


class _FakeDeepFace:
    calls = 0

    @classmethod
    def analyze(cls, *_args, **_kwargs):
        cls.calls += 1
        return {"emotion": {"surprise": 80, "fear": 0, "disgust": 0}}


class _FakeMTCNN:
    def detect_faces(self, _frame):
        return [{"box": [1, 1, 4, 4]}]


class VideoFeaturesTests(unittest.TestCase):
    def test_extract_samples_deepface_every_n_seconds(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        _FakeDeepFace.calls = 0

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", True), \
             patch.object(video_features, "DeepFace", _FakeDeepFace, create=True), \
             patch.object(video_features, "_MP_FACE_DETECTION", None), \
             patch.object(video_features, "_FACE_CASCADE", None), \
             patch.object(video_features, "_seek_frame", return_value=frame), \
             patch.object(video_features, "_hsv_hist", return_value=np.array([0.0], dtype=np.float32)), \
             patch.object(video_features.settings, "video_expression_sample_sec", 3):
            frames = video_features.extract(Path("fake.mp4"))

        self.assertEqual(len(frames), 5)
        self.assertEqual(_FakeDeepFace.calls, 2)
        self.assertEqual(frames[0].face_reaction_prob, 0.8)
        self.assertEqual(frames[1].face_reaction_prob, 0.8)
        self.assertEqual(frames[3].face_reaction_prob, 0.8)

    def test_refine_face_reactions_only_updates_shortlist_windows(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        _FakeDeepFace.calls = 0
        segments = [
            RichSegment(start=0.0, end=2.0, text="outside"),
            RichSegment(start=10.0, end=12.0, text="inside"),
        ]
        candidates = [Candidate(id="c1", start=10.0, end=12.0, source="reaction", anchor_time=11.0)]

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", True), \
             patch.object(video_features, "DeepFace", _FakeDeepFace, create=True), \
             patch.object(video_features, "_seek_frame", return_value=frame):
            updated = video_features.refine_face_reactions(
                segments,
                Path("fake.mp4"),
                candidates,
                sample_every_sec=1,
            )

        self.assertEqual(_FakeDeepFace.calls, 3)
        self.assertEqual(updated[0].max_face_reaction, 0.0)
        self.assertEqual(updated[1].max_face_reaction, 0.8)
        self.assertIn("face_reaction", updated[1].emotions)

    def test_extract_uses_mtcnn_when_mediapipe_unavailable(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", False), \
             patch.object(video_features, "_MP_FACE_DETECTION", None), \
             patch.object(video_features, "_MP_TASKS_FACE_DETECTOR", None), \
             patch.object(video_features, "_MTCNN_DETECTOR", _FakeMTCNN()), \
             patch.object(video_features, "_FACE_CASCADE", None), \
             patch.object(video_features, "_seek_frame", return_value=frame), \
             patch.object(video_features, "_hsv_hist", return_value=np.array([0.0], dtype=np.float32)):
            frames = video_features.extract(Path("fake.mp4"))

        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].face_count, 1)
        self.assertGreater(frames[0].face_size_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
