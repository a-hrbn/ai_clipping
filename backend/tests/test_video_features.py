import sys
import time
import types
import unittest
import os
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


class _NoCaptureCV2(_FakeCV2):
    @staticmethod
    def VideoCapture(_path: str) -> _FakeCap:
        raise AssertionError("VideoCapture should not be used for AV1 ffmpeg decode path")


class _FakeDeepFace:
    calls = 0

    @classmethod
    def analyze(cls, *_args, **_kwargs):
        cls.calls += 1
        return {"emotion": {"surprise": 80, "fear": 0, "disgust": 0}}


class _FakeDeepFaceWithEmptyResults:
    calls = 0

    @classmethod
    def analyze(cls, *_args, **_kwargs):
        cls.calls += 1
        if cls.calls == 1:
            return None
        if cls.calls == 2:
            return []
        return {"emotion": {"surprise": 60, "fear": 0, "disgust": 0}}


class _FakeDeepFaceWithBuildModel:
    build_calls = 0
    analyze_calls = 0

    @classmethod
    def build_model(cls, name: str):
        cls.build_calls += 1
        return {"name": name}

    @classmethod
    def analyze(cls, *_args, **_kwargs):
        cls.analyze_calls += 1
        return {"emotion": {"surprise": 80, "fear": 0, "disgust": 0}}


class _FakeDeepFaceWithTaskAwareBuildModel:
    build_calls = 0
    analyze_calls = 0
    build_args = None

    @classmethod
    def build_model(cls, model_name: str, task: str = "facial_recognition"):
        cls.build_calls += 1
        cls.build_args = (model_name, task)
        return {"task": task, "model_name": model_name}

    @classmethod
    def analyze(cls, *_args, **_kwargs):
        cls.analyze_calls += 1
        return {"emotion": {"surprise": 80, "fear": 0, "disgust": 0}}


class _FakeDeepFaceWithFailingBuildModel:
    build_calls = 0
    analyze_calls = 0

    @classmethod
    def build_model(cls, name: str):
        cls.build_calls += 1
        raise RuntimeError(f"boom:{name}")

    @classmethod
    def analyze(cls, *_args, **_kwargs):
        cls.analyze_calls += 1
        return {"emotion": {"surprise": 80, "fear": 0, "disgust": 0}}


class _FakeMTCNN:
    def detect_faces(self, _frame):
        return [{"box": [1, 1, 4, 4]}]


class _FakeBatchFFmpegProcess:
    def __init__(self, payload: bytes) -> None:
        self.stdout = payload
        self.stderr = b""
        self.returncode = 0

    class _Reader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self._offset = 0

        def read(self, size: int = -1) -> bytes:
            if size < 0:
                chunk = self._payload[self._offset:]
                self._offset = len(self._payload)
                return chunk
            chunk = self._payload[self._offset:self._offset + size]
            self._offset += len(chunk)
            return chunk

        def close(self) -> None:
            return None

    def wait(self) -> int:
        return self.returncode


class VideoFeaturesTests(unittest.TestCase):
    def test_decode_second_frames_ffmpeg_reads_multiple_frames_from_single_process(self) -> None:
        frame_a = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))
        frame_b = np.full((8, 8, 3), 7, dtype=np.uint8)
        payload = frame_a.tobytes() + frame_b.tobytes()

        class _Process:
            def __init__(self, payload: bytes) -> None:
                self.stdout = _FakeBatchFFmpegProcess._Reader(payload)
                self.stderr = _FakeBatchFFmpegProcess._Reader(b"")
                self.returncode = 0

            def wait(self) -> int:
                return self.returncode

        with patch.object(video_features.subprocess, "Popen", return_value=_Process(payload)):
            frames = video_features._decode_second_frames_ffmpeg(
                Path("fake-av1.mp4"),
                start_sec=0.0,
                sample_count=2,
                stream_info={"width": 8, "height": 8},
            )

        self.assertEqual(sorted(frames), [0, 1])
        self.assertTrue(np.array_equal(frames[0], frame_a))
        self.assertTrue(np.array_equal(frames[1], frame_b))

    def test_configure_opencv_capture_env_sets_software_decode_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            video_features._configure_opencv_capture_env()

            self.assertEqual(os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"), "hwaccel;none")

    def test_configure_opencv_capture_env_preserves_explicit_override(self) -> None:
        with patch.dict(os.environ, {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "hwaccel;cuda"}, clear=True):
            video_features._configure_opencv_capture_env()

            self.assertEqual(os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"), "hwaccel;cuda")

    def test_should_use_ffmpeg_frame_decode_for_noisy_codecs(self) -> None:
        self.assertTrue(video_features._should_use_ffmpeg_frame_decode({"codec_name": "av1"}))
        self.assertTrue(video_features._should_use_ffmpeg_frame_decode({"codec_name": "h264"}))
        self.assertTrue(video_features._should_use_ffmpeg_frame_decode({"codec_name": "hevc"}))
        self.assertFalse(video_features._should_use_ffmpeg_frame_decode({"codec_name": ""}))

    def test_extract_samples_deepface_every_n_seconds(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        _FakeDeepFace.calls = 0

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", True), \
             patch.object(video_features, "_DEEPFACE_MODEL_READY", False), \
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

    def test_extract_ignores_none_and_empty_deepface_payloads(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        _FakeDeepFaceWithEmptyResults.calls = 0

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", True), \
             patch.object(video_features, "_DEEPFACE_MODEL_READY", False), \
             patch.object(video_features, "DeepFace", _FakeDeepFaceWithEmptyResults, create=True), \
             patch.object(video_features, "_MP_FACE_DETECTION", None), \
             patch.object(video_features, "_FACE_CASCADE", None), \
             patch.object(video_features, "_seek_frame", return_value=frame), \
             patch.object(video_features, "_hsv_hist", return_value=np.array([0.0], dtype=np.float32)), \
             patch.object(video_features.settings, "video_expression_sample_sec", 2):
            frames = video_features.extract(Path("fake.mp4"))

        self.assertEqual(len(frames), 5)
        self.assertEqual(_FakeDeepFaceWithEmptyResults.calls, 3)
        self.assertEqual(frames[0].face_reaction_prob, 0.0)
        self.assertEqual(frames[1].face_reaction_prob, 0.0)
        self.assertEqual(frames[2].face_reaction_prob, 0.0)
        self.assertEqual(frames[4].face_reaction_prob, 0.6)

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
             patch.object(video_features, "_DEEPFACE_MODEL_READY", False), \
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

    def test_refine_face_reactions_ignores_none_and_empty_deepface_payloads(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        _FakeDeepFaceWithEmptyResults.calls = 0
        segments = [RichSegment(start=10.0, end=14.0, text="inside")]
        candidates = [Candidate(id="c1", start=10.0, end=14.0, source="reaction", anchor_time=11.0)]

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", True), \
             patch.object(video_features, "_DEEPFACE_MODEL_READY", False), \
             patch.object(video_features, "DeepFace", _FakeDeepFaceWithEmptyResults, create=True), \
             patch.object(video_features, "_seek_frame", return_value=frame):
            updated = video_features.refine_face_reactions(
                segments,
                Path("fake.mp4"),
                candidates,
                sample_every_sec=2,
            )

        self.assertEqual(_FakeDeepFaceWithEmptyResults.calls, 3)
        self.assertEqual(updated[0].max_face_reaction, 0.6)
        self.assertIn("face_reaction", updated[0].emotions)

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

    def test_extract_uses_ffmpeg_frame_decode_for_av1_streams(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _NoCaptureCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", False), \
             patch.object(video_features, "_MP_FACE_DETECTION", None), \
             patch.object(video_features, "_MP_TASKS_FACE_DETECTOR", None), \
             patch.object(video_features, "_MTCNN_DETECTOR", None), \
             patch.object(video_features, "_FACE_CASCADE", None), \
             patch.object(video_features, "_video_stream_info", return_value={"codec_name": "av1", "duration_sec": 5.0, "width": 8, "height": 8}), \
             patch.object(video_features, "_decode_second_frames_ffmpeg", return_value={index: frame for index in range(5)}) as decode_mock, \
             patch.object(video_features, "_hsv_hist", return_value=np.array([0.0], dtype=np.float32)):
            frames = video_features.extract(Path("fake-av1.mp4"))

        self.assertEqual(len(frames), 5)
        decode_mock.assert_called_once()

    def test_extract_uses_ffmpeg_frame_decode_for_h264_streams(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _NoCaptureCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", False), \
             patch.object(video_features, "_MP_FACE_DETECTION", None), \
             patch.object(video_features, "_MP_TASKS_FACE_DETECTOR", None), \
             patch.object(video_features, "_MTCNN_DETECTOR", None), \
             patch.object(video_features, "_FACE_CASCADE", None), \
             patch.object(video_features, "_video_stream_info", return_value={"codec_name": "h264", "duration_sec": 5.0, "width": 8, "height": 8}), \
             patch.object(video_features, "_decode_second_frames_ffmpeg", return_value={index: frame for index in range(5)}) as decode_mock, \
             patch.object(video_features, "_hsv_hist", return_value=np.array([0.0], dtype=np.float32)):
            frames = video_features.extract(Path("fake-h264.mp4"))

        self.assertEqual(len(frames), 5)
        decode_mock.assert_called_once()

    def test_decode_second_frames_ffmpeg_uses_fatal_loglevel(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        commands: list[list[str]] = []

        class _Process:
            def __init__(self, command, **_kwargs) -> None:
                commands.append(command)
                self.stdout = _FakeBatchFFmpegProcess._Reader(frame.tobytes())
                self.stderr = _FakeBatchFFmpegProcess._Reader(b"")
                self.returncode = 0

            def wait(self) -> int:
                return self.returncode

        with patch.object(video_features.subprocess, "Popen", side_effect=_Process):
            frames = video_features._decode_second_frames_ffmpeg(
                Path("fake-h264.mp4"),
                start_sec=0.0,
                sample_count=1,
                stream_info={"width": 8, "height": 8},
            )

        self.assertEqual(sorted(frames), [0])
        loglevel_index = commands[0].index("-loglevel")
        self.assertEqual(commands[0][loglevel_index + 1], "fatal")

    def test_decode_frame_ffmpeg_uses_configured_timeout(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        calls: list[dict] = []

        def _run(_command, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(stdout=frame.tobytes())

        with patch.object(video_features.subprocess, "run", side_effect=_run), \
             patch.object(video_features.settings, "video_feature_ffmpeg_timeout_sec", 12.5):
            decoded = video_features._decode_frame_ffmpeg(
                Path("fake-h264.mp4"),
                3.0,
                stream_info={"width": 8, "height": 8},
            )

        self.assertIsNotNone(decoded)
        self.assertEqual(calls[0]["timeout"], 12.5)
        self.assertIs(calls[0]["stderr"], video_features.subprocess.DEVNULL)

    def test_decode_second_frames_ffmpeg_times_out_stalled_process(self) -> None:
        real_popen = video_features.subprocess.Popen

        def _stalling_popen(_command, **kwargs):
            return real_popen(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                stdout=kwargs.get("stdout"),
                stderr=kwargs.get("stderr"),
            )

        started_at = time.monotonic()
        with patch.object(video_features.subprocess, "Popen", side_effect=_stalling_popen), \
             patch.object(video_features.settings, "video_feature_ffmpeg_timeout_sec", 0.05):
            frames = video_features._decode_second_frames_ffmpeg(
                Path("fake-h264.mp4"),
                start_sec=0.0,
                sample_count=1,
                stream_info={"width": 8, "height": 8},
            )

        self.assertEqual(frames, {})
        self.assertLess(time.monotonic() - started_at, 2.0)

    def test_extract_builds_deepface_model_once_per_process(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        _FakeDeepFaceWithBuildModel.build_calls = 0
        _FakeDeepFaceWithBuildModel.analyze_calls = 0

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", True), \
             patch.object(video_features, "_DEEPFACE_MODEL_READY", False), \
             patch.object(video_features, "DeepFace", _FakeDeepFaceWithBuildModel, create=True), \
             patch.object(video_features, "_MP_FACE_DETECTION", None), \
             patch.object(video_features, "_FACE_CASCADE", None), \
             patch.object(video_features, "_seek_frame", return_value=frame), \
             patch.object(video_features, "_hsv_hist", return_value=np.array([0.0], dtype=np.float32)), \
             patch.object(video_features.settings, "video_expression_sample_sec", 3):
            video_features.extract(Path("fake.mp4"))
            video_features.extract(Path("fake.mp4"))

        self.assertEqual(_FakeDeepFaceWithBuildModel.build_calls, 1)
        self.assertEqual(_FakeDeepFaceWithBuildModel.analyze_calls, 4)

    def test_extract_warms_task_aware_deepface_with_facial_attribute_emotion(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        _FakeDeepFaceWithTaskAwareBuildModel.build_calls = 0
        _FakeDeepFaceWithTaskAwareBuildModel.analyze_calls = 0
        _FakeDeepFaceWithTaskAwareBuildModel.build_args = None

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", True), \
             patch.object(video_features, "_DEEPFACE_MODEL_READY", False), \
             patch.object(video_features, "_DEEPFACE_WARMUP_ATTEMPTED", False), \
             patch.object(video_features, "DeepFace", _FakeDeepFaceWithTaskAwareBuildModel, create=True), \
             patch.object(video_features, "_MP_FACE_DETECTION", None), \
             patch.object(video_features, "_FACE_CASCADE", None), \
             patch.object(video_features, "_seek_frame", return_value=frame), \
             patch.object(video_features, "_hsv_hist", return_value=np.array([0.0], dtype=np.float32)), \
             patch.object(video_features.settings, "video_expression_sample_sec", 3):
            video_features.extract(Path("fake.mp4"))

        self.assertEqual(_FakeDeepFaceWithTaskAwareBuildModel.build_calls, 1)
        self.assertEqual(_FakeDeepFaceWithTaskAwareBuildModel.build_args, ("Emotion", "facial_attribute"))
        self.assertEqual(_FakeDeepFaceWithTaskAwareBuildModel.analyze_calls, 2)

    def test_extract_does_not_retry_failing_deepface_build_every_time(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        _FakeDeepFaceWithFailingBuildModel.build_calls = 0
        _FakeDeepFaceWithFailingBuildModel.analyze_calls = 0

        with patch.object(video_features, "_CV2_AVAILABLE", True), \
             patch.object(video_features, "cv2", _FakeCV2), \
             patch.object(video_features, "_DEEPFACE_AVAILABLE", True), \
             patch.object(video_features, "_DEEPFACE_MODEL_READY", False), \
             patch.object(video_features, "_DEEPFACE_WARMUP_ATTEMPTED", False), \
             patch.object(video_features, "DeepFace", _FakeDeepFaceWithFailingBuildModel, create=True), \
             patch.object(video_features, "_MP_FACE_DETECTION", None), \
             patch.object(video_features, "_FACE_CASCADE", None), \
             patch.object(video_features, "_seek_frame", return_value=frame), \
             patch.object(video_features, "_hsv_hist", return_value=np.array([0.0], dtype=np.float32)), \
             patch.object(video_features.settings, "video_expression_sample_sec", 3):
            video_features.extract(Path("fake.mp4"))
            video_features.extract(Path("fake.mp4"))

        self.assertEqual(_FakeDeepFaceWithFailingBuildModel.build_calls, 1)
        self.assertEqual(_FakeDeepFaceWithFailingBuildModel.analyze_calls, 4)


if __name__ == "__main__":
    unittest.main()
