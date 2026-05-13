import pathlib
import tempfile
import unittest
from unittest.mock import patch

from services import vision_language_features
from services.timeline_types import Candidate


class VisionLanguageFeatureTests(unittest.TestCase):
    def test_score_candidates_uses_local_model_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = pathlib.Path(tmp) / "video.mp4"
            video_path.write_bytes(b"fake")
            candidate = Candidate(id="c1", start=0.0, end=10.0, source="reaction", anchor_time=4.0)

            with patch.object(vision_language_features.settings, "enable_multimodal_embeddings", True), \
                 patch.object(vision_language_features, "_load_model", return_value=("processor", "model", "torch")), \
                 patch.object(vision_language_features, "_score_candidate", return_value={
                     "vl_prompt_max_score": 0.8,
                     "vl_clutch_score": 0.8,
                     "vl_comedy_score": 0.1,
                     "vl_reaction_face_score": 0.5,
                     "vl_scoreboard_score": 0.2,
                 }):
                scores = vision_language_features.score_candidates(video_path, [candidate])

        self.assertEqual(scores["c1"]["vl_prompt_max_score"], 0.8)

    def test_best_fingerprint_like_prompt_scores_are_empty_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = pathlib.Path(tmp) / "video.mp4"
            video_path.write_bytes(b"fake")
            candidate = Candidate(id="c1", start=0.0, end=10.0, source="reaction", anchor_time=4.0)

            with patch.object(vision_language_features.settings, "enable_multimodal_embeddings", False):
                scores = vision_language_features.score_candidates(video_path, [candidate])

        self.assertEqual(scores, {})

    def test_model_availability_reports_missing_local_model_without_download(self) -> None:
        with patch.object(vision_language_features.settings, "enable_multimodal_embeddings", True), \
             patch.object(vision_language_features.importlib.util, "find_spec", return_value=object()), \
             patch.object(vision_language_features, "_load_model", return_value=None):
            status = vision_language_features.model_availability()

        self.assertTrue(status["enabled"])
        self.assertFalse(status["available"])
        self.assertTrue(status["local_files_only"])
        self.assertEqual(status["reason"], "local_model_unavailable")


if __name__ == "__main__":
    unittest.main()
