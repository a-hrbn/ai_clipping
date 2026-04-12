import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import build_vector_reference_set
from services import clip_similarity
from services.transcriber import RichSegment


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            lower = str(text).lower()
            if "winner" in lower or "hook" in lower:
                vector = np.array([1.0, 0.0], dtype=np.float32)
            elif "loser" in lower or "flat" in lower:
                vector = np.array([0.0, 1.0], dtype=np.float32)
            else:
                vector = np.array([0.7, 0.3], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vector) or 1.0
                vector = vector / norm
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


class ClipSimilarityTests(unittest.TestCase):
    def test_build_reference_manifest_uses_text_and_labels(self) -> None:
        rows = [
            {
                "candidate_id": "winner-1",
                "label": 1,
                "transcript": "winner hook big payoff",
                "view_count": 1000,
                "like_count": 120,
                "comment_count": 30,
            },
            {
                "candidate_id": "loser-1",
                "label": 0,
                "transcript": "flat loser clip",
                "view_count": 100,
                "like_count": 2,
                "comment_count": 0,
            },
        ]

        manifest = build_vector_reference_set.build_reference_manifest(rows, "jynxzi")

        self.assertEqual(len(manifest), 2)
        self.assertEqual(manifest[0]["label"], "winner")
        self.assertIn("hook", manifest[0]["hook_text"])

    def test_score_candidate_window_prefers_winner_neighbors(self) -> None:
        artifact = {
            "rows": [
                {"_index": 0, "clip_id": "winner-1", "label": "winner", "quality_score": 1.6},
                {"_index": 1, "clip_id": "loser-1", "label": "loser", "quality_score": 0.1},
            ],
            "full": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "hook": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "payoff": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        }
        window = [
            RichSegment(start=0.0, end=2.0, text="winner hook", hype_score=2.0),
            RichSegment(start=2.0, end=4.0, text="winner payoff", hype_score=4.0),
        ]

        with patch("services.clip_similarity._load_artifact", return_value=artifact), \
             patch("services.clip_similarity._get_model", return_value=_FakeModel()):
            features = clip_similarity.score_candidate_window(window, 2.0, "jynxzi")

        self.assertGreater(features["winner_similarity_max"], features["loser_similarity_max"])
        self.assertGreater(features["similarity_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
