import tempfile
import unittest
from pathlib import Path

from ml import audit_chat_feature_coverage


class AuditChatFeatureCoverageTests(unittest.TestCase):
    def test_build_summary_counts_chat_features_and_cache(self) -> None:
        rows = [
            {
                "vod_id": "123",
                "dataset_source": "gold_eval_train",
                "gold_holdout_split": "train",
                "features": {
                    "has_chat_features": True,
                    "chat_peak": 0.8,
                    "chat_mean": 0.2,
                    "chat_hype_score": 0.8,
                },
            },
            {
                "vod_id": "456",
                "dataset_source": "gold_eval_holdout",
                "gold_holdout_split": "holdout",
                "features": {
                    "has_chat_features": False,
                    "chat_peak": 0.0,
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "vod_feature_cache"
            chat_dir = cache_dir / "123" / "chat"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.rechat.json").write_text("[]", encoding="utf-8")

            summary = audit_chat_feature_coverage.build_summary(
                rows,
                storage_dir=root / "storage",
                vod_feature_cache_dir=cache_dir,
            )

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["rows_with_chat_features"], 1)
        self.assertEqual(summary["chat_feature_coverage"], 0.5)
        self.assertEqual(summary["chat_category_nonzero_rows"]["chat_hype_score"], 1)
        self.assertEqual(summary["cache"]["chat_cached_vods"], 1)


if __name__ == "__main__":
    unittest.main()
