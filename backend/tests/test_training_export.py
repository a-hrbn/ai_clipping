import json
import tempfile
import unittest
from pathlib import Path

from services.training_export import export_training_rows


class TrainingExportTests(unittest.TestCase):
    def test_export_training_rows_merges_features_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            debug = storage / "job-1" / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "candidate_features.jsonl").write_text(
                json.dumps({"candidate_id": "c1", "max_hype": 5.0, "source": "reaction"}) + "\n",
                encoding="utf-8",
            )
            (debug / "feedback.jsonl").write_text(
                json.dumps({
                    "job_id": "job-1",
                    "candidate_id": "c1",
                    "selected": True,
                    "exported": False,
                    "final_start": 1.0,
                    "final_end": 10.0,
                    "rejected_reason": None,
                    "rejection_source": None,
                }) + "\n",
                encoding="utf-8",
            )
            output = storage / "feedback" / "training_rows.jsonl"

            count = export_training_rows(storage, output)

            self.assertEqual(count, 1)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["label"], 1)
            self.assertIsNone(rows[0]["rejection_source"])
            self.assertEqual(rows[0]["features"]["candidate_id"], "c1")
            self.assertEqual(rows[0]["supervision_tier"], "pipeline_feedback")
            self.assertEqual(rows[0]["label_class"], "pipeline_positive")
            self.assertEqual(rows[0]["ordinal_label"], 2)

    def test_export_training_rows_skips_post_processing_filtered_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            debug = storage / "job-1" / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "candidate_features.jsonl").write_text(
                "\n".join([
                    json.dumps({"candidate_id": "low", "max_hype": 1.0, "source": "reaction"}),
                    json.dumps({"candidate_id": "soft", "max_hype": 4.0, "source": "reaction"}),
                ]) + "\n",
                encoding="utf-8",
            )
            (debug / "feedback.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "job_id": "job-1",
                        "candidate_id": "low",
                        "selected": False,
                        "exported": False,
                        "final_start": None,
                        "final_end": None,
                        "rejected_reason": "low_rank",
                        "rejection_source": "low_rank",
                    }),
                    json.dumps({
                        "job_id": "job-1",
                        "candidate_id": "soft",
                        "selected": False,
                        "exported": False,
                        "final_start": None,
                        "final_end": None,
                        "rejected_reason": "post_processing_filtered",
                        "rejection_source": "post_processing_filtered",
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            output = storage / "feedback" / "training_rows.jsonl"

            count = export_training_rows(storage, output)

            self.assertEqual(count, 1)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["candidate_id"], "low")
            self.assertEqual(rows[0]["label"], 0)
            self.assertEqual(rows[0]["rejection_source"], "low_rank")
            self.assertEqual(rows[0]["supervision_tier"], "pipeline_feedback")
            self.assertEqual(rows[0]["label_class"], "pipeline_negative")

    def test_export_training_rows_prefers_real_outcome_feedback_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            debug = storage / "job-2" / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "candidate_features.jsonl").write_text(
                json.dumps({"candidate_id": "c2", "max_hype": 6.0, "source": "reaction"}) + "\n",
                encoding="utf-8",
            )
            (debug / "feedback.jsonl").write_text(
                json.dumps({
                    "job_id": "job-2",
                    "candidate_id": "c2",
                    "selected": False,
                    "exported": False,
                    "final_start": None,
                    "final_end": None,
                    "rejected_reason": "post_processing_filtered",
                    "rejection_source": "post_processing_filtered",
                    "feedback_rating": "good",
                }) + "\n",
                encoding="utf-8",
            )
            output = storage / "feedback" / "training_rows.jsonl"

            count = export_training_rows(storage, output)

            self.assertEqual(count, 1)
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["label"], 1)
            self.assertEqual(row["label_source"], "feedback_rating")
            self.assertGreater(row["sample_weight"], 1.0)
            self.assertEqual(row["supervision_tier"], "outcome_feedback")
            self.assertEqual(row["label_class"], "outcome_positive")
            self.assertEqual(row["ordinal_label"], 3)
            self.assertEqual(row["dataset_source"], "feedback_outcome_export")

    def test_export_training_rows_treats_medium_feedback_as_weak_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            debug = storage / "job-4" / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "candidate_features.jsonl").write_text(
                json.dumps({"candidate_id": "c4", "max_hype": 5.2, "source": "reaction"}) + "\n",
                encoding="utf-8",
            )
            (debug / "feedback.jsonl").write_text(
                json.dumps({
                    "job_id": "job-4",
                    "candidate_id": "c4",
                    "selected": True,
                    "exported": False,
                    "final_start": 2.0,
                    "final_end": 12.0,
                    "rejected_reason": None,
                    "rejection_source": None,
                    "feedback_rating": "medium",
                }) + "\n",
                encoding="utf-8",
            )
            output = storage / "feedback" / "training_rows.jsonl"

            count = export_training_rows(storage, output)

            self.assertEqual(count, 1)
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["label"], 1)
            self.assertEqual(row["label_source"], "feedback_rating_neutral")
            self.assertEqual(row["supervision_tier"], "feedback_rating_neutral")
            self.assertEqual(row["ordinal_label"], 1)
            self.assertEqual(row["feedback_rating"], "medium")

    def test_export_training_rows_preserves_variant_and_dynamic_feature_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            debug = storage / "job-3" / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "candidate_features.jsonl").write_text(
                json.dumps({
                    "candidate_id": "c3",
                    "max_hype": 4.2,
                    "source": "reaction",
                    "dynamic_trend_score": 0.61,
                    "projected_retention_score": 0.73,
                    "feature_build_version": 6,
                    "retrieved_memory_span_count": 2.0,
                    "setup_depth_score": 0.44,
                }) + "\n",
                encoding="utf-8",
            )
            (debug / "feedback.jsonl").write_text(
                json.dumps({
                    "job_id": "job-3",
                    "candidate_id": "c3",
                    "selected": True,
                    "exported": False,
                    "final_start": 3.0,
                    "final_end": 18.0,
                    "rejected_reason": None,
                    "rejection_source": None,
                    "publish_platform": "tiktok",
                    "final_variant_key": "aggressive_hook",
                    "feedback_rating": "good",
                }) + "\n",
                encoding="utf-8",
            )
            output = storage / "feedback" / "training_rows.jsonl"

            count = export_training_rows(storage, output)

            self.assertEqual(count, 1)
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["publish_platform"], "tiktok")
            self.assertEqual(row["final_variant_key"], "aggressive_hook")
            self.assertEqual(row["candidate_source"], "reaction")
            self.assertEqual(row["content_bucket"], "reaction")
            self.assertEqual(row["features"]["projected_retention_score"], 0.73)
            self.assertTrue(row["feature_provenance"]["has_dynamic_trend"])
            self.assertTrue(row["feature_provenance"]["has_variant_projection"])
            self.assertTrue(row["feature_provenance"]["has_vod_memory"])


if __name__ == "__main__":
    unittest.main()
