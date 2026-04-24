import csv
import json
import tempfile
import unittest
from pathlib import Path

from ml import export_ranker_hard_negatives


class ExportRankerHardNegativesTests(unittest.TestCase):
    def test_exports_high_scoring_false_positives_from_gold_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnosis_csv = root / "diagnosis.csv"
            gold_rows_path = root / "gold_rows.jsonl"
            with diagnosis_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "queue_id",
                        "job_id",
                        "candidate_id",
                        "vod_id",
                        "label",
                        "error_type",
                        "forced_score",
                        "heuristic_score",
                        "trained_score",
                        "group",
                        "rank",
                        "review_url",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "queue_id": "q-1",
                        "job_id": "job-1",
                        "candidate_id": "cand-1",
                        "vod_id": "vod-1",
                        "label": "0",
                        "error_type": "false_positive",
                        "forced_score": "9.5",
                        "heuristic_score": "8.0",
                        "trained_score": "10.0",
                        "group": "vod-1",
                        "rank": "1",
                        "review_url": "https://example.test/vod?t=10s",
                    }
                )
                writer.writerow(
                    {
                        "queue_id": "q-2",
                        "job_id": "job-1",
                        "candidate_id": "cand-2",
                        "vod_id": "vod-1",
                        "label": "0",
                        "error_type": "false_positive",
                        "forced_score": "-1.0",
                    }
                )
                writer.writerow(
                    {
                        "queue_id": "q-3",
                        "job_id": "job-1",
                        "candidate_id": "cand-3",
                        "vod_id": "vod-1",
                        "label": "1",
                        "error_type": "true_positive",
                        "forced_score": "11.0",
                    }
                )
            gold_rows = [
                {
                    "queue_id": "q-1",
                    "job_id": "job-1",
                    "candidate_id": "cand-1",
                    "vod_id": "vod-1",
                    "label": 0,
                    "label_class": "gold_negative",
                    "gold_holdout_split": "holdout",
                    "sample_weight": 1.7,
                    "candidate_source": "peak",
                    "features": {"hook_strength": 0.2},
                },
                {
                    "queue_id": "q-2",
                    "job_id": "job-1",
                    "candidate_id": "cand-2",
                    "vod_id": "vod-1",
                    "label": 0,
                    "gold_holdout_split": "holdout",
                    "features": {"hook_strength": 0.4},
                },
            ]
            gold_rows_path.write_text("\n".join(json.dumps(row) for row in gold_rows), encoding="utf-8")

            payload = export_ranker_hard_negatives.export_hard_negatives(
                diagnosis_csv=diagnosis_csv,
                gold_rows_path=gold_rows_path,
                profile="jynxzi",
                min_forced_score=0.0,
                max_rows=10,
            )

        self.assertEqual(payload["summary"]["selected_false_positive_rows"], 1)
        self.assertEqual(payload["summary"]["exported_rows"], 1)
        row = payload["rows"][0]
        self.assertEqual(row["queue_id"], "q-1")
        self.assertEqual(row["label"], 0)
        self.assertEqual(row["dataset_source"], "gold_eval_hard_negative_train")
        self.assertEqual(row["label_source"], "gold_eval")
        self.assertEqual(row["supervision_tier"], "gold_eval")
        self.assertEqual(row["gold_holdout_split"], "train")
        self.assertEqual(row["hard_negative_original_split"], "holdout")
        self.assertEqual(row["hard_negative_forced_score"], 9.5)
        self.assertEqual(row["hard_negative_original_sample_weight"], 1.7)
        self.assertEqual(row["sample_weight"], 1.0)
        self.assertEqual(row["features"]["hook_strength"], 0.2)


if __name__ == "__main__":
    unittest.main()
