import json
import tempfile
import unittest
from pathlib import Path

from ml import diagnose_rollout_readiness_coverage


class DiagnoseRolloutReadinessCoverageTests(unittest.TestCase):
    def test_diagnose_coverage_reports_split_and_holdout_only_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            holdout = root / "holdout.jsonl"
            rows = [
                _row("a-1", "vod-a", 1),
                _row("a-2", "vod-a", 0),
                _row("b-1", "vod-b", 1),
                _row("b-2", "vod-b", 0),
                _row("c-1", "vod-c", 1),
                _row("d-1", "vod-d", 0),
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            holdout.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        _row("a-holdout", "vod-a", 1),
                        _row("z-holdout", "vod-z", 1),
                    ]
                ),
                encoding="utf-8",
            )

            report = diagnose_rollout_readiness_coverage.diagnose_coverage(
                dataset_path=dataset,
                profile="jynxzi",
                holdout_dataset_path=holdout,
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["summary"]["rows_after_training_policy"], 6)
            self.assertEqual(report["summary"]["available_groups_after_policy"], 4)
            self.assertEqual(report["targets"]["mode"], "inventory_calibrated")
            self.assertEqual(report["targets"]["validation_groups"], 4)
            self.assertEqual(report["targets"]["raw_targets"]["validation_groups"], 40)
            self.assertIn("validation_groups_below_rollout_target", report["readiness"]["failures"])
            self.assertEqual(report["diagnosis"]["available_groups_needed"], 0)
            self.assertIn("split_validation_groups", report["diagnosis"]["blockers"])
            self.assertEqual(report["holdout_overlap"]["holdout_only_groups"], 1)
            self.assertEqual(report["holdout_overlap"]["holdout_only_sample_groups"], ["vod-z"])
            self.assertTrue(any(group["split"] == "validation" for group in report["groups"]))
            self.assertTrue(
                any(group["missing_reason"] == "not_selected_for_validation_by_current_split" for group in report["groups"])
            )

    def test_diagnose_coverage_reports_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "empty.jsonl"
            dataset.write_text("", encoding="utf-8")

            report = diagnose_rollout_readiness_coverage.diagnose_coverage(
                dataset_path=dataset,
                profile="jynxzi",
            )

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["reason"], "no_dataset_rows")

    def test_inventory_calibrated_targets_allow_maxed_out_jynxzi_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "train.jsonl"
            rows: list[dict] = []
            for group_idx in range(38):
                row_count = 2 if group_idx < 15 else 13
                for row_idx in range(row_count):
                    label = 1 if row_idx % 2 == 0 else 0
                    rows.append(_row(f"{group_idx}-{row_idx}", f"vod-{group_idx:02d}", label))
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            report = diagnose_rollout_readiness_coverage.diagnose_coverage(
                dataset_path=dataset,
                profile="jynxzi",
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["summary"]["available_groups_after_policy"], 38)
            self.assertEqual(report["targets"]["mode"], "inventory_calibrated")
            self.assertEqual(report["targets"]["training_rows"], 304)
            self.assertEqual(report["targets"]["positive_rows"], 152)
            self.assertEqual(report["targets"]["validation_groups"], 15)
            self.assertTrue(report["readiness"]["passed"])
            self.assertEqual(report["readiness"]["failures"], [])


def _row(candidate_id: str, vod_id: str, label: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "job_id": f"job-{vod_id}",
        "queue_id": candidate_id,
        "vod_id": vod_id,
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "creator_id": "jynxzi",
        "label": label,
        "candidate_source": "reaction",
        "source_bucket": "reaction",
        "content_bucket": "story",
        "publish_platform": "twitch",
        "gold_holdout_split": "train",
        "window_start": 10.0,
        "window_end": 35.0,
        "features": {
            "candidate_id": candidate_id,
            "duration": 25.0,
            "hook_strength": 0.8 if label else 0.1,
            "payoff_score": 0.7 if label else 0.1,
            "source": "reaction",
        },
    }


if __name__ == "__main__":
    unittest.main()
