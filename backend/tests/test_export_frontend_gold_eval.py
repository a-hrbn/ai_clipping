import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import export_frontend_gold_eval


class ExportFrontendGoldEvalTests(unittest.TestCase):
    def test_main_writes_empty_artifacts_when_no_gold_eval_rows_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            storage.mkdir(parents=True, exist_ok=True)
            rows_out = root / "gold_rows.jsonl"
            output = root / "gold_dataset.jsonl"
            summary = root / "gold_summary.json"

            argv = [
                "export_frontend_gold_eval.py",
                "--storage-dir",
                str(storage),
                "--rows-out",
                str(rows_out),
                "--output",
                str(output),
                "--summary-out",
                str(summary),
            ]
            with patch.object(sys, "argv", argv):
                code = export_frontend_gold_eval.main()

            self.assertEqual(code, 0)
            self.assertEqual(rows_out.read_text(encoding="utf-8"), "")
            self.assertEqual(output.read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"rows": 0, "eval_dataset": {}})

    def test_main_preserves_existing_outputs_when_requested_and_frontend_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            storage.mkdir(parents=True, exist_ok=True)
            rows_out = root / "gold_rows.jsonl"
            output = root / "gold_dataset.jsonl"
            summary = root / "gold_summary.json"
            existing_row = {
                "queue_id": "manual:c1",
                "job_id": "manual",
                "candidate_id": "c1",
                "vod_id": "vod-1",
                "creator_id": "jynxzi",
                "publish_platform": "twitch",
                "label": 1,
                "ordinal_label": 3,
                "gold_grade": 3,
                "label_source": "gold_eval",
                "label_class": "gold_positive",
                "supervision_tier": "gold_eval",
                "features": {"candidate_id": "c1", "source": "reaction"},
            }
            rows_out.write_text(json.dumps(existing_row), encoding="utf-8")

            argv = [
                "export_frontend_gold_eval.py",
                "--storage-dir",
                str(storage),
                "--rows-out",
                str(rows_out),
                "--output",
                str(output),
                "--summary-out",
                str(summary),
                "--preserve-existing-if-empty",
            ]
            with patch.object(sys, "argv", argv):
                code = export_frontend_gold_eval.main()

            self.assertEqual(code, 0)
            preserved_rows = [json.loads(line) for line in rows_out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(preserved_rows, [existing_row])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(summary_payload["preserved_existing"])
            self.assertEqual(summary_payload["rows"], 1)

    def test_main_merges_existing_outputs_with_frontend_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            feedback_dir = storage / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            rows_out = root / "gold_rows.jsonl"
            output = root / "gold_dataset.jsonl"
            summary = root / "gold_summary.json"
            existing_rows = [
                {
                    "queue_id": "job-1:c1",
                    "job_id": "job-1",
                    "candidate_id": "c1",
                    "vod_id": "vod-1",
                    "creator_id": "jynxzi",
                    "publish_platform": "twitch",
                    "label": 0,
                    "ordinal_label": 0,
                    "gold_grade": 0,
                    "label_source": "gold_eval",
                    "label_class": "gold_negative",
                    "supervision_tier": "gold_eval",
                    "dataset_source": "gold_eval_frontend",
                    "features": {"candidate_id": "c1", "source": "peak"},
                    "updated_at": 10,
                },
                {
                    "queue_id": "manual:c2",
                    "job_id": "manual",
                    "candidate_id": "c2",
                    "vod_id": "vod-2",
                    "creator_id": "jynxzi",
                    "publish_platform": "twitch",
                    "label": 1,
                    "ordinal_label": 3,
                    "gold_grade": 3,
                    "label_source": "gold_eval",
                    "label_class": "gold_positive",
                    "supervision_tier": "gold_eval",
                    "dataset_source": "gold_eval_manual",
                    "features": {"candidate_id": "c2", "source": "reaction"},
                },
            ]
            rows_out.write_text(
                "\n".join(json.dumps(row) for row in existing_rows),
                encoding="utf-8",
            )
            (feedback_dir / "gold_eval_updates.jsonl").write_text(
                json.dumps(
                    {
                        "queue_id": "job-1:c1",
                        "job_id": "job-1",
                        "candidate_id": "c1",
                        "vod_id": "vod-1",
                        "creator_id": "jynxzi",
                        "publish_platform": "twitch",
                        "label": 1,
                        "ordinal_label": 3,
                        "gold_grade": 3,
                        "label_source": "gold_eval",
                        "label_class": "gold_positive",
                        "supervision_tier": "gold_eval",
                        "dataset_source": "gold_eval_frontend",
                        "features": {"candidate_id": "c1", "source": "peak"},
                        "updated_at": 20,
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "export_frontend_gold_eval.py",
                "--storage-dir",
                str(storage),
                "--rows-out",
                str(rows_out),
                "--output",
                str(output),
                "--summary-out",
                str(summary),
                "--creator-id",
                "jynxzi",
                "--merge-existing",
            ]
            with patch.object(sys, "argv", argv):
                code = export_frontend_gold_eval.main()

            self.assertEqual(code, 0)
            merged_rows = [json.loads(line) for line in rows_out.read_text(encoding="utf-8").splitlines()]
            rows_by_queue = {row["queue_id"]: row for row in merged_rows}
            self.assertEqual(len(merged_rows), 2)
            self.assertEqual(rows_by_queue["job-1:c1"]["gold_grade"], 3)
            self.assertEqual(rows_by_queue["manual:c2"]["dataset_source"], "gold_eval_manual")
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(summary_payload["merged_existing"])
            self.assertEqual(summary_payload["source_rows"]["existing"], 2)
            self.assertEqual(summary_payload["source_rows"]["frontend"], 1)
            self.assertEqual(summary_payload["source_rows"]["merged"], 2)
            self.assertEqual(summary_payload["source_rows"]["frontend_updates"], 1)

    def test_main_merge_keeps_manual_row_when_frontend_has_no_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            feedback_dir = storage / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            rows_out = root / "gold_rows.jsonl"
            output = root / "gold_dataset.jsonl"
            summary = root / "gold_summary.json"
            rows_out.write_text(
                json.dumps(
                    {
                        "queue_id": "manual:c1",
                        "job_id": "manual",
                        "candidate_id": "c1",
                        "vod_id": "vod-1",
                        "creator_id": "jynxzi",
                        "publish_platform": "twitch",
                        "label": 1,
                        "ordinal_label": 3,
                        "gold_grade": 3,
                        "label_source": "gold_eval",
                        "label_class": "gold_positive",
                        "supervision_tier": "gold_eval",
                        "dataset_source": "gold_eval_manual",
                        "features": {"candidate_id": "c1", "source": "reaction"},
                    }
                ),
                encoding="utf-8",
            )
            (feedback_dir / "gold_eval_updates.jsonl").write_text(
                json.dumps(
                    {
                        "queue_id": "manual:c1",
                        "job_id": "manual",
                        "candidate_id": "c1",
                        "vod_id": "vod-1",
                        "creator_id": "jynxzi",
                        "publish_platform": "twitch",
                        "label": 0,
                        "ordinal_label": 0,
                        "gold_grade": 0,
                        "label_source": "gold_eval",
                        "label_class": "gold_negative",
                        "supervision_tier": "gold_eval",
                        "dataset_source": "gold_eval_frontend",
                        "features": {"candidate_id": "c1", "source": "reaction"},
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "export_frontend_gold_eval.py",
                "--storage-dir",
                str(storage),
                "--rows-out",
                str(rows_out),
                "--output",
                str(output),
                "--summary-out",
                str(summary),
                "--creator-id",
                "jynxzi",
                "--merge-existing",
            ]
            with patch.object(sys, "argv", argv):
                code = export_frontend_gold_eval.main()

            self.assertEqual(code, 0)
            merged_row = json.loads(rows_out.read_text(encoding="utf-8"))
            self.assertEqual(merged_row["dataset_source"], "gold_eval_manual")
            self.assertEqual(merged_row["gold_grade"], 3)

    def test_export_builds_gold_eval_dataset_from_debug_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            debug_dir = storage / "job-1" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "gold_eval.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "queue_id": "job-1:c1",
                                "job_id": "job-1",
                                "candidate_id": "c1",
                                "vod_id": "vod-1",
                                "creator_id": "jynxzi",
                                "publish_platform": "twitch",
                                "source_url": "https://www.twitch.tv/videos/123",
                                "window_start": 10.0,
                                "window_end": 24.0,
                                "label": 1,
                                "ordinal_label": 3,
                                "gold_grade": 3,
                                "label_source": "gold_eval",
                                "label_class": "gold_positive",
                                "sample_weight": 1.0,
                                "supervision_tier": "gold_eval",
                                "supervision_reason": "frontend_gold_eval",
                                "dataset_source": "gold_eval_frontend",
                                "selected": True,
                                "exported": False,
                                "candidate_source": "reaction",
                                "features": {"candidate_id": "c1", "source": "reaction"},
                                "updated_at": 100,
                            }
                        )
                    ]
                ),
                encoding="utf-8",
            )

            payload = export_frontend_gold_eval.export_frontend_gold_eval(storage)

            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(len(payload["eval_rows"]), 1)
            self.assertEqual(payload["eval_rows"][0]["dataset_role"], "gold_eval")
            self.assertEqual(payload["summary"]["eval_dataset"]["rows"], 1)

    def test_export_builds_dataset_from_snapshot_rows_after_job_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            feedback_dir = storage / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            (feedback_dir / "gold_eval_updates.jsonl").write_text(
                json.dumps(
                    {
                        "queue_id": "job-1:c1",
                        "job_id": "job-1",
                        "candidate_id": "c1",
                        "vod_id": "vod-1",
                        "creator_id": "jynxzi",
                        "publish_platform": "twitch",
                        "source_url": "https://www.twitch.tv/videos/123",
                        "window_start": 10.0,
                        "window_end": 24.0,
                        "label": 1,
                        "ordinal_label": 2,
                        "gold_grade": 2,
                        "label_source": "gold_eval",
                        "label_class": "gold_positive",
                        "sample_weight": 1.0,
                        "supervision_tier": "gold_eval",
                        "supervision_reason": "frontend_gold_eval",
                        "dataset_source": "gold_eval_frontend",
                        "selected": True,
                        "exported": False,
                        "candidate_source": "reaction",
                        "features": {"candidate_id": "c1", "source": "reaction"},
                        "updated_at": 100,
                    }
                ),
                encoding="utf-8",
            )

            payload = export_frontend_gold_eval.export_frontend_gold_eval(storage)

            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["candidate_id"], "c1")
            self.assertIn(payload["rows"][0]["gold_holdout_split"], {"train", "holdout"})
            self.assertEqual(payload["summary"]["eval_dataset"]["rows"], 1)

    def test_export_keeps_latest_row_per_job_and_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            debug_dir = storage / "job-1" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "gold_eval.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "job_id": "job-1",
                                "candidate_id": "c1",
                                "label": 0,
                                "ordinal_label": 0,
                                "gold_grade": 0,
                                "label_source": "gold_eval",
                                "supervision_tier": "gold_eval",
                                "features": {"candidate_id": "c1"},
                                "updated_at": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "job_id": "job-1",
                                "candidate_id": "c1",
                                "label": 1,
                                "ordinal_label": 3,
                                "gold_grade": 3,
                                "label_source": "gold_eval",
                                "supervision_tier": "gold_eval",
                                "features": {"candidate_id": "c1"},
                                "updated_at": 20,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            payload = export_frontend_gold_eval.export_frontend_gold_eval(storage)

            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["gold_grade"], 3)

    def test_export_filters_by_creator_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            feedback_dir = storage / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            (feedback_dir / "gold_eval_updates.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "job_id": "job-1",
                                "candidate_id": "c1",
                                "creator_id": "jynxzi",
                                "label": 1,
                                "ordinal_label": 3,
                                "gold_grade": 3,
                                "label_source": "gold_eval",
                                "supervision_tier": "gold_eval",
                                "features": {"candidate_id": "c1"},
                                "updated_at": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "job_id": "job-2",
                                "candidate_id": "c2",
                                "creator_id": "tarik",
                                "label": 1,
                                "ordinal_label": 2,
                                "gold_grade": 2,
                                "label_source": "gold_eval",
                                "supervision_tier": "gold_eval",
                                "features": {"candidate_id": "c2"},
                                "updated_at": 11,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            payload = export_frontend_gold_eval.export_frontend_gold_eval(storage, creator_id="jynxzi")

            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["creator_id"], "jynxzi")

    def test_export_prefers_newest_row_across_snapshot_and_debug_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            debug_dir = storage / "job-1" / "debug"
            feedback_dir = storage / "feedback"
            debug_dir.mkdir(parents=True, exist_ok=True)
            feedback_dir.mkdir(parents=True, exist_ok=True)
            (feedback_dir / "gold_eval_updates.jsonl").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "candidate_id": "c1",
                        "label": 0,
                        "ordinal_label": 0,
                        "gold_grade": 0,
                        "label_source": "gold_eval",
                        "supervision_tier": "gold_eval",
                        "features": {"candidate_id": "c1"},
                        "updated_at": 10,
                    }
                ),
                encoding="utf-8",
            )
            (debug_dir / "gold_eval.jsonl").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "candidate_id": "c1",
                        "label": 1,
                        "ordinal_label": 3,
                        "gold_grade": 3,
                        "label_source": "gold_eval",
                        "supervision_tier": "gold_eval",
                        "features": {"candidate_id": "c1"},
                        "updated_at": 20,
                    }
                ),
                encoding="utf-8",
            )

            payload = export_frontend_gold_eval.export_frontend_gold_eval(storage)

            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["gold_grade"], 3)
