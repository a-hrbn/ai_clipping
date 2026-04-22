import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import build_gold_eval_queue, import_gold_eval_labels


class GoldEvalWorkflowTests(unittest.TestCase):
    def test_build_gold_eval_queue_writes_joinable_csv_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            storage_dir = root / "storage"
            debug_dir = storage_dir / "job-1" / "debug"
            debug_dir.mkdir(parents=True)
            manifest.write_text(
                "job_id,vod_url,creator_id,platform,vod_id\n"
                "job-1,https://www.twitch.tv/videos/12345,jynxzi,twitch,12345\n",
                encoding="utf-8",
            )
            (debug_dir / "candidates_normalized.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "cand-1", "start": 12.0, "end": 24.0}),
                        json.dumps({"id": "cand-2", "start": 40.0, "end": 50.0}),
                    ]
                ),
                encoding="utf-8",
            )
            (debug_dir / "candidate_features.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"candidate_id": "cand-1", "source": "reaction", "hook_strength": 0.8}),
                        json.dumps({"candidate_id": "cand-2", "source": "quote", "hook_strength": 0.2}),
                    ]
                ),
                encoding="utf-8",
            )
            (debug_dir / "segments.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"start": 10.0, "end": 15.0, "text": "first line"}),
                        json.dumps({"start": 20.0, "end": 23.0, "text": "second line"}),
                    ]
                ),
                encoding="utf-8",
            )
            out_base = root / "gold_eval_queue"
            with patch(
                "sys.argv",
                [
                    "build_gold_eval_queue.py",
                    "--manifest",
                    str(manifest),
                    "--storage-dir",
                    str(storage_dir),
                    "--out",
                    str(out_base),
                    "--seed",
                    "7",
                ],
            ):
                rc = build_gold_eval_queue.main()

            self.assertEqual(rc, 0)
            queue_path = out_base.with_suffix(".csv")
            sidecar_path = queue_path.with_suffix(".features.jsonl")
            queue_rows = list(csv.DictReader(queue_path.open("r", encoding="utf-8", newline="")))
            sidecar_rows = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(queue_rows), 2)
        self.assertIn("queue_id", queue_rows[0])
        self.assertIn("job_id", queue_rows[0])
        self.assertTrue(queue_rows[0]["queue_id"].startswith("job-1:cand-"))
        self.assertEqual({row["queue_id"] for row in queue_rows}, {row["queue_id"] for row in sidecar_rows})
        self.assertEqual({row["job_id"] for row in sidecar_rows}, {"job-1"})

    def test_import_gold_eval_labels_rebuilds_gold_eval_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled_csv = root / "gold_eval_queue.csv"
            sidecar = labeled_csv.with_suffix(".features.jsonl")
            labeled_csv.write_text(
                "queue_id,job_id,candidate_id,vod_id,creator_id,platform,source_url,gold_grade,notes\n"
                "job-1:cand-1,job-1,cand-1,12345,jynxzi,twitch,https://www.twitch.tv/videos/12345,3,strong winner\n"
                "job-1:cand-2,job-1,cand-2,12345,jynxzi,twitch,https://www.twitch.tv/videos/12345,1,weak opener\n"
                "job-1:cand-3,job-1,cand-3,12345,jynxzi,twitch,https://www.twitch.tv/videos/12345,,skip for now\n",
                encoding="utf-8",
            )
            sidecar.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "queue_id": "job-1:cand-1",
                                "job_id": "job-1",
                                "candidate_id": "cand-1",
                                "vod_id": "12345",
                                "creator_id": "jynxzi",
                                "platform": "twitch",
                                "source_url": "https://www.twitch.tv/videos/12345",
                                "window_start": 12.0,
                                "window_end": 24.0,
                                "features": {"candidate_id": "cand-1", "source": "reaction", "hook_strength": 0.9},
                            }
                        ),
                        json.dumps(
                            {
                                "queue_id": "job-1:cand-2",
                                "job_id": "job-1",
                                "candidate_id": "cand-2",
                                "vod_id": "12345",
                                "creator_id": "jynxzi",
                                "platform": "twitch",
                                "source_url": "https://www.twitch.tv/videos/12345",
                                "window_start": 30.0,
                                "window_end": 42.0,
                                "features": {"candidate_id": "cand-2", "source": "quote", "hook_strength": 0.2},
                            }
                        ),
                        json.dumps(
                            {
                                "queue_id": "job-1:cand-3",
                                "job_id": "job-1",
                                "candidate_id": "cand-3",
                                "vod_id": "12345",
                                "creator_id": "jynxzi",
                                "platform": "twitch",
                                "source_url": "https://www.twitch.tv/videos/12345",
                                "window_start": 50.0,
                                "window_end": 61.0,
                                "features": {"candidate_id": "cand-3", "source": "reaction", "hook_strength": 0.4},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            payload = import_gold_eval_labels.import_gold_eval_labels(labeled_csv, sidecar)

        self.assertEqual(payload["summary"]["queue_rows"], 3)
        self.assertEqual(payload["summary"]["labeled_rows"], 2)
        self.assertEqual(payload["summary"]["skipped_unlabeled_rows"], 1)
        self.assertEqual(payload["summary"]["grade_distribution"]["3"], 1)
        self.assertEqual(payload["summary"]["grade_distribution"]["1"], 1)
        rows = payload["rows"]
        eval_rows = payload["eval_rows"]
        self.assertEqual(rows[0]["label_source"], "gold_eval")
        self.assertEqual(rows[0]["supervision_tier"], "gold_eval")
        self.assertIn(rows[0]["gold_holdout_split"], {"train", "holdout"})
        self.assertEqual(eval_rows[0]["dataset_role"], "gold_eval")
        self.assertEqual(eval_rows[0]["eval_group"], "12345")
        self.assertEqual(eval_rows[0]["source_bucket"], "reaction")
        self.assertEqual(eval_rows[1]["label"], 0)

    def test_import_gold_eval_labels_requires_notes_for_grade_three(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled_csv = root / "gold_eval_queue.csv"
            sidecar = labeled_csv.with_suffix(".features.jsonl")
            labeled_csv.write_text(
                "queue_id,candidate_id,gold_grade,notes\n"
                "job-1:cand-1,cand-1,3,\n",
                encoding="utf-8",
            )
            sidecar.write_text(
                json.dumps(
                    {
                        "queue_id": "job-1:cand-1",
                        "candidate_id": "cand-1",
                        "features": {"candidate_id": "cand-1", "source": "reaction"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "gold_notes are required"):
                import_gold_eval_labels.import_gold_eval_labels(labeled_csv, sidecar)

    def test_import_gold_eval_labels_requires_notes_for_high_scoring_grade_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled_csv = root / "gold_eval_queue.csv"
            sidecar = labeled_csv.with_suffix(".features.jsonl")
            labeled_csv.write_text(
                "queue_id,candidate_id,gold_grade,notes\n"
                "job-1:cand-1,cand-1,0,\n",
                encoding="utf-8",
            )
            sidecar.write_text(
                json.dumps(
                    {
                        "queue_id": "job-1:cand-1",
                        "candidate_id": "cand-1",
                        "heuristic_rank_score": 8.2,
                        "features": {"candidate_id": "cand-1", "source": "reaction"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "gold_notes are required"):
                import_gold_eval_labels.import_gold_eval_labels(labeled_csv, sidecar)

    def test_import_gold_eval_labels_requires_queue_id_when_candidate_ids_collide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled_csv = root / "gold_eval_queue.csv"
            sidecar = labeled_csv.with_suffix(".features.jsonl")
            labeled_csv.write_text(
                "candidate_id,gold_grade\n"
                "cand-1,3\n",
                encoding="utf-8",
            )
            sidecar.write_text(
                "\n".join(
                    [
                        json.dumps({"queue_id": "job-1:cand-1", "candidate_id": "cand-1", "features": {"candidate_id": "cand-1"}}),
                        json.dumps({"queue_id": "job-2:cand-1", "candidate_id": "cand-1", "features": {"candidate_id": "cand-1"}}),
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "queue_id is required"):
                import_gold_eval_labels.import_gold_eval_labels(labeled_csv, sidecar)


if __name__ == "__main__":
    unittest.main()
