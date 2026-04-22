import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import build_gold_eval_manifest


class BuildGoldEvalManifestTests(unittest.TestCase):
    def test_build_manifest_resolves_job_from_request_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            job_dir = storage / "job-1"
            debug_dir = job_dir / "debug"
            debug_dir.mkdir(parents=True)
            (debug_dir / "feedback.jsonl").write_text(
                json.dumps({"job_id": "job-1", "candidate_id": "c1", "selected": True}),
                encoding="utf-8",
            )
            (debug_dir / "candidates_normalized.jsonl").write_text(
                json.dumps({"id": "c1", "start": 1.0, "end": 2.0}),
                encoding="utf-8",
            )
            (job_dir / "request.json").write_text(
                json.dumps({"url": "https://www.twitch.tv/videos/1234567890"}),
                encoding="utf-8",
            )

            payload = build_gold_eval_manifest.build_manifest(storage)

        self.assertEqual(len(payload["resolved"]), 1)
        self.assertEqual(payload["resolved"][0]["job_id"], "job-1")
        self.assertEqual(payload["resolved"][0]["platform"], "twitch")
        self.assertEqual(payload["resolved"][0]["vod_id"], "1234567890")
        self.assertEqual(payload["resolved"][0]["selected_count"], 1)
        self.assertFalse(payload["unresolved"])

    def test_build_manifest_uses_job_map_for_missing_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            job_dir = storage / "job-2"
            debug_dir = job_dir / "debug"
            debug_dir.mkdir(parents=True)
            (debug_dir / "feedback.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"job_id": "job-2", "candidate_id": "c1", "selected": True}),
                        json.dumps({"job_id": "job-2", "candidate_id": "c2", "selected": False}),
                    ]
                ),
                encoding="utf-8",
            )
            (debug_dir / "candidates_normalized.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "c1", "start": 1.0, "end": 2.0}),
                        json.dumps({"id": "c2", "start": 3.0, "end": 4.0}),
                    ]
                ),
                encoding="utf-8",
            )
            job_map = root / "job_map.csv"
            job_map.write_text(
                "job_id,vod_url,creator_id,platform,vod_id\n"
                "job-2,https://youtu.be/abcdefghijk,caseoh_,youtube,abcdefghijk\n",
                encoding="utf-8",
            )

            payload = build_gold_eval_manifest.build_manifest(storage, job_map_path=job_map)

        self.assertEqual(len(payload["resolved"]), 1)
        self.assertEqual(payload["resolved"][0]["creator_id"], "caseoh_")
        self.assertEqual(payload["resolved"][0]["platform"], "youtube")
        self.assertFalse(payload["unresolved"])

    def test_build_manifest_reports_unresolved_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            job_dir = storage / "job-3"
            debug_dir = job_dir / "debug"
            debug_dir.mkdir(parents=True)
            (debug_dir / "feedback.jsonl").write_text(
                json.dumps({"job_id": "job-3", "candidate_id": "c1", "selected": True}),
                encoding="utf-8",
            )
            (debug_dir / "candidates_normalized.jsonl").write_text(
                json.dumps({"id": "c1", "start": 1.0, "end": 2.0}),
                encoding="utf-8",
            )

            payload = build_gold_eval_manifest.build_manifest(storage)

        self.assertFalse(payload["resolved"])
        self.assertEqual(len(payload["unresolved"]), 1)
        self.assertEqual(payload["unresolved"][0]["reason"], "missing_vod_url")

    def test_main_writes_manifest_and_unresolved_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            job_dir = storage / "job-1"
            debug_dir = job_dir / "debug"
            debug_dir.mkdir(parents=True)
            (debug_dir / "feedback.jsonl").write_text(
                json.dumps({"job_id": "job-1", "candidate_id": "c1", "selected": True}),
                encoding="utf-8",
            )
            (debug_dir / "candidates_normalized.jsonl").write_text(
                json.dumps({"id": "c1", "start": 1.0, "end": 2.0}),
                encoding="utf-8",
            )
            (job_dir / "source_url.txt").write_text(
                "https://www.twitch.tv/videos/1234567890",
                encoding="utf-8",
            )
            output = root / "gold_eval_manifest.csv"
            unresolved = root / "gold_eval_manifest_unresolved.csv"

            with patch(
                "sys.argv",
                [
                    "build_gold_eval_manifest.py",
                    "--storage-dir",
                    str(storage),
                    "--output",
                    str(output),
                    "--unresolved-out",
                    str(unresolved),
                ],
            ):
                rc = build_gold_eval_manifest.main()

            self.assertEqual(rc, 0)
            resolved_rows = list(csv.DictReader(output.open("r", encoding="utf-8", newline="")))
            unresolved_rows = unresolved.read_text(encoding="utf-8")

        self.assertEqual(len(resolved_rows), 1)
        self.assertEqual(resolved_rows[0]["job_id"], "job-1")
        self.assertEqual(unresolved_rows, "")


if __name__ == "__main__":
    unittest.main()
