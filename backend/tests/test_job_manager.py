import json
import tempfile
import time
import unittest
from pathlib import Path

import job_manager


class JobManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_jobs = dict(job_manager._jobs)
        self._orig_created_at = dict(job_manager._job_created_at)
        self._orig_jobs_file = job_manager._jobs_file
        self._orig_last_persisted_at = job_manager._last_persisted_at
        self._tmp = tempfile.TemporaryDirectory()

        job_manager._jobs.clear()
        job_manager._job_created_at.clear()
        job_manager._jobs_file = Path(self._tmp.name) / "jobs.json"
        job_manager._last_persisted_at = 0.0

    def tearDown(self) -> None:
        job_manager._jobs.clear()
        job_manager._jobs.update(self._orig_jobs)
        job_manager._job_created_at.clear()
        job_manager._job_created_at.update(self._orig_created_at)
        job_manager._jobs_file = self._orig_jobs_file
        job_manager._last_persisted_at = self._orig_last_persisted_at
        self._tmp.cleanup()

    def test_progress_updates_stay_in_memory_until_persist_boundary(self) -> None:
        job_manager.create_job("job-1", profile="general")

        queued_payload = json.loads(job_manager._jobs_file.read_text(encoding="utf-8"))
        self.assertEqual(queued_payload[0]["status"], "queued")

        job_manager.update_job("job-1", status="downloading", progress_pct=10)

        stored_payload = json.loads(job_manager._jobs_file.read_text(encoding="utf-8"))
        self.assertEqual(stored_payload[0]["status"], "queued")
        self.assertEqual(job_manager.get_job("job-1").status, "downloading")

        job_manager.update_job("job-1", status="done", progress_pct=100)

        final_payload = json.loads(job_manager._jobs_file.read_text(encoding="utf-8"))
        self.assertEqual(final_payload[0]["status"], "done")

    def test_get_job_evicts_expired_jobs_and_persists_cleanup(self) -> None:
        job_manager.create_job("job-expired", profile="general")
        job_manager._job_created_at["job-expired"] = time.monotonic() - job_manager._JOB_TTL_SECONDS - 1

        self.assertIsNone(job_manager.get_job("job-expired"))
        self.assertEqual(
            json.loads(job_manager._jobs_file.read_text(encoding="utf-8")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
