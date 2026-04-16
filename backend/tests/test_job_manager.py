import json
import tempfile
import time
import unittest
from pathlib import Path

import job_manager
from models import JobStatus


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

    def test_get_job_evicts_expired_terminal_jobs_and_persists_cleanup(self) -> None:
        job_manager.create_job("job-expired", profile="general")
        job_manager.update_job("job-expired", status="done", progress_pct=100)
        job_manager._job_created_at["job-expired"] = time.monotonic() - job_manager._JOB_TTL_SECONDS - 1

        self.assertIsNone(job_manager.get_job("job-expired"))
        self.assertEqual(
            json.loads(job_manager._jobs_file.read_text(encoding="utf-8")),
            [],
        )

    def test_active_jobs_are_not_evicted_after_original_ttl(self) -> None:
        job_manager.create_job("job-running", profile="general")
        job_manager._job_created_at["job-running"] = time.monotonic() - job_manager._JOB_TTL_SECONDS - 1

        job_manager.update_job("job-running", status="transcribing", progress_pct=40)

        job = job_manager.get_job("job-running")
        self.assertIsNotNone(job)
        assert isinstance(job, JobStatus)
        self.assertEqual(job.status, "transcribing")

    def test_terminal_transition_refreshes_retention_window_for_long_running_job(self) -> None:
        job_manager.create_job("job-finished", profile="general")
        job_manager._job_created_at["job-finished"] = time.monotonic() - job_manager._JOB_TTL_SECONDS - 1

        job_manager.update_job("job-finished", status="done", progress_pct=100)

        job = job_manager.get_job("job-finished")
        self.assertIsNotNone(job)
        assert isinstance(job, JobStatus)
        self.assertEqual(job.status, "done")

    def test_load_jobs_accepts_legacy_float_download_eta(self) -> None:
        payload = [
            {
                "job_id": "job-eta",
                "status": "downloading",
                "progress_pct": 12,
                "profile": "general",
                "clips": [],
                "error": None,
                "download_speed": 1024.5,
                "download_eta": 715.6143084933422,
                "download_cached": False,
            }
        ]
        job_manager._jobs_file.write_text(json.dumps(payload), encoding="utf-8")

        job_manager._jobs.clear()
        job_manager._job_created_at.clear()
        job_manager._load_jobs()

        job = job_manager.get_job("job-eta")
        self.assertIsNotNone(job)
        assert isinstance(job, JobStatus)
        self.assertEqual(job.download_eta, 716)

    def test_missing_job_updates_do_not_log_warning_noise(self) -> None:
        with self.assertNoLogs(job_manager.logger, level="WARNING"):
            job_manager.update_job("missing-job", status="downloading", progress_pct=10)


if __name__ == "__main__":
    unittest.main()
