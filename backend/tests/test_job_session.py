import time
import unittest

import job_manager
from models import JobStatus


class JobSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        job_manager._jobs.clear()
        job_manager._job_created_at.clear()
        job_manager._cancelled_jobs.clear()
        job_manager._job_session_id.clear()
        job_manager._job_session_expires_at.clear()

    def tearDown(self) -> None:
        job_manager._jobs.clear()
        job_manager._job_created_at.clear()
        job_manager._cancelled_jobs.clear()
        job_manager._job_session_id.clear()
        job_manager._job_session_expires_at.clear()

    def test_session_expiry_cancels_running_job(self) -> None:
        job_manager._jobs["job-1"] = JobStatus(job_id="job-1", status="downloading", progress_pct=10)
        job_manager._job_created_at["job-1"] = time.monotonic()
        job_manager.bind_job_session("job-1", "sess-1")
        job_manager._job_session_expires_at["job-1"] = time.monotonic() - 1

        cancelled = job_manager.is_job_cancelled("job-1")

        self.assertTrue(cancelled)
        self.assertEqual(job_manager._jobs["job-1"].status, "cancelled")
        self.assertEqual(job_manager._jobs["job-1"].error, "Job cancelled because the client session ended")

    def test_touch_job_session_renews_matching_session(self) -> None:
        job_manager._jobs["job-1"] = JobStatus(job_id="job-1", status="downloading", progress_pct=10)
        job_manager._job_created_at["job-1"] = time.monotonic()
        job_manager.bind_job_session("job-1", "sess-1")
        expires_before = job_manager._job_session_expires_at["job-1"]

        job_manager.touch_job_session("job-1", "sess-1")

        self.assertGreater(job_manager._job_session_expires_at["job-1"], expires_before)
