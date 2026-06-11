import pathlib
import tempfile
import time
import unittest

import job_manager
import session_lease
from models import JobStatus


class JobSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_jobs_file = job_manager._jobs_file
        self._orig_jobs_file_mtime_ns = job_manager._jobs_file_mtime_ns
        self._orig_time_source = session_lease._time_source
        self._tmp = tempfile.TemporaryDirectory()

        job_manager._jobs.clear()
        job_manager._job_created_at.clear()
        job_manager._cancelled_jobs.clear()
        session_lease._job_session_id.clear()
        session_lease._job_session_expires_at.clear()
        # Persist-Schreibvorgänge in ein Wegwerf-Verzeichnis lenken, damit der Test
        # hermetisch bleibt und nicht die echte jobs.json berührt.
        job_manager._jobs_file = pathlib.Path(self._tmp.name) / "jobs.json"
        job_manager._jobs_file_mtime_ns = 0

    def tearDown(self) -> None:
        job_manager._jobs.clear()
        job_manager._job_created_at.clear()
        job_manager._cancelled_jobs.clear()
        session_lease._job_session_id.clear()
        session_lease._job_session_expires_at.clear()
        session_lease._time_source = self._orig_time_source
        job_manager._jobs_file = self._orig_jobs_file
        job_manager._jobs_file_mtime_ns = self._orig_jobs_file_mtime_ns
        self._tmp.cleanup()

    def test_is_job_cancelled_is_pure_query_before_explicit_drive(self) -> None:
        # Reine Query: auf einem laufenden Job mit abgelaufener Lease — aber VOR dem
        # expliziten Request-Pfad-Drive — darf is_job_cancelled weder den Status
        # ändern noch persistieren.
        clock = {"t": 1000.0}
        session_lease._time_source = lambda: clock["t"]
        job_manager._jobs["job-1"] = JobStatus(job_id="job-1", status="downloading", progress_pct=10)
        job_manager._job_created_at["job-1"] = time.monotonic()
        job_manager.bind_job_session("job-1", "sess-1")
        clock["t"] = 1000.0 + session_lease._SESSION_TTL_SECONDS + 1

        cancelled = job_manager.is_job_cancelled("job-1")

        self.assertFalse(cancelled)
        self.assertEqual(job_manager._jobs["job-1"].status, "downloading")
        self.assertFalse(job_manager._jobs_file.exists())  # nichts persistiert

    def test_session_expiry_cancels_running_job_via_explicit_drive(self) -> None:
        # Ende-zu-Ende: eine abgelaufene, gebundene Lease eines laufenden Jobs führt
        # über den expliziten Request-Pfad-Drive (session_lease.drive_expiry →
        # job_manager.cancel_job) zu status="cancelled" mit der session-ended-Meldung.
        clock = {"t": 1000.0}
        session_lease._time_source = lambda: clock["t"]
        job_manager._jobs["job-1"] = JobStatus(job_id="job-1", status="downloading", progress_pct=10)
        job_manager._job_created_at["job-1"] = time.monotonic()
        job_manager.bind_job_session("job-1", "sess-1")
        clock["t"] = 1000.0 + session_lease._SESSION_TTL_SECONDS + 1

        drove = session_lease.drive_expiry("job-1")

        self.assertTrue(drove)
        self.assertEqual(job_manager._jobs["job-1"].status, "cancelled")
        self.assertEqual(
            job_manager._jobs["job-1"].error,
            "Job cancelled because the client session ended",
        )

    def test_drive_expiry_noop_when_lease_not_expired(self) -> None:
        clock = {"t": 1000.0}
        session_lease._time_source = lambda: clock["t"]
        job_manager._jobs["job-1"] = JobStatus(job_id="job-1", status="downloading", progress_pct=10)
        job_manager._job_created_at["job-1"] = time.monotonic()
        job_manager.bind_job_session("job-1", "sess-1")

        self.assertFalse(session_lease.drive_expiry("job-1"))
        self.assertEqual(job_manager._jobs["job-1"].status, "downloading")

    def test_cancel_job_reason_distinguishes_manual_and_session_cancel(self) -> None:
        # Beide Cancellations fließen durch denselben Pfad (cancel_job), bleiben aber
        # über ihre Meldung unterscheidbar.
        for job_id in ("manual", "session"):
            job_manager._jobs[job_id] = JobStatus(job_id=job_id, status="downloading", progress_pct=10)
            job_manager._job_created_at[job_id] = time.monotonic()

        job_manager.cancel_job("manual")
        job_manager.cancel_job("session", reason="Job cancelled because the client session ended")

        self.assertEqual(job_manager._jobs["manual"].error, "Job cancelled by client")
        self.assertEqual(
            job_manager._jobs["session"].error,
            "Job cancelled because the client session ended",
        )

    def test_touch_job_session_renews_matching_session(self) -> None:
        clock = {"t": 1000.0}
        session_lease._time_source = lambda: clock["t"]
        job_manager._jobs["job-1"] = JobStatus(job_id="job-1", status="downloading", progress_pct=10)
        job_manager._job_created_at["job-1"] = time.monotonic()
        job_manager.bind_job_session("job-1", "sess-1")
        expires_before = job_manager._job_session_expires_at["job-1"]

        clock["t"] = 1050.0
        job_manager.touch_job_session("job-1", "sess-1")

        self.assertGreater(job_manager._job_session_expires_at["job-1"], expires_before)


if __name__ == "__main__":
    unittest.main()
