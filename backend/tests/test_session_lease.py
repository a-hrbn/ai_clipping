import time
import unittest

import job_manager
import session_lease
from models import JobStatus


class SessionLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        job_manager._jobs.clear()
        job_manager._job_created_at.clear()
        job_manager._cancelled_jobs.clear()
        session_lease._job_session_id.clear()
        session_lease._job_session_expires_at.clear()
        self._orig_time_source = session_lease._time_source

    def tearDown(self) -> None:
        job_manager._jobs.clear()
        job_manager._job_created_at.clear()
        job_manager._cancelled_jobs.clear()
        session_lease._job_session_id.clear()
        session_lease._job_session_expires_at.clear()
        session_lease._time_source = self._orig_time_source

    def test_job_manager_reexports_lease_state_by_reference(self) -> None:
        # Single source of truth: job_manager aliases session_lease's dicts so the
        # lease state has one owner. Mutating one is observable through the other.
        self.assertIs(job_manager._job_session_id, session_lease._job_session_id)
        self.assertIs(job_manager._job_session_expires_at, session_lease._job_session_expires_at)

    def test_bind_and_touch_use_injectable_time_source(self) -> None:
        clock = {"t": 1000.0}
        session_lease._time_source = lambda: clock["t"]

        session_lease.bind("job-1", "sess-1")
        self.assertEqual(
            session_lease._job_session_expires_at["job-1"],
            1000.0 + session_lease._SESSION_TTL_SECONDS,
        )

        clock["t"] = 1050.0
        self.assertTrue(session_lease.touch("job-1", "sess-1"))
        self.assertEqual(
            session_lease._job_session_expires_at["job-1"],
            1050.0 + session_lease._SESSION_TTL_SECONDS,
        )

    def test_touch_ignores_mismatched_session(self) -> None:
        session_lease.bind("job-1", "sess-1")
        before = session_lease._job_session_expires_at["job-1"]

        self.assertFalse(session_lease.touch("job-1", "other-session"))
        self.assertEqual(session_lease._job_session_expires_at["job-1"], before)

    def test_is_expired_via_injected_clock(self) -> None:
        clock = {"t": 1000.0}
        session_lease._time_source = lambda: clock["t"]
        session_lease.bind("job-1", "sess-1")

        self.assertFalse(session_lease.is_expired("job-1"))

        clock["t"] = 1000.0 + session_lease._SESSION_TTL_SECONDS + 1
        self.assertTrue(session_lease.is_expired("job-1"))

    def test_is_expired_false_without_bound_session(self) -> None:
        self.assertFalse(session_lease.is_expired("unbound-job"))

    def test_clear_removes_lease_entries(self) -> None:
        session_lease.bind("job-1", "sess-1")

        session_lease.clear("job-1")

        self.assertNotIn("job-1", session_lease._job_session_id)
        self.assertNotIn("job-1", session_lease._job_session_expires_at)

    def test_eviction_clears_lease_via_hook(self) -> None:
        # An evicted terminal job must leave no orphaned lease entry behind.
        job_manager._jobs["job-1"] = JobStatus(job_id="job-1", status="done", progress_pct=100)
        job_manager._job_created_at["job-1"] = time.monotonic() - job_manager._JOB_TTL_SECONDS - 1
        session_lease.bind("job-1", "sess-1")
        self.assertIn("job-1", session_lease._job_session_id)

        job_manager._evict_expired()

        self.assertNotIn("job-1", session_lease._job_session_id)
        self.assertNotIn("job-1", session_lease._job_session_expires_at)


if __name__ == "__main__":
    unittest.main()
