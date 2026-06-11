import unittest
from typing import get_args

import job_lifecycle
from models import JobState


# The terminal statuses, written out independently of the module under test so
# the test pins the vocabulary rather than echoing the implementation.
_TERMINAL = {"done", "failed", "cancelled"}


class JobLifecycleTests(unittest.TestCase):
    def test_terminal_set_is_exactly_the_finished_statuses(self) -> None:
        self.assertEqual(set(job_lifecycle.TERMINAL), _TERMINAL)

    def test_terminal_is_a_frozenset(self) -> None:
        # An immutable single source of truth — callers must not mutate it.
        self.assertIsInstance(job_lifecycle.TERMINAL, frozenset)

    def test_is_terminal_true_for_each_terminal_status(self) -> None:
        for status in _TERMINAL:
            with self.subTest(status=status):
                self.assertTrue(job_lifecycle.is_terminal(status))

    def test_is_terminal_exhaustive_over_every_job_state(self) -> None:
        # Exhaustive across the whole JobState vocabulary: every declared status
        # is terminal iff it is in the terminal set, no status left untested.
        states = get_args(JobState)
        self.assertTrue(states, "JobState should declare at least one status")
        for status in states:
            with self.subTest(status=status):
                self.assertEqual(job_lifecycle.is_terminal(status), status in _TERMINAL)

    def test_is_terminal_false_for_unknown_status(self) -> None:
        self.assertFalse(job_lifecycle.is_terminal("totally-not-a-status"))

    def test_named_constants_agree_with_terminal_set(self) -> None:
        self.assertEqual(
            {job_lifecycle.DONE, job_lifecycle.FAILED, job_lifecycle.CANCELLED},
            _TERMINAL,
        )


if __name__ == "__main__":
    unittest.main()
