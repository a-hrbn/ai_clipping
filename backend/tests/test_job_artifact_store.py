"""Contract tests for the Job-Artifact-Store (PRD #39, tracer #42).

The store is the single owner of the job-artifact *names* and their
json/jsonl/text I/O. Two adapters make the seam real: ``RealFsJobArtifactStore``
(prod, under ``storage_dir/job_id/...``) and ``InMemoryJobArtifactStore`` (the
"DebugSink as a port" fake from docs/design/clip-pipeline-producers.md). Callers
pass only the ``job_id`` and an artifact role — never a path.

These tests assert the seam externally: a write through a role returns the same
value when read back through that role, and the two adapters agree. They never
look at on-disk layout — that knowledge belongs to the store alone.
"""
import json
import tempfile
import unittest
from pathlib import Path

from services.job_artifact_store import (
    InMemoryJobArtifactStore,
    RealFsJobArtifactStore,
)

_JOB = "job-1"

_JSON_ROLES = ("stage", "usage", "final_moments", "request_metadata", "progress")
_JSONL_ROLES = ("segments", "moments", "raw_moments", "feedback", "review_candidates")
_TEXT_ROLES = ("prompt", "response")


def _both_stores():
    """Yield (label, store) for both adapters; RealFs gets its own tmp dir."""
    mem = InMemoryJobArtifactStore()
    yield "in_memory", mem, None
    tmp = tempfile.TemporaryDirectory()
    yield "real_fs", RealFsJobArtifactStore(tmp.name), tmp


class RoundTripTests(unittest.TestCase):
    def test_json_roles_round_trip_on_both_adapters(self) -> None:
        payload = {"status": "ok", "count": 3, "nested": {"a": [1, 2]}}
        for label, store, tmp in _both_stores():
            with self.subTest(adapter=label):
                for role in _JSON_ROLES:
                    getattr(store, f"write_{role}")(_JOB, payload)
                    self.assertEqual(getattr(store, f"read_{role}")(_JOB), payload)
            if tmp:
                tmp.cleanup()

    def test_jsonl_roles_round_trip_on_both_adapters(self) -> None:
        rows = [{"rank": 1, "start_time": 0.0}, {"rank": 2, "start_time": 5.0}]
        for label, store, tmp in _both_stores():
            with self.subTest(adapter=label):
                for role in _JSONL_ROLES:
                    getattr(store, f"write_{role}")(_JOB, rows)
                    self.assertEqual(getattr(store, f"read_{role}")(_JOB), rows)
            if tmp:
                tmp.cleanup()

    def test_text_roles_round_trip_on_both_adapters(self) -> None:
        text = "line one\nline two\n"
        for label, store, tmp in _both_stores():
            with self.subTest(adapter=label):
                for role in _TEXT_ROLES:
                    getattr(store, f"write_{role}")(_JOB, text)
                    self.assertEqual(getattr(store, f"read_{role}")(_JOB), text)
            if tmp:
                tmp.cleanup()


class MissingArtifactTests(unittest.TestCase):
    def test_missing_json_returns_none(self) -> None:
        for label, store, tmp in _both_stores():
            with self.subTest(adapter=label):
                self.assertIsNone(store.read_stage(_JOB))
                self.assertIsNone(store.read_final_moments(_JOB))
            if tmp:
                tmp.cleanup()

    def test_missing_jsonl_returns_empty_list(self) -> None:
        for label, store, tmp in _both_stores():
            with self.subTest(adapter=label):
                self.assertEqual(store.read_segments(_JOB), [])
                self.assertEqual(store.read_moments(_JOB), [])
            if tmp:
                tmp.cleanup()

    def test_missing_text_returns_none(self) -> None:
        for label, store, tmp in _both_stores():
            with self.subTest(adapter=label):
                self.assertIsNone(store.read_prompt(_JOB))
                self.assertIsNone(store.read_response(_JOB))
            if tmp:
                tmp.cleanup()

    def test_malformed_json_reads_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RealFsJobArtifactStore(tmp)
            path = Path(tmp) / _JOB / "debug" / "llm_moment_finder_stage.json"
            path.parent.mkdir(parents=True)
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(store.read_stage(_JOB))


class DebugPresenceTests(unittest.TestCase):
    def test_debug_artifacts_present_tracks_debug_writes_only(self) -> None:
        for label, store, tmp in _both_stores():
            with self.subTest(adapter=label):
                self.assertFalse(store.debug_artifacts_present(_JOB))
                # request.json lives at the job root, not under debug/
                store.write_request_metadata(_JOB, {"url": "x"})
                self.assertFalse(store.debug_artifacts_present(_JOB))
                store.write_segments(_JOB, [{"start": 0.0}])
                self.assertTrue(store.debug_artifacts_present(_JOB))
            if tmp:
                tmp.cleanup()

    def test_real_fs_last_modified_is_iso_after_write_none_before(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RealFsJobArtifactStore(tmp)
            self.assertIsNone(store.debug_last_modified(_JOB))
            store.write_segments(_JOB, [{"start": 0.0}])
            stamp = store.debug_last_modified(_JOB)
            self.assertIsInstance(stamp, str)
            self.assertIn("T", stamp)

    def test_in_memory_last_modified_is_none(self) -> None:
        store = InMemoryJobArtifactStore()
        store.write_segments(_JOB, [{"start": 0.0}])
        self.assertIsNone(store.debug_last_modified(_JOB))


class RegistryAndCompatTests(unittest.TestCase):
    def test_jobs_registry_path_points_at_storage_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RealFsJobArtifactStore(tmp)
            self.assertEqual(store.jobs_registry_path(), Path(tmp) / "jobs.json")

    def test_in_memory_has_no_jobs_registry_path(self) -> None:
        with self.assertRaises(NotImplementedError):
            InMemoryJobArtifactStore().jobs_registry_path()

    def test_real_fs_reads_producer_written_jsonl_without_trailing_newline(self) -> None:
        # Producers write jsonl as "\n".join(...) with no trailing newline; the
        # store's reader must parse that faithfully (read-compat for the diagnostics
        # reader, which now reads producer-written artifacts through the store).
        with tempfile.TemporaryDirectory() as tmp:
            store = RealFsJobArtifactStore(tmp)
            path = Path(tmp) / _JOB / "debug" / "llm_moment_finder_moments.jsonl"
            path.parent.mkdir(parents=True)
            rows = [{"rank": 1}, {"rank": 2}, {"rank": 3}]
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=True) for r in rows),
                encoding="utf-8",
            )
            self.assertEqual(store.read_moments(_JOB), rows)


if __name__ == "__main__":
    unittest.main()
