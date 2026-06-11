"""Reader-migration tests for routers.pipeline_debug (PRD #39, tracer #44).

The training/debug router now reads its *job* artifacts (progress.json, the
``llm_moment_finder_*`` dumps, segments.jsonl) through the Job-Artifact-Store
instead of building paths and decoding jsonl itself. These tests seed the
in-memory fake store and assert the readers cross that seam — no tmp directory.

Workflow-level files (workflow.json, the review labels/events) are NOT job
artifacts and keep their own I/O; they are out of scope here.
"""
import asyncio
import unittest
from unittest.mock import patch

import pytest

# fastapi is a base dependency; skip cleanly when running without it installed.
pytest.importorskip("fastapi")

from routers import pipeline_debug as pd
from services.job_artifact_store import InMemoryJobArtifactStore


class ReadJobProgressTests(unittest.TestCase):
    def test_reads_progress_through_store(self) -> None:
        store = InMemoryJobArtifactStore()
        store.write_progress("job-1", {"status": "transcribing", "progress_pct": 42})

        self.assertEqual(pd._read_job_progress("job-1", store=store), ("transcribing", 42))

    def test_returns_none_when_progress_absent(self) -> None:
        store = InMemoryJobArtifactStore()
        self.assertIsNone(pd._read_job_progress("job-1", store=store))

    def test_returns_none_when_status_missing(self) -> None:
        store = InMemoryJobArtifactStore()
        store.write_progress("job-1", {"progress_pct": 10})
        self.assertIsNone(pd._read_job_progress("job-1", store=store))


class BuildReviewRowsTests(unittest.TestCase):
    def test_prefers_raw_moments_read_through_store(self) -> None:
        store = InMemoryJobArtifactStore()
        store.write_segments(
            "job-1", [{"start": 9.0, "end": 41.0, "text": "viral clip text"}]
        )
        store.write_raw_moments(
            "job-1",
            [{"rank": 1, "start_time": 10.0, "end_time": 40.0, "viral_score": 0.9, "reason": "hype"}],
        )
        # A filtered set exists too, but raw is preferred.
        store.write_moments(
            "job-1", [{"rank": 1, "start_time": 11.0, "end_time": 39.0}]
        )

        rows, raw_count = pd._build_review_rows(
            workflow_id="wf1", vod_id="123", job_id="job-1", store=store
        )

        self.assertEqual(raw_count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["start_time"], 10.0)
        self.assertEqual(rows[0]["end_time"], 40.0)
        # No clips on disk for the in-memory fake → unresolved clip.
        self.assertEqual(rows[0]["clip_status"], "missing")

    def test_falls_back_to_filtered_when_raw_absent(self) -> None:
        store = InMemoryJobArtifactStore()
        store.write_segments("job-1", [{"start": 0.0, "end": 5.0, "text": "hi"}])
        store.write_moments(
            "job-1", [{"rank": 1, "start_time": 1.0, "end_time": 4.0}]
        )

        rows, raw_count = pd._build_review_rows(
            workflow_id="wf1", vod_id="123", job_id="job-1", store=store
        )

        self.assertEqual(raw_count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["start_time"], 1.0)


class ComputeEvalTests(unittest.TestCase):
    def test_reads_predictions_through_store(self) -> None:
        store = InMemoryJobArtifactStore()
        store.write_raw_moments(
            "pipeline_debug_wf1_123",
            [{"rank": 1, "start_time": 10.0, "end_time": 40.0}],
        )
        status = {"workflow_id": "wf1", "status": "done", "request": {"vod_ids": ["123"]}}
        captured: dict = {}

        def fake_evaluate_all(predictions_by_vod, **kwargs):
            captured["predictions"] = predictions_by_vod
            return {"overall": {}, "per_vod": []}

        with patch("ml.eval_clip_finder.evaluate_all", side_effect=fake_evaluate_all), \
             patch("ml.eval_clip_finder.load_manifest", return_value={}):
            result = pd._compute_eval(status, store=store)

        self.assertEqual(result, {"overall": {}, "per_vod": []})
        self.assertIn("123", captured["predictions"])
        self.assertEqual(captured["predictions"]["123"][0]["start_time"], 10.0)


class RunPromptEndpointTests(unittest.TestCase):
    def test_reads_prompt_through_store(self) -> None:
        store = InMemoryJobArtifactStore()
        store.write_prompt("pipeline_debug_wf1abc_123", "HELLO PROMPT")

        with patch.object(pd, "_job_store", return_value=store):
            result = asyncio.run(pd.get_pipeline_debug_run_prompt("wf1abc", vod_id="123"))

        self.assertEqual(result, {"vod_id": "123", "prompt": "HELLO PROMPT", "exists": True})

    def test_reports_absent_prompt_through_store(self) -> None:
        store = InMemoryJobArtifactStore()

        with patch.object(pd, "_job_store", return_value=store):
            result = asyncio.run(pd.get_pipeline_debug_run_prompt("wf1abc", vod_id="123"))

        self.assertEqual(result, {"vod_id": "123", "prompt": "", "exists": False})


if __name__ == "__main__":
    unittest.main()
