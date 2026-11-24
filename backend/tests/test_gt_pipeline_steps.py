"""gt_pipeline step logic: matching/eval_manifest are only remembered as done
once repost downloads and VOD transcripts are complete (#129).

Steps 4+5 run against a real run dir + state file; only the matcher is
injected (and the canonical manifest is redirected into the temp dir).
"""
import pathlib
import tempfile
import unittest
from unittest import mock

from ml import gt_pipeline
from ml.row_identity import read_jsonl


def _match_row(repost: dict, start: float) -> dict:
    return {
        "vod_id": "v1",
        "source_url": repost["source_url"],
        "repost_title": repost["clip_id"],
        "views": 100,
        "window_start": start,
        "window_end": start + 20.0,
        "transcript_similarity": 0.9,
        "transcript_match_kind": "single_window",
    }


class _FakeMatcher:
    """Matches every repost it is given, each at a distinct VOD position."""

    def __init__(self) -> None:
        self.calls = 0

    def match_reposts_to_vods_with_diagnostics(self, vod_manifest, reposts, **_kw):
        self.calls += 1
        rows = [_match_row(r, 1000.0 * (i + 1)) for i, r in enumerate(reposts)]
        return rows, {}


def _repost(n: int) -> dict:
    return {"clip_id": f"c{n}", "source_url": f"https://youtu.be/c{n}", "video_path": f"/x/c{n}.mp4"}


class GtPipelineStepCachingTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        (self.root / "ml" / "artifacts").mkdir(parents=True)
        self.matcher = _FakeMatcher()
        for patch in (
            mock.patch.object(gt_pipeline, "_matcher", self.matcher),
            mock.patch.object(gt_pipeline, "BACKEND_ROOT", self.root),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _run_steps_4_5(self, *, reposts: list[dict], inputs_complete: bool) -> list[dict]:
        """One pipeline run of matching + eval_manifest, resuming from disk state."""
        state = gt_pipeline._load_state(self.run_dir)
        if inputs_complete:
            gt_pipeline._mark_done(state, "iboclipss_downloaded")
            gt_pipeline._mark_done(state, "vod_transcripts_all")
        matches = gt_pipeline.step_run_matcher(
            self.run_dir, state, vod_manifest=[{"vod_id": "v1"}], reposts=reposts,
            min_similarity=0.5, vod_cache_root=self.root / "cache", match_workers=1,
        )
        gt_pipeline._save_state(self.run_dir, state)
        gt_pipeline.step_build_eval_manifest(self.run_dir, state, matches=matches, min_similarity=0.5)
        gt_pipeline._save_state(self.run_dir, state)
        return matches

    def _manifest_urls(self) -> set[str]:
        rows = read_jsonl(self.run_dir / "eval_manifest.jsonl", strict=False)
        return {m["source_url"] for r in rows for m in r["gt_moments"]}

    def test_partial_run_does_not_mark_matching_or_eval_manifest_done(self) -> None:
        self._run_steps_4_5(reposts=[_repost(1)], inputs_complete=False)

        state = gt_pipeline._load_state(self.run_dir)
        self.assertFalse(gt_pipeline._is_done(state, "matching"))
        self.assertFalse(gt_pipeline._is_done(state, "eval_manifest"))

    def test_follow_up_run_after_missing_data_arrives_rematches(self) -> None:
        self._run_steps_4_5(reposts=[_repost(1)], inputs_complete=False)

        matches = self._run_steps_4_5(reposts=[_repost(1), _repost(2)], inputs_complete=True)

        self.assertEqual(self.matcher.calls, 2)
        self.assertEqual({m["source_url"] for m in matches},
                         {"https://youtu.be/c1", "https://youtu.be/c2"})
        self.assertEqual(self._manifest_urls(), {"https://youtu.be/c1", "https://youtu.be/c2"})

    def test_complete_run_caches_matching_and_eval_manifest(self) -> None:
        self._run_steps_4_5(reposts=[_repost(1)], inputs_complete=True)

        state = gt_pipeline._load_state(self.run_dir)
        self.assertTrue(gt_pipeline._is_done(state, "matching"))
        self.assertTrue(gt_pipeline._is_done(state, "eval_manifest"))

        self._run_steps_4_5(reposts=[_repost(1), _repost(2)], inputs_complete=True)
        self.assertEqual(self.matcher.calls, 1)
        self.assertEqual(self._manifest_urls(), {"https://youtu.be/c1"})


if __name__ == "__main__":
    unittest.main()
