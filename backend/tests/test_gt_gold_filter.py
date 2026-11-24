"""Both gold-minting paths apply the same match filter (#130).

``gt_pipeline`` (step 5) and the weekly flywheel core
(``gt_match_verify.match_and_verify``) are fed the same matcher output; the
moments that reach the manifest must be identical. Canonical rule: audio
fallback matches are kept, transcript matches need ``min_similarity``.
"""
import pathlib
import tempfile
import unittest
from unittest import mock

from ml import gt_match_verify
from ml import gt_pipeline
from ml.row_identity import read_jsonl

MIN_SIM = 0.58


def _row(start: float, *, url: str, sim: float, kind: str) -> dict:
    return {
        "vod_id": "V1",
        "window_start": start,
        "window_end": start + 20.0,
        "source_url": url,
        "repost_title": url,
        "views": 1000,
        "transcript_similarity": sim,
        "transcript_match_kind": kind,
    }


MATCHES = [
    _row(100.0, url="u-transcript", sim=0.9, kind="single_window"),
    # Low-speech clip found by audio: transcript similarity is near zero.
    _row(400.0, url="u-audio", sim=0.12, kind="audio_fingerprint_fallback"),
    # View-weighted transcript accept below the floor.
    _row(700.0, url="u-weak", sim=0.45, kind="single_window"),
]


def _moment_keys(manifest: list[dict]) -> set[tuple]:
    return {
        (row["vod_id"], m["start"], m["end"], m["source_url"])
        for row in manifest
        for m in row["gt_moments"]
    }


class SharedGoldFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        (self.root / "ml" / "artifacts").mkdir(parents=True)
        patch = mock.patch.object(gt_pipeline, "BACKEND_ROOT", self.root)
        patch.start()
        self.addCleanup(patch.stop)

    def _pipeline_gold(self) -> set[tuple]:
        run_dir = self.root / "run"
        run_dir.mkdir()
        gt_pipeline.step_build_eval_manifest(run_dir, {}, matches=[dict(r) for r in MATCHES], min_similarity=MIN_SIM)
        return _moment_keys(read_jsonl(run_dir / "eval_manifest.jsonl"))

    def _weekly_gold(self) -> set[tuple]:
        result = gt_match_verify.match_and_verify(
            [{"source_url": r["source_url"]} for r in MATCHES],
            ["V1"],
            cache_root=self.root / "captures",
            min_similarity=MIN_SIM,
            match_reposts=lambda vod_rows, reposts: [dict(r) for r in MATCHES],
            load_vod_fingerprint=lambda vod_id: None,
            verify_moment=lambda vod_id, moment, **kw: {"vod_id": vod_id, "gt_tier": "gold"},
        )
        return _moment_keys(result.gold_manifest)

    def test_both_paths_mint_the_same_gold_from_the_same_matches(self) -> None:
        expected = {
            ("V1", 100.0, 120.0, "u-transcript"),
            ("V1", 400.0, 420.0, "u-audio"),
        }
        self.assertEqual(self._pipeline_gold(), expected)
        self.assertEqual(self._weekly_gold(), expected)


if __name__ == "__main__":
    unittest.main()
