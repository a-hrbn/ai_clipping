"""Slice 3 (#105) — judge_recall_baseline.json: the production-ranker regression baseline.

Go-Live (#105 / ADR 0001) makes the Archetype-Judge the production ranker. Future serving
changes prove their recall delta against the judge, not the retired composite — so the judge
gets its OWN frozen baseline, ``judge_recall_baseline.json``.

It is deliberately a *separate* artifact from the composite ``finder_recall_baseline.json``:
that file stays the input the gap_decomposition / oof drift-gate framework re-mints and pins,
so re-pinning it in place would break those guards. Same gold-only ``directional_eval`` shape,
different ranker. These tests pin:

  * the net-free re-pin mechanism (``gold_baseline_from_verdict``): a gold-only
    ``directional_eval`` of Arm B's judged predictions, scored without a paid call;
  * the production regression guard: replaying the persisted Arm B raws (the canonical paid
    run) reproduces the committed judge baseline — so the artifact tracks the judge, and a
    serving change that moves the judge's recall is caught here.
"""
from __future__ import annotations

import json
import sys
import types
import unittest

# Defensive: the moment_finder package reaches RichSegment, which can pull the heavy
# transcription stack. Prior art (test_replay_harness) stubs whisper so the import stays
# cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import archetype_judge_retest as judge_retest
from ml import eval_clip_finder

_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_JUDGE_BASELINE_PATH = eval_clip_finder.BACKEND_ROOT / "ml" / "artifacts" / "judge_recall_baseline.json"
_ARM_B_DIR = _STORAGE_DIR / judge_retest.DEFAULT_ARTIFACTS_SUBDIR / "arm_b"


def _real_judge_artifacts_present() -> bool:
    """The guard needs the committed judge baseline, the replayable pools, AND the persisted
    Arm B raws (all gitignored bar the baseline) — so it runs only where the paid-run
    artifacts live, and skips in CI."""
    return (
        _JUDGE_BASELINE_PATH.exists()
        and any(_STORAGE_DIR.glob("pipeline_debug_*"))
        and any(_ARM_B_DIR.glob("*.raw.txt"))
    )


def _pred(start: float, end: float, score: float) -> dict:
    return {"start_time": start, "end_time": end, "composite_score": score,
            "viral_score": score, "rank": 1}


class GoldBaselineFromVerdictTests(unittest.TestCase):
    """Net-free (#105 AC): the re-pin mechanism re-scores a verdict over the GOLD tier
    only — the baseline's shape — with no replay and no paid call."""

    def test_scores_the_gold_tier_only(self) -> None:
        # V carries a gold moment Arm B's prediction covers; S carries only a silver one.
        # The gold-only baseline counts V and excludes S entirely.
        manifest = {
            "V": [{"start": 0.0, "end": 10.0, "views": 300_000, "gt_tier": "gold"}],
            "S": [{"start": 0.0, "end": 10.0, "views": 300_000, "gt_tier": "silver"}],
        }
        hit = [_pred(0.0, 10.0, 9.0)]
        verdict = {"arm_b": {
            "final_by_vod": {"V": hit, "S": hit},
            "raw_by_vod": {"V": hit, "S": hit},
        }}

        result = judge_retest.gold_baseline_from_verdict(verdict, manifest, ["V", "S"])

        self.assertEqual(result["gold_tiers"], ["gold"])
        self.assertEqual(result["floor"]["gt_count"], 1)        # only V's gold; S excluded
        self.assertEqual(result["floor"]["recall"]["20"], 1.0)  # V's gold is covered


class ProductionRankerBaselineGuardTests(unittest.TestCase):
    @unittest.skipUnless(
        _real_judge_artifacts_present(),
        "real pipeline_debug pools / Arm B raws / judge baseline not present (gitignored)",
    )
    def test_judge_replay_reproduces_frozen_judge_baseline(self) -> None:
        # The Go-Live regression guard: replaying the persisted Arm B raws (the canonical
        # paid run, 400-batch) through the SAME shared serving post-processing the judge
        # uses must reproduce the committed judge baseline's gold recall (floor & headline
        # @5/@10/@20). A serving change that moves the judge's recall — or eval/serving
        # skew — breaks here.
        manifest = eval_clip_finder.load_manifest()
        anchors = eval_clip_finder.read_anchor_vod_ids()
        artifacts_root = _STORAGE_DIR / judge_retest.DEFAULT_ARTIFACTS_SUBDIR

        result = judge_retest.replay_gold_baseline(
            anchors, manifest,
            storage=_STORAGE_DIR, predictions_dir=_STORAGE_DIR, artifacts_root=artifacts_root,
        )
        baseline = json.loads(_JUDGE_BASELINE_PATH.read_text(encoding="utf-8"))

        for k in ("5", "10", "20"):
            self.assertEqual(result["floor"]["recall"][k], baseline["floor"]["recall"][k])
            self.assertEqual(result["headline"]["recall"][k], baseline["headline"]["recall"][k])
        self.assertEqual(result["scored_vods"], baseline["scored_vods"])
        self.assertEqual(result["missing_prediction_vods"], baseline["missing_prediction_vods"])


if __name__ == "__main__":
    unittest.main()
