"""Tests for the B-narrow probe data foundation (ml/b_narrow_probe.py).

The free, offline Phase-1 foundation of PRD #64: it selects the *recoverable
subset* — the llm_skipped golds the #60 strong-model window-probe separated above
the delivered competition (within-set percentile >= 0.66, the same pre-fixed cut)
— mechanically from the #60-persisted scores, never hand-picked. No paid call.

These tests pin the subset selection through the public interface on hand-built
ratings (exact percentiles in, exact subset out) — never a network or paid call.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

# Defensive: the selection reaches strong_model_window_probe -> the serving
# post-processor, which can pull the heavy transcription stack. Prior art stubs
# whisper the same way so the import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import b_narrow_probe as bnp, strong_model_window_probe as swp


def _scored(vod_id, start, end, kind, score, **extra):
    return {"vod_id": vod_id, "start": float(start), "end": float(end),
            "kind": kind, "score": score, **extra}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — select_recoverable: golds with within-set percentile >= 0.66
# ─────────────────────────────────────────────────────────────────────────────

def test_select_recoverable_keeps_only_golds_at_or_above_the_066_cut() -> None:
    # V1: a clearly-separated headline gold (beats all 3 controls -> pct 1.0) and an
    #     exogenous one (beats none -> pct 0.0). V2: a boundary floor gold that beats
    #     2 of 3 controls -> pct 0.667 >= 0.66 (recoverable). Subset = {high, boundary}.
    rating = {"vods": {
        "V1": {"scored": [
            _scored("V1", 3000, 3030, "gold", 0.9, headline=True, desc="recoverable-high"),
            _scored("V1", 6000, 6030, "gold", 0.4, headline=True, desc="exogenous"),
            _scored("V1", 100, 120, "control", 0.5, rank=1),   # gold 0.9 beats all 3 (pct 1.0),
            _scored("V1", 200, 220, "control", 0.6, rank=2),   # gold 0.4 beats none (pct 0.0)
            _scored("V1", 300, 320, "control", 0.7, rank=3),
        ]},
        "V2": {"scored": [
            _scored("V2", 8000, 8030, "gold", 0.5, headline=False, desc="recoverable-boundary"),
            _scored("V2", 400, 420, "control", 0.1, rank=1),
            _scored("V2", 500, 520, "control", 0.2, rank=2),
            _scored("V2", 600, 620, "control", 0.9, rank=3),
        ]},
    }}

    subset = bnp.select_recoverable(rating)

    assert {w["desc"] for w in subset["recoverable"]} == {"recoverable-high", "recoverable-boundary"}
    assert all(w["percentile"] >= 0.66 for w in subset["recoverable"])
    # The exogenous gold (pct 0.0) is excluded — not hand-picked, the cut decides.
    assert "exogenous" not in {w["desc"] for w in subset["recoverable"]}
    # Distinct subset VODs + size surfaced (thin-n explicit: subset of all golds).
    assert subset["vods"] == ["V1", "V2"]
    assert subset["count"] == 2
    assert subset["headline_count"] == 1          # only recoverable-high is headline
    assert subset["total_golds"] == 3             # 3 llm_skipped golds considered


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — the CLI reads the persisted #60 scores and reports the subset
# ─────────────────────────────────────────────────────────────────────────────

def test_main_reads_persisted_scores_and_reports_subset(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    # One VOD's persisted #60 scores: a recoverable gold (beats both controls) + the
    # controls. The CLI re-derives the subset from disk, net-free.
    scores = [
        {"vod_id": "V", "start": 3000.0, "end": 3030.0, "kind": "gold",
         "headline": True, "rank": None, "desc": "KSI recoverable", "score": 0.9},
        {"vod_id": "V", "start": 100.0, "end": 120.0, "kind": "control",
         "headline": None, "rank": 1, "desc": None, "score": 0.2},
        {"vod_id": "V", "start": 200.0, "end": 220.0, "kind": "control",
         "headline": None, "rank": 2, "desc": None, "score": 0.3},
    ]
    (artifacts / "V.scores.json").write_text(json.dumps(scores), encoding="utf-8")
    out_path = tmp_path / "subset.json"

    rc = bnp.main(["--artifacts-dir", str(artifacts), "--output", str(out_path), "--json"])

    assert rc == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["count"] == 1
    assert artifact["vods"] == ["V"]
    assert artifact["recoverable"][0]["desc"] == "KSI recoverable"
    assert artifact["recoverable"][0]["percentile"] == 1.0
    assert "computed_at" in artifact


# ─────────────────────────────────────────────────────────────────────────────
# Guard — reconcile the recoverable subset against the #60 0.66 cut on real scores
# ─────────────────────────────────────────────────────────────────────────────

_ARTIFACTS_DIR = swp.DEFAULT_ARTIFACTS_DIR
# The 5 distinct VODs whose golds the #60 strong-model probe separated (pct >= 0.66).
_RECOVERABLE_VODS = ["2748783535", "2752903388", "2757793134", "2758541573", "2759304833"]


def _scores_present() -> bool:
    """The (gitignored) persisted #60 window-probe scores from the paid run."""
    return _ARTIFACTS_DIR.exists() and any(_ARTIFACTS_DIR.glob("*.scores.json"))


@pytest.mark.skipif(
    not _scores_present(),
    reason="no persisted #60 window-probe scores (no paid strong-model run yet)",
)
def test_recoverable_subset_pinned_against_60_cut_on_real_scores() -> None:
    # The recoverable subset, read mechanically from the real persisted #60 scores:
    # exactly the golds the strong model separated (pct >= 0.66) — the bimodal
    # recoverable half the #64 B-narrow re-gen targets.
    rating = swp.load_persisted_rating(_ARTIFACTS_DIR)
    subset = bnp.select_recoverable(rating)

    assert subset["count"] == 5                         # ~5 recoverable golds (PRD #64)
    assert subset["headline_count"] == 4
    assert subset["vods"] == _RECOVERABLE_VODS
    assert all(w["percentile"] >= 0.66 for w in subset["recoverable"])
    # ChainsFR (the named #60 exogenous prediction, pct 0.40) is NOT recoverable.
    assert all("ChainsFR" not in (w.get("desc") or "") for w in subset["recoverable"])
    # A subset of the 17 floor llm_skipped golds — thin-n existence set, not a stat.
    assert subset["total_golds"] == 17
