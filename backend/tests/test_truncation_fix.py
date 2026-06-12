"""Tests for the truncation_fix quick-win (#68, #56 routing-verdict floor win).

Two floor golds on VOD ``2769594798`` (98k "Darth Maul Sneakers", 43k "400k Live
Viewers") are #56 ``truncation`` hard_miss: their covering chunk hit our
``max_tokens=32000`` output cap (observed ``completion_tokens`` 31990 ≈ 32000) and
aborted mid-JSON, so the candidates were never generated. The model-independent fix
(no model swap — still ``gemini-2.5-flash-lite``) raises the production token budget
above that point so the covering chunk can finish.

These tests pin (a) the fix landing in the production generation config and (b) that
a completed chunk deterministically resolves the truncation the #56 detector saw —
hand-built fixtures, through the public interface, no network or paid call.
"""
from __future__ import annotations

import sys
import types

import pytest

# Defensive: hard_miss_decomposition reaches the serving post-processor, which can
# pull the heavy transcription stack. Prior art stubs whisper so the import stays cheap.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import eval_clip_finder, hard_miss_decomposition as hmd, llm_moment_finder as d0

# The observed truncation point on 2769594798: chunks 4 and 5 hit completion_tokens
# 31990 against our 32000 cap. The fix must request a budget comfortably above this.
_OBSERVED_TRUNCATION_TOKENS = 31990


def test_production_generation_config_requests_budget_above_truncation_point() -> None:
    # The fix lands in the production generation config: both the canonical code
    # default and the settings value request 64000 tokens — above the 31990 that
    # truncated, with headroom — so the covering chunk can complete. flash-lite
    # supports up to 64k output; model-independent (no model swap).
    from config import Settings

    assert d0.DEFAULT_MAX_TOKENS == 64000
    assert Settings().moment_finder_max_tokens == 64000
    # Above the observed truncation point — the cap was ours, not the model's.
    assert Settings().moment_finder_max_tokens > _OBSERVED_TRUNCATION_TOKENS


def _gold(start, end, views=0.0, desc=""):
    return {"start": float(start), "end": float(end), "views": float(views),
            "gt_tier": "gold", "desc": desc}


def _seg(start, end, text):
    return {"start": float(start), "end": float(end), "text": str(text)}


def _covers_gold(moments, gold) -> bool:
    """Does any parsed moment hit the gold window (#49 bidirectional is_hit)?"""
    return any(
        eval_clip_finder.is_hit(gold, {"start_time": m.start, "end_time": m.end})
        for m in moments)


# A single-chunk response (no CHUNK delimiter) the way the gemini generator emits it.
# Truncated: an earlier moment closes, then the gold's moment aborts mid-string — the
# exact shape that hit our 32000 cap. Fixed: the same chunk, the gold's moment closed.
_EARLIER = '{"start": 9000, "end": 9020, "hook": 0.5, "flow": 0.5, "value": 0.5, "trend": 0.5, "rationale": "earlier"}'
_TRUNCATED_RESPONSE = (
    '{"moments": [\n  ' + _EARLIER + ',\n'
    '  {"start": 10470, "end": 10482, "hook": 0.8, "flow": 0.8, "rationale": "the Darth Maul sneak')
_FIXED_RESPONSE = (
    '{"moments": [\n  ' + _EARLIER + ',\n'
    '  {"start": 10470, "end": 10482, "hook": 0.8, "flow": 0.8, "value": 0.8, "trend": 0.8, '
    '"rationale": "the Darth Maul sneakers bit"}\n]}')
_SEGMENTS = [
    _seg(9000, 9020, "an earlier complete moment well before the gold window"),
    _seg(10465, 10485, "lospollos asks jynxzi where does darth maul get his sneakers"),
]


def test_fix_deterministically_resolves_the_truncation_for_the_gold() -> None:
    # The Darth Maul gold (VOD 2769594798). With the OLD budget the covering chunk
    # aborted mid-moment: the #56 detector reads it as truncation AND no parsed
    # candidate covers the gold (its moment never closed). With the budget raised,
    # the SAME chunk completes: the detector clears it AND a candidate now covers the
    # gold — the truncation bucket resolves to 0. Deterministic, no LLM call.
    gold = _gold(10470, 10482, views=98_090, desc="Darth Maul Sneakers")

    # OLD budget → truncated chunk → truncation, gold uncovered.
    assert hmd.is_truncated(gold, _SEGMENTS, _TRUNCATED_RESPONSE) is True
    assert _covers_gold(d0.parse_moments(_TRUNCATED_RESPONSE), gold) is False

    # FIXED budget → completed chunk → not truncation, gold now has a candidate.
    assert hmd.is_truncated(gold, _SEGMENTS, _FIXED_RESPONSE) is False
    assert _covers_gold(d0.parse_moments(_FIXED_RESPONSE), gold) is True


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis guard — the two golds ARE truncation on the real artifacts (#56)
# ─────────────────────────────────────────────────────────────────────────────

_VOD = "2769594798"
_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
# The two floor-truncation golds from the #56 routing verdict (start seconds).
_TRUNCATION_GOLDS = {10470.0: "Darth Maul Sneakers", 13498.4: "400k Live Viewers"}


def _real_artifacts_present() -> bool:
    return any(_STORAGE_DIR.glob(f"pipeline_debug_*_{_VOD}"))


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real pipeline_debug artifacts not present (tmp/storage is gitignored)",
)
def test_two_golds_are_truncation_on_real_artifacts_before_the_fix() -> None:
    # AC #1, pinned on the frozen artifacts: the two 2769594798 golds are #56
    # `truncation` hard_miss (their covering chunk aborted mid-JSON). This is the
    # diagnosis the budget fix targets; the live re-gen verifies the bucket → 0.
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()
    raw, pool, seg, resp, cd, _gaps = hmd.build_inputs(anchors, _STORAGE_DIR)
    result = hmd.run(manifest, raw, pool, segments_by_vod=seg, response_by_vod=resp,
                     chunk_duration_by_vod=cd, anchor_vod_ids=anchors)

    truncation = {round(g["start"], 1): g for g in result["hard_miss"]["floor"]["golds"]
                  if g["vod_id"] == _VOD and g["bucket"] == hmd.TRUNCATION}
    assert set(truncation) == set(_TRUNCATION_GOLDS)
