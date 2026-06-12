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

def test_production_generation_config_raises_token_budget_to_64000() -> None:
    # The change lands in the production generation config: the canonical code
    # default and the settings value both request 64000 (was 32000), same model (no
    # swap). NOTE — net-positive headroom, NOT a proven truncation cure: a live
    # re-gen (#68) showed dense chunks fill 64000 too (completion 63990), because
    # max_moments=180 scales output to any budget. Kept as a harmless headroom bump.
    from config import Settings

    assert d0.DEFAULT_MAX_TOKENS == 64000
    assert Settings().moment_finder_max_tokens == 64000
    assert Settings().moment_finder_max_tokens > 32000   # strictly more headroom than before


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


def test_a_completed_covering_chunk_recovers_the_gold_a_truncated_one_loses_it() -> None:
    # The mechanism the truncation bucket rests on — NOT a claim that 64000 achieves
    # completion (the live #68 re-gen showed it does not: dense chunks fill 64000 too).
    # With a truncated covering chunk the #56 detector reads truncation AND no
    # candidate covers the Darth Maul gold (its moment never closed); with the SAME
    # chunk *completed* the detector clears it AND a candidate covers the gold. This
    # pins what truncation costs and what completion recovers — the model-independent
    # output-size fix (out of scope) is what would actually complete these chunks.
    gold = _gold(10470, 10482, views=98_090, desc="Darth Maul Sneakers")

    # Truncated chunk → truncation, gold uncovered.
    assert hmd.is_truncated(gold, _SEGMENTS, _TRUNCATED_RESPONSE) is True
    assert _covers_gold(d0.parse_moments(_TRUNCATED_RESPONSE), gold) is False

    # Completed chunk → not truncation, gold now has a candidate.
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
