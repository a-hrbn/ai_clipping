"""Archetype-Judge as a serving RankScoreFn — the #103 seam, plugged in (Slice 2, #104).

Reuses the offline-validated judge unchanged (``ml.archetype_judge_retest``): the
neutral repost base + the Jynxzi archetypes (``archetype_prompt_builder``), reading a
±12 s raw transcript excerpt per candidate, scoped from a per-candidate ``RerankFn`` to
the pool-level ``RankScoreFn`` by ``judge_shortlist_rerank(depth=None)`` — the full
deduped pool, no top-N cut. The ``ml`` imports are lazy (inside the call) because
``ml.replay_harness`` imports ``services.moment_finder`` at module load; deferring breaks
that cycle.

Two behaviours this serving adapter adds over the offline run:
  * **profile-gate** — a profile with no archetypes returns ``None`` (the caller ranks via
    the composite, unchanged), so the judge runs ONLY for archetype profiles.
  * **degrade-to-composite** — the offline run aborts LOUDLY on an empty/over-truncated
    judge response (``EmptyJudgeResponseError``) so a failed paid call never reads as a real
    0.0 verdict. Serving inverts that: it catches the empty response AND any provider
    error/timeout, logs the degrade, and returns the pool unchanged (the composite order) —
    never raising, never delivering an empty set.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# The symmetric ±pad raw-transcript excerpt around each candidate window (PRD #95 AC):
# narrow enough that the judge scores the moment, not a whole chunk.
EXCERPT_PAD_SEC = 12.0

# The Sonnet-class model both arms ran on offline; the serving judge keeps it.
DEFAULT_JUDGE_MODEL = "gemini-2.5-pro"

# rows -> rows: the pool-level ranking seam (== services.moment_finder.RankScoreFn).
RankScoreFn = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
# prompt in, raw response text out (== ml.repost_virality_judge.CallProviderFn).
CallProviderFn = Callable[[str], str]


def archetype_judge_rank_score_fn(
    *,
    profile_spec: Mapping[str, Any] | None,
    segments: Sequence[Any],
    provider: CallProviderFn,
    model: str = "",
    chunk_size: int | None = None,
    max_retries: int = 3,
    pad_sec: float = EXCERPT_PAD_SEC,
    stats: dict[str, Any] | None = None,
) -> RankScoreFn | None:
    """A pool-level ``RankScoreFn`` that re-ranks the deduped pool via the Archetype-Judge.

    Returns ``None`` when ``profile_spec`` carries no archetypes — the **profile-gate**:
    the caller then ranks via the composite (``post_process_pool`` under identity), so the
    judge never runs for a non-archetype profile.

    Otherwise it wires the offline ``neutral_repost_judge`` with the archetype prompt
    (``archetype_prompt_builder``) and the injected ``provider``, scopes it to the FULL
    deduped pool via ``judge_shortlist_rerank(depth=None)``, and wraps the result in the
    **degrade-to-composite** guard: an ``EmptyJudgeResponseError`` or any provider
    error/timeout is logged and the pool is returned unchanged (composite order) — serving
    never raises and never delivers an empty set. The judge score overwrites
    ``composite_score`` outright (**full override**, no blend). Judge provider/model,
    latency, and the degrade flag are recorded into ``stats`` for the job summary.
    """
    archetypes = list((profile_spec or {}).get("archetypes") or [])
    if not archetypes:
        return None

    # Lazy import: ml.replay_harness imports services.moment_finder at load time, so
    # importing the ml judge machinery at module top would be circular.
    from ml.archetype_judge_retest import archetype_prompt_builder, neutral_repost_judge
    from ml.judge_spike import judge_shortlist_rerank

    sink = stats if stats is not None else {}
    resolved_model = str(model or DEFAULT_JUDGE_MODEL)

    judge = neutral_repost_judge(
        call_provider=provider,
        persist_raw=lambda _text: None,  # serving doesn't persist raw responses
        segments=list(segments),
        pad_sec=pad_sec,
        build_prompt=archetype_prompt_builder(archetypes),
        chunk_size=chunk_size or None,
        max_retries=max_retries,
    )
    judge_rank = judge_shortlist_rerank(judge, depth=None)

    def _rank_score_fn(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sink.update({
            "archetype_judge": True,
            "judge_provider": "gemini",
            "judge_model": resolved_model,
            "judge_archetype_count": len(archetypes),
            "judge_degraded": False,
        })
        start = time.perf_counter()
        try:
            scored = judge_rank(rows)
            sink["judge_latency_sec"] = round(time.perf_counter() - start, 3)
            sink["judge_scored_count"] = len(scored)
            return scored
        # Catches EmptyJudgeResponseError (the offline LOUD abort, a RuntimeError) AND
        # any provider error/timeout — serving degrades, it never raises.
        except Exception as exc:  # degrade-to-composite
            sink["judge_degraded"] = True
            sink["judge_degrade_reason"] = type(exc).__name__
            sink["judge_latency_sec"] = round(time.perf_counter() - start, 3)
            sink["judge_scored_count"] = 0
            logger.warning(
                json.dumps(
                    {
                        "event": "archetype_judge_degraded_to_composite",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "pool_size": len(rows),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            return list(rows)

    return _rank_score_fn


def default_judge_provider(
    *,
    model: str = "",
    timeout_sec: float,
    retries: int,
    retry_backoff_sec: float,
) -> CallProviderFn:
    """The production provider seam: bind the judge to gemini-2.5-pro (temp 0).

    Lazily imported so serving callers that never enable the judge don't pull the offline
    module. Tests inject ``fake_gemini_provider`` instead and never reach this.
    """
    from ml.archetype_judge_retest import GEMINI_PRO_MODEL, gemini_pro_provider

    return gemini_pro_provider(
        model=model or GEMINI_PRO_MODEL,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
    )
