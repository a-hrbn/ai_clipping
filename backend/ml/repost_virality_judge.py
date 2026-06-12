"""Repost-Virality Judge — scores a deduped candidate set on *repost* virality.

A strong-model judge that rates an already-deduped candidate set on how likely
each moment is to go viral *as a repost* (the quiet, quotable, narrative beats the
@Iboclipss reposts surface) — NOT on live-chat arousal, which PRD #46 proved is
the wrong rank signal. It is shaped as a ``RerankFn`` so it drops straight into
the offline replay harness as the pluggable ``rank_score_fn``.

Two-Adapter-Disziplin, mirroring ``llm_moment_finder``:
  * ``repost_virality_judge`` — the real adapter: builds a compact per-candidate
    prompt (no long rationale by default — lessons-konform gegen JSON-Truncation),
    calls the injected provider (default Anthropic Sonnet, ``anthropic_model_topk``),
    persists the raw response BEFORE parsing, then parses compact scores.
  * ``fake_repost_virality_judge`` — deterministic canned scores, no network.

Fail-closed: a candidate the judge did not score is DROPPED, never back-filled with
a default that would distort top-k (PRD #46). The drop count is logged, so a
truncated response never reads as "scored everything".
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Mapping

from config import settings
from ml import llm_moment_finder as d0

logger = logging.getLogger(__name__)

# Compact scores for the deduped top-~200 shortlist cost ~15 tokens/candidate, so an
# 8k ceiling clears the whole shortlist with headroom — keeping per-call output well
# under the limit that truncates long JSON (the reason rationales are dropped).
DEFAULT_JUDGE_MAX_TOKENS = 8000

# A deduped serving-currency row (the shape ``llm_moments_to_post_processor_rows``
# emits): start_time/end_time/composite_score/reason/… The judge overwrites the
# rank-score (``composite_score``) and leaves the rest untouched.
Row = dict[str, Any]
RerankFn = Callable[[list[Row]], list[Row]]
# The single provider seam: prompt in, raw response text out. The real wiring binds
# it to Anthropic Sonnet; tests inject a canned-response fake (no network, no paid call).
CallProviderFn = Callable[[str], str]


def repost_virality_judge(
    *,
    call_provider: CallProviderFn,
    persist_raw: Callable[[str], None],
) -> RerankFn:
    """The real adapter: judge a deduped candidate set on repost-virality.

    Builds a compact per-candidate prompt, calls the injected provider, persists the
    raw response BEFORE parsing, then overwrites each scored row's rank-score.
    """

    def _rerank(rows: list[Row]) -> list[Row]:
        prompt = build_judge_prompt(rows)
        raw = call_provider(prompt)
        persist_raw(raw)
        scores = parse_judge_scores(raw)
        reranked = apply_scores(rows, scores)
        logger.info(
            json.dumps(
                {
                    "event": "repost_virality_judge_scored",
                    "candidates": len(rows),
                    "scored": len(reranked),
                    "dropped": len(rows) - len(reranked),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return reranked

    return _rerank


def anthropic_sonnet_provider(
    *,
    model: str = "",
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    timeout_sec: float = d0.DEFAULT_TIMEOUT_SEC,
    retries: int = d0.DEFAULT_API_RETRIES,
    retry_backoff_sec: float = d0.DEFAULT_RETRY_BACKOFF_SEC,
) -> CallProviderFn:
    """Default wiring: bind the provider seam to the configured strong model
    (Anthropic Sonnet, ``anthropic_model_topk``). Returns just the response text;
    the judge's own parser owns the structured decode."""
    resolved_model = str(model or settings.anthropic_model_topk or d0.DEFAULT_ANTHROPIC_MODEL)

    def _call(prompt: str) -> str:
        text, _usage = d0.call_anthropic_moment_finder(
            prompt,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        return text

    return _call


def build_judge_prompt(rows: list[Row]) -> str:
    """Render the compact judge prompt: one line per candidate (addressed by the same
    1-based id the parser maps back), asking for scores ONLY — no per-candidate
    rationale, which is what truncates long JSON responses (lessons-konform)."""
    candidate_lines = "\n".join(_format_candidate(index, row) for index, row in enumerate(rows, start=1))
    return f"""You are an expert judge of REPOST virality for short-form Twitch clips.

You are scoring an already-selected, deduplicated candidate set — do NOT propose new \
moments. Score each candidate on how likely the moment is to go viral as a REPOST: the \
quiet, quotable, narrative beats (a sharp explanation, a memorable anecdote, a clean hot \
take) that get reposted and re-shared days later. This is NOT live-chat hype — a loud \
live reaction (rage, clutch, chat spike) is by itself weak evidence of repost virality \
and must not inflate the score on its own.

Use the full 0.0-1.0 scale and score each candidate independently on its own merit.

Return raw JSON only, no markdown, scores only (no explanations or rationale), this exact shape:
{{"scores": [{{"id": <candidate id>, "repost_virality": <0.0-1.0>}}]}}

Candidates:
{candidate_lines}
"""


def _format_candidate(index: int, row: Row) -> str:
    start = d0.float_or_none(row.get("start_time"))
    end = d0.float_or_none(row.get("end_time"))
    window = f"t={start}s-{end}s" if start is not None and end is not None else "t=?"
    reason = str(row.get("reason") or "").strip()
    # A later harness slice may thread the transcript excerpt in via ``text``; include
    # it when present so the judge reads the actual content, not just the rationale.
    text = str(row.get("text") or "").strip()
    parts = [f"[{index}]", window]
    if reason:
        parts.append(reason)
    if text:
        parts.append(f"transcript: {text}")
    return " | ".join(parts)


def parse_judge_scores(text: str) -> dict[int, float]:
    """Parse the compact ``{"scores": [{"id", "repost_virality"}, ...]}`` response
    into an id -> 0-1 score map. Items missing either field are skipped (fail-closed
    at the item level — no fabricated scores)."""
    payload = _decode_scores_json(text)
    items = payload.get("scores") if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        return {}
    scores: dict[int, float] = {}
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        candidate_id = d0.int_or_none(raw.get("id"))
        score = d0.float_or_none(raw.get("repost_virality"))
        if candidate_id is None or score is None:
            continue
        scores[candidate_id] = max(0.0, min(float(score), 1.0))
    return scores


def _decode_scores_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        decoded = d0.parse_json_with_repair(stripped)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        return decoded
    # Truncated/malformed response: recover the complete balanced score items from
    # the known array instead of discarding the whole (paid) run.
    return d0.extract_partial_array_payload(stripped, "scores")


def fake_repost_virality_judge(scores: Mapping[int, float]) -> RerankFn:
    """Deterministic, network-free judge: canned per-candidate-id scores.

    ``scores`` maps a 1-based candidate id (the row's position in the input) to a
    0-1 repost-virality score, the same id scheme the real adapter's prompt uses.
    """
    canned = {int(k): float(v) for k, v in scores.items()}

    def _rerank(rows: list[Row]) -> list[Row]:
        return apply_scores(rows, canned)

    return _rerank


def apply_scores(rows: list[Row], scores: Mapping[int, float]) -> list[Row]:
    """Overwrite ``composite_score`` with the judge score for every scored row.

    Rows whose 1-based id is absent from ``scores`` are dropped (fail-closed).
    Returns fresh row dicts; the caller's rows are not mutated.
    """
    reranked: list[Row] = []
    for index, row in enumerate(rows, start=1):
        if index not in scores:
            continue
        scored = dict(row)
        scored["composite_score"] = _to_pipeline_scale(scores[index])
        reranked.append(scored)
    return reranked


def _to_pipeline_scale(value: float) -> float:
    """Map a 0-1 judge score onto the pipeline's 1-10 composite_score scale,
    matching ``_virality_to_pipeline_score`` so the judged rows sort comparably."""
    number = float(value)
    if number <= 1.0:
        number *= 10.0
    return max(1.0, min(round(number, 3), 10.0))
