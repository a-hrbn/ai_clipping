from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ml import llm_moment_finder as d0
from services.timeline_types import Candidate, CandidateFeatures, RankedCandidate
from . import _post_processor
from services.transcriber import RichSegment

# rows -> rows. The pool-level ranking seam (ADR 0001): a fn receives the deduped
# pool (every candidate that survived dedupe, in snap order) and returns rows with
# a possibly-overridden ``composite_score`` and/or reordered. Default identity
# keeps the current composite order. This is where the Archetype-Judge plugs in.
RankScoreFn = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

# Composite-score weights (must sum to 1.0). Tune against Holdout-Eval.
_COMPOSITE_W_VIRALITY = 0.45
_COMPOSITE_W_HOOK     = 0.25
_COMPOSITE_W_VALUE    = 0.15
_COMPOSITE_W_FLOW     = 0.10
_COMPOSITE_W_TREND    = 0.05


@dataclass(frozen=True)
class MomentFinderResult:
    ranked_candidates: list[RankedCandidate]
    post_processed_moments: list[dict[str, Any]]
    raw_moments: list[dict[str, Any]]
    prompt: str
    response_text: str
    usage: dict[str, Any]
    summary: dict[str, Any]


class MomentFinderError(RuntimeError):
    pass


def find_moments(
    segments: Sequence[RichSegment],
    *,
    profile: str,
    vod_id: str = "",
    provider: str = d0.DEFAULT_PROVIDER,
    model: str = "",
    max_moments: int = d0.DEFAULT_MAX_MOMENTS,
    min_duration_sec: float = d0.DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: float = d0.DEFAULT_MAX_DURATION_SEC,
    max_tokens: int = d0.DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    timeout_sec: float = d0.DEFAULT_TIMEOUT_SEC,
    api_retries: int = d0.DEFAULT_API_RETRIES,
    retry_backoff_sec: float = d0.DEFAULT_RETRY_BACKOFF_SEC,
    filter_transcript_noise: bool = True,
    chunk_duration_sec: float = d0.DEFAULT_CHUNK_DURATION_SEC,
    chunk_overlap_sec: float = d0.DEFAULT_CHUNK_OVERLAP_SEC,
    eval_recall: bool = False,
) -> MomentFinderResult:
    profile_spec = d0.load_clip_profile(profile)
    streamer_name = str((profile_spec or {}).get("streamer") or "")
    transcript_segments = rich_segments_to_transcript_segments(segments, streamer_name=streamer_name)
    prompt_transcript = d0.format_transcript_for_prompt(
        transcript_segments,
        filter_noise=filter_transcript_noise,
    )
    resolved_provider = d0.normalize_provider(provider)
    resolved_model = d0.resolve_provider_model(resolved_provider, model)

    # Split long VODs into time windows so each LLM call gets its own output-token
    # budget. A single call truncates at ~max_tokens and silently drops the back
    # half of a multi-hour stream — the dominant recall loss observed in eval.
    chunks = d0.split_segments_into_time_chunks(
        transcript_segments,
        chunk_duration_sec=chunk_duration_sec,
        chunk_overlap_sec=chunk_overlap_sec,
    ) or [transcript_segments]

    chunk_prompts: list[str] = []
    chunk_responses: list[str] = []
    chunk_usages: list[dict[str, Any]] = []
    collected: list[d0.LlmMoment] = []
    try:
        for chunk_segments in chunks:
            chunk_transcript = d0.format_transcript_for_prompt(
                chunk_segments,
                filter_noise=filter_transcript_noise,
            )
            chunk_prompt = d0.build_moment_finder_prompt(
                profile=profile,
                vod_id=vod_id or "serving",
                transcript_text=chunk_transcript.text,
                max_moments=max_moments,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                profile_spec=profile_spec,
            )
            chunk_response, chunk_usage = d0.call_moment_finder_provider(
                chunk_prompt,
                provider=resolved_provider,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                retries=api_retries,
                retry_backoff_sec=retry_backoff_sec,
            )
            collected.extend(d0.parse_moments(chunk_response))
            chunk_prompts.append(chunk_prompt)
            chunk_responses.append(chunk_response)
            chunk_usages.append(dict(chunk_usage or {}))
        max_time_sec = max((float(s.end) for s in transcript_segments), default=None)
        timestamp_repair = d0.repair_clock_encoded_timestamps(collected, max_time_sec=max_time_sec)
        moments = timestamp_repair.moments
        # Remove cross-chunk duplicates from the overlap regions (no-op for a
        # single chunk, so single-call VODs keep their exact prior behaviour).
        if len(chunks) > 1:
            moments, _ = d0.dedupe_moments_by_overlap(moments)
    except SystemExit as exc:
        raise MomentFinderError(str(exc)) from exc
    except Exception as exc:
        raise MomentFinderError(str(exc)) from exc

    prompt = _join_chunk_debug(chunk_prompts)
    response_text = _join_chunk_debug(chunk_responses)
    usage = _aggregate_chunk_usage(chunk_usages)

    length_discipline = d0.enforce_moment_length_discipline(
        moments,
        max_duration_sec=max_duration_sec,
        blob_drop_duration_sec=d0.DEFAULT_BLOB_DROP_DURATION_SEC,
    )
    # Cross-chunk rank normalization is DISABLED: the all_v5 eval showed every
    # per-chunk variant (percentile, z-score, blends) loses against raw composite
    # ranking (recall@50 20% -> 11-14%) because weak chunks' tops flood the global
    # top-K. _cross_chunk_rank_scores stays available for config-gated experiments.
    rank_score_fn = None
    rank_key = _llm_moment_composite

    # Serving currency: the post-processed dict-rows the Clipping producer consumes.
    # find_moments and the offline replay converge on ONE shared post-processing
    # function — snap -> dedupe -> pool-level rank_score_fn -> diversity -> rank
    # (ADR 0001). The identity rank_score_fn (here) keeps today's serving output;
    # it is the seam the offline-validated Archetype-Judge plugs into (Slice 2).
    # No top_k here: the Clipping producer truncates to clip_top_k downstream.
    moment_rows = llm_moments_to_post_processor_rows(length_discipline.moments)
    post_processed_rows = post_process_pool(moment_rows, list(segments))

    # Eval-only encodings, gated behind eval_recall (off for serving): the raw
    # pre-dedupe/pre-diversity branch (a second snap_boundaries pass measuring true
    # LLM recall) and the typed RankedCandidate (the 65-field CandidateFeatures
    # build). Serving consumes neither, so the default pays for neither.
    if eval_recall:
        raw_ranked = sorted(length_discipline.moments, key=rank_key, reverse=True)
        raw_rows = llm_moments_to_post_processor_rows(raw_ranked, rank_score_fn=rank_score_fn)
        raw_snapped = _post_processor.snap_boundaries(raw_rows, list(segments))
        raw_post_processed_rows = _rank_post_processed_rows(raw_snapped)
        ranked_candidates = post_processed_rows_to_ranked_candidates(post_processed_rows)
    else:
        raw_post_processed_rows = []
        ranked_candidates = []
    summary = {
        "provider": resolved_provider,
        "model": resolved_model,
        "vod_id": vod_id,
        "raw_moment_count": len(raw_post_processed_rows),
        "timestamp_repaired_moment_count": timestamp_repair.repaired_moment_count,
        "moment_count_after_length_policy": len(length_discipline.moments),
        "post_processed_moment_count": len(post_processed_rows),
        "ranked_candidate_count": len(ranked_candidates),
        "input_moment_count": length_discipline.input_moment_count,
        "capped_moment_count": length_discipline.capped_moment_count,
        "dropped_long_moment_count": len(length_discipline.dropped_blob_moments),
        "dropped_non_positive_duration_count": length_discipline.dropped_non_positive_duration_count,
        "max_duration_sec": round(float(length_discipline.max_duration_sec), 3),
        "blob_drop_duration_sec": round(float(length_discipline.blob_drop_duration_sec), 3),
        "transcript_input_segments": prompt_transcript.input_segments,
        "transcript_prompt_segments": prompt_transcript.kept_segments,
        "transcript_dropped_segments": prompt_transcript.dropped_segments,
        "transcript_filter_reasons": prompt_transcript.dropped_reasons,
        "prompt_chars": len(prompt),
        "chunk_count": len(chunks),
        "chunk_duration_sec": round(float(chunk_duration_sec), 3),
    }
    return MomentFinderResult(
        ranked_candidates=ranked_candidates,
        post_processed_moments=post_processed_rows,
        raw_moments=raw_post_processed_rows,
        prompt=prompt,
        response_text=response_text,
        usage=usage,
        summary=summary,
    )


def _join_chunk_debug(parts: list[str]) -> str:
    """Join per-chunk prompts/responses for the single debug artifact files."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    total = len(parts)
    return "\n\n".join(f"===== CHUNK {i + 1}/{total} =====\n{p}" for i, p in enumerate(parts))


def _aggregate_chunk_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum token usage across chunks; keep the per-chunk breakdown for debugging."""
    if not usages:
        return {}
    if len(usages) == 1:
        return usages[0]
    aggregated: dict[str, Any] = {"chunk_count": len(usages), "chunks": usages}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        aggregated[key] = sum(int(u.get(key) or 0) for u in usages)
    return aggregated


def _moment_key(moment: d0.LlmMoment) -> tuple[float, float]:
    return (round(float(moment.start), 3), round(float(moment.end), 3))


def _cross_chunk_rank_scores(
    moments: Sequence[d0.LlmMoment],
    chunk_duration_sec: float,
    t0: float,
) -> dict[tuple[float, float], float]:
    """Per-chunk percentile rank score, so each chunk's standouts compete fairly.

    Moments are grouped by their VOD-time window (derived from `start`, so the
    grouping survives repair/dedupe/length-discipline). Within each window they
    are ranked by raw composite; the returned score is the within-window
    percentile (top moment ~1.0) plus a tiny composite tiebreak. Sorting all
    moments by this score interleaves the chunks' best moments instead of letting
    one chunk's score inflation dominate the global top-K.
    """
    from collections import defaultdict

    groups: dict[int, list[d0.LlmMoment]] = defaultdict(list)
    for moment in moments:
        idx = int((float(moment.start) - t0) // chunk_duration_sec) if chunk_duration_sec > 0 else 0
        groups[idx].append(moment)

    scores: dict[tuple[float, float], float] = {}
    for group in groups.values():
        ordered = sorted(group, key=_llm_moment_composite, reverse=True)
        n = len(ordered)
        for rank, moment in enumerate(ordered):
            percentile = 1.0 - (rank / n) if n > 1 else 1.0
            # Composite tiebreak is kept < smallest percentile step (1/n, n<=~180)
            # and pulls DOWN from the percentile so the score stays strictly < 1.0:
            # _virality_to_pipeline_score treats values > 1.0 as already 1-10 scale,
            # which stranded every chunk-top moment (percentile 1.0) at ~1.0 while
            # the rest scaled to ~10 (all_v5: GT rank 1 -> 356).
            composite = _llm_moment_composite(moment)
            scores[_moment_key(moment)] = percentile - (1.0 + 1e-6 - composite) * 0.001
    return scores


def _render_speaker(raw_speaker: object, streamer_name: str) -> str:
    """Map the diarizer's STREAMER role to the streamer's name (WALL-Re-Test #92).

    STREAMER → ``streamer_name`` when one is configured (else the role label is
    kept); GUEST_n and UNKNOWN pass through unchanged. UNKNOWN renders no marker
    downstream, so an un-diarized transcript keeps its exact pre-diarization form.
    """
    speaker = str(raw_speaker or "UNKNOWN").strip() or "UNKNOWN"
    if speaker == "STREAMER" and streamer_name:
        return streamer_name
    return speaker


def rich_segments_to_transcript_segments(
    segments: Sequence[RichSegment], *, streamer_name: str = ""
) -> list[d0.TranscriptSegment]:
    streamer = str(streamer_name or "").strip()
    transcript_segments: list[d0.TranscriptSegment] = []
    for segment in segments:
        start = float(getattr(segment, "start", 0.0) or 0.0)
        end = float(getattr(segment, "end", 0.0) or 0.0)
        text = str(getattr(segment, "text", "") or "").strip()
        if end <= start or not text:
            continue
        transcript_segments.append(d0.TranscriptSegment(
            start=start,
            end=end,
            text=text,
            speaker=_render_speaker(getattr(segment, "speaker", "UNKNOWN"), streamer),
            chat_spike_score=float(getattr(segment, "chat_spike_score", 0.0) or 0.0),
            chat_hype_score=float(getattr(segment, "chat_hype_score", 0.0) or 0.0),
            chat_funny_score=float(getattr(segment, "chat_funny_score", 0.0) or 0.0),
            chat_shock_score=float(getattr(segment, "chat_shock_score", 0.0) or 0.0),
            voice_energy_burst_score=float(getattr(segment, "voice_energy_burst_score", 0.0) or 0.0),
        ))
    return transcript_segments


def llm_moments_to_post_processor_rows(
    moments: Sequence[d0.LlmMoment],
    *,
    rank_score_fn=None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, moment in enumerate(moments, start=1):
        hook = round(float(moment.hook), 6)
        flow = round(float(moment.flow), 6)
        value = round(float(moment.value), 6)
        trend = round(float(moment.trend), 6)
        virality = round(float(moment.virality), 6)
        # Ranking uses the cross-chunk-normalized score when provided; viral_score
        # and the sub-scores below stay the raw LLM values for display/features.
        composite = rank_score_fn(moment) if rank_score_fn is not None else _llm_moment_composite(moment)
        rows.append(
            {
                "rank": index,
                "start_time": round(float(moment.start), 3),
                "end_time": round(float(moment.end), 3),
                "viral_score": _virality_to_pipeline_score(virality),
                "composite_score": _virality_to_pipeline_score(composite),
                "llm_virality": virality,
                "hook": hook,
                "flow": flow,
                "value": value,
                "trend": trend,
                "reason": moment.rationale,
            }
        )
    return rows


def post_processed_rows_to_ranked_candidates(rows: Sequence[Mapping[str, Any]]) -> list[RankedCandidate]:
    ranked: list[RankedCandidate] = []
    for index, raw in enumerate(rows, start=1):
        row = dict(raw)
        start = _float_value(row.get("start_time"))
        end = _float_value(row.get("end_time"))
        if start is None or end is None or end <= start:
            continue
        rank_score = _float_value(row.get("viral_score"))
        if rank_score is None:
            rank_score = _virality_to_pipeline_score(row.get("llm_virality"))
        llm_virality = _float_value(row.get("llm_virality"))
        if llm_virality is None:
            llm_virality = max(0.0, min(float(rank_score) / 10.0, 1.0))
        candidate_id = _candidate_id(index, start, end)
        rationale = [
            "llm_moment_finder",
            f"hook={_float_value(row.get('hook')) or 0.0:.2f}",
            f"flow={_float_value(row.get('flow')) or 0.0:.2f}",
            f"value={_float_value(row.get('value')) or 0.0:.2f}",
            f"trend={_float_value(row.get('trend')) or 0.0:.2f}",
        ]
        reason = str(row.get("reason") or row.get("rationale") or "").strip()
        if reason:
            rationale.append(reason)
        candidate = Candidate(
            id=candidate_id,
            start=float(start),
            end=float(end),
            source="llm_moment_finder",
            anchor_time=(float(start) + float(end)) / 2.0,
            score_hint=float(rank_score),
            tags=["llm_moment_finder"],
        )
        features = _candidate_features_from_row(
            candidate_id=candidate_id,
            duration=float(end) - float(start),
            rank_score=float(rank_score),
            llm_virality=float(llm_virality),
            row=row,
        )
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                features=features,
                rank_score=float(rank_score),
                rationale=rationale,
                base_rank_score=float(rank_score),
                heuristic_rank_score=float(rank_score),
            )
        )
    ranked.sort(key=lambda item: item.rank_score, reverse=True)
    return ranked


def _rank_post_processed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows = [dict(row) for row in rows]
    ranked_rows.sort(
        key=lambda row: (
            _float_value(row.get("composite_score")) or _float_value(row.get("viral_score")) or 0.0,
            -(_float_value(row.get("start_time")) or 0.0),
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index
    return ranked_rows


def _identity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def _row_to_moment(row: Mapping[str, Any]) -> d0.LlmMoment:
    """Rebuild the LlmMoment the dedupe step consumes from a post-processed row.

    The inverse of ``llm_moments_to_post_processor_rows`` for the fields
    ``dedupe_moments_by_overlap`` reads (window + virality), so the shared
    post-processing calls the real dedupe instead of copying its overlap logic.
    """
    return d0.LlmMoment(
        start=float(row.get("start_time") or 0.0),
        end=float(row.get("end_time") or 0.0),
        hook=float(row.get("hook") or 0.0),
        flow=float(row.get("flow") or 0.0),
        value=float(row.get("value") or 0.0),
        trend=float(row.get("trend") or 0.0),
        virality=float(row.get("llm_virality") or 0.0),
        rationale=str(row.get("reason") or ""),
    )


def _dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop overlap-duplicate rows via the real dedupe, preserving input order.

    Reconstructs one LlmMoment per row, asks ``dedupe_moments_by_overlap`` which
    survive, and keeps the surviving rows in their incoming (snap) order — the
    order the downstream diversity/ranking expect. Survivors are matched back by
    object identity, so two distinct rows that snapped to the same window are
    deduped correctly (a window-key match would keep both).
    """
    moments = [_row_to_moment(row) for row in rows]
    kept, _ = d0.dedupe_moments_by_overlap(moments)
    kept_ids = {id(moment) for moment in kept}
    return [dict(row) for row, moment in zip(rows, moments) if id(moment) in kept_ids]


def snap_dedupe_rescore(
    raw_rows: Sequence[Mapping[str, Any]],
    segments: Sequence[RichSegment],
    rank_score_fn: RankScoreFn | None,
) -> list[dict[str, Any]]:
    """snap -> dedupe -> rank_score_fn: the shared pre-ranking base.

    The point where the rank-only pool (snap -> dedupe -> rank, the judge-shortlist
    currency) and the full ``post_process_pool`` (+ diversity + rank + top_k)
    diverge — every step a call into the real serving function, so both stay free
    of eval/serving skew.
    """
    snapped = _post_processor.snap_boundaries(list(raw_rows), list(segments))
    deduped = _dedupe_rows(snapped)
    return (rank_score_fn or _identity_rows)(deduped)


def post_process_pool(
    raw_rows: Sequence[Mapping[str, Any]],
    segments: Sequence[RichSegment],
    *,
    rank_score_fn: RankScoreFn | None = None,
    top_k: int = 0,
) -> list[dict[str, Any]]:
    """snap -> dedupe -> rank_score_fn(pool) -> diversity -> rank -> top_k.

    The single shared serving post-processing path: both ``find_moments`` and the
    offline ``replay_harness.replay_postprocess`` call it, so the offline number
    equals the serving number (no eval/serving skew) and the pool-level
    ``rank_score_fn`` seam (ADR 0001) steers both. Under the identity
    ``rank_score_fn`` the output equals today's serving output.

    ``rank_score_fn`` receives the deduped pool and may overwrite each row's
    ``composite_score`` and/or reorder. ``top_k > 0`` truncates to the K
    highest-ranked rows; ``top_k <= 0`` keeps the full ranked set (serving keeps
    all and lets the Clipping producer truncate to clip_top_k downstream).

    Note: dedupe runs on the *snapped* windows and ranks survivors by virality,
    while diversity ranks by composite. On real finder output these agree; a
    hand-crafted pool where the two orderings invert (a moment with high virality
    but low composite snapping onto a higher-composite neighbour) can pick a
    different survivor than the pre-snap dedupe order would — an accepted edge of
    converging on the snap->dedupe order (ADR 0001).
    """
    rescored = snap_dedupe_rescore(raw_rows, segments, rank_score_fn)
    diverse = _post_processor.enforce_diversity(rescored)
    ranked = _rank_post_processed_rows(diverse)
    return ranked[:top_k] if top_k > 0 else ranked


def _candidate_features_from_row(
    *,
    candidate_id: str,
    duration: float,
    rank_score: float,
    llm_virality: float,
    row: Mapping[str, Any],
) -> CandidateFeatures:
    _h = _float_value(row.get("hook"))
    _fl = _float_value(row.get("flow"))
    _v = _float_value(row.get("value"))
    _tr = _float_value(row.get("trend"))
    hook = llm_virality if _h is None else _h
    flow = llm_virality if _fl is None else _fl
    value = llm_virality if _v is None else _v
    trend = llm_virality if _tr is None else _tr
    return CandidateFeatures.create(
        candidate_id=candidate_id,
        duration=max(float(duration), 0.0),
        hook_strength=hook,
        payoff_score=value,
        quoteability_score=value,
        standalone_score=value,
        context_efficiency=flow,
        trend_prior_score=trend,
        projected_hook_score=hook,
        projected_retention_score=flow,
        semantic_density=value,
        has_llm_semantics=True,
        source="llm_moment_finder",
        clip_type="llm_moment",
        heuristic_score=rank_score,
        multimodal_proxy_score=rank_score,
    )


def _candidate_id(rank: int, start: float, end: float) -> str:
    return f"llm_moment:{int(rank)}:{int(round(start * 1000))}:{int(round(end * 1000))}"


def _virality_to_pipeline_score(value: Any) -> float:
    number = _float_value(value)
    if number is None:
        return 1.0
    if number <= 1.0:
        number *= 10.0
    return max(1.0, min(round(float(number), 3), 10.0))


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _llm_moment_composite(moment: d0.LlmMoment) -> float:
    """Weighted composite from a LlmMoment (0-1 scale).

    Audio-energy is a per-segment signal, not a per-moment LlmMoment field.
    It is factored in later via _composite_score on the row dict.
    """
    return (
        _COMPOSITE_W_VIRALITY * float(moment.virality)
        + _COMPOSITE_W_HOOK * float(moment.hook)
        + _COMPOSITE_W_VALUE * float(moment.value)
        + _COMPOSITE_W_FLOW * float(moment.flow)
        + _COMPOSITE_W_TREND * float(moment.trend)
    )
