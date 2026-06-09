from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ml import llm_moment_finder as d0
from services import post_processor
from services.timeline_types import Candidate, CandidateFeatures, RankedCandidate
from services.transcriber import RichSegment

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
) -> MomentFinderResult:
    transcript_segments = rich_segments_to_transcript_segments(segments)
    prompt_transcript = d0.format_transcript_for_prompt(
        transcript_segments,
        filter_noise=filter_transcript_noise,
    )
    resolved_provider = d0.normalize_provider(provider)
    resolved_model = d0.resolve_provider_model(resolved_provider, model)
    profile_spec = d0.load_clip_profile(profile)
    prompt = d0.build_moment_finder_prompt(
        profile=profile,
        vod_id=vod_id or "serving",
        transcript_text=prompt_transcript.text,
        max_moments=max_moments,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        profile_spec=profile_spec,
    )
    try:
        response_text, usage = d0.call_moment_finder_provider(
            prompt,
            provider=resolved_provider,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            retries=api_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        raw_moments = d0.parse_moments(response_text)
        timestamp_repair = d0.repair_clock_encoded_timestamps(raw_moments)
        moments = timestamp_repair.moments
    except SystemExit as exc:
        raise MomentFinderError(str(exc)) from exc
    except Exception as exc:
        raise MomentFinderError(str(exc)) from exc

    length_discipline = d0.enforce_moment_length_discipline(
        moments,
        max_duration_sec=max_duration_sec,
        blob_drop_duration_sec=d0.DEFAULT_BLOB_DROP_DURATION_SEC,
    )
    # Raw branch: pre-dedupe, pre-diversity — measures true LLM recall for eval.
    raw_ranked = sorted(length_discipline.moments, key=_llm_moment_composite, reverse=True)
    raw_rows = llm_moments_to_post_processor_rows(raw_ranked)
    raw_snapped = post_processor.snap_boundaries(raw_rows, list(segments))
    raw_post_processed_rows = _rank_post_processed_rows(raw_snapped)

    deduped_moments, duplicate_moments = d0.dedupe_moments_by_overlap(length_discipline.moments)
    ranked_moments = sorted(deduped_moments, key=_llm_moment_composite, reverse=True)
    moment_rows = llm_moments_to_post_processor_rows(ranked_moments)
    snapped_rows = post_processor.snap_boundaries(moment_rows, list(segments))
    diverse_rows = post_processor.enforce_diversity(snapped_rows)
    post_processed_rows = _rank_post_processed_rows(diverse_rows)
    ranked_candidates = post_processed_rows_to_ranked_candidates(post_processed_rows)
    summary = {
        "provider": resolved_provider,
        "model": resolved_model,
        "vod_id": vod_id,
        "raw_moment_count": len(raw_post_processed_rows),
        "timestamp_repaired_moment_count": timestamp_repair.repaired_moment_count,
        "moment_count_after_length_policy": len(ranked_moments),
        "duplicate_moment_count": len(duplicate_moments),
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


def rich_segments_to_transcript_segments(segments: Sequence[RichSegment]) -> list[d0.TranscriptSegment]:
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
            chat_spike_score=float(getattr(segment, "chat_spike_score", 0.0) or 0.0),
            chat_hype_score=float(getattr(segment, "chat_hype_score", 0.0) or 0.0),
            chat_funny_score=float(getattr(segment, "chat_funny_score", 0.0) or 0.0),
            chat_shock_score=float(getattr(segment, "chat_shock_score", 0.0) or 0.0),
            voice_energy_burst_score=float(getattr(segment, "voice_energy_burst_score", 0.0) or 0.0),
        ))
    return transcript_segments


def llm_moments_to_post_processor_rows(moments: Sequence[d0.LlmMoment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, moment in enumerate(moments, start=1):
        hook = round(float(moment.hook), 6)
        flow = round(float(moment.flow), 6)
        value = round(float(moment.value), 6)
        trend = round(float(moment.trend), 6)
        virality = round(float(moment.virality), 6)
        rows.append(
            {
                "rank": index,
                "start_time": round(float(moment.start), 3),
                "end_time": round(float(moment.end), 3),
                "viral_score": _virality_to_pipeline_score(virality),
                "composite_score": _virality_to_pipeline_score(_llm_moment_composite(moment)),
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
