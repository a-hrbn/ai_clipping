from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from services.timeline_types import Candidate, RankedCandidate

_SAMPLE_LIMIT = 8
_TRACE_SAMPLE_LIMIT = 25
_STAGE_ORDER = [
    "peak",
    "semantic",
    "dense",
    "dense_inventory",
    "fallback",
    "merged",
    "normalized",
    "ranked_pre_judge",
    "judge_scope",
    "clip_judge",
    "quality_gate",
    "judged_quality_kept",
    "unjudged_quality_kept",
    "quality_kept",
    "final_ranked",
    "final_moments",
]


def build_candidate_recall_report(
    *,
    stage_candidates: Mapping[str, Sequence[Candidate]],
    pre_judge_ranked: Sequence[RankedCandidate],
    judged_ranked: Sequence[RankedCandidate],
    quality_gate_rows: Sequence[Mapping[str, Any]],
    final_ranked: Sequence[RankedCandidate],
    final_moments: Sequence[Mapping[str, Any]],
    judge_top_k: int,
    judge_scope_ranked: Sequence[RankedCandidate] | None = None,
    positive_windows: Sequence[Mapping[str, Any]] | None = None,
    positive_overlap_ratio: float = 0.45,
) -> dict[str, Any]:
    """Build a compact funnel report for where serving candidates survive or drop."""

    stage_ids = [
        "peak",
        "semantic",
        "dense",
        "dense_inventory",
        "merged",
        "normalized",
        "ranked_pre_judge",
        "judge_scope",
        "judged_quality_kept",
        "unjudged_quality_kept",
        "quality_kept",
        "final_ranked",
        "final_moments",
    ]

    candidate_sets: dict[str, set[str]] = {
        stage: _candidate_ids(candidates)
        for stage, candidates in stage_candidates.items()
    }
    pre_judge_ids = {item.candidate.id for item in pre_judge_ranked}
    judge_scope_source = judge_scope_ranked if judge_scope_ranked is not None else pre_judge_ranked
    judge_scope_ids = {
        item.candidate.id
        for item in judge_scope_source[: max(int(judge_top_k or 0), 0)]
    }
    quality_kept_ids = {
        _row_candidate_id(row)
        for row in quality_gate_rows
        if (row.get("accepted") is True or row.get("kept") is True or row.get("status") == "kept")
        and _row_candidate_id(row)
    }
    judged_quality_kept_ids = {
        _row_candidate_id(row)
        for row in quality_gate_rows
        if (row.get("accepted") is True or row.get("kept") is True or row.get("status") == "kept")
        and _row_has_judge_result(row)
        and _row_candidate_id(row)
    }
    unjudged_quality_kept_ids = quality_kept_ids - judged_quality_kept_ids
    final_ranked_ids = {item.candidate.id for item in final_ranked}
    final_moment_ids = {
        _base_candidate_id(str(moment.get("candidate_id") or ""))
        for moment in final_moments
        if moment.get("candidate_id")
    }

    candidate_sets.update(
        {
            "ranked_pre_judge": pre_judge_ids,
            "judge_scope": judge_scope_ids,
            "judged_quality_kept": judged_quality_kept_ids,
            "unjudged_quality_kept": unjudged_quality_kept_ids,
            "quality_kept": quality_kept_ids,
            "final_ranked": final_ranked_ids,
            "final_moments": final_moment_ids,
        }
    )

    candidate_lookup: dict[str, dict[str, Any]] = {}
    for stage, candidates in stage_candidates.items():
        for candidate in candidates:
            candidate_lookup.setdefault(candidate.id, _candidate_sample(candidate, stage=stage))
    for ranked in pre_judge_ranked:
        sample = candidate_lookup.setdefault(
            ranked.candidate.id,
            _candidate_sample(ranked.candidate, stage="ranked_pre_judge"),
        )
        sample["pre_judge_rank_score"] = round(float(ranked.rank_score), 6)
    for ranked in judged_ranked:
        sample = candidate_lookup.setdefault(
            ranked.candidate.id,
            _candidate_sample(ranked.candidate, stage="judged"),
        )
        sample["judged_rank_score"] = round(float(ranked.rank_score), 6)
        sample["judge_used"] = ranked.judge_result is not None
    for row in quality_gate_rows:
        candidate_id = _row_candidate_id(row)
        if not candidate_id:
            continue
        sample = candidate_lookup.setdefault(candidate_id, _quality_row_sample(row, stage="quality_gate"))
        sample["quality_accepted"] = bool(row.get("accepted") is True or row.get("kept") is True)
        sample["rejection_reasons"] = list(row.get("rejection_reasons") or [])

    transitions = [
        ("merged_to_normalized", "merged", "normalized"),
        ("normalized_to_ranked", "normalized", "ranked_pre_judge"),
        ("ranked_to_judge_scope", "ranked_pre_judge", "judge_scope"),
        ("judge_scope_to_judged_quality_kept", "judge_scope", "judged_quality_kept"),
        ("judged_quality_kept_to_final_ranked", "judged_quality_kept", "final_ranked"),
        ("unjudged_quality_kept_to_final_ranked", "unjudged_quality_kept", "final_ranked"),
        ("final_ranked_to_final_moments", "final_ranked", "final_moments"),
    ]
    loss_rows = [
        _transition_loss(name, source, target, candidate_sets, candidate_lookup)
        for name, source, target in transitions
    ]

    quality_rejected = [
        row for row in quality_gate_rows
        if not (row.get("accepted") is True or row.get("kept") is True or row.get("status") == "kept")
    ]
    quality_rejected = sorted(
        quality_rejected,
        key=lambda row: float(row.get("rank_score") or row.get("judge_score") or 0.0),
        reverse=True,
    )
    judge_scope_limit = max(int(judge_top_k or 0), 0)
    normal_judge_ids = {
        item.candidate.id for item in pre_judge_ranked[:judge_scope_limit]
    }
    recall_reserved_samples = [
        _ranked_sample(item, stage="judge_recall_reserved")
        for item in judge_scope_source[:judge_scope_limit]
        if item.candidate.id not in normal_judge_ids
    ]

    positive_window_recall = _positive_window_recall_summary(
        positive_windows or [],
        stage_candidates=stage_candidates,
        pre_judge_ranked=pre_judge_ranked,
        judged_ranked=judged_ranked,
        quality_gate_rows=quality_gate_rows,
        final_ranked=final_ranked,
        final_moments=final_moments,
        judge_top_k=judge_top_k,
        judge_scope_ranked=judge_scope_ranked,
        overlap_ratio=positive_overlap_ratio,
    )

    return {
        "stage_counts": {stage: len(candidate_sets.get(stage, set())) for stage in stage_ids},
        "source_counts": {
            stage: dict(sorted(_source_counts(candidates).items()))
            for stage, candidates in stage_candidates.items()
        },
        "funnel": [
            {
                "stage": stage,
                "count": len(candidate_sets.get(stage, set())),
            }
            for stage in stage_ids
        ],
        "losses": loss_rows,
        "largest_losses": sorted(loss_rows, key=lambda row: int(row["lost_count"]), reverse=True)[:5],
        "ranker": {
            "ranked_rows": len(pre_judge_ranked),
            "judge_top_k": judge_scope_limit,
            "not_in_judge_scope_top": [
                _ranked_sample(item, stage="not_in_judge_scope")
                for item in pre_judge_ranked
                if item.candidate.id not in judge_scope_ids
            ][:_SAMPLE_LIMIT],
            "recall_reserved": recall_reserved_samples[:_SAMPLE_LIMIT],
        },
        "quality_gate": {
            "input_rows": len(quality_gate_rows),
            "kept_rows": len(quality_kept_ids),
            "judged_kept_rows": len(judged_quality_kept_ids),
            "unjudged_kept_rows": len(unjudged_quality_kept_ids),
            "rejected_rows": len(quality_rejected),
            "rejected_by_reason": dict(sorted(_quality_rejection_counts(quality_rejected).items())),
            "top_rejected": [_quality_row_sample(row, stage="quality_rejected") for row in quality_rejected[:_SAMPLE_LIMIT]],
        },
        "final": {
            "ranked_rows": len(final_ranked),
            "moment_rows": len(final_moments),
            "candidate_ids": sorted(final_moment_ids),
        },
        "positive_window_recall": positive_window_recall,
    }


def build_candidate_trace(
    *,
    stage_candidates: Mapping[str, Sequence[Candidate]],
    pre_judge_ranked: Sequence[RankedCandidate],
    judged_ranked: Sequence[RankedCandidate],
    quality_gate_rows: Sequence[Mapping[str, Any]],
    final_ranked: Sequence[RankedCandidate],
    final_moments: Sequence[Mapping[str, Any]],
    judge_top_k: int,
    judge_scope_ranked: Sequence[RankedCandidate] | None = None,
) -> list[dict[str, Any]]:
    """Build one consolidated trace row per candidate across the serving funnel."""

    trace: dict[str, dict[str, Any]] = {}

    def row_for(candidate_id: str) -> dict[str, Any]:
        candidate_id = _base_candidate_id(candidate_id)
        return trace.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "source": "",
                "start": None,
                "end": None,
                "duration": None,
                "score_hint": None,
                "tags": [],
                "stages": {},
                "stage_path": [],
                "rejection_reasons": [],
                "final_selected": False,
            },
        )

    def mark_stage(candidate_id: str, stage: str, status: str, **extra: Any) -> None:
        row = row_for(candidate_id)
        stage_row = {"status": status}
        stage_row.update({key: value for key, value in extra.items() if value is not None})
        row["stages"][stage] = stage_row

    for stage, candidates in stage_candidates.items():
        for candidate in candidates:
            row = row_for(candidate.id)
            _merge_candidate_identity(row, candidate)
            mark_stage(candidate.id, stage, "seen", score_hint=_round_optional(candidate.score_hint))

    judge_scope_limit = max(int(judge_top_k or 0), 0)
    judge_scope_source = judge_scope_ranked if judge_scope_ranked is not None else pre_judge_ranked
    judge_scope_rank_by_id = {
        _base_candidate_id(item.candidate.id): index
        for index, item in enumerate(judge_scope_source[:judge_scope_limit], start=1)
    }
    judge_scope_ids = set(judge_scope_rank_by_id)
    for index, ranked in enumerate(pre_judge_ranked, start=1):
        candidate_id = ranked.candidate.id
        base_candidate_id = _base_candidate_id(candidate_id)
        trace_row = row_for(candidate_id)
        _merge_candidate_identity(trace_row, ranked.candidate)
        trace_row["pre_judge_rank"] = index
        trace_row["pre_judge_rank_score"] = _round_optional(ranked.rank_score)
        trace_row["trained_model_used"] = bool(ranked.trained_model_used)
        mark_stage(candidate_id, "ranked_pre_judge", "ranked", rank=index, rank_score=_round_optional(ranked.rank_score))
        if base_candidate_id in judge_scope_ids:
            judge_scope_rank = judge_scope_rank_by_id.get(base_candidate_id)
            mark_stage(
                candidate_id,
                "judge_scope",
                "included",
                rank=judge_scope_rank,
                pre_judge_rank=index,
                recall_reserved=index > judge_scope_limit,
            )
            trace_row["judge_scope_rank"] = judge_scope_rank
            trace_row["judge_recall_reserved"] = bool(index > judge_scope_limit)
        else:
            mark_stage(candidate_id, "judge_scope", "excluded", rank=index)
            trace_row["rejection_reasons"].append("outside_judge_scope")

    for index, ranked in enumerate(judged_ranked, start=1):
        candidate_id = ranked.candidate.id
        trace_row = row_for(candidate_id)
        _merge_candidate_identity(trace_row, ranked.candidate)
        trace_row["judged_rank"] = index
        trace_row["judged_rank_score"] = _round_optional(ranked.rank_score)
        if ranked.judge_result is not None:
            judge = ranked.judge_result
            trace_row["judge_score"] = _round_optional(judge.judge_score)
            trace_row["judge_confidence"] = _round_optional(judge.judge_confidence)
            trace_row["judge_provider"] = judge.provider
            trace_row["judge_reason"] = judge.judge_reason
            mark_stage(
                candidate_id,
                "clip_judge",
                "judged",
                judge_score=_round_optional(judge.judge_score),
                judge_confidence=_round_optional(judge.judge_confidence),
                provider=judge.provider,
            )
        elif _base_candidate_id(candidate_id) in judge_scope_ids:
            mark_stage(candidate_id, "clip_judge", "missing_result")

    for row in quality_gate_rows:
        candidate_id = _row_candidate_id(row)
        if not candidate_id:
            continue
        trace_row = row_for(candidate_id)
        sample = _quality_row_sample(row, stage="quality_gate")
        _merge_sample_identity(trace_row, sample)
        accepted = bool(row.get("accepted") is True or row.get("kept") is True or row.get("status") == "kept")
        reasons = [str(reason) for reason in row.get("rejection_reasons") or []]
        trace_row["quality_accepted"] = accepted
        trace_row["quality_has_judge_result"] = _row_has_judge_result(row)
        trace_row["quality_rejection_reasons"] = reasons
        if not accepted:
            trace_row["rejection_reasons"].extend(reasons or ["quality_gate_rejected"])
        mark_stage(
            candidate_id,
            "quality_gate",
            "kept" if accepted else "rejected",
            rank_score=sample.get("rank_score"),
            judge_score=sample.get("judge_score"),
            rejection_reasons=reasons,
        )
        if accepted:
            mark_stage(candidate_id, "quality_kept", "kept")
            if _row_has_judge_result(row):
                mark_stage(candidate_id, "judged_quality_kept", "kept")
            else:
                mark_stage(candidate_id, "unjudged_quality_kept", "kept")

    for index, ranked in enumerate(final_ranked, start=1):
        candidate_id = ranked.candidate.id
        trace_row = row_for(candidate_id)
        _merge_candidate_identity(trace_row, ranked.candidate)
        trace_row["final_rank"] = index
        trace_row["final_rank_score"] = _round_optional(ranked.rank_score)
        mark_stage(candidate_id, "final_ranked", "selected", rank=index, rank_score=_round_optional(ranked.rank_score))

    for index, moment in enumerate(final_moments, start=1):
        candidate_id = _base_candidate_id(str(moment.get("candidate_id") or ""))
        if not candidate_id:
            continue
        trace_row = row_for(candidate_id)
        trace_row["final_selected"] = True
        trace_row["final_moment_index"] = index
        trace_row["final_start"] = _round_optional(moment.get("start_time") or moment.get("start"))
        trace_row["final_end"] = _round_optional(moment.get("end_time") or moment.get("end"))
        mark_stage(candidate_id, "final_moments", "selected", index=index)

    for trace_row in trace.values():
        _finalize_trace_row(trace_row)

    return sorted(
        trace.values(),
        key=lambda row: (
            not bool(row.get("final_selected")),
            _none_last(row.get("final_rank")),
            _none_last(row.get("pre_judge_rank")),
            -float(row.get("pre_judge_rank_score") or 0.0),
            str(row.get("candidate_id") or ""),
        ),
    )


def build_candidate_funnel_summary(
    report: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a UI-friendly compact summary from report and trace rows."""

    drop_counts = Counter(str(row.get("dropped_stage") or "unknown") for row in trace_rows if not row.get("final_selected"))
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    final_source_counts: Counter[str] = Counter()
    clip_type_counts: Counter[str] = Counter()
    final_clip_type_counts: Counter[str] = Counter()
    for row in trace_rows:
        source = str(row.get("source") or "unknown")
        source_counts[source] += 1
        clip_type = str(row.get("clip_type") or "general")
        clip_type_counts[clip_type] += 1
        if row.get("final_selected"):
            final_source_counts[source] += 1
            final_clip_type_counts[clip_type] += 1
        reasons = row.get("rejection_reasons") if isinstance(row.get("rejection_reasons"), list) else []
        reason_counts.update(str(reason) for reason in reasons)

    score_fields = [
        "pre_judge_rank_score",
        "judged_rank_score",
        "judge_score",
        "final_rank_score",
    ]
    return {
        "rows": len(trace_rows),
        "final_selected_rows": sum(1 for row in trace_rows if row.get("final_selected")),
        "stage_counts": dict(report.get("stage_counts") or {}),
        "losses": list(report.get("losses") or []),
        "largest_losses": list(report.get("largest_losses") or []),
        "drop_stage_counts": dict(sorted(drop_counts.items())),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "final_source_counts": dict(sorted(final_source_counts.items())),
        "clip_type_counts": dict(sorted(clip_type_counts.items())),
        "final_clip_type_counts": dict(sorted(final_clip_type_counts.items())),
        "score_distributions": {
            field: _distribution([row.get(field) for row in trace_rows])
            for field in score_fields
        },
        "quality_breakdown": _quality_breakdown(trace_rows),
        "ranker_lift": _ranker_lift_summary(trace_rows),
        "clip_type_lift": _clip_type_quality_breakdown(trace_rows),
        "positive_window_recall": dict(report.get("positive_window_recall") or {}),
        "trace_samples": _trace_samples(trace_rows),
    }


def _positive_window_recall_summary(
    positive_windows: Sequence[Mapping[str, Any]],
    *,
    stage_candidates: Mapping[str, Sequence[Candidate]],
    pre_judge_ranked: Sequence[RankedCandidate],
    judged_ranked: Sequence[RankedCandidate],
    quality_gate_rows: Sequence[Mapping[str, Any]],
    final_ranked: Sequence[RankedCandidate],
    final_moments: Sequence[Mapping[str, Any]],
    judge_top_k: int,
    judge_scope_ranked: Sequence[RankedCandidate] | None,
    overlap_ratio: float,
) -> dict[str, Any]:
    threshold = max(min(float(overlap_ratio or 0.0), 1.0), 0.0)
    stage_samples = _positive_recall_stage_samples(
        stage_candidates=stage_candidates,
        pre_judge_ranked=pre_judge_ranked,
        judged_ranked=judged_ranked,
        quality_gate_rows=quality_gate_rows,
        final_ranked=final_ranked,
        final_moments=final_moments,
        judge_top_k=judge_top_k,
        judge_scope_ranked=judge_scope_ranked,
    )
    details: list[dict[str, Any]] = []
    recall_counts: Counter[str] = Counter()
    for index, row in enumerate(positive_windows, start=1):
        positive = _positive_window_sample(row, index)
        if positive is None:
            continue
        matched_stages: list[str] = []
        best_by_stage: dict[str, dict[str, Any] | None] = {}
        for stage in _STAGE_ORDER:
            best = _best_positive_stage_match(positive, stage_samples.get(stage, []))
            if best is not None:
                best_by_stage[stage] = best
                if float(best.get("overlap_ratio") or 0.0) >= threshold:
                    matched_stages.append(stage)
                    recall_counts[stage] += 1
            else:
                best_by_stage[stage] = None

        last_matched_stage = matched_stages[-1] if matched_stages else None
        first_missed_after_match = None
        if last_matched_stage:
            seen_match = False
            for stage in _STAGE_ORDER:
                if stage == last_matched_stage:
                    seen_match = True
                    continue
                if seen_match and stage not in matched_stages:
                    first_missed_after_match = stage
                    break

        details.append(
            {
                **positive,
                "matched": bool(matched_stages),
                "matched_stages": matched_stages,
                "last_matched_stage": last_matched_stage,
                "first_missed_stage_after_last_match": first_missed_after_match,
                "best_by_stage": best_by_stage,
            }
        )

    return {
        "positive_windows": len(details),
        "overlap_ratio": threshold,
        "recall_by_stage": {
            stage: {
                "matched": int(recall_counts.get(stage, 0)),
                "total": len(details),
                "ratio": round(float(recall_counts.get(stage, 0)) / len(details), 6) if details else None,
            }
            for stage in _STAGE_ORDER
        },
        "details": details,
    }


def _positive_recall_stage_samples(
    *,
    stage_candidates: Mapping[str, Sequence[Candidate]],
    pre_judge_ranked: Sequence[RankedCandidate],
    judged_ranked: Sequence[RankedCandidate],
    quality_gate_rows: Sequence[Mapping[str, Any]],
    final_ranked: Sequence[RankedCandidate],
    final_moments: Sequence[Mapping[str, Any]],
    judge_top_k: int,
    judge_scope_ranked: Sequence[RankedCandidate] | None,
) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {
        stage: [_candidate_sample(candidate, stage=stage) for candidate in candidates]
        for stage, candidates in stage_candidates.items()
    }
    samples["ranked_pre_judge"] = [
        {**_ranked_sample(ranked, stage="ranked_pre_judge"), "pre_judge_rank": index}
        for index, ranked in enumerate(pre_judge_ranked, start=1)
    ]
    judge_scope_source = judge_scope_ranked if judge_scope_ranked is not None else pre_judge_ranked
    judge_limit = max(int(judge_top_k or 0), 0)
    samples["judge_scope"] = [
        {**_ranked_sample(ranked, stage="judge_scope"), "judge_scope_rank": index}
        for index, ranked in enumerate(judge_scope_source[:judge_limit], start=1)
    ]
    clip_judge_rows: list[dict[str, Any]] = []
    for index, ranked in enumerate(judged_ranked, start=1):
        row = {**_ranked_sample(ranked, stage="clip_judge"), "judged_rank": index}
        if ranked.judge_result is not None:
            row["judge_score"] = _round_optional(ranked.judge_result.judge_score)
            row["judge_confidence"] = _round_optional(ranked.judge_result.judge_confidence)
            row["judge_provider"] = ranked.judge_result.provider
        clip_judge_rows.append(row)
    samples["clip_judge"] = clip_judge_rows
    samples["quality_gate"] = [
        _quality_row_sample(row, stage="quality_gate")
        for row in quality_gate_rows
        if _row_candidate_id(row)
    ]
    samples["judged_quality_kept"] = [
        _quality_row_sample(row, stage="judged_quality_kept")
        for row in quality_gate_rows
        if (row.get("accepted") is True or row.get("kept") is True or row.get("status") == "kept")
        and _row_has_judge_result(row)
        and _row_candidate_id(row)
    ]
    samples["unjudged_quality_kept"] = [
        _quality_row_sample(row, stage="unjudged_quality_kept")
        for row in quality_gate_rows
        if (row.get("accepted") is True or row.get("kept") is True or row.get("status") == "kept")
        and not _row_has_judge_result(row)
        and _row_candidate_id(row)
    ]
    samples["quality_kept"] = [
        _quality_row_sample(row, stage="quality_kept")
        for row in quality_gate_rows
        if (row.get("accepted") is True or row.get("kept") is True or row.get("status") == "kept")
        and _row_candidate_id(row)
    ]
    samples["final_ranked"] = [
        {**_ranked_sample(ranked, stage="final_ranked"), "final_rank": index}
        for index, ranked in enumerate(final_ranked, start=1)
    ]
    samples["final_moments"] = [
        _final_moment_sample(moment, index)
        for index, moment in enumerate(final_moments, start=1)
    ]
    return samples


def _positive_window_sample(row: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    start = _safe_float(row.get("window_start"))
    end = _safe_float(row.get("window_end"))
    if start is None or end is None or end <= start:
        return None
    return {
        "index": index,
        "positive_id": str(
            row.get("candidate_id")
            or row.get("repost_source_url")
            or row.get("source_url")
            or f"positive:{index}"
        ),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "source_url": str(row.get("repost_source_url") or row.get("source_url") or ""),
        "title": str(row.get("repost_title") or row.get("short_title") or row.get("title") or ""),
        "views": _round_optional(row.get("repost_views")),
        "similarity": _round_optional(row.get("transcript_similarity") or row.get("confidence")),
    }


def _final_moment_sample(moment: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "candidate_id": _base_candidate_id(str(moment.get("candidate_id") or f"final:{index}")),
        "stage": "final_moments",
        "source": str(moment.get("source") or ""),
        "start": _round_optional(moment.get("start_time") if moment.get("start_time") is not None else moment.get("start")),
        "end": _round_optional(moment.get("end_time") if moment.get("end_time") is not None else moment.get("end")),
        "final_moment_index": index,
    }


def _best_positive_stage_match(
    positive: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_ratio = 0.0
    pos_start = _safe_float(positive.get("start"))
    pos_end = _safe_float(positive.get("end"))
    if pos_start is None or pos_end is None or pos_end <= pos_start:
        return None
    for sample in samples:
        start = _safe_float(sample.get("start"))
        end = _safe_float(sample.get("end"))
        if start is None or end is None or end <= start:
            continue
        overlap = max(0.0, min(end, pos_end) - max(start, pos_start))
        if overlap <= 0.0:
            continue
        candidate_duration = max(end - start, 0.001)
        positive_duration = max(pos_end - pos_start, 0.001)
        ratio = max(overlap / candidate_duration, overlap / positive_duration)
        if ratio <= best_ratio:
            continue
        best_ratio = ratio
        best = {
            key: value
            for key, value in sample.items()
            if key
            in {
                "candidate_id",
                "stage",
                "source",
                "start",
                "end",
                "score",
                "model_score",
                "heuristic_score",
                "accepted",
                "rank_score",
                "pre_judge_rank",
                "judge_scope_rank",
                "judged_rank",
                "judge_score",
                "quality_accepted",
                "has_judge_result",
                "rejection_reasons",
                "final_rank",
                "final_moment_index",
            }
        }
        best["overlap_ratio"] = round(ratio, 6)
    return best


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_ids(candidates: Sequence[Candidate]) -> set[str]:
    return {candidate.id for candidate in candidates}


def _base_candidate_id(candidate_id: str) -> str:
    return candidate_id.split("::", 1)[0]


def _row_candidate_id(row: Mapping[str, Any]) -> str:
    value = row.get("candidate_id") or row.get("id")
    if value:
        return _base_candidate_id(str(value))
    candidate = row.get("candidate")
    if isinstance(candidate, Mapping):
        nested = candidate.get("id") or candidate.get("candidate_id")
        if nested:
            return _base_candidate_id(str(nested))
    return ""


def _row_has_judge_result(row: Mapping[str, Any]) -> bool:
    if row.get("has_judge_result") is not None:
        return bool(row.get("has_judge_result"))
    return row.get("judge_score") not in (None, "")


def _source_counts(candidates: Sequence[Candidate]) -> Counter[str]:
    return Counter(candidate.source for candidate in candidates)


def _transition_loss(
    name: str,
    source: str,
    target: str,
    candidate_sets: Mapping[str, set[str]],
    candidate_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_ids = set(candidate_sets.get(source, set()))
    target_ids = set(candidate_sets.get(target, set()))
    retained_ids = source_ids & target_ids
    lost_ids = sorted(source_ids - target_ids)
    input_count = len(source_ids)
    output_count = len(retained_ids)
    return {
        "transition": name,
        "from_stage": source,
        "to_stage": target,
        "input_count": input_count,
        "output_count": output_count,
        "lost_count": len(lost_ids),
        "retention_ratio": round(output_count / input_count, 6) if input_count else None,
        "lost_samples": [
            dict(candidate_lookup.get(candidate_id, {"candidate_id": candidate_id}))
            for candidate_id in lost_ids[:_SAMPLE_LIMIT]
        ],
        "target_total_count": len(target_ids),
    }


def _candidate_sample(candidate: Candidate, *, stage: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "stage": stage,
        "source": candidate.source,
        "start": round(float(candidate.start), 3),
        "end": round(float(candidate.end), 3),
        "anchor_time": round(float(candidate.anchor_time), 3),
        "score_hint": round(float(candidate.score_hint), 6),
    }


def _ranked_sample(ranked: RankedCandidate, *, stage: str) -> dict[str, Any]:
    sample = _candidate_sample(ranked.candidate, stage=stage)
    sample["rank_score"] = round(float(ranked.rank_score), 6)
    sample["trained_model_used"] = bool(ranked.trained_model_used)
    return sample


def _quality_row_sample(row: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    return {
        "candidate_id": _row_candidate_id(row),
        "stage": stage,
        "source": str(row.get("source") or ""),
        "start": _round_optional(row.get("start")),
        "end": _round_optional(row.get("end")),
        "rank_score": _round_optional(row.get("rank_score")),
        "judge_score": _round_optional(row.get("judge_score")),
        "has_judge_result": _row_has_judge_result(row),
        "accepted": bool(row.get("accepted") is True or row.get("kept") is True),
        "rejection_reasons": list(row.get("rejection_reasons") or []),
    }


def _quality_rejection_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        reasons = row.get("rejection_reasons") or []
        if not reasons:
            counts["unspecified"] += 1
            continue
        counts.update(str(reason) for reason in reasons)
    return counts


def _merge_candidate_identity(row: dict[str, Any], candidate: Candidate) -> None:
    row["source"] = row.get("source") or candidate.source
    row["start"] = row.get("start") if row.get("start") is not None else _round_optional(candidate.start)
    row["end"] = row.get("end") if row.get("end") is not None else _round_optional(candidate.end)
    if row.get("duration") is None:
        row["duration"] = _round_optional(float(candidate.end) - float(candidate.start))
    row["score_hint"] = row.get("score_hint") if row.get("score_hint") is not None else _round_optional(candidate.score_hint)
    if not row.get("tags"):
        row["tags"] = list(candidate.tags)


def _merge_sample_identity(row: dict[str, Any], sample: Mapping[str, Any]) -> None:
    row["source"] = row.get("source") or str(sample.get("source") or "")
    row["start"] = row.get("start") if row.get("start") is not None else sample.get("start")
    row["end"] = row.get("end") if row.get("end") is not None else sample.get("end")
    if row.get("duration") is None and row.get("start") is not None and row.get("end") is not None:
        row["duration"] = _round_optional(float(row["end"]) - float(row["start"]))
    if not row.get("clip_type") and sample.get("clip_type"):
        row["clip_type"] = sample.get("clip_type")


def _finalize_trace_row(row: dict[str, Any]) -> None:
    stages = row.get("stages") if isinstance(row.get("stages"), dict) else {}
    ordered_stages = [
        {"stage": stage, **dict(stages[stage])}
        for stage in _STAGE_ORDER
        if stage in stages
    ]
    row["stage_path"] = ordered_stages
    row["first_seen_stage"] = ordered_stages[0]["stage"] if ordered_stages else None
    row["last_seen_stage"] = ordered_stages[-1]["stage"] if ordered_stages else None
    row["rejection_reasons"] = _dedupe_strings(row.get("rejection_reasons") or [])

    if row.get("final_selected"):
        row["status"] = "final_selected"
        row["dropped_stage"] = None
        row["drop_reason"] = None
        return

    if stages.get("quality_gate", {}).get("status") == "rejected":
        row["status"] = "rejected"
        row["dropped_stage"] = "quality_gate"
        row["drop_reason"] = ", ".join(row.get("quality_rejection_reasons") or []) or "quality_gate_rejected"
    elif stages.get("judge_scope", {}).get("status") == "excluded":
        row["status"] = "rejected"
        row["dropped_stage"] = "judge_scope"
        row["drop_reason"] = "outside_judge_scope"
    elif "ranked_pre_judge" not in stages and "normalized" in stages:
        row["status"] = "rejected"
        row["dropped_stage"] = "ranker"
        row["drop_reason"] = "not_ranked"
    elif "normalized" not in stages and "merged" in stages:
        row["status"] = "rejected"
        row["dropped_stage"] = "normalization"
        row["drop_reason"] = "not_normalized"
    elif "final_ranked" in stages:
        row["status"] = "kept_until_final_ranked"
        row["dropped_stage"] = "final_selection"
        row["drop_reason"] = "not_selected_as_final_moment"
    else:
        row["status"] = "incomplete_trace"
        row["dropped_stage"] = row.get("last_seen_stage")
        row["drop_reason"] = "trace_ended_before_final"


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _distribution(values: Sequence[Any]) -> dict[str, float | int | None]:
    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return {"count": 0, "min": None, "mean": None, "max": None}
    numeric.sort()
    return {
        "count": len(numeric),
        "min": round(numeric[0], 6),
        "mean": round(sum(numeric) / len(numeric), 6),
        "max": round(numeric[-1], 6),
    }


def _trace_samples(trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    final_rows = [row for row in trace_rows if row.get("final_selected")]
    rejected_rows = [row for row in trace_rows if not row.get("final_selected")]
    return {
        "final": [_compact_trace_sample(row) for row in final_rows[:_TRACE_SAMPLE_LIMIT]],
        "highest_ranked_rejected": [
            _compact_trace_sample(row)
            for row in sorted(
                rejected_rows,
                key=lambda row: (
                    _none_last(row.get("pre_judge_rank")),
                    -float(row.get("pre_judge_rank_score") or 0.0),
                    str(row.get("candidate_id") or ""),
                ),
            )[:_TRACE_SAMPLE_LIMIT]
        ],
        "quality_rejected": [
            _compact_trace_sample(row)
            for row in rejected_rows
            if row.get("dropped_stage") == "quality_gate"
        ][:_TRACE_SAMPLE_LIMIT],
    }


def _compact_trace_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id"),
        "source": row.get("source"),
        "start": row.get("start"),
        "end": row.get("end"),
        "status": row.get("status"),
        "clip_type": row.get("clip_type"),
        "dropped_stage": row.get("dropped_stage"),
        "drop_reason": row.get("drop_reason"),
        "pre_judge_rank": row.get("pre_judge_rank"),
        "pre_judge_rank_score": row.get("pre_judge_rank_score"),
        "judge_score": row.get("judge_score"),
        "quality_has_judge_result": row.get("quality_has_judge_result"),
        "final_rank": row.get("final_rank"),
    }


def _quality_breakdown(trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    judged_kept = 0
    unjudged_kept = 0
    judged_rejected = 0
    unjudged_rejected = 0
    for row in trace_rows:
        stages = row.get("stages") if isinstance(row.get("stages"), Mapping) else {}
        quality = stages.get("quality_gate") if isinstance(stages.get("quality_gate"), Mapping) else None
        if quality is None:
            continue
        has_judge = bool(row.get("quality_has_judge_result"))
        if quality.get("status") == "kept":
            if has_judge:
                judged_kept += 1
            else:
                unjudged_kept += 1
        elif quality.get("status") == "rejected":
            if has_judge:
                judged_rejected += 1
            else:
                unjudged_rejected += 1
    return {
        "judged_kept": judged_kept,
        "unjudged_kept": unjudged_kept,
        "judged_rejected": judged_rejected,
        "unjudged_rejected": unjudged_rejected,
    }


def _ranker_lift_summary(trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    high_judge_rows = [
        row for row in trace_rows
        if _optional_float(row.get("judge_score"), 0.0) >= 7.0
    ]
    top20 = sum(
        1 for row in high_judge_rows
        if _optional_float(row.get("pre_judge_rank"), 10**9) <= 20
    )
    top40 = sum(
        1 for row in high_judge_rows
        if _optional_float(row.get("pre_judge_rank"), 10**9) <= 40
    )
    final_rows = [row for row in trace_rows if row.get("final_selected")]
    return {
        "high_judge_rows": len(high_judge_rows),
        "high_judge_pre_judge_top20_ratio": _ratio(top20, len(high_judge_rows)),
        "high_judge_pre_judge_top40_ratio": _ratio(top40, len(high_judge_rows)),
        "high_judge_pre_judge_rank": _distribution([row.get("pre_judge_rank") for row in high_judge_rows]),
        "final_selected_pre_judge_rank": _distribution([row.get("pre_judge_rank") for row in final_rows]),
        "final_selected_judge_score": _distribution([row.get("judge_score") for row in final_rows]),
    }


def _clip_type_quality_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    by_type: dict[str, dict[str, int | float | None]] = {}
    for row in rows:
        clip_type = str(row.get("clip_type") or "general")
        bucket = by_type.setdefault(
            clip_type,
            {
                "rows": 0,
                "final_selected": 0,
                "high_judge": 0,
            },
        )
        bucket["rows"] = int(bucket["rows"] or 0) + 1
        if row.get("final_selected"):
            bucket["final_selected"] = int(bucket["final_selected"] or 0) + 1
        if _optional_float(row.get("judge_score"), 0.0) >= 7.0:
            bucket["high_judge"] = int(bucket["high_judge"] or 0) + 1
    return dict(sorted(by_type.items()))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _optional_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _none_last(value: Any) -> tuple[int, float]:
    if value is None:
        return (1, 0.0)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, 0.0)


def _round_optional(value: Any) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None
