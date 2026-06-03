from __future__ import annotations

import json
import pathlib
from urllib.parse import urlparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from collections.abc import Mapping

from models import ClipFeedbackPayload, ClipFeedbackResponse

from services.gold_eval_policy import annotate_gold_row, validate_gold_notes
from services.timeline_types import RankedCandidate

# Candidates with rank_score below this threshold are labelled "low_rank".
# Those above it were presumably decent but filtered by the LLM or diversity
# enforcement — a fundamentally different failure mode for training purposes.
_LOW_RANK_THRESHOLD = 3.0


@dataclass
class CandidateFeedback:
    job_id: str
    candidate_id: str
    selected: bool
    exported: bool
    rank_score: float = 0.0         # heuristic rank score from ranker.py
    proposed_start: float | None = None
    proposed_end: float | None = None
    final_start: float | None = None
    final_end: float | None = None
    rejected_reason: str | None = None   # kept for schema compatibility
    rejection_source: str | None = None  # richer label for training
    feedback_rating: str | None = None
    feedback_tags: list[str] = field(default_factory=list)
    feedback_notes: str | None = None
    publish_platform: str | None = None
    final_variant_key: str | None = None
    judge_score: float | None = None
    standalone_clarity: float | None = None
    payoff_strength: float | None = None
    narrative_completeness: float | None = None
    judge_confidence: float | None = None
    judge_provider: str | None = None
    judge_reason: str | None = None
    updated_at: int | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["feedback_tags"] = list(payload.get("feedback_tags") or [])
        return payload


def build_pipeline_feedback(
    job_id: str,
    ranked_candidates: list[RankedCandidate],
    final_moments: list[dict],
    quality_gate_rows: list[Mapping[str, object]] | None = None,
) -> list[CandidateFeedback]:
    feedback_by_id: dict[str, CandidateFeedback] = {}
    quality_rows_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in (quality_gate_rows or [])
        if str(row.get("candidate_id") or "").strip()
    }
    for ranked in ranked_candidates:
        selected_moment = _matching_moment(ranked.candidate.start, ranked.candidate.end, final_moments)
        selected = selected_moment is not None
        quality_row = quality_rows_by_id.get(ranked.candidate.id)

        if selected:
            rejected_reason = None
            rejection_source = None
        elif ranked.rank_score < _LOW_RANK_THRESHOLD:
            # Ranked poorly — likely a genuinely weak candidate
            rejected_reason = "low_rank"
            rejection_source = "low_rank"
        else:
            # Scored reasonably but not in the final output — LLM refinement or
            # diversity enforcement dropped it.  Training should treat this as a
            # weak negative (or exclude from training) rather than a hard negative.
            rejected_reason = "post_processing_filtered"
            rejection_source = "post_processing_filtered"

        feedback_by_id[ranked.candidate.id] = _feedback_row(
            job_id=job_id,
            candidate_id=ranked.candidate.id,
            selected=selected,
            rank_score=ranked.rank_score,
            proposed_start=ranked.candidate.start,
            proposed_end=ranked.candidate.end,
            selected_moment=selected_moment,
            rejected_reason=rejected_reason,
            rejection_source=rejection_source,
            quality_row=quality_row,
        )

    for candidate_id, quality_row in quality_rows_by_id.items():
        if not _quality_row_has_judge_result(quality_row):
            continue
        existing = feedback_by_id.get(candidate_id)
        if existing is not None:
            _apply_judge_fields(existing, quality_row)
            continue

        start = _as_float(quality_row.get("start"))
        end = _as_float(quality_row.get("end"))
        selected_moment = _matching_moment(start, end, final_moments)
        selected = selected_moment is not None
        accepted = bool(quality_row.get("accepted") is True or quality_row.get("kept") is True)
        reasons = [str(reason) for reason in quality_row.get("rejection_reasons") or []]
        if selected:
            rejected_reason = None
            rejection_source = None
        elif accepted:
            rejected_reason = "judge_quality_accepted"
            rejection_source = "judge_quality_accepted"
        else:
            rejected_reason = ", ".join(reasons) or "judge_quality_rejected"
            rejection_source = "judge_quality_rejected"

        feedback_by_id[candidate_id] = _feedback_row(
            job_id=job_id,
            candidate_id=candidate_id,
            selected=selected,
            rank_score=_as_float(quality_row.get("rank_score")),
            proposed_start=start,
            proposed_end=end,
            selected_moment=selected_moment,
            rejected_reason=rejected_reason,
            rejection_source=rejection_source,
            quality_row=quality_row,
        )

    return list(feedback_by_id.values())


def append_feedback_events(storage_dir: pathlib.Path, events: list[CandidateFeedback]) -> None:
    feedback_dir = storage_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    events_file = feedback_dir / "events.jsonl"
    with events_file.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")


def write_job_feedback(debug_dir: pathlib.Path, events: list[CandidateFeedback]) -> None:
    path = debug_dir / "feedback.jsonl"
    path.write_text(
        "\n".join(json.dumps(event.to_dict(), ensure_ascii=True) for event in events),
        encoding="utf-8",
    )


def update_job_feedback(
    storage_dir: pathlib.Path,
    job_id: str,
    payload: ClipFeedbackPayload,
) -> ClipFeedbackResponse:
    if payload.mode == "gold_eval":
        return update_job_gold_eval(storage_dir, job_id, payload)

    debug_dir = storage_dir / job_id / "debug"
    path = debug_dir / "feedback.jsonl"
    rows = _read_feedback_rows(path)
    target = next((row for row in rows if row.get("candidate_id") == payload.candidate_id), None)
    if target is None:
        raise KeyError(payload.candidate_id)

    target["feedback_rating"] = payload.rating
    target["feedback_tags"] = list(payload.feedback_tags or [])
    target["feedback_notes"] = payload.notes
    target["updated_at"] = int(datetime.now(timezone.utc).timestamp())

    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )
    append_feedback_snapshot(storage_dir, target)
    return ClipFeedbackResponse(
        job_id=job_id,
        candidate_id=payload.candidate_id,
        mode="training_feedback",
        rating=target.get("feedback_rating"),
        feedback_tags=list(target.get("feedback_tags") or []),
        notes=target.get("feedback_notes"),
    )


def update_job_gold_eval(
    storage_dir: pathlib.Path,
    job_id: str,
    payload: ClipFeedbackPayload,
) -> ClipFeedbackResponse:
    debug_dir = storage_dir / job_id / "debug"
    features_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in _read_feedback_rows(debug_dir / "candidate_features.jsonl")
        if str(row.get("candidate_id") or "").strip()
    }
    feature_row = features_by_id.get(payload.candidate_id)
    if feature_row is None:
        raise KeyError(payload.candidate_id)

    windows_by_id = {
        str(row.get("id") or "").strip(): row
        for row in _read_feedback_rows(debug_dir / "candidates_normalized.jsonl")
        if str(row.get("id") or "").strip()
    }
    window_row = windows_by_id.get(payload.candidate_id) or {}
    heuristic_rank_score = _heuristic_rank_score(debug_dir, payload.candidate_id)
    validate_gold_notes(
        int(payload.gold_grade or 0),
        payload.notes,
        heuristic_rank_score=heuristic_rank_score,
    )
    metadata = _job_metadata(storage_dir / job_id)
    start = _as_float(window_row.get("start"))
    end = _as_float(window_row.get("end"))
    label = 1 if int(payload.gold_grade or 0) >= 2 else 0
    row = annotate_gold_row(
        {
        "queue_id": f"{job_id}:{payload.candidate_id}",
        "job_id": job_id,
        "candidate_id": payload.candidate_id,
        "vod_id": metadata["vod_id"],
        "creator_id": metadata["creator_id"],
        "publish_platform": metadata["platform"],
        "source_url": metadata["source_url"],
        "window_start": start,
        "window_end": end,
        "label": label,
        "ordinal_label": int(payload.gold_grade or 0),
        "gold_grade": int(payload.gold_grade or 0),
        "gold_notes": payload.notes,
        "label_source": "gold_eval",
        "label_class": "gold_positive" if label == 1 else "gold_negative",
        "sample_weight": 1.0,
        "supervision_tier": "gold_eval",
        "supervision_reason": "frontend_gold_eval",
        "dataset_source": "gold_eval_frontend",
        "selected": bool(label),
        "exported": False,
        "candidate_source": str(feature_row.get("source") or "").strip(),
        "heuristic_rank_score": heuristic_rank_score,
        "features": dict(feature_row),
        "updated_at": int(datetime.now(timezone.utc).timestamp()),
        }
    )
    path = debug_dir / "gold_eval.jsonl"
    rows = _read_feedback_rows(path)
    rows = [existing for existing in rows if existing.get("candidate_id") != payload.candidate_id]
    rows.append(row)
    path.write_text(
        "\n".join(json.dumps(existing, ensure_ascii=True) for existing in rows),
        encoding="utf-8",
    )
    append_gold_eval_snapshot(storage_dir, row)
    return ClipFeedbackResponse(
        job_id=job_id,
        candidate_id=payload.candidate_id,
        mode="gold_eval",
        gold_grade=int(payload.gold_grade or 0),
        feedback_tags=list(payload.feedback_tags or []),
        notes=payload.notes,
    )


def append_feedback_snapshot(storage_dir: pathlib.Path, feedback_row: dict) -> None:
    feedback_dir = storage_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    snapshots_file = feedback_dir / "feedback_updates.jsonl"
    with snapshots_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(feedback_row, ensure_ascii=True) + "\n")


def append_gold_eval_snapshot(storage_dir: pathlib.Path, gold_eval_row: dict) -> None:
    feedback_dir = storage_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    snapshots_file = feedback_dir / "gold_eval_updates.jsonl"
    with snapshots_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(gold_eval_row, ensure_ascii=True) + "\n")


def _read_feedback_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _heuristic_rank_score(debug_dir: pathlib.Path, candidate_id: str) -> float | None:
    ranked_paths = (
        debug_dir / "ranked_candidates_pre_judge.jsonl",
        debug_dir / "ranked_candidates.jsonl",
    )
    for path in ranked_paths:
        for row in _read_feedback_rows(path):
            candidate = row.get("candidate") or {}
            row_candidate_id = str(candidate.get("id") or row.get("candidate_id") or "").strip()
            if row_candidate_id != candidate_id:
                continue
            value = row.get("heuristic_rank_score")
            if value not in (None, ""):
                return _as_float(value)
    return None


def _job_metadata(job_dir: pathlib.Path) -> dict[str, str]:
    request_path = job_dir / "request.json"
    payload: dict = {}
    if request_path.exists():
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    source_url = str(payload.get("url") or "").strip()
    platform = str(payload.get("platform") or "").strip().lower()
    profile = str(payload.get("profile") or "").strip().lower()
    return {
        "source_url": source_url,
        "platform": platform or _platform_from_url(source_url),
        "vod_id": _vod_id_from_url(source_url),
        "creator_id": _creator_id_from_url(source_url) or profile,
    }


def _platform_from_url(source_url: str) -> str:
    host = urlparse(source_url).netloc.lower()
    if "twitch.tv" in host:
        return "twitch"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "tiktok.com" in host:
        return "tiktok"
    if "instagram.com" in host:
        return "instagram_reels"
    return "general"


def _vod_id_from_url(source_url: str) -> str:
    parts = [part for part in urlparse(source_url).path.split("/") if part]
    if len(parts) >= 2 and parts[-2] == "videos":
        return parts[-1]
    if parts:
        return parts[-1]
    return ""


def _creator_id_from_url(source_url: str) -> str:
    parts = [part for part in urlparse(source_url).path.split("/") if part]
    if len(parts) >= 2 and parts[-2] == "videos":
        return ""
    if parts and parts[0].startswith("@"):
        return parts[0][1:]
    if parts:
        return parts[0]
    return ""


def _as_float(value) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _matching_moment(start: float, end: float, moments: list[dict]) -> dict | None:
    for moment in moments:
        overlap = min(end, moment["end_time"]) - max(start, moment["start_time"])
        if overlap <= 0:
            continue
        shorter = min(end - start, moment["end_time"] - moment["start_time"])
        if shorter > 0 and overlap / shorter >= 0.4:
            return moment
    return None


def _feedback_row(
    *,
    job_id: str,
    candidate_id: str,
    selected: bool,
    rank_score: float,
    proposed_start: float,
    proposed_end: float,
    selected_moment: dict | None,
    rejected_reason: str | None,
    rejection_source: str | None,
    quality_row: Mapping[str, object] | None,
) -> CandidateFeedback:
    row = CandidateFeedback(
        job_id=job_id,
        candidate_id=candidate_id,
        selected=selected,
        exported=False,
        rank_score=round(float(rank_score), 4),
        proposed_start=proposed_start,
        proposed_end=proposed_end,
        final_start=selected_moment["start_time"] if selected_moment else None,
        final_end=selected_moment["end_time"] if selected_moment else None,
        rejected_reason=rejected_reason,
        rejection_source=rejection_source,
        publish_platform=selected_moment.get("platform") if selected_moment else None,
        final_variant_key=selected_moment.get("variant_key") if selected_moment else None,
        updated_at=int(datetime.now(timezone.utc).timestamp()),
    )
    if quality_row is not None:
        _apply_judge_fields(row, quality_row)
    return row


def _apply_judge_fields(feedback: CandidateFeedback, quality_row: Mapping[str, object]) -> None:
    feedback.judge_score = _optional_float(quality_row.get("judge_score"))
    feedback.standalone_clarity = _optional_float(quality_row.get("standalone_clarity"))
    feedback.payoff_strength = _optional_float(quality_row.get("payoff_strength"))
    feedback.narrative_completeness = _optional_float(quality_row.get("narrative_completeness"))
    feedback.judge_confidence = _optional_float(quality_row.get("judge_confidence"))
    feedback.judge_provider = str(quality_row.get("judge_provider") or "") or None
    feedback.judge_reason = str(quality_row.get("judge_reason") or "") or None


def _quality_row_has_judge_result(row: Mapping[str, object]) -> bool:
    if row.get("has_judge_result") is not None:
        return bool(row.get("has_judge_result"))
    return row.get("judge_score") not in (None, "")


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
