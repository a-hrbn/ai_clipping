from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from models import ClipFeedbackPayload, ClipFeedbackResponse

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
    user_feedback_tags: list[str] | None = None
    feedback_notes: str | None = None
    publish_platform: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    share_count: int | None = None
    save_count: int | None = None
    impressions: int | None = None
    avg_watch_time: float | None = None
    retention_3s: float | None = None
    retention_50pct: float | None = None
    completion_rate: float | None = None
    views_1h: int | None = None
    views_24h: int | None = None
    baseline_view_rate: float | None = None
    observed_view_rate: float | None = None
    final_variant_key: str | None = None
    updated_at: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_pipeline_feedback(
    job_id: str,
    ranked_candidates: list[RankedCandidate],
    final_moments: list[dict],
) -> list[CandidateFeedback]:
    feedback: list[CandidateFeedback] = []
    for ranked in ranked_candidates:
        selected_moment = _matching_moment(ranked.candidate.start, ranked.candidate.end, final_moments)
        selected = selected_moment is not None

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

        feedback.append(
            CandidateFeedback(
                job_id=job_id,
                candidate_id=ranked.candidate.id,
                selected=selected,
                exported=False,
                rank_score=round(ranked.rank_score, 4),
                proposed_start=ranked.candidate.start,
                proposed_end=ranked.candidate.end,
                final_start=selected_moment["start_time"] if selected_moment else None,
                final_end=selected_moment["end_time"] if selected_moment else None,
                rejected_reason=rejected_reason,
                rejection_source=rejection_source,
                user_feedback_tags=[],
                publish_platform=selected_moment.get("platform") if selected_moment else None,
                final_variant_key=selected_moment.get("variant_key") if selected_moment else None,
                updated_at=int(datetime.now(timezone.utc).timestamp()),
            )
        )
    return feedback


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
    debug_dir = storage_dir / job_id / "debug"
    path = debug_dir / "feedback.jsonl"
    rows = _read_feedback_rows(path)
    target = next((row for row in rows if row.get("candidate_id") == payload.candidate_id), None)
    if target is None:
        raise KeyError(payload.candidate_id)

    target["user_feedback_tags"] = sorted(set(payload.tags))
    target["feedback_notes"] = payload.notes
    target["publish_platform"] = payload.publish_platform
    target["view_count"] = payload.view_count
    target["like_count"] = payload.like_count
    target["comment_count"] = payload.comment_count
    target["share_count"] = payload.share_count
    target["save_count"] = payload.save_count
    target["impressions"] = payload.impressions
    target["avg_watch_time"] = payload.avg_watch_time
    target["retention_3s"] = payload.retention_3s
    target["retention_50pct"] = payload.retention_50pct
    target["completion_rate"] = payload.completion_rate
    target["views_1h"] = payload.views_1h
    target["views_24h"] = payload.views_24h
    target["baseline_view_rate"] = payload.baseline_view_rate
    target["observed_view_rate"] = payload.observed_view_rate
    target["updated_at"] = int(datetime.now(timezone.utc).timestamp())

    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )
    append_feedback_snapshot(storage_dir, target)
    return ClipFeedbackResponse(
        job_id=job_id,
        candidate_id=payload.candidate_id,
        tags=list(target.get("user_feedback_tags") or []),
        notes=target.get("feedback_notes"),
        publish_platform=target.get("publish_platform"),
        view_count=target.get("view_count"),
        like_count=target.get("like_count"),
        comment_count=target.get("comment_count"),
        share_count=target.get("share_count"),
        save_count=target.get("save_count"),
        impressions=target.get("impressions"),
        avg_watch_time=target.get("avg_watch_time"),
        retention_3s=target.get("retention_3s"),
        retention_50pct=target.get("retention_50pct"),
        completion_rate=target.get("completion_rate"),
        views_1h=target.get("views_1h"),
        views_24h=target.get("views_24h"),
        baseline_view_rate=target.get("baseline_view_rate"),
        observed_view_rate=target.get("observed_view_rate"),
    )


def append_feedback_snapshot(storage_dir: pathlib.Path, feedback_row: dict) -> None:
    feedback_dir = storage_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    snapshots_file = feedback_dir / "feedback_updates.jsonl"
    with snapshots_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(feedback_row, ensure_ascii=True) + "\n")


def _read_feedback_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _matching_moment(start: float, end: float, moments: list[dict]) -> dict | None:
    for moment in moments:
        overlap = min(end, moment["end_time"]) - max(start, moment["start_time"])
        if overlap <= 0:
            continue
        shorter = min(end - start, moment["end_time"] - moment["start_time"])
        if shorter > 0 and overlap / shorter >= 0.4:
            return moment
    return None
