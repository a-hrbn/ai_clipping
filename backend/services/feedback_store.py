from __future__ import annotations

import json
import pathlib
from urllib.parse import urlparse
from datetime import datetime, timezone

from models import ClipFeedbackPayload, ClipFeedbackResponse

from services.gold_eval_policy import annotate_gold_row, validate_gold_notes

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
    if feature_row is not None:
        windows_by_id = {
            str(row.get("id") or "").strip(): row
            for row in _read_feedback_rows(debug_dir / "candidates_normalized.jsonl")
            if str(row.get("id") or "").strip()
        }
        window_row = windows_by_id.get(payload.candidate_id) or {}
    else:
        # Legacy candidate artifacts are gone since the old ML services were
        # removed: fall back to the clip's feedback row (written by Clipping).
        clip_row = next(
            (row for row in _read_feedback_rows(debug_dir / "feedback.jsonl")
             if row.get("candidate_id") == payload.candidate_id),
            None,
        )
        if clip_row is None:
            raise KeyError(payload.candidate_id)
        feature_row = {
            key: value for key, value in clip_row.items()
            if key == "rank" or key.endswith("_score")
        }
        feature_row["source"] = clip_row.get("ranking_mode")
        window_row = {"start": clip_row.get("final_start"), "end": clip_row.get("final_end")}
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
