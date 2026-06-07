from __future__ import annotations

import datetime
import json
import pathlib
import threading
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ml import build_llm_moment_precision_review as precision_review
from ml import run_llm_moment_finder_d0 as d0
from ml import run_serving_recall_holdout as holdout


router = APIRouter(prefix="/api/llm-moment-review", tags=["llm-moment-review"])

WorkflowStatus = Literal["queued", "running", "done", "failed"]
WorkflowStage = Literal["queued", "moment_finder", "precision_review", "done", "failed"]

WORKFLOW_ROOT = d0.DEFAULT_ARTIFACT_DIR / "llm_moment_review_workflows"
DEFAULT_WORKFLOW_NAME = "jynxzi_llm_moment_finder_d0"
LABELS_FILENAME = "review_labels.json"
LABEL_EVENTS_FILENAME = "review_label_events.jsonl"


class LlmMomentReviewStartRequest(BaseModel):
    profile: str = "jynxzi"
    vod_ids: list[str] = Field(default_factory=list)
    vod_limit: int = Field(default=3, ge=1, le=10)
    top_n: int = Field(default=d0.DEFAULT_TOP_N, ge=1, le=500)
    max_moments: int = Field(default=d0.DEFAULT_MAX_MOMENTS, ge=1, le=500)
    max_tokens: int = Field(default=d0.DEFAULT_MAX_TOKENS, ge=512, le=64000)
    provider: Literal["gemini", "anthropic"] = d0.DEFAULT_PROVIDER  # type: ignore[assignment]
    model: str = ""
    top_n_per_vod: int = Field(default=precision_review.DEFAULT_TOP_N_PER_VOD, ge=1, le=80)
    max_duration_sec: float = Field(default=precision_review.DEFAULT_MAX_DURATION_SEC, ge=1.0, le=300.0)
    exclude_known_reposts: bool = False
    phase2_min_non_repost_precision: float = Field(default=0.5, ge=0.0, le=1.0)
    cut_clips: bool = True
    allow_holdout: bool = False
    dry_run: bool = False


class LlmMomentReviewLabelRequest(BaseModel):
    review_id: str
    human_viral_label: Literal["yes", "no", "unsure", ""] = ""
    human_boundary_label: Literal["good", "too_wide", "too_short", "wrong", "unsure", ""] = ""
    human_notes: str = ""


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def workflow_id_now() -> str:
    return d0.resolve_artifact_run_id("")


def workflow_dir(workflow_id: str) -> pathlib.Path:
    safe_id = d0.safe_artifact_name(workflow_id)
    if not safe_id or safe_id != workflow_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    root = WORKFLOW_ROOT.resolve()
    path = (root / safe_id).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return path


def status_path(root: pathlib.Path) -> pathlib.Path:
    return root / "workflow.json"


def labels_path(root: pathlib.Path) -> pathlib.Path:
    return root / "review" / LABELS_FILENAME


def label_events_path(root: pathlib.Path) -> pathlib.Path:
    return root / "review" / LABEL_EVENTS_FILENAME


def review_jsonl_path(root: pathlib.Path) -> pathlib.Path:
    return root / "review" / "precision_review.jsonl"


def review_summary_path(root: pathlib.Path) -> pathlib.Path:
    return root / "review" / "precision_review.summary.json"


def read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError:
        return rows
    return rows


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def update_status(
    workflow_id: str,
    *,
    status: WorkflowStatus,
    stage: WorkflowStage,
    progress_pct: int,
    message: str,
    error: str | None = None,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = workflow_dir(workflow_id)
    current = read_json(status_path(root))
    payload = {
        **current,
        "workflow_id": workflow_id,
        "status": status,
        "stage": stage,
        "progress_pct": max(0, min(int(progress_pct), 100)),
        "message": message,
        "error": error,
        "updated_at": utc_now(),
    }
    if "created_at" not in payload:
        payload["created_at"] = payload["updated_at"]
    if patch:
        payload.update(patch)
    write_json(status_path(root), payload)
    return payload


@router.post("/workflows")
async def start_llm_moment_review(request: LlmMomentReviewStartRequest) -> dict[str, Any]:
    workflow_id = workflow_id_now()
    root = workflow_dir(workflow_id)
    root.mkdir(parents=True, exist_ok=True)
    payload = request.model_dump()
    initial = update_status(
        workflow_id,
        status="queued",
        stage="queued",
        progress_pct=0,
        message="Workflow queued.",
        patch={"request": payload, "workflow_dir": str(root)},
    )
    thread = threading.Thread(
        target=run_workflow,
        args=(workflow_id, payload),
        name=f"llm-moment-review-{workflow_id}",
        daemon=True,
    )
    thread.start()
    return hydrate_workflow_response(read_json(status_path(root)) or initial)


@router.get("/workflows")
async def list_llm_moment_review_workflows(limit: int = 20) -> dict[str, Any]:
    safe_limit = min(max(int(limit or 0), 1), 100)
    rows: list[dict[str, Any]] = []
    if WORKFLOW_ROOT.exists():
        for path in sorted(WORKFLOW_ROOT.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_dir():
                continue
            status = read_json(status_path(path))
            if not status:
                continue
            rows.append(hydrate_workflow_response(status, include_rows=False))
            if len(rows) >= safe_limit:
                break
    return {"workflows": rows}


@router.get("/workflows/{workflow_id}")
async def get_llm_moment_review_workflow(workflow_id: str) -> dict[str, Any]:
    root = workflow_dir(workflow_id)
    status = read_json(status_path(root))
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return hydrate_workflow_response(status)


@router.post("/workflows/{workflow_id}/labels")
async def save_llm_moment_review_label(workflow_id: str, request: LlmMomentReviewLabelRequest) -> dict[str, Any]:
    root = workflow_dir(workflow_id)
    if not status_path(root).exists():
        raise HTTPException(status_code=404, detail="Workflow not found")
    review_id = str(request.review_id or "").strip()
    if not review_id:
        raise HTTPException(status_code=400, detail="review_id is required")
    rows = read_review_rows(root)
    if review_id not in {str(row.get("review_id") or "") for row in rows}:
        raise HTTPException(status_code=404, detail="Review item not found")

    labels = read_json(labels_path(root))
    label = {
        "review_id": review_id,
        "human_viral_label": request.human_viral_label,
        "human_boundary_label": request.human_boundary_label,
        "human_notes": request.human_notes.strip(),
        "updated_at": utc_now(),
    }
    labels[review_id] = label
    write_json(labels_path(root), labels)
    with label_events_path(root).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label, ensure_ascii=True, sort_keys=True) + "\n")
    return {"workflow_id": workflow_id, "label": label}


@router.post("/workflows/{workflow_id}/recall-floor/latest")
async def attach_latest_recall_floor(workflow_id: str) -> dict[str, Any]:
    root = workflow_dir(workflow_id)
    status = read_json(status_path(root))
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    recall_floor = recall_floor_from_holdout_summary(holdout.DEFAULT_RUN_SUMMARY)
    if recall_floor.get("available") is not True:
        raise HTTPException(status_code=400, detail=recall_floor.get("reason") or "No valid LLM holdout summary found")
    update_status(
        workflow_id,
        status=str(status.get("status") or "done"),  # type: ignore[arg-type]
        stage=str(status.get("stage") or "done"),  # type: ignore[arg-type]
        progress_pct=int(status.get("progress_pct") or 100),
        message=str(status.get("message") or "Review ready."),
        patch={"recall_floor": recall_floor},
    )
    return hydrate_workflow_response(read_json(status_path(root)))


@router.get("/workflows/{workflow_id}/clips/{clip_path:path}")
def serve_llm_moment_review_clip(workflow_id: str, clip_path: str) -> FileResponse:
    root = workflow_dir(workflow_id)
    review_root = (root / "review").resolve()
    path = (review_root / clip_path).resolve()
    try:
        path.relative_to(review_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    if not path.is_file() or path.suffix.lower() not in {".mp4", ".webm", ".mov", ".m4v", ".mkv"}:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


def run_workflow(workflow_id: str, request: dict[str, Any]) -> None:
    root = workflow_dir(workflow_id)
    moment_dir = root / "moment_finder"
    review_dir = root / "review"
    output_prefix = moment_dir / DEFAULT_WORKFLOW_NAME
    selected_vod_ids = [str(vod_id).strip() for vod_id in request.get("vod_ids") or [] if str(vod_id).strip()]
    vod_id_arg = ",".join(selected_vod_ids)
    try:
        update_status(
            workflow_id,
            status="running",
            stage="moment_finder",
            progress_pct=5,
            message="Running LLM moment finder.",
        )
        provider = d0.normalize_provider(str(request.get("provider") or d0.DEFAULT_PROVIDER))
        report = d0.run_d0(
            profile=str(request.get("profile") or d0.DEFAULT_PROFILE),
            vod_id=vod_id_arg,
            positive_rows_path=d0.DEFAULT_POSITIVES,
            recent_holdout_policy_path=d0.DEFAULT_RECENT_HOLDOUT_POLICY,
            training_holdout_policy_path=d0.DEFAULT_TRAINING_HOLDOUT_POLICY,
            vod_cache_root=d0.DEFAULT_VOD_CACHE_ROOT,
            video_source_root=d0.DEFAULT_VIDEO_SOURCE_ROOT,
            output_prefix=output_prefix,
            provider=provider,
            model=d0.resolve_provider_model(provider, str(request.get("model") or "")),
            top_n=int(request.get("top_n") or d0.DEFAULT_TOP_N),
            max_moments=int(request.get("max_moments") or d0.DEFAULT_MAX_MOMENTS),
            vod_limit=int(request.get("vod_limit") or d0.DEFAULT_VOD_LIMIT),
            overlap_ratio=d0.DEFAULT_OVERLAP_RATIO,
            match_mode=d0.DEFAULT_MATCH_MODE,
            min_duration_sec=d0.DEFAULT_MIN_DURATION_SEC,
            max_duration_sec=d0.DEFAULT_MAX_DURATION_SEC,
            temperature=0.0,
            max_tokens=int(request.get("max_tokens") or d0.DEFAULT_MAX_TOKENS),
            timeout_sec=d0.DEFAULT_TIMEOUT_SEC,
            api_retries=d0.DEFAULT_API_RETRIES,
            retry_backoff_sec=d0.DEFAULT_RETRY_BACKOFF_SEC,
            gate_min_recall=d0.DEFAULT_GATE_MIN_RECALL,
            filter_transcript_noise=True,
            allow_holdout=bool(request.get("allow_holdout")),
            dry_run=bool(request.get("dry_run")),
            reuse_responses=False,
            mock_response_path=None,
            download_vod_preflight=not bool(request.get("dry_run")),
            download_cache_ttl_sec=-1,
        )
        report = d0.attach_cli_run_manifest(
            report,
            run_dir=moment_dir,
            run_id=workflow_id,
            output_prefix=output_prefix,
            mode="frontend_moment_review",
            argv=[],
        )
        update_status(
            workflow_id,
            status="running",
            stage="precision_review",
            progress_pct=70,
            message="Building precision review and raw clips.",
            patch={"moment_report": summarize_moment_report(report), "moment_run_dir": str(moment_dir)},
        )
        if not has_any_moment_artifacts(output_prefix, report):
            raise RuntimeError("Moment finder produced no moment artifacts.")
        summary = precision_review.build_review_queue(
            profile=str(request.get("profile") or precision_review.DEFAULT_PROFILE),
            vod_ids=[],
            moment_prefix=output_prefix,
            moment_run_dir=moment_dir,
            positive_rows_path=precision_review.DEFAULT_POSITIVE_ROWS,
            vod_cache_root=precision_review.DEFAULT_VOD_CACHE_ROOT,
            out_base=review_dir / "precision_review",
            top_n_per_vod=int(request.get("top_n_per_vod") or precision_review.DEFAULT_TOP_N_PER_VOD),
            max_duration_sec=float(request.get("max_duration_sec") or precision_review.DEFAULT_MAX_DURATION_SEC),
            exclude_known_reposts=bool(request.get("exclude_known_reposts", False)),
            excerpt_pad_sec=precision_review.DEFAULT_EXCERPT_PAD_SEC,
            excerpt_max_chars=precision_review.DEFAULT_EXCERPT_MAX_CHARS,
            overlap_ratio=precision_review.DEFAULT_OVERLAP_RATIO,
            match_mode=precision_review.DEFAULT_MATCH_MODE,
            cut_clips=bool(request.get("cut_clips", True)),
            download_missing_vods=bool(request.get("cut_clips", True)),
            video_source_root=precision_review.DEFAULT_VIDEO_SOURCE_ROOT,
            clip_output_dir=review_dir / "clips",
            clip_run_id=workflow_id,
            clip_pad_sec=precision_review.DEFAULT_CLIP_PAD_SEC,
            download_cache_ttl_sec=-1,
            ffmpeg_bin=None,
            require_clips=bool(request.get("cut_clips", True)),
        )
        if int(summary.get("queue_rows") or 0) <= 0:
            raise RuntimeError("Precision review produced no review rows.")
        if bool(request.get("cut_clips", True)) and summary.get("clip_summary", {}).get("all_clips_ready") is False:
            raise RuntimeError("Precision review clips are incomplete.")
        update_status(
            workflow_id,
            status="done",
            stage="done",
            progress_pct=100,
            message="Review ready.",
            patch={
                "review_summary": summary,
                "review_jsonl_path": str(review_jsonl_path(root)),
                "review_summary_path": str(review_summary_path(root)),
            },
        )
    except SystemExit as exc:
        update_status(
            workflow_id,
            status="failed",
            stage="failed",
            progress_pct=100,
            message="Workflow failed.",
            error=str(exc),
        )
    except Exception as exc:
        update_status(
            workflow_id,
            status="failed",
            stage="failed",
            progress_pct=100,
            message="Workflow failed.",
            error=str(exc),
        )


def summarize_moment_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": report.get("event"),
        "provider": report.get("provider"),
        "model": report.get("model"),
        "vod_ids": report.get("vod_ids") or ([report.get("vod_id")] if report.get("vod_id") else []),
        "vod_count": report.get("vod_count"),
        "moment_count": report.get("moment_count"),
        "recall_at_top_n": report.get("recall_at_top_n"),
        "failed_vod_count": report.get("failed_vod_count"),
        "failed_vod_ids": report.get("failed_vod_ids") or [],
    }


def has_any_moment_artifacts(output_prefix: pathlib.Path, report: dict[str, Any]) -> bool:
    vod_ids = report.get("vod_ids") or ([report.get("vod_id")] if report.get("vod_id") else [])
    for vod_id in vod_ids:
        if precision_review.moment_path_for_vod(output_prefix, str(vod_id)).exists():
            return True
    return output_prefix.with_suffix(".moments.json").exists()


def hydrate_workflow_response(status: dict[str, Any], *, include_rows: bool = True) -> dict[str, Any]:
    workflow_id = str(status.get("workflow_id") or "")
    root = workflow_dir(workflow_id)
    summary = read_json(review_summary_path(root))
    rows = hydrate_review_rows(root, workflow_id) if include_rows else []
    label_summary = summarize_labels(rows)
    phase2_gate = build_phase2_gate(status, summary, label_summary)
    return {
        **status,
        "summary": summary,
        "label_summary": label_summary,
        "phase2_gate": phase2_gate,
        "rows": rows,
        "row_count": len(rows) if include_rows else int(summary.get("queue_rows") or 0),
    }


def read_review_rows(root: pathlib.Path) -> list[dict[str, Any]]:
    return read_jsonl(review_jsonl_path(root))


def hydrate_review_rows(root: pathlib.Path, workflow_id: str) -> list[dict[str, Any]]:
    rows = read_review_rows(root)
    labels = read_json(labels_path(root))
    hydrated: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        review_id = str(copied.get("review_id") or "")
        label = labels.get(review_id) if isinstance(labels.get(review_id), dict) else {}
        if label:
            copied["human_viral_label"] = label.get("human_viral_label", copied.get("human_viral_label", ""))
            copied["human_boundary_label"] = label.get("human_boundary_label", copied.get("human_boundary_label", ""))
            copied["human_notes"] = label.get("human_notes", copied.get("human_notes", ""))
            copied["label_updated_at"] = label.get("updated_at")
        clip_rel_path = str(copied.get("clip_rel_path") or "").strip()
        copied["clip_url"] = f"/api/llm-moment-review/workflows/{workflow_id}/clips/{clip_rel_path}" if clip_rel_path else ""
        hydrated.append(copied)
    return hydrated


def summarize_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_repost_rows = [row for row in rows if not bool(row.get("known_repost_match"))]
    labeled_non_reposts = [
        row
        for row in non_repost_rows
        if str(row.get("human_viral_label") or "").strip() in {"yes", "no", "unsure"}
    ]
    decisive_non_reposts = [
        row
        for row in non_repost_rows
        if str(row.get("human_viral_label") or "").strip() in {"yes", "no"}
    ]
    yes_count = sum(1 for row in non_repost_rows if str(row.get("human_viral_label") or "").strip() == "yes")
    no_count = sum(1 for row in non_repost_rows if str(row.get("human_viral_label") or "").strip() == "no")
    unsure_count = sum(1 for row in non_repost_rows if str(row.get("human_viral_label") or "").strip() == "unsure")
    total_non_repost = len(non_repost_rows)
    decisive_count = len(decisive_non_reposts)
    labeled_count = len(labeled_non_reposts)
    return {
        "total_rows": len(rows),
        "known_repost_rows": sum(1 for row in rows if bool(row.get("known_repost_match"))),
        "non_repost_rows": total_non_repost,
        "labeled_non_repost_rows": labeled_count,
        "unlabeled_non_repost_rows": max(total_non_repost - labeled_count, 0),
        "decisive_non_repost_rows": decisive_count,
        "non_repost_yes": yes_count,
        "non_repost_no": no_count,
        "non_repost_unsure": unsure_count,
        "non_repost_yes_no_precision": round(yes_count / decisive_count, 6) if decisive_count else None,
        "non_repost_strict_precision": round(yes_count / total_non_repost, 6) if total_non_repost else None,
        "non_repost_label_coverage": round(labeled_count / total_non_repost, 6) if total_non_repost else None,
    }


def build_phase2_gate(status: dict[str, Any], summary: dict[str, Any], label_summary: dict[str, Any]) -> dict[str, Any]:
    request = status.get("request") if isinstance(status.get("request"), dict) else {}
    min_precision = float(request.get("phase2_min_non_repost_precision") or 0.5)
    total_non_repost = int(label_summary.get("non_repost_rows") or 0)
    unlabeled_non_repost = int(label_summary.get("unlabeled_non_repost_rows") or 0)
    precision = label_summary.get("non_repost_yes_no_precision")
    precision_value = float(precision) if isinstance(precision, (int, float)) else None
    precision_ready = total_non_repost > 0 and unlabeled_non_repost == 0 and precision_value is not None
    precision_passes = bool(precision_ready and precision_value >= min_precision)
    recall_floor = extract_recall_floor(status)
    return {
        "version": "phase2_llm_moment_finder_gate_v1",
        "ready": bool(precision_ready and recall_floor.get("available") is True),
        "passes": bool(precision_passes and recall_floor.get("passes") is True),
        "min_non_repost_precision": round(min_precision, 6),
        "non_repost_precision_ready": precision_ready,
        "non_repost_precision_passes": precision_passes,
        "recall_floor": recall_floor,
        "label_summary": label_summary,
        "review_queue": {
            "exclude_known_reposts": bool(summary.get("exclude_known_reposts")),
            "queue_rows": int(summary.get("queue_rows") or 0),
            "known_repost_matches_in_queue": int(summary.get("known_repost_matches_in_queue") or 0),
            "non_repost_rows_in_queue": int(summary.get("non_repost_rows_in_queue") or 0),
        },
        "blocking_reasons": phase2_blocking_reasons(
            recall_floor=recall_floor,
            precision_ready=precision_ready,
            precision_passes=precision_passes,
            unlabeled_non_repost=unlabeled_non_repost,
            total_non_repost=total_non_repost,
        ),
    }


def extract_recall_floor(status: dict[str, Any]) -> dict[str, Any]:
    raw = status.get("recall_floor")
    if not isinstance(raw, dict):
        return {
            "available": False,
            "passes": False,
            "reason": "Run scripts/run_serving_recall_holdout.sh --enable-llm-moment-finder and attach its summary before Phase 3.",
        }
    ratio = raw.get("ratio")
    baseline = float(raw.get("baseline_ratio") or 0.027)
    if not isinstance(ratio, (int, float)):
        return {"available": False, "passes": False, "reason": "recall_floor.ratio missing", **raw}
    return {
        **raw,
        "available": True,
        "baseline_ratio": baseline,
        "passes": float(ratio) > baseline,
    }


def recall_floor_from_holdout_summary(path: pathlib.Path) -> dict[str, Any]:
    payload = read_json(path)
    if not payload:
        return {"available": False, "passes": False, "reason": f"Holdout summary not found: {path}"}
    llm_config = payload.get("llm_moment_finder") if isinstance(payload.get("llm_moment_finder"), dict) else {}
    if llm_config.get("enabled") is not True:
        return {"available": False, "passes": False, "reason": "Latest holdout summary is not an LLM Moment-Finder run."}
    aggregate = payload.get("eval_aggregate") if isinstance(payload.get("eval_aggregate"), dict) else {}
    product_selection = aggregate.get("product_selection") if isinstance(aggregate.get("product_selection"), dict) else {}
    metric = product_selection.get("final_moments@k12")
    metric_name = "final_moments@k12"
    if not isinstance(metric, dict):
        metric = product_selection.get("final_ranked@k12")
        metric_name = "final_ranked@k12"
    if not isinstance(metric, dict) or not isinstance(metric.get("per_vod_recall"), (int, float)):
        return {"available": False, "passes": False, "reason": "Latest LLM holdout summary has no final @k12 recall metric."}
    ratio = float(metric.get("per_vod_recall") or 0.0)
    baseline = 0.027
    return {
        "available": True,
        "passes": ratio > baseline,
        "ratio": round(ratio, 6),
        "baseline_ratio": baseline,
        "metric": metric_name,
        "matched_positive_windows": int(metric.get("matched_positive_windows") or 0),
        "positive_windows": int(metric.get("positive_windows") or 0),
        "selected_candidate_rows": int(metric.get("selected_candidate_rows") or 0),
        "run_id": str(payload.get("run_id") or ""),
        "run_summary_path": str(path),
        "llm_moment_finder": llm_config,
    }


def phase2_blocking_reasons(
    *,
    recall_floor: dict[str, Any],
    precision_ready: bool,
    precision_passes: bool,
    unlabeled_non_repost: int,
    total_non_repost: int,
) -> list[str]:
    reasons: list[str] = []
    if recall_floor.get("available") is not True:
        reasons.append("recall_floor_missing")
    elif recall_floor.get("passes") is not True:
        reasons.append("recall_floor_below_baseline")
    if total_non_repost <= 0:
        reasons.append("no_non_repost_review_rows")
    elif not precision_ready:
        reasons.append(f"unlabeled_non_repost_rows:{unlabeled_non_repost}")
    elif not precision_passes:
        reasons.append("non_repost_precision_below_threshold")
    return reasons
