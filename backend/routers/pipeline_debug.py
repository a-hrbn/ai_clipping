from __future__ import annotations

import asyncio
import datetime
import json
import logging
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import settings
import shutil

from job_lifecycle import is_terminal
from job_manager import create_job, delete_job, get_job
from ml.llm_moment_finder import TranscriptSegment, transcript_excerpt
from pipeline_runner import run_job_subprocess
from services.job_artifact_store import JobArtifactStore, RealFsJobArtifactStore

logger = logging.getLogger(__name__)

_BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _job_store() -> JobArtifactStore:
    """The Job-Artifact-Store reader seam for this router (real-FS adapter).

    Job artifacts (progress.json, the ``llm_moment_finder_*`` dumps,
    segments.jsonl) are read through the store so this router never spells an
    artifact filename or re-implements the jsonl/json decode. Workflow-level
    files (workflow.json, the review labels/events) are NOT job artifacts and
    keep their own I/O below.
    """
    return RealFsJobArtifactStore(settings.storage_dir)


router = APIRouter(prefix="/api/pipeline-debug", tags=["pipeline-debug"])

WorkflowStatus = Literal["queued", "running", "done", "failed"]
WorkflowStage = Literal["queued", "transcribe", "moment_finder", "cut_clips", "done", "failed"]

# Map the worker's fine-grained JobState onto the coarse workflow stage shown in
# the review UI, so a long run reports its real phase instead of a frozen one.
_JOB_STATE_TO_WORKFLOW_STAGE: dict[str, WorkflowStage] = {
    "queued": "transcribe",
    "downloading": "transcribe",
    "downloading_chat": "transcribe",
    "extracting_audio": "transcribe",
    "transcribing": "transcribe",
    "extracting_features": "transcribe",
    "diarizing": "transcribe",
    "analyzing": "moment_finder",
    "judging": "moment_finder",
    "clipping": "cut_clips",
    "done": "cut_clips",
}

WORKFLOW_ROOT = pathlib.Path(settings.storage_dir).parent / "pipeline_debug_workflows"
LABELS_FILENAME = "review_labels.json"
LABEL_EVENTS_FILENAME = "review_label_events.jsonl"

_WORKFLOW_ID_RE = re.compile(r"^[0-9A-Za-z][\w\-]{0,63}$")

EXCERPT_PAD_SEC = 4.0
EXCERPT_MAX_CHARS = 500


class PipelineDebugStartRequest(BaseModel):
    profile: str = "jynxzi"
    vod_ids: list[str] = Field(default_factory=list)
    provider: Literal["gemini", "anthropic"] = "gemini"
    model: str = ""
    max_moments: int = Field(default=180, ge=1, le=500)
    name: str = ""


class PipelineDebugLabelRequest(BaseModel):
    review_id: str
    human_viral_label: Literal["yes", "no", "unsure", ""] = ""
    human_boundary_label: Literal["good", "too_wide", "too_short", "wrong", "unsure", ""] = ""
    human_notes: str = ""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def _workflow_id_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_workflow_id(workflow_id: str) -> str:
    if not _WORKFLOW_ID_RE.fullmatch(str(workflow_id or "")):
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow_id


def _workflow_dir(workflow_id: str) -> pathlib.Path:
    safe_id = _safe_workflow_id(workflow_id)
    root = WORKFLOW_ROOT.resolve()
    path = (root / safe_id).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return path


def _status_path(root: pathlib.Path) -> pathlib.Path:
    return root / "workflow.json"


def _labels_path(root: pathlib.Path) -> pathlib.Path:
    return root / "review" / LABELS_FILENAME


def _label_events_path(root: pathlib.Path) -> pathlib.Path:
    return root / "review" / LABEL_EVENTS_FILENAME


def _review_jsonl_path(root: pathlib.Path) -> pathlib.Path:
    return root / "review" / "precision_review.jsonl"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
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


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _update_status(
    workflow_id: str,
    *,
    status: WorkflowStatus,
    stage: WorkflowStage,
    progress_pct: int,
    message: str,
    error: str | None = None,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = _workflow_dir(workflow_id)
    current = _read_json(_status_path(root))
    payload: dict[str, Any] = {
        **current,
        "workflow_id": workflow_id,
        "status": status,
        "stage": stage,
        "progress_pct": max(0, min(int(progress_pct), 100)),
        "message": message,
        "error": error,
        "updated_at": _utc_now(),
    }
    if "created_at" not in payload:
        payload["created_at"] = payload["updated_at"]
    if patch:
        payload.update(patch)
    _write_json(_status_path(root), payload)
    return payload


@router.get("/active-profile")
async def get_active_clip_profile(profile: str = "jynxzi") -> dict[str, Any]:
    from ml import llm_moment_finder as d0
    profile_spec = d0.load_clip_profile(profile)
    if profile_spec is not None:
        return {
            "is_generated": True,
            "profile_name": profile,
            "rendered_rules": d0.render_profile_rules(profile_spec),
            **profile_spec,
        }
    return {
        "is_generated": False,
        "profile_name": profile,
        "rendered_rules": None,
        "summary": None,
        "archetypes": [],
        "selection_rules": [],
        "avoid": [],
        "source": None,
        "generated_at": None,
    }


@router.get("/workflows/{workflow_id}/prompt")
async def get_pipeline_debug_run_prompt(workflow_id: str, vod_id: str = "") -> dict[str, Any]:
    safe_id = _safe_workflow_id(workflow_id)
    vod_id_clean = re.sub(r"[^0-9a-zA-Z_-]", "", str(vod_id or "").strip())
    if not vod_id_clean:
        raise HTTPException(status_code=400, detail="vod_id is required")
    # job_id is built from the sanitized workflow_id + vod_id (no path-traversal
    # chars), so the store resolves it safely under storage_dir/<job_id>/.
    job_id = f"pipeline_debug_{safe_id}_{vod_id_clean}"
    prompt_text = _job_store().read_prompt(job_id)
    if prompt_text is None:
        return {"vod_id": vod_id_clean, "prompt": "", "exists": False}
    return {"vod_id": vod_id_clean, "prompt": prompt_text, "exists": True}


@router.get("/anchors")
async def get_pipeline_debug_anchors() -> dict[str, Any]:
    from ml.eval_clip_finder import read_anchor_vod_ids as _read_anchors
    ids = _read_anchors()
    return {
        "vod_ids": ids,
        "vods": [{"vod_id": v, "url": f"https://www.twitch.tv/videos/{v}"} for v in ids],
        "count": len(ids),
    }


@router.post("/workflows")
async def start_pipeline_debug_workflow(request: PipelineDebugStartRequest) -> dict[str, Any]:
    workflow_id = _workflow_id_now()
    root = _workflow_dir(workflow_id)
    root.mkdir(parents=True, exist_ok=True)
    req_dict = request.model_dump()
    # Validate client vod_ids against anchor CSV; fall back to all anchors if none supplied.
    try:
        from ml.eval_clip_finder import read_anchor_vod_ids as _read_anchors
        anchor_ids = _read_anchors()
        anchor_set = set(anchor_ids)
        client_ids = [v for v in (req_dict.get("vod_ids") or []) if v in anchor_set]
        req_dict["vod_ids"] = client_ids if client_ids else anchor_ids
        logger.info(
            "pipeline_debug_anchor_selection workflow_id=%s client_ids=%s resolved=%s",
            workflow_id,
            ",".join(req_dict.get("vod_ids") or []) or "(none)",
            ",".join(req_dict["vod_ids"]),
        )
    except Exception:
        logger.warning("Could not read anchor VOD IDs — using client-supplied vod_ids", exc_info=True)
    initial = _update_status(
        workflow_id,
        status="queued",
        stage="queued",
        progress_pct=0,
        message="Workflow queued.",
        patch={"request": req_dict, "workflow_dir": str(root), "name": req_dict.get("name", "").strip()},
    )
    # Spawn AFTER the queued status (with the request payload) is persisted — the
    # detached orchestrator reads its request straight back from workflow.json.
    _spawn_detached_orchestrator(workflow_id, root)
    return _hydrate_workflow_response(_read_json(_status_path(root)) or initial)


def _spawn_detached_orchestrator(workflow_id: str, root: pathlib.Path) -> int:
    """Launch the workflow orchestrator as a fully detached subprocess.

    Running in its own session (``start_new_session=True``) keeps the run alive
    across uvicorn ``--reload`` restarts and server crashes, so a long multi-VOD
    run is not aborted when source files change during development. This mirrors
    how ``pipeline_worker.py`` is already spawned. The request payload lives in
    ``workflow.json``, so the detached process is fully self-describing.
    """
    from pipeline_runner import build_subprocess_env

    log_handle = open(root / "orchestrator.log", "ab")
    try:
        process = subprocess.Popen(
            [sys.executable, str(_BACKEND_ROOT / "pipeline_debug_runner.py"), "--workflow-id", workflow_id],
            cwd=str(_BACKEND_ROOT),
            env=build_subprocess_env(),
            stdout=log_handle,
            stderr=log_handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()  # the child holds its own dup'd fd
    logger.info("pipeline_debug_orchestrator_spawned workflow_id=%s pid=%d", workflow_id, process.pid)
    return process.pid


@router.get("/workflows")
async def list_pipeline_debug_workflows(limit: int = 20) -> dict[str, Any]:
    safe_limit = min(max(int(limit or 0), 1), 100)
    rows: list[dict[str, Any]] = []
    if WORKFLOW_ROOT.exists():
        for path in sorted(WORKFLOW_ROOT.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_dir():
                continue
            status = _read_json(_status_path(path))
            if not status:
                continue
            rows.append(_hydrate_workflow_response(status, include_rows=False))
            if len(rows) >= safe_limit:
                break
    return {"workflows": rows}


@router.get("/workflows/{workflow_id}")
async def get_pipeline_debug_workflow(workflow_id: str) -> dict[str, Any]:
    root = _workflow_dir(workflow_id)
    status = _read_json(_status_path(root))
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return _hydrate_workflow_response(status)


@router.delete("/workflows/{workflow_id}")
async def delete_pipeline_debug_workflow(workflow_id: str) -> dict[str, Any]:
    root = _workflow_dir(workflow_id)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Workflow not found")
    status = _read_json(_status_path(root))
    if status.get("status") in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Workflow is still running")
    # Delete all job entries and their storage directories for this workflow
    storage_root = pathlib.Path(settings.storage_dir)
    prefix = f"pipeline_debug_{workflow_id}_"
    for job_dir in storage_root.iterdir():
        if not job_dir.name.startswith(prefix):
            continue
        job_id = job_dir.name
        job = get_job(job_id)
        if job is not None and not is_terminal(job.status):
            continue  # skip still-running jobs
        delete_job(job_id)
        if job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)
    # Delete the workflow directory itself
    shutil.rmtree(root, ignore_errors=True)
    return {"deleted": workflow_id}


@router.post("/workflows/{workflow_id}/labels")
async def save_pipeline_debug_label(workflow_id: str, request: PipelineDebugLabelRequest) -> dict[str, Any]:
    root = _workflow_dir(workflow_id)
    if not _status_path(root).exists():
        raise HTTPException(status_code=404, detail="Workflow not found")
    review_id = str(request.review_id or "").strip()
    if not review_id:
        raise HTTPException(status_code=400, detail="review_id is required")
    rows = _read_jsonl(_review_jsonl_path(root))
    if review_id not in {str(row.get("review_id") or "") for row in rows}:
        raise HTTPException(status_code=404, detail="Review item not found")
    labels = _read_json(_labels_path(root))
    label: dict[str, Any] = {
        "review_id": review_id,
        "human_viral_label": request.human_viral_label,
        "human_boundary_label": request.human_boundary_label,
        "human_notes": request.human_notes.strip(),
        "updated_at": _utc_now(),
    }
    labels[review_id] = label
    _write_json(_labels_path(root), labels)
    with _label_events_path(root).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label, ensure_ascii=True, sort_keys=True) + "\n")
    return {"workflow_id": workflow_id, "label": label}


@router.get("/workflows/{workflow_id}/clips/{clip_path:path}")
def serve_pipeline_debug_clip(workflow_id: str, clip_path: str) -> FileResponse:
    _safe_workflow_id(workflow_id)
    storage_root = pathlib.Path(settings.storage_dir).resolve()
    path = (storage_root / clip_path).resolve()
    try:
        path.relative_to(storage_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    if not path.is_file() or path.suffix.lower() not in {".mp4", ".webm", ".mov", ".m4v", ".mkv"}:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


async def _mirror_job_progress(
    *,
    workflow_id: str,
    job_id: str,
    vod_id: str,
    vod_index: int,
    vod_count: int,
    stop: asyncio.Event,
    poll_sec: float = 3.0,
) -> None:
    """Mirror the running worker's job status into the workflow status.

    The orchestrator otherwise blocks on the subprocess and the review UI would
    show a frozen 'transcribe' stage for the whole run. Polling the job store and
    reflecting status/progress makes a long run observable phase by phase.
    """
    base = 5 + int((vod_index / max(vod_count, 1)) * 80)
    span = max(int(80 / max(vod_count, 1)), 1)
    last: tuple[str, int] | None = None
    while not stop.is_set():
        # Read the worker's own progress side-file rather than the shared job
        # registry: jobs.json is mutated by multiple processes with unreliable
        # mtime-based reloads on bind mounts, so it can show a stale phase.
        prog = _read_job_progress(job_id)
        if prog is not None and prog != last:
            last = prog
            status, pct = prog
            stage = _JOB_STATE_TO_WORKFLOW_STAGE.get(status, "transcribe")
            progress = min(base + int((pct / 100.0) * span), 95)
            _update_status(
                workflow_id,
                status="running",
                stage=stage,
                progress_pct=progress,
                message=f"VOD {vod_index + 1}/{vod_count} ({vod_id}): {status} {pct}%",
            )
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_sec)
        except asyncio.TimeoutError:
            pass


def _read_job_progress(
    job_id: str, *, store: JobArtifactStore | None = None
) -> tuple[str, int] | None:
    """Return (status, progress_pct) from the worker's per-job progress artifact.

    None if the artifact is missing/unreadable (worker hasn't created its
    workspace yet) — the caller then keeps the last known status.
    """
    store = store or _job_store()
    data = store.read_progress(job_id)
    if not isinstance(data, dict):
        return None
    status = str(data.get("status") or "").strip()
    if not status:
        return None
    try:
        pct = int(data.get("progress_pct") or 0)
    except (TypeError, ValueError):
        pct = 0
    return status, pct


def _run_workflow_sync(workflow_id: str, request: dict[str, Any]) -> None:
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run_workflow(workflow_id, request))
    finally:
        loop.close()


async def _run_workflow(workflow_id: str, request: dict[str, Any]) -> None:
    root = _workflow_dir(workflow_id)
    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    profile = str(request.get("profile") or "jynxzi")
    vod_ids = [str(v).strip() for v in (request.get("vod_ids") or []) if str(v).strip()]
    max_moments = int(request.get("max_moments") or 180)
    provider = str(request.get("provider") or "gemini")
    strategy: dict[str, Any] = {"platform": "twitch"}

    all_rows: list[dict[str, Any]] = []
    failed_vods: list[str] = []
    vod_count = len(vod_ids)
    total_llm_moment_count = 0
    workflow_start = time.monotonic()

    logger.info(
        "pipeline_debug_start workflow_id=%s vod_count=%d profile=%s provider=%s max_moments=%d vods=%s",
        workflow_id, vod_count, profile, provider, max_moments, ",".join(vod_ids),
    )

    try:
        _update_status(
            workflow_id,
            status="running",
            stage="transcribe",
            progress_pct=5,
            message=f"Running pipeline for {vod_count} VOD(s).",
        )

        # Run VODs with bounded concurrency: Whisper is serialized per-GPU via a
        # cross-process lock in the transcriber, so a second worker overlaps its
        # download/ffmpeg/chat stages with the first worker's transcription.
        # Status messages flap between in-flight VODs; acceptable for a debug UI.
        vod_concurrency = int(getattr(settings, "pipeline_debug_vod_concurrency", 0) or 0)
        if vod_concurrency <= 0:
            from services.transcriber import transcription_slots
            vod_concurrency = min(4, transcription_slots() + 1)
        vod_semaphore = asyncio.Semaphore(max(vod_concurrency, 1))

        async def _run_one_vod(vod_index: int, vod_id: str) -> tuple[list[dict], int] | None:
            """Run one VOD's pipeline; returns (rows, llm_count) or None on failure."""
            job_id = f"pipeline_debug_{workflow_id}_{vod_id}"
            url = f"https://www.twitch.tv/videos/{vod_id}"
            async with vod_semaphore:
                progress = 5 + int((vod_index / max(vod_count, 1)) * 80)
                vod_start = time.monotonic()

                logger.info(
                    "pipeline_debug_vod_start workflow_id=%s vod=%s job_id=%s [%d/%d]",
                    workflow_id, vod_id, job_id, vod_index + 1, vod_count,
                )
                _update_status(
                    workflow_id,
                    status="running",
                    stage="transcribe",
                    progress_pct=progress,
                    message=f"VOD {vod_index + 1}/{vod_count}: {vod_id}",
                )

                create_job(job_id, profile=profile)

                stop_mirror = asyncio.Event()
                mirror_task = asyncio.create_task(
                    _mirror_job_progress(
                        workflow_id=workflow_id,
                        job_id=job_id,
                        vod_id=vod_id,
                        vod_index=vod_index,
                        vod_count=vod_count,
                        stop=stop_mirror,
                    )
                )
                try:
                    returncode = await run_job_subprocess(
                        job_id=job_id,
                        url=url,
                        profile=profile,
                        strategy=strategy,
                    )
                finally:
                    stop_mirror.set()
                    await mirror_task

                vod_elapsed = round(time.monotonic() - vod_start, 1)
                if returncode != 0:
                    job = get_job(job_id)
                    err = str(job.error if job else "") or f"exit code {returncode}"
                    logger.warning(
                        "pipeline_debug_vod_failed workflow_id=%s vod=%s job_id=%s elapsed=%.1fs error=%r",
                        workflow_id, vod_id, job_id, vod_elapsed, err,
                    )
                    return None

                vod_rows, vod_llm_count = _build_review_rows(workflow_id=workflow_id, vod_id=vod_id, job_id=job_id)
                clips_ready = sum(1 for r in vod_rows if r.get("clip_status") == "ready")
                logger.info(
                    "pipeline_debug_vod_done workflow_id=%s vod=%s elapsed=%.1fs"
                    " llm_moments=%d rows=%d clips_ready=%d clips_missing=%d",
                    workflow_id, vod_id, vod_elapsed,
                    vod_llm_count, len(vod_rows), clips_ready, len(vod_rows) - clips_ready,
                )
                return vod_rows, vod_llm_count

        results = await asyncio.gather(
            *(_run_one_vod(i, v) for i, v in enumerate(vod_ids)),
            return_exceptions=True,
        )
        for vod_id, result in zip(vod_ids, results):
            if isinstance(result, BaseException):
                logger.error(
                    "pipeline_debug_vod_crashed workflow_id=%s vod=%s error=%r",
                    workflow_id, vod_id, result,
                )
                failed_vods.append(vod_id)
            elif result is None:
                failed_vods.append(vod_id)
            else:
                vod_rows, vod_llm_count = result
                all_rows.extend(vod_rows)
                total_llm_moment_count += vod_llm_count

        _write_jsonl(_review_jsonl_path(root), all_rows)

        if not all_rows and vod_ids:
            raise RuntimeError(
                f"Pipeline produced no review rows. "
                f"Failed VODs: {failed_vods or 'none'}, "
                f"successful VODs yielded 0 moments."
            )

        total_elapsed = round(time.monotonic() - workflow_start, 1)
        logger.info(
            "pipeline_debug_done workflow_id=%s elapsed=%.1fs vods=%d failed=%d"
            " total_rows=%d llm_moments=%d",
            workflow_id, total_elapsed, vod_count, len(failed_vods),
            len(all_rows), total_llm_moment_count,
        )
        _update_status(
            workflow_id,
            status="done",
            stage="done",
            progress_pct=100,
            message="Review ready.",
            patch={
                "vod_count": vod_count,
                "failed_vod_count": len(failed_vods),
                "failed_vod_ids": failed_vods,
                "row_count": len(all_rows),
                "llm_moment_count": total_llm_moment_count,
                "elapsed_sec": total_elapsed,
            },
        )

    except SystemExit as exc:
        total_elapsed = round(time.monotonic() - workflow_start, 1)
        logger.error(
            "pipeline_debug_failed workflow_id=%s elapsed=%.1fs error=%r",
            workflow_id, total_elapsed, str(exc),
        )
        _update_status(
            workflow_id,
            status="failed",
            stage="failed",
            progress_pct=100,
            message="Workflow failed.",
            error=str(exc),
        )
    except Exception as exc:
        total_elapsed = round(time.monotonic() - workflow_start, 1)
        logger.exception(
            "pipeline_debug_failed workflow_id=%s elapsed=%.1fs error=%r",
            workflow_id, total_elapsed, str(exc),
        )
        _update_status(
            workflow_id,
            status="failed",
            stage="failed",
            progress_pct=100,
            message="Workflow failed.",
            error=str(exc),
        )


def _clip_overlap(start: float, end: float, clip_start: float, clip_end: float) -> float:
    """Overlap fraction of [start,end] covered by [clip_start,clip_end]."""
    dur = max(end - start, 0.001)
    overlap = max(0.0, min(end, clip_end) - max(start, clip_start))
    return overlap / dur


def _build_filtered_clip_map(clips_dir: pathlib.Path, filtered_moments: list[dict[str, Any]]) -> dict[tuple[float, float], str]:
    """Map (start_time, end_time) → clip filename for each filtered moment by position."""
    clip_map: dict[tuple[float, float], str] = {}
    for i, m in enumerate(filtered_moments, start=1):
        clip_map[(round(float(m.get("start_time") or 0), 3), round(float(m.get("end_time") or 0), 3))] = f"clip_{i:02d}.mp4"
    return clip_map


def _resolve_clip_for_raw_moment(
    start: float,
    end: float,
    clip_map: dict[tuple[float, float], str],
    clips_dir: pathlib.Path,
    job_id: str,
) -> tuple[str, str]:
    """Return (clip_status, clip_url) for a raw moment by matching against filtered clips."""
    best_filename: str | None = None
    best_overlap = 0.0
    for (cs, ce), fname in clip_map.items():
        ov = _clip_overlap(start, end, cs, ce)
        if ov > best_overlap:
            best_overlap = ov
            best_filename = fname
    if best_filename and best_overlap >= 0.9 and (clips_dir / best_filename).is_file():
        return "ready", f"/static/{job_id}/clips/{best_filename}"
    return "missing", ""


def _build_review_rows(
    *, workflow_id: str, vod_id: str, job_id: str, store: JobArtifactStore | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Return (rows, raw_llm_moment_count).

    Rows are built from the pre-diversity raw moments so the eval and review UI
    reflect true LLM recall. Clip URLs are resolved by matching raw moments against
    the filtered (clipped) set via overlap — only moments that survived diversity
    filtering will have a playable clip.
    """
    store = store or _job_store()
    clips_dir = pathlib.Path(settings.storage_dir) / job_id / "clips"

    # Prefer raw; fall back to filtered for backwards-compat with older workflow
    # runs that wrote no raw artifact (filtered ⊆ raw, so empty raw ⇒ empty filtered).
    filtered_moments = store.read_moments(job_id)
    raw_moments = store.read_raw_moments(job_id)
    moment_rows = raw_moments or filtered_moments
    raw_llm_count = len(moment_rows)
    if not moment_rows:
        logger.warning(
            "pipeline_debug_build_rows_no_moments workflow_id=%s vod=%s job_id=%s",
            workflow_id, vod_id, job_id,
        )
    segment_records = store.read_segments(job_id)
    logger.info(
        "pipeline_debug_build_rows workflow_id=%s vod=%s moments=%d segments=%d raw=%s",
        workflow_id, vod_id, raw_llm_count, len(segment_records), bool(raw_moments),
    )
    segments = [
        TranscriptSegment(start=float(s.get("start") or 0), end=float(s.get("end") or 0), text=str(s.get("text") or ""))
        for s in segment_records
        if s.get("start") is not None and s.get("end") is not None and s.get("text")
    ]

    # Build clip map from filtered moments (clips are only cut for filtered set).
    clip_map = _build_filtered_clip_map(clips_dir, filtered_moments)

    rows: list[dict[str, Any]] = []
    for i, moment in enumerate(moment_rows, start=1):
        start = float(moment.get("start_time") or 0)
        end = float(moment.get("end_time") or 0)
        if end <= start:
            continue
        duration = round(end - start, 3)
        clip_status, clip_url = _resolve_clip_for_raw_moment(start, end, clip_map, clips_dir, job_id)
        excerpt = transcript_excerpt(segments, start, end, pad_sec=EXCERPT_PAD_SEC, max_chars=EXCERPT_MAX_CHARS) if segments else ""
        review_id = f"pipeline_debug:{vod_id}:rank{i:02d}"
        rows.append({
            "review_id": review_id,
            "vod_id": vod_id,
            "job_id": job_id,
            "review_rank": i,
            "source_rank": int(moment.get("rank") or i),
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": duration,
            "virality": float(moment.get("viral_score") or 0),
            "composite_score": float(moment.get("composite_score") or moment.get("viral_score") or 0),
            "llm_virality": float(moment.get("llm_virality") or 0),
            "hook": float(moment.get("hook") or 0),
            "flow": float(moment.get("flow") or 0),
            "value": float(moment.get("value") or 0),
            "trend": float(moment.get("trend") or 0),
            "rationale": str(moment.get("reason") or "").strip(),
            "transcript_excerpt": excerpt,
            "clip_status": clip_status,
            "clip_url": clip_url,
            "human_viral_label": "",
            "human_boundary_label": "",
            "human_notes": "",
        })
    return rows, raw_llm_count


def _hydrate_workflow_response(status: dict[str, Any], *, include_rows: bool = True) -> dict[str, Any]:
    workflow_id = str(status.get("workflow_id") or "")
    root = _workflow_dir(workflow_id)
    rows = _hydrate_review_rows(root, workflow_id) if include_rows else []
    # Compute eval metrics live (only on detail GET when status=done; skipped on list endpoint)
    eval_result: dict[str, Any] = {}
    if include_rows and status.get("status") == "done":
        eval_result = _compute_eval(status)
        rows = _annotate_rows_with_gt(rows, eval_result)
    label_summary = _summarize_labels(rows)
    return {
        **status,
        "label_summary": label_summary,
        "eval": eval_result,
        "rows": rows,
        "row_count": len(rows) if include_rows else int(status.get("row_count") or 0),
    }


def _hydrate_review_rows(root: pathlib.Path, workflow_id: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(_review_jsonl_path(root))
    labels = _read_json(_labels_path(root))
    storage = pathlib.Path(settings.storage_dir)
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
        # Refresh clip_status from filesystem — baked-in value may be stale if
        # clipping was still in progress when the workflow state was written.
        # Only use the clip_url that was resolved during _build_review_rows (via
        # overlap-matching against filtered clips). Do NOT fall back to review_rank
        # as a position index — raw rows have different positions than filtered clips.
        job_id = str(copied.get("job_id") or "")
        clip_filename = str(copied.get("clip_url") or "").rsplit("/", 1)[-1] or ""
        if job_id and clip_filename:
            clip_file = storage / job_id / "clips" / clip_filename
            if clip_file.is_file():
                copied["clip_status"] = "ready"
                copied["clip_url"] = f"/static/{job_id}/clips/{clip_filename}"
            else:
                copied["clip_status"] = "missing"
                copied["clip_url"] = ""
        hydrated.append(copied)
    return hydrated


def _compute_eval(
    status: dict[str, Any], *, store: JobArtifactStore | None = None
) -> dict[str, Any]:
    """Compute baseline eval metrics live from disk. Never raises — returns {} on any error."""
    try:
        from ml.eval_clip_finder import (
            evaluate_all as _evaluate_all,
            load_manifest as _load_manifest,
            read_anchor_vod_ids as _read_anchors,
        )
        store = store or _job_store()
        workflow_id = str(status.get("workflow_id") or "")
        if not workflow_id:
            return {}
        # Use the VODs actually run in this workflow (may be a subset of all anchors).
        run_vod_ids = [str(v) for v in ((status.get("request") or {}).get("vod_ids") or [])]
        if not run_vod_ids:
            run_vod_ids = _read_anchors()
        predictions_by_vod: dict[str, list[dict[str, Any]]] = {}
        for vod_id in run_vod_ids:
            job_id = f"pipeline_debug_{workflow_id}_{vod_id}"
            # Prefer raw (pre-diversity) moments for eval; fall back to filtered for older runs.
            raw_moments = store.read_raw_moments(job_id)
            predictions_by_vod[vod_id] = raw_moments or store.read_moments(job_id)
        manifest = _load_manifest()
        manifest_vod_count = sum(1 for v in run_vod_ids if v in manifest)
        pred_counts = {v: len(p) for v, p in predictions_by_vod.items()}
        logger.info(
            "pipeline_debug_eval workflow_id=%s vods=%d manifest_coverage=%d/%d preds=%s",
            workflow_id, len(run_vod_ids), manifest_vod_count, len(run_vod_ids),
            ",".join(f"{v}:{n}" for v, n in pred_counts.items()),
        )
        result = _evaluate_all(predictions_by_vod, manifest=manifest, anchor_vod_ids=run_vod_ids, ks=(5, 10, 20, 50))
        overall = result.get("overall") or {}
        logger.info(
            "pipeline_debug_eval_result workflow_id=%s gt_count=%d hits=%d miss_rate=%s"
            " recall@5=%s recall@10=%s recall@20=%s avg_rank=%s",
            workflow_id,
            overall.get("gt_count", 0),
            overall.get("hit_count", 0),
            overall.get("miss_rate"),
            (overall.get("recall") or {}).get("5"),
            (overall.get("recall") or {}).get("10"),
            (overall.get("recall") or {}).get("20"),
            overall.get("avg_rank"),
        )
        return result
    except Exception:
        logger.warning("_compute_eval failed — skipping eval in response", exc_info=True)
        return {}


def _annotate_rows_with_gt(
    rows: list[dict[str, Any]],
    eval_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Stamp each row with gt_hit/gt_desc/gt_coverage/gt_views/gt_rank from eval_result.

    Joins on (vod_id, source_rank) == (vod_id, hit_rank) so the rank in the eval
    and the clip filename number always agree.
    """
    per_vod: list[dict[str, Any]] = eval_result.get("per_vod") or []
    # Build index: (vod_id, hit_rank) -> gt_hit entry (first hit wins if two GT share a rank)
    hit_index: dict[tuple[str, int], dict[str, Any]] = {}
    for vod_eval in per_vod:
        vod_id = str(vod_eval.get("vod_id") or "")
        for h in vod_eval.get("gt_hits") or []:
            if h.get("hit") and h.get("hit_rank") is not None:
                key = (vod_id, int(h["hit_rank"]))
                if key not in hit_index:
                    hit_index[key] = h

    annotated: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        vod_id = str(copied.get("vod_id") or "")
        source_rank = copied.get("source_rank")
        hit = hit_index.get((vod_id, int(source_rank))) if source_rank is not None else None
        copied["gt_hit"] = bool(hit)
        copied["gt_desc"] = str(hit["desc"]) if hit else ""
        copied["gt_coverage"] = float(hit["coverage"]) if hit else None
        copied["gt_views"] = float(hit["views"]) if hit else None
        copied["gt_rank"] = int(hit["hit_rank"]) if hit else None
        annotated.append(copied)
    return annotated


def _summarize_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [r for r in rows if str(r.get("human_viral_label") or "").strip() in {"yes", "no", "unsure"}]
    decisive = [r for r in rows if str(r.get("human_viral_label") or "").strip() in {"yes", "no"}]
    yes = sum(1 for r in rows if str(r.get("human_viral_label") or "").strip() == "yes")
    no = sum(1 for r in rows if str(r.get("human_viral_label") or "").strip() == "no")
    unsure = sum(1 for r in rows if str(r.get("human_viral_label") or "").strip() == "unsure")
    total = len(rows)
    decisive_count = len(decisive)
    return {
        "total_rows": total,
        "labeled_rows": len(labeled),
        "unlabeled_rows": max(total - len(labeled), 0),
        "yes": yes,
        "no": no,
        "unsure": unsure,
        "yes_precision": round(yes / decisive_count, 6) if decisive_count else None,
        "label_coverage": round(len(labeled) / total, 6) if total else None,
    }
