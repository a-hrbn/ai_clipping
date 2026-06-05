import asyncio
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from config import settings
from models import (
    JobRequest,
    JobStatus,
    ClipFeedbackPayload,
    ClipFeedbackResponse,
    ReviewCandidateResponse,
)
from job_manager import bind_job_session, cancel_job, create_job, get_job, touch_job_session
from job_manager import update_job
from services import clip_strategy, feedback_store

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

status_router = APIRouter(prefix="/api", tags=["status"])

# Cache: result is valid for 5 minutes to avoid burning API quota on every page load
import time as _time
_status_cache: dict = {}
_STATUS_TTL = 300
# Face-detection capability is stable for the life of the process (import-time
# dependency check), so we compute it once and reuse. The probe shells out to a
# subprocess that loads mediapipe/MTCNN/OpenCV, which is too expensive to rerun
# every 5 minutes on the status endpoint.
_face_detection_cache: dict[str, str | bool | None] | None = None
_SENTENCE_TRANSFORMERS_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_PYANNOTE_MODEL = "speaker-diarization-3.1"
_BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TERMINAL_STATES = {"done", "failed", "cancelled"}


def _read_json_file(path: pathlib.Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _jsonl_count(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _read_jsonl_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
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


def _metric(label: str, value) -> dict:
    return {"label": label, "value": value}


def _diagnostic_stage(
    stage_id: str,
    label: str,
    status: str,
    summary: str,
    metrics: list[dict] | None = None,
    details: list[str] | None = None,
) -> dict:
    return {
        "id": stage_id,
        "label": label,
        "status": status,
        "summary": summary,
        "metrics": metrics or [],
        "details": details or [],
    }


def _debug_last_modified(debug_dir: pathlib.Path) -> str | None:
    mtimes: list[float] = []
    try:
        for path in debug_dir.iterdir():
            if path.is_file():
                mtimes.append(path.stat().st_mtime)
    except OSError:
        return None
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()


def _job_pipeline_diagnostics(job_id: str, job: JobStatus | dict | None = None) -> dict:
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    debug_dir = job_dir / "debug"
    job_status = None
    job_error = None
    clip_count = 0
    if isinstance(job, dict):
        job_status = job.get("status")
        job_error = job.get("error")
        clip_count = len(job.get("clips") or [])
    elif job is not None:
        job_status = job.status
        job_error = job.error
        clip_count = len(job.clips)

    if not debug_dir.exists():
        return {
            "available": False,
            "job_status": job_status,
            "summary": "Noch keine Pipeline-Debugdaten vorhanden.",
            "stages": [],
        }

    candidate_counts = {
        "peak": _jsonl_count(debug_dir / "candidates_peak.jsonl"),
        "semantic": _jsonl_count(debug_dir / "candidates_semantic.jsonl"),
        "dense": _jsonl_count(debug_dir / "candidates_dense.jsonl"),
        "fallback": _jsonl_count(debug_dir / "candidates_fallback.jsonl"),
        "merged": _jsonl_count(debug_dir / "candidates_merged.jsonl"),
        "normalized": _jsonl_count(debug_dir / "candidates_normalized.jsonl"),
    }
    total_raw_candidates = (
        candidate_counts["peak"]
        + candidate_counts["semantic"]
        + candidate_counts["dense"]
        + candidate_counts["fallback"]
    )
    candidate_status = "pending"
    if candidate_counts["normalized"] > 0 or candidate_counts["merged"] > 0:
        candidate_status = "ok"
    elif job_status in _TERMINAL_STATES:
        candidate_status = "warning"

    recall_report = _read_json_file(debug_dir / "candidate_recall_report.json")
    funnel_summary = _read_json_file(debug_dir / "candidate_funnel_summary.json")
    trace_rows = _read_jsonl_rows(debug_dir / "candidate_trace.jsonl")
    trace_row_count = len(trace_rows)
    recall_status = "pending"
    recall_summary = "Recall-Funnel noch nicht geschrieben."
    recall_details: list[str] = []
    recall_metrics: list[dict] = []
    if isinstance(recall_report, dict) and recall_report:
        stage_counts = recall_report.get("stage_counts") if isinstance(recall_report.get("stage_counts"), dict) else {}
        largest_losses = recall_report.get("largest_losses") if isinstance(recall_report.get("largest_losses"), list) else []
        final_count_from_recall = int(stage_counts.get("final_moments") or 0)
        recall_status = "ok" if final_count_from_recall > 0 else ("warning" if job_status in _TERMINAL_STATES else "pending")
        recall_summary = (
            f"Recall-Funnel: {int(stage_counts.get('merged') or 0)} merged, "
            f"{int(stage_counts.get('ranked_pre_judge') or 0)} ranked, "
            f"{int(stage_counts.get('quality_kept') or 0)} Quality-kept, "
            f"{final_count_from_recall} final."
        )
        recall_metrics = [
            _metric("Dense Inventory", int(stage_counts.get("dense_inventory") or 0)),
            _metric("Merged", int(stage_counts.get("merged") or 0)),
            _metric("Ranked", int(stage_counts.get("ranked_pre_judge") or 0)),
            _metric("Judge Scope", int(stage_counts.get("judge_scope") or 0)),
            _metric("Quality Kept", int(stage_counts.get("quality_kept") or 0)),
            _metric("Final", final_count_from_recall),
            _metric("Trace Rows", trace_row_count),
        ]
        for loss in largest_losses[:3]:
            if not isinstance(loss, dict) or int(loss.get("lost_count") or 0) <= 0:
                continue
            recall_details.append(
                f"{loss.get('transition')}: {loss.get('lost_count')} verloren "
                f"({loss.get('input_count')} -> {loss.get('output_count')})."
            )
    elif job_status in _TERMINAL_STATES:
        recall_status = "disabled"
        recall_summary = "Kein Recall-Funnel-Artefakt gefunden; vermutlich alter Job oder vorzeitiger Abbruch."
    if trace_row_count > 0 and not isinstance(funnel_summary, dict):
        funnel_summary = {
            "rows": trace_row_count,
            "trace_samples": {
                "highest_ranked_rejected": trace_rows[:8],
            },
        }

    dense_inventory_rows = _jsonl_count(debug_dir / "candidates_dense_inventory.jsonl")
    dense_inventory_status = "pending"
    dense_inventory_details: list[str] = []
    dense_inventory_metrics = [
        _metric("Inventory", dense_inventory_rows),
    ]
    if dense_inventory_rows > 0:
        dense_inventory_status = "ok"
    elif job_status in _TERMINAL_STATES:
        dense_inventory_status = "warning"
        dense_inventory_details.append("Kein Dense-Inventory-Artefakt gefunden.")

    ranked_rows = _read_jsonl_rows(debug_dir / "ranked_candidates_v2.jsonl")
    if ranked_rows:
        ranked_count = len(ranked_rows)
        trained_rows = sum(1 for row in ranked_rows if row.get("trained_model_used"))
        judge_used_rows = sum(1 for row in ranked_rows if row.get("judge_used"))
        selected_rows = sum(1 for row in ranked_rows if row.get("selected_after_diversity"))
        best_pre_judge = max((float(row.get("pre_judge_rank_score") or 0.0) for row in ranked_rows), default=0.0)
        best_judged = max((float(row.get("judged_rank_score") or 0.0) for row in ranked_rows), default=0.0)
    else:
        ranked_count = _jsonl_count(debug_dir / "ranked_candidates_pre_judge.jsonl")
        trained_rows = 0
        judge_used_rows = 0
        selected_rows = _jsonl_count(debug_dir / "ranked_candidates.jsonl")
        best_pre_judge = 0.0
        best_judged = 0.0

    ranker_status = "pending"
    ranker_details: list[str] = []
    if ranked_count > 0:
        ranker_status = "ok" if trained_rows > 0 else "warning"
        if trained_rows == 0:
            ranker_details.append("Ranker-Ausgabe vorhanden, aber kein trainiertes Modell wurde in den Audit-Rows markiert.")
    elif job_status in _TERMINAL_STATES:
        ranker_status = "warning"

    judged_file_rows = _jsonl_count(debug_dir / "judged_candidates.jsonl")
    judge_stage = _read_json_file(debug_dir / "clip_judge_stage.json")
    judge_status = "pending"
    judge_details: list[str] = []
    judge_summary = f"{judge_used_rows or judged_file_rows} Kandidaten mit Judge-Signal."
    judged_count = judge_used_rows or judged_file_rows
    processed_candidates = 0
    attempted_candidates = 0
    recall_filtered_candidates = 0
    if isinstance(judge_stage, dict) and judge_stage:
        stage_status = str(judge_stage.get("status") or "")
        reason = judge_stage.get("reason")
        attempted = bool(judge_stage.get("attempted"))
        judged_count = int(judge_stage.get("judged_candidates") or 0)
        judge_top_k = int(judge_stage.get("judge_top_k") or 0)
        fallback_candidates = int(judge_stage.get("fallback_candidates") or 0)
        processed_candidates = int(judge_stage.get("processed_candidates") or 0)
        attempted_candidates = int(judge_stage.get("attempted_candidates") or 0)
        recall_filtered_candidates = int(judge_stage.get("recall_filtered_candidates") or 0)
        if stage_status == "ok":
            judge_status = "ok"
        elif stage_status == "disabled":
            judge_status = "disabled"
        elif stage_status in {"degraded", "failed"}:
            judge_status = "warning"
        elif stage_status:
            judge_status = "pending"
        if reason == "timeout":
            judge_details.append(
                f"Clip Judge ist nach {judge_stage.get('timeout_sec')}s getimeoutet; "
                f"Fallback auf {fallback_candidates} Pre-Judge Kandidaten."
            )
        elif reason == "llm_judge_failed":
            error = judge_stage.get("error")
            judge_details.append(f"Clip Judge Fehler: {error or judge_stage.get('error_type') or reason}.")
        elif reason == "empty_judge_result":
            judge_details.append("Clip Judge lief, lieferte aber kein nutzbares Judge-Ergebnis.")
        elif reason == "enable_clip_judge_false":
            judge_details.append("Clip Judge ist deaktiviert.")
        elif reason:
            judge_details.append(f"Clip Judge Status: {reason}.")
        judge_summary = (
            f"{processed_candidates or judged_count} von {judge_top_k} Clip-Judge Kandidaten verarbeitet."
            if attempted
            else "Clip Judge wurde nicht ausgeführt."
        )
    elif judge_used_rows > 0 or judged_file_rows > 0:
        judge_status = "ok"
    elif ranked_count > 0:
        judge_status = "warning"
        judge_details.append("Kein Clip-Judge Ergebnis gefunden; Auswahl basiert auf dem Pre-Judge Ranking.")
    if job_error and "judge" in str(job_error).lower():
        judge_status = "warning"
        judge_details.append(str(job_error))

    quality_rows = _read_jsonl_rows(debug_dir / "serving_quality_gate.jsonl")
    quality_count = len(quality_rows)
    quality_kept = sum(
        1
        for row in quality_rows
        if row.get("kept") is True or row.get("accepted") is True or row.get("status") == "kept"
    )
    quality_status = "pending"
    quality_details: list[str] = []
    if quality_count > 0:
        quality_status = "ok" if quality_kept > 0 else "warning"
        if quality_kept == 0:
            quality_details.append("Quality Gate hat keinen Kandidaten explizit als behalten markiert.")
    elif ranked_count > 0 and job_status in _TERMINAL_STATES:
        quality_status = "warning"
        quality_details.append("Kein Quality-Gate Audit gefunden.")

    final_moments = _read_json_file(debug_dir / "final_moments.json")
    final_count = len(final_moments) if isinstance(final_moments, list) else clip_count
    final_status = "pending"
    if final_count > 0 or clip_count > 0:
        final_status = "ok"
    elif job_status in _TERMINAL_STATES:
        final_status = "warning"

    stages = [
        _diagnostic_stage(
            "candidate_generation",
            "Kandidatengenerierung",
            candidate_status,
            f"{candidate_counts['normalized'] or candidate_counts['merged']} Kandidaten nach Merge/Normalisierung.",
            [
                _metric("Peak", candidate_counts["peak"]),
                _metric("Semantic", candidate_counts["semantic"]),
                _metric("Dense", candidate_counts["dense"]),
                _metric("Fallback", candidate_counts["fallback"]),
                _metric("Raw Total", total_raw_candidates),
            ],
        ),
        _diagnostic_stage(
            "candidate_recall",
            "Candidate Recall",
            recall_status,
            recall_summary,
            recall_metrics,
            recall_details,
        ),
        _diagnostic_stage(
            "dense_inventory",
            "Dense Inventory",
            dense_inventory_status,
            "Dense Inventory wird direkt in den Ranker-Pool gemerged.",
            dense_inventory_metrics,
            dense_inventory_details,
        ),
        _diagnostic_stage(
            "ranker",
            "Ranker",
            ranker_status,
            f"{ranked_count} Kandidaten gerankt, {trained_rows} mit trainiertem Ranker markiert.",
            [
                _metric("Ranked", ranked_count),
                _metric("ML genutzt", trained_rows),
                _metric("Top Pre-Judge", round(best_pre_judge, 3)),
            ],
            ranker_details,
        ),
        _diagnostic_stage(
            "clip_judge",
            "Clip Judge",
            judge_status,
            judge_summary,
            [
                _metric("Processed", processed_candidates),
                _metric("Attempted", attempted_candidates),
                _metric("Judged", judged_count or judge_used_rows or judged_file_rows),
                _metric("Filtered", recall_filtered_candidates),
                _metric("Top Judged", round(best_judged, 3)),
            ],
            judge_details,
        ),
        _diagnostic_stage(
            "quality_gate",
            "Quality Gate",
            quality_status,
            f"{quality_count} Audit-Rows, {quality_kept} explizit behalten.",
            [_metric("Audit Rows", quality_count), _metric("Behalten", quality_kept)],
            quality_details,
        ),
        _diagnostic_stage(
            "final_selection",
            "Finale Auswahl",
            final_status,
            f"{final_count or clip_count} finale Momente, {clip_count} geschnittene Clips.",
            [
                _metric("Momente", final_count),
                _metric("Clips", clip_count),
                _metric("Diversity Auswahl", selected_rows),
            ],
        ),
    ]
    warnings = sum(1 for stage in stages if stage["status"] == "warning")
    ok = sum(1 for stage in stages if stage["status"] == "ok")
    return {
        "available": True,
        "job_status": job_status,
        "last_modified": _debug_last_modified(debug_dir),
        "summary": f"{ok}/{len(stages)} Stages ok, {warnings} mit Warnung.",
        "stages": stages,
        "candidate_funnel": _candidate_funnel_payload(funnel_summary, trace_rows),
    }


def _candidate_funnel_payload(funnel_summary: dict | list | None, trace_rows: list[dict]) -> dict:
    if not isinstance(funnel_summary, dict) and not trace_rows:
        return {"available": False, "rows": 0, "samples": []}
    summary = funnel_summary if isinstance(funnel_summary, dict) else {}
    samples = []
    trace_samples = summary.get("trace_samples") if isinstance(summary.get("trace_samples"), dict) else {}
    for key in ("final", "highest_ranked_rejected", "quality_rejected"):
        values = trace_samples.get(key) if isinstance(trace_samples.get(key), list) else []
        for value in values:
            if isinstance(value, dict):
                sample = dict(value)
                sample.setdefault("sample_group", key)
                samples.append(sample)
    if not samples and trace_rows:
        samples = [dict(row) for row in trace_rows[:20]]
    return {
        "available": bool(summary or trace_rows),
        "rows": int(summary.get("rows") or len(trace_rows) or 0),
        "final_selected_rows": int(summary.get("final_selected_rows") or 0),
        "stage_counts": summary.get("stage_counts") if isinstance(summary.get("stage_counts"), dict) else {},
        "losses": summary.get("losses") if isinstance(summary.get("losses"), list) else [],
        "largest_losses": summary.get("largest_losses") if isinstance(summary.get("largest_losses"), list) else [],
        "drop_stage_counts": summary.get("drop_stage_counts") if isinstance(summary.get("drop_stage_counts"), dict) else {},
        "rejection_reason_counts": (
            summary.get("rejection_reason_counts")
            if isinstance(summary.get("rejection_reason_counts"), dict)
            else {}
        ),
        "source_counts": summary.get("source_counts") if isinstance(summary.get("source_counts"), dict) else {},
        "final_source_counts": (
            summary.get("final_source_counts") if isinstance(summary.get("final_source_counts"), dict) else {}
        ),
        "score_distributions": (
            summary.get("score_distributions") if isinstance(summary.get("score_distributions"), dict) else {}
        ),
        "quality_breakdown": (
            summary.get("quality_breakdown") if isinstance(summary.get("quality_breakdown"), dict) else {}
        ),
        "ranker_lift": (
            summary.get("ranker_lift") if isinstance(summary.get("ranker_lift"), dict) else {}
        ),
        "samples": samples[:40],
    }


def _write_job_request_metadata(job_id: str, payload: dict) -> None:
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "request.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _pipeline_subprocess_command(
    *,
    job_id: str,
    url: str | None,
    video_path: pathlib.Path | None,
    profile: str,
    strategy: dict,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_BACKEND_ROOT / "pipeline_worker.py"),
        "--job-id",
        job_id,
        "--profile",
        profile,
        "--strategy-json",
        json.dumps(_json_safe(strategy or {}), ensure_ascii=True),
    ]
    if video_path is not None:
        cmd.extend(["--video-path", str(video_path)])
    elif url:
        cmd.extend(["--url", url])
    return cmd


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return value


def _pipeline_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(_BACKEND_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast")
    env.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env["MPLCONFIGDIR"] = str(_writable_pipeline_cache_dir("matplotlib"))
    env.setdefault("WHISPER_LANGUAGE", "en")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    if (env.get("WHISPER_DEVICE") or "").strip().lower() == "cuda":
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault(
        "PIPELINE_ENABLE_DIARIZATION",
        "true" if settings.pipeline_enable_diarization else "false",
    )
    # Native face stacks still run in the isolated worker process, so a crash
    # cannot take down FastAPI. Operators can set this to 1 when a host is
    # unstable, but frontend jobs should use face signals by default.
    env.setdefault("AI_CLIPPING_DISABLE_FACE_DETECTION", "0")
    return env


def _writable_pipeline_cache_dir(name: str) -> pathlib.Path:
    preferred = pathlib.Path(settings.storage_dir).parent / name
    fallback = pathlib.Path(tempfile.gettempdir()) / f"ai_clipping_{name}"
    for candidate in (preferred, fallback):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return fallback


def _pipeline_subprocess_error(returncode: int) -> str:
    if returncode < 0:
        signal = -returncode
        if signal == 9:
            return "Pipeline subprocess exited from signal 9 (SIGKILL; likely OOM or external kill)"
        return f"Pipeline subprocess exited from signal {signal}"
    return f"Pipeline subprocess exited with code {returncode}"


def _cleanup_serving_job_source_media(job_id: str) -> None:
    if str(job_id or "").startswith(("twitch_vod", "judge_backfill_")):
        return

    job_dir = pathlib.Path(settings.storage_dir) / job_id
    for path in (job_dir / "video.mp4", job_dir / "audio.wav"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    chunks_dir = job_dir / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir, ignore_errors=True)


async def _run_pipeline_subprocess(
    *,
    job_id: str,
    url: str | None = None,
    video_path: pathlib.Path | None = None,
    profile: str,
    strategy: dict,
) -> None:
    cmd = _pipeline_subprocess_command(
        job_id=job_id,
        url=url,
        video_path=video_path,
        profile=profile,
        strategy=strategy,
    )
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_BACKEND_ROOT),
        env=_pipeline_subprocess_env(),
    )
    returncode = await process.wait()
    try:
        if returncode == 0:
            return
        job = get_job(job_id)
        if job is not None and job.status not in {"done", "failed", "cancelled"}:
            update_job(job_id, status="failed", error=_pipeline_subprocess_error(returncode))
    finally:
        _cleanup_serving_job_source_media(job_id)


def _quota_exhausted(message: str) -> bool:
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            "quota exceeded",
            "resource_exhausted",
            "billing details",
            "requestsperday",
            "inputtokenspermodelperminute",
            "limit: 0",
        )
    )


def _check_groq(api_key: str, model: str) -> tuple[bool, str | None]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True, None
    except Exception as e:
        msg = str(e)
        if "401" in msg or "auth" in msg.lower():
            return False, "Ungültiger API Key"
        if _quota_exhausted(msg):
            return False, "Groq-Quota erschöpft"
        if "429" in msg:
            return False, "Groq Rate Limit erreicht"
        return False, "Groq nicht erreichbar"


def _check_gemini(api_key: str, model: str) -> tuple[bool, str | None]:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True, None
    except Exception as e:
        msg = str(e)
        if "401" in msg or "API_KEY_INVALID" in msg:
            return False, "Ungültiger API Key"
        if _quota_exhausted(msg):
            return False, "Gemini-Quota erschöpft"
        if "429" in msg:
            return False, "Gemini Rate Limit erreicht"
        return False, "Gemini nicht erreichbar"


def _status_payload(
    *,
    label: str,
    name: str,
    model: str,
    active: bool,
    error: str | None,
    mode: str,
    fallback_name: str | None = None,
    fallback_model: str | None = None,
) -> dict[str, str | bool | None]:
    return {
        "label": label,
        "name": name,
        "model": model,
        "active": active,
        "error": error,
        "mode": mode,
        "fallback_name": fallback_name,
        "fallback_model": fallback_model,
    }


def _provider_payload(
    *,
    label: str,
    configured: bool,
    provider_name: str,
    provider_model: str,
    fallback_name: str,
    fallback_model: str,
    ok: bool,
    error: str | None,
) -> dict[str, str | bool | None]:
    if configured and ok:
        return _status_payload(
            label=label,
            name=provider_name,
            model=provider_model,
            active=True,
            error=None,
            mode="active",
        )
    if configured and error:
        return _status_payload(
            label=label,
            name=provider_name,
            model=provider_model,
            active=True,
            error=error,
            mode="fallback",
            fallback_name=fallback_name,
            fallback_model=fallback_model,
        )
    if not configured:
        return _status_payload(
            label=label,
            name=fallback_name,
            model=fallback_model,
            active=True,
            error=None,
            mode="fallback",
        )
    return _status_payload(
        label=label,
        name=provider_name,
        model=provider_model,
        active=False,
        error=error,
        mode="error",
    )


def _package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _face_detection_probe_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast")
    env.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")
    env["MPLCONFIGDIR"] = env.get(
        "MPLCONFIGDIR",
        str(_writable_pipeline_cache_dir("matplotlib")),
    )
    return env


def _run_face_detection_probe_script(script: str) -> tuple[dict[str, str | bool | None] | None, str | None]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=_face_detection_probe_env(),
        )
    except Exception as exc:
        return None, str(exc)

    payload = (result.stdout or "").strip().splitlines()
    if payload:
        try:
            data = json.loads(payload[-1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            if data.get("active"):
                return data, None
            error = data.get("error")
            return None, str(error) if error else "Face detection probe inactive"

    stderr = (result.stderr or "").strip().splitlines()
    stderr_tail = stderr[-1] if stderr else ""
    if result.returncode < 0:
        return None, f"Face detection probe exited with signal {-result.returncode}: {stderr_tail}"
    if result.returncode:
        return None, f"Face detection probe exited with code {result.returncode}: {stderr_tail}"
    return None, "Face detection probe produced no status"


def _probe_face_detection_status() -> dict[str, str | bool | None]:
    mediapipe_script = f"""
import json
import pathlib

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    else:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

        model_path = pathlib.Path({str(pathlib.Path(__file__).resolve().parents[1] / "face_detection_short_range.tflite")!r})
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe face model missing: {{model_path}}")
        FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=BaseOptions.Delegate.CPU,
                ),
                min_detection_confidence=0.5,
            )
        )
    print(json.dumps({{
        "label": "Face Detection",
        "name": "MediaPipe",
        "model": "face_detection",
        "active": True,
        "error": None,
        "mode": "active",
    }}))
except Exception as exc:
    print(json.dumps({{"active": False, "error": str(exc)}}))
"""

    mtcnn_script = """
import json

try:
    from mtcnn import MTCNN
    MTCNN()
    print(json.dumps({
        "label": "Face Detection",
        "name": "MTCNN",
        "model": "mtcnn",
        "active": True,
        "error": None,
        "mode": "fallback",
    }))
except Exception as exc:
    print(json.dumps({"active": False, "error": str(exc)}))
"""

    opencv_script = """
import json

try:
    import cv2
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Haar cascade not available")
    print(json.dumps({
        "label": "Face Detection",
        "name": "OpenCV",
        "model": "haar cascade",
        "active": True,
        "error": None,
        "mode": "fallback",
    }))
except Exception as exc:
    print(json.dumps({"active": False, "error": str(exc)}))
"""

    data, mediapipe_error = _run_face_detection_probe_script(mediapipe_script)
    if data:
        return data

    data, mtcnn_error = _run_face_detection_probe_script(mtcnn_script)
    if data:
        data["error"] = mediapipe_error or "MediaPipe nicht verfuegbar"
        return data

    data, opencv_error = _run_face_detection_probe_script(opencv_script)
    if data:
        data["error"] = mtcnn_error or mediapipe_error
        return data

    return _status_payload(
        label="Face Detection",
        name="Disabled",
        model="none",
        active=False,
        error=opencv_error or mtcnn_error or mediapipe_error or "Face detection probe failed",
        mode="error",
    )


def _pipeline_component_statuses() -> dict[str, dict[str, str | bool | None]]:
    whisper_available = _package_available("whisper")
    sentence_transformers_available = _package_available("sentence_transformers")
    deepface_available = _package_available("deepface")
    tf_keras_available = _package_available("tf_keras")
    pyannote_available = _package_available("pyannote.audio")

    embedding_status = (
        _status_payload(
            label="Candidate Embeddings",
            name="SentenceTransformers",
            model=_SENTENCE_TRANSFORMERS_MODEL,
            active=True,
            error=None,
            mode="active",
        )
        if sentence_transformers_available
        else _status_payload(
            label="Candidate Embeddings",
            name="Regex Fallback",
            model="heuristic",
            active=True,
            error="sentence-transformers nicht installiert",
            mode="fallback",
        )
    )

    expression_status = (
        _status_payload(
            label="Expression Analysis",
            name="DeepFace",
            model="emotion analyzer",
            active=True,
            error=None,
            mode="active",
        )
        if deepface_available and tf_keras_available
        else _status_payload(
            label="Expression Analysis",
            name="Disabled",
            model="none",
            active=False,
            error=(
                "tf-keras fehlt"
                if deepface_available and not tf_keras_available
                else "DeepFace nicht installiert"
            ),
            mode="inactive",
        )
    )

    diarization_status = (
        _status_payload(
            label="Diarization",
            name="Pyannote",
            model=_PYANNOTE_MODEL,
            active=True,
            error=None,
            mode="active",
        )
        if settings.hf_token and pyannote_available
        else _status_payload(
            label="Diarization",
            name="Disabled",
            model=_PYANNOTE_MODEL,
            active=False,
            error=(
                "HF token fehlt"
                if not settings.hf_token
                else "pyannote.audio nicht installiert"
            ),
            mode="inactive",
        )
    )

    global _face_detection_cache
    if _face_detection_cache is None:
        _face_detection_cache = _probe_face_detection_status()

    return {
        "transcription": (
            _status_payload(
                label="Transcription",
                name="Whisper",
                model=settings.whisper_model,
                active=True,
                error=None,
                mode="active",
            )
            if whisper_available
            else _status_payload(
                label="Transcription",
                name="Disabled",
                model=settings.whisper_model,
                active=False,
                error="openai-whisper nicht installiert",
                mode="error",
            )
        ),
        "semantic_embeddings": embedding_status,
        "face_detection": dict(_face_detection_cache),
        "expression_analysis": expression_status,
        "diarization": diarization_status,
    }


@status_router.get("/status")
async def get_status():
    global _status_cache
    now = _time.time()

    # Return cached result if still fresh
    if _status_cache and now - _status_cache.get("_ts", 0) < _STATUS_TTL:
        result = dict(_status_cache)
        result.pop("_ts", None)
        return result

    loop = asyncio.get_running_loop()
    gemini_configured = bool(settings.gemini_api_key)

    gemini_ok, gemini_err = (False, None)

    checks = []
    if gemini_configured:
        checks.append(loop.run_in_executor(None, _check_gemini, settings.gemini_api_key, settings.gemini_model))
    checks.append(loop.run_in_executor(None, _pipeline_component_statuses))

    results = await asyncio.gather(*checks)
    idx = 0
    if gemini_configured:
        gemini_ok, gemini_err = results[idx]
        idx += 1
    pipeline_components = results[idx]

    result = {
        "providers": {
            "semantic": _status_payload(
                label="Scoring",
                name="Ollama",
                model=settings.ollama_model,
                active=True,
                error=None,
                mode="active",
                fallback_name="Groq" if settings.groq_api_key else None,
                fallback_model=settings.groq_model if settings.groq_api_key else None,
            ),
            "refinement": _provider_payload(
                label="Refinement",
                configured=gemini_configured,
                provider_name="Gemini",
                provider_model=settings.gemini_model,
                fallback_name="Ollama",
                fallback_model=settings.ollama_model,
                ok=gemini_ok,
                error=gemini_err,
            ),
        },
        "pipeline": pipeline_components,
    }

    _status_cache = {**result, "_ts": now}
    return result


def _request_strategy(
    *,
    platform: str | None = None,
    goal: str | None = None,
    tone: str | None = None,
    topic: str | None = None,
    must_include: list[str] | None = None,
    exclude: list[str] | None = None,
    speaker: str | None = None,
    time_range: dict | None = None,
    clip_style: str | None = None,
    variant_count: int | None = None,
) -> dict:
    return clip_strategy.normalize_strategy(
        {
            "platform": platform,
            "goal": goal,
            "tone": tone,
            "topic": topic,
            "must_include": must_include or [],
            "exclude": exclude or [],
            "speaker": speaker,
            "time_range": time_range,
            "clip_style": clip_style,
            "variant_count": variant_count,
        }
    )


def _coerce_optional_text(value) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_optional_number(value) -> float | int | None:
    return value if isinstance(value, (int, float)) else None


def _coerce_csv_list(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@router.post("", response_model=JobStatus, status_code=202)
async def submit_job(request: JobRequest):
    job_id = str(uuid.uuid4())
    effective_profile = request.profile or settings.clip_profile
    job = create_job(job_id, profile=effective_profile)
    bind_job_session(job_id, request.session_id)
    strategy = _request_strategy(
        platform=request.platform,
        goal=request.goal,
        tone=request.tone,
        topic=request.topic,
        must_include=request.must_include,
        exclude=request.exclude,
        speaker=request.speaker,
        time_range=request.time_range.model_dump(exclude_none=True) if request.time_range else None,
        clip_style=request.clip_style,
        variant_count=request.variant_count,
    )
    _write_job_request_metadata(
        job_id,
        {
            "url": request.url,
            "profile": effective_profile,
            "session_id": request.session_id,
            "platform": request.platform,
            "goal": request.goal,
            "tone": request.tone,
            "topic": request.topic,
            "must_include": request.must_include,
            "exclude": request.exclude,
            "speaker": request.speaker,
            "time_range": request.time_range.model_dump(exclude_none=True) if request.time_range else None,
            "clip_style": request.clip_style,
            "variant_count": request.variant_count,
        },
    )

    asyncio.create_task(
        _run_pipeline_subprocess(
            job_id=job_id,
            url=request.url,
            profile=effective_profile,
            strategy=strategy,
        )
    )

    return job


@router.post("/upload", response_model=JobStatus, status_code=202)
async def upload_job(
    file: UploadFile = File(...),
    profile: str | None = Form(None),
    session_id: str | None = Form(None),
    platform: str | None = Form(None),
    goal: str | None = Form(None),
    tone: str | None = Form(None),
    topic: str | None = Form(None),
    must_include: str | None = Form(None),
    exclude: str | None = Form(None),
    speaker: str | None = Form(None),
    time_range_start: float | None = Form(None),
    time_range_end: float | None = Form(None),
    clip_style: str | None = Form(None),
    variant_count: int | None = Form(None),
):
    suffix = pathlib.Path(file.filename or "").suffix.lower()
    allowed_suffixes = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".flv"}
    if suffix and suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Nicht unterstütztes Dateiformat")

    spooled = file.file
    size_limit = settings.max_upload_size_bytes
    if size_limit is not None and size_limit > 0:
        spooled.seek(0, io.SEEK_END)
        size_bytes = spooled.tell()
        spooled.seek(0)
        if size_bytes > size_limit:
            try:
                spooled.close()
            except Exception:
                pass
            raise HTTPException(status_code=413, detail="Datei ist zu groß")

    job_id = str(uuid.uuid4())
    profile = _coerce_optional_text(profile)
    session_id = _coerce_optional_text(session_id)
    platform = _coerce_optional_text(platform)
    goal = _coerce_optional_text(goal)
    tone = _coerce_optional_text(tone)
    topic = _coerce_optional_text(topic)
    speaker = _coerce_optional_text(speaker)
    clip_style = _coerce_optional_text(clip_style)
    time_range_start = _coerce_optional_number(time_range_start)
    time_range_end = _coerce_optional_number(time_range_end)
    variant_count = _coerce_optional_number(variant_count)
    effective_profile = profile or settings.clip_profile
    job = create_job(job_id, profile=effective_profile)
    bind_job_session(job_id, session_id)
    strategy = _request_strategy(
        platform=platform,
        goal=goal,
        tone=tone,
        topic=topic,
        must_include=_coerce_csv_list(must_include),
        exclude=_coerce_csv_list(exclude),
        speaker=speaker,
        time_range={
            key: value
            for key, value in {
                "start_sec": time_range_start,
                "end_sec": time_range_end,
            }.items()
            if value is not None
        } or None,
        clip_style=clip_style,
        variant_count=variant_count,
    )

    job_dir = pathlib.Path(settings.storage_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "video.mp4"
    _write_job_request_metadata(
        job_id,
        {
            "url": "",
            "profile": effective_profile,
            "session_id": session_id,
            "platform": platform,
            "goal": goal,
            "tone": tone,
            "topic": topic,
            "must_include": _coerce_csv_list(must_include),
            "exclude": _coerce_csv_list(exclude),
            "speaker": speaker,
            "time_range": {
                key: value
                for key, value in {
                    "start_sec": time_range_start,
                    "end_sec": time_range_end,
                }.items()
                if value is not None
            } or None,
            "clip_style": clip_style,
            "variant_count": variant_count,
            "original_filename": file.filename,
        },
    )

    # Detach the SpooledTemporaryFile from FastAPI's lifecycle so we can copy
    # it in the background without blocking the HTTP response.
    file.file = io.BytesIO()  # replace so FastAPI cleanup doesn't touch our file

    async def _save_and_run():
        try:
            def _copy():
                spooled.seek(0)
                with video_path.open("wb") as f:
                    shutil.copyfileobj(spooled, f)
                try:
                    spooled.close()
                except Exception:
                    pass

            await asyncio.to_thread(_copy)
            await _run_pipeline_subprocess(
                job_id=job_id,
                video_path=video_path,
                profile=effective_profile,
                strategy=strategy,
            )
        except Exception as exc:
            update_job(job_id, status="failed", error=str(exc))
            shutil.rmtree(job_dir, ignore_errors=True)

    asyncio.create_task(_save_and_run())

    return job


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, session_id: str | None = None):
    job = touch_job_session(job_id, session_id)
    if job is None:
        job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    diagnostics = _job_pipeline_diagnostics(job_id, job)
    if isinstance(job, dict):
        payload = dict(job)
        payload["pipeline_diagnostics"] = diagnostics
        return payload
    if hasattr(job, "model_copy"):
        return job.model_copy(update={"pipeline_diagnostics": diagnostics})
    return job


@router.post("/{job_id}/cancel", response_model=JobStatus)
async def cancel_job_status(job_id: str):
    job = cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/{job_id}/feedback", response_model=ClipFeedbackResponse)
async def submit_clip_feedback(job_id: str, payload: ClipFeedbackPayload):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        return feedback_store.update_job_feedback(pathlib.Path(settings.storage_dir), job_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail="Candidate not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{job_id}/review-candidates", response_model=ReviewCandidateResponse)
async def get_review_candidates(job_id: str, limit: int = 80):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    safe_limit = min(max(int(limit or 0), 1), 200)
    debug_dir = pathlib.Path(settings.storage_dir) / job_id / "debug"
    rows = _read_jsonl_rows(debug_dir / "review_candidates.jsonl")
    if not rows:
        rows = _review_candidates_from_feedback(debug_dir)
    else:
        rows = _merge_review_feedback(rows, debug_dir)
    return {"job_id": job_id, "candidates": rows[:safe_limit]}


def _merge_review_feedback(rows: list[dict], debug_dir: pathlib.Path) -> list[dict]:
    feedback_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in _read_jsonl_rows(debug_dir / "feedback.jsonl")
        if str(row.get("candidate_id") or "").strip()
    }
    merged: list[dict] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        feedback = feedback_by_id.get(candidate_id) or {}
        if feedback:
            row = {
                **row,
                "feedback_rating": feedback.get("feedback_rating"),
                "feedback_tags": list(feedback.get("feedback_tags") or []),
                "feedback_notes": feedback.get("feedback_notes"),
            }
        merged.append(row)
    return merged


def _review_candidates_from_feedback(debug_dir: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for row in _read_jsonl_rows(debug_dir / "feedback.jsonl"):
        candidate_id = str(row.get("candidate_id") or "").strip()
        start = _as_optional_float(row.get("final_start") if row.get("selected") else row.get("proposed_start"))
        end = _as_optional_float(row.get("final_end") if row.get("selected") else row.get("proposed_end"))
        if start is None:
            start = _as_optional_float(row.get("proposed_start"))
        if end is None:
            end = _as_optional_float(row.get("proposed_end"))
        if not candidate_id or start is None or end is None or end <= start:
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "preview_url": None,
                "start_time": start,
                "end_time": end,
                "duration": end - start,
                "rank_score": row.get("rank_score"),
                "judge_score": row.get("judge_score"),
                "selected": bool(row.get("selected")),
                "rejected_reason": row.get("rejected_reason"),
                "rejection_source": row.get("rejection_source"),
                "judge_reason": row.get("judge_reason"),
                "feedback_rating": row.get("feedback_rating"),
                "feedback_tags": list(row.get("feedback_tags") or []),
                "feedback_notes": row.get("feedback_notes"),
            }
        )
    rows.sort(
        key=lambda item: (
            0 if item.get("selected") else 1,
            -_as_float(item.get("judge_score")),
            -_as_float(item.get("rank_score")),
            _as_float(item.get("start_time")),
        )
    )
    return rows


def _as_optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float:
    parsed = _as_optional_float(value)
    return parsed if parsed is not None else 0.0
