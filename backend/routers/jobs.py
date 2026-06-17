import asyncio
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import uuid
from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from config import settings
from models import (
    JobRequest,
    JobStatus,
    ClipFeedbackPayload,
    ClipFeedbackResponse,
    ReviewCandidateResponse,
)
import session_lease
from job_lifecycle import is_terminal
from job_manager import bind_job_session, cancel_job, create_job, delete_job, get_job, touch_job_session
from job_manager import update_job
from pipeline_runner import run_job_subprocess, writable_cache_dir
from services import clip_strategy, feedback_store
from services.job_artifact_store import JobArtifactStore, RealFsJobArtifactStore

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


def _gt_matches_for_vod(vod_id: str) -> list[dict]:
    """Search all gt_run eval manifests for GT moments belonging to vod_id."""
    gt_runs_root = _BACKEND_ROOT / "tmp" / "gt_runs"
    if not gt_runs_root.exists():
        return []
    for run_dir in sorted(gt_runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest = run_dir / "eval_manifest.jsonl"
        if not manifest.exists():
            continue
        try:
            for line in manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("vod_id", "")) == vod_id:
                    return list(row.get("gt_moments") or [])
        except OSError:
            continue
    return []


def _vod_id_from_url(url: str | None) -> str | None:
    """Extract Twitch VOD ID from a URL like https://www.twitch.tv/videos/12345."""
    import re
    if not url:
        return None
    m = re.search(r"/videos/(\d+)", url)
    return m.group(1) if m else None


def _job_pipeline_diagnostics(
    job_id: str,
    job: JobStatus | dict | None = None,
    *,
    store: JobArtifactStore | None = None,
) -> dict:
    store = store or RealFsJobArtifactStore(settings.storage_dir)
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

    if not store.debug_artifacts_present(job_id):
        return {
            "available": False,
            "job_status": job_status,
            "summary": "Noch keine Pipeline-Debugdaten vorhanden.",
            "stages": [],
        }

    segments_count = len(store.read_segments(job_id))
    transcript_status = "pending"
    transcript_details: list[str] = []
    if segments_count > 0:
        transcript_status = "ok"
    elif is_terminal(job_status):
        transcript_status = "warning"
        transcript_details.append("Keine Transcript-Segmente gefunden.")

    llm_stage = store.read_stage(job_id)
    llm_moments_count = len(store.read_moments(job_id))
    llm_status = "pending"
    llm_summary = "LLM Moment Finder noch nicht ausgeführt."
    llm_details: list[str] = []
    llm_metrics: list[dict] = []
    if isinstance(llm_stage, dict) and llm_stage:
        stage_status = str(llm_stage.get("status") or "")
        if stage_status == "ok":
            llm_status = "ok"
            llm_summary = f"{llm_moments_count} Momente gefunden."
        elif stage_status == "failed":
            llm_status = "warning"
            llm_summary = "LLM Moment Finder fehlgeschlagen."
        else:
            llm_summary = f"Status: {stage_status or 'running'}."
        if llm_stage.get("error"):
            llm_details.append(str(llm_stage["error"]))
        llm_metrics = [
            _metric("Momente", llm_moments_count),
            _metric("Provider", str(llm_stage.get("provider") or "-")),
            _metric("Model", str(llm_stage.get("model") or "default")),
            _metric("Max Momente", int(llm_stage.get("max_moments") or 0)),
        ]
    elif is_terminal(job_status):
        llm_status = "warning"
        llm_details.append("Kein LLM-Stage-Artefakt gefunden.")

    final_moments_data = store.read_final_moments(job_id)
    final_count = 0
    if isinstance(final_moments_data, dict):
        moments_list = final_moments_data.get("moments")
        final_count = int(final_moments_data.get("count") or (len(moments_list) if isinstance(moments_list, list) else 0))
    clip_status = "pending"
    clip_details: list[str] = []
    if clip_count > 0 or final_count > 0:
        clip_status = "ok"
    elif is_terminal(job_status):
        clip_status = "warning"
        clip_details.append("Keine Clips geschnitten.")

    stages = [
        _diagnostic_stage(
            "transcript",
            "Transkription",
            transcript_status,
            f"{segments_count} Transcript-Segmente.",
            [_metric("Segmente", segments_count)],
            transcript_details,
        ),
        _diagnostic_stage(
            "llm_moment_finder",
            "LLM Moment Finder",
            llm_status,
            llm_summary,
            llm_metrics,
            llm_details,
        ),
        _diagnostic_stage(
            "clipping",
            "Clips schneiden",
            clip_status,
            f"{final_count} Momente → {clip_count} Clips.",
            [_metric("Momente", final_count), _metric("Clips", clip_count)],
            clip_details,
        ),
    ]
    # GT matches stage — only shown for Twitch VOD jobs that have a match in the eval manifest
    request_meta = store.read_request_metadata(job_id)
    source_url = request_meta.get("url") if isinstance(request_meta, dict) else None
    vod_id = _vod_id_from_url(source_url)
    if vod_id:
        gt_moments = _gt_matches_for_vod(vod_id)
        if gt_moments:
            gt_details = []
            for m in sorted(gt_moments, key=lambda x: float(x.get("start") or 0)):
                start = int(float(m.get("start") or 0))
                end = int(float(m.get("end") or 0))
                sim = float(m.get("match_similarity") or 0)
                views = int(float(m.get("views") or 0))
                desc = str(m.get("desc") or "")[:60]
                twitch_link = f"https://www.twitch.tv/videos/{vod_id}?t={start}s"
                gt_details.append(
                    f"t={start}s–{end}s  sim={sim:.2f}  {views//1000}k views — {desc}  → {twitch_link}"
                )
            stages.append(_diagnostic_stage(
                "gt_matches",
                "Ground Truth Matches",
                "ok",
                f"{len(gt_moments)} verifizierte GT-Momente für VOD {vod_id}.",
                [
                    _metric("GT-Momente", len(gt_moments)),
                    _metric("VOD", vod_id),
                ],
                gt_details,
            ))

    warnings = sum(1 for stage in stages if stage["status"] == "warning")
    ok = sum(1 for stage in stages if stage["status"] == "ok")
    return {
        "available": True,
        "job_status": job_status,
        "last_modified": store.debug_last_modified(job_id),
        "summary": f"{ok}/{len(stages)} Stages ok, {warnings} mit Warnung.",
        "stages": stages,
        "candidate_funnel": {"available": False, "rows": 0, "samples": []},
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


def _cleanup_serving_job_source_media(job_id: str) -> None:
    if str(job_id or "").startswith(("twitch_vod", "judge_backfill_", "pipeline_debug_")):
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
    # Cross the shared pipeline_runner seam for the launch + failure marking;
    # the serving router only adds its own post-run source-media cleanup, which
    # the training adapter (routers.pipeline_debug) deliberately does not do.
    try:
        await run_job_subprocess(
            job_id=job_id,
            url=url,
            video_path=video_path,
            profile=profile,
            strategy=strategy,
        )
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
        str(writable_cache_dir("matplotlib")),
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
    archetype_judge: bool | None = None,
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
            "archetype_judge": archetype_judge,
        }
    )


def _coerce_optional_text(value) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_optional_number(value) -> float | int | None:
    return value if isinstance(value, (int, float)) else None


def _coerce_optional_bool(value) -> bool | None:
    # Tri-state: a real bool passes through; anything else (an unset Form sentinel when the
    # endpoint is called directly, a stray string) becomes None → inherit the global default.
    return value if isinstance(value, bool) else None


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
        archetype_judge=request.archetype_judge,
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
            "archetype_judge": request.archetype_judge,
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
    archetype_judge: bool | None = Form(None),
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
    archetype_judge = _coerce_optional_bool(archetype_judge)
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
        archetype_judge=archetype_judge,
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
            "archetype_judge": archetype_judge,
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
    # Explicit request-path drive: a matching session_id has just renewed the lease
    # above, so an expired lease here means the owning tab is gone — cancel the orphan.
    if session_lease.drive_expiry(job_id):
        job = get_job(job_id)
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


@router.delete("/{job_id}")
async def delete_job_run(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not is_terminal(job.status):
        raise HTTPException(status_code=409, detail="Job is still running — cancel it first")
    deleted = delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    return {"deleted": job_id}


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
    store = RealFsJobArtifactStore(settings.storage_dir)
    rows = store.read_review_candidates(job_id)
    if not rows:
        rows = _review_candidates_from_feedback(store, job_id)
    else:
        rows = _merge_review_feedback(rows, store, job_id)
    return {"job_id": job_id, "candidates": rows[:safe_limit]}


def _merge_review_feedback(rows: list[dict], store: JobArtifactStore, job_id: str) -> list[dict]:
    feedback_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in store.read_feedback(job_id)
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


def _review_candidates_from_feedback(store: JobArtifactStore, job_id: str) -> list[dict]:
    rows: list[dict] = []
    for row in store.read_feedback(job_id):
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
