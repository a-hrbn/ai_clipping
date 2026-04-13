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
from models import JobRequest, JobStatus, ClipFeedbackPayload, ClipFeedbackResponse
from job_manager import create_job, get_job
from services import feedback_store

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

status_router = APIRouter(prefix="/api", tags=["status"])

# Cache: result is valid for 5 minutes to avoid burning API quota on every page load
import time as _time
_status_cache: dict = {}
_STATUS_TTL = 300
_SENTENCE_TRANSFORMERS_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_PYANNOTE_MODEL = "speaker-diarization-3.1"


def _check_groq(api_key: str, model: str) -> tuple[bool, str | None]:
    try:
        from openai import OpenAI
        from openai import AuthenticationError, RateLimitError
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
        if "429" in msg:
            return True, None  # Key is valid, just rate-limited — treat as active
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
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return True, None  # Key is valid, just rate-limited — treat as active
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


def _probe_face_detection_status() -> dict[str, str | bool | None]:
    script = f"""
import json
import pathlib

mediapipe_error = None
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
    raise SystemExit(0)
except Exception as exc:
    mediapipe_error = str(exc)

try:
    from mtcnn import MTCNN
    MTCNN()
    print(json.dumps({{
        "label": "Face Detection",
        "name": "MTCNN",
        "model": "mtcnn",
        "active": True,
        "error": mediapipe_error or "MediaPipe nicht verfuegbar",
        "mode": "fallback",
    }}))
    raise SystemExit(0)
except Exception as exc:
    mtcnn_error = str(exc)

try:
    import cv2
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Haar cascade not available")
    print(json.dumps({{
        "label": "Face Detection",
        "name": "OpenCV",
        "model": "haar cascade",
        "active": True,
        "error": mediapipe_error or mtcnn_error,
        "mode": "fallback",
    }}))
except Exception as exc:
    print(json.dumps({{
        "label": "Face Detection",
        "name": "Disabled",
        "model": "none",
        "active": False,
        "error": str(exc) or mtcnn_error or mediapipe_error,
        "mode": "error",
    }}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env={
                **os.environ,
                "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"),
            },
        )
        payload = (result.stdout or "").strip().splitlines()
        if payload:
            data = json.loads(payload[-1])
            if isinstance(data, dict):
                return data
    except Exception as exc:
        return _status_payload(
            label="Face Detection",
            name="Disabled",
            model="none",
            active=False,
            error=str(exc),
            mode="error",
        )
    return _status_payload(
        label="Face Detection",
        name="Disabled",
        model="none",
        active=False,
        error="Face detection probe failed",
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
        "face_detection": _probe_face_detection_status(),
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
    groq_configured = bool(settings.groq_api_key)
    gemini_configured = bool(settings.gemini_api_key)

    groq_ok, groq_err = (False, None)
    gemini_ok, gemini_err = (False, None)

    checks = []
    if groq_configured:
        checks.append(loop.run_in_executor(None, _check_groq, settings.groq_api_key, settings.groq_model))
    if gemini_configured:
        checks.append(loop.run_in_executor(None, _check_gemini, settings.gemini_api_key, settings.gemini_model))
    checks.append(loop.run_in_executor(None, _pipeline_component_statuses))

    results = await asyncio.gather(*checks)
    idx = 0
    if groq_configured:
        groq_ok, groq_err = results[idx]; idx += 1
    if gemini_configured:
        gemini_ok, gemini_err = results[idx]
        idx += 1
    pipeline_components = results[idx]

    result = {
        "providers": {
            "semantic": _provider_payload(
                label="Scoring",
                configured=groq_configured,
                provider_name="Groq",
                provider_model=settings.groq_model,
                fallback_name="Ollama",
                fallback_model=settings.ollama_model,
                ok=groq_ok,
                error=groq_err,
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


@router.post("", response_model=JobStatus, status_code=202)
async def submit_job(request: JobRequest):
    job_id = str(uuid.uuid4())
    effective_profile = request.profile or settings.clip_profile
    job = create_job(job_id, profile=effective_profile)

    from pipeline import run_pipeline
    asyncio.create_task(run_pipeline(job_id, url=request.url, profile=effective_profile))

    return job


@router.post("/upload", response_model=JobStatus, status_code=202)
async def upload_job(file: UploadFile = File(...), profile: str | None = Form(None)):
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
    effective_profile = profile or settings.clip_profile
    job = create_job(job_id, profile=effective_profile)

    job_dir = pathlib.Path(settings.storage_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "video.mp4"

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
            from pipeline import run_pipeline
            await run_pipeline(job_id, video_path=video_path, profile=effective_profile)
        except Exception as exc:
            from job_manager import update_job
            update_job(job_id, status="failed", error=str(exc))
            shutil.rmtree(job_dir, ignore_errors=True)

    asyncio.create_task(_save_and_run())

    return job


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = get_job(job_id)
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
