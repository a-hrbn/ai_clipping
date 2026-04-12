import asyncio
import io
import pathlib
import shutil
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


@status_router.get("/status")
async def get_status():
    global _status_cache
    now = _time.time()

    # Return cached result if still fresh
    if _status_cache and now - _status_cache.get("_ts", 0) < _STATUS_TTL:
        result = dict(_status_cache)
        result.pop("_ts", None)
        return result

    loop = asyncio.get_event_loop()
    groq_configured = bool(settings.groq_api_key)
    gemini_configured = bool(settings.gemini_api_key)

    groq_ok, groq_err = (False, None)
    gemini_ok, gemini_err = (False, None)

    checks = []
    if groq_configured:
        checks.append(loop.run_in_executor(None, _check_groq, settings.groq_api_key, settings.groq_model))
    if gemini_configured:
        checks.append(loop.run_in_executor(None, _check_gemini, settings.gemini_api_key, settings.gemini_model))

    results = await asyncio.gather(*checks)
    idx = 0
    if groq_configured:
        groq_ok, groq_err = results[idx]; idx += 1
    if gemini_configured:
        gemini_ok, gemini_err = results[idx]

    result = {
        "providers": {
            "semantic": {
                "name": "Groq" if groq_configured else "Ollama",
                "model": settings.groq_model if groq_configured else settings.ollama_model,
                "active": groq_configured and groq_ok,
                "error": groq_err if groq_configured else None,
            },
            "refinement": {
                "name": "Gemini" if gemini_configured else "Ollama",
                "model": settings.gemini_model if gemini_configured else settings.ollama_model,
                "active": gemini_configured and gemini_ok,
                "error": gemini_err if gemini_configured else None,
            },
        }
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

    job_id = str(uuid.uuid4())
    effective_profile = profile or settings.clip_profile
    job = create_job(job_id, profile=effective_profile)

    job_dir = pathlib.Path(settings.storage_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "video.mp4"

    # Detach the SpooledTemporaryFile from FastAPI's lifecycle so we can copy
    # it in the background without blocking the HTTP response.
    spooled = file.file
    spooled.seek(0, io.SEEK_END)
    size_bytes = spooled.tell()
    spooled.seek(0)
    if size_bytes > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Datei ist zu groß")
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
