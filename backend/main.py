import os
import pathlib
import re
import certifi

# Fix SSL certificate verification on macOS — Python does not use the system
# keychain, so urllib (used by Whisper model downloads, yt-dlp, etc.) fails
# with CERTIFICATE_VERIFY_FAILED. Point it at certifi's bundled CA store.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from config import settings
from routers.jobs import router as jobs_router, status_router
from routers.llm_moment_review import router as llm_moment_review_router

# Ensure storage directory exists
pathlib.Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Clipping", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(status_router)
app.include_router(llm_moment_review_router)

# Only generated clip files are publicly addressable. The broader storage dir
# contains debug JSONL dumps, transcripts, chat replays, and jobs.json — none of
# which should be reachable via an unauthenticated static mount.
_JOB_ID_RE = re.compile(r"^[0-9a-zA-Z][0-9a-zA-Z\-_]{0,63}$")
_CLIP_FILENAME_RE = re.compile(r"^[0-9a-zA-Z][0-9a-zA-Z\-_.]{0,63}\.(mp4|webm|mov|m4v|mkv)$")


@app.get("/static/{job_id}/clips/{filename}")
def serve_clip(job_id: str, filename: str) -> FileResponse:
    if not _JOB_ID_RE.fullmatch(job_id) or not _CLIP_FILENAME_RE.fullmatch(filename):
        raise HTTPException(status_code=404, detail="Not found")
    storage_root = pathlib.Path(settings.storage_dir).resolve()
    clip_path = (storage_root / job_id / "clips" / filename).resolve()
    try:
        clip_path.relative_to(storage_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    if not clip_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(clip_path), media_type="video/mp4", filename=filename)


@app.get("/static/{job_id}/review_clips/{filename}")
def serve_review_clip(job_id: str, filename: str) -> FileResponse:
    if not _JOB_ID_RE.fullmatch(job_id) or not _CLIP_FILENAME_RE.fullmatch(filename):
        raise HTTPException(status_code=404, detail="Not found")
    storage_root = pathlib.Path(settings.storage_dir).resolve()
    clip_path = (storage_root / job_id / "review_clips" / filename).resolve()
    try:
        clip_path.relative_to(storage_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    if not clip_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(clip_path), media_type="video/mp4", filename=filename)
