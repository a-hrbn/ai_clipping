import os
import pathlib
import certifi

# Fix SSL certificate verification on macOS — Python does not use the system
# keychain, so urllib (used by Whisper model downloads, yt-dlp, etc.) fails
# with CERTIFICATE_VERIFY_FAILED. Point it at certifi's bundled CA store.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import settings
from routers.jobs import router as jobs_router, status_router

# Ensure storage directory exists
pathlib.Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Clipping", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(status_router)

app.mount("/static", StaticFiles(directory=settings.storage_dir), name="static")
