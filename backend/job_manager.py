import json
import logging
import pathlib
import threading
import time
from typing import Dict

from config import settings
from models import JobStatus

logger = logging.getLogger(__name__)

_jobs: Dict[str, JobStatus] = {}
_job_created_at: Dict[str, float] = {}   # job_id → creation timestamp (epoch seconds)
_cancelled_jobs: set[str] = set()
_job_session_id: Dict[str, str] = {}
_job_session_expires_at: Dict[str, float] = {}
_lock = threading.RLock()
_jobs_file = pathlib.Path(settings.storage_dir) / "jobs.json"
_last_persisted_at = 0.0

_JOB_TTL_SECONDS = 2 * 3600  # evict in-memory records after 2 hours
_PERSIST_INTERVAL_SECONDS = 5.0
# Browser tabs may heavily throttle timers/polling while unfocused. Keep the
# server-side session lease long enough that background tabs do not get treated
# as disconnected sessions, while still allowing stale abandoned jobs to stop.
_SESSION_TTL_SECONDS = 300.0
_TERMINAL_JOB_STATUSES = {"done", "failed", "cancelled"}


def _evict_expired(now: float | None = None) -> bool:
    """Remove jobs older than _JOB_TTL_SECONDS from the in-memory store."""
    now = time.monotonic() if now is None else now
    expired = [
        jid
        for jid, ts in _job_created_at.items()
        if now - ts > _JOB_TTL_SECONDS
        and _jobs.get(jid) is not None
        and _jobs[jid].status in _TERMINAL_JOB_STATUSES
    ]
    for jid in expired:
        _jobs.pop(jid, None)
        _job_created_at.pop(jid, None)
        _cancelled_jobs.discard(jid)
        _job_session_id.pop(jid, None)
        _job_session_expires_at.pop(jid, None)
    return bool(expired)


def _load_jobs() -> None:
    if not _jobs_file.exists():
        return
    try:
        raw = json.loads(_jobs_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load jobs from %s: %s — starting with empty job store", _jobs_file, exc)
        return

    if not isinstance(raw, list):
        logger.error("jobs.json has unexpected format (expected list) — starting with empty job store")
        return

    loaded = 0
    for item in raw:
        try:
            job = JobStatus.model_validate(item)
        except Exception as exc:
            logger.warning("Skipping malformed job record: %s", exc)
            continue
        _jobs[job.job_id] = job
        _job_created_at[job.job_id] = time.monotonic()
        loaded += 1

    if loaded:
        logger.info("Loaded %d job(s) from %s", loaded, _jobs_file)


def _persist_jobs() -> None:
    global _last_persisted_at
    _jobs_file.parent.mkdir(parents=True, exist_ok=True)
    payload = [job.model_dump(mode="json") for job in _jobs.values()]
    _jobs_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    _last_persisted_at = time.monotonic()


def _should_persist_update(kwargs: dict, now: float) -> bool:
    status = kwargs.get("status")
    if status in {"done", "failed", "cancelled"}:
        return True
    if any(key in kwargs for key in {"clips", "error", "profile"}):
        return True
    return now - _last_persisted_at >= _PERSIST_INTERVAL_SECONDS


_load_jobs()


def create_job(job_id: str, profile: str = "general") -> JobStatus:
    with _lock:
        _evict_expired()
        job = JobStatus(job_id=job_id, status="queued", progress_pct=0, profile=profile)
        _jobs[job_id] = job
        _job_created_at[job_id] = time.monotonic()
        _cancelled_jobs.discard(job_id)
        _persist_jobs()
        return job


def get_job(job_id: str) -> JobStatus | None:
    with _lock:
        if _evict_expired():
            _persist_jobs()
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        now = time.monotonic()
        evicted = _evict_expired(now)
        job = _jobs.get(job_id)
        if job is None:
            logger.warning("Ignoring update for missing job %s", job_id)
            return
        previous_status = job.status
        for k, v in kwargs.items():
            setattr(job, k, v)
        if job.status in _TERMINAL_JOB_STATUSES and previous_status not in _TERMINAL_JOB_STATUSES:
            # Retain completed jobs for the full TTL after they reach a terminal state,
            # even if the job itself ran longer than _JOB_TTL_SECONDS.
            _job_created_at[job_id] = now
        elif job_id not in _job_created_at:
            _job_created_at[job_id] = now
        if evicted or _should_persist_update(kwargs, now):
            _persist_jobs()


def cancel_job(job_id: str) -> JobStatus | None:
    with _lock:
        if _evict_expired():
            _persist_jobs()
        job = _jobs.get(job_id)
        if job is None:
            return None
        if job.status in {"done", "failed", "cancelled"}:
            return job
        _cancelled_jobs.add(job_id)
        _job_session_id.pop(job_id, None)
        _job_session_expires_at.pop(job_id, None)
        job.status = "cancelled"
        job.error = "Job cancelled by client"
        _job_created_at[job_id] = time.monotonic()
        _persist_jobs()
        return job


def is_job_cancelled(job_id: str) -> bool:
    with _lock:
        _evict_expired()
        if job_id in _cancelled_jobs:
            return True

        session_id = _job_session_id.get(job_id)
        expires_at = _job_session_expires_at.get(job_id)
        if session_id and expires_at is not None and time.monotonic() > expires_at:
            job = _jobs.get(job_id)
            if job is not None and job.status not in {"done", "failed", "cancelled"}:
                _cancelled_jobs.add(job_id)
                _job_session_id.pop(job_id, None)
                _job_session_expires_at.pop(job_id, None)
                job.status = "cancelled"
                job.error = "Job cancelled because the client session ended"
                _job_created_at[job_id] = time.monotonic()
                _persist_jobs()
            return True
        return False


def bind_job_session(job_id: str, session_id: str | None) -> None:
    if not session_id:
        return
    with _lock:
        _evict_expired()
        if job_id not in _jobs:
            return
        _job_session_id[job_id] = session_id
        _job_session_expires_at[job_id] = time.monotonic() + _SESSION_TTL_SECONDS


def touch_job_session(job_id: str, session_id: str | None) -> JobStatus | None:
    with _lock:
        if _evict_expired():
            _persist_jobs()
        job = _jobs.get(job_id)
        if job is None:
            return None
        if job.status in {"done", "failed", "cancelled"}:
            return job
        if session_id and _job_session_id.get(job_id) == session_id:
            _job_session_expires_at[job_id] = time.monotonic() + _SESSION_TTL_SECONDS
        return job
