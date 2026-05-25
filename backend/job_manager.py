import json
import logging
import os
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
# Dedicated write lock so disk I/O is serialized but does NOT hold the job-state
# `_lock`. Pipeline progress ticks call update_job() frequently; blocking
# in-memory reads on every fsync is needless back-pressure.
_write_lock = threading.Lock()
_jobs_file = pathlib.Path(settings.storage_dir) / "jobs.json"
_last_persisted_at = 0.0
_jobs_file_mtime_ns = 0

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
    global _jobs_file_mtime_ns
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
    seen_job_ids: set[str] = set()
    for item in raw:
        try:
            job = JobStatus.model_validate(item)
        except Exception as exc:
            logger.warning("Skipping malformed job record: %s", exc)
            continue
        _jobs[job.job_id] = job
        _job_created_at.setdefault(job.job_id, time.monotonic())
        seen_job_ids.add(job.job_id)
        loaded += 1

    for job_id in list(_jobs):
        if job_id not in seen_job_ids:
            _jobs.pop(job_id, None)
            _job_created_at.pop(job_id, None)
            _cancelled_jobs.discard(job_id)
            _job_session_id.pop(job_id, None)
            _job_session_expires_at.pop(job_id, None)

    try:
        _jobs_file_mtime_ns = _jobs_file.stat().st_mtime_ns
    except OSError:
        _jobs_file_mtime_ns = 0

    if loaded:
        logger.info("Loaded %d job(s) from %s", loaded, _jobs_file)


def _reload_jobs_from_disk_if_changed() -> None:
    try:
        mtime_ns = _jobs_file.stat().st_mtime_ns
    except OSError:
        return
    if mtime_ns <= _jobs_file_mtime_ns:
        return
    _load_jobs()


def _persist_jobs() -> None:
    """Flush current job state to disk without holding the state lock during I/O.

    Takes a quick snapshot under `_lock`, then writes under `_write_lock`. Atomic
    via tempfile + os.replace so a crash mid-write can't truncate jobs.json and
    wipe the store on next boot.
    """
    global _last_persisted_at, _jobs_file_mtime_ns
    _jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with _lock:
            payload = [job.model_dump(mode="json") for job in _jobs.values()]
        tmp_path = _jobs_file.with_suffix(_jobs_file.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(tmp_path, _jobs_file)
        _last_persisted_at = time.monotonic()
        try:
            _jobs_file_mtime_ns = _jobs_file.stat().st_mtime_ns
        except OSError:
            _jobs_file_mtime_ns = 0


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
        _reload_jobs_from_disk_if_changed()
        evicted = _evict_expired()
        job = _jobs.get(job_id)
    if evicted:
        _persist_jobs()
    return job


def update_job(job_id: str, **kwargs) -> None:
    should_persist = False
    with _lock:
        _reload_jobs_from_disk_if_changed()
        now = time.monotonic()
        evicted = _evict_expired(now)
        job = _jobs.get(job_id)
        if job is None:
            logger.debug("Ignoring update for missing job %s", job_id)
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
            should_persist = True
    if should_persist:
        _persist_jobs()


def cancel_job(job_id: str) -> JobStatus | None:
    should_persist = False
    with _lock:
        _reload_jobs_from_disk_if_changed()
        if _evict_expired():
            should_persist = True
        job = _jobs.get(job_id)
        if job is None:
            if should_persist:
                pass  # fall through to flush the eviction
            else:
                return None
        elif job.status in {"done", "failed", "cancelled"}:
            if should_persist:
                pass
            else:
                return job
        else:
            _cancelled_jobs.add(job_id)
            _job_session_id.pop(job_id, None)
            _job_session_expires_at.pop(job_id, None)
            job.status = "cancelled"
            job.error = "Job cancelled by client"
            _job_created_at[job_id] = time.monotonic()
            should_persist = True
    if should_persist:
        _persist_jobs()
    return job


def is_job_cancelled(job_id: str) -> bool:
    should_persist = False
    with _lock:
        _reload_jobs_from_disk_if_changed()
        _evict_expired()
        if job_id in _cancelled_jobs:
            return True

        cancelled_here = False
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
                should_persist = True
            cancelled_here = True
    if should_persist:
        _persist_jobs()
    return cancelled_here


def bind_job_session(job_id: str, session_id: str | None) -> None:
    if not session_id:
        return
    with _lock:
        _reload_jobs_from_disk_if_changed()
        _evict_expired()
        if job_id not in _jobs:
            return
        _job_session_id[job_id] = session_id
        _job_session_expires_at[job_id] = time.monotonic() + _SESSION_TTL_SECONDS


def touch_job_session(job_id: str, session_id: str | None) -> JobStatus | None:
    should_persist = False
    with _lock:
        _reload_jobs_from_disk_if_changed()
        if _evict_expired():
            should_persist = True
        job = _jobs.get(job_id)
        if job is None:
            if should_persist:
                pass
            else:
                return None
        elif job.status in {"done", "failed", "cancelled"}:
            pass
        elif session_id and _job_session_id.get(job_id) == session_id:
            _job_session_expires_at[job_id] = time.monotonic() + _SESSION_TTL_SECONDS
    if should_persist:
        _persist_jobs()
    return job
