import json
import logging
import os
import pathlib
import threading
import time
from typing import Dict

import session_lease
from config import settings
from job_lifecycle import is_terminal
from models import JobStatus

logger = logging.getLogger(__name__)

_jobs: Dict[str, JobStatus] = {}
_job_created_at: Dict[str, float] = {}   # job_id → creation timestamp (epoch seconds)
_cancelled_jobs: set[str] = set()
# Session-Lease-State wohnt jetzt in session_lease (single owner). Hier nur per
# Referenz re-exportiert, damit Aufrufer/Tests `job_manager._job_session_id` /
# `_job_session_expires_at` unverändert sehen. Alias — niemals neu binden, nur
# mutieren; alle Lease-Mutationen laufen über die session_lease-Funktionen.
_job_session_id = session_lease._job_session_id
_job_session_expires_at = session_lease._job_session_expires_at
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


def _evict_expired(now: float | None = None) -> bool:
    """Remove jobs older than _JOB_TTL_SECONDS from the in-memory store."""
    now = time.monotonic() if now is None else now
    expired = [
        jid
        for jid, ts in _job_created_at.items()
        if now - ts > _JOB_TTL_SECONDS
        and _jobs.get(jid) is not None
        and is_terminal(_jobs[jid].status)
    ]
    for jid in expired:
        _jobs.pop(jid, None)
        _job_created_at.pop(jid, None)
        _cancelled_jobs.discard(jid)
        session_lease.clear(jid)
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
            session_lease.clear(job_id)

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
        # Per-process tmp name: jobs.json is written by several worker processes
        # and _write_lock only serializes threads. A shared tmp path let one
        # process os.replace() the other's tmp away mid-write (FileNotFoundError).
        tmp_path = _jobs_file.parent / f"{_jobs_file.name}.{os.getpid()}.tmp"
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(tmp_path, _jobs_file)
        _last_persisted_at = time.monotonic()
        try:
            _jobs_file_mtime_ns = _jobs_file.stat().st_mtime_ns
        except OSError:
            _jobs_file_mtime_ns = 0


def _should_persist_update(kwargs: dict, now: float) -> bool:
    status = kwargs.get("status")
    if is_terminal(status):
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
    progress_snapshot: tuple[str, int, str | None] | None = None
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
        if is_terminal(job.status) and not is_terminal(previous_status):
            # Retain completed jobs for the full TTL after they reach a terminal state,
            # even if the job itself ran longer than _JOB_TTL_SECONDS.
            _job_created_at[job_id] = now
        elif job_id not in _job_created_at:
            _job_created_at[job_id] = now
        progress_snapshot = (job.status, int(job.progress_pct), job.error)
        if evicted or _should_persist_update(kwargs, now):
            should_persist = True
    # Side-channel progress file (single writer = this job's worker process).
    if progress_snapshot is not None:
        _write_job_progress_file(job_id, *progress_snapshot)
    if should_persist:
        _persist_jobs()


def _write_job_progress_file(job_id: str, status: str, progress_pct: int, error: str | None) -> None:
    """Write a per-job progress side-file for reliable cross-process status reads.

    jobs.json is a shared registry mutated by several processes with mtime-based
    reload that is unreliable on bind-mounted filesystems, so the detached debug
    orchestrator cannot observe a worker's live phase through it. This file has a
    single writer — the process running the job's pipeline — so a reader always
    sees the worker's true status. Best-effort; never raises, never creates dirs.
    """
    try:
        job_dir = pathlib.Path(settings.storage_dir) / job_id
        if not job_dir.is_dir():
            return  # only write once the job owns a workspace
        tmp = job_dir / "progress.json.tmp"
        tmp.write_text(
            json.dumps({"status": status, "progress_pct": int(progress_pct), "error": error}),
            encoding="utf-8",
        )
        os.replace(tmp, job_dir / "progress.json")
    except OSError:
        pass


def cancel_job(job_id: str, reason: str = "Job cancelled by client") -> JobStatus | None:
    """Cancel a non-terminal job. `reason` becomes the job's error message so that
    manual cancellation and session-expiry cancellation share this one path yet stay
    distinguishable by their message."""
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
        elif is_terminal(job.status):
            if should_persist:
                pass
            else:
                return job
        else:
            _cancelled_jobs.add(job_id)
            session_lease.clear(job_id)
            job.status = "cancelled"
            job.error = reason
            _job_created_at[job_id] = time.monotonic()
            should_persist = True
    if should_persist:
        _persist_jobs()
    return job


def is_job_cancelled(job_id: str) -> bool:
    """Pure query: True if the job is in the cancel-set or already carries the
    terminal "cancelled" status. Mutates nothing and never persists — a cheap,
    lock-friendly call for the Conductor hot path between producers.

    Lease expiry is no longer driven here; the explicit drive lives in the request
    path (`session_lease.drive_expiry` → `cancel_job`). The reload stays so a worker
    subprocess observes a cancellation the main process wrote to jobs.json."""
    with _lock:
        _reload_jobs_from_disk_if_changed()
        if job_id in _cancelled_jobs:
            return True
        job = _jobs.get(job_id)
        return job is not None and job.status == "cancelled"


def bind_job_session(job_id: str, session_id: str | None) -> None:
    if not session_id:
        return
    with _lock:
        _reload_jobs_from_disk_if_changed()
        _evict_expired()
        if job_id not in _jobs:
            return
        session_lease.bind(job_id, session_id)


def delete_job(job_id: str) -> bool:
    """Remove a terminal job from the in-memory store and persist.

    Returns True if the job was found and deleted, False if not found or still
    running (active jobs cannot be deleted — cancel first).
    """
    with _lock:
        _reload_jobs_from_disk_if_changed()
        job = _jobs.get(job_id)
        if job is None:
            return False
        if not is_terminal(job.status):
            return False
        _jobs.pop(job_id, None)
        _job_created_at.pop(job_id, None)
        _cancelled_jobs.discard(job_id)
        session_lease.clear(job_id)
    _persist_jobs()
    return True


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
        elif is_terminal(job.status):
            pass
        else:
            session_lease.touch(job_id, session_id)
    if should_persist:
        _persist_jobs()
    return job
