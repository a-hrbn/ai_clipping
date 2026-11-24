import fcntl
import json
import logging
import os
import pathlib
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator

import session_lease
from config import settings
from job_lifecycle import is_terminal
from models import JobStatus
from services.job_artifact_store import RealFsJobArtifactStore

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
# Cross-process registry lock (#116). Container-local path, NOT the bind mount —
# flock on drvfs/9p is unreliable (same reason as the transcription lock).
_REGISTRY_LOCK_PATH = pathlib.Path(tempfile.gettempdir()) / "ai_clipping_jobs.lock"
_last_persisted_at = 0.0
_jobs_file_mtime_ns = 0

_JOB_TTL_SECONDS = 2 * 3600  # evict in-memory records after 2 hours
_PERSIST_INTERVAL_SECONDS = 5.0

# update_job() kwargs applied in memory but not yet on disk (progress ticks are
# persisted at most every _PERSIST_INTERVAL_SECONDS). Replayed after every reload
# so neither a reload nor another process's write loses them — and replayed
# through the terminal-sticky rule, so a cancel written by another process wins.
_pending_updates: Dict[str, dict] = {}


def _expired_job_ids(now: float | None = None) -> list[str]:
    now = time.monotonic() if now is None else now
    return [
        jid
        for jid, ts in _job_created_at.items()
        if now - ts > _JOB_TTL_SECONDS
        and _jobs.get(jid) is not None
        and is_terminal(_jobs[jid].status)
    ]


# Debug, training and backfill workflows manage their own job folders (and
# source media); eviction and source-media cleanup leave them alone.
SELF_MANAGED_JOB_PREFIXES = ("pipeline_debug_", "twitch_vod", "judge_backfill_")


def is_self_managed_job(job_id: str) -> bool:
    return str(job_id or "").startswith(SELF_MANAGED_JOB_PREFIXES)


def _evict_expired(now: float | None = None) -> list[str]:
    """Remove terminal jobs older than _JOB_TTL_SECONDS from the in-memory store
    and return their ids. Runs inside every `_registry_write`; read paths only
    check `_expired_job_ids`."""
    expired = _expired_job_ids(now)
    for jid in expired:
        _jobs.pop(jid, None)
        _job_created_at.pop(jid, None)
        _cancelled_jobs.discard(jid)
        _pending_updates.pop(jid, None)
        session_lease.clear(jid)
    return expired


def _delete_job_folders(job_ids: list[str]) -> None:
    """Delete the storage folders of evicted serving jobs (#117)."""
    store = RealFsJobArtifactStore(_jobs_file.parent)
    for jid in job_ids:
        if not is_self_managed_job(jid):
            store.delete_job(jid)


def _apply_update(job_id: str, kwargs: dict, now: float) -> JobStatus | None:
    """Apply update_job kwargs in memory. None if the job is missing or terminal.

    Terminal is sticky: the run is over (done/failed/cancelled), so late progress
    ticks or a late `failed` must not resurrect or relabel it. Only create_job
    starts a fresh run."""
    job = _jobs.get(job_id)
    if job is None or is_terminal(job.status):
        return None
    for k, v in kwargs.items():
        setattr(job, k, v)
    if is_terminal(job.status):
        # Retain completed jobs for the full TTL after they reach a terminal state,
        # even if the job itself ran longer than _JOB_TTL_SECONDS.
        _job_created_at[job_id] = now
    elif job_id not in _job_created_at:
        _job_created_at[job_id] = now
    return job


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

    now = time.monotonic()
    for job_id, kwargs in list(_pending_updates.items()):
        if _apply_update(job_id, kwargs, now) is None:
            _pending_updates.pop(job_id, None)  # gone or terminal on disk: disk wins

    try:
        _jobs_file_mtime_ns = _jobs_file.stat().st_mtime_ns
    except OSError:
        _jobs_file_mtime_ns = 0

    if loaded:
        logger.debug("Loaded %d job(s) from %s", loaded, _jobs_file)


def _reload_jobs_from_disk_if_changed() -> None:
    """Cheap read-path refresh (mtime check, no lock wait). Writes don't rely on
    it: `_registry_write` always reloads."""
    try:
        mtime_ns = _jobs_file.stat().st_mtime_ns
    except OSError:
        return
    if mtime_ns <= _jobs_file_mtime_ns:
        return
    _load_jobs()


def _persist_jobs_locked() -> None:
    """Write the in-memory store to jobs.json. Caller holds `_registry_write`.

    Snapshot under `_lock`, I/O outside it so readers aren't blocked on fsync.
    Atomic via tempfile + os.replace so a crash mid-write can't truncate
    jobs.json and wipe the store on next boot.
    """
    global _last_persisted_at, _jobs_file_mtime_ns
    _jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        payload = [job.model_dump(mode="json") for job in _jobs.values()]
        _pending_updates.clear()  # all in this snapshot; later ticks stay pending
    # Per-process tmp name, belt and braces next to the flock.
    tmp_path = _jobs_file.parent / f"{_jobs_file.name}.{os.getpid()}.tmp"
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp_path, _jobs_file)
    _last_persisted_at = time.monotonic()
    try:
        _jobs_file_mtime_ns = _jobs_file.stat().st_mtime_ns
    except OSError:
        _jobs_file_mtime_ns = 0


@contextmanager
def _registry_write() -> Iterator[None]:
    """Run a registry mutation as reload → evict → change → write under one lock
    that spans processes (flock) and threads (`_write_lock`). Always reloads —
    mtime checks miss writes on coarse-mtime filesystems. Every mutation and the
    eviction persist go through here. Lock order: `_write_lock` → flock → `_lock`;
    never enter this while holding `_lock`."""
    with _write_lock, open(_REGISTRY_LOCK_PATH, "a") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            with _lock:
                _load_jobs()
                evicted = _evict_expired()
            # Still under the flock, so no other process can recreate the job
            # in between; outside `_lock` so readers don't wait on rmtree.
            _delete_job_folders(evicted)
            yield
            _persist_jobs_locked()
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _persist_jobs() -> None:
    """Flush pending updates and due evictions (reload → merge → write, locked)."""
    with _registry_write():
        pass


def _should_persist_update(kwargs: dict, now: float) -> bool:
    status = kwargs.get("status")
    if is_terminal(status):
        return True
    if any(key in kwargs for key in {"clips", "error", "profile"}):
        return True
    return now - _last_persisted_at >= _PERSIST_INTERVAL_SECONDS


_load_jobs()


def create_job(job_id: str, profile: str = "general") -> JobStatus:
    with _registry_write(), _lock:
        job = JobStatus(job_id=job_id, status="queued", progress_pct=0, profile=profile)
        _jobs[job_id] = job
        _job_created_at[job_id] = time.monotonic()
        _cancelled_jobs.discard(job_id)
        _pending_updates.pop(job_id, None)
    return job


def get_job(job_id: str) -> JobStatus | None:
    with _lock:
        _reload_jobs_from_disk_if_changed()
        eviction_due = bool(_expired_job_ids())
    if eviction_due:
        _persist_jobs()
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        _reload_jobs_from_disk_if_changed()
        now = time.monotonic()
        job = _apply_update(job_id, kwargs, now)
        if job is None:
            logger.debug("Ignoring update for missing or terminal job %s", job_id)
            return
        _pending_updates.setdefault(job_id, {}).update(kwargs)
        progress_snapshot = (job.status, int(job.progress_pct), job.error)
        should_persist = _should_persist_update(kwargs, now)
    # Side-channel progress file (single writer = this job's worker process).
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
    with _registry_write(), _lock:
        job = _jobs.get(job_id)
        if job is None or is_terminal(job.status):
            return job
        _cancelled_jobs.add(job_id)
        session_lease.clear(job_id)
        _pending_updates.pop(job_id, None)
        job.status = "cancelled"
        job.error = reason
        _job_created_at[job_id] = time.monotonic()
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
        if job_id not in _jobs:
            return
        session_lease.bind(job_id, session_id)


def delete_job(job_id: str) -> bool:
    """Remove a terminal job from the registry and persist.

    Returns True if the job was found and deleted, False if not found or still
    running (active jobs cannot be deleted — cancel first).
    """
    with _registry_write(), _lock:
        job = _jobs.get(job_id)
        if job is None or not is_terminal(job.status):
            return False
        _jobs.pop(job_id, None)
        _job_created_at.pop(job_id, None)
        _cancelled_jobs.discard(job_id)
        _pending_updates.pop(job_id, None)
        session_lease.clear(job_id)
        return True


def touch_job_session(job_id: str, session_id: str | None) -> JobStatus | None:
    with _lock:
        _reload_jobs_from_disk_if_changed()
        eviction_due = bool(_expired_job_ids())
    if eviction_due:
        _persist_jobs()
    with _lock:
        job = _jobs.get(job_id)
        if job is not None and not is_terminal(job.status):
            session_lease.touch(job_id, session_id)
        return job
