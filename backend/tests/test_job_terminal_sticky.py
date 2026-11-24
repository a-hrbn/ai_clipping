"""Terminal status is sticky (#115): once a job is done/failed/cancelled, later
status updates within the same run must not resurrect it. Tested at the job
registry API against a real temp storage dir; a second, freshly imported module
instance stands in for the worker process."""

import importlib.util
import json
import uuid
from pathlib import Path

import pytest

import job_manager as _jm_module

_JM_PATH = Path(_jm_module.__file__)


def _fresh_registry(storage_dir: Path):
    """Load an independent job_manager instance bound to `storage_dir`
    (its own in-memory store, like a separate process)."""
    from config import settings

    orig = settings.storage_dir
    settings.storage_dir = str(storage_dir)
    try:
        spec = importlib.util.spec_from_file_location(f"job_manager_{uuid.uuid4().hex}", _JM_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        settings.storage_dir = orig
    return module


def _status_on_disk(storage_dir: Path, job_id: str) -> dict:
    jobs = json.loads((storage_dir / "jobs.json").read_text(encoding="utf-8"))
    return next(j for j in jobs if j["job_id"] == job_id)


def test_cancelled_job_stays_cancelled_after_progress_updates(tmp_path: Path) -> None:
    reg = _fresh_registry(tmp_path)
    reg.create_job("job-1")
    reg.update_job("job-1", status="transcribing", progress_pct=30)
    reg.cancel_job("job-1", reason="Job cancelled by client")

    reg.update_job("job-1", status="transcribing", progress_pct=40)
    reg.update_job("job-1", status="clipping", clips=[])

    assert reg.get_job("job-1").status == "cancelled"
    assert reg.is_job_cancelled("job-1")
    assert _status_on_disk(tmp_path, "job-1")["status"] == "cancelled"


def test_late_failure_does_not_relabel_cancelled_job(tmp_path: Path) -> None:
    reg = _fresh_registry(tmp_path)
    reg.create_job("job-1")
    reg.update_job("job-1", status="analyzing", progress_pct=50)
    reg.cancel_job("job-1", reason="Session expired")

    reg.update_job("job-1", status="failed", error="Worker crashed")

    job = reg.get_job("job-1")
    assert job.status == "cancelled"
    assert job.error == "Session expired"
    on_disk = _status_on_disk(tmp_path, "job-1")
    assert (on_disk["status"], on_disk["error"]) == ("cancelled", "Session expired")


def test_done_job_does_not_fall_back_to_running(tmp_path: Path) -> None:
    reg = _fresh_registry(tmp_path)
    reg.create_job("job-1")
    reg.update_job("job-1", status="done", progress_pct=100, clips=[])

    reg.update_job("job-1", status="clipping", progress_pct=90)

    assert reg.get_job("job-1").status == "done"


def test_create_job_resets_a_terminal_job(tmp_path: Path) -> None:
    reg = _fresh_registry(tmp_path)
    reg.create_job("job-1")
    reg.cancel_job("job-1")

    reg.create_job("job-1")
    reg.update_job("job-1", status="downloading", progress_pct=5)

    assert reg.get_job("job-1").status == "downloading"
    assert not reg.is_job_cancelled("job-1")


def test_worker_sees_cancel_from_server_even_after_reporting_progress(tmp_path: Path) -> None:
    server = _fresh_registry(tmp_path)
    server.create_job("job-1")
    worker = _fresh_registry(tmp_path)  # loads the job from disk
    worker.update_job("job-1", status="transcribing", progress_pct=20)

    server.cancel_job("job-1", reason="Job cancelled by client")
    worker.update_job("job-1", status="transcribing", progress_pct=40)
    worker.update_job("job-1", status="clipping", clips=[])

    assert worker.is_job_cancelled("job-1")
    assert server.get_job("job-1").status == "cancelled"
    assert _status_on_disk(tmp_path, "job-1")["status"] == "cancelled"
