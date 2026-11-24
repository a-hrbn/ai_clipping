"""Evicting a job from the registry also deletes its storage folder (#117), so
the disk doesn't fill up. Self-managed workflows (debug, training, backfill)
keep their folders. Tested at the job registry API against a real temp storage
dir."""

from pathlib import Path

import pytest

from test_job_terminal_sticky import _fresh_registry


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reg = _fresh_registry(tmp_path)
    # Every terminal job is past its TTL immediately.
    monkeypatch.setattr(reg, "_JOB_TTL_SECONDS", -1)
    return reg


def _job_folder(storage_dir: Path, job_id: str) -> Path:
    folder = storage_dir / job_id
    (folder / "clips").mkdir(parents=True)
    (folder / "clips" / "clip_1.mp4").write_bytes(b"x")
    return folder


def test_eviction_deletes_serving_job_folder(registry, tmp_path: Path) -> None:
    registry.create_job("job-1")
    folder = _job_folder(tmp_path, "job-1")
    registry.update_job("job-1", status="done", progress_pct=100, clips=[])

    assert registry.get_job("job-1") is None
    assert not folder.exists()


@pytest.mark.parametrize(
    "job_id",
    ["pipeline_debug_wf1_123", "twitch_vod_123", "judge_backfill_abc"],
)
def test_eviction_keeps_self_managed_job_folders(registry, tmp_path: Path, job_id: str) -> None:
    registry.create_job(job_id)
    folder = _job_folder(tmp_path, job_id)
    registry.update_job(job_id, status="done", progress_pct=100, clips=[])

    assert registry.get_job(job_id) is None
    assert folder.exists()


def test_eviction_never_deletes_a_running_job(registry, tmp_path: Path) -> None:
    registry.create_job("job-running")
    folder = _job_folder(tmp_path, "job-running")
    registry.update_job("job-running", status="transcribing", progress_pct=30)

    registry.create_job("job-other")  # a registry write runs eviction

    assert registry.get_job("job-running").status == "transcribing"
    assert folder.exists()
