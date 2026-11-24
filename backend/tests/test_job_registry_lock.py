"""Job registry is safe across processes (#116): every mutation runs
reload → change → write under one cross-process lock. Tested at the registry
API against a real temp storage dir; two freshly imported module instances
stand in for two processes (e.g. a serving job and a pipeline-debug workflow)."""

import fcntl
import json
import multiprocessing
import os
import threading
from pathlib import Path

from models import ClipResult
from tests.test_job_terminal_sticky import _fresh_registry

_CLIP = ClipResult(
    candidate_id="c1", filename="clip_1.mp4", download_url="/clips/c1",
    start_time=10.0, end_time=40.0, viral_score=0.9, reason="hype",
)


def _jobs_on_disk(storage_dir: Path) -> dict[str, dict]:
    jobs = json.loads((storage_dir / "jobs.json").read_text(encoding="utf-8"))
    return {j["job_id"]: j for j in jobs}


def test_create_job_in_one_instance_keeps_jobs_created_by_another(tmp_path: Path) -> None:
    serving = _fresh_registry(tmp_path)
    debug = _fresh_registry(tmp_path)

    serving.create_job("serving-1")
    debug.create_job("pipeline_debug_1")

    assert set(_jobs_on_disk(tmp_path)) == {"serving-1", "pipeline_debug_1"}
    assert serving.get_job("serving-1") is not None
    assert debug.get_job("serving-1") is not None


def test_create_job_does_not_reset_a_job_another_instance_finished(tmp_path: Path) -> None:
    serving = _fresh_registry(tmp_path)
    serving.create_job("pipeline_debug_vod1")
    debug = _fresh_registry(tmp_path)  # sees vod1 as queued
    serving.update_job("pipeline_debug_vod1", status="done", progress_pct=100, clips=[_CLIP])

    debug.create_job("pipeline_debug_vod2")

    vod1 = _jobs_on_disk(tmp_path)["pipeline_debug_vod1"]
    assert vod1["status"] == "done"
    assert [c["candidate_id"] for c in vod1["clips"]] == ["c1"]
    assert debug.get_job("pipeline_debug_vod1").status == "done"


def test_concurrent_progress_updates_of_two_instances_both_survive(tmp_path: Path) -> None:
    a = _fresh_registry(tmp_path)
    a.create_job("job-a")
    a.create_job("job-b")
    b = _fresh_registry(tmp_path)

    a.update_job("job-a", status="analyzing", progress_pct=60)  # within 5 s: memory only
    b.update_job("job-b", status="transcribing", progress_pct=30)  # b's first tick hits disk
    a.update_job("job-a", clips=[_CLIP])  # forces a's flush

    on_disk = _jobs_on_disk(tmp_path)
    assert (on_disk["job-a"]["status"], on_disk["job-a"]["progress_pct"]) == ("analyzing", 60)
    assert (on_disk["job-b"]["status"], on_disk["job-b"]["progress_pct"]) == ("transcribing", 30)
    assert a.get_job("job-a").progress_pct == 60


def test_worker_flush_keeps_server_cancel_even_when_mtime_did_not_change(tmp_path: Path) -> None:
    # Coarse-mtime filesystems (bind mounts) can give the server's cancel write the
    # same mtime the worker last saw, so an mtime-based reload misses it.
    server = _fresh_registry(tmp_path)
    server.create_job("job-1")
    worker = _fresh_registry(tmp_path)
    worker.update_job("job-1", status="transcribing", progress_pct=20)
    jobs_file = tmp_path / "jobs.json"
    seen_mtime = jobs_file.stat().st_mtime_ns

    server.cancel_job("job-1", reason="Job cancelled by client")
    os.utime(jobs_file, ns=(seen_mtime, seen_mtime))
    worker.update_job("job-1", status="clipping", progress_pct=90, clips=[_CLIP])

    on_disk = _jobs_on_disk(tmp_path)["job-1"]
    assert (on_disk["status"], on_disk["error"]) == ("cancelled", "Job cancelled by client")
    assert worker.is_job_cancelled("job-1")


def _create_jobs_in_own_process(storage_dir: str, prefix: str, count: int) -> None:
    reg = _fresh_registry(Path(storage_dir))
    for i in range(count):
        reg.create_job(f"{prefix}-{i}")


def test_parallel_processes_creating_jobs_lose_none(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=_create_jobs_in_own_process, args=(str(tmp_path), f"p{n}", 25))
        for n in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    assert len(_jobs_on_disk(tmp_path)) == 100


def test_read_paths_do_not_wait_for_the_registry_lock(tmp_path: Path) -> None:
    reg = _fresh_registry(tmp_path)
    reg.create_job("job-1")
    reg.cancel_job("job-1")
    results: dict[str, object] = {}

    def read() -> None:
        results["job"] = reg.get_job("job-1")
        results["cancelled"] = reg.is_job_cancelled("job-1")

    with open(reg._REGISTRY_LOCK_PATH, "a") as held:  # another process mid-write
        fcntl.flock(held, fcntl.LOCK_EX)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(timeout=2)
        fcntl.flock(held, fcntl.LOCK_UN)

    assert not reader.is_alive(), "read path blocked on the registry lock"
    assert results["job"].status == "cancelled" and results["cancelled"] is True
