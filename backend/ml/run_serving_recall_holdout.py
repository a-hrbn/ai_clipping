from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from job_manager import create_job, get_job
from ml import evaluate_serving_recall_holdout
from ml.row_identity import read_jsonl, write_json

DEFAULT_PROFILE = "jynxzi"
DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_STORAGE_DIR = BACKEND_ROOT / "tmp" / "storage"
DEFAULT_HOLDOUT_POLICY = DEFAULT_ARTIFACT_DIR / "jynxzi_recent_vod_holdout_policy.jsonl"
DEFAULT_POSITIVE_ROWS = DEFAULT_ARTIFACT_DIR / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "jynxzi_serving_recall_holdout_eval.json"
DEFAULT_RUN_SUMMARY = DEFAULT_ARTIFACT_DIR / "jynxzi_serving_recall_holdout_run_summary.json"
MAX_WORKERS = 2
DEFAULT_PROGRESS_INTERVAL_SEC = 3.0
DEFAULT_HOLDOUT_JUDGE_SAMPLE_COUNT = 3
DEFAULT_ENABLE_JUDGE = False
DEFAULT_HOLDOUT_TWITCH_DOWNLOAD_CACHE_TTL_SEC = -1
PHASE_A_PIPELINE_ENV = {
    "ENABLE_DENSE_CANDIDATE_INVENTORY": "true",
    "DENSE_CANDIDATE_INVENTORY_SOURCE_CAP": "700",
}
DEFAULT_MOMENT_FINDER_PROVIDER = "gemini"
DEFAULT_MOMENT_FINDER_MODEL = "gemini-2.5-flash-lite"
DEFAULT_MOMENT_FINDER_MAX_MOMENTS = 180
DEFAULT_MOMENT_FINDER_MAX_TOKENS = 16000
DEFAULT_MOMENT_FINDER_MULTIMODAL_CONFIRM_TOP_N = 12

_PROGRESS_LOCK = threading.Lock()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all holdout VODs through the oracle-free serving pipeline and "
            "evaluate stage recall against cross-platform positive windows."
        )
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--holdout-policy-rows", type=pathlib.Path, default=DEFAULT_HOLDOUT_POLICY)
    parser.add_argument("--positive-rows", type=pathlib.Path, default=DEFAULT_POSITIVE_ROWS)
    parser.add_argument("--storage-dir", type=pathlib.Path, default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-summary-output", type=pathlib.Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop submitting new VOD jobs after the first failure. Enabled by default; use --no-fail-fast to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--positive-overlap-ratio", type=float, default=0.45)
    parser.add_argument("--progress-interval-sec", type=float, default=DEFAULT_PROGRESS_INTERVAL_SEC)
    parser.add_argument("--strategy-json", default="{}")
    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help=(
            "Run the non-deterministic Clip Judge during holdout evaluation. "
            "Default is judge-off so merge/final-selection experiments are deterministic."
        ),
    )
    parser.add_argument(
        "--judge-sample-count",
        type=int,
        default=DEFAULT_HOLDOUT_JUDGE_SAMPLE_COUNT,
        help="Judge samples per candidate when --enable-judge is used. Default: 3.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of VOD pipeline jobs to run in parallel. Capped at 2 for resource safety.",
    )
    parser.add_argument(
        "--enable-llm-moment-finder",
        action="store_true",
        help="Use the additive LLM Moment-Finder path for the serving holdout instead of the legacy ranker path.",
    )
    parser.add_argument("--moment-finder-provider", default=DEFAULT_MOMENT_FINDER_PROVIDER)
    parser.add_argument("--moment-finder-model", default=DEFAULT_MOMENT_FINDER_MODEL)
    parser.add_argument("--moment-finder-max-moments", type=int, default=DEFAULT_MOMENT_FINDER_MAX_MOMENTS)
    parser.add_argument("--moment-finder-max-tokens", type=int, default=DEFAULT_MOMENT_FINDER_MAX_TOKENS)
    parser.add_argument(
        "--moment-finder-confirm-top-n",
        type=int,
        default=DEFAULT_MOMENT_FINDER_MULTIMODAL_CONFIRM_TOP_N,
        help="Number of LLM-ranked moments to pass to multimodal confirmation when Judge is enabled.",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    holdout_vods = load_holdout_vods(args.holdout_policy_rows)
    if args.limit and args.limit > 0:
        holdout_vods = holdout_vods[: args.limit]

    workers = _bounded_workers(args.workers)
    strategy = _load_json_object(args.strategy_json, label="strategy-json")
    started_at = time.time()
    ordered_rows: dict[int, dict[str, Any]] = {}
    pending_jobs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    failed = False
    emit_progress(
        "holdout_run_started",
        profile=args.profile,
        run_id=run_id,
        total_vods=len(holdout_vods),
        workers=workers,
        fail_fast=bool(args.fail_fast),
        progress_interval_sec=float(args.progress_interval_sec),
        enable_clip_judge=bool(args.enable_judge),
        judge_sample_count=_bounded_judge_sample_count(args.judge_sample_count),
        phase_a_pipeline_env=phase_a_pipeline_env(),
        llm_moment_finder=llm_moment_finder_config(args),
        twitch_download_cache_ttl_sec=DEFAULT_HOLDOUT_TWITCH_DOWNLOAD_CACHE_TTL_SEC,
        holdout_policy_rows=str(args.holdout_policy_rows),
        positive_rows=str(args.positive_rows),
        dry_run=bool(args.dry_run),
    )

    for vod_index, vod in enumerate(holdout_vods, start=1):
        vod_id = str(vod.get("vod_id") or "").strip()
        url = str(vod.get("vod_url") or vod.get("source_url") or f"https://www.twitch.tv/videos/{vod_id}").strip()
        if not vod_id or not url:
            skipped["missing_vod_id_or_url"] += 1
            emit_progress(
                "holdout_vod_skipped",
                vod_index=vod_index,
                total_vods=len(holdout_vods),
                vod_id=vod_id,
                reason="missing_vod_id_or_url",
            )
            continue
        existing = find_existing_holdout_job(args.storage_dir, vod_id)
        if args.skip_existing and existing:
            emit_progress(
                "holdout_vod_reused",
                vod_index=vod_index,
                total_vods=len(holdout_vods),
                vod_id=vod_id,
                job_id=existing.name,
                debug_dir=str(existing / "debug"),
            )
            ordered_rows[vod_index] = {
                "vod_id": vod_id,
                "url": url,
                "job_id": existing.name,
                "status": "reused_existing",
                "debug_dir": str(existing / "debug"),
            }
            continue
        if args.dry_run:
            emit_progress(
                "holdout_vod_planned",
                vod_index=vod_index,
                total_vods=len(holdout_vods),
                vod_id=vod_id,
                url=url,
                job_id=planned_job_id(run_id, vod_id),
            )
            ordered_rows[vod_index] = {
                "vod_id": vod_id,
                "url": url,
                "job_id": planned_job_id(run_id, vod_id),
                "status": "dry_run",
            }
            continue

        job_id = planned_job_id(run_id, vod_id)
        pending_jobs.append(
            {
                "job_id": job_id,
                "url": url,
                "profile": args.profile,
                "strategy": strategy,
                "storage_dir": args.storage_dir,
                "positive_rows": args.positive_rows,
                "enable_clip_judge": bool(args.enable_judge),
                "judge_sample_count": _bounded_judge_sample_count(args.judge_sample_count),
                "llm_moment_finder": llm_moment_finder_config(args),
                "vod_index": vod_index,
                "total_vods": len(holdout_vods),
                "progress_interval_sec": max(float(args.progress_interval_sec or 0.0), 0.5),
            }
        )

    if pending_jobs:
        emit_progress(
            "holdout_pipeline_jobs_started",
            run_id=run_id,
            workers=workers,
            jobs=len(pending_jobs),
            fail_fast=bool(args.fail_fast),
            enable_clip_judge=bool(args.enable_judge),
            enable_llm_moment_finder=bool(args.enable_llm_moment_finder),
        )
        job_rows, job_failed = run_pending_pipeline_jobs(
            pending_jobs,
            workers=workers,
            fail_fast=bool(args.fail_fast),
        )
        ordered_rows.update(job_rows)
        failed = failed or job_failed

    run_rows = [ordered_rows[index] for index in sorted(ordered_rows)]

    eval_report: dict[str, Any] = {}
    if not args.dry_run:
        emit_progress(
            "holdout_eval_started",
            run_id=run_id,
            output=str(args.output),
            jobs_done=sum(1 for row in run_rows if row.get("status") == "done"),
            jobs_failed=sum(1 for row in run_rows if row.get("status") == "failed"),
        )
        vod_ids = {str(row.get("vod_id") or "").strip() for row in holdout_vods if str(row.get("vod_id") or "").strip()}
        eval_report = evaluate_serving_recall_holdout.evaluate_holdout(
            storage_dir=args.storage_dir,
            positive_rows_path=args.positive_rows,
            overlap_ratio=float(args.positive_overlap_ratio),
            vod_ids=vod_ids,
            latest_per_vod=True,
            require_known_oracle_free=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(eval_report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        emit_progress(
            "holdout_eval_done",
            run_id=run_id,
            output=str(args.output),
            jobs_evaluated=eval_report.get("jobs_evaluated"),
            positive_windows=(eval_report.get("aggregate") or {}).get("positive_windows"),
            jobs_skipped=eval_report.get("jobs_skipped"),
        )

    summary = {
        "event": "serving_recall_holdout_run_done",
        "profile": args.profile,
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "fail_fast": bool(args.fail_fast),
        "progress_interval_sec": float(args.progress_interval_sec),
        "enable_clip_judge": bool(args.enable_judge),
        "judge_sample_count": _bounded_judge_sample_count(args.judge_sample_count),
        "judge_policy": {
            "default": "judge_off",
            "reason": "deterministic holdout eval for merge/final-selection tuning",
            "reference_flag": "--enable-judge",
        },
        "phase_a_pipeline_env": phase_a_pipeline_env(),
        "llm_moment_finder": llm_moment_finder_config(args),
        "holdout_policy_rows": str(args.holdout_policy_rows),
        "positive_rows": str(args.positive_rows),
        "storage_dir": str(args.storage_dir),
        "output": str(args.output),
        "requested_vods": len(holdout_vods),
        "workers": workers,
        "jobs": run_rows,
        "jobs_by_status": dict(sorted(Counter(str(row.get("status") or "unknown") for row in run_rows).items())),
        "skipped": dict(sorted(skipped.items())),
        "elapsed_sec": round(time.time() - started_at, 3),
        "eval_aggregate": eval_report.get("aggregate") if isinstance(eval_report, dict) else {},
    }
    write_json(args.run_summary_output, summary)
    emit_progress(
        "holdout_run_done",
        run_id=run_id,
        status="failed" if failed else "done",
        requested_vods=len(holdout_vods),
        jobs_by_status=summary["jobs_by_status"],
        elapsed_sec=summary["elapsed_sec"],
        run_summary_output=str(args.run_summary_output),
        eval_output=str(args.output),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 1 if failed and args.fail_fast else 0


def run_pending_pipeline_jobs(
    pending_jobs: list[dict[str, Any]],
    *,
    workers: int,
    fail_fast: bool,
) -> tuple[dict[int, dict[str, Any]], bool]:
    worker_count = min(_bounded_workers(workers), len(pending_jobs))
    ordered_rows: dict[int, dict[str, Any]] = {}
    failed = False
    next_index = 0
    active: dict[Future[dict[str, Any]], dict[str, Any]] = {}

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal next_index
        if next_index >= len(pending_jobs):
            return
        job = pending_jobs[next_index]
        next_index += 1
        active[executor.submit(run_pipeline_job, **job)] = job

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="holdout-vod") as executor:
        for _ in range(worker_count):
            submit_next(executor)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                row = _pipeline_future_row(future, job)
                ordered_rows[int(job["vod_index"])] = row
                if row.get("status") != "done":
                    failed = True
                if not (failed and fail_fast):
                    submit_next(executor)
            if failed and fail_fast:
                emit_progress(
                    "holdout_fail_fast_waiting_for_inflight",
                    active_jobs=len(active),
                    completed_jobs=len(ordered_rows),
                )
    return ordered_rows, failed


def _pipeline_future_row(future: Future[dict[str, Any]], job: dict[str, Any]) -> dict[str, Any]:
    try:
        return future.result()
    except Exception as exc:  # pragma: no cover - defensive around subprocess orchestration
        return {
            "vod_id": _vod_id_from_url(str(job.get("url") or "")),
            "url": str(job.get("url") or ""),
            "job_id": str(job.get("job_id") or ""),
            "status": "failed",
            "returncode": None,
            "elapsed_sec": None,
            "debug_dir": str(pathlib.Path(job.get("storage_dir") or DEFAULT_STORAGE_DIR) / str(job.get("job_id") or "") / "debug"),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _bounded_workers(raw_workers: int) -> int:
    try:
        workers = int(raw_workers)
    except (TypeError, ValueError):
        workers = 1
    return max(1, min(workers, MAX_WORKERS))


def _bounded_judge_sample_count(raw_count: int) -> int:
    try:
        sample_count = int(raw_count)
    except (TypeError, ValueError):
        sample_count = DEFAULT_HOLDOUT_JUDGE_SAMPLE_COUNT
    return max(1, min(sample_count, 5))


def llm_moment_finder_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(args, "enable_llm_moment_finder", False)),
        "provider": str(getattr(args, "moment_finder_provider", DEFAULT_MOMENT_FINDER_PROVIDER) or DEFAULT_MOMENT_FINDER_PROVIDER),
        "model": str(getattr(args, "moment_finder_model", DEFAULT_MOMENT_FINDER_MODEL) or DEFAULT_MOMENT_FINDER_MODEL),
        "max_moments": max(int(getattr(args, "moment_finder_max_moments", DEFAULT_MOMENT_FINDER_MAX_MOMENTS) or 0), 1),
        "max_tokens": max(int(getattr(args, "moment_finder_max_tokens", DEFAULT_MOMENT_FINDER_MAX_TOKENS) or 0), 512),
        "multimodal_confirm_top_n": max(
            int(getattr(args, "moment_finder_confirm_top_n", DEFAULT_MOMENT_FINDER_MULTIMODAL_CONFIRM_TOP_N) or 0),
            0,
        ),
    }


def llm_moment_finder_config_from_mapping(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {
            "enabled": False,
            "provider": DEFAULT_MOMENT_FINDER_PROVIDER,
            "model": DEFAULT_MOMENT_FINDER_MODEL,
            "max_moments": DEFAULT_MOMENT_FINDER_MAX_MOMENTS,
            "max_tokens": DEFAULT_MOMENT_FINDER_MAX_TOKENS,
            "multimodal_confirm_top_n": DEFAULT_MOMENT_FINDER_MULTIMODAL_CONFIRM_TOP_N,
        }
    return {
        "enabled": bool(config.get("enabled")),
        "provider": str(config.get("provider") or DEFAULT_MOMENT_FINDER_PROVIDER),
        "model": str(config.get("model") or DEFAULT_MOMENT_FINDER_MODEL),
        "max_moments": max(int(config.get("max_moments") or DEFAULT_MOMENT_FINDER_MAX_MOMENTS), 1),
        "max_tokens": max(int(config.get("max_tokens") or DEFAULT_MOMENT_FINDER_MAX_TOKENS), 512),
        "multimodal_confirm_top_n": max(
            int(config.get("multimodal_confirm_top_n") or DEFAULT_MOMENT_FINDER_MULTIMODAL_CONFIRM_TOP_N),
            0,
        ),
    }


def load_holdout_vods(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        split = str(row.get("gold_holdout_split") or row.get("outcome_split") or row.get("dataset_role") or "").lower()
        source = str(row.get("dataset_source") or row.get("supervision_tier") or "").lower()
        if "holdout" not in split and "holdout" not in source:
            continue
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id:
            continue
        rows.append(dict(row))
    rows.sort(key=lambda row: (_holdout_rank(row), str(row.get("created_at") or ""), str(row.get("vod_id") or "")))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        vod_id = str(row.get("vod_id") or "").strip()
        if vod_id in seen:
            continue
        seen.add(vod_id)
        deduped.append(row)
    return deduped


def run_pipeline_job(
    *,
    job_id: str,
    url: str,
    profile: str,
    strategy: dict[str, Any],
    storage_dir: pathlib.Path,
    positive_rows: pathlib.Path,
    enable_clip_judge: bool,
    judge_sample_count: int,
    llm_moment_finder: dict[str, Any] | None,
    vod_index: int,
    total_vods: int,
    progress_interval_sec: float,
    runner_name: str = "serving_recall_holdout",
    event_prefix: str = "holdout",
) -> dict[str, Any]:
    storage_dir.mkdir(parents=True, exist_ok=True)
    job_dir = storage_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    create_job(job_id, profile=profile)
    request_payload = {
        "url": url,
        "profile": profile,
        "strategy": strategy,
        "runner": runner_name,
        "llm_moment_finder": llm_moment_finder_config_from_mapping(llm_moment_finder),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (job_dir / "request.json").write_text(json.dumps(request_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    cmd = [
        sys.executable,
        str(BACKEND_ROOT / "pipeline_worker.py"),
        "--job-id",
        job_id,
        "--url",
        url,
        "--profile",
        profile,
        "--strategy-json",
        json.dumps(strategy, ensure_ascii=True),
    ]
    started_at = time.time()
    emit_progress(
        f"{event_prefix}_vod_started",
        vod_index=vod_index,
        total_vods=total_vods,
        vod_id=_vod_id_from_url(url),
        url=url,
        job_id=job_id,
    )
    process = subprocess.Popen(
        cmd,
        cwd=str(BACKEND_ROOT),
        env=pipeline_env(
            positive_rows,
            enable_clip_judge=enable_clip_judge,
            judge_sample_count=judge_sample_count,
            llm_moment_finder=llm_moment_finder,
        ),
    )
    last_snapshot: tuple[str, int, str, bool, int | None] | None = None
    last_emit = 0.0
    while True:
        returncode = process.poll()
        job = get_job(job_id)
        snapshot = job_progress_snapshot(job)
        now = time.time()
        should_emit = snapshot != last_snapshot or now - last_emit >= progress_interval_sec
        if should_emit:
            emit_progress(
                f"{event_prefix}_vod_progress",
                vod_index=vod_index,
                total_vods=total_vods,
                vod_id=_vod_id_from_url(url),
                job_id=job_id,
                status=snapshot[0],
                progress_pct=snapshot[1],
                stage_label=snapshot[2],
                download_cached=snapshot[3],
                download_eta=snapshot[4],
                elapsed_sec=round(now - started_at, 3),
            )
            last_snapshot = snapshot
            last_emit = now
        if returncode is not None:
            break
        time.sleep(min(progress_interval_sec, 1.0))
    cleanup_source_media(job_dir)
    job = get_job(job_id)
    status = str(getattr(job, "status", "") or ("done" if process.returncode == 0 else "failed"))
    row = {
        "vod_id": _vod_id_from_url(url),
        "url": url,
        "job_id": job_id,
        "status": status,
        "returncode": process.returncode,
        "elapsed_sec": round(time.time() - started_at, 3),
        "debug_dir": str(job_dir / "debug"),
        "error": str(getattr(job, "error", "") or ""),
    }
    emit_progress(
        f"{event_prefix}_vod_done",
        vod_index=vod_index,
        total_vods=total_vods,
        vod_id=row["vod_id"],
        job_id=job_id,
        status=status,
        returncode=process.returncode,
        elapsed_sec=row["elapsed_sec"],
        clips=len(getattr(job, "clips", []) or []) if job is not None else 0,
        error=row["error"],
    )
    return row


def job_progress_snapshot(job) -> tuple[str, int, str, bool, int | None]:
    if job is None:
        return ("queued", 0, "queued", False, None)
    status = str(getattr(job, "status", "") or "queued")
    try:
        progress_pct = int(getattr(job, "progress_pct", 0) or 0)
    except (TypeError, ValueError):
        progress_pct = 0
    return (
        status,
        progress_pct,
        stage_label(status),
        bool(getattr(job, "download_cached", False)),
        getattr(job, "download_eta", None),
    )


def stage_label(status: str) -> str:
    labels = {
        "queued": "queued",
        "downloading": "download VOD",
        "downloading_chat": "download chat",
        "extracting_audio": "extract audio",
        "transcribing": "transcribe",
        "extracting_features": "extract features",
        "diarizing": "diarize",
        "analyzing": "candidate generation / ranker",
        "judging": "clip judge",
        "clipping": "clip previews/final clips",
        "done": "done",
        "cancelled": "cancelled",
        "failed": "failed",
    }
    return labels.get(status, status or "unknown")


def emit_progress(event: str, **payload: Any) -> None:
    with _PROGRESS_LOCK:
        print(
            json.dumps(
                {
                    "event": event,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    **payload,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            flush=True,
        )


def pipeline_env(
    positive_rows: pathlib.Path,
    *,
    enable_clip_judge: bool = DEFAULT_ENABLE_JUDGE,
    judge_sample_count: int = DEFAULT_HOLDOUT_JUDGE_SAMPLE_COUNT,
    llm_moment_finder: dict[str, Any] | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(BACKEND_ROOT), existing_pythonpath]) if existing_pythonpath else str(BACKEND_ROOT)
    env["SERVING_CROSS_PLATFORM_RECALL_ENABLED"] = "false"
    env["SERVING_CROSS_PLATFORM_FEATURE_BOOSTS_ENABLED"] = "false"
    env["SERVING_CROSS_PLATFORM_POSITIVE_ROWS_PATH"] = str(positive_rows)
    env["TWITCH_DOWNLOAD_CACHE_TTL_SEC"] = str(DEFAULT_HOLDOUT_TWITCH_DOWNLOAD_CACHE_TTL_SEC)
    env["ENABLE_CLIP_JUDGE"] = "true" if enable_clip_judge else "false"
    env.update(phase_a_pipeline_env())
    moment_config = llm_moment_finder_config_from_mapping(llm_moment_finder)
    env["ENABLE_LLM_MOMENT_FINDER"] = "true" if moment_config["enabled"] else "false"
    if moment_config["enabled"]:
        env["MOMENT_FINDER_PROVIDER"] = str(moment_config["provider"])
        env["MOMENT_FINDER_MODEL"] = str(moment_config["model"])
        env["MOMENT_FINDER_MAX_MOMENTS"] = str(moment_config["max_moments"])
        env["MOMENT_FINDER_MAX_TOKENS"] = str(moment_config["max_tokens"])
        env["MOMENT_FINDER_MULTIMODAL_CONFIRM_TOP_N"] = str(moment_config["multimodal_confirm_top_n"])
    if enable_clip_judge:
        sample_count = _bounded_judge_sample_count(judge_sample_count)
        env["CLIP_JUDGE_SAMPLE_COUNT"] = str(sample_count)
        env["CLIP_JUDGE_UNCERTAIN_SAMPLE_COUNT"] = str(sample_count)
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast")
    env.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    return env


def phase_a_pipeline_env() -> dict[str, str]:
    return dict(PHASE_A_PIPELINE_ENV)


def find_existing_holdout_job(storage_dir: pathlib.Path, vod_id: str) -> pathlib.Path | None:
    candidates: list[pathlib.Path] = []
    if not storage_dir.exists():
        return None
    for job_dir in storage_dir.iterdir():
        if not job_dir.is_dir() or not (job_dir / "debug" / "candidate_trace.jsonl").exists():
            continue
        request = _read_json(job_dir / "request.json")
        if _vod_id_from_url(str(request.get("url") or "")) != vod_id:
            continue
        funnel = _read_json(job_dir / "debug" / "candidate_funnel_summary.json")
        diagnostics = funnel.get("cross_platform_positive_diagnostics") if isinstance(funnel, dict) else {}
        if isinstance(diagnostics, dict) and diagnostics.get("enabled") is True:
            continue
        candidates.append(job_dir)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def cleanup_source_media(job_dir: pathlib.Path) -> None:
    for path in (job_dir / "video.mp4", job_dir / "audio.wav"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    chunks_dir = job_dir / "chunks"
    if chunks_dir.exists():
        import shutil

        shutil.rmtree(chunks_dir, ignore_errors=True)


def planned_job_id(run_id: str, vod_id: str) -> str:
    safe_run = "".join(char if char.isalnum() or char in "._-" else "_" for char in run_id).strip("_") or "run"
    safe_vod = "".join(char for char in str(vod_id) if char.isdigit()) or str(vod_id)
    return f"holdout_eval_{safe_run}_{safe_vod}"


def _load_json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid {label}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"Invalid {label}: expected a JSON object")
    return parsed


def _holdout_rank(row: dict[str, Any]) -> int:
    try:
        return int(row.get("holdout_rank") or 999999)
    except (TypeError, ValueError):
        return 999999


def _vod_id_from_url(url: str) -> str:
    marker = "twitch.tv/videos/"
    if marker not in url:
        return ""
    return url.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
