from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter
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
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--positive-overlap-ratio", type=float, default=0.45)
    parser.add_argument("--progress-interval-sec", type=float, default=10.0)
    parser.add_argument("--strategy-json", default="{}")
    args = parser.parse_args(argv)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    holdout_vods = load_holdout_vods(args.holdout_policy_rows)
    if args.limit and args.limit > 0:
        holdout_vods = holdout_vods[: args.limit]

    strategy = _load_json_object(args.strategy_json, label="strategy-json")
    started_at = time.time()
    run_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    failed = False
    emit_progress(
        "holdout_run_started",
        profile=args.profile,
        run_id=run_id,
        total_vods=len(holdout_vods),
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
            run_rows.append(
                {
                    "vod_id": vod_id,
                    "url": url,
                    "job_id": existing.name,
                    "status": "reused_existing",
                    "debug_dir": str(existing / "debug"),
                }
            )
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
            run_rows.append({"vod_id": vod_id, "url": url, "job_id": planned_job_id(run_id, vod_id), "status": "dry_run"})
            continue

        job_id = planned_job_id(run_id, vod_id)
        row = run_pipeline_job(
            job_id=job_id,
            url=url,
            profile=args.profile,
            strategy=strategy,
            storage_dir=args.storage_dir,
            positive_rows=args.positive_rows,
            vod_index=vod_index,
            total_vods=len(holdout_vods),
            progress_interval_sec=max(float(args.progress_interval_sec or 0.0), 0.5),
        )
        run_rows.append(row)
        if row.get("status") != "done":
            failed = True
            if args.fail_fast:
                break

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
        "holdout_policy_rows": str(args.holdout_policy_rows),
        "positive_rows": str(args.positive_rows),
        "storage_dir": str(args.storage_dir),
        "output": str(args.output),
        "requested_vods": len(holdout_vods),
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
    vod_index: int,
    total_vods: int,
    progress_interval_sec: float,
) -> dict[str, Any]:
    storage_dir.mkdir(parents=True, exist_ok=True)
    job_dir = storage_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    create_job(job_id, profile=profile)
    request_payload = {
        "url": url,
        "profile": profile,
        "strategy": strategy,
        "runner": "serving_recall_holdout",
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
        "holdout_vod_started",
        vod_index=vod_index,
        total_vods=total_vods,
        vod_id=_vod_id_from_url(url),
        url=url,
        job_id=job_id,
    )
    process = subprocess.Popen(cmd, cwd=str(BACKEND_ROOT), env=pipeline_env(positive_rows))
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
                "holdout_vod_progress",
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
        "holdout_vod_done",
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


def pipeline_env(positive_rows: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(BACKEND_ROOT), existing_pythonpath]) if existing_pythonpath else str(BACKEND_ROOT)
    env["SERVING_CROSS_PLATFORM_RECALL_ENABLED"] = "false"
    env["SERVING_CROSS_PLATFORM_FEATURE_BOOSTS_ENABLED"] = "false"
    env["SERVING_CROSS_PLATFORM_POSITIVE_ROWS_PATH"] = str(positive_rows)
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
