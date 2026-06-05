from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import export_serving_hard_negatives
from ml.collect_twitch_clip_labels import TwitchClient, _get_user, _get_vods
from ml.row_identity import read_jsonl, write_json
from ml.run_serving_recall_holdout import (
    _load_json_object,
    _read_json,
    _vod_id_from_url,
    emit_progress,
    run_pipeline_job,
)

DEFAULT_PROFILE = "jynxzi"
DEFAULT_DATE_FROM = "2026-04-27"
DEFAULT_DATE_TO = "2026-05-26"
DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_STORAGE_DIR = BACKEND_ROOT / "tmp" / "storage"
DEFAULT_VOD_INVENTORY = DEFAULT_ARTIFACT_DIR / "jynxzi_twitch_vod_inventory.jsonl"
DEFAULT_POSITIVE_ROWS = DEFAULT_ARTIFACT_DIR / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_HOLDOUT_POLICY = DEFAULT_ARTIFACT_DIR / "jynxzi_training_holdout_policy_rows.jsonl"
DEFAULT_ROWS_OUT = DEFAULT_ARTIFACT_DIR / "jynxzi_serving_hard_negative_train_rows.jsonl"
DEFAULT_EXPORT_SUMMARY = DEFAULT_ARTIFACT_DIR / "jynxzi_serving_hard_negative_train_summary.json"
DEFAULT_RUN_SUMMARY = DEFAULT_ARTIFACT_DIR / "jynxzi_serving_hard_negative_collection_summary.json"
RUNNER_NAME = "serving_hard_negative_collection"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a date range of Jynxzi VODs through the oracle-free serving pipeline "
            "and export high-ranked organic misses as hard negatives."
        )
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--date-from", default=DEFAULT_DATE_FROM)
    parser.add_argument("--date-to", default=DEFAULT_DATE_TO)
    parser.add_argument("--vod-inventory", type=pathlib.Path, default=DEFAULT_VOD_INVENTORY)
    parser.add_argument("--positive-rows", type=pathlib.Path, default=DEFAULT_POSITIVE_ROWS)
    parser.add_argument("--holdout-policy-rows", type=pathlib.Path, default=DEFAULT_HOLDOUT_POLICY)
    parser.add_argument("--storage-dir", type=pathlib.Path, default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--rows-out", type=pathlib.Path, default=DEFAULT_ROWS_OUT)
    parser.add_argument("--export-summary-out", type=pathlib.Path, default=DEFAULT_EXPORT_SUMMARY)
    parser.add_argument("--run-summary-output", type=pathlib.Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--no-twitch-refresh", action="store_true")
    parser.add_argument("--allow-holdout-vods", action="store_true")
    parser.add_argument("--progress-interval-sec", type=float, default=10.0)
    parser.add_argument("--strategy-json", default="{}")
    parser.add_argument("--pre-judge-top-k", type=int, default=500)
    parser.add_argument("--max-hard-negatives-per-vod", type=int, default=20)
    parser.add_argument(
        "--max-hard-negatives",
        type=int,
        default=0,
        help="Global export cap. 0 means uncapped; per-VOD cap still applies.",
    )
    parser.add_argument("--positive-exclusion-overlap", type=float, default=0.25)
    parser.add_argument("--min-rank-score", type=float, default=-999.0)
    args = parser.parse_args(argv)

    started_at = time.time()
    _validate_date_range(args.date_from, args.date_to)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    strategy = _load_json_object(args.strategy_json, label="strategy-json")
    selected_result = select_vods_for_date_range(
        inventory_path=args.vod_inventory,
        holdout_policy_rows_path=args.holdout_policy_rows,
        profile=args.profile,
        date_from=args.date_from,
        date_to=args.date_to,
        refresh_twitch=not bool(args.no_twitch_refresh),
        include_holdout_vods=bool(args.allow_holdout_vods),
    )
    selected_vods = selected_result["selected_vods"]
    if args.limit and args.limit > 0:
        selected_vods = selected_vods[: args.limit]

    emit_progress(
        "hard_negative_collection_started",
        profile=args.profile,
        run_id=run_id,
        date_from=args.date_from,
        date_to=args.date_to,
        total_vods=len(selected_vods),
        vod_inventory=str(args.vod_inventory),
        positive_rows=str(args.positive_rows),
        holdout_policy_rows=str(args.holdout_policy_rows),
        twitch_refresh=not bool(args.no_twitch_refresh),
        dry_run=bool(args.dry_run),
        selection_summary=selected_result["summary"],
    )
    if not selected_vods and not args.dry_run and not args.allow_empty:
        summary = {
            "event": "serving_hard_negative_collection_done",
            "profile": args.profile,
            "run_id": run_id,
            "date_from": args.date_from,
            "date_to": args.date_to,
            "dry_run": bool(args.dry_run),
            "status": "failed",
            "failure_reason": "no_vods_selected",
            "vod_inventory": str(args.vod_inventory),
            "positive_rows": str(args.positive_rows),
            "holdout_policy_rows": str(args.holdout_policy_rows),
            "storage_dir": str(args.storage_dir),
            "rows_out": str(args.rows_out),
            "export_summary_out": str(args.export_summary_out),
            "requested_vods": 0,
            "selected_vod_ids": [],
            "selection_summary": selected_result["summary"],
            "jobs": [],
            "jobs_by_status": {},
            "skipped": {},
            "elapsed_sec": round(time.time() - started_at, 3),
            "export_summary": {},
        }
        write_json(args.run_summary_output, summary)
        emit_progress(
            "hard_negative_collection_done",
            run_id=run_id,
            status="failed",
            failure_reason="no_vods_selected",
            requested_vods=0,
            elapsed_sec=summary["elapsed_sec"],
            run_summary_output=str(args.run_summary_output),
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
        return 1

    run_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    failed = False
    for vod_index, vod in enumerate(selected_vods, start=1):
        vod_id = str(vod.get("vod_id") or "").strip()
        url = str(vod.get("vod_url") or vod.get("source_url") or f"https://www.twitch.tv/videos/{vod_id}").strip()
        if not vod_id or not url:
            skipped["missing_vod_id_or_url"] += 1
            emit_progress(
                "hard_negative_collection_vod_skipped",
                vod_index=vod_index,
                total_vods=len(selected_vods),
                vod_id=vod_id,
                reason="missing_vod_id_or_url",
            )
            continue
        existing = find_existing_collection_job(args.storage_dir, vod_id)
        if args.skip_existing and existing:
            emit_progress(
                "hard_negative_collection_vod_reused",
                vod_index=vod_index,
                total_vods=len(selected_vods),
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
                    "created_at": str(vod.get("created_at") or ""),
                }
            )
            continue
        job_id = planned_job_id(run_id, vod_id)
        if args.dry_run:
            emit_progress(
                "hard_negative_collection_vod_planned",
                vod_index=vod_index,
                total_vods=len(selected_vods),
                vod_id=vod_id,
                url=url,
                job_id=job_id,
                created_at=str(vod.get("created_at") or ""),
                duration_sec=vod.get("vod_duration_seconds"),
            )
            run_rows.append(
                {
                    "vod_id": vod_id,
                    "url": url,
                    "job_id": job_id,
                    "status": "dry_run",
                    "created_at": str(vod.get("created_at") or ""),
                    "vod_duration_seconds": vod.get("vod_duration_seconds"),
                }
            )
            continue

        row = run_pipeline_job(
            job_id=job_id,
            url=url,
            profile=args.profile,
            strategy=strategy,
            storage_dir=args.storage_dir,
            positive_rows=args.positive_rows,
            enable_clip_judge=True,
            judge_sample_count=1,
            vod_index=vod_index,
            total_vods=len(selected_vods),
            progress_interval_sec=max(float(args.progress_interval_sec or 0.0), 0.5),
            runner_name=RUNNER_NAME,
            event_prefix="hard_negative_collection",
        )
        row["created_at"] = str(vod.get("created_at") or "")
        row["vod_duration_seconds"] = vod.get("vod_duration_seconds")
        run_rows.append(row)
        if row.get("status") != "done":
            failed = True
            if args.fail_fast:
                break

    export_summary: dict[str, Any] = {}
    selected_vod_ids = {str(vod.get("vod_id") or "").strip() for vod in selected_vods if str(vod.get("vod_id") or "").strip()}
    if not args.dry_run and selected_vod_ids:
        emit_progress(
            "hard_negative_export_started",
            run_id=run_id,
            vod_ids=sorted(selected_vod_ids),
            rows_out=str(args.rows_out),
            summary_out=str(args.export_summary_out),
        )
        export_payload = export_serving_hard_negatives.export_serving_hard_negatives(
            storage_dir=args.storage_dir,
            positive_rows_path=args.positive_rows,
            holdout_policy_rows_path=args.holdout_policy_rows,
            include_holdout_vods=bool(args.allow_holdout_vods),
            vod_ids=selected_vod_ids,
            profile=args.profile,
            pre_judge_top_k=max(int(args.pre_judge_top_k), 0),
            max_per_vod=max(int(args.max_hard_negatives_per_vod), 0),
            max_rows=max(int(args.max_hard_negatives), 0),
            positive_exclusion_overlap=max(min(float(args.positive_exclusion_overlap), 1.0), 0.0),
            min_rank_score=float(args.min_rank_score),
        )
        export_serving_hard_negatives._write_jsonl(args.rows_out, export_payload["rows"])
        export_serving_hard_negatives._write_json(args.export_summary_out, export_payload["summary"])
        export_summary = export_payload["summary"]
        emit_progress(
            "hard_negative_export_done",
            run_id=run_id,
            exported_rows=export_summary.get("exported_rows"),
            unique_vods=export_summary.get("unique_vods"),
            skipped=export_summary.get("skipped"),
            rows_out=str(args.rows_out),
            summary_out=str(args.export_summary_out),
        )

    summary = {
        "event": "serving_hard_negative_collection_done",
        "profile": args.profile,
        "run_id": run_id,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "dry_run": bool(args.dry_run),
        "vod_inventory": str(args.vod_inventory),
        "positive_rows": str(args.positive_rows),
        "holdout_policy_rows": str(args.holdout_policy_rows),
        "storage_dir": str(args.storage_dir),
        "rows_out": str(args.rows_out),
        "export_summary_out": str(args.export_summary_out),
        "requested_vods": len(selected_vods),
        "selected_vod_ids": sorted(selected_vod_ids),
        "selection_summary": selected_result["summary"],
        "jobs": run_rows,
        "jobs_by_status": dict(sorted(Counter(str(row.get("status") or "unknown") for row in run_rows).items())),
        "skipped": dict(sorted(skipped.items())),
        "elapsed_sec": round(time.time() - started_at, 3),
        "export_summary": export_summary,
    }
    write_json(args.run_summary_output, summary)
    emit_progress(
        "hard_negative_collection_done",
        run_id=run_id,
        status="failed" if failed else "done",
        requested_vods=len(selected_vods),
        jobs_by_status=summary["jobs_by_status"],
        exported_rows=export_summary.get("exported_rows"),
        elapsed_sec=summary["elapsed_sec"],
        run_summary_output=str(args.run_summary_output),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 1 if failed and args.fail_fast else 0


def select_vods_for_date_range(
    *,
    inventory_path: pathlib.Path,
    holdout_policy_rows_path: pathlib.Path,
    profile: str,
    date_from: str,
    date_to: str,
    refresh_twitch: bool,
    include_holdout_vods: bool,
) -> dict[str, Any]:
    inventory_rows = read_jsonl(inventory_path)
    api_rows: list[dict[str, Any]] = []
    api_error = ""
    if refresh_twitch:
        try:
            api_rows = fetch_twitch_vods(profile=profile, date_from=date_from, date_to=date_to)
        except Exception as exc:  # noqa: BLE001 - surfaced in summary so fallback remains useful.
            api_error = f"{type(exc).__name__}: {exc}"
            api_rows = []
    merged = _merge_vod_rows([*inventory_rows, *api_rows])
    holdout_vod_ids = load_holdout_vod_ids(holdout_policy_rows_path)
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in merged:
        vod_id = str(row.get("vod_id") or "").strip()
        created_at = str(row.get("created_at") or "").strip()
        created_date = _date_key(created_at)
        if not created_date:
            skipped["missing_created_at"] += 1
            continue
        if not (date_from <= created_date <= date_to):
            skipped["outside_date_range"] += 1
            continue
        if vod_id in holdout_vod_ids and not include_holdout_vods:
            skipped["holdout_vod_id"] += 1
            continue
        selected.append(dict(row))
    selected.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("vod_id") or "")))
    return {
        "selected_vods": selected,
        "summary": {
            "date_from": date_from,
            "date_to": date_to,
            "inventory_rows": len(inventory_rows),
            "twitch_api_rows": len(api_rows),
            "twitch_api_error": api_error,
            "merged_rows": len(merged),
            "holdout_vods": len(holdout_vod_ids),
            "include_holdout_vods": bool(include_holdout_vods),
            "selected_vods": len(selected),
            "selected_vod_ids": [str(row.get("vod_id") or "") for row in selected],
            "skipped": dict(sorted(skipped.items())),
        },
    }


def fetch_twitch_vods(*, profile: str, date_from: str, date_to: str) -> list[dict[str, Any]]:
    if not settings.twitch_client_id or not settings.twitch_client_secret:
        raise RuntimeError("missing_twitch_credentials")
    client = TwitchClient(settings.twitch_client_id, settings.twitch_client_secret)
    user = _get_user(client, profile)
    rows: list[dict[str, Any]] = []
    for vod in _get_vods(client, str(user["id"]), date_from, date_to):
        vod_id = str(vod.get("id") or vod.get("vod_id") or "").strip()
        if not vod_id:
            continue
        rows.append(
            {
                "vod_id": vod_id,
                "vod_url": str(vod.get("url") or f"https://www.twitch.tv/videos/{vod_id}"),
                "vod_duration_seconds": _optional_float(vod.get("duration_seconds")),
                "created_at": str(vod.get("created_at") or ""),
                "title": str(vod.get("title") or ""),
                "broadcaster": profile,
                "streamer": profile,
                "inventory_source": "twitch_vod_api",
            }
        )
    return rows


def load_holdout_vod_ids(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    vod_ids: set[str] = set()
    for row in read_jsonl(path):
        vod_id = str(row.get("vod_id") or row.get("twitch_vod_id") or "").strip()
        if vod_id:
            vod_ids.add(vod_id)
    return vod_ids


def find_existing_collection_job(storage_dir: pathlib.Path, vod_id: str) -> pathlib.Path | None:
    candidates: list[pathlib.Path] = []
    if not storage_dir.exists():
        return None
    for job_dir in storage_dir.iterdir():
        if not job_dir.is_dir() or not (job_dir / "debug" / "ranked_candidates_pre_judge.jsonl").exists():
            continue
        request = _read_json(job_dir / "request.json")
        if str(request.get("runner") or "") != RUNNER_NAME:
            continue
        if _vod_id_from_url(str(request.get("url") or "")) != vod_id:
            continue
        candidates.append(job_dir)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def planned_job_id(run_id: str, vod_id: str) -> str:
    safe_run = "".join(char if char.isalnum() or char in "._-" else "_" for char in run_id).strip("_") or "run"
    safe_vod = "".join(char for char in str(vod_id) if char.isdigit()) or str(vod_id)
    return f"hardneg_collect_{safe_run}_{safe_vod}"


def _merge_vod_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_vod: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        vod_id = str(row.get("vod_id") or row.get("id") or "").strip()
        if not vod_id:
            continue
        current = merged_by_vod.get(vod_id, {})
        copied = dict(row)
        copied["vod_id"] = vod_id
        copied["vod_url"] = str(copied.get("vod_url") or copied.get("url") or f"https://www.twitch.tv/videos/{vod_id}")
        copied["_input_index"] = index
        merged = {**current, **copied}
        for key in ("created_at", "title", "vod_duration_seconds"):
            if current.get(key) not in (None, "", 0, 0.0) and copied.get(key) in (None, "", 0, 0.0):
                merged[key] = current[key]
        merged_by_vod[vod_id] = merged
    return sorted(merged_by_vod.values(), key=lambda row: (_created_timestamp(row.get("created_at")), str(row.get("vod_id") or "")))


def _date_key(value: str) -> str:
    parsed = _parse_datetime(value)
    if parsed is None:
        return ""
    return parsed.date().isoformat()


def _created_timestamp(value: Any) -> float:
    parsed = _parse_datetime(value)
    return parsed.timestamp() if parsed is not None else 0.0


def _parse_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_date_range(date_from: str, date_to: str) -> None:
    try:
        start = datetime.fromisoformat(date_from).date()
        end = datetime.fromisoformat(date_to).date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date range: {exc}") from exc
    if end < start:
        raise SystemExit("date-to must be on or after date-from")


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
