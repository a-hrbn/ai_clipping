from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import pathlib
import re
import sys
import time
from datetime import date
from urllib.parse import urlparse

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from job_manager import create_job, get_job
from ml.collect_twitch_clip_labels import (
    TwitchClient,
    _get_user,
    _get_vods,
    _resolved_broadcasters,
)

DEFAULT_SUMMARY_PATH = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_clip_judge_backfill_summary.json"
DEFAULT_SNAPSHOT_PATH = pathlib.Path(settings.storage_dir) / "feedback" / "mm_judge_snapshots.jsonl"
DEFAULT_SOURCE_MANIFEST = BACKEND_ROOT / "ml" / "streamer_sources.csv"
DEFAULT_LABEL_MANIFEST = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_twitch_clip_labels.jsonl"
DEFAULT_CACHED_VOD_MANIFEST = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_vods.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill LLM Clip-Judge snapshots by sending Twitch VODs through "
            "the serving pipeline path."
        )
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--broadcaster", default="", help="Explicit Twitch broadcaster login.")
    parser.add_argument("--date-from", default="", help="YYYY-MM-DD. Used when discovering VODs via Twitch.")
    parser.add_argument("--date-to", default="", help="YYYY-MM-DD. Used when discovering VODs via Twitch.")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument(
        "--label-manifest",
        default=str(DEFAULT_LABEL_MANIFEST),
        help="Existing Twitch label JSONL used as fallback for older date ranges missing from Twitch archive discovery.",
    )
    parser.add_argument(
        "--fallback-label-manifest",
        dest="fallback_label_manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When date-range Twitch discovery returns no VODs, recover VOD URLs from --label-manifest.",
    )
    parser.add_argument("--vod-url", action="append", default=[], help="Twitch VOD URL. Can be passed multiple times.")
    parser.add_argument(
        "--vod-manifest",
        default="",
        help="Optional CSV/JSONL/TXT manifest containing vod_url/source_url/url/vod_id values.",
    )
    parser.add_argument(
        "--cached-vod-manifest",
        default=str(DEFAULT_CACHED_VOD_MANIFEST),
        help=(
            "JSONL manifest with cached scoped window video_path rows. "
            "Used to avoid re-downloading old/deleted Twitch VODs."
        ),
    )
    parser.add_argument(
        "--use-cached-windows",
        dest="use_cached_windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer local scoped window source.mp4 files over Twitch VOD downloads when available.",
    )
    parser.add_argument(
        "--max-cached-windows-per-vod",
        type=int,
        default=0,
        help="Maximum cached scoped windows per VOD. 0 means all available cached windows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum planned jobs to process after VOD de-dupe and cached-window expansion.",
    )
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--snapshot-path", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument(
        "--disable-face-detection",
        dest="disable_face_detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable MediaPipe/MTCNN/DeepFace for headless backfills before "
            "importing the serving pipeline. Keeps scene/motion features."
        ),
    )
    parser.add_argument(
        "--skip-existing-snapshots",
        dest="skip_existing_snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip VODs whose ID already appears in the snapshot file.",
    )
    parser.add_argument(
        "--strict-judge",
        dest="strict_judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force ENABLE_CLIP_JUDGE + fail-fast + no provider fallback for this process.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later VODs after a failed pipeline job. Default aborts on first failure.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan and write summary without running pipeline jobs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started_at = time.time()
    snapshot_path = pathlib.Path(args.snapshot_path)
    summary_path = pathlib.Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.disable_face_detection:
        _enable_headless_video_feature_mode()
    if args.strict_judge:
        _force_strict_judge_settings()

    source_urls = _source_vod_urls(args)
    existing_snapshot_vods = _snapshot_vod_ids(snapshot_path) if args.skip_existing_snapshots else set()
    deduped_urls = _dedupe_vod_urls(source_urls)
    eligible_vod_plan = _planned_vods(
        source_urls,
        skip_vod_ids=existing_snapshot_vods,
        limit=0,
    )
    eligible_plan = _expand_cached_windows(
        eligible_vod_plan,
        cached_manifest=pathlib.Path(args.cached_vod_manifest),
        use_cached_windows=bool(args.use_cached_windows),
        max_windows_per_vod=max(int(args.max_cached_windows_per_vod or 0), 0),
    )
    limit = max(int(args.limit or 0), 0)
    plan = eligible_plan[:limit] if limit > 0 else eligible_plan
    skipped_existing = sum(
        1 for url in deduped_urls if _vod_id_from_url(url) in existing_snapshot_vods
    )
    before_snapshot_rows = _jsonl_row_count(snapshot_path)
    planned_vod_ids = _distinct_vod_ids(plan)
    eligible_vod_ids = _distinct_vod_ids(eligible_plan)
    planned_cached_windows = sum(1 for item in plan if item.get("input_mode") == "cached_window")
    eligible_cached_windows = sum(1 for item in eligible_plan if item.get("input_mode") == "cached_window")
    planned_full_vods = sum(1 for item in plan if item.get("input_mode") == "vod_url")

    summary: dict[str, object] = {
        "status": "planned" if args.dry_run else "running",
        "profile": args.profile,
        "broadcaster": args.broadcaster,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "strict_judge": bool(args.strict_judge),
        "disable_face_detection": bool(args.disable_face_detection),
        "continue_on_error": bool(args.continue_on_error),
        "skip_existing_snapshots": bool(args.skip_existing_snapshots),
        "use_cached_windows": bool(args.use_cached_windows),
        "cached_vod_manifest": str(pathlib.Path(args.cached_vod_manifest)),
        "max_cached_windows_per_vod": max(int(args.max_cached_windows_per_vod or 0), 0),
        "snapshot_path": str(snapshot_path),
        "summary_path": str(summary_path),
        "input_vods": len(deduped_urls),
        "eligible_vods": len(eligible_vod_plan),
        "eligible_items": len(eligible_plan),
        "eligible_cached_windows": eligible_cached_windows,
        "eligible_full_vods": sum(1 for item in eligible_plan if item.get("input_mode") == "vod_url"),
        "eligible_vod_ids": eligible_vod_ids,
        "planned_vods": len(planned_vod_ids),
        "planned_items": len(plan),
        "planned_cached_windows": planned_cached_windows,
        "planned_full_vods": planned_full_vods,
        "planned_vod_ids": planned_vod_ids,
        "planned_cached_window_ids": [
            str(item.get("scoped_window_id") or "") for item in plan if item.get("input_mode") == "cached_window"
        ],
        "skipped_existing_snapshot_vods": max(skipped_existing, 0),
        "skipped_by_limit": max(len(eligible_plan) - len(plan), 0),
        "snapshot_rows_before": before_snapshot_rows,
        "jobs": [],
    }
    _log_event("clip_judge_backfill_plan", **{key: value for key, value in summary.items() if key != "jobs"})

    if args.dry_run:
        summary["status"] = "dry_run"
        summary["elapsed_sec"] = round(time.time() - started_at, 3)
        _write_json(summary_path, summary)
        return 0

    if not plan:
        summary["status"] = "no_vods"
        summary["elapsed_sec"] = round(time.time() - started_at, 3)
        _write_json(summary_path, summary)
        _log_event("clip_judge_backfill_done", status="no_vods", summary_path=str(summary_path))
        return 0

    failed = False
    for index, item in enumerate(plan, start=1):
        result = asyncio.run(_run_one_vod(item, profile=args.profile, index=index, total=len(plan)))
        summary["jobs"].append(result)
        _write_json(summary_path, summary)
        if result["status"] != "done":
            failed = True
            if not args.continue_on_error:
                break

    after_snapshot_rows = _jsonl_row_count(snapshot_path)
    summary["snapshot_rows_after"] = after_snapshot_rows
    summary["snapshot_rows_added"] = max(after_snapshot_rows - before_snapshot_rows, 0)
    summary["elapsed_sec"] = round(time.time() - started_at, 3)
    summary["status"] = "failed" if failed else "done"
    _write_json(summary_path, summary)
    _log_event(
        "clip_judge_backfill_done",
        status=summary["status"],
        planned_vods=len(planned_vod_ids),
        planned_items=len(plan),
        planned_cached_windows=planned_cached_windows,
        planned_full_vods=planned_full_vods,
        completed_jobs=sum(1 for job in summary["jobs"] if job.get("status") == "done"),
        failed_jobs=sum(1 for job in summary["jobs"] if job.get("status") != "done"),
        snapshot_rows_added=summary["snapshot_rows_added"],
        summary_path=str(summary_path),
    )
    return 1 if failed else 0


async def _run_one_vod(item: dict[str, object], *, profile: str, index: int, total: int) -> dict[str, object]:
    from job_manager import create_job, get_job
    from pipeline import run_pipeline

    vod_id = str(item["vod_id"])
    vod_url = str(item["vod_url"])
    input_mode = str(item.get("input_mode") or "vod_url")
    scoped_window_id = str(item.get("scoped_window_id") or "")
    video_path_text = str(item.get("video_path") or "")
    job_suffix = _safe_job_component(scoped_window_id or input_mode)
    job_id = f"judge_backfill_{vod_id}_{job_suffix}_{index}_{int(time.time())}"
    create_job(job_id, profile=profile)
    started_at = time.time()
    event_fields = {
        "job_id": job_id,
        "vod_id": vod_id,
        "vod_url": vod_url,
        "input_mode": input_mode,
        "video_path": video_path_text,
        "scoped_window_id": scoped_window_id,
        "source_start_offset": item.get("source_start_offset"),
        "source_end_offset": item.get("source_end_offset"),
        "index": index,
        "total": total,
    }
    _log_event("clip_judge_backfill_job_start", **event_fields)
    if video_path_text:
        await run_pipeline(
            job_id=job_id,
            url=vod_url,
            video_path=pathlib.Path(video_path_text),
            profile=profile,
        )
    else:
        await run_pipeline(job_id=job_id, url=vod_url, profile=profile)
    job = get_job(job_id)
    status = str(getattr(job, "status", "") or "missing")
    error = str(getattr(job, "error", "") or "")
    result = {
        "job_id": job_id,
        "vod_id": vod_id,
        "vod_url": vod_url,
        "input_mode": input_mode,
        "video_path": video_path_text,
        "scoped_window_id": scoped_window_id,
        "source_start_offset": item.get("source_start_offset"),
        "source_end_offset": item.get("source_end_offset"),
        "status": status,
        "error": error,
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    _log_event("clip_judge_backfill_job_done", **result)
    return result


def _force_strict_judge_settings() -> None:
    settings.enable_clip_judge = True
    settings.clip_judge_fail_fast = True
    settings.judge_provider_fallback_enabled = False
    settings.judge_all_candidates = True
    if not (settings.anthropic_api_key or settings.gemini_api_key or settings.groq_api_key):
        raise SystemExit(
            "Strict judge backfill requires at least one configured remote LLM provider "
            "(ANTHROPIC_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY)."
        )


def _enable_headless_video_feature_mode() -> None:
    os.environ.setdefault("AI_CLIPPING_DISABLE_FACE_DETECTION", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")


def _source_vod_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    urls.extend(str(url or "").strip() for url in args.vod_url)
    if args.vod_manifest:
        urls.extend(_read_vod_manifest(pathlib.Path(args.vod_manifest)))
    used_discovery = bool(args.date_from or args.date_to)
    if args.date_from or args.date_to:
        if not args.date_from or not args.date_to:
            raise SystemExit("--date-from and --date-to must be provided together.")
        urls.extend(
            _discover_vod_urls(
                broadcaster=args.broadcaster,
                profile=args.profile,
                date_from=args.date_from,
                date_to=args.date_to,
                sources=pathlib.Path(args.sources),
            )
        )
        if not urls and bool(args.fallback_label_manifest):
            urls.extend(
                _discover_vod_urls_from_label_manifest(
                    pathlib.Path(args.label_manifest),
                    broadcaster=args.broadcaster,
                    date_from=args.date_from,
                    date_to=args.date_to,
                )
            )
    urls = [url for url in urls if url]
    if not urls:
        if used_discovery:
            return []
        raise SystemExit("No VODs provided. Use --vod-url, --vod-manifest, or --date-from/--date-to.")
    return urls


def _discover_vod_urls(
    *,
    broadcaster: str,
    profile: str,
    date_from: str,
    date_to: str,
    sources: pathlib.Path,
) -> list[str]:
    if not settings.twitch_client_id or not settings.twitch_client_secret:
        raise SystemExit("TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET are required for date-range discovery.")
    explicit_broadcaster = str(broadcaster or "").strip().lower()
    broadcasters = (
        [explicit_broadcaster]
        if explicit_broadcaster
        else _resolved_broadcasters(
            broadcaster="",
            profile=profile,
            manifest_path=sources,
        )
    )
    if not broadcasters:
        raise SystemExit("No Twitch broadcasters resolved for the requested profile/broadcaster.")
    client = TwitchClient(settings.twitch_client_id, settings.twitch_client_secret)
    urls: list[str] = []
    for login in broadcasters:
        try:
            user = _get_user(client, login)
        except ValueError as exc:
            _log_event(
                "clip_judge_backfill_broadcaster_unavailable",
                broadcaster=login,
                error=str(exc),
            )
            continue
        vods = _get_vods(client, str(user["id"]), date_from, date_to)
        _log_event("clip_judge_backfill_vods_discovered", broadcaster=login, vods=len(vods))
        urls.extend(f"https://www.twitch.tv/videos/{vod['id']}" for vod in vods if str(vod.get("id") or "").strip())
    return urls


def _discover_vod_urls_from_label_manifest(
    path: pathlib.Path,
    *,
    broadcaster: str,
    date_from: str,
    date_to: str,
) -> list[str]:
    if not path.exists():
        _log_event(
            "clip_judge_backfill_label_manifest_missing",
            path=str(path),
        )
        return []
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    target_broadcaster = str(broadcaster or "").strip().lower()
    urls: dict[str, str] = {}
    for row in _read_jsonl(path):
        if target_broadcaster and str(row.get("broadcaster") or "").strip().lower() != target_broadcaster:
            continue
        created_date = _date_from_row(row)
        if created_date is None or not (start <= created_date <= end):
            continue
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id:
            vod_id = _vod_id_from_text(str(row.get("vod_url") or row.get("source_url") or ""))
        if not vod_id:
            continue
        urls.setdefault(vod_id, f"https://www.twitch.tv/videos/{vod_id}")
    _log_event(
        "clip_judge_backfill_label_manifest_vods_discovered",
        path=str(path),
        broadcaster=target_broadcaster,
        date_from=date_from,
        date_to=date_to,
        vods=len(urls),
    )
    return list(urls.values())


def _read_vod_manifest(path: pathlib.Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"VOD manifest does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
        return [_url_from_manifest_row(row) for row in rows]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [_url_from_manifest_row(row) for row in csv.DictReader(handle)]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _url_from_manifest_row(row: dict) -> str:
    for key in ("vod_url", "source_url", "url"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    vod_id = str(row.get("vod_id") or "").strip()
    if vod_id:
        return f"https://www.twitch.tv/videos/{vod_id}"
    return ""


def _planned_vods(urls: list[str], *, skip_vod_ids: set[str], limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for url in _dedupe_vod_urls(urls):
        vod_id = _vod_id_from_url(url)
        if not vod_id:
            continue
        if vod_id in skip_vod_ids:
            continue
        rows.append(
            {
                "vod_id": vod_id,
                "vod_url": f"https://www.twitch.tv/videos/{vod_id}",
                "input_mode": "vod_url",
            }
        )
    if limit > 0:
        return rows[:limit]
    return rows


def _expand_cached_windows(
    vod_plan: list[dict[str, str]],
    *,
    cached_manifest: pathlib.Path,
    use_cached_windows: bool,
    max_windows_per_vod: int,
) -> list[dict[str, object]]:
    if not use_cached_windows or not vod_plan:
        return [dict(item) for item in vod_plan]
    vod_ids = {str(item.get("vod_id") or "") for item in vod_plan if str(item.get("vod_id") or "")}
    windows_by_vod = _cached_windows_by_vod(cached_manifest, vod_ids=vod_ids)
    expanded: list[dict[str, object]] = []
    cached_vods = 0
    cached_windows = 0
    for vod_item in vod_plan:
        vod_id = str(vod_item.get("vod_id") or "")
        windows = windows_by_vod.get(vod_id, [])
        if max_windows_per_vod > 0:
            windows = windows[:max_windows_per_vod]
        if not windows:
            expanded.append(dict(vod_item))
            continue
        cached_vods += 1
        cached_windows += len(windows)
        for window in windows:
            item = dict(vod_item)
            item.update(window)
            item["input_mode"] = "cached_window"
            expanded.append(item)
    _log_event(
        "clip_judge_backfill_cached_windows_resolved",
        manifest_path=str(cached_manifest),
        requested_vods=len(vod_ids),
        cached_vods=cached_vods,
        cached_windows=cached_windows,
        fallback_vod_downloads=sum(1 for item in expanded if item.get("input_mode") == "vod_url"),
        max_cached_windows_per_vod=max_windows_per_vod,
    )
    return expanded


def _cached_windows_by_vod(cached_manifest: pathlib.Path, *, vod_ids: set[str]) -> dict[str, list[dict[str, object]]]:
    if not cached_manifest.exists():
        _log_event("clip_judge_backfill_cached_manifest_missing", path=str(cached_manifest))
        return {}
    rows_by_vod: dict[str, list[dict[str, object]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in _read_jsonl(cached_manifest):
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id or vod_id not in vod_ids:
            continue
        video_path = _resolve_cached_video_path(str(row.get("video_path") or ""))
        if video_path is None:
            continue
        scoped_window_id = str(row.get("scoped_window_id") or video_path.parent.name).strip()
        key = (vod_id, scoped_window_id, str(video_path))
        if key in seen:
            continue
        seen.add(key)
        rows_by_vod.setdefault(vod_id, []).append(
            {
                "vod_id": vod_id,
                "vod_url": str(row.get("vod_url") or f"https://www.twitch.tv/videos/{vod_id}"),
                "video_path": str(video_path),
                "scoped_window_id": scoped_window_id,
                "source_start_offset": row.get("source_start_offset"),
                "source_end_offset": row.get("source_end_offset"),
                "source_window_start": row.get("source_window_start"),
                "source_window_end": row.get("source_window_end"),
                "manifest_source": row.get("manifest_source") or "cached_vod_manifest",
            }
        )
    for rows in rows_by_vod.values():
        rows.sort(key=lambda item: (_as_float(item.get("source_start_offset")), str(item.get("scoped_window_id") or "")))
    return rows_by_vod


def _resolve_cached_video_path(value: str) -> pathlib.Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    direct = pathlib.Path(text)
    if direct.exists():
        return direct
    app_backend_prefix = "/app/backend/"
    if text.startswith(app_backend_prefix):
        repaired = BACKEND_ROOT / text[len(app_backend_prefix):]
        if repaired.exists():
            return repaired
    app_root_prefix = "/app/"
    if text.startswith(app_root_prefix):
        repaired = BACKEND_ROOT.parent / text[len(app_root_prefix):]
        if repaired.exists():
            return repaired
    return None


def _distinct_vod_ids(rows: list[dict[str, object]]) -> list[str]:
    vod_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        vod_id = str(row.get("vod_id") or "")
        if not vod_id or vod_id in seen:
            continue
        seen.add(vod_id)
        vod_ids.append(vod_id)
    return vod_ids


def _safe_job_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned[:48] or "vod"


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe_vod_urls(urls: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        vod_id = _vod_id_from_url(url)
        key = vod_id or url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(url)
    return deduped


def _snapshot_vod_ids(path: pathlib.Path) -> set[str]:
    vod_ids: set[str] = set()
    for row in _read_jsonl(path):
        for key in ("vod_id", "source_url", "vod_url", "candidate_id"):
            value = str(row.get(key) or "")
            vod_id = _vod_id_from_text(value)
            if vod_id:
                vod_ids.add(vod_id)
    return vod_ids


def _vod_id_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    text = f"{parsed.netloc}{parsed.path}" if parsed.netloc else str(url or "")
    return _vod_id_from_text(text)


def _vod_id_from_text(text: str) -> str:
    match = re.search(r"(?:twitch\.tv/videos/|twitch_vod_|vod_)?(\d{6,})", str(text or ""))
    return match.group(1) if match else ""


def _date_from_row(row: dict) -> date | None:
    for key in ("created_at", "vod_created_at", "published_at"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            continue
    return None


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _jsonl_row_count(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _log_event(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
