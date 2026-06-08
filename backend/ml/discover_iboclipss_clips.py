"""Discover and download @Iboclipss YouTube Shorts that match Jynxzi VOD dates.

Usage:
  python ml/discover_iboclipss_clips.py --vods ml/jynxzi_gt_vods.csv
  python ml/discover_iboclipss_clips.py --vods ml/jynxzi_gt_vods.csv --discover-only
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import certifi
import yt_dlp

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from services.downloader import ensure_cached_download
from services.yt_dlp_auth import apply_yt_dlp_auth_options

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHANNEL = "https://www.youtube.com/@Iboclipss/shorts"
DEFAULT_VOD_CSV = BACKEND_ROOT / "ml" / "jynxzi_gt_vods.csv"
DEFAULT_DISCOVERED_OUT = BACKEND_ROOT / "ml" / "artifacts" / "iboclipss_discovered.jsonl"
DEFAULT_REPOSTS_OUT = BACKEND_ROOT / "ml" / "artifacts" / "iboclipss_reposts.jsonl"
DEFAULT_MAX_CANDIDATES = 250
DEFAULT_WINDOW_DAYS = 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover and download @Iboclipss Shorts matching Jynxzi VOD dates."
    )
    parser.add_argument("--vods", default=str(DEFAULT_VOD_CSV), help="Path to jynxzi_gt_vods.csv")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--discovered-out", default=str(DEFAULT_DISCOVERED_OUT))
    parser.add_argument("--reposts-out", default=str(DEFAULT_REPOSTS_OUT))
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only write discovered manifest, skip download.",
    )
    args = parser.parse_args()

    vod_rows = _read_vod_csv(pathlib.Path(args.vods))
    if not vod_rows:
        logger.error("No VOD rows in %s", args.vods)
        return 1

    date_windows = _build_date_windows(vod_rows, window_days=args.window_days)
    logger.info("Loaded %d VODs, %d dates in windows", len(vod_rows), len(date_windows))

    logger.info("Listing channel %s (max %d)...", args.channel, args.max_candidates)
    entries = _list_channel_entries(args.channel, args.max_candidates)
    if not entries:
        logger.error("No entries returned from channel %s", args.channel)
        return 1
    logger.info("Fetched %d channel entries", len(entries))

    entries = _enrich_with_upload_dates(entries, args.channel)
    matched = _filter_to_windows(entries, date_windows)
    logger.info("Matched %d entries within date windows", len(matched))

    discovered_out = pathlib.Path(args.discovered_out)
    discovered_out.parent.mkdir(parents=True, exist_ok=True)
    discovered_out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, default=str) for row in matched),
        encoding="utf-8",
    )
    logger.info("Wrote %d discovered rows → %s", len(matched), discovered_out)

    if args.discover_only or not matched:
        return 0

    reposts = _download_clips(matched)
    reposts_out = pathlib.Path(args.reposts_out)
    reposts_out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, default=str) for row in reposts),
        encoding="utf-8",
    )
    logger.info("Wrote %d repost rows → %s", len(reposts), reposts_out)
    return 0 if reposts else 1


# ── VOD CSV ───────────────────────────────────────────────────────────────────

def _read_vod_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        logger.error("VOD CSV not found: %s", path)
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vod_id = (row.get("vod_id") or "").strip()
            if not vod_id or vod_id.startswith("#"):
                continue
            rows.append({
                "vod_id": vod_id,
                "vod_date": (row.get("vod_date") or "").strip(),
                "note": (row.get("note") or "").strip(),
            })
    return rows


def _build_date_windows(vod_rows: list[dict], window_days: int) -> dict[date, list[str]]:
    """Map each date in [vod_date, vod_date+window_days] to the list of vod_ids it covers."""
    windows: dict[date, list[str]] = {}
    for row in vod_rows:
        vod_date_str = row.get("vod_date", "").strip()
        if not vod_date_str:
            logger.warning("VOD %s has no vod_date — cannot build date window", row["vod_id"])
            continue
        try:
            vod_date = datetime.strptime(vod_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("Invalid vod_date %r for VOD %s — skipping", vod_date_str, row["vod_id"])
            continue
        for delta in range(window_days + 1):
            d = vod_date + timedelta(days=delta)
            windows.setdefault(d, []).append(row["vod_id"])
    return windows


# ── Channel listing ───────────────────────────────────────────────────────────

def _list_channel_entries(channel_url: str, max_candidates: int) -> list[dict]:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max(max_candidates, 1),
        "skip_download": True,
    }
    apply_yt_dlp_auth_options(opts)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
    except Exception:
        logger.exception("Channel listing failed for %s", channel_url)
        return []
    if not info or not isinstance(info, dict):
        return []
    return [e for e in (info.get("entries") or []) if e]


# ── Date enrichment ───────────────────────────────────────────────────────────

def _enrich_with_upload_dates(entries: list[dict], channel_url: str) -> list[dict]:
    ids_needing_dates = {
        str(e.get("id") or "").strip()
        for e in entries
        if str(e.get("id") or "").strip()
        and not _coerce_upload_date(e)
        and _coerce_publish_timestamp(e) <= 0
    }
    if not ids_needing_dates:
        return entries

    logger.info("Fetching publish dates for %d entries without dates...", len(ids_needing_dates))
    try:
        html_text = _fetch_text(channel_url)
    except Exception:
        logger.debug("HTML channel page fetch failed for %s", channel_url, exc_info=True)
        html_text = ""

    publish_dates = _youtube_channel_publish_dates_from_html(html_text)
    enriched: list[dict] = []
    html_hits = meta_fetches = meta_failures = 0

    for entry in entries:
        video_id = str(entry.get("id") or "").strip()
        if not video_id:
            enriched.append(entry)
            continue

        published_at = publish_dates.get(video_id)
        if published_at is not None:
            row = dict(entry)
            if _coerce_publish_timestamp(row) <= 0:
                row["timestamp"] = int(published_at.timestamp())
            if not _coerce_upload_date(row):
                row["upload_date"] = published_at.strftime("%Y%m%d")
            html_hits += 1
            enriched.append(row)
            continue

        if video_id not in ids_needing_dates:
            enriched.append(entry)
            continue

        meta_url = f"https://www.youtube.com/shorts/{video_id}"
        metadata = _fetch_single_video_metadata(meta_url)
        meta_fetches += 1
        if metadata is None:
            meta_failures += 1
            enriched.append(entry)
        else:
            merged = {**entry, **{k: v for k, v in metadata.items() if v not in (None, "", [], {})}}
            enriched.append(merged)

    logger.info(
        "Date enrichment: html_hits=%d meta_fetches=%d meta_failures=%d total=%d",
        html_hits, meta_fetches, meta_failures, len(entries),
    )
    return enriched


def _filter_to_windows(entries: list[dict], date_windows: dict[date, list[str]]) -> list[dict]:
    matched = []
    skipped_no_date = 0
    for entry in entries:
        entry_date = _entry_date(entry)
        if entry_date is None:
            skipped_no_date += 1
            continue
        candidate_vod_ids = date_windows.get(entry_date)
        if not candidate_vod_ids:
            continue
        video_id = str(entry.get("id") or "").strip()
        source_url = (
            str(entry.get("webpage_url") or "").strip()
            or str(entry.get("url") or "").strip()
            or (f"https://www.youtube.com/shorts/{video_id}" if video_id else "")
        )
        if not source_url:
            continue
        matched.append({
            "source_url": source_url,
            "platform": "youtube",
            "clip_id": f"ibo_{video_id}" if video_id else f"ibo_{len(matched)}",
            "title": str(entry.get("title") or "").strip(),
            "upload_date": entry_date.strftime("%Y-%m-%d"),
            "view_count": int(entry.get("view_count") or 0),
            "duration": int(entry.get("duration") or 0),
            "candidate_vod_ids": list(candidate_vod_ids),
        })
    if skipped_no_date:
        logger.info("Skipped %d entries with no resolvable upload date", skipped_no_date)
    return matched


def _entry_date(entry: dict) -> date | None:
    upload_date_str = _coerce_upload_date(entry)
    if upload_date_str:
        try:
            return datetime.strptime(upload_date_str, "%Y%m%d").date()
        except ValueError:
            pass
    ts = _coerce_publish_timestamp(entry)
    if ts > 0:
        return datetime.fromtimestamp(ts, tz=timezone.utc).date()
    return None


# ── Download ──────────────────────────────────────────────────────────────────

def _download_clips(discovered: list[dict]) -> list[dict]:
    reposts = []
    for row in discovered:
        url = (row.get("source_url") or "").strip()
        if not url:
            logger.warning("Skipping %s: no source_url", row.get("clip_id", ""))
            continue
        try:
            video_path = ensure_cached_download(url, cache_namespace="short_form", cache_ttl_sec=-1)
        except Exception:
            logger.warning("Download failed for %s", url, exc_info=True)
            continue
        reposts.append({
            "source_url": url,
            "video_path": str(video_path),
            "platform": "youtube",
            "views": row.get("view_count", 0),
            "title": row.get("title", ""),
            "clip_id": row.get("clip_id", ""),
            "upload_date": row.get("upload_date", ""),
            "candidate_vod_ids": row.get("candidate_vod_ids", []),
        })
        logger.info("Downloaded %s → %s", url, video_path)
    return reposts


# ── Date coercion helpers (from auto_train_short_form.py) ─────────────────────

def _coerce_upload_date(item: dict) -> str:
    value = str(item.get("upload_date") or "").strip()
    if len(value) == 8 and value.isdigit():
        return value
    return ""


def _coerce_publish_timestamp(item: dict) -> int:
    value = item.get("timestamp")
    if value not in (None, ""):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    upload_date = _coerce_upload_date(item)
    if upload_date:
        try:
            dt = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            pass
    return 0


# ── YouTube HTML date extraction (from auto_train_short_form.py) ──────────────

def _youtube_channel_publish_dates_from_html(
    html_text: str, *, now: datetime | None = None
) -> dict[str, datetime]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows: dict[str, datetime] = {}
    pattern = re.compile(
        r'"videoId"\s*:\s*"(?P<id>[A-Za-z0-9_-]{6,})"(?P<body>.{0,4000}?)(?="videoId"\s*:|$)',
        re.DOTALL,
    )
    for match in pattern.finditer(html_text or ""):
        video_id = match.group("id")
        if video_id in rows:
            continue
        relative_text = _youtube_published_time_text(match.group("body"))
        published_at = _parse_youtube_relative_publish_time(relative_text, now=now)
        if published_at is not None:
            rows[video_id] = published_at
    return rows


def _youtube_published_time_text(block: str) -> str:
    for pattern in (
        r'"publishedTimeText"\s*:\s*\{\s*"simpleText"\s*:\s*"(?P<text>[^"]+)"',
        r'"publishedTimeText"\s*:\s*\{\s*"runs"\s*:\s*\[\s*\{\s*"text"\s*:\s*"(?P<text>[^"]+)"',
    ):
        m = re.search(pattern, block or "")
        if m:
            return _unescape_json_text(m.group("text"))
    return ""


def _parse_youtube_relative_publish_time(
    text: str, *, now: datetime | None = None
) -> datetime | None:
    value = _unescape_json_text(text).strip().lower()
    if not value:
        return None
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if value in {"today", "just now"}:
        return now
    m = re.search(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", value)
    if not m:
        return None
    amount = int(m.group(1))
    unit = m.group(2)
    days_by_unit: dict[str, float] = {
        "second": amount / 86400,
        "minute": amount / 1440,
        "hour": amount / 24,
        "day": float(amount),
        "week": amount * 7,
        "month": amount * 30,
        "year": amount * 365,
    }
    return now - timedelta(days=days_by_unit[unit])


def _unescape_json_text(text: str) -> str:
    try:
        return json.loads(f'"{text}"')
    except Exception:
        return text


def _fetch_text(url: str, proxy: str | None = None) -> str:
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    handlers = [urllib.request.HTTPSHandler(context=context)]
    if proxy:
        handlers.insert(0, urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    with opener.open(request, timeout=20) as response:
        return response.read().decode("utf-8", "ignore")


def _fetch_single_video_metadata(url: str) -> dict | None:
    opts: dict = {"quiet": True, "no_warnings": True, "skip_download": True}
    apply_yt_dlp_auth_options(opts)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            metadata = ydl.extract_info(url, download=False)
        if isinstance(metadata, dict):
            cooldown = float(settings.yt_dlp_short_form_min_sleep_interval_sec or 5.0)
            if cooldown > 0:
                time.sleep(cooldown)
            return metadata
        return None
    except Exception:
        logger.debug("Per-video metadata fetch failed for %s", url, exc_info=True)
        return None


if __name__ == "__main__":
    sys.exit(main())
