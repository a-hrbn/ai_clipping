from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import math
import pathlib
import random
import re
import socket
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import certifi
import yt_dlp

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import collect_viral_clips, train_ranker, train_vod_ranker
from services.yt_dlp_auth import apply_yt_dlp_auth_options

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_MIN_POSITIVE_DURATION = 10
_MAX_POSITIVE_DURATION = 75
_MIN_MEDIAN_DENOMINATOR = 1.0
_SHORT_FORM_METADATA_CACHE_VERSION = 1
YOUTUBE_RATE_LIMIT_ERROR_MARKERS = (
    "this content isn't available, try again later",
    "this content is not available, try again later",
    "rate-limited by youtube",
    "rate limited by youtube",
    "exceeding the rate limit",
    "add a delay between video requests",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover short-form links for multiple streamers, build a labeled dataset, collect features, and train the ranker.",
    )
    parser.add_argument("--sources", default=str(BACKEND_ROOT / "ml" / "streamer_sources.csv"))
    parser.add_argument("--manifest-out", default=str(BACKEND_ROOT / "ml" / "artifacts" / "discovered_short_form_manifest.csv"))
    parser.add_argument("--dataset-out", default=str(BACKEND_ROOT / "ml" / "artifacts" / "real_short_form_training_rows.jsonl"))
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--work-dir", default=str(BACKEND_ROOT / "tmp" / "short_form_training"))
    parser.add_argument("--max-duration", type=int, default=180)
    parser.add_argument("--min-duration", type=int, default=8)
    parser.add_argument("--top-fraction", type=float, default=0.3)
    parser.add_argument("--bottom-fraction", type=float, default=0.3)
    parser.add_argument("--proxy", default="", help="Optional proxy URL forwarded to yt-dlp and Bing RSS fetches.")
    parser.add_argument("--skip-platforms", default="", help="Comma-separated platforms to skip (for example: tiktok).")
    parser.add_argument("--only-streamers", default="", help="Comma-separated streamer names to keep from the sources manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Only discover and write the manifest, do not download clips or train.")
    parser.add_argument("--per-streamer", action="store_true", help="Train a separate artifact for each unique streamer in the sources CSV.")
    parser.add_argument(
        "--training-mode",
        choices=("auto", "vod", "short-form"),
        default="auto",
        help="Prefer VOD-first training when Twitch labels are available, or force a specific training path.",
    )
    parser.add_argument(
        "--twitch-labels",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_clip_labels.jsonl"),
        help="Twitch label dataset used for VOD-first training when training-mode is auto or vod.",
    )
    parser.add_argument(
        "--short-form-weight",
        type=float,
        default=0.35,
        help="Weight of short-form supplemental rows when routed into VOD-first training.",
    )
    args = parser.parse_args()
    args.skip_platforms = {
        item.strip().lower()
        for item in str(args.skip_platforms or "").split(",")
        if item.strip()
    }
    args.only_streamers = {
        item.strip().lower()
        for item in str(args.only_streamers or "").split(",")
        if item.strip()
    }

    seeds = _read_source_manifest(pathlib.Path(args.sources))
    seeds = _filter_seeds_by_streamers(seeds, args.only_streamers)
    if not seeds:
        logger.error("No streamer sources found in %s", args.sources)
        return 1

    if args.per_streamer:
        return _run_per_streamer(seeds, args)

    return _run_single_profile(seeds, args)


def _discover_seeds(
    seeds: list[dict[str, str]],
    args,
) -> list[dict[str, str]]:
    """Discover and cross-deduplicate manifest rows from a list of seeds."""
    manifest_rows: list[dict[str, str]] = []
    for seed in seeds:
        if seed["platform"].strip().lower() in args.skip_platforms:
            logger.info(
                "Discovery skipped_unavailable for %s (%s): platform explicitly skipped",
                seed["streamer"],
                seed["platform"],
            )
            continue
        try:
            discovered = _discover_seed(
                seed,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                top_fraction=args.top_fraction,
                bottom_fraction=args.bottom_fraction,
                proxy=(args.proxy or "").strip() or None,
            )
        except _PlatformUnavailableError as exc:
            logger.warning(
                "Discovery skipped_unavailable for %s (%s): %s",
                seed["streamer"],
                seed["platform"],
                exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Discovery failed for %s (%s): %s",
                seed["streamer"],
                seed["platform"],
                exc,
            )
            continue
        manifest_rows.extend(discovered)
        logger.info(
            "Discovered %d labeled clips for %s (%s)",
            len(discovered),
            seed["streamer"],
            seed["platform"],
        )

    # Deduplicate across sources: keep the first occurrence of each URL.
    seen_urls: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in manifest_rows:
        if row["url"] not in seen_urls:
            seen_urls.add(row["url"])
            deduped.append(row)
    if len(deduped) < len(manifest_rows):
        logger.info(
            "Cross-source dedup removed %d duplicate URLs (%d → %d rows)",
            len(manifest_rows) - len(deduped),
            len(manifest_rows),
            len(deduped),
        )
    return deduped


def _run_single_profile(seeds: list[dict[str, str]], args) -> int:
    manifest_rows = _discover_seeds(seeds, args)

    if not manifest_rows:
        logger.error("No short-form clips discovered from the configured sources.")
        return 1

    manifest_out = pathlib.Path(args.manifest_out)
    _write_manifest(manifest_out, manifest_rows)
    logger.info("Wrote discovery manifest to %s", manifest_out)

    if args.dry_run:
        _log_manifest_preview(manifest_rows)
        logger.info("Dry run complete. No clips were downloaded and no training was started.")
        return 0

    collected_rows = collect_viral_clips.collect_rows(
        manifest_rows,
        args.profile,
        pathlib.Path(args.work_dir),
        proxy=(args.proxy or "").strip() or None,
        skip_platforms=args.skip_platforms,
    )
    if not collected_rows:
        logger.error("Collection produced no training rows.")
        return 1

    dataset_out = pathlib.Path(args.dataset_out)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    dataset_out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in collected_rows),
        encoding="utf-8",
    )
    logger.info("Wrote collected dataset to %s", dataset_out)

    return _train_preferred_path(args, dataset_out)


def _run_per_streamer(seeds: list[dict[str, str]], args) -> int:
    """Train a separate artifact for each unique streamer in the seeds list."""
    from collections import defaultdict
    streamer_seeds: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        name = seed["streamer"].strip().lower()
        if name:
            streamer_seeds[name].append(seed)

    if not streamer_seeds:
        logger.error("No named streamers found in the sources manifest.")
        return 1

    logger.info("Per-streamer mode: will train artifacts for %s", sorted(streamer_seeds))
    artifact_dir = pathlib.Path(args.artifact_dir)
    exit_codes: list[int] = []

    for streamer, streamer_seed_list in sorted(streamer_seeds.items()):
        logger.info("=== Streamer: %s (%d sources) ===", streamer, len(streamer_seed_list))

        manifest_rows = _discover_seeds(streamer_seed_list, args)
        if not manifest_rows:
            logger.warning("No clips discovered for %s — skipping", streamer)
            exit_codes.append(1)
            continue

        manifest_out = artifact_dir / f"{streamer}_manifest.csv"
        _write_manifest(manifest_out, manifest_rows)
        logger.info("Wrote manifest for %s to %s", streamer, manifest_out)

        if args.dry_run:
            _log_manifest_preview(manifest_rows)
            continue

        collected_rows = collect_viral_clips.collect_rows(
            manifest_rows,
            streamer,
            pathlib.Path(args.work_dir) / streamer,
            proxy=(args.proxy or "").strip() or None,
            skip_platforms=args.skip_platforms,
        )
        if not collected_rows:
            logger.warning("Collection produced no rows for %s — skipping", streamer)
            exit_codes.append(1)
            continue

        dataset_out = artifact_dir / f"{streamer}_training_rows.jsonl"
        dataset_out.parent.mkdir(parents=True, exist_ok=True)
        dataset_out.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True) for row in collected_rows),
            encoding="utf-8",
        )
        logger.info("Wrote dataset for %s to %s", streamer, dataset_out)

        code = _train_preferred_path(
            args,
            dataset_out,
            profile=streamer,
            broadcaster_filter=streamer,
        )
        exit_codes.append(code)
        logger.info("Streamer %s: training exit code %d", streamer, code)

    if args.dry_run:
        logger.info("Dry run complete. No clips were downloaded and no training was started.")
        return 0

    failures = sum(1 for c in exit_codes if c != 0)
    logger.info(
        "Per-streamer training complete: %d/%d succeeded",
        len(exit_codes) - failures,
        len(exit_codes),
    )
    return 0 if failures == 0 else 1


def _read_source_manifest(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return []

    delimiter = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(lines, delimiter=delimiter)
    seeds: list[dict[str, str]] = []
    for row in reader:
        source_url = (row.get("source_url") or row.get("url") or "").strip()
        if not source_url:
            continue
        seeds.append(
            {
                "streamer": (row.get("streamer") or "").strip(),
                "platform": (row.get("platform") or "").strip().lower(),
                "source_url": source_url,
                "language": (row.get("language") or "").strip().lower(),
                "tags": (row.get("tags") or "").strip(),
                "max_clips": (row.get("max_clips") or "").strip(),
            }
        )
    return seeds


def _filter_seeds_by_streamers(
    seeds: list[dict[str, str]],
    only_streamers: set[str],
) -> list[dict[str, str]]:
    if not only_streamers:
        return seeds
    return [
        seed for seed in seeds
        if str(seed.get("streamer") or "").strip().lower() in only_streamers
    ]


def _train_preferred_path(
    args,
    short_form_dataset: pathlib.Path,
    profile: str | None = None,
    broadcaster_filter: str | None = None,
) -> int:
    effective_profile = profile or args.profile
    artifact_dir = pathlib.Path(args.artifact_dir)
    labels_path = pathlib.Path(args.twitch_labels)
    should_use_vod = args.training_mode == "vod" or (
        args.training_mode == "auto" and labels_path.exists()
    )

    if should_use_vod:
        logger.info(
            "Training route: VOD-first (%s) with short-form supplemental rows from %s",
            labels_path,
            short_form_dataset,
        )
        rows = train_vod_ranker._read_jsonl(labels_path)
        if broadcaster_filter:
            rows = _filter_twitch_labels_for_broadcaster(rows, broadcaster_filter)
        if not rows:
            if broadcaster_filter:
                logger.warning(
                    "VOD-first training requested but no Twitch labels for broadcaster=%s were found in %s",
                    broadcaster_filter,
                    labels_path,
                )
            else:
                logger.warning("VOD-first training requested but no Twitch labels were found in %s", labels_path)
        else:
            composite = train_vod_ranker.build_composite_dataset(
                rows,
                profile=effective_profile,
                work_dir=pathlib.Path(args.work_dir) / "vod_training" / effective_profile,
                supplemental_rows=train_vod_ranker._read_jsonl(short_form_dataset),
                supplemental_weight=args.short_form_weight,
            )
            dataset_out = artifact_dir / f"{effective_profile}_vod_training_rows.jsonl"
            train_vod_ranker._write_jsonl(dataset_out, composite)
            logger.info("Wrote VOD-first composite dataset to %s", dataset_out)
            logger.info("VOD-first dataset summary: %s", json.dumps(train_vod_ranker.dataset_summary(composite), ensure_ascii=True))
            return train_vod_ranker._train_models_for_dataset(
                dataset_out,
                effective_profile,
                artifact_dir,
            )

    logger.info("Training route: short-form primary (%s)", short_form_dataset)
    return train_ranker.train_from_dataset(short_form_dataset, effective_profile, artifact_dir)


def _filter_twitch_labels_for_broadcaster(rows: list[dict], broadcaster: str) -> list[dict]:
    normalized = str(broadcaster or "").strip().lower()
    if not normalized:
        return rows
    return [
        row for row in rows
        if str(row.get("broadcaster") or "").strip().lower() == normalized
    ]


def _discover_seed(
    seed: dict[str, str],
    *,
    min_duration: int,
    max_duration: int,
    top_fraction: float,
    bottom_fraction: float,
    proxy: str | None = None,
) -> list[dict[str, str]]:
    raw_entries = _extract_source_entries(seed["source_url"], _max_candidates(seed), proxy=proxy)
    candidates = [
        entry for entry in (_normalize_entry(item, seed) for item in raw_entries)
        if entry is not None and _is_short_form_candidate(entry, seed["platform"], min_duration, max_duration)
    ]
    if not candidates:
        return []

    deduped = _dedupe_candidates(candidates)
    labeled = _label_candidates(deduped, top_fraction=top_fraction, bottom_fraction=bottom_fraction)
    return [
        {
            "url": entry["url"],
            "streamer": seed["streamer"],
            "platform": seed["platform"],
            "language": seed["language"],
            "tags": seed["tags"],
            "label": str(entry["label"]),
            "quality_bucket": entry["quality_bucket"],
            "label_reason": entry["label_reason"],
            "viral_score": _format_float(entry["viral_score"]),
            "view_count": str(entry["view_count"]),
            "like_count": str(entry["like_count"]),
            "comment_count": str(entry["comment_count"]),
            "duration": str(entry["duration"]),
            "publish_timestamp": str(entry["publish_timestamp"]),
            "upload_date": entry["upload_date"],
            "title": entry["title"],
            "uploader": entry["uploader"],
            "channel": entry["channel"],
            "channel_id": entry["channel_id"],
            "description": entry["description"],
            "discovery_source_url": seed["source_url"],
            "notes": (
                "auto_discovered "
                f"viral_score={_format_float(entry['viral_score'])} "
                f"views={entry['view_count']} likes={entry['like_count']} "
                f"comments={entry['comment_count']} rank={entry['rank']}"
            ),
        }
        for entry in labeled
    ]


def _extract_source_entries(source_url: str, max_candidates: int, proxy: str | None = None) -> list[dict]:
    if source_url.strip().lower().startswith("ytsearchdate"):
        return _extract_youtube_date_search_entries(source_url, max_candidates, proxy=proxy)
    if source_url.strip().lower().startswith("tiktoksearch"):
        return _extract_tiktok_search_entries_via_bing(source_url, max_candidates, proxy=proxy)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max(max_candidates, 1),
        "skip_download": True,
    }
    apply_yt_dlp_auth_options(ydl_opts)
    if proxy:
        ydl_opts["proxy"] = proxy
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_url, download=False)
    except Exception as exc:
        _cooldown_after_short_form_yt_dlp_error(exc, stage="source_discovery", url=source_url)
        if _is_platform_unavailable_error(exc, source_url):
            raise _PlatformUnavailableError(str(exc)) from exc
        if "tiktok.com/@" in source_url:
            return _extract_tiktok_entries_via_bing(source_url, max_candidates, proxy=proxy)
        raise
    if not info:
        if "tiktok.com/@" in source_url:
            return _extract_tiktok_entries_via_bing(source_url, max_candidates, proxy=proxy)
        return []
    if isinstance(info, dict) and info.get("entries"):
        entries = [entry for entry in info["entries"] if entry]
        if not entries and "tiktok.com/@" in source_url:
            return _extract_tiktok_entries_via_bing(source_url, max_candidates, proxy=proxy)
        if _is_youtube_channel_shorts_source(source_url):
            entries = _enrich_youtube_channel_entries_with_publish_dates(entries, source_url, proxy=proxy)
        return entries
    return [info]


def _extract_youtube_date_search_entries(source_url: str, max_candidates: int, proxy: str | None = None) -> list[dict]:
    query = _extract_prefixed_search_query(source_url)
    if not query:
        return []
    search_url = (
        "https://www.youtube.com/results?"
        + urllib.parse.urlencode({"search_query": query})
        + "&sp=CAI%253D"
    )
    entries = _extract_youtube_search_page_entries(search_url, max_candidates, proxy=proxy)
    if entries:
        return entries
    return _extract_youtube_search_page_entries(
        f"ytsearch{max(max_candidates, 1)}:{query}",
        max_candidates,
        proxy=proxy,
    )


def _extract_youtube_search_page_entries(search_url: str, max_candidates: int, proxy: str | None = None) -> list[dict]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max(max_candidates, 1),
        "skip_download": True,
    }
    apply_yt_dlp_auth_options(ydl_opts)
    if proxy:
        ydl_opts["proxy"] = proxy
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
    except Exception as exc:
        _cooldown_after_short_form_yt_dlp_error(exc, stage="youtube_date_search", url=search_url)
        logger.debug("YouTube date-search extraction failed for %s", search_url, exc_info=True)
        return []
    if not info:
        return []
    if isinstance(info, dict) and info.get("entries"):
        return [entry for entry in info["entries"] if entry][:max_candidates]
    return [info]


def _is_youtube_channel_shorts_source(source_url: str) -> bool:
    parsed = urllib.parse.urlparse(str(source_url or ""))
    host = parsed.netloc.lower()
    path = parsed.path.strip("/").lower()
    return "youtube.com" in host and path.endswith("/shorts")


def _enrich_youtube_channel_entries_with_publish_dates(
    entries: list[dict],
    source_url: str,
    proxy: str | None = None,
) -> list[dict]:
    if not entries:
        return entries
    ids_needing_dates = {
        str(entry.get("id") or "").strip()
        for entry in entries
        if str(entry.get("id") or "").strip() and not _coerce_upload_date(entry) and _coerce_publish_timestamp(entry) <= 0
    }
    if not ids_needing_dates:
        return entries
    try:
        html_text = _fetch_text(source_url, proxy=proxy)
    except Exception:
        logger.debug("YouTube channel publish-date page extraction failed for %s", source_url, exc_info=True)
        html_text = ""

    publish_dates = _youtube_channel_publish_dates_from_html(html_text)
    enriched: list[dict] = []
    html_date_rows = 0
    metadata_cache_hits = 0
    metadata_fetches = 0
    metadata_failures = 0
    metadata_date_rows = 0
    for entry in entries:
        video_id = str(entry.get("id") or "").strip()
        published_at = publish_dates.get(video_id)
        if published_at is not None:
            row = dict(entry)
            if _coerce_publish_timestamp(row) <= 0:
                row["timestamp"] = int(published_at.timestamp())
            if not _coerce_upload_date(row):
                row["upload_date"] = published_at.strftime("%Y%m%d")
            html_date_rows += 1
            enriched.append(row)
            continue

        metadata_url = _metadata_url_for_entry(entry, source_url)
        if not metadata_url:
            enriched.append(entry)
            continue
        metadata, cache_status = _extract_single_entry_metadata_with_cache_status(metadata_url, proxy=proxy)
        if cache_status == "hit":
            metadata_cache_hits += 1
        elif cache_status == "fetched":
            metadata_fetches += 1
        elif cache_status == "failed":
            metadata_failures += 1
        if metadata is None:
            enriched.append(entry)
            continue
        row = _merge_flat_entry_with_metadata(entry, metadata)
        if _coerce_upload_date(row) or _coerce_publish_timestamp(row) > 0:
            metadata_date_rows += 1
        enriched.append(row)

    logger.info(
        "youtube_channel_short_metadata_enrichment source=%s rows=%s html_date_rows=%s "
        "metadata_cache_hits=%s metadata_fetches=%s metadata_failures=%s metadata_date_rows=%s",
        source_url,
        len(entries),
        html_date_rows,
        metadata_cache_hits,
        metadata_fetches,
        metadata_failures,
        metadata_date_rows,
    )
    return enriched


def _metadata_url_for_entry(entry: dict, source_url: str) -> str:
    for key in ("webpage_url", "url"):
        value = str(entry.get(key) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    video_id = str(entry.get("id") or "").strip()
    if not video_id:
        return ""
    if _is_youtube_channel_shorts_source(source_url):
        return f"https://www.youtube.com/shorts/{video_id}"
    return f"https://www.youtube.com/watch?v={video_id}"


def _merge_flat_entry_with_metadata(entry: dict, metadata: dict) -> dict:
    merged = dict(entry)
    for key, value in metadata.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    if not str(merged.get("webpage_url") or "").strip():
        merged["webpage_url"] = (
            str(metadata.get("webpage_url") or "")
            or str(entry.get("webpage_url") or "")
            or str(entry.get("url") or "")
        )
    return merged


def _youtube_channel_publish_dates_from_html(html_text: str, *, now: datetime | None = None) -> dict[str, datetime]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows: dict[str, datetime] = {}
    pattern = re.compile(r'"videoId"\s*:\s*"(?P<id>[A-Za-z0-9_-]{6,})"(?P<body>.{0,4000}?)(?="videoId"\s*:|$)', re.DOTALL)
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
        match = re.search(pattern, block or "")
        if match:
            return _unescape_json_text(match.group("text"))
    return ""


def _parse_youtube_relative_publish_time(text: str, *, now: datetime | None = None) -> datetime | None:
    value = _unescape_json_text(text).strip().lower()
    if not value:
        return None
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if value in {"today", "just now"}:
        return now
    match = re.search(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", value)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    days_by_unit = {
        "second": amount / 86400,
        "minute": amount / 1440,
        "hour": amount / 24,
        "day": amount,
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


def _extract_tiktok_entries_via_bing(source_url: str, max_candidates: int, proxy: str | None = None) -> list[dict]:
    username = _extract_tiktok_username(source_url)
    query = f"site:tiktok.com/@{username}/video {username}"
    return _extract_tiktok_bing_query_entries(query, max_candidates, proxy=proxy)


def _extract_tiktok_search_entries_via_bing(
    source_url: str,
    max_candidates: int,
    proxy: str | None = None,
) -> list[dict]:
    query = _extract_tiktok_search_query(source_url)
    if not query:
        return []
    entries: list[dict] = []
    seen_links: set[str] = set()
    for bing_query in _tiktok_keyword_bing_queries(query):
        entries.extend(
            _extract_tiktok_bing_query_entries(
                bing_query,
                max_candidates - len(entries),
                proxy=proxy,
                seen_links=seen_links,
            )
        )
        if len(entries) >= max_candidates:
            break
    return entries


def _tiktok_keyword_bing_queries(query: str) -> list[str]:
    quoted = f'"{query}"'
    return [
        f"site:tiktok.com/@ inurl:/video {quoted}",
        f"site:tiktok.com/@ {quoted}",
        f"site:tiktok.com {quoted} TikTok",
    ]


def _extract_tiktok_bing_query_entries(
    query: str,
    max_candidates: int,
    proxy: str | None = None,
    seen_links: set[str] | None = None,
) -> list[dict]:
    if max_candidates <= 0:
        return []
    seen_links = seen_links if seen_links is not None else set()
    candidates = _extract_tiktok_bing_rss_candidates(query, max_candidates, proxy=proxy, seen_links=seen_links)
    if len(candidates) < max_candidates:
        candidates.extend(
            _extract_tiktok_bing_html_candidates(
                query,
                max_candidates - len(candidates),
                proxy=proxy,
                seen_links=seen_links,
            )
        )

    entries: list[dict] = []
    for link, fallback in candidates:
        metadata = _extract_single_entry_metadata(link, proxy=proxy)
        if metadata is not None:
            entries.append(metadata)
        else:
            entries.append(fallback)
    return entries


def _extract_tiktok_bing_rss_candidates(
    query: str,
    max_candidates: int,
    proxy: str | None = None,
    seen_links: set[str] | None = None,
) -> list[tuple[str, dict]]:
    rss_url = "https://www.bing.com/search?format=rss&q=" + urllib.parse.quote(query)
    try:
        xml_text = _fetch_text(rss_url, proxy=proxy)
        root = ET.fromstring(xml_text)
    except Exception:
        logger.debug("TikTok Bing RSS extraction failed for %s", query, exc_info=True)
        return []

    seen_links = seen_links if seen_links is not None else set()
    candidates: list[tuple[str, dict]] = []
    for item in root.findall("./channel/item"):
        link = _canonical_tiktok_video_url(item.findtext("link") or "")
        if not link or link in seen_links:
            continue
        seen_links.add(link)
        candidates.append((link, _tiktok_rss_fallback_entry(link, item)))
        if len(candidates) >= max_candidates:
            break
    return candidates


def _extract_tiktok_bing_html_candidates(
    query: str,
    max_candidates: int,
    proxy: str | None = None,
    seen_links: set[str] | None = None,
) -> list[tuple[str, dict]]:
    if max_candidates <= 0:
        return []
    search_url = "https://www.bing.com/search?q=" + urllib.parse.quote(query)
    try:
        html_text = _fetch_text(search_url, proxy=proxy)
    except Exception:
        logger.debug("TikTok Bing HTML extraction failed for %s", query, exc_info=True)
        return []
    seen_links = seen_links if seen_links is not None else set()
    candidates: list[tuple[str, dict]] = []
    for link in _extract_tiktok_links_from_html(html_text):
        if link in seen_links:
            continue
        seen_links.add(link)
        candidates.append((link, _tiktok_html_fallback_entry(link, html_text)))
        if len(candidates) >= max_candidates:
            break
    return candidates


def _extract_tiktok_links_from_html(html_text: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"https?://[^\s\"'<>]+", html_text):
        link = _canonical_tiktok_video_url(raw)
        if not link or link in seen:
            continue
        seen.add(link)
        links.append(link)
    return links


def _canonical_tiktok_video_url(raw_url: str) -> str:
    value = unescape(str(raw_url or "")).strip()
    if not value:
        return ""
    parsed = urllib.parse.urlparse(value)
    if "bing.com" in parsed.netloc and parsed.path.startswith("/ck/a"):
        query = urllib.parse.parse_qs(parsed.query)
        for key in ("u", "url"):
            for encoded in query.get(key, []):
                decoded = urllib.parse.unquote(encoded)
                if decoded.startswith("a1"):
                    try:
                        payload = decoded[2:]
                        payload += "=" * (-len(payload) % 4)
                        decoded = base64.urlsafe_b64decode(payload).decode("utf-8", "ignore")
                    except Exception:
                        pass
                canonical = _canonical_tiktok_video_url(decoded)
                if canonical:
                    return canonical
    if "tiktok.com" not in parsed.netloc.lower() or "/video/" not in parsed.path:
        return ""
    return urllib.parse.urlunparse((parsed.scheme or "https", parsed.netloc, parsed.path, "", "", ""))


def _extract_tiktok_search_query(source_url: str) -> str:
    return _extract_prefixed_search_query(source_url)


def _extract_prefixed_search_query(source_url: str) -> str:
    raw = source_url.strip()
    if ":" not in raw:
        return ""
    return raw.split(":", 1)[1].strip()


def _tiktok_rss_fallback_entry(link: str, item: ET.Element) -> dict:
    timestamp, upload_date = _rss_publish_fields(
        _rss_item_text(item, "pubDate") or _rss_item_text(item, "date")
    )
    uploader = _extract_tiktok_username(link)
    return {
        "webpage_url": link,
        "url": link,
        "duration": 0,
        "view_count": 0,
        "like_count": 0,
        "comment_count": 0,
        "timestamp": timestamp,
        "upload_date": upload_date,
        "title": _rss_item_text(item, "title"),
        "description": _rss_item_text(item, "description"),
        "uploader": uploader,
        "channel": uploader,
    }


def _tiktok_html_fallback_entry(link: str, html_text: str) -> dict:
    uploader = _extract_tiktok_username(link)
    title = _extract_html_title_near_url(html_text, link) or f"TikTok video by {uploader}"
    return {
        "webpage_url": link,
        "url": link,
        "duration": 0,
        "view_count": 0,
        "like_count": 0,
        "comment_count": 0,
        "timestamp": 0,
        "upload_date": "",
        "title": title,
        "description": title,
        "uploader": uploader,
        "channel": uploader,
    }


def _extract_html_title_near_url(html_text: str, link: str) -> str:
    idx = html_text.find(link)
    if idx < 0:
        return ""
    snippet = html_text[max(0, idx - 500): idx + 500]
    match = re.search(r"<a\b[^>]*>(.*?)</a>", snippet, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(match.group(1)))).strip()


def _rss_item_text(item: ET.Element, local_name: str) -> str:
    for child in item:
        if child.tag.rsplit("}", 1)[-1] == local_name:
            return (child.text or "").strip()
    return ""


def _rss_publish_fields(raw: str) -> tuple[int, str]:
    value = raw.strip()
    if not value:
        return 0, ""
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0, ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp()), dt.strftime("%Y%m%d")


def _extract_single_entry_metadata(url: str, proxy: str | None = None) -> dict | None:
    metadata, _cache_status = _extract_single_entry_metadata_with_cache_status(url, proxy=proxy)
    return metadata


def _extract_single_entry_metadata_with_cache_status(
    url: str,
    proxy: str | None = None,
) -> tuple[dict | None, str]:
    cached = _read_short_form_metadata_cache(url)
    if cached is not None:
        return cached, "hit"

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    apply_yt_dlp_auth_options(opts)
    if proxy:
        opts["proxy"] = proxy
    outcome = "fetched"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            metadata = ydl.extract_info(url, download=False)
        if isinstance(metadata, dict):
            _write_short_form_metadata_cache(url, metadata)
            return metadata, outcome
        return None, "failed"
    except Exception as exc:
        outcome = "failed"
        _cooldown_after_short_form_yt_dlp_error(exc, stage="metadata_extraction", url=url)
        if _is_platform_unavailable_error(exc, url):
            raise _PlatformUnavailableError(str(exc)) from exc
        logger.debug("Metadata extraction failed for %s", url, exc_info=True)
        return None, outcome
    finally:
        if outcome == "fetched":
            cooldown_sec = _short_form_sleep_range_sample()
            logger.info(
                "short_form_yt_dlp_metadata_request_cooldown outcome=%s cooldown_sec=%.3f url=%s",
                outcome,
                cooldown_sec,
                url,
            )
            if cooldown_sec > 0.0:
                time.sleep(cooldown_sec)


def _short_form_metadata_cache_dir() -> pathlib.Path:
    return pathlib.Path(settings.storage_dir) / "_download_cache" / "short_form_metadata"


def _short_form_metadata_cache_path(url: str) -> pathlib.Path:
    digest = hashlib.sha256(str(url or "").strip().encode("utf-8")).hexdigest()
    return _short_form_metadata_cache_dir() / f"{digest}.json"


def _read_short_form_metadata_cache(url: str) -> dict | None:
    path = _short_form_metadata_cache_path(url)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version") or 0) != _SHORT_FORM_METADATA_CACHE_VERSION:
        return None
    if str(payload.get("url") or "") != str(url or "").strip():
        return None
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def _write_short_form_metadata_cache(url: str, metadata: dict) -> None:
    path = _short_form_metadata_cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        "version": _SHORT_FORM_METADATA_CACHE_VERSION,
        "url": str(url or "").strip(),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        logger.debug("Failed to write short-form metadata cache for %s", url, exc_info=True)


def _cooldown_after_short_form_yt_dlp_error(exc: BaseException, *, stage: str, url: str) -> None:
    cooldown_sec = _short_form_error_cooldown_sec()
    rate_limited = _is_youtube_rate_limited_error(exc)
    logger.warning(
        "short_form_yt_dlp_error_cooldown stage=%s rate_limited=%s cooldown_sec=%.3f url=%s error=%s",
        stage,
        rate_limited,
        cooldown_sec,
        url,
        _short_error(exc, limit=500),
    )
    if cooldown_sec > 0.0:
        time.sleep(cooldown_sec)


def _short_form_error_cooldown_sec() -> float:
    return _short_form_sleep_range_sample()


def _short_form_sleep_range_sample() -> float:
    try:
        min_sleep = float(getattr(settings, "yt_dlp_short_form_min_sleep_interval_sec", 5.0) or 5.0)
    except (TypeError, ValueError):
        min_sleep = 5.0
    try:
        max_sleep = float(getattr(settings, "yt_dlp_short_form_max_sleep_interval_sec", 10.0) or 10.0)
    except (TypeError, ValueError):
        max_sleep = 10.0
    min_sleep = max(min_sleep, 0.0)
    max_sleep = max(max_sleep, min_sleep)
    if max_sleep <= min_sleep:
        return min_sleep
    return random.uniform(min_sleep, max_sleep)


def _is_youtube_rate_limited_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "youtube" in text and any(marker in text for marker in YOUTUBE_RATE_LIMIT_ERROR_MARKERS)


def _short_error(exc: BaseException, *, limit: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def _normalize_entry(item: dict, seed: dict[str, str]) -> dict | None:
    if _is_live_or_upcoming_entry(item):
        return None
    url = _extract_entry_url(item, seed["platform"])
    if not url:
        return None
    return {
        "url": url,
        "duration": int(item.get("duration") or 0),
        "view_count": int(item.get("view_count") or 0),
        "like_count": int(item.get("like_count") or 0),
        "comment_count": int(item.get("comment_count") or 0),
        "publish_timestamp": _coerce_publish_timestamp(item),
        "upload_date": _coerce_upload_date(item),
        "title": str(item.get("title") or ""),
        "uploader": str(item.get("uploader") or item.get("channel") or ""),
        "channel": str(item.get("channel") or item.get("uploader") or ""),
        "channel_id": str(item.get("channel_id") or item.get("uploader_id") or ""),
        "description": str(item.get("description") or ""),
    }


def _is_live_or_upcoming_entry(item: dict) -> bool:
    live_status = str(item.get("live_status") or "").strip().lower()
    if live_status in {"is_live", "is_upcoming", "post_live"}:
        return True
    if bool(item.get("is_live")) or bool(item.get("was_live")):
        return True
    duration = int(item.get("duration") or 0)
    title = str(item.get("title") or "").strip().lower()
    if duration <= 0 and any(token in title for token in (" live ", "premiere", "upcoming")):
        return True
    return False


def _extract_entry_url(item: dict, platform: str) -> str:
    webpage_url = str(item.get("webpage_url") or item.get("url") or "").strip()
    if webpage_url.startswith("http://") or webpage_url.startswith("https://"):
        return webpage_url
    if platform == "youtube":
        video_id = str(item.get("id") or "").strip()
        if not video_id:
            return ""
        return f"https://www.youtube.com/watch?v={video_id}"
    return ""


def _extract_tiktok_username(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/")
    if path.startswith("@"):
        return path.split("/", 1)[0][1:]
    return path


def _fetch_text(url: str, proxy: str | None = None) -> str:
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy else urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )
    with opener.open(request, timeout=20) as response:
        return response.read().decode("utf-8", "ignore")


class _PlatformUnavailableError(RuntimeError):
    pass


def _is_platform_unavailable_error(exc: Exception, url: str) -> bool:
    if "tiktok.com" not in str(url or "").lower():
        return False
    if isinstance(exc, socket.gaierror):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "temporary failure in name resolution",
            "name or service not known",
            "nodename nor servname provided",
            "failed to resolve",
            "dns",
        )
    )


def _is_short_form_candidate(entry: dict, platform: str, min_duration: int, max_duration: int) -> bool:
    duration = int(entry.get("duration") or 0)
    if duration and (duration < min_duration or duration > max_duration):
        return False
    url = entry["url"].lower()
    if platform == "youtube":
        return "/shorts/" in url or "watch?v=" in url
    if platform == "tiktok":
        return "tiktok.com/" in url
    return True


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    enriched = [{**item, "viral_score": _viral_score(item)} for item in candidates]
    for item in sorted(enriched, key=lambda row: (row["viral_score"], row["view_count"], row["url"]), reverse=True):
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        result.append(item)
    return result


def _label_candidates(candidates: list[dict], *, top_fraction: float, bottom_fraction: float) -> list[dict]:
    if not candidates:
        return []
    baselines = _candidate_baselines(candidates)
    enriched = [
        {
            **item,
            "viral_score": float(item.get("viral_score", _viral_score(item, baselines))),
            "positive_eligible": _is_positive_candidate(item),
        }
        for item in candidates
    ]
    ordered = sorted(enriched, key=lambda row: (row["viral_score"], row["view_count"], row["url"]), reverse=True)
    total = len(ordered)
    eligible_positive = [row for row in ordered if row["positive_eligible"]]
    positive_n = max(1, min(len(eligible_positive) or total, int(round(total * top_fraction))))
    negative_n = max(1, min(total - positive_n, int(round(total * bottom_fraction))))

    labeled: list[dict] = []
    positive_pool = eligible_positive or ordered
    for idx, item in enumerate(positive_pool[:positive_n], start=1):
        labeled.append({
            **item,
            "label": 1,
            "rank": idx,
            "quality_bucket": "positive",
            "label_reason": "top_ranked_by_viral_score",
        })

    negative_slice = ordered[-negative_n:] if negative_n > 0 else []
    for offset, item in enumerate(negative_slice, start=1):
        if any(existing["url"] == item["url"] for existing in labeled):
            continue
        labeled.append({
            **item,
            "label": 0,
            "rank": total - negative_n + offset,
            "quality_bucket": "negative",
            "label_reason": "bottom_ranked_by_viral_score",
        })

    return labeled


def _candidate_baselines(candidates: list[dict]) -> dict[str, float]:
    views = sorted(max(int(item.get("view_count") or 0), 0) for item in candidates)
    likes = sorted(max(int(item.get("like_count") or 0), 0) for item in candidates)
    comments = sorted(max(int(item.get("comment_count") or 0), 0) for item in candidates)
    return {
        "median_views": max(_median(views), _MIN_MEDIAN_DENOMINATOR),
        "median_likes": max(_median(likes), _MIN_MEDIAN_DENOMINATOR),
        "median_comments": max(_median(comments), _MIN_MEDIAN_DENOMINATOR),
    }


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2.0


def _is_positive_candidate(entry: dict) -> bool:
    duration = int(entry.get("duration") or 0)
    if duration <= 0:
        return True
    return _MIN_POSITIVE_DURATION <= duration <= _MAX_POSITIVE_DURATION


def _viral_score(entry: dict, baselines: dict[str, float] | None = None) -> float:
    baselines = baselines or {
        "median_views": 1.0,
        "median_likes": 1.0,
        "median_comments": 1.0,
    }
    views = max(int(entry.get("view_count") or 0), 0)
    likes = max(int(entry.get("like_count") or 0), 0)
    comments = max(int(entry.get("comment_count") or 0), 0)
    duration = max(int(entry.get("duration") or 0), 1)
    publish_timestamp = _coerce_publish_timestamp(entry)
    age_days = max((datetime.now(timezone.utc).timestamp() - publish_timestamp) / 86400.0, 0.0) if publish_timestamp else 30.0
    age_boost = 1.0 / max(age_days + 3.0, 3.0)
    relative_views = views / max(float(baselines["median_views"]), _MIN_MEDIAN_DENOMINATOR)
    relative_likes = likes / max(float(baselines["median_likes"]), _MIN_MEDIAN_DENOMINATOR)
    relative_comments = comments / max(float(baselines["median_comments"]), _MIN_MEDIAN_DENOMINATOR)
    like_rate = likes / max(views, 1)
    comment_rate = comments / max(views, 1)
    duration_penalty = 0.0
    if duration > _MAX_POSITIVE_DURATION:
        duration_penalty = 0.75 + 0.02 * (duration - _MAX_POSITIVE_DURATION)
    score = (
        0.75 * math.log1p(relative_views)
        + 1.15 * math.log1p(relative_likes)
        + 1.35 * math.log1p(relative_comments)
        + 0.9 * math.log1p(views * age_boost)
        + 18.0 * like_rate
        + 32.0 * comment_rate
        - 0.10 * math.log1p(duration)
        - duration_penalty
    )
    return round(score, 6)


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
            return 0
    return 0


def _coerce_upload_date(item: dict) -> str:
    value = str(item.get("upload_date") or "").strip()
    if len(value) == 8 and value.isdigit():
        return value
    return ""


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _max_candidates(seed: dict[str, str]) -> int:
    raw = str(seed.get("max_clips") or "").strip()
    if raw.isdigit():
        return max(int(raw) * 4, 8)
    return 24


def _write_manifest(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "url",
        "streamer",
        "platform",
        "language",
        "tags",
        "label",
        "quality_bucket",
        "label_reason",
        "viral_score",
        "view_count",
        "like_count",
        "comment_count",
        "duration",
        "publish_timestamp",
        "upload_date",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _log_manifest_preview(rows: list[dict[str, str]], limit: int = 12) -> None:
    positives = sum(1 for row in rows if str(row.get("label")) == "1")
    negatives = sum(1 for row in rows if str(row.get("label")) == "0")
    logger.info("Preview: %d manifest rows (%d positive, %d negative)", len(rows), positives, negatives)
    for row in rows[:limit]:
        logger.info(
            "  %s | %s | %s | %s",
            row.get("streamer", ""),
            row.get("platform", ""),
            row.get("label", ""),
            row.get("url", ""),
        )


if __name__ == "__main__":
    raise SystemExit(main())
