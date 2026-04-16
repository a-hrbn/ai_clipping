from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
import socket
import ssl
import sys
from datetime import datetime, timezone
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import certifi
import yt_dlp

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import collect_viral_clips, train_ranker, train_vod_ranker

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_MIN_POSITIVE_DURATION = 10
_MAX_POSITIVE_DURATION = 75
_MIN_MEDIAN_DENOMINATOR = 1.0


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
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max(max_candidates, 1),
        "skip_download": True,
    }
    if proxy:
        ydl_opts["proxy"] = proxy
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_url, download=False)
    except Exception as exc:
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
        return entries
    return [info]


def _extract_tiktok_entries_via_bing(source_url: str, max_candidates: int, proxy: str | None = None) -> list[dict]:
    username = _extract_tiktok_username(source_url)
    query = f"site:tiktok.com/@{username}/video {username}"
    rss_url = "https://www.bing.com/search?format=rss&q=" + urllib.parse.quote(query)
    xml_text = _fetch_text(rss_url, proxy=proxy)
    root = ET.fromstring(xml_text)

    links: list[str] = []
    for item in root.findall("./channel/item"):
        link = (item.findtext("link") or "").strip()
        if not link or "/video/" not in link or "tiktok.com/@" not in link:
            continue
        links.append(link)

    unique_links: list[str] = []
    seen: set[str] = set()
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        unique_links.append(link)
        if len(unique_links) >= max_candidates:
            break

    entries: list[dict] = []
    for link in unique_links:
        metadata = _extract_single_entry_metadata(link, proxy=proxy)
        if metadata is not None:
            entries.append(metadata)
    return entries


def _extract_single_entry_metadata(url: str, proxy: str | None = None) -> dict | None:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    if proxy:
        opts["proxy"] = proxy
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception as exc:
        if _is_platform_unavailable_error(exc, url):
            raise _PlatformUnavailableError(str(exc)) from exc
        logger.debug("Metadata extraction failed for %s", url, exc_info=True)
        return None


def _normalize_entry(item: dict, seed: dict[str, str]) -> dict | None:
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
    }


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
        - 0.10 * math.log(duration)
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
