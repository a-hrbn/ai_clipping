from __future__ import annotations

import argparse
import csv
import json
import pathlib
import ssl
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta, timezone

import certifi

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml.row_identity import read_jsonl, write_jsonl

_TWITCH_AUTH_URL = "https://id.twitch.tv/oauth2/token"
_TWITCH_API_URL = "https://api.twitch.tv/helix"
_NEGATIVE_WINDOW_SECONDS = 45
_NEGATIVE_WINDOW_MIN_SECONDS = 18
_NEGATIVE_WINDOW_MAX_SECONDS = 75
_NEGATIVE_BUFFER_SECONDS = 20
_HARD_NEGATIVE_OFFSETS = (6.0, 12.0, 24.0, 40.0, 60.0)
_BACKGROUND_NEGATIVE_RATIO = 3.0
_BACKGROUND_STRIDE_MULTIPLIER = 1.2
_CLIP_REGION_MERGE_GAP_SECONDS = 5.0
_CLIP_REGION_AMBIGUOUS_CLIP_COUNT = 5
_CLIP_REGION_AMBIGUOUS_WINDOW_SECONDS = 95.0
_CLIP_REGION_STRONG_SHRINK_MAX_SECONDS = 75.0
_PROXY_CALIBRATION_VERSION = "vod_ratio_creator_v1"
_VOD_RELATIVE_WEAK_RATIO = 0.05
_VOD_RELATIVE_POSITIVE_RATIO = 0.15
_VOD_RELATIVE_VIRAL_RATIO = 0.40
_CREATOR_DENSITY_WINDOW_SECONDS = 90.0
_DEFAULT_SOURCE_MANIFEST = BACKEND_ROOT / "ml" / "streamer_sources.csv"
_SOURCE_MANIFEST_REQUIRED_COLUMNS = {
    "streamer",
    "platform",
    "source_url",
    "language",
    "tags",
    "max_clips",
    "training_profiles",
    "twitch_broadcaster",
}


class TwitchClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str | None = None
        self._token_expiry: float = 0.0
        context = ssl.create_default_context(cafile=certifi.where())
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context),
        )

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        query = urllib.parse.urlencode(
            {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            }
        )
        request = urllib.request.Request(
            _TWITCH_AUTH_URL,
            data=query.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "ignore").strip()
            except Exception:
                detail = ""
            if exc.code in {401, 403}:
                raise RuntimeError(
                    "Twitch auth failed. Check TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in backend/.env."
                ) from exc
            raise RuntimeError(
                f"Twitch auth request failed with HTTP {exc.code}"
                + (f": {detail}" if detail else "")
            ) from exc
        self._token = str(data["access_token"])
        self._token_expiry = time.time() + int(data.get("expires_in", 3600))
        return self._token

    def get_json(self, path: str, **params) -> dict:
        query = urllib.parse.urlencode({key: value for key, value in params.items() if value not in (None, "")})
        url = f"{_TWITCH_API_URL}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Client-Id": self.client_id,
                "Authorization": f"Bearer {self._get_token()}",
            },
        )
        with self._opener.open(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Twitch clip labels and derive canonical VOD windows.")
    parser.add_argument("--broadcaster", default="")
    parser.add_argument("--profile", default="", help="Resolve a broadcaster pool from streamer_sources.csv.")
    parser.add_argument("--date-from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--sources", default=str(_DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--cache-dir", default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_cache"))
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_clip_labels.jsonl"))
    parser.add_argument(
        "--progress-file",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_clip_label_progress.json"),
        help="Tracks already collected date ranges per broadcaster for incremental collection.",
    )
    parser.add_argument(
        "--merge-output",
        action="store_true",
        help="Merge newly collected labels into an existing output JSONL instead of replacing it.",
    )
    parser.add_argument(
        "--skip-covered",
        action="store_true",
        help="Skip already collected date ranges recorded in the progress file.",
    )
    args = parser.parse_args()

    resolved_broadcasters = _resolved_broadcasters(
        broadcaster=args.broadcaster,
        profile=args.profile,
        manifest_path=pathlib.Path(args.sources),
    )
    if not resolved_broadcasters:
        print("Either --broadcaster or --profile must resolve to at least one Twitch broadcaster.")
        return 1

    if not settings.twitch_client_id or not settings.twitch_client_secret:
        print("TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be configured.")
        return 1

    client = TwitchClient(settings.twitch_client_id, settings.twitch_client_secret)
    cache_dir = pathlib.Path(args.cache_dir)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = pathlib.Path(args.progress_file)
    requested_start = _parse_cli_date(args.date_from)
    requested_end = _parse_cli_date(args.date_to)
    if requested_end < requested_start:
        print("date-to must be on or after date-from.")
        return 1

    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    print(f"Trainiere VOD von {args.date_from} bis {args.date_to}")
    rows = list(existing_rows) if (output_path.exists() and (args.merge_output or args.skip_covered)) else []
    collected_any = False
    failures: list[str] = []
    for broadcaster in resolved_broadcasters:
        missing_ranges = [(requested_start, requested_end)]
        skipped_ranges: list[tuple[date, date]] = []
        if args.skip_covered and output_path.exists():
            progress = _read_progress_file(progress_path)
            raw_covered_ranges = _covered_date_ranges(progress, broadcaster)
            covered_ranges = _validated_covered_date_ranges(
                raw_covered_ranges,
                existing_rows,
                broadcaster,
            )
            if covered_ranges != raw_covered_ranges:
                progress = _replace_collected_ranges(progress, broadcaster, covered_ranges)
                _write_progress_file(progress_path, progress)
                print(
                    json.dumps(
                        {
                            "event": "twitch_clip_label_progress_reconciled",
                            "broadcaster": broadcaster,
                            "raw_covered_ranges": _date_range_payload(raw_covered_ranges),
                            "validated_covered_ranges": _date_range_payload(covered_ranges),
                            "existing_label_rows": len(existing_rows),
                            "existing_label_vods": _label_vod_count(existing_rows, broadcaster),
                        },
                        ensure_ascii=True,
                    )
                )
            skipped_ranges = _intersecting_date_ranges(requested_start, requested_end, covered_ranges)
            missing_ranges = _missing_date_ranges(requested_start, requested_end, covered_ranges)

        for skipped_start, skipped_end in skipped_ranges:
            print(
                f"Überspringe {broadcaster}: {skipped_start.isoformat()} bis {skipped_end.isoformat()} "
                "(bereits trainiert)"
            )

        chunk_ranges = _collection_chunks(missing_ranges)
        if args.skip_covered and not missing_ranges:
            print(
                "Überspringe Twitch-Sammlung, Bereich bereits vollständig trainiert: "
                f"{broadcaster} {args.date_from} -> {args.date_to}"
            )
            continue

        for range_start, range_end in chunk_ranges:
            print(
                "Sammle Twitch-Labels für "
                f"{broadcaster}: {range_start.isoformat()} bis {range_end.isoformat()}"
            )
            try:
                range_rows, collection_summary = collect_clip_labels_with_summary(
                    client,
                    broadcaster=broadcaster,
                    date_from=range_start.isoformat(),
                    date_to=range_end.isoformat(),
                    cache_dir=cache_dir,
                )
                print(json.dumps({"event": "twitch_clip_label_collection_summary", **collection_summary}, ensure_ascii=True))
            except Exception as exc:
                message = f"{broadcaster}: {exc}"
                failures.append(message)
                print(f"Warnung: Twitch-Sammlung übersprungen für {message}", file=sys.stderr)
                continue
            before_count = len(rows)
            rows = _merge_label_rows(rows, range_rows)
            collected_any = collected_any or bool(range_rows)
            if output_path.exists() or rows:
                write_jsonl(output_path, rows)
            progress = _read_progress_file(progress_path)
            progress = _record_collected_range(progress, broadcaster, range_start, range_end)
            _write_progress_file(progress_path, progress)
            print(
                f"Checkpoint gespeichert für {broadcaster} {range_start.isoformat()} bis {range_end.isoformat()} "
                f"({before_count} -> {len(rows)} Rows)"
            )

    if output_path.exists() and rows == existing_rows and not collected_any:
        print(f"Label-Datei unverändert: {output_path}")
    if failures:
        print(json.dumps({"event": "twitch_pool_failures", "failures": failures}, ensure_ascii=True), file=sys.stderr)
    print(f"Wrote {len(rows)} label rows to {output_path}")
    if failures and not rows and not collected_any:
        return 1
    return 0


def _resolved_broadcasters(
    *,
    broadcaster: str,
    profile: str,
    manifest_path: pathlib.Path,
) -> list[str]:
    explicit = str(broadcaster or "").strip().lower()
    broadcasters: list[str] = []
    if explicit:
        broadcasters.append(explicit)
    profile_name = str(profile or "").strip().lower()
    if profile_name:
        manifest_rows = _read_source_manifest(manifest_path)
        for row in manifest_rows:
            profiles = {
                item.strip().lower()
                for item in str(row.get("training_profiles") or "").split(",")
                if item.strip()
            }
            if profile_name not in profiles:
                continue
            candidate = str(row.get("twitch_broadcaster") or row.get("streamer") or "").strip().lower()
            if candidate:
                broadcasters.append(candidate)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in broadcasters:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


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
    _validate_source_manifest_header(reader.fieldnames or [], path)
    rows: list[dict[str, str]] = []
    for row in reader:
        normalized = {str(key or ""): str(value or "").strip() for key, value in row.items()}
        _validate_source_manifest_row(normalized, path)
        rows.append(normalized)
    return rows


def _validate_source_manifest_header(fieldnames: list[str], path: pathlib.Path) -> None:
    normalized = {str(name or "").strip() for name in fieldnames}
    missing = sorted(_SOURCE_MANIFEST_REQUIRED_COLUMNS - normalized)
    if missing:
        raise ValueError(
            f"Invalid streamer source manifest {path}: missing required columns {missing}"
        )


def _validate_source_manifest_row(row: dict[str, str], path: pathlib.Path) -> None:
    streamer = str(row.get("streamer") or "").strip()
    platform = str(row.get("platform") or "").strip().lower()
    source_url = str(row.get("source_url") or "").strip()
    training_profiles = str(row.get("training_profiles") or "").strip()
    twitch_broadcaster = str(row.get("twitch_broadcaster") or "").strip()
    if not streamer:
        raise ValueError(f"Invalid streamer source manifest {path}: row missing streamer")
    if not platform:
        raise ValueError(f"Invalid streamer source manifest {path}: row for {streamer} missing platform")
    if not training_profiles:
        raise ValueError(
            f"Invalid streamer source manifest {path}: row for {streamer} missing training_profiles"
        )
    if platform == "twitch_pool":
        if not twitch_broadcaster:
            raise ValueError(
                f"Invalid streamer source manifest {path}: twitch_pool row for {streamer} missing twitch_broadcaster"
            )
        return
    if not source_url:
        raise ValueError(
            f"Invalid streamer source manifest {path}: row for {streamer} missing source_url"
        )


def collect_clip_labels(
    client: TwitchClient,
    broadcaster: str,
    date_from: str,
    date_to: str,
    cache_dir: pathlib.Path,
) -> list[dict]:
    labels, _summary = collect_clip_labels_with_summary(
        client,
        broadcaster=broadcaster,
        date_from=date_from,
        date_to=date_to,
        cache_dir=cache_dir,
    )
    return labels


def collect_clip_labels_with_summary(
    client: TwitchClient,
    broadcaster: str,
    date_from: str,
    date_to: str,
    cache_dir: pathlib.Path,
) -> tuple[list[dict], dict[str, object]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    user = _get_user(client, broadcaster)
    broadcaster_id = str(user["id"])
    vods = _get_vods(client, broadcaster_id, date_from, date_to)
    clips = _get_clips(client, broadcaster_id, date_from, date_to)

    _write_cache(cache_dir / f"{broadcaster}_user.json", user)
    _write_cache(cache_dir / f"{broadcaster}_vods.json", {"data": vods})
    _write_cache(cache_dir / f"{broadcaster}_clips.json", {"data": clips})

    labels = build_label_rows(broadcaster, vods, clips)
    summary = _collection_input_summary(
        broadcaster=broadcaster,
        date_from=date_from,
        date_to=date_to,
        vods=vods,
        clips=clips,
        labels=labels,
    )
    _write_cache(cache_dir / f"{broadcaster}_labels.json", {"data": labels, "summary": summary})
    return labels, summary


def _collection_input_summary(
    *,
    broadcaster: str,
    date_from: str,
    date_to: str,
    vods: list[dict],
    clips: list[dict],
    labels: list[dict],
) -> dict[str, object]:
    vod_ids = {str(vod.get("id") or "").strip() for vod in vods if str(vod.get("id") or "").strip()}
    clip_video_ids = [str(clip.get("video_id") or "").strip() for clip in clips]
    clips_with_video_id = [video_id for video_id in clip_video_ids if video_id]
    clips_matching_vod = [video_id for video_id in clips_with_video_id if video_id in vod_ids]
    label_classes: dict[str, int] = {}
    for row in labels:
        label_class = str(row.get("label_class") or "unknown").strip() or "unknown"
        label_classes[label_class] = label_classes.get(label_class, 0) + 1
    labels_positive = sum(
        1
        for row in labels
        if str(row.get("label_class") or "").strip() in {"weak_positive", "positive", "viral_positive"}
    )
    labels_negative = sum(1 for row in labels if str(row.get("label_class") or "").strip() == "hard_negative")
    diagnosis = "ok"
    if clips and not vods:
        diagnosis = "clips_seen_but_no_vods"
    elif clips and not clips_with_video_id:
        diagnosis = "clips_seen_without_video_id"
    elif clips_with_video_id and not clips_matching_vod:
        diagnosis = "clip_video_ids_without_matching_vods"
    elif clips and not labels:
        diagnosis = "clips_seen_but_no_labels_built"
    return {
        "broadcaster": broadcaster,
        "date_from": date_from,
        "date_to": date_to,
        "diagnosis": diagnosis,
        "vods_seen": len(vods),
        "vod_ids_seen": len(vod_ids),
        "clips_seen": len(clips),
        "clips_with_video_id": len(clips_with_video_id),
        "clips_without_video_id": len(clips) - len(clips_with_video_id),
        "clips_with_matching_vod": len(clips_matching_vod),
        "clips_with_unmatched_video_id": len(clips_with_video_id) - len(clips_matching_vod),
        "labels_built": len(labels),
        "positive_labels_built": labels_positive,
        "negative_labels_built": labels_negative,
        "label_classes_built": dict(sorted(label_classes.items())),
    }


def build_label_rows(broadcaster: str, vods: list[dict], clips: list[dict]) -> list[dict]:
    vod_map = {str(vod["id"]): vod for vod in vods}
    clips_by_vod: dict[str, list[dict]] = {}
    for clip in clips:
        vod_id = str(clip.get("video_id") or "").strip()
        if not vod_id or vod_id not in vod_map:
            continue
        clip_row = {
            "clip_id": str(clip["id"]),
            "vod_id": vod_id,
            "vod_offset": float(clip.get("vod_offset") or 0.0),
            "duration": float(clip.get("duration") or 0.0),
            "view_count": int(clip.get("view_count") or 0),
            "creator_id": str(clip.get("creator_id") or clip.get("creator_name") or "").strip(),
            "created_at": str(clip.get("created_at") or ""),
            "url": str(clip.get("url") or ""),
        }
        clips_by_vod.setdefault(vod_id, []).append(clip_row)

    view_thresholds = _view_count_thresholds(
        [
            clip_row
            for clip_rows in clips_by_vod.values()
            for clip_row in clip_rows
        ]
    )
    vod_thresholds = _per_vod_view_count_thresholds(clips_by_vod, fallback_thresholds=view_thresholds)
    view_ratios = _per_vod_view_ratios(clips_by_vod)
    for vod_id, clip_rows in clips_by_vod.items():
        for clip_row in clip_rows:
            clip_row["view_ratio"] = view_ratios.get(str(clip_row["clip_id"]), 0.0)
            clip_row["unique_creators_in_window"] = _unique_creators_near_clip(clip_row, clip_rows)

    rows: list[dict] = []
    for vod_id, vod in vod_map.items():
        merged = _merge_clip_regions(
            clips_by_vod.get(vod_id, []),
            thresholds=vod_thresholds.get(vod_id, view_thresholds),
        )
        for item in merged:
            item.update(
                {
                    "broadcaster": broadcaster,
                    "platform_source": "twitch_clip",
                    "label_source": "twitch_clip_api",
                    "vod_url": f"https://www.twitch.tv/videos/{vod_id}",
                    "vod_duration_seconds": int(vod.get("duration_seconds") or 0),
                    "created_at": item["cluster_created_at"],
                }
            )
            rows.append(item)
        rows.extend(_sample_negative_rows(broadcaster, vod, merged))
    return sorted(rows, key=lambda row: (row["vod_id"], float(row["window_start"]), row["label_class"]))


def _get_user(client: TwitchClient, broadcaster: str) -> dict:
    data = client.get_json("/users", login=broadcaster)
    rows = data.get("data") or []
    if not rows:
        raise ValueError(f"Unknown broadcaster: {broadcaster}")
    return rows[0]


def _get_vods(client: TwitchClient, broadcaster_id: str, date_from: str, date_to: str) -> list[dict]:
    start = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) + timedelta(days=1)
    after: str | None = None
    rows: list[dict] = []
    while True:
        payload = client.get_json(
            "/videos",
            user_id=broadcaster_id,
            type="archive",
            first=100,
            after=after,
        )
        batch = payload.get("data") or []
        for vod in batch:
            created_at = _parse_iso_datetime(str(vod.get("created_at") or ""))
            if created_at is None:
                continue
            if start <= created_at < end:
                vod_row = dict(vod)
                vod_row["duration_seconds"] = _parse_twitch_duration(str(vod.get("duration") or ""))
                rows.append(vod_row)
        after = ((payload.get("pagination") or {}).get("cursor"))
        if not after or not batch:
            break
    return rows


def _get_clips(client: TwitchClient, broadcaster_id: str, date_from: str, date_to: str) -> list[dict]:
    start = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    end = (datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    after: str | None = None
    rows: list[dict] = []
    while True:
        payload = client.get_json(
            "/clips",
            broadcaster_id=broadcaster_id,
            started_at=start,
            ended_at=end,
            first=100,
            after=after,
        )
        batch = payload.get("data") or []
        rows.extend(batch)
        after = ((payload.get("pagination") or {}).get("cursor"))
        if not after or not batch:
            break
    return rows


def _merge_clip_regions(clips: list[dict], thresholds: dict[str, int] | None = None) -> list[dict]:
    if not clips:
        return []
    ordered = sorted(clips, key=lambda row: (row["vod_offset"], row["duration"], row["clip_id"]))
    regions: list[dict] = []
    current: dict | None = None
    for clip in ordered:
        start, end = _clip_window_bounds(clip)
        bucket, label_class, label_score = _classify_clip_views(
            int(clip["view_count"]),
            thresholds=thresholds,
            view_ratio=float(clip.get("view_ratio") or 0.0),
            unique_creators=int(clip.get("unique_creators_in_window") or 0),
        )
        if current is None or not _should_merge_clip_into_region(current, clip):
            current = _new_clip_region(clip, start, end, bucket, label_class, label_score)
            regions.append(current)
            continue

        current["window_start"] = round(min(float(current["window_start"]), start), 2)
        current["window_end"] = round(max(float(current["window_end"]), end), 2)
        current["cluster_last_clip_end"] = max(float(current["cluster_last_clip_end"]), _clip_merge_end(clip))
        current["clip_ids"].append(clip["clip_id"])
        current["view_ratio"] = round(max(float(current.get("view_ratio") or 0.0), float(clip.get("view_ratio") or 0.0)), 4)
        current["unique_creators_in_window"] = max(
            int(current.get("unique_creators_in_window") or 0),
            int(clip.get("unique_creators_in_window") or 0),
        )
        if int(clip["view_count"]) > int(current["view_count"]):
            current["view_count"] = int(clip["view_count"])
            current["quality_bucket"] = bucket
            current["view_bucket"] = bucket
            current["label_class"] = label_class
            current["label_score"] = label_score
            current["cluster_created_at"] = clip["created_at"]
            current["anchor_clip_id"] = clip["clip_id"]
            current["anchor_vod_offset"] = float(clip["vod_offset"])
            current["anchor_duration"] = float(clip["duration"] or 0.0)

    return [_finalize_clip_region(region, thresholds) for region in regions]


def _clip_window_bounds(clip: dict) -> tuple[float, float]:
    start = max(float(clip["vod_offset"]) - 3.0, 0.0)
    end = float(clip["vod_offset"]) + max(float(clip["duration"]), 8.0) + 3.0
    return start, end


def _clip_merge_end(clip: dict) -> float:
    return float(clip["vod_offset"]) + max(float(clip["duration"]), 8.0)


def _new_clip_region(
    clip: dict,
    start: float,
    end: float,
    bucket: str,
    label_class: str,
    label_score: float,
) -> dict:
    return {
        "vod_id": clip["vod_id"],
        "window_start": round(start, 2),
        "window_end": round(end, 2),
        "label_class": label_class,
        "label_score": label_score,
        "quality_bucket": bucket,
        "view_bucket": bucket,
        "clip_ids": [clip["clip_id"]],
        "view_count": int(clip["view_count"]),
        "view_ratio": round(float(clip.get("view_ratio") or 0.0), 4),
        "unique_creators_in_window": int(clip.get("unique_creators_in_window") or 0),
        "proxy_calibration_version": _PROXY_CALIBRATION_VERSION,
        "cluster_created_at": clip["created_at"],
        "cluster_last_clip_end": _clip_merge_end(clip),
        "anchor_clip_id": clip["clip_id"],
        "anchor_vod_offset": float(clip["vod_offset"]),
        "anchor_duration": float(clip["duration"] or 0.0),
    }


def _should_merge_clip_into_region(region: dict, clip: dict) -> bool:
    clip_start = float(clip["vod_offset"])
    return clip_start <= float(region["cluster_last_clip_end"]) + _CLIP_REGION_MERGE_GAP_SECONDS


def _finalize_clip_region(region: dict, thresholds: dict[str, int] | None = None) -> dict:
    finalized = dict(region)
    region_duration = float(finalized["window_end"]) - float(finalized["window_start"])
    if (
        len(finalized["clip_ids"]) >= _CLIP_REGION_AMBIGUOUS_CLIP_COUNT
        or region_duration > _CLIP_REGION_AMBIGUOUS_WINDOW_SECONDS
    ):
        shrunk = _shrink_strong_positive_region(finalized, thresholds=thresholds)
        if shrunk is not None:
            finalized = shrunk
        else:
            finalized["label_class"] = "ambiguous"
            finalized["label_score"] = 0.0
            finalized["quality_bucket"] = "ambiguous"
    return _public_clip_region(finalized)


def _shrink_strong_positive_region(region: dict, thresholds: dict[str, int] | None = None) -> dict | None:
    resolved = thresholds or _default_view_thresholds()
    if str(region.get("label_class") or "") not in {"positive", "viral_positive"}:
        if int(region.get("view_count") or 0) < int(resolved["positive_min"]):
            return None

    anchor_clip = {
        "vod_offset": float(region["anchor_vod_offset"]),
        "duration": float(region["anchor_duration"]),
    }
    start, end = _clip_window_bounds(anchor_clip)
    if end - start > _CLIP_REGION_STRONG_SHRINK_MAX_SECONDS:
        end = start + _CLIP_REGION_STRONG_SHRINK_MAX_SECONDS

    shrunk = dict(region)
    shrunk["window_start"] = round(start, 2)
    shrunk["window_end"] = round(max(end, start + 8.0), 2)
    shrunk["clip_ids"] = [str(region["anchor_clip_id"])]
    return shrunk


def _public_clip_region(region: dict) -> dict:
    public = dict(region)
    public.pop("cluster_last_clip_end", None)
    public.pop("anchor_clip_id", None)
    public.pop("anchor_vod_offset", None)
    public.pop("anchor_duration", None)
    return public


def _sample_negative_rows(broadcaster: str, vod: dict, positives: list[dict]) -> list[dict]:
    duration = int(vod.get("duration_seconds") or 0)
    if duration <= _NEGATIVE_WINDOW_SECONDS:
        return []
    target_durations = _negative_target_durations(positives)
    positive_like = [
        row for row in positives
        if str(row.get("label_class") or "") in {"viral_positive", "positive", "weak_positive"}
    ]
    blocked: list[tuple[float, float]] = []
    for row in positives:
        start = float(row["window_start"]) - _NEGATIVE_BUFFER_SECONDS
        end = float(row["window_end"]) + _NEGATIVE_BUFFER_SECONDS
        blocked.append((max(start, 0.0), min(end, float(duration))))
    rows: list[dict] = []
    rows.extend(_mine_boundary_negatives(broadcaster, vod, positives, blocked))
    rows.extend(_mine_interstitial_negatives(broadcaster, vod, positives, blocked))
    target_negative_rows = max(
        int(max(len(positive_like), len(positives), 1) * _BACKGROUND_NEGATIVE_RATIO),
        len(rows),
        4,
    )
    target_background_rows = max(
        int(max(len(positive_like), len(positives), 1) * 1.5),
        2,
    )
    rows.extend(
        _mine_background_negatives(
            broadcaster,
            vod,
            blocked,
            target_durations=target_durations,
            target_count=max(target_negative_rows - len(rows), target_background_rows),
        )
    )
    return _dedupe_negative_rows(rows)


def _mine_boundary_negatives(
    broadcaster: str,
    vod: dict,
    positives: list[dict],
    blocked: list[tuple[float, float]],
) -> list[dict]:
    duration = float(vod.get("duration_seconds") or 0)
    rows: list[dict] = []
    index = 0
    for positive in positives:
        start = float(positive["window_start"])
        end = float(positive["window_end"])
        target_duration = _duration_for_positive_window(positive)
        for offset in _HARD_NEGATIVE_OFFSETS:
            candidates = [
                (max(start - target_duration - offset, 0.0), max(start - offset, 0.0)),
                (min(end + offset, duration), min(end + offset + target_duration, duration)),
            ]
            for window_start, window_end in candidates:
                if window_end - window_start < target_duration * 0.8:
                    continue
                window = (window_start, window_end)
                if _overlaps_any(window, blocked):
                    continue
                rows.append(
                    {
                        "broadcaster": broadcaster,
                        "platform_source": "twitch_vod_negative",
                        "label_source": "twitch_clip_api",
                        "vod_id": str(vod["id"]),
                        "vod_url": f"https://www.twitch.tv/videos/{vod['id']}",
                        "vod_duration_seconds": int(vod.get("duration_seconds") or 0),
                        "window_start": round(window_start, 2),
                        "window_end": round(window_end, 2),
                        "label_class": "hard_negative",
                        "label_score": 0.0,
                        "quality_bucket": "negative",
                        "view_bucket": "none",
                        "clip_ids": [],
                        "view_count": 0,
                        "created_at": str(vod.get("created_at") or ""),
                        "negative_index": index,
                        "negative_source": "boundary_mined",
                    }
                )
                index += 1
    return rows


def _mine_interstitial_negatives(
    broadcaster: str,
    vod: dict,
    positives: list[dict],
    blocked: list[tuple[float, float]],
) -> list[dict]:
    if len(positives) < 2:
        return []

    ordered = sorted(positives, key=lambda row: float(row["window_start"]))
    duration = float(vod.get("duration_seconds") or 0.0)
    rows: list[dict] = []
    index = 0
    for left, right in zip(ordered, ordered[1:]):
        target_duration = _blended_gap_duration(left, right)
        gap_start = float(left["window_end"])
        gap_end = float(right["window_start"])
        gap = gap_end - gap_start
        if gap < target_duration * 1.35:
            continue
        for window_start in _interstitial_window_starts(gap_start, gap_end, target_duration):
            window_end = min(window_start + target_duration, duration)
            if window_end - window_start < target_duration * 0.8:
                continue
            window = (window_start, window_end)
            if _overlaps_any(window, blocked):
                continue
            rows.append(
                {
                    "broadcaster": broadcaster,
                    "platform_source": "twitch_vod_negative",
                    "label_source": "twitch_clip_api",
                    "vod_id": str(vod["id"]),
                    "vod_url": f"https://www.twitch.tv/videos/{vod['id']}",
                    "vod_duration_seconds": int(vod.get("duration_seconds") or 0),
                    "window_start": round(window_start, 2),
                    "window_end": round(window_end, 2),
                    "label_class": "hard_negative",
                    "label_score": 0.0,
                    "quality_bucket": "negative",
                    "view_bucket": "none",
                    "clip_ids": [],
                    "view_count": 0,
                    "created_at": str(vod.get("created_at") or ""),
                    "negative_index": index,
                    "negative_source": "between_positive_clusters",
                }
            )
            index += 1
    return rows


def _mine_background_negatives(
    broadcaster: str,
    vod: dict,
    blocked: list[tuple[float, float]],
    *,
    target_durations: list[float],
    target_count: int,
) -> list[dict]:
    duration = float(vod.get("duration_seconds") or 0.0)
    if target_count <= 0 or duration <= 0.0:
        return []
    offsets = [0.0]
    for item in target_durations[:3]:
        offsets.extend([round(item * 0.45, 2), round(item * 0.9, 2)])

    rows: list[dict] = []
    negative_index = 0
    for base_offset in offsets:
        cursor = min(max(base_offset, 0.0), max(duration - min(target_durations), 0.0))
        duration_index = negative_index
        while cursor + min(target_durations) <= duration and len(rows) < target_count:
            target_duration = _cycled_duration(target_durations, duration_index)
            window_end = min(cursor + target_duration, duration)
            if window_end - cursor < target_duration * 0.8:
                break
            window = (cursor, window_end)
            if not _overlaps_any(window, blocked):
                rows.append(
                    {
                        "broadcaster": broadcaster,
                        "platform_source": "twitch_vod_negative",
                        "label_source": "twitch_clip_api",
                        "vod_id": str(vod["id"]),
                        "vod_url": f"https://www.twitch.tv/videos/{vod['id']}",
                        "vod_duration_seconds": int(vod.get("duration_seconds") or 0),
                        "window_start": round(window[0], 2),
                        "window_end": round(window[1], 2),
                        "label_class": "background_negative",
                        "label_score": 0.0,
                        "quality_bucket": "negative",
                        "view_bucket": "none",
                        "clip_ids": [],
                        "view_count": 0,
                        "created_at": str(vod.get("created_at") or ""),
                        "negative_index": negative_index,
                        "negative_source": "background_sample",
                    }
                )
                negative_index += 1
            duration_index += 1
            cursor += max(target_duration * _BACKGROUND_STRIDE_MULTIPLIER, float(_NEGATIVE_WINDOW_MIN_SECONDS))
    return rows


def _interstitial_window_starts(gap_start: float, gap_end: float, target_duration: float) -> list[float]:
    gap = max(gap_end - gap_start, 0.0)
    if gap <= 0.0:
        return []
    slack = max(gap - target_duration, 0.0)
    starts = [gap_start + (slack / 2.0)]
    if slack >= target_duration * 0.6:
        starts.extend([
            gap_start + (slack * 0.2),
            gap_start + (slack * 0.8),
        ])
    deduped: list[float] = []
    seen: set[float] = set()
    for start in sorted(round(max(item, 0.0), 2) for item in starts):
        if start in seen:
            continue
        seen.add(start)
        deduped.append(start)
    return deduped


def _negative_target_durations(positives: list[dict]) -> list[float]:
    durations = [
        _clamp_negative_duration(float(row["window_end"]) - float(row["window_start"]))
        for row in positives
        if float(row["window_end"]) > float(row["window_start"])
    ]
    if not durations:
        return [float(_NEGATIVE_WINDOW_SECONDS)]
    return sorted(durations)


def _duration_for_positive_window(row: dict) -> float:
    return _clamp_negative_duration(float(row["window_end"]) - float(row["window_start"]))


def _blended_gap_duration(left: dict, right: dict) -> float:
    left_duration = _duration_for_positive_window(left)
    right_duration = _duration_for_positive_window(right)
    return _clamp_negative_duration((left_duration + right_duration) / 2.0)


def _clamp_negative_duration(value: float) -> float:
    return round(
        min(max(float(value), float(_NEGATIVE_WINDOW_MIN_SECONDS)), float(_NEGATIVE_WINDOW_MAX_SECONDS)),
        2,
    )


def _cycled_duration(durations: list[float], index: int) -> float:
    if not durations:
        return float(_NEGATIVE_WINDOW_SECONDS)
    return float(durations[index % len(durations)])


def _dedupe_negative_rows(rows: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for row in sorted(rows, key=lambda item: (item.get("label_class", ""), float(item["window_start"]), float(item["window_end"]))):
        key = (int(float(row["window_start"])), int(float(row["window_end"])))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _classify_clip_views(
    view_count: int,
    thresholds: dict[str, int] | None = None,
    *,
    view_ratio: float | None = None,
    unique_creators: int = 0,
) -> tuple[str, str, float]:
    resolved = thresholds or _default_view_thresholds()
    absolute_level = _view_count_level(view_count, resolved)
    if view_ratio is None:
        return _level_to_classification(absolute_level, view_count, resolved)

    ratio_level = _view_ratio_level(float(view_ratio))
    combined_level = int(round((0.70 * ratio_level) + (0.30 * absolute_level)))
    combined_level = max(absolute_level if absolute_level >= 2 else combined_level, combined_level)
    creator_bonus = _creator_density_bonus(unique_creators)
    if ratio_level <= 1 and absolute_level == 0:
        creator_bonus = min(creator_bonus, 1)
    combined_level = min(combined_level + creator_bonus, 3)
    if combined_level <= 0 and view_count < resolved["hard_negative_lt"]:
        return "weak", "hard_negative", 0.0
    if combined_level <= 0:
        return "weak", "ambiguous", 0.0
    return _level_to_classification(combined_level, view_count, resolved)


def _default_view_thresholds() -> dict[str, int]:
    return {
        "hard_negative_lt": 50,
        "weak_positive_min": 250,
        "positive_min": 1_000,
        "viral_min": 10_000,
    }


def _view_count_thresholds(clips: list[dict]) -> dict[str, int]:
    defaults = _default_view_thresholds()
    counts = sorted(
        max(int(clip.get("view_count") or 0), 0)
        for clip in clips
    )
    if not counts:
        return defaults

    hard_negative_lt = int(min(max(_percentile_value(counts, 0.15), 25), 250))
    weak_positive_min = int(max(_percentile_value(counts, 0.50), defaults["weak_positive_min"]))
    positive_min = int(max(_percentile_value(counts, 0.75), defaults["positive_min"]))
    viral_min = int(max(_percentile_value(counts, 0.90), defaults["viral_min"]))

    weak_positive_min = max(weak_positive_min, hard_negative_lt + 25)
    positive_min = max(positive_min, weak_positive_min + 100)
    viral_min = max(viral_min, positive_min + 500)

    return {
        "hard_negative_lt": hard_negative_lt,
        "weak_positive_min": weak_positive_min,
        "positive_min": positive_min,
        "viral_min": viral_min,
    }


def _per_vod_view_count_thresholds(
    clips_by_vod: dict[str, list[dict]],
    *,
    fallback_thresholds: dict[str, int],
    min_clips: int = 4,
) -> dict[str, dict[str, int]]:
    thresholds_by_vod: dict[str, dict[str, int]] = {}
    defaults = _default_view_thresholds()
    for vod_id, clips in clips_by_vod.items():
        if len(clips) < min_clips:
            thresholds_by_vod[vod_id] = dict(fallback_thresholds)
            continue
        local = _view_count_thresholds(clips)
        thresholds_by_vod[vod_id] = {
            "hard_negative_lt": min(int(local["hard_negative_lt"]), int(fallback_thresholds["hard_negative_lt"])),
            "weak_positive_min": max(int(local["weak_positive_min"]), int(defaults["weak_positive_min"])),
            "positive_min": max(int(local["positive_min"]), int(defaults["positive_min"])),
            "viral_min": max(int(local["viral_min"]), int(defaults["viral_min"])),
        }
    return thresholds_by_vod


def _per_vod_view_ratios(clips_by_vod: dict[str, list[dict]]) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for clips in clips_by_vod.values():
        max_views = max((max(int(clip.get("view_count") or 0), 0) for clip in clips), default=0)
        if max_views <= 0:
            continue
        for clip in clips:
            ratios[str(clip.get("clip_id") or "")] = round(max(int(clip.get("view_count") or 0), 0) / max_views, 4)
    return ratios


def _unique_creators_near_clip(clip: dict, clips: list[dict]) -> int:
    start = float(clip.get("vod_offset") or 0.0) - _CREATOR_DENSITY_WINDOW_SECONDS / 2.0
    end = float(clip.get("vod_offset") or 0.0) + _CREATOR_DENSITY_WINDOW_SECONDS / 2.0
    creators = {
        str(item.get("creator_id") or "").strip().lower()
        for item in clips
        if start <= float(item.get("vod_offset") or 0.0) <= end
        and str(item.get("creator_id") or "").strip()
    }
    return len(creators)


def _view_count_level(view_count: int, thresholds: dict[str, int]) -> int:
    if view_count >= thresholds["viral_min"]:
        return 3
    if view_count >= thresholds["positive_min"]:
        return 2
    if view_count >= thresholds["weak_positive_min"]:
        return 1
    return 0


def _view_ratio_level(view_ratio: float) -> int:
    if view_ratio >= _VOD_RELATIVE_VIRAL_RATIO:
        return 3
    if view_ratio >= _VOD_RELATIVE_POSITIVE_RATIO:
        return 2
    if view_ratio >= _VOD_RELATIVE_WEAK_RATIO:
        return 1
    return 0


def _creator_density_bonus(unique_creators: int) -> int:
    if unique_creators >= 5:
        return 2
    if unique_creators >= 3:
        return 1
    return 0


def _level_to_classification(
    level: int,
    view_count: int,
    thresholds: dict[str, int],
) -> tuple[str, str, float]:
    if level >= 3:
        return "viral", "viral_positive", 1.0
    if level == 2:
        return "good", "positive", 0.8
    if level == 1:
        return "okay", "weak_positive", 0.55
    if view_count < thresholds["hard_negative_lt"]:
        return "weak", "hard_negative", 0.0
    return "weak", "ambiguous", 0.0


def _percentile_value(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    position = max(0.0, min(percentile, 1.0)) * (len(values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    lower = float(values[lower_index])
    upper = float(values[upper_index])
    fraction = position - lower_index
    return int(round(lower + (upper - lower) * fraction))


def _parse_twitch_duration(raw: str) -> int:
    total = 0
    current = ""
    units = {"h": 3600, "m": 60, "s": 1}
    for char in raw.strip().lower():
        if char.isdigit():
            current += char
            continue
        if char in units and current:
            total += int(current) * units[char]
            current = ""
    return total


def _parse_iso_datetime(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _overlaps_any(window: tuple[float, float], blocked: list[tuple[float, float]]) -> bool:
    return any(window[0] < end and window[1] > start for start, end in blocked)


def _label_row_key(row: dict) -> tuple:
    clip_ids = sorted(str(item) for item in (row.get("clip_ids") or []) if str(item).strip())
    if clip_ids:
        return (
            "positive_cluster",
            str(row.get("vod_id") or ""),
            tuple(clip_ids),
        )
    return (
        "window",
        str(row.get("vod_id") or ""),
        str(row.get("label_class") or ""),
        str(row.get("negative_source") or ""),
        round(float(row.get("window_start") or 0.0), 2),
        round(float(row.get("window_end") or 0.0), 2),
    )


def _merge_label_rows(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for row in existing_rows:
        merged[_label_row_key(row)] = dict(row)
    for row in new_rows:
        merged[_label_row_key(row)] = dict(row)
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("vod_id") or ""),
            float(row.get("window_start") or 0.0),
            str(row.get("label_class") or ""),
            str(row.get("negative_source") or ""),
        ),
    )


def _parse_cli_date(raw: str) -> date:
    return date.fromisoformat(raw)


def _merge_date_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[date, date]] = [ordered[0]]
    for current_start, current_end in ordered[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end + timedelta(days=1):
            merged[-1] = (last_start, max(last_end, current_end))
            continue
        merged.append((current_start, current_end))
    return merged


def _missing_date_ranges(
    requested_start: date,
    requested_end: date,
    covered_ranges: list[tuple[date, date]],
) -> list[tuple[date, date]]:
    if requested_end < requested_start:
        return []
    current = requested_start
    missing: list[tuple[date, date]] = []
    for covered_start, covered_end in _merge_date_ranges(covered_ranges):
        if covered_end < current:
            continue
        if covered_start > requested_end:
            break
        if covered_start > current:
            missing.append((current, min(requested_end, covered_start - timedelta(days=1))))
        current = max(current, covered_end + timedelta(days=1))
        if current > requested_end:
            break
    if current <= requested_end:
        missing.append((current, requested_end))
    return missing


def _intersecting_date_ranges(
    requested_start: date,
    requested_end: date,
    covered_ranges: list[tuple[date, date]],
) -> list[tuple[date, date]]:
    intersections: list[tuple[date, date]] = []
    for covered_start, covered_end in _merge_date_ranges(covered_ranges):
        start = max(requested_start, covered_start)
        end = min(requested_end, covered_end)
        if start <= end:
            intersections.append((start, end))
    return intersections


def _validated_covered_date_ranges(
    covered_ranges: list[tuple[date, date]],
    existing_rows: list[dict],
    broadcaster: str,
) -> list[tuple[date, date]]:
    if not covered_ranges:
        return []
    label_ranges = _label_date_ranges(existing_rows, broadcaster)
    if not label_ranges:
        return []
    validated: list[tuple[date, date]] = []
    for covered_start, covered_end in _merge_date_ranges(covered_ranges):
        validated.extend(_intersecting_date_ranges(covered_start, covered_end, label_ranges))
    return _merge_date_ranges(validated)


def _label_date_ranges(rows: list[dict], broadcaster: str) -> list[tuple[date, date]]:
    dates = sorted({
        row_date
        for row in rows
        if _row_matches_broadcaster(row, broadcaster)
        for row_date in [_row_label_date(row)]
        if row_date is not None
    })
    return _merge_date_ranges([(item, item) for item in dates])


def _row_matches_broadcaster(row: dict, broadcaster: str) -> bool:
    expected = str(broadcaster or "").strip().lower()
    if not expected:
        return True
    actual = str(row.get("broadcaster") or row.get("creator_id") or row.get("streamer") or "").strip().lower()
    return not actual or actual == expected


def _row_label_date(row: dict) -> date | None:
    for key in ("created_at", "vod_created_at", "publish_timestamp", "upload_date", "date"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        if key == "publish_timestamp":
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            return datetime.fromtimestamp(value, tz=timezone.utc).date()
        text = str(raw).strip()
        if not text:
            continue
        try:
            return datetime.fromisoformat(text[:10]).date()
        except ValueError:
            try:
                parsed = _parse_iso_datetime(text)
            except ValueError:
                parsed = None
            if parsed is not None:
                return parsed.date()
    return None


def _collection_chunks(
    ranges: list[tuple[date, date]],
    days_per_chunk: int = 1,
) -> list[tuple[date, date]]:
    chunk_days = max(int(days_per_chunk), 1)
    chunks: list[tuple[date, date]] = []
    for range_start, range_end in ranges:
        current = range_start
        while current <= range_end:
            chunk_end = min(range_end, current + timedelta(days=chunk_days - 1))
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
    return chunks


def _read_progress_file(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"version": 1, "broadcasters": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "broadcasters": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "broadcasters": {}}
    broadcasters = payload.get("broadcasters")
    if not isinstance(broadcasters, dict):
        broadcasters = {}
    return {
        "version": int(payload.get("version") or 1),
        "broadcasters": broadcasters,
    }


def _write_progress_file(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _covered_date_ranges(payload: dict, broadcaster: str) -> list[tuple[date, date]]:
    rows = payload.get("broadcasters", {}).get(broadcaster.strip().lower(), [])
    covered: list[tuple[date, date]] = []
    for row in rows:
        try:
            covered.append((_parse_cli_date(str(row["date_from"])), _parse_cli_date(str(row["date_to"]))))
        except (KeyError, TypeError, ValueError):
            continue
    return _merge_date_ranges(covered)


def _record_collected_range(payload: dict, broadcaster: str, date_from: date, date_to: date) -> dict:
    normalized = broadcaster.strip().lower()
    ranges = _covered_date_ranges(payload, normalized)
    ranges.append((date_from, date_to))
    merged = _merge_date_ranges(ranges)
    updated = {
        "version": 1,
        "broadcasters": dict(payload.get("broadcasters") or {}),
    }
    updated["broadcasters"][normalized] = [
        {"date_from": start.isoformat(), "date_to": end.isoformat()}
        for start, end in merged
    ]
    return updated


def _replace_collected_ranges(payload: dict, broadcaster: str, ranges: list[tuple[date, date]]) -> dict:
    normalized = broadcaster.strip().lower()
    updated = {
        "version": 1,
        "broadcasters": dict(payload.get("broadcasters") or {}),
    }
    merged = _merge_date_ranges(ranges)
    if merged:
        updated["broadcasters"][normalized] = [
            {"date_from": start.isoformat(), "date_to": end.isoformat()}
            for start, end in merged
        ]
    else:
        updated["broadcasters"].pop(normalized, None)
    return updated


def _date_range_payload(ranges: list[tuple[date, date]]) -> list[dict[str, str]]:
    return [{"date_from": start.isoformat(), "date_to": end.isoformat()} for start, end in ranges]


def _label_vod_count(rows: list[dict], broadcaster: str) -> int:
    vods = {
        str(row.get("vod_id") or "").strip()
        for row in rows
        if _row_matches_broadcaster(row, broadcaster) and str(row.get("vod_id") or "").strip()
    }
    return len(vods)


def _write_cache(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
