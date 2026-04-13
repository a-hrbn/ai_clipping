from __future__ import annotations

import argparse
import json
import pathlib
import ssl
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

import certifi

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings

_TWITCH_AUTH_URL = "https://id.twitch.tv/oauth2/token"
_TWITCH_API_URL = "https://api.twitch.tv/helix"
_NEGATIVE_WINDOW_SECONDS = 45
_NEGATIVE_WINDOW_MIN_SECONDS = 18
_NEGATIVE_WINDOW_MAX_SECONDS = 75
_NEGATIVE_BUFFER_SECONDS = 20
_HARD_NEGATIVE_OFFSETS = (12.0, 24.0, 40.0)


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
    parser.add_argument("--broadcaster", required=True)
    parser.add_argument("--date-from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--cache-dir", default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_cache"))
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_clip_labels.jsonl"))
    args = parser.parse_args()

    if not settings.twitch_client_id or not settings.twitch_client_secret:
        print("TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be configured.")
        return 1

    client = TwitchClient(settings.twitch_client_id, settings.twitch_client_secret)
    cache_dir = pathlib.Path(args.cache_dir)
    rows = collect_clip_labels(
        client,
        broadcaster=args.broadcaster,
        date_from=args.date_from,
        date_to=args.date_to,
        cache_dir=cache_dir,
    )
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} label rows to {output_path}")
    return 0


def collect_clip_labels(
    client: TwitchClient,
    broadcaster: str,
    date_from: str,
    date_to: str,
    cache_dir: pathlib.Path,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    user = _get_user(client, broadcaster)
    broadcaster_id = str(user["id"])
    vods = _get_vods(client, broadcaster_id, date_from, date_to)
    clips = _get_clips(client, broadcaster_id, date_from, date_to)

    _write_cache(cache_dir / f"{broadcaster}_user.json", user)
    _write_cache(cache_dir / f"{broadcaster}_vods.json", {"data": vods})
    _write_cache(cache_dir / f"{broadcaster}_clips.json", {"data": clips})

    labels = build_label_rows(broadcaster, vods, clips)
    _write_cache(cache_dir / f"{broadcaster}_labels.json", {"data": labels})
    return labels


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
            "created_at": str(clip.get("created_at") or ""),
            "url": str(clip.get("url") or ""),
        }
        clips_by_vod.setdefault(vod_id, []).append(clip_row)

    rows: list[dict] = []
    for vod_id, vod in vod_map.items():
        merged = _merge_clip_regions(clips_by_vod.get(vod_id, []))
        for item in merged:
            item.update(
                {
                    "broadcaster": broadcaster,
                    "platform_source": "twitch_clip",
                    "label_source": "twitch_clip_api",
                    "vod_url": f"https://www.twitch.tv/videos/{vod_id}",
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


def _merge_clip_regions(clips: list[dict]) -> list[dict]:
    if not clips:
        return []
    ordered = sorted(clips, key=lambda row: (row["vod_offset"], row["duration"], row["clip_id"]))
    regions: list[dict] = []
    current: dict | None = None
    for clip in ordered:
        start = max(float(clip["vod_offset"]) - 3.0, 0.0)
        end = float(clip["vod_offset"]) + max(float(clip["duration"]), 8.0) + 3.0
        bucket, label_class, label_score = _classify_clip_views(int(clip["view_count"]))
        if current is None or start > float(current["window_end"]) + 12.0:
            current = {
                "vod_id": clip["vod_id"],
                "window_start": round(start, 2),
                "window_end": round(end, 2),
                "label_class": label_class,
                "label_score": label_score,
                "quality_bucket": bucket,
                "view_bucket": bucket,
                "clip_ids": [clip["clip_id"]],
                "view_count": int(clip["view_count"]),
                "cluster_created_at": clip["created_at"],
            }
            regions.append(current)
            continue

        current["window_start"] = round(min(float(current["window_start"]), start), 2)
        current["window_end"] = round(max(float(current["window_end"]), end), 2)
        current["clip_ids"].append(clip["clip_id"])
        if int(clip["view_count"]) > int(current["view_count"]):
            current["view_count"] = int(clip["view_count"])
            current["quality_bucket"] = bucket
            current["view_bucket"] = bucket
            current["label_class"] = label_class
            current["label_score"] = label_score
            current["cluster_created_at"] = clip["created_at"]

    for region in regions:
        if len(region["clip_ids"]) >= 5 or float(region["window_end"]) - float(region["window_start"]) > 95.0:
            region["label_class"] = "ambiguous"
            region["label_score"] = 0.0
            region["quality_bucket"] = "ambiguous"
    return regions


def _sample_negative_rows(broadcaster: str, vod: dict, positives: list[dict]) -> list[dict]:
    duration = int(vod.get("duration_seconds") or 0)
    if duration <= _NEGATIVE_WINDOW_SECONDS:
        return []
    target_durations = _negative_target_durations(positives)
    blocked: list[tuple[float, float]] = []
    for row in positives:
        start = float(row["window_start"]) - _NEGATIVE_BUFFER_SECONDS
        end = float(row["window_end"]) + _NEGATIVE_BUFFER_SECONDS
        blocked.append((max(start, 0.0), min(end, float(duration))))
    rows: list[dict] = []
    rows.extend(_mine_boundary_negatives(broadcaster, vod, positives, blocked))
    rows.extend(_mine_interstitial_negatives(broadcaster, vod, positives, blocked))
    cursor = 0.0
    negative_index = 0
    while cursor + min(target_durations) <= duration:
        target_duration = _cycled_duration(target_durations, negative_index)
        window_end = min(cursor + target_duration, float(duration))
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
        cursor += target_duration * 2.5
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
        center = (gap_start + gap_end) / 2.0
        window_start = max(center - (target_duration / 2.0), 0.0)
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


def _classify_clip_views(view_count: int) -> tuple[str, str, float]:
    if view_count >= 10_000:
        return "viral", "viral_positive", 1.0
    if view_count >= 1_000:
        return "good", "positive", 0.8
    if view_count >= 100:
        return "okay", "weak_positive", 0.55
    return "weak", "weak_positive", 0.35


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


def _write_cache(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
