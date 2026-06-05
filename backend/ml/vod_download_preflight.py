from __future__ import annotations

import pathlib
from typing import Any, Callable, Iterable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_SOURCE_ROOT = BACKEND_ROOT / "tmp" / "storage" / "_download_cache" / "twitch"

Downloader = Callable[..., pathlib.Path]


class VodDownloadPreflightError(RuntimeError):
    def __init__(self, summary: dict[str, Any]):
        self.summary = summary
        failed = ", ".join(str(vod_id) for vod_id in summary.get("failed_vod_ids", []))
        super().__init__(f"Full-VOD preflight failed for VODs: {failed}")


def twitch_vod_url(vod_id: str) -> str:
    return f"https://www.twitch.tv/videos/{str(vod_id).strip()}"


def source_video_path(video_source_root: pathlib.Path, vod_id: str) -> pathlib.Path | None:
    if not str(vod_id or "").strip():
        return None
    candidates = [
        video_source_root / f"vod_{vod_id}.mp4",
        video_source_root / f"{vod_id}.mp4",
    ]
    for path in candidates:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def ensure_vod_sources(
    vod_ids: Iterable[str],
    *,
    video_source_root: pathlib.Path = DEFAULT_VIDEO_SOURCE_ROOT,
    download_missing: bool,
    require_all: bool,
    cache_ttl_sec: int = -1,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    unique_vod_ids = []
    for raw_vod_id in vod_ids:
        vod_id = str(raw_vod_id or "").strip()
        if vod_id and vod_id not in unique_vod_ids:
            unique_vod_ids.append(vod_id)

    by_vod: dict[str, dict[str, Any]] = {}
    for vod_id in unique_vod_ids:
        existing_path = source_video_path(video_source_root, vod_id)
        if existing_path is not None:
            by_vod[vod_id] = {
                "status": "ready",
                "source_video_path": str(existing_path),
                "downloaded": False,
                "url": twitch_vod_url(vod_id),
                "error": "",
            }
            continue

        if not download_missing:
            by_vod[vod_id] = {
                "status": "missing_source",
                "source_video_path": "",
                "downloaded": False,
                "url": twitch_vod_url(vod_id),
                "error": f"Missing full VOD source: {video_source_root / f'vod_{vod_id}.mp4'}",
            }
            continue

        try:
            downloaded_path = resolve_downloader(downloader)(
                twitch_vod_url(vod_id),
                cache_namespace="twitch",
                cache_ttl_sec=int(cache_ttl_sec),
            )
            downloaded_path = pathlib.Path(downloaded_path)
            if not downloaded_path.exists() or downloaded_path.stat().st_size <= 0:
                raise FileNotFoundError(f"Downloader returned unusable path: {downloaded_path}")
            by_vod[vod_id] = {
                "status": "downloaded",
                "source_video_path": str(downloaded_path),
                "downloaded": True,
                "url": twitch_vod_url(vod_id),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 - caller needs exact availability failure.
            by_vod[vod_id] = {
                "status": "download_failed",
                "source_video_path": "",
                "downloaded": False,
                "url": twitch_vod_url(vod_id),
                "error": str(exc),
            }

    failed_vod_ids = [
        vod_id
        for vod_id, row in by_vod.items()
        if row["status"] not in {"ready", "downloaded"}
    ]
    summary = {
        "event": "full_vod_download_preflight",
        "video_source_root": str(video_source_root),
        "download_missing": bool(download_missing),
        "require_all": bool(require_all),
        "cache_ttl_sec": int(cache_ttl_sec),
        "vod_count": len(unique_vod_ids),
        "available_count": len(unique_vod_ids) - len(failed_vod_ids),
        "downloaded_count": sum(1 for row in by_vod.values() if row["status"] == "downloaded"),
        "missing_source_count": sum(1 for row in by_vod.values() if row["status"] == "missing_source"),
        "download_failed_count": sum(1 for row in by_vod.values() if row["status"] == "download_failed"),
        "all_available": len(failed_vod_ids) == 0,
        "failed_vod_ids": failed_vod_ids,
        "by_vod": by_vod,
    }
    if require_all and failed_vod_ids:
        raise VodDownloadPreflightError(summary)
    return summary


def resolve_downloader(downloader: Downloader | None) -> Downloader:
    if downloader is not None:
        return downloader
    from services.downloader import ensure_cached_download

    return ensure_cached_download
