import pathlib
import shutil
import time
from hashlib import sha256

import yt_dlp
from config import settings
from job_manager import update_job


class DownloadUnavailableError(RuntimeError):
    pass


def _is_twitch_url(url: str) -> bool:
    value = url.strip().lower()
    return "twitch.tv/" in value


def _cache_dir() -> pathlib.Path:
    return pathlib.Path(settings.storage_dir) / "_download_cache" / "twitch"


def _cache_path_for_url(url: str) -> pathlib.Path:
    digest = sha256(url.strip().encode("utf-8")).hexdigest()
    return _cache_dir() / f"{digest}.mp4"


def _evict_expired_cache(cache_dir: pathlib.Path) -> None:
    ttl = max(settings.twitch_download_cache_ttl_sec, 0)
    if ttl <= 0 or not cache_dir.exists():
        return

    cutoff = time.time() - ttl
    for path in cache_dir.glob("*.mp4"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _copy_into_job(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def download(url: str, job_id: str, proxy: str | None = None) -> pathlib.Path:
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "video.mp4"
    use_twitch_cache = _is_twitch_url(url)

    if use_twitch_cache:
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        _evict_expired_cache(cache_dir)
        cache_path = _cache_path_for_url(url)
        if cache_path.exists():
            _copy_into_job(cache_path, output_path)
            try:
                update_job(
                    job_id,
                    progress_pct=19,
                    download_speed=None,
                    download_eta=0,
                    download_cached=True,
                )
            except KeyError:
                pass
            return output_path

    def progress_hook(d: dict) -> None:
        if d["status"] != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes", 0)
        speed = d.get("speed")  # bytes/s, may be None
        eta = d.get("eta")      # seconds, may be None

        if total and total > 0:
            # Map download progress to 5–19% of overall pipeline
            download_fraction = downloaded / total
            pct = 5 + int(download_fraction * 14)
        else:
            pct = 5

        try:
            update_job(
                job_id,
                progress_pct=pct,
                download_speed=speed,
                download_eta=eta,
                download_cached=False,
            )
        except KeyError:
            # Training/collection scripts may reuse downloader without creating
            # a JobStatus entry first. In that case progress updates are best-effort.
            pass

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_path),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "progress_hooks": [progress_hook],
    }
    if proxy:
        ydl_opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        message = str(exc).lower()
        if "tiktok.com" in url.lower() and any(
            token in message
            for token in (
                "temporary failure in name resolution",
                "name or service not known",
                "nodename nor servname provided",
                "failed to resolve",
                "dns",
            )
        ):
            raise DownloadUnavailableError(str(exc)) from exc
        raise

    if use_twitch_cache:
        try:
            _copy_into_job(output_path, cache_path)
        except OSError:
            pass

    try:
        update_job(job_id, download_speed=None, download_eta=None, download_cached=False)
    except KeyError:
        pass
    return output_path
