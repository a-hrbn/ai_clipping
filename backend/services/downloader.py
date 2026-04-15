import pathlib
import shutil
import time
from hashlib import sha256

import yt_dlp
from config import settings
from job_manager import is_job_cancelled, update_job


class DownloadUnavailableError(RuntimeError):
    pass


class DownloadCancelledError(RuntimeError):
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


def _twitch_cache_enabled(url: str) -> bool:
    return _is_twitch_url(url) and max(settings.twitch_download_cache_ttl_sec, 0) > 0


def _copy_into_job(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _resolve_downloaded_video(job_dir: pathlib.Path, preferred_path: pathlib.Path) -> pathlib.Path:
    if preferred_path.exists():
        return preferred_path

    candidates = [
        path for path in job_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".ts", ".m4v"}
        and not path.name.endswith((".part", ".ytdl"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"yt-dlp completed but no video file was found in {job_dir}"
        )

    chosen = max(candidates, key=lambda path: path.stat().st_size)
    if chosen != preferred_path:
        chosen.replace(preferred_path)
    return preferred_path


def _coerce_eta_seconds(value: float | int | None) -> int | None:
    if value is None:
        return None
    return max(int(round(float(value))), 0)


def download(url: str, job_id: str, proxy: str | None = None) -> pathlib.Path:
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "video.mp4"
    use_twitch_cache = _twitch_cache_enabled(url)

    if is_job_cancelled(job_id):
        raise DownloadCancelledError(f"Download cancelled for job {job_id}")

    if use_twitch_cache:
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        _evict_expired_cache(cache_dir)
        cache_path = _cache_path_for_url(url)
        if cache_path.exists():
            if is_job_cancelled(job_id):
                raise DownloadCancelledError(f"Download cancelled for job {job_id}")
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
        if is_job_cancelled(job_id):
            raise DownloadCancelledError(f"Download cancelled for job {job_id}")
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
                download_eta=_coerce_eta_seconds(eta),
                download_cached=False,
            )
        except KeyError:
            # Training/collection scripts may reuse downloader without creating
            # a JobStatus entry first. In that case progress updates are best-effort.
            pass

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(job_dir / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "progress_hooks": [progress_hook],
    }
    if proxy:
        ydl_opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadCancelledError:
        raise
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

    reported_paths: list[pathlib.Path] = []
    if isinstance(info, dict):
        filepath = info.get("filepath") or info.get("_filename")
        if filepath:
            reported_paths.append(pathlib.Path(str(filepath)))
        for item in info.get("requested_downloads") or []:
            candidate = item.get("filepath")
            if candidate:
                reported_paths.append(pathlib.Path(str(candidate)))

    resolved_output = next((path for path in reported_paths if path.exists()), None)
    if resolved_output is None:
        resolved_output = _resolve_downloaded_video(job_dir, output_path)
    elif resolved_output != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        resolved_output.replace(output_path)
        resolved_output = output_path

    if use_twitch_cache:
        try:
            _copy_into_job(resolved_output, cache_path)
        except OSError:
            pass

    try:
        update_job(job_id, download_speed=None, download_eta=None, download_cached=False)
    except KeyError:
        pass
    return resolved_output
