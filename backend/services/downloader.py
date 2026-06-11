import pathlib
import json
import shutil
import time
from hashlib import sha256
import os
import re
import threading
import uuid

from config import settings
from job_manager import is_job_cancelled, update_job
from services.yt_dlp_auth import apply_yt_dlp_auth_options


def __getattr__(name: str):
    # Lazily expose yt_dlp as a module attribute so importing this module stays
    # cheap, while callers and tests can still reference `downloader.yt_dlp`.
    if name == "yt_dlp":
        import yt_dlp

        return yt_dlp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class DownloadUnavailableError(RuntimeError):
    pass


class DownloadPausedError(RuntimeError):
    pass


class DownloadCancelledError(RuntimeError):
    pass


_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: dict[str, threading.Lock] = {}
_VIDEO_FILE_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".ts", ".m4v"}
_CACHE_METADATA_VERSION = 1


def _is_twitch_url(url: str) -> bool:
    value = url.strip().lower()
    return "twitch.tv/" in value


def _aria2c_available() -> bool:
    return shutil.which("aria2c") is not None


def _format_selector(*, max_height: int = 0) -> str:
    height = max(int(max_height or 0), 0)
    if height <= 0:
        return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}][ext=mp4]/"
        f"best[height<={height}]"
    )


def _twitch_max_height_for_job(job_id: str) -> int:
    if not str(job_id or "").startswith("twitch_vod"):
        try:
            return max(int(settings.twitch_download_max_height or 0), 0)
        except (TypeError, ValueError):
            return 0
    try:
        return max(int(settings.training_vod_max_height or 0), 0)
    except (TypeError, ValueError):
        return 0


def _base_ydl_opts(job_dir: pathlib.Path, *, max_height: int = 0) -> dict[str, object]:
    return apply_yt_dlp_auth_options({
        "format": _format_selector(max_height=max_height),
        "outtmpl": str(job_dir / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    })


def _apply_short_form_sleep_opts(opts: dict[str, object]) -> None:
    sleep_requests = max(float(settings.yt_dlp_short_form_sleep_requests_sec or 0.0), 0.0)
    min_sleep = max(float(settings.yt_dlp_short_form_min_sleep_interval_sec or 0.0), 0.0)
    max_sleep = max(float(settings.yt_dlp_short_form_max_sleep_interval_sec or 0.0), 0.0)
    if sleep_requests > 0:
        opts["sleep_interval_requests"] = sleep_requests
    if min_sleep > 0:
        opts["sleep_interval"] = min_sleep
        opts["max_sleep_interval"] = max(max_sleep, min_sleep)


def _twitch_download_ydl_opts(job_dir: pathlib.Path, *, max_height: int = 0) -> dict[str, object]:
    opts = _base_ydl_opts(job_dir, max_height=max_height)
    opts.update(
        {
            "concurrent_fragment_downloads": max(int(settings.twitch_download_concurrent_fragments or 0), 1),
            "fragment_retries": 10,
            "skip_unavailable_fragments": True,
            "keep_fragments": False,
        }
    )
    if settings.twitch_download_use_aria2c and _aria2c_available():
        split = max(int(settings.twitch_download_aria2_connections or 0), 1)
        opts["external_downloader"] = "aria2c"
        opts["external_downloader_args"] = [
            "--split",
            str(split),
            "--max-connection-per-server",
            str(split),
            "--min-split-size",
            "1M",
            "--file-allocation=none",
            "--summary-interval=0",
            "--console-log-level=error",
            # Silence the realtime "[#hash MiB/MiB(%)]" readout that otherwise
            # floods orchestrator.log with thousands of lines; errors still log.
            "--show-console-readout=false",
        ]
    return opts


def _cache_dir(namespace: str) -> pathlib.Path:
    return pathlib.Path(settings.storage_dir) / "_download_cache" / namespace


def _twitch_vod_id(url: str) -> str | None:
    match = re.search(r"twitch\.tv/videos/(\d+)", url.strip().lower())
    return match.group(1) if match else None


def _cache_path_for_url(url: str, *, namespace: str = "twitch") -> pathlib.Path:
    vod_id = _twitch_vod_id(url)
    if namespace == "twitch" and vod_id:
        return _cache_dir(namespace) / f"vod_{vod_id}.mp4"
    digest = sha256(url.strip().encode("utf-8")).hexdigest()
    return _cache_dir(namespace) / f"{digest}.mp4"


def _cache_metadata_path(cache_path: pathlib.Path) -> pathlib.Path:
    return cache_path.with_suffix(f"{cache_path.suffix}.metadata.json")


def _cache_metadata_payload(url: str, namespace: str, cache_path: pathlib.Path) -> dict[str, object]:
    stat = cache_path.stat()
    payload: dict[str, object] = {
        "version": _CACHE_METADATA_VERSION,
        "namespace": namespace,
        "url": url.strip(),
        "size": int(stat.st_size),
        "cache_file": cache_path.name,
        "complete": True,
        "cached_at": time.time(),
    }
    vod_id = _twitch_vod_id(url)
    if vod_id:
        payload["vod_id"] = vod_id
    return payload


def _write_cache_metadata(url: str, namespace: str, cache_path: pathlib.Path) -> None:
    metadata_path = _cache_metadata_path(cache_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(_cache_metadata_payload(url, namespace, cache_path), ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(metadata_path)


def _cache_entry_usable(cache_path: pathlib.Path, url: str, namespace: str) -> bool:
    try:
        if cache_path.stat().st_size <= 0:
            return False
    except OSError:
        return False
    metadata_path = _cache_metadata_path(cache_path)
    if not metadata_path.exists():
        # Backward-compatible legacy cache hit. New writes create metadata.
        return True
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    if int(metadata.get("version") or 0) != _CACHE_METADATA_VERSION:
        return False
    if str(metadata.get("namespace") or "") != namespace:
        return False
    if bool(metadata.get("complete")) is not True:
        return False
    try:
        if int(metadata.get("size") or -1) != int(cache_path.stat().st_size):
            return False
    except OSError:
        return False
    if namespace == "twitch":
        return str(metadata.get("vod_id") or "") == str(_twitch_vod_id(url) or "")
    return str(metadata.get("url") or "") == url.strip()


def _evict_expired_cache(cache_dir: pathlib.Path, ttl: int) -> None:
    if ttl <= 0 or not cache_dir.exists():
        return

    cutoff = time.time() - ttl
    for path in cache_dir.glob("*.mp4"):
        try:
            metadata_path = _cache_metadata_path(path)
            reference_time = path.stat().st_mtime
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(metadata, dict):
                        reference_time = float(metadata.get("cached_at") or reference_time)
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            if reference_time < cutoff:
                path.unlink(missing_ok=True)
                _cache_metadata_path(path).unlink(missing_ok=True)
        except OSError:
            continue


def _twitch_cache_enabled(url: str, job_id: str | None = None) -> bool:
    return _is_twitch_url(url) and int(settings.twitch_download_cache_ttl_sec or 0) != 0


def _cache_lock(cache_path: pathlib.Path) -> threading.Lock:
    key = str(cache_path)
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def _copy_into_job(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _link_or_copy_into_job(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _cleanup_redundant_job_download_artifacts(job_dir: pathlib.Path, keep_path: pathlib.Path) -> None:
    keep_path = keep_path.resolve(strict=False)
    for path in job_dir.iterdir():
        if not path.is_file() or path.resolve(strict=False) == keep_path:
            continue
        name = path.name
        suffix = path.suffix.lower()
        is_video_sibling = name.startswith("video") and suffix in _VIDEO_FILE_SUFFIXES
        is_transient_fragment = name.startswith("video") and (
            ".part" in name
            or name.endswith(".aria2")
            or name.endswith(".ytdl")
        )
        if not is_video_sibling and not is_transient_fragment:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _store_in_cache(source: pathlib.Path, cache_path: pathlib.Path, *, url: str, namespace: str) -> pathlib.Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    needs_refresh = not cache_path.exists()
    if not needs_refresh:
        try:
            needs_refresh = (
                cache_path.stat().st_size != source.stat().st_size
                or not _cache_entry_usable(cache_path, url, namespace)
            )
        except OSError:
            needs_refresh = True
    if needs_refresh:
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        tmp_path.unlink(missing_ok=True)
        try:
            # Hardlink is instant and shares storage with the job copy — the
            # cache->job direction already links. A 10GB shutil.copy2 over the
            # drvfs bind mount (~10MB/s) costs 15-20 min per VOD otherwise.
            os.link(source, tmp_path)
        except OSError:
            shutil.copy2(source, tmp_path)
        tmp_path.replace(cache_path)
    _write_cache_metadata(url, namespace, cache_path)
    return cache_path


def _resolve_downloaded_video(job_dir: pathlib.Path, preferred_path: pathlib.Path) -> pathlib.Path:
    if preferred_path.exists():
        return preferred_path

    candidates = [
        path for path in job_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _VIDEO_FILE_SUFFIXES
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


def _is_transient_network_error(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "temporary failure in name resolution",
            "name or service not known",
            "nodename nor servname provided",
            "failed to resolve",
            "dns",
            "network is unreachable",
            "no route to host",
            "connection refused",
            "connection reset",
            "connection reset by peer",
            "connection aborted",
            "remote end closed connection",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "proxy error",
            "unable to connect",
        )
    )


def _ffmpeg_downloader_exit_code(message: str) -> int | None:
    """Return ffmpeg's exit code if this is a yt-dlp ffmpeg-fragment-downloader
    failure, else None.

    Twitch VODs are HLS-only, so yt-dlp downloads fragments through ffmpeg. The
    ffmpeg downloader aborts the *whole* download on the first hard segment error
    (a Twitch HTTP 403 surfaces as ``exited with code 8``; a crash as ``-11``)
    instead of retrying per-fragment — so we fall back to yt-dlp's native HLS
    downloader, which honors ``fragment_retries`` / ``skip_unavailable_fragments``.
    """
    match = re.search(r"ffmpeg exited with code (-?\d+)", (message or "").lower())
    return int(match.group(1)) if match else None


def _extract_yt_dlp_info(ydl_opts: dict[str, object], url: str) -> dict | object:
    import yt_dlp  # deferred: heavy import, only needed when actually downloading

    ydl = yt_dlp.YoutubeDL(ydl_opts)
    primary_exc: BaseException | None = None
    try:
        return ydl.extract_info(url, download=True)
    except BaseException as exc:
        primary_exc = exc
        raise
    finally:
        try:
            ydl.close()
        except Exception:
            if primary_exc is None:
                raise


def _download_uncached(
    url: str,
    job_id: str,
    proxy: str | None = None,
    *,
    ydl_opt_overrides: dict[str, object] | None = None,
    disable_external_downloader: bool = False,
) -> pathlib.Path:
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "video.mp4"

    if is_job_cancelled(job_id):
        raise DownloadCancelledError(f"Download cancelled for job {job_id}")

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

    max_height = _twitch_max_height_for_job(job_id)
    ydl_opts = (
        _twitch_download_ydl_opts(job_dir, max_height=max_height)
        if _is_twitch_url(url)
        else _base_ydl_opts(job_dir, max_height=max_height)
    )
    if ydl_opt_overrides:
        ydl_opts.update(ydl_opt_overrides)
    if disable_external_downloader:
        ydl_opts.pop("external_downloader", None)
        ydl_opts.pop("external_downloader_args", None)
    ydl_opts["progress_hooks"] = [progress_hook]
    if proxy:
        ydl_opts["proxy"] = proxy

    try:
        info = _extract_yt_dlp_info(ydl_opts, url)
    except DownloadCancelledError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if _is_transient_network_error(message):
            raise DownloadPausedError(str(exc)) from exc
        ffmpeg_exit_code = _ffmpeg_downloader_exit_code(message)
        if (
            _is_twitch_url(url)
            and not disable_external_downloader
            and ffmpeg_exit_code is not None
        ):
            _cleanup_redundant_job_download_artifacts(job_dir, output_path)
            retry_overrides = dict(ydl_opt_overrides or {})
            # A -11 segfault is format-specific, so also collapse to a single
            # muxed format. Other ffmpeg-downloader failures (e.g. code 8 on a
            # Twitch HLS 403) just need the native HLS downloader — keep the
            # height-capped format so the retry doesn't balloon to a "best" pull.
            if ffmpeg_exit_code == -11:
                retry_overrides["format"] = "best[ext=mp4]/best"
            return _download_uncached(
                url,
                job_id,
                proxy=proxy,
                ydl_opt_overrides=retry_overrides,
                disable_external_downloader=True,
            )
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

    try:
        update_job(job_id, download_speed=None, download_eta=None, download_cached=False)
    except KeyError:
        pass
    return resolved_output


def ensure_cached_download(
    url: str,
    *,
    cache_namespace: str,
    cache_ttl_sec: int = -1,
    proxy: str | None = None,
) -> pathlib.Path:
    cache_dir = _cache_dir(cache_namespace)
    cache_path = _cache_path_for_url(url, namespace=cache_namespace)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with _cache_lock(cache_path):
        _evict_expired_cache(cache_dir, int(cache_ttl_sec or 0))
        if cache_path.exists() and _cache_entry_usable(cache_path, url, cache_namespace):
            cache_path.touch()
            return cache_path

        temp_job_id = f"_cache_fill_{cache_namespace}_{uuid.uuid4().hex[:12]}"
        temp_job_dir = pathlib.Path(settings.storage_dir) / temp_job_id
        ydl_opt_overrides: dict[str, object] = {}
        if cache_namespace == "short_form":
            _apply_short_form_sleep_opts(ydl_opt_overrides)
            # Retry transient HTTP errors (incl. 403) up to 3 times before raising.
            ydl_opt_overrides.setdefault("retries", 3)
            ydl_opt_overrides.setdefault("fragment_retries", 3)
        downloaded_path = _download_uncached(
            url,
            temp_job_id,
            proxy=proxy,
            ydl_opt_overrides=ydl_opt_overrides,
        )
        try:
            _store_in_cache(downloaded_path, cache_path, url=url, namespace=cache_namespace)
        finally:
            shutil.rmtree(temp_job_dir, ignore_errors=True)
        return cache_path


def download(url: str, job_id: str, proxy: str | None = None) -> pathlib.Path:
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "video.mp4"

    if _twitch_cache_enabled(url, job_id):
        cache_dir = _cache_dir("twitch")
        cache_path = _cache_path_for_url(url, namespace="twitch")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with _cache_lock(cache_path):
            _evict_expired_cache(cache_dir, int(settings.twitch_download_cache_ttl_sec or 0))
            if cache_path.exists() and _cache_entry_usable(cache_path, url, "twitch"):
                if is_job_cancelled(job_id):
                    raise DownloadCancelledError(f"Download cancelled for job {job_id}")
                _link_or_copy_into_job(cache_path, output_path)
                _cleanup_redundant_job_download_artifacts(job_dir, output_path)
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

            resolved_output = _download_uncached(url, job_id, proxy=proxy)
            try:
                _store_in_cache(resolved_output, cache_path, url=url, namespace="twitch")
                resolved_output = _link_or_copy_into_job(cache_path, output_path)
            except OSError:
                pass
            _cleanup_redundant_job_download_artifacts(job_dir, output_path)
            return resolved_output

    return _download_uncached(url, job_id, proxy=proxy)


def download_audio_only(url: str, job_id: str, proxy: str | None = None) -> pathlib.Path:
    """Download only the audio track of ``url`` (no video).

    GT capture needs only audio (fingerprint + full-VOD transcript), so a
    multi-hour Twitch VOD drops from tens of GiB of 720p video to a few hundred
    MB of audio — far less disk, bandwidth and wall-clock, and it sidesteps the
    934G download-cache disk-exhaustion risk. Goes straight through
    ``_download_uncached`` (no shared serving cache) so the audio-only artifact
    never collides with the 720p video that the frontend serving path caches
    under the same VOD URL.
    """
    return _download_uncached(
        url,
        job_id,
        proxy=proxy,
        ydl_opt_overrides={"format": "bestaudio/best"},
    )


def download_range(
    url: str,
    job_id: str,
    *,
    start_sec: float,
    end_sec: float,
    proxy: str | None = None,
) -> pathlib.Path:
    from yt_dlp.utils import download_range_func  # deferred: heavy import

    start = max(float(start_sec), 0.0)
    end = max(float(end_sec), start + 1.0)
    return _download_uncached(
        url,
        job_id,
        proxy=proxy,
        disable_external_downloader=True,
        ydl_opt_overrides={
            "download_ranges": download_range_func(None, [(start, end)]),
            "force_keyframes_at_cuts": False,
        },
    )
