"""Filesystem adapters wiring the untouched services behind the Producer seams.

``FsAudioCache`` is the production :class:`~producers.seam.AudioCache`. It owns the
audio-cache machinery drained from ``pipeline.py``: digest keying, metadata
staleness matching, the 44-byte empty-wav guard, atomic tmp-replace writes, and
the hardlink-or-copy fallback. The cache-key and duration-probe leaf ops are
injected so this module stays free of heavy service imports.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import uuid
from dataclasses import asdict, is_dataclass
from typing import Callable

logger = logging.getLogger(__name__)

_AUDIO_CACHE_VERSION = 1

CacheKeyFn = Callable[["str | pathlib.Path | None"], "str | None"]
ProbeFn = Callable[[pathlib.Path], "float | None"]


def _metadata_matches(cached: object, expected: dict) -> bool:
    if not isinstance(cached, dict):
        return False
    for key, value in expected.items():
        if cached.get(key) != value:
            return False
    return True


def _json_safe_cache_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe_cache_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe_cache_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_cache_value(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe_cache_value(tolist())
        except (TypeError, ValueError):
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_cache_value(item())
        except (TypeError, ValueError):
            pass

    return str(value)


def _link_or_copy_cache_file(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source.resolve(strict=False) == target.resolve(strict=False):
            return target
    except OSError:
        pass
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


class FsAudioCache:
    """Production audio cache: a per-source 16 kHz wav under
    ``<storage_dir>/_audio_cache/<digest>/``, materialized into the job dir on a
    hit. Never raises on a stale/corrupt entry — it reports a miss with ``None``.
    """

    def __init__(
        self,
        storage_dir: str | pathlib.Path,
        *,
        cache_key: CacheKeyFn,
        probe: ProbeFn,
    ) -> None:
        self._storage_dir = pathlib.Path(storage_dir)
        self._cache_key = cache_key
        self._probe = probe

    def load(
        self, cache_source: str | None, *, video_path: pathlib.Path, job_id: str
    ) -> pathlib.Path | None:
        path = self._cache_path(cache_source)
        if path is None or not path.exists():
            return None
        metadata_path = self._metadata_path(path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if path.stat().st_size <= 44:
                return None
        except (OSError, json.JSONDecodeError):
            logger.warning("audio_cache_read_failed path=%s", path, exc_info=True)
            return None
        expected = self._metadata(str(cache_source), video_path)
        if not _metadata_matches(metadata, expected):
            logger.info("audio_cache_stale source=%s path=%s", cache_source, path)
            return None
        target = self._job_target(job_id)
        try:
            _link_or_copy_cache_file(path, target)
        except OSError:
            logger.warning(
                "audio_cache_link_failed source=%s path=%s", cache_source, path, exc_info=True
            )
            return None
        logger.info(
            "audio_cache_hit job_id=%s source=%s path=%s", job_id, cache_source, path
        )
        return target

    def store(
        self,
        cache_source: str | None,
        *,
        video_path: pathlib.Path,
        audio_path: pathlib.Path,
    ) -> None:
        path = self._cache_path(cache_source)
        if path is None:
            return
        try:
            if not audio_path.exists() or audio_path.stat().st_size <= 44:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            write_token = f"{os.getpid()}.{uuid.uuid4().hex}"
            tmp_path = path.with_suffix(path.suffix + f".tmp.{write_token}")
            _link_or_copy_cache_file(audio_path, tmp_path)
            tmp_path.replace(path)
            metadata = self._metadata(str(cache_source), video_path)
            metadata_path = self._metadata_path(path)
            tmp_metadata_path = metadata_path.with_suffix(
                metadata_path.suffix + f".tmp.{write_token}"
            )
            tmp_metadata_path.write_text(
                json.dumps(_json_safe_cache_value(metadata), ensure_ascii=True),
                encoding="utf-8",
            )
            tmp_metadata_path.replace(metadata_path)
            logger.info(
                "audio_cache_written source=%s path=%s size_bytes=%s",
                cache_source,
                path,
                path.stat().st_size,
            )
        except (OSError, TypeError, ValueError):
            logger.warning(
                "audio_cache_write_failed source=%s path=%s", cache_source, path, exc_info=True
            )

    def _cache_path(self, cache_source: str | None) -> pathlib.Path | None:
        digest = self._cache_key(cache_source)
        if not digest:
            return None
        return self._storage_dir / "_audio_cache" / digest / "audio.wav"

    @staticmethod
    def _metadata_path(audio_cache_path: pathlib.Path) -> pathlib.Path:
        return audio_cache_path.with_name("audio_metadata.json")

    def _metadata(self, cache_source: str, video_path: pathlib.Path | None) -> dict:
        metadata: dict[str, object] = {
            "version": _AUDIO_CACHE_VERSION,
            "source": cache_source,
            "sample_rate_hz": 16000,
            "channels": 1,
            "codec": "pcm_s16le",
        }
        if video_path is not None:
            duration_sec = self._probe(video_path)
            metadata["duration_sec"] = (
                round(duration_sec, 3) if duration_sec is not None else None
            )
        return metadata

    def _job_target(self, job_id: str) -> pathlib.Path:
        return self._storage_dir / job_id / "audio.wav"
