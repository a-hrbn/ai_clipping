"""AudioResource — the lazy, memoized 16 kHz wav for one job.

The keystone Resource: transcription and voice-energy both pull it, yet audio is
extracted at most ONCE across all consumers. Pure — the extractor and cache are
injected; it never reads settings or touches job state. The Conductor owns
teardown; Producers never tear it down.
"""
from __future__ import annotations

import logging
import pathlib
import shutil

from .seam import AudioCache, ExtractFn

logger = logging.getLogger(__name__)


class AudioResource:
    def __init__(
        self,
        *,
        video_path: pathlib.Path,
        job_id: str,
        cache_source: str | None,
        extract: ExtractFn,
        cache: AudioCache,
    ) -> None:
        self._video_path = video_path
        self._job_id = job_id
        self._cache_source = cache_source
        self._extract = extract
        self._cache = cache
        self._resolved = False
        self._path: pathlib.Path | None = None

    def path(self) -> pathlib.Path | None:
        """Resolve the wav, extracting on the first call only.

        Ordering: ``cache.load`` (hit → return) → ``extract`` → ``cache.store``.
        Idempotent: N calls → at most ONE ``extract`` (memoized), so a second
        consumer pulling it never re-extracts. ``cache_source`` of ``None`` skips
        the cache and extracts once. A stale/corrupt cache entry surfaces as a
        miss (``cache.load`` returns ``None``) → extraction; an ``extract``
        failure propagates.
        """
        if self._resolved:
            return self._path
        self._resolved = True

        if self._cache_source is not None:
            cached = self._cache.load(
                self._cache_source, video_path=self._video_path, job_id=self._job_id
            )
            if cached is not None:
                self._path = cached
                return cached
            logger.info(
                "audio_resource_cache_miss job_id=%s source=%s",
                self._job_id,
                self._cache_source,
            )

        audio_path = self._extract(self._video_path, self._job_id)
        self._path = audio_path
        if self._cache_source is not None and audio_path is not None:
            self._cache.store(
                self._cache_source, video_path=self._video_path, audio_path=audio_path
            )
        return audio_path

    def materialized(self) -> pathlib.Path | None:
        """Read-only peek — the wav if it was resolved into the job dir, else
        ``None``. Lets the Conductor decide progress/teardown without forcing
        extraction."""
        return self._path

    def teardown(self) -> None:
        """Delete the materialized wav and its sibling ``chunks/`` dir.
        Idempotent; Conductor-only."""
        audio_path = self._path
        if audio_path is None:
            return
        self._path = None
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "audio_resource_cleanup_failed path=%s", audio_path, exc_info=True
            )
        chunks_dir = audio_path.parent / "chunks"
        if chunks_dir.exists():
            try:
                shutil.rmtree(chunks_dir)
            except OSError:
                logger.warning(
                    "audio_resource_chunks_cleanup_failed path=%s",
                    chunks_dir,
                    exc_info=True,
                )
