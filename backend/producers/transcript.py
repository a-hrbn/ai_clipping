"""TranscriptResource — the lazy, memoized transcript for one job.

Resolves cheapest-first: on a transcript cache hit the shared AudioResource is
never pulled, so no audio is extracted; only a miss pulls audio and transcribes.
The media-duration probe is folded in here — a missing/failed probe is a local
miss, not a silent cache no-op leaking to the Conductor as an ordering rule.
Pure: probe / load_cached / transcribe are injected; it never reads settings or
touches job state. Progress flows out as a 0..1 fraction (the Conductor maps it
onto an absolute % band); the leaf ops' ``(done, total, chunk)`` callbacks are
remapped to that fraction internally.
"""
from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

from .audio import AudioResource
from .seam import LoadCachedFn, ProbeFn, Progress, TranscribeFn

if TYPE_CHECKING:
    from services.transcriber import RichSegment

logger = logging.getLogger(__name__)


def _noop_progress(_fraction: float) -> None:
    pass


class TranscriptResource:
    def __init__(
        self,
        *,
        audio: AudioResource,
        cache_source: str | None,
        video_path: pathlib.Path,
        isolate_chunks: bool,
        enable_diarization: bool,
        load_cached: LoadCachedFn,
        transcribe: TranscribeFn,
        probe: ProbeFn,
    ) -> None:
        self._audio = audio
        self._cache_source = cache_source
        self._video_path = video_path
        self._isolate_chunks = isolate_chunks
        self._enable_diarization = enable_diarization
        self._load_cached = load_cached
        self._transcribe = transcribe
        self._probe = probe
        self._resolved = False
        self._segments: list["RichSegment"] = []
        self._from_cache = False

    def segments(self, *, progress: Progress = _noop_progress) -> list["RichSegment"]:
        """Resolve the transcript, cheapest-first.

        On a cache hit the shared :class:`AudioResource` is never pulled (so no
        audio is extracted — the locality win); a miss pulls it then transcribes.
        Idempotent: N calls → at most ONE resolution (memoized). The leaf ops'
        ``(done, total, chunk)`` progress is remapped to a 0..1 fraction here.
        Errors: a ``transcribe`` failure propagates; ``load_cached`` returning
        ``None`` (or a missing probe duration) is a MISS, not an error.
        """
        if self._resolved:
            return self._segments
        self._resolved = True

        def on_chunk(done: int, total: int, _chunk_path=None) -> None:
            if total > 0:
                progress(min(max(done / total, 0.0), 1.0))

        cached = self._load_from_cache(on_chunk)
        if cached is not None:
            self._segments = cached
            self._from_cache = True
            logger.info(
                "transcript_resource_cache_hit source=%s segments=%s",
                self._cache_source,
                len(cached),
            )
            return cached

        audio_path = self._audio.path()
        transcript = self._transcribe(
            audio_path,
            on_chunk,
            cache_key=self._cache_source,
            isolate_chunks=self._isolate_chunks,
            enable_diarization=self._enable_diarization,
        )
        self._segments = transcript
        return transcript

    def from_cache(self) -> bool:
        """Whether the resolved transcript came from the cache (audio untouched).
        ``False`` until :meth:`segments` has run."""
        return self._from_cache

    def _load_from_cache(self, on_chunk) -> "list[RichSegment] | None":
        """Probe duration then consult the transcript cache. A missing
        ``cache_source`` or a failed probe short-circuits to a miss (``None``)
        without consulting the cache — keeping the no-duration decision local."""
        if self._cache_source is None:
            return None
        duration_sec = self._probe(self._video_path)
        if duration_sec is None:
            return None
        return self._load_cached(
            self._cache_source,
            expected_duration_sec=duration_sec,
            progress_cb=on_chunk,
        )
