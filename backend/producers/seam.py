"""Seam currency shared by the clip-pipeline Producers.

Only the types the currently-extracted Producers need live here; the set grows
as more Producers are drained from ``pipeline.py`` (see
docs/design/clip-pipeline-producers.md). Kept deliberately import-light — nothing
here may pull a heavy leaf dependency (torch/whisper), so Producers stay
unit-testable in isolation.
"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:  # keep runtime import-light — RichSegment pulls torch/whisper
    from models import ClipResult
    from services.moment_finder import MomentFinderResult
    from services.transcriber import RichSegment


class ProducerError(RuntimeError):
    """Base for failures a Producer raises. The Conductor maps these to a
    ``failed`` job status. Cancellation is NOT a ProducerError."""


# Producer-local progress: the Producer reports a 0..1 fraction of its OWN work;
# the Conductor maps it onto an absolute % band. Monotonic, optional to call.
Progress = Callable[[float], None]


# Leaf-op port: the audio extractor. The Conductor wires
# ``audio_extractor.extract``; tests pass a call-counting fake.
ExtractFn = Callable[[pathlib.Path, str], "pathlib.Path | None"]

# Leaf-op ports for the transcript step. The Conductor wires the untouched
# ``transcriber`` service; tests pass canned fakes. The ``(done, total, chunk)``
# progress callbacks are remapped to a 0..1 fraction inside TranscriptResource.
TranscribeFn = Callable[..., "list[RichSegment]"]  # transcriber.transcribe_with_progress
LoadCachedFn = Callable[..., "list[RichSegment] | None"]  # transcriber.load_cached_transcript
ProbeFn = Callable[[pathlib.Path], "float | None"]  # ffprobe media duration

# Leaf-op ports for the voice-energy step. ``compute`` runs on the WAV resolved
# from the shared AudioResource; ``load``/``save`` persist the per-segment score
# under the transcript cache so later cache-hit runs keep the signal.
ComputeEnergyFn = Callable[[pathlib.Path, "list[RichSegment]"], "list[float]"]  # audio_energy.compute_segment_energy
LoadVoiceEnergyFn = Callable[["str | None", "list[RichSegment]"], "list[float] | None"]  # transcriber.load_voice_energy
SaveVoiceEnergyFn = Callable[["str | None", "list[RichSegment]", "list[float]"], None]  # transcriber.save_voice_energy

# Leaf-op ports for the chat-enrichment step. The Conductor owns the prefetch
# thread and hands the Producer a ``chat_source`` callable that blocks on the
# prefetched result OR downloads — so the Producer never sees a thread/future.
# ``apply_cached`` reuses the persisted enrichment; ``enrich`` annotates from a
# freshly downloaded chat path; ``save`` persists it under the transcript cache.
ChatSourceFn = Callable[[], "pathlib.Path | None"]  # prefetch result OR chat_signal.download_chat
ApplyChatEnrichmentFn = Callable[["str | None", "list[RichSegment]"], "list[RichSegment] | None"]  # transcriber.apply_chat_enrichment
EnrichFn = Callable[["list[RichSegment]", pathlib.Path], "list[RichSegment]"]  # chat_signal.enrich_segments
SaveChatEnrichmentFn = Callable[["str | None", "list[RichSegment]"], None]  # transcriber.save_chat_enrichment

# Leaf-op port for the moment-finding step. The Conductor wires the untouched
# ``moment_finder.find_moments`` (kwargs bound from the injected MomentParams);
# tests pass a canned ``MomentFinderResult``. Importing the real service pulls
# torch via RichSegment, so MomentFinding keeps it behind this Callable alias.
FindMomentsFn = Callable[..., "MomentFinderResult"]  # moment_finder.find_moments

# Leaf-op port for the clipping step. The Conductor wires ``clipper.cut_clips``
# (wrapped to stream partial clips via update_job); tests pass a canned
# ``list[ClipResult]``. Shape: cut(video_path, moments, job_id, *, progress_cb)
# where progress_cb receives ``(so_far, done, total)``.
CutClipsFn = Callable[..., "list[ClipResult]"]  # clipper.cut_clips


class AudioCache(Protocol):
    """The per-source audio cache. Two adapters justify the port: ``FsAudioCache``
    (prod) and the in-memory / null fakes used in tests. Implementations never
    raise on a stale/corrupt entry — they report a miss by returning ``None``."""

    def load(
        self, cache_source: str | None, *, video_path: pathlib.Path, job_id: str
    ) -> pathlib.Path | None: ...

    def store(
        self,
        cache_source: str | None,
        *,
        video_path: pathlib.Path,
        audio_path: pathlib.Path,
    ) -> None: ...
