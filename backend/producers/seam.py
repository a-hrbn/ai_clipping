"""Seam currency shared by the clip-pipeline Producers.

Only the types the currently-extracted Producers need live here; the set grows
as more Producers are drained from ``pipeline.py`` (see
docs/design/clip-pipeline-producers.md). Kept deliberately import-light — nothing
here may pull a heavy leaf dependency (torch/whisper), so Producers stay
unit-testable in isolation.
"""
from __future__ import annotations

import pathlib
from typing import Callable, Protocol


class ProducerError(RuntimeError):
    """Base for failures a Producer raises. The Conductor maps these to a
    ``failed`` job status. Cancellation is NOT a ProducerError."""


# Leaf-op port: the audio extractor. The Conductor wires
# ``audio_extractor.extract``; tests pass a call-counting fake.
ExtractFn = Callable[[pathlib.Path, str], "pathlib.Path | None"]


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
