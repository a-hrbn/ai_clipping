"""VoiceEnergy — stamps a per-segment voice-energy burst score onto a transcript.

Loads the score from the transcript cache when possible; on a miss it pulls the
**shared** :class:`AudioResource` (so audio is extracted at most once across this
and transcription) and computes it, then persists it so later transcript-cache
hits keep the signal. Pure: ``load`` / ``save`` / ``compute`` are injected; it
never reads settings or touches job state. When no audio is available the
transcript is returned unchanged — matching today's no-audio warning path.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from .audio import AudioResource
from .seam import ComputeEnergyFn, LoadVoiceEnergyFn, SaveVoiceEnergyFn

if TYPE_CHECKING:
    from services.transcriber import RichSegment

logger = logging.getLogger(__name__)


class VoiceEnergy:
    def __init__(
        self,
        *,
        audio: AudioResource,
        cache_source: str | None,
        load: LoadVoiceEnergyFn,
        save: SaveVoiceEnergyFn,
        compute: ComputeEnergyFn,
    ) -> None:
        self._audio = audio
        self._cache_source = cache_source
        self._load = load
        self._save = save
        self._compute = compute

    def apply(self, transcript: list["RichSegment"]) -> list["RichSegment"]:
        """Resolve the per-segment voice-energy score and stamp it onto a new
        transcript.

        Ordering: ``load`` cache hit returns its scores → miss pulls the shared
        :class:`AudioResource` and ``compute``s, then ``save``s. The audio pull
        is memoized, so a transcription that already extracted the WAV is reused
        — no second extraction. Returns the transcript **unchanged** when no
        score can be obtained (no audio, or an empty score list).
        """
        energy_scores = self._load(self._cache_source, transcript)
        if energy_scores is None:
            audio_path = self._audio.path()
            if audio_path is None:
                logger.warning("voice_energy_skipped reason=no_audio")
                return transcript
            energy_scores = self._compute(audio_path, transcript)
            if energy_scores:
                self._save(self._cache_source, transcript, energy_scores)
        if not energy_scores:
            return transcript
        return [
            dataclasses.replace(seg, voice_energy_burst_score=energy_scores[i])
            for i, seg in enumerate(transcript)
        ]
