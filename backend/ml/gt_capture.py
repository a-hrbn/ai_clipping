"""GT-flywheel capture core — fingerprint-first capture of a single VOD.

The timing-critical half of the GT flywheel (issue #5 / PRD #3). Given one Twitch
VOD id this:

  1. Downloads the VOD media (the only retention-sensitive step).
  2. Computes the audio fingerprint **first**, validated to cover the full VOD
     duration, and persists it — so even if transcription later fails or the box
     dies mid-run we keep a usable artifact for this VOD.
  3. Extracts audio and transcribes the full VOD on the local CUDA path
     (base model, chunk-isolated subprocesses, diarization off).
  4. Persists the transcript plus a provenance record (id/date/duration/captured_at).
  5. Deletes the source media, so the 934G download-cache disk-exhaustion incident
     never repeats.

Storing the transcript — not just the fingerprint — is what keeps later repost
matches against this VOD transcript-confirmable (gold) after Twitch deletes it.

Scheduling/polling live in the daily capture job (#6); this module is the
reusable, dependency-injectable core. Heavy services (downloader, audio
extractor, transcriber) are imported lazily inside the default collaborators so
the pure decision seams stay testable without the ML stack installed.
"""
from __future__ import annotations

import json
import logging
import pathlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone

from ml import gt_transcript_artifact
from ml import match_cross_platform_audio_fingerprints as audio_matcher

logger = logging.getLogger(__name__)

FINGERPRINT_FILENAME = "fingerprint.json"
TRANSCRIPT_FILENAME = "transcript.json"
PROVENANCE_FILENAME = "provenance.json"
AUDIO_FILENAME = "audio.wav"
CHUNKS_DIRNAME = "chunks"


@dataclass
class CaptureResult:
    vod_id: str
    fingerprint_path: pathlib.Path
    transcript_path: pathlib.Path
    provenance_path: pathlib.Path
    provenance: dict
    fingerprint_duration: float
    fingerprint_items: int
    transcript_segments: int


# ---------------------------------------------------------------------------
# Pure decision seams
# ---------------------------------------------------------------------------
def validate_full_duration_fingerprint(
    num_items: int,
    duration_sec: float,
    *,
    min_items_per_sec: float = audio_matcher._MIN_FINGERPRINT_ITEMS_PER_SEC,
) -> None:
    """Reject prefix-only fingerprints.

    Chromaprint emits ~8 items/sec of decoded audio; far fewer means fpcalc only
    covered its default ~120s prefix, which would silently map the whole VOD onto
    its first two minutes. Zero/unknown duration is a no-op (nothing to check).
    """
    if duration_sec > 0 and num_items < duration_sec * min_items_per_sec:
        raise ValueError(
            f"prefix-only fingerprint: {num_items} items for {duration_sec:.0f}s "
            f"(expected >= {duration_sec * min_items_per_sec:.0f})"
        )


def build_provenance(
    vod_id: str,
    vod_date: str,
    duration_sec: float,
    captured_at: str,
) -> dict:
    """Compact, auditable provenance record persisted alongside the artifacts."""
    return {
        "vod_id": str(vod_id),
        "vod_date": str(vod_date or ""),
        "duration_sec": round(float(duration_sec), 3),
        "captured_at": str(captured_at),
    }


# ---------------------------------------------------------------------------
# Default collaborators (lazy heavy imports keep this module torch-free)
# ---------------------------------------------------------------------------
def _default_download(vod_url: str, job_id: str) -> pathlib.Path:
    from services import downloader

    # Capture only needs audio (fingerprint + transcript), so pull the audio-only
    # track: a multi-hour VOD shrinks from tens of GiB of 720p video to a few
    # hundred MB, avoiding the disk-exhaustion risk and slashing download time.
    return downloader.download_audio_only(vod_url, job_id)


def _default_extract_audio(
    video_path: pathlib.Path, job_id: str, output_path: pathlib.Path
) -> pathlib.Path:
    from services import audio_extractor

    return audio_extractor.extract(video_path, job_id, output_path=output_path)


def _default_transcribe(audio_path: pathlib.Path, vod_id: str):
    from services import transcriber

    return transcriber.transcribe_with_progress(
        audio_path,
        enable_diarization=False,
        isolate_chunks=True,
        cache_key=f"twitch_vod:{vod_id}",
    )


def _default_delete_media(paths) -> None:
    for raw in paths:
        path = pathlib.Path(raw)
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            logger.warning("gt_capture: failed to delete media %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Capture core
# ---------------------------------------------------------------------------
def capture_vod(
    vod_id: str,
    *,
    cache_dir: pathlib.Path,
    vod_date: str = "",
    captured_at: str | None = None,
    fpcalc: str | None = None,
    download=_default_download,
    extract_audio=_default_extract_audio,
    fingerprint_to_file=audio_matcher.fingerprint_media_to_file,
    transcribe=_default_transcribe,
    delete_media=_default_delete_media,
) -> CaptureResult:
    """Capture one VOD into fingerprint + transcript + provenance artifacts.

    Collaborators are injectable for testing; defaults wire the real download /
    extract / fingerprint / transcribe / cleanup path.
    """
    vod_id = str(vod_id)
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if captured_at is None:
        captured_at = datetime.now(timezone.utc).isoformat()
    if fpcalc is None:
        fpcalc = shutil.which("fpcalc")
        if fpcalc is None:
            raise RuntimeError(
                "fpcalc not found. Install chromaprint-tools to enable VOD fingerprinting."
            )

    vod_url = f"https://www.twitch.tv/videos/{vod_id}"
    job_id = f"gt_capture_{vod_id}"
    fingerprint_path = cache_dir / FINGERPRINT_FILENAME
    transcript_path = cache_dir / TRANSCRIPT_FILENAME
    provenance_path = cache_dir / PROVENANCE_FILENAME
    audio_path = cache_dir / AUDIO_FILENAME

    # 1. Download media — the only retention-sensitive step.
    logger.info("VOD %s: downloading media...", vod_id)
    video_path = pathlib.Path(download(vod_url, job_id))

    # 2. Fingerprint FIRST, full-duration-validated, so a usable artifact exists
    #    even if transcription later fails.
    logger.info("VOD %s: fingerprinting...", vod_id)
    fingerprint = fingerprint_to_file(video_path, fingerprint_path, fpcalc)
    try:
        validate_full_duration_fingerprint(len(fingerprint.values), fingerprint.duration)
    except ValueError:
        # A prefix-only fingerprint is corrupt, not "degraded but usable" — drop it.
        fingerprint_path.unlink(missing_ok=True)
        raise
    logger.info(
        "VOD %s: fingerprint persisted (%d items, %.0fs) → %s",
        vod_id,
        len(fingerprint.values),
        fingerprint.duration,
        fingerprint_path,
    )

    # 3. Extract audio + full-VOD transcript (local CUDA, base, chunk-isolated).
    logger.info("VOD %s: extracting audio + transcribing...", vod_id)
    extract_audio(video_path, job_id, audio_path)
    segments = transcribe(audio_path, vod_id)
    gt_transcript_artifact.write_transcript_cache(transcript_path, segments)
    logger.info("VOD %s: transcript persisted (%d segments)", vod_id, len(segments))

    # 4. Provenance.
    provenance = build_provenance(vod_id, vod_date, fingerprint.duration, captured_at)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=True, sort_keys=True), encoding="utf-8"
    )

    # 5. Delete source media now that both compact artifacts are persisted.
    delete_media([video_path, audio_path, cache_dir / CHUNKS_DIRNAME])
    logger.info("VOD %s: media deleted — capture complete", vod_id)

    return CaptureResult(
        vod_id=vod_id,
        fingerprint_path=fingerprint_path,
        transcript_path=transcript_path,
        provenance_path=provenance_path,
        provenance=provenance,
        fingerprint_duration=float(fingerprint.duration),
        fingerprint_items=len(fingerprint.values),
        transcript_segments=len(segments),
    )
