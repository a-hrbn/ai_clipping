from __future__ import annotations

import json
import logging
import pathlib
import re
import subprocess
import threading
from dataclasses import dataclass, field

import whisper
from config import settings

logger = logging.getLogger(__name__)

# Chunk length in seconds.
# These chunks are non-overlapping; timestamp offsets must match that exactly.
CHUNK_SECONDS = 600  # 10 minutes

# Module-level Whisper singleton — loaded once on first use, reused across jobs.
_whisper_model: whisper.Whisper | None = None
_whisper_runtime_logged = False
_whisper_lock = threading.Lock()


def _model_device_name(model: whisper.Whisper) -> str:
    try:
        parameter = next(model.parameters())
        return str(parameter.device)
    except Exception:
        device = getattr(model, "device", None)
        return str(device or "unknown")


def runtime_summary() -> dict[str, object]:
    if _whisper_model is None:
        return {
            "whisper_model": settings.whisper_model,
            "device": "unloaded",
            "precision": "unknown",
        }
    device_name = _model_device_name(_whisper_model)
    precision = "fp16" if device_name.startswith("cuda") else "fp32"
    return {
        "whisper_model": settings.whisper_model,
        "device": device_name,
        "precision": precision,
    }


def _get_model() -> whisper.Whisper:
    global _whisper_model, _whisper_runtime_logged
    # Double-checked locking: the load is a multi-second disk+CUDA op; without
    # the lock two concurrent first-transcriptions would each download and load
    # the model into memory, then silently race to bind _whisper_model.
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                logger.info("transcriber_init whisper_model=%s", settings.whisper_model)
                whisper_cache_dir = pathlib.Path(settings.storage_dir).parent / "cache" / "whisper"
                whisper_cache_dir.mkdir(parents=True, exist_ok=True)
                _whisper_model = whisper.load_model(
                    settings.whisper_model,
                    download_root=str(whisper_cache_dir),
                )
    if not _whisper_runtime_logged:
        summary = runtime_summary()
        logger.info(
            "transcriber_runtime whisper_model=%s device=%s precision=%s",
            summary["whisper_model"],
            summary["device"],
            summary["precision"],
        )
        _whisper_runtime_logged = True
    return _whisper_model

# Legacy type alias for backward compatibility
Segment = tuple[float, float, str]  # (start, end, text)

# Emotion detection patterns
_LAUGHTER_RE = re.compile(r"\b(ha(ha)+|hehe|lol|lmao|xd|kekw|lulw)\b", re.IGNORECASE)
_EXCLAMATION_RE = re.compile(r"[!]{1,}$|(!)\s")
_QUESTION_RE = re.compile(r"\?")


@dataclass
class RichSegment:
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"
    emotions: list[str] = field(default_factory=list)  # "exclamation", "question", "laughter"
    words_per_second: float = 0.0
    # Numerical audio features (populated by audio_features.enrich_segments)
    laughter_prob: float = 0.0
    max_rms_db: float = 0.0
    mean_rms_db: float = 0.0   # average over non-silent frames (0.0 = not set)
    max_pitch_hz: float = 0.0
    max_onset_rate: float = 0.0
    # Composite virality score (populated by scorer.score_segments)
    hype_score: float = 0.0
    # Emotional arc score of the best candidate window this segment belongs to
    # (populated by scorer.extract_candidates)
    arc_score: float = 0.0
    # Visual features (populated by video_features.enrich_segments)
    max_motion_score: float = 0.0    # peak optical flow magnitude (0–1, normalised)
    mean_face_size: float = 0.0      # largest face bbox area / frame area
    max_face_count: int = 0          # peak simultaneous faces in segment
    max_face_reaction: float = 0.0   # peak DeepFace surprise+fear+disgust prob
    has_scene_change: bool = False   # any hard cut / overlay event in segment
    visual_activity: float = 0.0    # composite visual activity level
    # Chat-replay signal (populated by chat_analyzer.enrich_segments)
    chat_spike_score: float = 0.0   # normalised chat message-rate spike intensity (0–1)

    def to_tuple(self) -> Segment:
        return (self.start, self.end, self.text)


def transcribe(audio_path: pathlib.Path) -> list[RichSegment]:
    return transcribe_with_progress(audio_path)


def transcribe_with_progress(
    audio_path: pathlib.Path,
    progress_cb=None,
) -> list[RichSegment]:
    chunks_dir = audio_path.parent / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    # Split audio into fixed, non-overlapping chunks via ffmpeg CLI.
    # Use explicit PCM re-encoding (not -c copy): the segment muxer does not
    # write correct WAV headers when stream-copying, which produces truncated
    # or malformed chunks that cause "Broken pipe" when Whisper reads them.
    chunk_pattern = str(chunks_dir / "chunk_%03d.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(CHUNK_SECONDS),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_pattern,
        ],
        check=True,
        capture_output=True,
    )

    chunk_files = sorted(chunks_dir.glob("chunk_*.wav"))
    model = _get_model()

    raw_segments: list[RichSegment] = []
    total_chunks = len(chunk_files)

    for idx, chunk_file in enumerate(chunk_files):
        time_offset = idx * CHUNK_SECONDS
        result = model.transcribe(str(chunk_file), word_timestamps=True, verbose=False)

        for seg in result.get("segments", []):
            start = seg["start"] + time_offset
            end = seg["end"] + time_offset
            text = seg["text"].strip()
            if not text:
                continue

            wps = _calc_tempo(seg, time_offset)
            emotions = _tag_emotions(text)

            raw_segments.append(RichSegment(
                start=start,
                end=end,
                text=text,
                emotions=emotions,
                words_per_second=wps,
            ))

        if progress_cb is not None and total_chunks > 0:
            progress_cb(idx + 1, total_chunks, chunk_file)

    deduped = _deduplicate(raw_segments)

    # Speaker diarization (optional — requires HF token)
    if settings.hf_token:
        deduped = _diarize(audio_path, deduped)

    return deduped


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------

def _calc_tempo(seg: dict, time_offset: float) -> float:
    """Words per second using the full segment duration as the denominator."""
    words = seg.get("words", [])
    segment_duration = seg["end"] - seg["start"]
    if words:
        word_count = len(words)
    else:
        word_count = len(seg["text"].split())

    # Sub-500 ms segments are too noisy for tempo scoring and frequently
    # represent single-word interjections rather than sustained excited speech.
    if segment_duration < 0.5:
        return 0.0

    return round(word_count / segment_duration, 2) if segment_duration > 0 else 0.0


# ---------------------------------------------------------------------------
# Emotional markers
# ---------------------------------------------------------------------------

def _tag_emotions(text: str) -> list[str]:
    tags: list[str] = []
    if _LAUGHTER_RE.search(text):
        tags.append("laughter")
    if _EXCLAMATION_RE.search(text):
        tags.append("exclamation")
    if _QUESTION_RE.search(text):
        tags.append("question")
    return tags


# ---------------------------------------------------------------------------
# Speaker diarization via pyannote
# ---------------------------------------------------------------------------

def _diarize(audio_path: pathlib.Path, segments: list[RichSegment]) -> list[RichSegment]:
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        return segments  # pyannote not installed — skip silently

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=settings.hf_token,
    )

    diarization = pipeline(str(audio_path))

    # Build speaker timeline: list of (start, end, speaker_label)
    speaker_timeline: list[tuple[float, float, str]] = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    # Determine which pyannote label maps to the streamer
    # Heuristic: the speaker with the most total speaking time is the streamer
    speaker_duration: dict[str, float] = {}
    for s, e, spk in speaker_timeline:
        speaker_duration[spk] = speaker_duration.get(spk, 0.0) + (e - s)
    streamer_label = max(speaker_duration, key=lambda k: speaker_duration[k]) if speaker_duration else None

    for seg in segments:
        dominant = _dominant_speaker(seg.start, seg.end, speaker_timeline)
        if dominant is None:
            seg.speaker = "UNKNOWN"
        elif dominant == streamer_label:
            seg.speaker = "STREAMER"
        else:
            # Stable guest numbering: GUEST_1, GUEST_2, ...
            guests = sorted(k for k in speaker_duration if k != streamer_label)
            idx = guests.index(dominant) + 1 if dominant in guests else 0
            seg.speaker = f"GUEST_{idx}"

    return segments


def _dominant_speaker(
    seg_start: float,
    seg_end: float,
    timeline: list[tuple[float, float, str]],
) -> str | None:
    """Return the speaker with the most overlap with [seg_start, seg_end]."""
    overlap: dict[str, float] = {}
    for t_start, t_end, spk in timeline:
        o = min(seg_end, t_end) - max(seg_start, t_start)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0.0) + o
    return max(overlap, key=lambda k: overlap[k]) if overlap else None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(segments: list[RichSegment]) -> list[RichSegment]:
    if not segments:
        return []
    result = [segments[0]]
    for seg in segments[1:]:
        prev_end = result[-1].end
        if seg.start < prev_end - 1.0:
            continue
        result.append(seg)
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_transcript(
    segments: list[RichSegment] | list[Segment],
    use_seconds: bool = False,
) -> str:
    """Format segments as a transcript string.

    Args:
        segments: List of RichSegment or legacy (start, end, text) tuples.
        use_seconds: When True, emit timestamps as raw float seconds so the
            LLM can copy them directly into JSON without any conversion.
            When False (default), emit HH:MM:SS for human readability.
    """
    lines = []
    for seg in segments:
        if isinstance(seg, RichSegment):
            if use_seconds:
                ts = f"[{seg.start:.2f} - {seg.end:.2f}]"
            else:
                ts = f"[{_fmt(seg.start)} - {_fmt(seg.end)}]"
            parts = [ts]
            if seg.speaker != "UNKNOWN":
                parts.append(f"[{seg.speaker}]")
            if seg.emotions:
                parts.append(f"[{','.join(seg.emotions).upper()}]")
            if seg.words_per_second:
                parts.append(f"[{seg.words_per_second:.1f}w/s]")
            if seg.hype_score > 0:
                parts.append(f"[HYPE:{seg.hype_score:.1f}]")
            if seg.arc_score > 0:
                parts.append(f"[ARC:{seg.arc_score:.1f}]")
            if seg.chat_spike_score > 0:
                parts.append(f"[CHAT:{seg.chat_spike_score:.2f}]")
            if seg.visual_activity > 0:
                parts.append(f"[VIS:{seg.visual_activity:.1f}]")
            parts.append(seg.text)
            lines.append(" ".join(parts))
        else:
            # Legacy tuple support
            start, end, text = seg
            if use_seconds:
                ts = f"[{start:.2f} - {end:.2f}]"
            else:
                ts = f"[{_fmt(start)} - {_fmt(end)}]"
            lines.append(f"{ts} {text}")
    return "\n".join(lines)


def format_jsonl(segments: list[RichSegment]) -> str:
    """Return one JSON object per line, preserving all numerical features."""
    lines = []
    for seg in segments:
        if not isinstance(seg, RichSegment):
            continue
        lines.append(json.dumps({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "speaker": seg.speaker,
            "text": seg.text,
            "tags": seg.emotions,
            "words_per_second": seg.words_per_second,
            "laughter_prob": seg.laughter_prob,
            "max_rms_db": seg.max_rms_db,
            "mean_rms_db": seg.mean_rms_db,
            "max_pitch_hz": seg.max_pitch_hz,
            "max_onset_rate": seg.max_onset_rate,
            "hype_score": seg.hype_score,
            "arc_score": seg.arc_score,
            "max_motion_score": seg.max_motion_score,
            "mean_face_size": seg.mean_face_size,
            "max_face_count": seg.max_face_count,
            "max_face_reaction": seg.max_face_reaction,
            "has_scene_change": seg.has_scene_change,
            "visual_activity": seg.visual_activity,
            "chat_spike_score": seg.chat_spike_score,
        }))
    return "\n".join(lines)


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
