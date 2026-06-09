"""Compute per-segment RMS energy from a WAV file.

Uses only the Python standard library (wave) and numpy (already a core dep).
No librosa or heavy ML deps required.

The key insight: absolute loudness is less meaningful than a *relative* burst
vs. the local baseline.  A shout in a quiet passage is more clip-worthy than
the same absolute level in an already-loud section.  We therefore normalise
using the same sliding-window baseline approach as chat_signal.py.
"""
from __future__ import annotations

import pathlib
import struct
import wave
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from services.transcriber import RichSegment

# Baseline window for relative energy normalisation (seconds each side of segment).
_ENERGY_BASELINE_WINDOW_SEC = 120
# Ratio at which the relative energy score saturates to ~1.0.
_ENERGY_SATURATION_RATIO = 5.0


def compute_segment_energy(
    wav_path: pathlib.Path | None,
    segments: list[RichSegment],
) -> list[float]:
    """Return a per-segment relative RMS energy score in [0, 1].

    Returns a list of 0.0 on any error (missing/corrupt WAV, empty segments)
    so the pipeline always gets a safe fallback without raising.
    """
    n = len(segments)
    if not n:
        return []
    zeros = [0.0] * n

    if wav_path is None or not wav_path.exists():
        return zeros

    try:
        samples, sample_rate = _read_wav_mono_float(wav_path)
    except Exception:
        return zeros

    if sample_rate <= 0 or len(samples) == 0:
        return zeros

    rms_by_sec = _rms_per_second(samples, sample_rate)
    if not rms_by_sec:
        return zeros

    scores: list[float] = []
    for seg in segments:
        start_s = int(max(getattr(seg, "start", 0.0) or 0.0, 0))
        end_s   = int(getattr(seg, "end",   0.0) or 0.0)
        if end_s <= start_s:
            scores.append(0.0)
            continue

        seg_rms = _mean_rms(rms_by_sec, start_s, end_s)
        score = _relative_energy_score(rms_by_sec, start_s, end_s, seg_rms)
        scores.append(round(score, 4))

    return scores


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_wav_mono_float(path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Read WAV file as a normalised float32 mono array."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    fmt = {1: "b", 2: "<h", 4: "<i"}.get(sample_width)
    if fmt is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    samples = np.frombuffer(raw, dtype=dtype_map[sample_width])

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    max_val = float(2 ** (8 * sample_width - 1))
    return samples.astype(np.float32) / max_val, frame_rate


def _rms_per_second(samples: np.ndarray, sample_rate: int) -> dict[int, float]:
    """Compute RMS energy for each integer second of audio."""
    result: dict[int, float] = {}
    total_seconds = max(int(len(samples) / sample_rate), 1)
    for sec in range(total_seconds):
        start = sec * sample_rate
        end = start + sample_rate
        chunk = samples[start:end]
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        result[sec] = rms
    return result


def _mean_rms(rms_by_sec: dict[int, float], start_s: int, end_s: int) -> float:
    vals = [rms_by_sec[s] for s in range(start_s, end_s) if s in rms_by_sec]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _relative_energy_score(
    rms_by_sec: dict[int, float],
    start_s: int,
    end_s: int,
    seg_rms: float,
) -> float:
    """Normalise seg_rms vs. local baseline (sliding window ±_ENERGY_BASELINE_WINDOW_SEC)."""
    window_start = max(0, start_s - _ENERGY_BASELINE_WINDOW_SEC)
    window_end   = end_s + _ENERGY_BASELINE_WINDOW_SEC
    baseline_vals = [
        rms_by_sec[s]
        for s in range(window_start, window_end)
        if s in rms_by_sec and not (start_s <= s < end_s)
    ]
    if len(baseline_vals) < 3:
        return 0.0
    baseline = float(np.median(baseline_vals))
    if baseline < 1e-9:
        return 0.0
    ratio = seg_rms / baseline
    return float(min(ratio / _ENERGY_SATURATION_RATIO, 1.0))
