"""
Per-second audio feature extraction — runs in parallel with Whisper transcription.

For each second of audio, computes:
  - pitch_hz       : dominant fundamental frequency via YIN (0 = unvoiced / silent)
  - rms_db         : loudness in dBFS
  - is_silence     : True when rms_db < SILENCE_THRESHOLD_DB
  - laughter_prob  : 0–1 from LaughterDetector
  - onset_rate     : onsets/second — proxy for speech tempo from raw audio

After extraction, `enrich_segments` merges frame-level features back into
RichSegment objects by appending new emotion tags:
  "audio_laughter"  — laughter_prob ≥ 0.6 in segment window
  "energy_burst"    — RMS notably above video median
  "pitch_spike"     — pitch notably above speaker median
  "silence_before"  — ≥ 2 silent seconds immediately before segment
  "fast_speech_audio" — onset_rate notably above video median

`diarize_speakers` assigns STREAMER / GUEST_N labels via MFCC-based KMeans
clustering — works with no external API token.
"""

import pathlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import librosa

# Avoid heavy numba JIT output in server logs
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


SILENCE_THRESHOLD_DB = -35.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatureFrame:
    second: int
    pitch_hz: float        # Hz, 0 if unvoiced
    rms_db: float          # dBFS
    is_silence: bool
    laughter_prob: float   # 0.0–1.0
    onset_rate: float      # onsets/second


# ---------------------------------------------------------------------------
# Laughter detector
# ---------------------------------------------------------------------------

class LaughterDetector:
    """
    Rule-based acoustic laughter classifier.

    Laughter differs from speech in three measurable ways:
    1. High zero-crossing rate  — breathy, partly unvoiced bursts
    2. High spectral flatness   — noise-like rather than tonal
    3. Pulsed amplitude envelope — rhythmic "ha-ha" modulation at ~3-8 Hz

    Each feature is passed through a sigmoid to produce a 0-1 score,
    then combined via a weighted sum.
    """

    _HOP = 512

    def predict(self, y: np.ndarray, sr: int) -> float:
        if len(y) < sr // 10:
            return 0.0
        if _rms_db(y) < SILENCE_THRESHOLD_DB:
            return 0.0

        # Feature 1 — zero-crossing rate
        zcr = float(np.mean(
            librosa.feature.zero_crossing_rate(y, hop_length=self._HOP)
        ))

        # Feature 2 — spectral flatness (Wiener entropy)
        flatness = float(np.mean(
            librosa.feature.spectral_flatness(y=y, hop_length=self._HOP)
        ))

        # Feature 3 — amplitude modulation rate (ZCR of the RMS envelope)
        rms_env = librosa.feature.rms(y=y, hop_length=self._HOP)[0]
        if rms_env.max() > 1e-8:
            rms_norm = rms_env / rms_env.max()
            mod_rate = float(np.mean(
                librosa.feature.zero_crossing_rate(rms_norm)
            ))
        else:
            mod_rate = 0.0

        score = (
            _sigmoid(zcr,      center=0.10, scale=25) * 0.30 +
            _sigmoid(flatness, center=0.05, scale=35) * 0.35 +
            _sigmoid(mod_rate, center=0.25, scale=18) * 0.35
        )

        return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def _extract_frames(y: np.ndarray, sr: int) -> list[AudioFeatureFrame]:
    duration_s = int(len(y) / sr)
    detector = LaughterDetector()
    frames: list[AudioFeatureFrame] = []

    for sec in range(duration_s):
        s0 = sec * sr
        s1 = s0 + sr
        y_frame = y[s0:s1]

        if len(y_frame) == 0:
            continue

        # --- Pitch via YIN ---
        try:
            f0 = librosa.yin(
                y_frame,
                fmin=librosa.note_to_hz("C2"),   # ~65 Hz
                fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
                sr=sr,
            )
            voiced = f0[(f0 > 50) & (f0 < 2000)]
            pitch_hz = float(np.median(voiced)) if len(voiced) > 0 else 0.0
        except Exception:
            pitch_hz = 0.0

        # --- RMS energy ---
        rms_db_val = _rms_db(y_frame)
        is_silence = rms_db_val < SILENCE_THRESHOLD_DB

        # --- Onset rate (speech tempo proxy) ---
        try:
            onset_frames = librosa.onset.onset_detect(y=y_frame, sr=sr, units="frames")
            onset_rate = float(len(onset_frames))
        except Exception:
            onset_rate = 0.0

        # --- Laughter probability ---
        laughter_prob = detector.predict(y_frame, sr)

        frames.append(AudioFeatureFrame(
            second=sec,
            pitch_hz=round(pitch_hz, 1),
            rms_db=round(rms_db_val, 1),
            is_silence=is_silence,
            laughter_prob=round(laughter_prob, 3),
            onset_rate=round(onset_rate, 1),
        ))

    return frames


def extract(audio_path: pathlib.Path) -> list[AudioFeatureFrame]:
    """
    Load audio and return one AudioFeatureFrame per second.

    Expects 16 kHz mono WAV (output of audio_extractor.extract).
    """
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    return _extract_frames(y, sr)


def extract_window_from_signal(
    y: np.ndarray,
    sr: int,
    *,
    start_sec: float,
    end_sec: float,
) -> list[AudioFeatureFrame]:
    start_sample = max(int(round(float(start_sec) * sr)), 0)
    end_sample = max(int(round(float(end_sec) * sr)), start_sample)
    return _extract_frames(y[start_sample:end_sample], sr)


# ---------------------------------------------------------------------------
# Segment enrichment
# ---------------------------------------------------------------------------

def enrich_segments(segments, frames: list[AudioFeatureFrame]):
    """
    Append audio-derived emotion tags to each RichSegment.

    Tags added (appended to segment.emotions):
      "audio_laughter"    — laughter_prob ≥ 0.6 within segment
      "energy_burst"      — max RMS ≥ median + 1.2σ across video
      "pitch_spike"       — max pitch ≥ median + 1.5σ of voiced frames
      "silence_before"    — ≥ 2 silent seconds just before segment start
      "fast_speech_audio" — onset_rate ≥ median + 1.5σ across video
    """
    if not frames or not segments:
        return segments

    frame_map: dict[int, AudioFeatureFrame] = {f.second: f for f in frames}

    # Global thresholds (computed once)
    voiced_pitches = [f.pitch_hz for f in frames if f.pitch_hz > 0]
    rms_values     = [f.rms_db   for f in frames if not f.is_silence]
    onset_values   = [f.onset_rate for f in frames if not f.is_silence]

    pitch_threshold  = _threshold(voiced_pitches, sigma=1.5) if voiced_pitches else 9999
    energy_threshold = _threshold(rms_values,     sigma=1.2) if rms_values     else 0
    onset_threshold  = _threshold(onset_values,   sigma=1.5) if onset_values   else 9999

    for idx, seg in enumerate(segments):
        seg_seconds = range(int(seg.start), int(seg.end) + 1)
        seg_frames  = [frame_map[s] for s in seg_seconds if s in frame_map]

        if not seg_frames:
            continue

        max_laughter = max(f.laughter_prob for f in seg_frames)
        max_rms      = max(f.rms_db for f in seg_frames)
        max_pitch    = max((f.pitch_hz for f in seg_frames), default=0.0)
        max_onset    = max((f.onset_rate for f in seg_frames), default=0.0)
        voiced_frames = [
            f for f in seg_frames
            if not f.is_silence and (f.pitch_hz > 0.0 or f.onset_rate > 0.0)
        ]
        pitch_values = [f.pitch_hz for f in voiced_frames if f.pitch_hz > 0.0]

        # Mean RMS over active (non-silent) frames — used for arc contrast scoring.
        # Stays at 0.0 (sentinel) if the whole segment is silent.
        active_frames = [f for f in seg_frames if not f.is_silence]
        mean_rms = (
            sum(f.rms_db for f in active_frames) / len(active_frames)
            if active_frames else 0.0
        )

        # Store raw numerical values on the segment for JSONL and scoring
        seg.laughter_prob  = round(max_laughter, 3)
        seg.max_rms_db     = round(max_rms, 1)
        seg.mean_rms_db    = round(mean_rms, 1)
        seg.max_pitch_hz   = round(max_pitch, 1)
        seg.pitch_std_hz   = round(float(np.std(pitch_values)) if len(pitch_values) >= 2 else 0.0, 1)
        seg.max_onset_rate = round(max_onset, 1)
        seg.voice_activity_score = round(len(voiced_frames) / max(len(seg_frames), 1), 3)
        seg.rms_burst_score = round(float(np.clip((max_rms - energy_threshold) / 12.0 + 0.5, 0.0, 1.0)), 3)

        # Audio laughter
        if max_laughter >= 0.6:
            seg.emotions.append("audio_laughter")

        # Energy burst
        if max_rms >= energy_threshold:
            seg.emotions.append("energy_burst")

        # Pitch spike
        if max_pitch >= pitch_threshold:
            seg.emotions.append("pitch_spike")

        # Silence before requires both:
        # 1. a real transcript gap immediately before the segment, and
        # 2. a contiguous run of silent audio frames ending right before start.
        prev_end = segments[idx - 1].end if idx > 0 else None
        transcript_gap = seg.start - prev_end if prev_end is not None else seg.start
        if transcript_gap >= 1.0 and _trailing_silence_seconds(frame_map, seg.start) >= 2:
            seg.emotions.append("silence_before")

        # Fast speech from audio
        if max_onset >= onset_threshold:
            seg.emotions.append("fast_speech_audio")

    return segments


# ---------------------------------------------------------------------------
# Speaker diarization — MFCC-based KMeans clustering (no auth token needed)
# ---------------------------------------------------------------------------

def diarize_speakers(segments, audio_path: pathlib.Path) -> list:
    """
    Assign STREAMER / GUEST_N labels to each RichSegment using voice clustering.

    Strategy:
    - Extract 20 MFCCs + their deltas per segment as a speaker embedding
    - Cluster all non-silent segments with KMeans
    - Auto-detect k=2..4 via silhouette score
    - Longest total speaking time → STREAMER; rest → GUEST_1, GUEST_2, ...

    Falls back gracefully: if audio is too short or clustering fails,
    segments remain labelled UNKNOWN.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception:
        return segments

    feature_vecs: list[np.ndarray] = []
    valid_idx:    list[int]         = []

    for i, seg in enumerate(segments):
        s0 = int(seg.start * sr)
        s1 = int(seg.end   * sr)
        y_seg = y[s0:s1]

        # Skip segments that are too short or silent
        min_samples = sr // 4  # 250 ms
        if len(y_seg) < min_samples or _rms_db(y_seg) < SILENCE_THRESHOLD_DB:
            continue

        mfcc       = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)

        # Concatenate mean + std of MFCCs and their deltas → 80-dim vector
        vec = np.concatenate([
            np.mean(mfcc,       axis=1),
            np.std(mfcc,        axis=1),
            np.mean(mfcc_delta, axis=1),
            np.std(mfcc_delta,  axis=1),
        ])

        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            continue

        feature_vecs.append(vec)
        valid_idx.append(i)

    if len(feature_vecs) < 4:
        return segments

    X = StandardScaler().fit_transform(np.array(feature_vecs))

    # Find optimal number of speakers (2–4) via silhouette score
    n_speakers = _best_k(X, k_range=range(2, min(5, len(feature_vecs))))

    labels: np.ndarray = KMeans(
        n_clusters=n_speakers, n_init=15, random_state=42
    ).fit_predict(X)

    # Cluster with longest cumulative speech time = STREAMER
    cluster_duration: dict[int, float] = defaultdict(float)
    for list_pos, seg_idx in enumerate(valid_idx):
        seg = segments[seg_idx]
        cluster_duration[int(labels[list_pos])] += seg.end - seg.start

    streamer_cluster = max(cluster_duration, key=cluster_duration.__getitem__)
    guest_clusters   = sorted(c for c in cluster_duration if c != streamer_cluster)

    cluster_to_label: dict[int, str] = {streamer_cluster: "STREAMER"}
    for guest_num, c in enumerate(guest_clusters, start=1):
        cluster_to_label[c] = f"GUEST_{guest_num}"

    for list_pos, seg_idx in enumerate(valid_idx):
        segments[seg_idx].speaker = cluster_to_label[int(labels[list_pos])]

    return segments


def _best_k(X: np.ndarray, k_range: range) -> int:
    """Return k with the highest silhouette score over the given range."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_score = k_range.start, -1.0
    for k in k_range:
        if k >= len(X):
            break
        labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(X)
        # Silhouette is undefined if all samples fall in one cluster
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels, sample_size=min(500, len(X))))
        if score > best_score:
            best_score, best_k = score, k

    return best_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))
    return 20.0 * np.log10(rms)


def _sigmoid(x: float, center: float, scale: float) -> float:
    return 1.0 / (1.0 + np.exp(-scale * (x - center)))


def _threshold(values: list[float], sigma: float) -> float:
    arr = np.array(values, dtype=np.float64)
    return float(np.median(arr) + sigma * np.std(arr))


def _trailing_silence_seconds(
    frame_map: dict[int, AudioFeatureFrame],
    seg_start: float,
) -> int:
    """Count consecutive silent whole-second frames ending immediately before seg_start."""
    count = 0
    sec = int(seg_start) - 1
    while sec >= 0:
        frame = frame_map.get(sec)
        if frame is None or not frame.is_silence:
            break
        count += 1
        sec -= 1
    return count
