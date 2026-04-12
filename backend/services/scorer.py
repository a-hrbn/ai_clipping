"""
Pre-scoring and candidate window extraction.

Runs before the LLM analysis step to:
1. Attach a composite `hype_score` to every segment (numerical, not binary).
2. Compute an `arc_score` for each candidate window — rewards emotional build-up,
   sustained peaks, and a payoff/resolution, not just isolated spikes.
3. Identify the top-N high-signal anchors and expand time windows around them.
   Only these candidate segments are forwarded to the LLM, keeping the prompt
   focused and within token limits for long VODs.
"""

from __future__ import annotations

from services.timeline_types import SegmentScore
from services.transcriber import RichSegment


def score_segments(segments: list[RichSegment]) -> list[RichSegment]:
    """
    Compute a composite hype_score for each segment and store it in-place.

    Score components (max ~13.0):
      laughter_prob × 2.0     — acoustic laughter (strong signal, but not dominant alone)
      energy_burst  × 2.0     — RMS loudness spike
      pitch_spike   × 2.0     — pitch well above baseline (excitement / emphasis)
      audio_laughter× 1.0     — acoustic tag (reduced to limit double-count with laughter_prob)
      silence_before× 1.0     — dramatic pause → big reaction pattern (structure signal)
      exclamation   × 1.0     — text-based excitement
      words/sec cap × 1.5     — speech tempo capped to avoid over-weighting
      chat_spike    × 1.5     — audience chat engagement (useful signal, not the arbiter)
    """
    for seg in segments:
        tag_set = {t.lower() for t in seg.emotions}
        seg.hype_score = round(
            # --- Audio signals ---
            seg.laughter_prob * 2.0               # reduced: relevant for all content, not only gaming
            + (2.0 if "energy_burst"    in tag_set else 0.0)
            + (2.0 if "pitch_spike"     in tag_set else 0.0)
            + (1.0 if "audio_laughter"  in tag_set else 0.0)
            + (1.0 if "silence_before"  in tag_set else 0.0)
            + (1.0 if "exclamation"     in tag_set else 0.0)
            + min((seg.words_per_second or 0.0) / 4.0, 1.5)
            # --- Chat-replay signal ------------------------------------------
            + seg.chat_spike_score * 1.5          # reduced: absent on many non-gaming streams
            # --- Visual signals (adds up to ~4.5 pts, preserves calibration) ---
            + (seg.max_face_reaction * 2.0)                   # 0–2.0 continuous
            + (1.5 if "high_motion"    in tag_set else 0.0)   # physical reaction
            + (1.0 if "face_reaction"  in tag_set else 0.0)   # bonus on top of continuous
            + (0.8 if "scene_change"   in tag_set else 0.0)   # visual event / cut
            + (0.7 if "multi_face"     in tag_set else 0.0)   # collab / guest
            + (0.5 if "prominent_face" in tag_set else 0.0)   # close-up reaction
            + min(seg.max_motion_score * 1.5, 1.5),           # 0–1.5 continuous
            3,
        )
    return segments


def build_score_timeline(segments: list[RichSegment]) -> list[SegmentScore]:
    if not segments:
        return []

    timeline: list[SegmentScore] = []
    for idx, seg in enumerate(segments):
        hook_score = _hook_signal(seg)
        payoff_score = _payoff_signal(seg)
        audience_score = _audience_signal(seg)
        semantic_score = _semantic_signal(seg)
        virality_score = round(
            seg.hype_score
            + hook_score
            + payoff_score
            + audience_score
            + semantic_score,
            3,
        )
        timeline.append(
            SegmentScore(
                segment_index=idx,
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                virality_score=virality_score,
                hook_score=hook_score,
                payoff_score=payoff_score,
                audience_score=audience_score,
                semantic_score=semantic_score,
            )
        )

    smoothed = _smooth_scores([item.virality_score for item in timeline])
    for item, smooth in zip(timeline, smoothed):
        item.smoothed_score = smooth

    return timeline


def _hook_signal(seg: RichSegment) -> float:
    score = 0.0
    if "exclamation" in seg.emotions:
        score += 0.35
    if "question" in seg.emotions:
        score += 0.25
    if seg.words_per_second >= 2.8:
        score += 0.2
    return round(score, 3)


def _payoff_signal(seg: RichSegment) -> float:
    score = 0.0
    if "audio_laughter" in seg.emotions or "laughter" in seg.emotions:
        score += 0.45
    if "face_reaction" in seg.emotions:
        score += 0.3
    if "high_motion" in seg.emotions:
        score += 0.2
    return round(score, 3)


def _audience_signal(seg: RichSegment) -> float:
    return round(seg.chat_spike_score * 0.8, 3)


def _semantic_signal(seg: RichSegment) -> float:
    text = seg.text.strip()
    if not text:
        return 0.0

    score = 0.0
    if text.endswith(("?", "!")):
        score += 0.25
    word_count = len(text.split())
    if word_count >= 8:
        score += 0.15
    lower = text.lower()
    if any(token in lower for token in ("wait", "look", "listen", "bro", "what", "why", "how")):
        score += 0.15
    return round(score, 3)


def _smooth_scores(values: list[float], radius: int = 1) -> list[float]:
    smoothed: list[float] = []
    for idx in range(len(values)):
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        window = values[left:right]
        smoothed.append(round(sum(window) / len(window), 3))
    return smoothed


def score_arc(window: list[RichSegment]) -> float:
    """
    Measure the emotional arc quality of a candidate clip window.

    Supports multiple narrative shapes — not just the gaming Setup→Peak→Reaction
    pattern.  Two arc shapes are scored and the higher one wins:

    Shape A — Reactive arc (gaming, IRL reactions, hype moments):
      Setup      — relatively quiet / low energy (first ~30%)
      Peak       — the viral moment: high energy, hype, or laughter (middle ~40%)
      Resolution — payoff / reaction: laughter, exclamation, or energy settling (last ~30%)

    Shape B — Sustained engagement (Just Chatting, podcast, interview):
      Rewards clips where energy stays consistently above a baseline throughout,
      without requiring a dramatic rise.  Catches insightful discussions,
      storytelling moments, and emotional conversations that don't spike.

    Score components (shared):
      hype_contrast   — how much hype_score rises from setup to peak (×2.0)
      rms_contrast    — dB energy rise from setup to peak, normalized (×1.0, max 2.0)
      has_payoff      — emotional/verbal payoff in the resolution phase (+2.0)
      quiet_setup     — setup is genuinely calm (bonus +0.8, prevents "two peaks" clips)
      sustained_peak  — more than one high-hype segment in the peak phase (+0.5 per extra)
      sustained_eng   — Shape B bonus: consistent mid-level engagement throughout (+2.5)
    """
    if len(window) < 4:
        return 0.0

    n = len(window)
    setup_end      = max(1, n * 3 // 10)       # first 30%
    peak_end       = max(setup_end + 1, n * 7 // 10)  # middle 40%
    setup      = window[:setup_end]
    peak       = window[setup_end:peak_end]
    resolution = window[peak_end:]

    # --- Hype contrast -------------------------------------------------------
    setup_hype = sum(s.hype_score for s in setup) / len(setup)
    peak_hypes = [s.hype_score for s in peak]
    peak_hype  = max(peak_hypes)
    hype_contrast = max(0.0, peak_hype - setup_hype)

    # Number of segments in peak phase that score above setup level + 1.0
    sustained = sum(1 for h in peak_hypes if h > setup_hype + 1.0) - 1  # -1 for the max itself
    sustained_bonus = max(0.0, sustained) * 0.5

    # --- RMS energy contrast (dBFS values are negative; higher = louder) -----
    # Only use segments where mean_rms_db was populated (< 0.0 means real audio)
    setup_rms_vals = [s.mean_rms_db for s in setup if s.mean_rms_db < 0.0]
    peak_rms_vals  = [s.max_rms_db  for s in peak  if s.max_rms_db  < 0.0]

    if setup_rms_vals and peak_rms_vals:
        setup_rms_mean = sum(setup_rms_vals) / len(setup_rms_vals)
        peak_rms_max   = max(peak_rms_vals)
        rms_contrast   = max(0.0, peak_rms_max - setup_rms_mean)
        # 10 dB rise → 1.0 point, 20 dB rise → 2.0 points (capped)
        rms_score = min(rms_contrast / 10.0, 2.0)
    else:
        rms_score = 0.0

    # --- Resolution / payoff -------------------------------------------------
    # Broad payoff definition: covers reactive (laughter, motion) AND verbal
    # (question, statement end) payoffs so dialogue-heavy content scores fairly.
    resolution_tags = {e for s in resolution for e in s.emotions}
    has_payoff = bool(
        resolution_tags & {
            "laughter", "audio_laughter", "exclamation", "question",
            "face_reaction", "high_motion",  # visible/audible reaction
            "dialogue", "statement",         # verbal conclusion / insight payoff
        }
    )

    # --- Quiet setup bonus ---------------------------------------------------
    quiet_setup = setup_hype < 1.5

    # --- Shape A: reactive arc -----------------------------------------------
    arc_a = (
        hype_contrast  * 2.0
        + rms_score
        + (2.0 if has_payoff else 0.0)
        + (0.8 if quiet_setup else 0.0)
        + sustained_bonus
    )

    # --- Shape B: sustained engagement (Just Chatting / podcast / interview) -
    # Scores well when all phases maintain consistent mid-level energy,
    # even without a dramatic reactive spike.
    all_hypes = [s.hype_score for s in window]
    mean_hype = sum(all_hypes) / len(all_hypes)
    above_floor = sum(1 for h in all_hypes if h >= 1.0) / len(all_hypes)  # fraction above baseline
    arc_b = 0.0
    if above_floor >= 0.6 and mean_hype >= 1.2:
        # Reward consistent engagement: 0–2.5 based on mean hype level
        arc_b = min(mean_hype / 2.0, 1.25) * 2.0
        if has_payoff:
            arc_b += 1.0

    return round(max(arc_a, arc_b), 2)


def extract_candidates(
    segments: list[RichSegment],
    max_window: float = 90.0,
    top_n: int = 10,
    score_threshold: float = 2.5,
) -> list[RichSegment]:
    """
    Return the subset of segments that fall inside high-scoring windows.

    For each candidate window:
    - The arc_score is computed and assigned to every segment in the window
      (taking the maximum if a segment belongs to multiple windows).
    - This propagates arc quality to the LLM prompt via [ARC:X.X] tags.

    Falls back to the full segment list if no anchor clears the threshold,
    so the pipeline never returns zero results.
    """
    if not segments:
        return segments

    # Scale top_n with video duration: at least one anchor per 5 minutes.
    duration_sec = segments[-1].end - segments[0].start
    top_n = max(top_n, int(duration_sec / 300))

    anchors = [s for s in segments if s.hype_score >= score_threshold]
    if not anchors:
        return segments  # LLM decides on the full transcript

    # Sort by hype score descending, then enforce minimum spacing so that
    # no two anchors fall within one window-width of each other.
    # Without this, clustered high-hype moments produce overlapping candidate
    # windows → the LLM sees the same peak repeatedly → dedup collapses
    # all clips to 1.
    anchors_sorted = sorted(anchors, key=lambda s: s.hype_score, reverse=True)
    spread_anchors: list[RichSegment] = []
    for anchor in anchors_sorted:
        too_close = any(
            abs(anchor.start - kept.start) < max_window
            for kept in spread_anchors
        )
        if not too_close:
            spread_anchors.append(anchor)
        if len(spread_anchors) >= top_n:
            break
    anchors = spread_anchors

    # Map from segment index → set of indices in its window, keyed by anchor
    selected: set[int] = set()

    for anchor in anchors:
        mid     = (anchor.start + anchor.end) / 2.0
        w_start = mid - max_window / 2.0
        w_end   = mid + max_window / 2.0

        window_indices = [
            i for i, seg in enumerate(segments)
            if seg.end >= w_start and seg.start <= w_end
        ]
        if not window_indices:
            continue

        window_segs = [segments[i] for i in window_indices]
        arc         = score_arc(window_segs)

        # Assign arc_score to every segment in the window (keep maximum)
        for i in window_indices:
            if arc > segments[i].arc_score:
                segments[i].arc_score = arc

        selected.update(window_indices)

    return [segments[i] for i in sorted(selected)]
