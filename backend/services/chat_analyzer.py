"""
Chat-replay analysis: spike detection + emote/sentiment classification.

Downloads chat replay for a VOD via yt-dlp (Twitch rechat, YouTube live chat,
etc.), aggregates messages into per-second activity frames, detects message-rate
spikes above a rolling baseline, and classifies the dominant sentiment
(hype / laugh / shock / sadge).  The result is a `chat_spike_score` (0–1,
normalized) on each RichSegment plus optional emotion-category tags — both feed
into the composite hype_score computed by scorer.score_segments.

Works with any platform that yt-dlp can extract chat from. Sentiment
classification uses both platform-specific emote names and universal text
patterns so it degrades gracefully when emote metadata is unavailable.

Usage
-----
chat_path = download_chat(url, job_id, out_dir)   # returns None on failure
segments  = enrich_segments(segments, chat_path)  # always safe to call
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field

from config import settings
from services.transcriber import RichSegment

logger = logging.getLogger(__name__)


def _is_twitch_vod_url(url: str) -> bool:
    value = url.strip().lower()
    return "twitch.tv/videos/" in value


def _is_missing_twitch_chat_replay(message: str) -> bool:
    normalized = message.lower()
    return (
        "unable to download video subtitles for 'rechat'" in normalized
        and "http error 404" in normalized
    ) or (
        "unable to download video subtitles for 'live_chat'" in normalized
        and "http error 404" in normalized
    )


# ---------------------------------------------------------------------------
# Emote / sentiment dictionaries
# ---------------------------------------------------------------------------

# Platform-specific emote names (Twitch) → category
# These are matched by exact token; the regex patterns below catch everything
# else (universal text, YouTube emoji descriptions, etc.)
_EMOTE_HYPE: set[str] = {
    # Twitch
    "PogChamp", "Pog", "POGGIES", "POGGERS", "PogU", "PogBones",
    "peepoRiot", "peepoHype", "Kreygasm", "EZ",
    "trihard", "LETSGO", "popCorn",
}
_EMOTE_LAUGH: set[str] = {
    # Twitch
    "KEKW", "LUL", "LULW", "4Head", "OMEGALUL", "LULGE",
    "KEKWait", "Jebaited", "FeelsDankMan", "BATCHEST",
}
_EMOTE_SHOCK: set[str] = {
    # Twitch
    "monkaS", "monkaW", "monkaHmm", "OhMyDog", "WeirdChamp",
    "PauseChamp",
}
_EMOTE_SADGE: set[str] = {
    # Twitch
    "Sadge", "peepoSad", "FeelsBadMan", "PepeHands", "notlikethis",
}

# Normalised lookup: lowercase name → (category, weight)
_EMOTE_MAP: dict[str, tuple[str, float]] = {}
for _name in _EMOTE_HYPE:
    _EMOTE_MAP[_name.lower()] = ("hype", 1.0)
for _name in _EMOTE_LAUGH:
    _EMOTE_MAP[_name.lower()] = ("laugh", 1.0)
for _name in _EMOTE_SHOCK:
    _EMOTE_MAP[_name.lower()] = ("shock", 0.8)
for _name in _EMOTE_SADGE:
    _EMOTE_MAP[_name.lower()] = ("sadge", 0.3)

# Universal text patterns — platform-agnostic.
# Covers Twitch emote text, YouTube emoji descriptions, plain chat reactions,
# and common multilingual expressions used across any live stream format.
_HYPE_RE = re.compile(
    r"\b("
    # Twitch emote tokens
    r"pog(?:gers|champ|u|bones|gies)?"
    r"|peeporiot|peepoHype|kreygasm|trihard"
    # Universal hype expressions (EN + DE)
    r"|letsgo+|hype+"
    r"|gg(?:wp)?"
    r"|omg|wow+"
    r"|lets\s*go|let'?s\s*go"
    r"|sheesh|yooo+|goated?|insane|fire+"
    r"|🔥|🚀|💯|👑|🎉"
    r"|wahnsinn|krass|alter|unfassbar"
    r")\b",
    re.IGNORECASE,
)
_LAUGH_RE = re.compile(
    r"\b("
    # Twitch emote tokens
    r"kekw|lulw|omegalul|4head|jebaited|batchest"
    # Universal laugh expressions (EN + DE)
    r"|lmao|lol+|haha+|hehe+|xd+|💀|😂|🤣"
    r"|rofl|ded|dead\b|im\s*dead"
    r"|hahaha|haha+|hehe"
    r"|😹|🤣"
    r"|hahah|hahaa|hahaah"
    r")\b",
    re.IGNORECASE,
)
_SHOCK_RE = re.compile(
    r"\b("
    # Twitch emote tokens
    r"monkas?w?|monkahmm|weirdchamp|ohmydog"
    # Universal shock expressions (EN + DE)
    r"|wtf|what\s*the|no\s*way|noway"
    r"|😱|🤯|😮|😲"
    r"|bruh|bro\b|wait\s*what"
    r"|oh\s*my\s*god|omg"
    r"|nicht\s*wahr|das\s*kann\s*nicht|echt\s*jetzt"
    r")\b",
    re.IGNORECASE,
)
_SADGE_RE = re.compile(
    r"\b("
    # Twitch emote tokens
    r"sadge|peeposad|feelsbadman|pepeHands|notlikethis"
    # Universal sad/disappointed expressions
    r"|😢|😭|💔|🥺"
    r"|rip\b|f\s*in\s*chat|\bf\b"
    r"|schade|traurig|arme?r?\s"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChatFrame:
    """Aggregated chat activity for a single second of the stream."""
    t: float                                    # second index (float-aligned)
    msgs: int = 0                               # total message count
    emote_weight: float = 0.0                   # weighted emote contribution
    categories: dict[str, int] = field(default_factory=dict)


@dataclass
class ChatSpike:
    """A detected chat message-rate spike window."""
    t_start: float
    t_end: float
    intensity: float            # normalised 0–1 (higher = stronger spike vs baseline)
    dominant_category: str      # hype | laugh | shock | sadge | neutral


# ---------------------------------------------------------------------------
# Chat download
# ---------------------------------------------------------------------------

def download_chat(url: str, job_id: str, out_dir: pathlib.Path) -> pathlib.Path | None:
    """
    Download live chat replay via yt-dlp (Twitch, YouTube, etc.).

    Writes a JSON chat export to *out_dir* and returns its path.
    Returns None silently on any failure (network error, no chat available,
    platform does not support chat export, yt-dlp unavailable) so the rest
    of the pipeline is unaffected.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "chat")
    is_twitch_vod = _is_twitch_vod_url(url)

    if is_twitch_vod:
        logger.info("Starting Twitch chat download for job %s: %s", job_id, url)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--skip-download",
                "--write-subs",
                "--sub-langs",
                "rechat,live_chat",
                "--sub-format",
                "json3/json",
                "-o",
                out_template,
                url,
            ],
            capture_output=True,
            text=True,
            timeout=settings.chat_download_timeout_sec,
            check=False,          # we handle failure ourselves
        )
    except FileNotFoundError:
        if is_twitch_vod:
            logger.warning(
                "Twitch chat download skipped for job %s because yt-dlp is unavailable",
                job_id,
            )
        return None
    except subprocess.TimeoutExpired:
        if is_twitch_vod:
            logger.warning(
                "Twitch chat download timed out for job %s after %ss",
                job_id,
                settings.chat_download_timeout_sec,
            )
        return None

    # yt-dlp names the file chat.rechat.json / chat.live_chat.json3 (or similar)
    candidates = sorted(out_dir.glob("chat*.json*"))
    if candidates:
        if is_twitch_vod:
            logger.info(
                "Twitch chat download finished for job %s: %s",
                job_id,
                candidates[0],
            )
        return candidates[0]

    if not candidates and result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        if is_twitch_vod:
            if _is_missing_twitch_chat_replay(message):
                logger.info(
                    "Twitch chat replay unavailable for job %s (exit=%s); continuing without chat enrichment",
                    job_id,
                    result.returncode,
                )
            else:
                logger.warning(
                    "Twitch chat download failed for job %s (exit=%s): %s",
                    job_id,
                    result.returncode,
                    message,
                )
        else:
            logger.info(
                "Chat download failed for %s (exit=%s): %s",
                job_id,
                result.returncode,
                message,
            )
        return None

    if is_twitch_vod:
        logger.info(
            "Twitch chat download produced no chat file for job %s (exit=%s)",
            job_id,
            result.returncode,
        )
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_messages(path: pathlib.Path) -> list[dict]:
    """
    Parse a yt-dlp chat file into a flat list of ``{t, text}`` dicts.

    Handles two formats:
    - json3 (YouTube-style): ``{"events": [{tStartMs, segs: [{utf8}]}]}``
    - Twitch native list: ``[{content_offset_seconds, message: {body}}]``
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return []

    messages: list[dict] = []

    if isinstance(raw, dict) and "events" in raw:
        # json3 format — segs[*].utf8 holds "username: message text\n"
        for event in raw.get("events", []):
            t_sec = event.get("tStartMs", 0) / 1000.0
            text = "".join(seg.get("utf8", "") for seg in event.get("segs", [])).strip()
            if not text:
                continue
            # Strip leading "username: " prefix when present
            if ": " in text:
                text = text.split(": ", 1)[1]
            messages.append({"t": t_sec, "text": text})
        return messages

    if isinstance(raw, list):
        # Twitch native API format
        for msg in raw:
            t = float(msg.get("content_offset_seconds", 0))
            body = ""
            fragments: list[dict] = []
            m = msg.get("message", {})
            if isinstance(m, dict):
                body = m.get("body", "")
                fragments = m.get("fragments", [])
            emotes = [
                frag["emoticon"]["emoticon_id"]
                for frag in fragments
                if isinstance(frag.get("emoticon"), dict)
            ]
            messages.append({"t": t, "text": body, "emotes": emotes})
        return messages

    return []


# ---------------------------------------------------------------------------
# Emote / sentiment classification
# ---------------------------------------------------------------------------

def _classify(text: str, emote_ids: list[str] | None = None) -> tuple[float, str]:
    """
    Score a single message and return ``(weight, dominant_category)``.

    weight         — contribution to window hype (0.0–3.0, capped)
    dominant_cat   — hype | laugh | shock | sadge | neutral
    """
    cats: dict[str, float] = {"hype": 0.0, "laugh": 0.0, "shock": 0.0, "sadge": 0.0}

    # Named emote IDs (Twitch native format)
    for eid in (emote_ids or []):
        entry = _EMOTE_MAP.get(eid.lower())
        if entry:
            cats[entry[0]] += entry[1]

    # Token-based pattern matching (works for json3 plain-text format too)
    for token in re.findall(r"\S+", text):
        entry = _EMOTE_MAP.get(token.lower())
        if entry:
            cats[entry[0]] += entry[1] * 0.9   # slightly lower weight than ID match

    # Regex catch-all for variations not in the dict
    if _HYPE_RE.search(text):
        cats["hype"] += 0.6
    if _LAUGH_RE.search(text):
        cats["laugh"] += 0.6
    if _SHOCK_RE.search(text):
        cats["shock"] += 0.5
    if _SADGE_RE.search(text):
        cats["sadge"] += 0.2

    total = sum(cats.values())
    weight = min(total, 3.0)

    if total > 0:
        dominant = max(cats, key=lambda k: cats[k])
        dominant_cat = dominant if cats[dominant] > 0 else "neutral"
    else:
        dominant_cat = "neutral"

    return weight, dominant_cat


# ---------------------------------------------------------------------------
# Frame building
# ---------------------------------------------------------------------------

def build_frames(messages: list[dict]) -> list[ChatFrame]:
    """Aggregate messages into per-second ChatFrames, sorted by time.

    Identical messages within a 1-second window are deduplicated to suppress
    bot spam (e.g. 50 identical "!lurk" lines that would inflate spike scores).
    """
    # Dedup: drop messages whose exact text appeared in the same second
    seen: dict[int, set[str]] = {}
    deduped: list[dict] = []
    for msg in messages:
        t_int = int(msg["t"])
        text = msg.get("text", "")
        if text and text in seen.get(t_int, set()):
            continue
        seen.setdefault(t_int, set()).add(text)
        deduped.append(msg)

    frames: dict[int, ChatFrame] = {}
    for msg in deduped:
        t_int = int(msg["t"])
        if t_int not in frames:
            frames[t_int] = ChatFrame(t=float(t_int))
        f = frames[t_int]
        f.msgs += 1
        w, cat = _classify(msg.get("text", ""), msg.get("emotes"))
        f.emote_weight += w
        f.categories[cat] = f.categories.get(cat, 0) + 1

    return sorted(frames.values(), key=lambda f: f.t)


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

def detect_spikes(
    frames: list[ChatFrame],
    window_sec: int = 10,
    spike_threshold: float = 2.5,
    merge_gap_sec: float = 15.0,
) -> list[ChatSpike]:
    """
    Sliding-window spike detection over per-second message counts.

    A spike is any contiguous run where the *window_sec*-wide rolling
    message count exceeds ``spike_threshold × baseline``.  Adjacent spikes
    separated by less than *merge_gap_sec* are merged into one.

    Returns a list of ChatSpike objects sorted by t_start.
    """
    if not frames:
        return []

    frame_map: dict[int, ChatFrame] = {int(f.t): f for f in frames}
    t_min = int(frames[0].t)
    t_max = int(frames[-1].t)

    # Rolling window counts (msgs only — emote weight used for category, not spike)
    counts: list[tuple[int, float]] = []
    for t in range(t_min, t_max + 1):
        c = sum(frame_map[tt].msgs for tt in range(t, t + window_sec) if tt in frame_map)
        counts.append((t, float(c)))

    if not counts:
        return []

    raw_counts = [c for _, c in counts]
    sorted_raw = sorted(raw_counts)
    baseline = sorted_raw[len(sorted_raw) // 2] or 1.0     # median, min 1
    spike_min = baseline * spike_threshold

    # --- Identify spike intervals -------------------------------------------
    raw_spikes: list[tuple[float, float, float]] = []   # (t_start, t_end, peak)
    in_spike = False
    spike_start = 0.0
    peak_count = 0.0

    for t, c in counts:
        if c >= spike_min:
            if not in_spike:
                in_spike = True
                spike_start = float(t)
                peak_count = c
            else:
                peak_count = max(peak_count, c)
        else:
            if in_spike:
                raw_spikes.append((spike_start, float(t) + window_sec, peak_count))
                in_spike = False

    if in_spike:
        raw_spikes.append((spike_start, float(t_max) + window_sec, peak_count))

    # --- Merge adjacent spikes ----------------------------------------------
    merged: list[tuple[float, float, float]] = []
    for s, e, pk in raw_spikes:
        if merged and s - merged[-1][1] <= merge_gap_sec:
            prev_s, prev_e, prev_pk = merged[-1]
            merged[-1] = (prev_s, max(prev_e, e), max(prev_pk, pk))
        else:
            merged.append((s, e, pk))

    # --- Build ChatSpike objects with category info -------------------------
    spikes: list[ChatSpike] = []
    for s, e, pk in merged:
        intensity = min((pk - baseline) / max(baseline, 1.0), 5.0) / 5.0  # 0–1

        # Collect categories for the spike window
        cats: dict[str, int] = {}
        for t in range(int(s), int(e) + 1):
            if t in frame_map:
                for cat, cnt in frame_map[t].categories.items():
                    cats[cat] = cats.get(cat, 0) + cnt

        dominant = "neutral"
        if cats:
            best = max(cats, key=lambda k: cats[k])
            if cats[best] > 0 and best != "neutral":
                dominant = best

        spikes.append(ChatSpike(
            t_start=s,
            t_end=e,
            intensity=round(intensity, 3),
            dominant_category=dominant,
        ))

    return spikes


# ---------------------------------------------------------------------------
# Segment enrichment
# ---------------------------------------------------------------------------

def enrich_segments(
    segments: list[RichSegment],
    chat_path: pathlib.Path | None,
) -> list[RichSegment]:
    """
    Attach ``chat_spike_score`` and optional chat emotion tags to each segment.

    Always safe to call — returns segments unchanged if *chat_path* is None
    or if parsing / spike detection fails for any reason.
    """
    if chat_path is None or not chat_path.exists():
        return segments

    try:
        messages = load_messages(chat_path)
        if not messages:
            return segments
        frames = build_frames(messages)
        spikes = detect_spikes(frames)
    except Exception:
        logger.exception("Chat enrichment failed for %s", chat_path)
        return segments

    if not spikes:
        return segments

    for seg in segments:
        best_intensity = 0.0
        dominant_cat = "neutral"

        for spike in spikes:
            overlap = min(seg.end, spike.t_end) - max(seg.start, spike.t_start)
            if overlap > 0:
                if spike.intensity > best_intensity:
                    best_intensity = spike.intensity
                    dominant_cat = spike.dominant_category

        if best_intensity > 0:
            seg.chat_spike_score = round(best_intensity, 3)
            # Add category tag to emotions so the LLM sees e.g. [CHAT_HYPE]
            cat_tag = f"chat_{dominant_cat}"
            if dominant_cat != "neutral" and cat_tag not in seg.emotions:
                seg.emotions.append(cat_tag)

    return segments
