from __future__ import annotations

import json
import pathlib
import re
import statistics
from typing import TYPE_CHECKING, Sequence

import requests

if TYPE_CHECKING:
    from services.transcriber import RichSegment

_TWITCH_GQL_URL = "https://gql.twitch.tv/gql"
_TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
_VOD_ID_RE = re.compile(r"(?:twitch\.tv/videos?/|^)(\d{7,12})")

# Curated emote/keyword sets — deliberately small; only signals that correlate with viral moments
_HYPE_TOKENS = frozenset([
    "pog", "pogchamp", "pogu", "pogcrazy", "pogs", "hypepog",
    "monkaw", "monkas", "hype", "letsgo", "omg", "bro", "insane",
    "clip", "clipped", "w", "ww", "www", "goat", "clutch",
])
_FUNNY_TOKENS = frozenset([
    "lul", "lulw", "kekw", "lmao", "lol", "loool", "xd", "haha", "😂", "💀",
    "dead", "ded", "kekw",
])
_SHOCK_TOKENS = frozenset([
    "wtf", "what", "noway", "noooo", "noo", "ohno",
    "peeposad", "sadge", "pepehands", "dno",
])

CHAT_SPIKE_BASELINE_WINDOW_SEC = 60
CHAT_SPIKE_SATURATION_RATIO = 8.0  # ratio at which spike_score reaches ~1.0
# Viewers type ~5-10s after the moment; look forward by this many seconds when
# attributing a chat spike to the segment that caused it.
CHAT_REACTION_DELAY_SEC = 8


def _extract_vod_id(url: str) -> str | None:
    m = _VOD_ID_RE.search(str(url or ""))
    return m.group(1) if m else None


def download_chat(
    url: str,
    job_id: str,
    out_dir: pathlib.Path,
    *,
    timeout_sec: int = 30,
) -> pathlib.Path | None:
    """Download Twitch VOD chat via GraphQL and write compact chat.json.

    Returns path to chat.json on success, None on non-Twitch URL or any error.
    Output format: list of {"t": offset_sec, "text": str, "emotes": [name, ...]}.
    """
    vod_id = _extract_vod_id(url)
    if not vod_id:
        return None

    cache_path = pathlib.Path(out_dir) / "chat.json"
    if cache_path.exists():
        return cache_path

    messages: list[dict] = []
    cursor = None
    offset_sec = 0

    try:
        while True:
            payload = _build_gql_payload(vod_id, offset_sec, cursor)
            resp = requests.post(
                _TWITCH_GQL_URL,
                json=payload,
                headers={"Client-ID": _TWITCH_CLIENT_ID},
                timeout=timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()

            comments_data = (
                data.get("data", {})
                .get("video", {})
                .get("comments", {})
            )
            if not comments_data:
                break

            edges = comments_data.get("edges", [])
            for edge in edges:
                node = edge.get("node", {})
                msg = _parse_comment_node(node)
                if msg is not None:
                    messages.append(msg)

            page_info = comments_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break

            # Advance cursor to last seen offset + 1 to avoid duplicate pages
            if edges:
                last_node = edges[-1].get("node", {})
                last_offset = last_node.get("contentOffsetSeconds")
                if last_offset is not None:
                    offset_sec = int(last_offset) + 1
                    cursor = None
                else:
                    cursor = edges[-1].get("cursor")
                    if not cursor:
                        break
            else:
                break

    except Exception:
        return None

    if not messages:
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return cache_path


def _build_gql_payload(vod_id: str, offset_sec: int, cursor: str | None) -> dict:
    if cursor:
        variables: dict = {"videoID": vod_id, "cursor": cursor, "first": 100}
    else:
        variables = {"videoID": vod_id, "contentOffsetSeconds": offset_sec, "first": 100}
    return {
        "operationName": "VideoCommentsByOffsetOrCursor",
        "variables": variables,
        "extensions": {
            "persistedQuery": {
                "version": 1,
                "sha256Hash": "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a",
            }
        },
    }


def _parse_comment_node(node: dict) -> dict | None:
    offset = node.get("contentOffsetSeconds")
    if offset is None:
        return None
    msg_body = node.get("message", {})
    text_fragments = msg_body.get("fragments", [])
    text_parts: list[str] = []
    emote_names: list[str] = []
    for frag in text_fragments:
        frag_text = frag.get("text", "")
        if frag_text:
            text_parts.append(frag_text)
        emote = frag.get("emote")
        if emote:
            emote_names.append(frag_text or emote.get("emoteID", ""))
    text = " ".join(text_parts).strip()
    if not text:
        return None
    result: dict = {"t": int(offset), "text": text}
    if emote_names:
        result["emotes"] = emote_names
    return result


def enrich_segments(
    segments: Sequence[RichSegment],
    chat_path: pathlib.Path | None,
) -> list[RichSegment]:
    """Attach chat_spike_score, chat_message_count, chat_category to each segment.

    Safe: returns segments unchanged on any error or missing chat data.
    """
    if not segments:
        return list(segments)
    if not chat_path or not chat_path.exists():
        return list(segments)

    try:
        messages = json.loads(chat_path.read_text(encoding="utf-8"))
    except Exception:
        return list(segments)

    if not messages:
        return list(segments)

    per_sec = _build_per_second_counts(messages)
    spike_scores = _compute_spike_scores(per_sec)

    result: list[RichSegment] = []
    for seg in segments:
        start = int(getattr(seg, "start", 0.0) or 0.0)
        end = int(getattr(seg, "end", 0.0) or 0.0)
        # Extend the lookup window forward to catch chat reactions that arrive
        # after the moment (viewers type ~5-10s after what they see).
        chat_end = end + CHAT_REACTION_DELAY_SEC

        # message count for this window (raw, no delay extension — counts what chat said during segment)
        count = sum(per_sec.get(s, 0) for s in range(start, max(start + 1, end)))
        # max spike score over the delay-extended window
        spike = max((spike_scores.get(s, 0.0) for s in range(start, max(start + 1, chat_end))), default=0.0)
        # dominant emotion category over the delay-extended window
        category = _dominant_category(messages, start, chat_end)

        result.append(
            seg.__class__(
                **{
                    **{
                        f.name: getattr(seg, f.name)
                        for f in seg.__dataclass_fields__.values()  # type: ignore[attr-defined]
                    },
                    "chat_spike_score": round(min(spike, 1.0), 4),
                    "chat_message_count": count,
                    "chat_category": category,
                }
            )
        )
    return result


def _build_per_second_counts(messages: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for msg in messages:
        t = msg.get("t")
        if isinstance(t, int):
            counts[t] = counts.get(t, 0) + 1
    return counts


def _compute_spike_scores(per_sec: dict[int, int]) -> dict[int, float]:
    if not per_sec:
        return {}
    all_times = sorted(per_sec)
    scores: dict[int, float] = {}
    half_window = CHAT_SPIKE_BASELINE_WINDOW_SEC // 2
    for t in all_times:
        window_rates = [
            per_sec[s]
            for s in range(t - half_window, t + half_window + 1)
            if s in per_sec
        ]
        if len(window_rates) < 3:
            scores[t] = 0.0
            continue
        baseline = statistics.median(window_rates)
        rate = per_sec[t]
        # Normalise: ratio = rate / (baseline + eps), saturate at SATURATION_RATIO
        ratio = rate / (baseline + 1e-6)
        scores[t] = min(ratio / CHAT_SPIKE_SATURATION_RATIO, 1.0)
    return scores


def _dominant_category(messages: list[dict], start: int, end: int) -> str:
    hype = funny = shock = 0
    for msg in messages:
        t = msg.get("t")
        if not isinstance(t, int) or not (start <= t < max(start + 1, end)):
            continue
        text_lower = msg.get("text", "").lower()
        emotes_lower = [e.lower() for e in msg.get("emotes", [])]
        tokens = set(text_lower.split()) | set(emotes_lower)
        hype += bool(tokens & _HYPE_TOKENS)
        funny += bool(tokens & _FUNNY_TOKENS)
        shock += bool(tokens & _SHOCK_TOKENS)
    total = hype + funny + shock
    if total == 0:
        return ""
    # Require at least 2 matching messages to assign a category
    best_count = max(hype, funny, shock)
    if best_count < 2:
        return ""
    if hype == best_count:
        return "hype"
    if funny == best_count:
        return "funny"
    return "shock"
