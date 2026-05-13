"""
Chat-replay analysis: activity density, spike detection, and sentiment classification.

Downloads Twitch VOD chat replay directly from Twitch's GraphQL comments API,
aggregates messages into per-second activity frames, attaches density/rate
signals, detects message-rate spikes above a rolling baseline, and classifies
the dominant sentiment (hype / laugh / shock / sadge).  The result is a
`chat_spike_score` plus density fields on each RichSegment and optional
emotion-category tags — both feed into candidate features.

Sentiment classification uses both platform-specific emote names and universal
text patterns so it degrades gracefully when emote metadata is unavailable.

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
import time
from dataclasses import dataclass, field
from typing import Any
from urllib import error, parse, request

from config import settings
from services.transcriber import RichSegment

logger = logging.getLogger(__name__)

_DEFAULT_TWITCH_WEB_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
_TWITCH_GRAPHQL_API = "https://gql.twitch.tv/gql"
_TWITCH_COMMENTS_QUERY = """
query VideoCommentsByOffsetOrCursor($videoID: ID!, $contentOffsetSeconds: Int) {
  video(id: $videoID) {
    comments(contentOffsetSeconds: $contentOffsetSeconds, first: 100) {
      edges {
        node {
          id
          contentOffsetSeconds
          commenter {
            displayName
            login
          }
          message {
            fragments {
              text
              emote {
                emoteID
                id
              }
            }
          }
        }
      }
      pageInfo {
        hasNextPage
      }
    }
  }
}
""".strip()
_CHAT_DENSITY_REFERENCE_MSG_PER_SEC = 5.0
_CHAT_EMOTE_REFERENCE_WEIGHT_PER_SEC = 3.0


def _is_twitch_vod_url(url: str) -> bool:
    return _twitch_vod_id(url) is not None


def _twitch_vod_id(url: str) -> str | None:
    parsed = parse.urlparse(url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "twitch.tv":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0].lower() == "videos" and parts[1].isdigit():
        return parts[1]
    return None


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
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    rate_derivative: float = 0.0
    ema_crossing_score: float = 0.0


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

class _TwitchRechatUnavailable(Exception):
    """Raised when Twitch reports that the replay comments are unavailable."""


def _twitch_headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Client-ID": _DEFAULT_TWITCH_WEB_CLIENT_ID,
        "User-Agent": "Mozilla/5.0",
    }


def _fetch_graphql_comments(
    vod_id: str,
    *,
    job_id: str,
    offset_seconds: int,
    headers: dict[str, str],
    timeout_sec: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    offset_seconds = max(int(offset_seconds), 0)
    logger.info(
        "twitch_graphql_rechat_request_start job_id=%s vod_id=%s offset=%s timeout_sec=%.1f client_id_source=web",
        job_id,
        vod_id,
        offset_seconds,
        timeout_sec,
    )
    payload = [
        {
            "operationName": "VideoCommentsByOffsetOrCursor",
            "variables": {
                "videoID": str(vod_id),
                "contentOffsetSeconds": offset_seconds,
            },
            "query": _TWITCH_COMMENTS_QUERY,
        }
    ]
    req = request.Request(
        _TWITCH_GRAPHQL_API,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            status = int(getattr(response, "status", 200))
            body = response.read()
    except error.HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        if exc.code == 404:
            logger.info(
                "twitch_graphql_rechat_unavailable job_id=%s vod_id=%s offset=%s status=%s reason=not_found elapsed_ms=%s",
                job_id,
                vod_id,
                offset_seconds,
                exc.code,
                elapsed_ms,
            )
            raise _TwitchRechatUnavailable("twitch_graphql_404") from exc
        try:
            error_body = exc.read(1000).decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        logger.warning(
            "twitch_graphql_rechat_http_error job_id=%s vod_id=%s offset=%s status=%s reason=%s body=%s elapsed_ms=%s",
            job_id,
            vod_id,
            offset_seconds,
            exc.code,
            exc.reason,
            error_body,
            elapsed_ms,
        )
        raise ValueError(
            f"Twitch GraphQL HTTP {exc.code}: {error_body or exc.reason}"
        ) from exc
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        "twitch_graphql_rechat_response job_id=%s vod_id=%s offset=%s status=%s bytes=%s elapsed_ms=%s",
        job_id,
        vod_id,
        offset_seconds,
        status,
        len(body),
        elapsed_ms,
    )
    if status == 404:
        logger.info(
            "twitch_graphql_rechat_unavailable job_id=%s vod_id=%s offset=%s status=%s reason=not_found elapsed_ms=%s",
            job_id,
            vod_id,
            offset_seconds,
            status,
            elapsed_ms,
        )
        raise _TwitchRechatUnavailable("twitch_graphql_404")
    if status >= 400:
        raise error.HTTPError(_TWITCH_GRAPHQL_API, status, f"HTTP {status}", hdrs=None, fp=None)

    raw = json.loads(body.decode("utf-8", errors="replace"))
    if isinstance(raw, list) and raw:
        response_payload = raw[0]
    else:
        response_payload = raw
    if not isinstance(response_payload, dict):
        raise ValueError("Twitch GraphQL response is not a JSON object")

    errors = response_payload.get("errors")
    data = response_payload.get("data")
    if isinstance(data, dict):
        video = data.get("video")
    else:
        video = None
    if video is None:
        logger.info(
            "twitch_graphql_rechat_unavailable job_id=%s vod_id=%s offset=%s status=%s reason=video_missing",
            job_id,
            vod_id,
            offset_seconds,
            status,
        )
        raise _TwitchRechatUnavailable("twitch_graphql_video_missing")
    if isinstance(errors, list) and errors:
        messages = ", ".join(
            str(item.get("message") or "")
            for item in errors
            if isinstance(item, dict)
        )
        logger.warning(
            "twitch_graphql_rechat_payload_error job_id=%s vod_id=%s offset=%s status=%s errors=%s",
            job_id,
            vod_id,
            offset_seconds,
            status,
            messages,
        )
        raise ValueError(f"Twitch GraphQL returned errors: {messages}")
    if not isinstance(video, dict):
        raise ValueError("Twitch GraphQL video payload is not an object")
    comments = video.get("comments")
    if comments is None:
        logger.info(
            "twitch_graphql_rechat_unavailable job_id=%s vod_id=%s offset=%s status=%s reason=comments_missing",
            job_id,
            vod_id,
            offset_seconds,
            status,
        )
        raise _TwitchRechatUnavailable("twitch_graphql_comments_missing")
    if not isinstance(comments, dict):
        raise ValueError("Twitch GraphQL comments payload is not an object")
    return comments


def _comment_text(message: dict[str, Any]) -> str:
    body = message.get("body")
    if isinstance(body, str) and body.strip():
        return body.strip()
    fragments = message.get("fragments")
    if isinstance(fragments, list):
        return "".join(
            str(fragment.get("text", ""))
            for fragment in fragments
            if isinstance(fragment, dict)
        ).strip()
    return ""


def _comment_emotes(message: dict[str, Any]) -> list[str]:
    emotes: list[str] = []
    fragments = message.get("fragments")
    if not isinstance(fragments, list):
        return emotes
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        emoticon = fragment.get("emoticon")
        if not isinstance(emoticon, dict):
            emoticon = fragment.get("emote")
        if not isinstance(emoticon, dict):
            continue
        for key in ("emoticon_id", "emoteID", "id", "name"):
            value = emoticon.get(key)
            if isinstance(value, str) and value:
                emotes.append(value)
                break
        else:
            text = fragment.get("text")
            if isinstance(text, str) and text:
                emotes.append(text)
    return emotes


def _normalize_twitch_comment(comment: dict[str, Any]) -> dict[str, Any] | None:
    try:
        t = float(comment.get("content_offset_seconds", comment.get("contentOffsetSeconds", 0.0)))
    except (TypeError, ValueError):
        return None
    message = comment.get("message")
    if not isinstance(message, dict):
        return None
    text = _comment_text(message)
    if not text:
        return None
    commenter = comment.get("commenter")
    user = ""
    if isinstance(commenter, dict):
        display_name = commenter.get("display_name") or commenter.get("displayName")
        login = commenter.get("name") or commenter.get("login")
        user = str(display_name or login or "")
    row: dict[str, Any] = {
        "t": t,
        "text": text,
        "emotes": _comment_emotes(message),
    }
    comment_id = comment.get("_id") or comment.get("id")
    if comment_id:
        row["id"] = str(comment_id)
    if user:
        row["user"] = user
    return row


def _normalize_target_offsets(target_offsets: list[float] | tuple[float, ...] | None) -> list[int]:
    if not target_offsets:
        return []
    offsets: set[int] = set()
    for value in target_offsets:
        try:
            offset = int(float(value))
        except (TypeError, ValueError):
            continue
        offsets.add(max(offset, 0))
    return sorted(offsets)


def _fetch_twitch_rechat(
    vod_id: str,
    job_id: str,
    *,
    target_offsets: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    headers = _twitch_headers()
    per_request_timeout = max(1.0, min(float(settings.chat_download_timeout_sec), 15.0))
    deadline = time.monotonic() + max(float(settings.chat_download_timeout_sec), 1.0)
    offsets = _normalize_target_offsets(target_offsets)
    offset_index = 0
    current_offset = offsets[0] if offsets else 0
    seen_messages: set[tuple[str, float, str]] = set()
    messages: list[dict[str, Any]] = []
    partial = False
    page_count = 0
    mode = "target_offsets" if offsets else "crawl"
    attempted_offsets: list[int] = []
    logger.info(
        "twitch_rechat_fetch_plan job_id=%s vod_id=%s mode=%s target_offsets=%s first_offset=%s last_offset=%s timeout_sec=%s per_request_timeout_sec=%.1f",
        job_id,
        vod_id,
        mode,
        len(offsets),
        offsets[0] if offsets else current_offset,
        offsets[-1] if offsets else current_offset,
        settings.chat_download_timeout_sec,
        per_request_timeout,
    )

    while True:
        if time.monotonic() >= deadline:
            partial = True
            logger.warning(
                "twitch_rechat_fetch_timeout job_id=%s vod_id=%s pages=%s messages=%s timeout_sec=%s",
                job_id,
                vod_id,
                page_count,
                len(messages),
                settings.chat_download_timeout_sec,
            )
            break

        payload = _fetch_graphql_comments(
            vod_id,
            job_id=job_id,
            offset_seconds=current_offset,
            headers=headers,
            timeout_sec=per_request_timeout,
        )
        attempted_offsets.append(int(current_offset))
        page_count += 1
        edges = payload.get("edges")
        if not isinstance(edges, list):
            raise ValueError("Twitch GraphQL rechat response is missing edges list")

        page_messages: list[dict[str, Any]] = []
        page_new_messages = 0
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            comment = edge.get("node")
            if not isinstance(comment, dict):
                continue
            normalized = _normalize_twitch_comment(comment)
            if normalized is None:
                continue
            page_messages.append(normalized)
            dedupe_id = str(normalized.get("id", ""))
            dedupe_key = (dedupe_id, float(normalized["t"]), str(normalized["text"]))
            if dedupe_key in seen_messages:
                continue
            seen_messages.add(dedupe_key)
            messages.append(normalized)
            page_new_messages += 1

        page_info = payload.get("pageInfo")
        has_next = bool(page_info.get("hasNextPage")) if isinstance(page_info, dict) else False
        first_page_t = min((float(row.get("t", 0.0)) for row in page_messages), default=0.0)
        last_page_t = max((float(row.get("t", 0.0)) for row in page_messages), default=0.0)
        logger.info(
            "twitch_rechat_fetch_page job_id=%s vod_id=%s page=%s offset=%s edges=%s normalized=%s new_messages=%s total_messages=%s first_t=%.3f last_t=%.3f has_next=%s mode=%s",
            job_id,
            vod_id,
            page_count,
            current_offset,
            len(edges),
            len(page_messages),
            page_new_messages,
            len(messages),
            first_page_t,
            last_page_t,
            has_next,
            mode,
        )

        if offsets:
            offset_index += 1
            if offset_index >= len(offsets):
                break
            current_offset = offsets[offset_index]
            continue

        if not has_next or not page_messages:
            logger.info(
                "twitch_rechat_fetch_stop job_id=%s vod_id=%s reason=%s page=%s offset=%s messages=%s has_next=%s",
                job_id,
                vod_id,
                "empty_page" if not page_messages else "no_next_page",
                page_count,
                current_offset,
                len(messages),
                has_next,
            )
            break
        next_offset = int(max(float(row.get("t", 0.0)) for row in page_messages)) + 1
        if next_offset <= current_offset:
            logger.warning(
                "twitch_rechat_fetch_stopped job_id=%s vod_id=%s reason=no_offset_progress offset=%s",
                job_id,
                vod_id,
                current_offset,
            )
            partial = True
            break
        current_offset = next_offset

    messages.sort(key=lambda row: float(row.get("t", 0.0)))
    logger.info(
        "twitch_rechat_fetch_done job_id=%s vod_id=%s status=%s pages=%s messages=%s first_t=%.3f last_t=%.3f mode=%s",
        job_id,
        vod_id,
        "partial" if partial else "complete",
        page_count,
        len(messages),
        float(messages[0].get("t", 0.0)) if messages else 0.0,
        float(messages[-1].get("t", 0.0)) if messages else 0.0,
        mode,
    )
    missing_offsets = [offset for offset in offsets if offset not in set(attempted_offsets)]
    fetch_metadata = {
        "mode": mode,
        "target_offsets": offsets,
        "attempted_offsets": attempted_offsets,
        "covered_target_offsets": attempted_offsets if offsets else [],
        "missing_target_offsets": missing_offsets,
        "partial": bool(partial or missing_offsets),
        "pages": page_count,
        "messages": len(messages),
        "first_t": float(messages[0].get("t", 0.0)) if messages else None,
        "last_t": float(messages[-1].get("t", 0.0)) if messages else None,
    }
    return messages, partial, fetch_metadata


def download_chat(
    url: str,
    job_id: str,
    out_dir: pathlib.Path,
    *,
    target_offsets: list[float] | tuple[float, ...] | None = None,
) -> pathlib.Path | None:
    """
    Download Twitch VOD chat replay directly from Twitch.

    Writes a normalized JSON chat export to *out_dir* and returns its path.
    Returns None on any failure (network error, no chat available, unsupported
    URL) so the rest of the pipeline is unaffected.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vod_id = _twitch_vod_id(url)
    if vod_id is None:
        logger.info(
            "chat_download_skipped job_id=%s reason=unsupported_url url=%s",
            job_id,
            url,
        )
        return None

    normalized_offsets = _normalize_target_offsets(target_offsets)
    logger.info(
        "twitch_chat_download_start job_id=%s vod_id=%s url=%s out_dir=%s target_offsets=%s first_offset=%s last_offset=%s",
        job_id,
        vod_id,
        url,
        out_dir,
        len(normalized_offsets),
        normalized_offsets[0] if normalized_offsets else 0,
        normalized_offsets[-1] if normalized_offsets else 0,
    )
    try:
        messages, partial, fetch_metadata = _fetch_twitch_rechat(vod_id, job_id, target_offsets=target_offsets)
    except _TwitchRechatUnavailable as exc:
        reason = str(exc) or "missing_rechat"
        _write_fetch_metadata(
            out_dir,
            {
                "mode": "target_offsets" if normalized_offsets else "crawl",
                "target_offsets": normalized_offsets,
                "attempted_offsets": [],
                "covered_target_offsets": [],
                "missing_target_offsets": normalized_offsets,
                "partial": True,
                "pages": 0,
                "messages": 0,
                "unavailable": True,
                "reason": reason,
            },
        )
        logger.info(
            "twitch_chat_download_unavailable job_id=%s vod_id=%s reason=%s",
            job_id,
            vod_id,
            reason,
        )
        return None
    except (TimeoutError, error.URLError, error.HTTPError, json.JSONDecodeError, ValueError) as exc:
        logger.exception(
            "twitch_chat_download_failed job_id=%s vod_id=%s url=%s error_type=%s error=%s",
            job_id,
            vod_id,
            url,
            type(exc).__name__,
            exc,
        )
        return None

    if not messages:
        logger.info(
            "twitch_chat_download_empty job_id=%s vod_id=%s reason=no_messages_after_fetch",
            job_id,
            vod_id,
        )
        return None

    chat_path = out_dir / "chat.twitch_rechat.json"
    chat_path.write_text(json.dumps(messages, ensure_ascii=True), encoding="utf-8")
    _write_fetch_metadata(out_dir, fetch_metadata)
    first_t = float(messages[0].get("t", 0.0))
    last_t = float(messages[-1].get("t", 0.0))
    logger.info(
        "twitch_chat_download_success job_id=%s vod_id=%s chat_path=%s messages=%s first_t=%.3f last_t=%.3f partial=%s",
        job_id,
        vod_id,
        chat_path,
        len(messages),
        first_t,
        last_t,
        partial,
    )
    return chat_path


def _write_fetch_metadata(out_dir: pathlib.Path, payload: dict[str, Any]) -> None:
    metadata_path = out_dir / "chat.twitch_rechat.fetch_metadata.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_messages(path: pathlib.Path) -> list[dict]:
    """
    Parse a chat file into a flat list of ``{t, text, emotes}`` dicts.

    Handles two formats:
    - normalized direct Twitch export: ``[{t, text, emotes}]``
    - legacy json3-style export: ``{"events": [{tStartMs, segs: [{utf8}]}]}``
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
        # Normalized direct export, or Twitch native API format.
        for msg in raw:
            if not isinstance(msg, dict):
                continue
            if "t" in msg or "text" in msg:
                try:
                    t = float(msg.get("t", 0.0))
                except (TypeError, ValueError):
                    continue
                body = str(msg.get("text", ""))
                raw_emotes = msg.get("emotes", [])
                emotes = [str(emote) for emote in raw_emotes] if isinstance(raw_emotes, list) else []
                row = {"t": t, "text": body, "emotes": emotes}
                if msg.get("id"):
                    row["id"] = str(msg.get("id"))
                if msg.get("user"):
                    row["user"] = str(msg.get("user"))
                messages.append(row)
                continue
            try:
                t = float(msg.get("content_offset_seconds", 0))
            except (TypeError, ValueError):
                continue
            m = msg.get("message", {})
            if not isinstance(m, dict):
                continue
            body = _comment_text(m)
            emotes = _comment_emotes(m)
            row = {"t": t, "text": body, "emotes": emotes}
            if msg.get("_id") or msg.get("id"):
                row["id"] = str(msg.get("_id") or msg.get("id"))
            commenter = msg.get("commenter") if isinstance(msg.get("commenter"), dict) else {}
            if commenter.get("display_name") or commenter.get("name"):
                row["user"] = str(commenter.get("display_name") or commenter.get("name"))
            messages.append(row)
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
    # Dedup bot replays without collapsing real multi-user emote floods.
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict] = []
    for msg in messages:
        t_int = int(msg["t"])
        key = _chat_dedupe_key(msg, t_int)
        if key in seen:
            continue
        seen.add(key)
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

    sorted_frames = sorted(frames.values(), key=lambda f: f.t)
    _annotate_chat_ema(sorted_frames)
    return sorted_frames


def _annotate_chat_ema(frames: list[ChatFrame], *, fast_sec: int = 5, slow_sec: int = 30) -> None:
    if not frames:
        return
    frame_map = {int(frame.t): frame for frame in frames}
    alpha_fast = 2.0 / (max(int(fast_sec), 1) + 1.0)
    alpha_slow = 2.0 / (max(int(slow_sec), fast_sec + 1) + 1.0)
    fast = 0.0
    slow = 0.0
    prev_fast = 0.0
    prev_slow = 0.0
    for second in range(int(frames[0].t), int(frames[-1].t) + 1):
        frame = frame_map.get(second)
        rate = float(frame.msgs if frame else 0)
        prev_fast, prev_slow = fast, slow
        fast = alpha_fast * rate + (1.0 - alpha_fast) * fast
        slow = alpha_slow * rate + (1.0 - alpha_slow) * slow
        if frame is None:
            continue
        derivative = max(0.0, fast - prev_fast)
        crossing_margin = max(0.0, fast - slow)
        crossing_started = fast > slow and prev_fast <= prev_slow
        frame.ema_fast = round(fast, 3)
        frame.ema_slow = round(slow, 3)
        frame.rate_derivative = round(derivative, 3)
        frame.ema_crossing_score = round(
            min(crossing_margin / max(slow, 1.0) + (0.25 if crossing_started else 0.0), 1.0),
            3,
        )


def _chat_dedupe_key(msg: dict, t_int: int) -> tuple[str, ...]:
    message_id = str(msg.get("id") or "").strip()
    if message_id:
        return ("id", message_id)
    text = str(msg.get("text") or "")
    user = str(msg.get("user") or "").strip()
    if user:
        return ("user_text", str(t_int), user, text)
    return ("legacy_text", str(t_int), text)


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

    frame_map = {int(frame.t): frame for frame in frames}
    for seg in segments:
        _attach_chat_activity(seg, frame_map)

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


def _attach_chat_activity(segment: RichSegment, frame_map: dict[int, ChatFrame]) -> None:
    start = float(segment.start)
    end = float(segment.end)
    if end <= start:
        return
    frames = [
        frame
        for second, frame in frame_map.items()
        if float(second) + 1.0 > start and float(second) < end
    ]
    if not frames:
        return

    duration = max(end - start, 1e-6)
    message_count = sum(frame.msgs for frame in frames)
    emote_weight = sum(frame.emote_weight for frame in frames)
    message_rate = message_count / duration
    emote_rate = emote_weight / duration
    density_score = _activity_score(message_rate, _CHAT_DENSITY_REFERENCE_MSG_PER_SEC)
    emote_score = _activity_score(emote_rate, _CHAT_EMOTE_REFERENCE_WEIGHT_PER_SEC)

    segment.chat_message_count = int(message_count)
    segment.chat_message_rate = round(message_rate, 3)
    segment.chat_density_score = round(max(density_score, emote_score * 0.75), 3)
    segment.chat_emote_score = round(emote_score, 3)
    segment.chat_ema_rate = round(max((frame.ema_fast for frame in frames), default=0.0), 3)
    segment.chat_rate_derivative = round(max((frame.rate_derivative for frame in frames), default=0.0), 3)
    segment.chat_ema_crossing_score = round(max((frame.ema_crossing_score for frame in frames), default=0.0), 3)

    category_weights = _activity_category_scores(frames)
    segment.chat_hype_score = round(category_weights["hype"] * segment.chat_density_score, 3)
    segment.chat_laugh_score = round(category_weights["laugh"] * segment.chat_density_score, 3)
    segment.chat_shock_score = round(category_weights["shock"] * segment.chat_density_score, 3)
    segment.chat_sadge_score = round(category_weights["sadge"] * segment.chat_density_score, 3)

    dominant_cat = _dominant_activity_category(frames)
    if dominant_cat != "neutral" and max(segment.chat_density_score, segment.chat_emote_score) > 0.05:
        cat_tag = f"chat_{dominant_cat}"
        if cat_tag not in segment.emotions:
            segment.emotions.append(cat_tag)


def _activity_score(rate: float, reference_rate: float) -> float:
    if rate <= 0.0:
        return 0.0
    return min(rate / max(reference_rate, 1e-6), 1.0)


def _dominant_activity_category(frames: list[ChatFrame]) -> str:
    categories: dict[str, int] = {}
    for frame in frames:
        for category, count in frame.categories.items():
            if category == "neutral":
                continue
            categories[category] = categories.get(category, 0) + int(count)
    if not categories:
        return "neutral"
    best = max(categories, key=lambda key: categories[key])
    return best if categories[best] > 0 else "neutral"


def _activity_category_scores(frames: list[ChatFrame]) -> dict[str, float]:
    categories = {"hype": 0.0, "laugh": 0.0, "shock": 0.0, "sadge": 0.0}
    total = 0.0
    for frame in frames:
        for category in categories:
            count = float(frame.categories.get(category, 0) or 0.0)
            categories[category] += count
            total += count
    if total <= 0.0:
        return categories
    return {category: min(value / total, 1.0) for category, value in categories.items()}
