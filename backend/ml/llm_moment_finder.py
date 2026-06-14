from __future__ import annotations

import json
import pathlib
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings


DEFAULT_MAX_MOMENTS = 180
# 64000 (was 32000): a 45-min/180-moment chunk's JSON can exceed the budget and abort
# mid-moment — the #56 `truncation` hard_miss (VOD 2769594798 chunks hit completion
# 31990 ≈ 32000). 64000 adds headroom (no model swap) but does NOT eliminate
# truncation: a live re-gen (#68) showed dense chunks fill 64000 too (completion
# 63990) — max_moments=180 scales output to any budget. The model-independent cure is
# fewer moments per chunk (chunk_duration/max_moments); out of scope for #68's
# floor-only quick-win, so 64000 stays as net-positive headroom, not a proven fix.
DEFAULT_MAX_TOKENS = 64000
# Long VODs are split into time windows so each LLM call gets its own output-token
# budget — a single call truncates at ~max_tokens and silently drops the back of a
# multi-hour stream. ~45 min/window keeps each call's output under the ceiling.
DEFAULT_CHUNK_DURATION_SEC = 2700.0
DEFAULT_CHUNK_OVERLAP_SEC = 120.0
DEFAULT_TIMEOUT_SEC = 900.0
DEFAULT_API_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SEC = 20.0
DEFAULT_MIN_DURATION_SEC = 8.0
DEFAULT_MAX_DURATION_SEC = 90.0
DEFAULT_BLOB_DROP_DURATION_SEC = 300.0
DEFAULT_PROVIDER = "gemini"
PROVIDERS = {"anthropic", "gemini"}
DEFAULT_GEMINI_MODEL = str(getattr(settings, "gemini_model", "") or "gemini-2.5-flash-lite")
DEFAULT_ANTHROPIC_MODEL = str(getattr(settings, "anthropic_model_topk", "") or "claude-sonnet-4-6")
_FILLER_TOKENS = {
    "you", "yeah", "uh", "um", "huh", "oh", "ah", "okay", "ok", "music", "[music]",
}
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"



@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"
    chat_spike_score: float = 0.0
    chat_hype_score: float = 0.0
    chat_funny_score: float = 0.0
    chat_shock_score: float = 0.0
    voice_energy_burst_score: float = 0.0


@dataclass(frozen=True)
class PromptTranscript:
    text: str
    input_segments: int
    kept_segments: int
    dropped_segments: int
    input_chars: int
    kept_chars: int
    dropped_reasons: dict[str, int]


@dataclass(frozen=True)
class LlmMoment:
    start: float
    end: float
    hook: float
    flow: float
    value: float
    trend: float
    virality: float
    rationale: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class MomentLengthDisciplineResult:
    moments: list[LlmMoment]
    input_moment_count: int
    dropped_non_positive_duration_count: int
    capped_moment_count: int
    dropped_blob_moments: list[LlmMoment]
    max_duration_sec: float
    blob_drop_duration_sec: float


@dataclass(frozen=True)
class TimestampRepairResult:
    moments: list[LlmMoment]
    repaired_moment_count: int


def _build_chat_marker(segment: "TranscriptSegment") -> str:
    """Build inline annotations for a transcript line.

    Examples:
      chat_spike=0.82, dominant=hype, vol=0.9  → ' | chat=0.82 hype | vol=0.90'
      chat_spike=0.74, no class                → ' | chat=0.74'
      vol=0.85, no spike                       → ' | vol=0.85'
      all low                                  → ''
    """
    parts: list[str] = []

    spike = float(getattr(segment, "chat_spike_score", 0.0) or 0.0)
    # 0.25 instead of 0.4: the all_v5 miss analysis showed most ground-truth
    # viral windows spike at 0.23-0.38 — at 0.4 they carried no marker at all.
    if spike >= 0.25:
        hype  = float(getattr(segment, "chat_hype_score",  0.0) or 0.0)
        funny = float(getattr(segment, "chat_funny_score", 0.0) or 0.0)
        shock = float(getattr(segment, "chat_shock_score", 0.0) or 0.0)
        dominant_score = max(hype, funny, shock)
        if dominant_score >= 0.25:
            label = "hype" if dominant_score == hype else ("funny" if dominant_score == funny else "shock")
            parts.append(f"chat={spike:.2f} {label}")
        else:
            parts.append(f"chat={spike:.2f}")

    vol = float(getattr(segment, "voice_energy_burst_score", 0.0) or 0.0)
    if vol >= 0.5:
        parts.append(f"vol={vol:.2f}")

    return (" | " + " | ".join(parts)) if parts else ""


def _build_speaker_marker(segment: "TranscriptSegment") -> str:
    """Render the diarized speaker as ` | spk=<name>`.

    Empty for UNKNOWN (the un-diarized default) so a base transcript keeps the
    exact pre-diarization line format — the WALL-Re-Test (#92) base-vs-neu split
    stays attributable to the diarization lever, not a format drift.
    """
    speaker = str(getattr(segment, "speaker", "") or "").strip()
    if not speaker or speaker == "UNKNOWN":
        return ""
    return f" | spk={speaker}"


def normalize_provider(provider: str) -> str:
    normalized = str(provider or DEFAULT_PROVIDER).strip().lower()
    if normalized not in PROVIDERS:
        return DEFAULT_PROVIDER
    return normalized


def resolve_provider_model(provider: str, model: str = "") -> str:
    explicit_model = str(model or "").strip()
    if explicit_model:
        return explicit_model
    if normalize_provider(provider) == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    return DEFAULT_GEMINI_MODEL


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_clip_profile(profile_name: str) -> dict[str, Any] | None:
    """Load ml/artifacts/{profile_name}_profile.json. Returns None on missing/invalid file."""
    safe_name = re.sub(r"[^a-z0-9_-]", "", str(profile_name or "").strip().lower())
    if not safe_name:
        return None
    path = BACKEND_ROOT / "ml" / "artifacts" / f"{safe_name}_profile.json"
    return read_json(path)


def render_profile_rules(profile_spec: Mapping[str, Any]) -> str:
    """Format a loaded clip profile into bullet-point rules for the moment-finder prompt."""
    lines: list[str] = []

    archetypes = sorted(
        [a for a in (profile_spec.get("archetypes") or []) if isinstance(a, dict)],
        key=lambda a: float(a.get("weight") or 0),
        reverse=True,
    )
    if archetypes:
        parts: list[str] = []
        for arch in archetypes:
            name = str(arch.get("name") or "").strip()
            desc = str(arch.get("description") or "").strip()
            signals = [str(s) for s in (arch.get("transcript_signals") or []) if s]
            if not name:
                continue
            entry = f"{name}: {desc}"
            if signals:
                entry += f" (listen for: {'; '.join(signals)})"
            parts.append(entry)
        if parts:
            lines.append("- Optimize recall for these clip archetypes: " + " | ".join(parts) + ".")

    for rule in (profile_spec.get("selection_rules") or []):
        rule_str = str(rule).strip()
        if rule_str:
            lines.append(f"- {rule_str}")

    avoid_rules = [str(r).strip() for r in (profile_spec.get("avoid") or []) if str(r).strip()]
    if avoid_rules:
        lines.append("- Avoid: " + "; ".join(avoid_rules) + ".")

    return "\n".join(lines)


def format_transcript_for_prompt(
    segments: Sequence[TranscriptSegment],
    *,
    filter_noise: bool = True,
) -> PromptTranscript:
    lines: list[str] = []
    input_chars = 0
    kept_chars = 0
    dropped_reasons: Counter[str] = Counter()
    for segment in segments:
        text = " ".join(segment.text.split())
        input_chars += len(text)
        if not text:
            dropped_reasons["empty"] += 1
            continue
        drop_reason = transcript_noise_reason(segment, text) if filter_noise else None
        if drop_reason:
            dropped_reasons[drop_reason] += 1
            continue
        chat_marker = _build_chat_marker(segment)
        speaker_marker = _build_speaker_marker(segment)
        lines.append(
            "["
            f"t={format_seconds_for_prompt(segment.start)}s - t={format_seconds_for_prompt(segment.end)}s"
            f" | clock={format_time(segment.start)} - {format_time(segment.end)}"
            f"{speaker_marker}"
            f"{chat_marker}"
            f"] {text}"
        )
        kept_chars += len(text)
    return PromptTranscript(
        text="\n".join(lines),
        input_segments=len(segments),
        kept_segments=len(lines),
        dropped_segments=max(len(segments) - len(lines), 0),
        input_chars=input_chars,
        kept_chars=kept_chars,
        dropped_reasons=dict(sorted(dropped_reasons.items())),
    )


def split_segments_into_time_chunks(
    segments: Sequence[TranscriptSegment],
    *,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    chunk_overlap_sec: float = DEFAULT_CHUNK_OVERLAP_SEC,
) -> list[list[TranscriptSegment]]:
    """Split transcript segments into overlapping VOD-time windows.

    Each window spans ``chunk_duration_sec`` of VOD time plus a
    ``chunk_overlap_sec`` lead-in from the previous window, so a moment that
    straddles a boundary can still be captured in the later window (the
    resulting cross-chunk duplicate is removed afterwards by overlap dedupe).
    Returns a single window when the transcript is shorter than one window.
    """
    segs = list(segments)
    if not segs:
        return []
    t0 = min(float(s.start) for s in segs)
    t_end = max(float(s.end) for s in segs)
    if chunk_duration_sec <= 0 or (t_end - t0) <= chunk_duration_sec:
        return [segs]
    overlap = max(float(chunk_overlap_sec), 0.0)
    chunks: list[list[TranscriptSegment]] = []
    start = t0
    while start < t_end:
        end = start + chunk_duration_sec
        lead = start - overlap
        window = [s for s in segs if float(s.start) < end and float(s.end) > lead]
        if window:
            chunks.append(window)
        start = end
    return chunks or [segs]


def transcript_noise_reason(segment: TranscriptSegment, text: str) -> str | None:
    normalized = normalize_transcript_text(text)
    if not normalized:
        return "empty_after_normalize"
    tokens = normalized.split()
    duration = max(float(segment.end) - float(segment.start), 0.0)
    if len(tokens) == 1 and tokens[0] in _FILLER_TOKENS and duration <= 3.5:
        return "single_filler_token"
    if len(tokens) <= 3 and len(set(tokens)) == 1 and tokens[0] in _FILLER_TOKENS:
        return "repeated_filler_token"
    if len(tokens) <= 2 and normalized in {"thank you", "thanks", "bye", "subtitles"} and duration <= 3.5:
        return "short_boilerplate"
    return None


def normalize_transcript_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[\[\]\(\){}]", " ", lowered)
    lowered = re.sub(r"[^a-z0-9']+", " ", lowered)
    return " ".join(lowered.split())


def build_moment_finder_prompt(
    *,
    profile: str,
    vod_id: str,
    transcript_text: str,
    max_moments: int,
    min_duration_sec: float,
    max_duration_sec: float,
    profile_spec: Mapping[str, Any] | None = None,
) -> str:
    if profile_spec is not None:
        content_rules = render_profile_rules(profile_spec)
    else:
        content_rules = (
            "- Optimize recall for viral Jynxzi-style clips: rage, clutch, absurd chat reactions, "
            "fanmail/unboxing reveals, strong arguments, quotable hot takes, streamer shock, and clear payoff beats.\n"
            "- Penalize generic talking, long setup with no payoff, moments that need too much missing context, "
            "clipping only the punchline without its setup, and splitting one bit across multiple windows."
        )
    return f"""You are an expert short-form clip producer for Twitch creator {profile}.

Read the entire VOD transcript and find the strongest viral clip moments. This is a moment-finder pass, not a judge for preselected clips. Look for the moments an OpusClip-style system would surface: immediate hook, clear reaction or conflict, payoff, standalone value, short-form trend fit, replay value.

Return raw JSON only, no markdown. Use this exact shape:
{{
  "moments": [
    {{
      "start": <absolute_seconds_from_vod_start>,
      "end": <absolute_seconds_from_vod_start>,
      "hook": <0.0-1.0>,
      "flow": <0.0-1.0>,
      "value": <0.0-1.0>,
      "trend": <0.0-1.0>,
      "virality": <0.0-1.0>,
      "rationale": "<one short sentence>"
    }}
  ]
}}

Rules:
- Segments annotated with `chat=N.NN` had a live audience spike (message-rate N.NN × above local baseline, scale 0–1). Treat these as strong evidence that something viral happened — weigh them heavily. The optional label after the score tells you the *kind* of reaction the audience had: `hype` = clutch/win/pop-off; `funny` = comedy/laughing moment; `shock` = rage/meltdown/drama. Use the label to inform the clip archetype and score — e.g. a `funny` spike raises `hook` and `value` for a comedy clip, a `shock` spike raises `virality` for a rage/drama clip. A segment with no `chat=` annotation had a normal or low chat rate.
- Segments with `chat=` ≥ 0.40 are strong recall signals — the live audience voted in real time. For each such cluster, you should almost always include a moment that covers the content that triggered the spike, even if the transcript text seems brief, confusing, or context-free. Anchor the moment so it starts just before the spike and ends after the audience reaction dies down. Use your normal virality-scoring judgment — do not artificially inflate or cap scores for spike regions.
- Segments with `chat=` between 0.25 and 0.40 mark mildly elevated chat activity. Many proven viral moments only reach this band, so treat such a marker as a recall nudge: when the surrounding transcript shows any reaction, reveal, joke, or conflict, prefer including a borderline moment there over skipping it.
- Segments annotated with `vol=N.NN` had a significant audio energy burst (relative loudness N.NN × above the local audio baseline, scale 0–1). This signals a shout, rage outburst, or sudden audio peak. A `vol=` burst on its own (without `chat=`) is still meaningful — it often marks the onset of a reaction before the audience has typed. A combined `chat=... shock` + `vol=` is a near-certain rage/meltdown moment.
- Return at most {max_moments} moments.
- **Virality scoring calibration:** Use the full 0.0–1.0 scale. Most moments should score 0.60–0.80. Reserve 0.85–0.95 for genuinely exceptional moments: rare verbal tics, massive chat spikes (≥ 0.50), emotional breakdowns, shocking reveals, or moments that match one of the priority archetypes above. Do not cluster moments at the same value — assign each score independently based on its own merit.
- Treat one continuous bit (setup -> reaction -> payoff) as ONE moment only if its beats are within ~30s of each other. Never merge two reactions into one window just because they are thematically related.
- If two distinct reactions are more than ~30s apart, return them as SEPARATE moments even if they share the same theme or game context.
- Each moment should be between {min_duration_sec:.0f} and {max_duration_sec:.0f} seconds. Capture the whole arc within this range; do not trim a clip down to only the punchline.
- Use the `t=...s` absolute second values from the transcript brackets. Do not copy display clock values like `19:44` as `1944`; `19:44` means `1184` seconds.
- Start a few seconds before the hook, on the segment that sets it up (the line that makes the payoff land), and end on the LAST reaction/payoff segment of the bit, not the first one.
- Do not use rounded minute boundaries like 3600, 4980, or 8100 unless the transcript itself has that exact boundary.
- Prefer one self-contained clip per distinct moment over many fragments of the same moment. If the entire payoff fits in 10–20s (a verbal tic, a visual reveal, a single reaction), return it at that length — do not pad it. If a moment requires setup + payoff to land, extend to 25–60s. Let the content dictate the length.
- Err on the side of including a borderline moment — it is better to return a candidate that might not be viral than to miss one that is. A later ranking stage handles precision.
- Scan the **entire** transcript from start to finish before you output anything. Do not stop generating candidates early. The best moments can appear anywhere — early, mid, or late in a multi-hour stream.
{content_rules}

VOD ID: {vod_id}

Transcript:
{transcript_text}
"""


def transcript_excerpt(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
    *,
    pad_sec: float,
    max_chars: int,
) -> str:
    lower = float(start) - float(pad_sec)
    upper = float(end) + float(pad_sec)
    lines: list[str] = []
    for segment in segments:
        if segment.end < lower:
            continue
        if segment.start > upper:
            break
        lines.append(f"[{format_time(segment.start)} - {format_time(segment.end)}] {segment.text}")
    text = " ".join(" ".join(lines).split())
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 3, 0)].rstrip() + "..."


def call_moment_finder_provider(
    prompt: str,
    *,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    retries: int = DEFAULT_API_RETRIES,
    retry_backoff_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
) -> tuple[str, dict[str, Any]]:
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "gemini":
        return call_gemini_moment_finder(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
    return call_anthropic_moment_finder(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
    )


def call_gemini_moment_finder(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    retries: int = DEFAULT_API_RETRIES,
    retry_backoff_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
) -> tuple[str, dict[str, Any]]:
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is required for the moment finder.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for Gemini's OpenAI-compatible endpoint.") from exc

    client = OpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_BASE_URL, timeout=timeout_sec if timeout_sec > 0 else None)
    attempts = max(int(retries or 0), 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You return strict JSON for clip-finding."},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                response_format={"type": "json_object"},
            )
            content = str(response.choices[0].message.content or "").strip()
            usage = openai_usage_to_dict(getattr(response, "usage", None))
            return content, usage
        except Exception as exc:
            if not is_retryable_llm_error(exc) or attempt >= attempts:
                raise RuntimeError(f"Gemini moment finder call failed after {attempt} attempt(s): {type(exc).__name__}: {exc}") from exc
            delay = float(retry_backoff_sec) * attempt
            print_retry_event(provider="gemini", attempt=attempt, attempts=attempts, timeout_sec=timeout_sec, retry_after_sec=delay, error=exc)
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("Gemini moment finder call failed before receiving a response.")


def call_anthropic_moment_finder(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    retries: int = DEFAULT_API_RETRIES,
    retry_backoff_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
) -> tuple[str, dict[str, Any]]:
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is required for the moment finder.")
    # Opus 4.7+ removed temperature/top_p/top_k as API params
    _no_sampling = model.startswith(("claude-opus-4-7", "claude-opus-4-8"))
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": [
            {
                "type": "text",
                "text": "You return strict JSON for clip-finding.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    if not _no_sampling:
        payload["temperature"] = float(temperature)
    request = urllib.request.Request(
        ANTHROPIC_MESSAGES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": str(settings.anthropic_api_key),
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    attempts = max(int(retries or 0), 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec if timeout_sec > 0 else None) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"Anthropic moment finder call failed: status={exc.code} body={body[:500]}") from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            if attempt >= attempts:
                raise RuntimeError(f"Anthropic moment finder call failed after {attempts} attempts: {type(exc).__name__}: {exc}") from exc
            delay = float(retry_backoff_sec) * attempt
            print_retry_event(provider="anthropic", attempt=attempt, attempts=attempts, timeout_sec=timeout_sec, retry_after_sec=delay, error=exc)
            if delay > 0:
                time.sleep(delay)
    else:
        raise RuntimeError("Anthropic moment finder call failed before receiving a response.")
    text = anthropic_text_content(response_payload)
    usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
    return text, dict(usage or {})


def print_retry_event(
    *,
    provider: str,
    attempt: int,
    attempts: int,
    timeout_sec: float,
    retry_after_sec: float,
    error: Exception,
) -> None:
    print(
        json.dumps(
            {
                "event": "llm_moment_finder_retry",
                "provider": normalize_provider(provider),
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "attempts": attempts,
                "timeout_sec": timeout_sec,
                "retry_after_sec": retry_after_sec,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout, urllib.error.URLError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ["timeout", "rate limit", "rate_limit", "temporarily unavailable", "503"])


def openai_usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())
    payload: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            payload[key] = value
    return payload


def anthropic_text_content(response_payload: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    for block in response_payload.get("content") or []:
        if isinstance(block, Mapping) and block.get("type") == "text":
            chunks.append(str(block.get("text") or ""))
    return "\n".join(chunks).strip()


def parse_moments(text: str) -> list[LlmMoment]:
    payload = parse_json_payload(text)
    raw_moments = payload.get("moments") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_moments, list):
        raise ValueError("LLM response does not contain a moments list.")
    moments: list[LlmMoment] = []
    for raw in raw_moments:
        if not isinstance(raw, Mapping):
            continue
        start = float_or_none(raw.get("start"))
        end = float_or_none(raw.get("end"))
        if start is None or end is None or end <= start:
            continue
        moments.append(
            LlmMoment(
                start=round(float(start), 3),
                end=round(float(end), 3),
                hook=score01(raw.get("hook")),
                flow=score01(raw.get("flow")),
                value=score01(raw.get("value")),
                trend=score01(raw.get("trend")),
                virality=score01(raw.get("virality")),
                rationale=str(raw.get("rationale") or "").strip(),
            )
        )
    return moments


def repair_clock_encoded_timestamps(
    moments: Sequence[LlmMoment],
    *,
    max_time_sec: float | None = None,
) -> TimestampRepairResult:
    repaired: list[LlmMoment] = []
    repaired_count = 0
    for moment in moments:
        repaired_moment = repair_clock_encoded_moment(moment, max_time_sec=max_time_sec)
        if repaired_moment != moment:
            repaired_count += 1
        repaired.append(repaired_moment)
    return TimestampRepairResult(moments=repaired, repaired_moment_count=repaired_count)


def repair_clock_encoded_moment(moment: LlmMoment, *, max_time_sec: float | None = None) -> LlmMoment:
    converted_start = compact_clock_to_seconds(moment.start)
    converted_end = compact_clock_to_seconds(moment.end)
    if converted_start is None or converted_end is None or converted_end <= converted_start:
        return moment
    # When the VOD length is known, only treat a value as clock-encoded if the
    # literal seconds are IMPOSSIBLE (beyond the VOD). Otherwise a legitimate
    # large absolute-second position — common once long VODs are chunked — would
    # be wrongly shrunk (e.g. 2600s misread as 26:00 = 1560s).
    if max_time_sec is not None and float(moment.start) <= float(max_time_sec) + 1.0:
        return moment
    return LlmMoment(
        start=round(float(converted_start), 3),
        end=round(float(converted_end), 3),
        hook=moment.hook,
        flow=moment.flow,
        value=moment.value,
        trend=moment.trend,
        virality=moment.virality,
        rationale=moment.rationale,
    )


def compact_clock_to_seconds(value: float) -> float | None:
    if abs(float(value) - round(float(value))) > 0.001:
        return None
    raw = int(round(float(value)))
    if raw < 1000:
        return None
    seconds = raw % 100
    if seconds >= 60:
        return None
    if raw >= 10000:
        hours = raw // 10000
        minutes = (raw // 100) % 100
        if minutes >= 60:
            return None
        converted = float((hours * 3600) + (minutes * 60) + seconds)
    else:
        minutes = raw // 100
        converted = float((minutes * 60) + seconds)
    if abs(converted - float(value)) < 0.001:
        return None
    return converted


def enforce_moment_length_discipline(
    moments: Sequence[LlmMoment],
    *,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    blob_drop_duration_sec: float = DEFAULT_BLOB_DROP_DURATION_SEC,
) -> MomentLengthDisciplineResult:
    max_duration = max(float(max_duration_sec or 0.0), 0.001)
    blob_drop_duration = max(float(blob_drop_duration_sec or 0.0), max_duration)
    kept: list[LlmMoment] = []
    dropped_blob_moments: list[LlmMoment] = []
    dropped_non_positive_duration_count = 0
    capped_moment_count = 0
    for moment in moments:
        duration = moment.duration
        if duration <= 0.0:
            dropped_non_positive_duration_count += 1
            continue
        if duration > blob_drop_duration:
            dropped_blob_moments.append(moment)
            continue
        if duration > max_duration:
            capped_moment_count += 1
            kept.append(LlmMoment(
                start=moment.start,
                end=moment.start + max_duration,
                hook=moment.hook,
                flow=moment.flow,
                value=moment.value,
                trend=moment.trend,
                virality=moment.virality,
                rationale=moment.rationale,
            ))
            continue
        kept.append(moment)
    return MomentLengthDisciplineResult(
        moments=kept,
        input_moment_count=len(moments),
        dropped_non_positive_duration_count=dropped_non_positive_duration_count,
        capped_moment_count=capped_moment_count,
        dropped_blob_moments=dropped_blob_moments,
        max_duration_sec=max_duration,
        blob_drop_duration_sec=blob_drop_duration,
    )


def dedupe_moments_by_overlap(
    moments: Sequence[LlmMoment],
    *,
    overlap_ratio: float = 0.8,
    min_gap_sec: float = 30.0,
    min_gap_dedupe_duration_sec: float = 60.0,
) -> tuple[list[LlmMoment], list[LlmMoment]]:
    threshold = bounded_ratio(overlap_ratio)
    min_gap = max(float(min_gap_sec), 0.0)
    min_gap_dedupe_duration = max(float(min_gap_dedupe_duration_sec), 0.0)
    ordered = sorted(enumerate(moments), key=lambda item: (-item[1].virality, item[0]))
    kept: list[tuple[int, LlmMoment]] = []
    duplicates: list[LlmMoment] = []
    for original_index, moment in ordered:
        is_duplicate = any(
            moment_is_duplicate_of(moment, kept_moment, threshold=threshold, min_gap_sec=min_gap, min_gap_dedupe_duration_sec=min_gap_dedupe_duration)
            for _, kept_moment in kept
        )
        if is_duplicate:
            duplicates.append(moment)
            continue
        kept.append((original_index, moment))
    return [moment for _, moment in kept], duplicates


def moment_is_duplicate_of(
    moment: LlmMoment,
    other: LlmMoment,
    *,
    threshold: float,
    min_gap_sec: float,
    min_gap_dedupe_duration_sec: float,
) -> bool:
    if window_overlap_ratio(moment.start, moment.end, other.start, other.end) >= threshold:
        return True
    if min(moment.duration, other.duration) < min_gap_dedupe_duration_sec:
        return False
    gap = max(max(moment.start, other.start) - min(moment.end, other.end), 0.0)
    return gap <= min_gap_sec


def moment_row_duration(row: Mapping[str, Any]) -> float | None:
    duration = float_or_none(row.get("duration"))
    if duration is not None:
        return float(duration)
    start = float_or_none(row.get("start"))
    end = float_or_none(row.get("end"))
    if start is None or end is None:
        return None
    return max(float(end) - float(start), 0.0)


def parse_json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    last_decode_error: json.JSONDecodeError | None = None

    def try_parse(candidate: str) -> Any | None:
        nonlocal last_decode_error
        try:
            return parse_json_with_repair(candidate)
        except json.JSONDecodeError as exc:
            last_decode_error = exc
            return None

    parsed = try_parse(stripped)
    if parsed is not None:
        return parsed
    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        parsed = try_parse(object_match.group(0))
        if parsed is not None:
            return parsed
    array_match = re.search(r"\[.*\]", stripped, flags=re.DOTALL)
    if array_match:
        parsed = try_parse(array_match.group(0))
        if parsed is not None:
            return parsed
    partial_payload = extract_partial_array_payload(stripped, "moments")
    if partial_payload is not None:
        return partial_payload
    if last_decode_error is not None:
        raise last_decode_error
    raise ValueError("No JSON object or array found in LLM response.")


def parse_json_with_repair(text: str) -> Any | None:
    candidates = [text, repair_common_json_breaks(text)]
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return None


def repair_common_json_breaks(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r"(\})(\s*)(\{)", r"\1,\2\3", repaired)
    repaired = re.sub(r"(\])(\s*)(\")", r"\1,\2\3", repaired)
    repaired = re.sub(r"(\})(\s*)(\")", r"\1,\2\3", repaired)
    repaired = re.sub(r"(\")(\s*\n\s*)(\")", r"\1,\2\3", repaired)
    return repaired


def extract_partial_array_payload(text: str, key: str) -> dict[str, list[dict[str, Any]]] | None:
    list_start = find_json_array_start(text, key)
    if list_start is None:
        return None
    raw_objects = balanced_json_objects_from_list(text, list_start)
    rows: list[dict[str, Any]] = []
    for raw_object in raw_objects:
        try:
            parsed = parse_json_with_repair(raw_object)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    if not rows:
        return None
    return {key: rows}


def find_json_array_start(text: str, key: str) -> int | None:
    escaped_key = re.escape(str(key))
    match = re.search(rf'"{escaped_key}"\s*:\s*\[', text)
    if not match:
        return None
    return match.end()


def balanced_json_objects_from_list(text: str, list_start: int) -> list[str]:
    objects: list[str] = []
    depth = 0
    start_index: int | None = None
    in_string = False
    escape_next = False
    for index in range(list_start, len(text)):
        char = text[index]
        if start_index is None:
            if char == "]":
                break
            if char == "{":
                start_index = index
                depth = 1
                in_string = False
                escape_next = False
            continue
        if in_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start_index : index + 1])
                start_index = None
    return objects


def format_time(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_seconds_for_prompt(seconds: float) -> str:
    return f"{max(float(seconds), 0.0):.3f}".rstrip("0").rstrip(".")


def score01(value: Any) -> float:
    number = float_or_none(value)
    if number is None:
        return 0.0
    if number > 1.0 and number <= 10.0:
        number = number / 10.0
    return bounded_ratio(number)


def bounded_ratio(value: Any) -> float:
    number = float_or_none(value)
    if number is None:
        return 0.0
    return max(0.0, min(float(number), 1.0))


def float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def window_overlap_ratio(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    left_end = max(float(left_end), float(left_start) + 0.001)
    right_end = max(float(right_end), float(right_start) + 0.001)
    overlap = max(0.0, min(left_end, right_end) - max(float(left_start), float(right_start)))
    shortest = max(0.001, min(left_end - float(left_start), right_end - float(right_start)))
    return overlap / shortest
