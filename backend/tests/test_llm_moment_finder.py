from __future__ import annotations

import json
import sys
import types

from ml import llm_moment_finder as d0


def test_format_transcript_filters_short_filler_noise() -> None:
    prompt_transcript = d0.format_transcript_for_prompt(
        [
            d0.TranscriptSegment(start=0.0, end=1.0, text="you"),
            d0.TranscriptSegment(start=2.0, end=4.0, text="you you you"),
            d0.TranscriptSegment(start=10.0, end=13.0, text="real reaction"),
        ],
        filter_noise=True,
    )

    assert "real reaction" in prompt_transcript.text
    assert "you you you" not in prompt_transcript.text
    assert prompt_transcript.kept_segments == 1
    assert prompt_transcript.dropped_segments == 2
    assert prompt_transcript.dropped_reasons == {
        "repeated_filler_token": 1,
        "single_filler_token": 1,
    }


def test_format_transcript_uses_absolute_seconds_as_authoritative_timestamp() -> None:
    prompt_transcript = d0.format_transcript_for_prompt(
        [
            d0.TranscriptSegment(start=1184.0, end=1186.0, text="Is this even legal?"),
        ],
        filter_noise=True,
    )

    assert "[t=1184s - t=1186s | clock=19:44 - 19:46] Is this even legal?" in prompt_transcript.text


def test_parse_moments_repairs_missing_commas_between_objects() -> None:
    moments = d0.parse_moments(
        """
        {
          "moments": [
            {"start": 100, "end": 120, "hook": 0.9, "flow": 0.8, "value": 0.9, "trend": 0.7, "virality": 0.95, "rationale": "first"}
            {"start": 300, "end": 330, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "second"}
          ]
        }
        """
    )

    assert [moment.start for moment in moments] == [100.0, 300.0]


def test_parse_moments_recovers_after_stray_character_between_objects() -> None:
    moments = d0.parse_moments(
        """
        {
          "moments": [
            {"start": 100, "end": 120, "hook": 0.9, "flow": 0.8, "value": 0.9, "trend": 0.7, "virality": 0.95, "rationale": "first"},
        s   {"start": 300, "end": 330, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "second"}
          ]
        }
        """
    )

    assert [moment.start for moment in moments] == [100.0, 300.0]


def test_parse_moments_recovers_complete_objects_from_truncated_response() -> None:
    moments = d0.parse_moments(
        """
        {
          "moments": [
            {"start": 100, "end": 120, "hook": 0.9, "flow": 0.8, "value": 0.9, "trend": 0.7, "virality": 0.95, "rationale": "first"},
            {"start": 300, "end": 330, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "second"},
            {"start": 500, "end": 530, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "unterminated
        """
    )

    assert [moment.start for moment in moments] == [100.0, 300.0]


def test_repair_clock_encoded_timestamps_converts_compact_mmss_without_rationale_matching() -> None:
    moments = [
        d0.LlmMoment(
            start=1938.0,
            end=2036.0,
            hook=0.8,
            flow=0.7,
            value=0.9,
            trend=0.7,
            virality=0.9,
            rationale="Streamer receives an airsoft shotgun as fan mail, initially mistaking it for a real firearm.",
        )
    ]

    repaired = d0.repair_clock_encoded_timestamps(moments)

    assert repaired.repaired_moment_count == 1
    assert repaired.moments[0].start == 1178.0
    assert repaired.moments[0].end == 1236.0


def test_repair_clock_encoded_timestamps_keeps_invalid_clock_values() -> None:
    moments = [
        d0.LlmMoment(
            start=1968.0,
            end=2036.0,
            hook=0.8,
            flow=0.7,
            value=0.9,
            trend=0.7,
            virality=0.9,
            rationale="Seconds part above 59 cannot be compact clock time.",
        )
    ]

    repaired = d0.repair_clock_encoded_timestamps(moments)

    assert repaired.repaired_moment_count == 0
    assert repaired.moments[0].start == 1968.0
    assert repaired.moments[0].end == 2036.0


def test_enforce_moment_length_discipline_caps_windows_and_drops_blobs() -> None:
    moments = [
        d0.LlmMoment(start=0.0, end=30.0, hook=1, flow=1, value=1, trend=1, virality=0.5, rationale="tight"),
        d0.LlmMoment(start=100.0, end=205.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="cap"),
        d0.LlmMoment(start=300.0, end=500.0, hook=1, flow=1, value=1, trend=1, virality=1.0, rationale="blob"),
        d0.LlmMoment(start=600.0, end=600.0, hook=1, flow=1, value=1, trend=1, virality=0.1, rationale="zero"),
    ]

    result = d0.enforce_moment_length_discipline(moments, max_duration_sec=90.0, blob_drop_duration_sec=120.0)

    assert result.input_moment_count == 4
    assert len(result.moments) == 2
    assert result.capped_moment_count == 1
    assert len(result.dropped_blob_moments) == 1
    assert result.dropped_non_positive_duration_count == 1
    assert result.moments[1].start == 100.0
    assert result.moments[1].end == 190.0
    assert result.moments[1].duration == 90.0


def test_default_length_discipline_caps_moderate_blobs_before_dropping_marathon_windows() -> None:
    moments = [
        d0.LlmMoment(start=5006.0, end=5216.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="pokemon"),
        d0.LlmMoment(start=0.0, end=600.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="marathon"),
    ]

    result = d0.enforce_moment_length_discipline(moments)

    assert len(result.moments) == 1
    assert result.moments[0].start == 5006.0
    assert result.moments[0].end == 5096.0
    assert result.capped_moment_count == 1
    assert len(result.dropped_blob_moments) == 1


def test_dedupe_moments_by_overlap_removes_repeated_llm_windows() -> None:
    moments = [
        d0.LlmMoment(start=100.0, end=160.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="first"),
        d0.LlmMoment(start=100.0, end=160.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="repeat"),
        d0.LlmMoment(start=112.0, end=160.0, hook=1, flow=1, value=1, trend=1, virality=0.8, rationale="overlap"),
        d0.LlmMoment(start=300.0, end=330.0, hook=1, flow=1, value=1, trend=1, virality=0.7, rationale="distinct"),
    ]

    kept, duplicates = d0.dedupe_moments_by_overlap(moments, overlap_ratio=0.8)

    assert [(moment.start, moment.end) for moment in kept] == [(100.0, 160.0), (300.0, 330.0)]
    assert len(duplicates) == 2


def test_call_anthropic_retries_read_timeout(monkeypatch) -> None:
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "content": [{"type": "text", "text": "{\"moments\": []}"}],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("The read operation timed out")
        return FakeResponse()

    monkeypatch.setattr(d0.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(d0.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(d0.time, "sleep", lambda seconds: None)

    text, usage = d0.call_anthropic_moment_finder(
        "prompt",
        model="mock-sonnet",
        temperature=0.0,
        max_tokens=16000,
        timeout_sec=900.0,
        retries=1,
        retry_backoff_sec=0.0,
    )

    assert calls["count"] == 2
    assert text == "{\"moments\": []}"
    assert usage["input_tokens"] == 10


def test_call_gemini_uses_openai_compatible_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content='{"moments": []}')
            choice = types.SimpleNamespace(message=message)
            usage = types.SimpleNamespace(prompt_tokens=20, completion_tokens=4, total_tokens=24)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class FakeClient:
        def __init__(self, *, api_key, base_url, timeout):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake_openai = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(d0.settings, "gemini_api_key", "gem-key")

    text, usage = d0.call_gemini_moment_finder(
        "prompt",
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_tokens=16000,
        timeout_sec=900.0,
        retries=0,
        retry_backoff_sec=0.0,
    )

    assert captured["base_url"] == d0.GEMINI_OPENAI_BASE_URL
    assert captured["model"] == "gemini-2.5-flash-lite"
    assert captured["response_format"] == {"type": "json_object"}
    assert text == '{"moments": []}'
    assert usage == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}


def test_provider_model_defaults_to_gemini_flash_lite() -> None:
    assert d0.normalize_provider("") == "gemini"
    assert d0.resolve_provider_model("gemini", "") == "gemini-2.5-flash-lite"
    assert d0.resolve_provider_model("anthropic", "") == d0.DEFAULT_ANTHROPIC_MODEL
