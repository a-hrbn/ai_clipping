"""WALL-Re-Test (#92) end-to-end with fakes — no real Whisper/pyannote/network.

Drives the production half through its real code paths (transcriber.transcribe +
moment_finder.find_moments) with only the three heavy leaves faked — the Whisper
model, the pyannote diarizer and the finder LLM — to prove the two cheap levers
compose end-to-end:

  * the Jynxzi glossary reaches the Whisper call (initial_prompt), and
  * a diarized segment carries its speaker (STREAMER → the streamer's name) all
    the way into the finder transcript prompt (spk=…).
"""
from __future__ import annotations

import json
import pathlib
import sys
import types
from unittest.mock import patch

# Lazy heavy deps that importing the transcriber must not require.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda *_a, **_k: types.SimpleNamespace(total_memory=0),
        )
    ),
)
# pyannote is imported lazily inside transcriber._diarize; inject a stub so the
# import succeeds and the real diarization-mapping logic runs against our fake.
sys.modules.setdefault("pyannote", types.ModuleType("pyannote"))
sys.modules.setdefault("pyannote.audio", types.SimpleNamespace(Pipeline=object))

from services import transcriber  # noqa: E402
from services import moment_finder  # noqa: E402
from ml.wall_retest_rerun import JYNXZI_GLOSSARY  # noqa: E402


class _FakeWhisper:
    """Stands in for the loaded Whisper model; records the transcribe kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, _path, **kwargs):
        self.calls.append(kwargs)
        return {
            "segments": [
                {"start": 10.0, "end": 20.0, "text": "Hivise tries the en passant"},
                {"start": 20.0, "end": 35.0, "text": "and the chat goes absolutely wild"},
            ]
        }


class _Turn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeDiarization:
    def itertracks(self, yield_label: bool = True):
        # SPEAKER_00 dominates total time → mapped to STREAMER; SPEAKER_01 → GUEST_1.
        return [
            (_Turn(0.0, 25.0), None, "SPEAKER_00"),
            (_Turn(25.0, 40.0), None, "SPEAKER_01"),
        ]


class _FakePyannotePipeline:
    def __call__(self, _audio_path):
        return _FakeDiarization()


def test_glossary_and_speaker_compose_end_to_end(monkeypatch) -> None:
    fake_model = _FakeWhisper()

    with patch("services.transcriber.subprocess.run", return_value=None), \
         patch.object(transcriber, "_whisper_model", None), \
         patch.object(transcriber, "_whisper_runtime_logged", False), \
         patch.object(transcriber, "_get_model", return_value=fake_model), \
         patch.object(transcriber.settings, "whisper_backend", "openai"), \
         patch("services.transcriber._model_device_name", return_value="cpu"), \
         patch.object(transcriber.settings, "hf_token", "fake-token"), \
         patch.object(transcriber, "_load_pyannote_pipeline", return_value=_FakePyannotePipeline()), \
         patch.object(transcriber, "_load_diarization_audio", return_value=(object(), 16000)), \
         patch("pathlib.Path.glob", return_value=[pathlib.Path("/tmp/chunk_000.wav")]), \
         patch("services.transcriber._chunk_frame_count", return_value=16000):
        segments = transcriber.transcribe(
            pathlib.Path("/tmp/audio.wav"),
            enable_diarization=True,
            initial_prompt=JYNXZI_GLOSSARY,
        )

    # Lever (a): the glossary reached the (faked) Whisper call.
    assert fake_model.calls[0]["initial_prompt"] == JYNXZI_GLOSSARY
    # Lever (b): the diarizer assigned the dominant speaker as STREAMER, the other
    # as GUEST_1 — proven through the real transcriber diarization mapping.
    assert [s.speaker for s in segments] == ["STREAMER", "GUEST_1"]

    captured: dict[str, str] = {}

    def fake_call(prompt: str, **_kwargs):
        captured["prompt"] = prompt
        return (
            json.dumps(
                {"moments": [
                    {"start": 10, "end": 20, "hook": 0.8, "flow": 0.7,
                     "value": 0.9, "trend": 0.6, "virality": 0.95, "rationale": "banter"}
                ]}
            ),
            {"provider": "fake"},
        )

    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", fake_call)

    moment_finder.find_moments(
        segments,
        profile="jynxzi",
        vod_id="vod-e2e",
        max_tokens=1000,
    )

    prompt = captured["prompt"]
    # STREAMER rendered as the jynxzi profile's streamer name; guest kept as GUEST_1.
    assert "spk=jynxzi" in prompt
    assert "spk=GUEST_1" in prompt
