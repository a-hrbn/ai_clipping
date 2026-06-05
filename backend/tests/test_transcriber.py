import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import transcriber
from services.transcriber import CHUNK_SECONDS


class TranscriberChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._backend_patch = patch.object(transcriber.settings, "whisper_backend", "openai")
        self._backend_patch.start()

    def tearDown(self) -> None:
        self._backend_patch.stop()

    def test_non_overlapping_chunks_use_exact_chunk_offsets(self) -> None:
        offsets = [idx * CHUNK_SECONDS for idx in range(4)]
        self.assertEqual(offsets, [0, 180, 360, 540])

    def test_whisper_model_uses_project_local_cache_dir(self) -> None:
        with patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=object()) as load_model_mock, \
             patch.object(transcriber.settings, "storage_dir", str((Path.cwd() / "backend" / "tmp" / "storage").resolve())):
            transcriber._get_model()

        _, kwargs = load_model_mock.call_args
        self.assertIn("download_root", kwargs)
        self.assertTrue(str(kwargs["download_root"]).endswith("backend/tmp/cache/whisper"))

    def test_whisper_model_uses_cuda_when_available_and_requested(self) -> None:
        with patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch.object(transcriber.settings, "whisper_device", "auto"), \
             patch("services.transcriber.torch.cuda.is_available", return_value=True), \
             patch("services.transcriber.whisper.load_model", return_value=object()) as load_model_mock:
            transcriber._get_model()

        _, kwargs = load_model_mock.call_args
        self.assertEqual(kwargs["device"], "cuda")

    def test_transcribe_chunk_passes_configured_language_hint(self) -> None:
        class _Model:
            def __init__(self) -> None:
                self.calls = []

            def transcribe(self, path, **kwargs):
                self.calls.append((path, kwargs))
                return {"segments": []}

        model = _Model()
        with patch.object(transcriber.settings, "whisper_language", "en"), \
             patch("services.transcriber._model_device_name", return_value="cpu"):
            transcriber._transcribe_chunk(model, Path("/tmp/chunk.wav"))

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][1]["language"], "en")
        self.assertFalse(model.calls[0][1]["fp16"])
        self.assertFalse(model.calls[0][1]["word_timestamps"])

    def test_faster_whisper_backend_converts_segments(self) -> None:
        class _Segment:
            start = 0.5
            end = 1.75
            text = " hello "
            words = []

        class _Model:
            def __init__(self) -> None:
                self.calls = []

            def transcribe(self, path, **kwargs):
                self.calls.append((path, kwargs))
                return iter([_Segment()]), object()

        model = _Model()
        with patch.object(transcriber, "_whisper_model_backend", "faster-whisper"), \
             patch.object(transcriber.settings, "whisper_language", "en"), \
             patch.object(transcriber.settings, "whisper_word_timestamps", False):
            result = transcriber._transcribe_chunk(model, Path("/tmp/chunk.wav"))

        self.assertEqual(result["segments"], [{"start": 0.5, "end": 1.75, "text": "hello"}])
        self.assertEqual(model.calls[0][1]["language"], "en")
        self.assertFalse(model.calls[0][1]["word_timestamps"])

    def test_load_pyannote_pipeline_prefers_current_token_kwarg(self) -> None:
        class _Pipeline:
            calls = []

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                cls.calls.append((model_id, kwargs))
                if "use_auth_token" in kwargs:
                    raise TypeError("unexpected legacy token kwarg")
                return object()

        transcriber._load_pyannote_pipeline(_Pipeline, "hf_secret")

        self.assertEqual(
            _Pipeline.calls,
            [("pyannote/speaker-diarization-3.1", {"token": "hf_secret"})],
        )

    def test_load_pyannote_pipeline_falls_back_to_legacy_use_auth_token_kwarg(self) -> None:
        class _Pipeline:
            calls = []

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                cls.calls.append((model_id, kwargs))
                if "token" in kwargs:
                    raise TypeError(
                        "Pipeline.from_pretrained() got an unexpected keyword argument 'token'"
                    )
                return object()

        transcriber._load_pyannote_pipeline(_Pipeline, "hf_secret")

        self.assertEqual(
            _Pipeline.calls,
            [
                ("pyannote/speaker-diarization-3.1", {"token": "hf_secret"}),
                ("pyannote/speaker-diarization-3.1", {"use_auth_token": "hf_secret"}),
            ],
        )

    def test_diarize_skips_gated_pyannote_access_failure(self) -> None:
        audio_module = types.ModuleType("pyannote.audio")

        class _Pipeline:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                raise RuntimeError(
                    "403 Client Error. Cannot access gated repo. "
                    "Access to model pyannote/speaker-diarization-3.1 is restricted "
                    "and you are not in the authorized list."
                )

        audio_module.Pipeline = _Pipeline
        segments = [transcriber.RichSegment(start=0.0, end=1.0, text="hello")]

        with patch.dict(
            sys.modules,
            {
                "pyannote": types.ModuleType("pyannote"),
                "pyannote.audio": audio_module,
            },
        ), patch.object(transcriber.settings, "hf_token", "bad_token"), \
             patch("services.transcriber.logger.warning") as warning_mock:
            result = transcriber._diarize(Path("/tmp/audio.wav"), segments)

        self.assertIs(result, segments)
        self.assertEqual(result[0].speaker, "UNKNOWN")
        self.assertTrue(
            any(
                "transcriber_diarization_unavailable" in str(call.args[0])
                and "hf_gated_model_access" in str(call.args)
                for call in warning_mock.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
