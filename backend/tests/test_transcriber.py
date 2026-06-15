import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda *_a, **_k: types.SimpleNamespace(total_memory=0),
        )
    ),
)

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

    def test_transcribe_chunk_passes_configured_glossary_as_initial_prompt(self) -> None:
        # WALL-Re-Test (#92): the Jynxzi glossary is biased into Whisper via
        # initial_prompt so verbal-tic catchphrases survive instead of being
        # smoothed to dictionary words. The model stays large-v3-turbo.
        class _Model:
            def __init__(self) -> None:
                self.calls = []

            def transcribe(self, path, **kwargs):
                self.calls.append((path, kwargs))
                return {"segments": []}

        glossary = "Balright, Darth Maura, Me ho, Hivise"
        model = _Model()
        with patch.object(transcriber.settings, "whisper_initial_prompt", glossary), \
             patch("services.transcriber._model_device_name", return_value="cpu"):
            transcriber._transcribe_chunk(model, Path("/tmp/chunk.wav"))

        self.assertEqual(model.calls[0][1]["initial_prompt"], glossary)

    def test_transcribe_chunk_omits_initial_prompt_when_glossary_blank(self) -> None:
        class _Model:
            def __init__(self) -> None:
                self.calls = []

            def transcribe(self, path, **kwargs):
                self.calls.append((path, kwargs))
                return {"segments": []}

        model = _Model()
        with patch.object(transcriber.settings, "whisper_initial_prompt", ""), \
             patch("services.transcriber._model_device_name", return_value="cpu"):
            transcriber._transcribe_chunk(model, Path("/tmp/chunk.wav"))

        self.assertNotIn("initial_prompt", model.calls[0][1])

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

    def test_diarization_annotation_unwraps_pyannote4_diarize_output(self) -> None:
        # pyannote 4.x returns a DiarizeOutput; the speaker Annotation (the thing
        # with .itertracks the timeline builder needs) is .speaker_diarization.
        annotation = object()

        class _DiarizeOutput:
            speaker_diarization = annotation

        self.assertIs(transcriber._diarization_annotation(_DiarizeOutput()), annotation)

    def test_diarization_annotation_passes_through_legacy_annotation(self) -> None:
        # pyannote 3.x returned the Annotation directly (no .speaker_diarization).
        class _Annotation:
            def itertracks(self, yield_label=False):
                return []

        annotation = _Annotation()
        self.assertIs(transcriber._diarization_annotation(annotation), annotation)

    def test_move_pipeline_to_device_uses_cuda_when_available(self) -> None:
        class _Pipeline:
            moved_to = None

            def to(self, device):
                self.moved_to = device
                return self

        # torch is imported lazily inside the helper — inject a fake so the test
        # is independent of whether a real torch/GPU is present in the test env.
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        fake_torch.device = lambda d: d

        pipeline = _Pipeline()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            returned = transcriber._move_pipeline_to_device(pipeline)

        self.assertIs(returned, pipeline)
        self.assertEqual(pipeline.moved_to, "cuda")

    def test_move_pipeline_to_device_tolerates_pipeline_without_to(self) -> None:
        # A fake/None pipeline (no .to) must not crash — GPU placement is best-effort.
        sentinel = object()
        self.assertIs(transcriber._move_pipeline_to_device(sentinel), sentinel)
        self.assertIsNone(transcriber._move_pipeline_to_device(None))


if __name__ == "__main__":
    unittest.main()
