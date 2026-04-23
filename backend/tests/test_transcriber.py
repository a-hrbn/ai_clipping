import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import transcriber
from services.transcriber import CHUNK_SECONDS


class TranscriberChunkingTests(unittest.TestCase):
    def test_non_overlapping_chunks_use_exact_chunk_offsets(self) -> None:
        offsets = [idx * CHUNK_SECONDS for idx in range(4)]
        self.assertEqual(offsets, [0, 600, 1200, 1800])

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


if __name__ == "__main__":
    unittest.main()
