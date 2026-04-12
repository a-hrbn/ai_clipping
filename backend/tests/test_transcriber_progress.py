import pathlib
import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import transcriber


class TranscriberProgressTests(unittest.TestCase):
    def test_progress_callback_receives_chunk_updates(self) -> None:
        updates = []

        class _Model:
            def transcribe(self, *_args, **_kwargs):
                return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

        fake_chunks = [pathlib.Path(f"/tmp/chunk_{idx:03d}.wav") for idx in range(3)]

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch("services.transcriber.whisper.load_model", return_value=_Model()), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks):
            run_mock.return_value = None
            result = transcriber.transcribe_with_progress(
                pathlib.Path("/tmp/audio.wav"),
                lambda done, total, chunk: updates.append((done, total, chunk.name)),
            )

        self.assertEqual(len(result), 3)
        self.assertEqual(updates, [(1, 3, "chunk_000.wav"), (2, 3, "chunk_001.wav"), (3, 3, "chunk_002.wav")])


if __name__ == "__main__":
    unittest.main()
