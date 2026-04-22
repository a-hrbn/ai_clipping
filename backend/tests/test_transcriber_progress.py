import pathlib
import sys
import threading
import types
import unittest
from time import sleep
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
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=_Model()), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            result = transcriber.transcribe_with_progress(
                pathlib.Path("/tmp/audio.wav"),
                lambda done, total, chunk: updates.append((done, total, chunk.name)),
            )

        self.assertEqual(len(result), 3)
        self.assertEqual(updates, [(1, 3, "chunk_000.wav"), (2, 3, "chunk_001.wav"), (3, 3, "chunk_002.wav")])

    def test_skips_empty_chunks_without_shifting_offsets(self) -> None:
        seen_paths = []

        class _Model:
            def transcribe(self, path, *_args, **_kwargs):
                seen_paths.append(path)
                return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

        fake_chunks = [pathlib.Path(f"/tmp/chunk_{idx:03d}.wav") for idx in range(3)]

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=_Model()), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", side_effect=[16000, 0, 16000]):
            run_mock.return_value = None
            result = transcriber.transcribe_with_progress(pathlib.Path("/tmp/audio.wav"))

        self.assertEqual(
            [segment.start for segment in result],
            [0.0, float(2 * transcriber.CHUNK_SECONDS)],
        )
        self.assertEqual(
            seen_paths,
            [str(fake_chunks[0]), str(fake_chunks[2])],
        )

    def test_skips_known_empty_tensor_runtime_error_for_chunk(self) -> None:
        class _Model:
            def __init__(self) -> None:
                self.calls = 0

            def transcribe(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError(
                        "cannot reshape tensor of 0 elements into shape [1, 0, 8, -1]"
                    )
                return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

        model = _Model()
        fake_chunks = [pathlib.Path(f"/tmp/chunk_{idx:03d}.wav") for idx in range(3)]

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=model), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            result = transcriber.transcribe_with_progress(pathlib.Path("/tmp/audio.wav"))

        self.assertEqual(len(result), 2)
        self.assertEqual(
            [segment.start for segment in result],
            [0.0, float(2 * transcriber.CHUNK_SECONDS)],
        )

    def test_raises_on_malformed_segment_payloads(self) -> None:
        class _Model:
            def transcribe(self, *_args, **_kwargs):
                return {
                    "segments": [
                        None,
                        {"start": 0.0, "end": 1.0, "text": "hello"},
                        {"start": "bad", "end": 2.0, "text": "broken"},
                    ]
                }

        fake_chunks = [pathlib.Path("/tmp/chunk_000.wav")]

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=_Model()), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            with self.assertRaisesRegex(ValueError, "expected mapping, got NoneType"):
                transcriber.transcribe_with_progress(pathlib.Path("/tmp/audio.wav"))

    def test_clamps_zero_duration_segment_instead_of_skipping_window(self) -> None:
        class _Model:
            def transcribe(self, *_args, **_kwargs):
                return {
                    "segments": [
                        {"start": 1.25, "end": 1.25, "text": "hello there"},
                        {"start": 2.0, "end": 3.0, "text": "still valid"},
                    ]
                }

        fake_chunks = [pathlib.Path("/tmp/chunk_000.wav")]

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=_Model()), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000), \
             patch("services.transcriber.logger.warning") as warning_mock:
            run_mock.return_value = None
            result = transcriber.transcribe_with_progress(pathlib.Path("/tmp/audio.wav"))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].start, 1.25)
        self.assertAlmostEqual(
            result[0].end,
            1.25 + transcriber._MIN_SEGMENT_DURATION_SECONDS,
            places=6,
        )
        self.assertEqual(result[0].text, "hello there")
        self.assertTrue(any("transcriber_clamp_zero_duration_segment" in str(call.args[0]) for call in warning_mock.call_args_list))

    def test_serializes_shared_whisper_inference(self) -> None:
        state = {"active": 0, "max_active": 0}
        state_lock = threading.Lock()
        start_event = threading.Event()

        class _Model:
            def transcribe(self, *_args, **_kwargs):
                with state_lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                sleep(0.05)
                with state_lock:
                    state["active"] -= 1
                return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

        fake_chunks = [pathlib.Path("/tmp/chunk_000.wav")]
        errors = []

        def _run_worker() -> None:
            try:
                start_event.wait(timeout=5)
                transcriber.transcribe_with_progress(pathlib.Path("/tmp/audio.wav"))
            except Exception as exc:  # pragma: no cover - test helper
                errors.append(exc)

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", _Model()), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            left = threading.Thread(target=_run_worker)
            right = threading.Thread(target=_run_worker)
            left.start()
            right.start()
            start_event.set()
            left.join(timeout=5)
            right.join(timeout=5)

        self.assertFalse(errors)
        self.assertEqual(state["max_active"], 1)


if __name__ == "__main__":
    unittest.main()
