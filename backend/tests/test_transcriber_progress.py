import json
import pathlib
import sys
import threading
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from time import sleep
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import transcriber


class TranscriberProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self._backend_patch = patch.object(transcriber.settings, "whisper_backend", "openai")
        self._backend_patch.start()

    def tearDown(self) -> None:
        self._backend_patch.stop()

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

    def test_reuses_durable_chunk_cache_without_loading_model(self) -> None:
        class _Model:
            def transcribe(self, path, *_args, **_kwargs):
                chunk_name = pathlib.Path(path).stem
                return {"segments": [{"start": 0.0, "end": 1.0, "text": chunk_name}]}

        fake_chunks = [pathlib.Path(f"/tmp/chunk_{idx:03d}.wav") for idx in range(2)]

        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=_Model()) as load_model, \
             patch.object(transcriber.settings, "storage_dir", tmp), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            first = transcriber.transcribe_with_progress(
                pathlib.Path("/tmp/audio.wav"),
                cache_key="url:https://www.twitch.tv/videos/123",
            )
            self.assertEqual([segment.text for segment in first], ["chunk_000", "chunk_001"])
            self.assertEqual(load_model.call_count, 1)

            with patch.object(transcriber, "_whisper_model", None), \
                 patch.object(transcriber, "_whisper_runtime_logged", False), \
                 patch("services.transcriber.whisper.load_model", side_effect=AssertionError("model should not load")):
                second = transcriber.transcribe_with_progress(
                    pathlib.Path("/tmp/audio.wav"),
                    cache_key="url:https://www.twitch.tv/videos/123",
                )

        self.assertEqual([segment.text for segment in second], ["chunk_000", "chunk_001"])

    def test_load_cached_transcript_requires_all_expected_chunks(self) -> None:
        class _Model:
            def transcribe(self, path, *_args, **_kwargs):
                return {"segments": [{"start": 0.0, "end": 1.0, "text": pathlib.Path(path).stem}]}

        fake_chunks = [pathlib.Path(f"/tmp/chunk_{idx:03d}.wav") for idx in range(2)]
        updates = []

        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=_Model()), \
             patch.object(transcriber.settings, "storage_dir", tmp), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            transcriber.transcribe_with_progress(
                pathlib.Path("/tmp/audio.wav"),
                cache_key="twitch_vod:123",
            )

            cached = transcriber.load_cached_transcript(
                "twitch_vod:123",
                expected_duration_sec=2 * transcriber.CHUNK_SECONDS,
                progress_cb=lambda done, total, path: updates.append((done, total, path.name)),
            )
            incomplete = transcriber.load_cached_transcript(
                "twitch_vod:123",
                expected_duration_sec=3 * transcriber.CHUNK_SECONDS,
            )

        self.assertIsNotNone(cached)
        self.assertEqual([segment.text for segment in cached or []], ["chunk_000", "chunk_001"])
        self.assertEqual(updates, [(1, 2, "chunk_000.json"), (2, 2, "chunk_001.json")])
        self.assertIsNone(incomplete)

    def test_cache_key_changes_do_not_reuse_chunks(self) -> None:
        class _Model:
            def __init__(self) -> None:
                self.calls = 0

            def transcribe(self, *_args, **_kwargs):
                self.calls += 1
                return {"segments": [{"start": 0.0, "end": 1.0, "text": f"call {self.calls}"}]}

        model = _Model()
        fake_chunks = [pathlib.Path("/tmp/chunk_000.wav")]

        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=model), \
             patch.object(transcriber.settings, "storage_dir", tmp), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            first = transcriber.transcribe_with_progress(pathlib.Path("/tmp/audio.wav"), cache_key="left")
            second = transcriber.transcribe_with_progress(pathlib.Path("/tmp/audio.wav"), cache_key="right")

        self.assertEqual([segment.text for segment in first], ["call 1"])
        self.assertEqual([segment.text for segment in second], ["call 2"])

    def test_isolated_chunk_transcription_avoids_parent_model_load(self) -> None:
        fake_chunks = [pathlib.Path(f"/tmp/chunk_{idx:03d}.wav") for idx in range(2)]

        def fake_isolated(chunks):
            return [
                (chunk_idx, {"segments": [{"start": 0.0, "end": 1.0, "text": chunk_file.stem}]})
                for chunk_idx, chunk_file in chunks
            ]

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", side_effect=AssertionError("parent model should not load")), \
             patch("services.transcriber._transcribe_chunks_isolated", side_effect=fake_isolated) as isolated_mock, \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            result = transcriber.transcribe_with_progress(
                pathlib.Path("/tmp/audio.wav"),
                isolate_chunks=True,
            )

        self.assertEqual([segment.text for segment in result], ["chunk_000", "chunk_001"])
        self.assertEqual(isolated_mock.call_count, 1)
        self.assertEqual([chunk.name for _idx, chunk in isolated_mock.call_args.args[0]], ["chunk_000.wav", "chunk_001.wav"])

    def test_isolated_chunk_transcription_splits_worker_batches(self) -> None:
        fake_chunks = [pathlib.Path(f"/tmp/chunk_{idx:03d}.wav") for idx in range(3)]
        seen_batches = []

        def fake_isolated(chunks):
            seen_batches.append([chunk.name for _idx, chunk in chunks])
            for chunk_idx, chunk_file in chunks:
                yield chunk_idx, {"segments": [{"start": 0.0, "end": 1.0, "text": chunk_file.stem}]}

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", side_effect=AssertionError("parent model should not load")), \
             patch("services.transcriber._transcribe_chunks_isolated", side_effect=fake_isolated), \
             patch.object(transcriber.settings, "transcription_worker_batch_size", 1), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            result = transcriber.transcribe_with_progress(
                pathlib.Path("/tmp/audio.wav"),
                isolate_chunks=True,
            )

        self.assertEqual([segment.text for segment in result], ["chunk_000", "chunk_001", "chunk_002"])
        self.assertEqual(seen_batches, [["chunk_000.wav"], ["chunk_001.wav"], ["chunk_002.wav"]])

    def test_isolated_chunk_worker_writes_segments(self) -> None:
        class _Model:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "chunk.json"
            chunk = pathlib.Path(tmp) / "chunk_000.wav"
            chunk.write_bytes(b"audio")
            with patch.object(transcriber, "_get_model", return_value=_Model()), \
                 patch.object(
                     transcriber,
                     "_transcribe_chunk",
                     return_value={"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
                 ):
                code = transcriber._transcribe_chunk_worker_main(
                    ["--transcribe-chunk", str(chunk), str(output)]
                )

            payload = json.loads(output.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["segments"][0]["text"], "hello")

    def test_isolated_chunk_worker_batch_loads_model_once(self) -> None:
        class _Model:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            first = pathlib.Path(tmp) / "chunk_000.wav"
            second = pathlib.Path(tmp) / "chunk_001.wav"
            first.write_bytes(b"audio")
            second.write_bytes(b"audio")
            manifest = pathlib.Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"chunk_index": 0, "chunk_path": str(first)},
                        {"chunk_index": 1, "chunk_path": str(second)},
                    ]
                ),
                encoding="utf-8",
            )
            output = StringIO()

            def fake_transcribe(_model, chunk_file):
                return {
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": pathlib.Path(chunk_file).stem}
                    ]
                }

            with patch.object(transcriber, "_get_model", return_value=_Model()) as get_model, \
                 patch.object(transcriber, "_transcribe_chunk", side_effect=fake_transcribe), \
                 redirect_stdout(output):
                code = transcriber._transcribe_chunk_worker_main(
                    ["--transcribe-chunks", str(manifest)]
                )

            rows = []
            for line in output.getvalue().splitlines():
                self.assertTrue(line.startswith(transcriber._WORKER_RESULT_PREFIX))
                rows.append(json.loads(line[len(transcriber._WORKER_RESULT_PREFIX):]))

        self.assertEqual(code, 0)
        self.assertEqual(get_model.call_count, 1)
        self.assertEqual([row["chunk_index"] for row in rows], [0, 1])
        self.assertEqual([row["segments"][0]["text"] for row in rows], ["chunk_000", "chunk_001"])

    def test_progress_path_forwards_configured_language_to_whisper(self) -> None:
        seen_kwargs = []

        class _Model:
            def transcribe(self, *_args, **kwargs):
                seen_kwargs.append(kwargs)
                return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

        fake_chunks = [pathlib.Path("/tmp/chunk_000.wav")]

        with patch("services.transcriber.subprocess.run") as run_mock, \
             patch.object(transcriber, "_whisper_model", None), \
             patch.object(transcriber, "_whisper_runtime_logged", False), \
             patch("services.transcriber.whisper.load_model", return_value=_Model()), \
             patch.object(transcriber.settings, "whisper_language", "en"), \
             patch("services.transcriber.settings.hf_token", None), \
             patch("pathlib.Path.glob", return_value=fake_chunks), \
             patch("services.transcriber._chunk_frame_count", return_value=16000):
            run_mock.return_value = None
            transcriber.transcribe_with_progress(pathlib.Path("/tmp/audio.wav"))

        self.assertEqual(len(seen_kwargs), 1)
        self.assertEqual(seen_kwargs[0]["language"], "en")

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
