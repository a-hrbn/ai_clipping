import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from ml import collect_viral_clips
from services import downloader


class CollectViralClipsTests(unittest.TestCase):
    @staticmethod
    def _sample_training_row(*, candidate_id: str, label: int = 1) -> dict:
        return {
            "candidate_id": candidate_id,
            "label": label,
            "features": {
                "hook_strength": 0.5,
                "max_hype": 7.0,
                "arc_score": 1.2,
            },
        }

    def test_cut_video_windows_uses_single_ffmpeg_process_within_batch_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(collect_viral_clips.settings, "training_ffmpeg_video_codec", "libx264"), \
             patch.object(collect_viral_clips.settings, "training_ffmpeg_audio_codec", "aac"), \
             patch.object(collect_viral_clips.settings, "training_ffmpeg_preset", "veryfast"), \
             patch.object(collect_viral_clips.settings, "training_ffmpeg_threads", 1), \
             patch("ml.collect_viral_clips._run_ffmpeg_command", return_value=SimpleNamespace(returncode=0, stderr="")) as run_mock:
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / "window_0001" / "window.mp4", 10.0, 20.0),
                (Path(tmp) / "window_0002" / "window.mp4", 30.0, 45.0),
            ]

            result = collect_viral_clips.cut_video_windows(video_path, outputs)

        self.assertEqual(result, [item[0] for item in outputs])
        command = run_mock.call_args.args[0]
        self.assertEqual(command[:4], ["ffmpeg", "-y", "-nostdin", "-loglevel"])
        self.assertEqual(command.count("-i"), 2)
        self.assertIn(str(outputs[0][0]), command)
        self.assertIn(str(outputs[1][0]), command)
        self.assertNotIn("-movflags", command)
        self.assertEqual(command.count("-threads"), 2)
        self.assertEqual(run_mock.call_count, 1)

    def test_cut_video_windows_uses_explicit_ffmpeg_settings_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(collect_viral_clips.settings, "training_ffmpeg_video_codec", "h264_videotoolbox"), \
             patch.object(collect_viral_clips.settings, "training_ffmpeg_audio_codec", "copy"), \
             patch.object(collect_viral_clips.settings, "training_ffmpeg_preset", ""), \
             patch.object(collect_viral_clips.settings, "training_ffmpeg_threads", 3), \
             patch("ml.collect_viral_clips._run_ffmpeg_command", return_value=SimpleNamespace(returncode=0, stderr="")) as run_mock:
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / "window_0001" / "window.mp4", 10.0, 20.0),
            ]

            collect_viral_clips.cut_video_windows(video_path, outputs)

        command = run_mock.call_args.args[0]
        self.assertIn("h264_videotoolbox", command)
        self.assertIn("copy", command)
        self.assertNotIn("libx264", command)
        self.assertNotIn("aac", command)
        self.assertNotIn("-preset", command)
        self.assertEqual(command[command.index("-threads") + 1], "3")

    def test_cut_video_windows_splits_large_batches_across_multiple_ffmpeg_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips._run_ffmpeg_command", return_value=SimpleNamespace(returncode=0, stderr="")) as run_mock:
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / f"window_{index:04d}" / "window.mp4", float(index * 10), float(index * 10 + 8))
                for index in range(1, collect_viral_clips._MAX_BATCH_CUT_WINDOWS + 3)
            ]

            result = collect_viral_clips.cut_video_windows(video_path, outputs)

        self.assertEqual(result, [item[0] for item in outputs])
        self.assertEqual(run_mock.call_count, 2)
        first_command = run_mock.call_args_list[0].args[0]
        second_command = run_mock.call_args_list[1].args[0]
        self.assertEqual(first_command.count("-i"), collect_viral_clips._MAX_BATCH_CUT_WINDOWS)
        self.assertEqual(second_command.count("-i"), 2)
        self.assertIn(str(outputs[0][0]), first_command)
        self.assertIn(str(outputs[-1][0]), second_command)

    def test_cut_video_windows_retries_failed_batch_by_splitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / f"window_{index:04d}" / "window.mp4", float(index * 10), float(index * 10 + 8))
                for index in range(1, 5)
            ]

            def _fake_run(command, **_kwargs):
                input_count = command.count("-i")
                if input_count == 4:
                    return SimpleNamespace(returncode=1, stderr="pthread_create failed")
                return SimpleNamespace(returncode=0, stderr="")

            with patch("ml.collect_viral_clips._run_ffmpeg_command", side_effect=_fake_run) as run_mock:
                result = collect_viral_clips.cut_video_windows(video_path, outputs)

        self.assertEqual(result, [item[0] for item in outputs])
        self.assertEqual(run_mock.call_count, 3)
        self.assertEqual(run_mock.call_args_list[0].args[0].count("-i"), 4)
        self.assertEqual(run_mock.call_args_list[1].args[0].count("-i"), 2)
        self.assertEqual(run_mock.call_args_list[2].args[0].count("-i"), 2)

    def test_cut_video_windows_raises_when_single_window_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch(
                 "ml.collect_viral_clips._run_ffmpeg_command",
                 return_value=SimpleNamespace(returncode=1, stderr="decoder blew up"),
             ):
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / "window_0001" / "window.mp4", 10.0, 20.0),
            ]

            with self.assertRaisesRegex(RuntimeError, "decoder blew up"):
                collect_viral_clips.cut_video_windows(video_path, outputs)

    def test_cut_video_windows_logs_batch_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips._run_ffmpeg_command", return_value=SimpleNamespace(returncode=0, stderr="")), \
             patch("ml.collect_viral_clips.logger.info") as info_mock:
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / f"window_{index:04d}" / "window.mp4", float(index * 10), float(index * 10 + 8))
                for index in range(1, 4)
            ]

            collect_viral_clips.cut_video_windows(video_path, outputs)

        messages = [call.args[0] for call in info_mock.call_args_list]
        self.assertIn("training_vod_batch_cut_plan video=%s windows=%s batches=%s batch_limit=%s", messages)
        self.assertIn(
            "training_vod_batch_cut_start video=%s batch=%s/%s windows=%s-%s total_windows=%s batch_media_seconds=%.1f",
            messages,
        )
        self.assertIn(
            "training_vod_batch_cut_done video=%s batch=%s/%s completed_windows=%s/%s elapsed_sec=%.1f",
            messages,
        )

    def test_read_entries_supports_manifest_metadata_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clips.csv"
            path.write_text(
                "\n".join([
                    "url,streamer,platform,language,tags,label,notes",
                    'https://www.youtube.com/shorts/abc,shroud,youtube,en,"fps,clutch",1,viral',
                    'https://www.tiktok.com/@tarik/video/123,tarik,tiktok,en,"reaction,chatting",0,flat',
                ]),
                encoding="utf-8",
            )

            entries = collect_viral_clips._read_entries(path)

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["streamer"], "shroud")
            self.assertEqual(entries[1]["label"], "0")
            self.assertEqual(entries[1]["notes"], "flat")

    def test_read_entries_dedupes_cross_source_urls_by_canonical_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clips.csv"
            path.write_text(
                "\n".join([
                    "url,streamer,platform,language,tags,label,notes",
                    "https://www.youtube.com/shorts/abc?feature=share,jynxzi,youtube,en,gaming,1,first",
                    "https://www.youtube.com/shorts/abc,jynxzi,youtube,en,gaming,0,duplicate",
                    "https://www.tiktok.com/@jynxzi/video/123/,jynxzi,tiktok,en,gaming,1,unique",
                ]),
                encoding="utf-8",
            )

            entries = collect_viral_clips._read_entries(path)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["notes"], "first")

    def test_canonical_entry_url_drops_query_and_trailing_slash(self) -> None:
        self.assertEqual(
            collect_viral_clips._canonical_entry_url("https://www.youtube.com/shorts/abc/?feature=share"),
            "https://www.youtube.com/shorts/abc",
        )

    def test_parse_label_accepts_positive_and_negative_aliases(self) -> None:
        self.assertEqual(collect_viral_clips._parse_label("1"), 1)
        self.assertEqual(collect_viral_clips._parse_label("negative"), 0)
        self.assertEqual(collect_viral_clips._parse_label("weak"), 0)
        self.assertEqual(collect_viral_clips._parse_label("viral"), 1)

    def test_infer_platform_accepts_manifest_entry_dict(self) -> None:
        self.assertEqual(
            collect_viral_clips._infer_platform(
                {"url": "https://www.tiktok.com/@tarik/video/123", "platform": "tiktok"}
            ),
            "tiktok",
        )
        self.assertEqual(
            collect_viral_clips._infer_platform(
                {"url": "https://www.youtube.com/shorts/abc", "platform": ""}
            ),
            "youtube",
        )

    def test_collect_rows_pauses_and_retries_same_clip_after_transient_network_failure(self) -> None:
        entry = {
            "url": "https://www.tiktok.com/@tarik/video/123",
            "streamer": "tarik",
            "platform": "tiktok",
            "language": "en",
            "tags": "",
            "label": "1",
            "notes": "",
        }
        recovered_row = {
            "label": 1,
            "features": {
                "hook_strength": 0.5,
                "max_hype": 7.0,
                "arc_score": 1.2,
            },
        }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips._ensure_short_form_media_cached", return_value=Path("/tmp/cached.mp4")), \
             patch("ml.collect_viral_clips._process_clip", side_effect=[
                 downloader.DownloadPausedError("Temporary failure in name resolution"),
                 recovered_row,
             ]) as process_mock, \
             patch("ml.collect_viral_clips.time.sleep") as sleep_mock:
            rows = collect_viral_clips.collect_rows([entry], "gaming", Path(tmp))

        self.assertEqual(rows, [recovered_row])
        self.assertEqual(process_mock.call_count, 2)
        sleep_mock.assert_called_once_with(30)

    def test_collect_rows_reuses_cached_row_without_reprocessing(self) -> None:
        entry = {
            "url": "https://www.youtube.com/shorts/abc",
            "streamer": "tarik",
            "platform": "youtube",
            "language": "en",
            "tags": "",
            "label": "1",
            "notes": "",
        }
        cached_row = self._sample_training_row(candidate_id="cached-1")

        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips._ensure_short_form_media_cached", return_value=Path("/tmp/cached.mp4")), \
             patch("ml.collect_viral_clips._warm_clip_worker_runtime"), \
             patch("ml.collect_viral_clips._process_clip", return_value=cached_row) as process_mock:
            rows = collect_viral_clips.collect_rows([entry], "gaming", Path(tmp))
            self.assertEqual(rows, [cached_row])
            self.assertEqual(process_mock.call_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = collect_viral_clips._short_form_row_cache_dir(Path(tmp))
            cache_fingerprint = collect_viral_clips._short_form_cache_fingerprint("gaming")
            collect_viral_clips._write_cached_row(cache_dir, entry, cache_fingerprint, cached_row)

            with patch("ml.collect_viral_clips._warm_clip_worker_runtime"), \
                 patch("ml.collect_viral_clips._process_clip", side_effect=AssertionError("cache miss")):
                rows = collect_viral_clips.collect_rows([entry], "gaming", Path(tmp))

        self.assertEqual(rows, [cached_row])

    def test_collect_rows_parallel_preserves_input_order(self) -> None:
        first = {
            "url": "https://www.youtube.com/shorts/first",
            "streamer": "tarik",
            "platform": "youtube",
            "language": "en",
            "tags": "",
            "label": "1",
            "notes": "",
        }
        second = {
            "url": "https://www.youtube.com/shorts/second",
            "streamer": "tarik",
            "platform": "youtube",
            "language": "en",
            "tags": "",
            "label": "0",
            "notes": "",
        }
        row_first = self._sample_training_row(candidate_id="row-first", label=1)
        row_second = self._sample_training_row(candidate_id="row-second", label=0)

        def _fake_process(entry, job_id, clip_dir, profile, proxy=None, media_path=None):
            if entry["url"].endswith("/first"):
                time.sleep(0.05)
                return row_first
            return row_second

        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips._ensure_short_form_media_cached", return_value=Path("/tmp/cached.mp4")), \
             patch("ml.collect_viral_clips._warm_clip_worker_runtime"), \
             patch("ml.collect_viral_clips._resolved_short_form_collection_workers", return_value=2), \
             patch("ml.collect_viral_clips._process_clip", side_effect=_fake_process):
            rows = collect_viral_clips.collect_rows([first, second], "gaming", Path(tmp))

        self.assertEqual(rows, [row_first, row_second])

    def test_resolved_short_form_collection_workers_defaults_to_three(self) -> None:
        with patch.object(collect_viral_clips.settings, "short_form_collection_workers", 3):
            self.assertEqual(collect_viral_clips._resolved_short_form_collection_workers(1), 1)
            self.assertEqual(collect_viral_clips._resolved_short_form_collection_workers(2), 2)
            self.assertEqual(collect_viral_clips._resolved_short_form_collection_workers(10), 3)

    def test_resolved_short_form_prefetch_workers_defaults_to_two(self) -> None:
        with patch.object(collect_viral_clips.settings, "short_form_prefetch_workers", 2):
            self.assertEqual(collect_viral_clips._resolved_short_form_prefetch_workers(1), 1)
            self.assertEqual(collect_viral_clips._resolved_short_form_prefetch_workers(2), 2)
            self.assertEqual(collect_viral_clips._resolved_short_form_prefetch_workers(10), 2)

    def test_collect_row_for_entry_uses_prefetched_media_path(self) -> None:
        entry = {
            "url": "https://www.youtube.com/shorts/prefetched",
            "streamer": "tarik",
            "platform": "youtube",
            "language": "en",
            "tags": "",
            "label": "1",
            "notes": "",
        }
        cached_media = Path("/tmp/prefetched.mp4")
        row = self._sample_training_row(candidate_id="prefetched-row")

        class _FakePrefetcher:
            def __init__(self) -> None:
                self.scheduled = []

            def schedule_entries(self, entries):
                self.scheduled.extend(entries)

            def get(self, requested_entry):
                self.requested_entry = requested_entry
                return cached_media

            def close(self):
                return None

        prefetcher = _FakePrefetcher()
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips._process_clip", return_value=row) as process_mock:
            outcome = collect_viral_clips._collect_row_for_entry(
                entry,
                entries=[entry],
                index=1,
                total_entries=1,
                profile="gaming",
                work_dir=Path(tmp),
                cache_dir=collect_viral_clips._short_form_row_cache_dir(Path(tmp)),
                cache_fingerprint=collect_viral_clips._short_form_cache_fingerprint("gaming"),
                proxy=None,
                skipped_platforms=set(),
                prefetcher=prefetcher,
                prefetch_width=2,
            )

        self.assertEqual(outcome["row"], row)
        self.assertEqual(prefetcher.requested_entry, entry)
        self.assertEqual(process_mock.call_args.kwargs["media_path"], cached_media)

    def test_warm_clip_worker_runtime_primes_available_models(self) -> None:
        transcriber = SimpleNamespace(_get_model=lambda: None)
        video_features = SimpleNamespace(_ensure_deepface_ready=lambda: None, runtime_summary=lambda: {})
        semantic_candidates = SimpleNamespace(
            _get_model=lambda: object(),
            _get_pattern_embeddings=lambda model: None,
        )

        with patch("ml.collect_viral_clips._import_service_module", side_effect=[
            transcriber,
            video_features,
            semantic_candidates,
        ]), \
             patch.object(transcriber, "_get_model") as whisper_mock, \
             patch.object(video_features, "_ensure_deepface_ready") as deepface_mock, \
             patch.object(video_features, "runtime_summary"), \
             patch.object(semantic_candidates, "_get_model", return_value=object()) as semantic_model_mock, \
             patch.object(semantic_candidates, "_get_pattern_embeddings") as semantic_embeddings_mock:
            collect_viral_clips._warm_clip_worker_runtime("gaming")

        whisper_mock.assert_called_once_with()
        deepface_mock.assert_called_once_with()
        semantic_model_mock.assert_called_once_with()
        semantic_embeddings_mock.assert_called_once()

    def test_collect_clip_modalities_parallelizes_independent_extractors(self) -> None:
        executor_instances: list[object] = []

        class _ImmediateFuture:
            def __init__(self, value) -> None:
                self._value = value

            def result(self):
                return self._value

        class _RecordingExecutor:
            def __init__(self, max_workers, *args, **kwargs) -> None:
                self.max_workers = max_workers
                self.submissions = []
                executor_instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                self.submissions.append((fn, args, kwargs))
                return _ImmediateFuture(fn(*args, **kwargs))

        transcriber = SimpleNamespace(transcribe=lambda audio_path: ["seg"])
        audio_features = SimpleNamespace(extract=lambda audio_path: ["audio"])
        video_features = SimpleNamespace(extract=lambda video_path: ["video"])

        with patch("ml.collect_viral_clips._resolved_short_form_clip_step_workers", return_value=3), \
             patch("ml.collect_viral_clips.concurrent.futures.ThreadPoolExecutor", _RecordingExecutor):
            transcript, audio_frames, visual_frames = collect_viral_clips._collect_clip_modalities(
                Path("/tmp/audio.wav"),
                Path("/tmp/video.mp4"),
                transcriber=transcriber,
                audio_features=audio_features,
                video_features=video_features,
            )

        self.assertEqual(transcript, ["seg"])
        self.assertEqual(audio_frames, ["audio"])
        self.assertEqual(visual_frames, ["video"])
        self.assertEqual(len(executor_instances), 1)
        self.assertEqual(executor_instances[0].max_workers, 3)
        self.assertEqual(len(executor_instances[0].submissions), 3)


if __name__ == "__main__":
    unittest.main()
