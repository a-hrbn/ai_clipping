import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import sys
import types

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from config import settings
from ml import build_twitch_training_dataset
from services.timeline_types import Candidate
from services.transcriber import RichSegment


class BuildTwitchTrainingDatasetTests(unittest.TestCase):
    def test_semantic_provider_summary_reports_auto_fallback_chain(self) -> None:
        with patch.object(settings, "groq_api_key", "secret"), \
             patch.object(settings, "groq_model", "llama-groq"), \
             patch.object(settings, "ollama_model", "llama-local"):
            summary = build_twitch_training_dataset._semantic_provider_summary("auto")

        self.assertEqual(summary, "auto->ollama:llama-local|fallback=groq:llama-groq")

    def test_semantic_provider_summary_reports_groq_fallback_when_unconfigured(self) -> None:
        with patch.object(settings, "groq_api_key", None), \
             patch.object(settings, "groq_model", "llama-groq"), \
             patch.object(settings, "ollama_model", "llama-local"):
            summary = build_twitch_training_dataset._semantic_provider_summary("groq")

        self.assertEqual(summary, "groq_unavailable->ollama:llama-local")

    def test_dataset_summary_counts_classes(self) -> None:
        summary = build_twitch_training_dataset.dataset_summary([
            {"vod_id": "1", "label_class": "positive", "label": 1, "platform_source": "twitch_clip"},
            {"vod_id": "1", "label_class": "hard_negative", "label": 0, "platform_source": "twitch_vod_negative"},
            {"vod_id": "2", "label_class": "viral_positive", "label": 1, "platform_source": "twitch_clip"},
        ])
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["vod_count"], 2)
        self.assertEqual(summary["label_classes"]["positive"], 1)
        self.assertEqual(summary["positive_rows"], 2)
        self.assertEqual(summary["negative_rows"], 1)

    def test_dataset_summary_reports_duplicate_windows_and_platform_drift(self) -> None:
        summary = build_twitch_training_dataset.dataset_summary([
            {
                "vod_id": "1",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label": 1,
                "platform_source": "twitch_clip",
                "candidate_id": "a",
            },
            {
                "vod_id": "1",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "hard_negative",
                "label": 0,
                "platform_source": "twitch_vod_negative",
                "candidate_id": "b",
            },
            {
                "vod_id": "2",
                "window_start": 30.0,
                "window_end": 40.0,
                "label_class": "positive",
                "label": 1,
                "platform_source": "twitch_clip",
                "candidate_id": "c",
            },
        ])

        self.assertEqual(summary["duplicate_windows"]["count"], 1)
        self.assertEqual(summary["platform_source_drift"]["twitch_clip"]["positive_rows"], 2)
        self.assertEqual(summary["platform_source_drift"]["twitch_vod_negative"]["negative_rows"], 1)
        self.assertIn("vod_overlap_count", summary["split_leak_check"])
        self.assertIn("candidate_source_mix", summary)

    def test_cap_candidate_source_mix_limits_peak_share(self) -> None:
        rows = [
            {"candidate_id": "r1", "candidate_source": "reaction"},
            {"candidate_id": "r2", "candidate_source": "reaction"},
            {"candidate_id": "r3", "candidate_source": "reaction"},
            {"candidate_id": "r4", "candidate_source": "reaction"},
            {"candidate_id": "r5", "candidate_source": "reaction"},
            {"candidate_id": "q1", "candidate_source": "quote"},
            {"candidate_id": "q2", "candidate_source": "quote"},
            {"candidate_id": "m1", "candidate_source": "moment"},
            {"candidate_id": "m2", "candidate_source": "moment"},
        ]

        capped = build_twitch_training_dataset._cap_candidate_source_mix(rows)
        summary = build_twitch_training_dataset._candidate_source_mix_summary(capped)

        self.assertLessEqual(summary["peak_share"], 0.4)
        self.assertLess(len(capped), len(rows))

    def test_entry_from_label_maps_positive_and_negative(self) -> None:
        pos = build_twitch_training_dataset._entry_from_label({
            "broadcaster": "jynxzi",
            "vod_url": "https://www.twitch.tv/videos/1",
            "label_class": "viral_positive",
            "vod_id": "1",
            "window_start": 10,
            "window_end": 30,
            "clip_ids": ["abc"],
        })
        neg = build_twitch_training_dataset._entry_from_label({
            "broadcaster": "jynxzi",
            "vod_url": "https://www.twitch.tv/videos/1",
            "label_class": "hard_negative",
            "vod_id": "1",
            "window_start": 40,
            "window_end": 70,
            "clip_ids": [],
        })
        self.assertEqual(pos["label"], "1")
        self.assertEqual(neg["label"], "0")

    def test_build_dataset_processes_non_ambiguous_rows(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 30.0,
                "window_end": 40.0,
                "label_class": "ambiguous",
                "label_score": 0.0,
                "clip_ids": [],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]
        fake_row = {
            "label": 1,
            "label_source": "twitch_clip_api",
            "features": {
                "candidate_id": "x",
                "duration": 10.0,
                "hook_delay_sec": 0.0,
                "hook_strength": 0.1,
                "max_hype": 1.0,
                "mean_hype": 1.0,
                "arc_score": 0.0,
                "chat_peak": 0.0,
                "motion_peak": 0.0,
                "face_reaction_peak": 0.0,
                "speaker_turns": 0,
                "exclamation_count": 0,
                "laughter_count": 0,
                "question_open": False,
                "context_cost": 0.0,
                "payoff_score": 0.0,
                "context_efficiency": 1.0,
                "quoteability_score": 0.0,
                "surprise_score": 0.0,
                "contradiction_score": 0.0,
                "standalone_score": 0.0,
                "semantic_density": 0.0,
                "profile_fit_score": 0.0,
                "pre_hook_hype": 0.0,
                "post_hook_hype": 0.0,
                "hype_delta": 0.0,
                "local_peak_density": 0.0,
                "vod_progress": 0.0,
                "boundary_left_context_sec": 0.0,
                "boundary_right_context_sec": 0.0,
                "neighbor_positive_density": 0.0,
                "clip_overlap_count": 0.0,
                "source": "training",
            },
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", return_value=[fake_row]):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_class"], "positive")
        self.assertIn("vod_progress", rows[0]["features"])
        self.assertIn("boundary_left_context_sec", rows[0]["features"])

    def test_apply_vod_context_features_uses_neighboring_windows(self) -> None:
        row = {"features": {}}
        label = {"window_start": 100.0, "window_end": 130.0, "clip_ids": ["a", "b"]}
        context_index = {
            "duration": 400.0,
            "ordered": [
                {"window_start": 40.0, "window_end": 70.0, "label_class": "hard_negative"},
                {"window_start": 100.0, "window_end": 130.0, "label_class": "positive"},
                {"window_start": 180.0, "window_end": 220.0, "label_class": "positive"},
            ],
            "positives": [
                {"window_start": 100.0, "window_end": 130.0, "label_class": "positive"},
                {"window_start": 180.0, "window_end": 220.0, "label_class": "positive"},
            ],
        }
        build_twitch_training_dataset._apply_vod_context_features(row, label, context_index)
        self.assertGreater(row["features"]["vod_progress"], 0.0)
        self.assertEqual(row["features"]["clip_overlap_count"], 2.0)

    def test_extract_clip_modalities_collects_transcript_audio_and_video(self) -> None:
        transcript = [RichSegment(start=0.0, end=1.0, text="test")]
        audio_frames = [SimpleNamespace(second=0)]
        visual_frames = [SimpleNamespace(second=0)]

        with patch("services.transcriber.transcribe", return_value=transcript) as transcribe_mock, \
             patch("services.audio_features.extract", return_value=audio_frames) as audio_mock, \
             patch("services.video_features.extract", return_value=visual_frames) as video_mock:
            result = build_twitch_training_dataset._extract_clip_modalities(
                Path("clip.mp4"),
                Path("audio.wav"),
            )

        self.assertEqual(result, (transcript, audio_frames, visual_frames))
        transcribe_mock.assert_called_once_with(Path("audio.wav"))
        audio_mock.assert_called_once_with(Path("audio.wav"))
        video_mock.assert_called_once_with(Path("clip.mp4"))

    def test_build_dataset_uses_vod_checkpoint_when_present(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]
        checkpoint_rows = [{"label": 1, "label_class": "positive", "features": {"candidate_id": "cached"}}]

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            build_twitch_training_dataset._write_vod_checkpoint(work_dir, "jynxzi", "123", labels, checkpoint_rows)
            with patch("ml.build_twitch_training_dataset._materialize_vod") as materialize_mock, \
                 patch("ml.build_twitch_training_dataset._build_rows_for_label") as build_rows_mock:
                rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", work_dir)

        self.assertEqual(rows, checkpoint_rows)
        materialize_mock.assert_not_called()
        build_rows_mock.assert_not_called()

    def test_build_dataset_writes_vod_checkpoint_after_processing(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]
        built_rows = [
            {
                "label": 1,
                "label_class": "positive",
                "label_source": "twitch_clip_api",
                "features": {"candidate_id": "x"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", return_value=built_rows):
            work_dir = Path(tmp) / "work"
            build_twitch_training_dataset.build_dataset(labels, "jynxzi", work_dir)
            checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", labels)

        self.assertEqual(checkpoint, built_rows)

    def test_build_dataset_invalidates_checkpoint_when_fingerprint_changes(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]
        checkpoint_rows = [{"label": 1, "label_class": "positive", "features": {"candidate_id": "cached"}}]

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            build_twitch_training_dataset._write_vod_checkpoint(work_dir, "jynxzi", "123", labels, checkpoint_rows)
            with patch("ml.build_twitch_training_dataset._dataset_build_fingerprint", side_effect=["fingerprint-a", "fingerprint-b"]):
                checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", labels)

        self.assertIsNone(checkpoint)

    def test_materialize_vod_always_downloads_directly_without_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "storage_dir", tmp):
            downloaded = Path(tmp) / "job-1" / "video.mp4"
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            downloaded.write_bytes(b"video")
            with patch("ml.build_twitch_training_dataset.downloader.download", return_value=downloaded) as download_mock:
                path = build_twitch_training_dataset._materialize_vod("https://www.twitch.tv/videos/456", "job-1")

        self.assertEqual(path, downloaded)
        download_mock.assert_called_once_with("https://www.twitch.tv/videos/456", "job-1")

    def test_build_dataset_skips_failed_window_and_keeps_processing(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 30.0,
                "window_end": 40.0,
                "label_class": "hard_negative",
                "label_score": 0.0,
                "clip_ids": [],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]
        fake_row = {
            "label": 0,
            "label_source": "twitch_clip_api",
            "features": {
                "candidate_id": "x",
                "duration": 10.0,
                "hook_strength": 0.1,
            },
        }

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_label_workers", 1), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch(
                 "ml.build_twitch_training_dataset._build_rows_for_label",
                 side_effect=[RuntimeError("ffmpeg failed"), [fake_row]],
             ) as build_rows_mock:
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_class"], "hard_negative")
        self.assertEqual(build_rows_mock.call_count, 2)

    def test_build_dataset_parallel_label_workers_preserve_label_order(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 30.0,
                "window_end": 40.0,
                "label_class": "hard_negative",
                "label_score": 0.0,
                "clip_ids": [],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]

        def _fake_build_rows_for_label(label: dict, **_kwargs) -> list[dict]:
            if float(label["window_start"]) == 10.0:
                time.sleep(0.05)
            return [{
                "label": 1 if label["label_class"] == "positive" else 0,
                "label_source": "twitch_clip_api",
                "features": {"candidate_id": f"cand-{int(label['window_start'])}"},
            }]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "training_label_workers", 2), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=_fake_build_rows_for_label):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual([row["window_start"] for row in rows], [10.0, 30.0])
        self.assertEqual([row["label_class"] for row in rows], ["positive", "hard_negative"])

    def test_build_dataset_parallel_label_workers_keep_processing_after_failure(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 30.0,
                "window_end": 40.0,
                "label_class": "hard_negative",
                "label_score": 0.0,
                "clip_ids": [],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]

        def _fake_build_rows_for_label(label: dict, **_kwargs) -> list[dict]:
            if float(label["window_start"]) == 10.0:
                time.sleep(0.05)
                raise RuntimeError("ffmpeg failed")
            return [{
                "label": 0,
                "label_source": "twitch_clip_api",
                "features": {"candidate_id": "cand-30"},
            }]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "training_label_workers", 2), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=_fake_build_rows_for_label):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["window_start"], 30.0)
        self.assertEqual(rows[0]["label_class"], "hard_negative")

    def test_build_dataset_batches_vod_window_cuts_once_per_vod(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 30.0,
                "window_end": 40.0,
                "label_class": "hard_negative",
                "label_score": 0.0,
                "clip_ids": [],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]

        def _fake_build_rows_for_label(label: dict, *, clip_path=None, **_kwargs) -> list[dict]:
            self.assertIsNotNone(clip_path)
            return [{
                "label": 1 if label["label_class"] == "positive" else 0,
                "label_source": "twitch_clip_api",
                "features": {"candidate_id": f"cand-{int(label['window_start'])}"},
            }]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "training_label_workers", 1), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset.collect_viral_clips.cut_video_windows") as batch_cut_mock, \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=_fake_build_rows_for_label):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 2)
        self.assertEqual(batch_cut_mock.call_count, 1)
        batch_windows = batch_cut_mock.call_args.args[1]
        self.assertEqual(len(batch_windows), 2)

    def test_build_dataset_falls_back_to_per_label_cut_when_batch_cut_fails(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 30.0,
                "window_end": 40.0,
                "label_class": "hard_negative",
                "label_score": 0.0,
                "clip_ids": [],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]

        def _fake_build_rows_for_label(label: dict, *, clip_path=None, **_kwargs) -> list[dict]:
            self.assertIsNone(clip_path)
            return [{
                "label": 1 if label["label_class"] == "positive" else 0,
                "label_source": "twitch_clip_api",
                "features": {"candidate_id": f"cand-{int(label['window_start'])}"},
            }]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "training_label_workers", 1), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch(
                 "ml.build_twitch_training_dataset.collect_viral_clips.cut_video_windows",
                 side_effect=RuntimeError("ffmpeg failed"),
             ), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=_fake_build_rows_for_label):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 2)

    def test_build_dataset_starts_prefetch_for_next_vod(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "456",
                "vod_url": "https://www.twitch.tv/videos/456",
                "window_start": 30.0,
                "window_end": 42.0,
                "label_class": "positive",
                "label_score": 0.7,
                "clip_ids": ["c2"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]
        fake_row = {
            "label": 1,
            "label_source": "twitch_clip_api",
            "features": {"candidate_id": "x"},
        }
        materialize_calls: list[str] = []

        def _fake_materialize(url: str, job_id: str) -> Path:
            materialize_calls.append(job_id)
            return Path("/tmp") / f"{job_id}.mp4"

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch("ml.build_twitch_training_dataset._materialize_vod", side_effect=_fake_materialize), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=lambda *args, **kwargs: [dict(fake_row)]):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 2)
        self.assertIn("twitch_vod_prefetch_456", materialize_calls)

    def test_build_dataset_parallel_workers_preserve_vod_order(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "456",
                "vod_url": "https://www.twitch.tv/videos/456",
                "window_start": 30.0,
                "window_end": 42.0,
                "label_class": "positive",
                "label_score": 0.7,
                "clip_ids": ["c2"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]

        def _fake_build_rows_for_vod(vod_id: str, *_args, **_kwargs):
            if vod_id == "123":
                time.sleep(0.05)
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": vod_id}}]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 2), \
             patch("ml.build_twitch_training_dataset._build_rows_for_vod", side_effect=_fake_build_rows_for_vod):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual([row["vod_id"] for row in rows], ["123", "456"])

    def test_build_dataset_parallel_workers_prefetch_download_futures(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "456",
                "vod_url": "https://www.twitch.tv/videos/456",
                "window_start": 30.0,
                "window_end": 42.0,
                "label_class": "positive",
                "label_score": 0.7,
                "clip_ids": ["c2"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "789",
                "vod_url": "https://www.twitch.tv/videos/789",
                "window_start": 50.0,
                "window_end": 62.0,
                "label_class": "positive",
                "label_score": 0.6,
                "clip_ids": ["c3"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]
        seen_prefetch_flags: list[tuple[str, bool]] = []

        def _fake_build_rows_for_vod(vod_id: str, *_args, **kwargs):
            vod_path_future = kwargs.get("vod_path_future")
            seen_prefetch_flags.append((vod_id, vod_path_future is not None))
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": vod_id}}]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 2), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._build_rows_for_vod", side_effect=_fake_build_rows_for_vod):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual([row["vod_id"] for row in rows], ["123", "456", "789"])
        self.assertCountEqual(seen_prefetch_flags, [("123", True), ("456", True), ("789", True)])

    def test_build_dataset_parallel_path_skips_prefetch_for_checkpoint_hits(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "456",
                "vod_url": "https://www.twitch.tv/videos/456",
                "window_start": 30.0,
                "window_end": 42.0,
                "label_class": "positive",
                "label_score": 0.7,
                "clip_ids": ["c2"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]

        def _fake_load_checkpoint(_work_dir, _profile, vod_id, _vod_rows):
            if vod_id == "123":
                return [{"vod_id": "123", "label": 1, "features": {"candidate_id": "cached-123"}}]
            return None

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 2), \
             patch("ml.build_twitch_training_dataset._load_vod_checkpoint", side_effect=_fake_load_checkpoint), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4") as materialize_mock, \
             patch(
                 "ml.build_twitch_training_dataset._build_rows_for_vod",
                 return_value=[{"vod_id": "456", "label": 1, "features": {"candidate_id": "456"}}],
             ):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual([row["vod_id"] for row in rows], ["123", "456"])
        self.assertEqual(materialize_mock.call_count, 1)
        self.assertEqual(materialize_mock.call_args.args[1], "twitch_vod_456")

    def test_refined_candidates_prefer_overlapping_generator_windows(self) -> None:
        proxy = build_twitch_training_dataset._proxy_candidate(
            {"window_start": 100.0, "window_end": 132.0},
            90.0,
            "job-1",
        )
        best = build_twitch_training_dataset._clone_candidate(
            build_twitch_training_dataset._proxy_candidate(
                {"window_start": 102.0, "window_end": 126.0},
                90.0,
                "seed",
            ),
            "job-1",
            0,
        )
        best.source = "story_payoff"
        best.score_hint = 6.0
        weaker = build_twitch_training_dataset._clone_candidate(
            build_twitch_training_dataset._proxy_candidate(
                {"window_start": 95.0, "window_end": 120.0},
                90.0,
                "seed",
            ),
            "job-1",
            1,
        )
        weaker.source = "reaction"
        weaker.score_hint = 3.0

        with patch("services.candidate_generators.generate_semantic_candidates", return_value=[best]), \
             patch("services.candidate_generators.generate_fallback_candidates", return_value=[weaker]), \
             patch("services.candidate_generators.merge_prioritized_candidates", return_value=[best, weaker]):
            selected = build_twitch_training_dataset._refined_candidates_for_label(
                {"label_class": "positive"},
                transcript=[],
                proxy_candidate=proxy,
                job_prefix="job-1",
            )

        self.assertEqual(selected[0].source, "story_payoff")
        self.assertGreaterEqual(len(selected), 2)

    def test_training_row_from_label_uses_refined_candidate_window(self) -> None:
        candidate = build_twitch_training_dataset._proxy_candidate(
            {"window_start": 120.0, "window_end": 145.0},
            100.0,
            "job-1",
        )
        candidate.start = 121.5
        candidate.end = 139.0
        candidate.source = "story_payoff"

        class _Features:
            candidate_id = "feat-1"

            def to_dict(self):
                return {"candidate_id": "feat-1", "duration": 17.5, "source": "story_payoff"}

        row = build_twitch_training_dataset._training_row_from_label(
            {
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "broadcaster": "jynxzi",
                "label_class": "positive",
                "label_source": "twitch_clip_api",
                "window_start": 120.0,
                "window_end": 145.0,
                "clip_ids": ["a"],
            },
            candidate,
            _Features(),
            sample_weight_scale=1.0,
            dataset_source="twitch_vod_primary",
            supervision_reason_suffix="",
            window_origin="proxy_label_refined",
        )

        self.assertEqual(row["window_start"], 121.5)
        self.assertEqual(row["proxy_window_start"], 120.0)
        self.assertEqual(row["candidate_source"], "story_payoff")

    def test_dedupe_ranker_rows_prefers_primary_proxy_window_per_clip_cluster(self) -> None:
        primary = {
            "candidate_id": "cand-primary",
            "label": 1,
            "label_class": "positive",
            "vod_id": "vod-1",
            "clip_ids": ["clip-a"],
            "dataset_source": "twitch_vod_primary",
            "window_origin": "proxy_label_refined",
            "sample_weight": 1.5,
        }
        supplemental = {
            "candidate_id": "cand-supplemental",
            "label": 1,
            "label_class": "positive",
            "vod_id": "vod-1",
            "clip_ids": ["clip-a"],
            "dataset_source": "twitch_vod_candidate_supplemental",
            "window_origin": "candidate_generator_overlap",
            "sample_weight": 0.975,
        }
        other_positive = {
            "candidate_id": "cand-other",
            "label": 1,
            "label_class": "positive",
            "vod_id": "vod-2",
            "clip_ids": ["clip-b"],
            "dataset_source": "twitch_vod_primary",
            "window_origin": "proxy_label_refined",
            "sample_weight": 1.5,
        }
        negative = {
            "candidate_id": "cand-negative",
            "label": 0,
            "label_class": "hard_negative",
            "vod_id": "vod-1",
            "clip_ids": [],
            "dataset_source": "twitch_vod_primary",
            "window_origin": "proxy_label_refined",
            "sample_weight": 1.8,
        }

        rows = build_twitch_training_dataset._dedupe_ranker_rows([primary, supplemental, other_positive, negative])

        self.assertEqual([row["candidate_id"] for row in rows], ["cand-primary", "cand-other", "cand-negative"])

    def test_boundary_perturbation_candidates_generate_valid_near_misses(self) -> None:
        transcript = [
            RichSegment(start=0.0, end=2.0, text="Setup.", speaker="STREAMER"),
            RichSegment(start=2.0, end=4.0, text="Wait what?", speaker="STREAMER", emotions=["question"], hype_score=2.5),
            RichSegment(start=4.0, end=6.0, text="No way!", speaker="STREAMER", emotions=["exclamation"], hype_score=5.0),
            RichSegment(start=6.0, end=9.0, text="Everyone laughs.", speaker="STREAMER", emotions=["laughter"], hype_score=4.2),
            RichSegment(start=9.0, end=12.0, text="Wrap up.", speaker="STREAMER"),
        ]
        reference = Candidate(
            id="cand",
            start=2.0,
            end=9.0,
            source="reaction",
            anchor_time=5.0,
            score_hint=5.0,
            tags=[],
        )

        candidates = build_twitch_training_dataset._boundary_perturbation_candidates(
            {"label_class": "positive"},
            transcript,
            reference,
            "job-1",
        )

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.start < reference.anchor_time < candidate.end for candidate in candidates))
        self.assertTrue(all(candidate.source == "boundary_perturbation" for candidate in candidates))
        self.assertTrue(any(candidate.start != reference.start or candidate.end != reference.end for candidate in candidates))

    def test_training_row_can_capture_boundary_reference_metadata(self) -> None:
        candidate = Candidate(
            id="cand",
            start=121.5,
            end=139.0,
            source="boundary_perturbation",
            anchor_time=130.0,
            score_hint=1.0,
            tags=["boundary_near_miss"],
        )
        features = SimpleNamespace(candidate_id="feat-1", to_dict=lambda: {"candidate_id": "feat-1", "duration": 17.5})

        row = build_twitch_training_dataset._training_row_from_label(
            {
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "broadcaster": "jynxzi",
                "label_class": "positive",
                "label_source": "twitch_clip_api",
                "window_start": 120.0,
                "window_end": 145.0,
                "clip_ids": ["a"],
            },
            candidate,
            features,
            sample_weight_scale=0.45,
            dataset_source="twitch_vod_boundary_near_miss",
            supervision_reason_suffix=":boundary_near_miss:start_late_small",
            window_origin="boundary_perturbed",
            label_class_override="boundary_near_miss_negative",
            label_value_override=0,
            supervision_tier="boundary_near_miss",
            rejected_reason_override="boundary_near_miss",
            reference_window=(120.0, 145.0),
            perturbation_kind="start_late_small",
        )

        self.assertEqual(row["label"], 0)
        self.assertEqual(row["label_class"], "boundary_near_miss_negative")
        self.assertEqual(row["boundary_reference_start"], 120.0)
        self.assertEqual(row["boundary_reference_end"], 145.0)
        self.assertEqual(row["boundary_error_start_sec"], 1.5)
        self.assertEqual(row["boundary_error_end_sec"], -6.0)
        self.assertEqual(row["boundary_perturbation_kind"], "start_late_small")
        self.assertEqual(row["supervision_tier"], "boundary_near_miss")


if __name__ == "__main__":
    unittest.main()
