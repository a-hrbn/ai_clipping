import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import sys
import types

import numpy as np

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from config import settings
from ml import build_twitch_training_dataset
from services.timeline_types import Candidate
from services.transcriber import RichSegment


class BuildTwitchTrainingDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._extended_window_patcher = patch.object(settings, "training_vod_extended_window_scoped", False)
        self._extended_window_patcher.start()

    def tearDown(self) -> None:
        self._extended_window_patcher.stop()

    def _fake_vod_feature_context(self, root: Path) -> build_twitch_training_dataset.VodFeatureContext:
        cache_dir = root / "vod_feature_cache" / "123"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return build_twitch_training_dataset.VodFeatureContext(
            transcript=[RichSegment(start=0.0, end=60.0, text="context")],
            audio_signal=[],
            audio_sample_rate=16000,
            cache_dir=cache_dir,
        )

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
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
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
                Path("vod.mp4"),
                Path("audio.wav"),
                window_start=10.0,
                window_end=28.0,
            )

        self.assertEqual(result, (transcript, audio_frames, visual_frames))
        transcribe_mock.assert_called_once_with(Path("audio.wav"))
        audio_mock.assert_called_once_with(Path("audio.wav"))
        video_mock.assert_called_once_with(Path("vod.mp4"), start_sec=10.0, end_sec=28.0)

    def test_slice_transcript_for_window_clamps_and_rebases_times(self) -> None:
        transcript = [
            RichSegment(start=8.0, end=12.0, text="before"),
            RichSegment(start=12.5, end=15.0, text="inside"),
            RichSegment(start=19.0, end=24.0, text="after"),
        ]

        sliced = build_twitch_training_dataset._slice_transcript_for_window(
            transcript,
            window_start=10.0,
            window_end=20.0,
        )

        self.assertEqual(
            [(segment.text, segment.start, segment.end) for segment in sliced],
            [("before", 0.0, 2.0), ("inside", 2.5, 5.0), ("after", 9.0, 10.0)],
        )

    def test_window_scoped_label_detection_requires_all_top_viewed_positive_labels(self) -> None:
        top_viewed = {
            "vod_id": "123",
            "vod_url": "https://www.twitch.tv/videos/123",
            "label_class": "positive",
            "manual_gold_review_source": "top_viewed_twitch_30d",
        }
        weak_top_viewed = {
            **top_viewed,
            "label_class": "weak_positive",
        }
        regular_positive = {
            **top_viewed,
            "manual_gold_review_source": "",
        }
        hard_negative = {
            **top_viewed,
            "label_class": "hard_negative",
        }

        self.assertTrue(
            build_twitch_training_dataset._should_use_window_scoped_label_contexts(
                [top_viewed, weak_top_viewed]
            )
        )
        self.assertFalse(
            build_twitch_training_dataset._should_use_window_scoped_label_contexts(
                [top_viewed, regular_positive]
            )
        )
        self.assertFalse(
            build_twitch_training_dataset._should_use_window_scoped_label_contexts(
                [top_viewed, hard_negative]
            )
        )

    def test_local_context_training_window_rebases_scoped_context_offsets(self) -> None:
        label = {
            "window_start": 100.0,
            "window_end": 110.0,
            "label_class": "positive",
        }
        context = build_twitch_training_dataset.VodFeatureContext(
            transcript=[],
            audio_signal=[],
            audio_sample_rate=16000,
            cache_dir=Path("cache"),
            source_start_sec=94.0,
            source_end_sec=118.0,
            source_kind="window_scoped_top_viewed",
        )

        absolute_start, absolute_end, local_start, local_end = (
            build_twitch_training_dataset._local_context_training_window(label, context)
        )

        self.assertEqual((absolute_start, absolute_end), (94.0, 118.0))
        self.assertEqual((local_start, local_end), (0.0, 24.0))

    def test_local_context_training_window_uses_full_extended_scoped_context(self) -> None:
        label = {
            "window_start": 100.0,
            "window_end": 110.0,
            "label_class": "positive",
        }
        context = build_twitch_training_dataset.VodFeatureContext(
            transcript=[],
            audio_signal=[],
            audio_sample_rate=16000,
            cache_dir=Path("cache"),
            source_start_sec=40.0,
            source_end_sec=410.0,
            source_kind="extended_window_scoped",
        )

        absolute_start, absolute_end, local_start, local_end = (
            build_twitch_training_dataset._local_context_training_window(label, context)
        )

        self.assertEqual((absolute_start, absolute_end), (40.0, 410.0))
        self.assertEqual((local_start, local_end), (0.0, 370.0))

    def test_write_rebased_chat_file_shifts_absolute_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chat_path = Path(tmp) / "chat.json"
            output_path = Path(tmp) / "rebased" / "chat.json"
            chat_path.write_text(
                json.dumps(
                    [
                        {"t": 90.0, "text": "before"},
                        {"t": 100.5, "text": "inside"},
                        {"t": 118.0, "text": "edge"},
                        {"t": 120.0, "text": "after"},
                    ]
                ),
                encoding="utf-8",
            )

            result = build_twitch_training_dataset._write_rebased_chat_file(
                chat_path,
                output_path,
                source_start_sec=94.0,
                source_end_sec=118.0,
            )

            self.assertEqual(result, output_path)
            rebased = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [(row["t"], row["text"]) for row in rebased],
                [(6.5, "inside"), (24.0, "edge")],
            )

    def test_window_scoped_vod_labels_use_one_scoped_context_per_label(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 100.0,
                "window_end": 110.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "manual_gold_review_source": "top_viewed_twitch_30d",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 200.0,
                "window_end": 212.0,
                "label_class": "weak_positive",
                "label_score": 0.55,
                "clip_ids": ["c2"],
                "manual_gold_review_source": "top_viewed_twitch_30d",
            },
        ]
        scoped_paths = [Path("scoped-1.mp4"), Path("scoped-2.mp4")]
        contexts = [
            build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path("cache-1"),
                source_start_sec=94.0,
                source_kind="window_scoped_top_viewed",
            ),
            build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path("cache-2"),
                source_start_sec=194.0,
                source_kind="window_scoped_top_viewed",
            ),
        ]
        calls: list[tuple[str, Path]] = []

        def _fake_scoped_context(_vod_id: str, _label: dict, **_kwargs):
            index = len(calls)
            return scoped_paths[index], contexts[index]

        def _fake_process(vod_id: str, index: int, label: dict, *, vod_path: Path, **_kwargs):
            calls.append((str(label["clip_ids"][0]), vod_path))
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": f"row-{index}"}}]

        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context", side_effect=_fake_scoped_context) as scoped_mock, \
             patch("ml.build_twitch_training_dataset._process_vod_label", side_effect=_fake_process):
            rows = build_twitch_training_dataset._build_rows_for_vod_labels(
                "123",
                labels,
                vod_path=Path(tmp) / "vod.mp4",
                vod_dir=Path(tmp) / "job",
                job_id="twitch_vod_123",
                profile="jynxzi",
                context_index={},
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(calls, [("c1", scoped_paths[0]), ("c2", scoped_paths[1])])
        self.assertEqual(scoped_mock.call_count, 2)

    def test_window_scoped_vod_unavailable_skips_remaining_labels(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "2708442778",
                "vod_url": "https://www.twitch.tv/videos/2708442778",
                "window_start": 410.0,
                "window_end": 497.6,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "manual_gold_review_source": "top_viewed_twitch_30d",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "2708442778",
                "vod_url": "https://www.twitch.tv/videos/2708442778",
                "window_start": 600.0,
                "window_end": 660.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c2"],
                "manual_gold_review_source": "top_viewed_twitch_30d",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp, \
             patch(
                 "ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context",
                 side_effect=RuntimeError("ERROR: [twitch:vod] 2708442778: Video 2708442778 does not exist"),
             ) as scoped_mock, \
             patch("ml.build_twitch_training_dataset._process_vod_label") as process_mock:
            with self.assertLogs("ml.build_twitch_training_dataset", level="WARNING") as logs:
                rows = build_twitch_training_dataset._build_rows_for_vod_labels(
                    "2708442778",
                    labels,
                    vod_path=None,
                    vod_dir=Path(tmp) / "job",
                    job_id="twitch_vod_2708442778",
                    profile="jynxzi",
                    context_index={},
                )

        self.assertEqual(rows, [])
        self.assertEqual(scoped_mock.call_count, 1)
        process_mock.assert_not_called()
        self.assertIn("training_vod_window_scoped_vod_unavailable", "\n".join(logs.output))

    def test_window_scoped_context_without_vod_path_downloads_only_range(self) -> None:
        label = {
            "broadcaster": "jynxzi",
            "vod_id": "123",
            "vod_url": "https://www.twitch.tv/videos/123",
            "window_start": 100.0,
            "window_end": 110.0,
            "label_class": "positive",
            "label_score": 0.8,
            "clip_ids": ["c1"],
            "manual_gold_review_source": "top_viewed_twitch_30d",
        }

        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(tmp) / "downloaded.mp4"
            downloaded.write_bytes(b"scoped-video")
            context = build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path(tmp) / "cache",
                source_start_sec=94.0,
                source_kind="window_scoped_top_viewed",
            )
            calls: dict[str, object] = {}

            def _fake_download_range(url: str, job_id: str, **kwargs):
                calls["url"] = url
                calls["job_id"] = job_id
                calls["kwargs"] = kwargs
                return downloaded

            with patch("services.downloader.download_range", side_effect=_fake_download_range) as range_mock, \
                 patch("ml.build_twitch_training_dataset._vod_feature_cache_dir", return_value=Path(tmp) / "vod-cache"), \
                 patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=context) as context_mock:
                scoped_path, result_context = (
                    build_twitch_training_dataset._load_or_build_window_scoped_feature_context(
                        "123",
                        label,
                        vod_path=None,
                        job_id="twitch_vod_123:window_scoped:1",
                    )
                )

                self.assertEqual(result_context, context)
                self.assertEqual(scoped_path.read_bytes(), b"scoped-video")
                self.assertEqual(calls["url"], "https://www.twitch.tv/videos/123")
                self.assertEqual(calls["kwargs"], {"start_sec": 94.0, "end_sec": 118.0})
                range_mock.assert_called_once()
                context_mock.assert_called_once()
                self.assertEqual(context_mock.call_args.kwargs["vod_path"], scoped_path)
                self.assertEqual(context_mock.call_args.kwargs["source_start_sec"], 94.0)
                self.assertEqual(context_mock.call_args.kwargs["source_end_sec"], 118.0)

    def test_build_dataset_uses_scoped_range_without_full_vod_for_top_viewed_labels(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 100.0,
                "window_end": 110.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "manual_gold_review_source": "top_viewed_twitch_30d",
            },
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 200.0,
                "window_end": 212.0,
                "label_class": "weak_positive",
                "label_score": 0.55,
                "clip_ids": ["c2"],
                "manual_gold_review_source": "top_viewed_twitch_30d",
            },
        ]
        contexts = [
            build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path("cache-1"),
                source_start_sec=94.0,
                source_kind="window_scoped_top_viewed",
            ),
            build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path("cache-2"),
                source_start_sec=194.0,
                source_kind="window_scoped_top_viewed",
            ),
        ]

        def _fake_scoped_context(_vod_id: str, _label: dict, **_kwargs):
            index = 0 if _label["clip_ids"] == ["c1"] else 1
            return Path(f"scoped-{index}.mp4"), contexts[index]

        def _fake_process(vod_id: str, index: int, _label: dict, **_kwargs):
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": f"row-{index}"}}]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "enable_chat_analysis", False), \
             patch("ml.build_twitch_training_dataset._materialize_vod") as materialize_mock, \
             patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context", side_effect=_fake_scoped_context) as scoped_mock, \
             patch("ml.build_twitch_training_dataset._process_vod_label", side_effect=_fake_process):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual([row["features"]["candidate_id"] for row in rows], ["row-1", "row-2"])
        materialize_mock.assert_not_called()
        self.assertEqual(scoped_mock.call_count, 2)

    def test_build_dataset_uses_extended_scoped_range_without_full_vod_for_regular_labels(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 100.0,
                "window_end": 110.0,
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
                "window_start": 200.0,
                "window_end": 212.0,
                "label_class": "hard_negative",
                "label_score": 0.1,
                "clip_ids": ["c2"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]
        seen_source_kinds: list[str] = []
        seen_context_label_counts: list[int] = []
        seen_scoped_windows: list[tuple[float, float]] = []

        def _fake_scoped_context(_vod_id: str, _label: dict, **kwargs):
            seen_source_kinds.append(str(kwargs.get("source_kind")))
            seen_context_label_counts.append(len(kwargs.get("context_labels") or []))
            seen_scoped_windows.append(kwargs.get("scoped_window"))
            return Path(f"scoped-{len(seen_source_kinds)}.mp4"), build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path("cache"),
                source_start_sec=0.0,
                source_kind=str(kwargs.get("source_kind")),
            )

        def _fake_process(vod_id: str, index: int, _label: dict, **_kwargs):
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": f"row-{index}"}}]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "training_vod_extended_window_scoped", True), \
             patch.object(settings, "enable_chat_analysis", False), \
             patch("ml.build_twitch_training_dataset._materialize_vod") as materialize_mock, \
             patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context", side_effect=_fake_scoped_context), \
             patch("ml.build_twitch_training_dataset._process_vod_label", side_effect=_fake_process):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual([row["features"]["candidate_id"] for row in rows], ["row-1", "row-2"])
        self.assertEqual(seen_source_kinds, ["extended_window_scoped"])
        self.assertEqual(seen_context_label_counts, [2])
        self.assertEqual(seen_scoped_windows, [(0.0, 512.0)])
        materialize_mock.assert_not_called()

    def test_extended_window_scoped_groups_overlapping_windows_with_max_duration_cap(self) -> None:
        labels = [
            {
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 100.0,
                "window_end": 110.0,
                "label_class": "positive",
            },
            {
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 200.0,
                "window_end": 212.0,
                "label_class": "hard_negative",
            },
            {
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 1200.0,
                "window_end": 1210.0,
                "label_class": "positive",
            },
        ]

        with patch.object(settings, "training_vod_extended_window_scoped", True), \
             patch.object(settings, "training_vod_extended_context_sec", 300), \
             patch.object(settings, "training_vod_extended_group_max_sec", 900):
            groups = build_twitch_training_dataset._extended_window_scoped_label_groups(labels)

        self.assertEqual(len(groups), 2)
        self.assertEqual(
            [(group["expanded_start"], group["expanded_end"], len(group["items"])) for group in groups],
            [(0.0, 512.0, 2), (900.0, 1510.0, 1)],
        )

    def test_extended_window_scoped_prefetches_next_group_source_only(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": start,
                "window_end": start + 10.0,
                "label_class": label_class,
                "label_score": 0.8,
                "clip_ids": [f"c{index}"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            }
            for index, (start, label_class) in enumerate(
                [(100.0, "positive"), (1200.0, "hard_negative"), (2400.0, "positive")],
                start=1,
            )
        ]
        source_job_ids: list[str] = []
        context_job_ids: list[str] = []

        def _fake_source(_vod_id: str, _label: dict, **kwargs):
            source_job_ids.append(str(kwargs["job_id"]))
            source_path = Path(f"source-{len(source_job_ids)}.mp4")
            return build_twitch_training_dataset.WindowScopedSourcePlan(
                scoped_kind=str(kwargs["source_kind"]),
                expanded_start=float(kwargs["scoped_window"][0]),
                expanded_end=float(kwargs["scoped_window"][1]),
                cache_dir=Path(f"cache-{len(source_job_ids)}"),
                scoped_video_path=source_path,
                metadata_path=Path(f"metadata-{len(source_job_ids)}.json"),
                metadata_payload={},
                uses_local_vod_source=False,
            )

        def _fake_source_plan(_vod_id: str, _label: dict, **kwargs):
            return build_twitch_training_dataset.WindowScopedSourcePlan(
                scoped_kind=str(kwargs["source_kind"]),
                expanded_start=float(kwargs["scoped_window"][0]),
                expanded_end=float(kwargs["scoped_window"][1]),
                cache_dir=Path("source-plan-cache"),
                scoped_video_path=Path("source-plan.mp4"),
                metadata_path=Path("source-plan.metadata.json"),
                metadata_payload={},
                uses_local_vod_source=False,
            )

        def _fake_context(_vod_id: str, _label: dict, **kwargs):
            context_job_ids.append(str(kwargs["job_id"]))
            return Path(f"context-{len(context_job_ids)}.mp4"), build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path(f"context-cache-{len(context_job_ids)}"),
                source_start_sec=float(kwargs["scoped_window"][0]),
                source_end_sec=float(kwargs["scoped_window"][1]),
                source_kind=str(kwargs["source_kind"]),
            )

        def _fake_process(vod_id: str, index: int, _label: dict, **_kwargs):
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": f"row-{index}"}}]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_extended_window_scoped", True), \
             patch.object(settings, "training_vod_group_source_prefetch", True), \
             patch.object(settings, "training_vod_extended_context_sec", 300), \
             patch.object(settings, "training_vod_extended_group_max_sec", 900), \
             patch("ml.build_twitch_training_dataset._window_scoped_source_plan", side_effect=_fake_source_plan), \
             patch("ml.build_twitch_training_dataset._window_scoped_source_plan_is_usable", return_value=False), \
             patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_source", side_effect=_fake_source), \
             patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context", side_effect=_fake_context), \
             patch("ml.build_twitch_training_dataset._process_vod_label", side_effect=_fake_process):
            rows = build_twitch_training_dataset._build_rows_for_vod_labels(
                "123",
                labels,
                vod_path=None,
                vod_dir=Path(tmp) / "job",
                job_id="twitch_vod_123",
                profile="jynxzi",
                context_index={},
            )

        self.assertEqual([row["features"]["candidate_id"] for row in rows], ["row-1", "row-2", "row-3"])
        self.assertEqual(
            source_job_ids,
            [
                "twitch_vod_123:window_scoped_group:1:source",
                "twitch_vod_123:window_scoped_group:2:source_prefetch",
                "twitch_vod_123:window_scoped_group:3:source_prefetch",
            ],
        )
        self.assertEqual(
            context_job_ids,
            [
                "twitch_vod_123:window_scoped_group:1",
                "twitch_vod_123:window_scoped_group:2",
                "twitch_vod_123:window_scoped_group:3",
            ],
        )

    def test_extended_window_scoped_source_prefetch_can_be_disabled(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": start,
                "window_end": start + 10.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": [f"c{index}"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            }
            for index, start in enumerate([100.0, 1200.0], start=1)
        ]
        context_job_ids: list[str] = []

        def _fake_context(_vod_id: str, _label: dict, **kwargs):
            context_job_ids.append(str(kwargs["job_id"]))
            return Path(f"context-{len(context_job_ids)}.mp4"), build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path(f"context-cache-{len(context_job_ids)}"),
                source_start_sec=float(kwargs["scoped_window"][0]),
                source_end_sec=float(kwargs["scoped_window"][1]),
                source_kind=str(kwargs["source_kind"]),
            )

        def _fake_process(vod_id: str, index: int, _label: dict, **_kwargs):
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": f"row-{index}"}}]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_extended_window_scoped", True), \
             patch.object(settings, "training_vod_group_source_prefetch", False), \
             patch.object(settings, "training_vod_extended_context_sec", 300), \
             patch.object(settings, "training_vod_extended_group_max_sec", 900), \
             patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_source") as source_mock, \
             patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context", side_effect=_fake_context), \
             patch("ml.build_twitch_training_dataset._process_vod_label", side_effect=_fake_process):
            rows = build_twitch_training_dataset._build_rows_for_vod_labels(
                "123",
                labels,
                vod_path=None,
                vod_dir=Path(tmp) / "job",
                job_id="twitch_vod_123",
                profile="jynxzi",
                context_index={},
            )

        self.assertEqual([row["features"]["candidate_id"] for row in rows], ["row-1", "row-2"])
        self.assertEqual(
            context_job_ids,
            [
                "twitch_vod_123:window_scoped_group:1",
                "twitch_vod_123:window_scoped_group:2",
            ],
        )
        source_mock.assert_not_called()

    def test_extended_window_scoped_writes_and_reuses_group_checkpoints(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": start,
                "window_end": start + 10.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": [f"c{index}"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            }
            for index, start in enumerate([100.0, 1200.0], start=1)
        ]
        context_calls: list[str] = []

        def _fake_context(_vod_id: str, _label: dict, **kwargs):
            context_calls.append(str(kwargs["job_id"]))
            return Path(f"context-{len(context_calls)}.mp4"), build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path(f"context-cache-{len(context_calls)}"),
                source_start_sec=float(kwargs["scoped_window"][0]),
                source_end_sec=float(kwargs["scoped_window"][1]),
                source_kind=str(kwargs["source_kind"]),
            )

        def _fake_process(vod_id: str, index: int, _label: dict, **_kwargs):
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": f"row-{index}"}}]

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            with patch.object(settings, "training_vod_extended_window_scoped", True), \
                 patch.object(settings, "training_vod_group_source_prefetch", False), \
                 patch.object(settings, "training_vod_extended_context_sec", 300), \
                 patch.object(settings, "training_vod_extended_group_max_sec", 900), \
                 patch.object(settings, "enable_chat_analysis", False), \
                 patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context", side_effect=_fake_context), \
                 patch("ml.build_twitch_training_dataset._process_vod_label", side_effect=_fake_process):
                first_rows = build_twitch_training_dataset._build_rows_for_vod_labels(
                    "123",
                    labels,
                    vod_path=None,
                    vod_dir=work_dir / "twitch_vod_123",
                    job_id="twitch_vod_123",
                    profile="jynxzi",
                    context_index={},
                )

            with patch.object(settings, "training_vod_extended_window_scoped", True), \
                 patch.object(settings, "training_vod_group_source_prefetch", False), \
                 patch.object(settings, "training_vod_extended_context_sec", 300), \
                 patch.object(settings, "training_vod_extended_group_max_sec", 900), \
                 patch.object(settings, "enable_chat_analysis", False), \
                 patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context") as context_mock, \
                 patch("ml.build_twitch_training_dataset._process_vod_label") as process_mock:
                second_rows = build_twitch_training_dataset._build_rows_for_vod_labels(
                    "123",
                    labels,
                    vod_path=None,
                    vod_dir=work_dir / "twitch_vod_123",
                    job_id="twitch_vod_123",
                    profile="jynxzi",
                    context_index={},
                )

        self.assertEqual([row["features"]["candidate_id"] for row in first_rows], ["row-1", "row-2"])
        self.assertEqual(second_rows, first_rows)
        self.assertEqual(
            context_calls,
            [
                "twitch_vod_123:window_scoped_group:1",
                "twitch_vod_123:window_scoped_group:2",
            ],
        )
        context_mock.assert_not_called()
        process_mock.assert_not_called()

    def test_extended_window_scoped_group_checkpoint_invalidates_when_label_window_changes(self) -> None:
        labels = [
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 100.0,
                "window_end": 110.0,
                "label_class": "positive",
                "label_score": 0.8,
                "clip_ids": ["c1"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            }
        ]
        changed_labels = [dict(labels[0], window_end=111.0)]

        def _fake_context(_vod_id: str, _label: dict, **kwargs):
            return Path("context.mp4"), build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path("context-cache"),
                source_start_sec=float(kwargs["scoped_window"][0]),
                source_end_sec=float(kwargs["scoped_window"][1]),
                source_kind=str(kwargs["source_kind"]),
            )

        def _fake_process(vod_id: str, index: int, _label: dict, **_kwargs):
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": f"row-{index}"}}]

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            with patch.object(settings, "training_vod_extended_window_scoped", True), \
                 patch.object(settings, "training_vod_group_source_prefetch", False), \
                 patch.object(settings, "training_vod_extended_context_sec", 300), \
                 patch.object(settings, "enable_chat_analysis", False), \
                 patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context", side_effect=_fake_context), \
                 patch("ml.build_twitch_training_dataset._process_vod_label", side_effect=_fake_process):
                build_twitch_training_dataset._build_rows_for_vod_labels(
                    "123",
                    labels,
                    vod_path=None,
                    vod_dir=work_dir / "twitch_vod_123",
                    job_id="twitch_vod_123",
                    profile="jynxzi",
                    context_index={},
                )

            with patch.object(settings, "training_vod_extended_window_scoped", True), \
                 patch.object(settings, "training_vod_group_source_prefetch", False), \
                 patch.object(settings, "training_vod_extended_context_sec", 300), \
                 patch.object(settings, "enable_chat_analysis", False), \
                 patch("ml.build_twitch_training_dataset._load_or_build_window_scoped_feature_context", side_effect=_fake_context) as context_mock, \
                 patch("ml.build_twitch_training_dataset._process_vod_label", side_effect=_fake_process):
                build_twitch_training_dataset._build_rows_for_vod_labels(
                    "123",
                    changed_labels,
                    vod_path=None,
                    vod_dir=work_dir / "twitch_vod_123",
                    job_id="twitch_vod_123",
                    profile="jynxzi",
                    context_index={},
                )

        context_mock.assert_called_once()

    def test_load_or_download_chat_reuses_cached_rechat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            chat_dir = cache_dir / "chat"
            chat_dir.mkdir(parents=True)
            cached = chat_dir / "chat.rechat.json"
            cached.write_text("[]", encoding="utf-8")

            with patch("services.chat_analyzer.download_chat") as download_mock, \
                 self.assertLogs(build_twitch_training_dataset.logger, level="INFO") as logs:
                result = build_twitch_training_dataset._load_or_download_chat(
                    "https://www.twitch.tv/videos/123",
                    job_id="job-1",
                    cache_dir=cache_dir,
                )
            metadata = json.loads((chat_dir / "chat.metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(result, cached)
        download_mock.assert_not_called()
        self.assertIn("training_vod_chat_cache_hit", "\n".join(logs.output))
        self.assertEqual(metadata["kind"], "vod_chat")
        self.assertEqual(metadata["vod_id"], "123")
        self.assertEqual(metadata["cache_status"], "hit")
        self.assertEqual(metadata["chat_path"], str(cached))

    def test_load_or_download_chat_downloads_and_persists_cache_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"

            def _fake_download(_url, _job_id, out_dir, **_kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "chat.rechat.json"
                path.write_text("[]", encoding="utf-8")
                return path

            with patch("services.chat_analyzer.download_chat", side_effect=_fake_download) as download_mock, \
                 self.assertLogs(build_twitch_training_dataset.logger, level="INFO") as logs:
                result = build_twitch_training_dataset._load_or_download_chat(
                    "https://www.twitch.tv/videos/123",
                    job_id="job-1",
                    cache_dir=cache_dir,
                )
            metadata = json.loads((cache_dir / "chat" / "chat.metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(result, cache_dir / "chat" / "chat.rechat.json")
        download_mock.assert_called_once()
        self.assertEqual(download_mock.call_args.kwargs["target_offsets"], None)
        log_text = "\n".join(logs.output)
        self.assertIn("training_vod_chat_download_start", log_text)
        self.assertIn("training_vod_chat_download_success", log_text)
        self.assertEqual(metadata["cache_status"], "downloaded")
        self.assertEqual(metadata["vod_url"], "https://www.twitch.tv/videos/123")

    def test_load_or_download_chat_passes_target_offsets_to_downloader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"

            def _fake_download(_url, _job_id, out_dir, **_kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "chat.rechat.json"
                path.write_text("[]", encoding="utf-8")
                return path

            with patch("services.chat_analyzer.download_chat", side_effect=_fake_download) as download_mock:
                build_twitch_training_dataset._load_or_download_chat(
                    "https://www.twitch.tv/videos/123",
                    job_id="job-1",
                    cache_dir=cache_dir,
                    target_offsets=[10.0, 20.0],
                )

        self.assertEqual(download_mock.call_args.kwargs["target_offsets"], [10.0, 20.0])

    def test_load_or_download_chat_hits_complete_target_offset_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            chat_dir = cache_dir / "chat"
            chat_dir.mkdir(parents=True)
            cached = chat_dir / "chat.twitch_rechat.json"
            cached.write_text('[{"t": 100.0, "text": "cached", "emotes": []}]', encoding="utf-8")
            build_twitch_training_dataset._write_chat_cache_metadata(
                cache_dir,
                vod_url="https://www.twitch.tv/videos/123",
                job_id="job-old",
                chat_path=cached,
                cache_status="downloaded",
                target_offsets=[90.0, 100.0],
                covered_target_offsets=[90.0, 100.0],
                missing_target_offsets=[],
            )

            with patch("services.chat_analyzer.download_chat") as download_mock:
                result = build_twitch_training_dataset._load_or_download_chat(
                    "https://www.twitch.tv/videos/123",
                    job_id="job-1",
                    cache_dir=cache_dir,
                    target_offsets=[90.0, 100.0],
                )
            metadata = json.loads((chat_dir / "chat.metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(result, cached)
        download_mock.assert_not_called()
        self.assertEqual(metadata["cache_status"], "hit")
        self.assertEqual(metadata["coverage_status"], "complete")
        self.assertEqual(metadata["missing_target_offsets"], [])

    def test_load_or_download_chat_refills_stale_cache_missing_target_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            chat_dir = cache_dir / "chat"
            chat_dir.mkdir(parents=True)
            cached = chat_dir / "chat.twitch_rechat.json"
            cached.write_text('[{"id": "old", "t": 10.0, "text": "old", "emotes": []}]', encoding="utf-8")

            def _fake_download(_url, _job_id, out_dir, **_kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "chat.twitch_rechat.json"
                path.write_text(
                    '[{"id": "new", "t": 105.0, "text": "new", "emotes": []}]',
                    encoding="utf-8",
                )
                return path

            with patch("services.chat_analyzer.download_chat", side_effect=_fake_download) as download_mock, \
                 self.assertLogs(build_twitch_training_dataset.logger, level="INFO") as logs:
                result = build_twitch_training_dataset._load_or_download_chat(
                    "https://www.twitch.tv/videos/123",
                    job_id="job-1",
                    cache_dir=cache_dir,
                    target_offsets=[100.0, 110.0],
                )
            merged = json.loads(cached.read_text(encoding="utf-8"))
            metadata = json.loads((chat_dir / "chat.metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(result, cached)
        self.assertEqual(download_mock.call_args.kwargs["target_offsets"], [100.0, 110.0])
        self.assertEqual([row["id"] for row in merged], ["old", "new"])
        self.assertEqual(metadata["cache_status"], "refilled")
        self.assertEqual(metadata["coverage_status"], "complete")
        self.assertEqual(metadata["target_offsets"], [100, 110])
        self.assertEqual(metadata["covered_target_offsets"], [100, 110])
        self.assertEqual(metadata["missing_target_offsets"], [])
        log_text = "\n".join(logs.output)
        self.assertIn("training_vod_chat_cache_partial", log_text)
        self.assertIn("training_vod_chat_refill_success", log_text)

    def test_load_or_download_chat_marks_partial_when_refill_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            chat_dir = cache_dir / "chat"
            chat_dir.mkdir(parents=True)
            cached = chat_dir / "chat.twitch_rechat.json"
            cached.write_text('[{"id": "old", "t": 10.0, "text": "old", "emotes": []}]', encoding="utf-8")
            build_twitch_training_dataset._write_chat_cache_metadata(
                cache_dir,
                vod_url="https://www.twitch.tv/videos/123",
                job_id="job-old",
                chat_path=cached,
                cache_status="downloaded",
                target_offsets=[10.0],
                covered_target_offsets=[10.0],
                missing_target_offsets=[],
            )

            with patch("services.chat_analyzer.download_chat", return_value=None):
                result = build_twitch_training_dataset._load_or_download_chat(
                    "https://www.twitch.tv/videos/123",
                    job_id="job-1",
                    cache_dir=cache_dir,
                    target_offsets=[10.0, 100.0],
                )
            metadata = json.loads((chat_dir / "chat.metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(result, cached)
        self.assertEqual(metadata["cache_status"], "partial")
        self.assertEqual(metadata["coverage_status"], "partial")
        self.assertTrue(metadata["partial"])
        self.assertEqual(metadata["covered_target_offsets"], [10])
        self.assertEqual(metadata["missing_target_offsets"], [100])

    def test_unscoped_chat_cache_allows_vod_checkpoint_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            chat_dir = cache_dir / "chat"
            chat_dir.mkdir(parents=True)
            cached = chat_dir / "chat.twitch_rechat.json"
            cached.write_text('[{"t": 10.0, "text": "cached", "emotes": []}]', encoding="utf-8")
            build_twitch_training_dataset._write_chat_cache_metadata(
                cache_dir,
                vod_url="https://www.twitch.tv/videos/123",
                job_id="job-old",
                chat_path=cached,
                cache_status="hit",
                target_offsets=[],
                covered_target_offsets=[],
                missing_target_offsets=[],
            )

            with patch.object(build_twitch_training_dataset, "_vod_feature_cache_dir", return_value=cache_dir), \
                 patch.object(build_twitch_training_dataset.settings, "enable_chat_analysis", True):
                can_write = build_twitch_training_dataset._can_write_vod_checkpoint("123")

        self.assertTrue(can_write)

    def test_partial_unscoped_chat_cache_blocks_vod_checkpoint_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            chat_dir = cache_dir / "chat"
            chat_dir.mkdir(parents=True)
            cached = chat_dir / "chat.twitch_rechat.json"
            cached.write_text('[{"t": 10.0, "text": "cached", "emotes": []}]', encoding="utf-8")
            build_twitch_training_dataset._write_chat_cache_metadata(
                cache_dir,
                vod_url="https://www.twitch.tv/videos/123",
                job_id="job-old",
                chat_path=cached,
                cache_status="partial",
                target_offsets=[],
                covered_target_offsets=[],
                missing_target_offsets=[100],
            )

            with patch.object(build_twitch_training_dataset, "_vod_feature_cache_dir", return_value=cache_dir), \
                 patch.object(build_twitch_training_dataset.settings, "enable_chat_analysis", True):
                can_write = build_twitch_training_dataset._can_write_vod_checkpoint("123")

        self.assertFalse(can_write)

    def test_partial_chat_cache_allows_checkpoint_when_rows_have_scoped_chat_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            chat_dir = cache_dir / "chat"
            chat_dir.mkdir(parents=True)
            cached = chat_dir / "chat.local.json"
            cached.write_text('[{"t": 10.0, "text": "local", "emotes": []}]', encoding="utf-8")
            build_twitch_training_dataset._write_chat_cache_metadata(
                cache_dir,
                vod_url="https://www.twitch.tv/videos/123",
                job_id="job-old",
                chat_path=cached,
                cache_status="partial",
                target_offsets=[10, 100],
                covered_target_offsets=[10],
                missing_target_offsets=[100],
            )
            rows = [
                {
                    "feature_provenance": {"has_chat_features": True},
                    "features": {"chat_hype_score": 0.0},
                }
            ]

            with patch.object(build_twitch_training_dataset, "_vod_feature_cache_dir", return_value=cache_dir), \
                 patch.object(build_twitch_training_dataset.settings, "enable_chat_analysis", True):
                can_write = build_twitch_training_dataset._can_write_vod_checkpoint("123", rows)

        self.assertTrue(can_write)

    def test_chat_target_offsets_for_labels_cover_expanded_windows(self) -> None:
        labels = [
            {
                "label_class": "positive",
                "window_start": 100.0,
                "window_end": 120.0,
            },
            {
                "label_class": "hard_negative",
                "window_start": 180.0,
                "window_end": 190.0,
            },
            {
                "label_class": "background_negative",
                "window_start": 1000.0,
                "window_end": 1010.0,
            },
            {
                "label_class": "ambiguous",
                "window_start": 1200.0,
                "window_end": 1210.0,
            },
        ]

        offsets = build_twitch_training_dataset._chat_target_offsets_for_labels(labels)

        self.assertIn(87.0, offsets)
        self.assertIn(134.0, offsets)
        self.assertIn(175.0, offsets)
        self.assertIn(195.0, offsets)
        self.assertIn(995.0, offsets)
        self.assertEqual(offsets, sorted(offsets))

    def test_chat_target_offsets_cap_hard_negatives_near_positives(self) -> None:
        labels = [
            {
                "label_class": "positive",
                "window_start": 100.0,
                "window_end": 120.0,
            }
        ]
        labels.extend(
            {
                "label_class": "hard_negative",
                "window_start": 200.0 + index * 100.0,
                "window_end": 210.0 + index * 100.0,
            }
            for index in range(12)
        )

        target_labels = build_twitch_training_dataset._chat_target_labels_for_offsets(labels)

        self.assertEqual(len(target_labels), 1 + build_twitch_training_dataset._MAX_VOD_NEGATIVE_ROWS_PER_POSITIVE)
        self.assertEqual(
            sum(1 for label in target_labels if label["label_class"] == "hard_negative"),
            build_twitch_training_dataset._MAX_VOD_NEGATIVE_ROWS_PER_POSITIVE,
        )
        self.assertLess(
            max(float(label["window_start"]) for label in target_labels if label["label_class"] == "hard_negative"),
            1000.0,
        )

    def test_chat_target_offsets_cover_negative_only_groups(self) -> None:
        labels = [
            {
                "label_class": "hard_negative",
                "window_start": 100.0,
                "window_end": 120.0,
            },
            {
                "label_class": "background_negative",
                "window_start": 140.0,
                "window_end": 160.0,
            },
        ]

        offsets = build_twitch_training_dataset._chat_target_offsets_for_labels(labels)
        target_labels = build_twitch_training_dataset._chat_target_labels_for_offsets(labels)

        self.assertEqual(len(target_labels), 2)
        self.assertGreater(len(offsets), 0)
        self.assertIn(105.0, offsets)
        self.assertIn(145.0, offsets)

    def test_load_or_download_chat_persists_unavailable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"

            with patch("services.chat_analyzer.download_chat", return_value=None), \
                 self.assertLogs(build_twitch_training_dataset.logger, level="INFO") as logs:
                result = build_twitch_training_dataset._load_or_download_chat(
                    "https://www.twitch.tv/videos/123",
                    job_id="job-1",
                    cache_dir=cache_dir,
                )
            metadata = json.loads((cache_dir / "chat" / "chat.metadata.json").read_text(encoding="utf-8"))

        self.assertIsNone(result)
        self.assertEqual(metadata["cache_status"], "unavailable")
        self.assertEqual(metadata["reason"], "download_returned_none_see_twitch_chat_logs")
        self.assertEqual(metadata["chat_path"], "")
        self.assertIn("training_vod_chat_download_unavailable", "\n".join(logs.output))

    def test_load_or_download_chat_skips_permanent_unavailable_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            build_twitch_training_dataset._write_chat_cache_miss_metadata(
                cache_dir,
                vod_url="https://www.twitch.tv/videos/123",
                job_id="job-old",
                cache_status="unavailable",
                reason="twitch_graphql_video_missing",
                target_offsets=[100.0, 200.0],
                missing_target_offsets=[100.0, 200.0],
            )

            with patch("services.chat_analyzer.download_chat") as download_mock:
                result = build_twitch_training_dataset._load_or_download_chat(
                    "https://www.twitch.tv/videos/123",
                    job_id="job-1",
                    cache_dir=cache_dir,
                    target_offsets=[100.0],
                )

        self.assertIsNone(result)
        download_mock.assert_not_called()

    def test_cached_chat_path_ignores_metadata_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            chat_dir = cache_dir / "chat"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.metadata.json").write_text("{}", encoding="utf-8")

            self.assertIsNone(build_twitch_training_dataset._cached_chat_path(cache_dir))

    def test_load_or_build_vod_feature_context_reuses_cached_transcript_and_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "storage"
            storage_dir.mkdir(parents=True, exist_ok=True)
            vod_path = storage_dir / "twitch_vod_123" / "video.mp4"
            vod_path.parent.mkdir(parents=True, exist_ok=True)
            vod_path.write_bytes(b"video")
            transcript = [RichSegment(start=0.0, end=1.0, text="cached")]
            extract_calls: list[Path] = []

            def _fake_extract(_source, _job, *, output_path, **_kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"audio")
                extract_calls.append(output_path)
                return output_path

            with patch.object(settings, "storage_dir", str(storage_dir)), \
                 patch.dict(sys.modules, {"services.audio_extractor": types.SimpleNamespace(extract=_fake_extract)}), \
                 patch("services.transcriber.transcribe", return_value=transcript) as transcribe_mock, \
                 patch("librosa.load", return_value=(["samples"], 16000)):
                first = build_twitch_training_dataset._load_or_build_vod_feature_context(
                    "123",
                    vod_path=vod_path,
                    job_id="job-1",
                )
                second = build_twitch_training_dataset._load_or_build_vod_feature_context(
                    "123",
                    vod_path=vod_path,
                    job_id="job-1",
                )

        self.assertEqual([segment.text for segment in first.transcript], ["cached"])
        self.assertEqual([segment.text for segment in second.transcript], ["cached"])
        self.assertEqual(len(extract_calls), 1)
        transcribe_mock.assert_called_once()

    def test_load_or_build_vod_feature_context_migrates_old_source_bound_transcript_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "storage"
            storage_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = storage_dir.parent / "vod_feature_cache" / "123"
            cache_dir.mkdir(parents=True, exist_ok=True)
            audio_path = cache_dir / "audio.wav"
            audio_path.write_bytes(b"audio")

            old_vod_path = storage_dir / "old_job" / "video.mp4"
            old_vod_path.parent.mkdir(parents=True, exist_ok=True)
            old_vod_path.write_bytes(b"video")
            old_payload = {
                "version": build_twitch_training_dataset._VOD_FEATURE_CACHE_VERSION,
                "kind": "transcript",
                "vod_id": "123",
                "source": build_twitch_training_dataset._source_video_signature(old_vod_path),
                "whisper_model": str(settings.whisper_model),
            }
            cached_transcript = [RichSegment(start=0.0, end=1.0, text="old cached")]
            old_transcript_path = build_twitch_training_dataset._transcript_cache_path(cache_dir, old_payload)
            build_twitch_training_dataset._write_transcript_cache(
                old_transcript_path,
                old_payload,
                cached_transcript,
            )

            new_vod_path = storage_dir / "new_job" / "video.mp4"
            new_vod_path.parent.mkdir(parents=True, exist_ok=True)
            new_vod_path.write_bytes(b"video")
            stable_payload = build_twitch_training_dataset._transcript_cache_payload("123")
            stable_transcript_path = build_twitch_training_dataset._transcript_cache_path(cache_dir, stable_payload)

            with patch.object(settings, "storage_dir", str(storage_dir)), \
                 patch.dict(sys.modules, {"services.audio_extractor": types.SimpleNamespace(extract=lambda *_args, **_kwargs: audio_path)}), \
                 patch("services.transcriber.transcribe") as transcribe_mock, \
                 patch.dict(sys.modules, {"librosa": types.SimpleNamespace(load=lambda *_args, **_kwargs: (["samples"], 16000))}):
                context = build_twitch_training_dataset._load_or_build_vod_feature_context(
                    "123",
                    vod_path=new_vod_path,
                    job_id="job-1",
                )

                self.assertEqual([segment.text for segment in context.transcript], ["old cached"])
                transcribe_mock.assert_not_called()
                self.assertTrue(stable_transcript_path.exists())
                self.assertFalse(old_transcript_path.exists())

    def test_load_or_build_vod_feature_context_prunes_old_transcripts_on_stable_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "storage"
            storage_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = storage_dir.parent / "vod_feature_cache" / "123"
            cache_dir.mkdir(parents=True, exist_ok=True)
            audio_path = cache_dir / "audio.wav"
            audio_path.write_bytes(b"audio")

            stable_payload = build_twitch_training_dataset._transcript_cache_payload("123")
            stable_transcript_path = build_twitch_training_dataset._transcript_cache_path(cache_dir, stable_payload)
            build_twitch_training_dataset._write_transcript_cache(
                stable_transcript_path,
                stable_payload,
                [RichSegment(start=0.0, end=1.0, text="stable")],
            )
            chunks_dir = cache_dir / "chunks"
            chunks_dir.mkdir()
            (chunks_dir / "chunk_000.wav").write_bytes(b"chunk")

            old_vod_path = storage_dir / "old_job" / "video.mp4"
            old_vod_path.parent.mkdir(parents=True, exist_ok=True)
            old_vod_path.write_bytes(b"video")
            old_payload = {
                "version": build_twitch_training_dataset._VOD_FEATURE_CACHE_VERSION,
                "kind": "transcript",
                "vod_id": "123",
                "source": build_twitch_training_dataset._source_video_signature(old_vod_path),
                "whisper_model": str(settings.whisper_model),
            }
            old_transcript_path = build_twitch_training_dataset._transcript_cache_path(cache_dir, old_payload)
            build_twitch_training_dataset._write_transcript_cache(
                old_transcript_path,
                old_payload,
                [RichSegment(start=0.0, end=1.0, text="old cached")],
            )

            vod_path = storage_dir / "new_job" / "video.mp4"
            vod_path.parent.mkdir(parents=True, exist_ok=True)
            vod_path.write_bytes(b"video")

            with patch.object(settings, "storage_dir", str(storage_dir)), \
                 patch.dict(sys.modules, {"services.audio_extractor": types.SimpleNamespace(extract=lambda *_args, **_kwargs: audio_path)}), \
                 patch("services.transcriber.transcribe") as transcribe_mock, \
                 patch.dict(sys.modules, {"librosa": types.SimpleNamespace(load=lambda *_args, **_kwargs: (["samples"], 16000))}):
                context = build_twitch_training_dataset._load_or_build_vod_feature_context(
                    "123",
                    vod_path=vod_path,
                    job_id="job-1",
                )

                self.assertEqual([segment.text for segment in context.transcript], ["stable"])
                transcribe_mock.assert_not_called()
                self.assertTrue(stable_transcript_path.exists())
                self.assertFalse(old_transcript_path.exists())
                self.assertFalse(chunks_dir.exists())

    def test_window_cache_payload_uses_stable_twitch_cache_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "storage"
            cache_dir = storage_dir / "_download_cache" / "twitch"
            cache_dir.mkdir(parents=True)
            cached_vod = cache_dir / "vod_123.mp4"
            cached_vod.write_bytes(b"same-video")
            first_path = storage_dir / "twitch_vod_123" / "video.mp4"
            second_path = storage_dir / "another_job" / "video.mp4"
            first_path.parent.mkdir(parents=True)
            second_path.parent.mkdir(parents=True)
            first_path.write_bytes(b"same-video")
            second_path.write_bytes(b"same-video")
            old_time = time.time() - 100
            first_path.touch()
            second_path.touch()

            with patch.object(settings, "storage_dir", str(storage_dir)):
                first = build_twitch_training_dataset._window_feature_cache_payload(
                    "123",
                    first_path,
                    window_start=10.0,
                    window_end=20.0,
                )
                second = build_twitch_training_dataset._window_feature_cache_payload(
                    "123",
                    second_path,
                    window_start=10.0,
                    window_end=20.0,
                )

        self.assertEqual(first["source"], second["source"])
        self.assertEqual(first["source"]["kind"], "twitch_download_cache")

    def test_load_or_build_window_modalities_uses_cache_after_first_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            cache_dir.mkdir(parents=True, exist_ok=True)
            context = build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=["samples"],
                audio_sample_rate=16000,
                cache_dir=cache_dir,
            )
            vod_path = Path(tmp) / "video.mp4"
            vod_path.write_bytes(b"video")
            audio_frames = [
                SimpleNamespace(
                    second=0,
                    pitch_hz=100.0,
                    rms_db=-10.0,
                    is_silence=False,
                    laughter_prob=0.2,
                    onset_rate=1.0,
                )
            ]
            visual_frames = [
                SimpleNamespace(
                    second=0,
                    scene_change=False,
                    motion_score=0.4,
                    face_count=1,
                    face_size_ratio=0.3,
                    face_reaction_prob=0.1,
                    visual_activity=0.5,
                )
            ]

            with patch("services.audio_features.extract_window_from_signal", return_value=audio_frames) as audio_mock, \
                 patch("services.video_features.extract", return_value=visual_frames) as video_mock:
                first = build_twitch_training_dataset._load_or_build_window_modalities(
                    "123",
                    vod_path=vod_path,
                    vod_context=context,
                    window_start=10.0,
                    window_end=20.0,
                )
                second = build_twitch_training_dataset._load_or_build_window_modalities(
                    "123",
                    vod_path=vod_path,
                    vod_context=context,
                    window_start=10.0,
                    window_end=20.0,
                )

        self.assertEqual(first[0][0].second, 0)
        self.assertEqual(first[1][0].second, 0)
        self.assertEqual(second[0][0].second, 0)
        self.assertEqual(second[1][0].second, 0)
        audio_mock.assert_called_once()
        video_mock.assert_called_once()

    def test_load_or_build_window_modalities_serializes_numpy_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            cache_dir.mkdir(parents=True, exist_ok=True)
            context = build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=["samples"],
                audio_sample_rate=16000,
                cache_dir=cache_dir,
            )
            vod_path = Path(tmp) / "video.mp4"
            vod_path.write_bytes(b"video")
            audio_frames = [
                SimpleNamespace(
                    second=np.int64(0),
                    pitch_hz=np.float32(100.0),
                    rms_db=np.float64(-10.0),
                    is_silence=np.bool_(False),
                    laughter_prob=np.float32(0.2),
                    onset_rate=np.float32(1.0),
                )
            ]
            visual_frames = [
                SimpleNamespace(
                    second=np.int64(0),
                    scene_change=np.bool_(True),
                    motion_score=np.float32(0.4),
                    face_count=np.int64(1),
                    face_size_ratio=np.float32(0.3),
                    face_reaction_prob=np.float32(0.1),
                    visual_activity=np.float32(0.5),
                )
            ]

            with patch("services.audio_features.extract_window_from_signal", return_value=audio_frames), \
                 patch("services.video_features.extract", return_value=visual_frames):
                build_twitch_training_dataset._load_or_build_window_modalities(
                    "123",
                    vod_path=vod_path,
                    vod_context=context,
                    window_start=10.0,
                    window_end=20.0,
                )

            cache_files = list((cache_dir / "windows").glob("*.json"))
            self.assertEqual(len(cache_files), 1)
            cached = json.loads(cache_files[0].read_text(encoding="utf-8"))

        self.assertIs(cached["audio_frames"][0]["is_silence"], False)
        self.assertIs(cached["visual_frames"][0]["scene_change"], True)
        self.assertEqual(cached["visual_frames"][0]["face_count"], 1)

    def test_load_or_build_window_modalities_treats_corrupt_cache_as_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vod_feature_cache" / "123"
            cache_dir.mkdir(parents=True, exist_ok=True)
            context = build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=["samples"],
                audio_sample_rate=16000,
                cache_dir=cache_dir,
            )
            vod_path = Path(tmp) / "video.mp4"
            vod_path.write_bytes(b"video")
            payload = build_twitch_training_dataset._window_feature_cache_payload(
                "123",
                vod_path,
                window_start=10.0,
                window_end=20.0,
            )
            cache_path = build_twitch_training_dataset._window_feature_cache_path(cache_dir, payload)
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text("{not-json", encoding="utf-8")
            audio_frames = [
                SimpleNamespace(
                    second=0,
                    pitch_hz=100.0,
                    rms_db=-10.0,
                    is_silence=False,
                    laughter_prob=0.2,
                    onset_rate=1.0,
                )
            ]
            visual_frames = [
                SimpleNamespace(
                    second=0,
                    scene_change=False,
                    motion_score=0.4,
                    face_count=1,
                    face_size_ratio=0.3,
                    face_reaction_prob=0.1,
                    visual_activity=0.5,
                )
            ]

            with patch("services.audio_features.extract_window_from_signal", return_value=audio_frames) as audio_mock, \
                 patch("services.video_features.extract", return_value=visual_frames) as video_mock:
                result = build_twitch_training_dataset._load_or_build_window_modalities(
                    "123",
                    vod_path=vod_path,
                    vod_context=context,
                    window_start=10.0,
                    window_end=20.0,
                )

        self.assertEqual(result[0][0].second, 0)
        audio_mock.assert_called_once()
        video_mock.assert_called_once()

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

    def test_build_dataset_skips_serial_prefetch_when_next_vod_has_checkpoint(self) -> None:
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
        checkpoint_123 = [{"vod_id": "123", "label": 1, "features": {"candidate_id": "cached-123"}}]
        checkpoint_456 = [{"vod_id": "456", "label": 1, "features": {"candidate_id": "cached-456"}}]

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            build_twitch_training_dataset._write_vod_checkpoint(work_dir, "jynxzi", "123", [labels[0]], checkpoint_123)
            build_twitch_training_dataset._write_vod_checkpoint(work_dir, "jynxzi", "456", [labels[1]], checkpoint_456)
            with patch.object(settings, "training_vod_workers", 1), \
                 patch("ml.build_twitch_training_dataset._materialize_vod") as materialize_mock:
                rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", work_dir)

        self.assertEqual(rows, checkpoint_123 + checkpoint_456)
        materialize_mock.assert_not_called()

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
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
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
            with patch("ml.build_twitch_training_dataset._raw_vod_rows_fingerprint", side_effect=["fingerprint-a", "fingerprint-b"]):
                checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", labels)

        self.assertIsNone(checkpoint)

    def test_vod_checkpoint_survives_top_viewed_metadata_on_mixed_vod(self) -> None:
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
                "view_count": 100,
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
        promoted_labels = [dict(row) for row in labels]
        promoted_labels[0].update(
            {
                "manual_gold_review": True,
                "manual_gold_review_source": "top_viewed_twitch_30d",
                "manual_gold_review_notes": "top viewed",
                "gold_candidate_status": "needs_feature_build",
                "intended_gold_grade": 3,
            }
        )
        checkpoint_rows = [{"label": 1, "label_class": "positive", "features": {"candidate_id": "cached"}}]

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            build_twitch_training_dataset._write_vod_checkpoint(work_dir, "jynxzi", "123", labels, checkpoint_rows)
            checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", promoted_labels)

        self.assertEqual(checkpoint, checkpoint_rows)

    def test_vod_checkpoint_invalidates_when_vod_becomes_top_viewed_scoped_only(self) -> None:
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
                "view_count": 100,
            },
        ]
        promoted_labels = [
            {
                **labels[0],
                "manual_gold_review": True,
                "manual_gold_review_source": "top_viewed_twitch_30d",
                "gold_candidate_status": "needs_feature_build",
                "intended_gold_grade": 3,
            }
        ]
        checkpoint_rows = [{"label": 1, "label_class": "positive", "features": {"candidate_id": "cached"}}]

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            build_twitch_training_dataset._write_vod_checkpoint(work_dir, "jynxzi", "123", labels, checkpoint_rows)
            checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", promoted_labels)

        self.assertIsNone(checkpoint)

    def test_vod_checkpoint_invalidates_when_chat_enrichment_enabled_changes(self) -> None:
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
            with patch.object(settings, "enable_chat_analysis", False):
                build_twitch_training_dataset._write_vod_checkpoint(work_dir, "jynxzi", "123", labels, checkpoint_rows)
            with patch.object(settings, "enable_chat_analysis", True):
                checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", labels)

        self.assertIsNone(checkpoint)

    def test_build_dataset_does_not_checkpoint_when_chat_enabled_without_chat_file(self) -> None:
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
             patch.object(settings, "enable_chat_analysis", True), \
             patch.object(settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", return_value=built_rows):
            work_dir = Path(tmp) / "work"
            build_twitch_training_dataset.build_dataset(labels, "jynxzi", work_dir)
            checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", labels)

        self.assertIsNone(checkpoint)

    def test_vod_checkpoint_survives_dataset_postprocessing_fingerprint_change(self) -> None:
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
            with patch("ml.build_twitch_training_dataset._dataset_build_fingerprint", return_value="postprocess-changed"):
                checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", labels)

        self.assertEqual(checkpoint, checkpoint_rows)

    def test_vod_checkpoint_accepts_legacy_dataset_fingerprint_without_raw_key(self) -> None:
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
            path = build_twitch_training_dataset._vod_checkpoint_path(work_dir, "123")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "version": build_twitch_training_dataset._VOD_CHECKPOINT_VERSION,
                        "profile": "jynxzi",
                        "vod_id": "123",
                        "dataset_build_fingerprint": next(iter(build_twitch_training_dataset._legacy_vod_checkpoint_fingerprints("jynxzi"))),
                        "label_signature": build_twitch_training_dataset._vod_checkpoint_signature(labels),
                        "rows": checkpoint_rows,
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            checkpoint = build_twitch_training_dataset._load_vod_checkpoint(work_dir, "jynxzi", "123", labels)

        self.assertEqual(checkpoint, checkpoint_rows)

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
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
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
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
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
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=_fake_build_rows_for_label):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["window_start"], 30.0)
        self.assertEqual(rows[0]["label_class"], "hard_negative")

    def test_build_dataset_processes_label_windows_directly_from_vod(self) -> None:
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
            self.assertNotIn("clip_path", _kwargs)
            return [{
                "label": 1 if label["label_class"] == "positive" else 0,
                "label_source": "twitch_clip_api",
                "features": {"candidate_id": f"cand-{int(label['window_start'])}"},
            }]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "training_label_workers", 1), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=_fake_build_rows_for_label):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["window_start"] for row in rows], [10.0, 30.0])

    def test_build_dataset_no_longer_uses_batch_cut_fallback_path(self) -> None:
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
            self.assertNotIn("clip_path", _kwargs)
            return [{
                "label": 1 if label["label_class"] == "positive" else 0,
                "label_source": "twitch_clip_api",
                "features": {"candidate_id": f"cand-{int(label['window_start'])}"},
            }]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "training_label_workers", 1), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=_fake_build_rows_for_label):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 2)

    def test_build_dataset_logs_label_phase_progress(self) -> None:
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
            {
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 50.0,
                "window_end": 60.0,
                "label_class": "positive",
                "label_score": 0.7,
                "clip_ids": ["c2"],
                "label_source": "twitch_clip_api",
                "platform_source": "twitch_clip",
            },
        ]

        def _fake_build_rows_for_label(label: dict, **_kwargs) -> list[dict]:
            self.assertNotIn("clip_path", _kwargs)
            return [{
                "label": 1 if label["label_class"] == "positive" else 0,
                "label_source": "twitch_clip_api",
                "features": {"candidate_id": f"cand-{int(label['window_start'])}"},
            }]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 1), \
             patch.object(settings, "training_label_workers", 2), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=_fake_build_rows_for_label), \
             patch("ml.build_twitch_training_dataset.logger.info") as info_mock:
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 3)
        messages = [call.args[0] for call in info_mock.call_args_list]
        self.assertIn("training_vod_label_phase_start vod_id=%s labels=%s label_workers=%s", messages)
        self.assertIn("training_vod_label_window_source vod_id=%s mode=%s labels=%s", messages)
        self.assertIn(
            "training_vod_label_progress vod_id=%s completed=%s/%s latest_window=%s rows=%s elapsed_sec=%.1f mode=parallel",
            messages,
        )
        self.assertIn("training_vod_label_phase_done vod_id=%s completed=%s rows=%s elapsed_sec=%.1f", messages)

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
             patch.object(settings, "training_vod_prefetch", True), \
             patch.object(settings, "training_vod_extended_window_scoped", False), \
             patch.object(settings, "training_vod_workers", 1), \
             patch("ml.build_twitch_training_dataset._materialize_vod", side_effect=_fake_materialize), \
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=lambda *args, **kwargs: [dict(fake_row)]):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 2)
        self.assertIn("twitch_vod_prefetch_456", materialize_calls)

    def test_build_dataset_skips_serial_prefetch_when_disabled(self) -> None:
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
        materialize_calls: list[str] = []

        def _fake_materialize(url: str, job_id: str) -> Path:
            materialize_calls.append(job_id)
            return Path("/tmp") / f"{job_id}.mp4"

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_prefetch", False), \
             patch.object(settings, "training_vod_extended_window_scoped", False), \
             patch.object(settings, "training_vod_workers", 1), \
             patch("ml.build_twitch_training_dataset._materialize_vod", side_effect=_fake_materialize), \
             patch("ml.build_twitch_training_dataset._load_or_build_vod_feature_context", return_value=self._fake_vod_feature_context(Path(tmp))), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", side_effect=lambda *args, **kwargs: [{"label": 1, "features": {"candidate_id": "x"}}]):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 2)
        self.assertEqual(materialize_calls, ["twitch_vod_123", "twitch_vod_456"])

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
             patch.object(settings, "training_vod_extended_window_scoped", False), \
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch("ml.build_twitch_training_dataset._build_rows_for_vod", side_effect=_fake_build_rows_for_vod):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual([row["vod_id"] for row in rows], ["123", "456", "789"])
        self.assertCountEqual(seen_prefetch_flags, [("123", True), ("456", True), ("789", True)])

    def test_build_dataset_parallel_workers_skip_full_vod_for_top_viewed_scoped_vods(self) -> None:
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
                "manual_gold_review_source": "top_viewed_twitch_30d",
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
                "manual_gold_review_source": "top_viewed_twitch_30d",
            },
        ]
        seen_prefetch_flags: list[tuple[str, bool]] = []

        def _fake_build_rows_for_vod(vod_id: str, *_args, **kwargs):
            vod_path_future = kwargs.get("vod_path_future")
            seen_prefetch_flags.append((vod_id, vod_path_future is not None))
            return [{"vod_id": vod_id, "label": 1, "features": {"candidate_id": vod_id}}]

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "training_vod_workers", 2), \
             patch("ml.build_twitch_training_dataset._materialize_vod") as materialize_mock, \
             patch("ml.build_twitch_training_dataset._build_rows_for_vod", side_effect=_fake_build_rows_for_vod):
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual([row["vod_id"] for row in rows], ["123", "456"])
        self.assertCountEqual(seen_prefetch_flags, [("123", False), ("456", False)])
        materialize_mock.assert_not_called()

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

    def test_balance_vod_negative_rows_caps_negative_heavy_mixed_vod(self) -> None:
        rows = [{
            "candidate_id": "pos",
            "label": 1,
            "label_class": "positive",
            "vod_id": "vod-heavy",
            "dataset_source": "twitch_vod_primary",
            "candidate_source": "reaction",
            "sample_weight": 1.5,
            "window_start": 100.0,
            "window_end": 130.0,
        }]
        rows.extend(
            {
                "candidate_id": f"primary-neg-{index}",
                "label": 0,
                "label_class": "hard_negative",
                "vod_id": "vod-heavy",
                "dataset_source": "twitch_vod_primary",
                "candidate_source": "twitch_label_proxy",
                "sample_weight": 1.8,
                "window_start": 20.0 + index * 20.0,
                "window_end": 40.0 + index * 20.0,
            }
            for index in range(12)
        )
        rows.extend(
            {
                "candidate_id": f"boundary-neg-{index}",
                "label": 0,
                "label_class": "boundary_near_miss_negative",
                "vod_id": "vod-heavy",
                "dataset_source": "twitch_vod_boundary_near_miss",
                "candidate_source": "boundary_perturbation",
                "sample_weight": 0.45,
                "window_start": 101.0 + index,
                "window_end": 129.0 + index,
            }
            for index in range(4)
        )

        balanced = build_twitch_training_dataset._balance_vod_negative_rows(
            rows,
            max_negative_rows_per_positive=8,
        )

        self.assertEqual(sum(1 for row in balanced if int(row["label"]) == 1), 1)
        self.assertEqual(sum(1 for row in balanced if int(row["label"]) == 0), 8)
        self.assertEqual(
            sum(1 for row in balanced if row["dataset_source"] == "twitch_vod_boundary_near_miss"),
            1,
        )

    def test_compute_vod_quiet_thresholds_uses_feature_percentiles(self) -> None:
        rows = [
            {
                "features": {
                    "chat_message_rate_mean": float(index),
                    "chat_message_rate_peak": float(index + 1),
                    "max_hype": float(index + 2),
                    "face_reaction_peak": float(index) / 100.0,
                    "laughter_count": float(index % 3),
                    "audio_pop_score": float(index) / 10.0,
                    "chat_laugh_score": float(index) / 20.0,
                }
            }
            for index in range(8)
        ]

        thresholds = build_twitch_training_dataset._compute_vod_quiet_thresholds(rows)

        self.assertEqual(thresholds["mode"], "relative")
        self.assertEqual(thresholds["chat_message_rate_mean"], 2.0)
        self.assertEqual(thresholds["max_hype"], 4.0)
        self.assertIn("max_hype_p95", thresholds)

    def test_background_features_are_quiet_accepts_relative_low_activity(self) -> None:
        thresholds = {
            "mode": "relative",
            "chat_message_rate_mean": 10.0,
            "chat_message_rate_peak": 20.0,
            "max_hype": 5.0,
            "face_reaction_peak": 0.3,
            "laughter_count": 2.0,
            "audio_pop_score": 0.5,
            "chat_laugh_score": 0.5,
            "max_hype_p95": 10.0,
        }
        features = SimpleNamespace(
            chat_message_rate_mean=8.0,
            chat_message_rate_peak=18.0,
            max_hype=4.5,
            face_reaction_peak=0.2,
            laughter_count=1.0,
            audio_pop_score=0.4,
            chat_laugh_score=0.7,
        )

        self.assertTrue(build_twitch_training_dataset._background_features_are_quiet(features, thresholds))

    def test_background_features_are_quiet_rejects_hard_floor_reaction(self) -> None:
        thresholds = {
            "mode": "relative",
            "chat_message_rate_mean": 100.0,
            "chat_message_rate_peak": 100.0,
            "max_hype": 100.0,
            "face_reaction_peak": 100.0,
            "laughter_count": 100.0,
            "audio_pop_score": 100.0,
            "chat_laugh_score": 100.0,
            "max_hype_p95": 100.0,
        }
        features = SimpleNamespace(
            chat_message_rate_mean=1.0,
            chat_message_rate_peak=1.0,
            max_hype=1.0,
            face_reaction_peak=0.5,
            laughter_count=0.0,
            audio_pop_score=0.0,
            chat_laugh_score=0.0,
        )

        self.assertFalse(build_twitch_training_dataset._background_features_are_quiet(features, thresholds))

    def test_background_quiet_fallback_accepts_least_noisy_relative_candidate(self) -> None:
        thresholds = {
            "mode": "relative",
            "chat_message_rate_mean": 4.0,
            "chat_message_rate_peak": 12.0,
            "max_hype": 3.0,
            "face_reaction_peak": 0.3,
            "laughter_count": 1.0,
            "audio_pop_score": 0.5,
            "chat_laugh_score": 0.4,
            "max_hype_p95": 6.0,
        }
        features = SimpleNamespace(
            chat_message_rate_mean=3.0,
            chat_message_rate_peak=10.0,
            max_hype=3.4,
            face_reaction_peak=0.5,
            laughter_count=0.0,
            audio_pop_score=0.8,
            chat_laugh_score=0.2,
        )

        assessment = build_twitch_training_dataset._background_quiet_assessment(features, thresholds)
        self.assertFalse(assessment["accepted"])
        self.assertIsNotNone(
            build_twitch_training_dataset._background_quiet_fallback_score(
                features,
                thresholds,
                assessment,
            )
        )

    def test_background_negative_builder_logs_quiet_filter_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = build_twitch_training_dataset.VodFeatureContext(
                transcript=[RichSegment(start=0.0, end=30.0, text="quiet context")],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=root,
            )
            label = {
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 100.0,
                "window_end": 130.0,
                "vod_duration_seconds": 1000.0,
                "label_class": "positive",
            }
            noisy_features = SimpleNamespace(
                candidate_id="background-cand",
                chat_message_rate_mean=99.0,
                chat_message_rate_peak=99.0,
                max_hype=99.0,
                face_reaction_peak=0.99,
                laughter_count=9.0,
                audio_pop_score=9.0,
                chat_laugh_score=9.0,
            )

            with patch.object(
                build_twitch_training_dataset.sample_background_negatives,
                "sample_background_windows",
                return_value=[build_twitch_training_dataset.sample_background_negatives.BackgroundWindow(0.0, 30.0)],
            ), \
                 patch.object(build_twitch_training_dataset, "_load_or_build_window_modalities", return_value=([], [])), \
                 patch("services.audio_features.enrich_segments", side_effect=lambda transcript, _frames: transcript), \
                 patch("services.video_features.enrich_segments", side_effect=lambda transcript, _frames: transcript), \
                 patch("services.scorer.score_segments", side_effect=lambda transcript: transcript), \
                 patch("services.feature_builder.build_features", return_value=[noisy_features]), \
                 self.assertLogs(build_twitch_training_dataset.logger, level="INFO") as logs:
                rows = build_twitch_training_dataset._build_background_negative_rows_for_vod(
                    "123",
                    [label],
                    [{"features": {"max_hype": 1.0, "face_reaction_peak": 0.1}}],
                    vod_path=root / "source.mp4",
                    vod_dir=root,
                    vod_feature_context=context,
                    job_id="job-123",
                    profile="jynxzi",
                    context_index={},
                )

        self.assertEqual(rows, [])
        log_text = "\n".join(logs.output)
        self.assertIn("training_vod_background_negative_diagnostics", log_text)
        self.assertIn("candidate_windows=1", log_text)
        self.assertIn("quiet_filter_rejected=1", log_text)

    def test_extended_window_scoped_background_rows_are_capped_per_vod(self) -> None:
        rows: list[dict] = []
        for group_index in range(3):
            context = build_twitch_training_dataset.VodFeatureContext(
                transcript=[],
                audio_signal=[],
                audio_sample_rate=16000,
                cache_dir=Path("/tmp"),
                source_start_sec=group_index * 100.0,
                source_end_sec=group_index * 100.0 + 90.0,
                source_kind=build_twitch_training_dataset._WINDOW_SCOPED_EXTENDED_KIND,
            )
            group_labels = [
                {
                    "vod_id": "vod-1",
                    "vod_url": "https://www.twitch.tv/videos/vod-1",
                    "window_start": group_index * 100.0 + 10.0,
                    "window_end": group_index * 100.0 + 20.0,
                    "vod_duration_seconds": 400.0,
                    "label_class": "weak_positive",
                }
            ]
            group_rows = [{"features": {"max_hype": 2.0}, "label": 1, "window_start": group_index * 100.0 + 10.0, "window_end": group_index * 100.0 + 20.0}]
            with patch.object(build_twitch_training_dataset, "_build_background_negative_rows_for_vod", return_value=[{"supervision_tier": "background_negative"}] * 7) as build_mock:
                remaining = max(build_twitch_training_dataset._BACKGROUND_NEGATIVE_WINDOWS_PER_VOD - len(rows), 0)
                background = build_twitch_training_dataset._build_background_negative_rows_for_vod(
                    "vod-1",
                    group_labels,
                    group_rows,
                    vod_path=Path("/tmp/source.mp4"),
                    vod_dir=Path("/tmp"),
                    vod_feature_context=context,
                    job_id=f"job-{group_index}",
                    profile="jynxzi",
                    context_index={},
                    count=remaining,
                )
            rows.extend(background[:remaining])
            self.assertLessEqual(build_mock.call_args.kwargs["count"], build_twitch_training_dataset._BACKGROUND_NEGATIVE_WINDOWS_PER_VOD)

        self.assertEqual(len(rows), build_twitch_training_dataset._BACKGROUND_NEGATIVE_WINDOWS_PER_VOD)

    def test_prepare_ranker_rows_preserves_mixed_vod_groups_when_source_capping(self) -> None:
        rows = []
        for index, source in enumerate(("quote", "reaction", "tension_build"), start=1):
            rows.append({
                "candidate_id": f"vod-{index}-pos",
                "label": 1,
                "label_class": "positive",
                "vod_id": f"vod-{index}",
                "clip_ids": [f"clip-{index}"],
                "dataset_source": "twitch_vod_primary",
                "candidate_source": source,
                "sample_weight": 1.5,
                "window_start": index * 100.0,
                "window_end": index * 100.0 + 30.0,
            })
            rows.extend(
                {
                    "candidate_id": f"vod-{index}-boundary-{neg_index}",
                    "label": 0,
                    "label_class": "boundary_near_miss_negative",
                    "vod_id": f"vod-{index}",
                    "dataset_source": "twitch_vod_boundary_near_miss",
                    "candidate_source": "boundary_perturbation",
                    "sample_weight": 0.45,
                    "window_start": index * 100.0 + neg_index,
                    "window_end": index * 100.0 + 30.0 + neg_index,
                }
                for neg_index in range(4)
            )
        rows.extend(
            {
                "candidate_id": f"extra-{index}",
                "label": 0,
                "label_class": "hard_negative",
                "vod_id": f"extra-{index}",
                "dataset_source": "twitch_vod_primary",
                "candidate_source": "twitch_label_proxy" if index % 2 else "moment",
                "sample_weight": 1.8,
                "window_start": 1000.0 + index,
                "window_end": 1030.0 + index,
            }
            for index in range(8)
        )

        prepared = build_twitch_training_dataset._prepare_ranker_rows(rows)
        mixed_vods = {
            row["vod_id"]
            for row in prepared
            if row["vod_id"].startswith("vod-")
        }

        self.assertEqual(mixed_vods, {"vod-1", "vod-2", "vod-3"})
        for vod_id in mixed_vods:
            group = [row for row in prepared if row["vod_id"] == vod_id]
            self.assertTrue(any(int(row["label"]) == 1 for row in group))
            self.assertTrue(any(int(row["label"]) == 0 for row in group))

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
