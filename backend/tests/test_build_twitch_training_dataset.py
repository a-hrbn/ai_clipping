import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import settings
from ml import build_twitch_training_dataset


class BuildTwitchTrainingDatasetTests(unittest.TestCase):
    def test_dataset_summary_counts_classes(self) -> None:
        summary = build_twitch_training_dataset.dataset_summary([
            {"vod_id": "1", "label_class": "positive"},
            {"vod_id": "1", "label_class": "hard_negative"},
            {"vod_id": "2", "label_class": "viral_positive"},
        ])
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["vod_count"], 2)
        self.assertEqual(summary["label_classes"]["positive"], 1)

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

    def test_build_dataset_uses_training_cache_when_present(self) -> None:
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
        fake_row = {
            "label": 1,
            "label_source": "twitch_clip_api",
            "features": {"candidate_id": "x"},
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "storage_dir", tmp), \
             patch.object(settings, "training_vod_cache_ttl_sec", 3600), \
             patch("ml.build_twitch_training_dataset._build_rows_for_label", return_value=[fake_row]) as build_rows_mock, \
             patch("ml.build_twitch_training_dataset.downloader.download") as download_mock:
            cache_path = build_twitch_training_dataset._training_cache_path(labels[0]["vod_url"])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"cached")
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp) / "work")

        self.assertEqual(len(rows), 1)
        download_mock.assert_not_called()
        self.assertEqual(build_rows_mock.call_args.kwargs["vod_path"], cache_path)

    def test_materialize_vod_writes_training_cache_after_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "storage_dir", tmp), \
             patch.object(settings, "training_vod_cache_ttl_sec", 3600):
            downloaded = Path(tmp) / "job" / "video.mp4"
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            downloaded.write_bytes(b"video")
            with patch("ml.build_twitch_training_dataset.downloader.download", return_value=downloaded):
                path = build_twitch_training_dataset._materialize_vod("https://www.twitch.tv/videos/456", "job-1")

            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), b"video")
            self.assertEqual(path, build_twitch_training_dataset._training_cache_path("https://www.twitch.tv/videos/456"))

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
             patch("ml.build_twitch_training_dataset._materialize_vod", return_value=Path(tmp) / "vod.mp4"), \
             patch(
                 "ml.build_twitch_training_dataset._build_rows_for_label",
                 side_effect=[RuntimeError("ffmpeg failed"), [fake_row]],
             ) as build_rows_mock:
            rows = build_twitch_training_dataset.build_dataset(labels, "jynxzi", Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_class"], "hard_negative")
        self.assertEqual(build_rows_mock.call_count, 2)

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


if __name__ == "__main__":
    unittest.main()
