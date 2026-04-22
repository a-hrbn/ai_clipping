import tempfile
import unittest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ml import auto_train_short_form


class AutoTrainShortFormTests(unittest.TestCase):
    def test_read_source_manifest_parses_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.csv"
            path.write_text(
                "\n".join([
                    "streamer,platform,source_url,language,tags,max_clips",
                    'shroud,youtube,https://www.youtube.com/@shroud/shorts,en,"fps,clutch",12',
                    'tarik,tiktok,https://www.tiktok.com/@tarik,en,"reaction,chatting",8',
                ]),
                encoding="utf-8",
            )

            rows = auto_train_short_form._read_source_manifest(path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["streamer"], "shroud")
            self.assertEqual(rows[1]["platform"], "tiktok")
            self.assertEqual(rows[1]["max_clips"], "8")

    def test_filter_seeds_by_streamers_keeps_requested_streamer(self) -> None:
        seeds = [
            {"streamer": "jynxzi", "platform": "youtube", "source_url": "a"},
            {"streamer": "tarik", "platform": "tiktok", "source_url": "b"},
        ]

        filtered = auto_train_short_form._filter_seeds_by_streamers(seeds, {"jynxzi"})

        self.assertEqual(filtered, [{"streamer": "jynxzi", "platform": "youtube", "source_url": "a"}])

    def test_label_candidates_uses_top_and_bottom_buckets(self) -> None:
        candidates = [
            {"url": f"https://example.com/{i}", "view_count": views, "like_count": 0, "comment_count": 0, "duration": 20, "title": str(i)}
            for i, views in enumerate([1000, 900, 800, 200, 100, 50], start=1)
        ]

        labeled = auto_train_short_form._label_candidates(
            candidates,
            top_fraction=0.34,
            bottom_fraction=0.34,
        )

        by_url = {row["url"]: row["label"] for row in labeled}
        self.assertEqual(by_url["https://example.com/1"], 1)
        self.assertEqual(by_url["https://example.com/2"], 1)
        self.assertEqual(by_url["https://example.com/5"], 0)
        self.assertEqual(by_url["https://example.com/6"], 0)
        self.assertNotIn("https://example.com/3", by_url)

    def test_label_candidates_uses_engagement_not_only_views(self) -> None:
        candidates = [
            {"url": "https://example.com/a", "view_count": 1000, "like_count": 5, "comment_count": 0, "duration": 20, "title": "a"},
            {"url": "https://example.com/b", "view_count": 900, "like_count": 300, "comment_count": 40, "duration": 20, "title": "b"},
            {"url": "https://example.com/c", "view_count": 100, "like_count": 1, "comment_count": 0, "duration": 20, "title": "c"},
        ]

        labeled = auto_train_short_form._label_candidates(candidates, top_fraction=0.34, bottom_fraction=0.34)
        by_url = {row["url"]: row["label"] for row in labeled}

        self.assertEqual(by_url["https://example.com/b"], 1)
        self.assertEqual(by_url["https://example.com/c"], 0)

    def test_label_candidates_filters_overlong_clips_from_positive_bucket(self) -> None:
        now = datetime.now(timezone.utc)
        candidates = [
            {
                "url": "https://example.com/long",
                "view_count": 5000,
                "like_count": 900,
                "comment_count": 120,
                "duration": 140,
                "title": "long",
                "publish_timestamp": int((now - timedelta(days=2)).timestamp()),
            },
            {
                "url": "https://example.com/short",
                "view_count": 3200,
                "like_count": 700,
                "comment_count": 80,
                "duration": 24,
                "title": "short",
                "publish_timestamp": int((now - timedelta(days=2)).timestamp()),
            },
            {
                "url": "https://example.com/weak",
                "view_count": 200,
                "like_count": 10,
                "comment_count": 1,
                "duration": 20,
                "title": "weak",
                "publish_timestamp": int((now - timedelta(days=30)).timestamp()),
            },
        ]

        labeled = auto_train_short_form._label_candidates(candidates, top_fraction=0.34, bottom_fraction=0.34)
        by_url = {row["url"]: row["label"] for row in labeled}

        self.assertEqual(by_url["https://example.com/short"], 1)
        self.assertNotIn("https://example.com/long", {url for url, label in by_url.items() if label == 1})

    def test_viral_score_uses_log1p_duration_penalty(self) -> None:
        base = {
            "view_count": 1_000,
            "like_count": 150,
            "comment_count": 30,
            "duration": 1,
        }
        longer = dict(base, duration=2)
        baselines = {"median_views": 100.0, "median_likes": 10.0, "median_comments": 5.0}

        score_delta = auto_train_short_form._viral_score(base, baselines) - auto_train_short_form._viral_score(longer, baselines)

        self.assertAlmostEqual(
            score_delta,
            0.10 * (auto_train_short_form.math.log1p(2) - auto_train_short_form.math.log1p(1)),
            places=6,
        )

    def test_short_form_filter_requires_supported_short_content(self) -> None:
        youtube_short = {"url": "https://www.youtube.com/shorts/abc", "duration": 30}
        youtube_long = {"url": "https://www.youtube.com/watch?v=abc", "duration": 400}
        tiktok_clip = {"url": "https://www.tiktok.com/@name/video/123", "duration": 45}

        self.assertTrue(auto_train_short_form._is_short_form_candidate(youtube_short, "youtube", 8, 180))
        self.assertFalse(auto_train_short_form._is_short_form_candidate(youtube_long, "youtube", 8, 180))
        self.assertTrue(auto_train_short_form._is_short_form_candidate(tiktok_clip, "tiktok", 8, 180))

    def test_extract_tiktok_username_handles_profile_urls(self) -> None:
        self.assertEqual(
            auto_train_short_form._extract_tiktok_username("https://www.tiktok.com/@jynxziontwitch"),
            "jynxziontwitch",
        )

    def test_log_manifest_preview_summarizes_rows(self) -> None:
        rows = [
            {"streamer": "shroud", "platform": "youtube", "label": "1", "url": "https://example.com/1"},
            {"streamer": "tarik", "platform": "tiktok", "label": "0", "url": "https://example.com/2"},
        ]
        with patch.object(auto_train_short_form.logger, "info") as info_mock:
            auto_train_short_form._log_manifest_preview(rows, limit=2)

        self.assertGreaterEqual(info_mock.call_count, 3)

    def test_main_continues_when_one_seed_discovery_fails(self) -> None:
        seeds = [
            {"streamer": "jynxzi", "platform": "youtube", "source_url": "y", "language": "en", "tags": "x", "max_clips": "4"},
            {"streamer": "jynxzi", "platform": "tiktok", "source_url": "t", "language": "en", "tags": "x", "max_clips": "4"},
        ]
        discovered_row = {
            "url": "https://www.youtube.com/watch?v=abc",
            "streamer": "jynxzi",
            "platform": "youtube",
            "language": "en",
            "tags": "x",
            "label": "1",
            "notes": "ok",
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.auto_train_short_form._read_source_manifest", return_value=seeds), \
             patch("ml.auto_train_short_form._discover_seed", side_effect=[[discovered_row], RuntimeError("boom")]), \
             patch("ml.auto_train_short_form._log_manifest_preview"), \
             patch("sys.argv", ["auto_train_short_form.py", "--dry-run", "--manifest-out", f"{tmp}/manifest.csv"]):
            rc = auto_train_short_form.main()

        self.assertEqual(rc, 0)

    def test_main_skips_requested_platforms(self) -> None:
        seeds = [
            {"streamer": "jynxzi", "platform": "youtube", "source_url": "y", "language": "en", "tags": "x", "max_clips": "4"},
            {"streamer": "jynxzi", "platform": "tiktok", "source_url": "t", "language": "en", "tags": "x", "max_clips": "4"},
        ]
        discovered_row = {
            "url": "https://www.youtube.com/watch?v=abc",
            "streamer": "jynxzi",
            "platform": "youtube",
            "language": "en",
            "tags": "x",
            "label": "1",
            "notes": "ok",
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.auto_train_short_form._read_source_manifest", return_value=seeds), \
             patch("ml.auto_train_short_form._discover_seed", return_value=[discovered_row]) as discover_mock, \
             patch("ml.auto_train_short_form._log_manifest_preview"), \
             patch("sys.argv", ["auto_train_short_form.py", "--dry-run", "--skip-platforms", "tiktok", "--manifest-out", f"{tmp}/manifest.csv"]):
            rc = auto_train_short_form.main()

        self.assertEqual(rc, 0)
        self.assertEqual(discover_mock.call_count, 1)

    def test_discover_seed_threads_proxy_to_extractor(self) -> None:
        seed = {
            "streamer": "jynxzi",
            "platform": "youtube",
            "source_url": "https://www.youtube.com/@Jynxzi/shorts",
            "language": "en",
            "tags": "x",
            "max_clips": "4",
        }
        raw_entry = {
            "id": "abc",
            "duration": 30,
            "view_count": 1000,
            "like_count": 50,
            "comment_count": 10,
            "title": "clip",
        }
        with patch("ml.auto_train_short_form._extract_source_entries", return_value=[raw_entry]) as extract_mock:
            rows = auto_train_short_form._discover_seed(
                seed,
                min_duration=8,
                max_duration=180,
                top_fraction=0.5,
                bottom_fraction=0.5,
                proxy="http://proxy.local:8080",
            )

        self.assertTrue(rows)
        self.assertEqual(extract_mock.call_args.kwargs["proxy"], "http://proxy.local:8080")

    def test_write_manifest_includes_rich_metadata_columns(self) -> None:
        row = {
            "url": "https://example.com/1",
            "streamer": "jynxzi",
            "platform": "youtube",
            "language": "en",
            "tags": "gaming",
            "label": "1",
            "quality_bucket": "positive",
            "label_reason": "top_ranked_by_viral_score",
            "viral_score": "14.2",
            "view_count": "1000",
            "like_count": "80",
            "comment_count": "12",
            "duration": "22",
            "publish_timestamp": "1710000000",
            "upload_date": "20240309",
            "notes": "ok",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            auto_train_short_form._write_manifest(path, [row])
            text = path.read_text(encoding="utf-8")

        self.assertIn("quality_bucket", text)
        self.assertIn("publish_timestamp", text)

    def test_train_preferred_path_prefers_vod_when_labels_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {
                "profile": "jynxzi",
                "artifact_dir": tmp,
                "twitch_labels": str(Path(tmp) / "twitch_labels.jsonl"),
                "training_mode": "auto",
                "work_dir": tmp,
                "short_form_weight": 0.35,
            })()
            short_form_dataset = Path(tmp) / "short_form.jsonl"
            short_form_dataset.write_text(json.dumps({"candidate_id": "sf-1", "label": 1, "features": {}}) + "\n", encoding="utf-8")
            Path(args.twitch_labels).write_text(json.dumps({"broadcaster": "jynxzi"}) + "\n", encoding="utf-8")

            with patch("ml.auto_train_short_form.train_vod_ranker.build_composite_dataset", return_value=[{"candidate_id": "vod-1", "label": 1, "features": {}}]), \
                 patch("ml.auto_train_short_form.train_vod_ranker._read_jsonl", side_effect=[[{"broadcaster": "jynxzi"}], [{"candidate_id": "sf-1", "label": 1, "features": {}}]]), \
                 patch("ml.auto_train_short_form.train_vod_ranker._write_jsonl") as write_mock, \
                 patch("ml.auto_train_short_form.train_vod_ranker._train_models_for_dataset", return_value=0) as train_mock:
                rc = auto_train_short_form._train_preferred_path(args, short_form_dataset)

        self.assertEqual(rc, 0)
        write_mock.assert_called_once()
        train_mock.assert_called_once_with(Path(tmp) / "jynxzi_vod_training_rows.jsonl", "jynxzi", Path(tmp))

    def test_train_preferred_path_can_force_short_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {
                "profile": "jynxzi",
                "artifact_dir": tmp,
                "twitch_labels": str(Path(tmp) / "missing.jsonl"),
                "training_mode": "short-form",
                "work_dir": tmp,
                "short_form_weight": 0.35,
            })()
            short_form_dataset = Path(tmp) / "short_form.jsonl"
            short_form_dataset.write_text(json.dumps({"candidate_id": "sf-1", "label": 1, "features": {}}) + "\n", encoding="utf-8")

            with patch("ml.auto_train_short_form.train_ranker.train_from_dataset", return_value=0) as train_mock:
                rc = auto_train_short_form._train_preferred_path(args, short_form_dataset)

        self.assertEqual(rc, 0)
        train_mock.assert_called_once_with(short_form_dataset, "jynxzi", Path(tmp))

    def test_train_preferred_path_filters_vod_labels_for_streamer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {
                "profile": "gaming",
                "artifact_dir": tmp,
                "twitch_labels": str(Path(tmp) / "twitch_labels.jsonl"),
                "training_mode": "vod",
                "work_dir": tmp,
                "short_form_weight": 0.35,
            })()
            short_form_dataset = Path(tmp) / "short_form.jsonl"
            short_form_dataset.write_text(json.dumps({"candidate_id": "sf-1", "label": 1, "features": {}}) + "\n", encoding="utf-8")
            Path(args.twitch_labels).write_text(
                "\n".join([
                    json.dumps({"broadcaster": "jynxzi", "vod_id": "vod-a"}),
                    json.dumps({"broadcaster": "tarik", "vod_id": "vod-b"}),
                ]),
                encoding="utf-8",
            )

            with patch("ml.auto_train_short_form.train_vod_ranker.build_composite_dataset", return_value=[{"candidate_id": "vod-1", "label": 1, "features": {}}]) as build_mock, \
                 patch("ml.auto_train_short_form.train_vod_ranker._read_jsonl", side_effect=[
                     [
                         {"broadcaster": "jynxzi", "vod_id": "vod-a"},
                         {"broadcaster": "tarik", "vod_id": "vod-b"},
                     ],
                     [{"candidate_id": "sf-1", "label": 1, "features": {}}],
                 ]), \
                 patch("ml.auto_train_short_form.train_vod_ranker._write_jsonl"), \
                 patch("ml.auto_train_short_form.train_vod_ranker._train_models_for_dataset", return_value=0) as train_mock:
                rc = auto_train_short_form._train_preferred_path(
                    args,
                    short_form_dataset,
                    profile="tarik",
                    broadcaster_filter="tarik",
                )

        self.assertEqual(rc, 0)
        filtered_rows = build_mock.call_args.args[0]
        self.assertEqual(filtered_rows, [{"broadcaster": "tarik", "vod_id": "vod-b"}])
        train_mock.assert_called_once_with(Path(tmp) / "tarik_vod_training_rows.jsonl", "tarik", Path(tmp))

    def test_train_preferred_path_falls_back_when_streamer_has_no_vod_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {
                "profile": "gaming",
                "artifact_dir": tmp,
                "twitch_labels": str(Path(tmp) / "twitch_labels.jsonl"),
                "training_mode": "vod",
                "work_dir": tmp,
                "short_form_weight": 0.35,
            })()
            short_form_dataset = Path(tmp) / "short_form.jsonl"
            short_form_dataset.write_text(json.dumps({"candidate_id": "sf-1", "label": 1, "features": {}}) + "\n", encoding="utf-8")
            Path(args.twitch_labels).write_text(json.dumps({"broadcaster": "jynxzi", "vod_id": "vod-a"}) + "\n", encoding="utf-8")

            with patch("ml.auto_train_short_form.train_vod_ranker.build_composite_dataset") as build_mock, \
                 patch("ml.auto_train_short_form.train_vod_ranker._read_jsonl", return_value=[{"broadcaster": "jynxzi", "vod_id": "vod-a"}]), \
                 patch("ml.auto_train_short_form.train_ranker.train_from_dataset", return_value=0) as train_mock:
                rc = auto_train_short_form._train_preferred_path(
                    args,
                    short_form_dataset,
                    profile="tarik",
                    broadcaster_filter="tarik",
                )

        self.assertEqual(rc, 0)
        build_mock.assert_not_called()
        train_mock.assert_called_once_with(short_form_dataset, "tarik", Path(tmp))


if __name__ == "__main__":
    unittest.main()
