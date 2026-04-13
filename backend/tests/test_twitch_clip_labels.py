import unittest

from ml import collect_twitch_clip_labels


class TwitchClipLabelsTests(unittest.TestCase):
    def test_parse_twitch_duration(self) -> None:
        self.assertEqual(collect_twitch_clip_labels._parse_twitch_duration("3h21m15s"), 12075)

    def test_classify_clip_views(self) -> None:
        self.assertEqual(collect_twitch_clip_labels._classify_clip_views(50)[1], "weak_positive")
        self.assertEqual(collect_twitch_clip_labels._classify_clip_views(5_000)[1], "positive")
        self.assertEqual(collect_twitch_clip_labels._classify_clip_views(20_000)[1], "viral_positive")

    def test_merge_clip_regions_merges_overlaps(self) -> None:
        clips = [
            {"id": "a", "video_id": "1", "vod_offset": 100.0, "duration": 30.0, "view_count": 2000, "created_at": "2025-01-01T00:00:00Z"},
            {"id": "b", "video_id": "1", "vod_offset": 120.0, "duration": 25.0, "view_count": 15000, "created_at": "2025-01-01T00:05:00Z"},
        ]
        rows = collect_twitch_clip_labels._merge_clip_regions([
            {
                "clip_id": c["id"],
                "vod_id": c["video_id"],
                "vod_offset": c["vod_offset"],
                "duration": c["duration"],
                "view_count": c["view_count"],
                "created_at": c["created_at"],
                "url": "",
            }
            for c in clips
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_class"], "viral_positive")
        self.assertEqual(sorted(rows[0]["clip_ids"]), ["a", "b"])

    def test_sample_negative_rows_avoids_positive_windows(self) -> None:
        vod = {"id": "1", "duration_seconds": 300, "created_at": "2025-01-01T00:00:00Z"}
        positives = [{"window_start": 90.0, "window_end": 140.0}]
        rows = collect_twitch_clip_labels._sample_negative_rows("jynxzi", vod, positives)
        self.assertTrue(rows)
        self.assertTrue(all(not (float(row["window_start"]) < 160 and float(row["window_end"]) > 70) for row in rows))
        self.assertIn("hard_negative", {row["label_class"] for row in rows})
        self.assertIn("background_negative", {row["label_class"] for row in rows})

    def test_boundary_negative_mining_prefers_nearby_windows(self) -> None:
        vod = {"id": "1", "duration_seconds": 500, "created_at": "2025-01-01T00:00:00Z"}
        positives = [{"window_start": 200.0, "window_end": 240.0}]
        blocked = [(180.0, 260.0)]
        rows = collect_twitch_clip_labels._mine_boundary_negatives("jynxzi", vod, positives, blocked)
        self.assertTrue(rows)
        self.assertTrue(all(row["negative_source"] == "boundary_mined" for row in rows))

    def test_interstitial_negative_mining_uses_gaps_between_positive_clusters(self) -> None:
        vod = {"id": "1", "duration_seconds": 600, "created_at": "2025-01-01T00:00:00Z"}
        positives = [
            {"window_start": 100.0, "window_end": 140.0},
            {"window_start": 260.0, "window_end": 300.0},
        ]
        blocked = [(80.0, 160.0), (240.0, 320.0)]
        rows = collect_twitch_clip_labels._mine_interstitial_negatives("jynxzi", vod, positives, blocked)
        self.assertTrue(rows)
        self.assertTrue(all(row["negative_source"] == "between_positive_clusters" for row in rows))

    def test_negative_target_durations_follow_positive_window_lengths(self) -> None:
        positives = [
            {"window_start": 100.0, "window_end": 126.0},
            {"window_start": 220.0, "window_end": 278.0},
        ]
        durations = collect_twitch_clip_labels._negative_target_durations(positives)
        self.assertEqual(durations, [26.0, 58.0])

    def test_background_negative_sampling_is_not_fixed_to_45_seconds(self) -> None:
        vod = {"id": "1", "duration_seconds": 500, "created_at": "2025-01-01T00:00:00Z"}
        positives = [
            {"window_start": 100.0, "window_end": 126.0},
            {"window_start": 220.0, "window_end": 278.0},
        ]
        rows = collect_twitch_clip_labels._sample_negative_rows("jynxzi", vod, positives)
        background = [row for row in rows if row["negative_source"] == "background_sample"]
        self.assertTrue(background)
        self.assertGreater(len({round(float(row["window_end"]) - float(row["window_start"]), 2) for row in background}), 1)


if __name__ == "__main__":
    unittest.main()
