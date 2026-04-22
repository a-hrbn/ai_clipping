import unittest
import tempfile
from pathlib import Path

from ml import collect_twitch_clip_labels


class TwitchClipLabelsTests(unittest.TestCase):
    def test_parse_twitch_duration(self) -> None:
        self.assertEqual(collect_twitch_clip_labels._parse_twitch_duration("3h21m15s"), 12075)

    def test_classify_clip_views(self) -> None:
        self.assertEqual(collect_twitch_clip_labels._classify_clip_views(25)[1], "hard_negative")
        self.assertEqual(collect_twitch_clip_labels._classify_clip_views(100)[1], "ambiguous")
        self.assertEqual(collect_twitch_clip_labels._classify_clip_views(300)[1], "weak_positive")
        self.assertEqual(collect_twitch_clip_labels._classify_clip_views(5_000)[1], "positive")
        self.assertEqual(collect_twitch_clip_labels._classify_clip_views(20_000)[1], "viral_positive")

    def test_build_label_rows_propagates_vod_duration_seconds(self) -> None:
        vods = [{"id": "1", "duration_seconds": 3600}]
        clips = [
            {
                "id": "clip-a",
                "video_id": "1",
                "vod_offset": 120.0,
                "duration": 24.0,
                "view_count": 15_000,
                "created_at": "2025-01-01T00:00:00Z",
                "url": "https://clips.twitch.tv/A",
            }
        ]

        rows = collect_twitch_clip_labels.build_label_rows("jynxzi", vods, clips)

        self.assertTrue(rows)
        self.assertTrue(all(row["vod_duration_seconds"] == 3600 for row in rows))

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

    def test_merge_clip_regions_does_not_chain_through_padded_gap(self) -> None:
        rows = collect_twitch_clip_labels._merge_clip_regions(
            [
                {
                    "clip_id": "a",
                    "vod_id": "1",
                    "vod_offset": 100.0,
                    "duration": 30.0,
                    "view_count": 5_000,
                    "created_at": "2025-01-01T00:00:00Z",
                    "url": "",
                },
                {
                    "clip_id": "b",
                    "vod_id": "1",
                    "vod_offset": 136.0,
                    "duration": 20.0,
                    "view_count": 6_000,
                    "created_at": "2025-01-01T00:05:00Z",
                    "url": "",
                },
            ]
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["clip_ids"] for row in rows], [["a"], ["b"]])

    def test_merge_clip_regions_shrinks_large_strong_cluster_to_anchor(self) -> None:
        rows = collect_twitch_clip_labels._merge_clip_regions(
            [
                {
                    "clip_id": clip_id,
                    "vod_id": "1",
                    "vod_offset": offset,
                    "duration": 30.0,
                    "view_count": view_count,
                    "created_at": f"2025-01-01T00:0{index}:00Z",
                    "url": "",
                }
                for index, (clip_id, offset, view_count) in enumerate(
                    [
                        ("a", 100.0, 1_500),
                        ("b", 108.0, 2_000),
                        ("c", 116.0, 5_000),
                        ("d", 124.0, 1_800),
                        ("e", 132.0, 1_700),
                    ]
                )
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_class"], "positive")
        self.assertEqual(rows[0]["clip_ids"], ["c"])
        self.assertLess(float(rows[0]["window_end"]) - float(rows[0]["window_start"]), 50.0)

    def test_merge_clip_regions_keeps_large_weak_cluster_ambiguous(self) -> None:
        rows = collect_twitch_clip_labels._merge_clip_regions(
            [
                {
                    "clip_id": clip_id,
                    "vod_id": "1",
                    "vod_offset": offset,
                    "duration": 30.0,
                    "view_count": 300,
                    "created_at": f"2025-01-01T00:0{index}:00Z",
                    "url": "",
                }
                for index, (clip_id, offset) in enumerate(
                    [
                        ("a", 100.0),
                        ("b", 108.0),
                        ("c", 116.0),
                        ("d", 124.0),
                        ("e", 132.0),
                    ]
                )
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_class"], "ambiguous")

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

    def test_background_negative_sampling_targets_roughly_three_to_one_ratio(self) -> None:
        vod = {"id": "1", "duration_seconds": 800, "created_at": "2025-01-01T00:00:00Z"}
        positives = [
            {"window_start": 100.0, "window_end": 140.0, "label_class": "positive"},
            {"window_start": 260.0, "window_end": 300.0, "label_class": "positive"},
        ]

        rows = collect_twitch_clip_labels._sample_negative_rows("jynxzi", vod, positives)

        negatives = [row for row in rows if row["label_class"] in {"hard_negative", "background_negative"}]
        self.assertGreaterEqual(len(negatives), 6)

    def test_resolved_broadcasters_merges_explicit_and_profile_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "streamer_sources.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "streamer,platform,source_url,language,tags,max_clips,training_profiles,twitch_broadcaster",
                        "jynxzi,youtube,https://example.com,en,gaming,5,\"jynxzi,gaming\",jynxzi",
                        "caseoh_,twitch_pool,,en,gaming,0,\"jynxzi,gaming\",caseoh_",
                        "shroud,twitch_pool,,en,gaming,0,gaming,shroud",
                    ]
                ),
                encoding="utf-8",
            )

            broadcasters = collect_twitch_clip_labels._resolved_broadcasters(
                broadcaster="jynxzi",
                profile="jynxzi",
                manifest_path=manifest,
            )

        self.assertEqual(broadcasters, ["jynxzi", "caseoh_"])

    def test_read_source_manifest_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "streamer_sources.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "streamer,platform,language,tags,max_clips,training_profiles,twitch_broadcaster",
                        "jynxzi,youtube,en,gaming,5,jynxzi,jynxzi",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                collect_twitch_clip_labels._read_source_manifest(manifest)

    def test_read_source_manifest_rejects_non_pool_rows_without_source_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "streamer_sources.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "streamer,platform,source_url,language,tags,max_clips,training_profiles,twitch_broadcaster",
                        "jynxzi,youtube,,en,gaming,5,jynxzi,jynxzi",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing source_url"):
                collect_twitch_clip_labels._read_source_manifest(manifest)

    def test_read_source_manifest_rejects_twitch_pool_rows_without_broadcaster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "streamer_sources.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "streamer,platform,source_url,language,tags,max_clips,training_profiles,twitch_broadcaster",
                        "caseoh_,twitch_pool,,en,gaming,0,jynxzi,",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing twitch_broadcaster"):
                collect_twitch_clip_labels._read_source_manifest(manifest)

    def test_merge_label_rows_dedupes_positive_clusters_by_clip_ids(self) -> None:
        existing = [
            {
                "vod_id": "1",
                "window_start": 10.0,
                "window_end": 20.0,
                "label_class": "positive",
                "clip_ids": ["clip-a", "clip-b"],
            }
        ]
        new = [
            {
                "vod_id": "1",
                "window_start": 11.0,
                "window_end": 21.0,
                "label_class": "viral_positive",
                "clip_ids": ["clip-b", "clip-a"],
            }
        ]

        merged = collect_twitch_clip_labels._merge_label_rows(existing, new)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["label_class"], "viral_positive")
        self.assertEqual(merged[0]["window_start"], 11.0)

    def test_merge_label_rows_dedupes_negative_windows(self) -> None:
        existing = [
            {
                "vod_id": "1",
                "window_start": 30.0,
                "window_end": 60.0,
                "label_class": "hard_negative",
                "negative_source": "boundary_mined",
                "clip_ids": [],
            }
        ]
        new = [
            {
                "vod_id": "1",
                "window_start": 30.0,
                "window_end": 60.0,
                "label_class": "hard_negative",
                "negative_source": "boundary_mined",
                "clip_ids": [],
                "view_count": 0,
            }
        ]

        merged = collect_twitch_clip_labels._merge_label_rows(existing, new)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["view_count"], 0)

    def test_write_and_read_jsonl_round_trip(self) -> None:
        rows = [
            {"vod_id": "1", "window_start": 10.0, "window_end": 20.0, "label_class": "positive"},
            {"vod_id": "2", "window_start": 30.0, "window_end": 40.0, "label_class": "hard_negative"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            collect_twitch_clip_labels._write_jsonl(path, rows)
            loaded = collect_twitch_clip_labels._read_jsonl(path)

        self.assertEqual(loaded, rows)

    def test_merge_date_ranges_merges_overlapping_and_adjacent_ranges(self) -> None:
        merged = collect_twitch_clip_labels._merge_date_ranges(
            [
                (collect_twitch_clip_labels._parse_cli_date("2026-03-10"), collect_twitch_clip_labels._parse_cli_date("2026-03-15")),
                (collect_twitch_clip_labels._parse_cli_date("2026-03-01"), collect_twitch_clip_labels._parse_cli_date("2026-03-05")),
                (collect_twitch_clip_labels._parse_cli_date("2026-03-06"), collect_twitch_clip_labels._parse_cli_date("2026-03-09")),
            ]
        )

        self.assertEqual(
            merged,
            [
                (
                    collect_twitch_clip_labels._parse_cli_date("2026-03-01"),
                    collect_twitch_clip_labels._parse_cli_date("2026-03-15"),
                )
            ],
        )

    def test_missing_date_ranges_skips_covered_middle_segment(self) -> None:
        missing = collect_twitch_clip_labels._missing_date_ranges(
            collect_twitch_clip_labels._parse_cli_date("2026-03-01"),
            collect_twitch_clip_labels._parse_cli_date("2026-03-15"),
            [
                (
                    collect_twitch_clip_labels._parse_cli_date("2026-03-05"),
                    collect_twitch_clip_labels._parse_cli_date("2026-03-10"),
                )
            ],
        )

        self.assertEqual(
            missing,
            [
                (
                    collect_twitch_clip_labels._parse_cli_date("2026-03-01"),
                    collect_twitch_clip_labels._parse_cli_date("2026-03-04"),
                ),
                (
                    collect_twitch_clip_labels._parse_cli_date("2026-03-11"),
                    collect_twitch_clip_labels._parse_cli_date("2026-03-15"),
                ),
            ],
        )

    def test_record_collected_range_merges_ranges_per_broadcaster(self) -> None:
        payload = {"version": 1, "broadcasters": {"jynxzi": [{"date_from": "2026-03-01", "date_to": "2026-03-15"}]}}

        updated = collect_twitch_clip_labels._record_collected_range(
            payload,
            "jynxzi",
            collect_twitch_clip_labels._parse_cli_date("2026-03-16"),
            collect_twitch_clip_labels._parse_cli_date("2026-03-20"),
        )

        self.assertEqual(
            updated["broadcasters"]["jynxzi"],
            [{"date_from": "2026-03-01", "date_to": "2026-03-20"}],
        )

    def test_collection_chunks_split_requested_window_into_days(self) -> None:
        chunks = collect_twitch_clip_labels._collection_chunks(
            [
                (
                    collect_twitch_clip_labels._parse_cli_date("2026-04-01"),
                    collect_twitch_clip_labels._parse_cli_date("2026-04-03"),
                )
            ]
        )

        self.assertEqual(
            chunks,
            [
                (
                    collect_twitch_clip_labels._parse_cli_date("2026-04-01"),
                    collect_twitch_clip_labels._parse_cli_date("2026-04-01"),
                ),
                (
                    collect_twitch_clip_labels._parse_cli_date("2026-04-02"),
                    collect_twitch_clip_labels._parse_cli_date("2026-04-02"),
                ),
                (
                    collect_twitch_clip_labels._parse_cli_date("2026-04-03"),
                    collect_twitch_clip_labels._parse_cli_date("2026-04-03"),
                ),
            ],
        )

    def test_progress_file_round_trip_preserves_ranges(self) -> None:
        payload = {
            "version": 1,
            "broadcasters": {
                "jynxzi": [{"date_from": "2026-03-01", "date_to": "2026-03-15"}]
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            collect_twitch_clip_labels._write_progress_file(path, payload)
            loaded = collect_twitch_clip_labels._read_progress_file(path)

        self.assertEqual(loaded, payload)


if __name__ == "__main__":
    unittest.main()
