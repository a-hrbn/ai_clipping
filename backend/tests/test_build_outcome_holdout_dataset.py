import unittest

from ml import build_outcome_holdout_dataset


class BuildOutcomeHoldoutDatasetTests(unittest.TestCase):
    def test_builds_positive_cross_platform_and_negative_auto_outcome_holdout(self) -> None:
        payload = build_outcome_holdout_dataset.build_outcome_holdout_dataset(
            positive_rows=[
                _cross_platform_row("vod-1", "candidate-a", "clip-a", "holdout"),
                _cross_platform_row("vod-2", "candidate-b", "clip-b", "train"),
                {**_cross_platform_row("vod-3", "candidate-c", "clip-c", "holdout"), "features": {}},
            ],
            auto_negative_rows=[
                _auto_negative_row("vod-4", "candidate-d", "holdout"),
                _auto_negative_row("vod-5", "candidate-e", "train"),
            ],
            target_positive_rows=3,
        )

        rows = payload["rows"]
        summary = payload["summary"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["positive_rows"], 1)
        self.assertEqual(summary["negative_rows"], 1)
        self.assertEqual(summary["dataset_roles"], {"outcome_holdout": 2})
        self.assertEqual(summary["by_supervision_tier"], {"outcome_holdout": 2})
        self.assertEqual(summary["cross_platform_positive_rows"], 1)
        self.assertEqual(summary["top_viewed_positive_rows"], 0)
        self.assertEqual(summary["auto_negative_rows"], 1)
        self.assertEqual(summary["target_positive_rows"], 3)
        self.assertFalse(summary["target_positive_rows_met"])
        self.assertEqual(summary["target_positive_rows_needed"], 2)
        self.assertEqual(rows[0]["dataset_source"], "cross_platform_outcome_holdout")
        self.assertEqual(rows[0]["label_class"], "cross_platform_outcome_positive")
        self.assertEqual(rows[1]["dataset_source"], "outcome_auto_negative_holdout")
        self.assertEqual(rows[1]["label_class"], "outcome_auto_negative")

    def test_skips_duplicate_identities_across_positive_and_negative_sources(self) -> None:
        payload = build_outcome_holdout_dataset.build_outcome_holdout_dataset(
            positive_rows=[
                _cross_platform_row("vod-1", "candidate-a", "clip-a", "holdout"),
            ],
            auto_negative_rows=[
                {
                    **_auto_negative_row("vod-1", "candidate-a", "holdout"),
                    "clip_ids": ["clip-a"],
                },
            ],
        )

        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["summary"]["skipped"]["outcome_auto_negative_holdout_duplicate_identity"], 1)

    def test_builds_outcome_train_split(self) -> None:
        payload = build_outcome_holdout_dataset.build_outcome_holdout_dataset(
            positive_rows=[
                _cross_platform_row("vod-1", "candidate-a", "clip-a", "train"),
                _cross_platform_row("vod-2", "candidate-b", "clip-b", "holdout"),
            ],
            auto_negative_rows=[
                _auto_negative_row("vod-3", "candidate-c", "train"),
                _auto_negative_row("vod-4", "candidate-d", "holdout"),
            ],
            split="train",
        )

        rows = payload["rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(payload["summary"]["dataset_roles"], {"outcome_train": 2})
        self.assertEqual(payload["summary"]["outcome_split"], "train")
        self.assertEqual({row["gold_holdout_split"] for row in rows}, {"train"})
        self.assertEqual({row["supervision_tier"] for row in rows}, {"outcome_train"})
        self.assertEqual(
            {row["dataset_source"] for row in rows},
            {"cross_platform_outcome_train", "outcome_auto_negative_train"},
        )

    def test_replays_features_for_cross_platform_rows_before_filtering(self) -> None:
        raw_positive = {
            **_cross_platform_row("vod-1", "candidate-a", "clip-a", "train"),
            "features": {},
        }
        feature_source = {
            **_cross_platform_row("vod-1", "candidate-a", "clip-a", "train"),
            "features": {"source": "reaction", "hook_strength": 0.91},
        }

        payload = build_outcome_holdout_dataset.build_outcome_holdout_dataset(
            positive_rows=[raw_positive],
            auto_negative_rows=[],
            feature_source_rows=[feature_source],
            split="train",
            target_positive_rows=1,
        )

        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["features"]["hook_strength"], 0.91)
        self.assertEqual(payload["rows"][0]["candidate_source"], "reaction")
        self.assertEqual(payload["summary"]["positive_rows"], 1)
        self.assertEqual(payload["summary"]["feature_replay"]["applied_rows"], 1)
        self.assertEqual(payload["summary"]["target_positive_rows_met"], True)
        self.assertNotIn("cross_platform_outcome_train_missing_features", payload["summary"]["skipped"])

    def test_auto_split_keeps_same_vod_in_one_split(self) -> None:
        rows = [
            {**_cross_platform_row("vod-1", "candidate-a", "clip-a", ""), "features": {"source": "quote"}},
            {
                **_cross_platform_row("vod-1", "candidate-b", "clip-b", ""),
                "features": {"source": "reaction"},
                "window_start": 120.0,
                "window_end": 140.0,
            },
        ]

        train = build_outcome_holdout_dataset.build_outcome_holdout_dataset(
            positive_rows=rows,
            auto_negative_rows=[],
            split="train",
            cross_platform_holdout_ratio=0.5,
        )
        holdout = build_outcome_holdout_dataset.build_outcome_holdout_dataset(
            positive_rows=rows,
            auto_negative_rows=[],
            split="holdout",
            cross_platform_holdout_ratio=0.5,
        )

        self.assertIn((len(train["rows"]), len(holdout["rows"])), {(2, 0), (0, 2)})

    def test_ignores_top_viewed_rows(self) -> None:
        payload = build_outcome_holdout_dataset.build_outcome_holdout_dataset(
            top_viewed_rows=[_top_viewed_row("vod-1", "candidate-a", "clip-a", "holdout")],
            auto_negative_rows=[],
        )

        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["summary"]["skipped"]["top_viewed_rows_ignored"], 1)


def _cross_platform_row(vod_id: str, candidate_id: str, clip_id: str, split: str) -> dict:
    return {
        "queue_id": f"queue-{candidate_id}",
        "job_id": f"twitch_vod_{vod_id}",
        "candidate_id": candidate_id,
        "vod_id": vod_id,
        "vod_url": f"https://www.twitch.tv/videos/{vod_id}",
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "window_start": 10.0,
        "window_end": 30.0,
        "label": 1,
        "label_class": "weak_positive",
        "dataset_source": "cross_platform_audio_fingerprint",
        "supervision_tier": "cross_platform_outcome",
        "supervision_reason": "audio_fingerprint_matched_shortform_outcome",
        "gold_holdout_split": split,
        "features": {"source": "hype_peak", "duration": 20.0},
        "clip_ids": [clip_id],
    }


def _top_viewed_row(vod_id: str, candidate_id: str, clip_id: str, split: str) -> dict:
    row = _cross_platform_row(vod_id, candidate_id, clip_id, split)
    row.update(
        {
            "dataset_source": "gold_eval_top_viewed_twitch",
            "supervision_tier": "gold_eval",
            "supervision_reason": "manual_top_viewed_twitch_gold_positive",
            "gold_holdout_policy": "top_viewed_auto",
        }
    )
    return row


def _auto_negative_row(vod_id: str, candidate_id: str, split: str) -> dict:
    return {
        "job_id": f"twitch_vod_{vod_id}",
        "candidate_id": candidate_id,
        "vod_id": vod_id,
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "window_start": 60.0,
        "window_end": 80.0,
        "label": 0,
        "label_class": "auto_outcome_background_negative",
        "dataset_source": f"auto_outcome_negative_{split}",
        "supervision_tier": "auto_outcome_negative",
        "gold_holdout_policy": "auto_outcome_negative",
        "gold_holdout_split": split,
        "features": {"source": "dense_window", "duration": 20.0},
    }


if __name__ == "__main__":
    unittest.main()
