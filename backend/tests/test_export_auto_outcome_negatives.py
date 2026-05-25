from __future__ import annotations

import unittest
from datetime import date

from ml import export_auto_outcome_negatives


class ExportAutoOutcomeNegativesTest(unittest.TestCase):
    def test_exports_train_and_holdout_negatives_with_positive_proximity_and_dedupe(self) -> None:
        candidates = [
            _candidate("cand-hard", "vod-1", 500.0, 530.0, score=9.2, created_at="2026-03-01T10:00:00Z"),
            _candidate("cand-background", "vod-2", 100.0, 125.0, score=2.0, created_at="2026-03-01T10:00:00Z"),
            _candidate("cand-nonpositive-input-match", "vod-6", 40.0, 55.0, score=8.3, created_at="2026-03-01T10:00:00Z"),
            _candidate("cand-overlap", "vod-3", 190.0, 210.0, score=9.9, created_at="2026-03-01T10:00:00Z"),
            _candidate("cand-duplicate", "vod-4", 10.0, 25.0, score=8.0, created_at="2026-03-01T10:00:00Z"),
            _candidate("cand-recent", "vod-5", 10.0, 25.0, score=8.0, created_at="2026-04-25T10:00:00Z"),
        ]
        positives = [
            {
                "candidate_id": "pos-overlap",
                "vod_id": "vod-3",
                "label": 1,
                "label_class": "viral_positive",
                "window_start": 100.0,
                "window_end": 130.0,
                "features": {"hook_strength": 1.0},
            },
            {
                **_candidate(
                    "cand-nonpositive-input-match",
                    "vod-6",
                    40.0,
                    55.0,
                    score=0.1,
                    created_at="2026-03-01T10:00:00Z",
                ),
                "label_class": "hard_negative",
            },
        ]
        existing = [
            {
                "candidate_id": "cand-duplicate",
                "vod_id": "vod-4",
                "window_start": 10.0,
                "window_end": 25.0,
                "features": {"hook_strength": 0.1},
            }
        ]

        payload = export_auto_outcome_negatives.build_auto_outcome_negative_rows(
            candidate_rows=candidates,
            positive_rows=positives,
            existing_rows=existing,
            as_of_date=date(2026, 4, 29),
            min_age_days=14,
            holdout_ratio=0.5,
            positive_exclusion_sec=90.0,
            hard_score_threshold=7.5,
            max_rows=20,
        )

        rows = payload["rows"]
        summary = payload["summary"]
        self.assertEqual(
            {row["candidate_id"] for row in rows},
            {"cand-hard", "cand-background", "cand-nonpositive-input-match"},
        )
        self.assertEqual(summary["exported_rows"], 3)
        self.assertEqual(summary["skipped"]["positive_proximity"], 1)
        self.assertEqual(summary["skipped"]["existing_or_positive_identity"], 1)
        self.assertEqual(summary["skipped"]["candidate_too_recent"], 1)
        rows_by_id = {row["candidate_id"]: row for row in rows}
        self.assertEqual(rows_by_id["cand-hard"]["label_class"], "auto_outcome_hard_negative")
        self.assertEqual(rows_by_id["cand-background"]["label_class"], "auto_outcome_background_negative")
        self.assertEqual(rows_by_id["cand-hard"]["label_source"], "auto_outcome_negative")
        self.assertEqual(rows_by_id["cand-hard"]["sample_weight"], 1.0)
        self.assertEqual(rows_by_id["cand-background"]["auto_outcome_negative_recommended_sample_weight"], 0.5)
        self.assertIn(rows_by_id["cand-hard"]["gold_holdout_split"], {"train", "holdout"})
        self.assertEqual(
            export_auto_outcome_negatives.build_auto_outcome_negative_rows(
                candidate_rows=candidates,
                positive_rows=positives,
                existing_rows=existing,
                as_of_date=date(2026, 4, 29),
                min_age_days=14,
                holdout_ratio=0.5,
                positive_exclusion_sec=90.0,
                hard_score_threshold=7.5,
                max_rows=20,
            )["rows"],
            rows,
        )

    def test_skip_undated_can_enforce_age_requirement(self) -> None:
        payload = export_auto_outcome_negatives.build_auto_outcome_negative_rows(
            candidate_rows=[_candidate("cand-undated", "vod-1", 10.0, 25.0, score=8.0, created_at="")],
            positive_rows=[],
            as_of_date=date(2026, 4, 29),
            min_age_days=14,
            allow_undated=False,
        )

        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["summary"]["skipped"]["undated_candidate"], 1)

    def test_holdout_floor_promotes_negatives_when_stable_split_is_train_heavy(self) -> None:
        candidates = [
            _candidate(f"cand-{idx}", f"vod-{idx}", float(idx * 40), float(idx * 40 + 20), score=2.0, created_at="")
            for idx in range(10)
        ]

        payload = export_auto_outcome_negatives.build_auto_outcome_negative_rows(
            candidate_rows=candidates,
            positive_rows=[],
            as_of_date=date(2026, 4, 29),
            holdout_ratio=0.2,
            max_rows=10,
        )

        rows = payload["rows"]
        summary = payload["summary"]
        self.assertEqual(summary["split_balance"]["target_holdout_rows"], 2)
        self.assertEqual(summary["holdout_rows"], 2)
        self.assertEqual(summary["train_rows"], 8)
        self.assertEqual(summary["by_dataset_source"]["auto_outcome_negative_holdout"], 2)
        self.assertTrue(
            all(
                row["dataset_source"] == f"auto_outcome_negative_{row['gold_holdout_split']}"
                for row in rows
            )
        )


def _candidate(
    candidate_id: str,
    vod_id: str,
    start: float,
    end: float,
    *,
    score: float,
    created_at: str,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "job_id": f"job-{vod_id}",
        "vod_id": vod_id,
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "window_start": start,
        "window_end": end,
        "label": 0,
        "label_class": "hard_negative",
        "heuristic_rank_score": score,
        "created_at": created_at,
        "features": {"hook_strength": score / 10.0},
    }
