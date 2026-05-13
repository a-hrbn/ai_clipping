import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import build_rollout_review_queue


class BuildRolloutReviewQueueTests(unittest.TestCase):
    def test_build_queue_excludes_gold_and_holdout_vods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "rows.jsonl"
            gold_rows = root / "gold.jsonl"
            out = root / "rollout_queue"
            _write_artifact(artifact_dir)
            rows = [
                _row("vod-a", "top-a", 3.0, start=10.0),
                _row("vod-a", "near-a", 2.7, start=20.0),
                _row("vod-a", "second-a", 2.0, start=90.0),
                _row("vod-a", "below-review-a", 1.9, start=130.0),
                _row("vod-b", "top-b", 4.0, start=30.0),
                _row("vod-holdout", "holdout-candidate", 9.0, start=40.0),
                _row("vod-a", "already-gold", 8.0, start=160.0),
            ]
            _write_jsonl(dataset, rows)
            _write_jsonl(
                gold_rows,
                [
                    {**rows[-1], "queue_id": "existing-gold"},
                    {"vod_id": "vod-holdout", "gold_holdout_split": "holdout"},
                ],
            )

            summary = build_rollout_review_queue.build_queue(
                dataset_path=dataset,
                existing_gold_rows_path=gold_rows,
                holdout_rows_path=gold_rows,
                profile="gaming",
                artifact_dir=artifact_dir,
                out_base=out,
                top_k_per_vod=2,
                max_total=10,
                min_gap_sec=20.0,
                review_threshold=7.4,
            )

            with (root / "rollout_queue.csv").open("r", encoding="utf-8", newline="") as handle:
                queue_rows = list(csv.DictReader(handle))
            sidecar_rows = [json.loads(line) for line in (root / "rollout_queue.features.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["review_required"], True)
        self.assertEqual(summary["auto_publish_allowed"], True)
        self.assertEqual(summary["excluded_gold_holdout_vod_rows"], 1)
        self.assertEqual(summary["excluded_existing_gold_rows"], 1)
        self.assertEqual(summary["eligible_rows"], 4)
        self.assertEqual(summary["queue_rows"], 3)
        self.assertEqual(summary["serving_threshold"], 9.0)
        self.assertEqual(summary["review_threshold"], 7.4)
        self.assertEqual(summary["trained_runtime_uses_artifact"], True)
        self.assertEqual(summary["artifact_status"], "usable")
        self.assertEqual(summary["auto_accept_enabled"], True)
        self.assertEqual(summary["by_rollout_bucket"], {"auto_accept_candidate": 2, "review_candidate": 1})
        self.assertEqual(len(queue_rows), 3)
        self.assertEqual(len(sidecar_rows), 3)
        self.assertEqual({row["queue_id"] for row in queue_rows}, {row["queue_id"] for row in sidecar_rows})
        self.assertNotIn("holdout-candidate", {row["candidate_id"] for row in queue_rows})
        self.assertNotIn("already-gold", {row["candidate_id"] for row in queue_rows})
        self.assertNotIn("below-review-a", {row["candidate_id"] for row in queue_rows})
        self.assertEqual([row["candidate_id"] for row in queue_rows if row["vod_id"] == "vod-a"], ["top-a", "second-a"])
        queue_by_id = {row["candidate_id"]: row for row in queue_rows}
        self.assertEqual(queue_by_id["top-a"]["rollout_bucket"], "auto_accept_candidate")
        self.assertEqual(queue_by_id["top-a"]["review_required"], "false")
        self.assertEqual(queue_by_id["top-a"]["auto_publish_allowed"], "true")
        self.assertEqual(queue_by_id["second-a"]["rollout_bucket"], "review_candidate")
        self.assertEqual(queue_by_id["second-a"]["review_required"], "true")
        self.assertEqual(queue_by_id["second-a"]["auto_publish_allowed"], "false")
        self.assertEqual(queue_by_id["second-a"]["review_threshold"], "7.4")
        sidecar_by_id = {row["candidate_id"]: row for row in sidecar_rows}
        self.assertTrue(sidecar_by_id["top-a"]["auto_publish_allowed"])
        self.assertTrue(sidecar_by_id["top-a"]["trained_runtime_uses_artifact"])
        self.assertEqual(sidecar_by_id["top-a"]["artifact_status"], "usable")
        self.assertFalse(sidecar_by_id["top-a"]["review_required"])
        self.assertFalse(sidecar_by_id["second-a"]["auto_publish_allowed"])
        self.assertTrue(sidecar_by_id["second-a"]["review_required"])
        self.assertTrue(all("transcript" not in row for row in sidecar_rows))
        self.assertTrue(all("prior_context" not in row for row in sidecar_rows))

    def test_rollout_thresholds_include_high_heuristic_model_disagreement(self) -> None:
        row = _row("vod-a", "disagreement-a", 0.0, start=10.0)
        scored = [
            {
                "row": row,
                "rank_score": -2.0,
                "heuristic_score": 10.5,
                "trained_score": -11.0,
                "blend": 0.55,
                "serving_threshold": 4.0,
                "score_margin": -6.0,
                "model_positive": False,
                "rationale": [],
            }
        ]

        enriched = build_rollout_review_queue._apply_rollout_thresholds(
            scored,
            review_threshold=3.0,
            disagreement_heuristic_threshold=8.0,
            disagreement_min_gap=4.0,
        )

        item = enriched[0]
        self.assertTrue(item["rollout_eligible"])
        self.assertTrue(item["review_required"])
        self.assertFalse(item["auto_publish_allowed"])
        self.assertEqual(item["rollout_bucket"], "model_heuristic_disagreement_review")
        self.assertEqual(item["disagreement_gap"], 12.5)
        self.assertGreater(float(item["rollout_priority_score"]), 100.0)

    def test_rollout_thresholds_disable_auto_accept_when_artifact_is_not_usable(self) -> None:
        scored = [
            {
                "row": _row("vod-a", "high-score-a", 0.0, start=10.0),
                "rank_score": 5.0,
                "heuristic_score": 5.0,
                "trained_score": None,
                "blend": 0.0,
                "serving_threshold": 4.0,
                "score_margin": 1.0,
                "model_positive": True,
                "trained_runtime_uses_artifact": False,
                "artifact_status": "rejected",
                "artifact_reason": "quality_gate_failed",
                "rationale": [],
            }
        ]

        enriched = build_rollout_review_queue._apply_rollout_thresholds(
            scored,
            review_threshold=3.0,
            disagreement_heuristic_threshold=8.0,
            disagreement_min_gap=4.0,
            auto_accept_enabled=False,
        )

        item = enriched[0]
        self.assertTrue(item["rollout_eligible"])
        self.assertTrue(item["review_required"])
        self.assertFalse(item["auto_publish_allowed"])
        self.assertEqual(item["rollout_bucket"], "review_candidate")

    def test_build_queue_is_review_only_when_ranker_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "rows.jsonl"
            out = root / "rollout_queue"
            _write_rejected_artifact(artifact_dir)
            _write_jsonl(dataset, [_row("vod-a", "high-score-a", 9.0, start=10.0)])

            summary = build_rollout_review_queue.build_queue(
                dataset_path=dataset,
                existing_gold_rows_path=None,
                holdout_rows_path=None,
                profile="gaming",
                artifact_dir=artifact_dir,
                out_base=out,
                top_k_per_vod=2,
                max_total=10,
                min_gap_sec=20.0,
                review_threshold=3.0,
            )

            with (root / "rollout_queue.csv").open("r", encoding="utf-8", newline="") as handle:
                queue_rows = list(csv.DictReader(handle))

        self.assertEqual(summary["trained_runtime_uses_artifact"], False)
        self.assertEqual(summary["artifact_status"], "rejected")
        self.assertEqual(summary["artifact_reason"], "validation_filter_precision_below_minimum")
        self.assertEqual(summary["auto_accept_enabled"], False)
        self.assertEqual(summary["auto_accept_disabled_reason"], "ranker_artifact_not_usable")
        self.assertEqual(summary["recommended_action"], "review_only_until_ranker_artifact_usable")
        self.assertEqual(summary["by_rollout_bucket"], {"review_candidate": 1})
        self.assertEqual(queue_rows[0]["rollout_bucket"], "review_candidate")
        self.assertEqual(queue_rows[0]["auto_publish_allowed"], "false")
        self.assertEqual(queue_rows[0]["artifact_status"], "rejected")

    def test_rollout_thresholds_drop_weak_below_review_candidates(self) -> None:
        row = _row("vod-a", "weak-a", 0.0, start=10.0)
        scored = [
            {
                "row": row,
                "rank_score": 2.0,
                "heuristic_score": 5.0,
                "trained_score": -1.0,
                "blend": 0.55,
                "serving_threshold": 4.0,
                "score_margin": -2.0,
                "model_positive": False,
                "rationale": [],
            }
        ]

        enriched = build_rollout_review_queue._apply_rollout_thresholds(
            scored,
            review_threshold=3.0,
            disagreement_heuristic_threshold=8.0,
            disagreement_min_gap=4.0,
        )

        item = enriched[0]
        self.assertFalse(item["rollout_eligible"])
        self.assertFalse(item["review_required"])
        self.assertEqual(item["rollout_bucket"], "below_review_threshold")

    def test_select_top_by_vod_caps_disagreement_without_capping_other_buckets(self) -> None:
        scored = [
            _scored_item("vod-a", "auto-a", rank_score=5.0, heuristic_score=5.0, start=10.0),
            _scored_item("vod-a", "disagreement-a", rank_score=-2.0, heuristic_score=10.5, start=50.0),
            _scored_item("vod-a", "disagreement-b", rank_score=-3.0, heuristic_score=10.0, start=90.0),
            _scored_item("vod-a", "disagreement-c", rank_score=-4.0, heuristic_score=9.5, start=130.0),
        ]
        eligible = [
            item
            for item in build_rollout_review_queue._apply_rollout_thresholds(
                scored,
                review_threshold=3.0,
                disagreement_heuristic_threshold=8.0,
                disagreement_min_gap=4.0,
            )
            if item["rollout_eligible"]
        ]

        selected = build_rollout_review_queue._select_top_by_vod(
            eligible,
            top_k_per_vod=10,
            max_total=10,
            min_gap_sec=0.0,
            max_disagreement_per_vod=2,
        )

        selected_ids = [item["row"]["candidate_id"] for item in selected]
        selected_buckets = [item["rollout_bucket"] for item in selected]
        self.assertEqual(selected_ids, ["auto-a", "disagreement-a", "disagreement-b"])
        self.assertEqual(selected_buckets.count("auto_accept_candidate"), 1)
        self.assertEqual(selected_buckets.count("model_heuristic_disagreement_review"), 2)
        self.assertNotIn("disagreement-c", selected_ids)


def _write_artifact(artifact_dir: Path) -> None:
    (artifact_dir / "gaming_ranker.json").write_text(
        json.dumps(
            {
                "version": 3,
                "profile": "gaming",
                "model_type": "linear",
                "feature_names": ["hook_strength"],
                "weights": {"hook_strength": 1.0},
                "scales": {"hook_strength": {"mean": 0.0, "std": 1.0}},
                "training_stats": {
                    "rows": 100,
                    "positive_rows": 40,
                    "negative_rows": 60,
                    "train_rows": 80,
                    "validation_rows": 20,
                },
                "validation": {
                    "accuracy": 0.75,
                    "precision": 0.7,
                    "recall": 0.8,
                },
                "intercept": 0.0,
                "blend": 1.0,
                "serving_threshold": 9.0,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_rejected_artifact(artifact_dir: Path) -> None:
    (artifact_dir / "gaming_ranker.json").write_text(
        json.dumps(
            {
                "version": 3,
                "profile": "gaming",
                "model_type": "linear",
                "feature_names": ["hook_strength"],
                "weights": {"hook_strength": 1.0},
                "scales": {"hook_strength": {"mean": 0.0, "std": 1.0}},
                "training_stats": {
                    "rows": 100,
                    "positive_rows": 40,
                    "negative_rows": 60,
                    "train_rows": 80,
                    "validation_rows": 20,
                },
                "validation": {
                    "accuracy": 0.75,
                    "precision": 0.7,
                    "recall": 0.8,
                },
                "filter_gate": {
                    "precision": 0.10,
                    "recall": 0.8,
                    "f1": 0.18,
                    "selected_rate": 0.2,
                },
                "intercept": 0.0,
                "blend": 1.0,
                "serving_threshold": 9.0,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _row(vod_id: str, candidate_id: str, hook_strength: float, *, start: float) -> dict:
    return {
        "job_id": f"job-{vod_id}",
        "candidate_id": candidate_id,
        "vod_id": vod_id,
        "creator_id": "jynxzi",
        "platform": "twitch",
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "window_start": start,
        "window_end": start + 25.0,
        "label": 0,
        "label_class": "review_candidate",
        "candidate_source": "test_source",
        "features": {
            "candidate_id": candidate_id,
            "hook_strength": hook_strength,
            "source": "test_source",
        },
    }


def _scored_item(vod_id: str, candidate_id: str, *, rank_score: float, heuristic_score: float, start: float) -> dict:
    return {
        "row": _row(vod_id, candidate_id, 0.0, start=start),
        "rank_score": rank_score,
        "heuristic_score": heuristic_score,
        "trained_score": rank_score,
        "blend": 0.55,
        "serving_threshold": 4.0,
        "score_margin": rank_score - 4.0,
        "model_positive": rank_score >= 4.0,
        "rationale": [],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
