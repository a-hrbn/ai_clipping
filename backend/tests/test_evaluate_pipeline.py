import json
import tempfile
import unittest
from pathlib import Path

from ml import evaluate_pipeline


class EvaluatePipelineTests(unittest.TestCase):
    def test_evaluate_pipeline_rows_reports_topk_metrics(self) -> None:
        rows = [
            {
                "job_id": "job-1",
                "candidate_id": "strong",
                "label": 1,
                "eval_group": "job-1",
                "platform_bucket": "tiktok",
                "content_bucket": "reaction",
                "source_bucket": "reaction",
                "features": {
                    "candidate_id": "strong",
                    "hook_strength": 0.9,
                    "payoff_score": 6.0,
                    "max_hype": 6.2,
                    "context_efficiency": 0.8,
                    "standalone_score": 0.8,
                    "profile_fit_score": 0.8,
                    "boundary_variant_score": 0.9,
                    "boundary_variant_margin_score": 0.4,
                    "boundary_variant_count": 8.0,
                    "has_hook_rewrite": True,
                    "hook_rewrite_score": 0.9,
                    "hook_curiosity_gap_score": 0.8,
                    "source": "reaction",
                },
            },
            {
                "job_id": "job-1",
                "candidate_id": "weak",
                "label": 0,
                "eval_group": "job-1",
                "platform_bucket": "tiktok",
                "content_bucket": "quote_driven",
                "source_bucket": "quote",
                "features": {
                    "candidate_id": "weak",
                    "hook_strength": 0.1,
                    "payoff_score": 1.0,
                    "max_hype": 1.2,
                    "context_efficiency": 0.2,
                    "standalone_score": 0.2,
                    "profile_fit_score": 0.2,
                    "source": "quote",
                },
            },
        ]

        report = evaluate_pipeline.evaluate_pipeline_rows(rows, profile="general")

        self.assertEqual(report["evaluation_kind"], "heuristic_pipeline")
        self.assertFalse(report["trained_model_used"])
        self.assertIn("heuristic pipeline ranker", report["interpretation_warning"])
        self.assertEqual(report["eval_scope"], "all_non_boundary_rows")
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["metrics"]["top1_positive_rate"], 1.0)
        self.assertEqual(report["metrics"]["recall_at_10"], 1.0)
        self.assertGreater(report["metrics"]["mrr"], 0.0)
        self.assertIn("tiktok", report["by_platform"])
        self.assertEqual(report["serving_reranker"]["hook_reranker_boost"], 1)
        self.assertEqual(report["serving_reranker"]["boundary_variant_set_max"], 1)
        self.assertIn("hook_reranker_boost", report["top_candidates"][0]["rationale"])
        self.assertIn("boundary_variant_set_max", report["top_candidates"][0]["rationale"])

    def test_evaluate_pipeline_rows_can_exclude_twitch_proxy_labels(self) -> None:
        rows = [
            {
                "job_id": "job-1",
                "candidate_id": "proxy",
                "label": 1,
                "eval_group": "job-1",
                "platform_bucket": "twitch",
                "content_bucket": "general",
                "source_bucket": "twitch_label_proxy",
                "dataset_source": "twitch_vod_primary",
                "supervision_tier": "twitch_clip_proxy",
                "features": {
                    "candidate_id": "proxy",
                    "source": "twitch_label_proxy",
                    "hook_strength": 0.8,
                    "payoff_score": 5.0,
                },
            },
            {
                "job_id": "job-1",
                "candidate_id": "outcome",
                "label": 0,
                "eval_group": "job-1",
                "platform_bucket": "twitch",
                "content_bucket": "background",
                "source_bucket": "background_negative",
                "dataset_source": "auto_outcome_negative_train",
                "supervision_tier": "auto_outcome_negative",
                "features": {
                    "candidate_id": "outcome",
                    "source": "background_negative",
                    "hook_strength": 0.1,
                    "payoff_score": 1.0,
                },
            },
            {
                "job_id": "job-1",
                "candidate_id": "boundary",
                "label": 0,
                "eval_group": "job-1",
                "platform_bucket": "twitch",
                "content_bucket": "general",
                "source_bucket": "boundary_perturbation",
                "features": {
                    "candidate_id": "boundary",
                    "source": "boundary_perturbation",
                },
            },
        ]

        report = evaluate_pipeline.evaluate_pipeline_rows(
            rows,
            profile="general",
            exclude_proxy_labels=True,
        )

        self.assertEqual(report["eval_scope"], "no_twitch_clip_proxy")
        self.assertEqual(report["input_rows"], 3)
        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["excluded_rows"], 2)
        self.assertEqual(
            report["excluded_rows_by_reason"],
            {"boundary_perturbation": 1, "twitch_clip_proxy": 1},
        )
        self.assertEqual(report["top_candidates"][0]["candidate_id"], "outcome")

    def test_trained_product_mode_applies_threshold_and_grouped_diversity(self) -> None:
        rows = [
            _row("g1-good", 1, "vod-1", 10.0, hook_strength=1.0),
            _row("g1-low", 1, "vod-1", 12.0, hook_strength=0.0),
            _row("g2-good", 1, "vod-2", 10.0, hook_strength=1.0),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_linear_artifact(artifact_dir)

            report = evaluate_pipeline.evaluate_pipeline_rows(
                rows,
                profile="general",
                artifact_dir=artifact_dir,
                evaluation_mode="trained_product",
                selection_top_k=1,
            )

        self.assertEqual(report["evaluation_kind"], "trained_product_pipeline")
        self.assertTrue(report["trained_model_used"])
        self.assertTrue(report["model_evaluation"])
        self.assertEqual(report["artifact"]["status"], "usable")
        self.assertEqual(report["metrics"]["classification"]["selected_count"], 2)
        self.assertEqual(report["metrics"]["classification"]["recall"], 0.667)
        self.assertEqual(report["metrics"]["diverse_selection"]["selected_rows"], 2)
        self.assertEqual(report["metrics"]["diverse_selection"]["selected_positive_rows"], 2)
        selected_ids = {
            item["candidate_id"]
            for item in report["top_candidates"]
            if item["selected_after_diversity"]
        }
        self.assertEqual(selected_ids, {"g1-good", "g2-good"})
        low = next(item for item in report["top_candidates"] if item["candidate_id"] == "g1-low")
        self.assertFalse(low["passes_serving_threshold"])


def _row(candidate_id: str, label: int, group: str, start: float, *, hook_strength: float) -> dict:
    return {
        "job_id": group,
        "candidate_id": candidate_id,
        "label": label,
        "eval_group": group,
        "window_start": start,
        "window_end": start + 10.0,
        "platform_bucket": "twitch",
        "content_bucket": "general",
        "source_bucket": "quote",
        "features": {
            "candidate_id": candidate_id,
            "duration": 10.0,
            "hook_strength": hook_strength,
            "source": "quote",
        },
    }


def _write_linear_artifact(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "general_ranker.json").write_text(
        json.dumps(
            {
                "version": 3,
                "profile": "general",
                "model_type": "linear",
                "selected_model": "logistic",
                "feature_names": ["hook_strength"],
                "weights": {"hook_strength": 5.0},
                "scales": {"hook_strength": {"mean": 0.0, "std": 1.0}},
                "intercept": 0.0,
                "blend": 1.0,
                "serving_threshold": 2.0,
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
                    "precision_at_5": 0.8,
                    "ndcg_at_5": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
