import unittest

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

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["metrics"]["top1_positive_rate"], 1.0)
        self.assertEqual(report["metrics"]["recall_at_10"], 1.0)
        self.assertGreater(report["metrics"]["mrr"], 0.0)
        self.assertIn("tiktok", report["by_platform"])


if __name__ == "__main__":
    unittest.main()
