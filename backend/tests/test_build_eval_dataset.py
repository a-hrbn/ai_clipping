import unittest

from ml import build_eval_dataset


class BuildEvalDatasetTests(unittest.TestCase):
    def test_build_eval_dataset_adds_bucket_metadata(self) -> None:
        rows = [
            {
                "job_id": "job-1",
                "candidate_id": "c1",
                "label": 1,
                "selected": True,
                "label_class": "outcome_positive",
                "supervision_tier": "outcome_feedback",
                "publish_platform": "tiktok",
                "final_variant_key": "aggressive_hook",
                "candidate_source": "reaction",
                "features": {"source": "reaction"},
                "retention_3s": 0.82,
                "completion_rate": 0.56,
            },
            {
                "job_id": "job-1",
                "candidate_id": "c2",
                "label": 0,
                "selected": False,
                "label_class": "pipeline_negative",
                "supervision_tier": "pipeline_feedback",
                "candidate_source": "quote",
                "features": {"source": "quote"},
            },
        ]

        eval_rows = build_eval_dataset.build_eval_dataset(rows)

        self.assertEqual(eval_rows[0]["eval_group"], "job-1")
        self.assertEqual(eval_rows[0]["platform_bucket"], "tiktok")
        self.assertEqual(eval_rows[0]["variant_bucket"], "aggressive_hook")
        self.assertEqual(eval_rows[0]["content_bucket"], "reaction")
        self.assertTrue(eval_rows[0]["has_outcome_feedback"])
        self.assertEqual(eval_rows[1]["content_bucket"], "quote_driven")

    def test_eval_dataset_summary_counts_platforms_and_roles(self) -> None:
        rows = build_eval_dataset.build_eval_dataset(
            [
                {
                    "job_id": "job-1",
                    "candidate_id": "c1",
                    "label": 1,
                    "selected": True,
                    "label_class": "outcome_positive",
                    "supervision_tier": "outcome_feedback",
                    "publish_platform": "youtube_shorts",
                    "candidate_source": "reaction",
                    "features": {"source": "reaction"},
                    "retention_3s": 0.8,
                },
                {
                    "job_id": "job-2",
                    "candidate_id": "c2",
                    "label": 0,
                    "selected": False,
                    "label_class": "pipeline_negative",
                    "supervision_tier": "pipeline_feedback",
                    "candidate_source": "quote",
                    "features": {"source": "quote"},
                },
            ]
        )

        summary = build_eval_dataset.eval_dataset_summary(rows)

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["by_platform"]["general"], 1)
        self.assertEqual(summary["by_platform"]["youtube_shorts"], 1)
        self.assertEqual(summary["dataset_roles"]["outcome_eval"], 1)
        self.assertEqual(summary["dataset_roles"]["pipeline_negative_eval"], 1)


if __name__ == "__main__":
    unittest.main()
