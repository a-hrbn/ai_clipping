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
                "outcome_view_tier": "viral",
                "rank_target": 4,
                "ordinal_label": 2,
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
        self.assertEqual(eval_rows[0]["outcome_view_tier"], "viral")
        self.assertEqual(eval_rows[0]["rank_target"], 4)
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
                    "candidate_source": "chat_velocity",
                    "features": {"source": "chat_velocity"},
                    "retention_3s": 0.8,
                    "outcome_view_tier": "strong",
                    "rank_target": 3,
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
        self.assertEqual(summary["by_content_bucket"]["reaction"], 1)
        self.assertEqual(summary["by_outcome_view_tier"]["strong"], 1)
        self.assertEqual(summary["by_rank_target"]["3"], 1)

    def test_outcome_holdout_role_is_explicit(self) -> None:
        rows = build_eval_dataset.build_eval_dataset(
            [
                {
                    "job_id": "job-1",
                    "candidate_id": "c1",
                    "label": 1,
                    "selected": True,
                    "label_class": "outcome_top_viewed_positive",
                    "supervision_tier": "outcome_holdout",
                    "dataset_source": "outcome_top_viewed_holdout",
                    "candidate_source": "hype_peak",
                    "features": {"source": "hype_peak"},
                }
            ]
        )

        self.assertEqual(rows[0]["dataset_role"], "outcome_holdout")

    def test_outcome_train_role_is_explicit(self) -> None:
        rows = build_eval_dataset.build_eval_dataset(
            [
                {
                    "job_id": "job-1",
                    "candidate_id": "c1",
                    "label": 1,
                    "selected": True,
                    "label_class": "outcome_top_viewed_positive",
                    "supervision_tier": "outcome_train",
                    "dataset_source": "outcome_top_viewed_train",
                    "candidate_source": "hype_peak",
                    "features": {"source": "hype_peak"},
                }
            ]
        )

        self.assertEqual(rows[0]["dataset_role"], "outcome_train")


if __name__ == "__main__":
    unittest.main()
