import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from services import trend_prior
from services.transcriber import RichSegment


class TrendPriorTests(unittest.TestCase):
    def test_scores_hook_and_topic_matches(self) -> None:
        window = [
            RichSegment(start=0.0, end=2.0, text="No way, what happened here?"),
            RichSegment(start=2.0, end=5.0, text="This reveal is actually insane and chat went crazy."),
        ]

        scores = trend_prior.score_candidate_window(window, profile="general")

        self.assertGreater(scores["trend_hook_score"], 0.0)
        self.assertGreater(scores["trend_topic_score"], 0.0)
        self.assertGreater(scores["trend_prior_score"], 0.0)
        self.assertIn("disbelief_reaction", scores["matched_labels"])

    def test_can_disable_trend_prior(self) -> None:
        window = [RichSegment(start=0.0, end=2.0, text="No way, what happened here?")]

        with patch.object(trend_prior.settings, "enable_trend_prior", False):
            scores = trend_prior.score_candidate_window(window, profile="general")

        self.assertEqual(scores["trend_hook_score"], 0.0)
        self.assertEqual(scores["trend_topic_score"], 0.0)
        self.assertEqual(scores["trend_prior_score"], 0.0)
        self.assertEqual(scores["matched_labels"], [])

    def test_dynamic_feedback_memory_boosts_matching_windows(self) -> None:
        trend_prior._load_dynamic_memory.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            feedback_dir = storage / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            (feedback_dir / "feedback_updates.jsonl").write_text(
                json.dumps(
                    {
                        "job_id": "job-memory",
                        "candidate_id": "clip-1",
                        "selected": True,
                        "publish_platform": "tiktok",
                        "user_feedback_tags": ["strong_hook"],
                        "retention_3s": 0.86,
                        "completion_rate": 0.58,
                        "view_count": 1000,
                        "share_count": 40,
                        "save_count": 35,
                        "final_start": 0.0,
                        "final_end": 8.0,
                    }
                ) + "\n",
                encoding="utf-8",
            )
            transcripts_dir = storage / "transcripts"
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            (transcripts_dir / "job-memory.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"start": 0.0, "end": 2.0, "text": "No way this happened"}),
                        json.dumps({"start": 2.0, "end": 5.0, "text": "chat went crazy after the reveal"}),
                        json.dumps({"start": 5.0, "end": 8.0, "text": "everyone started laughing"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            window = [
                RichSegment(start=10.0, end=12.0, text="No way this happened"),
                RichSegment(start=12.0, end=15.0, text="chat went crazy after the reveal"),
            ]

            with patch.object(trend_prior.settings, "storage_dir", str(storage)), \
                 patch.object(trend_prior, "_DEFAULT_TRANSCRIPTS_DIR", transcripts_dir):
                scores = trend_prior.score_candidate_window(
                    window,
                    profile="general",
                    strategy={"platform": "tiktok", "goal": "funny reaction"},
                )

        self.assertGreater(scores["dynamic_trend_score"], 0.0)
        self.assertGreater(scores["outcome_prior_score"], 0.0)
        self.assertIn("strong_hook", scores["matched_feedback_labels"])


if __name__ == "__main__":
    unittest.main()
