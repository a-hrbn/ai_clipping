import unittest
import tempfile
from pathlib import Path

from models import ClipFeedbackPayload
from services import feedback_store
from services.timeline_types import Candidate, CandidateFeatures, RankedCandidate


class FeedbackStoreTests(unittest.TestCase):
    def test_build_pipeline_feedback_marks_selected_candidates(self) -> None:
        ranked = [
            RankedCandidate(
                candidate=Candidate(id="a", start=0.0, end=10.0, source="reaction", anchor_time=5.0),
                features=CandidateFeatures(
                    candidate_id="a",
                    duration=10.0,
                    hook_delay_sec=1.0,
                    hook_strength=0.8,
                    max_hype=5.0,
                    mean_hype=3.0,
                    arc_score=4.0,
                    chat_peak=0.5,
                    motion_peak=0.5,
                    face_reaction_peak=0.2,
                    speaker_turns=1,
                    exclamation_count=1,
                    laughter_count=0,
                    question_open=False,
                    context_cost=0.1,
                    payoff_score=5.0,
                    context_efficiency=0.8,
                    quoteability_score=0.2,
                    surprise_score=0.6,
                    contradiction_score=0.0,
                    standalone_score=0.7,
                    semantic_density=0.3,
                    profile_fit_score=0.7,
                    source="reaction",
                ),
                rank_score=8.0,
                rationale=[],
            )
        ]
        feedback = feedback_store.build_pipeline_feedback(
            "job-1",
            ranked,
            [
                {
                    "start_time": 2.0,
                    "end_time": 9.0,
                    "viral_score": 8.0,
                    "reason": "story",
                    "platform": "tiktok",
                    "variant_key": "aggressive_hook",
                }
            ],
        )

        self.assertEqual(len(feedback), 1)
        self.assertTrue(feedback[0].selected)
        self.assertFalse(feedback[0].exported)
        self.assertEqual(feedback[0].publish_platform, "tiktok")
        self.assertEqual(feedback[0].final_variant_key, "aggressive_hook")

    def test_update_job_feedback_persists_outcome_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            debug = storage / "job-1" / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            feedback_store.write_job_feedback(
                debug,
                [
                    feedback_store.CandidateFeedback(
                        job_id="job-1",
                        candidate_id="c1",
                        selected=True,
                        exported=False,
                    )
                ],
            )

            response = feedback_store.update_job_feedback(
                storage,
                "job-1",
                ClipFeedbackPayload(
                    candidate_id="c1",
                    tags=["strong_hook"],
                    notes="guter opener",
                    publish_platform="youtube_shorts",
                    impressions=4000,
                    avg_watch_time=17.5,
                    retention_3s=0.82,
                    retention_50pct=0.46,
                    completion_rate=0.58,
                    views_1h=1200,
                    views_24h=7200,
                    share_count=12,
                    save_count=20,
                ),
            )

            self.assertEqual(response.candidate_id, "c1")
            self.assertEqual(response.tags, ["strong_hook"])
            self.assertEqual(response.publish_platform, "youtube_shorts")
            self.assertEqual(response.retention_3s, 0.82)
            rows = feedback_store._read_feedback_rows(debug / "feedback.jsonl")
            self.assertEqual(rows[0]["views_24h"], 7200)
            self.assertEqual(rows[0]["impressions"], 4000)
            self.assertEqual(rows[0]["completion_rate"], 0.58)


if __name__ == "__main__":
    unittest.main()
