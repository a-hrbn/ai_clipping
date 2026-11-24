import json
import unittest
import tempfile
from pathlib import Path

from models import ClipFeedbackPayload
from services import feedback_store


class FeedbackStoreTests(unittest.TestCase):
    def test_update_job_feedback_persists_outcome_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            debug = storage / "job-1" / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "feedback.jsonl").write_text(
                json.dumps({"job_id": "job-1", "candidate_id": "c1", "selected": True}),
                encoding="utf-8",
            )

            response = feedback_store.update_job_feedback(
                storage,
                "job-1",
                ClipFeedbackPayload(
                    candidate_id="c1",
                    rating="good",
                    feedback_tags=["good_clip"],
                    notes="guter opener",
                ),
            )

            self.assertEqual(response.candidate_id, "c1")
            self.assertEqual(response.rating, "good")
            self.assertEqual(response.notes, "guter opener")
            rows = feedback_store._read_feedback_rows(debug / "feedback.jsonl")
            self.assertEqual(rows[0]["feedback_rating"], "good")
            self.assertEqual(rows[0]["feedback_tags"], ["good_clip"])
            self.assertEqual(rows[0]["feedback_notes"], "guter opener")

    def test_update_job_feedback_persists_gold_eval_rows_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            job_dir = storage / "job-1"
            debug = job_dir / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (job_dir / "request.json").write_text(
                '{"url":"https://www.twitch.tv/videos/123","profile":"jynxzi"}',
                encoding="utf-8",
            )
            (debug / "candidate_features.jsonl").write_text(
                '{"candidate_id":"c1","source":"reaction","duration":10.0}',
                encoding="utf-8",
            )
            (debug / "candidates_normalized.jsonl").write_text(
                '{"id":"c1","start":12.5,"end":22.5,"source":"reaction"}',
                encoding="utf-8",
            )

            response = feedback_store.update_job_feedback(
                storage,
                "job-1",
                ClipFeedbackPayload(
                    candidate_id="c1",
                    mode="gold_eval",
                    gold_grade=3,
                    notes="klarer winner",
                ),
            )

            self.assertEqual(response.mode, "gold_eval")
            self.assertEqual(response.gold_grade, 3)
            rows = feedback_store._read_feedback_rows(debug / "gold_eval.jsonl")
            self.assertEqual(rows[0]["gold_grade"], 3)
            self.assertEqual(rows[0]["source_url"], "https://www.twitch.tv/videos/123")
            self.assertEqual(rows[0]["creator_id"], "jynxzi")
            self.assertEqual(rows[0]["window_start"], 12.5)
            self.assertIn(rows[0]["gold_holdout_split"], {"train", "holdout"})

    def test_update_job_feedback_requires_notes_for_high_gold_grades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            job_dir = storage / "job-1"
            debug = job_dir / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (job_dir / "request.json").write_text(
                '{"url":"https://www.twitch.tv/videos/123","profile":"jynxzi"}',
                encoding="utf-8",
            )
            (debug / "candidate_features.jsonl").write_text(
                '{"candidate_id":"c1","source":"reaction","duration":10.0}',
                encoding="utf-8",
            )
            (debug / "candidates_normalized.jsonl").write_text(
                '{"id":"c1","start":12.5,"end":22.5,"source":"reaction"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "gold_notes are required"):
                feedback_store.update_job_feedback(
                    storage,
                    "job-1",
                    ClipFeedbackPayload(
                        candidate_id="c1",
                        mode="gold_eval",
                        gold_grade=3,
                        notes="",
                    ),
                )

    def test_update_job_feedback_requires_notes_for_high_scoring_gold_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            job_dir = storage / "job-1"
            debug = job_dir / "debug"
            debug.mkdir(parents=True, exist_ok=True)
            (job_dir / "request.json").write_text(
                '{"url":"https://www.twitch.tv/videos/123","profile":"jynxzi"}',
                encoding="utf-8",
            )
            (debug / "candidate_features.jsonl").write_text(
                '{"candidate_id":"c1","source":"reaction","duration":10.0}',
                encoding="utf-8",
            )
            (debug / "candidates_normalized.jsonl").write_text(
                '{"id":"c1","start":12.5,"end":22.5,"source":"reaction"}',
                encoding="utf-8",
            )
            (debug / "ranked_candidates_pre_judge.jsonl").write_text(
                '{"candidate":{"id":"c1"},"heuristic_rank_score":8.7}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "gold_notes are required"):
                feedback_store.update_job_feedback(
                    storage,
                    "job-1",
                    ClipFeedbackPayload(
                        candidate_id="c1",
                        mode="gold_eval",
                        gold_grade=0,
                        notes="",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
