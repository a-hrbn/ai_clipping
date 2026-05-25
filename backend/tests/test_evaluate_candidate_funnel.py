import json
import tempfile
import unittest
from pathlib import Path

from ml import evaluate_candidate_funnel


class EvaluateCandidateFunnelTests(unittest.TestCase):
    def test_summarize_debug_dir_reads_trace_and_lift_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp) / "debug"
            debug_dir.mkdir()
            (debug_dir / "candidate_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "candidate_id": "winner",
                                "source": "peak",
                                "final_selected": True,
                                "proposal_score": 0.8,
                                "pre_judge_rank": 3,
                                "judge_score": 8.0,
                            }
                        ),
                        json.dumps(
                            {
                                "candidate_id": "missed",
                                "source": "dense_inventory",
                                "final_selected": False,
                                "proposal_score": 0.2,
                                "pre_judge_rank": 42,
                                "judge_score": 7.5,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (debug_dir / "candidate_funnel_summary.json").write_text(
                json.dumps(
                    {
                        "stage_counts": {"final_moments": 1},
                        "quality_breakdown": {"judged_kept": 1},
                        "proposal_lift": {"precision_at_positive_k": 0.5},
                        "ranker_lift": {"high_judge_pre_judge_top20_ratio": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            summary = evaluate_candidate_funnel.summarize_debug_dir(debug_dir)

        self.assertEqual(summary["trace_rows"], 2)
        self.assertEqual(summary["final_selected_rows"], 1)
        self.assertEqual(summary["high_judge_rows"], 2)
        self.assertEqual(summary["quality_breakdown"], {"judged_kept": 1})
        self.assertEqual(summary["proposal_lift"]["precision_at_positive_k"], 0.5)
        self.assertEqual(summary["score_summary"]["final_pre_judge_rank_mean"], 3.0)
        self.assertEqual(summary["high_judge_not_final"][0]["candidate_id"], "missed")

    def test_delta_summary_compares_current_against_baseline(self) -> None:
        current = {
            "final_selected_rows": 5,
            "judged_rows": 80,
            "high_judge_rows": 12,
            "score_summary": {"final_judge_score_mean": 7.4},
        }
        baseline = {
            "final_selected_rows": 4,
            "judged_rows": 40,
            "high_judge_rows": 6,
            "score_summary": {"final_judge_score_mean": 6.9},
        }

        delta = evaluate_candidate_funnel._delta_summary(current, baseline)

        self.assertEqual(delta["final_selected_delta"], 1)
        self.assertEqual(delta["judged_rows_delta"], 40)
        self.assertEqual(delta["high_judge_rows_delta"], 6)
        self.assertEqual(delta["score_summary_delta"]["final_judge_score_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
