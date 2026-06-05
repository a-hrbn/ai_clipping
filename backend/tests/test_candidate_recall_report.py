import unittest

from services import candidate_recall_report
from services.timeline_types import Candidate, RankedCandidate


def _candidate(candidate_id: str, source: str, start: float) -> Candidate:
    return Candidate(
        id=candidate_id,
        source=source,
        start=start,
        end=start + 20.0,
        anchor_time=start + 10.0,
        score_hint=1.0,
    )


def _ranked(candidate: Candidate, score: float) -> RankedCandidate:
    return RankedCandidate(
        candidate=candidate,
        features=None,
        rank_score=score,
        rationale=["test"],
        trained_model_used=True,
    )


class CandidateRecallReportTests(unittest.TestCase):
    def test_report_tracks_stage_losses_and_top_rejections(self) -> None:
        peak = _candidate("p1", "peak", 10.0)
        dense = _candidate("d1", "dense_inventory", 30.0)
        dropped = _candidate("d2", "dense_inventory", 60.0)
        ranked = [_ranked(peak, 9.0), _ranked(dense, 8.0)]

        report = candidate_recall_report.build_candidate_recall_report(
            stage_candidates={
                "peak": [peak],
                "semantic": [],
                "dense": [],
                "dense_inventory": [dense, dropped],
                "fallback": [],
                "merged": [peak, dense, dropped],
                "normalized": [peak, dense],
            },
            pre_judge_ranked=ranked,
            judged_ranked=ranked,
            quality_gate_rows=[
                {"candidate_id": "p1", "source": "peak", "accepted": True, "rank_score": 9.0},
                {
                    "candidate_id": "d1",
                    "source": "dense_inventory",
                    "accepted": False,
                    "rank_score": 8.0,
                    "rejection_reasons": ["payoff_strength_below_min"],
                },
            ],
            final_ranked=[ranked[0]],
            final_moments=[{"candidate_id": "p1::balanced"}],
            judge_top_k=1,
        )

        self.assertEqual(report["stage_counts"]["merged"], 3)
        self.assertEqual(report["stage_counts"]["normalized"], 2)
        self.assertEqual(report["stage_counts"]["quality_kept"], 1)
        self.assertEqual(report["stage_counts"]["judged_quality_kept"], 0)
        self.assertEqual(report["stage_counts"]["unjudged_quality_kept"], 1)
        self.assertEqual(report["stage_counts"]["final_moments"], 1)

        losses = {row["transition"]: row for row in report["losses"]}
        self.assertEqual(losses["merged_to_normalized"]["lost_count"], 1)
        self.assertEqual(losses["ranked_to_judge_scope"]["lost_count"], 1)
        self.assertEqual(losses["judge_scope_to_judged_quality_kept"]["output_count"], 0)
        self.assertEqual(
            report["quality_gate"]["rejected_by_reason"],
            {"payoff_strength_below_min": 1},
        )
        self.assertEqual(report["quality_gate"]["judged_kept_rows"], 0)
        self.assertEqual(report["quality_gate"]["unjudged_kept_rows"], 1)

    def test_report_tracks_positive_window_recall_by_stage(self) -> None:
        matched = _candidate("dense-positive", "dense_inventory", 30.0)
        unrelated = _candidate("peak", "peak", 90.0)
        ranked = [_ranked(unrelated, 9.0)]

        report = candidate_recall_report.build_candidate_recall_report(
            stage_candidates={
                "peak": [unrelated],
                "semantic": [],
                "dense": [],
                "dense_inventory": [matched],
                "fallback": [],
                "merged": [matched, unrelated],
                "normalized": [matched, unrelated],
            },
            pre_judge_ranked=ranked,
            judged_ranked=[],
            quality_gate_rows=[],
            final_ranked=[],
            final_moments=[],
            judge_top_k=1,
            positive_windows=[
                {
                    "candidate_id": "short-a",
                    "repost_source_url": "https://youtube.test/shorts/a",
                    "window_start": 31.0,
                    "window_end": 48.0,
                    "transcript_similarity": 0.8,
                }
            ],
            positive_overlap_ratio=0.45,
        )

        recall = report["positive_window_recall"]
        self.assertEqual(recall["positive_windows"], 1)
        self.assertEqual(recall["recall_by_stage"]["dense_inventory"]["matched"], 1)
        self.assertEqual(recall["recall_by_stage"]["normalized"]["matched"], 1)
        self.assertEqual(recall["recall_by_stage"]["ranked_pre_judge"]["matched"], 0)
        detail = recall["details"][0]
        self.assertEqual(detail["last_matched_stage"], "normalized")
        self.assertEqual(detail["first_missed_stage_after_last_match"], "ranked_pre_judge")
        self.assertEqual(detail["best_by_stage"]["normalized"]["candidate_id"], "dense-positive")

    def test_candidate_trace_tracks_drop_stage_scores_and_final_selection(self) -> None:
        peak = _candidate("p1", "peak", 10.0)
        dense = _candidate("d1", "dense_inventory", 30.0)
        dropped = _candidate("d2", "dense_inventory", 60.0)
        ranked = [_ranked(peak, 9.0), _ranked(dense, 8.0)]

        trace = candidate_recall_report.build_candidate_trace(
            stage_candidates={
                "peak": [peak],
                "semantic": [],
                "dense": [],
                "dense_inventory": [dense, dropped],
                "fallback": [],
                "merged": [peak, dense, dropped],
                "normalized": [peak, dense],
            },
            pre_judge_ranked=ranked,
            judged_ranked=ranked,
            quality_gate_rows=[
                {"candidate_id": "p1", "source": "peak", "accepted": True, "rank_score": 9.0},
                {
                    "candidate_id": "d1",
                    "source": "dense_inventory",
                    "accepted": False,
                    "rank_score": 8.0,
                    "rejection_reasons": ["payoff_strength_below_min"],
                },
            ],
            final_ranked=[ranked[0]],
            final_moments=[{"candidate_id": "p1::balanced"}],
            judge_top_k=1,
        )

        rows = {row["candidate_id"]: row for row in trace}
        self.assertTrue(rows["p1"]["final_selected"])
        self.assertEqual(rows["p1"]["status"], "final_selected")
        self.assertEqual(rows["d1"]["dropped_stage"], "quality_gate")
        self.assertEqual(rows["d1"]["drop_reason"], "payoff_strength_below_min")
        self.assertEqual(rows["d2"]["dropped_stage"], "normalization")
        self.assertEqual(rows["d2"]["drop_reason"], "not_normalized")

        summary = candidate_recall_report.build_candidate_funnel_summary(
            {"stage_counts": {"final_moments": 1}, "losses": [], "largest_losses": []},
            trace,
        )
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["final_selected_rows"], 1)
        self.assertEqual(summary["drop_stage_counts"]["quality_gate"], 1)
        self.assertEqual(summary["rejection_reason_counts"]["payoff_strength_below_min"], 1)
        self.assertEqual(summary["quality_breakdown"]["unjudged_kept"], 1)
        self.assertIn("ranker_lift", summary)
        self.assertIn("clip_type_lift", summary)

    def test_quality_split_counts_judged_and_unjudged_kept_separately(self) -> None:
        judged = _candidate("j1", "peak", 10.0)
        unjudged = _candidate("u1", "dense_inventory", 40.0)
        ranked = [_ranked(judged, 8.0), _ranked(unjudged, 7.0)]

        report = candidate_recall_report.build_candidate_recall_report(
            stage_candidates={
                "peak": [judged],
                "semantic": [],
                "dense": [],
                "dense_inventory": [unjudged],
                "fallback": [],
                "merged": [judged, unjudged],
                "normalized": [judged, unjudged],
            },
            pre_judge_ranked=ranked,
            judged_ranked=ranked,
            quality_gate_rows=[
                {
                    "candidate_id": "j1",
                    "source": "peak",
                    "accepted": True,
                    "has_judge_result": True,
                    "judge_score": 7.2,
                },
                {
                    "candidate_id": "u1",
                    "source": "dense_inventory",
                    "accepted": True,
                    "has_judge_result": False,
                },
            ],
            final_ranked=[ranked[0]],
            final_moments=[{"candidate_id": "j1"}],
            judge_top_k=1,
        )

        self.assertEqual(report["stage_counts"]["quality_kept"], 2)
        self.assertEqual(report["stage_counts"]["judged_quality_kept"], 1)
        self.assertEqual(report["stage_counts"]["unjudged_quality_kept"], 1)
        losses = {row["transition"]: row for row in report["losses"]}
        self.assertEqual(losses["judge_scope_to_judged_quality_kept"]["retention_ratio"], 1.0)
        self.assertLessEqual(losses["judged_quality_kept_to_final_ranked"]["retention_ratio"], 1.0)

    def test_judge_scope_can_use_recall_reserved_input_order(self) -> None:
        top = _candidate("top", "peak", 10.0)
        reserved = _candidate("reserved", "dense_inventory", 40.0)
        dropped = _candidate("dropped", "dense_inventory", 70.0)
        ranked = [_ranked(top, 9.0), _ranked(dropped, 8.0), _ranked(reserved, 7.0)]
        judge_scope_ranked = [_ranked(top, 9.0), _ranked(reserved, 7.0)]

        report = candidate_recall_report.build_candidate_recall_report(
            stage_candidates={
                "peak": [top],
                "semantic": [],
                "dense": [],
                "dense_inventory": [reserved, dropped],
                "fallback": [],
                "merged": [top, reserved, dropped],
                "normalized": [top, reserved, dropped],
            },
            pre_judge_ranked=ranked,
            judged_ranked=judge_scope_ranked,
            quality_gate_rows=[],
            final_ranked=[],
            final_moments=[],
            judge_top_k=2,
            judge_scope_ranked=judge_scope_ranked,
        )
        trace = candidate_recall_report.build_candidate_trace(
            stage_candidates={
                "peak": [top],
                "semantic": [],
                "dense": [],
                "dense_inventory": [reserved, dropped],
                "fallback": [],
                "merged": [top, reserved, dropped],
                "normalized": [top, reserved, dropped],
            },
            pre_judge_ranked=ranked,
            judged_ranked=judge_scope_ranked,
            quality_gate_rows=[],
            final_ranked=[],
            final_moments=[],
            judge_top_k=2,
            judge_scope_ranked=judge_scope_ranked,
        )

        self.assertEqual(report["stage_counts"]["judge_scope"], 2)
        losses = {row["transition"]: row for row in report["losses"]}
        self.assertEqual(losses["ranked_to_judge_scope"]["lost_count"], 1)
        self.assertEqual(report["ranker"]["recall_reserved"][0]["candidate_id"], "reserved")
        rows = {row["candidate_id"]: row for row in trace}
        self.assertEqual(rows["reserved"]["stages"]["judge_scope"]["status"], "included")
        self.assertTrue(rows["reserved"]["judge_recall_reserved"])
        self.assertNotIn("outside_judge_scope", rows["reserved"]["rejection_reasons"])
        self.assertEqual(rows["dropped"]["stages"]["judge_scope"]["status"], "excluded")

    def test_summary_reports_ranker_lift(self) -> None:
        trace = [
            {
                "candidate_id": "winner",
                "source": "dense_inventory",
                "pre_judge_rank": 24,
                "judge_score": 7.4,
                "final_selected": True,
                "stages": {},
            },
            {
                "candidate_id": "top-ranked",
                "source": "quote",
                "pre_judge_rank": 2,
                "judge_score": 4.0,
                "final_selected": False,
                "stages": {},
            },
        ]

        summary = candidate_recall_report.build_candidate_funnel_summary(
            {"stage_counts": {}, "losses": [], "largest_losses": []},
            trace,
        )

        self.assertEqual(summary["ranker_lift"]["high_judge_pre_judge_top20_ratio"], 0.0)
        self.assertEqual(summary["ranker_lift"]["final_selected_pre_judge_rank"]["max"], 24.0)


if __name__ == "__main__":
    unittest.main()
