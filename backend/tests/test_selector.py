from unittest.mock import patch

from services import selector
from services.timeline_types import Candidate, CandidateFeatures, JudgeResult, RankedCandidate


def _ranked(
    candidate_id: str,
    *,
    rank_score: float = 7.0,
    judge_result: JudgeResult | None = None,
    rationale: list[str] | None = None,
    start: float = 0.0,
    end: float = 20.0,
    source: str = "peak",
) -> RankedCandidate:
    return RankedCandidate(
        candidate=Candidate(
            id=candidate_id,
            start=start,
            end=end,
            source=source,
            anchor_time=(start + end) / 2.0,
        ),
        features=CandidateFeatures(candidate_id=candidate_id, source=source),
        rank_score=rank_score,
        rationale=list(rationale or []),
        judge_result=judge_result,
    )


def _judge(
    *,
    score: float,
    clarity: float = 0.8,
    payoff: float = 0.8,
    narrative: float = 0.8,
    confidence: float = 0.8,
) -> JudgeResult:
    return JudgeResult(
        judge_score=score,
        scroll_stop_power=0.8,
        standalone_clarity=clarity,
        payoff_strength=payoff,
        narrative_completeness=narrative,
        platform_fit_shortform=0.8,
        judge_reason="test",
        judge_confidence=confidence,
        provider="gemini",
        llm_used=True,
    )


def test_quality_gate_rejects_judged_low_quality_candidates() -> None:
    strong = _ranked("strong", judge_result=_judge(score=7.2))
    weak = _ranked("weak", judge_result=_judge(score=5.1, payoff=0.3))
    unjudged = _ranked("unjudged", judge_result=None)

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_min_judge_score", 6.0), \
         patch.object(selector.settings, "serving_quality_min_payoff_strength", 0.55), \
         patch.object(selector.settings, "serving_quality_subscore_gate_mode", "hard"):
        result = selector.apply_serving_quality_gate([strong, weak, unjudged])

    assert [item.candidate.id for item in result.kept] == ["strong", "unjudged"]
    weak_audit = next(row for row in result.audit_rows if row["candidate_id"] == "weak")
    assert weak_audit["accepted"] is False
    assert "high_recall_protection" in weak_audit
    assert "score" in weak_audit["high_recall_protection"]
    assert "judge_score_below_min" in weak_audit["rejection_reasons"]
    assert "payoff_strength_below_min" in weak_audit["rejection_reasons"]
    assert result.summary["rejected_candidates"] == 1
    assert result.summary["kept_without_judge_result"] == 1


def test_quality_gate_does_not_refill_from_rejected_judged_candidates() -> None:
    strong = _ranked("strong", judge_result=_judge(score=7.2))
    weak = _ranked("weak", judge_result=_judge(score=4.5))

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_min_judge_score", 6.0), \
         patch.object(selector.settings, "serving_quality_subscore_gate_mode", "hard"):
        result = selector.apply_serving_quality_gate([strong, weak])

    selected = selector.select_diverse_ranked_candidates(result.kept, top_k=5)

    assert [item.candidate.id for item in selected] == ["strong"]


def test_quality_gate_rejects_recall_filtered_candidates_without_judge() -> None:
    skipped = _ranked("skipped", judge_result=None, rationale=["judge_recall_filtered"])

    with patch.object(selector.settings, "serving_quality_gate_enabled", True):
        result = selector.apply_serving_quality_gate([skipped])

    assert result.kept == []
    assert result.audit_rows[0]["rejection_reasons"] == ["judge_recall_filtered"]


def test_quality_gate_keeps_verified_cross_platform_recall_target_when_judge_underrates_it() -> None:
    target = _ranked(
        "cross-platform-target",
        judge_result=_judge(score=3.8, clarity=0.4, payoff=0.2, narrative=0.3, confidence=0.8),
        rationale=["cross_platform_recall_target"],
    )

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_min_judge_score", 6.0), \
         patch.object(selector.settings, "serving_cross_platform_quality_min_judge_score", 3.0):
        result = selector.apply_serving_quality_gate([target])

    assert [item.candidate.id for item in result.kept] == ["cross-platform-target"]
    assert result.audit_rows[0]["accepted"] is True
    assert result.audit_rows[0]["cross_platform_recall_target"] is True


def test_quality_gate_keeps_calibrated_jynxzi_reaction_scores() -> None:
    reaction = _ranked(
        "jynxzi-reaction",
        judge_result=_judge(score=4.48, clarity=0.4, payoff=0.4, narrative=0.5, confidence=0.8),
    )

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_min_judge_score", 4.4), \
         patch.object(selector.settings, "serving_quality_min_standalone_clarity", 0.35), \
         patch.object(selector.settings, "serving_quality_min_payoff_strength", 0.35):
        result = selector.apply_serving_quality_gate([reaction])

    assert [item.candidate.id for item in result.kept] == ["jynxzi-reaction"]
    assert result.audit_rows[0]["accepted"] is True


def test_quality_gate_allows_organic_ranker_threshold_pass_without_cross_platform_target() -> None:
    ranked = _ranked(
        "ranker-pass",
        rank_score=5.1,
        judge_result=_judge(score=5.1, clarity=0.3, payoff=0.3, narrative=0.5, confidence=0.8),
        rationale=["ranker_serving_threshold_pass"],
    )
    ranked.base_rank_score = 8.8

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_ranker_bypass_enabled", True), \
         patch.object(selector.settings, "serving_quality_ranker_bypass_min_base_score", 7.5):
        result = selector.apply_serving_quality_gate([ranked])

    assert [item.candidate.id for item in result.kept] == ["ranker-pass"]
    assert result.audit_rows[0]["accepted"] is True
    assert result.audit_rows[0]["ranker_quality_bypass"] is True
    assert result.audit_rows[0]["cross_platform_recall_target"] is False


def test_rank_candidates_marks_near_serving_threshold_for_ranker_bypass() -> None:
    class FakeRanker:
        def score_details(self, features, profile, strategy=None):
            return [
                {
                    "candidate_id": "near",
                    "score": 7.627,
                    "heuristic_score": 7.1,
                    "trained_score": 7.9,
                    "blend": 0.65,
                    "serving_threshold": 7.637,
                    "passes_serving_threshold": False,
                    "trained_model_used": True,
                    "rationale": ["ml_ranked"],
                }
            ]

    candidate = Candidate(id="near", start=0.0, end=20.0, source="peak", anchor_time=10.0)
    feature = CandidateFeatures(candidate_id="near", source="peak")

    with patch.object(selector, "Ranker", return_value=FakeRanker()), \
         patch.object(selector.settings, "serving_quality_ranker_bypass_threshold_margin", 0.15), \
         patch.object(selector.settings, "serving_quality_ranker_bypass_threshold_ratio", 0.98):
        ranked = selector.rank_candidates([candidate], [feature], profile="jynxzi")

    assert ranked[0].candidate.id == "near"
    assert "ranker_serving_threshold_near_pass" in ranked[0].rationale


def test_quality_gate_allows_near_threshold_ranker_bypass() -> None:
    ranked = _ranked(
        "near-ranker-pass",
        rank_score=7.627,
        judge_result=_judge(score=5.8, clarity=0.3, payoff=0.3, narrative=0.4, confidence=0.9),
        rationale=["ranker_serving_threshold_near_pass"],
    )

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_ranker_bypass_enabled", True):
        result = selector.apply_serving_quality_gate([ranked])

    assert [item.candidate.id for item in result.kept] == ["near-ranker-pass"]
    assert result.audit_rows[0]["accepted"] is True
    assert result.audit_rows[0]["ranker_quality_bypass"] is True
    assert result.audit_rows[0]["ranker_quality_bypass_signal"] is True


def test_quality_gate_soft_accepts_single_narrative_miss_for_strong_judge() -> None:
    candidate = _ranked(
        "single-narrative-miss",
        judge_result=_judge(score=7.62, clarity=0.7, payoff=0.9, narrative=0.4, confidence=0.9),
    )

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_min_judge_score", 4.4), \
         patch.object(selector.settings, "serving_quality_min_standalone_clarity", 0.35), \
         patch.object(selector.settings, "serving_quality_min_payoff_strength", 0.35), \
         patch.object(selector.settings, "serving_quality_min_narrative_completeness", 0.45), \
         patch.object(selector.settings, "serving_quality_soft_single_subscore_miss_enabled", True), \
         patch.object(selector.settings, "serving_quality_soft_single_subscore_min_judge_score", 6.0), \
         patch.object(selector.settings, "serving_quality_soft_single_subscore_min_axis_mean", 0.55):
        result = selector.apply_serving_quality_gate([candidate])

    assert [item.candidate.id for item in result.kept] == ["single-narrative-miss"]
    assert result.audit_rows[0]["accepted"] is True
    assert result.audit_rows[0]["soft_judge_quality_gate"] is True


def test_quality_gate_soft_mode_uses_judge_score_floor_not_subscore_cutoffs() -> None:
    noisy = _ranked(
        "noisy-but-not-dead",
        judge_result=_judge(score=4.9, clarity=0.2, payoff=0.25, narrative=0.2, confidence=0.8),
    )
    dead = _ranked(
        "dead",
        judge_result=_judge(score=2.8, clarity=0.8, payoff=0.8, narrative=0.8, confidence=0.8),
    )

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_subscore_gate_mode", "soft"), \
         patch.object(selector.settings, "serving_quality_hard_min_judge_score", 3.0), \
         patch.object(selector.settings, "serving_quality_min_standalone_clarity", 0.35), \
         patch.object(selector.settings, "serving_quality_min_payoff_strength", 0.35), \
         patch.object(selector.settings, "serving_quality_min_narrative_completeness", 0.45):
        result = selector.apply_serving_quality_gate([noisy, dead])

    assert [item.candidate.id for item in result.kept] == ["noisy-but-not-dead"]
    assert result.audit_rows[0]["rejection_reasons"] == []
    assert result.audit_rows[0]["serving_quality_subscore_gate_mode"] == "soft"
    assert result.audit_rows[1]["rejection_reasons"] == ["judge_score_below_hard_min"]


def test_post_judge_final_selection_prioritizes_judged_candidates_over_unjudged_fillers() -> None:
    judged_a = _ranked(
        "judged-a",
        rank_score=6.0,
        judge_result=_judge(score=7.0, confidence=0.8),
        start=0.0,
        end=20.0,
        source="dense_inventory",
    )
    judged_b = _ranked(
        "judged-b",
        rank_score=5.0,
        judge_result=_judge(score=6.8, confidence=0.8),
        start=80.0,
        end=100.0,
        source="dense_inventory",
    )
    unjudged = _ranked(
        "unjudged-high-ranker",
        rank_score=99.0,
        judge_result=None,
        start=160.0,
        end=180.0,
        source="peak",
    )

    result = selector.select_post_judge_final_candidates(
        [unjudged, judged_b, judged_a],
        top_k=2,
    )

    assert [item.candidate.id for item in result.selected] == ["judged-a", "judged-b"]
    assert result.summary["selected_judged_candidates"] == 2
    assert result.summary["selected_unjudged_candidates"] == 0
    assert result.summary["source_cap_enabled"] is False
    assert result.summary["unjudged_fallback_enabled"] is False


def test_post_judge_final_selection_sorts_by_blended_rank_score_before_judge_score() -> None:
    ranker_favored = _ranked(
        "ranker-favored",
        rank_score=8.0,
        judge_result=_judge(score=6.8, confidence=0.8),
        start=0.0,
        end=20.0,
    )
    judge_favored = _ranked(
        "judge-favored",
        rank_score=6.2,
        judge_result=_judge(score=8.4, confidence=0.8),
        start=80.0,
        end=100.0,
    )

    result = selector.select_post_judge_final_candidates(
        [judge_favored, ranker_favored],
        top_k=2,
    )

    assert [item.candidate.id for item in result.selected] == ["ranker-favored", "judge-favored"]
    assert result.summary["policy"] == "blended_ranker_judge_temporal_diversity"


def test_post_judge_final_selection_uses_ranker_bypass_unjudged_candidates() -> None:
    judged = _ranked(
        "judged",
        rank_score=7.2,
        judge_result=_judge(score=7.2, confidence=0.8),
        start=0.0,
        end=20.0,
        source="dense_inventory",
    )
    unjudged_ranker = _ranked(
        "unjudged-ranker",
        rank_score=7.6,
        judge_result=None,
        rationale=["ranker_serving_threshold_near_pass"],
        start=120.0,
        end=140.0,
        source="peak",
    )
    unjudged_filler = _ranked(
        "unjudged-filler",
        rank_score=99.0,
        judge_result=None,
        start=240.0,
        end=260.0,
        source="peak",
    )

    with patch.object(selector.settings, "serving_quality_ranker_bypass_enabled", True):
        result = selector.select_post_judge_final_candidates(
            [unjudged_filler, unjudged_ranker, judged],
            top_k=2,
        )

    assert [item.candidate.id for item in result.selected] == ["judged", "unjudged-ranker"]
    assert result.summary["selected_unjudged_candidates"] == 1
    assert result.summary["ranker_bypass_unjudged_candidates"] == 1
    assert result.summary["ranker_bypass_unjudged_selected"] == 1
    assert "post_judge_final_ranker_bypass" in result.selected[1].rationale


def test_post_judge_final_selection_returns_fewer_clips_instead_of_unjudged_fillers() -> None:
    judged = _ranked(
        "judged",
        rank_score=4.0,
        judge_result=_judge(score=7.2, confidence=0.8),
        start=0.0,
        end=20.0,
        source="dense_inventory",
    )
    unjudged_a = _ranked(
        "unjudged-a",
        rank_score=99.0,
        judge_result=None,
        start=80.0,
        end=100.0,
        source="peak",
    )
    unjudged_b = _ranked(
        "unjudged-b",
        rank_score=98.0,
        judge_result=None,
        start=160.0,
        end=180.0,
        source="dense_window",
    )

    result = selector.select_post_judge_final_candidates(
        [unjudged_a, unjudged_b, judged],
        top_k=3,
    )

    assert [item.candidate.id for item in result.selected] == ["judged"]
    assert result.summary["selected_candidates"] == 1
    assert result.summary["selected_judged_candidates"] == 1
    assert result.summary["selected_unjudged_candidates"] == 0
    assert result.summary["fallback_unjudged_candidates"] == 0
    assert result.summary["unjudged_fallback_enabled"] is False
    assert result.summary["missing_final_slots"] == 2
    assert all("post_judge_final_fallback_unjudged" not in item.rationale for item in [unjudged_a, unjudged_b])


def test_post_judge_final_selection_does_not_apply_source_cap_after_judge() -> None:
    ranked = [
        _ranked(
            f"dense-{index}",
            rank_score=6.0 - index * 0.01,
            judge_result=_judge(score=7.5 - index * 0.1, confidence=0.8),
            start=index * 60.0,
            end=index * 60.0 + 20.0,
            source="dense_inventory",
        )
        for index in range(4)
    ]

    result = selector.select_post_judge_final_candidates(ranked, top_k=4)

    assert [item.candidate.id for item in result.selected] == [
        "dense-0",
        "dense-1",
        "dense-2",
        "dense-3",
    ]
    assert result.summary["selected_judged_candidates"] == 4


def test_post_judge_final_selection_drops_overlapping_lower_judge_candidate() -> None:
    winner = _ranked(
        "winner",
        judge_result=_judge(score=7.4, confidence=0.8),
        start=100.0,
        end=130.0,
    )
    overlap = _ranked(
        "overlap",
        judge_result=_judge(score=7.2, confidence=0.8),
        start=120.0,
        end=150.0,
    )
    far = _ranked(
        "far",
        judge_result=_judge(score=7.0, confidence=0.8),
        start=220.0,
        end=240.0,
    )

    result = selector.select_post_judge_final_candidates(
        [overlap, far, winner],
        top_k=3,
        min_gap_sec=20.0,
    )

    assert [item.candidate.id for item in result.selected] == ["winner", "far"]
    assert result.summary["dropped_judged_samples"][0]["candidate_id"] == "overlap"
