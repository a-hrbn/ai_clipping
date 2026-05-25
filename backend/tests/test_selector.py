from unittest.mock import patch

from services import selector
from services.timeline_types import Candidate, CandidateFeatures, JudgeResult, RankedCandidate


def _ranked(
    candidate_id: str,
    *,
    rank_score: float = 7.0,
    judge_result: JudgeResult | None = None,
    rationale: list[str] | None = None,
) -> RankedCandidate:
    return RankedCandidate(
        candidate=Candidate(
            id=candidate_id,
            start=0.0,
            end=20.0,
            source="peak",
            anchor_time=10.0,
        ),
        features=CandidateFeatures(candidate_id=candidate_id, source="peak"),
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
         patch.object(selector.settings, "serving_quality_min_payoff_strength", 0.55):
        result = selector.apply_serving_quality_gate([strong, weak, unjudged])

    assert [item.candidate.id for item in result.kept] == ["strong", "unjudged"]
    weak_audit = next(row for row in result.audit_rows if row["candidate_id"] == "weak")
    assert weak_audit["accepted"] is False
    assert "judge_score_below_min" in weak_audit["rejection_reasons"]
    assert "payoff_strength_below_min" in weak_audit["rejection_reasons"]
    assert result.summary["rejected_candidates"] == 1
    assert result.summary["kept_without_judge_result"] == 1


def test_quality_gate_does_not_refill_from_rejected_judged_candidates() -> None:
    strong = _ranked("strong", judge_result=_judge(score=7.2))
    weak = _ranked("weak", judge_result=_judge(score=4.5))

    with patch.object(selector.settings, "serving_quality_gate_enabled", True), \
         patch.object(selector.settings, "serving_quality_min_judge_score", 6.0):
        result = selector.apply_serving_quality_gate([strong, weak])

    selected = selector.select_diverse_ranked_candidates(result.kept, top_k=5)

    assert [item.candidate.id for item in selected] == ["strong"]


def test_quality_gate_rejects_recall_filtered_candidates_without_judge() -> None:
    skipped = _ranked("skipped", judge_result=None, rationale=["judge_recall_filtered"])

    with patch.object(selector.settings, "serving_quality_gate_enabled", True):
        result = selector.apply_serving_quality_gate([skipped])

    assert result.kept == []
    assert result.audit_rows[0]["rejection_reasons"] == ["judge_recall_filtered"]

