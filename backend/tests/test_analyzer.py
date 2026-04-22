import json
import unittest
import sys
import types
from unittest.mock import patch

sys.modules.setdefault("ollama", types.SimpleNamespace(Client=object))
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import analyzer
from services.timeline_types import Candidate, CandidateFeatures, JudgeResult, RankedCandidate
from services.transcriber import RichSegment


class AnalyzerValidationTests(unittest.TestCase):
    def test_validate_snaps_times_to_known_boundaries(self) -> None:
        segments = [
            RichSegment(start=0.0, end=5.0, text="a"),
            RichSegment(start=5.0, end=10.0, text="b"),
        ]

        clips = analyzer._validate(
            {
                "clips": [
                    {
                        "rank": 1,
                        "start_time": 0.4,
                        "end_time": 9.6,
                        "viral_score": 11.5,
                        "reason": "peak",
                    }
                ]
            },
            segments,
        )

        self.assertIsNotNone(clips)
        self.assertEqual(clips[0]["start_time"], 0.0)
        self.assertEqual(clips[0]["end_time"], 10.0)
        self.assertEqual(clips[0]["viral_score"], 10.0)

    def test_validate_rejects_invalid_ranges(self) -> None:
        segments = [RichSegment(start=10.0, end=20.0, text="a")]

        clips = analyzer._validate(
            {
                "clips": [
                    {
                        "rank": 1,
                        "start_time": 25.0,
                        "end_time": 26.0,
                        "viral_score": 6.0,
                        "reason": "bad",
                    }
                ]
            },
            segments,
        )

        self.assertIsNone(clips)

    def test_enforce_story_shape_expands_peak_only_clip(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup."),
            RichSegment(start=2.0, end=4.0, text="Something is happening."),
            RichSegment(start=4.0, end=5.0, text="No way!", emotions=["exclamation"]),
            RichSegment(start=5.0, end=8.0, text="That actually worked."),
            RichSegment(start=8.0, end=12.0, text="Everyone laughs.", emotions=["laughter"]),
        ]
        candidate = Candidate(
            id="c1",
            start=0.0,
            end=12.0,
            source="reaction",
            anchor_time=4.5,
            score_hint=5.0,
            tags=[],
        )

        moment = analyzer._enforce_story_shape(
            {"start_time": 4.0, "end_time": 5.0, "viral_score": 8.0, "reason": "peak"},
            candidate,
            segments,
        )

        self.assertLessEqual(moment["start_time"], 2.0)
        self.assertGreaterEqual(moment["end_time"], 8.0)
        self.assertGreaterEqual(moment["end_time"] - moment["start_time"], 8.0)

    def test_refine_candidates_v2_prefers_clean_open_and_complete_payoff(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="and then he says", speaker="STREAMER", hype_score=0.8),
            RichSegment(start=2.0, end=4.0, text="Wait what happened?", speaker="STREAMER", emotions=["question"], hype_score=2.8),
            RichSegment(start=4.0, end=6.0, text="No way!", speaker="STREAMER", emotions=["exclamation"], hype_score=5.6),
            RichSegment(start=6.0, end=10.0, text="Everyone bursts out laughing.", speaker="STREAMER", emotions=["laughter"], hype_score=4.9),
        ]
        ranked = RankedCandidate(
            candidate=Candidate(
                id="c1",
                start=0.0,
                end=10.0,
                source="reaction",
                anchor_time=5.0,
                score_hint=5.0,
                tags=[],
            ),
            features=None,
            rank_score=6.0,
            rationale=["strong_hook"],
        )

        with patch.object(analyzer.settings, "enable_boundary_refinement_v2", True), \
             patch.object(analyzer.settings, "enable_boundary_llm_pair_rerank", False):
            moments = analyzer.refine_candidates([ranked], segments, limit=1)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0]["start_time"], 2.0)
        self.assertEqual(moments[0]["end_time"], 10.0)
        self.assertIn("closure", moments[0]["reason"])

    def test_refine_candidates_v2_pair_rerank_does_not_call_legacy_timestamp_llm(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup.", speaker="STREAMER", hype_score=1.0),
            RichSegment(start=2.0, end=4.0, text="Wait what happened?", speaker="STREAMER", emotions=["question"], hype_score=3.0),
            RichSegment(start=4.0, end=6.0, text="No way!", speaker="STREAMER", emotions=["exclamation"], hype_score=5.8),
            RichSegment(start=6.0, end=10.0, text="Everyone bursts out laughing.", speaker="STREAMER", emotions=["laughter"], hype_score=4.9),
        ]
        ranked = RankedCandidate(
            candidate=Candidate(
                id="c2",
                start=0.0,
                end=10.0,
                source="reaction",
                anchor_time=5.0,
                score_hint=5.0,
                tags=[],
            ),
            features=None,
            rank_score=6.0,
            rationale=["strong_hook"],
        )

        with patch.object(analyzer.settings, "enable_boundary_refinement_v2", True), \
             patch.object(analyzer.settings, "enable_boundary_llm_pair_rerank", True), \
             patch.object(analyzer, "_call_ollama", side_effect=AssertionError("legacy timestamp llm should not be used")), \
             patch.object(
                 analyzer,
                 "_call_pair_rerank",
                 return_value={"start_time": 2.0, "end_time": 10.0, "viral_score": 8.5, "reason": "LLM selected pair."},
             ):
            moments = analyzer.refine_candidates([ranked], segments, limit=1)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0]["start_time"], 2.0)
        self.assertEqual(moments[0]["end_time"], 10.0)
        self.assertEqual(moments[0]["reason"], "LLM selected pair.")

    def test_refine_candidates_adds_structured_selection_reasons(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup.", speaker="STREAMER", hype_score=1.2),
            RichSegment(start=2.0, end=4.0, text="Wait what happened?", speaker="STREAMER", emotions=["question"], hype_score=3.2),
            RichSegment(start=4.0, end=6.0, text="No way!", speaker="STREAMER", emotions=["exclamation"], hype_score=5.8),
            RichSegment(start=6.0, end=10.0, text="Everyone bursts out laughing.", speaker="STREAMER", emotions=["laughter"], hype_score=5.0),
        ]
        ranked = RankedCandidate(
            candidate=Candidate(
                id="c3",
                start=0.0,
                end=10.0,
                source="reaction",
                anchor_time=5.0,
                score_hint=5.0,
                tags=[],
            ),
            features=CandidateFeatures(
                candidate_id="c3",
                hook_strength=0.82,
                payoff_score=5.4,
                standalone_score=0.72,
                context_efficiency=0.74,
                chat_peak=0.1,
                face_reaction_peak=0.52,
                source="reaction",
            ),
            rank_score=6.4,
            rationale=["strong_hook", "clear_payoff", "standalone"],
            judge_result=JudgeResult(
                judge_score=8.1,
                scroll_stop_power=0.88,
                standalone_clarity=0.79,
                payoff_strength=0.86,
                narrative_completeness=0.7,
                platform_fit_shortform=0.73,
                judge_reason="Strong short-form candidate.",
                judge_confidence=0.72,
            ),
        )

        with patch.object(analyzer.settings, "enable_boundary_refinement_v2", True), \
             patch.object(analyzer.settings, "enable_boundary_llm_pair_rerank", False):
            moments = analyzer.refine_candidates([ranked], segments, limit=1)

        self.assertEqual(len(moments), 1)
        selection_reasons = moments[0]["selection_reasons"]
        self.assertGreaterEqual(len(selection_reasons), 3)
        self.assertTrue(any("hook" in reason.lower() for reason in selection_reasons))
        self.assertTrue(any("Judge-Check" in reason for reason in selection_reasons))
        self.assertTrue(any("Clip startet" in reason or "Payoff" in reason for reason in selection_reasons))

    def test_refine_candidates_legacy_timestamp_llm_requires_explicit_flag(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup.", hype_score=1.0),
            RichSegment(start=2.0, end=5.0, text="Hook!", emotions=["exclamation"], hype_score=4.0),
            RichSegment(start=5.0, end=9.0, text="Payoff.", emotions=["laughter"], hype_score=4.8),
        ]
        ranked = RankedCandidate(
            candidate=Candidate(
                id="legacy",
                start=0.0,
                end=9.0,
                source="reaction",
                anchor_time=3.5,
                score_hint=4.0,
                tags=[],
            ),
            features=None,
            rank_score=5.0,
            rationale=["hook"],
        )

        with patch.object(analyzer.settings, "enable_boundary_refinement_v2", False), \
             patch.object(analyzer.settings, "enable_legacy_boundary_refinement_llm", False), \
             patch.object(analyzer, "_call_ollama", side_effect=AssertionError("legacy path should stay disabled")):
            moments = analyzer.refine_candidates([ranked], segments, limit=1)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0]["start_time"], 0.0)
        self.assertEqual(moments[0]["end_time"], 9.0)
        self.assertIn("Selected from", moments[0]["reason"])

    def test_expand_variants_emits_multiple_platform_aware_versions(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", hype_score=1.2),
            RichSegment(start=2.0, end=4.0, text="wait what happened", emotions=["question"], hype_score=3.4),
            RichSegment(start=4.0, end=6.0, text="no way", emotions=["exclamation"], hype_score=5.8),
            RichSegment(start=6.0, end=10.0, text="everyone starts laughing", emotions=["laughter"], hype_score=5.0),
        ]
        ranked = RankedCandidate(
            candidate=Candidate(
                id="variant",
                start=0.0,
                end=10.0,
                source="reaction",
                anchor_time=5.0,
                score_hint=5.0,
                tags=[],
            ),
            features=CandidateFeatures(
                candidate_id="variant",
                hook_strength=0.78,
                context_efficiency=0.66,
                projected_hook_score=0.74,
                projected_retention_score=0.69,
                source="reaction",
            ),
            rank_score=6.5,
            rationale=["strong_hook", "projected_retention"],
        )
        moments = [
            {
                "start_time": 2.0,
                "end_time": 10.0,
                "viral_score": 8.1,
                "reason": "reaction moment",
                "rank": 1,
                "candidate_id": "variant",
                "platform": "tiktok",
                "variant_key": "aggressive_hook",
                "variant_label": "Aggressive Hook · Tiktok",
                "projected_hook_score": 0.74,
                "projected_retention_score": 0.69,
            }
        ]

        expanded = analyzer.expand_variants(
            moments,
            [ranked],
            segments,
            strategy={"platform": "tiktok", "goal": "funny reaction", "clip_style": "aggressive_hook", "variant_count": 3},
        )

        self.assertEqual(len(expanded), 3)
        self.assertEqual({moment["variant_key"] for moment in expanded}, {"reaction_first", "balanced", "aggressive_hook"})
        self.assertTrue(all("Tiktok" in moment["variant_label"] for moment in expanded))
        self.assertTrue(any(moment["projected_hook_score"] > 0.74 for moment in expanded))

    def test_enforce_story_shape_respects_minimum_required_setup_start(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup starts."),
            RichSegment(start=2.0, end=4.0, text="The bet is on."),
            RichSegment(start=4.0, end=6.0, text="Still building."),
            RichSegment(start=6.0, end=8.0, text="No way!", emotions=["exclamation"]),
            RichSegment(start=8.0, end=12.0, text="Everyone laughs.", emotions=["laughter"]),
        ]
        candidate = Candidate(
            id="memory-shape",
            start=0.0,
            end=12.0,
            source="story_payoff",
            anchor_time=7.0,
            score_hint=6.0,
            tags=[],
        )

        moment = analyzer._enforce_story_shape(
            {
                "start_time": 6.0,
                "end_time": 12.0,
                "viral_score": 8.0,
                "reason": "payoff only",
                "minimum_required_setup_start": 2.0,
            },
            candidate,
            segments,
        )

        self.assertLessEqual(moment["start_time"], 2.0)


class AnalyzerHelperTests(unittest.TestCase):
    def test_snap_time_returns_nearest_boundary_within_tolerance(self) -> None:
        boundaries = [0.0, 2.5, 5.0, 10.0]
        self.assertEqual(analyzer._snap_time(2.6, boundaries), 2.5)
        self.assertEqual(analyzer._snap_time(5.3, boundaries), 5.0)

    def test_snap_time_raises_when_outside_tolerance(self) -> None:
        with self.assertRaises(KeyError):
            analyzer._snap_time(20.0, [0.0, 5.0], tolerance=3.0)

    def test_allowed_boundaries_dedupes_and_sorts(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="a"),
            RichSegment(start=2.0, end=5.0, text="b"),
            RichSegment(start=5.0, end=9.0, text="c"),
        ]
        self.assertEqual(analyzer._allowed_boundaries(segments), [0.0, 2.0, 5.0, 9.0])

    def test_append_unique_reason_dedupes_case_insensitively(self) -> None:
        reasons: list[str] = []
        analyzer._append_unique_reason(reasons, "Starker Hook")
        analyzer._append_unique_reason(reasons, "starker hook")
        analyzer._append_unique_reason(reasons, "")
        analyzer._append_unique_reason(reasons, "   ")
        analyzer._append_unique_reason(reasons, "Zweiter Grund")
        self.assertEqual(reasons, ["Starker Hook", "Zweiter Grund"])


class AnalyzerPairSelectionTests(unittest.TestCase):
    def _pair(self, pair_id: str = "pair-1"):
        from services.timeline_types import BoundaryPair

        return BoundaryPair(
            id=pair_id,
            candidate_id="cand",
            start_boundary_id="b0",
            end_boundary_id="b1",
            start_time=2.0,
            end_time=10.0,
            duration=8.0,
            pair_score=0.8,
            start_score=0.7,
            end_score=0.9,
            hook_score=0.7,
            payoff_score=0.8,
            narrative_score=0.75,
            reason="clean sentence start; complete payoff",
        )

    def test_returns_payload_for_known_pair(self) -> None:
        pair = self._pair()
        content = json.dumps({"selected_pair_id": "pair-1", "viral_score": 8.5, "reason": "great beat"})
        result = analyzer._parse_pair_selection(content, [pair])
        self.assertIsNotNone(result)
        self.assertEqual(result["start_time"], 2.0)
        self.assertEqual(result["end_time"], 10.0)
        self.assertEqual(result["viral_score"], 8.5)
        self.assertEqual(result["reason"], "great beat")
        self.assertEqual(result["selection_boundary_reason"], pair.reason)

    def test_embedded_json_is_extracted(self) -> None:
        pair = self._pair()
        content = "Here is the answer:\n" + json.dumps({"selected_pair_id": "pair-1", "viral_score": 7.0, "reason": "x"}) + "\nthanks"
        result = analyzer._parse_pair_selection(content, [pair])
        self.assertIsNotNone(result)
        self.assertEqual(result["viral_score"], 7.0)

    def test_malformed_json_returns_none(self) -> None:
        self.assertIsNone(analyzer._parse_pair_selection("not json", [self._pair()]))

    def test_unknown_pair_id_returns_none(self) -> None:
        pair = self._pair()
        content = json.dumps({"selected_pair_id": "missing", "viral_score": 5.0, "reason": "x"})
        self.assertIsNone(analyzer._parse_pair_selection(content, [pair]))

    def test_invalid_viral_score_falls_back_to_pair_score(self) -> None:
        pair = self._pair()
        content = json.dumps({"selected_pair_id": "pair-1", "viral_score": "bogus", "reason": "x"})
        result = analyzer._parse_pair_selection(content, [pair])
        self.assertIsNotNone(result)
        self.assertEqual(result["viral_score"], round(pair.pair_score * 10.0, 2))

    def test_viral_score_is_clamped(self) -> None:
        pair = self._pair()
        content = json.dumps({"selected_pair_id": "pair-1", "viral_score": 100.0, "reason": "x"})
        result = analyzer._parse_pair_selection(content, [pair])
        self.assertEqual(result["viral_score"], 10.0)


class AnalyzerBoundaryReasonTests(unittest.TestCase):
    def test_none_reason_returns_none(self) -> None:
        self.assertIsNone(analyzer._boundary_selection_reason(None))
        self.assertIsNone(analyzer._boundary_selection_reason(""))

    def test_maps_known_start_and_end_reasons(self) -> None:
        text = analyzer._boundary_selection_reason("clean sentence start; complete payoff")
        self.assertIsNotNone(text)
        self.assertIn("sauberen Satzanfang", text)
        self.assertIn("Payoff", text)

    def test_unknown_reason_falls_back_to_verbatim_note(self) -> None:
        text = analyzer._boundary_selection_reason("mystery reason; penalties: x")
        self.assertIsNotNone(text)
        self.assertIn("Hinweise:", text)


class AnalyzerDedupMomentsTests(unittest.TestCase):
    def _moment(self, start: float, end: float, score: float) -> dict:
        return {
            "start_time": start,
            "end_time": end,
            "viral_score": score,
            "rank": 1,
            "reason": "r",
        }

    def test_keeps_disjoint_moments_and_reassigns_rank(self) -> None:
        moments = [
            self._moment(0.0, 4.0, 6.0),
            self._moment(10.0, 15.0, 8.0),
        ]
        kept = analyzer._dedup_moments(moments)
        self.assertEqual(len(kept), 2)
        # Higher score should end up ranked 1.
        self.assertEqual(kept[0]["viral_score"], 8.0)
        self.assertEqual(kept[0]["rank"], 1)
        self.assertEqual(kept[1]["rank"], 2)

    def test_drops_duplicate_with_high_overlap(self) -> None:
        moments = [
            self._moment(0.0, 10.0, 6.0),
            self._moment(1.0, 9.5, 8.0),  # fully overlapping a shorter window
        ]
        kept = analyzer._dedup_moments(moments)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["viral_score"], 8.0)

    def test_drops_moments_with_nonpositive_duration(self) -> None:
        kept = analyzer._dedup_moments([self._moment(5.0, 5.0, 9.0)])
        self.assertEqual(kept, [])


class AnalyzerApplyVariantSpecTests(unittest.TestCase):
    def _candidate(self) -> Candidate:
        return Candidate(
            id="variant",
            start=0.0,
            end=12.0,
            source="reaction",
            anchor_time=6.0,
            score_hint=5.0,
            tags=[],
        )

    def test_returns_unchanged_when_no_boundaries(self) -> None:
        moment = {"start_time": 2.0, "end_time": 10.0, "viral_score": 7.0, "reason": "r"}
        result = analyzer._apply_variant_spec(moment, self._candidate(), [], {"key": "balanced"})
        self.assertEqual(result["start_time"], 2.0)
        self.assertEqual(result["end_time"], 10.0)

    def test_aggressive_hook_trims_left_context(self) -> None:
        boundaries = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        moment = {"start_time": 2.0, "end_time": 10.0, "viral_score": 7.0, "reason": "r"}
        result = analyzer._apply_variant_spec(
            moment,
            self._candidate(),
            boundaries,
            {"key": "aggressive_hook", "start_bias_sec": 2.0, "end_bias_sec": 0.0, "context_multiplier": 0.5},
        )
        self.assertGreaterEqual(result["start_time"], 2.0)
        self.assertLessEqual(result["start_time"], boundaries[-1])
        self.assertGreater(result["end_time"], result["start_time"])

    def test_context_balanced_pulls_left_boundary_earlier(self) -> None:
        boundaries = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        moment = {"start_time": 4.0, "end_time": 8.0, "viral_score": 7.0, "reason": "r"}
        result = analyzer._apply_variant_spec(
            moment,
            self._candidate(),
            boundaries,
            {"key": "context_balanced", "start_bias_sec": 0.0, "end_bias_sec": 0.0, "context_multiplier": 2.0},
        )
        self.assertLessEqual(result["start_time"], 4.0)
        self.assertGreaterEqual(result["end_time"], 8.0)


if __name__ == "__main__":
    unittest.main()
