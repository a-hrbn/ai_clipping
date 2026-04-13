import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())
sys.modules.setdefault("ollama", types.SimpleNamespace(Client=lambda *_args, **_kwargs: None))
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import clip_judge
from services.keyframe_sampler import SampledKeyframe
from services.timeline_types import Candidate, CandidateFeatures, RankedCandidate
from services.transcriber import RichSegment


class ClipJudgeTests(unittest.TestCase):
    def test_heuristic_judge_scores_strong_candidate_higher(self) -> None:
        strong = RankedCandidate(
            candidate=Candidate(id="strong", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
            features=CandidateFeatures(
                candidate_id="strong",
                hook_strength=0.9,
                max_hype=7.0,
                chat_peak=0.7,
                face_reaction_peak=0.8,
                payoff_score=7.0,
                context_efficiency=0.85,
                standalone_score=0.8,
                arc_score=4.0,
                profile_fit_score=0.85,
                post_hook_hype=4.0,
                source="reaction",
            ),
            rank_score=7.0,
        )
        weak = RankedCandidate(
            candidate=Candidate(id="weak", start=30.0, end=75.0, source="quote", anchor_time=40.0),
            features=CandidateFeatures(
                candidate_id="weak",
                hook_strength=0.1,
                max_hype=2.0,
                chat_peak=0.0,
                payoff_score=1.0,
                context_efficiency=0.2,
                standalone_score=0.2,
                arc_score=1.0,
                profile_fit_score=0.2,
                source="quote",
            ),
            rank_score=5.0,
        )
        segments = [
            RichSegment(start=0.0, end=2.0, text="what is happening?", emotions=["question"], hype_score=2.5),
            RichSegment(start=2.0, end=5.0, text="no way!", emotions=["exclamation", "laughter"], hype_score=6.5),
            RichSegment(start=5.0, end=9.0, text="everyone reacts", emotions=["laughter"], hype_score=5.0),
        ]

        with patch("services.clip_judge._call_llm_judge", return_value=None):
            strong_result = clip_judge.judge_candidate(strong, segments)
            weak_result = clip_judge.judge_candidate(weak, segments)

        self.assertGreater(strong_result.judge_score, weak_result.judge_score)
        self.assertFalse(strong_result.llm_used)

    def test_rerank_candidates_attaches_judge_metadata(self) -> None:
        ranked = [
            RankedCandidate(
                candidate=Candidate(id="a", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
                features=CandidateFeatures(candidate_id="a", hook_strength=0.9, payoff_score=7.0, standalone_score=0.8, context_efficiency=0.8, profile_fit_score=0.8, source="reaction"),
                rank_score=6.0,
                rationale=["baseline"],
            ),
            RankedCandidate(
                candidate=Candidate(id="b", start=30.0, end=48.0, source="quote", anchor_time=33.0),
                features=CandidateFeatures(candidate_id="b", hook_strength=0.4, payoff_score=3.0, standalone_score=0.4, context_efficiency=0.4, profile_fit_score=0.4, source="quote"),
                rank_score=5.5,
                rationale=["baseline"],
            ),
        ]
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", hype_score=1.5),
            RichSegment(start=2.0, end=5.0, text="payoff", emotions=["exclamation"], hype_score=5.0),
        ]

        with patch("services.clip_judge._call_llm_judge", return_value=None):
            reranked = clip_judge.rerank_candidates(ranked, segments, shortlist_top_k=1)

        judged = next(item for item in reranked if item.candidate.id == "a")
        untouched = next(item for item in reranked if item.candidate.id == "b")
        self.assertIsNotNone(judged.judge_result)
        self.assertEqual(judged.base_rank_score, 6.0)
        self.assertIn("judge_reranked", judged.rationale)
        self.assertIsNone(untouched.judge_result)

    def test_llm_result_is_used_when_available(self) -> None:
        ranked = RankedCandidate(
            candidate=Candidate(id="a", start=0.0, end=15.0, source="reaction", anchor_time=3.0),
            features=CandidateFeatures(candidate_id="a", hook_strength=0.8, payoff_score=6.0, standalone_score=0.7, context_efficiency=0.7, profile_fit_score=0.7, source="reaction"),
            rank_score=6.0,
        )
        segments = [RichSegment(start=0.0, end=2.0, text="hook", hype_score=3.0)]

        with patch("services.clip_judge._call_llm_judge") as llm_mock:
            llm_mock.return_value = clip_judge.JudgeResult(
                judge_score=8.5,
                scroll_stop_power=0.9,
                standalone_clarity=0.8,
                payoff_strength=0.8,
                narrative_completeness=0.7,
                platform_fit_shortform=0.8,
                judge_reason="LLM approved.",
                judge_confidence=0.75,
                provider="gemini",
                llm_used=True,
            )
            result = clip_judge.judge_candidate(ranked, segments, profile="general")

        self.assertTrue(result.llm_used)
        self.assertEqual(result.provider, "gemini")

    def test_multimodal_gemini_path_is_preferred_when_keyframes_exist(self) -> None:
        ranked = RankedCandidate(
            candidate=Candidate(id="a", start=0.0, end=15.0, source="reaction", anchor_time=3.0),
            features=CandidateFeatures(candidate_id="a", hook_strength=0.8, payoff_score=6.0, standalone_score=0.7, context_efficiency=0.7, profile_fit_score=0.7, source="reaction"),
            rank_score=6.0,
        )
        segments = [RichSegment(start=0.0, end=2.0, text="hook", hype_score=3.0)]
        keyframes = [
            SampledKeyframe(candidate_id="a", label="anchor", time_sec=3.0, path=str(Path("/tmp/frame.jpg")))
        ]

        with patch.object(clip_judge.settings, "gemini_api_key", "test-key"), \
             patch.object(clip_judge.settings, "enable_multimodal_clip_judge", True), \
             patch("services.clip_judge._call_gemini_multimodal") as multimodal_mock:
            multimodal_mock.return_value = clip_judge.JudgeResult(
                judge_score=8.7,
                scroll_stop_power=0.9,
                standalone_clarity=0.8,
                payoff_strength=0.9,
                narrative_completeness=0.8,
                platform_fit_shortform=0.8,
                judge_reason="Visuals strengthen the candidate.",
                judge_confidence=0.8,
                provider="gemini_multimodal",
                llm_used=True,
            )
            result = clip_judge.judge_candidate(ranked, segments, keyframes=keyframes)

        self.assertEqual(result.provider, "gemini_multimodal")
        self.assertTrue(multimodal_mock.called)

    def test_heuristic_judge_uses_chapter_signals(self) -> None:
        chapter_strong = RankedCandidate(
            candidate=Candidate(id="strong", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
            features=CandidateFeatures(
                candidate_id="strong",
                hook_strength=0.6,
                max_hype=5.0,
                payoff_score=5.5,
                context_efficiency=0.65,
                standalone_score=0.6,
                arc_score=3.5,
                profile_fit_score=0.7,
                chapter_coherence_score=0.85,
                topic_shift_strength=0.65,
                chapter_local_density=0.6,
                chapter_boundary_distance=2.0,
                clip_overlap_count=0.0,
                trend_hook_score=0.7,
                trend_topic_score=0.65,
                trend_prior_score=0.75,
                source="reaction",
            ),
            rank_score=6.0,
        )
        chapter_weak = RankedCandidate(
            candidate=Candidate(id="weak", start=20.0, end=38.0, source="reaction", anchor_time=24.0),
            features=CandidateFeatures(
                candidate_id="weak",
                hook_strength=0.6,
                max_hype=5.0,
                payoff_score=5.5,
                context_efficiency=0.65,
                standalone_score=0.6,
                arc_score=3.5,
                profile_fit_score=0.7,
                chapter_coherence_score=0.2,
                topic_shift_strength=0.05,
                chapter_local_density=0.2,
                chapter_boundary_distance=14.0,
                clip_overlap_count=2.5,
                trend_hook_score=0.0,
                trend_topic_score=0.0,
                trend_prior_score=0.0,
                source="reaction",
            ),
            rank_score=6.0,
        )
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", hype_score=2.0),
            RichSegment(start=2.0, end=6.0, text="hook and reaction", emotions=["exclamation"], hype_score=5.0),
        ]

        with patch("services.clip_judge._call_llm_judge", return_value=None):
            strong_result = clip_judge.judge_candidate(chapter_strong, segments)
            weak_result = clip_judge.judge_candidate(chapter_weak, segments)

        self.assertGreater(strong_result.judge_score, weak_result.judge_score)
        self.assertIn("clean chapter coherence", strong_result.judge_reason)
        self.assertIn("matches proven short-form patterns", strong_result.judge_reason)
        self.assertIn("overlaps nearby candidates", weak_result.judge_reason)

    def test_heuristic_judge_penalizes_slow_context_dependent_clips(self) -> None:
        risky = RankedCandidate(
            candidate=Candidate(id="risky", start=0.0, end=24.0, source="quote", anchor_time=8.0),
            features=CandidateFeatures(
                candidate_id="risky",
                hook_strength=0.2,
                payoff_score=2.0,
                payoff_clarity_score=0.2,
                standalone_score=0.25,
                context_efficiency=0.3,
                profile_fit_score=0.5,
                slow_open_penalty=0.8,
                context_dependency_score=0.75,
                weak_resolution_penalty=0.7,
                source="quote",
            ),
            rank_score=5.0,
        )
        clean = RankedCandidate(
            candidate=Candidate(id="clean", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
            features=CandidateFeatures(
                candidate_id="clean",
                hook_strength=0.8,
                payoff_score=6.0,
                payoff_clarity_score=0.8,
                standalone_score=0.8,
                context_efficiency=0.8,
                profile_fit_score=0.7,
                slow_open_penalty=0.1,
                context_dependency_score=0.1,
                weak_resolution_penalty=0.1,
                source="reaction",
            ),
            rank_score=5.0,
        )
        segments = [
            RichSegment(start=0.0, end=2.0, text="hook", emotions=["question"], hype_score=2.5),
            RichSegment(start=2.0, end=5.0, text="payoff", emotions=["exclamation"], hype_score=5.0),
        ]

        with patch("services.clip_judge._call_llm_judge", return_value=None):
            risky_result = clip_judge.judge_candidate(risky, segments)
            clean_result = clip_judge.judge_candidate(clean, segments)

        self.assertGreater(clean_result.judge_score, risky_result.judge_score)
        self.assertIn("slow opening risk", risky_result.judge_reason)
        self.assertIn("needs prior stream context", risky_result.judge_reason)


if __name__ == "__main__":
    unittest.main()
