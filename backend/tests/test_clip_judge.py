import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())
sys.modules.setdefault("ollama", types.SimpleNamespace(Client=lambda *_args, **_kwargs: None))
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import clip_judge
from services.keyframe_sampler import SampledKeyframe
from services.timeline_types import Candidate, CandidateFeatures, JudgeResult, RankedCandidate
from services.transcriber import RichSegment


class ClipJudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        patches = [
            patch.object(clip_judge.settings, "anthropic_api_key", None),
            patch.object(clip_judge.settings, "clip_judge_fail_fast", False),
            patch.object(clip_judge.settings, "clip_judge_cache_enabled", False),
            patch.object(clip_judge.settings, "clip_judge_sample_count", 1),
            patch.object(clip_judge.settings, "clip_judge_uncertain_sample_count", 1),
            patch.object(clip_judge.settings, "judge_provider_fallback_enabled", True),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

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

    def test_rerank_candidates_reports_live_progress(self) -> None:
        ranked = [
            RankedCandidate(
                candidate=Candidate(id="a", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
                features=CandidateFeatures(candidate_id="a", hook_strength=0.9, payoff_score=7.0, source="reaction"),
                rank_score=6.0,
            ),
            RankedCandidate(
                candidate=Candidate(id="b", start=30.0, end=48.0, source="quote", anchor_time=33.0),
                features=CandidateFeatures(candidate_id="b", hook_strength=0.6, payoff_score=5.0, source="quote"),
                rank_score=5.5,
            ),
        ]
        segments = [RichSegment(start=0.0, end=60.0, text="setup then payoff", hype_score=3.0)]
        progress: list[dict] = []

        with patch("services.clip_judge._call_llm_judge", return_value=None):
            clip_judge.rerank_candidates(
                ranked,
                segments,
                shortlist_top_k=2,
                progress_cb=progress.append,
            )

        self.assertGreaterEqual(len(progress), 3)
        self.assertEqual(progress[0]["processed_candidates"], 0)
        self.assertEqual(progress[-1]["processed_candidates"], 2)
        self.assertEqual(progress[-1]["total_candidates"], 2)
        self.assertEqual(progress[-1]["attempted_candidates"], 2)
        self.assertEqual(progress[-1]["judged_candidates"], 2)

    def test_cross_platform_recall_target_gets_judge_score_floor(self) -> None:
        ranked = [
            RankedCandidate(
                candidate=Candidate(id="cp", start=0.0, end=18.0, source="dense_inventory", anchor_time=3.0),
                features=CandidateFeatures(
                    candidate_id="cp",
                    source="dense_inventory",
                    cross_platform_recall_target=True,
                    cross_platform_positive_overlap_ratio=1.0,
                ),
                rank_score=-3.0,
                rationale=["cross_platform_recall_target"],
            )
        ]
        segments = [RichSegment(start=0.0, end=18.0, text="verified repost transcript", hype_score=2.0)]
        judge_result = JudgeResult(
            judge_score=3.7,
            scroll_stop_power=0.4,
            standalone_clarity=0.5,
            payoff_strength=0.2,
            narrative_completeness=0.3,
            platform_fit_shortform=0.6,
            judge_reason="underrated by llm",
            judge_confidence=0.8,
            provider="ollama",
            llm_used=True,
        )

        with patch.object(clip_judge.settings, "serving_cross_platform_judge_score_floor", 7.2), \
             patch.object(clip_judge, "judge_candidate", return_value=judge_result):
            reranked = clip_judge.rerank_candidates(ranked, segments, shortlist_top_k=1)

        self.assertGreaterEqual(reranked[0].rank_score, 7.15)
        self.assertEqual(reranked[0].judge_result.judge_score, 3.7)
        self.assertIn("cross_platform_judge_calibrated", reranked[0].rationale)

    def test_rerank_candidates_batches_ollama_judge(self) -> None:
        ranked = [
            RankedCandidate(
                candidate=Candidate(id="a", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
                features=CandidateFeatures(candidate_id="a", hook_strength=0.9, payoff_score=7.0, source="reaction"),
                rank_score=6.0,
                rationale=["baseline"],
            ),
            RankedCandidate(
                candidate=Candidate(id="b", start=20.0, end=38.0, source="quote", anchor_time=25.0),
                features=CandidateFeatures(candidate_id="b", hook_strength=0.7, payoff_score=5.0, source="quote"),
                rank_score=5.5,
                rationale=["baseline"],
            ),
        ]
        segments = [RichSegment(start=0.0, end=60.0, text="setup then payoff", hype_score=3.0)]
        batch_results = {
            "a": JudgeResult(8.0, 0.8, 0.7, 0.8, 0.7, 0.8, "Strong.", 0.8, provider="ollama_batch", llm_used=True),
            "b": JudgeResult(7.0, 0.7, 0.65, 0.7, 0.6, 0.75, "Works.", 0.7, provider="ollama_batch", llm_used=True),
        }

        with patch.object(clip_judge.settings, "clip_judge_batch_size", 2), \
             patch("services.clip_judge._judge_ollama_batch", return_value=batch_results) as batch_mock, \
             patch("services.clip_judge.judge_candidate") as single_mock:
            reranked = clip_judge.rerank_candidates(
                ranked,
                segments,
                provider="ollama",
                shortlist_top_k=2,
            )

        self.assertEqual(batch_mock.call_count, 1)
        self.assertFalse(single_mock.called)
        self.assertTrue(all(item.judge_result is not None for item in reranked))
        self.assertTrue(any("judge_batch" in item.rationale for item in reranked))

    def test_rerank_candidates_falls_back_when_ollama_batch_misses_result(self) -> None:
        ranked = [
            RankedCandidate(
                candidate=Candidate(id="a", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
                features=CandidateFeatures(candidate_id="a", hook_strength=0.9, payoff_score=7.0, source="reaction"),
                rank_score=6.0,
            )
        ]
        segments = [RichSegment(start=0.0, end=30.0, text="setup then payoff", hype_score=3.0)]
        fallback = JudgeResult(6.5, 0.6, 0.6, 0.6, 0.6, 0.6, "Fallback.", 0.6, provider="ollama", llm_used=True)

        with patch.object(clip_judge.settings, "clip_judge_batch_size", 4), \
             patch("services.clip_judge._judge_ollama_batch", return_value={}), \
             patch("services.clip_judge.judge_candidate", return_value=fallback) as single_mock:
            reranked = clip_judge.rerank_candidates(
                ranked,
                segments,
                provider="ollama",
                shortlist_top_k=1,
            )

        self.assertEqual(single_mock.call_count, 1)
        self.assertEqual(reranked[0].judge_result.provider, "ollama")

    def test_llm_first_mode_uses_judge_score_as_primary_rank(self) -> None:
        high_rank_low_judge = RankedCandidate(
            candidate=Candidate(id="old-top", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
            features=CandidateFeatures(candidate_id="old-top", hook_strength=0.6, payoff_score=4.0, source="reaction"),
            rank_score=9.0,
            rationale=["ml_ranked"],
        )
        low_rank_high_judge = RankedCandidate(
            candidate=Candidate(id="llm-top", start=20.0, end=38.0, source="quote", anchor_time=23.0),
            features=CandidateFeatures(candidate_id="llm-top", hook_strength=0.6, payoff_score=4.0, source="quote"),
            rank_score=3.0,
            rationale=["ml_ranked"],
        )
        results = {
            "old-top": clip_judge.JudgeResult(
                judge_score=4.0,
                scroll_stop_power=0.4,
                standalone_clarity=0.4,
                payoff_strength=0.4,
                narrative_completeness=0.4,
                platform_fit_shortform=0.4,
                judge_reason="Weak as short form.",
                judge_confidence=0.8,
                provider="gemini",
                llm_used=True,
            ),
            "llm-top": clip_judge.JudgeResult(
                judge_score=8.5,
                scroll_stop_power=0.9,
                standalone_clarity=0.8,
                payoff_strength=0.9,
                narrative_completeness=0.8,
                platform_fit_shortform=0.85,
                judge_reason="Strong as short form.",
                judge_confidence=0.8,
                provider="gemini",
                llm_used=True,
            ),
        }

        def fake_judge(ranked, *_args, **_kwargs):
            return results[ranked.candidate.id]

        with patch.object(clip_judge.settings, "judge_llm_first", True), \
             patch("services.clip_judge.judge_candidate", side_effect=fake_judge):
            reranked = clip_judge.rerank_candidates(
                [high_rank_low_judge, low_rank_high_judge],
                [RichSegment(start=0.0, end=2.0, text="hook")],
                shortlist_top_k=2,
            )

        self.assertEqual(reranked[0].candidate.id, "llm-top")
        self.assertIn("judge_primary_ranked", reranked[0].rationale)
        self.assertTrue(reranked[0].features.has_judge_score)
        self.assertEqual(reranked[0].features.judge_score, 8.5)
        self.assertTrue(reranked[0].features.judge_llm_used)

    def test_llm_first_mode_falls_back_to_blend_for_low_confidence_judge(self) -> None:
        ranked = RankedCandidate(
            candidate=Candidate(id="a", start=0.0, end=18.0, source="reaction", anchor_time=3.0),
            features=CandidateFeatures(candidate_id="a", hook_strength=0.6, payoff_score=4.0, source="reaction"),
            rank_score=9.0,
        )
        judge_result = clip_judge.JudgeResult(
            judge_score=2.0,
            scroll_stop_power=0.2,
            standalone_clarity=0.2,
            payoff_strength=0.2,
            narrative_completeness=0.2,
            platform_fit_shortform=0.2,
            judge_reason="Uncertain.",
            judge_confidence=0.1,
            provider="gemini",
            llm_used=True,
        )

        with patch.object(clip_judge.settings, "judge_llm_first", True), \
             patch.object(clip_judge.settings, "judge_llm_first_min_confidence", 0.35), \
             patch("services.clip_judge.judge_candidate", return_value=judge_result):
            reranked = clip_judge.rerank_candidates(
                [ranked],
                [RichSegment(start=0.0, end=2.0, text="hook")],
                shortlist_top_k=1,
            )

        self.assertNotEqual(reranked[0].rank_score, 2.0)
        self.assertIn("judge_low_confidence_blended", reranked[0].rationale)

    def test_recall_filter_skips_dead_candidates_before_judge(self) -> None:
        dead = RankedCandidate(
            candidate=Candidate(id="dead", start=0.0, end=18.0, source="background", anchor_time=3.0),
            features=CandidateFeatures(candidate_id="dead", source="background"),
            rank_score=-1.0,
        )

        with patch.object(clip_judge.settings, "judge_recall_filter_enabled", True), \
             patch.object(clip_judge.settings, "judge_recall_filter_min_score", 0.0), \
             patch("services.clip_judge.judge_candidate") as judge_mock:
            reranked = clip_judge.rerank_candidates(
                [dead],
                [RichSegment(start=0.0, end=2.0, text="")],
                shortlist_top_k=1,
            )

        self.assertFalse(judge_mock.called)
        self.assertIsNone(reranked[0].judge_result)
        self.assertIn("judge_recall_filtered", reranked[0].rationale)

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

    def test_ollama_judge_requests_json_format(self) -> None:
        captured = {}

        class FakeClient:
            def __init__(self, *, host: str) -> None:
                captured["host"] = host

            def chat(self, **kwargs):
                captured.update(kwargs)
                return {
                    "message": {
                        "content": (
                            '{"scroll_stop_power":0.8,"standalone_clarity":0.7,'
                            '"payoff_strength":0.75,"narrative_completeness":0.65,'
                            '"platform_fit_shortform":0.8,"judge_reason":"Works.",'
                            '"judge_confidence":0.74}'
                        )
                    }
                }

        with patch.object(clip_judge.ollama, "Client", FakeClient), \
             patch.object(clip_judge.settings, "ollama_host", "http://ollama.local"), \
             patch.object(clip_judge.settings, "ollama_model", "local-model"):
            result = clip_judge._call_ollama("judge this")

        self.assertIsNotNone(result)
        self.assertEqual(captured["host"], "http://ollama.local")
        self.assertEqual(captured["model"], "local-model")
        self.assertEqual(captured["format"], "json")
        self.assertEqual(captured["options"], {"temperature": 0.0})

    def test_auto_provider_fallback_to_ollama_is_logged(self) -> None:
        keyframes = [
            SampledKeyframe(candidate_id="a", label="anchor", time_sec=3.0, path=str(Path("/tmp/frame.jpg")))
        ]
        ollama_result = clip_judge.JudgeResult(
            judge_score=7.8,
            scroll_stop_power=0.8,
            standalone_clarity=0.7,
            payoff_strength=0.8,
            narrative_completeness=0.7,
            platform_fit_shortform=0.8,
            judge_reason="Local fallback works.",
            judge_confidence=0.7,
            provider="ollama",
            llm_used=True,
        )

        with patch.object(clip_judge.settings, "anthropic_api_key", None), \
             patch.object(clip_judge.settings, "gemini_api_key", "test-key"), \
             patch.object(clip_judge.settings, "groq_api_key", "test-key"), \
             patch.object(clip_judge.settings, "enable_multimodal_clip_judge", True), \
             patch.object(clip_judge.settings, "judge_provider_fallback_enabled", True), \
             patch("services.clip_judge._call_gemini_multimodal_from_optional_keyframes", return_value=None), \
             patch("services.clip_judge._call_gemini_text", return_value=None), \
             patch("services.clip_judge._call_groq_text", return_value=None), \
             patch("services.clip_judge._call_ollama_text", return_value=ollama_result), \
             self.assertLogs("services.clip_judge", level="INFO") as captured:
            result = clip_judge._call_text_judge_prompt("judge this", provider=None, keyframes=keyframes)

        events = [json.loads(record.getMessage()) for record in captured.records]
        event_names = [event["event"] for event in events]
        self.assertEqual(result.provider, "ollama")
        self.assertIn("clip_judge_provider_failed", event_names)
        self.assertIn("clip_judge_provider_fallback", event_names)
        self.assertIn("clip_judge_provider_selected", event_names)
        self.assertEqual(events[-1]["provider"], "ollama")
        self.assertEqual(events[-1]["fallback_from"], ["gemini_multimodal", "gemini", "groq"])

    def test_auto_provider_does_not_fallback_when_disabled(self) -> None:
        with patch.object(clip_judge.settings, "anthropic_api_key", "test-key"), \
             patch.object(clip_judge.settings, "gemini_api_key", "test-key"), \
             patch.object(clip_judge.settings, "judge_provider_fallback_enabled", False), \
             patch.object(clip_judge.settings, "clip_judge_fail_fast", False), \
             patch("services.clip_judge._call_anthropic_text", return_value=None) as anthropic_mock, \
             patch("services.clip_judge._call_gemini_text") as gemini_mock, \
             self.assertLogs("services.clip_judge", level="INFO") as captured:
            result = clip_judge._call_text_judge_prompt("judge this", provider=None, keyframes=None)

        events = [json.loads(record.getMessage()) for record in captured.records]
        event_names = [event["event"] for event in events]
        self.assertIsNone(result)
        self.assertTrue(anthropic_mock.called)
        self.assertFalse(gemini_mock.called)
        self.assertIn("clip_judge_provider_failed", event_names)
        self.assertNotIn("clip_judge_provider_fallback", event_names)
        self.assertIn("clip_judge_all_providers_failed", event_names)

    def test_fail_fast_raises_when_llm_provider_fails(self) -> None:
        with patch.object(clip_judge.settings, "anthropic_api_key", "test-key"), \
             patch.object(clip_judge.settings, "judge_provider_fallback_enabled", False), \
             patch.object(clip_judge.settings, "clip_judge_fail_fast", True), \
             patch("services.clip_judge._call_anthropic_text", return_value=None), \
             self.assertLogs("services.clip_judge", level="INFO") as captured:
            with self.assertRaises(clip_judge.ClipJudgeError):
                clip_judge._call_text_judge_prompt("judge this", provider=None, keyframes=None)

        events = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual(events[-1]["event"], "clip_judge_all_providers_failed")
        self.assertTrue(events[-1]["fail_fast"])

    def test_rerank_candidates_logs_summary(self) -> None:
        ranked = [
            RankedCandidate(
                candidate=Candidate(id="a", start=0.0, end=15.0, source="reaction", anchor_time=3.0),
                features=CandidateFeatures(candidate_id="a", hook_strength=0.8, payoff_score=7.0, standalone_score=0.7),
                rank_score=6.0,
            )
        ]
        segments = [RichSegment(start=0.0, end=2.0, text="hook", hype_score=3.0)]
        judge_result = clip_judge.JudgeResult(
            judge_score=8.0,
            scroll_stop_power=0.8,
            standalone_clarity=0.8,
            payoff_strength=0.8,
            narrative_completeness=0.8,
            platform_fit_shortform=0.8,
            judge_reason="Strong.",
            judge_confidence=0.8,
            provider="anthropic",
            llm_used=True,
        )

        with patch("services.clip_judge.judge_candidate", return_value=judge_result), \
             patch("services.clip_judge._append_judge_snapshot"), \
             self.assertLogs("services.clip_judge", level="INFO") as captured:
            clip_judge.rerank_candidates(ranked, segments, shortlist_top_k=1)

        events = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual(events[0]["event"], "clip_judge_rerank_start")
        self.assertEqual(events[-1]["event"], "clip_judge_rerank_done")
        self.assertEqual(events[-1]["counters"]["judge_attempted"], 1)
        self.assertEqual(events[-1]["counters"]["llm_used"], 1)
        self.assertEqual(events[-1]["provider_counts"]["anthropic"], 1)

    def test_auto_provider_prefers_anthropic_when_configured(self) -> None:
        anthropic_result = clip_judge.JudgeResult(
            judge_score=8.0,
            scroll_stop_power=0.8,
            standalone_clarity=0.8,
            payoff_strength=0.8,
            narrative_completeness=0.8,
            platform_fit_shortform=0.8,
            judge_reason="Claude selected this.",
            judge_confidence=0.75,
            provider="anthropic",
            llm_used=True,
            model="claude-haiku-4-5-20251001",
        )

        with patch.object(clip_judge.settings, "anthropic_api_key", "test-key"), \
             patch.object(clip_judge.settings, "gemini_api_key", "test-key"), \
             patch("services.clip_judge._call_anthropic_text", return_value=anthropic_result) as anthropic_mock, \
             patch("services.clip_judge._call_gemini_text") as gemini_mock:
            result = clip_judge._call_text_judge_prompt("judge this", provider=None, keyframes=None)

        self.assertEqual(result.provider, "anthropic")
        self.assertTrue(anthropic_mock.called)
        self.assertFalse(gemini_mock.called)

    def test_explicit_ollama_provider_ignores_configured_cloud_providers(self) -> None:
        ollama_result = clip_judge.JudgeResult(
            judge_score=7.4,
            scroll_stop_power=0.75,
            standalone_clarity=0.7,
            payoff_strength=0.75,
            narrative_completeness=0.7,
            platform_fit_shortform=0.75,
            judge_reason="Ollama selected this.",
            judge_confidence=0.7,
            provider="ollama",
            llm_used=True,
            model="local-model",
        )

        with patch.object(clip_judge.settings, "anthropic_api_key", "test-key"), \
             patch.object(clip_judge.settings, "gemini_api_key", "test-key"), \
             patch.object(clip_judge.settings, "groq_api_key", "test-key"), \
             patch("services.clip_judge._call_anthropic_text") as anthropic_mock, \
             patch("services.clip_judge._call_gemini_text") as gemini_mock, \
             patch("services.clip_judge._call_groq_text") as groq_mock, \
             patch("services.clip_judge._call_ollama_text", return_value=ollama_result) as ollama_mock:
            result = clip_judge._call_text_judge_prompt("judge this", provider="ollama", keyframes=None)

        self.assertEqual(result.provider, "ollama")
        self.assertTrue(ollama_mock.called)
        self.assertFalse(anthropic_mock.called)
        self.assertFalse(gemini_mock.called)
        self.assertFalse(groq_mock.called)

    def test_anthropic_judge_uses_prompt_cache_control(self) -> None:
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    '{"scroll_stop_power":0.8,"standalone_clarity":0.7,'
                                    '"payoff_strength":0.75,"narrative_completeness":0.65,'
                                    '"platform_fit_shortform":0.8,"judge_reason":"Works.",'
                                    '"judge_confidence":0.74}'
                                ),
                            }
                        ],
                        "usage": {
                            "input_tokens": 120,
                            "output_tokens": 40,
                            "cache_creation_input_tokens": 80,
                            "cache_read_input_tokens": 20,
                        },
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        prompt = clip_judge._JUDGE_PROMPT.format(
            profile="jynxzi",
            rank_score="6.000",
            source="reaction",
            duration="18.00",
            feature_summary="hook_strength=0.9",
            context_summary="Limited prior context.",
            transcript="[0.0s] no way",
        )

        with patch.object(clip_judge.settings, "anthropic_api_key", "test-key"), \
             patch.object(clip_judge.settings, "anthropic_model_bulk", "claude-haiku-4-5-20251001"), \
             patch.object(clip_judge.settings, "anthropic_prompt_caching", True), \
             patch.object(clip_judge.settings, "clip_judge_timeout_sec", 12.0), \
             patch("services.clip_judge.urllib.request.urlopen", side_effect=fake_urlopen), \
             self.assertLogs("services.clip_judge", level="INFO") as logs:
            result = clip_judge._call_anthropic_text(prompt)

        self.assertIsNotNone(result)
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.model, "claude-haiku-4-5-20251001")
        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured["timeout"], 12.0)
        self.assertEqual(captured["payload"]["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(
            captured["payload"]["system"][0]["cache_control"],
            {"type": "ephemeral"},
        )
        self.assertIn("Profile: jynxzi", captured["payload"]["messages"][0]["content"][0]["text"])
        usage_events = [
            json.loads(record.getMessage())
            for record in logs.records
            if json.loads(record.getMessage()).get("event") == "clip_judge_provider_usage"
        ]
        self.assertEqual(usage_events[0]["cache_read_input_tokens"], 20)
        self.assertEqual(usage_events[0]["cache_creation_input_tokens"], 80)

    def test_anthropic_multimodal_uses_topk_model_and_image_blocks(self) -> None:
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    '{"scroll_stop_power":0.9,"standalone_clarity":0.8,'
                                    '"payoff_strength":0.85,"narrative_completeness":0.75,'
                                    '"platform_fit_shortform":0.9,"judge_reason":"Visual works.",'
                                    '"judge_confidence":0.8}'
                                ),
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        with tempfile.NamedTemporaryFile(suffix=".jpg") as frame:
            frame.write(b"jpeg-bytes")
            frame.flush()
            keyframes = [
                SampledKeyframe(
                    candidate_id="a",
                    label="anchor",
                    time_sec=3.0,
                    path=frame.name,
                    mime_type="image/jpeg",
                )
            ]
            with patch.object(clip_judge.settings, "anthropic_api_key", "test-key"), \
                 patch.object(clip_judge.settings, "anthropic_model_topk", "claude-sonnet-4-6"), \
                 patch("services.clip_judge.urllib.request.urlopen", side_effect=fake_urlopen):
                result = clip_judge._call_anthropic_multimodal_from_optional_keyframes("Profile: jynxzi", keyframes)

        self.assertIsNotNone(result)
        self.assertEqual(result.provider, "anthropic_multimodal")
        self.assertEqual(result.model, "claude-sonnet-4-6")
        self.assertEqual(captured["payload"]["model"], "claude-sonnet-4-6")
        content = captured["payload"]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["source"]["type"], "base64")
        self.assertEqual(content[0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(content[0]["source"]["data"], "anBlZy1ieXRlcw==")

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

    def test_judge_prompt_values_reaction_spikes_without_full_story_requirement(self) -> None:
        self.assertIn("reaction spike", clip_judge._JUDGE_PROMPT_STATIC)
        self.assertIn("Do not require a complete stream story", clip_judge._JUDGE_PROMPT_STATIC)
        batch_prompt = clip_judge._batch_judge_prompt(profile="jynxzi", items=[])
        self.assertIn("reaction spike", batch_prompt)
        self.assertIn("do not over-penalize short reaction clips", batch_prompt)

    def test_judge_candidate_uses_cache_for_repeatable_results(self) -> None:
        ranked = RankedCandidate(
            candidate=Candidate(id="cache-me", start=0.0, end=15.0, source="reaction", anchor_time=4.0),
            features=CandidateFeatures(candidate_id="cache-me", hook_strength=0.8, payoff_score=6.0),
            rank_score=6.0,
        )
        segments = [RichSegment(start=0.0, end=3.0, text="no way", hype_score=4.0)]
        result = JudgeResult(
            judge_score=8.0,
            scroll_stop_power=0.8,
            standalone_clarity=0.7,
            payoff_strength=0.8,
            narrative_completeness=0.6,
            platform_fit_shortform=0.8,
            judge_reason="Strong cached result.",
            judge_confidence=0.8,
            provider="gemini",
            llm_used=True,
            model="test-model",
        )

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(clip_judge.settings, "storage_dir", str(Path(tmpdir) / "storage")), \
             patch.object(clip_judge.settings, "clip_judge_cache_enabled", True), \
             patch.object(clip_judge, "_call_text_judge_prompt", return_value=result) as llm_mock:
            first = clip_judge.judge_candidate(ranked, segments, provider="gemini")
            second = clip_judge.judge_candidate(ranked, segments, provider="gemini")

        self.assertEqual(first.judge_score, 8.0)
        self.assertEqual(second.judge_score, 8.0)
        self.assertEqual(llm_mock.call_count, 1)

    def test_uncertain_judge_result_uses_median_consensus(self) -> None:
        ranked = RankedCandidate(
            candidate=Candidate(id="median-me", start=0.0, end=15.0, source="reaction", anchor_time=4.0),
            features=CandidateFeatures(candidate_id="median-me", hook_strength=0.8, payoff_score=6.0),
            rank_score=6.0,
        )
        segments = [RichSegment(start=0.0, end=3.0, text="no way", hype_score=4.0)]
        samples = [
            JudgeResult(7.0, 0.7, 0.4, 0.8, 0.4, 0.8, "low narrative", 0.8, provider="gemini", llm_used=True),
            JudgeResult(8.0, 0.8, 0.7, 0.9, 0.7, 0.8, "strong", 0.9, provider="gemini", llm_used=True),
            JudgeResult(6.0, 0.6, 0.6, 0.7, 0.6, 0.7, "middle", 0.7, provider="gemini", llm_used=True),
        ]

        with patch.object(clip_judge.settings, "clip_judge_cache_enabled", False), \
             patch.object(clip_judge.settings, "clip_judge_sample_count", 1), \
             patch.object(clip_judge.settings, "clip_judge_uncertain_sample_count", 3), \
             patch.object(clip_judge.settings, "clip_judge_uncertain_margin", 0.08), \
             patch.object(clip_judge, "_call_text_judge_prompt", side_effect=samples) as llm_mock:
            result = clip_judge.judge_candidate(ranked, segments, provider="gemini")

        self.assertEqual(llm_mock.call_count, 3)
        self.assertAlmostEqual(result.standalone_clarity, 0.6)
        self.assertAlmostEqual(result.narrative_completeness, 0.6)
        self.assertEqual(result.provider, "gemini")


if __name__ == "__main__":
    unittest.main()
