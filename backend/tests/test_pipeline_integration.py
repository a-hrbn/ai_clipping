import asyncio
import json
import importlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import types

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))


class _FFmpegStub:
    def input(self, *_args, **_kwargs):
        return self

    def output(self, *_args, **_kwargs):
        return self

    def overwrite_output(self):
        return self

    def run(self, **_kwargs):
        return None


sys.modules.setdefault("ffmpeg", _FFmpegStub())
sys.modules.setdefault("ollama", types.SimpleNamespace(Client=lambda *_args, **_kwargs: None))
sys.modules.setdefault("librosa", types.SimpleNamespace())

from models import ClipResult
from services.timeline_types import Candidate, CandidateFeatures, Chapter, JudgeResult, RankedCandidate
from services.transcriber import RichSegment


def _identity(*args, **kwargs):
    return args[0]


class PipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_await_with_optional_timeout_skips_wait_for_when_disabled(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        loop = asyncio.get_running_loop()

        future_none = loop.create_future()
        future_none.set_result("none")
        future_zero = loop.create_future()
        future_zero.set_result("zero")

        with patch.object(pipeline.asyncio, "wait_for", side_effect=AssertionError("wait_for should not run")):
            result_none = await pipeline._await_with_optional_timeout(future_none, None)
            result_zero = await pipeline._await_with_optional_timeout(future_zero, 0)

        self.assertEqual(result_none, "none")
        self.assertEqual(result_zero, "zero")

    async def test_run_pipeline_prefers_peak_path_and_writes_debug_artifacts(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            audio_path = tmp_path / "audio.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")

            job_updates: list[dict] = []
            cleanup_coroutines = []

            transcript = [
                RichSegment(start=0.0, end=2.0, text="quiet setup", words_per_second=1.0),
                RichSegment(
                    start=2.0,
                    end=4.0,
                    text="wait what happened?!",
                    emotions=["exclamation", "question", "audio_laughter", "face_reaction"],
                    words_per_second=3.6,
                    laughter_prob=0.6,
                    chat_spike_score=0.7,
                    max_face_reaction=0.5,
                ),
                RichSegment(start=4.0, end=6.0, text="payoff lands", emotions=["laughter"], words_per_second=2.0),
            ]

            semantic_candidates = [
                Candidate(
                    id="quote:0:250",
                    start=1.5,
                    end=9.5,
                    source="quote",
                    anchor_time=2.5,
                    score_hint=4.0,
                )
            ]
            chapters = [
                Chapter(id="chapter_00", start=0.0, end=6.0, index=0, label="reaction", confidence=0.8)
            ]

            def fake_update_job(job_id: str, **kwargs) -> None:
                job_updates.append({"job_id": job_id, **kwargs})

            build_feature_calls: list[tuple[bool, set[str] | None, list[Chapter] | None]] = []

            def fake_build_features(
                candidates,
                _segments,
                _profile,
                use_llm_semantics=True,
                llm_candidate_ids=None,
                llm_provider=None,
                chapters=None,
                strategy=None,
            ):
                del llm_provider, strategy
                build_feature_calls.append((use_llm_semantics, llm_candidate_ids, chapters))
                return [
                    CandidateFeatures(
                        candidate_id=candidate.id,
                        hook_strength=0.9 if candidate.source == "peak" else 0.5,
                        max_hype=6.0,
                        payoff_score=5.0,
                        context_efficiency=0.8,
                        source=candidate.source,
                    )
                    for candidate in candidates
                ]

            def fake_rank_candidates(candidates, features, profile="general", strategy=None):
                del strategy
                feature_map = {feature.candidate_id: feature for feature in features}
                ranked = [
                    RankedCandidate(
                        candidate=candidate,
                        features=feature_map[candidate.id],
                        rank_score=9.0 if candidate.source == "peak" else 5.0,
                        rationale=["peak_first" if candidate.source == "peak" else "semantic"],
                    )
                    for candidate in candidates
                ]
                ranked.sort(key=lambda item: item.rank_score, reverse=True)
                return ranked

            def fake_judge_rerank(ranked_candidates, _segments, _profile="general", shortlist_top_k=None, provider=None, keyframes_by_candidate=None):
                del provider, keyframes_by_candidate
                updated = []
                limit = shortlist_top_k or len(ranked_candidates)
                for index, ranked in enumerate(ranked_candidates):
                    judge_result = None
                    score = ranked.rank_score
                    rationale = list(ranked.rationale)
                    if index < limit:
                        judge_result = JudgeResult(
                            judge_score=8.8 if ranked.candidate.source == "peak" else 5.9,
                            scroll_stop_power=0.9,
                            standalone_clarity=0.8,
                            payoff_strength=0.9,
                            narrative_completeness=0.8,
                            platform_fit_shortform=0.85,
                            judge_reason="Strong short-form candidate.",
                            judge_confidence=0.7,
                        )
                        score += 0.4 if ranked.candidate.source == "peak" else 0.1
                        rationale.append("judge_reranked")
                    updated.append(
                        RankedCandidate(
                            candidate=ranked.candidate,
                            features=ranked.features,
                            rank_score=score,
                            rationale=rationale,
                            base_rank_score=ranked.rank_score,
                            judge_result=judge_result,
                        )
                    )
                updated.sort(key=lambda item: item.rank_score, reverse=True)
                return updated

            def fake_select_diverse_ranked_candidates(ranked_candidates, top_k, min_gap_sec=20.0, strategy=None):
                del min_gap_sec, strategy
                return ranked_candidates[:top_k]

            def fake_refine_candidates(ranked_candidates, _segments, limit=5, profile="general", strategy=None):
                del strategy
                moments = []
                for index, ranked in enumerate(ranked_candidates[:limit], start=1):
                    moments.append(
                        {
                            "start_time": ranked.candidate.start,
                            "end_time": ranked.candidate.end,
                            "viral_score": 9.0 if ranked.candidate.source == "peak" else 6.0,
                            "reason": f"{ranked.candidate.source} moment",
                            "rank": index,
                            "ranking_mode": "heuristic",
                            "candidate_id": ranked.candidate.id,
                        }
                    )
                return moments

            def fake_cut_clips(_video_path, moments, job_id):
                return [
                    ClipResult(
                        candidate_id=moment["candidate_id"],
                        filename=f"clip_{idx:02d}.mp4",
                        download_url=f"/static/{job_id}/clips/clip_{idx:02d}.mp4",
                        start_time=moment["start_time"],
                        end_time=moment["end_time"],
                        viral_score=moment["viral_score"],
                        reason=moment["reason"],
                        ranking_mode=moment.get("ranking_mode", "heuristic"),
                    )
                    for idx, moment in enumerate(moments, start=1)
                ]

            def fake_create_task(coro):
                cleanup_coroutines.append(coro)
                coro.close()
                return None

            refine_calls: list[list[str]] = []

            def fake_refine_face_reactions(segments, _video_path, candidates, _sample_sec):
                refine_calls.append([candidate.id for candidate in candidates])
                return segments

            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline, "update_job", side_effect=fake_update_job))
                stack.enter_context(patch.object(pipeline.audio_extractor, "extract", return_value=audio_path))
                stack.enter_context(patch.object(pipeline.transcriber, "transcribe_with_progress", return_value=transcript))
                stack.enter_context(patch.object(pipeline.audio_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.video_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.audio_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.video_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(
                    patch.object(
                        pipeline.video_features,
                        "refine_face_reactions",
                        side_effect=fake_refine_face_reactions,
                        create=True,
                    )
                )
                stack.enter_context(patch.object(pipeline.audio_features, "diarize_speakers", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.chat_analyzer, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.chapter_segmenter, "segment_chapters", return_value=chapters))
                stack.enter_context(
                    patch.object(pipeline.candidate_generators, "generate_semantic_candidates", return_value=semantic_candidates)
                )
                stack.enter_context(
                    patch.object(pipeline.candidate_generators, "generate_fallback_candidates", return_value=[])
                )
                stack.enter_context(patch.object(pipeline.feature_builder, "build_features", side_effect=fake_build_features))
                stack.enter_context(patch.object(pipeline.selector, "rank_candidates", side_effect=fake_rank_candidates))
                stack.enter_context(
                    patch.object(
                        pipeline.selector,
                        "select_diverse_ranked_candidates",
                        side_effect=fake_select_diverse_ranked_candidates,
                    )
                )
                stack.enter_context(patch.object(pipeline.clip_judge, "rerank_candidates", side_effect=fake_judge_rerank))
                stack.enter_context(patch.object(pipeline.analyzer, "refine_candidates", side_effect=fake_refine_candidates))
                stack.enter_context(patch.object(pipeline.analyzer, "expand_variants", side_effect=lambda moments, *_args, **_kwargs: moments))
                stack.enter_context(patch.object(pipeline.post_processor, "snap_boundaries", side_effect=lambda moments, _segments: moments))
                stack.enter_context(patch.object(pipeline.post_processor, "enforce_diversity", side_effect=lambda moments: moments))
                stack.enter_context(patch.object(pipeline.clipper, "cut_clips", side_effect=fake_cut_clips))
                stack.enter_context(patch.object(pipeline.asyncio, "create_task", side_effect=fake_create_task))

                await pipeline.run_pipeline("job-peak", video_path=video_path, profile="general")

            self.assertTrue(any(update.get("status") == "done" for update in job_updates))
            self.assertEqual(len(refine_calls), 1)
            self.assertEqual(len(refine_calls[0]), 1)
            self.assertTrue(refine_calls[0][0].startswith("peak:"))
            self.assertEqual(build_feature_calls[0], (False, None, chapters))
            self.assertEqual(build_feature_calls[1][0], True)
            self.assertEqual(len(build_feature_calls[1][1] or set()), 1)
            self.assertEqual(build_feature_calls[1][2], chapters)
            debug_dir = tmp_path / "job-peak" / "debug"
            peak_rows = _read_jsonl(debug_dir / "candidates_peak.jsonl")
            semantic_rows = _read_jsonl(debug_dir / "candidates_semantic.jsonl")
            fallback_rows = _read_jsonl(debug_dir / "candidates_fallback.jsonl")
            merged_rows = _read_jsonl(debug_dir / "candidates_merged.jsonl")
            chapter_rows = _read_jsonl(debug_dir / "chapters.jsonl")
            judged_rows = _read_jsonl(debug_dir / "judged_candidates.jsonl")
            ranked_rows = _read_jsonl(debug_dir / "ranked_candidates.jsonl")

            self.assertTrue(peak_rows)
            self.assertTrue(semantic_rows)
            self.assertEqual(fallback_rows, [])
            self.assertEqual(merged_rows[0]["source"], "peak")
            self.assertEqual(chapter_rows[0]["label"], "reaction")
            self.assertTrue(judged_rows)
            self.assertEqual(judged_rows[0]["judge_result"]["judge_reason"], "Strong short-form candidate.")
            self.assertEqual(ranked_rows[0]["candidate"]["source"], "peak")

    async def test_run_pipeline_uses_fallback_candidates_when_no_peaks_exist(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            audio_path = tmp_path / "audio.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")

            job_updates: list[dict] = []

            transcript = [
                RichSegment(start=0.0, end=2.0, text="calm intro", words_per_second=0.8),
                RichSegment(start=2.0, end=4.0, text="plain statement", words_per_second=1.0),
            ]
            semantic_candidates: list[Candidate] = []
            fallback_candidates = [
                Candidate(
                    id="reaction:0:200",
                    start=0.0,
                    end=10.0,
                    source="reaction",
                    anchor_time=2.0,
                    score_hint=3.0,
                )
            ]
            chapters = [
                Chapter(id="chapter_00", start=0.0, end=4.0, index=0, label="mixed", confidence=0.6)
            ]

            def fake_update_job(job_id: str, **kwargs) -> None:
                job_updates.append({"job_id": job_id, **kwargs})

            def fake_build_features(
                candidates,
                _segments,
                _profile,
                use_llm_semantics=True,
                llm_candidate_ids=None,
                llm_provider=None,
                chapters=None,
                strategy=None,
            ):
                del use_llm_semantics, llm_candidate_ids, llm_provider, chapters, strategy
                return [
                    CandidateFeatures(
                        candidate_id=candidate.id,
                        hook_strength=0.4,
                        max_hype=2.0,
                        payoff_score=2.0,
                        context_efficiency=0.5,
                        source=candidate.source,
                    )
                    for candidate in candidates
                ]

            def fake_rank_candidates(candidates, features, profile="general", strategy=None):
                del strategy
                feature_map = {feature.candidate_id: feature for feature in features}
                return [
                    RankedCandidate(
                        candidate=candidate,
                        features=feature_map[candidate.id],
                        rank_score=4.0,
                        rationale=["fallback"],
                    )
                    for candidate in candidates
                ]

            def fake_judge_rerank(ranked_candidates, _segments, _profile="general", shortlist_top_k=None, provider=None, keyframes_by_candidate=None):
                del shortlist_top_k, provider, keyframes_by_candidate
                return [
                    RankedCandidate(
                        candidate=ranked.candidate,
                        features=ranked.features,
                        rank_score=ranked.rank_score + 0.2,
                        rationale=list(ranked.rationale) + ["judge_reranked"],
                        base_rank_score=ranked.rank_score,
                        judge_result=JudgeResult(
                            judge_score=6.0,
                            scroll_stop_power=0.5,
                            standalone_clarity=0.6,
                            payoff_strength=0.6,
                            narrative_completeness=0.5,
                            platform_fit_shortform=0.6,
                            judge_reason="Fallback candidate still works as a short.",
                            judge_confidence=0.6,
                        ),
                    )
                    for ranked in ranked_candidates
                ]

            def fake_select_diverse_ranked_candidates(ranked_candidates, top_k, min_gap_sec=20.0, strategy=None):
                del min_gap_sec, strategy
                return ranked_candidates[:top_k]

            def fake_refine_candidates(ranked_candidates, _segments, limit=5, profile="general", strategy=None):
                del strategy
                return [
                    {
                        "start_time": ranked_candidates[0].candidate.start,
                        "end_time": ranked_candidates[0].candidate.end,
                        "viral_score": 4.0,
                        "reason": "fallback moment",
                        "rank": 1,
                        "ranking_mode": "heuristic",
                        "candidate_id": ranked_candidates[0].candidate.id,
                    }
                ]

            def fake_create_task(coro):
                coro.close()
                return None

            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline, "update_job", side_effect=fake_update_job))
                stack.enter_context(patch.object(pipeline.audio_extractor, "extract", return_value=audio_path))
                stack.enter_context(patch.object(pipeline.transcriber, "transcribe_with_progress", return_value=transcript))
                stack.enter_context(patch.object(pipeline.audio_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.video_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.audio_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.video_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.audio_features, "diarize_speakers", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.chat_analyzer, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.chapter_segmenter, "segment_chapters", return_value=chapters))
                stack.enter_context(
                    patch.object(pipeline.peak_detector, "detect_peaks", return_value=[])
                )
                stack.enter_context(
                    patch.object(pipeline.peak_detector, "build_peak_candidates", return_value=[])
                )
                stack.enter_context(
                    patch.object(pipeline.candidate_generators, "generate_semantic_candidates", return_value=semantic_candidates)
                )
                stack.enter_context(
                    patch.object(pipeline.candidate_generators, "generate_fallback_candidates", return_value=fallback_candidates)
                )
                stack.enter_context(patch.object(pipeline.feature_builder, "build_features", side_effect=fake_build_features))
                stack.enter_context(patch.object(pipeline.selector, "rank_candidates", side_effect=fake_rank_candidates))
                stack.enter_context(
                    patch.object(
                        pipeline.selector,
                        "select_diverse_ranked_candidates",
                        side_effect=fake_select_diverse_ranked_candidates,
                    )
                )
                stack.enter_context(patch.object(pipeline.clip_judge, "rerank_candidates", side_effect=fake_judge_rerank))
                stack.enter_context(patch.object(pipeline.analyzer, "refine_candidates", side_effect=fake_refine_candidates))
                stack.enter_context(patch.object(pipeline.analyzer, "expand_variants", side_effect=lambda moments, *_args, **_kwargs: moments))
                stack.enter_context(patch.object(pipeline.post_processor, "snap_boundaries", side_effect=lambda moments, _segments: moments))
                stack.enter_context(patch.object(pipeline.post_processor, "enforce_diversity", side_effect=lambda moments: moments))
                stack.enter_context(
                    patch.object(
                        pipeline.clipper,
                        "cut_clips",
                        return_value=[
                            ClipResult(
                                candidate_id="reaction:0:200",
                                filename="clip_01.mp4",
                                download_url="/static/job-fallback/clips/clip_01.mp4",
                                start_time=0.0,
                                end_time=10.0,
                                viral_score=4.0,
                                reason="fallback moment",
                                ranking_mode="heuristic",
                            )
                        ],
                    )
                )
                stack.enter_context(patch.object(pipeline.asyncio, "create_task", side_effect=fake_create_task))

                await pipeline.run_pipeline("job-fallback", video_path=video_path, profile="general")

            self.assertTrue(any(update.get("status") == "done" for update in job_updates))
            debug_dir = tmp_path / "job-fallback" / "debug"
            peak_rows = _read_jsonl(debug_dir / "candidates_peak.jsonl")
            semantic_rows = _read_jsonl(debug_dir / "candidates_semantic.jsonl")
            fallback_rows = _read_jsonl(debug_dir / "candidates_fallback.jsonl")
            merged_rows = _read_jsonl(debug_dir / "candidates_merged.jsonl")

            self.assertEqual(peak_rows, [])
            self.assertEqual(semantic_rows, [])
            self.assertTrue(fallback_rows)
            self.assertEqual(merged_rows[0]["source"], "reaction")

    async def test_run_pipeline_disables_chat_download_and_keeps_noop_enrichment(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            audio_path = tmp_path / "audio.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")

            transcript = [
                RichSegment(start=0.0, end=2.0, text="setup", words_per_second=1.0),
                RichSegment(start=2.0, end=4.0, text="payoff", words_per_second=2.5, hype_score=4.0),
            ]
            chapters = [
                Chapter(id="chapter_00", start=0.0, end=4.0, index=0, label="mixed", confidence=0.6)
            ]
            fallback_candidates = [
                Candidate(
                    id="reaction:0:200",
                    start=0.0,
                    end=8.0,
                    source="reaction",
                    anchor_time=2.5,
                    score_hint=3.0,
                )
            ]
            enrich_chat_args: list[Path | None] = []
            job_updates: list[dict] = []

            def fake_build_features(
                candidates,
                _segments,
                _profile,
                use_llm_semantics=True,
                llm_candidate_ids=None,
                llm_provider=None,
                chapters=None,
                strategy=None,
            ):
                del use_llm_semantics, llm_candidate_ids, llm_provider, chapters, strategy
                return [
                    CandidateFeatures(
                        candidate_id=candidate.id,
                        hook_strength=0.4,
                        max_hype=2.0,
                        payoff_score=2.0,
                        context_efficiency=0.5,
                        source=candidate.source,
                    )
                    for candidate in candidates
                ]

            def fake_rank_candidates(candidates, features, profile="general", strategy=None):
                del strategy
                feature_map = {feature.candidate_id: feature for feature in features}
                return [
                    RankedCandidate(
                        candidate=candidate,
                        features=feature_map[candidate.id],
                        rank_score=4.0,
                        rationale=["fallback"],
                    )
                    for candidate in candidates
                ]

            def fake_select_diverse_ranked_candidates(ranked_candidates, top_k, min_gap_sec=20.0, strategy=None):
                del min_gap_sec, strategy
                return ranked_candidates[:top_k]

            def fake_refine_candidates(ranked_candidates, _segments, limit=5, profile="general", strategy=None):
                del strategy
                return [
                    {
                        "start_time": ranked_candidates[0].candidate.start,
                        "end_time": ranked_candidates[0].candidate.end,
                        "viral_score": 4.0,
                        "reason": "fallback moment",
                        "rank": 1,
                        "ranking_mode": "heuristic",
                        "candidate_id": ranked_candidates[0].candidate.id,
                    }
                ]

            def fake_enrich_chat(segments, chat_path):
                enrich_chat_args.append(chat_path)
                return segments

            def fake_update_job(job_id: str, **kwargs) -> None:
                job_updates.append({"job_id": job_id, **kwargs})

            def fake_create_task(coro):
                coro.close()
                return None

            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline, "update_job", side_effect=fake_update_job))
                stack.enter_context(patch.object(pipeline.audio_extractor, "extract", return_value=audio_path))
                stack.enter_context(patch.object(pipeline.transcriber, "transcribe_with_progress", return_value=transcript))
                stack.enter_context(patch.object(pipeline.audio_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.video_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.audio_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.video_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.audio_features, "diarize_speakers", side_effect=_identity))
                download_chat_mock = stack.enter_context(patch.object(pipeline.chat_analyzer, "download_chat"))
                stack.enter_context(patch.object(pipeline.chat_analyzer, "enrich_segments", side_effect=fake_enrich_chat))
                stack.enter_context(patch.object(pipeline.chapter_segmenter, "segment_chapters", return_value=chapters))
                stack.enter_context(
                    patch.object(pipeline.peak_detector, "detect_peaks", return_value=[])
                )
                stack.enter_context(
                    patch.object(pipeline.peak_detector, "build_peak_candidates", return_value=[])
                )
                stack.enter_context(
                    patch.object(pipeline.candidate_generators, "generate_semantic_candidates", return_value=[])
                )
                stack.enter_context(
                    patch.object(pipeline.candidate_generators, "generate_fallback_candidates", return_value=fallback_candidates)
                )
                stack.enter_context(patch.object(pipeline.feature_builder, "build_features", side_effect=fake_build_features))
                stack.enter_context(patch.object(pipeline.selector, "rank_candidates", side_effect=fake_rank_candidates))
                stack.enter_context(
                    patch.object(
                        pipeline.selector,
                        "select_diverse_ranked_candidates",
                        side_effect=fake_select_diverse_ranked_candidates,
                    )
                )
                stack.enter_context(patch.object(pipeline.clip_judge, "rerank_candidates", side_effect=lambda ranked, *_args, **_kwargs: ranked))
                stack.enter_context(patch.object(pipeline.analyzer, "refine_candidates", side_effect=fake_refine_candidates))
                stack.enter_context(patch.object(pipeline.analyzer, "expand_variants", side_effect=lambda moments, *_args, **_kwargs: moments))
                stack.enter_context(patch.object(pipeline.post_processor, "snap_boundaries", side_effect=lambda moments, _segments: moments))
                stack.enter_context(patch.object(pipeline.post_processor, "enforce_diversity", side_effect=lambda moments: moments))
                stack.enter_context(
                    patch.object(
                        pipeline.clipper,
                        "cut_clips",
                        return_value=[
                            ClipResult(
                                candidate_id="reaction:0:200",
                                filename="clip_01.mp4",
                                download_url="/static/job-slow-chat/clips/clip_01.mp4",
                                start_time=0.0,
                                end_time=8.0,
                                viral_score=4.0,
                                reason="fallback moment",
                                ranking_mode="heuristic",
                            )
                        ],
                    )
                )
                stack.enter_context(patch.object(pipeline.asyncio, "create_task", side_effect=fake_create_task))

                await pipeline.run_pipeline(
                    "job-slow-chat",
                    url="https://www.twitch.tv/videos/123",
                    video_path=video_path,
                    profile="general",
                )

            download_chat_mock.assert_not_called()
            self.assertEqual(enrich_chat_args, [None])
            self.assertTrue(any(update.get("status") == "done" for update in job_updates))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _import_pipeline_with_stubs():
    def _stub_module(name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    _stub_module("services.downloader", download=lambda *_args, **_kwargs: None)
    _stub_module("services.audio_extractor", extract=lambda *_args, **_kwargs: None)
    _stub_module(
        "services.audio_features",
        extract=lambda *_args, **_kwargs: [],
        enrich_segments=lambda segments, *_args, **_kwargs: segments,
        diarize_speakers=lambda segments, *_args, **_kwargs: segments,
    )
    _stub_module(
        "services.video_features",
        extract=lambda *_args, **_kwargs: [],
        enrich_segments=lambda segments, *_args, **_kwargs: segments,
        refine_face_reactions=lambda segments, *_args, **_kwargs: segments,
    )
    _stub_module(
        "services.chat_analyzer",
        download_chat=lambda *_args, **_kwargs: None,
        enrich_segments=lambda segments, *_args, **_kwargs: segments,
    )
    _stub_module("services.chapter_segmenter", segment_chapters=lambda *_args, **_kwargs: [], segments_for_chapter=lambda segments, *_args, **_kwargs: segments)
    _stub_module(
        "services.analyzer",
        refine_candidates=lambda *_args, **_kwargs: [],
        expand_variants=lambda moments, *_args, **_kwargs: moments,
    )
    _stub_module("services.clipper", cut_clips=lambda *_args, **_kwargs: [])
    _stub_module("services.feature_builder", build_features=lambda *_args, **_kwargs: [])
    _stub_module("services.clip_judge", rerank_candidates=lambda ranked, *_args, **_kwargs: ranked)
    _stub_module(
        "services.selector",
        rank_candidates=lambda *_args, **_kwargs: [],
        select_diverse_ranked_candidates=lambda ranked, *_args, **_kwargs: ranked,
        select_top_candidates=lambda *_args, **_kwargs: [],
    )

    sys.modules.pop("pipeline", None)
    return importlib.import_module("pipeline")


if __name__ == "__main__":
    unittest.main()
