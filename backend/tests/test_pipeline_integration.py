import asyncio
import json
import importlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
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


async def _noop_cleanup_later(_job_id: str) -> None:
    return None


@dataclass
class _AudioFeatureFrameStub:
    second: int
    pitch_hz: float = 0.0
    rms_db: float = 0.0
    is_silence: bool = False
    laughter_prob: float = 0.0
    onset_rate: float = 0.0


@dataclass
class _VideoFeatureFrameStub:
    second: int
    scene_change: bool = False
    motion_score: float = 0.0
    face_count: int = 0
    face_size_ratio: float = 0.0
    face_reaction_prob: float = 0.0
    visual_activity: float = 0.0


class _NumpyLikeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class PipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_phase_a_defaults_keep_inventory_enabled(self) -> None:
        pipeline = _import_pipeline_with_stubs()

        self.assertTrue(pipeline.settings.enable_dense_candidate_inventory)

    async def test_run_pipeline_uses_llm_moment_finder_when_enabled(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            audio_path = tmp_path / "audio.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")
            transcript = [
                RichSegment(start=10.0, end=20.0, text="actual hook."),
                RichSegment(start=20.0, end=35.0, text="clear payoff!"),
            ]
            llm_candidate = Candidate(
                id="llm_moment:1:10000:35000",
                start=10.0,
                end=35.0,
                source="llm_moment_finder",
                anchor_time=22.5,
                score_hint=9.5,
                tags=["llm_moment_finder"],
            )
            llm_ranked = RankedCandidate(
                candidate=llm_candidate,
                features=CandidateFeatures.create(
                    candidate_id=llm_candidate.id,
                    duration=25.0,
                    has_llm_semantics=True,
                    source="llm_moment_finder",
                ),
                rank_score=9.5,
                rationale=["llm_moment_finder"],
            )

            def fake_refine_candidates(ranked_candidates, _segments, limit=5, profile="general", strategy=None):
                del limit, profile, strategy
                return [
                    {
                        "start_time": ranked_candidates[0].candidate.start,
                        "end_time": ranked_candidates[0].candidate.end,
                        "viral_score": ranked_candidates[0].rank_score,
                        "reason": "llm moment",
                        "rank": 1,
                        "ranking_mode": "llm_moment_finder",
                        "candidate_id": ranked_candidates[0].candidate.id,
                    }
                ]

            def fake_cut_clips(_video_path, moments, job_id):
                return [
                    ClipResult(
                        candidate_id=moment["candidate_id"],
                        filename="clip_01.mp4",
                        download_url=f"/static/{job_id}/clips/clip_01.mp4",
                        start_time=moment["start_time"],
                        end_time=moment["end_time"],
                        viral_score=moment["viral_score"],
                        reason=moment["reason"],
                        ranking_mode=moment.get("ranking_mode", "llm_moment_finder"),
                    )
                    for moment in moments
                ]

            llm_result = types.SimpleNamespace(
                ranked_candidates=[llm_ranked],
                post_processed_moments=[
                    {
                        "rank": 1,
                        "start_time": 10.0,
                        "end_time": 35.0,
                        "viral_score": 9.5,
                    }
                ],
                prompt="prompt",
                response_text='{"moments":[]}',
                usage={"provider": "fake"},
                summary={"ranked_candidate_count": 1, "post_processed_moment_count": 1},
            )
            job_updates: list[dict] = []

            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline.settings, "enable_clip_judge", False))
                stack.enter_context(patch.object(pipeline.settings, "enable_chat_analysis", False))
                stack.enter_context(patch.object(pipeline.settings, "enable_whole_vod_dense_candidates", False))
                stack.enter_context(patch.object(pipeline.settings, "enable_dense_candidate_inventory", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_transcript_cache_enabled", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_feature_cache_enabled", False))
                stack.enter_context(patch.object(pipeline.settings, "pipeline_overlap_feature_extraction", False))
                stack.enter_context(patch.object(pipeline, "_probe_media_duration_sec", return_value=45.0))
                stack.enter_context(
                    patch.object(
                        pipeline,
                        "update_job",
                        side_effect=lambda job_id, **kwargs: job_updates.append({"job_id": job_id, **kwargs}),
                    )
                )
                stack.enter_context(patch.object(pipeline.audio_extractor, "extract", return_value=audio_path))
                stack.enter_context(patch.object(pipeline.transcriber, "transcribe_with_progress", return_value=transcript))
                stack.enter_context(patch.object(pipeline.audio_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.video_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.audio_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.video_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.chat_analyzer, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.moment_finder, "find_moments", return_value=llm_result))
                stack.enter_context(patch.object(pipeline.analyzer, "refine_candidates", side_effect=fake_refine_candidates))
                stack.enter_context(patch.object(pipeline.analyzer, "expand_variants", side_effect=lambda moments, *_args, **_kwargs: moments))
                stack.enter_context(patch.object(pipeline.post_processor, "snap_boundaries", side_effect=lambda moments, _segments: moments))
                stack.enter_context(patch.object(pipeline.post_processor, "enforce_diversity", side_effect=lambda moments: moments))
                stack.enter_context(patch.object(pipeline.clipper, "cut_clips", side_effect=fake_cut_clips))
                stack.enter_context(patch.object(pipeline, "_cleanup_later", side_effect=_noop_cleanup_later))

                await pipeline.run_pipeline("job-llm", video_path=video_path, profile="jynxzi")

            debug_dir = tmp_path / "job-llm" / "debug"
            stage = json.loads((debug_dir / "llm_moment_finder_stage.json").read_text(encoding="utf-8"))
            ranked_rows = _read_jsonl(debug_dir / "ranked_candidates_pre_judge.jsonl")

            self.assertEqual(stage["status"], "ok")
            self.assertEqual(ranked_rows[0]["candidate"]["source"], "llm_moment_finder")
            self.assertTrue(
                any(update.get("status") == "analyzing" and update.get("progress_pct") == 73 for update in job_updates)
            )
            self.assertFalse(any(update.get("status") == "finding_moments" for update in job_updates))
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            job_dir = tmp_path / "frontend-job"
            clips_dir = job_dir / "clips"
            debug_dir = job_dir / "debug"
            chunks_dir = job_dir / "chunks"
            clips_dir.mkdir(parents=True)
            debug_dir.mkdir(parents=True)
            chunks_dir.mkdir(parents=True)
            video_path = job_dir / "video.mp4"
            audio_path = job_dir / "audio.wav"
            clip_path = clips_dir / "clip_01.mp4"
            debug_path = debug_dir / "candidate_funnel_summary.json"
            chunk_path = chunks_dir / "chunk_000.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")
            clip_path.write_bytes(b"clip")
            debug_path.write_text("{}", encoding="utf-8")
            chunk_path.write_bytes(b"chunk")

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline.settings, "frontend_job_cache_enabled", False):
                pipeline._cleanup_serving_source_media(
                    job_id="frontend-job",
                    video_path=video_path,
                    audio_path=audio_path,
                )

            self.assertFalse(video_path.exists())
            self.assertFalse(audio_path.exists())
            self.assertFalse(chunks_dir.exists())
            self.assertTrue(clip_path.exists())
            self.assertTrue(debug_path.exists())

    async def test_serving_source_media_cleanup_preserves_training_vod(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            job_dir = tmp_path / "twitch_vod_123"
            job_dir.mkdir(parents=True)
            video_path = job_dir / "video.mp4"
            audio_path = job_dir / "audio.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline.settings, "frontend_job_cache_enabled", False):
                pipeline._cleanup_serving_source_media(
                    job_id="twitch_vod_123",
                    video_path=video_path,
                    audio_path=audio_path,
                )

            self.assertTrue(video_path.exists())
            self.assertTrue(audio_path.exists())

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

    async def test_frontend_cache_setting_disables_serving_job_cache_keys(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with patch.object(pipeline.settings, "frontend_job_cache_enabled", False):
            self.assertFalse(pipeline._frontend_cache_enabled_for_job("frontend-job"))
            self.assertTrue(pipeline._frontend_cache_enabled_for_job("twitch_vod_123"))

    async def test_frontend_transcript_cache_uses_stable_twitch_vod_key_without_media_cache(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with patch.object(pipeline.settings, "frontend_job_cache_enabled", False), \
             patch.object(pipeline.settings, "frontend_transcript_cache_enabled", True):
            self.assertFalse(pipeline._frontend_cache_enabled_for_job("frontend-job"))
            self.assertTrue(
                pipeline._transcript_cache_enabled_for_job(
                    "frontend-job",
                    "https://www.twitch.tv/videos/123",
                )
            )
            self.assertEqual(
                pipeline._transcript_cache_source("https://www.twitch.tv/videos/123", Path("/tmp/job/video.mp4")),
                "twitch_vod:123",
            )

        with patch.object(pipeline.settings, "frontend_job_cache_enabled", False), \
             patch.object(pipeline.settings, "frontend_transcript_cache_enabled", False):
            self.assertFalse(
                pipeline._transcript_cache_enabled_for_job(
                    "frontend-job",
                    "https://www.twitch.tv/videos/123",
                )
            )

    async def test_frontend_feature_cache_uses_stable_twitch_vod_key_without_media_cache(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with patch.object(pipeline.settings, "frontend_job_cache_enabled", False), \
             patch.object(pipeline.settings, "frontend_feature_cache_enabled", True):
            self.assertTrue(
                pipeline._feature_cache_enabled_for_job(
                    "frontend-job",
                    "https://www.twitch.tv/videos/123",
                )
            )
            self.assertEqual(
                pipeline._feature_cache_source(
                    job_id="frontend-job",
                    url="https://www.twitch.tv/videos/123",
                    video_path=Path("/tmp/job/video.mp4"),
                ),
                "twitch_vod:123",
            )
            self.assertIsNone(
                pipeline._feature_cache_source(
                    job_id="frontend-job",
                    url="https://www.youtube.com/watch?v=abc",
                    video_path=Path("/tmp/job/video.mp4"),
                )
            )

        with patch.object(pipeline.settings, "frontend_job_cache_enabled", False), \
             patch.object(pipeline.settings, "frontend_feature_cache_enabled", False):
            self.assertFalse(
                pipeline._feature_cache_enabled_for_job(
                    "frontend-job",
                    "https://www.twitch.tv/videos/123",
                )
            )
    async def test_judge_recall_reserve_score_prefers_context_complete_variant(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        context_candidate = Candidate(
            id="dense_inventory:60:1:0:1000",
            start=10.0,
            end=70.0,
            source="dense_inventory",
            anchor_time=35.0,
            score_hint=2.0,
        )
        weak_candidate = Candidate(
            id="dense_inventory:60:1:1:2000",
            start=20.0,
            end=80.0,
            source="dense_inventory",
            anchor_time=45.0,
            score_hint=4.0,
        )
        context_features = CandidateFeatures(
            candidate_id=context_candidate.id,
            context_completeness_score=0.9,
            context_strength_score=0.85,
            setup_present_score=1.0,
            payoff_present_score=1.0,
            context_pattern_score=0.8,
            context_pattern_similarity_margin=0.08,
            payoff_score=7.5,
            projected_retention_score=0.6,
            standalone_score=0.75,
        )
        weak_features = CandidateFeatures(
            candidate_id=weak_candidate.id,
            context_completeness_score=0.2,
            context_strength_score=0.2,
            setup_present_score=0.1,
            payoff_present_score=0.25,
            payoff_score=3.0,
            projected_retention_score=0.35,
            standalone_score=0.35,
        )

        context_score = pipeline._judge_recall_reserve_score(
            RankedCandidate(context_candidate, context_features, rank_score=-1.0, heuristic_rank_score=-1.0),
            pre_judge_rank=120,
            pool_k=240,
        )
        weak_score = pipeline._judge_recall_reserve_score(
            RankedCandidate(weak_candidate, weak_features, rank_score=1.0, heuristic_rank_score=1.0),
            pre_judge_rank=80,
            pool_k=240,
        )

        self.assertGreater(context_score, weak_score)

    async def test_audio_feature_cache_hit_skips_audio_extraction(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            cache_source = "twitch_vod:123"

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline, "_probe_media_duration_sec", return_value=12.0):
                metadata = pipeline._feature_cache_metadata("audio", cache_source, video_path)
                pipeline._write_feature_frame_cache(
                    cache_source,
                    "audio",
                    [_AudioFeatureFrameStub(second=3, rms_db=-12.5)],
                    metadata,
                )
                with patch.object(
                    pipeline.audio_extractor,
                    "extract",
                    side_effect=AssertionError("audio should not be extracted on cache hit"),
                ), patch.object(
                    pipeline.audio_features,
                    "extract",
                    side_effect=AssertionError("audio features should not be recomputed on cache hit"),
                ):
                    frames, audio_path = pipeline._extract_audio_features_cached(
                        video_path,
                        audio_path=None,
                        job_id="frontend-job",
                        cache_source=cache_source,
                    )

            self.assertIsNone(audio_path)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].second, 3)
            self.assertEqual(frames[0].rms_db, -12.5)

    async def test_audio_cache_reuses_extracted_wav_between_jobs(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            cache_source = "twitch_vod:123"

            def fake_extract(_video_path: Path, job_id: str) -> Path:
                audio_path = tmp_path / job_id / "audio.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"RIFF" + (b"\0" * 128))
                return audio_path

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline, "_probe_media_duration_sec", return_value=12.0), \
                 patch.object(pipeline.audio_extractor, "extract", side_effect=fake_extract) as extract_mock:
                first = pipeline._extract_audio_cached(
                    video_path,
                    "job-one",
                    cache_source=cache_source,
                )
                second = pipeline._extract_audio_cached(
                    video_path,
                    "job-two",
                    cache_source=cache_source,
                )

            extract_mock.assert_called_once()
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(second.read_bytes(), b"RIFF" + (b"\0" * 128))

    async def test_video_feature_cache_miss_writes_reusable_frames(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            cache_source = "twitch_vod:123"
            extracted = [_VideoFeatureFrameStub(second=4, motion_score=0.7)]

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline, "_probe_media_duration_sec", return_value=12.0), \
                 patch.object(pipeline.video_features, "extract", return_value=extracted) as extract_mock:
                first = pipeline._extract_video_features_cached(video_path, cache_source=cache_source)
                second = pipeline._extract_video_features_cached(video_path, cache_source=cache_source)

            extract_mock.assert_called_once()
            self.assertEqual(first[0].motion_score, 0.7)
            self.assertEqual(second[0].motion_score, 0.7)

    async def test_video_feature_cache_hit_is_used_when_long_vod_extraction_is_disabled(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            cache_source = "twitch_vod:123"

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline, "_probe_media_duration_sec", return_value=100_000.0):
                metadata = pipeline._feature_cache_metadata("video", cache_source, video_path)
                pipeline._write_feature_frame_cache(
                    cache_source,
                    "video",
                    [_VideoFeatureFrameStub(second=9, motion_score=0.42)],
                    metadata,
                )
                with patch.object(
                    pipeline.video_features,
                    "extract",
                    side_effect=AssertionError("video features should not be recomputed on cache hit"),
                ):
                    cached = pipeline._extract_video_features_cached(
                        video_path,
                        cache_source=cache_source,
                        allow_extract=False,
                    )

            self.assertEqual(len(cached), 1)
            self.assertEqual(cached[0].second, 9)
            self.assertEqual(cached[0].motion_score, 0.42)

    async def test_video_feature_cache_skip_avoids_long_vod_extraction_when_cache_missing(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline, "_probe_media_duration_sec", return_value=100_000.0), \
                 patch.object(
                     pipeline.video_features,
                     "extract",
                     side_effect=AssertionError("long VOD visual extraction should be skipped"),
                 ):
                frames = pipeline._extract_video_features_cached(
                    video_path,
                    cache_source="twitch_vod:missing",
                    allow_extract=False,
                )

            self.assertEqual(frames, [])

    async def test_feature_cache_write_normalizes_numpy_like_scalars(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            cache_source = "twitch_vod:123"
            frames = [
                _VideoFeatureFrameStub(
                    second=4,
                    scene_change=_NumpyLikeScalar(True),
                    motion_score=_NumpyLikeScalar(0.7),
                    face_count=_NumpyLikeScalar(2),
                )
            ]

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline, "_probe_media_duration_sec", return_value=12.0):
                metadata = pipeline._feature_cache_metadata("video", cache_source, video_path)
                metadata["native_bool"] = _NumpyLikeScalar(False)
                pipeline._write_feature_frame_cache(cache_source, "video", frames, metadata)
                cached = pipeline._load_feature_frame_cache(
                    cache_source,
                    "video",
                    _VideoFeatureFrameStub,
                    pipeline._feature_cache_metadata("video", cache_source, video_path) | {"native_bool": False},
                )

            self.assertIsNotNone(cached)
            self.assertEqual(len(cached), 1)
            self.assertIs(cached[0].scene_change, True)
            self.assertEqual(cached[0].motion_score, 0.7)
            self.assertEqual(cached[0].face_count, 2)

    async def test_judge_recall_reserve_promotes_deeper_feature_candidates(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        ranked = []
        for index in range(1, 9):
            candidate_id = f"dense:{index}"
            candidate = Candidate(
                id=candidate_id,
                start=float(index * 100),
                end=float(index * 100 + 20),
                source="dense_inventory",
                anchor_time=float(index * 100 + 10),
                score_hint=1.0,
            )
            ranked.append(
                RankedCandidate(
                    candidate=candidate,
                    features=CandidateFeatures(
                        candidate_id=candidate_id,
                        source="dense_inventory",
                        payoff_score=4.0 if index >= 6 else 1.0,
                        projected_retention_score=0.6 if index >= 6 else 0.1,
                    ),
                    rank_score=10.0 - index,
                    heuristic_rank_score=6.0,
                    rationale=["ranker"],
                )
            )

        with patch.object(pipeline.settings, "judge_recall_reserve_enabled", True), \
             patch.object(pipeline.settings, "judge_recall_reserve_slots", 2), \
             patch.object(pipeline.settings, "judge_recall_reserve_pool_k", 8), \
             patch.object(pipeline.settings, "judge_recall_reserve_min_score", 0.1):
            judge_input, summary = pipeline._build_judge_recall_reserved_order(
                ranked,
                [],
                4,
            )

        self.assertEqual([item.candidate.id for item in judge_input[:2]], ["dense:1", "dense:2"])
        self.assertEqual({"dense:6", "dense:7"}, {item.candidate.id for item in judge_input[2:4]})
        self.assertEqual(summary["normal_slots"], 2)
        self.assertEqual(summary["reserve_slots"], 2)
        self.assertEqual(summary["selected"], 2)
        self.assertTrue(all("judge_recall_reserved" in item.rationale for item in judge_input[2:4]))
    async def test_serving_cross_platform_recall_disabled_does_not_load_oracle_rows(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "positives.jsonl"
            rows_path.write_text(
                json.dumps(
                    {
                        "vod_id": "123",
                        "window_start": 10.0,
                        "window_end": 20.0,
                        "transcript_similarity": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(pipeline.settings, "serving_cross_platform_recall_enabled", False), \
                 patch.object(pipeline.settings, "serving_cross_platform_positive_rows_path", str(rows_path)):
                rows, summary = pipeline._load_cross_platform_positive_windows(
                    "jynxzi",
                    "https://www.twitch.tv/videos/123",
                )

        self.assertEqual(rows, [])
        self.assertEqual(summary["status"], "disabled")
        self.assertEqual(summary["matched_rows"], 0)

    async def test_cross_platform_positive_windows_can_load_for_diagnostics_without_serving_boost(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "positives.jsonl"
            rows_path.write_text(
                json.dumps(
                    {
                        "vod_id": "123",
                        "window_start": 10.0,
                        "window_end": 20.0,
                        "transcript_similarity": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(pipeline.settings, "serving_cross_platform_recall_enabled", False), \
                 patch.object(pipeline.settings, "serving_cross_platform_positive_rows_path", str(rows_path)):
                rows, summary = pipeline._load_cross_platform_positive_windows(
                    "jynxzi",
                    "https://www.twitch.tv/videos/123",
                    force_load=True,
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["status"], "loaded")
        self.assertFalse(summary["enabled"])
        self.assertTrue(summary["active"])
        self.assertTrue(summary["diagnostic_only"])
        self.assertEqual(summary["matched_rows"], 1)
    async def test_clip_judge_timeout_degrades_to_pre_judge_candidates_even_when_fail_fast(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        loop = asyncio.get_running_loop()
        never_finishes = loop.create_future()
        fallback = [
            RankedCandidate(
                candidate=Candidate(id="peak:0:100", start=0.0, end=10.0, source="peak", anchor_time=5.0),
                features=CandidateFeatures(candidate_id="peak:0:100", source="peak"),
                rank_score=8.0,
                rationale=["pre_judge"],
            )
        ]

        with patch.object(pipeline.settings, "clip_judge_fail_fast", True):
            result, degraded = await pipeline._await_clip_judge_or_fallback(
                never_finishes,
                timeout_sec=0.001,
                fallback_candidates=fallback,
                job_id="job-timeout",
            )

        never_finishes.cancel()
        self.assertTrue(degraded)
        self.assertEqual(result, fallback)

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
            llm_candidate = Candidate(
                id="llm_moment:1:0:8000",
                start=0.0,
                end=8.0,
                source="llm_moment_finder",
                anchor_time=4.0,
                score_hint=8.0,
            )
            llm_result = types.SimpleNamespace(
                ranked_candidates=[
                    RankedCandidate(
                        candidate=llm_candidate,
                        features=CandidateFeatures.create(candidate_id=llm_candidate.id, duration=8.0),
                        rank_score=8.0,
                        rationale=["llm_moment_finder"],
                    )
                ],
                post_processed_moments=[{"rank": 1, "start_time": 0.0, "end_time": 8.0, "viral_score": 8.0}],
                prompt="prompt",
                response_text='{"moments":[]}',
                usage={"provider": "fake"},
                summary={"ranked_candidate_count": 1, "post_processed_moment_count": 1},
            )
            enrich_chat_args: list[Path | None] = []
            job_updates: list[dict] = []

            def fake_refine_candidates(ranked_candidates, _segments, limit=5, profile="general", strategy=None):
                del strategy
                return [
                    {
                        "start_time": ranked_candidates[0].candidate.start,
                        "end_time": ranked_candidates[0].candidate.end,
                        "viral_score": 8.0,
                        "reason": "llm moment",
                        "rank": 1,
                        "ranking_mode": "llm_moment_finder",
                        "candidate_id": ranked_candidates[0].candidate.id,
                    }
                ]

            def fake_enrich_chat(segments, chat_path):
                enrich_chat_args.append(chat_path)
                return segments

            def fake_update_job(job_id: str, **kwargs) -> None:
                job_updates.append({"job_id": job_id, **kwargs})

            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline.settings, "enable_chat_analysis", False))
                stack.enter_context(patch.object(pipeline.settings, "enable_clip_judge", False))
                stack.enter_context(patch.object(pipeline.settings, "enable_dense_candidate_inventory", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_transcript_cache_enabled", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_feature_cache_enabled", False))
                stack.enter_context(patch.object(pipeline.settings, "hf_token", "hf_secret"))
                stack.enter_context(patch.object(pipeline.settings, "pipeline_enable_diarization", False))
                stack.enter_context(patch.object(pipeline, "_probe_media_duration_sec", return_value=12.0))
                stack.enter_context(patch.object(pipeline, "update_job", side_effect=fake_update_job))
                stack.enter_context(patch.object(pipeline.audio_extractor, "extract", return_value=audio_path))
                transcribe_mock = stack.enter_context(
                    patch.object(pipeline.transcriber, "transcribe_with_progress", return_value=transcript)
                )
                stack.enter_context(patch.object(pipeline.audio_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.video_features, "extract", return_value=[]))
                stack.enter_context(patch.object(pipeline.audio_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.video_features, "enrich_segments", side_effect=_identity))
                diarize_mock = stack.enter_context(
                    patch.object(pipeline.audio_features, "diarize_speakers", side_effect=_identity)
                )
                download_chat_mock = stack.enter_context(patch.object(pipeline.chat_analyzer, "download_chat"))
                stack.enter_context(patch.object(pipeline.chat_analyzer, "enrich_segments", side_effect=fake_enrich_chat))
                stack.enter_context(patch.object(pipeline.moment_finder, "find_moments", return_value=llm_result))
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
                                candidate_id=llm_candidate.id,
                                filename="clip_01.mp4",
                                download_url="/static/job-slow-chat/clips/clip_01.mp4",
                                start_time=0.0,
                                end_time=8.0,
                                viral_score=8.0,
                                reason="llm moment",
                                ranking_mode="llm_moment_finder",
                            )
                        ],
                    )
                )
                stack.enter_context(patch.object(pipeline, "_cleanup_later", side_effect=_noop_cleanup_later))

                await pipeline.run_pipeline(
                    "job-slow-chat",
                    url="https://www.twitch.tv/videos/123",
                    video_path=video_path,
                    profile="general",
                )

            download_chat_mock.assert_not_called()
            self.assertFalse(transcribe_mock.call_args.kwargs["enable_diarization"])
            diarize_mock.assert_not_called()
            self.assertEqual(enrich_chat_args, [None])
            self.assertTrue(any(update.get("status") == "done" for update in job_updates))

    async def test_run_pipeline_uses_complete_transcript_cache_without_retranscribing(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")

            transcript = [
                RichSegment(start=0.0, end=2.0, text="cached setup", words_per_second=1.0),
                RichSegment(start=2.0, end=4.0, text="cached payoff", words_per_second=2.0),
            ]
            llm_candidate = Candidate(
                id="llm_moment:1:0:8000",
                start=0.0,
                end=8.0,
                source="llm_moment_finder",
                anchor_time=4.0,
                score_hint=8.0,
            )
            llm_result = types.SimpleNamespace(
                ranked_candidates=[
                    RankedCandidate(
                        candidate=llm_candidate,
                        features=CandidateFeatures.create(candidate_id=llm_candidate.id, duration=8.0),
                        rank_score=8.0,
                        rationale=["llm_moment_finder"],
                    )
                ],
                post_processed_moments=[{"rank": 1, "start_time": 0.0, "end_time": 8.0, "viral_score": 8.0}],
                prompt="prompt",
                response_text='{"moments":[]}',
                usage={"provider": "fake"},
                summary={"ranked_candidate_count": 1, "post_processed_moment_count": 1},
            )
            job_updates: list[dict] = []

            def fake_refine_candidates(ranked_candidates, *_args, **_kwargs):
                return [
                    {
                        "start_time": ranked_candidates[0].candidate.start,
                        "end_time": ranked_candidates[0].candidate.end,
                        "viral_score": 8.0,
                        "reason": "cached transcript moment",
                        "rank": 1,
                        "ranking_mode": "llm_moment_finder",
                        "candidate_id": ranked_candidates[0].candidate.id,
                    }
                ]

            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline.settings, "enable_chat_analysis", False))
                stack.enter_context(patch.object(pipeline.settings, "enable_clip_judge", False))
                stack.enter_context(patch.object(pipeline.settings, "enable_dense_candidate_inventory", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_job_cache_enabled", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_transcript_cache_enabled", True))
                stack.enter_context(patch.object(pipeline.settings, "pipeline_full_visual_feature_max_duration_sec", 7200.0))
                stack.enter_context(patch.object(pipeline, "update_job", side_effect=lambda job_id, **kwargs: job_updates.append({"job_id": job_id, **kwargs})))
                stack.enter_context(patch.object(pipeline, "_probe_media_duration_sec", return_value=16720.0))
                cache_mock = stack.enter_context(
                    patch.object(pipeline, "_load_complete_cached_transcript", return_value=transcript)
                )
                transcribe_mock = stack.enter_context(
                    patch.object(pipeline.transcriber, "transcribe_with_progress", side_effect=AssertionError("transcription should be skipped"))
                )
                audio_extract_mock = stack.enter_context(
                    patch.object(pipeline.audio_extractor, "extract", return_value=tmp_path / "audio.wav")
                )
                audio_features_mock = stack.enter_context(
                    patch.object(pipeline.audio_features, "extract", return_value=[])
                )
                visual_features_mock = stack.enter_context(
                    patch.object(pipeline.video_features, "extract", side_effect=AssertionError("full-VOD visual features should be skipped"))
                )
                stack.enter_context(patch.object(pipeline.audio_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.video_features, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.chat_analyzer, "enrich_segments", side_effect=_identity))
                stack.enter_context(patch.object(pipeline.moment_finder, "find_moments", return_value=llm_result))
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
                                candidate_id=llm_candidate.id,
                                filename="clip_01.mp4",
                                download_url="/static/job-cache/clips/clip_01.mp4",
                                start_time=0.0,
                                end_time=8.0,
                                viral_score=8.0,
                                reason="cached transcript moment",
                                ranking_mode="llm_moment_finder",
                            )
                        ],
                    )
                )
                stack.enter_context(patch.object(pipeline, "_cleanup_later", side_effect=_noop_cleanup_later))

                await pipeline.run_pipeline(
                    "job-cache",
                    url="https://www.twitch.tv/videos/123",
                    video_path=video_path,
                    profile="jynxzi",
                )

            cache_mock.assert_called_once()
            transcribe_mock.assert_not_called()
            audio_extract_mock.assert_called_once()
            audio_features_mock.assert_called_once()
            visual_features_mock.assert_not_called()
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

    _stub_module(
        "services.downloader",
        DownloadCancelledError=type("DownloadCancelledError", (RuntimeError,), {}),
        download=lambda *_args, **_kwargs: None,
    )
    _stub_module("services.audio_extractor", extract=lambda *_args, **_kwargs: None)
    _stub_module(
        "services.transcriber",
        RichSegment=RichSegment,
        transcribe_with_progress=lambda *_args, **_kwargs: [],
    )
    _stub_module(
        "services.audio_features",
        AudioFeatureFrame=_AudioFeatureFrameStub,
        extract=lambda *_args, **_kwargs: [],
        enrich_segments=lambda segments, *_args, **_kwargs: segments,
        diarize_speakers=lambda segments, *_args, **_kwargs: segments,
    )
    _stub_module(
        "services.video_features",
        VideoFeatureFrame=_VideoFeatureFrameStub,
        extract=lambda *_args, **_kwargs: [],
        enrich_segments=lambda segments, *_args, **_kwargs: segments,
        refine_face_reactions=lambda segments, *_args, **_kwargs: segments,
        runtime_summary=lambda: {"face_detector": "stub", "expression_sample_sec": 3},
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
    _stub_module(
        "services.clipper",
        cut_clips=lambda *_args, **_kwargs: [],
        cut_review_clips=lambda _video_path, rows, *_args, **_kwargs: rows,
    )
    _stub_module("services.feature_builder", build_features=lambda *_args, **_kwargs: [])
    _stub_module("services.semantic_candidates", preload_model=lambda: True)
    _stub_module("services.clip_judge", rerank_candidates=lambda ranked, *_args, **_kwargs: ranked)
    _stub_module(
        "services.selector",
        rank_candidates=lambda *_args, **_kwargs: [],
        select_diverse_ranked_candidates=lambda ranked, *_args, **_kwargs: ranked,
        select_top_candidates=lambda *_args, **_kwargs: [],
        apply_serving_quality_gate=lambda ranked, *_args, **_kwargs: types.SimpleNamespace(
            kept=ranked,
            audit_rows=[],
            summary={
                "input_candidates": len(ranked),
                "kept_candidates": len(ranked),
                "rejected_candidates": 0,
            },
        ),
        select_post_judge_final_candidates=lambda ranked, *_args, **_kwargs: types.SimpleNamespace(
            selected=ranked,
            summary={
                "input_candidates": len(ranked),
                "selected_candidates": len(ranked),
            },
        ),
    )

    sys.modules.pop("pipeline", None)
    pipeline = importlib.import_module("pipeline")
    return pipeline


if __name__ == "__main__":
    unittest.main()
