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

from models import ClipResult
from services.transcriber import RichSegment


async def _noop_cleanup_later(_job_id: str) -> None:
    return None


class PipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_pipeline_lean_path(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            transcript = [
                RichSegment(start=10.0, end=20.0, text="actual hook."),
                RichSegment(start=20.0, end=35.0, text="clear payoff!"),
            ]
            llm_result = types.SimpleNamespace(
                raw_moments=[],
                ranked_candidates=[object()],
                post_processed_moments=[
                    {"rank": 1, "start_time": 10.0, "end_time": 35.0, "viral_score": 9.5, "reason": "hook"},
                ],
                prompt="prompt text",
                response_text='{"moments":[]}',
                usage={"provider": "gemini", "input_tokens": 100, "output_tokens": 20},
                summary={"ranked_candidate_count": 1, "post_processed_moment_count": 1},
            )

            def fake_cut_clips(_video_path, moments, job_id, **_kwargs):
                return [
                    ClipResult(
                        candidate_id=m["candidate_id"],
                        filename="clip_01.mp4",
                        download_url=f"/static/{job_id}/clips/clip_01.mp4",
                        start_time=m["start_time"],
                        end_time=m["end_time"],
                        viral_score=m["viral_score"],
                        reason=m["reason"],
                        ranking_mode="llm_moment_finder",
                    )
                    for m in moments
                ]

            job_updates: list[dict] = []

            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline.settings, "frontend_transcript_cache_enabled", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_job_cache_enabled", False))
                stack.enter_context(
                    patch.object(
                        pipeline,
                        "update_job",
                        side_effect=lambda job_id, **kwargs: job_updates.append({"job_id": job_id, **kwargs}),
                    )
                )
                stack.enter_context(patch.object(pipeline.audio_extractor, "extract", return_value=tmp_path / "audio.wav"))
                stack.enter_context(patch.object(pipeline.transcriber, "transcribe_with_progress", return_value=transcript))
                stack.enter_context(patch.object(pipeline.moment_finder, "find_moments", return_value=llm_result))
                stack.enter_context(patch.object(pipeline.clipper, "cut_clips", side_effect=fake_cut_clips))
                cleanup_mock = stack.enter_context(
                    patch.object(pipeline, "_cleanup_later", side_effect=_noop_cleanup_later)
                )

                await pipeline.run_pipeline("lean-job", video_path=video_path, profile="jynxzi")

            debug_dir = tmp_path / "lean-job" / "debug"
            stage = json.loads((debug_dir / "llm_moment_finder_stage.json").read_text(encoding="utf-8"))
            moments_rows = _read_jsonl(debug_dir / "llm_moment_finder_moments.jsonl")
            final = json.loads((debug_dir / "final_moments.json").read_text(encoding="utf-8"))

            self.assertEqual(stage["status"], "ok")
            self.assertEqual(len(moments_rows), 1)
            self.assertEqual(moments_rows[0]["start_time"], 10.0)
            self.assertEqual(len(final["moments"]), 1)
            self.assertEqual(final["moments"][0]["candidate_id"], "llm_moment_01")
            self.assertTrue(any(u.get("status") == "done" for u in job_updates))
            cleanup_mock.assert_called_once_with("lean-job")

    async def test_pipeline_debug_job_skips_cleanup_later(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            transcript = [RichSegment(start=0.0, end=10.0, text="moment")]
            llm_result = types.SimpleNamespace(
                raw_moments=[],
                ranked_candidates=[object()],
                post_processed_moments=[{"rank": 1, "start_time": 0.0, "end_time": 10.0, "viral_score": 1.0}],
                prompt="p",
                response_text="{}",
                usage={},
                summary={"ranked_candidate_count": 1, "post_processed_moment_count": 1},
            )
            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline.settings, "frontend_transcript_cache_enabled", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_job_cache_enabled", False))
                stack.enter_context(patch.object(pipeline, "update_job", side_effect=lambda *_a, **_kw: None))
                stack.enter_context(patch.object(pipeline.audio_extractor, "extract", return_value=tmp_path / "audio.wav"))
                stack.enter_context(patch.object(pipeline.transcriber, "transcribe_with_progress", return_value=transcript))
                stack.enter_context(patch.object(pipeline.moment_finder, "find_moments", return_value=llm_result))
                stack.enter_context(patch.object(pipeline.clipper, "cut_clips", return_value=[]))
                cleanup_mock = stack.enter_context(
                    patch.object(pipeline, "_cleanup_later", side_effect=_noop_cleanup_later)
                )

                debug_job_id = "pipeline_debug_wf1_vod123"
                await pipeline.run_pipeline(debug_job_id, video_path=video_path, profile="jynxzi")

            cleanup_mock.assert_not_called()

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

    async def test_serving_source_media_cleanup_preserves_pipeline_debug_job(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            job_id = "pipeline_debug_wf1_vod123"
            job_dir = tmp_path / job_id
            job_dir.mkdir(parents=True)
            video_path = job_dir / "video.mp4"
            audio_path = job_dir / "audio.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline.settings, "frontend_job_cache_enabled", False):
                pipeline._cleanup_serving_source_media(
                    job_id=job_id,
                    video_path=video_path,
                    audio_path=audio_path,
                )

            self.assertTrue(video_path.exists())
            self.assertTrue(audio_path.exists())

    async def test_frontend_cache_setting_disables_serving_job_cache_keys(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with patch.object(pipeline.settings, "frontend_job_cache_enabled", False):
            self.assertFalse(pipeline._frontend_cache_enabled_for_job("frontend-job"))
            self.assertTrue(pipeline._frontend_cache_enabled_for_job("twitch_vod_123"))
            self.assertTrue(pipeline._frontend_cache_enabled_for_job("pipeline_debug_wf1_vod123"))

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
            llm_result = types.SimpleNamespace(
                raw_moments=[],
                ranked_candidates=[object()],
                post_processed_moments=[{"rank": 1, "start_time": 0.0, "end_time": 8.0, "viral_score": 8.0}],
                prompt="p",
                response_text="{}",
                usage={},
                summary={"ranked_candidate_count": 1, "post_processed_moment_count": 1},
            )
            job_updates: list[dict] = []

            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
                stack.enter_context(patch.object(pipeline.settings, "frontend_job_cache_enabled", False))
                stack.enter_context(patch.object(pipeline.settings, "frontend_transcript_cache_enabled", True))
                stack.enter_context(
                    patch.object(pipeline, "update_job", side_effect=lambda job_id, **kwargs: job_updates.append({"job_id": job_id, **kwargs}))
                )
                stack.enter_context(patch.object(pipeline, "_probe_media_duration_sec", return_value=16720.0))
                # TranscriptResource resolves the cache via the injected leaf op;
                # a hit must skip transcription AND never pull the audio.
                cache_mock = stack.enter_context(
                    patch.object(pipeline.transcriber, "load_cached_transcript", return_value=transcript)
                )
                transcribe_mock = stack.enter_context(
                    patch.object(pipeline.transcriber, "transcribe_with_progress", side_effect=AssertionError("transcription should be skipped"))
                )
                stack.enter_context(patch.object(pipeline.moment_finder, "find_moments", return_value=llm_result))
                stack.enter_context(patch.object(pipeline.clipper, "cut_clips", return_value=[]))
                stack.enter_context(patch.object(pipeline, "_cleanup_later", side_effect=_noop_cleanup_later))

                await pipeline.run_pipeline(
                    "job-cache",
                    url="https://www.twitch.tv/videos/123",
                    video_path=video_path,
                    profile="jynxzi",
                )

            cache_mock.assert_called_once()
            transcribe_mock.assert_not_called()
            self.assertTrue(any(u.get("status") == "done" for u in job_updates))


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
        load_cached_transcript=lambda *_args, **_kwargs: None,
        stable_cache_key_for_source=lambda source: source,
        format_transcript=lambda segments: "",
        format_jsonl=lambda segments: "",
    )
    _stub_module(
        "services.clipper",
        cut_clips=lambda *_args, **_kwargs: [],
    )
    _stub_module(
        "services.clip_strategy",
        normalize_strategy=lambda strategy: strategy or {},
    )
    _stub_module(
        "services.moment_finder",
        find_moments=lambda *_args, **_kwargs: None,
        MomentFinderError=type("MomentFinderError", (RuntimeError,), {}),
    )

    sys.modules.pop("pipeline", None)
    pipeline = importlib.import_module("pipeline")
    return pipeline


if __name__ == "__main__":
    unittest.main()
