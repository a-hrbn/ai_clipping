import importlib
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

# Torch-free harness. The Conductor's Producers are substituted with fakes here
# (each Producer's real behavior is covered by tests/test_producers_*.py), so
# this integration suite exercises pure orchestration and needs none of the ML
# wheels. The defensive stubs below keep any transitive import light; the real
# chat_signal / audio_energy (requests / numpy only) import fine.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))
sys.modules.setdefault("ollama", types.SimpleNamespace(Client=lambda *_args, **_kwargs: None))


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


# An opaque, torch-free transcript handed between the faked Producers — its
# element type is irrelevant because every Producer that reads it is a fake.
_SEGMENTS = [types.SimpleNamespace(start=0.0, end=5.0, text="segment")]


class _FakeAudioResource:
    def __init__(self, **_kwargs) -> None:
        self.teardowns = 0

    def path(self):
        return None

    def materialized(self):
        return None

    def teardown(self) -> None:
        self.teardowns += 1


class _FakeTranscriptResource:
    def __init__(self, **_kwargs) -> None:
        pass

    def segments(self, *, progress=lambda _f: None):
        progress(1.0)
        return _SEGMENTS

    def from_cache(self) -> bool:
        return False


class _Passthrough:
    """VoiceEnergy / ChatEnrichment stand-in: returns the transcript unchanged."""

    def __init__(self, **_kwargs) -> None:
        pass

    def apply(self, transcript):
        return transcript


class _FakeMomentFinding:
    def __init__(self, *, result, raises=None) -> None:
        self._result = result
        self._raises = raises

    def find(self, _transcript, *, progress=lambda _f: None):
        progress(1.0)
        if self._raises is not None:
            raise self._raises
        return self._result


class _FakeClipping:
    def __init__(self, *, clips) -> None:
        self._clips = clips

    def run(self, _video_path, _result, *, progress=lambda _f: None):
        progress(1.0)
        return self._clips


def _patch_conductor_producers(stack, pipeline, *, result, clips, moments_raises=None) -> None:
    """Swap every Producer the Conductor builds for a fake, so the test exercises
    orchestration only (Producer behavior lives in tests/test_producers_*.py)."""
    stack.enter_context(patch.object(pipeline, "AudioResource", _FakeAudioResource))
    stack.enter_context(patch.object(pipeline, "TranscriptResource", _FakeTranscriptResource))
    stack.enter_context(patch.object(pipeline, "VoiceEnergy", _Passthrough))
    stack.enter_context(patch.object(pipeline, "ChatEnrichment", _Passthrough))
    stack.enter_context(
        patch.object(
            pipeline, "MomentFinding",
            lambda **_kw: _FakeMomentFinding(result=result, raises=moments_raises),
        )
    )
    stack.enter_context(patch.object(pipeline, "Clipping", lambda **_kw: _FakeClipping(clips=clips)))


def _conductor_stack(stack, pipeline, tmp_path, job_updates) -> None:
    stack.enter_context(patch.object(pipeline.settings, "storage_dir", str(tmp_path)))
    stack.enter_context(patch.object(pipeline.settings, "frontend_transcript_cache_enabled", False))
    stack.enter_context(patch.object(pipeline.settings, "frontend_job_cache_enabled", False))
    stack.enter_context(
        patch.object(
            pipeline, "update_job",
            side_effect=lambda job_id, **kwargs: job_updates.append({"job_id": job_id, **kwargs}),
        )
    )


_MISSING = object()

_STUBBED_MODULE_NAMES = (
    "services.downloader",
    "services.audio_extractor",
    "services.transcriber",
    "services.clipper",
    "services.clip_strategy",
    "services.moment_finder",
    "pipeline",
)


class _PipelineStubTestCase(unittest.IsolatedAsyncioTestCase):
    """Restore the sys.modules entries and parent-package attributes that
    _import_pipeline_with_stubs() clobbers, so the stubs don't leak into later
    test modules (e.g. test_transcriber) and corrupt them."""

    def setUp(self) -> None:
        self._saved = []
        for name in _STUBBED_MODULE_NAMES:
            module = sys.modules.get(name, _MISSING)
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name) if parent_name else None
            attr = getattr(parent, child, _MISSING) if parent is not None else _MISSING
            self._saved.append((name, module, parent, child, attr))

    def tearDown(self) -> None:
        for name, module, parent, child, attr in self._saved:
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
            if parent is not None:
                if attr is _MISSING:
                    if hasattr(parent, child):
                        delattr(parent, child)
                else:
                    setattr(parent, child, attr)


class ConductorOrchestrationTests(_PipelineStubTestCase):
    async def test_runs_producers_in_band_order_to_done(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            result = types.SimpleNamespace(
                post_processed_moments=[{"start_time": 0.0, "end_time": 5.0}]
            )
            clips = [types.SimpleNamespace(filename="clip_01.mp4")]
            job_updates: list[dict] = []

            with ExitStack() as stack:
                _conductor_stack(stack, pipeline, tmp_path, job_updates)
                _patch_conductor_producers(stack, pipeline, result=result, clips=clips)

                await pipeline.run_pipeline("lean-job", video_path=video_path, profile="jynxzi")

            statuses = [u.get("status") for u in job_updates if u.get("status")]
            band_order = [s for s in statuses if s in {
                "transcribing", "extracting_audio", "analyzing", "clipping", "done"
            }]
            seen: list[str] = []
            for s in band_order:
                if s not in seen:
                    seen.append(s)
            self.assertEqual(
                seen, ["transcribing", "extracting_audio", "analyzing", "clipping", "done"]
            )
            done = [u for u in job_updates if u.get("status") == "done"][-1]
            self.assertEqual(done["progress_pct"], 100)
            self.assertEqual(done["clips"], clips)

    async def test_run_writes_nothing_outside_storage_dir(self) -> None:
        # The run's only transcript copy is segments.jsonl in the job artifact
        # store; nothing lands in a repo-level transcripts/ dir nobody reads.
        pipeline = _import_pipeline_with_stubs()
        job_id = "no-stray-writes-job"
        stray_dir = Path(pipeline.__file__).resolve().parents[1] / "transcripts"
        stray = [stray_dir / f"{job_id}.txt", stray_dir / f"{job_id}.jsonl"]
        for path in stray:
            path.unlink(missing_ok=True)
        self.addCleanup(lambda: [path.unlink(missing_ok=True) for path in stray])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            job_updates: list[dict] = []

            with ExitStack() as stack:
                _conductor_stack(stack, pipeline, tmp_path, job_updates)
                _patch_conductor_producers(
                    stack, pipeline,
                    result=types.SimpleNamespace(post_processed_moments=[]), clips=[],
                )

                await pipeline.run_pipeline(job_id, video_path=video_path, profile="jynxzi")

            self.assertEqual(job_updates[-1].get("status"), "done")
            self.assertEqual([path for path in stray if path.exists()], [])

    async def test_done_job_has_one_feedback_row_per_delivered_clip(self) -> None:
        # The REAL Clipping producer runs (only the ffmpeg cutter is faked), and the
        # feedback rows are read back from disk through the job-artifact store — the
        # same reader the feedback/review routes use.
        from models import ClipResult
        from services.job_artifact_store import RealFsJobArtifactStore

        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            result = types.SimpleNamespace(post_processed_moments=[
                {"start_time": 10.0, "end_time": 30.0, "viral_score": 9.0,
                 "composite_score": 8.5, "reason": "a", "hook": 7.0},
                {"start_time": 40.0, "end_time": 60.0, "viral_score": 6.0, "reason": "b"},
                {"start_time": 70.0, "end_time": 90.0, "viral_score": 5.0, "reason": "c"},
            ])

            def fake_cut(_video, moments, job_id, *, progress_cb=None):
                # Moment 2 fails to cut: only clips 1 and 3 are delivered. Clip 1's
                # final window differs from the proposed one (cutter clamping).
                delivered = [(moments[0], 10.0, 25.0), (moments[2], 70.0, 90.0)]
                return [
                    ClipResult(
                        candidate_id=m["candidate_id"], filename=f"{m['candidate_id']}.mp4",
                        download_url="/x", start_time=start, end_time=end,
                        viral_score=m["viral_score"], reason=m["reason"],
                        composite_score=m["composite_score"],
                    )
                    for m, start, end in delivered
                ]

            job_updates: list[dict] = []
            with ExitStack() as stack:
                _conductor_stack(stack, pipeline, tmp_path, job_updates)
                stack.enter_context(patch.object(pipeline.settings, "clip_top_k", 0))
                stack.enter_context(patch.object(pipeline, "AudioResource", _FakeAudioResource))
                stack.enter_context(patch.object(pipeline, "TranscriptResource", _FakeTranscriptResource))
                stack.enter_context(patch.object(pipeline, "VoiceEnergy", _Passthrough))
                stack.enter_context(patch.object(pipeline, "ChatEnrichment", _Passthrough))
                stack.enter_context(
                    patch.object(pipeline, "MomentFinding", lambda **_kw: _FakeMomentFinding(result=result))
                )
                stack.enter_context(patch.object(pipeline.clipper, "cut_clips", fake_cut))

                await pipeline.run_pipeline("lean-job", video_path=video_path, profile="jynxzi")

            self.assertEqual([u.get("status") for u in job_updates if u.get("status")][-1], "done")
            rows = RealFsJobArtifactStore(tmp_path).read_feedback("lean-job")
            by_id = {row["candidate_id"]: row for row in rows}
            self.assertEqual(sorted(by_id), ["llm_moment_01", "llm_moment_03"])
            first = by_id["llm_moment_01"]
            self.assertTrue(first["selected"])
            self.assertEqual(first["job_id"], "lean-job")
            self.assertEqual((first["proposed_start"], first["proposed_end"]), (10.0, 30.0))
            self.assertEqual((first["final_start"], first["final_end"]), (10.0, 25.0))
            self.assertEqual(first["rank"], 1)
            self.assertEqual(first["viral_score"], 9.0)
            self.assertEqual(first["composite_score"], 8.5)
            self.assertEqual(first["projected_hook_score"], 7.0)
            self.assertEqual(by_id["llm_moment_03"]["rank"], 3)

    async def test_producer_failure_sets_failed_status(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            job_updates: list[dict] = []

            with ExitStack() as stack:
                _conductor_stack(stack, pipeline, tmp_path, job_updates)
                _patch_conductor_producers(
                    stack, pipeline, result=None, clips=[],
                    moments_raises=pipeline.moment_finder.MomentFinderError("boom"),
                )

                await pipeline.run_pipeline("fail-job", video_path=video_path, profile="jynxzi")

            statuses = [u.get("status") for u in job_updates]
            self.assertIn("failed", statuses)  # any exception → failed
            self.assertNotIn("done", statuses)
            failed = [u for u in job_updates if u.get("status") == "failed"][-1]
            self.assertEqual(failed["error"], "boom")

    async def test_cancellation_does_not_set_failed_status(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            job_updates: list[dict] = []
            calls = {"n": 0}

            def _cancelled(_job_id: str) -> bool:
                calls["n"] += 1
                return calls["n"] >= 2  # pass entry check, cancel at the first run() between producers

            with ExitStack() as stack:
                _conductor_stack(stack, pipeline, tmp_path, job_updates)
                stack.enter_context(patch.object(pipeline, "is_job_cancelled", side_effect=_cancelled))
                _patch_conductor_producers(
                    stack, pipeline,
                    result=types.SimpleNamespace(post_processed_moments=[]), clips=[],
                )

                await pipeline.run_pipeline("cancel-job", video_path=video_path, profile="jynxzi")

            statuses = [u.get("status") for u in job_updates]
            self.assertNotIn("failed", statuses)  # cancel → no failed status
            self.assertNotIn("done", statuses)


class ConductorCacheSourceTests(_PipelineStubTestCase):
    """Verify the Conductor-owned cache-source helpers directly; these survive
    the Producer extraction unchanged."""

    async def test_frontend_transcript_cache_uses_stable_twitch_vod_key_without_media_cache(self) -> None:
        pipeline = _import_pipeline_with_stubs()
        with patch.object(pipeline.settings, "frontend_job_cache_enabled", False), \
             patch.object(pipeline.settings, "frontend_transcript_cache_enabled", True):
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


def _import_pipeline_with_stubs():
    def _stub_module(name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        # Override the parent-package attribute too: when another test module has
        # already imported the real submodule, `from services import X` resolves
        # to that cached attribute and would bypass this sys.modules stub. Setting
        # the attribute makes the stub win regardless of suite ordering.
        parent_name, _, child = name.rpartition(".")
        if parent_name:
            setattr(importlib.import_module(parent_name), child, module)

    _stub_module(
        "services.downloader",
        DownloadCancelledError=type("DownloadCancelledError", (RuntimeError,), {}),
        download=lambda *_args, **_kwargs: None,
    )
    _stub_module("services.audio_extractor", extract=lambda *_args, **_kwargs: None)
    # The Conductor binds these as injected leaf ops when it constructs the (now
    # faked) Producers, so every name the constructors reference must exist.
    _stub_module(
        "services.transcriber",
        transcribe_with_progress=lambda *_args, **_kwargs: [],
        load_cached_transcript=lambda *_args, **_kwargs: None,
        stable_cache_key_for_source=lambda source: source,
        has_chat_enrichment=lambda *_args, **_kwargs: False,
        load_voice_energy=lambda *_args, **_kwargs: None,
        save_voice_energy=lambda *_args, **_kwargs: None,
        apply_chat_enrichment=lambda *_args, **_kwargs: None,
        save_chat_enrichment=lambda *_args, **_kwargs: None,
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
