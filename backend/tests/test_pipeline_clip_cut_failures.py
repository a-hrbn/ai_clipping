"""A broken clip cut must not sink the whole job (#120).

Seam: the Conductor ``run_pipeline``. Download/transcription/LLM are faked (as in
``test_pipeline_integration``); the REAL Clipping Producer and the REAL
``services.clipper.cut_clips`` run, and only the per-clip ffmpeg invocation
(``cut_raw_segment``) is replaced by a fake cutter that fails for chosen windows.
"""
import importlib
import logging
import sys
import tempfile
import types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from tests.test_pipeline_integration import (
    _FakeAudioResource,
    _FakeMomentFinding,
    _FakeTranscriptResource,
    _Passthrough,
    _PipelineStubTestCase,
    _conductor_stack,
    _import_pipeline_with_stubs,
)

_ROWS = [
    {"start_time": 100.0, "end_time": 130.0, "viral_score": 9.0, "reason": "first"},
    {"start_time": 200.0, "end_time": 230.0, "viral_score": 8.0, "reason": "second"},
    {"start_time": 300.0, "end_time": 330.0, "viral_score": 7.0, "reason": "third"},
]


def _real_clipper():
    # _import_pipeline_with_stubs() put a stub at services.clipper; load the real
    # module fresh (the test case's tearDown restores whatever was there before).
    # test_pipeline_integration may have parked an ffmpeg stub in sys.modules; the
    # real ffmpeg-python (a declared dep) is bound for this load only.
    sys.modules.pop("services.clipper", None)
    with patch.dict(sys.modules):
        sys.modules.pop("ffmpeg", None)
        return importlib.import_module("services.clipper")


def _ffmpeg_error(clipper, stderr: bytes) -> Exception:
    return clipper.ffmpeg.Error("ffmpeg", b"", stderr)


class ClipCutFailureTests(_PipelineStubTestCase):
    async def _run(self, failing_starts: set[float]):
        pipeline = _import_pipeline_with_stubs()
        clipper = _real_clipper()
        job_updates: list[dict] = []
        cut_calls: list[tuple[float, float, Path]] = []

        def fake_cut(_video, start, end, output_path):
            cut_calls.append((start, end, output_path))
            if start in failing_starts:
                raise _ffmpeg_error(
                    clipper, b"frame=0\nError opening input: missing VOD fragment\n"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"clip")

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_path = Path(tmp.name)
        self.storage_dir = tmp_path
        job_dir = tmp_path / "clip-job"
        job_dir.mkdir()
        video_path = job_dir / "video.mp4"
        video_path.write_bytes(b"video")
        result = types.SimpleNamespace(post_processed_moments=_ROWS)

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
            stack.enter_context(patch.object(pipeline, "clipper", clipper))
            stack.enter_context(patch.object(clipper, "cut_raw_segment", fake_cut))
            stack.enter_context(patch.object(clipper, "_video_output_kwargs", lambda _p: {}))
            logs = stack.enter_context(self.assertLogs("services.clipper", level=logging.WARNING)) \
                if failing_starts else None
            await pipeline.run_pipeline("clip-job", video_path=video_path, profile="jynxzi")

        statuses = [u.get("status") for u in job_updates if u.get("status")]
        return job_updates, statuses, cut_calls, logs

    async def test_one_failing_clip_still_finishes_job_with_remaining_clips(self) -> None:
        job_updates, statuses, cut_calls, logs = await self._run(failing_starts={200.0})

        self.assertEqual(len(cut_calls), 3)
        self.assertNotIn("failed", statuses)
        done = [u for u in job_updates if u.get("status") == "done"][-1]
        clips = done["clips"]
        # Survivors keep their original rank/filename — clip_02 is simply missing.
        self.assertEqual([c.filename for c in clips], ["clip_01.mp4", "clip_03.mp4"])
        self.assertEqual([c.start_time for c in clips], [100.0, 300.0])
        self.assertEqual(
            [c.download_url for c in clips],
            ["/static/clip-job/clips/clip_01.mp4", "/static/clip-job/clips/clip_03.mp4"],
        )

        # The log names the failed clip's index, time window and the ffmpeg error.
        failure_log = "\n".join(logs.output)
        self.assertIn("index=1", failure_log)
        self.assertIn("rank=2", failure_log)
        self.assertIn("start=200.0", failure_log)
        self.assertIn("end=230.0", failure_log)
        self.assertIn("missing VOD fragment", failure_log)

        # Feedback (#119) only covers clips that were actually cut.
        from services.job_artifact_store import RealFsJobArtifactStore

        rows = RealFsJobArtifactStore(self.storage_dir).read_feedback("clip-job")
        self.assertEqual(
            sorted((r["candidate_id"], r["rank"]) for r in rows),
            [("llm_moment_01", 1), ("llm_moment_03", 3)],
        )

    async def test_all_clips_failing_fails_job_with_clear_message(self) -> None:
        job_updates, statuses, _cut_calls, _logs = await self._run(
            failing_starts={100.0, 200.0, 300.0}
        )

        self.assertNotIn("done", statuses)
        failed = [u for u in job_updates if u.get("status") == "failed"][-1]
        self.assertIn("All 3 clips failed to cut", failed["error"])
        self.assertIn("missing VOD fragment", failed["error"])

    async def test_no_failures_cuts_every_clip(self) -> None:
        job_updates, statuses, _cut_calls, _logs = await self._run(failing_starts=set())

        self.assertNotIn("failed", statuses)
        done = [u for u in job_updates if u.get("status") == "done"][-1]
        self.assertEqual(
            [c.filename for c in done["clips"]], ["clip_01.mp4", "clip_02.mp4", "clip_03.mp4"]
        )
