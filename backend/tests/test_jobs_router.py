import asyncio
import io
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pytest

# fastapi is a base dependency; skip cleanly (rather than error the whole
# collection) when running without it installed.
pytest.importorskip("fastapi")

from fastapi import HTTPException, UploadFile

from routers import jobs
from models import ClipFeedbackPayload, ClipFeedbackResponse, JobStatus


class JobsRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        jobs._status_cache.clear()

    def tearDown(self) -> None:
        jobs._status_cache.clear()

    def test_status_reports_local_fallback_when_external_keys_are_absent(self) -> None:
        with patch.object(jobs.settings, "groq_api_key", None), \
             patch.object(jobs.settings, "gemini_api_key", None), \
             patch.object(jobs.settings, "ollama_model", "llama3.2"), \
             patch.object(
                 jobs,
                 "_pipeline_component_statuses",
                 return_value={
                     "face_detection": {
                         "label": "Face Detection",
                         "name": "MTCNN",
                         "model": "mtcnn",
                         "active": True,
                         "error": "MediaPipe nicht verfuegbar",
                         "mode": "fallback",
                     }
                 },
             ):
            result = asyncio.run(jobs.get_status())

        semantic = result["providers"]["semantic"]
        refinement = result["providers"]["refinement"]

        self.assertEqual(semantic["name"], "Ollama")
        self.assertEqual(semantic["mode"], "active")
        self.assertTrue(semantic["active"])
        self.assertIsNone(semantic["error"])
        self.assertIsNone(semantic["fallback_name"])

        self.assertEqual(refinement["name"], "Ollama")
        self.assertEqual(refinement["mode"], "fallback")
        self.assertTrue(refinement["active"])
        self.assertIsNone(refinement["error"])
        self.assertIsNone(refinement["fallback_name"])
        self.assertEqual(result["pipeline"]["face_detection"]["name"], "MTCNN")

    def test_status_reports_local_semantic_scoring_even_when_groq_is_configured(self) -> None:
        with patch.object(jobs.settings, "groq_api_key", "bad-key"), \
             patch.object(jobs.settings, "gemini_api_key", None), \
             patch.object(jobs.settings, "groq_model", "llama-3.3"), \
             patch.object(jobs.settings, "ollama_model", "llama3.2"), \
             patch.object(jobs, "_pipeline_component_statuses", return_value={}):
            result = asyncio.run(jobs.get_status())

        semantic = result["providers"]["semantic"]
        self.assertEqual(semantic["name"], "Ollama")
        self.assertEqual(semantic["model"], "llama3.2")
        self.assertEqual(semantic["mode"], "active")
        self.assertTrue(semantic["active"])
        self.assertIsNone(semantic["error"])
        self.assertEqual(semantic["fallback_name"], "Groq")
        self.assertEqual(semantic["fallback_model"], "llama-3.3")

    def test_status_reports_ollama_fallback_when_gemini_quota_is_exhausted(self) -> None:
        with patch.object(jobs.settings, "groq_api_key", None), \
             patch.object(jobs.settings, "gemini_api_key", "gem-key"), \
             patch.object(jobs.settings, "gemini_model", "gemini-2.0-flash"), \
             patch.object(jobs.settings, "ollama_model", "llama3.2"), \
             patch.object(jobs, "_check_gemini", return_value=(False, "Gemini-Quota erschöpft")), \
             patch.object(jobs, "_pipeline_component_statuses", return_value={}):
            result = asyncio.run(jobs.get_status())

        refinement = result["providers"]["refinement"]
        self.assertEqual(refinement["name"], "Gemini")
        self.assertEqual(refinement["model"], "gemini-2.0-flash")
        self.assertEqual(refinement["mode"], "fallback")
        self.assertTrue(refinement["active"])
        self.assertEqual(refinement["error"], "Gemini-Quota erschöpft")
        self.assertEqual(refinement["fallback_name"], "Ollama")
        self.assertEqual(refinement["fallback_model"], "llama3.2")

    def test_face_probe_falls_back_when_mediapipe_subprocess_crashes(self) -> None:
        crashed = subprocess.CompletedProcess(
            args=[],
            returncode=-11,
            stdout="",
            stderr="mediapipe segfault",
        )
        mtcnn_ok = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"label":"Face Detection","name":"MTCNN","model":"mtcnn","active":true,"error":null,"mode":"fallback"}\n',
            stderr="",
        )

        with patch.object(jobs.subprocess, "run", side_effect=[crashed, mtcnn_ok]):
            result = jobs._probe_face_detection_status()

        self.assertEqual(result["name"], "MTCNN")
        self.assertTrue(result["active"])
        self.assertIn("signal 11", result["error"])

    def test_face_probe_falls_back_to_opencv_when_mediapipe_and_mtcnn_fail(self) -> None:
        mediapipe_failed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"active":false,"error":"mediapipe unavailable"}\n',
            stderr="",
        )
        mtcnn_failed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"active":false,"error":"mtcnn unavailable"}\n',
            stderr="",
        )
        opencv_ok = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"label":"Face Detection","name":"OpenCV","model":"haar cascade","active":true,"error":null,"mode":"fallback"}\n',
            stderr="",
        )

        with patch.object(
            jobs.subprocess,
            "run",
            side_effect=[mediapipe_failed, mtcnn_failed, opencv_ok],
        ):
            result = jobs._probe_face_detection_status()

        self.assertEqual(result["name"], "OpenCV")
        self.assertTrue(result["active"])
        self.assertEqual(result["error"], "mtcnn unavailable")

    def test_upload_accepts_large_file_when_limit_is_disabled(self) -> None:
        upload = UploadFile(filename="video.mp4", file=io.BytesIO(b"123456"))
        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return None

        fake_job_id = "job-1"
        fake_job = {"id": fake_job_id}
        with patch.object(jobs.settings, "max_upload_size_bytes", None), \
             patch.object(jobs.uuid, "uuid4", return_value=fake_job_id), \
             patch.object(jobs, "create_job", return_value=fake_job) as create_job_mock, \
             patch.object(jobs, "bind_job_session") as bind_session_mock, \
             patch.object(jobs.asyncio, "create_task", side_effect=fake_create_task):
            result = asyncio.run(jobs.upload_job(upload, session_id="sess-1"))

        self.assertEqual(result, fake_job)
        create_job_mock.assert_called_once()
        bind_session_mock.assert_called_once_with(fake_job_id, "sess-1")
        self.assertEqual(len(scheduled), 1)

    def test_upload_rejects_oversized_file_before_creating_job_when_limit_is_set(self) -> None:
        upload = UploadFile(filename="video.mp4", file=io.BytesIO(b"123456"))

        with patch.object(jobs.settings, "max_upload_size_bytes", 5), \
             patch.object(jobs, "create_job") as create_job_mock:
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(jobs.upload_job(upload))

        self.assertEqual(ctx.exception.status_code, 413)
        create_job_mock.assert_not_called()

    def test_cancel_job_returns_updated_job_status(self) -> None:
        cancelled = {"job_id": "job-1", "status": "cancelled"}

        with patch.object(jobs, "cancel_job", return_value=cancelled):
            result = asyncio.run(jobs.cancel_job_status("job-1"))

        self.assertEqual(result, cancelled)

    def test_cancel_job_raises_404_for_unknown_job(self) -> None:
        with patch.object(jobs, "cancel_job", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(jobs.cancel_job_status("missing"))

        self.assertEqual(ctx.exception.status_code, 404)

    def test_submit_job_binds_client_session(self) -> None:
        captured = []

        def fake_create_task(coro):
            captured.append(coro)
            coro.close()
            return None

        fake_job_id = "job-1"
        fake_job = {"job_id": fake_job_id}
        with patch.object(jobs.uuid, "uuid4", return_value=fake_job_id), \
             patch.object(jobs, "create_job", return_value=fake_job) as create_job_mock, \
             patch.object(jobs, "bind_job_session") as bind_session_mock, \
             patch.object(jobs.asyncio, "create_task", side_effect=fake_create_task):
            result = asyncio.run(jobs.submit_job(jobs.JobRequest(url="https://www.twitch.tv/videos/123", session_id="sess-1")))

        self.assertEqual(result, fake_job)
        create_job_mock.assert_called_once_with(fake_job_id, profile="general")
        bind_session_mock.assert_called_once_with(fake_job_id, "sess-1")

    def test_pipeline_command_runs_worker_in_subprocess(self) -> None:
        cmd = jobs._pipeline_subprocess_command(
            job_id="job-1",
            url="https://www.twitch.tv/videos/123",
            video_path=None,
            profile="jynxzi",
            strategy={"clip_style": "balanced"},
        )

        self.assertEqual(cmd[0], jobs.sys.executable)
        self.assertTrue(cmd[1].endswith("pipeline_worker.py"))
        self.assertIn("--url", cmd)
        self.assertIn("https://www.twitch.tv/videos/123", cmd)
        self.assertIn("--strategy-json", cmd)

    def test_pipeline_command_serializes_set_valued_strategy_fields(self) -> None:
        cmd = jobs._pipeline_subprocess_command(
            job_id="job-1",
            url="https://www.twitch.tv/videos/123",
            video_path=None,
            profile="jynxzi",
            strategy={
                "keywords": {"rage", "funny"},
                "nested": {"exclude_keywords": {"spoiler"}},
            },
        )

        raw_strategy = cmd[cmd.index("--strategy-json") + 1]
        payload = jobs.json.loads(raw_strategy)
        self.assertEqual(payload["keywords"], ["funny", "rage"])
        self.assertEqual(payload["nested"]["exclude_keywords"], ["spoiler"])

    def test_pipeline_subprocess_signal_failure_marks_job_failed(self) -> None:
        captured = {}

        class _Process:
            async def wait(self):
                return -11

        async def fake_create_subprocess_exec(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _Process()

        with patch.object(jobs.asyncio, "create_subprocess_exec", side_effect=fake_create_subprocess_exec), \
             patch.object(jobs, "get_job", return_value=JobStatus(job_id="job-1", status="downloading")), \
             patch.object(jobs, "update_job") as update_mock:
            asyncio.run(
                jobs._run_pipeline_subprocess(
                    job_id="job-1",
                    url="https://www.twitch.tv/videos/123",
                    profile="general",
                    strategy={},
                )
            )

        self.assertTrue(captured["args"][1].endswith("pipeline_worker.py"))
        self.assertEqual(captured["kwargs"]["cwd"], str(jobs._BACKEND_ROOT))
        update_mock.assert_called_once_with(
            "job-1",
            status="failed",
            error="Pipeline subprocess exited from signal 11",
        )

    def test_pipeline_subprocess_failure_deletes_serving_source_media(self) -> None:
        class _Process:
            async def wait(self):
                return -9

        async def fake_create_subprocess_exec(*_args, **_kwargs):
            return _Process()

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            job_dir = storage_dir / "job-1"
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

            with patch.object(jobs.settings, "storage_dir", str(storage_dir)), \
                 patch.object(jobs.asyncio, "create_subprocess_exec", side_effect=fake_create_subprocess_exec), \
                 patch.object(jobs, "get_job", return_value=JobStatus(job_id="job-1", status="downloading")), \
                 patch.object(jobs, "update_job"):
                asyncio.run(
                    jobs._run_pipeline_subprocess(
                        job_id="job-1",
                        url="https://www.twitch.tv/videos/123",
                        profile="general",
                        strategy={},
                    )
                )

            self.assertFalse(video_path.exists())
            self.assertFalse(audio_path.exists())
            self.assertFalse(chunks_dir.exists())
            self.assertTrue(clip_path.exists())
            self.assertTrue(debug_path.exists())

    def test_subprocess_parent_cleanup_preserves_training_source_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            job_dir = storage_dir / "twitch_vod_123"
            job_dir.mkdir(parents=True)
            video_path = job_dir / "video.mp4"
            audio_path = job_dir / "audio.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")

            with patch.object(jobs.settings, "storage_dir", str(storage_dir)):
                jobs._cleanup_serving_job_source_media("twitch_vod_123")

            self.assertTrue(video_path.exists())
            self.assertTrue(audio_path.exists())

    def test_pipeline_subprocess_sigkill_error_mentions_oom(self) -> None:
        self.assertEqual(
            jobs._pipeline_subprocess_error(-9),
            "Pipeline subprocess exited from signal 9 (SIGKILL; likely OOM or external kill)",
        )

    def test_pipeline_subprocess_env_enables_face_detection_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(jobs.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(jobs.os.environ, {}, clear=True):
            env = jobs._pipeline_subprocess_env()
            self.assertTrue(Path(env["MPLCONFIGDIR"]).exists())

        self.assertEqual(env["AI_CLIPPING_DISABLE_FACE_DETECTION"], "0")
        self.assertIn(str(jobs._BACKEND_ROOT), env["PYTHONPATH"].split(jobs.os.pathsep))
        self.assertEqual(env["LIBGL_ALWAYS_SOFTWARE"], "1")
        self.assertEqual(env["MESA_LOADER_DRIVER_OVERRIDE"], "swrast")
        self.assertEqual(env["OPENCV_FFMPEG_CAPTURE_OPTIONS"], "hwaccel;none")
        self.assertEqual(env["PYTHONFAULTHANDLER"], "1")
        self.assertEqual(env["TORCHDYNAMO_DISABLE"], "1")
        self.assertEqual(env["TORCH_COMPILE_DISABLE"], "1")
        self.assertEqual(env["PIPELINE_ENABLE_DIARIZATION"], "false")
        self.assertEqual(env["WHISPER_LANGUAGE"], "en")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["MKL_NUM_THREADS"], "1")
        self.assertEqual(env["NUMEXPR_NUM_THREADS"], "1")
        self.assertEqual(env["VECLIB_MAXIMUM_THREADS"], "1")

    def test_pipeline_subprocess_env_matches_cuda_training_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(jobs.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(jobs.os.environ, {"WHISPER_DEVICE": "cuda"}, clear=True):
            env = jobs._pipeline_subprocess_env()

        self.assertEqual(env["WHISPER_LANGUAGE"], "en")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_pipeline_subprocess_env_can_enable_diarization_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(jobs.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.object(jobs.settings, "pipeline_enable_diarization", True), \
             patch.dict(jobs.os.environ, {}, clear=True):
            env = jobs._pipeline_subprocess_env()

        self.assertEqual(env["PIPELINE_ENABLE_DIARIZATION"], "true")

    def test_pipeline_subprocess_env_respects_explicit_face_detection_disable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(jobs.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(jobs.os.environ, {"AI_CLIPPING_DISABLE_FACE_DETECTION": "1"}, clear=True):
            env = jobs._pipeline_subprocess_env()

        self.assertEqual(env["AI_CLIPPING_DISABLE_FACE_DETECTION"], "1")

    def test_face_detection_probe_env_uses_writable_matplotlib_and_headless_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(jobs.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(jobs.os.environ, {}, clear=True):
            env = jobs._face_detection_probe_env()
            self.assertTrue(Path(env["MPLCONFIGDIR"]).exists())

        self.assertEqual(env["LIBGL_ALWAYS_SOFTWARE"], "1")
        self.assertEqual(env["MESA_LOADER_DRIVER_OVERRIDE"], "swrast")
        self.assertEqual(env["OPENCV_FFMPEG_CAPTURE_OPTIONS"], "hwaccel;none")

    def test_get_job_status_touches_client_session(self) -> None:
        job = {"job_id": "job-1", "status": "downloading"}

        with patch.object(jobs, "touch_job_session", return_value=job) as touch_mock, \
             patch.object(jobs, "_job_pipeline_diagnostics", return_value={"available": False}) as diagnostics_mock:
            result = asyncio.run(jobs.get_job_status("job-1", session_id="sess-1"))

        self.assertEqual(result["job_id"], "job-1")
        self.assertEqual(result["pipeline_diagnostics"], {"available": False})
        touch_mock.assert_called_once_with("job-1", "sess-1")
        diagnostics_mock.assert_called_once_with("job-1", job)

    def test_job_pipeline_diagnostics_summarizes_debug_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(jobs.settings, "storage_dir", tmp):
            debug_dir = Path(tmp) / "job-1" / "debug"
            debug_dir.mkdir(parents=True)
            (debug_dir / "segments.jsonl").write_text(
                '{"start":0.0,"end":5.0,"text":"hello"}\n{"start":5.0,"end":10.0,"text":"world"}\n',
                encoding="utf-8",
            )
            (debug_dir / "llm_moment_finder_stage.json").write_text(
                jobs.json.dumps(
                    {
                        "attempted": True,
                        "status": "ok",
                        "provider": "gemini",
                        "model": "gemini-2.5-flash-lite",
                        "max_moments": 180,
                        "ranked_candidate_count": 3,
                        "post_processed_moment_count": 3,
                    }
                ),
                encoding="utf-8",
            )
            (debug_dir / "llm_moment_finder_moments.jsonl").write_text(
                '{"rank":1,"start_time":10.0,"end_time":40.0}\n'
                '{"rank":2,"start_time":60.0,"end_time":90.0}\n'
                '{"rank":3,"start_time":120.0,"end_time":150.0}\n',
                encoding="utf-8",
            )
            (debug_dir / "final_moments.json").write_text(
                jobs.json.dumps({"moments": [{"candidate_id": "llm_moment_01"}, {"candidate_id": "llm_moment_02"}, {"candidate_id": "llm_moment_03"}], "count": 3}),
                encoding="utf-8",
            )

            result = jobs._job_pipeline_diagnostics(
                "job-1",
                JobStatus(job_id="job-1", status="done", clips=[]),
            )

        self.assertTrue(result["available"])
        stages = {stage["id"]: stage for stage in result["stages"]}
        self.assertEqual(stages["transcript"]["status"], "ok")
        self.assertIn("2", stages["transcript"]["summary"])
        self.assertEqual(stages["llm_moment_finder"]["status"], "ok")
        self.assertIn("3 Momente", stages["llm_moment_finder"]["summary"])
        self.assertEqual(stages["clipping"]["status"], "ok")
        self.assertIn("Stages ok", result["summary"])
        self.assertFalse(result["candidate_funnel"]["available"])

    def test_job_pipeline_diagnostics_reports_llm_moment_finder_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(jobs.settings, "storage_dir", tmp):
            debug_dir = Path(tmp) / "job-1" / "debug"
            debug_dir.mkdir(parents=True)
            (debug_dir / "segments.jsonl").write_text(
                '{"start":0.0,"end":5.0,"text":"hello"}\n',
                encoding="utf-8",
            )
            (debug_dir / "llm_moment_finder_stage.json").write_text(
                jobs.json.dumps(
                    {
                        "attempted": True,
                        "status": "failed",
                        "provider": "gemini",
                        "model": "gemini-2.5-flash-lite",
                        "max_moments": 180,
                        "error": "API quota exceeded",
                    }
                ),
                encoding="utf-8",
            )

            result = jobs._job_pipeline_diagnostics(
                "job-1",
                JobStatus(job_id="job-1", status="failed", clips=[]),
            )

        stages = {stage["id"]: stage for stage in result["stages"]}
        llm = stages["llm_moment_finder"]
        self.assertEqual(llm["status"], "warning")
        self.assertIn("fehlgeschlagen", llm["summary"])
        self.assertIn("API quota exceeded", llm["details"][0])

    def test_submit_clip_feedback_accepts_gold_eval_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = ClipFeedbackResponse(
                job_id="job-1",
                candidate_id="c1",
                mode="gold_eval",
                gold_grade=2,
                notes="brauchbar",
            )
            with patch.object(jobs.settings, "storage_dir", tmp), \
                 patch.object(jobs, "get_job", return_value={"job_id": "job-1"}), \
                 patch.object(jobs.feedback_store, "update_job_feedback", return_value=response) as update_mock:
                result = asyncio.run(
                    jobs.submit_clip_feedback(
                        "job-1",
                        ClipFeedbackPayload(
                            candidate_id="c1",
                            mode="gold_eval",
                            gold_grade=2,
                            notes="brauchbar",
                        ),
                    )
                )

        self.assertEqual(result.mode, "gold_eval")
        self.assertEqual(result.gold_grade, 2)
        update_mock.assert_called_once_with(Path(tmp), "job-1", unittest.mock.ANY)

    def test_submit_clip_feedback_maps_validation_errors_to_http_400(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(jobs.settings, "storage_dir", tmp), \
                 patch.object(jobs, "get_job", return_value={"job_id": "job-1"}), \
                 patch.object(jobs.feedback_store, "update_job_feedback", side_effect=ValueError("gold_notes are required")):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(
                        jobs.submit_clip_feedback(
                            "job-1",
                            ClipFeedbackPayload(
                                candidate_id="c1",
                                mode="gold_eval",
                                gold_grade=3,
                                notes="",
                            ),
                        )
                    )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("gold_notes are required", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
