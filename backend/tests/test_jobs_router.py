import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from routers import jobs


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
        self.assertEqual(semantic["mode"], "fallback")
        self.assertTrue(semantic["active"])
        self.assertIsNone(semantic["error"])
        self.assertIsNone(semantic["fallback_name"])

        self.assertEqual(refinement["name"], "Ollama")
        self.assertEqual(refinement["mode"], "fallback")
        self.assertTrue(refinement["active"])
        self.assertIsNone(refinement["error"])
        self.assertIsNone(refinement["fallback_name"])
        self.assertEqual(result["pipeline"]["face_detection"]["name"], "MTCNN")

    def test_status_reports_ollama_fallback_when_groq_check_fails(self) -> None:
        with patch.object(jobs.settings, "groq_api_key", "bad-key"), \
             patch.object(jobs.settings, "gemini_api_key", None), \
             patch.object(jobs.settings, "groq_model", "llama-3.3"), \
             patch.object(jobs.settings, "ollama_model", "llama3.2"), \
             patch.object(jobs, "_check_groq", return_value=(False, "Ungültiger API Key")), \
             patch.object(jobs, "_pipeline_component_statuses", return_value={}):
            result = asyncio.run(jobs.get_status())

        semantic = result["providers"]["semantic"]
        self.assertEqual(semantic["name"], "Groq")
        self.assertEqual(semantic["model"], "llama-3.3")
        self.assertEqual(semantic["mode"], "fallback")
        self.assertTrue(semantic["active"])
        self.assertEqual(semantic["error"], "Ungültiger API Key")
        self.assertEqual(semantic["fallback_name"], "Ollama")
        self.assertEqual(semantic["fallback_model"], "llama3.2")

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

    def test_get_job_status_touches_client_session(self) -> None:
        job = {"job_id": "job-1", "status": "downloading"}

        with patch.object(jobs, "touch_job_session", return_value=job) as touch_mock:
            result = asyncio.run(jobs.get_job_status("job-1", session_id="sess-1"))

        self.assertEqual(result, job)
        touch_mock.assert_called_once_with("job-1", "sess-1")


if __name__ == "__main__":
    unittest.main()
