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
             patch.object(jobs.settings, "ollama_model", "llama3.2"):
            result = asyncio.run(jobs.get_status())

        semantic = result["providers"]["semantic"]
        refinement = result["providers"]["refinement"]

        self.assertEqual(semantic["name"], "Ollama")
        self.assertEqual(semantic["mode"], "fallback")
        self.assertTrue(semantic["active"])
        self.assertIsNone(semantic["error"])

        self.assertEqual(refinement["name"], "Ollama")
        self.assertEqual(refinement["mode"], "fallback")
        self.assertTrue(refinement["active"])
        self.assertIsNone(refinement["error"])

    def test_status_reports_ollama_fallback_when_groq_check_fails(self) -> None:
        with patch.object(jobs.settings, "groq_api_key", "bad-key"), \
             patch.object(jobs.settings, "gemini_api_key", None), \
             patch.object(jobs.settings, "groq_model", "llama-3.3"), \
             patch.object(jobs.settings, "ollama_model", "llama3.2"), \
             patch.object(jobs, "_check_groq", return_value=(False, "Ungültiger API Key")):
            result = asyncio.run(jobs.get_status())

        semantic = result["providers"]["semantic"]
        self.assertEqual(semantic["name"], "Ollama")
        self.assertEqual(semantic["model"], "llama3.2")
        self.assertEqual(semantic["mode"], "fallback")
        self.assertTrue(semantic["active"])
        self.assertEqual(semantic["error"], "Ungültiger API Key")

    def test_upload_accepts_large_file_when_limit_is_disabled(self) -> None:
        upload = UploadFile(filename="video.mp4", file=io.BytesIO(b"123456"))
        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return None

        fake_job = {"id": "job-1"}
        with patch.object(jobs.settings, "max_upload_size_bytes", None), \
             patch.object(jobs, "create_job", return_value=fake_job) as create_job_mock, \
             patch.object(jobs.asyncio, "create_task", side_effect=fake_create_task):
            result = asyncio.run(jobs.upload_job(upload))

        self.assertEqual(result, fake_job)
        create_job_mock.assert_called_once()
        self.assertEqual(len(scheduled), 1)

    def test_upload_rejects_oversized_file_before_creating_job_when_limit_is_set(self) -> None:
        upload = UploadFile(filename="video.mp4", file=io.BytesIO(b"123456"))

        with patch.object(jobs.settings, "max_upload_size_bytes", 5), \
             patch.object(jobs, "create_job") as create_job_mock:
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(jobs.upload_job(upload))

        self.assertEqual(ctx.exception.status_code, 413)
        create_job_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
