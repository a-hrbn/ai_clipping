import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))


class JobCancellationTests(unittest.TestCase):
    def test_downloader_aborts_when_job_is_cancelled(self) -> None:
        from services import downloader

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/123456"

            class _CancelAwareYDL:
                def __init__(self, opts: dict):
                    self.opts = opts

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def extract_info(self, _url: str, download: bool = True):
                    assert download is True
                    hook = self.opts["progress_hooks"][0]
                    hook(
                        {
                            "status": "downloading",
                            "downloaded_bytes": 1024,
                            "total_bytes": 4096,
                            "speed": 256,
                            "eta": 12,
                        }
                    )
                    return {}

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader, "is_job_cancelled", return_value=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _CancelAwareYDL):
                with self.assertRaises(downloader.DownloadCancelledError):
                    downloader.download(url, "job-cancelled")

    def test_pipeline_stops_with_cancelled_status(self) -> None:
        sys.modules.pop("pipeline", None)
        pipeline = importlib.import_module("pipeline")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")

            updates: list[dict] = []

            def fake_update_job(job_id: str, **kwargs) -> None:
                updates.append({"job_id": job_id, **kwargs})

            with patch.object(pipeline.settings, "storage_dir", str(tmp_path)), \
                 patch.object(pipeline, "update_job", side_effect=fake_update_job), \
                 patch.object(pipeline, "is_job_cancelled", side_effect=[False, True]), \
                 patch.object(pipeline.audio_extractor, "extract", return_value=tmp_path / "audio.wav"):
                asyncio_module = __import__("asyncio")
                asyncio_module.run(pipeline.run_pipeline("job-cancelled", video_path=video_path, profile="general"))

        self.assertTrue(any(update.get("status") == "downloading" for update in updates))
        self.assertFalse(any(update.get("status") == "failed" for update in updates))
