import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import downloader


class _FakeYoutubeDL:
    def __init__(self, opts: dict):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def download(self, urls: list[str]) -> None:
        target = Path(self.opts["outtmpl"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"downloaded:{urls[0]}".encode("utf-8"))


class DownloaderTests(unittest.TestCase):
    def test_reuses_recent_twitch_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/123456"
            cache_path = storage_dir / "_download_cache" / "twitch"
            cache_path.mkdir(parents=True, exist_ok=True)
            cached_file = cache_path / f"{downloader.sha256(url.encode('utf-8')).hexdigest()}.mp4"
            cached_file.write_bytes(b"cached-video")

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader, "update_job", create=True) as update_job_mock, \
                 patch.object(downloader.yt_dlp, "YoutubeDL", side_effect=AssertionError("network should not run")):
                output_path = downloader.download(url, "job-cache-hit")

            self.assertEqual(output_path.read_bytes(), b"cached-video")
            update_job_mock.assert_called_with(
                "job-cache-hit",
                progress_pct=19,
                download_speed=None,
                download_eta=0,
                download_cached=True,
            )

    def test_stores_fresh_twitch_download_in_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/7890"
            expected_cache_file = storage_dir / "_download_cache" / "twitch" / f"{downloader.sha256(url.encode('utf-8')).hexdigest()}.mp4"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _FakeYoutubeDL):
                output_path = downloader.download(url, "job-cache-miss")

            self.assertTrue(output_path.exists())
            self.assertTrue(expected_cache_file.exists())
            self.assertEqual(expected_cache_file.read_bytes(), output_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
