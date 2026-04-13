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

    def extract_info(self, url: str, download: bool = True):
        assert download is True
        target = Path(self.opts["outtmpl"])
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized = Path(str(target).replace("%(ext)s", "mp4"))
        normalized.write_bytes(f"downloaded:{url}".encode("utf-8"))
        return {"filepath": str(normalized)}


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

    def test_resolves_downloaded_file_with_nonstandard_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            expected = job_dir / "video.mp4"
            actual = job_dir / "video.mkv"
            actual.write_bytes(b"video-data")

            resolved = downloader._resolve_downloaded_video(job_dir, expected)

            self.assertEqual(resolved, expected)
            self.assertTrue(expected.exists())
            self.assertFalse(actual.exists())
            self.assertEqual(expected.read_bytes(), b"video-data")

    def test_uses_reported_yt_dlp_filepath_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/111"

            class _ReportedPathYDL(_FakeYoutubeDL):
                def extract_info(self, url: str, download: bool = True):
                    target = storage_dir / "job-reported" / "nested" / "video.mkv"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"reported-video")
                    return {"requested_downloads": [{"filepath": str(target)}]}

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _ReportedPathYDL):
                output_path = downloader.download(url, "job-reported")

            self.assertEqual(output_path, storage_dir / "job-reported" / "video.mp4")
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes(), b"reported-video")


if __name__ == "__main__":
    unittest.main()
