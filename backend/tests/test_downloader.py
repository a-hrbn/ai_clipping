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
    def test_coerces_float_eta_to_integer_seconds(self) -> None:
        self.assertEqual(downloader._coerce_eta_seconds(715.6143084933422), 716)
        self.assertEqual(downloader._coerce_eta_seconds(0.2), 0)
        self.assertIsNone(downloader._coerce_eta_seconds(None))

    def test_twitch_download_opts_enable_parallel_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            with patch.object(downloader.settings, "twitch_download_concurrent_fragments", 9), \
                 patch.object(downloader.settings, "twitch_download_use_aria2c", False):
                opts = downloader._twitch_download_ydl_opts(job_dir)

        self.assertEqual(opts["concurrent_fragment_downloads"], 9)
        self.assertEqual(opts["fragment_retries"], 10)
        self.assertTrue(opts["skip_unavailable_fragments"])
        self.assertFalse(opts["keep_fragments"])
        self.assertNotIn("external_downloader", opts)

    def test_twitch_download_opts_use_aria2c_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            with patch.object(downloader.settings, "twitch_download_use_aria2c", True), \
                 patch.object(downloader.settings, "twitch_download_aria2_connections", 12), \
                 patch.object(downloader, "_aria2c_available", return_value=True):
                opts = downloader._twitch_download_ydl_opts(job_dir)

        self.assertEqual(opts["external_downloader"], "aria2c")
        self.assertEqual(
            opts["external_downloader_args"],
            [
                "--split",
                "12",
                "--max-connection-per-server",
                "12",
                "--min-split-size",
                "1M",
                "--file-allocation=none",
                "--summary-interval=0",
            ],
        )

    def test_reuses_recent_twitch_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/123456"
            cache_path = storage_dir / "_download_cache" / "twitch"
            cache_path.mkdir(parents=True, exist_ok=True)
            cached_file = cache_path / f"{downloader.sha256(url.encode('utf-8')).hexdigest()}.mp4"
            cached_file.write_bytes(b"cached-video")

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", 3600), \
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
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", 3600), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _FakeYoutubeDL):
                output_path = downloader.download(url, "job-cache-miss")

            self.assertTrue(output_path.exists())
            self.assertTrue(expected_cache_file.exists())
            self.assertEqual(expected_cache_file.read_bytes(), output_path.read_bytes())

    def test_skips_twitch_cache_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/555"
            expected_cache_dir = storage_dir / "_download_cache" / "twitch"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", 0), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _FakeYoutubeDL):
                output_path = downloader.download(url, "job-no-cache")

            self.assertTrue(output_path.exists())
            self.assertFalse(expected_cache_dir.exists())

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

    def test_transient_network_errors_pause_download_instead_of_marking_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.tiktok.com/@jynxzi/video/123"

            class _FailingYDL:
                def __init__(self, opts: dict):
                    self.opts = opts

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def extract_info(self, url: str, download: bool = True):
                    raise RuntimeError("Temporary failure in name resolution")

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _FailingYDL):
                with self.assertRaises(downloader.DownloadPausedError):
                    downloader.download(url, "job-network-pause")


if __name__ == "__main__":
    unittest.main()
