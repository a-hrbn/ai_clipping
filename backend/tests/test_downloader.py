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


class _FakeYoutubeDLWithTemp(_FakeYoutubeDL):
    def extract_info(self, url: str, download: bool = True):
        info = super().extract_info(url, download=download)
        target = Path(self.opts["outtmpl"])
        target.parent.mkdir(parents=True, exist_ok=True)
        (target.parent / "video.temp.mp4").write_bytes(b"stale-temp-copy")
        (target.parent / "video.mp4.part-Frag1").write_bytes(b"stale-fragment")
        return info


class _CaptureOptsYoutubeDL(_FakeYoutubeDL):
    captured_opts: dict | None = None

    def __init__(self, opts: dict):
        type(self).captured_opts = opts
        super().__init__(opts)


class DownloaderTests(unittest.TestCase):
    def test_coerces_float_eta_to_integer_seconds(self) -> None:
        self.assertEqual(downloader._coerce_eta_seconds(715.6143084933422), 716)
        self.assertEqual(downloader._coerce_eta_seconds(0.2), 0)
        self.assertIsNone(downloader._coerce_eta_seconds(None))

    def test_twitch_cache_path_uses_canonical_vod_id(self) -> None:
        path = downloader._cache_path_for_url("https://www.twitch.tv/videos/123456789?t=1h2m3s")
        self.assertEqual(path.name, "vod_123456789.mp4")

    def test_negative_ttl_keeps_twitch_cache_enabled(self) -> None:
        with patch.object(downloader.settings, "twitch_download_cache_ttl_sec", -1):
            self.assertTrue(downloader._twitch_cache_enabled("https://www.twitch.tv/videos/123456"))

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

    def test_twitch_training_download_opts_cap_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/555"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "training_vod_max_height", 720), \
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", 0), \
                 patch.object(downloader.settings, "twitch_download_use_aria2c", False), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _CaptureOptsYoutubeDL):
                downloader.download(url, "twitch_vod_555")

        self.assertIsNotNone(_CaptureOptsYoutubeDL.captured_opts)
        self.assertIn("height<=720", str(_CaptureOptsYoutubeDL.captured_opts["format"]))

    def test_non_training_download_opts_do_not_cap_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/555"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "training_vod_max_height", 720), \
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", 0), \
                 patch.object(downloader.settings, "twitch_download_use_aria2c", False), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _CaptureOptsYoutubeDL):
                downloader.download(url, "regular-job")

        self.assertIsNotNone(_CaptureOptsYoutubeDL.captured_opts)
        self.assertNotIn("height<=720", str(_CaptureOptsYoutubeDL.captured_opts["format"]))

    def test_download_opts_include_ytdlp_cookiefile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.youtube.com/shorts/abc"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "yt_dlp_cookiefile", "/tmp/youtube.cookies.txt"), \
                 patch.object(downloader.settings, "yt_dlp_cookies_from_browser", ""), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _CaptureOptsYoutubeDL):
                downloader.download(url, "regular-job")

        self.assertIsNotNone(_CaptureOptsYoutubeDL.captured_opts)
        self.assertEqual(_CaptureOptsYoutubeDL.captured_opts["cookiefile"], "/tmp/youtube.cookies.txt")

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
            cached_file = cache_path / "vod_123456.mp4"
            cached_file.write_bytes(b"cached-video")
            job_dir = storage_dir / "job-cache-hit"
            job_dir.mkdir()
            (job_dir / "video.temp.mp4").write_bytes(b"old-temp")

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", -1), \
                 patch.object(downloader, "update_job", create=True) as update_job_mock, \
                 patch.object(downloader.yt_dlp, "YoutubeDL", side_effect=AssertionError("network should not run")):
                output_path = downloader.download(url, "job-cache-hit")

            self.assertEqual(output_path.read_bytes(), b"cached-video")
            self.assertFalse((job_dir / "video.temp.mp4").exists())
            update_job_mock.assert_called_with(
                "job-cache-hit",
                progress_pct=19,
                download_speed=None,
                download_eta=0,
                download_cached=True,
            )

    def test_ensure_cached_download_reuses_existing_generic_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.youtube.com/shorts/abc"
            cache_dir = storage_dir / "_download_cache" / "short_form"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_file = cache_dir / f"{downloader.sha256(url.encode('utf-8')).hexdigest()}.mp4"
            cached_file.write_bytes(b"cached-short-form")

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", side_effect=AssertionError("network should not run")):
                output_path = downloader.ensure_cached_download(url, cache_namespace="short_form")

            self.assertEqual(output_path, cached_file)
            self.assertEqual(output_path.read_bytes(), b"cached-short-form")

    def test_ensure_cached_download_populates_generic_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.tiktok.com/@jynxzi/video/123"
            expected_cache_file = storage_dir / "_download_cache" / "short_form" / f"{downloader.sha256(url.encode('utf-8')).hexdigest()}.mp4"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _FakeYoutubeDL):
                output_path = downloader.ensure_cached_download(url, cache_namespace="short_form")

            self.assertEqual(output_path, expected_cache_file)
            self.assertTrue(expected_cache_file.exists())
            self.assertEqual(expected_cache_file.read_bytes(), b"downloaded:https://www.tiktok.com/@jynxzi/video/123")
            metadata = downloader.json.loads(downloader._cache_metadata_path(expected_cache_file).read_text(encoding="utf-8"))
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["namespace"], "short_form")

    def test_short_form_cache_download_uses_ytdlp_sleep_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.youtube.com/shorts/abc"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "yt_dlp_short_form_sleep_requests_sec", 6.0), \
                 patch.object(downloader.settings, "yt_dlp_short_form_min_sleep_interval_sec", 7.0), \
                 patch.object(downloader.settings, "yt_dlp_short_form_max_sleep_interval_sec", 10.0), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _CaptureOptsYoutubeDL):
                downloader.ensure_cached_download(url, cache_namespace="short_form")

        self.assertIsNotNone(_CaptureOptsYoutubeDL.captured_opts)
        self.assertEqual(_CaptureOptsYoutubeDL.captured_opts["sleep_interval_requests"], 6.0)
        self.assertEqual(_CaptureOptsYoutubeDL.captured_opts["sleep_interval"], 7.0)
        self.assertEqual(_CaptureOptsYoutubeDL.captured_opts["max_sleep_interval"], 10.0)

    def test_non_short_form_cache_download_does_not_use_sleep_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://example.com/video.mp4"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "yt_dlp_short_form_sleep_requests_sec", 6.0), \
                 patch.object(downloader.settings, "yt_dlp_short_form_min_sleep_interval_sec", 7.0), \
                 patch.object(downloader.settings, "yt_dlp_short_form_max_sleep_interval_sec", 10.0), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _CaptureOptsYoutubeDL):
                downloader.ensure_cached_download(url, cache_namespace="generic_media")

        self.assertIsNotNone(_CaptureOptsYoutubeDL.captured_opts)
        self.assertNotIn("sleep_interval_requests", _CaptureOptsYoutubeDL.captured_opts)
        self.assertNotIn("sleep_interval", _CaptureOptsYoutubeDL.captured_opts)
        self.assertNotIn("max_sleep_interval", _CaptureOptsYoutubeDL.captured_opts)

    def test_stores_fresh_twitch_download_in_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/7890"
            expected_cache_file = storage_dir / "_download_cache" / "twitch" / "vod_7890.mp4"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", -1), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _FakeYoutubeDLWithTemp):
                output_path = downloader.download(url, "job-cache-miss")

            self.assertTrue(output_path.exists())
            self.assertTrue(expected_cache_file.exists())
            self.assertEqual(expected_cache_file.read_bytes(), output_path.read_bytes())
            self.assertFalse((storage_dir / "job-cache-miss" / "video.temp.mp4").exists())
            self.assertFalse((storage_dir / "job-cache-miss" / "video.mp4.part-Frag1").exists())

            output_stat = output_path.stat()
            cache_stat = expected_cache_file.stat()
            self.assertEqual((output_stat.st_dev, output_stat.st_ino), (cache_stat.st_dev, cache_stat.st_ino))
            metadata = downloader.json.loads(downloader._cache_metadata_path(expected_cache_file).read_text(encoding="utf-8"))
            self.assertEqual(metadata["vod_id"], "7890")
            self.assertEqual(metadata["size"], expected_cache_file.stat().st_size)

    def test_invalid_cache_metadata_forces_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/7890"
            cache_file = storage_dir / "_download_cache" / "twitch" / "vod_7890.mp4"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"stale")
            downloader._cache_metadata_path(cache_file).write_text(
                downloader.json.dumps({"version": 1, "namespace": "wrong", "size": 5, "complete": True}),
                encoding="utf-8",
            )

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", -1), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _FakeYoutubeDL):
                output_path = downloader.download(url, "job-refresh")

            self.assertEqual(output_path.read_bytes(), b"downloaded:https://www.twitch.tv/videos/7890")
            self.assertEqual(cache_file.read_bytes(), output_path.read_bytes())

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

    def test_download_range_passes_yt_dlp_range_options_without_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            url = "https://www.twitch.tv/videos/555"

            with patch.object(downloader.settings, "storage_dir", str(storage_dir)), \
                 patch.object(downloader.settings, "twitch_download_cache_ttl_sec", -1), \
                 patch.object(downloader.settings, "twitch_download_use_aria2c", True), \
                 patch.object(downloader, "_aria2c_available", return_value=True), \
                 patch.object(downloader, "update_job", create=True), \
                 patch.object(downloader.yt_dlp, "YoutubeDL", _CaptureOptsYoutubeDL):
                output_path = downloader.download_range(url, "job-range", start_sec=12.5, end_sec=45.0)

            self.assertTrue(output_path.exists())
            self.assertFalse((storage_dir / "_download_cache" / "twitch").exists())
            self.assertIsNotNone(_CaptureOptsYoutubeDL.captured_opts)
            self.assertIn("download_ranges", _CaptureOptsYoutubeDL.captured_opts)
            self.assertFalse(_CaptureOptsYoutubeDL.captured_opts["force_keyframes_at_cuts"])
            self.assertNotIn("external_downloader", _CaptureOptsYoutubeDL.captured_opts)
            self.assertNotIn("external_downloader_args", _CaptureOptsYoutubeDL.captured_opts)

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
