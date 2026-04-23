from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml import cache_cleanup


class CacheCleanupTests(unittest.TestCase):
    def test_cleanup_removes_runtime_artifacts_and_keeps_persistent_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend_root = Path(tmp) / "backend"
            storage_dir = backend_root / "tmp" / "storage"
            download_cache = storage_dir / "_download_cache" / "twitch"
            vod_feature_cache = backend_root / "tmp" / "vod_feature_cache" / "123"
            row_cache = backend_root / "tmp" / "short_form_training" / "_row_cache"
            model_cache = backend_root / "tmp" / "cache" / "whisper"
            legacy_short_form_download = storage_dir / "viral_abcd1234"
            cached_legacy_short_form_download = storage_dir / "viral_beef1234"
            uncached_legacy_short_form_download = storage_dir / "viral_feed1234"
            short_form_job = backend_root / "tmp" / "short_form_training" / "viral_abcd1234"

            for path in (
                download_cache,
                storage_dir / "_download_cache" / "short_form",
                vod_feature_cache,
                row_cache,
                model_cache,
                legacy_short_form_download,
                cached_legacy_short_form_download,
                uncached_legacy_short_form_download,
                storage_dir / "_cache_fill_short_form_deadbeef",
                storage_dir / "twitch_vod_123",
                storage_dir / "twitch_vod_prefetch_456",
                storage_dir / "twitch_vod_999",
                short_form_job,
                backend_root / "tmp" / "short_form_backfill_artifacts",
                backend_root / "tmp" / "cache_deepface",
            ):
                path.mkdir(parents=True, exist_ok=True)
            (download_cache / "vod_123.mp4").write_bytes(b"cached-vod")
            (vod_feature_cache / "audio.wav").write_bytes(b"audio")
            (row_cache / "row.json").write_text("{}", encoding="utf-8")
            (model_cache / "base.pt").write_bytes(b"model")
            (legacy_short_form_download / "video.mp4").write_bytes(b"legacy-short-form-raw")
            (cached_legacy_short_form_download / "video.mp4").write_bytes(b"cached-short-form-raw")
            (storage_dir / "_download_cache" / "short_form" / "cached.mp4").write_bytes(b"cached-short-form-raw")
            (uncached_legacy_short_form_download / "video.mp4").write_bytes(b"only-local-short-form")
            (storage_dir / "twitch_vod_123" / "video.mp4").write_bytes(b"cached-vod")
            (storage_dir / "twitch_vod_999" / "video.mp4").write_bytes(b"only-local-vod")
            (storage_dir / "download.tmp").write_bytes(b"tmp")
            (storage_dir / "video.mp4.part-Frag1").write_bytes(b"part")

            result = cache_cleanup.cleanup_backend_runtime_caches(backend_root)

            self.assertGreaterEqual(result.removed_count, 7)
            self.assertEqual(result.failed_count, 0)
            self.assertFalse((storage_dir / "_cache_fill_short_form_deadbeef").exists())
            self.assertFalse((storage_dir / "twitch_vod_123").exists())
            self.assertFalse((storage_dir / "twitch_vod_prefetch_456").exists())
            self.assertFalse(cached_legacy_short_form_download.exists())
            self.assertFalse(short_form_job.exists())
            self.assertFalse((backend_root / "tmp" / "short_form_backfill_artifacts").exists())
            self.assertFalse((backend_root / "tmp" / "cache_deepface").exists())
            self.assertFalse((storage_dir / "download.tmp").exists())
            self.assertFalse((storage_dir / "video.mp4.part-Frag1").exists())

            self.assertTrue((download_cache / "vod_123.mp4").exists())
            self.assertTrue((legacy_short_form_download / "video.mp4").exists())
            self.assertTrue((uncached_legacy_short_form_download / "video.mp4").exists())
            self.assertTrue((storage_dir / "twitch_vod_999" / "video.mp4").exists())
            self.assertTrue((vod_feature_cache / "audio.wav").exists())
            self.assertTrue((row_cache / "row.json").exists())
            self.assertTrue((model_cache / "base.pt").exists())

    def test_cleanup_keeps_nonempty_legacy_deepface_fallback_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend_root = Path(tmp) / "backend"
            legacy_cache = backend_root / "tmp" / "cache_deepface"
            legacy_cache.mkdir(parents=True)
            (legacy_cache / "weights.h5").write_bytes(b"weights")

            result = cache_cleanup.cleanup_backend_runtime_caches(backend_root)

            self.assertEqual(result.failed_count, 0)
            self.assertTrue((legacy_cache / "weights.h5").exists())

    def test_dry_run_reports_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend_root = Path(tmp) / "backend"
            job_dir = backend_root / "tmp" / "storage" / "_cache_fill_short_form_abcd1234"
            job_dir.mkdir(parents=True)

            result = cache_cleanup.cleanup_backend_runtime_caches(backend_root, dry_run=True)

            self.assertEqual(result.removed_paths, ["tmp/storage/_cache_fill_short_form_abcd1234"])
            self.assertTrue(job_dir.exists())


if __name__ == "__main__":
    unittest.main()
