import contextlib
import io
import json
import unittest
from unittest.mock import patch

from ml import match_cross_platform_audio_fingerprints as matcher


class CrossPlatformFingerprintTests(unittest.TestCase):
    def test_best_fingerprint_alignment_finds_repost_offset(self) -> None:
        needle = [10, 20, 30, 40]
        haystack = [1, 2, 3, *needle, 99, 100]

        confidence, offset = matcher.best_fingerprint_alignment(needle, haystack)

        self.assertEqual(offset, 3)
        self.assertEqual(confidence, 1.0)

    def test_fingerprint_similarity_penalizes_bit_distance(self) -> None:
        perfect = matcher._fingerprint_similarity([0b1111], [0b1111])
        mismatch = matcher._fingerprint_similarity([0b1111], [0b0000])

        self.assertGreater(perfect, mismatch)

    def test_match_reposts_to_vods_emits_cross_platform_training_row(self) -> None:
        def fake_fingerprint(path, _fpcalc):
            if str(path).endswith("repost.mp4"):
                return matcher.Fingerprint(duration=20.0, values=[10, 20, 30, 40])
            return matcher.Fingerprint(duration=100.0, values=[1, 2, 10, 20, 30, 40, 99])

        vod_rows = [{
            "vod_id": "123",
            "vod_url": "https://www.twitch.tv/videos/123",
            "video_path": "/tmp/vod.mp4",
            "source_start_offset": 100.0,
        }]
        repost_rows = [{
            "video_path": "/tmp/repost.mp4",
            "platform": "tiktok",
            "views": 150000,
            "profile_filter_version": 7,
        }]
        with patch.object(matcher.shutil, "which", return_value="/usr/bin/fpcalc"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch.object(matcher, "_fingerprint", side_effect=fake_fingerprint):
            rows = matcher.match_reposts_to_vods(vod_rows, repost_rows, min_confidence=0.7)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_source"], "cross_platform_audio_fingerprint")
        self.assertEqual(rows[0]["supervision_tier"], "cross_platform_outcome")
        self.assertEqual(rows[0]["label"], 1)
        self.assertGreaterEqual(rows[0]["window_start"], 100.0)
        self.assertGreaterEqual(rows[0]["fingerprint_confidence"], 0.7)
        self.assertEqual(rows[0]["profile_filter_version"], 7)
        self.assertEqual(rows[0]["outcome_view_tier"], "viral")
        self.assertEqual(rows[0]["outcome_view_sample_weight"], 2.0)
        self.assertEqual(rows[0]["sample_weight"], 2.0)

    def test_match_reposts_to_vods_skips_missing_vod_media(self) -> None:
        repost_rows = [{"video_path": "/tmp/repost.mp4", "platform": "youtube"}]
        vod_rows = [{"vod_id": "missing", "video_path": "/tmp/missing-vod.mp4"}]

        with patch.object(matcher.shutil, "which", return_value="/usr/bin/fpcalc"), \
             patch.object(matcher, "_resolve_media_path", side_effect=[None, "/tmp/repost.mp4"]), \
             patch.object(matcher, "_fingerprint", return_value=matcher.Fingerprint(duration=10.0, values=[1, 2, 3])):
            rows = matcher.match_reposts_to_vods(vod_rows, repost_rows)

        self.assertEqual(rows, [])

    def test_match_reposts_to_vods_uses_cached_vod_fingerprint_path(self) -> None:
        def fake_fingerprint(path, _fpcalc):
            if str(path).endswith("repost.mp4"):
                return matcher.Fingerprint(duration=20.0, values=[10, 20, 30, 40])
            raise AssertionError("VOD media should not be fingerprinted when fingerprint_path exists")

        vod_rows = [{
            "vod_id": "123",
            "vod_url": "https://www.twitch.tv/videos/123",
            "fingerprint_path": "/tmp/vod.fpcalc.json",
            "source_start_offset": 0.0,
        }]
        repost_rows = [{"video_path": "/tmp/repost.mp4", "platform": "youtube"}]

        with patch.object(matcher.shutil, "which", return_value="/usr/bin/fpcalc"), \
             patch.object(matcher, "_resolve_media_path", side_effect=lambda raw: raw if raw else None), \
             patch.object(matcher, "_read_fingerprint_file", return_value=matcher.Fingerprint(duration=100.0, values=[1, 2, 10, 20, 30, 40, 99])), \
             patch.object(matcher, "_fingerprint", side_effect=fake_fingerprint):
            rows = matcher.match_reposts_to_vods(vod_rows, repost_rows, min_confidence=0.7)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vod_id"], "123")
        self.assertGreaterEqual(rows[0]["fingerprint_confidence"], 0.7)

    def test_match_diagnostics_report_below_threshold_distribution(self) -> None:
        vod_rows = [{"vod_id": "123", "video_path": "/tmp/vod.mp4", "source_start_offset": 10.0}]
        repost_rows = [{"video_path": "/tmp/repost.mp4", "platform": "youtube", "source_url": "https://youtu.be/short"}]

        with patch.object(matcher.shutil, "which", return_value="/usr/bin/fpcalc"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch.object(matcher, "_fingerprint", return_value=matcher.Fingerprint(duration=10.0, values=[1, 2, 3, 4])), \
             patch.object(matcher, "best_fingerprint_alignment", return_value=(0.56, 0)):
            rows, diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_confidence=0.72,
            )

        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["matched_rows"], 0)
        self.assertEqual(diagnostics["confidence_distribution"]["median"], 0.56)
        self.assertEqual(diagnostics["threshold_ladder"]["0.55"], 1)
        self.assertEqual(diagnostics["threshold_ladder"]["0.72"], 0)
        self.assertEqual(diagnostics["recommendation"]["status"], "consider_lowering_threshold")
        self.assertEqual(len(diagnostics["best_match_rows"]), 1)

    def test_match_progress_logs_are_emitted_to_stderr(self) -> None:
        vod_rows = [{"vod_id": "123", "video_path": "/tmp/vod.mp4"}]
        repost_rows = [{"video_path": "/tmp/repost.mp4", "platform": "youtube"}]
        stderr = io.StringIO()

        with patch.object(matcher.shutil, "which", return_value="/usr/bin/fpcalc"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch.object(matcher, "_fingerprint", return_value=matcher.Fingerprint(duration=10.0, values=[1, 2, 3, 4])), \
             patch.object(matcher, "best_fingerprint_alignment", return_value=(0.8, 0)), \
             contextlib.redirect_stderr(stderr):
            rows, _diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_confidence=0.72,
                log_progress=True,
                vod_progress_interval=1,
                repost_progress_interval=1,
            )

        events = [
            json.loads(line)["event"]
            for line in stderr.getvalue().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertIn("cross_platform_fingerprint_match_start", events)
        self.assertIn("cross_platform_fingerprint_vod_progress", events)
        self.assertIn("cross_platform_fingerprint_repost_progress", events)
        self.assertIn("cross_platform_fingerprint_match_done", events)


if __name__ == "__main__":
    unittest.main()
