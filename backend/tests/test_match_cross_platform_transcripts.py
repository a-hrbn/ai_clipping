import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import match_cross_platform_transcripts as matcher


def segment(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text)


class CrossPlatformTranscriptMatchTests(unittest.TestCase):
    def test_match_reposts_to_vods_emits_transcript_training_row(self) -> None:
        vod_rows = [{"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"}]
        repost_rows = [{
            "video_path": "/tmp/repost.mp4",
            "platform": "youtube",
            "views": 150000,
            "source_url": "https://youtube.com/shorts/repost",
            "profile_filter_version": 8,
        }]
        vod_transcript = [
            segment(10.0, 14.0, "nothing important happens here"),
            segment(90.0, 95.0, "that was the funniest clutch I have ever seen"),
            segment(95.0, 100.0, "chat is losing it after that play"),
        ]
        repost_transcript = [
            segment(0.0, 5.0, "funniest clutch I have ever seen"),
            segment(5.0, 9.0, "chat is losing it"),
        ]

        with patch.object(matcher, "_load_vod_transcript") as load_vod, \
             patch.object(matcher.audio_matcher, "_resolve_media_path", return_value="/tmp/repost.mp4"), \
             patch.object(matcher, "_load_or_build_repost_transcript", return_value=(repost_transcript, True)):
            load_vod.return_value = matcher.VodTranscript(
                row=vod_rows[0],
                segments=vod_transcript,
                segment_tokens=[matcher._tokens(row.text) for row in vod_transcript],
                token_to_indices={
                    token: [idx]
                    for idx, row in enumerate(vod_transcript)
                    for token in set(matcher._tokens(row.text))
                },
            )
            rows, diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_similarity=0.6,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_source"], "cross_platform_transcript_match")
        self.assertEqual(rows[0]["label_source"], "cross_platform_transcript_match")
        self.assertEqual(rows[0]["supervision_tier"], "cross_platform_outcome")
        self.assertEqual(rows[0]["window_start"], 90.0)
        self.assertGreaterEqual(rows[0]["transcript_similarity"], 0.6)
        self.assertNotIn("fingerprint_confidence", rows[0])
        self.assertEqual(diagnostics["verification_mode"], "transcript")
        self.assertEqual(diagnostics["matched_rows"], 1)

    def test_diagnostics_report_below_threshold_transcript_candidates(self) -> None:
        vod_rows = [{"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"}]
        repost_rows = [{"video_path": "/tmp/repost.mp4", "platform": "youtube"}]
        vod_transcript = [segment(10.0, 20.0, "alpha beta gamma delta")]
        repost_transcript = [segment(0.0, 8.0, "alpha beta unrelated words")]

        with patch.object(matcher, "_load_vod_transcript") as load_vod, \
             patch.object(matcher.audio_matcher, "_resolve_media_path", return_value="/tmp/repost.mp4"), \
             patch.object(matcher, "_load_or_build_repost_transcript", return_value=(repost_transcript, False)):
            load_vod.return_value = matcher.VodTranscript(
                row=vod_rows[0],
                segments=vod_transcript,
                segment_tokens=[matcher._tokens(vod_transcript[0].text)],
                token_to_indices={token: [0] for token in set(matcher._tokens(vod_transcript[0].text))},
            )
            rows, diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_similarity=0.95,
            )

        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["matched_rows"], 0)
        self.assertEqual(len(diagnostics["best_match_rows"]), 1)
        self.assertEqual(diagnostics["recommendation"]["status"], "threshold_too_high_or_no_text_match")

    def test_parallel_repost_matching_matches_serial_results(self) -> None:
        vod_rows = [{"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"}]
        repost_rows = [
            {
                "video_path": "/tmp/repost-a.mp4",
                "platform": "youtube",
                "source_url": "https://youtube.com/shorts/a",
                "profile_filter_version": 8,
            },
            {
                "video_path": "/tmp/repost-b.mp4",
                "platform": "youtube",
                "source_url": "https://youtube.com/shorts/b",
                "profile_filter_version": 8,
            },
        ]
        vod_transcript = [
            segment(10.0, 14.0, "nothing important happens here"),
            segment(90.0, 95.0, "that was the funniest clutch I have ever seen"),
            segment(95.0, 100.0, "chat is losing it after that play"),
            segment(150.0, 155.0, "prime age for gaming is seventeen to nineteen"),
        ]
        repost_transcripts = [
            [segment(0.0, 5.0, "funniest clutch I have ever seen"), segment(5.0, 9.0, "chat is losing it")],
            [segment(0.0, 5.0, "prime age for gaming is seventeen to nineteen")],
        ]
        vod_loaded = matcher.VodTranscript(
            row=vod_rows[0],
            segments=vod_transcript,
            segment_tokens=[matcher._tokens(row.text) for row in vod_transcript],
            token_to_indices={
                token: [idx]
                for idx, row in enumerate(vod_transcript)
                for token in set(matcher._tokens(row.text))
            },
        )

        def load_repost(repost: dict, _path: Path, *, cache_root: Path):
            index = 0 if repost["source_url"].endswith("/a") else 1
            return repost_transcripts[index], True

        with patch.object(matcher, "_load_vod_transcript", return_value=vod_loaded), \
             patch.object(matcher.audio_matcher, "_resolve_media_path", side_effect=lambda path: Path(path)), \
             patch.object(matcher, "_load_or_build_repost_transcript", side_effect=load_repost):
            serial_rows, serial_diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_similarity=0.5,
                transcript_match_workers=1,
            )
            parallel_rows, parallel_diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_similarity=0.5,
                transcript_match_workers=2,
            )

        self.assertEqual(parallel_rows, serial_rows)
        self.assertEqual(parallel_diagnostics["best_match_rows"], serial_diagnostics["best_match_rows"])
        self.assertEqual(parallel_diagnostics["transcript_match_workers"], 2)

    def test_repost_transcription_skips_diarization(self) -> None:
        repost = {"source_url": "https://youtube.com/shorts/abc"}
        with self.subTest("cache miss"):
            with tempfile.TemporaryDirectory() as tmp, \
                 patch.object(matcher.audio_extractor, "extract") as extract_mock, \
                 patch.object(matcher.transcriber, "transcribe", return_value=[]) as transcribe_mock:
                video_path = Path(tmp) / "repost.mp4"
                video_path.write_bytes(b"video")
                extract_mock.return_value = Path(tmp) / "audio.wav"
                transcript, cache_hit = matcher._load_or_build_repost_transcript(
                    repost,
                    video_path,
                    cache_root=Path(tmp) / "transcript-cache",
                )

        self.assertEqual(transcript, [])
        self.assertFalse(cache_hit)
        transcribe_mock.assert_called_once()
        self.assertFalse(transcribe_mock.call_args.kwargs["enable_diarization"])


if __name__ == "__main__":
    unittest.main()
