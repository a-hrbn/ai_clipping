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


def transcript_match(
    vod_id: str,
    start: float,
    end: float,
    *,
    score: float = 0.8,
    kind: str = "multi_window_chunk",
    chunk_index: int | None = 1,
) -> matcher.TranscriptMatch:
    return matcher.TranscriptMatch(
        score=score,
        token_recall=score,
        bigram_recall=score,
        sequence_ratio=score,
        start=start,
        end=end,
        vod={"vod_id": vod_id, "vod_url": f"https://www.twitch.tv/videos/{vod_id}"},
        vod_excerpt=f"vod {vod_id} {start}",
        repost_excerpt=f"repost {start}",
        match_kind=kind,
        acceptance_reason="transcript_threshold",
        acceptance_threshold=0.5,
        source_score=score,
        chunk_index=chunk_index,
        chunk_count=4,
    )


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
        self.assertEqual(rows[0]["outcome_view_tier"], "viral")
        self.assertEqual(rows[0]["outcome_view_sample_weight"], 2.0)
        self.assertEqual(rows[0]["sample_weight"], 2.0)
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

    def test_multi_window_chunks_emit_separate_positive_rows_for_recaps(self) -> None:
        vod_rows = [{"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"}]
        repost_rows = [{
            "video_path": "/tmp/repost.mp4",
            "platform": "youtube",
            "views": 4_000_000,
            "source_url": "https://youtube.com/shorts/recap",
        }]
        first_moment = "red box unboxing signed jersey gold chain surprise reaction"
        second_moment = "blue box unboxing rare controller chat screamed loud reaction"
        vod_transcript = [
            segment(100.0, 106.0, first_moment),
            segment(300.0, 306.0, second_moment),
        ]
        repost_transcript = [
            segment(0.0, 5.0, first_moment),
            segment(5.0, 10.0, second_moment),
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
                min_similarity=0.9,
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["transcript_match_kind"] for row in rows}, {"multi_window_chunk"})
        self.assertEqual([row["window_start"] for row in rows], [100.0, 300.0])
        self.assertEqual({row["transcript_chunk_count"] for row in rows}, {2})
        self.assertEqual(diagnostics["multi_window_chunk_matches"], 2)

    def test_multi_window_cleanup_merges_near_duplicate_windows(self) -> None:
        matches = [
            transcript_match("123", 100.0, 110.0, score=0.82, chunk_index=1),
            transcript_match("123", 123.0, 133.0, score=0.78, chunk_index=2),
            transcript_match("123", 300.0, 310.0, score=0.81, chunk_index=3),
        ]

        cleaned, stats = matcher._clean_accepted_matches(matches)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(stats["multi_window_near_duplicate_merges"], 1)
        self.assertEqual(stats["multi_window_cross_vod_filtered"], 0)
        self.assertEqual(cleaned[0].match_kind, "multi_window_merged")
        self.assertEqual(cleaned[0].start, 100.0)
        self.assertEqual(cleaned[0].end, 133.0)
        self.assertEqual(cleaned[1].start, 300.0)

    def test_multi_window_cleanup_keeps_only_top_vod(self) -> None:
        matches = [
            transcript_match("vod-a", 100.0, 108.0, score=0.82, chunk_index=1),
            transcript_match("vod-a", 500.0, 508.0, score=0.80, chunk_index=2),
            transcript_match("vod-b", 200.0, 208.0, score=0.50, chunk_index=3),
            transcript_match("vod-b", 700.0, 708.0, score=0.30, chunk_index=4),
            transcript_match("vod-c", 900.0, 908.0, score=0.40, kind="single_window", chunk_index=None),
        ]

        cleaned, stats = matcher._clean_accepted_matches(matches)

        self.assertEqual({match.vod["vod_id"] for match in cleaned}, {"vod-a"})
        self.assertEqual(stats["multi_window_cross_vod_filtered"], 3)
        self.assertEqual(stats["multi_window_near_duplicate_merges"], 0)

    def test_cleanup_filters_weak_cross_vod_match_when_best_match_is_single_window(self) -> None:
        matches = [
            transcript_match("vod-a", 100.0, 120.0, score=0.92, kind="single_window", chunk_index=None),
            transcript_match("vod-b", 500.0, 508.0, score=0.49, kind="multi_window_chunk", chunk_index=1),
        ]

        cleaned, stats = matcher._clean_accepted_matches(matches)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].vod["vod_id"], "vod-a")
        self.assertEqual(stats["multi_window_cross_vod_filtered"], 1)

    def test_cleanup_keeps_strong_secondary_vod_match(self) -> None:
        matches = [
            transcript_match("vod-a", 100.0, 108.0, score=0.82, chunk_index=1),
            transcript_match("vod-a", 500.0, 508.0, score=0.80, chunk_index=2),
            transcript_match("vod-b", 900.0, 908.0, score=0.57, kind="single_window", chunk_index=None),
        ]

        cleaned, stats = matcher._clean_accepted_matches(matches)

        self.assertEqual({match.vod["vod_id"] for match in cleaned}, {"vod-a", "vod-b"})
        self.assertEqual(stats["multi_window_cross_vod_filtered"], 0)

    def test_high_view_near_miss_uses_view_weighted_threshold(self) -> None:
        vod_rows = [{"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"}]
        repost_rows = [{
            "video_path": "/tmp/repost.mp4",
            "platform": "youtube",
            "views": 750_000,
            "source_url": "https://youtube.com/shorts/high-view-near-miss",
        }]
        vod_transcript = [segment(10.0, 18.0, "alpha beta gamma delta epsilon zeta eta theta")]
        repost_transcript = [segment(0.0, 8.0, "alpha beta gamma delta extra one two three")]

        with patch.object(matcher, "_load_vod_transcript") as load_vod, \
             patch.object(matcher.audio_matcher, "_resolve_media_path", return_value="/tmp/repost.mp4"), \
             patch.object(matcher, "_load_or_build_repost_transcript", return_value=(repost_transcript, True)):
            load_vod.return_value = matcher.VodTranscript(
                row=vod_rows[0],
                segments=vod_transcript,
                segment_tokens=[matcher._tokens(vod_transcript[0].text)],
                token_to_indices={token: [0] for token in set(matcher._tokens(vod_transcript[0].text))},
            )
            rows, diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_similarity=0.6,
            )

        self.assertEqual(len(rows), 1)
        self.assertLess(rows[0]["transcript_similarity"], 0.6)
        self.assertEqual(rows[0]["transcript_acceptance_reason"], "view_weighted_transcript_threshold")
        self.assertEqual(diagnostics["view_weighted_accepts"], 1)

    def test_audio_fallback_accepts_transcript_near_miss(self) -> None:
        vod_rows = [{"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"}]
        repost_rows = [{
            "video_path": "/tmp/repost.mp4",
            "platform": "youtube",
            "views": 10_000,
            "source_url": "https://youtube.com/shorts/audio-fallback",
        }]
        vod_transcript = [segment(10.0, 18.0, "alpha beta gamma delta epsilon zeta eta theta")]
        repost_transcript = [segment(0.0, 8.0, "alpha beta gamma delta extra one two three")]
        fallback_match = matcher.TranscriptMatch(
            score=0.47,
            token_recall=0.5,
            bigram_recall=0.42,
            sequence_ratio=0.35,
            start=50.0,
            end=58.0,
            vod=vod_rows[0],
            vod_excerpt="alpha beta gamma delta",
            repost_excerpt="alpha beta gamma delta extra",
            match_kind="audio_fingerprint_fallback",
            source_score=0.47,
            audio_confidence=0.62,
        )

        with patch.object(matcher, "_load_vod_transcript") as load_vod, \
             patch.object(matcher.audio_matcher, "_resolve_media_path", return_value="/tmp/repost.mp4"), \
             patch.object(matcher, "_load_or_build_repost_transcript", return_value=(repost_transcript, True)), \
             patch.object(matcher, "_load_audio_vod_fingerprints", return_value=[(vod_rows[0], object())]), \
             patch.object(matcher, "_audio_fingerprint_fallback", return_value=fallback_match):
            load_vod.return_value = matcher.VodTranscript(
                row=vod_rows[0],
                segments=vod_transcript,
                segment_tokens=[matcher._tokens(vod_transcript[0].text)],
                token_to_indices={token: [0] for token in set(matcher._tokens(vod_transcript[0].text))},
            )
            rows, diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_similarity=0.6,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transcript_match_kind"], "audio_fingerprint_fallback")
        self.assertEqual(rows[0]["transcript_acceptance_reason"], "audio_fingerprint_fallback")
        self.assertEqual(rows[0]["fingerprint_confidence"], 0.62)
        self.assertEqual(diagnostics["audio_fallback_attempts"], 1)
        self.assertEqual(diagnostics["audio_fallback_matches"], 1)

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
