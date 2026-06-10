import random
import unittest
from types import SimpleNamespace

from ml import match_cross_platform_audio_fingerprints as audio_matcher
from ml import match_cross_platform_transcripts as matcher


def _random_walk_values(count: int, seed: int = 7) -> list[int]:
    rng = random.Random(seed)
    values = [rng.getrandbits(32)]
    for _ in range(count - 1):
        values.append(values[-1] ^ (1 << rng.randrange(32)) ^ (1 << rng.randrange(32)))
    return values


def _segment(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text)


_CATCHPHRASE = "chat that is actually insane bro no shot"


def _vod_with_catchphrase_decoy(vod_id: str = "123") -> matcher.VodTranscript:
    """A VOD where the catchphrase occurs everywhere but the unique content
    ('gotham teaches en passant rook gambit') occurs exactly once at 500s."""
    segments = []
    t = 0.0
    for _ in range(40):
        segments.append(_segment(t, t + 10.0, _CATCHPHRASE))
        t += 10.0
    segments.append(_segment(500.0, 510.0, f"{_CATCHPHRASE} gotham teaches en passant rook gambit"))
    t = 510.0
    for _ in range(40):
        segments.append(_segment(t, t + 10.0, _CATCHPHRASE))
        t += 10.0
    return matcher._build_vod_transcript({"vod_id": vod_id}, segments)


class IdfRankingTests(unittest.TestCase):
    def test_best_match_prefers_content_specific_window_over_catchphrases(self) -> None:
        vod = _vod_with_catchphrase_decoy()
        repost_text = f"{_CATCHPHRASE} gotham teaches en passant rook gambit"
        repost_tokens = matcher._tokens(repost_text)

        best = matcher._best_transcript_match(
            {"views": 0},
            repost_text=repost_text,
            repost_tokens=repost_tokens,
            repost_duration=10.0,
            vod_transcripts=[vod],
        )

        self.assertIsNotNone(best)
        self.assertEqual(best.start, 500.0)

    def test_pure_catchphrase_repost_is_flagged_position_ambiguous(self) -> None:
        vod = _vod_with_catchphrase_decoy()
        repost_text = _CATCHPHRASE
        repost_tokens = matcher._tokens(repost_text)

        best = matcher._best_transcript_match(
            {"views": 0},
            repost_text=repost_text,
            repost_tokens=repost_tokens,
            repost_duration=10.0,
            vod_transcripts=[vod],
        )

        self.assertIsNotNone(best)
        self.assertTrue(best.position_ambiguous)

    def test_unique_content_match_is_not_flagged(self) -> None:
        vod = _vod_with_catchphrase_decoy()
        repost_text = "gotham teaches en passant rook gambit opening theory lecture"
        repost_tokens = matcher._tokens(repost_text)

        best = matcher._best_transcript_match(
            {"views": 0},
            repost_text=repost_text,
            repost_tokens=repost_tokens,
            repost_duration=10.0,
            vod_transcripts=[vod],
        )

        self.assertIsNotNone(best)
        self.assertEqual(best.start, 500.0)
        self.assertFalse(best.position_ambiguous)


class MergeSpanCapTests(unittest.TestCase):
    def _match(self, start: float, end: float, score: float) -> matcher.TranscriptMatch:
        return matcher.TranscriptMatch(
            score=score, token_recall=score, bigram_recall=0.0, sequence_ratio=0.0,
            start=start, end=end, vod={"vod_id": "123"},
            vod_excerpt="", repost_excerpt="", match_kind="multi_window_chunk",
        )

    def test_merge_cluster_caps_span_to_repost_length(self) -> None:
        stats = {"multi_window_near_duplicate_merges": 0, "multi_window_merge_span_capped": 0}
        cluster = [self._match(100.0, 120.0, 0.9), self._match(150.0, 184.0, 0.6)]

        merged = matcher._merge_match_cluster(cluster, stats, repost_duration=30.0)

        # cap = 30 * 1.2 + 5 = 41s, anchored on the best window [100, 120]
        self.assertLessEqual(merged.end - merged.start, 41.0)
        self.assertLessEqual(merged.start, 100.0)
        self.assertGreaterEqual(merged.end, 120.0)
        self.assertEqual(stats["multi_window_merge_span_capped"], 1)

    def test_merge_cluster_keeps_uncapped_span_without_repost_duration(self) -> None:
        stats = {"multi_window_near_duplicate_merges": 0, "multi_window_merge_span_capped": 0}
        cluster = [self._match(100.0, 120.0, 0.9), self._match(150.0, 184.0, 0.6)]

        merged = matcher._merge_match_cluster(cluster, stats, repost_duration=0.0)

        self.assertEqual(merged.start, 100.0)
        self.assertEqual(merged.end, 184.0)
        self.assertEqual(stats["multi_window_merge_span_capped"], 0)


class AudioPeakMarginTests(unittest.TestCase):
    def test_unique_occurrence_yields_sharp_peak(self) -> None:
        vod_values = _random_walk_values(8000)
        needle = vod_values[4000:4160]
        vod_fp = audio_matcher.Fingerprint(duration=1000.0, values=vod_values)
        repost_fp = audio_matcher.Fingerprint(duration=20.0, values=needle)

        confidence, index = audio_matcher.best_fingerprint_alignment(needle, vod_values)
        margin = matcher._audio_peak_margin(repost_fp, vod_fp, confidence, index)

        self.assertEqual(index, 4000)
        self.assertGreater(margin, matcher._AUDIO_FALLBACK_MIN_PEAK_MARGIN)

    def test_repeated_occurrence_yields_no_margin(self) -> None:
        vod_values = _random_walk_values(8000)
        needle = vod_values[4000:4160]
        vod_values[6000:6160] = needle  # same audio occurs twice, >60s apart
        vod_fp = audio_matcher.Fingerprint(duration=1000.0, values=vod_values)
        repost_fp = audio_matcher.Fingerprint(duration=20.0, values=needle)

        confidence, index = audio_matcher.best_fingerprint_alignment(needle, vod_values)
        margin = matcher._audio_peak_margin(repost_fp, vod_fp, confidence, index)

        self.assertLess(margin, matcher._AUDIO_FALLBACK_MIN_PEAK_MARGIN)


if __name__ == "__main__":
    unittest.main()
