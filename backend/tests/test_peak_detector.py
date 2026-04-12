import sys
import types
import unittest

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import peak_detector
from services.timeline_types import SegmentScore
from services.transcriber import RichSegment


class PeakDetectorTests(unittest.TestCase):
    def test_detect_peaks_keeps_strong_local_maximum(self) -> None:
        timeline = [
            SegmentScore(segment_index=0, start=0.0, end=2.0, virality_score=1.5, smoothed_score=1.5),
            SegmentScore(segment_index=1, start=2.0, end=4.0, virality_score=5.4, smoothed_score=5.4),
            SegmentScore(segment_index=2, start=4.0, end=6.0, virality_score=1.8, smoothed_score=1.8),
        ]

        peaks = peak_detector.detect_peaks(timeline, min_peak_score=3.0, min_prominence=0.5, min_gap_sec=5.0)

        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0].segment_index, 1)
        self.assertEqual(peaks[0].peak_time, 3.0)

    def test_detect_peaks_falls_back_to_top_segments_when_no_true_peak_exists(self) -> None:
        timeline = [
            SegmentScore(segment_index=0, start=0.0, end=2.0, virality_score=2.1, smoothed_score=2.1),
            SegmentScore(segment_index=1, start=2.0, end=4.0, virality_score=2.0, smoothed_score=2.0),
            SegmentScore(segment_index=2, start=4.0, end=6.0, virality_score=1.9, smoothed_score=1.9),
        ]

        peaks = peak_detector.detect_peaks(timeline, min_peak_score=3.0, min_prominence=0.5, min_gap_sec=1.0)

        self.assertTrue(peaks)
        self.assertEqual(peaks[0].segment_index, 0)

    def test_build_peak_candidates_uses_peak_windows_and_anchor_times(self) -> None:
        peaks = [
            peak_detector.DetectedPeak(
                peak_time=12.0,
                peak_score=5.2,
                left_base_time=6.0,
                right_base_time=18.0,
                prominence=1.5,
                window_start=6.0,
                window_end=18.0,
                segment_index=2,
            )
        ]
        segments = [
            RichSegment(start=0.0, end=4.0, text="a"),
            RichSegment(start=4.0, end=8.0, text="b"),
            RichSegment(start=8.0, end=12.0, text="c"),
            RichSegment(start=12.0, end=16.0, text="d"),
            RichSegment(start=16.0, end=20.0, text="e"),
        ]

        candidates = peak_detector.build_peak_candidates(peaks, segments)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source, "peak")
        self.assertEqual(candidates[0].anchor_time, 12.0)
        self.assertLessEqual(candidates[0].start, 6.0)
        self.assertGreaterEqual(candidates[0].end, 18.0)


if __name__ == "__main__":
    unittest.main()
