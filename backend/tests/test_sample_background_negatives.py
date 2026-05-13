import unittest

from ml.sample_background_negatives import sample_background_windows


class SampleBackgroundNegativesTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_uses_middle_80_percent(self) -> None:
        first = sample_background_windows("vod-1", duration_seconds=1000, count=6)
        second = sample_background_windows("vod-1", duration_seconds=1000, count=6)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        for window in first:
            self.assertGreaterEqual(window.start, 100.0)
            self.assertLessEqual(window.end, 900.0)
            self.assertAlmostEqual(window.end - window.start, 30.0, places=1)

    def test_sampling_rejects_blocked_overlap(self) -> None:
        windows = sample_background_windows(
            "vod-2",
            duration_seconds=300,
            blocked_windows=[(120.0, 180.0)],
            count=10,
            window_seconds=30.0,
            max_overlap_ratio=0.20,
        )

        for window in windows:
            overlap = min(window.end, 180.0) - max(window.start, 120.0)
            self.assertLessEqual(max(0.0, overlap) / 30.0, 0.20)


if __name__ == "__main__":
    unittest.main()
