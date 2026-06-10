import unittest

from ml.gt_pipeline import _carry_over_verification


class CarryOverVerificationTests(unittest.TestCase):
    def test_carries_tier_and_refined_boundaries_for_same_repost(self) -> None:
        previous = [{
            "source_url": "https://youtu.be/a",
            "start": 500.0, "end": 520.0,
            "transcript_start": 495.0, "transcript_end": 521.0,
            "gt_tier": "gold", "boundary_source": "audio_refined",
        }]
        new = [{"source_url": "https://youtu.be/a", "start": 494.0, "end": 522.0}]

        _carry_over_verification(previous, new)

        self.assertEqual(new[0]["gt_tier"], "gold")
        self.assertEqual(new[0]["start"], 500.0)
        self.assertEqual(new[0]["end"], 520.0)
        self.assertEqual(new[0]["transcript_start"], 494.0)
        self.assertEqual(new[0]["boundary_source"], "audio_refined")

    def test_does_not_carry_across_different_positions(self) -> None:
        previous = [{
            "source_url": "https://youtu.be/a",
            "start": 500.0, "end": 520.0, "gt_tier": "gold",
        }]
        new = [{"source_url": "https://youtu.be/a", "start": 4000.0, "end": 4026.0}]

        _carry_over_verification(previous, new)

        self.assertNotIn("gt_tier", new[0])

    def test_does_not_carry_unverified_or_foreign_moments(self) -> None:
        previous = [
            {"source_url": "https://youtu.be/a", "start": 500.0, "end": 520.0},
            {"source_url": "https://youtu.be/b", "start": 100.0, "end": 120.0, "gt_tier": "silver"},
        ]
        new = [{"source_url": "https://youtu.be/a", "start": 500.0, "end": 520.0}]

        _carry_over_verification(previous, new)

        self.assertNotIn("gt_tier", new[0])


if __name__ == "__main__":
    unittest.main()
