import unittest

from ml import eval_clip_finder


class FilterManifestTiersTests(unittest.TestCase):
    def test_filter_keeps_only_requested_tiers(self) -> None:
        manifest = {
            "123": [
                {"start": 1.0, "end": 2.0, "gt_tier": "gold"},
                {"start": 3.0, "end": 4.0, "gt_tier": "review"},
                {"start": 5.0, "end": 6.0},  # unverified — excluded under a filter
            ],
            "456": [
                {"start": 7.0, "end": 8.0, "gt_tier": "silver"},
            ],
        }

        filtered = eval_clip_finder.filter_manifest_tiers(manifest, {"gold", "silver"})

        self.assertEqual(len(filtered["123"]), 1)
        self.assertEqual(filtered["123"][0]["gt_tier"], "gold")
        self.assertEqual(len(filtered["456"]), 1)


if __name__ == "__main__":
    unittest.main()
