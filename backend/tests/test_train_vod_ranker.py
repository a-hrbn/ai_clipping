import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import train_vod_ranker


class TrainVodRankerTests(unittest.TestCase):
    def test_build_composite_dataset_marks_short_form_as_supplemental(self) -> None:
        vod_rows = [
            {
                "candidate_id": "vod-1",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod",
                "supervision_tier": "twitch_clip_proxy",
                "feature_provenance": {"has_vod_context": True},
                "sample_weight": 1.5,
                "features": {},
            }
        ]
        short_form_rows = [
            {
                "candidate_id": "sf-1",
                "label": 1,
                "streamer": "jynxzi",
                "feature_provenance": {"has_llm_semantics": True},
                "features": {},
            }
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=vod_rows):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                supplemental_rows=short_form_rows,
                supplemental_weight=0.35,
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dataset_source"], "twitch_vod")
        self.assertEqual(rows[1]["dataset_source"], "short_form_supplemental")
        self.assertEqual(rows[1]["label_class"], "short_form_positive")
        self.assertEqual(rows[1]["supervision_tier"], "short_form_discovered")
        self.assertEqual(rows[1]["ordinal_label"], 3)
        self.assertTrue(rows[1]["feature_provenance"]["has_llm_semantics"])
        self.assertAlmostEqual(rows[1]["sample_weight"], 0.35)

    def test_filter_short_form_rows_matches_broadcaster(self) -> None:
        rows = [
            {"streamer": "jynxzi", "label": 1},
            {"creator_id": "tarik", "label": 0},
        ]
        filtered = train_vod_ranker._filter_short_form_rows(rows, "jynxzi")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["streamer"], "jynxzi")

    def test_train_per_streamer_writes_streamer_specific_dataset(self) -> None:
        labels = [
            {"broadcaster": "jynxzi", "label_class": "positive"},
            {"broadcaster": "tarik", "label_class": "hard_negative"},
        ]

        def _fake_build_composite_dataset(streamer_labels, profile, work_dir, supplemental_rows, supplemental_weight):
            return [{
                "candidate_id": f"{profile}-1",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod",
                "sample_weight": 1.5,
                "streamer": profile,
                "features": {},
            }]

        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.train_vod_ranker.build_composite_dataset", side_effect=_fake_build_composite_dataset), \
             patch("ml.train_vod_ranker.train_ranker.train_from_dataset", return_value=0) as train_mock:
            rc = train_vod_ranker._train_per_streamer(
                labels,
                short_form_rows=[{"streamer": "jynxzi", "label": 1, "features": {}}],
                artifact_dir=Path(tmp),
                work_dir=Path(tmp) / "work",
                short_form_weight=0.35,
            )

            self.assertEqual(rc, 0)
            self.assertEqual(train_mock.call_count, 2)
            self.assertTrue((Path(tmp) / "jynxzi_vod_training_rows.jsonl").exists())
            self.assertTrue((Path(tmp) / "tarik_vod_training_rows.jsonl").exists())
            dataset_rows = [
                json.loads(line)
                for line in (Path(tmp) / "jynxzi_vod_training_rows.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(dataset_rows[0]["streamer"], "jynxzi")


if __name__ == "__main__":
    unittest.main()
