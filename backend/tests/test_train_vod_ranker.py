import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import train_vod_ranker


class TrainVodRankerTests(unittest.TestCase):
    def test_main_builds_dataset_and_trains_without_exercising_downloader(self) -> None:
        dataset_rows = [
            {
                "candidate_id": "vod-1",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "supervision_tier": "twitch_clip_proxy",
                "feature_provenance": {"has_vod_context": True},
                "features": {"hook_strength": 0.8},
            }
        ]
        labels_path = Path(tempfile.gettempdir()) / "test_labels.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.jsonl"
            labels_path.write_text(json.dumps({"broadcaster": "jynxzi", "vod_id": "123"}) + "\n", encoding="utf-8")
            dataset_out = Path(tmp) / "dataset.jsonl"
            artifact_dir = Path(tmp) / "artifacts"

            with patch("ml.train_vod_ranker.build_composite_dataset", return_value=dataset_rows) as build_mock, \
                 patch("ml.train_vod_ranker._train_models_for_dataset", return_value=0) as train_mock, \
                 patch(
                     "sys.argv",
                     [
                         "train_vod_ranker.py",
                         "--labels", str(labels_path),
                         "--dataset-out", str(dataset_out),
                         "--profile", "jynxzi",
                         "--artifact-dir", str(artifact_dir),
                         "--work-dir", str(Path(tmp) / "work"),
                     ],
                 ):
                rc = train_vod_ranker.main()

            self.assertEqual(rc, 0)
            build_mock.assert_called_once()
            train_mock.assert_called_once_with(dataset_out, "jynxzi", artifact_dir)
            written_rows = [
                json.loads(line)
                for line in dataset_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(written_rows, dataset_rows)

    def test_train_models_for_dataset_trains_boundary_ranker_when_dataset_has_references(self) -> None:
        rows = [
            {
                "candidate_id": "vod-1",
                "label": 1,
                "label_class": "positive",
                "boundary_reference_start": 10.0,
                "boundary_reference_end": 22.0,
                "boundary_abs_start_error_sec": 0.0,
                "boundary_abs_end_error_sec": 0.0,
                "boundary_abs_total_error_sec": 0.0,
                "features": {},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset.jsonl"
            dataset_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            fake_boundary_eval = type("BoundaryEval", (), {
                "_has_boundary_reference": staticmethod(lambda row: True),
                "_boundary_metrics": staticmethod(lambda rows: {"mean_abs_total_error_sec": 0.0}),
            })
            fake_boundary_train = type("BoundaryTrain", (), {
                "train_from_rows": staticmethod(lambda rows, profile, artifact_dir: 0),
            })

            with patch("ml.train_vod_ranker.train_ranker.train_from_dataset", return_value=0) as ranker_mock, \
                 patch("ml.train_vod_ranker.ranker_dataset_quality", return_value={"ok": True}), \
                 patch("ml.train_vod_ranker._boundary_training_modules", return_value=(fake_boundary_eval, fake_boundary_train)), \
                 patch.object(fake_boundary_train, "train_from_rows", return_value=0) as boundary_mock:
                rc = train_vod_ranker._train_models_for_dataset(dataset_path, "jynxzi", Path(tmp) / "artifacts")

        self.assertEqual(rc, 0)
        ranker_mock.assert_called_once()
        boundary_mock.assert_called_once()

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
             patch("ml.train_vod_ranker._train_models_for_dataset", return_value=0) as train_mock:
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

    def test_ranker_dataset_quality_rejects_single_vod_single_clip_dataset(self) -> None:
        rows = [
            {
                "candidate_id": "cand-1",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "vod_id": "vod-1",
                "clip_ids": ["clip-a"],
            },
            {
                "candidate_id": "cand-2",
                "label": 0,
                "label_class": "hard_negative",
                "dataset_source": "twitch_vod_primary",
                "vod_id": "vod-1",
                "clip_ids": [],
            },
        ]

        quality = train_vod_ranker.ranker_dataset_quality(rows)

        self.assertFalse(quality["ok"])
        self.assertIn("primary_unique_vods_below_minimum", quality["violations"])
        self.assertIn("primary_positive_clusters_below_minimum", quality["violations"])
        self.assertIn("primary_ranking_groups_below_minimum", quality["violations"])


if __name__ == "__main__":
    unittest.main()
