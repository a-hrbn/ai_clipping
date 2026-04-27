import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

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

            with patch(
                "ml.train_vod_ranker.build_composite_dataset_with_report",
                return_value=(dataset_rows, {"enabled": False, "filtered_rows": 0}),
            ) as build_mock, \
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
                short_form_mode="supplemental",
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dataset_source"], "twitch_vod")
        self.assertEqual(rows[1]["dataset_source"], "short_form_supplemental")
        self.assertEqual(rows[1]["label_class"], "short_form_positive")
        self.assertEqual(rows[1]["supervision_tier"], "short_form_discovered")
        self.assertEqual(rows[1]["ordinal_label"], 3)
        self.assertTrue(rows[1]["feature_provenance"]["has_llm_semantics"])
        self.assertAlmostEqual(rows[1]["sample_weight"], 0.35)

    def test_build_composite_dataset_defaults_to_vod_only(self) -> None:
        vod_rows = [
            {
                "candidate_id": "vod-1",
                "label": 1,
                "dataset_source": "twitch_vod_primary",
                "features": {},
            }
        ]
        short_form_rows = [
            {"candidate_id": "sf-1", "label": 1, "features": {}}
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=vod_rows):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                supplemental_rows=short_form_rows,
            )

        self.assertEqual(rows, vod_rows)

    def test_build_composite_dataset_merges_gold_train_and_excludes_holdout(self) -> None:
        vod_rows = [
            {
                "queue_id": "q-1",
                "candidate_id": "cand-1",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 0,
                "label_class": "hard_negative",
                "dataset_source": "twitch_vod_primary",
                "supervision_tier": "twitch_clip_proxy",
                "sample_weight": 1.0,
                "features": {"hook_strength": 0.1},
            },
            {
                "queue_id": "q-2",
                "candidate_id": "cand-2",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 0,
                "label_class": "hard_negative",
                "dataset_source": "twitch_vod_primary",
                "supervision_tier": "twitch_clip_proxy",
                "features": {"hook_strength": 0.2},
            },
        ]
        gold_rows = [
            {
                "queue_id": "q-1",
                "candidate_id": "cand-1",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 1,
                "label_class": "gold_positive",
                "supervision_tier": "gold_eval",
                "label_source": "gold_eval",
                "gold_holdout_split": "train",
                "sample_weight": 1.5,
                "features": {"hook_strength": 0.9},
            },
            {
                "queue_id": "q-3",
                "candidate_id": "cand-3",
                "job_id": "job-1",
                "vod_id": "vod-2",
                "label": 1,
                "supervision_tier": "gold_eval",
                "label_source": "gold_eval",
                "gold_holdout_split": "holdout",
                "features": {"hook_strength": 1.0},
            },
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=vod_rows):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                gold_train_rows=gold_rows,
                gold_train_weight=2.0,
            )

        self.assertEqual(len(rows), 2)
        rows_by_queue = {row["queue_id"]: row for row in rows}
        self.assertEqual(rows_by_queue["q-1"]["dataset_source"], "gold_eval_train")
        self.assertEqual(rows_by_queue["q-1"]["label"], 1)
        self.assertEqual(rows_by_queue["q-1"]["sample_weight"], 3.0)
        self.assertEqual(rows_by_queue["q-1"]["features"]["hook_strength"], 0.9)
        self.assertEqual(rows_by_queue["q-2"]["dataset_source"], "twitch_vod_primary")
        self.assertNotIn("q-3", rows_by_queue)

    def test_build_composite_dataset_excludes_gold_holdout_vods_from_all_training_sources(self) -> None:
        captured_labels = []
        labels = [
            {
                "vod_id": "vod-train",
                "vod_url": "https://www.twitch.tv/videos/111",
                "label_class": "positive",
            },
            {
                "vod_id": "vod-holdout",
                "vod_url": "https://www.twitch.tv/videos/222",
                "label_class": "positive",
            },
        ]
        gold_rows = [
            {
                "queue_id": "gold-holdout",
                "candidate_id": "holdout-cand",
                "vod_id": "vod-holdout",
                "label": 1,
                "gold_holdout_split": "holdout",
                "window_start": 10.0,
                "window_end": 25.0,
                "features": {"hook_strength": 1.0},
            },
            {
                "queue_id": "gold-train-same-vod",
                "candidate_id": "train-on-holdout-vod",
                "vod_id": "vod-holdout",
                "label": 1,
                "gold_holdout_split": "train",
                "window_start": 40.0,
                "window_end": 55.0,
                "features": {"hook_strength": 0.9},
            },
            {
                "queue_id": "gold-train-clean",
                "candidate_id": "train-clean",
                "vod_id": "vod-train",
                "label": 1,
                "gold_holdout_split": "train",
                "features": {"hook_strength": 0.8},
            },
        ]

        def _fake_build_dataset(input_labels, _profile, _work_dir):
            captured_labels.extend(input_labels)
            return [
                {
                    "queue_id": "vod-clean",
                    "candidate_id": "vod-clean",
                    "vod_id": "vod-train",
                    "label": 1,
                    "dataset_source": "twitch_vod_primary",
                    "features": {},
                },
                {
                    "queue_id": "vod-leak",
                    "candidate_id": "vod-leak",
                    "vod_id": "vod-holdout",
                    "label": 1,
                    "dataset_source": "twitch_vod_primary",
                    "features": {},
                },
            ]

        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", side_effect=_fake_build_dataset):
            rows, report = train_vod_ranker.build_composite_dataset_with_report(
                labels=labels,
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                gold_train_rows=gold_rows,
                gold_train_weight=2.0,
            )

        self.assertEqual([row["vod_id"] for row in captured_labels], ["vod-train"])
        rows_by_queue = {row["queue_id"]: row for row in rows}
        self.assertIn("vod-clean", rows_by_queue)
        self.assertIn("gold-train-clean", rows_by_queue)
        self.assertNotIn("vod-leak", rows_by_queue)
        self.assertNotIn("gold-train-same-vod", rows_by_queue)
        self.assertNotIn("gold-holdout", rows_by_queue)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["holdout_vod_count"], 1)
        self.assertGreaterEqual(report["filtered_by_stage"]["twitch_label_input"], 1)
        self.assertGreaterEqual(report["filtered_by_reason"]["holdout_vod_id"], 1)

    def test_build_composite_dataset_excludes_promoted_holdout_hard_negatives(self) -> None:
        hard_negative_rows = [
            {
                "queue_id": "hn-holdout",
                "candidate_id": "hn-holdout",
                "vod_id": "vod-a",
                "label": 0,
                "hard_negative_original_split": "holdout",
                "features": {"hook_strength": 0.2},
            },
            {
                "queue_id": "hn-train",
                "candidate_id": "hn-train",
                "vod_id": "vod-b",
                "label": 0,
                "hard_negative_original_split": "train",
                "features": {"hook_strength": 0.1},
            },
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=[]):
            rows, report = train_vod_ranker.build_composite_dataset_with_report(
                labels=[],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                hard_negative_rows=hard_negative_rows,
            )

        rows_by_queue = {row["queue_id"]: row for row in rows}
        self.assertNotIn("hn-holdout", rows_by_queue)
        self.assertIn("hn-train", rows_by_queue)
        self.assertEqual(report["filtered_by_reason"]["promoted_holdout_hard_negative"], 1)

    def test_holdout_training_exclusion_detects_window_overlap_when_vod_is_not_globally_blocked(self) -> None:
        policy = {
            "enabled": True,
            "holdout_vod_ids": [],
            "holdout_source_keys": [],
            "holdout_windows_by_vod": {"vod-a": [(10.0, 20.0)]},
        }
        overlapping = {
            "vod_id": "vod-a",
            "window_start": 18.0,
            "window_end": 22.0,
        }
        non_overlapping = {
            "vod_id": "vod-a",
            "window_start": 25.0,
            "window_end": 35.0,
        }

        self.assertEqual(
            train_vod_ranker._holdout_training_exclusion_reason(overlapping, policy),
            "holdout_window_overlap",
        )
        self.assertEqual(
            train_vod_ranker._holdout_training_exclusion_reason(non_overlapping, policy),
            "",
        )

    def test_build_composite_dataset_merges_weighted_hard_negatives(self) -> None:
        vod_rows = [
            {
                "queue_id": "q-1",
                "candidate_id": "cand-1",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "sample_weight": 1.0,
                "features": {"hook_strength": 0.9},
            },
            {
                "queue_id": "q-2",
                "candidate_id": "cand-2",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 1,
                "dataset_source": "twitch_vod_primary",
                "features": {"hook_strength": 0.7},
            },
        ]
        hard_negative_rows = [
            {
                "queue_id": "q-2",
                "candidate_id": "cand-2",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 0,
                "gold_holdout_split": "train",
                "sample_weight": 1.2,
                "features": {"hook_strength": 0.1},
            }
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=vod_rows):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                hard_negative_rows=hard_negative_rows,
                hard_negative_weight=2.5,
            )

        self.assertEqual(len(rows), 2)
        rows_by_queue = {row["queue_id"]: row for row in rows}
        self.assertEqual(rows_by_queue["q-2"]["label"], 0)
        self.assertEqual(rows_by_queue["q-2"]["dataset_source"], "gold_eval_hard_negative_train")
        self.assertEqual(rows_by_queue["q-2"]["label_class"], "gold_hard_negative")
        self.assertEqual(rows_by_queue["q-2"]["supervision_tier"], "gold_eval")
        self.assertEqual(rows_by_queue["q-2"]["gold_holdout_split"], "train")
        self.assertEqual(rows_by_queue["q-2"]["sample_weight"], 3.0)
        self.assertEqual(rows_by_queue["q-2"]["features"]["hook_strength"], 0.1)

    def test_build_composite_dataset_defaults_to_stronger_hard_negative_weight(self) -> None:
        hard_negative_rows = [
            {
                "queue_id": "q-1",
                "candidate_id": "cand-1",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 0,
                "gold_holdout_split": "train",
                "sample_weight": 1.0,
                "features": {"hook_strength": 0.1},
            }
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=[]):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                hard_negative_rows=hard_negative_rows,
            )

        self.assertEqual(rows[0]["dataset_source"], "gold_eval_hard_negative_train")
        self.assertEqual(rows[0]["sample_weight"], 4.0)

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

        def _fake_build_composite_dataset(
            streamer_labels,
            profile,
            work_dir,
            supplemental_rows,
            supplemental_weight,
            short_form_mode,
            gold_train_rows,
            gold_train_weight,
            hard_negative_rows,
            hard_negative_weight,
        ):
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
                hard_negative_rows=[{"streamer": "jynxzi", "label": 0, "features": {}}],
                artifact_dir=Path(tmp),
                work_dir=Path(tmp) / "work",
                short_form_weight=0.35,
                short_form_mode="supplemental",
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
        self.assertIn("primary_mixed_ranking_groups_below_minimum", quality["violations"])

    def test_ranker_dataset_quality_accepts_multiple_mixed_vod_groups_without_short_form(self) -> None:
        rows = []
        for vod in ("vod-a", "vod-b", "vod-c"):
            rows.append({
                "candidate_id": f"{vod}-pos",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "vod_id": vod,
                "clip_ids": [f"clip-{vod}"],
                "candidate_source": "quote",
            })
            rows.append({
                "candidate_id": f"{vod}-neg",
                "label": 0,
                "label_class": "hard_negative",
                "dataset_source": "twitch_vod_primary",
                "vod_id": vod,
                "clip_ids": [],
                "candidate_source": "twitch_label_proxy",
            })
        rows.extend([
            {
                "candidate_id": "vod-d-pos",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "vod_id": "vod-d",
                "clip_ids": ["clip-vod-d"],
                "candidate_source": "tension_build",
            },
            {
                "candidate_id": "vod-e-pos",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "vod_id": "vod-e",
                "clip_ids": ["clip-vod-e"],
                "candidate_source": "narrative_reveal",
            },
            {
                "candidate_id": "vod-f-pos",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "vod_id": "vod-f",
                "clip_ids": ["clip-vod-f"],
                "candidate_source": "reaction",
            },
        ])

        quality = train_vod_ranker.ranker_dataset_quality(rows)

        self.assertTrue(quality["ok"])
        self.assertEqual(quality["mixed_ranking_groups"], 3)
        self.assertEqual(quality["targets"]["mixed_ranking_groups"], 20)
        self.assertEqual(quality["vod_contrast_balance"]["target_negative_rows_per_positive_max"], 8)
        self.assertEqual(quality["vod_contrast_balance"]["worst_negative_rows_per_positive"], 1.0)
        self.assertEqual(quality["vod_contrast_balance"]["over_limit_group_count"], 0)

    def test_ranker_dataset_quality_rejects_candidate_source_dominance(self) -> None:
        rows = [
            {
                "candidate_id": f"cand-r{i}",
                "label": 1 if i % 3 == 0 else 0,
                "label_class": "positive" if i % 3 == 0 else "hard_negative",
                "dataset_source": "twitch_vod_primary",
                "vod_id": f"vod-{i}",
                "clip_ids": [f"clip-{i}"] if i % 3 == 0 else [],
                "candidate_source": "reaction",
            }
            for i in range(9)
        ] + [
            {
                "candidate_id": "cand-q1",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "vod_id": "vod-q1",
                "clip_ids": ["clip-q1"],
                "candidate_source": "quote",
            },
            {
                "candidate_id": "cand-m1",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "vod_id": "vod-m1",
                "clip_ids": ["clip-m1"],
                "candidate_source": "moment",
            },
        ]

        quality = train_vod_ranker.ranker_dataset_quality(rows)

        self.assertFalse(quality["ok"])
        self.assertIn("candidate_source_peak_share_above_maximum", quality["violations"])


class TrainVodRankerWorkflowStateTests(unittest.TestCase):
    def _setup_common(self, tmp: str, *, write_artifact: bool, write_dataset: bool) -> dict:
        root = Path(tmp)
        labels_path = root / "labels.jsonl"
        labels_path.write_text(json.dumps({"broadcaster": "jynxzi"}) + "\n", encoding="utf-8")
        dataset_out = root / "dataset.jsonl"
        if write_dataset:
            dataset_out.write_text("{}\n", encoding="utf-8")
        artifact_dir = root / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if write_artifact:
            (artifact_dir / "jynxzi_ranker.json").write_text("{}", encoding="utf-8")
        state_path = artifact_dir / "workflow_state.json"
        return {
            "labels": labels_path,
            "dataset_out": dataset_out,
            "artifact_dir": artifact_dir,
            "state_path": state_path,
            "work_dir": root / "work",
        }

    def _argv(self, paths: dict, state_json: str | None) -> list[str]:
        argv = [
            "train_vod_ranker.py",
            "--labels", str(paths["labels"]),
            "--dataset-out", str(paths["dataset_out"]),
            "--profile", "jynxzi",
            "--artifact-dir", str(paths["artifact_dir"]),
            "--work-dir", str(paths["work_dir"]),
            "--workflow-state-path", str(paths["state_path"]),
        ]
        if state_json is not None:
            argv.extend(["--workflow-state-json", state_json])
        return argv

    def test_skips_when_state_matches_and_artifacts_exist(self) -> None:
        current_state = {"workflow": "jynxzi_vod_training", "code_hash": "same"}
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._setup_common(tmp, write_artifact=True, write_dataset=True)
            # Stored state already matches current, artifacts are newer than labels.
            paths["state_path"].write_text(json.dumps(current_state), encoding="utf-8")
            import os as _os
            _os.utime(paths["labels"], (100, 100))
            _os.utime(paths["dataset_out"], (300, 300))
            _os.utime(paths["artifact_dir"] / "jynxzi_ranker.json", (300, 300))

            with patch("ml.train_vod_ranker.build_composite_dataset") as build_mock, \
                 patch("ml.train_vod_ranker._train_models_for_dataset") as train_mock, \
                 patch("sys.argv", self._argv(paths, json.dumps(current_state))):
                rc = train_vod_ranker.main()

        self.assertEqual(rc, 0)
        build_mock.assert_not_called()
        train_mock.assert_not_called()

    def test_retrains_and_writes_state_when_training_succeeds(self) -> None:
        current_state = {"workflow": "jynxzi_vod_training", "code_hash": "new"}
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._setup_common(tmp, write_artifact=False, write_dataset=False)

            with patch(
                "ml.train_vod_ranker.build_composite_dataset_with_report",
                return_value=([{"candidate_id": "c1", "label": 1, "features": {}}], {"enabled": False}),
            ), \
                 patch("ml.train_vod_ranker._train_models_for_dataset", return_value=0) as train_mock, \
                 patch("sys.argv", self._argv(paths, json.dumps(current_state))):
                rc = train_vod_ranker.main()

            self.assertEqual(rc, 0)
            train_mock.assert_called_once()
            self.assertTrue(paths["state_path"].exists())
            self.assertEqual(
                json.loads(paths["state_path"].read_text(encoding="utf-8")),
                current_state,
            )

    def test_state_not_written_when_training_fails(self) -> None:
        current_state = {"workflow": "jynxzi_vod_training", "code_hash": "new"}
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._setup_common(tmp, write_artifact=False, write_dataset=False)

            with patch(
                "ml.train_vod_ranker.build_composite_dataset_with_report",
                return_value=([{"candidate_id": "c1", "label": 1, "features": {}}], {"enabled": False}),
            ), \
                 patch("ml.train_vod_ranker._train_models_for_dataset", return_value=2), \
                 patch("sys.argv", self._argv(paths, json.dumps(current_state))):
                rc = train_vod_ranker.main()

            self.assertEqual(rc, 2)
            self.assertFalse(paths["state_path"].exists())

    def test_rejects_invalid_workflow_state_json(self) -> None:
        with self.assertRaises(SystemExit):
            train_vod_ranker._load_workflow_state("not-json")

    def test_rejects_non_object_workflow_state_json(self) -> None:
        with self.assertRaises(SystemExit):
            train_vod_ranker._load_workflow_state("[1, 2, 3]")

    def test_empty_workflow_state_returns_none(self) -> None:
        self.assertIsNone(train_vod_ranker._load_workflow_state(None))
        self.assertIsNone(train_vod_ranker._load_workflow_state(""))
        self.assertIsNone(train_vod_ranker._load_workflow_state("   "))


if __name__ == "__main__":
    unittest.main()
