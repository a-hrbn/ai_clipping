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

    def test_cross_platform_fingerprint_rows_are_sent_through_vod_feature_builder(self) -> None:
        captured_labels = []
        vod_rows = [
            {
                "candidate_id": "cross-1",
                "label": 1,
                "label_class": "weak_positive",
                "dataset_source": "cross_platform_audio_fingerprint_train",
                "supervision_tier": "cross_platform_outcome",
                "features": {"hook_strength": 0.8},
            }
        ]

        def _fake_build_dataset(labels, profile, work_dir):
            captured_labels.extend(labels)
            return vod_rows

        cross_platform_rows = [
            {
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "candidate_id": "repost-1",
                "window_start": 10.0,
                "window_end": 25.0,
                "label": 1,
                "dataset_source": "cross_platform_audio_fingerprint",
                "supervision_tier": "cross_platform_outcome",
                "fingerprint_confidence": 0.91,
            }
        ]

        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", side_effect=_fake_build_dataset):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{
                    "broadcaster": "jynxzi",
                    "vod_id": "456",
                    "vod_url": "https://www.twitch.tv/videos/456",
                    "window_start": 1.0,
                    "window_end": 3.0,
                    "label_class": "hard_negative",
                }],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                cross_platform_rows=cross_platform_rows,
            )

        self.assertEqual(rows, vod_rows)
        cross_labels = [
            label for label in captured_labels
            if label.get("training_dataset_source") == "cross_platform_audio_fingerprint_train"
        ]
        self.assertEqual(len(cross_labels), 1)
        self.assertEqual(cross_labels[0]["label_class"], "cross_platform_outcome_positive")
        self.assertEqual(cross_labels[0]["label_source"], "cross_platform_audio_fingerprint")
        self.assertEqual(cross_labels[0]["training_supervision_tier"], "cross_platform_outcome")
        self.assertTrue(cross_labels[0]["cross_platform_recall_target"])

    def test_mm_judge_proxy_rows_merge_as_weak_training_rows(self) -> None:
        vod_rows = [
            {
                "candidate_id": "vod-1",
                "label": 1,
                "dataset_source": "twitch_vod_primary",
                "features": {"hook_strength": 0.8},
            }
        ]
        judge_rows = [
            {
                "candidate_id": "judge-1",
                "label": 1,
                "judge_score": 8.6,
                "features": {"hook_strength": 0.9, "payoff_score": 0.8},
                "profile": "jynxzi",
            }
        ]

        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=vod_rows):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                mm_judge_rows=judge_rows,
                mm_judge_weight=0.4,
            )

        judge = [row for row in rows if row.get("candidate_id") == "judge-1"][0]
        self.assertEqual(judge["dataset_source"], "mm_judge_proxy_train")
        self.assertEqual(judge["supervision_tier"], "mm_judge_proxy")
        self.assertEqual(judge["label_class"], "mm_judge_positive")
        self.assertAlmostEqual(judge["sample_weight"], 0.4)
        self.assertTrue(judge["feature_provenance"]["mm_judge_proxy"])

    def test_mm_judge_snapshots_replay_judge_features_into_matching_training_rows(self) -> None:
        vod_rows = [
            {
                "candidate_id": "candidate-a",
                "label": 1,
                "dataset_source": "cross_platform_audio_fingerprint_train",
                "features": {"hook_strength": 0.8},
            }
        ]
        judge_rows = [
            {
                "candidate_id": "candidate-a",
                "label": 1,
                "judge_score": 8.6,
                "judge_confidence": 0.72,
                "judge_llm_used": True,
                "features": {
                    "hook_strength": 0.9,
                    "judge_score": 8.6,
                    "judge_confidence": 0.72,
                    "judge_scroll_stop_power": 0.91,
                    "judge_payoff_strength": 0.83,
                    "has_judge_score": True,
                    "judge_llm_used": True,
                },
                "profile": "jynxzi",
            }
        ]

        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=vod_rows):
            rows, report = train_vod_ranker.build_composite_dataset_with_report(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                mm_judge_rows=judge_rows,
                mm_judge_weight=0.4,
            )

        vod_row = [row for row in rows if row.get("dataset_source") == "cross_platform_audio_fingerprint_train"][0]
        self.assertEqual(vod_row["features"]["judge_score"], 8.6)
        self.assertEqual(vod_row["features"]["judge_confidence"], 0.72)
        self.assertEqual(vod_row["features"]["judge_scroll_stop_power"], 0.91)
        self.assertTrue(vod_row["features"]["has_judge_score"])
        self.assertTrue(vod_row["features"]["judge_llm_used"])
        self.assertTrue(vod_row["feature_provenance"]["mm_judge_feature_replay"])
        self.assertEqual(report["mm_judge_feature_replay"]["applied_rows"], 2)

    def test_main_merges_targeted_backfill_labels_before_vod_feature_builder(self) -> None:
        dataset_rows = [
            {
                "candidate_id": "backfilled-1",
                "label": 1,
                "dataset_source": "twitch_vod_primary",
                "features": {},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_path = root / "labels.jsonl"
            backfill_path = root / "targeted_backfill.jsonl"
            dataset_out = root / "dataset.jsonl"
            artifact_dir = root / "artifacts"
            labels_path.write_text(
                json.dumps({
                    "broadcaster": "jynxzi",
                    "vod_id": "111",
                    "vod_url": "https://www.twitch.tv/videos/111",
                    "window_start": 1.0,
                    "window_end": 10.0,
                    "label_class": "negative",
                }) + "\n",
                encoding="utf-8",
            )
            backfill_path.write_text(
                json.dumps({
                    "candidate_id": "queued-1",
                    "vod_id": "222",
                    "window_start": 20.0,
                    "window_end": 46.0,
                    "label_class": "positive",
                    "view_count": 9000,
                }) + "\n",
                encoding="utf-8",
            )

            with patch(
                "ml.train_vod_ranker.build_composite_dataset_with_report",
                return_value=(dataset_rows, {"enabled": False, "filtered_rows": 0}),
            ) as build_mock, \
                 patch("ml.train_vod_ranker._train_models_for_dataset", return_value=0), \
                 patch(
                     "sys.argv",
                     [
                         "train_vod_ranker.py",
                         "--labels", str(labels_path),
                         "--targeted-backfill-labels", str(backfill_path),
                         "--dataset-out", str(dataset_out),
                         "--profile", "jynxzi",
                         "--artifact-dir", str(artifact_dir),
                         "--work-dir", str(root / "work"),
                     ],
                 ):
                rc = train_vod_ranker.main()

        self.assertEqual(rc, 0)
        merged_labels = build_mock.call_args.args[0]
        self.assertEqual(len(merged_labels), 2)
        queued = [row for row in merged_labels if row.get("candidate_id") == "queued-1"][0]
        self.assertEqual(queued["vod_url"], "https://www.twitch.tv/videos/222")
        self.assertEqual(queued["manual_gold_review_source"], "targeted_backfill")
        self.assertEqual(queued["gold_candidate_status"], "needs_feature_build")
        self.assertTrue(queued["targeted_feature_backfill"])

    def test_dataset_summary_reports_feature_signal_preflight(self) -> None:
        summary = train_vod_ranker.dataset_summary([
            {
                "candidate_id": "row-1",
                "label": 1,
                "label_class": "positive",
                "dataset_source": "twitch_vod_primary",
                "features": {
                    "chat_ema_crossing_score": 0.7,
                    "chat_rate_derivative_score": 0.5,
                    "streamer_shout_score": 0.6,
                    "f0_sigma_score": 0.4,
                    "vad_voice_ratio": 0.9,
                    "shout_onset_score": 0.3,
                    "boundary_variant_score": 0.8,
                    "boundary_variant_margin_score": 0.2,
                    "boundary_variant_count": 4,
                    "hook_rewrite_score": 0.8,
                    "hook_curiosity_gap_score": 0.7,
                    "has_hook_rewrite": True,
                    "vl_prompt_max_score": 0.6,
                    "vl_clutch_score": 0.5,
                    "game_event_score": 0.4,
                    "is_clutch_window": True,
                    "kill_density_in_window": 0.2,
                },
            }
        ])

        preflight = summary["feature_signal_preflight"]
        self.assertEqual(preflight["nonzero_counts"]["chat_ema_crossing_score"], 1)
        self.assertEqual(preflight["nonzero_counts"]["boundary_variant_score"], 1)
        self.assertEqual(preflight["nonzero_counts"]["hook_rewrite_score"], 1)
        self.assertEqual(preflight["nonzero_counts"]["vl_prompt_max_score"], 1)
        self.assertEqual(preflight["nonzero_counts"]["game_event_score"], 1)
        self.assertIn("vl_scoreboard_score", preflight["dark_features"])

    def test_cross_platform_input_summary_reports_empty_file_state(self) -> None:
        summary = train_vod_ranker._cross_platform_input_summary(
            Path("/tmp/missing-cross-platform.jsonl"),
            [],
        )

        self.assertFalse(summary["exists"])
        self.assertEqual(summary["rows"], 0)

    def test_cross_platform_input_summary_includes_prepare_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_path = root / "jynxzi_cross_platform_fingerprint_rows.jsonl"
            summary_path = root / "jynxzi_cross_platform_fingerprint_summary.json"
            rows_path.write_text("", encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "missing_inputs",
                        "vod_rows": 0,
                        "repost_rows": 0,
                        "matched_rows": 0,
                        "diagnostic_rows": 12,
                        "missing_inputs": ["repost_manifest"],
                        "match_diagnostics": {
                            "confidence_distribution": {"count": 12, "median": 0.43},
                            "threshold_ladder": {"0.50": 2, "0.72": 0},
                            "recommendation": {"status": "chromaprint_signal_weak"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = train_vod_ranker._cross_platform_input_summary(rows_path, [])

        self.assertEqual(summary["prepare_status"], "missing_inputs")
        self.assertEqual(summary["prepare_missing_inputs"], ["repost_manifest"])
        self.assertEqual(summary["prepare_diagnostic_rows"], 12)
        self.assertEqual(summary["prepare_confidence_distribution"]["median"], 0.43)
        self.assertEqual(summary["prepare_threshold_ladder"]["0.72"], 0)
        self.assertEqual(summary["prepare_match_recommendation"]["status"], "chromaprint_signal_weak")

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

    def test_build_composite_dataset_weights_bad_rollout_reviews_as_hard_negatives(self) -> None:
        gold_rows = [
            {
                "queue_id": "bad-auto",
                "candidate_id": "bad-auto",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 0,
                "gold_grade": 0,
                "dataset_source": "gold_eval_rollout_review",
                "supervision_reason": "manual_rollout_auto_accept_review",
                "rollout_bucket": "auto_accept_candidate",
                "gold_holdout_split": "train",
                "sample_weight": 1.0,
                "features": {"hook_strength": 0.1},
            },
            {
                "queue_id": "good-auto",
                "candidate_id": "good-auto",
                "job_id": "job-2",
                "vod_id": "vod-2",
                "label": 1,
                "gold_grade": 2,
                "dataset_source": "gold_eval_rollout_review",
                "supervision_reason": "manual_rollout_auto_accept_review",
                "rollout_bucket": "auto_accept_candidate",
                "gold_holdout_split": "train",
                "sample_weight": 1.0,
                "features": {"hook_strength": 0.8},
            },
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=[]):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                gold_train_rows=gold_rows,
                gold_train_weight=2.0,
                rollout_review_negative_weight=4.0,
            )

        rows_by_queue = {row["queue_id"]: row for row in rows}
        self.assertEqual(rows_by_queue["bad-auto"]["dataset_source"], "gold_eval_rollout_review_hard_negative_train")
        self.assertEqual(rows_by_queue["bad-auto"]["label_class"], "gold_rollout_review_hard_negative")
        self.assertEqual(rows_by_queue["bad-auto"]["supervision_reason"], "manual_rollout_auto_accept_false_positive")
        self.assertEqual(rows_by_queue["bad-auto"]["sample_weight"], 4.0)
        self.assertEqual(rows_by_queue["good-auto"]["dataset_source"], "gold_eval_train")
        self.assertEqual(rows_by_queue["good-auto"]["label_class"], "gold_positive")
        self.assertEqual(rows_by_queue["good-auto"]["sample_weight"], 2.0)

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

    def test_top_viewed_auto_holdout_blocks_window_not_entire_vod(self) -> None:
        captured_labels = []
        labels = [
            {
                "vod_id": "vod-top",
                "vod_url": "https://www.twitch.tv/videos/222",
                "label_class": "positive",
                "window_start": 40.0,
                "window_end": 50.0,
            },
            {
                "vod_id": "vod-clean",
                "vod_url": "https://www.twitch.tv/videos/111",
                "label_class": "positive",
                "window_start": 5.0,
                "window_end": 15.0,
            },
        ]
        gold_rows = [
            {
                "queue_id": "top-viewed-holdout",
                "candidate_id": "holdout-cand",
                "vod_id": "vod-top",
                "label": 1,
                "gold_holdout_policy": "top_viewed_auto",
                "gold_holdout_split": "holdout",
                "window_start": 10.0,
                "window_end": 25.0,
                "features": {"hook_strength": 1.0},
            },
            {
                "queue_id": "top-viewed-train-same-vod",
                "candidate_id": "train-same-vod",
                "vod_id": "vod-top",
                "label": 1,
                "gold_holdout_policy": "top_viewed_auto",
                "gold_holdout_split": "train",
                "window_start": 60.0,
                "window_end": 75.0,
                "features": {"hook_strength": 0.9},
            },
        ]

        def _fake_build_dataset(input_labels, _profile, _work_dir):
            captured_labels.extend(input_labels)
            return [
                {
                    "queue_id": "vod-overlap",
                    "candidate_id": "vod-overlap",
                    "vod_id": "vod-top",
                    "label": 1,
                    "dataset_source": "twitch_vod_primary",
                    "window_start": 12.0,
                    "window_end": 20.0,
                    "features": {},
                },
                {
                    "queue_id": "vod-same-vod-clean-window",
                    "candidate_id": "vod-same-vod-clean-window",
                    "vod_id": "vod-top",
                    "label": 1,
                    "dataset_source": "twitch_vod_primary",
                    "window_start": 40.0,
                    "window_end": 50.0,
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

        self.assertEqual({row["vod_id"] for row in captured_labels}, {"vod-top", "vod-clean"})
        rows_by_queue = {row["queue_id"]: row for row in rows}
        self.assertNotIn("top-viewed-holdout", rows_by_queue)
        self.assertNotIn("vod-overlap", rows_by_queue)
        self.assertIn("vod-same-vod-clean-window", rows_by_queue)
        self.assertIn("top-viewed-train-same-vod", rows_by_queue)
        self.assertEqual(report["holdout_vod_count"], 0)
        self.assertEqual(report["window_only_holdout_rows"], 1)
        self.assertEqual(report["filtered_by_reason"]["holdout_window_overlap"], 1)

    def test_build_composite_dataset_logs_vod_usage_overview_by_date(self) -> None:
        labels = [
            {
                "vod_id": "vod-full-block",
                "vod_url": "https://www.twitch.tv/videos/100",
                "label_class": "positive",
                "created_at": "2026-03-01T10:00:00Z",
                "window_start": 10.0,
                "window_end": 20.0,
            },
            {
                "vod_id": "vod-partial",
                "vod_url": "https://www.twitch.tv/videos/200",
                "label_class": "positive",
                "created_at": "2026-03-02T10:00:00Z",
                "window_start": 40.0,
                "window_end": 55.0,
            },
            {
                "vod_id": "vod-used",
                "vod_url": "https://www.twitch.tv/videos/300",
                "label_class": "positive",
                "created_at": "2026-03-03T10:00:00Z",
                "window_start": 70.0,
                "window_end": 85.0,
            },
        ]
        gold_rows = [
            {
                "queue_id": "strict-holdout",
                "candidate_id": "strict-holdout",
                "vod_id": "vod-full-block",
                "label": 1,
                "gold_holdout_split": "holdout",
                "window_start": 10.0,
                "window_end": 20.0,
                "features": {},
            },
            {
                "queue_id": "window-holdout",
                "candidate_id": "window-holdout",
                "vod_id": "vod-partial",
                "label": 1,
                "gold_holdout_policy": "top_viewed_auto",
                "gold_holdout_split": "holdout",
                "window_start": 10.0,
                "window_end": 20.0,
                "features": {},
            },
        ]

        def _fake_build_dataset(input_labels, _profile, _work_dir):
            return [
                {
                    "queue_id": f"row-{row['vod_id']}",
                    "candidate_id": f"row-{row['vod_id']}",
                    "vod_id": row["vod_id"],
                    "label": 1,
                    "dataset_source": "twitch_vod_primary",
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "features": {},
                }
                for row in input_labels
            ]

        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", side_effect=_fake_build_dataset), \
             self.assertLogs(train_vod_ranker.logger, level="INFO") as logs:
            train_vod_ranker.build_composite_dataset_with_report(
                labels=labels,
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                gold_train_rows=gold_rows,
                gold_train_weight=2.0,
            )

        joined = "\n".join(logs.output)
        self.assertIn("training_vod_usage_overview_summary current_label_vods=3 complete_blocked=1 partially_blocked=1 fully_used=1", joined)
        self.assertIn("training_vod_usage_overview_complete_blocked vod_date=2026-03-01", joined)
        self.assertIn("training_vod_usage_overview_partially_blocked vod_date=2026-03-02", joined)
        self.assertIn("training_vod_usage_overview_fully_used vod_date=2026-03-03", joined)
        self.assertNotIn("vod-full-block", joined)
        self.assertNotIn("https://www.twitch.tv/videos/100", joined)

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

    def test_build_composite_dataset_keeps_promoted_holdout_hard_negatives(self) -> None:
        gold_rows = [
            {
                "queue_id": "holdout-positive",
                "candidate_id": "cand-holdout",
                "job_id": "job-holdout",
                "vod_id": "vod-holdout",
                "label": 1,
                "label_class": "gold_positive",
                "supervision_tier": "gold_eval",
                "label_source": "gold_eval",
                "gold_holdout_split": "holdout",
                "features": {"hook_strength": 1.0},
            }
        ]
        hard_negative_rows = [
            {
                "queue_id": "hard-negative",
                "candidate_id": "cand-hard",
                "job_id": "job-holdout",
                "vod_id": "vod-holdout",
                "label": 0,
                "gold_holdout_split": "train",
                "hard_negative_original_split": "holdout",
                "hard_negative_source": "ranker_holdout_diagnosis",
                "sample_weight": 1.0,
                "features": {"hook_strength": 0.1},
            }
        ]

        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=[]):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                gold_train_rows=gold_rows,
                hard_negative_rows=hard_negative_rows,
                hard_negative_weight=4.0,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["queue_id"], "hard-negative")
        self.assertEqual(rows[0]["dataset_source"], "gold_eval_hard_negative_train")
        self.assertEqual(rows[0]["sample_weight"], 4.0)

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

    def test_build_composite_dataset_merges_auto_outcome_train_negatives_only(self) -> None:
        auto_rows = [
            {
                "queue_id": "auto-hard",
                "candidate_id": "auto-hard",
                "vod_id": "vod-a",
                "label": 0,
                "gold_holdout_split": "train",
                "auto_outcome_negative_tier": "hard_negative",
                "sample_weight": 1.0,
                "features": {"hook_strength": 0.1},
            },
            {
                "queue_id": "auto-background",
                "candidate_id": "auto-background",
                "vod_id": "vod-b",
                "label": 0,
                "gold_holdout_split": "train",
                "auto_outcome_negative_tier": "background_negative",
                "sample_weight": 1.0,
                "features": {"hook_strength": 0.05},
            },
            {
                "queue_id": "auto-holdout",
                "candidate_id": "auto-holdout",
                "vod_id": "vod-c",
                "label": 0,
                "gold_holdout_split": "holdout",
                "auto_outcome_negative_tier": "hard_negative",
                "features": {"hook_strength": 0.2},
            },
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=[]):
            rows, report = train_vod_ranker.build_composite_dataset_with_report(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                auto_outcome_negative_rows=auto_rows,
                auto_outcome_hard_negative_weight=1.0,
                auto_outcome_background_negative_weight=0.25,
            )

        rows_by_queue = {row["queue_id"]: row for row in rows}
        self.assertEqual(set(rows_by_queue), {"auto-hard", "auto-background"})
        self.assertEqual(rows_by_queue["auto-hard"]["dataset_source"], "auto_outcome_negative_train")
        self.assertEqual(rows_by_queue["auto-hard"]["label_source"], "auto_outcome_negative")
        self.assertEqual(rows_by_queue["auto-hard"]["supervision_tier"], "auto_outcome_negative")
        self.assertEqual(rows_by_queue["auto-hard"]["label_class"], "auto_outcome_hard_negative")
        self.assertEqual(rows_by_queue["auto-hard"]["sample_weight"], 1.0)
        self.assertEqual(rows_by_queue["auto-background"]["label_class"], "auto_outcome_background_negative")
        self.assertEqual(rows_by_queue["auto-background"]["sample_weight"], 0.25)
        self.assertNotIn("auto-holdout", rows_by_queue)
        self.assertEqual(report["filtered_by_reason"].get("gold_holdout_row", 0), 0)

    def test_build_composite_dataset_merges_cross_platform_outcome_train_positives_without_gold(self) -> None:
        outcome_rows = [
            {
                "queue_id": "outcome-positive",
                "candidate_id": "outcome-positive",
                "vod_id": "vod-a",
                "label": 1,
                "gold_holdout_split": "train",
                "dataset_source": "cross_platform_outcome_train",
                "label_source": "cross_platform_audio_fingerprint",
                "supervision_tier": "cross_platform_outcome",
                "features": {"hook_strength": 0.9},
            },
            {
                "queue_id": "outcome-holdout",
                "candidate_id": "outcome-holdout",
                "vod_id": "vod-b",
                "label": 1,
                "gold_holdout_split": "holdout",
                "dataset_source": "cross_platform_outcome_holdout",
                "supervision_tier": "outcome_holdout",
                "features": {"hook_strength": 0.8},
            },
            {
                "queue_id": "top-viewed-proxy",
                "candidate_id": "top-viewed-proxy",
                "vod_id": "vod-c",
                "label": 1,
                "gold_holdout_split": "train",
                "dataset_source": "outcome_top_viewed_train",
                "supervision_tier": "outcome_train",
                "features": {"hook_strength": 0.7},
            },
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=[]):
            rows, report = train_vod_ranker.build_composite_dataset_with_report(
                labels=[{"broadcaster": "jynxzi"}],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                outcome_positive_rows=outcome_rows,
                outcome_positive_weight=0.55,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["queue_id"], "outcome-positive")
        self.assertEqual(rows[0]["dataset_source"], "cross_platform_outcome_train")
        self.assertEqual(rows[0]["label_source"], "cross_platform_audio_fingerprint")
        self.assertEqual(rows[0]["supervision_tier"], "cross_platform_outcome")
        self.assertEqual(rows[0]["label_class"], "cross_platform_outcome_positive")
        self.assertNotIn("top_viewed_weak_positive", rows[0])
        self.assertEqual(rows[0]["sample_weight"], 0.55)
        self.assertEqual(report["filtered_by_reason"].get("gold_holdout_row", 0), 0)

    def test_build_composite_dataset_weights_cross_platform_positives_by_views(self) -> None:
        outcome_rows = [
            {
                "queue_id": "viral-positive",
                "candidate_id": "viral-positive",
                "vod_id": "vod-a",
                "label": 1,
                "gold_holdout_split": "train",
                "dataset_source": "cross_platform_outcome_train",
                "label_source": "cross_platform_transcript_match",
                "supervision_tier": "cross_platform_outcome",
                "views": 600000,
                "features": {"hook_strength": 0.9},
            },
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=[]):
            rows, _report = train_vod_ranker.build_composite_dataset_with_report(
                labels=[],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                outcome_positive_rows=outcome_rows,
                outcome_positive_weight=0.55,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome_view_tier"], "breakout_viral")
        self.assertEqual(rows[0]["label_quality_class"], "cross_platform_viral_positive")
        self.assertEqual(rows[0]["rank_target"], 5)
        self.assertAlmostEqual(rows[0]["sample_weight"], 1.375)

    def test_build_composite_dataset_merges_serving_hard_negatives(self) -> None:
        hard_rows = [
            {
                "queue_id": "serving-hard",
                "candidate_id": "serving-hard",
                "job_id": "job-1",
                "vod_id": "vod-1",
                "label": 0,
                "gold_holdout_split": "train",
                "sample_weight": 1.25,
                "features": {"hook_strength": 0.4, "source_prior": 0.8},
            }
        ]
        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", return_value=[]):
            rows = train_vod_ranker.build_composite_dataset(
                labels=[],
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                serving_hard_negative_rows=hard_rows,
                serving_hard_negative_weight=2.0,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_source"], "serving_trace_hard_negative_train")
        self.assertEqual(rows[0]["label_source"], "serving_trace")
        self.assertEqual(rows[0]["supervision_tier"], "serving_hard_negative")
        self.assertEqual(rows[0]["label_class"], "serving_trace_hard_negative")
        self.assertEqual(rows[0]["sample_weight"], 2.5)
        self.assertTrue(rows[0]["feature_provenance"]["serving_trace_hard_negative"])

    def test_outcome_only_policy_ignores_twitch_labels_and_uses_cross_platform_labels(self) -> None:
        labels = [
            {"queue_id": "twitch-positive", "candidate_id": "twitch-positive", "label": 1, "broadcaster": "jynxzi"}
        ]
        cross_platform_rows = [
            {
                "queue_id": "cross-positive",
                "candidate_id": "cross-positive",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 25.0,
                "label": 1,
                "dataset_source": "cross_platform_audio_fingerprint",
                "supervision_tier": "cross_platform_outcome",
            }
        ]

        captured_labels = []

        def _fake_build_dataset(training_labels, profile, work_dir):
            captured_labels.extend(training_labels)
            return []

        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", side_effect=_fake_build_dataset):
            rows, report = train_vod_ranker.build_composite_dataset_with_report(
                labels=labels,
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                cross_platform_rows=cross_platform_rows,
                primary_label_policy="outcome_only",
            )

        self.assertEqual(rows, [])
        self.assertEqual(report["twitch_label_rows_ignored_by_policy"], 1)
        self.assertEqual(len(captured_labels), 1)
        self.assertEqual(captured_labels[0]["training_dataset_source"], "cross_platform_audio_fingerprint_train")

    def test_outcome_only_targeted_backfill_builds_features_without_training_rows(self) -> None:
        labels = [
            {
                "candidate_id": "twitch-positive",
                "label": 1,
                "broadcaster": "jynxzi",
                "vod_id": "123",
                "vod_url": "https://www.twitch.tv/videos/123",
                "window_start": 10.0,
                "window_end": 25.0,
            },
            {
                "candidate_id": "targeted-backfill",
                "label": 1,
                "broadcaster": "jynxzi",
                "vod_id": "456",
                "vod_url": "https://www.twitch.tv/videos/456",
                "window_start": 30.0,
                "window_end": 45.0,
                "targeted_feature_backfill": True,
            },
        ]
        captured_labels = []

        def _fake_build_dataset(training_labels, profile, work_dir):
            captured_labels.extend(training_labels)
            return [
                {
                    "candidate_id": "targeted-row",
                    "vod_id": "456",
                    "label": 1,
                    "dataset_source": "targeted_feature_backfill",
                    "targeted_feature_backfill": True,
                    "features": {"hook_strength": 0.5},
                }
            ]

        with patch("ml.train_vod_ranker.build_twitch_training_dataset.build_dataset", side_effect=_fake_build_dataset):
            rows, report = train_vod_ranker.build_composite_dataset_with_report(
                labels=labels,
                profile="jynxzi",
                work_dir=Path("/tmp/ignored"),
                primary_label_policy="outcome_only",
            )

        self.assertEqual(rows, [])
        self.assertEqual(len(captured_labels), 1)
        self.assertEqual(captured_labels[0]["candidate_id"], "targeted-backfill")
        self.assertEqual(report["targeted_feature_backfill_label_rows"], 1)
        self.assertEqual(report["targeted_feature_backfill_rows_built"], 1)

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
            holdout_policy_rows,
            rollout_review_negative_weight,
            hard_negative_rows,
            hard_negative_weight,
            serving_hard_negative_rows,
            serving_hard_negative_weight,
            outcome_positive_rows,
            outcome_positive_weight,
            auto_outcome_negative_rows,
            auto_outcome_hard_negative_weight,
            auto_outcome_background_negative_weight,
            cross_platform_rows=None,
            cross_platform_weight=1.0,
            primary_label_policy="all",
            mm_judge_rows=None,
            mm_judge_weight=0.4,
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
            "--include-cross-platform-outcome-train", str(Path(paths["artifact_dir"]) / "missing_cross_platform.jsonl"),
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
