import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import training_workflow_state


class TrainingWorkflowStateTests(unittest.TestCase):
    def test_should_retrain_when_state_mismatch_even_if_artifacts_are_newer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            dataset = root / "dataset.jsonl"
            artifact = root / "artifact.json"
            state = root / "state.json"
            for path in (labels, dataset, artifact):
                path.write_text("x", encoding="utf-8")
            os.utime(labels, (100, 100))
            os.utime(dataset, (200, 200))
            os.utime(artifact, (300, 300))
            training_workflow_state.write_state(state, {"code_hash": "old"})

            self.assertTrue(
                training_workflow_state.should_retrain(
                    labels_path=labels,
                    dataset_path=dataset,
                    artifact_path=artifact,
                    state_path=state,
                    current_state={"code_hash": "new"},
                )
            )

    def test_should_reevaluate_when_state_mismatch_even_if_outputs_are_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            artifact = root / "artifact.json"
            outputs = [root / "eval.json", root / "ranker.json"]
            state = root / "state.json"
            for path in (dataset, artifact, *outputs):
                path.write_text("x", encoding="utf-8")
                os.utime(path, (300, 300))
            training_workflow_state.write_state(state, {"code_hash": "old"})

            self.assertTrue(
                training_workflow_state.should_reevaluate(
                    dataset_path=dataset,
                    artifact_path=artifact,
                    output_paths=outputs,
                    state_path=state,
                    current_state={"code_hash": "new"},
                )
            )

    def test_should_reevaluate_when_dependency_is_newer_than_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            artifact = root / "artifact.json"
            dependency = root / "gold_eval_updates.jsonl"
            outputs = [root / "eval.json", root / "ranker.json"]
            state = root / "state.json"
            current_state = {"code_hash": "same"}
            for path in (dataset, artifact, dependency, *outputs):
                path.write_text("x", encoding="utf-8")
            os.utime(dataset, (100, 100))
            os.utime(artifact, (100, 100))
            os.utime(outputs[0], (200, 200))
            os.utime(outputs[1], (200, 200))
            os.utime(dependency, (300, 300))
            training_workflow_state.write_state(state, current_state)

            self.assertTrue(
                training_workflow_state.should_reevaluate(
                    dataset_path=dataset,
                    artifact_path=artifact,
                    output_paths=outputs,
                    dependency_paths=[dependency],
                    state_path=state,
                    current_state=current_state,
                )
            )

    def test_should_retrain_uses_label_mtime_when_state_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            dataset = root / "dataset.jsonl"
            artifact = root / "artifact.json"
            state = root / "state.json"
            current_state = {"code_hash": "same"}
            for path in (labels, dataset, artifact):
                path.write_text("x", encoding="utf-8")
            os.utime(labels, (500, 500))
            os.utime(dataset, (200, 200))
            os.utime(artifact, (300, 300))
            training_workflow_state.write_state(state, current_state)

            self.assertTrue(
                training_workflow_state.should_retrain(
                    labels_path=labels,
                    dataset_path=dataset,
                    artifact_path=artifact,
                    state_path=state,
                    current_state=current_state,
                )
            )

    def test_should_retrain_uses_dependency_mtime_when_state_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            gold_rows = root / "gold_rows.jsonl"
            dataset = root / "dataset.jsonl"
            artifact = root / "artifact.json"
            state = root / "state.json"
            current_state = {"code_hash": "same"}
            for path in (labels, gold_rows, dataset, artifact):
                path.write_text("x", encoding="utf-8")
            os.utime(labels, (100, 100))
            os.utime(dataset, (200, 200))
            os.utime(artifact, (300, 300))
            os.utime(gold_rows, (500, 500))
            training_workflow_state.write_state(state, current_state)

            self.assertTrue(
                training_workflow_state.should_retrain(
                    labels_path=labels,
                    dataset_path=dataset,
                    artifact_path=artifact,
                    state_path=state,
                    current_state=current_state,
                    dependency_paths=[gold_rows],
                )
            )

    def test_build_jynxzi_workflow_state_includes_runtime_knobs(self) -> None:
        with patch.object(training_workflow_state.settings, "whisper_model", "tiny"), \
             patch.object(training_workflow_state.settings, "storage_dir", "/tmp/storage"), \
             patch.object(training_workflow_state.settings, "enable_clip_judge", True), \
             patch.object(training_workflow_state.settings, "clip_judge_fail_fast", True), \
             patch.object(training_workflow_state.settings, "judge_provider_fallback_enabled", False), \
             patch.object(training_workflow_state.settings, "enable_multimodal_embeddings", True), \
             patch.object(training_workflow_state.feature_builder, "FEATURE_BUILD_VERSION", 11), \
             patch("ml.training_workflow_state._code_hash", return_value="abc123"), \
             patch.dict(
                 os.environ,
                 {
                     "TRAINING_DATASET_LLM_PROVIDER": "groq",
                     "VOD_RANKER_SHORT_FORM_MODE": "supplemental",
                     "VOD_RANKER_GOLD_TRAIN_WEIGHT": "2.5",
                     "VOD_RANKER_HARD_NEGATIVE_WEIGHT": "3.5",
                     "VOD_RANKER_USE_USER_LABELS": "1",
                     "VOD_RANKER_USE_MM_JUDGE_PROXY": "1",
                     "VOD_RANKER_MM_JUDGE_WEIGHT": "0.6",
                     "RANKER_MAX_FEATURES": "32",
                     "RANKER_PREFER_LIGHTGBM": "1",
                     "RANKER_LIGHTGBM_SELECTION_TOLERANCE": "0.05",
                     "VOD_RANKER_ROLLOUT_REVIEW_NEGATIVE_WEIGHT": "4.5",
                     "VOD_RANKER_ROLLOUT_THRESHOLD_MIN_RECALL": "0.9",
                     "VOD_RANKER_ROLLOUT_THRESHOLD_MAX_SELECTED_RATE": "0.7",
                     "VOD_RANKER_ROLLOUT_THRESHOLD_TARGET_PRECISION": "0.88",
                     "VOD_RANKER_PRIMARY_LABEL_POLICY": "outcome_only",
                     "VOD_RANKER_REQUIRE_CROSS_PLATFORM_OUTCOME": "1",
                     "VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS": "320",
                     "OUTCOME_HOLDOUT_TARGET_POSITIVE_ROWS": "350",
                     "ROLLOUT_READINESS_TARGET_NEW_VOD_GROUPS": "25",
                     "ROLLOUT_READINESS_TARGET_AVAILABLE_GROUPS": "45",
                     "ROLLOUT_READINESS_MAX_CANDIDATES_PER_VOD": "5",
                     "CROSS_PLATFORM_FINGERPRINT_ENABLED": "1",
                     "CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE": "0.80",
                     "CROSS_PLATFORM_VOD_MANIFEST_PATH": "/tmp/cross_vods.jsonl",
                     "CROSS_PLATFORM_REPOST_MANIFEST_PATH": "/tmp/cross_reposts.jsonl",
                     "CROSS_PLATFORM_FINGERPRINT_ROWS_PATH": "/tmp/cross_rows.jsonl",
                     "CROSS_PLATFORM_FINGERPRINT_DIAGNOSTICS_PATH": "/tmp/cross_diagnostics.jsonl",
                     "CROSS_PLATFORM_SOURCE_MANIFEST_PATH": "/tmp/cross_sources.csv",
                     "CROSS_PLATFORM_URL_MANIFEST_PATH": "/tmp/cross_urls.txt",
                     "CROSS_PLATFORM_MAX_REPOSTS": "40",
                     "CROSS_PLATFORM_MIN_REPOSTS": "30",
                     "CROSS_PLATFORM_MIN_MATCHED_ROWS": "12",
                     "CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS": "7",
                     "CROSS_PLATFORM_MIN_MATCHED_VODS": "5",
                     "CROSS_PLATFORM_REFRESH_VODS": "1",
                     "CROSS_PLATFORM_SECOND_PASS_AFTER_VOD_BUILD": "0",
                     "CROSS_PLATFORM_REFRESH_REPOSTS": "1",
                     "CROSS_PLATFORM_REUSE_EXISTING_ROWS": "0",
                     "CROSS_PLATFORM_HOLDOUT_RATIO": "0.25",
                     "CROSS_PLATFORM_DISCOVER_REPOSTS": "0",
                     "CROSS_PLATFORM_DOWNLOAD_REPOSTS": "1",
                     "CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS": "1",
                     "CROSS_PLATFORM_FULL_VOD_AUDIO_MAX_VODS": "9",
                     "CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS": "3",
                     "CROSS_PLATFORM_KEEP_FULL_VOD_AUDIO": "1",
                     "CROSS_PLATFORM_COARSE_VOD_TILING": "1",
                     "CROSS_PLATFORM_COARSE_VOD_TILE_SEC": "25",
                     "CROSS_PLATFORM_COARSE_VOD_STRIDE_SEC": "300",
                     "CROSS_PLATFORM_COARSE_VOD_MAX_TILES_PER_VOD": "12",
                     "CROSS_PLATFORM_COARSE_VOD_MAX_VODS": "8",
                     "CROSS_PLATFORM_COARSE_VOD_HORIZON_PADDING_SEC": "600",
                },
                 clear=False,
             ):
            state = training_workflow_state.build_jynxzi_workflow_state(
                date_from="2026-04-01",
                date_to="2026-04-10",
                skip_covered=True,
                merge_output=False,
            )

        self.assertEqual(
            state,
            {
                "workflow": "jynxzi_vod_training",
                "date_from": "2026-04-01",
                "date_to": "2026-04-10",
                "skip_covered": True,
                "merge_output": False,
                "whisper_model": "tiny",
                "dataset_llm_provider": "groq",
                "enable_clip_judge": "true",
                "clip_judge_fail_fast": "true",
                "judge_provider_fallback_enabled": "false",
                "ranker_short_form_mode": "supplemental",
                "ranker_gold_train_weight": "2.5",
                "ranker_use_gold_train": "1",
                "ranker_hard_negative_weight": "3.5",
                "ranker_use_gold_hard_negatives": "0",
                "ranker_use_user_labels": "1",
                "ranker_rollout_review_negative_weight": "4.5",
                "ranker_outcome_positive_weight": "2.0",
                "ranker_use_mm_judge_proxy": "1",
                "ranker_mm_judge_weight": "0.6",
                "ranker_max_features": "32",
                "ranker_prefer_lightgbm": "1",
                "ranker_lightgbm_selection_tolerance": "0.05",
                "ranker_rollout_threshold_min_recall": "0.9",
                "ranker_rollout_threshold_max_selected_rate": "0.7",
                "ranker_rollout_threshold_target_precision": "0.88",
                "ranker_primary_label_policy": "outcome_only",
                "ranker_require_cross_platform_outcome": "1",
                "ranker_min_cross_platform_train_rows": "320",
                "outcome_holdout_target_positive_rows": "350",
                "rollout_readiness_target_new_vod_groups": "25",
                "rollout_readiness_target_available_groups": "45",
                "rollout_readiness_max_candidates_per_vod": "5",
                "targeted_backfill_require_all_vods": "1",
                "cross_platform_fingerprint_enabled": "1",
                "cross_platform_fingerprint_min_confidence": "0.80",
                "cross_platform_vod_manifest_path": "/tmp/cross_vods.jsonl",
                "cross_platform_repost_manifest_path": "/tmp/cross_reposts.jsonl",
                "cross_platform_rows_path": "/tmp/cross_rows.jsonl",
                "cross_platform_diagnostics_path": "/tmp/cross_diagnostics.jsonl",
                "cross_platform_source_manifest_path": "/tmp/cross_sources.csv",
                "cross_platform_url_manifest_path": "/tmp/cross_urls.txt",
                "cross_platform_max_reposts": "40",
                "cross_platform_min_reposts": "30",
                "cross_platform_min_matched_rows": "12",
                "cross_platform_seed_min_matched_rows": "7",
                "cross_platform_min_matched_vods": "5",
                "cross_platform_refresh_vods": "1",
                "cross_platform_second_pass_after_vod_build": "0",
                "cross_platform_refresh_reposts": "1",
                "cross_platform_reuse_existing_rows": "0",
                "cross_platform_holdout_ratio": "0.25",
                "cross_platform_discover_reposts": "0",
                "cross_platform_download_reposts": "1",
                "cross_platform_full_vod_audio_fingerprints": "1",
                "cross_platform_full_vod_audio_max_vods": "9",
                "cross_platform_full_vod_audio_workers": "3",
                "cross_platform_keep_full_vod_audio": "1",
                "cross_platform_coarse_vod_tiling": "1",
                "cross_platform_coarse_vod_tile_sec": "25",
                "cross_platform_coarse_vod_stride_sec": "300",
                "cross_platform_coarse_vod_max_tiles_per_vod": "12",
                "cross_platform_coarse_vod_max_vods": "8",
                "cross_platform_coarse_vod_horizon_padding_sec": "600",
                "storage_dir": "/tmp/storage",
                "feature_build_version": 11,
                "enable_multimodal_embeddings": True,
                "code_hash": "abc123",
            },
        )

    def test_build_jynxzi_workflow_state_defaults_to_stronger_hard_negative_weight(self) -> None:
        with patch.object(training_workflow_state.settings, "whisper_model", "tiny"), \
             patch.object(training_workflow_state.settings, "storage_dir", "/tmp/storage"), \
             patch("ml.training_workflow_state._code_hash", return_value="abc123"), \
             patch.dict(os.environ, {}, clear=True):
            state = training_workflow_state.build_jynxzi_workflow_state(
                date_from="2026-04-01",
                date_to="2026-04-10",
                skip_covered=True,
                merge_output=False,
            )

        self.assertEqual(state["ranker_hard_negative_weight"], "4.0")
        self.assertEqual(state["ranker_rollout_review_negative_weight"], "4.0")
        self.assertEqual(state["ranker_use_user_labels"], "1")
        self.assertEqual(state["ranker_use_gold_train"], "1")
        self.assertEqual(state["ranker_rollout_threshold_min_recall"], "0.85")
        self.assertEqual(state["ranker_rollout_threshold_max_selected_rate"], "0.75")
        self.assertEqual(state["ranker_rollout_threshold_target_precision"], "0.85")
        self.assertEqual(state["ranker_primary_label_policy"], "outcome_only")
        self.assertEqual(state["ranker_require_cross_platform_outcome"], "1")
        self.assertEqual(state["ranker_min_cross_platform_train_rows"], "70")
        self.assertEqual(state["outcome_holdout_target_positive_rows"], "300")
        self.assertEqual(state["rollout_readiness_target_new_vod_groups"], "20")
        self.assertEqual(state["rollout_readiness_target_available_groups"], "40")
        self.assertEqual(state["rollout_readiness_max_candidates_per_vod"], "3")
        self.assertEqual(state["targeted_backfill_require_all_vods"], "1")
        self.assertEqual(state["cross_platform_fingerprint_enabled"], "1")
        self.assertEqual(state["ranker_use_mm_judge_proxy"], "1")
        self.assertEqual(state["ranker_mm_judge_weight"], "0.4")
        self.assertEqual(state["ranker_max_features"], "40")
        self.assertEqual(state["ranker_prefer_lightgbm"], "1")
        self.assertEqual(state["ranker_lightgbm_selection_tolerance"], "0.02")
        self.assertEqual(state["cross_platform_fingerprint_min_confidence"], "0.50")
        self.assertTrue(str(state["cross_platform_rows_path"]).endswith("jynxzi_cross_platform_fingerprint_rows.jsonl"))
        self.assertTrue(str(state["cross_platform_diagnostics_path"]).endswith("jynxzi_cross_platform_fingerprint_diagnostics.jsonl"))
        self.assertEqual(state["cross_platform_max_reposts"], "3600")
        self.assertEqual(state["cross_platform_min_reposts"], "0")
        self.assertEqual(state["cross_platform_min_matched_rows"], "70")
        self.assertEqual(state["cross_platform_seed_min_matched_rows"], "70")
        self.assertEqual(state["cross_platform_min_matched_vods"], "20")
        self.assertEqual(state["cross_platform_refresh_vods"], "0")
        self.assertEqual(state["cross_platform_second_pass_after_vod_build"], "1")
        self.assertEqual(state["cross_platform_refresh_reposts"], "0")
        self.assertEqual(state["cross_platform_reuse_existing_rows"], "0")
        self.assertEqual(state["cross_platform_holdout_ratio"], "0.20")
        self.assertEqual(state["cross_platform_discover_reposts"], "1")
        self.assertEqual(state["cross_platform_download_reposts"], "1")
        self.assertEqual(state["cross_platform_full_vod_audio_fingerprints"], "1")
        self.assertEqual(state["cross_platform_full_vod_audio_max_vods"], "0")
        self.assertEqual(state["cross_platform_full_vod_audio_workers"], "2")
        self.assertEqual(state["cross_platform_keep_full_vod_audio"], "0")
        self.assertEqual(state["cross_platform_coarse_vod_tiling"], "0")
        self.assertEqual(state["cross_platform_coarse_vod_tile_sec"], "360")
        self.assertEqual(state["cross_platform_coarse_vod_stride_sec"], "180")
        self.assertEqual(state["cross_platform_coarse_vod_max_tiles_per_vod"], "0")
        self.assertEqual(state["cross_platform_coarse_vod_max_vods"], "0")
        self.assertEqual(state["cross_platform_coarse_vod_horizon_padding_sec"], "900")
        self.assertIn(state["enable_clip_judge"], {"true", "false"})
        self.assertIn(state["clip_judge_fail_fast"], {"true", "false"})
        self.assertIn(state["judge_provider_fallback_enabled"], {"true", "false"})
