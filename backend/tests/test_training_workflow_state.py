import json
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

    def test_build_jynxzi_workflow_state_includes_runtime_knobs(self) -> None:
        with patch.object(training_workflow_state.settings, "whisper_model", "tiny"), \
             patch.object(training_workflow_state.settings, "storage_dir", "/tmp/storage"), \
             patch("ml.training_workflow_state._code_hash", return_value="abc123"), \
             patch.dict(os.environ, {"TRAINING_DATASET_LLM_PROVIDER": "groq"}, clear=False):
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
                "storage_dir": "/tmp/storage",
                "code_hash": "abc123",
            },
        )
