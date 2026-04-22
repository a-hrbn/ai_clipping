from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.pipelines.components import (
    build_training_dataset_op,
    collect_twitch_clips_op,
    promote_op,
    train_ranker_op,
)


class CollectComponentEnvTests(unittest.TestCase):
    def test_run_passes_twitch_credentials_for_subprocess(self) -> None:
        captured: dict[str, object] = {}

        def _fake_run_python_module(module: str, args: list[str], **kwargs) -> None:
            captured["module"] = module
            captured["args"] = list(args)
            captured["env"] = dict(kwargs["env"])
            output_path = Path(args[args.index("--output") + 1])
            output_path.write_text('{"clip_id":"c1"}\n', encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_uri = str(Path(tmp_dir) / "labels.jsonl")
            with patch.object(
                collect_twitch_clips_op,
                "run_python_module",
                side_effect=_fake_run_python_module,
            ):
                collect_twitch_clips_op.run(
                    streamer="jynxzi",
                    date_from="2026-03-14",
                    date_to="2026-03-20",
                    output_uri=output_uri,
                    twitch_client_id_secret="local-client-id",
                    twitch_client_secret_secret="local-client-secret",
                )

        self.assertEqual(captured["module"], "ml.collect_twitch_clip_labels")
        env = captured["env"]
        self.assertEqual(env["TWITCH_CLIENT_ID"], "local-client-id")
        self.assertEqual(env["TWITCH_CLIENT_SECRET"], "local-client-secret")


class BuildDatasetComponentEnvTests(unittest.TestCase):
    def test_run_passes_dataset_provider_and_llm_credentials(self) -> None:
        captured: dict[str, object] = {}

        def _fake_run_python_module(module: str, args: list[str], **kwargs) -> None:
            captured["module"] = module
            captured["args"] = list(args)
            captured["env"] = dict(kwargs["env"])
            output_path = Path(args[args.index("--output") + 1])
            output_path.write_text('{"label":1}\n', encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            labels_uri = str(Path(tmp_dir) / "labels.jsonl")
            Path(labels_uri).write_text('{"clip_id":"c1"}\n', encoding="utf-8")
            output_uri = str(Path(tmp_dir) / "training_rows.jsonl")
            with patch.object(
                build_training_dataset_op,
                "run_python_module",
                side_effect=_fake_run_python_module,
            ):
                build_training_dataset_op.run(
                    labels_uri=labels_uri,
                    output_uri=output_uri,
                    profile="jynxzi",
                    dataset_llm_provider="none",
                    groq_api_key_secret="local-groq-key",
                    ollama_host="http://ollama.internal:11434",
                )

        env = captured["env"]
        self.assertEqual(env["TRAINING_DATASET_LLM_PROVIDER"], "none")
        self.assertEqual(env["GROQ_API_KEY"], "local-groq-key")
        self.assertEqual(env["OLLAMA_HOST"], "http://ollama.internal:11434")


class TrainRankerComponentEnvTests(unittest.TestCase):
    def test_run_passes_mlflow_and_context_env_to_subprocess(self) -> None:
        captured: dict[str, object] = {}

        def _fake_run_python_module(module: str, args: list[str], **kwargs) -> None:
            captured["module"] = module
            captured["args"] = list(args)
            captured["env"] = dict(kwargs["env"])
            artifact_dir = Path(args[args.index("--artifact-dir") + 1])
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "jynxzi_ranker.json").write_text(
                '{"selected_model":"xgb","validation":{"ndcg_at_5":0.8},"training_stats":{"rows":12},"mlflow":{"run_id":"run-1"}}',
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_rows_uri = str(Path(tmp_dir) / "training_rows.jsonl")
            Path(training_rows_uri).write_text('{"label":1}\n', encoding="utf-8")
            artifact_bundle_uri = str(Path(tmp_dir) / "artifact_bundle")
            training_stats_uri = str(Path(tmp_dir) / "training_stats.json")
            with patch.object(
                train_ranker_op,
                "run_python_module",
                side_effect=_fake_run_python_module,
            ):
                train_ranker_op.run(
                    training_rows_uri=training_rows_uri,
                    artifact_bundle_uri=artifact_bundle_uri,
                    training_stats_uri=training_stats_uri,
                    profile="jynxzi",
                    streamer="jynxzi",
                    date_from="2026-03-14",
                    date_to="2026-03-20",
                    mlflow_tracking_uri_secret="http://mlflow.local",
                )

        env = captured["env"]
        self.assertEqual(env["MLFLOW_TRACKING_URI"], "http://mlflow.local")
        self.assertEqual(env["TRAINING_STREAMER"], "jynxzi")
        self.assertEqual(env["DATE_FROM"], "2026-03-14")
        self.assertEqual(env["DATE_TO"], "2026-03-20")
        self.assertNotIn("LOCAL_DOCKER_IMAGE", env)


class PromoteComponentEnvTests(unittest.TestCase):
    def test_run_applies_mlflow_tracking_uri_for_in_process_registration(self) -> None:
        captured: dict[str, str] = {}

        def _fake_register_existing_artifact_bundle(**kwargs):
            del kwargs
            captured["tracking_uri"] = os.environ.get("MLFLOW_TRACKING_URI", "")
            return {"registered_version": "1"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_dir = root / "artifact_bundle"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "jynxzi_ranker.json").write_text("{}", encoding="utf-8")
            (artifact_dir / "jynxzi_artifact_health.json").write_text("{}", encoding="utf-8")
            for filename in ("proxy_eval.json", "gold_eval.json", "pipeline_eval.json"):
                (root / filename).write_text("{}", encoding="utf-8")
            output_uri = str(root / "registry_report.json")

            with patch.object(
                promote_op.mlflow_tracking,
                "register_existing_artifact_bundle",
                side_effect=_fake_register_existing_artifact_bundle,
            ):
                promote_op.run(
                    artifact_bundle_uri=str(artifact_dir),
                    proxy_eval_report_uri=str(root / "proxy_eval.json"),
                    gold_eval_report_uri=str(root / "gold_eval.json"),
                    pipeline_eval_report_uri=str(root / "pipeline_eval.json"),
                    output_uri=output_uri,
                    profile="jynxzi",
                    mlflow_tracking_uri_secret="http://mlflow.local",
                )

        self.assertEqual(captured["tracking_uri"], "http://mlflow.local")


if __name__ == "__main__":
    unittest.main()
