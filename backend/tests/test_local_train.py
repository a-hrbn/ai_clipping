from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import local_train


class LocalTrainTests(unittest.TestCase):
    def test_configure_runtime_env_prefers_gpu_and_fast_local_defaults(self) -> None:
        args = types.SimpleNamespace(
            whisper_model="base",
            whisper_device="cuda",
            semantic_device="cuda",
            semantic_batch_size=0,
            storage_dir="tmp/storage",
            train_tmpdir="tmp/training_runs",
            dataset_llm_provider="none",
            ollama_host="http://host.docker.internal:11434",
            training_vod_workers=0,
            training_label_workers=0,
            video_expression_sample_sec=6,
        )
        with patch("ml.local_train.os.cpu_count", return_value=12):
            env = local_train._configure_runtime_env(args)

        self.assertEqual(env["WHISPER_DEVICE"], "cuda")
        self.assertEqual(env["SEMANTIC_DEVICE"], "cuda")
        self.assertEqual(env["TRAINING_DATASET_LLM_PROVIDER"], "none")
        self.assertEqual(env["TRAINING_VOD_WORKERS"], "2")
        self.assertEqual(env["TRAINING_LABEL_WORKERS"], "3")
        self.assertEqual(env["SEMANTIC_BATCH_SIZE"], "256")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_main_writes_summary_and_invokes_training_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            args = types.SimpleNamespace(
                date_from="2026-03-14",
                date_to="2026-03-20",
                streamer="jynxzi",
                summary_path=str(summary_path),
                storage_dir="tmp/storage",
                train_tmpdir="tmp/training_runs",
                whisper_model="base",
                whisper_device="cuda",
                semantic_device="cuda",
                semantic_batch_size=0,
                dataset_llm_provider="none",
                ollama_host="http://host.docker.internal:11434",
                training_vod_workers=2,
                training_label_workers=2,
                video_expression_sample_sec=6,
                artifact_only=True,
                skip_covered=True,
                merge_output=True,
            )
            with patch.object(local_train, "_parse_args", return_value=args), \
                 patch.object(local_train, "_configure_runtime_env", return_value={"PATH": "", "WHISPER_DEVICE": "cuda", "SEMANTIC_DEVICE": "cuda", "TRAINING_VOD_WORKERS": "2", "TRAINING_LABEL_WORKERS": "2", "SEMANTIC_BATCH_SIZE": "256", "TRAINING_DATASET_LLM_PROVIDER": "none", "VIDEO_EXPRESSION_SAMPLE_SEC": "6", "STORAGE_DIR": "tmp/storage", "AI_CLIPPING_TRAIN_TMPDIR": "tmp/training_runs"}), \
                 patch.object(local_train.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run_mock:
                code = local_train.main()
                summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        run_mock.assert_called_once()
        self.assertEqual(summary["training_return_code"], 0)
        self.assertTrue(summary["artifact_only"])
        self.assertEqual(summary["hardware_profile"]["whisper_device"], "cuda")


if __name__ == "__main__":
    unittest.main()
