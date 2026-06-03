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
            whisper_language="en",
            semantic_device="cuda",
            semantic_batch_size=0,
            storage_dir="tmp/storage",
            train_tmpdir="tmp/training_runs",
            dataset_llm_provider="none",
            ollama_host="http://host.docker.internal:11434",
            training_vod_workers=0,
            training_label_workers=0,
            training_ffmpeg_video_codec="h264_nvenc",
            training_ffmpeg_audio_codec="aac",
            training_ffmpeg_preset="",
            training_ffmpeg_threads=0,
            video_expression_sample_sec=6,
        )
        with patch("ml.local_train.os.cpu_count", return_value=12), \
             patch("ml.local_train._available_memory_gib", return_value=27.4):
            env = local_train._configure_runtime_env(args)

        self.assertEqual(env["WHISPER_DEVICE"], "cuda")
        self.assertEqual(env["WHISPER_LANGUAGE"], "en")
        self.assertEqual(env["SEMANTIC_DEVICE"], "cuda")
        self.assertEqual(env["TRAINING_DATASET_LLM_PROVIDER"], "none")
        self.assertEqual(env["TRAINING_VOD_WORKERS"], "2")
        self.assertEqual(env["TRAINING_LABEL_WORKERS"], "2")
        self.assertEqual(env["TRAINING_FFMPEG_VIDEO_CODEC"], "h264_nvenc")
        self.assertEqual(env["TRAINING_FFMPEG_AUDIO_CODEC"], "aac")
        self.assertEqual(env["TRAINING_FFMPEG_PRESET"], "")
        self.assertEqual(env["TRAINING_FFMPEG_THREADS"], "0")
        self.assertEqual(env["VOD_RANKER_SHORT_FORM_MODE"], "off")
        self.assertEqual(env["SEMANTIC_BATCH_SIZE"], "256")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["MALLOC_ARENA_MAX"], "2")
        self.assertEqual(env["TF_FORCE_GPU_ALLOW_GROWTH"], "true")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_recommended_worker_plan_allows_more_parallelism_on_large_memory_hosts(self) -> None:
        plan = local_train._recommended_worker_plan(24, 64.0)

        self.assertEqual(plan.vod_workers, 4)
        self.assertEqual(plan.label_workers, 3)

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
                whisper_language="en",
                semantic_device="cuda",
                semantic_batch_size=0,
                dataset_llm_provider="none",
                ollama_host="http://host.docker.internal:11434",
                training_vod_workers=2,
                training_label_workers=2,
                training_ffmpeg_video_codec="h264_nvenc",
                training_ffmpeg_audio_codec="aac",
                training_ffmpeg_preset="",
                training_ffmpeg_threads=0,
                video_expression_sample_sec=6,
                artifact_only=True,
                skip_covered=True,
                merge_output=True,
            )
            with patch.object(local_train, "_parse_args", return_value=args), \
                 patch.object(local_train, "_configure_runtime_env", return_value={"PATH": "", "WHISPER_DEVICE": "cuda", "WHISPER_LANGUAGE": "en", "SEMANTIC_DEVICE": "cuda", "TRAINING_VOD_WORKERS": "2", "TRAINING_LABEL_WORKERS": "2", "TRAINING_FFMPEG_VIDEO_CODEC": "h264_nvenc", "TRAINING_FFMPEG_AUDIO_CODEC": "aac", "TRAINING_FFMPEG_PRESET": "", "TRAINING_FFMPEG_THREADS": "0", "SEMANTIC_BATCH_SIZE": "256", "TRAINING_DATASET_LLM_PROVIDER": "none", "VIDEO_EXPRESSION_SAMPLE_SEC": "6", "STORAGE_DIR": "tmp/storage", "AI_CLIPPING_TRAIN_TMPDIR": "tmp/training_runs"}), \
                 patch.object(local_train.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run_mock:
                code = local_train.main()
                summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        run_mock.assert_called_once()
        self.assertEqual(summary["training_return_code"], 0)
        self.assertTrue(summary["artifact_only"])
        self.assertEqual(summary["hardware_profile"]["whisper_device"], "cuda")
        self.assertEqual(summary["hardware_profile"]["whisper_language"], "en")
        self.assertEqual(summary["hardware_profile"]["training_ffmpeg_video_codec"], "h264_nvenc")
        self.assertEqual(summary["hardware_profile"]["training_ffmpeg_threads"], "0")
        self.assertEqual(summary["hardware_profile"]["ranker_short_form_mode"], "off")


if __name__ == "__main__":
    unittest.main()
