import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_runner


class BuildSubprocessCommandTests(unittest.TestCase):
    def test_url_job_passes_url_and_omits_video_path(self) -> None:
        cmd = pipeline_runner.build_subprocess_command(
            job_id="job-1",
            url="https://www.twitch.tv/videos/123",
            video_path=None,
            profile="jynxzi",
            strategy={"clip_style": "balanced"},
        )

        self.assertEqual(cmd[0], pipeline_runner.sys.executable)
        self.assertTrue(cmd[1].endswith("pipeline_worker.py"))
        self.assertIn("--url", cmd)
        self.assertIn("https://www.twitch.tv/videos/123", cmd)
        self.assertNotIn("--video-path", cmd)

    def test_video_path_job_passes_path_and_omits_url(self) -> None:
        cmd = pipeline_runner.build_subprocess_command(
            job_id="job-1",
            url=None,
            video_path=Path("/data/video.mp4"),
            profile="general",
            strategy={},
        )

        self.assertIn("--video-path", cmd)
        self.assertIn(str(Path("/data/video.mp4")), cmd)
        self.assertNotIn("--url", cmd)

    def test_video_path_wins_when_both_url_and_path_are_given(self) -> None:
        cmd = pipeline_runner.build_subprocess_command(
            job_id="job-1",
            url="https://www.twitch.tv/videos/123",
            video_path=Path("/data/video.mp4"),
            profile="general",
            strategy={},
        )

        self.assertIn("--video-path", cmd)
        self.assertNotIn("--url", cmd)

    def test_profile_is_passed_through(self) -> None:
        cmd = pipeline_runner.build_subprocess_command(
            job_id="job-1",
            url="https://www.twitch.tv/videos/123",
            video_path=None,
            profile="jynxzi",
            strategy={},
        )

        self.assertEqual(cmd[cmd.index("--profile") + 1], "jynxzi")

    def test_strategy_is_serialized_as_json(self) -> None:
        cmd = pipeline_runner.build_subprocess_command(
            job_id="job-1",
            url="https://www.twitch.tv/videos/123",
            video_path=None,
            profile="general",
            strategy={"clip_style": "balanced", "max_clips": 5},
        )

        raw_strategy = cmd[cmd.index("--strategy-json") + 1]
        payload = pipeline_runner.json.loads(raw_strategy)
        self.assertEqual(payload, {"clip_style": "balanced", "max_clips": 5})

    def test_strategy_set_fields_are_serialized_as_sorted_lists(self) -> None:
        cmd = pipeline_runner.build_subprocess_command(
            job_id="job-1",
            url="https://www.twitch.tv/videos/123",
            video_path=None,
            profile="general",
            strategy={
                "keywords": {"rage", "funny"},
                "nested": {"exclude_keywords": {"spoiler"}},
            },
        )

        raw_strategy = cmd[cmd.index("--strategy-json") + 1]
        payload = pipeline_runner.json.loads(raw_strategy)
        self.assertEqual(payload["keywords"], ["funny", "rage"])
        self.assertEqual(payload["nested"]["exclude_keywords"], ["spoiler"])

    def test_none_strategy_serializes_to_empty_object(self) -> None:
        cmd = pipeline_runner.build_subprocess_command(
            job_id="job-1",
            url="https://www.twitch.tv/videos/123",
            video_path=None,
            profile="general",
            strategy=None,
        )

        raw_strategy = cmd[cmd.index("--strategy-json") + 1]
        self.assertEqual(pipeline_runner.json.loads(raw_strategy), {})


class BuildSubprocessEnvTests(unittest.TestCase):
    def test_pythonpath_prepends_backend_root_and_preserves_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline_runner.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(pipeline_runner.os.environ, {"PYTHONPATH": "/existing/path"}, clear=True):
            env = pipeline_runner.build_subprocess_env()

        parts = env["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(parts[0], str(pipeline_runner._BACKEND_ROOT))
        self.assertIn("/existing/path", parts)

    def test_pythonpath_is_just_backend_root_when_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline_runner.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(pipeline_runner.os.environ, {}, clear=True):
            env = pipeline_runner.build_subprocess_env()

        self.assertEqual(env["PYTHONPATH"], str(pipeline_runner._BACKEND_ROOT))

    def test_thread_caps_and_headless_defaults_are_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline_runner.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(pipeline_runner.os.environ, {}, clear=True):
            env = pipeline_runner.build_subprocess_env()
            self.assertTrue(Path(env["MPLCONFIGDIR"]).exists())

        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["MKL_NUM_THREADS"], "1")
        self.assertEqual(env["NUMEXPR_NUM_THREADS"], "1")
        self.assertEqual(env["VECLIB_MAXIMUM_THREADS"], "1")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(env["WHISPER_LANGUAGE"], "en")

    def test_face_detection_is_enabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline_runner.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(pipeline_runner.os.environ, {}, clear=True):
            env = pipeline_runner.build_subprocess_env()

        self.assertEqual(env["AI_CLIPPING_DISABLE_FACE_DETECTION"], "0")

    def test_diarization_reflects_disabled_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline_runner.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.object(pipeline_runner.settings, "pipeline_enable_diarization", False), \
             patch.dict(pipeline_runner.os.environ, {}, clear=True):
            env = pipeline_runner.build_subprocess_env()

        self.assertEqual(env["PIPELINE_ENABLE_DIARIZATION"], "false")

    def test_diarization_reflects_enabled_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline_runner.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.object(pipeline_runner.settings, "pipeline_enable_diarization", True), \
             patch.dict(pipeline_runner.os.environ, {}, clear=True):
            env = pipeline_runner.build_subprocess_env()

        self.assertEqual(env["PIPELINE_ENABLE_DIARIZATION"], "true")

    def test_setdefault_does_not_override_preset_values(self) -> None:
        preset = {
            "OMP_NUM_THREADS": "8",
            "AI_CLIPPING_DISABLE_FACE_DETECTION": "1",
            "PIPELINE_ENABLE_DIARIZATION": "true",
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline_runner.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.object(pipeline_runner.settings, "pipeline_enable_diarization", False), \
             patch.dict(pipeline_runner.os.environ, preset, clear=True):
            env = pipeline_runner.build_subprocess_env()

        self.assertEqual(env["OMP_NUM_THREADS"], "8")
        self.assertEqual(env["AI_CLIPPING_DISABLE_FACE_DETECTION"], "1")
        self.assertEqual(env["PIPELINE_ENABLE_DIARIZATION"], "true")

    def test_cuda_device_defaults_are_added_when_whisper_device_is_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline_runner.settings, "storage_dir", str(Path(tmp) / "storage")), \
             patch.dict(pipeline_runner.os.environ, {"WHISPER_DEVICE": "cuda"}, clear=True):
            env = pipeline_runner.build_subprocess_env()

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")


class SubprocessErrorTests(unittest.TestCase):
    def test_sigkill_signal_mentions_oom(self) -> None:
        self.assertEqual(
            pipeline_runner._subprocess_error(-9),
            "Pipeline subprocess exited from signal 9 (SIGKILL; likely OOM or external kill)",
        )

    def test_generic_signal_is_reported(self) -> None:
        self.assertEqual(
            pipeline_runner._subprocess_error(-11),
            "Pipeline subprocess exited from signal 11",
        )

    def test_positive_returncode_reports_exit_code(self) -> None:
        self.assertEqual(
            pipeline_runner._subprocess_error(2),
            "Pipeline subprocess exited with code 2",
        )


if __name__ == "__main__":
    unittest.main()
