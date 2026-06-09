from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ml.pipelines.backends import local_shell


class LocalShellPipelineTests(unittest.TestCase):
    def test_run_local_pipeline_invokes_all_component_modules_in_order(self) -> None:
        calls: list[tuple[str, list[str]]] = []

        def _fake_run_python_module(module: str, args: list[str], **kwargs) -> None:
            del kwargs
            calls.append((module, list(args)))

        with patch.object(local_shell, "run_python_module", side_effect=_fake_run_python_module):
            outputs = local_shell.run_local_pipeline(
                streamer="jynxzi",
                profile="jynxzi",
                date_from="2026-03-14",
                date_to="2026-03-20",
                pipeline_root="/tmp/pipeline-root",
            )

        self.assertEqual(
            [module for module, _ in calls],
            [
                "ml.pipelines.components.collect_twitch_clips_op",
                "ml.pipelines.components.build_training_dataset_op",
                "ml.pipelines.components.build_eval_dataset_op",
            ],
        )
        self.assertIn("labels_uri", outputs)
        self.assertIn("training_rows_uri", outputs)
        self.assertIn("--skip-covered", calls[0][1])
        self.assertIn("--merge-output", calls[0][1])
        self.assertEqual(Path(calls[1][1][1]), Path("/tmp/pipeline-root/datasets/jynxzi_twitch_labels.jsonl"))


if __name__ == "__main__":
    unittest.main()
