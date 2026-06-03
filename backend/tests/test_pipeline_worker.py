import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import services

import pipeline_worker


class PipelineWorkerTests(unittest.TestCase):
    def test_preload_semantic_candidates_before_pipeline_import_uses_service_preload(self) -> None:
        calls: list[str] = []
        semantic_module = types.ModuleType("services.semantic_candidates")
        semantic_module.preload_model = lambda: calls.append("preload") or True

        with patch.object(services, "semantic_candidates", semantic_module, create=True), \
             patch.dict(sys.modules, {"services.semantic_candidates": semantic_module}):
            loaded = pipeline_worker._preload_semantic_candidates_before_pipeline_import()

        self.assertTrue(loaded)
        self.assertEqual(calls, ["preload"])


if __name__ == "__main__":
    unittest.main()
