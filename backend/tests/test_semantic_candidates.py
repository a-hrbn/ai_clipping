import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from services import semantic_candidates


class _FakeSentenceTransformer:
    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls.append((args, kwargs))


class SemanticCandidatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_model = semantic_candidates._model
        self._previous_unavailable_reasons = set(semantic_candidates._model_unavailable_logged_reasons)
        semantic_candidates._model = None
        semantic_candidates._model_unavailable_logged_reasons.clear()
        _FakeSentenceTransformer.calls.clear()

    def tearDown(self) -> None:
        semantic_candidates._model = self._previous_model
        semantic_candidates._model_unavailable_logged_reasons.clear()
        semantic_candidates._model_unavailable_logged_reasons.update(self._previous_unavailable_reasons)
        _FakeSentenceTransformer.calls.clear()

    def test_get_model_prefers_offline_load_when_local_cache_exists(self) -> None:
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = _FakeSentenceTransformer

        with patch.dict(sys.modules, {"sentence_transformers": module}), \
             patch.object(semantic_candidates.settings, "semantic_device", "cuda"), \
             patch("services.semantic_candidates.torch.cuda.is_available", return_value=True):
            with patch.object(
                semantic_candidates,
                "_materialize_local_sentence_transformer_model",
                return_value=Path("/tmp/local-model"),
            ):
                model = semantic_candidates._get_model()

        self.assertIsInstance(model, _FakeSentenceTransformer)
        _args, kwargs = _FakeSentenceTransformer.calls[-1]
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(kwargs["cache_folder"], str(semantic_candidates._sentence_transformer_cache_folder()))
        self.assertEqual(kwargs["device"], "cuda")

    def test_get_model_fails_open_when_local_cache_missing(self) -> None:
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = _FakeSentenceTransformer

        with patch.dict(sys.modules, {"sentence_transformers": module}), \
             patch.object(semantic_candidates.settings, "semantic_device", "auto"), \
             patch("services.semantic_candidates.torch.cuda.is_available", return_value=False):
            with patch.object(
                semantic_candidates,
                "_materialize_local_sentence_transformer_model",
                return_value=None,
            ):
                with self.assertLogs(semantic_candidates.logger, level="WARNING") as logs:
                    model = semantic_candidates._get_model()

        self.assertIsNone(model)
        self.assertEqual(_FakeSentenceTransformer.calls, [])
        self.assertIn("semantic_candidates_model_unavailable", "\n".join(logs.output))

    def test_get_model_logs_missing_local_snapshot_once(self) -> None:
        with patch.object(
            semantic_candidates,
            "_materialize_local_sentence_transformer_model",
            return_value=None,
        ):
            with self.assertLogs(semantic_candidates.logger, level="DEBUG") as logs:
                self.assertIsNone(semantic_candidates._get_model())
                self.assertIsNone(semantic_candidates._get_model())

        log_text = "\n".join(logs.output)
        self.assertEqual(log_text.count("semantic_candidates_model_unavailable model="), 1)
        self.assertIn("semantic_candidates_model_unavailable_suppressed", log_text)

    def test_materializes_local_sentence_transformer_model_from_split_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            hf_home = cache_root / "huggingface"
            hub_snapshot_dir = (
                hf_home
                / "hub"
                / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
                / "snapshots"
                / "revision-1"
            )
            transformer_snapshot_dir = (
                hf_home
                / "transformers"
                / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
                / "snapshots"
                / "revision-1"
            )
            (hub_snapshot_dir / "1_Pooling").mkdir(parents=True)
            transformer_snapshot_dir.mkdir(parents=True)
            (hub_snapshot_dir / "modules.json").write_text("[]", encoding="utf-8")
            (hub_snapshot_dir / "config_sentence_transformers.json").write_text("{}", encoding="utf-8")
            (hub_snapshot_dir / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
            (transformer_snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
            (transformer_snapshot_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (transformer_snapshot_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (transformer_snapshot_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            with patch.object(semantic_candidates, "DEFAULT_CACHE_DIR", cache_root):
                with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                    model_dir = semantic_candidates._materialize_local_sentence_transformer_model(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )

            self.assertIsNotNone(model_dir)
            assert model_dir is not None
            self.assertTrue((model_dir / "modules.json").exists())
            self.assertTrue((model_dir / "config_sentence_transformers.json").exists())
            self.assertTrue((model_dir / "1_Pooling" / "config.json").exists())
            self.assertTrue((model_dir / "config.json").exists())
            self.assertTrue((model_dir / "tokenizer_config.json").exists())
            self.assertTrue((model_dir / "tokenizer.json").exists())
            self.assertTrue((model_dir / "model.safetensors").exists())
            self.assertTrue(
                semantic_candidates._materialized_model_complete(model_dir)
            )

    def test_materialize_returns_none_when_split_cache_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            hf_home = cache_root / "huggingface"
            hub_snapshot_dir = (
                hf_home
                / "hub"
                / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
                / "snapshots"
                / "revision-1"
            )
            (hub_snapshot_dir / "1_Pooling").mkdir(parents=True)
            (hub_snapshot_dir / "modules.json").write_text("[]", encoding="utf-8")
            (hub_snapshot_dir / "config_sentence_transformers.json").write_text("{}", encoding="utf-8")
            (hub_snapshot_dir / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")

            with patch.object(semantic_candidates, "DEFAULT_CACHE_DIR", cache_root):
                with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                    model_dir = semantic_candidates._materialize_local_sentence_transformer_model(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )

            self.assertIsNone(model_dir)

    def test_materializes_local_sentence_transformer_model_from_hub_only_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            hf_home = cache_root / "huggingface"
            hub_snapshot_dir = (
                hf_home
                / "hub"
                / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
                / "snapshots"
                / "revision-1"
            )
            (hub_snapshot_dir / "1_Pooling").mkdir(parents=True)
            (hub_snapshot_dir / "modules.json").write_text("[]", encoding="utf-8")
            (hub_snapshot_dir / "config_sentence_transformers.json").write_text("{}", encoding="utf-8")
            (hub_snapshot_dir / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
            (hub_snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
            (hub_snapshot_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (hub_snapshot_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (hub_snapshot_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            with patch.object(semantic_candidates, "DEFAULT_CACHE_DIR", cache_root):
                with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                    model_dir = semantic_candidates._materialize_local_sentence_transformer_model(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )

            self.assertIsNotNone(model_dir)
            assert model_dir is not None
            self.assertTrue(
                semantic_candidates._materialized_model_complete(model_dir)
            )


if __name__ == "__main__":
    unittest.main()
