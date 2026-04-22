import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import semantic_service
from config import settings
from services.transcriber import RichSegment


class SemanticServiceTests(unittest.TestCase):
    def test_semantic_scores_detect_surprise_and_quoteability(self) -> None:
        window = [
            RichSegment(start=0.0, end=2.0, text="Listen, the truth is this is crazy!", emotions=["exclamation"]),
            RichSegment(start=2.0, end=5.0, text="I thought we lose, but no way this worked."),
        ]

        self.assertGreater(semantic_service.quoteability_score(window), 0.4)
        self.assertGreater(semantic_service.surprise_score(window), 0.3)
        self.assertGreater(semantic_service.contradiction_score(window), 0.2)
        self.assertGreater(semantic_service.standalone_score(window, 1.0), 0.5)

    def test_call_llm_prefers_ollama_by_default_even_when_groq_is_configured(self) -> None:
        with patch.object(settings, "groq_api_key", "secret"), \
             patch.object(semantic_service, "_call_ollama_semantic", return_value={"hook": 0.7}) as ollama_mock, \
             patch.object(semantic_service, "_call_groq", return_value={"hook": 0.2}) as groq_mock:
            result = semantic_service._call_llm("test")

        self.assertEqual(result, {"hook": 0.7})
        ollama_mock.assert_called_once_with("test")
        groq_mock.assert_not_called()

    def test_call_llm_uses_groq_as_fallback_when_ollama_returns_none(self) -> None:
        with patch.object(settings, "groq_api_key", "secret"), \
             patch.object(semantic_service, "_call_ollama_semantic", return_value=None) as ollama_mock, \
             patch.object(semantic_service, "_call_groq", return_value={"hook": 0.2}) as groq_mock:
            result = semantic_service._call_llm("test")

        self.assertEqual(result, {"hook": 0.2})
        ollama_mock.assert_called_once_with("test")
        groq_mock.assert_called_once_with("test")


if __name__ == "__main__":
    unittest.main()
