import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from producers.moments import MomentFinding, MomentParams


class _FakeMomentFinderError(RuntimeError):
    """Stand-in for the injected ``moment_finder.MomentFinderError`` — keeps the
    producer torch-free (the real one pulls RichSegment → torch)."""


def _params(**over) -> MomentParams:
    base = dict(
        provider="gemini",
        model="gemini-2.0",
        max_moments=180,
        min_duration_sec=8.0,
        max_duration_sec=90.0,
        max_tokens=32000,
        chunk_duration_sec=2700.0,
        chunk_overlap_sec=120.0,
        timeout_sec=900.0,
        api_retries=2,
        retry_backoff_sec=20.0,
        filter_transcript_noise=True,
    )
    base.update(over)
    return MomentParams(**base)


def _result(ranked: list) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        ranked_candidates=ranked,
        post_processed_moments=[{"rank": 1, "start_time": 0.0, "end_time": 5.0}],
        raw_moments=[{"raw": 1}],
        prompt="PROMPT",
        response_text="RESPONSE",
        usage={"provider": "gemini", "input_tokens": 10},
        summary={"ranked_candidate_count": len(ranked), "post_processed_moment_count": 1},
    )


class MomentFindingTests(unittest.TestCase):
    def test_success_writes_running_then_ok_stage_with_summary_and_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            debug = Path(tmp)
            find_calls: list = []
            mf = MomentFinding(
                profile="jynxzi",
                vod_id="123",
                debug_dir=debug,
                params=_params(),
                find=lambda _segs, **kw: find_calls.append(kw) or _result([object()]),
                error=_FakeMomentFinderError,
            )

            out = mf.find([object()])

            stage = json.loads((debug / "llm_moment_finder_stage.json").read_text(encoding="utf-8"))
            self.assertEqual(stage["status"], "ok")
            self.assertEqual(stage["provider"], "gemini")
            self.assertEqual(stage["model"], "gemini-2.0")
            self.assertEqual(stage["max_moments"], 180)
            self.assertEqual(stage["ranked_candidate_count"], 1)  # merged-in summary
            self.assertEqual((debug / "llm_moment_finder_prompt.txt").read_text(encoding="utf-8"), "PROMPT")
            self.assertEqual((debug / "llm_moment_finder_response.txt").read_text(encoding="utf-8"), "RESPONSE")
            self.assertEqual(
                len((debug / "llm_moment_finder_moments.jsonl").read_text(encoding="utf-8").splitlines()), 1
            )
            self.assertTrue((debug / "llm_moment_finder_moments_raw.jsonl").exists())
            self.assertTrue((debug / "llm_moment_finder_usage.json").exists())
            self.assertIs(out.ranked_candidates[0], find_calls[0] and out.ranked_candidates[0])

    def test_params_and_identity_are_passed_through_to_find(self) -> None:
        # purity: every find_moments arg is resolved from the injected params /
        # identity, never from config.settings.
        with TemporaryDirectory() as tmp:
            captured: dict = {}
            mf = MomentFinding(
                profile="elias",
                vod_id="vod-9",
                debug_dir=Path(tmp),
                params=_params(provider="ollama", max_moments=42, temperature=0.0),
                find=lambda _segs, **kw: captured.update(kw) or _result([object()]),
                error=_FakeMomentFinderError,
            )

            mf.find([object()])

            self.assertEqual(captured["profile"], "elias")
            self.assertEqual(captured["vod_id"], "vod-9")
            self.assertEqual(captured["provider"], "ollama")
            self.assertEqual(captured["max_moments"], 42)
            self.assertEqual(captured["min_duration_sec"], 8.0)
            self.assertEqual(captured["filter_transcript_noise"], True)
            self.assertEqual(captured["temperature"], 0.0)

    def test_empty_ranked_candidates_raises_and_marks_stage_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            debug = Path(tmp)
            mf = MomentFinding(
                profile="jynxzi",
                vod_id="123",
                debug_dir=debug,
                params=_params(),
                find=lambda _segs, **kw: _result([]),  # no ranked candidates
                error=_FakeMomentFinderError,
            )

            with self.assertRaises(_FakeMomentFinderError):
                mf.find([object()])

            stage = json.loads((debug / "llm_moment_finder_stage.json").read_text(encoding="utf-8"))
            self.assertEqual(stage["status"], "failed")
            self.assertEqual(stage["error_type"], "_FakeMomentFinderError")
            self.assertIn("no ranked candidates", stage["error"])
            # ok-path artifacts are not written on the failure path
            self.assertFalse((debug / "llm_moment_finder_moments.jsonl").exists())
            self.assertFalse((debug / "llm_moment_finder_prompt.txt").exists())

    def test_find_failure_marks_stage_failed_and_reraises(self) -> None:
        with TemporaryDirectory() as tmp:
            debug = Path(tmp)

            def boom(_segs, **_kw):
                raise _FakeMomentFinderError("provider exploded")

            mf = MomentFinding(
                profile="jynxzi",
                vod_id="123",
                debug_dir=debug,
                params=_params(),
                find=boom,
                error=_FakeMomentFinderError,
            )

            with self.assertRaises(_FakeMomentFinderError):
                mf.find([object()])

            stage = json.loads((debug / "llm_moment_finder_stage.json").read_text(encoding="utf-8"))
            self.assertEqual(stage["status"], "failed")
            self.assertIn("provider exploded", stage["error"])


if __name__ == "__main__":
    unittest.main()
