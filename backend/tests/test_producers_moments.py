import types
import unittest

from producers.moments import MomentFinding, MomentParams
from services.job_artifact_store import InMemoryJobArtifactStore

_JOB = "job-1"


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


def _result(ranked: list, *, post_processed=None, raw=None) -> types.SimpleNamespace:
    # post_processed (the dict-rows) is the serving currency the empty check now
    # guards; default it to mirror the ranked count so a non-empty `ranked`
    # produces a non-empty serving set. raw_moments is eval-only and empty unless
    # the run opted into eval_recall.
    if post_processed is None:
        post_processed = [
            {"rank": i + 1, "start_time": 0.0, "end_time": 5.0} for i in range(len(ranked))
        ]
    if raw is None:
        raw = [{"raw": 1}] if ranked else []
    return types.SimpleNamespace(
        ranked_candidates=ranked,
        post_processed_moments=post_processed,
        raw_moments=raw,
        prompt="PROMPT",
        response_text="RESPONSE",
        usage={"provider": "gemini", "input_tokens": 10},
        summary={
            "ranked_candidate_count": len(ranked),
            "post_processed_moment_count": len(post_processed),
        },
    )


class MomentFindingTests(unittest.TestCase):
    def test_success_writes_running_then_ok_stage_with_summary_and_artifacts(self) -> None:
        # Artifacts are asserted in-memory against the injected fake store, never a
        # tmp directory: the producer owns no filenames, only artifact roles.
        store = InMemoryJobArtifactStore()
        find_calls: list = []
        mf = MomentFinding(
            profile="jynxzi",
            vod_id="123",
            job_id=_JOB,
            store=store,
            params=_params(),
            find=lambda _segs, **kw: find_calls.append(kw) or _result([object()]),
            error=_FakeMomentFinderError,
        )

        out = mf.find([object()])

        stage = store.read_stage(_JOB)
        self.assertEqual(stage["status"], "ok")
        self.assertEqual(stage["provider"], "gemini")
        self.assertEqual(stage["model"], "gemini-2.0")
        self.assertEqual(stage["max_moments"], 180)
        self.assertEqual(stage["ranked_candidate_count"], 1)  # merged-in summary
        self.assertEqual(store.read_prompt(_JOB), "PROMPT")
        self.assertEqual(store.read_response(_JOB), "RESPONSE")
        self.assertEqual(len(store.read_moments(_JOB)), 1)
        # eval_recall is off by default, so the eval-only raw artifact is not written.
        self.assertEqual(find_calls[0]["eval_recall"], False)
        self.assertEqual(store.read_raw_moments(_JOB), [])
        self.assertIsNotNone(store.read_usage(_JOB))
        self.assertIs(out.ranked_candidates[0], find_calls[0] and out.ranked_candidates[0])

    def test_eval_recall_passes_flag_and_writes_raw_artifact(self) -> None:
        store = InMemoryJobArtifactStore()
        captured: dict = {}
        mf = MomentFinding(
            profile="jynxzi",
            vod_id="123",
            job_id=_JOB,
            store=store,
            params=_params(eval_recall=True),
            find=lambda _segs, **kw: captured.update(kw) or _result([object()]),
            error=_FakeMomentFinderError,
        )

        mf.find([object()])

        self.assertEqual(captured["eval_recall"], True)
        self.assertEqual(store.read_raw_moments(_JOB), [{"raw": 1}])

    def test_params_and_identity_are_passed_through_to_find(self) -> None:
        # purity: every find_moments arg is resolved from the injected params /
        # identity, never from config.settings.
        store = InMemoryJobArtifactStore()
        captured: dict = {}
        mf = MomentFinding(
            profile="elias",
            vod_id="vod-9",
            job_id=_JOB,
            store=store,
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
        self.assertEqual(captured["eval_recall"], False)

    def test_empty_post_processed_moments_raises_and_marks_stage_failed(self) -> None:
        store = InMemoryJobArtifactStore()
        mf = MomentFinding(
            profile="jynxzi",
            vod_id="123",
            job_id=_JOB,
            store=store,
            params=_params(),
            find=lambda _segs, **kw: _result([]),  # no serving moments
            error=_FakeMomentFinderError,
        )

        with self.assertRaises(_FakeMomentFinderError):
            mf.find([object()])

        stage = store.read_stage(_JOB)
        self.assertEqual(stage["status"], "failed")
        self.assertEqual(stage["error_type"], "_FakeMomentFinderError")
        self.assertIn("no moments", stage["error"])
        # ok-path artifacts are not written on the failure path
        self.assertEqual(store.read_moments(_JOB), [])
        self.assertIsNone(store.read_prompt(_JOB))

    def test_find_failure_marks_stage_failed_and_reraises(self) -> None:
        store = InMemoryJobArtifactStore()

        def boom(_segs, **_kw):
            raise _FakeMomentFinderError("provider exploded")

        mf = MomentFinding(
            profile="jynxzi",
            vod_id="123",
            job_id=_JOB,
            store=store,
            params=_params(),
            find=boom,
            error=_FakeMomentFinderError,
        )

        with self.assertRaises(_FakeMomentFinderError):
            mf.find([object()])

        stage = store.read_stage(_JOB)
        self.assertEqual(stage["status"], "failed")
        self.assertIn("provider exploded", stage["error"])


if __name__ == "__main__":
    unittest.main()
