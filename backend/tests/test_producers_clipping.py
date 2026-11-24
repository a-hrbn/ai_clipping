import types
import unittest
from pathlib import Path

from producers.clipping import Clipping
from services.job_artifact_store import InMemoryJobArtifactStore

_JOB = "job"

_ROWS = [
    {"start_time": 0.0, "end_time": 5.0, "viral_score": 9.0, "reason": "a"},
    {"start_time": 5.0, "end_time": 10.0, "viral_score": 8.0, "reason": "b"},
    {"start_time": 10.0, "end_time": 15.0, "viral_score": 7.0, "reason": "c"},
]


def _result(rows: list) -> types.SimpleNamespace:
    return types.SimpleNamespace(post_processed_moments=rows)


def _echo_cut(_video_path, moments, _job_id, *, progress_cb=None):
    """Fake CutClipsFn: returns one stub clip per moment it was handed."""
    return [
        types.SimpleNamespace(
            candidate_id=m["candidate_id"], start_time=m["start_time"], end_time=m["end_time"]
        )
        for m in moments
    ]


class ClippingTests(unittest.TestCase):
    def test_top_k_truncates_and_writes_final_moments(self) -> None:
        # final_moments is asserted in-memory through the injected fake store, never
        # a tmp directory: the producer owns no filename, only the artifact role.
        store = InMemoryJobArtifactStore()
        cut_calls: list = []

        def cut(video_path, moments, job_id, *, progress_cb=None):
            cut_calls.append(moments)
            return _echo_cut(video_path, moments, job_id)

        clipping = Clipping(
            job_id=_JOB, store=store, top_k=2, platform="tiktok", cut=cut
        )

        out = clipping.run(Path("/v.mp4"), _result(_ROWS))

        final = store.read_final_moments(_JOB)
        self.assertEqual(final["count"], 2)  # top-2 of 3
        self.assertEqual(len(final["moments"]), 2)
        self.assertEqual(final["moments"][0]["candidate_id"], "llm_moment_01")
        self.assertEqual(final["moments"][1]["candidate_id"], "llm_moment_02")
        self.assertEqual(final["moments"][0]["platform"], "tiktok")
        self.assertEqual(final["moments"][0]["ranking_mode"], "llm_moment_finder")
        self.assertEqual(len(cut_calls[0]), 2)  # cutter only sees the top-2
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].candidate_id, "llm_moment_01")

    def test_top_k_zero_keeps_all_moments(self) -> None:
        store = InMemoryJobArtifactStore()
        clipping = Clipping(
            job_id=_JOB, store=store, top_k=0, platform="general", cut=_echo_cut
        )

        out = clipping.run(Path("/v.mp4"), _result(_ROWS))

        final = store.read_final_moments(_JOB)
        self.assertEqual(final["count"], 3)  # 0 → no truncation
        self.assertEqual(len(out), 3)

    def test_progress_maps_cutter_callback_to_fraction(self) -> None:
        store = InMemoryJobArtifactStore()
        fractions: list = []

        def fake_cut(_video_path, moments, _job_id, *, progress_cb=None):
            progress_cb([], 1, 4)
            progress_cb([], 4, 4)
            return []

        clipping = Clipping(
            job_id=_JOB, store=store, top_k=0, platform="general", cut=fake_cut
        )

        clipping.run(Path("/v.mp4"), _result(_ROWS), progress=lambda f: fractions.append(f))

        self.assertIn(0.25, fractions)  # done=1/total=4
        self.assertIn(1.0, fractions)  # done=4/total=4

    def test_never_reads_settings_top_k_and_platform_are_injected(self) -> None:
        # purity tripwire: a non-default platform/top_k passed in must surface in
        # final_moments.json without any config.settings lookup.
        store = InMemoryJobArtifactStore()
        clipping = Clipping(
            job_id=_JOB, store=store, top_k=1, platform="youtube_shorts", cut=_echo_cut
        )

        clipping.run(Path("/v.mp4"), _result(_ROWS))

        final = store.read_final_moments(_JOB)
        self.assertEqual(final["count"], 1)
        self.assertEqual(final["moments"][0]["platform"], "youtube_shorts")


if __name__ == "__main__":
    unittest.main()
