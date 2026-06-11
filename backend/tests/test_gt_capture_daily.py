import json
import pathlib
import tempfile
import unittest

from ml import gt_capture_daily


# --------------------------------------------------------------------------
# Pure decision seam: new-VOD selection diff
# --------------------------------------------------------------------------
class SelectNewVodsTests(unittest.TestCase):
    def test_excludes_already_captured_and_returns_only_new(self) -> None:
        vods = [{"id": "C"}, {"id": "B"}, {"id": "A"}]
        new = gt_capture_daily.select_new_vods(vods, {"A", "B"})

        self.assertEqual([v["id"] for v in new], ["C"])

    def test_empty_inputs_yield_nothing(self) -> None:
        self.assertEqual(gt_capture_daily.select_new_vods([], set()), [])
        self.assertEqual(gt_capture_daily.select_new_vods([], {"A"}), [])

    def test_deduplicates_keeping_first_occurrence(self) -> None:
        vods = [{"id": "C", "n": 1}, {"id": "C", "n": 2}, {"id": "D"}]
        new = gt_capture_daily.select_new_vods(vods, set())

        self.assertEqual([(v["id"], v.get("n")) for v in new], [("C", 1), ("D", None)])

    def test_skips_records_without_a_usable_id(self) -> None:
        vods = [{"id": ""}, {"id": None}, {"foo": "bar"}, {"id": "E"}]
        new = gt_capture_daily.select_new_vods(vods, set())

        self.assertEqual([v["id"] for v in new], ["E"])

    def test_already_captured_id_types_are_normalized(self) -> None:
        # Helix ids arrive as strings; persisted state may carry ints — both match.
        vods = [{"id": "123"}, {"id": "456"}]
        new = gt_capture_daily.select_new_vods(vods, {123})

        self.assertEqual([v["id"] for v in new], ["456"])


# --------------------------------------------------------------------------
# State persistence (resumable / idempotent)
# --------------------------------------------------------------------------
class CapturedStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = pathlib.Path(self._tmp.name) / "state.json"

    def test_missing_state_loads_empty(self) -> None:
        state = gt_capture_daily.load_captured_state(self.state_path)
        self.assertEqual(gt_capture_daily.captured_ids(state), set())

    def test_save_then_load_roundtrips_and_is_atomic(self) -> None:
        state = gt_capture_daily.load_captured_state(self.state_path)
        state.setdefault("captured", {})["A"] = {"vod_id": "A", "duration_sec": 10.0}
        gt_capture_daily.save_captured_state(self.state_path, state)

        reloaded = gt_capture_daily.load_captured_state(self.state_path)
        self.assertEqual(gt_capture_daily.captured_ids(reloaded), {"A"})
        self.assertEqual(reloaded["captured"]["A"]["duration_sec"], 10.0)
        # No stray temp file left behind by the atomic replace.
        self.assertFalse((self.state_path.parent / "state.json.tmp").exists())

    def test_unreadable_state_falls_back_to_empty(self) -> None:
        self.state_path.write_text("{ not json", encoding="utf-8")
        state = gt_capture_daily.load_captured_state(self.state_path)
        self.assertEqual(gt_capture_daily.captured_ids(state), set())


# --------------------------------------------------------------------------
# Daily capture orchestration (thin shell exercised through injected fakes)
# --------------------------------------------------------------------------
class _FakeCaptureResult:
    def __init__(self, vod_id, provenance) -> None:
        self.vod_id = vod_id
        self.provenance = provenance


class RunDailyCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name)
        self.state_path = root / "state.json"
        self.cache_root = root / "vod_cache"
        self.captured_calls: list[str] = []

    def _capture(self, vod_id, *, cache_dir, vod_date, captured_at):
        self.captured_calls.append(str(vod_id))
        return _FakeCaptureResult(
            str(vod_id),
            {
                "vod_id": str(vod_id),
                "vod_date": vod_date,
                "duration_sec": 10.0,
                "captured_at": captured_at,
            },
        )

    def _run(self, vods, *, capture=None, **overrides):
        kwargs = dict(
            list_vods=lambda: vods,
            state_path=self.state_path,
            cache_root=self.cache_root,
            capture=capture or self._capture,
            captured_at="2026-06-11T03:00:00+00:00",
        )
        kwargs.update(overrides)
        return gt_capture_daily.run_daily_capture(**kwargs)

    def test_captures_only_the_new_vod_and_skips_the_rest(self) -> None:
        # Demoable: a day with one new VOD captures exactly that VOD.
        gt_capture_daily.save_captured_state(
            self.state_path,
            {"captured": {"A": {"vod_id": "A"}, "B": {"vod_id": "B"}}},
        )

        summary = self._run([{"id": "C"}, {"id": "B"}, {"id": "A"}])

        self.assertEqual(self.captured_calls, ["C"])
        self.assertEqual(summary["captured"], ["C"])
        self.assertEqual(summary["skipped"], 2)
        self.assertEqual(summary["failed"], [])

    def test_records_provenance_and_is_idempotent_across_restart(self) -> None:
        self._run([{"id": "C", "created_at": "2026-06-10T20:00:00Z"}])

        # Persisted provenance is keyed by VOD id and survives a fresh load.
        state = gt_capture_daily.load_captured_state(self.state_path)
        self.assertEqual(gt_capture_daily.captured_ids(state), {"C"})
        self.assertEqual(state["captured"]["C"]["vod_id"], "C")

        # A second run (simulating tomorrow's cron) re-reads state from disk and
        # captures nothing new — idempotent.
        self.captured_calls.clear()
        summary = self._run([{"id": "C"}])
        self.assertEqual(self.captured_calls, [])
        self.assertEqual(summary["captured"], [])

    def test_state_persists_after_each_capture_so_a_crash_is_resumable(self) -> None:
        def flaky_capture(vod_id, *, cache_dir, vod_date, captured_at):
            self.captured_calls.append(str(vod_id))
            if vod_id == "D":
                raise RuntimeError("box died mid-transcribe")
            return _FakeCaptureResult(str(vod_id), {"vod_id": str(vod_id)})

        summary = self._run([{"id": "C"}, {"id": "D"}], capture=flaky_capture)

        # C committed before D blew up; D reported as a failure, not silently lost.
        self.assertEqual(summary["captured"], ["C"])
        self.assertEqual(summary["failed"], ["D"])
        self.assertEqual(
            gt_capture_daily.captured_ids(gt_capture_daily.load_captured_state(self.state_path)),
            {"C"},
        )

        # Re-run with a healthy capture: only the unfinished D is retried.
        self.captured_calls.clear()
        summary = self._run([{"id": "C"}, {"id": "D"}])
        self.assertEqual(self.captured_calls, ["D"])
        self.assertEqual(summary["captured"], ["D"])

    def test_concurrent_run_is_rejected_while_lock_is_held_by_live_pid(self) -> None:
        # A lock file naming a *live* PID (is_alive=True) means another run is in
        # progress → the second run must refuse and capture nothing.
        lock_path = self.state_path.parent / (self.state_path.name + gt_capture_daily.LOCK_SUFFIX)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("4242", encoding="utf-8")

        with self.assertRaises(gt_capture_daily.CaptureAlreadyRunning):
            self._run([{"id": "C"}], is_alive=lambda pid: True)

        self.assertEqual(self.captured_calls, [])

    def test_stale_lock_from_dead_pid_is_taken_over(self) -> None:
        # A lock left behind by a crashed/killed run (dead PID) must not wedge the
        # job forever — it is taken over and the run proceeds.
        lock_path = self.state_path.parent / (self.state_path.name + gt_capture_daily.LOCK_SUFFIX)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("4242", encoding="utf-8")

        summary = self._run([{"id": "C"}], is_alive=lambda pid: False)

        self.assertEqual(summary["captured"], ["C"])
        # The lock is released after a clean run.
        self.assertFalse(lock_path.exists())

    def test_lock_is_released_after_a_normal_run(self) -> None:
        self._run([{"id": "C"}])
        lock_path = self.state_path.parent / (self.state_path.name + gt_capture_daily.LOCK_SUFFIX)
        self.assertFalse(lock_path.exists())

    def test_passes_vod_date_and_cache_dir_per_vod_to_capture_core(self) -> None:
        seen = {}

        def recording_capture(vod_id, *, cache_dir, vod_date, captured_at):
            seen["cache_dir"] = cache_dir
            seen["vod_date"] = vod_date
            return _FakeCaptureResult(str(vod_id), {"vod_id": str(vod_id)})

        self._run(
            [{"id": "C", "created_at": "2026-06-10T20:00:00Z"}],
            capture=recording_capture,
        )

        self.assertEqual(pathlib.Path(seen["cache_dir"]), self.cache_root / "C")
        self.assertEqual(seen["vod_date"], "2026-06-10T20:00:00Z")


if __name__ == "__main__":
    unittest.main()
