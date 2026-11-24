import json
import pathlib
import random
import tempfile
import unittest
from unittest.mock import patch

from ml import match_cross_platform_audio_fingerprints as audio_matcher
from ml import verify_gt_matches as verifier


def _pseudo_random_values(count: int, seed: int = 7) -> list[int]:
    """Random walk over 32-bit values: like real chromaprint output, adjacent
    items differ by a few bits while distant items are decorrelated."""
    rng = random.Random(seed)
    values = [rng.getrandbits(32)]
    for _ in range(count - 1):
        values.append(values[-1] ^ (1 << rng.randrange(32)) ^ (1 << rng.randrange(32)))
    return values


def _full_vod_fingerprint(duration: float = 1000.0, rate: int = 8) -> audio_matcher.Fingerprint:
    return audio_matcher.Fingerprint(duration=duration, values=_pseudo_random_values(int(duration * rate)))


class VerifyGtMatchesHelperTests(unittest.TestCase):
    def test_slice_fingerprint_values_maps_seconds_to_items(self) -> None:
        fp = _full_vod_fingerprint(duration=100.0, rate=8)

        values, actual_start = verifier.slice_fingerprint_values(fp, 10.0, 20.0)

        self.assertEqual(len(values), 80)
        self.assertEqual(values, fp.values[80:160])
        self.assertEqual(actual_start, 10.0)

    def test_null_window_start_avoids_claimed_scope(self) -> None:
        start = verifier.null_window_start(400.0, 600.0, 10_000.0, "vod:400.00:600.00")

        scope_len = 200.0
        self.assertTrue(start + scope_len <= 400.0 or start >= 600.0)

    def test_is_confirmed_combines_margin_and_position_agreement(self) -> None:
        self.assertTrue(verifier.is_confirmed(0.65, 0.59, 1.0))    # small margin, exact position
        self.assertTrue(verifier.is_confirmed(0.69, 0.58, 8.0))    # solid margin, near position
        self.assertFalse(verifier.is_confirmed(0.65, 0.64, 1.0))   # no margin over null
        self.assertFalse(verifier.is_confirmed(0.69, 0.58, 40.0))  # alignment far from claim
        self.assertFalse(verifier.is_confirmed(0.54, 0.30, 0.0))   # below absolute floor

    def test_is_confirmed_tolerates_positive_range_download_bias(self) -> None:
        self.assertTrue(verifier.is_confirmed(0.69, 0.58, 18.0, boundary_basis="range_approx"))
        self.assertFalse(verifier.is_confirmed(0.69, 0.58, -18.0, boundary_basis="range_approx"))

    def test_assign_tier_policy(self) -> None:
        self.assertEqual(verifier.assign_tier(available=False, confirmed=False, match_similarity=0.9), "unverifiable")
        self.assertEqual(verifier.assign_tier(available=True, confirmed=True, match_similarity=0.3), "gold")
        self.assertEqual(verifier.assign_tier(available=True, confirmed=False, match_similarity=0.70), "silver")
        self.assertEqual(verifier.assign_tier(available=True, confirmed=False, match_similarity=0.55), "review")


class VerifyMomentTests(unittest.TestCase):
    def test_verify_moment_confirms_and_refines_true_position(self) -> None:
        vod_fp = _full_vod_fingerprint(duration=1000.0, rate=8)
        # The repost truly lives at 500.0s–520.0s in the VOD.
        repost_fp = audio_matcher.Fingerprint(duration=20.0, values=vod_fp.values[4000:4160])
        moment = {"start": 495.0, "end": 521.0, "source_url": "https://youtu.be/x", "match_similarity": 0.6}

        with patch.object(verifier, "_repost_fingerprint", return_value=repost_fp):
            row = verifier.verify_moment(
                "123", moment, fpcalc="/usr/bin/fpcalc", full_vod_fp=vod_fp, null_source_fp=None,
            )

        self.assertEqual(row["verdict"], "confirmed")
        self.assertEqual(row["gt_tier"], "gold")
        self.assertEqual(row["refined_start"], 500.0)
        self.assertEqual(row["refined_end"], 520.0)
        self.assertEqual(row["boundary_basis"], "full_fingerprint")
        self.assertGreater(row["confirm_confidence"], row["null_confidence"] + 0.15)

    def test_verify_moment_rejects_wrong_claimed_position(self) -> None:
        vod_fp = _full_vod_fingerprint(duration=1000.0, rate=8)
        repost_fp = audio_matcher.Fingerprint(duration=20.0, values=vod_fp.values[4000:4160])
        # Claim says 100s although the repost audio lives at 500s.
        moment = {"start": 100.0, "end": 126.0, "source_url": "https://youtu.be/x", "match_similarity": 0.70}

        with patch.object(verifier, "_repost_fingerprint", return_value=repost_fp):
            row = verifier.verify_moment(
                "123", moment, fpcalc="/usr/bin/fpcalc", full_vod_fp=vod_fp, null_source_fp=None,
            )

        self.assertEqual(row["verdict"], "not_confirmed")
        self.assertEqual(row["gt_tier"], "silver")

    def test_verify_moment_marks_unavailable_repost(self) -> None:
        vod_fp = _full_vod_fingerprint(duration=1000.0, rate=8)
        moment = {"start": 10.0, "end": 30.0, "source_url": "https://youtu.be/gone", "match_similarity": 0.8}

        with patch.object(verifier, "_repost_fingerprint", side_effect=RuntimeError("HTTP 404")):
            row = verifier.verify_moment(
                "123", moment, fpcalc="/usr/bin/fpcalc", full_vod_fp=vod_fp, null_source_fp=None,
            )

        self.assertEqual(row["verdict"], "repost_unavailable")
        self.assertEqual(row["gt_tier"], "unverifiable")


class ApplyTiersTests(unittest.TestCase):
    def test_apply_tiers_writes_tiers_and_refines_gold_boundaries(self) -> None:
        manifest_row = {
            "vod_id": "123",
            "gt_moments": [
                {"start": 495.0, "end": 521.0, "source_url": "https://youtu.be/a", "match_similarity": 0.6},
                {"start": 100.0, "end": 126.0, "source_url": "https://youtu.be/b", "match_similarity": 0.5},
            ],
        }
        verification_rows = [
            {
                "key": "123:495.00:521.00", "gt_tier": "gold", "boundary_basis": "full_fingerprint",
                "refined_start": 500.0, "refined_end": 520.0,
            },
            {"key": "123:100.00:126.00", "gt_tier": "review"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = pathlib.Path(tmp) / "manifest.jsonl"
            manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

            result = verifier.apply_tiers(manifest_path, verification_rows, apply_boundaries=True)
            updated = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(result, {"tiers_applied": 2, "boundaries_applied": 1})
        gold, review = updated["gt_moments"]
        self.assertEqual(gold["gt_tier"], "gold")
        self.assertEqual(gold["start"], 500.0)
        self.assertEqual(gold["end"], 520.0)
        self.assertEqual(gold["transcript_start"], 495.0)
        self.assertEqual(gold["boundary_source"], "audio_refined")
        self.assertEqual(review["gt_tier"], "review")
        self.assertEqual(review["start"], 100.0)
        self.assertNotIn("boundary_source", review)

    def test_second_boundary_run_keeps_transcript_window_of_refined_moment(self) -> None:
        # A second verification run re-verifies the already refined window
        # (its key changed with start/end) and confirms it as gold again.
        first_pass = [{
            "key": "123:495.00:521.00", "gt_tier": "gold", "boundary_basis": "full_fingerprint",
            "refined_start": 500.0, "refined_end": 520.0,
        }]
        second_pass = first_pass + [{
            "key": "123:500.00:520.00", "gt_tier": "gold", "boundary_basis": "full_fingerprint",
            "refined_start": 500.12, "refined_end": 520.12,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = pathlib.Path(tmp) / "manifest.jsonl"
            manifest_path.write_text(json.dumps({
                "vod_id": "123",
                "gt_moments": [{"start": 495.0, "end": 521.0, "source_url": "https://youtu.be/a"}],
            }) + "\n", encoding="utf-8")

            verifier.apply_tiers(manifest_path, first_pass, apply_boundaries=True)
            result = verifier.apply_tiers(manifest_path, second_pass, apply_boundaries=True)
            moment = json.loads(manifest_path.read_text(encoding="utf-8"))["gt_moments"][0]

        self.assertEqual(result["boundaries_applied"], 0)
        self.assertEqual((moment["start"], moment["end"]), (500.0, 520.0))
        self.assertEqual((moment["transcript_start"], moment["transcript_end"]), (495.0, 521.0))
        self.assertEqual(moment["boundary_source"], "audio_refined")

    def test_malformed_manifest_line_aborts_instead_of_being_dropped(self) -> None:
        good = json.dumps({"vod_id": "123", "gt_moments": [{"start": 495.0, "end": 521.0}]})
        original = good + "\n" + '{"vod_id": "456", "gt_moments": [' + "\n"
        verification_rows = [{"key": "123:495.00:521.00", "gt_tier": "gold"}]
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = pathlib.Path(tmp) / "manifest.jsonl"
            manifest_path.write_text(original, encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                verifier.apply_tiers(manifest_path, verification_rows, apply_boundaries=False)
            after = manifest_path.read_text(encoding="utf-8")

        self.assertIn(f"{manifest_path}:2", str(caught.exception))
        self.assertEqual(after, original)


class MainVerificationFileTests(unittest.TestCase):
    def test_rejudge_aborts_on_malformed_verification_line_without_rewriting(self) -> None:
        manifest = json.dumps({"vod_id": "123", "gt_moments": [{"start": 495.0, "end": 521.0}]}) + "\n"
        verification = json.dumps({"key": "123:495.00:521.00", "verdict": "not_confirmed"}) + "\n" + '{"key": "123:1' + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = pathlib.Path(tmp) / "manifest.jsonl"
            output_path = pathlib.Path(tmp) / "verification.jsonl"
            manifest_path.write_text(manifest, encoding="utf-8")
            output_path.write_text(verification, encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                verifier.main(["--manifest", str(manifest_path), "--output", str(output_path), "--rejudge"])
            after = output_path.read_text(encoding="utf-8")

        self.assertIn(f"{output_path}:2", str(caught.exception))
        self.assertEqual(after, verification)

    def test_rejudge_rewrites_verification_rows_under_current_rule(self) -> None:
        manifest = json.dumps({"vod_id": "123", "gt_moments": [{"start": 495.0, "end": 521.0}]}) + "\n"
        stale = {
            "key": "123:495.00:521.00", "verdict": "not_confirmed", "gt_tier": "review",
            "confirm_confidence": 0.9, "null_confidence": 0.5, "refined_offset_sec": 1.0,
            "boundary_basis": "full_fingerprint", "match_similarity": 0.4,
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = pathlib.Path(tmp) / "manifest.jsonl"
            output_path = pathlib.Path(tmp) / "verification.jsonl"
            manifest_path.write_text(manifest, encoding="utf-8")
            output_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")

            with patch.object(verifier.shutil, "which", return_value=None):
                with self.assertRaises(RuntimeError):  # stops at the fpcalc check, after rejudging
                    verifier.main(["--manifest", str(manifest_path), "--output", str(output_path), "--rejudge"])
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([(r["verdict"], r["gt_tier"]) for r in rows], [("confirmed", "gold")])


if __name__ == "__main__":
    unittest.main()
