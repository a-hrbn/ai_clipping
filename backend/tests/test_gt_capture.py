import json
import pathlib
import tempfile
import unittest

from ml import gt_capture
from ml import gt_transcript_artifact
from ml import match_cross_platform_audio_fingerprints as audio_matcher


class _Segment:
    """Minimal stand-in for transcriber.RichSegment (attribute access only)."""

    def __init__(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


def _full_duration_values(duration_sec: float, rate: int = 8) -> list[int]:
    return list(range(int(duration_sec * rate)))


# --------------------------------------------------------------------------
# Pure decision seam: full-duration fingerprint guard
# --------------------------------------------------------------------------
class ValidateFullDurationFingerprintTests(unittest.TestCase):
    def test_rejects_prefix_only_fingerprint(self) -> None:
        # 10 items for a 3600s VOD == fpcalc only covered its ~120s prefix.
        with self.assertRaises(ValueError):
            gt_capture.validate_full_duration_fingerprint(10, 3600.0)

    def test_accepts_full_duration_fingerprint(self) -> None:
        # ~8 items/sec over the whole VOD is the healthy Chromaprint rate.
        gt_capture.validate_full_duration_fingerprint(int(3600 * 8), 3600.0)

    def test_zero_duration_is_a_noop(self) -> None:
        # Unknown/zero duration must not divide-by-zero or falsely reject.
        gt_capture.validate_full_duration_fingerprint(0, 0.0)


# --------------------------------------------------------------------------
# Pure decision seam: provenance record
# --------------------------------------------------------------------------
class BuildProvenanceTests(unittest.TestCase):
    def test_records_id_date_duration_and_captured_at(self) -> None:
        provenance = gt_capture.build_provenance(
            "2468013579", "2026-06-01", 3600.5, "2026-06-02T03:00:00+00:00"
        )

        self.assertEqual(provenance["vod_id"], "2468013579")
        self.assertEqual(provenance["vod_date"], "2026-06-01")
        self.assertEqual(provenance["duration_sec"], 3600.5)
        self.assertEqual(provenance["captured_at"], "2026-06-02T03:00:00+00:00")


# --------------------------------------------------------------------------
# Shared transcript artifact serializer (torch-free single source of truth)
# --------------------------------------------------------------------------
class TranscriptArtifactTests(unittest.TestCase):
    def test_serialize_keeps_known_fields_and_drops_none(self) -> None:
        rows = gt_transcript_artifact.serialize_segments(
            [_Segment(start=0.0, end=2.0, text="hello", hype_score=None)]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "hello")
        self.assertNotIn("hype_score", rows[0])  # None values are pruned

    def test_write_transcript_cache_roundtrips_and_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "transcript.json"
            gt_transcript_artifact.write_transcript_cache(
                path, [_Segment(start=0.0, end=1.0, text="hi", word_count=1)]
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["metadata"]["version"],
                gt_transcript_artifact.TRANSCRIPT_CACHE_VERSION,
            )
            self.assertEqual(payload["transcript"][0]["text"], "hi")
            # No stray temp file left behind by the atomic replace.
            self.assertFalse((pathlib.Path(tmp) / "transcript.tmp").exists())


# --------------------------------------------------------------------------
# Capture core orchestration (thin shell exercised through injected fakes)
# --------------------------------------------------------------------------
class CaptureVodTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name)
        self.cache_dir = root / "vod_cache" / "2468013579"
        self.media_dir = root / "downloads"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.media_dir / "source.mp4"
        self.duration = 12.0
        self.calls: list[str] = []
        self.fingerprint_present_at_transcribe = None

    # -- fakes -----------------------------------------------------------
    def _download(self, vod_url, job_id):
        self.calls.append("download")
        self.video_path.write_bytes(b"fake-video-bytes")
        return self.video_path

    def _extract_audio(self, video_path, job_id, output_path):
        self.calls.append("extract_audio")
        output_path.write_bytes(b"fake-audio-bytes")
        return output_path

    def _fingerprint_to_file(self, media_path, output_path, fpcalc, *, values=None):
        self.calls.append("fingerprint")
        vals = values if values is not None else _full_duration_values(self.duration)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"duration": self.duration, "fingerprint": vals}),
            encoding="utf-8",
        )
        return audio_matcher.Fingerprint(duration=self.duration, values=vals)

    def _transcribe(self, audio_path, vod_id):
        self.calls.append("transcribe")
        self.fingerprint_present_at_transcribe = (
            self.cache_dir / "fingerprint.json"
        ).exists()
        return [_Segment(start=0.0, end=2.0, text="clip moment", word_count=2)]

    def _capture(self, **overrides):
        kwargs = dict(
            vod_date="2026-06-01",
            cache_dir=self.cache_dir,
            captured_at="2026-06-02T03:00:00+00:00",
            fpcalc="/usr/bin/fpcalc",
            download=self._download,
            extract_audio=self._extract_audio,
            fingerprint_to_file=self._fingerprint_to_file,
            transcribe=self._transcribe,
        )
        kwargs.update(overrides)
        return gt_capture.capture_vod("2468013579", **kwargs)

    # -- tests -----------------------------------------------------------
    def test_persists_artifacts_with_provenance_and_deletes_media(self) -> None:
        result = self._capture()

        fingerprint_path = self.cache_dir / "fingerprint.json"
        transcript_path = self.cache_dir / "transcript.json"
        provenance_path = self.cache_dir / "provenance.json"

        self.assertTrue(fingerprint_path.exists())
        self.assertTrue(transcript_path.exists())
        self.assertTrue(provenance_path.exists())

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["vod_id"], "2468013579")
        self.assertEqual(provenance["vod_date"], "2026-06-01")
        self.assertEqual(provenance["duration_sec"], self.duration)
        self.assertEqual(provenance["captured_at"], "2026-06-02T03:00:00+00:00")

        # Source media is gone; compact artifacts remain.
        self.assertFalse(self.video_path.exists())
        self.assertFalse((self.cache_dir / "audio.wav").exists())

        self.assertEqual(result.vod_id, "2468013579")
        self.assertEqual(result.transcript_segments, 1)
        self.assertEqual(result.fingerprint_items, int(self.duration * 8))

    def test_fingerprint_persisted_before_transcription_begins(self) -> None:
        self._capture()

        # Observable ordering: the fingerprint artifact already existed on disk
        # by the time transcription started, and fingerprint ran first.
        self.assertTrue(self.fingerprint_present_at_transcribe)
        self.assertLess(self.calls.index("fingerprint"), self.calls.index("transcribe"))

    def test_prefix_only_fingerprint_is_rejected_before_transcription(self) -> None:
        prefix = lambda media_path, output_path, fpcalc: self._fingerprint_to_file(
            media_path, output_path, fpcalc, values=[1, 2, 3]
        )

        with self.assertRaises(ValueError):
            self._capture(fingerprint_to_file=prefix)

        # Transcription never ran, and the corrupt prefix-only fingerprint was
        # not left behind as a usable artifact.
        self.assertNotIn("transcribe", self.calls)
        self.assertFalse((self.cache_dir / "transcript.json").exists())
        self.assertFalse((self.cache_dir / "fingerprint.json").exists())


if __name__ == "__main__":
    unittest.main()
