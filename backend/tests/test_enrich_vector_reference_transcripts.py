import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import enrich_vector_reference_transcripts
from services.transcriber import RichSegment


class EnrichVectorReferenceTranscriptTests(unittest.TestCase):
    def test_build_transcript_fields_splits_hook_and_payoff(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="first hook line", hype_score=1.0),
            RichSegment(start=2.0, end=5.0, text="second hook line", hype_score=1.5),
            RichSegment(start=5.0, end=8.0, text="big payoff happens", hype_score=6.0),
            RichSegment(start=8.0, end=10.0, text="reaction after payoff", hype_score=4.0),
        ]

        fields = enrich_vector_reference_transcripts.build_transcript_fields(segments)

        self.assertIn("first hook line", fields["hook_text"])
        self.assertIn("big payoff happens", fields["payoff_text"])
        self.assertIn("reaction after payoff", fields["payoff_text"])
        self.assertIn("second hook line", fields["transcript"])

    def test_enrich_reference_row_merges_transcript_fields(self) -> None:
        row = {"source_url": "https://example.com/clip", "candidate_id": "c1"}
        fake_segments = [
            RichSegment(start=0.0, end=2.0, text="hook", hype_score=1.0),
            RichSegment(start=2.0, end=4.0, text="payoff", hype_score=5.0),
        ]
        with TemporaryDirectory() as tmp, patch(
            "ml.enrich_vector_reference_transcripts._transcribe_reference_clip",
            return_value=fake_segments,
        ):
            enriched = enrich_vector_reference_transcripts.enrich_reference_row(
                row,
                Path(tmp),
                "jynxzi",
            )

        self.assertIsNotNone(enriched)
        assert enriched is not None
        self.assertEqual(enriched["transcript"], "hook payoff")
        self.assertEqual(enriched["transcript_segment_count"], 2)


if __name__ == "__main__":
    unittest.main()
