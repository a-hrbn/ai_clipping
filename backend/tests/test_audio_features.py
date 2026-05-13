import unittest

from services import audio_features
from services.transcriber import RichSegment


class AudioFeatureEnrichmentTests(unittest.TestCase):
    def test_enrich_segments_adds_voice_and_f0_stats(self) -> None:
        frames = [
            audio_features.AudioFeatureFrame(second=0, pitch_hz=180.0, rms_db=-18.0, is_silence=False, laughter_prob=0.0, onset_rate=3.0),
            audio_features.AudioFeatureFrame(second=1, pitch_hz=420.0, rms_db=-9.0, is_silence=False, laughter_prob=0.0, onset_rate=8.0),
            audio_features.AudioFeatureFrame(second=2, pitch_hz=0.0, rms_db=-40.0, is_silence=True, laughter_prob=0.0, onset_rate=0.0),
        ]
        segments = [RichSegment(start=0.0, end=2.0, text="NO WAY", speaker="STREAMER")]

        enriched = audio_features.enrich_segments(segments, frames)

        self.assertGreater(enriched[0].pitch_std_hz, 0.0)
        self.assertGreater(enriched[0].voice_activity_score, 0.0)
        self.assertGreater(enriched[0].rms_burst_score, 0.0)


if __name__ == "__main__":
    unittest.main()
