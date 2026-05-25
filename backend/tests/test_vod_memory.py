import unittest

from services import vod_memory
from services.timeline_types import Candidate
from services.transcriber import RichSegment


class VodMemoryTests(unittest.TestCase):
    def test_build_memory_spans_detects_setup_payoff_and_callback(self) -> None:
        segments = [
            RichSegment(start=0.0, end=3.0, text="Wait, if he loses this round he owes chat five gifted.", speaker="STREAMER", hype_score=2.2),
            RichSegment(start=3.0, end=6.0, text="Everyone keeps talking about the gifted bet.", speaker="CHATTER", hype_score=1.8),
            RichSegment(start=18.0, end=22.0, text="No way, he actually lost the bet!", speaker="STREAMER", emotions=["exclamation"], hype_score=5.1, chat_spike_score=0.55),
            RichSegment(start=22.0, end=26.0, text="Chat is losing it laughing right now.", speaker="STREAMER", emotions=["laughter"], hype_score=4.8),
        ]

        spans = vod_memory.build_memory_spans(segments)
        kinds = {span.kind for span in spans}

        self.assertIn("setup_span", kinds)
        self.assertIn("payoff_span", kinds)
        self.assertIn("callback_span", kinds)

    def test_candidate_memory_context_returns_required_setup_and_scores(self) -> None:
        segments = [
            RichSegment(start=0.0, end=3.0, text="Remember the 1v1 bet if he throws this?", speaker="STREAMER", hype_score=2.4),
            RichSegment(start=3.0, end=6.0, text="Chat keeps spamming about that bet.", speaker="STREAMER", hype_score=2.0, chat_spike_score=0.25),
            RichSegment(start=20.0, end=23.0, text="No way he actually threw the round.", speaker="STREAMER", emotions=["exclamation"], hype_score=5.5, chat_spike_score=0.6),
            RichSegment(start=23.0, end=26.0, text="That means the bet pays off right now.", speaker="STREAMER", emotions=["laughter"], hype_score=4.6),
        ]
        candidate = Candidate(id="c1", start=20.0, end=26.0, source="story_payoff", anchor_time=22.0)

        spans = vod_memory.build_memory_spans(segments)
        context = vod_memory.candidate_memory_context(candidate, segments, spans)

        self.assertLess(context["minimum_required_setup_start"], candidate.start)
        self.assertGreater(context["setup_depth_score"], 0.0)
        self.assertGreater(context["retrieved_memory_span_count"], 0.0)
        self.assertIn("Setup:", context["context_summary"])

    def test_candidate_memory_context_ignores_stale_global_callbacks(self) -> None:
        segments = [
            RichSegment(start=0.0, end=3.0, text="Wait bro, remember this unrelated bet?", speaker="STREAMER", hype_score=2.4),
            RichSegment(start=400.0, end=403.0, text="No way bro, that was wild.", speaker="STREAMER", emotions=["exclamation"], hype_score=5.2),
            RichSegment(start=403.0, end=406.0, text="Everybody is reacting now.", speaker="STREAMER", emotions=["laughter"], hype_score=4.5),
        ]
        candidate = Candidate(id="late", start=400.0, end=406.0, source="dense_inventory", anchor_time=402.0)

        spans = vod_memory.build_memory_spans(segments)
        context = vod_memory.candidate_memory_context(candidate, segments, spans)

        self.assertIsNone(context["minimum_required_setup_start"])
        self.assertEqual(context["required_setup_lead_sec"], 0.0)
        self.assertNotIn("Callback:", context["context_summary"])
        self.assertNotIn("unrelated bet", context["context_summary"])


if __name__ == "__main__":
    unittest.main()
