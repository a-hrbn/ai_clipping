"""Unit tests for services/chat_signal.py (no network, no torch required)."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from services.chat_signal import (
    _build_per_second_counts,
    _compute_spike_scores,
    _dominant_category,
    _extract_vod_id,
    _parse_comment_node,
)


class TestExtractVodId(unittest.TestCase):
    def test_twitch_url(self):
        self.assertEqual(_extract_vod_id("https://www.twitch.tv/videos/2777621798"), "2777621798")

    def test_twitch_video_url(self):
        self.assertEqual(_extract_vod_id("https://twitch.tv/video/987654321"), "987654321")

    def test_non_twitch(self):
        self.assertIsNone(_extract_vod_id("https://youtube.com/watch?v=abc"))

    def test_empty(self):
        self.assertIsNone(_extract_vod_id(""))


class TestPerSecondCounts(unittest.TestCase):
    def test_basic(self):
        msgs = [{"t": 10, "text": "a"}, {"t": 10, "text": "b"}, {"t": 11, "text": "c"}]
        counts = _build_per_second_counts(msgs)
        self.assertEqual(counts[10], 2)
        self.assertEqual(counts[11], 1)

    def test_missing_t(self):
        msgs = [{"text": "no time"}, {"t": 5, "text": "ok"}]
        counts = _build_per_second_counts(msgs)
        self.assertEqual(counts.get(5), 1)
        self.assertNotIn(None, counts)


class TestSpikeScores(unittest.TestCase):
    def test_spike_at_burst(self):
        per_sec = {100: 10, 101: 2, 102: 2, 103: 1, 104: 1, 105: 1}
        scores = _compute_spike_scores(per_sec)
        self.assertGreater(scores[100], 0.5)

    def test_flat_baseline_no_spike(self):
        per_sec = {t: 2 for t in range(100, 130)}
        scores = _compute_spike_scores(per_sec)
        for v in scores.values():
            self.assertLessEqual(v, 0.3)

    def test_saturation_at_one(self):
        per_sec = {t: 1 for t in range(50)}
        per_sec[25] = 1000  # extreme burst
        scores = _compute_spike_scores(per_sec)
        self.assertAlmostEqual(scores[25], 1.0)


class TestDominantCategory(unittest.TestCase):
    def _make_msgs(self, texts, start=0):
        return [{"t": start + i, "text": t, "emotes": []} for i, t in enumerate(texts)]

    def test_hype_wins(self):
        msgs = self._make_msgs(["pog", "omg", "lol"], start=100)
        cat = _dominant_category(msgs, 99, 104)
        self.assertEqual(cat, "hype")

    def test_funny_wins(self):
        msgs = self._make_msgs(["lol", "lmao", "pog"], start=200)
        cat = _dominant_category(msgs, 199, 204)
        self.assertEqual(cat, "funny")

    def test_single_match_returns_empty(self):
        msgs = self._make_msgs(["pog", "random stuff"], start=300)
        cat = _dominant_category(msgs, 299, 303)
        self.assertEqual(cat, "")

    def test_no_messages_in_window(self):
        msgs = [{"t": 999, "text": "pog", "emotes": []}]
        cat = _dominant_category(msgs, 0, 10)
        self.assertEqual(cat, "")


class TestParseCommentNode(unittest.TestCase):
    def test_basic_node(self):
        node = {
            "contentOffsetSeconds": 42,
            "message": {
                "fragments": [
                    {"text": "hello", "emote": None},
                    {"text": "PogChamp", "emote": {"emoteID": "305954156"}},
                ]
            },
        }
        result = _parse_comment_node(node)
        self.assertIsNotNone(result)
        self.assertEqual(result["t"], 42)
        self.assertIn("PogChamp", result.get("emotes", []))

    def test_missing_offset_returns_none(self):
        node = {"message": {"fragments": [{"text": "hi"}]}}
        self.assertIsNone(_parse_comment_node(node))


class _FakeField:
    def __init__(self, name):
        self.name = name


class _FakeRichSegment:
    """Minimal stand-in for RichSegment in enrich_segments tests (no torch required)."""

    _FIELDS = ["start", "end", "text", "speaker", "emotions", "words_per_second",
               "chat_spike_score", "chat_message_count", "chat_category"]

    __dataclass_fields__ = {f: _FakeField(f) for f in _FIELDS}

    def __init__(self, **kwargs):
        for f in self._FIELDS:
            setattr(self, f, kwargs.get(f, "" if f in ("text", "speaker", "chat_category") else
                                        [] if f == "emotions" else
                                        False if f == "has_scene_change" else 0))


def _make_seg(start, end, text="hello"):
    return _FakeRichSegment(
        start=start, end=end, text=text, speaker="UNKNOWN", emotions=[],
        words_per_second=0.0, chat_spike_score=0.0, chat_message_count=0, chat_category="",
    )


class TestEnrichSegments(unittest.TestCase):
    """Smoke-tests for enrich_segments using _FakeRichSegment (avoids importing torch)."""

    def test_returns_unchanged_on_no_chat(self):
        from services.chat_signal import enrich_segments
        seg = _make_seg(0.0, 5.0)
        result = enrich_segments([seg], None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chat_spike_score, 0.0)

    def test_enriches_spike_segment(self):
        from services.chat_signal import enrich_segments
        msgs = [{"t": i, "text": "pog"} for i in range(1, 11)]  # 10 msg in first 10s
        msgs += [{"t": i, "text": "x"} for i in range(60, 120)]  # baseline ~1/s

        with tempfile.TemporaryDirectory() as tmp:
            chat_path = pathlib.Path(tmp) / "chat.json"
            chat_path.write_text(json.dumps(msgs), encoding="utf-8")
            seg = _make_seg(0.0, 12.0)
            result = enrich_segments([seg], chat_path)
        self.assertGreater(result[0].chat_spike_score, 0.0)
        self.assertGreaterEqual(result[0].chat_message_count, 1)

    def test_empty_chat_file_returns_segments_unchanged(self):
        from services.chat_signal import enrich_segments
        with tempfile.TemporaryDirectory() as tmp:
            chat_path = pathlib.Path(tmp) / "chat.json"
            chat_path.write_text("[]", encoding="utf-8")
            seg = _make_seg(0.0, 5.0)
            result = enrich_segments([seg], chat_path)
        self.assertEqual(result[0].chat_spike_score, 0.0)


if __name__ == "__main__":
    unittest.main()
