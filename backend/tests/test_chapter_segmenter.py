import sys
import types
import unittest

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import chapter_segmenter
from services.transcriber import RichSegment


class ChapterSegmenterTests(unittest.TestCase):
    def test_segment_chapters_splits_on_strong_topic_breaks(self) -> None:
        segments = [
            RichSegment(start=0.0, end=10.0, text="we are setting up the match", speaker="STREAMER"),
            RichSegment(start=10.0, end=20.0, text="the round is getting intense", speaker="STREAMER", chat_spike_score=0.2),
            RichSegment(start=20.0, end=30.0, text="huge reaction from everyone", speaker="STREAMER", has_scene_change=True, chat_spike_score=0.8),
            RichSegment(start=35.0, end=45.0, text="now let me tell you a personal story about school", speaker="GUEST_1"),
            RichSegment(start=45.0, end=55.0, text="it completely changed how i think about this", speaker="GUEST_1"),
            RichSegment(start=55.0, end=65.0, text="and that is why i still remember it", speaker="GUEST_1"),
        ]

        chapters = chapter_segmenter.segment_chapters(segments)

        self.assertEqual(len(chapters), 2)
        self.assertLess(chapters[0].end, chapters[1].start + 20.0)
        self.assertTrue(all(chapter.start < chapter.end for chapter in chapters))

    def test_segments_for_chapter_returns_only_overlapping_segments(self) -> None:
        segments = [
            RichSegment(start=0.0, end=10.0, text="a"),
            RichSegment(start=10.0, end=20.0, text="b"),
            RichSegment(start=20.0, end=30.0, text="c"),
        ]
        chapter = chapter_segmenter.Chapter(id="chapter_00", start=9.0, end=21.0, index=0)

        sliced = chapter_segmenter.segments_for_chapter(segments, chapter)

        self.assertEqual([seg.text for seg in sliced], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
