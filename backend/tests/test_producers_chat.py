import dataclasses
import unittest
from pathlib import Path

from producers.chat import ChatEnrichment

CHAT = Path("/jobs/job/chat.json")


@dataclasses.dataclass
class _Seg:
    """A torch-free stand-in for RichSegment — ChatEnrichment only treats the
    transcript as an opaque list it hands to the injected leaf ops."""

    start: float
    text: str = ""


def _segs() -> list[_Seg]:
    return [_Seg(start=0.0, text="a"), _Seg(start=2.0, text="b")]


class _CountingSource:
    """Call-counting ChatSourceFn so the at-most-once assertion can bite."""

    def __init__(self, result: Path | None = CHAT) -> None:
        self._result = result
        self.calls = 0

    def __call__(self) -> Path | None:
        self.calls += 1
        return self._result


class ChatEnrichmentTests(unittest.TestCase):
    def test_cache_hit_returns_enriched_without_pulling_chat_source(self) -> None:
        enriched = _segs()
        source = _CountingSource()
        ce = ChatEnrichment(
            cache_source="twitch_vod:1",
            chat_source=source,
            apply_cached=lambda _src, _segs: enriched,
            enrich=lambda *_a: self.fail("must not enrich on a cache hit"),
            save=lambda *_a: self.fail("must not save on a cache hit"),
        )

        out = ce.apply(_segs())

        self.assertIs(out, enriched)  # the cached enrichment, verbatim
        self.assertEqual(source.calls, 0)  # hit → chat_source (prefetch) never pulled

    def test_miss_downloads_enriches_and_saves(self) -> None:
        source = _CountingSource(CHAT)
        enriched = _segs()
        enrich_args: list = []
        saved: list = []
        ce = ChatEnrichment(
            cache_source="twitch_vod:1",
            chat_source=source,
            apply_cached=lambda _src, _segs: None,
            enrich=lambda segs, path: enrich_args.append((segs, path)) or enriched,
            save=lambda src, segs: saved.append((src, segs)),
        )
        segs = _segs()

        out = ce.apply(segs)

        self.assertIs(out, enriched)
        self.assertEqual(source.calls, 1)  # at most once
        self.assertEqual(enrich_args, [(segs, CHAT)])
        self.assertEqual(saved, [("twitch_vod:1", enriched)])

    def test_miss_without_chat_path_returns_transcript_unchanged(self) -> None:
        source = _CountingSource(None)  # prefetch/download yielded nothing
        segs = _segs()
        ce = ChatEnrichment(
            cache_source="twitch_vod:1",
            chat_source=source,
            apply_cached=lambda _src, _segs: None,
            enrich=lambda *_a: self.fail("must not enrich without a chat path"),
            save=lambda *_a: self.fail("must not save without a chat path"),
        )

        out = ce.apply(segs)

        self.assertIs(out, segs)  # returned unchanged, not rebuilt
        self.assertEqual(source.calls, 1)

    def test_miss_without_cache_source_enriches_but_does_not_save(self) -> None:
        # A path-only run (no transcript cache) still enriches, but there is no
        # cache key to persist under — matches the `if transcript_cache_source`
        # guard in the old inline block.
        enriched = _segs()
        ce = ChatEnrichment(
            cache_source=None,
            chat_source=_CountingSource(CHAT),
            apply_cached=lambda _src, _segs: None,
            enrich=lambda _segs, _path: enriched,
            save=lambda *_a: self.fail("must not save without a cache_source"),
        )

        out = ce.apply(_segs())

        self.assertIs(out, enriched)


if __name__ == "__main__":
    unittest.main()
