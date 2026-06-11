"""ChatEnrichment — stamps Twitch-chat signal onto a transcript.

The Conductor owns the chat prefetch (a daemon thread); it hands this Producer a
``chat_source`` callable that either blocks on the prefetched result or downloads
— so the Producer never sees a thread / future / asyncio. It applies the cached
enrichment on a hit, else enriches from the downloaded chat path and saves. Pure:
``chat_source`` / ``apply_cached`` / ``enrich`` / ``save`` are injected; it never
reads settings or touches job state. When no chat is available the transcript is
returned unchanged — matching today's no-chat path.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .seam import ApplyChatEnrichmentFn, ChatSourceFn, EnrichFn, SaveChatEnrichmentFn

if TYPE_CHECKING:
    from services.transcriber import RichSegment

logger = logging.getLogger(__name__)


class ChatEnrichment:
    def __init__(
        self,
        *,
        cache_source: str | None,
        chat_source: ChatSourceFn,
        apply_cached: ApplyChatEnrichmentFn,
        enrich: EnrichFn,
        save: SaveChatEnrichmentFn,
    ) -> None:
        self._cache_source = cache_source
        self._chat_source = chat_source
        self._apply_cached = apply_cached
        self._enrich = enrich
        self._save = save

    def apply(self, transcript: list["RichSegment"]) -> list["RichSegment"]:
        """Resolve chat enrichment cheapest-first.

        Ordering:
          1. ``apply_cached`` hit → return the enriched transcript WITHOUT pulling
             ``chat_source`` (so the Conductor may cancel the still-running
             prefetch).
          2. Miss → ``chat_source`` **once**; a path → ``enrich`` then ``save``
             (only when a ``cache_source`` is set, matching the old inline guard);
             no path → the transcript is returned **unchanged**.
        """
        cached = self._apply_cached(self._cache_source, transcript)
        if cached is not None:
            return cached
        chat_path = self._chat_source()
        if chat_path is None:
            return transcript
        enriched = self._enrich(transcript, chat_path)
        if self._cache_source:
            self._save(self._cache_source, enriched)
        return enriched
