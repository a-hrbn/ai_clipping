"""Compact VOD transcript artifact format — single, torch-free source of truth.

Both the legacy `build_gt_vod_transcripts` pipeline and the decoupled GT-flywheel
capture core (`gt_capture`) persist VOD transcripts in this exact shape so that
`match_cross_platform_transcripts` can read them interchangeably:

    {"metadata": {"version": 1}, "transcript": [{field: value, ...}, ...]}

Keeping the serializer here (no Whisper/torch imports) lets the capture core be
unit-tested without the ML stack installed.
"""
from __future__ import annotations

import json
import pathlib

TRANSCRIPT_CACHE_VERSION = 1
SERIALIZE_FIELDS = (
    "start",
    "end",
    "text",
    "words_per_second",
    "hype_score",
    "speaker_id",
    "word_count",
)


def serialize_segments(segments) -> list[dict]:
    """Project transcriber segments onto the compact, JSON-safe artifact rows."""
    serialized: list[dict] = []
    for seg in segments:
        row: dict = {}
        for field in SERIALIZE_FIELDS:
            value = getattr(seg, field, None)
            if value is not None:
                row[field] = value
        if row:
            serialized.append(row)
    return serialized


def write_transcript_cache(path: pathlib.Path, segments) -> None:
    """Atomically write the transcript artifact for `segments` to `path`."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"version": TRANSCRIPT_CACHE_VERSION},
        "transcript": serialize_segments(segments),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)
