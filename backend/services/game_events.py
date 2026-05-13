from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

from config import settings


@dataclass(frozen=True)
class GameEvent:
    vod_id: str
    start: float
    end: float
    event_type: str
    confidence: float = 0.0
    source: str = "ocr"


def cache_path(vod_id: str) -> pathlib.Path:
    return pathlib.Path(settings.storage_dir) / "game_events" / f"{vod_id}.jsonl"


def load_cached_events(vod_id: str) -> list[GameEvent]:
    path = cache_path(vod_id)
    if not path.exists():
        return []
    events: list[GameEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            events.append(GameEvent(**payload))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return events


def write_cached_events(vod_id: str, events: list[GameEvent]) -> pathlib.Path:
    path = cache_path(vod_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(asdict(event), ensure_ascii=True) for event in events),
        encoding="utf-8",
    )
    return path


def detect_events(vod_id: str, *_args, **_kwargs) -> list[GameEvent]:
    """Fail-open placeholder for OCR-backed game event extraction.

    The ranker can consume cached events immediately, while environments without
    PaddleOCR or game-specific templates keep returning no events.
    """
    return load_cached_events(vod_id)
