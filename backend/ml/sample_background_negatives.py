from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import window_overlap_ratio  # noqa: E402


@dataclass(frozen=True)
class BackgroundWindow:
    start: float
    end: float
    source: str = "background_negative"


def sample_background_windows(
    vod_id: str,
    *,
    duration_seconds: float,
    blocked_windows: list[tuple[float, float]] | tuple[tuple[float, float], ...] = (),
    count: int = 12,
    window_seconds: float = 30.0,
    middle_fraction: float = 0.80,
    max_overlap_ratio: float = 0.20,
    max_attempts: int | None = None,
) -> list[BackgroundWindow]:
    duration = max(0.0, float(duration_seconds or 0.0))
    window = max(1.0, float(window_seconds))
    if duration < window + 2.0 or count <= 0:
        return []

    margin = max(0.0, (1.0 - max(0.0, min(float(middle_fraction), 1.0))) * 0.5 * duration)
    lower = min(max(0.0, margin), max(0.0, duration - window))
    upper = max(lower, duration - margin - window)
    if upper <= lower:
        lower = 0.0
        upper = max(0.0, duration - window)

    rng = random.Random(_stable_seed(vod_id))
    attempts = int(max_attempts or max(count * 24, 120))
    sampled: list[BackgroundWindow] = []
    blocked = [(float(start), float(end)) for start, end in blocked_windows if float(end) > float(start)]
    for _ in range(attempts):
        if len(sampled) >= count:
            break
        start = round(rng.uniform(lower, upper), 2)
        end = round(min(start + window, duration), 2)
        if end - start < window * 0.8:
            continue
        candidate = (start, end)
        if any(_overlap_ratio(candidate, blocked_window) > max_overlap_ratio for blocked_window in blocked):
            continue
        if any(_overlap_ratio(candidate, (existing.start, existing.end)) > max_overlap_ratio for existing in sampled):
            continue
        sampled.append(BackgroundWindow(start=start, end=end))
    return sorted(sampled, key=lambda item: item.start)


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _overlap_ratio(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_start, left_end = left
    right_start, right_end = right
    return window_overlap_ratio(left_start, left_end, right_start, right_end)
