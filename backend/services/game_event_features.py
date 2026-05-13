from __future__ import annotations

from services.game_events import GameEvent


def features_for_window(events: list[GameEvent], *, start: float, end: float) -> dict[str, float | bool | str]:
    overlapping = [
        event
        for event in events
        if float(event.end) > float(start) and float(event.start) < float(end)
    ]
    if not overlapping:
        return {
            "game_event_score": 0.0,
            "game_event_type": "",
            "kill_density_in_window": 0.0,
            "is_clutch_window": False,
            "is_match_endpoint": False,
        }
    duration = max(float(end) - float(start), 1.0)
    kill_events = [
        event
        for event in overlapping
        if event.event_type in {"kill", "multi_kill", "clutch"}
    ]
    strongest = max(overlapping, key=lambda event: float(event.confidence))
    return {
        "game_event_score": round(max(float(event.confidence) for event in overlapping), 3),
        "game_event_type": strongest.event_type,
        "kill_density_in_window": round(len(kill_events) / duration, 4),
        "is_clutch_window": any(event.event_type == "clutch" for event in overlapping),
        "is_match_endpoint": any(event.event_type in {"round_win", "match_endpoint", "rank_up"} for event in overlapping),
    }
