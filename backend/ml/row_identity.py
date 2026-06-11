from __future__ import annotations

import json
import pathlib
from typing import Iterable


POSITIVE_LABEL_CLASSES = {"positive", "weak_positive", "viral_positive", "gold_positive", "cross_platform_outcome_positive"}


def read_jsonl(path: pathlib.Path, *, strict: bool = True) -> list[dict]:
    """Read a JSONL file into a list of dicts; missing file -> ``[]``.

    ``strict=True`` (default) raises on a malformed line — the historical
    behaviour. ``strict=False`` makes the lenient decision explicit at the
    call-site: blank and malformed lines are skipped and the valid rows are
    returned.
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if strict:
            rows.append(json.loads(line))
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_json(path: pathlib.Path, *, strict: bool = True) -> dict:
    """Pendant to ``write_json``: missing file -> ``{}``.

    A missing file yields an empty dict. ``strict=True`` (default) parses the
    present file with ``json.loads`` and raises on malformed content — the
    historical behaviour, mirroring ``read_jsonl``. ``strict=False`` makes the
    lenient decision explicit at the call-site: a corrupt file yields ``{}``
    instead of raising. A valid but non-dict top-level payload yields ``{}``.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if strict:
        payload = json.loads(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    _replace_text(path, payload + ("\n" if payload else ""))


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_text(path, json.dumps(payload, ensure_ascii=True, indent=2))


def window_overlap_ratio(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Overlap of two time windows as a fraction of the *shorter* one.

    Returns ``overlap / min(duration_a, duration_b)`` and ``0.0`` for
    non-overlapping, zero-length, or inverted windows. Commutative in (a, b).

    This unifies the two mathematically equivalent expressions of the same
    measure previously scattered across the eval/manifest scripts
    (``overlap / min(dur)`` and ``max(overlap / dur_a, overlap / dur_b)``).
    """
    overlap = min(end_a, end_b) - max(start_a, start_b)
    if overlap <= 0:
        return 0.0
    min_dur = min(end_a - start_a, end_b - start_b)
    if min_dur <= 0:
        return 0.0
    return overlap / min_dur


def row_identity_keys(row: dict) -> set[str]:
    keys: set[str] = set()
    row_clip_ids = clip_ids(row)
    if row_clip_ids:
        keys.add("clips:" + ",".join(row_clip_ids))
        keys.update(f"clip:{clip_id}" for clip_id in row_clip_ids)

    queue_id = str(row.get("queue_id") or "").strip()
    if queue_id:
        keys.add(f"queue:{queue_id}")

    candidate_id = str(row.get("candidate_id") or "").strip()
    job_id = str(row.get("job_id") or "").strip()
    if candidate_id and job_id:
        keys.add(f"candidate:{job_id}:{candidate_id}")
    elif candidate_id:
        keys.add(f"candidate:{candidate_id}")

    vod_id = str(row.get("vod_id") or "").strip()
    source_url = str(row.get("source_url") or row.get("vod_url") or "").strip()
    for start_key, end_key in (
        ("proxy_window_start", "proxy_window_end"),
        ("window_start", "window_end"),
    ):
        start = _optional_float(row.get(start_key))
        end = _optional_float(row.get(end_key))
        if start is None or end is None:
            continue
        if vod_id:
            keys.add(f"vod_window:{vod_id}:{start:.2f}:{end:.2f}")
        if source_url:
            keys.add(f"url_window:{source_url}:{start:.2f}:{end:.2f}")
    return keys


def primary_identity_key(row: dict) -> str:
    row_clip_ids = clip_ids(row)
    if row_clip_ids:
        return "clips:" + ",".join(row_clip_ids)
    vod_id = str(row.get("vod_id") or "").strip()
    source_url = str(row.get("source_url") or row.get("vod_url") or "").strip()
    start = _optional_float(row.get("window_start")) or 0.0
    end = _optional_float(row.get("window_end")) or 0.0
    owner = vod_id or source_url or "unknown"
    return f"window:{owner}:{start:.2f}:{end:.2f}"


def clip_ids(row: dict) -> tuple[str, ...]:
    raw = row.get("clip_ids") or []
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def view_count(row: dict) -> int:
    try:
        return int(float(row.get("view_count") or 0))
    except (TypeError, ValueError):
        return 0


def is_positive_label(row: dict) -> bool:
    return str(row.get("label_class") or "").strip() in POSITIVE_LABEL_CLASSES


def _replace_text(path: pathlib.Path, payload: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
