from __future__ import annotations

import json
import pathlib


def append_annotations(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_annotations(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def merge_annotations(dataset_rows: list[dict], annotations: list[dict]) -> list[dict]:
    by_sample = {str(row.get("sample_id") or _sample_id(row)): row for row in dataset_rows}
    for annotation in annotations:
        sample_id = str(annotation.get("sample_id") or "")
        if not sample_id or sample_id not in by_sample:
            continue
        row = by_sample[sample_id]
        row["manual_annotation"] = {
            "is_good_clip": bool(annotation.get("is_good_clip")),
            "quality_score": int(annotation.get("quality_score", 0) or 0),
            "start_time": annotation.get("start_time"),
            "end_time": annotation.get("end_time"),
            "standalone": annotation.get("standalone"),
            "hook_strength": annotation.get("hook_strength"),
            "notes": annotation.get("notes", ""),
        }
        _apply_manual_label(row, annotation)
    return list(by_sample.values())


def _sample_id(row: dict) -> str:
    return f"{row.get('vod_id','')}:{row.get('window_start','')}:{row.get('window_end','')}"


def _apply_manual_label(row: dict, annotation: dict) -> None:
    is_good_clip = annotation.get("is_good_clip")
    quality_score = int(annotation.get("quality_score", 0) or 0)
    if is_good_clip is None and quality_score <= 0:
        return

    good = bool(is_good_clip)
    if not good and quality_score >= 4:
        good = True

    row["label"] = 1 if good else 0
    row["label_source"] = "manual_annotation"
    row["supervision_tier"] = "manual_annotation"
    row["supervision_reason"] = str(annotation.get("notes") or "").strip() or "manual_review"
    row["sample_weight"] = 3.0
    row["ordinal_label"] = _manual_ordinal_label(good, quality_score)
    row["label_class"] = _manual_label_class(good, quality_score)


def _manual_ordinal_label(good: bool, quality_score: int) -> int:
    if not good:
        return 0 if quality_score <= 2 else 1
    if quality_score >= 5:
        return 4
    if quality_score >= 4:
        return 3
    return 2


def _manual_label_class(good: bool, quality_score: int) -> str:
    if not good:
        return "hard_negative" if quality_score >= 2 else "background_negative"
    if quality_score >= 5:
        return "viral_positive"
    if quality_score >= 4:
        return "positive"
    return "weak_positive"
