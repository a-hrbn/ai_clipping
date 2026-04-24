from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.gold_eval_policy import GOLD_HOLDOUT_SPLIT_FIELD


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export high-scoring ranker false positives as hard-negative training rows."
    )
    parser.add_argument(
        "--diagnosis-csv",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_holdout_diagnosis.csv"),
    )
    parser.add_argument(
        "--gold-rows",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gold_eval_rows.jsonl"),
    )
    parser.add_argument(
        "--rows-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_hard_negative_train_rows.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_hard_negative_train_summary.json"),
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--min-forced-score", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=100)
    args = parser.parse_args()

    payload = export_hard_negatives(
        diagnosis_csv=pathlib.Path(args.diagnosis_csv),
        gold_rows_path=pathlib.Path(args.gold_rows),
        profile=args.profile,
        min_forced_score=args.min_forced_score,
        max_rows=args.max_rows,
    )
    _write_jsonl(pathlib.Path(args.rows_out), payload["rows"])
    _write_json(pathlib.Path(args.summary_out), payload["summary"])
    print(json.dumps(payload["summary"], ensure_ascii=True))
    return 0


def export_hard_negatives(
    *,
    diagnosis_csv: pathlib.Path,
    gold_rows_path: pathlib.Path,
    profile: str,
    min_forced_score: float = 0.0,
    max_rows: int = 100,
) -> dict[str, object]:
    diagnosis_rows = _read_csv(diagnosis_csv)
    gold_rows = _read_jsonl(gold_rows_path)
    gold_index = _gold_row_index(gold_rows)
    selected = _select_false_positives(
        diagnosis_rows,
        min_forced_score=float(min_forced_score),
        max_rows=max(int(max_rows), 0),
    )
    rows: list[dict] = []
    missing_keys: list[str] = []
    seen_keys: set[tuple[str, ...]] = set()
    for diagnosis in selected:
        key = _diagnosis_key(diagnosis)
        if key is None:
            missing_keys.append("")
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        source = gold_index.get(key)
        if source is None:
            missing_keys.append("|".join(key))
            continue
        rows.append(_hard_negative_row(source, diagnosis, profile))

    summary = {
        "profile": profile,
        "diagnosis_csv": str(diagnosis_csv),
        "gold_rows": str(gold_rows_path),
        "diagnosis_rows": len(diagnosis_rows),
        "gold_rows_read": len(gold_rows),
        "selected_false_positive_rows": len(selected),
        "exported_rows": len(rows),
        "missing_source_rows": len(missing_keys),
        "min_forced_score": float(min_forced_score),
        "max_rows": int(max_rows),
        "dataset_sources": dict(Counter(str(row.get("dataset_source") or "") for row in rows)),
        "candidate_sources": dict(Counter(str(row.get("candidate_source") or "") for row in rows)),
        "original_gold_splits": dict(Counter(str(row.get("hard_negative_original_split") or "") for row in rows)),
        "leakage_note": (
            "These rows are promoted into training. Do not treat the same candidates as untouched holdout "
            "when evaluating the next artifact."
        ),
    }
    if missing_keys:
        summary["missing_source_keys"] = missing_keys[:20]
    return {"rows": rows, "summary": summary}


def _select_false_positives(rows: list[dict], *, min_forced_score: float, max_rows: int) -> list[dict]:
    selected: list[dict] = []
    for row in rows:
        if str(row.get("error_type") or "").strip() != "false_positive":
            continue
        if str(row.get("label") or "").strip() != "0":
            continue
        try:
            forced_score = float(row.get("forced_score") or 0.0)
        except ValueError:
            continue
        if forced_score < min_forced_score:
            continue
        selected.append(row)
    selected.sort(
        key=lambda row: (
            -_as_float(row.get("forced_score")),
            str(row.get("group") or ""),
            _as_int(row.get("rank")),
        )
    )
    return selected[:max_rows] if max_rows else []


def _hard_negative_row(source: dict, diagnosis: dict, profile: str) -> dict:
    copied = dict(source)
    copied["features"] = dict(source.get("features") or {})
    copied["label"] = 0
    copied["selected"] = False
    copied["exported"] = False
    copied["dataset_source"] = "gold_eval_hard_negative_train"
    copied["label_class"] = "gold_hard_negative"
    copied["label_source"] = "gold_eval"
    copied["supervision_tier"] = "gold_eval"
    copied["supervision_reason"] = "ranker_diagnosis_false_positive"
    copied["profile_hint"] = str(copied.get("creator_id") or copied.get("streamer") or profile).strip().lower() or profile
    copied["hard_negative_source"] = "ranker_holdout_diagnosis"
    copied["hard_negative_original_split"] = str(source.get(GOLD_HOLDOUT_SPLIT_FIELD) or "")
    copied["hard_negative_forced_score"] = _as_float(diagnosis.get("forced_score"))
    copied["hard_negative_heuristic_score"] = _as_float(diagnosis.get("heuristic_score"))
    copied["hard_negative_trained_score"] = _as_float(diagnosis.get("trained_score"))
    copied["hard_negative_diagnosis_group"] = str(diagnosis.get("group") or "")
    copied["hard_negative_diagnosis_rank"] = _as_int(diagnosis.get("rank"))
    copied["hard_negative_review_url"] = str(diagnosis.get("review_url") or "")
    copied["hard_negative_original_sample_weight"] = _as_float(source.get("sample_weight")) or 1.0
    copied["sample_weight"] = 1.0
    copied[GOLD_HOLDOUT_SPLIT_FIELD] = "train"
    return copied


def _gold_row_index(rows: list[dict]) -> dict[tuple[str, ...], dict]:
    index: dict[tuple[str, ...], dict] = {}
    for row in rows:
        for key in _row_keys(row):
            index.setdefault(key, row)
    return index


def _diagnosis_key(row: dict) -> tuple[str, ...] | None:
    queue_id = str(row.get("queue_id") or "").strip()
    if queue_id:
        return ("queue_id", queue_id)
    job_id = str(row.get("job_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or "").strip()
    if job_id and candidate_id:
        return ("job_candidate", job_id, candidate_id)
    vod_id = str(row.get("vod_id") or "").strip()
    if vod_id and candidate_id:
        return ("vod_candidate", vod_id, candidate_id)
    return None


def _row_keys(row: dict) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []
    queue_id = str(row.get("queue_id") or "").strip()
    if queue_id:
        keys.append(("queue_id", queue_id))
    job_id = str(row.get("job_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or "").strip()
    if job_id and candidate_id:
        keys.append(("job_candidate", job_id, candidate_id))
    vod_id = str(row.get("vod_id") or "").strip()
    if vod_id and candidate_id:
        keys.append(("vod_candidate", vod_id, candidate_id))
    return keys


def _read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    _write_text_if_changed(path, payload + ("\n" if payload else ""))


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_if_changed(path, json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _write_text_if_changed(path: pathlib.Path, payload: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return
    path.write_text(payload, encoding="utf-8")


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
