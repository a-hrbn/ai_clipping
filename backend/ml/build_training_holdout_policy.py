from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import primary_identity_key, read_jsonl

DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "jynxzi_training_holdout_policy_rows.jsonl"
DEFAULT_SUMMARY = DEFAULT_ARTIFACT_DIR / "jynxzi_training_holdout_policy_summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge holdout-policy JSONL files used to exclude rows from training.")
    parser.add_argument("--input", dest="inputs", action="append", type=pathlib.Path, default=[])
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=pathlib.Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)

    rows, summary = merge_holdout_rows(args.inputs)
    summary.update(
        {
            "event": "training_holdout_policy_built",
            "inputs": [str(path) for path in args.inputs],
            "output": str(args.output),
        }
    )
    _write_jsonl_if_changed(args.output, rows)
    _write_json_if_changed(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


def merge_holdout_rows(paths: list[pathlib.Path]) -> tuple[list[dict], dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    input_rows = 0
    source_counts: Counter[str] = Counter()
    vod_ids: set[str] = set()
    for path in paths:
        if not pathlib.Path(path).exists():
            continue
        for row in read_jsonl(pathlib.Path(path)):
            input_rows += 1
            if not _is_holdout_row(row):
                continue
            copied = dict(row)
            copied.setdefault("gold_holdout_split", "holdout")
            copied.setdefault("dataset_source", "training_holdout_policy")
            key = _row_key(copied)
            if key in seen:
                continue
            seen.add(key)
            merged.append(copied)
            source_counts[str(copied.get("dataset_source") or "unknown")] += 1
            vod_id = str(copied.get("vod_id") or "").strip()
            if vod_id:
                vod_ids.add(vod_id)
    merged.sort(key=lambda row: (str(row.get("vod_id") or ""), _as_float(row.get("window_start")), _as_float(row.get("window_end")), _row_key(row)))
    return merged, {
        "input_rows": input_rows,
        "rows": len(merged),
        "unique_holdout_vods": len(vod_ids),
        "holdout_vod_ids": sorted(vod_ids),
        "by_dataset_source": dict(sorted(source_counts.items())),
    }


def _is_holdout_row(row: dict) -> bool:
    split = str(row.get("gold_holdout_split") or row.get("outcome_split") or row.get("dataset_role") or "").lower()
    if "holdout" in split:
        return True
    source = str(row.get("dataset_source") or row.get("supervision_tier") or "").lower()
    return "holdout" in source


def _row_key(row: dict) -> str:
    vod_id = str(row.get("vod_id") or "").strip()
    if vod_id and row.get("window_start") in (None, "") and row.get("window_end") in (None, ""):
        return f"vod:{vod_id}"
    return primary_identity_key(row)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_jsonl_if_changed(path: pathlib.Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    if payload:
        payload += "\n"
    _write_text_if_changed(path, payload)


def _write_json_if_changed(path: pathlib.Path, payload: dict) -> None:
    _write_text_if_changed(path, json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


def _write_text_if_changed(path: pathlib.Path, payload: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
