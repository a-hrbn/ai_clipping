from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_eval_dataset
from services.gold_eval_policy import annotate_gold_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export direct frontend gold-eval ratings from tmp/storage into a gold eval dataset."
    )
    parser.add_argument(
        "--storage-dir",
        default=str(BACKEND_ROOT / "tmp" / "storage"),
    )
    parser.add_argument(
        "--rows-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gold_eval_rows.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gold_eval_dataset.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gold_eval_summary.json"),
    )
    parser.add_argument("--creator-id", default="")
    parser.add_argument(
        "--preserve-existing-if-empty",
        action="store_true",
        help="Keep existing output rows when no frontend gold-eval rows are available.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge existing output rows with frontend gold-eval rows instead of replacing them.",
    )
    args = parser.parse_args()

    payload = export_frontend_gold_eval(
        pathlib.Path(args.storage_dir),
        creator_id=(args.creator_id or "").strip().lower() or None,
    )

    rows_out = pathlib.Path(args.rows_out)
    if args.merge_existing and rows_out.exists():
        existing_rows = _read_jsonl(rows_out)
        if existing_rows:
            payload = _merge_with_existing_rows(existing_rows, payload)
    elif args.preserve_existing_if_empty and not payload["rows"] and rows_out.exists():
        existing_rows = _read_jsonl(rows_out)
        if existing_rows:
            existing_eval_rows = build_eval_dataset.build_eval_dataset(existing_rows)
            payload = {
                "rows": existing_rows,
                "eval_rows": existing_eval_rows,
                "summary": {
                    "rows": len(existing_rows),
                    "eval_dataset": build_eval_dataset.eval_dataset_summary(existing_eval_rows),
                    "preserved_existing": True,
                },
            }

    rows_out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(rows_out, payload["rows"])

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, payload["eval_rows"])

    summary_out = pathlib.Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_if_changed(summary_out, json.dumps(payload["summary"], ensure_ascii=True, indent=2))
    if payload["summary"].get("preserved_existing"):
        print("No frontend gold-eval rows found; preserved existing gold-eval artifacts.", file=sys.stderr)
    elif payload["summary"].get("merged_existing"):
        print("Merged frontend gold-eval rows with existing gold-eval artifacts.", file=sys.stderr)
    elif not payload["rows"]:
        print("No frontend gold-eval rows found; wrote empty gold-eval artifacts.", file=sys.stderr)
    print(json.dumps(payload["summary"], ensure_ascii=True))
    return 0


def export_frontend_gold_eval(
    storage_dir: pathlib.Path,
    *,
    creator_id: str | None = None,
) -> dict[str, object]:
    rows: list[dict] = []
    if not storage_dir.exists():
        return {"rows": rows, "eval_rows": [], "summary": _summary(rows, [])}

    rows.extend(_read_jsonl(storage_dir / "feedback" / "gold_eval_updates.jsonl"))

    for job_dir in sorted(path for path in storage_dir.iterdir() if path.is_dir() and path.name != "feedback"):
        gold_eval_path = job_dir / "debug" / "gold_eval.jsonl"
        if not gold_eval_path.exists():
            continue
        rows.extend(_read_jsonl(gold_eval_path))

    deduped: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("job_id") or ""), str(row.get("candidate_id") or ""))
        existing = deduped.get(key)
        if existing is None or int(row.get("updated_at") or 0) >= int(existing.get("updated_at") or 0):
            deduped[key] = row

    deduped_rows = list(deduped.values())
    if creator_id:
        deduped_rows = [
            row
            for row in deduped_rows
            if str(row.get("creator_id") or "").strip().lower() == creator_id
        ]
    deduped_rows = annotate_gold_rows(deduped_rows)
    eval_rows = build_eval_dataset.build_eval_dataset(deduped_rows)
    return {
        "rows": deduped_rows,
        "eval_rows": eval_rows,
        "summary": _summary(deduped_rows, eval_rows),
    }


def _merge_with_existing_rows(existing_rows: list[dict], frontend_payload: dict[str, object]) -> dict[str, object]:
    frontend_rows = list(frontend_payload.get("rows") or [])
    merged_by_key: dict[tuple[str, ...], dict] = {}

    for row in existing_rows:
        merged_by_key[_gold_row_key(row)] = dict(row)

    frontend_updates = 0
    frontend_additions = 0
    for row in frontend_rows:
        key = _gold_row_key(row)
        existing = merged_by_key.get(key)
        if existing is None:
            frontend_additions += 1
            merged_by_key[key] = dict(row)
            continue
        if _should_replace_existing_gold_row(existing, row):
            frontend_updates += 1
            merged_by_key[key] = dict(row)

    merged_rows = annotate_gold_rows(list(merged_by_key.values()))
    eval_rows = build_eval_dataset.build_eval_dataset(merged_rows)
    summary = _summary(merged_rows, eval_rows)
    summary.update(
        {
            "merged_existing": True,
            "source_rows": {
                "existing": len(existing_rows),
                "frontend": len(frontend_rows),
                "merged": len(merged_rows),
                "frontend_additions": frontend_additions,
                "frontend_updates": frontend_updates,
                "duplicates_skipped": len(existing_rows) + len(frontend_rows) - len(merged_rows),
            },
        }
    )
    if not frontend_rows:
        summary["preserved_existing"] = True
    return {"rows": merged_rows, "eval_rows": eval_rows, "summary": summary}


def _summary(rows: list[dict], eval_rows: list[dict]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "eval_dataset": build_eval_dataset.eval_dataset_summary(eval_rows) if eval_rows else {},
    }


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    _write_text_if_changed(
        path,
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
    )


def _write_text_if_changed(path: pathlib.Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _gold_row_key(row: dict) -> tuple[str, ...]:
    queue_id = str(row.get("queue_id") or "").strip()
    if queue_id:
        return ("queue_id", queue_id)

    job_id = str(row.get("job_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or "").strip()
    if job_id and candidate_id:
        return ("job_candidate", job_id, candidate_id)

    vod_id = str(row.get("vod_id") or "").strip()
    window_start = _stable_time_value(row.get("window_start"))
    window_end = _stable_time_value(row.get("window_end"))
    if vod_id and candidate_id:
        return ("vod_candidate_window", vod_id, candidate_id, window_start, window_end)

    source_url = str(row.get("source_url") or "").strip()
    if source_url and candidate_id:
        return ("source_candidate_window", source_url, candidate_id, window_start, window_end)

    return ("row", json.dumps(row, ensure_ascii=True, sort_keys=True))


def _stable_time_value(value) -> str:
    try:
        return f"{float(value or 0.0):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _should_replace_existing_gold_row(existing: dict, incoming: dict) -> bool:
    existing_updated_at = _updated_at(existing)
    incoming_updated_at = _updated_at(incoming)
    if incoming_updated_at is not None or existing_updated_at is not None:
        return (incoming_updated_at or 0) >= (existing_updated_at or 0)
    existing_source = str(existing.get("dataset_source") or "")
    incoming_source = str(incoming.get("dataset_source") or "")
    return existing_source != "gold_eval_manual" and incoming_source == "gold_eval_frontend"


def _updated_at(row: dict) -> int | None:
    value = row.get("updated_at")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
