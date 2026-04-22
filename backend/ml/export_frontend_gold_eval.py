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
    args = parser.parse_args()

    payload = export_frontend_gold_eval(
        pathlib.Path(args.storage_dir),
        creator_id=(args.creator_id or "").strip().lower() or None,
    )
    if not payload["rows"]:
        print("No frontend gold-eval rows found.", file=sys.stderr)
        return 1

    rows_out = pathlib.Path(args.rows_out)
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(rows_out, payload["rows"])

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, payload["eval_rows"])

    summary_out = pathlib.Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        json.dumps(payload["summary"], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
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
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
