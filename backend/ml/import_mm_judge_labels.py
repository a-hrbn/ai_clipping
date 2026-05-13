from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import persisted multimodal judge snapshots as weak proxy labels.")
    parser.add_argument("--snapshots", default=str(BACKEND_ROOT / "tmp" / "storage" / "feedback" / "mm_judge_snapshots.jsonl"))
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_mm_judge_proxy.jsonl"))
    args = parser.parse_args()
    rows = import_rows(pathlib.Path(args.snapshots))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows), encoding="utf-8")
    print(json.dumps({"input": args.snapshots, "output": str(output), "rows": len(rows)}, ensure_ascii=True))
    return 0


def import_rows(path: pathlib.Path) -> list[dict]:
    best_by_candidate: dict[str, dict] = {}
    for row in _read_jsonl(path):
        candidate_id = str(row.get("candidate_id") or row.get("queue_id") or "").strip()
        if not candidate_id:
            continue
        score = _safe_float(row.get("judge_score"))
        if score >= 7.5:
            label = 1
            label_class = "mm_judge_positive"
        elif score <= 3.0:
            label = 0
            label_class = "mm_judge_negative"
        else:
            continue
        copied = dict(row)
        copied.update(
            {
                "candidate_id": candidate_id,
                "label": label,
                "selected": label == 1,
                "label_class": label_class,
                "dataset_source": "mm_judge_proxy",
                "label_source": "mm_judge_proxy",
                "supervision_tier": "mm_judge_proxy",
                "sample_weight": 0.5,
            }
        )
        current = best_by_candidate.get(candidate_id)
        if current is None or abs(score - 5.0) > abs(_safe_float(current.get("judge_score")) - 5.0):
            best_by_candidate[candidate_id] = copied
    return list(best_by_candidate.values())


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
