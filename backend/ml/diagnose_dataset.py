from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

def _read_jsonl(path: pathlib.Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _numeric_feature_names(features: dict) -> list:
    return [k for k, v in features.items() if isinstance(v, (int, float))]


def _row_timestamp(row: dict) -> float:
    for key in ("timestamp", "vod_timestamp", "start_time", "created_at"):
        val = row.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a training dataset and highlight weak or non-informative features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top", type=int, default=8, help="How many strongest separating features to print.")
    args = parser.parse_args()

    rows = _read_jsonl(pathlib.Path(args.dataset))
    if not rows:
        print("No dataset rows found.")
        return 1

    feature_names = _numeric_feature_names(rows[0]["features"])
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    creator_counts = Counter(str(row.get("creator_id") or row.get("streamer") or "unknown") for row in rows)
    timestamps = [_row_timestamp(row) for row in rows if _row_timestamp(row) > 0]

    print(json.dumps({
        "rows": len(rows),
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "creators": dict(creator_counts.most_common()),
        "time_split_possible": len(timestamps) >= max(4, len(rows) // 2),
        "time_range": {
            "min_timestamp": min(timestamps) if timestamps else 0,
            "max_timestamp": max(timestamps) if timestamps else 0,
        },
    }, ensure_ascii=True))

    diagnostics = []
    for name in feature_names:
        pos_vals = [float(row["features"].get(name, 0.0)) for row in positives]
        neg_vals = [float(row["features"].get(name, 0.0)) for row in negatives]
        all_vals = pos_vals + neg_vals
        if not all_vals:
            continue
        mean_all = sum(all_vals) / len(all_vals)
        variance = sum((value - mean_all) ** 2 for value in all_vals) / max(len(all_vals), 1)
        pos_mean = sum(pos_vals) / max(len(pos_vals), 1)
        neg_mean = sum(neg_vals) / max(len(neg_vals), 1)
        diagnostics.append({
            "feature": name,
            "positive_mean": round(pos_mean, 3),
            "negative_mean": round(neg_mean, 3),
            "delta": round(pos_mean - neg_mean, 3),
            "std": round(variance ** 0.5, 3),
            "zero_fraction": round(sum(1 for value in all_vals if abs(value) < 1e-9) / len(all_vals), 3),
        })

    strongest = sorted(diagnostics, key=lambda row: abs(row["delta"]), reverse=True)[: max(args.top, 1)]
    weakest = sorted(
        diagnostics,
        key=lambda row: (row["std"] > 0.05, abs(row["delta"]), row["zero_fraction"]),
    )[: max(args.top, 1)]
    print(json.dumps({"strongest_features": strongest, "weak_features": weakest}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
