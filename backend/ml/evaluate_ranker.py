from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.ranker import Ranker
from services.timeline_types import CandidateFeatures
from ml import train_ranker


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the current ranker artifact on exported training rows.")
    parser.add_argument("--dataset", default="/tmp/ai_clipping/feedback/training_rows.jsonl")
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--top-groups", type=int, default=5)
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    rows = _read_jsonl(dataset_path)
    if not rows:
        print("No dataset rows found.")
        return 1

    ranker = Ranker(artifact_dir=pathlib.Path(args.artifact_dir))
    artifact = ranker._load_artifact(args.profile)
    features = [CandidateFeatures(**row["features"]) for row in rows]
    scores = {candidate_id: score for candidate_id, score in ranker.score(features, args.profile)}

    scored_rows: list[dict] = []
    for row in rows:
        scored_rows.append({"label": int(row["label"]), "score": float(scores.get(row["candidate_id"], 0.0))})

    metrics = train_ranker._evaluate_scores(
        train_ranker.np.array([row["label"] for row in scored_rows], dtype=int),
        train_ranker.np.array([row["score"] for row in scored_rows], dtype=float),
        score_kind="margin",
    )
    split = train_ranker._split_rows(rows)
    print(
        json.dumps(
            {
                "artifact": {
                    "available": artifact is not None,
                    "selected_model": artifact.get("selected_model") if artifact else "",
                    "model_type": artifact.get("model_type") if artifact else "",
                    "training_stats": artifact.get("training_stats", {}) if artifact else {},
                    "validation": artifact.get("validation", {}) if artifact else {},
                },
                "metrics": metrics,
                "rows": len(rows),
                "positive_rows": sum(1 for row in rows if int(row["label"]) == 1),
                "negative_rows": sum(1 for row in rows if int(row["label"]) == 0),
                "split_strategy": split["strategy"],
                "split_unit": split.get("unit", ""),
                "leakage_risk": split.get("leakage_risk", "unknown"),
                "by_label_class": _bucket_metrics(rows, scores, "label_class"),
                "by_supervision_tier": _bucket_metrics(rows, scores, "supervision_tier"),
                "top_groups": _group_metrics(rows, scores, split.get("unit", ""), args.top_groups),
            },
            ensure_ascii=True,
        )
    )
    return 0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if row.get("label") in {0, 1}:
                rows.append(row)
    return rows


def _bucket_metrics(rows: list[dict], scores: dict[str, float], key: str) -> dict[str, dict]:
    bucketed: dict[str, dict] = {}
    values = sorted({str(row.get(key) or "").strip() for row in rows if str(row.get(key) or "").strip()})
    for value in values:
        subset = [row for row in rows if str(row.get(key) or "").strip() == value]
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_scores = train_ranker.np.array([float(scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        bucketed[value] = {
            "rows": len(subset),
            "metrics": train_ranker._evaluate_scores(labels, subset_scores, score_kind="margin"),
        }
    return bucketed


def _group_metrics(rows: list[dict], scores: dict[str, float], split_unit: str, top_k: int) -> list[dict]:
    unit = split_unit or "creator_id"
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        group = train_ranker._ranking_group_key(row, unit)
        grouped.setdefault(group, []).append(row)

    summaries: list[dict] = []
    for group, subset in grouped.items():
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_scores = train_ranker.np.array([float(scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        summaries.append(
            {
                "group": group,
                "rows": len(subset),
                "metrics": train_ranker._evaluate_scores(labels, subset_scores, score_kind="margin"),
            }
        )
    summaries.sort(key=lambda item: (item["rows"], item["group"]), reverse=True)
    return summaries[: max(int(top_k), 0)]


if __name__ == "__main__":
    raise SystemExit(main())
