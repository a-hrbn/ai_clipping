from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import train_ranker


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Twitch-derived dataset with bucketed label metrics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--top-groups", type=int, default=5)
    args = parser.parse_args()

    rows = train_ranker._read_jsonl(pathlib.Path(args.dataset))
    if not rows:
        print("No dataset rows found.")
        return 1
    from services.ranker import Ranker
    from services.timeline_types import CandidateFeatures

    split = train_ranker._split_rows(rows)
    validation = split["validation"]
    ranker = Ranker(artifact_dir=pathlib.Path(args.artifact_dir))
    artifact = ranker._load_artifact(args.profile)
    feature_names = train_ranker._numeric_feature_names(validation[0]["features"])
    scales = train_ranker._compute_scales(rows, feature_names)
    weights = train_ranker._mean_difference_weights(
        [row["features"] for row in rows if int(row["label"]) == 1],
        [row["features"] for row in rows if int(row["label"]) == 0],
        feature_names,
        scales,
    )
    baseline_scores = train_ranker._linear_raw_scores(validation, feature_names, weights, scales, 0.0)
    runtime_features = [CandidateFeatures.from_dict(row["features"]) for row in validation]
    runtime_scores = {
        candidate_id: score
        for candidate_id, score in ranker.score(runtime_features, args.profile)
    }
    runtime_score_array = train_ranker.np.array(
        [float(runtime_scores.get(row["candidate_id"], 0.0)) for row in validation],
        dtype=float,
    )
    labels = train_ranker.np.array([int(row["label"]) for row in validation], dtype=int)
    metrics = {
        "baseline_linear": train_ranker._evaluate_scores(labels, baseline_scores),
        "runtime_artifact": train_ranker._evaluate_scores(labels, runtime_score_array, score_kind="margin"),
    }
    bucketed = {}
    for label_class in sorted({str(row.get("label_class", "")) for row in validation}):
        subset = [row for row in validation if str(row.get("label_class", "")) == label_class]
        if not subset:
            continue
        subset_labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_baseline_scores = train_ranker._linear_raw_scores(subset, feature_names, weights, scales, 0.0)
        subset_runtime_scores = train_ranker.np.array(
            [float(runtime_scores.get(row["candidate_id"], 0.0)) for row in subset],
            dtype=float,
        )
        bucketed[label_class] = {
            "rows": len(subset),
            "baseline_linear": train_ranker._evaluate_scores(subset_labels, subset_baseline_scores),
            "runtime_artifact": train_ranker._evaluate_scores(subset_labels, subset_runtime_scores, score_kind="margin"),
        }
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
                "validation": metrics,
                "bucketed": bucketed,
                "split_strategy": split["strategy"],
                "split_unit": split.get("unit", ""),
                "leakage_risk": split.get("leakage_risk", "unknown"),
                "top_groups": _group_metrics(validation, runtime_scores, split.get("unit", ""), args.top_groups),
            },
            ensure_ascii=True,
        )
    )
    return 0


def _group_metrics(rows: list[dict], runtime_scores: dict[str, float], split_unit: str, top_k: int) -> list[dict]:
    unit = split_unit or "vod_id"
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        group = train_ranker._ranking_group_key(row, unit)
        grouped.setdefault(group, []).append(row)

    summaries: list[dict] = []
    for group, subset in grouped.items():
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        scores = train_ranker.np.array([float(runtime_scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        summaries.append(
            {
                "group": group,
                "rows": len(subset),
                "runtime_artifact": train_ranker._evaluate_scores(labels, scores, score_kind="margin"),
            }
        )
    summaries.sort(key=lambda item: (item["rows"], item["group"]), reverse=True)
    return summaries[: max(int(top_k), 0)]


if __name__ == "__main__":
    raise SystemExit(main())
