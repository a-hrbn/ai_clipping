from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_eval_dataset, train_ranker
from services.ranker import Ranker
from services.timeline_types import CandidateFeatures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate heuristic pipeline ranking on an eval dataset.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "eval_dataset.jsonl"),
    )
    parser.add_argument("--profile", default="general")
    parser.add_argument("--top-groups", type=int, default=5)
    args = parser.parse_args()

    rows = train_ranker._read_jsonl(pathlib.Path(args.dataset))
    if not rows:
        print("No eval dataset rows found.")
        return 1

    result = evaluate_pipeline_rows(rows, profile=args.profile, top_groups=args.top_groups)
    print(json.dumps(result, ensure_ascii=True))
    return 0


def evaluate_pipeline_rows(rows: list[dict], *, profile: str, top_groups: int = 5) -> dict[str, object]:
    ranker = Ranker(enable_trained_model=False)
    features = [CandidateFeatures.from_dict(row["features"]) for row in rows]
    scores = {
        candidate_id: score
        for candidate_id, score in ranker.score(features, profile=profile)
    }
    labels = train_ranker.np.array([int(row["label"]) for row in rows], dtype=int)
    score_array = train_ranker.np.array([float(scores.get(row["candidate_id"], 0.0)) for row in rows], dtype=float)
    groups = [str(row.get("eval_group") or build_eval_dataset._eval_group(row)) for row in rows]

    return {
        "rows": len(rows),
        "positive_rows": int(sum(labels)),
        "negative_rows": int(len(rows) - sum(labels)),
        "metrics": {
            "classification": train_ranker._evaluate_scores(labels, score_array, score_kind="margin", groups=groups),
            "top1_positive_rate": round(_top1_positive_rate(rows, scores), 3),
            "recall_at_10": round(_group_recall_at_k(rows, scores, 10), 3),
            "recall_at_25": round(_group_recall_at_k(rows, scores, 25), 3),
            "mrr": round(_mean_reciprocal_rank(rows, scores), 3),
        },
        "by_platform": _bucket_metrics(rows, scores, "platform_bucket"),
        "by_content_bucket": _bucket_metrics(rows, scores, "content_bucket"),
        "by_source": _bucket_metrics(rows, scores, "source_bucket"),
        "top_groups": _top_groups(rows, scores, top_groups),
    }


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        group = str(row.get("eval_group") or build_eval_dataset._eval_group(row))
        grouped.setdefault(group, []).append(row)
    return grouped


def _top1_positive_rate(rows: list[dict], scores: dict[str, float]) -> float:
    grouped = _group_rows(rows)
    hits = 0
    considered = 0
    for subset in grouped.values():
        if not subset:
            continue
        ordered = sorted(subset, key=lambda row: float(scores.get(row["candidate_id"], 0.0)), reverse=True)
        hits += int(ordered[0].get("label", 0) == 1)
        considered += 1
    return hits / max(considered, 1)


def _group_recall_at_k(rows: list[dict], scores: dict[str, float], k: int) -> float:
    grouped = _group_rows(rows)
    values: list[float] = []
    for subset in grouped.values():
        positives = [row for row in subset if int(row.get("label", 0)) == 1]
        if not positives:
            continue
        ordered = sorted(subset, key=lambda row: float(scores.get(row["candidate_id"], 0.0)), reverse=True)
        top = ordered[: min(k, len(ordered))]
        hits = sum(1 for row in top if int(row.get("label", 0)) == 1)
        values.append(hits / max(len(positives), 1))
    return sum(values) / max(len(values), 1)


def _mean_reciprocal_rank(rows: list[dict], scores: dict[str, float]) -> float:
    grouped = _group_rows(rows)
    values: list[float] = []
    for subset in grouped.values():
        ordered = sorted(subset, key=lambda row: float(scores.get(row["candidate_id"], 0.0)), reverse=True)
        reciprocal = 0.0
        for index, row in enumerate(ordered, start=1):
            if int(row.get("label", 0)) == 1:
                reciprocal = 1.0 / index
                break
        values.append(reciprocal)
    return sum(values) / max(len(values), 1)


def _bucket_metrics(rows: list[dict], scores: dict[str, float], key: str) -> dict[str, dict[str, object]]:
    values = sorted({str(row.get(key) or "").strip() for row in rows if str(row.get(key) or "").strip()})
    bucketed: dict[str, dict[str, object]] = {}
    for value in values:
        subset = [row for row in rows if str(row.get(key) or "").strip() == value]
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_scores = train_ranker.np.array([float(scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        groups = [str(row.get("eval_group") or build_eval_dataset._eval_group(row)) for row in subset]
        bucketed[value] = {
            "rows": len(subset),
            "classification": train_ranker._evaluate_scores(labels, subset_scores, score_kind="margin", groups=groups),
            "top1_positive_rate": round(_top1_positive_rate(subset, scores), 3),
            "recall_at_10": round(_group_recall_at_k(subset, scores, 10), 3),
        }
    return bucketed


def _top_groups(rows: list[dict], scores: dict[str, float], top_k: int) -> list[dict[str, object]]:
    grouped = _group_rows(rows)
    summaries: list[dict[str, object]] = []
    for group, subset in grouped.items():
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_scores = train_ranker.np.array([float(scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        summaries.append(
            {
                "group": group,
                "rows": len(subset),
                "top1_positive_rate": round(_top1_positive_rate(subset, scores), 3),
                "mrr": round(_mean_reciprocal_rank(subset, scores), 3),
                "classification": train_ranker._evaluate_scores(
                    labels,
                    subset_scores,
                    score_kind="margin",
                    groups=[group for _ in subset],
                ),
            }
        )
    summaries.sort(key=lambda item: (item["rows"], item["group"]), reverse=True)
    return summaries[: max(int(top_k), 0)]


if __name__ == "__main__":
    raise SystemExit(main())
