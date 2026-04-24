from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pathlib
import sys
import types
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

if importlib.util.find_spec("whisper") is None:
    sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import train_ranker
from services.ranker import OFFLINE_ONLY_FEATURE_NAMES, Ranker
from services.timeline_types import CandidateFeatures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose rejected ranker artifacts by force-scoring gold holdout rows."
    )
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gold_eval_rows.jsonl"),
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--split-field", default="gold_holdout_split")
    parser.add_argument("--split-value", default="holdout")
    parser.add_argument(
        "--group-field",
        default="",
        help="Primary grouping field. Defaults to vod_id, then eval_group, job_id, source_url.",
    )
    parser.add_argument("--top-groups", type=int, default=10)
    parser.add_argument("--min-group-rows", type=int, default=2)
    parser.add_argument("--threshold", default="artifact", help="'artifact', 'zero', or a numeric threshold.")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    artifact_dir = pathlib.Path(args.artifact_dir)
    csv_out = pathlib.Path(args.csv_out) if args.csv_out else artifact_dir / f"{args.profile}_ranker_holdout_diagnosis.csv"
    report_out = (
        pathlib.Path(args.report_out)
        if args.report_out
        else artifact_dir / f"{args.profile}_ranker_holdout_diagnosis.json"
    )
    report = diagnose_holdout(
        dataset_path=pathlib.Path(args.dataset),
        profile=args.profile,
        artifact_dir=artifact_dir,
        split_field=args.split_field,
        split_value=args.split_value,
        group_field=args.group_field,
        top_groups=args.top_groups,
        min_group_rows=args.min_group_rows,
        threshold_arg=args.threshold,
        csv_out=csv_out,
    )
    _write_json(report_out, report)
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report.get("status") != "failed" else 1


def diagnose_holdout(
    *,
    dataset_path: pathlib.Path,
    profile: str,
    artifact_dir: pathlib.Path,
    split_field: str = "gold_holdout_split",
    split_value: str = "holdout",
    group_field: str = "",
    top_groups: int = 10,
    min_group_rows: int = 2,
    threshold_arg: str = "artifact",
    csv_out: pathlib.Path | None = None,
) -> dict[str, object]:
    rows = _read_jsonl(dataset_path)
    selected_rows = _filter_rows(rows, split_field, split_value)
    baseline_ranker = Ranker(artifact_dir=artifact_dir, enable_trained_model=False)
    artifact_ranker = Ranker(artifact_dir=artifact_dir, enable_trained_model=True)
    inspection = artifact_ranker.inspect_artifact(profile)
    artifact = inspection.get("artifact") if isinstance(inspection.get("artifact"), dict) else None
    if artifact is None:
        report = _empty_report(
            profile=profile,
            dataset_path=dataset_path,
            rows=rows,
            selected_rows=selected_rows,
            inspection=inspection,
            reason="artifact_missing",
        )
        if csv_out is not None:
            _write_csv(csv_out, [])
            report["csv_out"] = str(csv_out)
        return report

    scored_rows = _score_rows(
        selected_rows,
        profile=profile,
        ranker=baseline_ranker,
        artifact=artifact,
        threshold_arg=threshold_arg,
    )
    group_key = group_field.strip()
    grouped = _group_scored_rows(scored_rows, group_key)
    threshold = _threshold_from_artifact(artifact, threshold_arg)
    group_summaries = _group_summaries(grouped, min_group_rows=max(int(min_group_rows), 1), threshold=threshold)
    group_summaries.sort(
        key=lambda item: (
            float(item["forced_artifact"].get("ranking_metric_value", 0.0)),
            float(item["forced_artifact"].get("ndcg_at_5", 0.0)),
            float(item["forced_artifact"].get("mrr", 0.0)),
            -int(item["positive_rows"]),
            str(item["group"]),
        )
    )
    ranking_groups = [group for group in group_summaries if int(group.get("positive_rows", 0)) > 0]
    false_positive_groups = [
        group for group in group_summaries if int(group.get("positive_rows", 0)) == 0 and int(group.get("false_positive_rows", 0)) > 0
    ]
    false_positive_groups.sort(
        key=lambda item: (-int(item.get("false_positive_rows", 0)), -int(item.get("rows", 0)), str(item.get("group", "")))
    )
    worst_groups = ranking_groups[: max(int(top_groups), 0)]
    worst_group_ids = {str(group["group"]) for group in worst_groups[: max(int(top_groups), 0)]}
    worst_group_ids.update(str(group["group"]) for group in false_positive_groups[: max(int(top_groups), 0)])
    csv_rows = [
        csv_row
        for group, group_rows in grouped.items()
        if group in worst_group_ids
        for csv_row in _diagnostic_csv_rows(group, group_rows)
    ]
    if csv_out is not None:
        _write_csv(csv_out, csv_rows)

    labels = train_ranker.np.array([int(item["row"]["label"]) for item in scored_rows], dtype=int)
    baseline_scores = train_ranker.np.array([float(item["heuristic_score"]) for item in scored_rows], dtype=float)
    forced_scores = train_ranker.np.array([float(item["forced_score"]) for item in scored_rows], dtype=float)
    groups = [str(item["group"]) for item in scored_rows]
    baseline_metrics = train_ranker._evaluate_scores(labels, baseline_scores, score_kind="margin", groups=groups)
    forced_metrics = train_ranker._evaluate_scores(
        labels,
        forced_scores,
        score_kind="margin",
        groups=groups,
        threshold=threshold,
    )
    return {
        "status": "ok",
        "profile": profile,
        "dataset": str(dataset_path),
        "rows": len(rows),
        "selected_rows": len(selected_rows),
        "split_field": split_field,
        "split_value": split_value,
        "group_field": group_key or "vod_id|eval_group|job_id|source_url",
        "artifact": {
            "inspection_status": inspection.get("status", "missing"),
            "inspection_reason": inspection.get("reason", ""),
            "selected_model": artifact.get("selected_model", ""),
            "model_type": artifact.get("model_type", ""),
            "artifact_blend": artifact.get("blend"),
            "safe_blend": baseline_ranker._safe_blend(artifact),
            "serving_threshold": artifact.get("serving_threshold"),
            "diagnosis_threshold": threshold,
            "training_stats": artifact.get("training_stats", {}),
            "validation": artifact.get("validation", {}),
        },
        "metrics": {
            "baseline_heuristic": baseline_metrics,
            "forced_artifact": forced_metrics,
            "delta": _metric_delta(forced_metrics, baseline_metrics),
        },
        "groups": group_summaries,
        "worst_groups": worst_groups,
        "all_negative_false_positive_groups": false_positive_groups[: max(int(top_groups), 0)],
        "csv_rows": len(csv_rows),
        "csv_out": str(csv_out) if csv_out is not None else "",
    }


def _score_rows(
    rows: list[dict],
    *,
    profile: str,
    ranker: Ranker,
    artifact: dict,
    threshold_arg: str,
) -> list[dict]:
    if not rows:
        return []
    features = [
        CandidateFeatures.from_dict(
            {
                "candidate_id": str(row.get("candidate_id") or "unknown"),
                **dict(row.get("features") or {}),
            }
        )
        for row in rows
    ]
    heuristic_pairs = ranker._heuristic_scores(features, profile)
    for feature, (heuristic_score, _rationale) in zip(features, heuristic_pairs):
        feature.heuristic_score = float(heuristic_score)
    trained_scores = ranker._trained_scores(features, artifact)
    blend = ranker._safe_blend(artifact)
    threshold = _threshold_from_artifact(artifact, threshold_arg)
    scored: list[dict] = []
    for row, feature, (heuristic_score, rationale), trained_score in zip(rows, features, heuristic_pairs, trained_scores):
        forced_score = ranker._clamp_score(float(heuristic_score) * (1.0 - blend) + float(trained_score) * blend)
        predicted = 1 if forced_score >= threshold else 0
        label = int(row.get("label", 0))
        scored.append(
            {
                "row": row,
                "feature": feature,
                "group": _group_key(row, ""),
                "label": label,
                "predicted": predicted,
                "error_type": _error_type(label, predicted),
                "heuristic_score": float(heuristic_score),
                "trained_score": float(trained_score),
                "forced_score": float(forced_score),
                "rationale": list(rationale),
                "top_positive_features": _top_feature_contributions(row, artifact, positive=True),
                "top_negative_features": _top_feature_contributions(row, artifact, positive=False),
            }
        )
    return scored


def _group_scored_rows(scored_rows: list[dict], group_field: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in scored_rows:
        group = _group_key(item["row"], group_field)
        item["group"] = group
        grouped.setdefault(group, []).append(item)
    return grouped


def _group_summaries(grouped: dict[str, list[dict]], *, min_group_rows: int, threshold: float) -> list[dict]:
    summaries: list[dict] = []
    for group, group_rows in grouped.items():
        if len(group_rows) < min_group_rows:
            continue
        labels = train_ranker.np.array([int(item["label"]) for item in group_rows], dtype=int)
        baseline_scores = train_ranker.np.array([float(item["heuristic_score"]) for item in group_rows], dtype=float)
        forced_scores = train_ranker.np.array([float(item["forced_score"]) for item in group_rows], dtype=float)
        group_labels = [str(group) for _item in group_rows]
        baseline = train_ranker._evaluate_scores(labels, baseline_scores, score_kind="margin", groups=group_labels)
        forced = train_ranker._evaluate_scores(
            labels,
            forced_scores,
            score_kind="margin",
            groups=group_labels,
            threshold=threshold,
        )
        summaries.append(
            {
                "group": group,
                "rows": len(group_rows),
                "positive_rows": int(labels.sum()),
                "negative_rows": int(len(group_rows) - int(labels.sum())),
                "false_positive_rows": sum(1 for item in group_rows if item["error_type"] == "false_positive"),
                "false_negative_rows": sum(1 for item in group_rows if item["error_type"] == "false_negative"),
                "baseline_heuristic": baseline,
                "forced_artifact": forced,
                "delta": _metric_delta(forced, baseline),
            }
        )
    return summaries


def _diagnostic_csv_rows(group: str, group_rows: list[dict]) -> list[dict]:
    ranked = sorted(group_rows, key=lambda item: float(item["forced_score"]), reverse=True)
    rows: list[dict] = []
    group_labels = train_ranker.np.array([int(item["label"]) for item in ranked], dtype=int)
    group_scores = train_ranker.np.array([float(item["forced_score"]) for item in ranked], dtype=float)
    group_metrics = train_ranker._evaluate_scores(
        group_labels,
        group_scores,
        score_kind="margin",
        groups=[group for _item in ranked],
    )
    for index, item in enumerate(ranked, start=1):
        row = item["row"]
        rows.append(
            {
                "group": group,
                "rank": index,
                "group_rows": len(ranked),
                "group_positive_rows": int(group_labels.sum()),
                "group_mrr": group_metrics.get("mrr", 0.0),
                "group_ndcg_at_5": group_metrics.get("ndcg_at_5", 0.0),
                "label": item["label"],
                "predicted": item["predicted"],
                "error_type": item["error_type"],
                "forced_score": round(float(item["forced_score"]), 6),
                "heuristic_score": round(float(item["heuristic_score"]), 6),
                "trained_score": round(float(item["trained_score"]), 6),
                "queue_id": row.get("queue_id", ""),
                "job_id": row.get("job_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "vod_id": row.get("vod_id", ""),
                "candidate_source": row.get("candidate_source") or row.get("source_bucket") or "",
                "content_bucket": row.get("content_bucket", ""),
                "gold_grade": row.get("gold_grade", ""),
                "window_start": row.get("window_start", ""),
                "window_end": row.get("window_end", ""),
                "source_url": row.get("source_url", ""),
                "review_url": _review_url(str(row.get("source_url") or ""), row.get("window_start")),
                "top_positive_features": ";".join(item["top_positive_features"]),
                "top_negative_features": ";".join(item["top_negative_features"]),
                "rationale": ";".join(str(value) for value in item["rationale"]),
            }
        )
    return rows


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("label") in {0, 1} and isinstance(row.get("features"), dict):
            rows.append(row)
    return rows


def _filter_rows(rows: list[dict], split_field: str, split_value: str) -> list[dict]:
    field = str(split_field or "").strip()
    value = str(split_value or "").strip()
    if not field or not value:
        return list(rows)
    return [row for row in rows if str(row.get(field) or "").strip() == value]


def _group_key(row: dict, group_field: str) -> str:
    if group_field:
        value = str(row.get(group_field) or "").strip()
        if value:
            return value
    for field in ("vod_id", "eval_group", "job_id", "source_url"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return "unknown"


def _threshold_from_artifact(artifact: dict, threshold_arg: str) -> float:
    value = str(threshold_arg or "artifact").strip().lower()
    if value == "artifact":
        threshold = train_ranker._artifact_serving_threshold(artifact)
        return float(threshold) if threshold is not None else 0.0
    if value == "zero":
        return 0.0
    try:
        threshold = float(value)
    except ValueError:
        return 0.0
    return threshold if math.isfinite(threshold) else 0.0


def _top_feature_contributions(row: dict, artifact: dict, *, positive: bool, limit: int = 5) -> list[str]:
    feature_names = [str(name) for name in artifact.get("feature_names") or []]
    weights = artifact.get("weights") if isinstance(artifact.get("weights"), dict) else {}
    scales = artifact.get("scales") if isinstance(artifact.get("scales"), dict) else {}
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    contributions: list[tuple[str, float]] = []
    for name in feature_names:
        if name in OFFLINE_ONLY_FEATURE_NAMES:
            continue
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            continue
        contribution = train_ranker._normalized_feature_value(features, name, scales) * weight
        if positive and contribution > 0:
            contributions.append((name, contribution))
        elif not positive and contribution < 0:
            contributions.append((name, contribution))
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return [f"{name}:{value:.3f}" for name, value in contributions[: max(int(limit), 0)]]


def _review_url(source_url: str, start: object) -> str:
    if not source_url:
        return ""
    try:
        seconds = int(max(float(start), 0.0))
    except (TypeError, ValueError):
        return source_url
    if seconds <= 0:
        return source_url
    split = urlsplit(source_url)
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    query["t"] = _twitch_timestamp(seconds)
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))


def _twitch_timestamp(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    value = ""
    if hours:
        value += f"{hours}h"
    if minutes:
        value += f"{minutes}m"
    value += f"{secs}s"
    return value


def _error_type(label: int, predicted: int) -> str:
    if label == 1 and predicted == 1:
        return "true_positive"
    if label == 0 and predicted == 1:
        return "false_positive"
    if label == 1 and predicted == 0:
        return "false_negative"
    return "true_negative"


def _metric_delta(current: dict, baseline: dict) -> dict[str, float]:
    tracked = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "ndcg_at_10",
        "mrr",
        "ranking_metric_value",
    )
    return {
        key: round(float(current.get(key, 0.0)) - float(baseline.get(key, 0.0)), 3)
        for key in tracked
    }


def _empty_report(
    *,
    profile: str,
    dataset_path: pathlib.Path,
    rows: list[dict],
    selected_rows: list[dict],
    inspection: dict,
    reason: str,
) -> dict[str, object]:
    return {
        "status": "failed",
        "reason": reason,
        "profile": profile,
        "dataset": str(dataset_path),
        "rows": len(rows),
        "selected_rows": len(selected_rows),
        "artifact": {
            "inspection_status": inspection.get("status", "missing"),
            "inspection_reason": inspection.get("reason", ""),
        },
        "metrics": {},
        "groups": [],
        "worst_groups": [],
        "all_negative_false_positive_groups": [],
        "csv_rows": 0,
    }


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "rank",
        "group_rows",
        "group_positive_rows",
        "group_mrr",
        "group_ndcg_at_5",
        "label",
        "predicted",
        "error_type",
        "forced_score",
        "heuristic_score",
        "trained_score",
        "queue_id",
        "job_id",
        "candidate_id",
        "vod_id",
        "candidate_source",
        "content_bucket",
        "gold_grade",
        "window_start",
        "window_end",
        "source_url",
        "review_url",
        "top_positive_features",
        "top_negative_features",
        "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
