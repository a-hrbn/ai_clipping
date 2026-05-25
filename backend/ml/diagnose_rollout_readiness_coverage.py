from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Iterable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import train_ranker
from services.gold_eval_policy import GOLD_HOLDOUT_SPLIT_FIELD


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose rollout-readiness coverage for a ranker dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--holdout-dataset", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    report = diagnose_coverage(
        dataset_path=pathlib.Path(args.dataset),
        profile=args.profile,
        holdout_dataset_path=pathlib.Path(args.holdout_dataset) if args.holdout_dataset else None,
    )
    if args.report_out:
        path = pathlib.Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report.get("status") == "ok" else 1


def diagnose_coverage(
    *,
    dataset_path: pathlib.Path,
    profile: str,
    holdout_dataset_path: pathlib.Path | None = None,
) -> dict[str, object]:
    raw_rows = _read_jsonl(dataset_path)
    if not raw_rows:
        return {
            "status": "failed",
            "reason": "no_dataset_rows",
            "dataset_path": str(dataset_path),
            "profile": profile,
        }

    prepared_rows, policy = train_ranker._prepare_ranker_training_rows(raw_rows)
    if not prepared_rows:
        return {
            "status": "failed",
            "reason": "no_rows_after_training_policy",
            "dataset_path": str(dataset_path),
            "profile": profile,
            "input_rows": len(raw_rows),
            "training_row_policy": policy,
        }

    split = train_ranker._split_rows(prepared_rows)
    train_rows = list(split.get("train") or [])
    validation_rows = list(split.get("validation") or [])
    split_unit = str(split.get("unit") or "")
    group_summaries = _group_summaries(
        raw_rows=raw_rows,
        prepared_rows=prepared_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        split_unit=split_unit,
    )
    holdout_rows = _read_jsonl(holdout_dataset_path) if holdout_dataset_path else []
    holdout_summary = _holdout_overlap_summary(group_summaries, holdout_rows, split_unit) if holdout_rows else {}
    targets = _targets(profile, available_groups=len(group_summaries))
    readiness = _readiness_summary(
        prepared_rows=prepared_rows,
        validation_rows=validation_rows,
        split_unit=split_unit,
        targets=targets,
    )
    diagnosis = _diagnosis(
        readiness=readiness,
        group_summaries=group_summaries,
        holdout_summary=holdout_summary,
        split=split,
        targets=targets,
    )
    return {
        "status": "ok",
        "dataset_path": str(dataset_path),
        "profile": profile,
        "summary": {
            "input_rows": len(raw_rows),
            "rows_after_training_policy": len(prepared_rows),
            "policy_removed_rows": len(raw_rows) - len(prepared_rows),
            "positive_rows_after_policy": sum(1 for row in prepared_rows if int(row.get("label", 0)) == 1),
            "negative_rows_after_policy": sum(1 for row in prepared_rows if int(row.get("label", 0)) == 0),
            "split_strategy": str(split.get("strategy") or ""),
            "split_unit": split_unit,
            "split_leakage_risk": str(split.get("leakage_risk") or ""),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_groups": len({train_ranker._ranking_group_key(row, split_unit) for row in train_rows}),
            "validation_groups": readiness["validation_groups"],
            "available_groups_after_policy": len(group_summaries),
            "raw_groups": len({_row_group(row, split_unit) for row in raw_rows}),
        },
        "targets": targets,
        "readiness": readiness,
        "diagnosis": diagnosis,
        "training_row_policy": policy,
        "holdout_overlap": holdout_summary,
        "groups": group_summaries,
    }


def _read_jsonl(path: pathlib.Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("label") in {0, 1}:
            rows.append(row)
    return rows


def _targets(profile: str, *, available_groups: int | None = None) -> dict[str, object]:
    return train_ranker.rollout_readiness_targets(profile, available_groups=available_groups)


def _readiness_summary(
    *,
    prepared_rows: list[dict],
    validation_rows: list[dict],
    split_unit: str,
    targets: dict[str, object],
) -> dict[str, object]:
    positive_rows = sum(1 for row in prepared_rows if int(row.get("label", 0)) == 1)
    validation_groups = len(set(train_ranker._ranking_group_labels(validation_rows, split_unit)))
    training_target = int(targets.get("training_rows", 0))
    positive_target = int(targets.get("positive_rows", 0))
    validation_group_target = int(targets.get("validation_groups", 0))
    failures: list[str] = []
    if training_target and len(prepared_rows) < training_target:
        failures.append("training_rows_below_rollout_target")
    if positive_target and positive_rows < positive_target:
        failures.append("positive_rows_below_rollout_target")
    if validation_group_target and validation_groups < validation_group_target:
        failures.append("validation_groups_below_rollout_target")
    return {
        "passed": not failures,
        "failures": failures,
        "training_rows": len(prepared_rows),
        "training_rows_needed": max(0, training_target - len(prepared_rows)),
        "positive_rows": positive_rows,
        "positive_rows_needed": max(0, positive_target - positive_rows),
        "validation_groups": validation_groups,
        "validation_groups_needed": max(0, validation_group_target - validation_groups),
    }


def _group_summaries(
    *,
    raw_rows: list[dict],
    prepared_rows: list[dict],
    train_rows: list[dict],
    validation_rows: list[dict],
    split_unit: str,
) -> list[dict[str, object]]:
    train_ids = {_row_identity(row) for row in train_rows}
    validation_ids = {_row_identity(row) for row in validation_rows}
    raw_by_group = _rows_by_group(raw_rows, split_unit)
    prepared_by_group = _rows_by_group(prepared_rows, split_unit)
    groups: list[dict[str, object]] = []
    for group_id in sorted(set(raw_by_group) | set(prepared_by_group)):
        rows = prepared_by_group.get(group_id, [])
        raw_group_rows = raw_by_group.get(group_id, [])
        train_group_rows = [row for row in rows if _row_identity(row) in train_ids]
        validation_group_rows = [row for row in rows if _row_identity(row) in validation_ids]
        split_scope = "validation" if validation_group_rows else "train" if train_group_rows else "policy_removed"
        groups.append(
            {
                "group_id": group_id,
                "split": split_scope,
                "rows": len(rows),
                "raw_rows": len(raw_group_rows),
                "policy_removed_rows": max(0, len(raw_group_rows) - len(rows)),
                "train_rows": len(train_group_rows),
                "validation_rows": len(validation_group_rows),
                "positives": _label_count(rows, 1),
                "negatives": _label_count(rows, 0),
                "raw_positives": _label_count(raw_group_rows, 1),
                "raw_negatives": _label_count(raw_group_rows, 0),
                "vod_count": len(_values(rows or raw_group_rows, "vod_id")),
                "source_url_count": len(_values(rows or raw_group_rows, "source_url")),
                "gold_train_rows": _count_value(rows, GOLD_HOLDOUT_SPLIT_FIELD, "train"),
                "gold_holdout_rows": _count_value(rows, GOLD_HOLDOUT_SPLIT_FIELD, "holdout"),
                "candidate_sources": _top_counts(rows or raw_group_rows, "candidate_source"),
                "source_buckets": _top_counts(rows or raw_group_rows, "source_bucket"),
                "content_buckets": _top_counts(rows or raw_group_rows, "content_bucket"),
                "platforms": _top_counts(rows or raw_group_rows, "publish_platform"),
                "window_start_min": _safe_min(row.get("window_start") for row in rows),
                "window_end_max": _safe_max(row.get("window_end") for row in rows),
                "sample_candidate_ids": [str(row.get("candidate_id") or "") for row in rows[:5]],
                "missing_reason": _group_missing_reason(
                    rows=rows,
                    raw_rows=raw_group_rows,
                    validation_rows=validation_group_rows,
                ),
            }
        )
    return sorted(groups, key=lambda item: (str(item["split"]) != "validation", -int(item["rows"]), str(item["group_id"])))


def _holdout_overlap_summary(
    group_summaries: list[dict[str, object]],
    holdout_rows: list[dict],
    split_unit: str,
) -> dict[str, object]:
    train_groups = {str(group["group_id"]) for group in group_summaries if int(group.get("rows") or 0) > 0}
    holdout_by_group = _rows_by_group(holdout_rows, split_unit)
    holdout_groups = set(holdout_by_group)
    holdout_only = sorted(holdout_groups - train_groups)
    shared = sorted(holdout_groups & train_groups)
    return {
        "holdout_rows": len(holdout_rows),
        "holdout_positive_rows": _label_count(holdout_rows, 1),
        "holdout_negative_rows": _label_count(holdout_rows, 0),
        "holdout_groups": len(holdout_groups),
        "shared_groups": len(shared),
        "holdout_only_groups": len(holdout_only),
        "holdout_only_positive_rows": sum(_label_count(rows, 1) for group, rows in holdout_by_group.items() if group in holdout_only),
        "holdout_only_sample_groups": holdout_only[:20],
    }


def _diagnosis(
    *,
    readiness: dict[str, object],
    group_summaries: list[dict[str, object]],
    holdout_summary: dict[str, object],
    split: dict[str, object],
    targets: dict[str, object],
) -> dict[str, object]:
    available_groups = sum(1 for group in group_summaries if int(group.get("rows") or 0) > 0)
    validation_groups = int(readiness.get("validation_groups") or 0)
    target_validation_groups = int(targets.get("validation_groups") or 0)
    holdout_only_groups = int(holdout_summary.get("holdout_only_groups") or 0)
    recommendations: list[str] = []
    blockers: list[str] = []
    if int(readiness.get("training_rows_needed") or 0) > 0:
        blockers.append("training_rows")
        recommendations.append("add_more_policy_surviving_training_rows")
    if int(readiness.get("positive_rows_needed") or 0) > 0:
        blockers.append("positive_rows")
        recommendations.append("add_more_outcome_positive_training_rows")
    if target_validation_groups and available_groups < target_validation_groups:
        blockers.append("available_validation_groups")
        recommendations.append("collect_more_distinct_vod_groups_for_training")
    elif target_validation_groups and validation_groups < target_validation_groups:
        blockers.append("split_validation_groups")
        recommendations.append("increase_group_holdout_validation_group_target")
    if holdout_only_groups:
        recommendations.append("promote_or_resplit_holdout_only_outcome_groups_without_leaking_final_eval")
    primary = "ready" if not blockers else blockers[0]
    return {
        "primary_blocker": primary,
        "blockers": blockers,
        "split_strategy": str(split.get("strategy") or ""),
        "split_unit": str(split.get("unit") or ""),
        "target_mode": str(targets.get("mode") or ""),
        "available_groups_after_policy": available_groups,
        "available_groups_needed": max(0, target_validation_groups - available_groups),
        "validation_groups_unused_by_split": max(0, available_groups - validation_groups),
        "holdout_only_groups": holdout_only_groups,
        "recommendations": recommendations,
    }


def _rows_by_group(rows: Iterable[dict], split_unit: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_row_group(row, split_unit)].append(row)
    return dict(grouped)


def _row_group(row: dict, split_unit: str) -> str:
    return train_ranker._ranking_group_key(row, split_unit)


def _row_identity(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("candidate_id") or ""),
        str(row.get("job_id") or ""),
        str(row.get("queue_id") or ""),
    )


def _label_count(rows: Iterable[dict], label: int) -> int:
    return sum(1 for row in rows if int(row.get("label", 0)) == int(label))


def _values(rows: Iterable[dict], key: str) -> set[str]:
    return {str(row.get(key) or "").strip() for row in rows if str(row.get(key) or "").strip()}


def _count_value(rows: Iterable[dict], key: str, value: str) -> int:
    expected = str(value).strip().lower()
    return sum(1 for row in rows if str(row.get(key) or "").strip().lower() == expected)


def _top_counts(rows: Iterable[dict], key: str, limit: int = 8) -> dict[str, int]:
    counter = Counter(str(row.get(key) or "unknown").strip().lower() or "unknown" for row in rows)
    return dict(counter.most_common(limit))


def _safe_float_values(values: Iterable[object]) -> list[float]:
    floats: list[float] = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            floats.append(float(value))
        except (TypeError, ValueError):
            continue
    return floats


def _safe_min(values: Iterable[object]) -> float | None:
    floats = _safe_float_values(values)
    return round(min(floats), 3) if floats else None


def _safe_max(values: Iterable[object]) -> float | None:
    floats = _safe_float_values(values)
    return round(max(floats), 3) if floats else None


def _group_missing_reason(*, rows: list[dict], raw_rows: list[dict], validation_rows: list[dict]) -> str:
    if validation_rows:
        positives = _label_count(validation_rows, 1)
        negatives = _label_count(validation_rows, 0)
        if positives and negatives:
            return ""
        if positives:
            return "validation_group_has_no_negative_rows"
        if negatives:
            return "validation_group_has_no_positive_rows"
        return "validation_group_empty"
    if not rows and raw_rows:
        return "removed_by_training_policy"
    if rows:
        return "not_selected_for_validation_by_current_split"
    return "missing_rows"


if __name__ == "__main__":
    raise SystemExit(main())
