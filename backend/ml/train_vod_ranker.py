from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_twitch_training_dataset, train_ranker, training_workflow_state
from services.gold_eval_policy import GOLD_HOLDOUT_SPLIT_FIELD

_MIN_PRIMARY_POSITIVE_ROWS = 6
_MIN_PRIMARY_UNIQUE_VODS = 2
_MIN_PRIMARY_POSITIVE_CLUSTERS = 3
_MIN_PRIMARY_RANKING_GROUPS = 2
_MIN_PRIMARY_MIXED_RANKING_GROUPS = 2
_TARGET_PRIMARY_MIXED_RANKING_GROUPS = 20
_TARGET_NEGATIVE_ROWS_PER_POSITIVE_MAX = 8
_MAX_CANDIDATE_SOURCE_SHARE = 0.40
_SHORT_FORM_MODES = {"off", "supplemental"}


def _load_workflow_state(raw: str | None) -> dict[str, object] | None:
    if not raw:
        return None
    payload = raw.strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --workflow-state-json payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--workflow-state-json must decode to a JSON object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a VOD-first training dataset from Twitch clip labels, optionally "
            "blend in short-form rows as low-weight supplemental data, and train the ranker."
        )
    )
    parser.add_argument(
        "--labels",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_clip_labels.jsonl"),
    )
    parser.add_argument(
        "--dataset-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_vod_training_rows.jsonl"),
    )
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--work-dir", default=str(BACKEND_ROOT / "tmp" / "training_runs"))
    parser.add_argument(
        "--include-short-form",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "real_short_form_training_rows.jsonl"),
        help="Optional supplemental short-form JSONL dataset. Missing files are ignored.",
    )
    parser.add_argument(
        "--short-form-weight",
        type=float,
        default=0.35,
        help="Relative weight multiplier for supplemental short-form rows.",
    )
    parser.add_argument(
        "--short-form-mode",
        choices=sorted(_SHORT_FORM_MODES),
        default="off",
        help=(
            "Use 'off' for VOD-only contrast training. Use 'supplemental' to append "
            "short-form rows as low-weight auxiliary data."
        ),
    )
    parser.add_argument(
        "--include-gold-train",
        default="",
        help=(
            "Optional gold-eval JSONL rows. Only rows with gold_holdout_split=train "
            "are merged into the ranker training dataset; holdout rows are ignored."
        ),
    )
    parser.add_argument(
        "--gold-train-weight",
        type=float,
        default=2.0,
        help="Sample-weight multiplier for gold-train rows merged into ranker training.",
    )
    parser.add_argument(
        "--include-hard-negative-train",
        default="",
        help=(
            "Optional hard-negative JSONL rows, usually exported from ranker holdout diagnosis. "
            "Rows are forced into ranker training with --hard-negative-weight."
        ),
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=2.5,
        help="Sample-weight multiplier for diagnosis-mined hard-negative rows.",
    )
    parser.add_argument(
        "--per-streamer",
        action="store_true",
        help="Train one artifact per broadcaster using that broadcaster's VOD windows as the primary data.",
    )
    parser.add_argument(
        "--workflow-state-path",
        default=None,
        help=(
            "Path to the workflow-state fingerprint JSON. When combined with --workflow-state-json, "
            "training is skipped if the stored fingerprint matches the current state and all artifacts "
            "already exist. After a successful training run the fingerprint is rewritten to this path."
        ),
    )
    parser.add_argument(
        "--workflow-state-json",
        default=None,
        help=(
            "JSON-encoded current workflow state. When omitted, fingerprint gating is disabled. "
            "When provided alongside --workflow-state-path, training uses "
            "training_workflow_state.should_retrain to decide whether to skip."
        ),
    )
    args = parser.parse_args()

    labels_path = pathlib.Path(args.labels)
    labels = _read_jsonl(labels_path)
    if not labels:
        print(f"No Twitch label rows found in {labels_path}")
        return 1

    short_form_path = pathlib.Path(args.include_short_form) if args.include_short_form else None
    short_form_rows = _read_jsonl(short_form_path) if short_form_path and short_form_path.exists() else []
    gold_train_path = pathlib.Path(args.include_gold_train) if args.include_gold_train else None
    gold_train_rows = _read_jsonl(gold_train_path) if gold_train_path and gold_train_path.exists() else []
    hard_negative_path = pathlib.Path(args.include_hard_negative_train) if args.include_hard_negative_train else None
    hard_negative_rows = _read_jsonl(hard_negative_path) if hard_negative_path and hard_negative_path.exists() else []

    if args.per_streamer:
        return _train_per_streamer(
            labels,
            short_form_rows,
            artifact_dir=pathlib.Path(args.artifact_dir),
            work_dir=pathlib.Path(args.work_dir),
            short_form_weight=args.short_form_weight,
            short_form_mode=args.short_form_mode,
            gold_train_rows=gold_train_rows,
            gold_train_weight=args.gold_train_weight,
            hard_negative_rows=hard_negative_rows,
            hard_negative_weight=args.hard_negative_weight,
        )

    artifact_dir = pathlib.Path(args.artifact_dir)
    dataset_out = pathlib.Path(args.dataset_out)
    artifact_path = artifact_dir / f"{args.profile}_ranker.json"

    workflow_state = _load_workflow_state(args.workflow_state_json)
    state_path = pathlib.Path(args.workflow_state_path) if args.workflow_state_path else None
    if workflow_state is not None and state_path is not None:
        if not training_workflow_state.should_retrain(
            labels_path=labels_path,
            dataset_path=dataset_out,
            artifact_path=artifact_path,
            state_path=state_path,
            current_state=workflow_state,
            dependency_paths=[
                path for path in (gold_train_path, hard_negative_path)
                if path is not None and path.exists()
            ],
        ):
            print(
                json.dumps(
                    {
                        "event": "train_vod_ranker_skipped",
                        "reason": "workflow_state_unchanged",
                        "profile": args.profile,
                        "state_path": str(state_path),
                    },
                    ensure_ascii=True,
                )
            )
            return 0

    rows = build_composite_dataset(
        labels,
        profile=args.profile,
        work_dir=pathlib.Path(args.work_dir),
        supplemental_rows=short_form_rows,
        supplemental_weight=args.short_form_weight,
        short_form_mode=args.short_form_mode,
        gold_train_rows=gold_train_rows,
        gold_train_weight=args.gold_train_weight,
        hard_negative_rows=hard_negative_rows,
        hard_negative_weight=args.hard_negative_weight,
    )
    _write_jsonl(dataset_out, rows)
    print(json.dumps(dataset_summary(rows), ensure_ascii=True))
    code = _train_models_for_dataset(
        dataset_out,
        args.profile,
        artifact_dir,
    )
    if code == 0 and workflow_state is not None and state_path is not None:
        training_workflow_state.write_state(state_path, workflow_state)
    return code


def build_composite_dataset(
    labels: list[dict],
    profile: str,
    work_dir: pathlib.Path,
    supplemental_rows: list[dict] | None = None,
    supplemental_weight: float = 0.35,
    short_form_mode: str = "off",
    gold_train_rows: list[dict] | None = None,
    gold_train_weight: float = 2.0,
    hard_negative_rows: list[dict] | None = None,
    hard_negative_weight: float = 2.5,
) -> list[dict]:
    vod_rows = build_twitch_training_dataset.build_dataset(labels, profile, work_dir)
    merged = list(vod_rows)
    if _normalize_short_form_mode(short_form_mode) != "off" and supplemental_rows:
        merged.extend(_prepare_short_form_rows(supplemental_rows, profile, supplemental_weight))
    if gold_train_rows:
        merged = _merge_gold_train_rows(
            merged,
            _prepare_gold_train_rows(gold_train_rows, profile, gold_train_weight),
        )
    if hard_negative_rows:
        merged = _merge_gold_train_rows(
            merged,
            _prepare_hard_negative_rows(hard_negative_rows, profile, hard_negative_weight),
        )
    return merged


def dataset_summary(rows: list[dict]) -> dict:
    summary = build_twitch_training_dataset.dataset_summary(rows)
    by_source: dict[str, int] = defaultdict(int)
    by_label_class: dict[str, int] = defaultdict(int)
    by_supervision_tier: dict[str, int] = defaultdict(int)
    provenance_coverage: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[str(row.get("dataset_source") or "unknown")] += 1
        label_class = str(row.get("label_class") or "")
        if label_class:
            by_label_class[label_class] += 1
        supervision_tier = str(row.get("supervision_tier") or "")
        if supervision_tier:
            by_supervision_tier[supervision_tier] += 1
        provenance = row.get("feature_provenance") or {}
        for key in ("has_llm_semantics", "has_chat_features", "has_vector_similarity", "has_vod_context"):
            if provenance.get(key):
                provenance_coverage[key] += 1
    summary.update(
        {
            "dataset_sources": dict(sorted(by_source.items())),
            "label_classes": dict(sorted(by_label_class.items())),
            "supervision_tiers": dict(sorted(by_supervision_tier.items())),
            "feature_provenance_coverage": dict(sorted(provenance_coverage.items())),
        }
    )
    return summary


def ranker_dataset_quality(rows: list[dict]) -> dict[str, object]:
    primary_rows = [row for row in rows if str(row.get("dataset_source") or "") != "short_form_supplemental"]
    evaluated_rows = primary_rows or list(rows)
    positive_rows = [row for row in evaluated_rows if int(row.get("label", 0)) == 1]
    negative_rows = [row for row in evaluated_rows if int(row.get("label", 0)) == 0]
    unique_vods = {
        str(row.get("vod_id") or row.get("source_url") or "").strip()
        for row in evaluated_rows
        if str(row.get("vod_id") or row.get("source_url") or "").strip()
    }
    positive_clusters = {
        _positive_cluster_key(row)
        for row in positive_rows
        if _positive_cluster_key(row) is not None
    }
    ranking_groups = {
        train_ranker._ranking_group_key(row, "vod_id")
        for row in evaluated_rows
        if str(train_ranker._ranking_group_key(row, "vod_id")).strip()
    }
    mixed_ranking_groups = _mixed_ranking_groups(evaluated_rows, "vod_id")
    contrast_balance = _vod_contrast_balance(evaluated_rows)
    source_mix = _candidate_source_mix(evaluated_rows)

    violations: list[str] = []
    if len(positive_rows) < _MIN_PRIMARY_POSITIVE_ROWS:
        violations.append("primary_positive_rows_below_minimum")
    if len(negative_rows) == 0:
        violations.append("primary_negative_rows_missing")
    if len(unique_vods) < _MIN_PRIMARY_UNIQUE_VODS:
        violations.append("primary_unique_vods_below_minimum")
    if len(positive_clusters) < _MIN_PRIMARY_POSITIVE_CLUSTERS:
        violations.append("primary_positive_clusters_below_minimum")
    if len(ranking_groups) < _MIN_PRIMARY_RANKING_GROUPS:
        violations.append("primary_ranking_groups_below_minimum")
    if len(mixed_ranking_groups) < _MIN_PRIMARY_MIXED_RANKING_GROUPS:
        violations.append("primary_mixed_ranking_groups_below_minimum")
    if source_mix["peak_share"] > _MAX_CANDIDATE_SOURCE_SHARE:
        violations.append("candidate_source_peak_share_above_maximum")

    return {
        "ok": not violations,
        "evaluated_rows": len(evaluated_rows),
        "primary_rows": len(primary_rows),
        "positive_rows": len(positive_rows),
        "negative_rows": len(negative_rows),
        "unique_vods": len(unique_vods),
        "unique_positive_clusters": len(positive_clusters),
        "ranking_groups": len(ranking_groups),
        "mixed_ranking_groups": len(mixed_ranking_groups),
        "thresholds": {
            "positive_rows": _MIN_PRIMARY_POSITIVE_ROWS,
            "unique_vods": _MIN_PRIMARY_UNIQUE_VODS,
            "unique_positive_clusters": _MIN_PRIMARY_POSITIVE_CLUSTERS,
            "ranking_groups": _MIN_PRIMARY_RANKING_GROUPS,
            "mixed_ranking_groups": _MIN_PRIMARY_MIXED_RANKING_GROUPS,
            "candidate_source_peak_share": _MAX_CANDIDATE_SOURCE_SHARE,
        },
        "targets": {
            "mixed_ranking_groups": _TARGET_PRIMARY_MIXED_RANKING_GROUPS,
            "negative_rows_per_positive_max": _TARGET_NEGATIVE_ROWS_PER_POSITIVE_MAX,
        },
        "vod_contrast_balance": contrast_balance,
        "candidate_source_mix": source_mix,
        "violations": violations,
    }


def _candidate_source_mix(rows: list[dict]) -> dict[str, object]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source = str(row.get("candidate_source") or "").strip().lower() or "unknown"
        counts[source] += 1
    total = sum(counts.values()) or 1
    share_by_source = {
        source: round(count / total, 3)
        for source, count in sorted(counts.items())
    }
    peak_source = max(share_by_source, key=share_by_source.get) if share_by_source else ""
    return {
        "rows": total if rows else 0,
        "distinct_sources": len(counts),
        "peak_source": peak_source,
        "peak_share": float(share_by_source.get(peak_source, 0.0)) if peak_source else 0.0,
        "by_source": dict(sorted(counts.items())),
        "share_by_source": share_by_source,
    }


def _vod_contrast_balance(rows: list[dict]) -> dict[str, object]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(train_ranker._ranking_group_key(row, "vod_id") or "").strip()
        if key:
            groups[key].append(row)

    mixed_groups = 0
    worst_ratio = 0.0
    over_limit_groups: list[dict[str, object]] = []
    for key, group_rows in groups.items():
        positive_count = sum(1 for row in group_rows if int(row.get("label", 0)) == 1)
        negative_count = sum(1 for row in group_rows if int(row.get("label", 0)) == 0)
        if positive_count <= 0 or negative_count <= 0:
            continue
        mixed_groups += 1
        ratio = negative_count / positive_count
        worst_ratio = max(worst_ratio, ratio)
        if ratio > _TARGET_NEGATIVE_ROWS_PER_POSITIVE_MAX:
            over_limit_groups.append(
                {
                    "group": key,
                    "positive_rows": positive_count,
                    "negative_rows": negative_count,
                    "negative_rows_per_positive": round(ratio, 3),
                }
            )

    over_limit_groups.sort(
        key=lambda item: (
            -float(item["negative_rows_per_positive"]),
            str(item["group"]),
        )
    )
    return {
        "mixed_ranking_groups": mixed_groups,
        "target_mixed_ranking_groups": _TARGET_PRIMARY_MIXED_RANKING_GROUPS,
        "target_negative_rows_per_positive_max": _TARGET_NEGATIVE_ROWS_PER_POSITIVE_MAX,
        "worst_negative_rows_per_positive": round(worst_ratio, 3),
        "over_limit_group_count": len(over_limit_groups),
        "over_limit_groups": over_limit_groups[:10],
    }


def _train_per_streamer(
    labels: list[dict],
    short_form_rows: list[dict],
    *,
    artifact_dir: pathlib.Path,
    work_dir: pathlib.Path,
    short_form_weight: float,
    short_form_mode: str,
    gold_train_rows: list[dict] | None = None,
    gold_train_weight: float = 2.0,
    hard_negative_rows: list[dict] | None = None,
    hard_negative_weight: float = 2.5,
) -> int:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        broadcaster = str(row.get("broadcaster") or "").strip().lower()
        if broadcaster:
            grouped[broadcaster].append(row)

    if not grouped:
        print("No broadcaster values found in Twitch labels.")
        return 1

    artifact_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for broadcaster, broadcaster_labels in sorted(grouped.items()):
        dataset_rows = build_composite_dataset(
            broadcaster_labels,
            profile=broadcaster,
            work_dir=work_dir / broadcaster,
            supplemental_rows=_filter_short_form_rows(short_form_rows, broadcaster),
            supplemental_weight=short_form_weight,
            short_form_mode=short_form_mode,
            gold_train_rows=_filter_gold_train_rows(gold_train_rows or [], broadcaster),
            gold_train_weight=gold_train_weight,
            hard_negative_rows=_filter_gold_train_rows(hard_negative_rows or [], broadcaster),
            hard_negative_weight=hard_negative_weight,
        )
        dataset_out = artifact_dir / f"{broadcaster}_vod_training_rows.jsonl"
        _write_jsonl(dataset_out, dataset_rows)
        print(
            json.dumps(
                {
                    "profile": broadcaster,
                    "dataset": str(dataset_out),
                    "summary": dataset_summary(dataset_rows),
                },
                ensure_ascii=True,
            )
        )
        code = _train_models_for_dataset(dataset_out, broadcaster, artifact_dir)
        if code != 0:
            failures += 1
    return 0 if failures == 0 else 1


def _train_models_for_dataset(
    dataset_path: pathlib.Path,
    profile: str,
    artifact_dir: pathlib.Path,
) -> int:
    evaluate_boundary_refinement, train_boundary_ranker = _boundary_training_modules()
    rows = _read_jsonl(dataset_path)
    quality = ranker_dataset_quality(rows)
    if not quality["ok"]:
        print(
            json.dumps(
                {
                    "profile": profile,
                    "ranker": {
                        "trained": False,
                        "reason": "insufficient_dataset_quality",
                        "quality": quality,
                    },
                },
                ensure_ascii=True,
            )
        )
        return 1

    code = train_ranker.train_from_dataset(dataset_path, profile, artifact_dir)
    if code != 0:
        return code

    boundary_rows = [row for row in rows if evaluate_boundary_refinement._has_boundary_reference(row)]
    if not boundary_rows:
        print(
            json.dumps(
                {
                    "profile": profile,
                    "boundary_ranker": {
                        "trained": False,
                        "reason": "no_boundary_reference_rows",
                    },
                },
                ensure_ascii=True,
            )
        )
        return 0

    print(
        json.dumps(
            {
                "profile": profile,
                "boundary_dataset": {
                    "rows": len(rows),
                    "boundary_rows": len(boundary_rows),
                    "metrics": evaluate_boundary_refinement._boundary_metrics(boundary_rows),
                },
            },
            ensure_ascii=True,
        )
    )
    return train_boundary_ranker.train_from_rows(rows, profile, artifact_dir)


def _boundary_training_modules():
    from ml import evaluate_boundary_refinement, train_boundary_ranker

    return evaluate_boundary_refinement, train_boundary_ranker


def _prepare_short_form_rows(
    rows: list[dict],
    profile: str,
    supplemental_weight: float,
) -> list[dict]:
    weight = max(0.0, min(float(supplemental_weight), 1.0))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", 0)) not in {0, 1}:
            continue
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        copied["dataset_source"] = "short_form_supplemental"
        if not copied.get("label_class"):
            copied["label_class"] = "short_form_positive" if int(copied["label"]) == 1 else "short_form_negative"
        copied["supervision_tier"] = str(copied.get("supervision_tier") or "short_form_discovered")
        copied["supervision_reason"] = str(copied.get("supervision_reason") or "manifest_label")
        copied["ordinal_label"] = int(copied.get("ordinal_label", 3 if int(copied["label"]) == 1 else 0))
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        creator = str(copied.get("creator_id") or copied.get("streamer") or "").strip().lower()
        copied["profile_hint"] = creator or profile
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        prepared.append(copied)
    return prepared


def _prepare_gold_train_rows(
    rows: list[dict],
    profile: str,
    gold_train_weight: float,
) -> list[dict]:
    weight = max(0.0, float(gold_train_weight))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", -1)) not in {0, 1}:
            continue
        if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() != "train":
            continue
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        copied["dataset_source"] = "gold_eval_train"
        copied["label_source"] = "gold_eval"
        copied["supervision_tier"] = "gold_eval"
        copied["supervision_reason"] = str(copied.get("supervision_reason") or "manual_gold_eval_train")
        if not copied.get("label_class"):
            copied["label_class"] = "gold_positive" if int(copied["label"]) == 1 else "gold_negative"
        copied["ordinal_label"] = int(copied.get("ordinal_label", copied.get("gold_grade", 3 if int(copied["label"]) == 1 else 0)))
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        creator = str(copied.get("creator_id") or copied.get("streamer") or "").strip().lower()
        copied["profile_hint"] = creator or profile
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        prepared.append(copied)
    return prepared


def _prepare_hard_negative_rows(
    rows: list[dict],
    profile: str,
    hard_negative_weight: float,
) -> list[dict]:
    weight = max(0.0, float(hard_negative_weight))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", -1)) != 0:
            continue
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        copied["label"] = 0
        copied["dataset_source"] = "gold_eval_hard_negative_train"
        copied["label_source"] = "gold_eval"
        copied["supervision_tier"] = "gold_eval"
        copied["supervision_reason"] = str(copied.get("supervision_reason") or "ranker_diagnosis_false_positive")
        copied["label_class"] = "gold_hard_negative"
        copied["ordinal_label"] = int(copied.get("ordinal_label", copied.get("gold_grade", 0)) or 0)
        copied[GOLD_HOLDOUT_SPLIT_FIELD] = "train"
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        creator = str(copied.get("creator_id") or copied.get("streamer") or "").strip().lower()
        copied["profile_hint"] = creator or profile
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        prepared.append(copied)
    return prepared


def _merge_gold_train_rows(base_rows: list[dict], gold_rows: list[dict]) -> list[dict]:
    if not gold_rows:
        return list(base_rows)
    merged: list[dict] = []
    index_by_key: dict[tuple[str, ...], int] = {}
    for row in base_rows:
        key = _training_row_key(row)
        if key:
            index_by_key[key] = len(merged)
        merged.append(row)
    for row in gold_rows:
        key = _training_row_key(row)
        if key and key in index_by_key:
            merged[index_by_key[key]] = row
            continue
        if key:
            index_by_key[key] = len(merged)
        merged.append(row)
    return merged


def _filter_short_form_rows(rows: list[dict], broadcaster: str) -> list[dict]:
    if not rows:
        return []
    normalized = broadcaster.strip().lower()
    filtered = [
        row for row in rows
        if str(row.get("creator_id") or row.get("streamer") or "").strip().lower() == normalized
    ]
    return filtered


def _filter_gold_train_rows(rows: list[dict], broadcaster: str) -> list[dict]:
    if not rows:
        return []
    normalized = broadcaster.strip().lower()
    return [
        row
        for row in rows
        if str(row.get("creator_id") or row.get("streamer") or row.get("profile_hint") or "").strip().lower()
        in {"", normalized}
    ]


def _training_row_key(row: dict) -> tuple[str, ...] | None:
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
    return None


def _stable_time_value(value) -> str:
    try:
        return f"{float(value or 0.0):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _normalize_short_form_mode(mode: str) -> str:
    normalized = str(mode or "off").strip().lower()
    if normalized not in _SHORT_FORM_MODES:
        raise ValueError(f"Unsupported short-form mode: {mode!r}")
    return normalized


def _mixed_ranking_groups(rows: list[dict], split_unit: str) -> set[str]:
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        key = train_ranker._ranking_group_key(row, split_unit)
        if not str(key).strip():
            continue
        labels_by_group[key].add(int(row.get("label", 0)))
    return {
        group
        for group, labels in labels_by_group.items()
        if 0 in labels and 1 in labels
    }


def _positive_cluster_key(row: dict) -> tuple[str, str] | None:
    if int(row.get("label", 0)) != 1:
        return None
    vod_id = str(row.get("vod_id") or row.get("source_url") or "").strip()
    if not vod_id:
        return None
    clip_ids = sorted(str(item) for item in (row.get("clip_ids") or []) if str(item).strip())
    if clip_ids:
        return vod_id, ",".join(clip_ids)
    candidate_id = str(row.get("candidate_id") or "").strip()
    if not candidate_id:
        return None
    return vod_id, f"candidate:{candidate_id}"


def _read_jsonl(path: pathlib.Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows), encoding="utf-8")
    tmp_path.replace(path)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
