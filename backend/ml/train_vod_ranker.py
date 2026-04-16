from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_twitch_training_dataset, train_ranker

_MIN_PRIMARY_POSITIVE_ROWS = 6
_MIN_PRIMARY_UNIQUE_VODS = 2
_MIN_PRIMARY_POSITIVE_CLUSTERS = 3
_MIN_PRIMARY_RANKING_GROUPS = 2


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
        "--per-streamer",
        action="store_true",
        help="Train one artifact per broadcaster using that broadcaster's VOD windows as the primary data.",
    )
    args = parser.parse_args()

    labels_path = pathlib.Path(args.labels)
    labels = _read_jsonl(labels_path)
    if not labels:
        print(f"No Twitch label rows found in {labels_path}")
        return 1

    short_form_path = pathlib.Path(args.include_short_form) if args.include_short_form else None
    short_form_rows = _read_jsonl(short_form_path) if short_form_path and short_form_path.exists() else []

    if args.per_streamer:
        return _train_per_streamer(
            labels,
            short_form_rows,
            artifact_dir=pathlib.Path(args.artifact_dir),
            work_dir=pathlib.Path(args.work_dir),
            short_form_weight=args.short_form_weight,
        )

    rows = build_composite_dataset(
        labels,
        profile=args.profile,
        work_dir=pathlib.Path(args.work_dir),
        supplemental_rows=short_form_rows,
        supplemental_weight=args.short_form_weight,
    )
    dataset_out = pathlib.Path(args.dataset_out)
    _write_jsonl(dataset_out, rows)
    print(json.dumps(dataset_summary(rows), ensure_ascii=True))
    return _train_models_for_dataset(
        dataset_out,
        args.profile,
        pathlib.Path(args.artifact_dir),
    )


def build_composite_dataset(
    labels: list[dict],
    profile: str,
    work_dir: pathlib.Path,
    supplemental_rows: list[dict] | None = None,
    supplemental_weight: float = 0.35,
) -> list[dict]:
    vod_rows = build_twitch_training_dataset.build_dataset(labels, profile, work_dir)
    if not supplemental_rows:
        return vod_rows
    merged = list(vod_rows)
    merged.extend(_prepare_short_form_rows(supplemental_rows, profile, supplemental_weight))
    return merged


def dataset_summary(rows: list[dict]) -> dict:
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
    return {
        "rows": len(rows),
        "dataset_sources": dict(sorted(by_source.items())),
        "label_classes": dict(sorted(by_label_class.items())),
        "supervision_tiers": dict(sorted(by_supervision_tier.items())),
        "feature_provenance_coverage": dict(sorted(provenance_coverage.items())),
    }


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

    return {
        "ok": not violations,
        "evaluated_rows": len(evaluated_rows),
        "primary_rows": len(primary_rows),
        "positive_rows": len(positive_rows),
        "negative_rows": len(negative_rows),
        "unique_vods": len(unique_vods),
        "unique_positive_clusters": len(positive_clusters),
        "ranking_groups": len(ranking_groups),
        "thresholds": {
            "positive_rows": _MIN_PRIMARY_POSITIVE_ROWS,
            "unique_vods": _MIN_PRIMARY_UNIQUE_VODS,
            "unique_positive_clusters": _MIN_PRIMARY_POSITIVE_CLUSTERS,
            "ranking_groups": _MIN_PRIMARY_RANKING_GROUPS,
        },
        "violations": violations,
    }


def _train_per_streamer(
    labels: list[dict],
    short_form_rows: list[dict],
    *,
    artifact_dir: pathlib.Path,
    work_dir: pathlib.Path,
    short_form_weight: float,
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


def _filter_short_form_rows(rows: list[dict], broadcaster: str) -> list[dict]:
    if not rows:
        return []
    normalized = broadcaster.strip().lower()
    filtered = [
        row for row in rows
        if str(row.get("creator_id") or row.get("streamer") or "").strip().lower() == normalized
    ]
    return filtered


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
