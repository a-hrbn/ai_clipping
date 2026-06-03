from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import audit_clipfinding_artifacts, build_proposal_training_rows, materialize_proposal_training_debug, train_proposal_scorer
from ml.row_identity import read_jsonl, write_json, write_jsonl
from services import context_patterns

DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_STORAGE_DIR = BACKEND_ROOT / "tmp" / "storage"
DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize proposal debug data from cached positive VOD transcripts, "
            "build proposal training rows, then train the proposal scorer."
        )
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--storage-dir", type=pathlib.Path, default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--positive-rows", type=pathlib.Path, default=None)
    parser.add_argument(
        "--holdout-policy-rows",
        type=pathlib.Path,
        default=None,
        help="Optional JSONL policy rows; VOD ids marked holdout are excluded from proposal training.",
    )
    parser.add_argument("--training-rows", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--summary-output", type=pathlib.Path, default=None)
    parser.add_argument("--false-negative-output", type=pathlib.Path, default=None)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--vod-cache-root", type=pathlib.Path, default=DEFAULT_VOD_CACHE_ROOT)
    parser.add_argument("--materialized-storage-dir", type=pathlib.Path, default=None)
    parser.add_argument("--materialize-max-vods", type=int, default=20)
    parser.add_argument("--materialize-refresh", action="store_true")
    parser.add_argument("--durations-sec", default="12,20,30,45,60")
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--max-candidates-per-vod", type=int, default=5000)
    parser.add_argument("--positive-overlap-ratio", type=float, default=0.45)
    parser.add_argument("--near-miss-sec", type=float, default=180.0)
    parser.add_argument("--background-negatives-per-positive", type=int, default=4)
    parser.add_argument("--max-background-negatives-per-vod", type=int, default=120)
    parser.add_argument("--context-pattern-output", type=pathlib.Path, default=None)
    parser.add_argument(
        "--context-pattern-rows",
        type=pathlib.Path,
        action="append",
        default=None,
        help=(
            "JSONL rows used only for context/pattern discovery. Defaults to "
            "cross-platform positives, repost manifest, and match diagnostics."
        ),
    )
    parser.add_argument("--context-pattern-min-support", type=int, default=2)
    parser.add_argument("--context-pattern-max-patterns", type=int, default=48)
    parser.add_argument("--context-pattern-cluster-similarity", type=float, default=0.22)
    parser.add_argument(
        "--context-pattern-semantic-cluster-similarity",
        type=float,
        default=context_patterns.SEMANTIC_CLUSTER_SIMILARITY,
    )
    parser.add_argument("--date-from", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--date-to", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    positive_rows = args.positive_rows or DEFAULT_ARTIFACT_DIR / f"{args.profile}_cross_platform_fingerprint_rows.jsonl"
    training_rows = args.training_rows or DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_training_rows.jsonl"
    output = args.output or DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_scorer.json"
    summary_output = args.summary_output or DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_training_pipeline_summary.json"
    false_negative_output = (
        args.false_negative_output
        or DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_false_negative_training_rows.jsonl"
    )
    context_pattern_output = args.context_pattern_output or output.parent / f"{args.profile}_context_patterns.json"
    materialized_storage_dir = args.materialized_storage_dir or args.storage_dir / "_proposal_training_materialized"
    holdout_vod_ids = _holdout_vod_ids(args.holdout_policy_rows)
    filtered_positive_rows = materialized_storage_dir / "_holdout_filtered_cross_platform_rows.jsonl"
    positive_rows_for_training, positive_filter_summary = _write_holdout_filtered_positive_rows(
        positive_rows,
        filtered_positive_rows,
        holdout_vod_ids,
    )

    context_pattern_rows, context_pattern_input_summary = _load_context_pattern_rows(
        _context_pattern_input_paths(
            profile=args.profile,
            artifact_dir=output.parent,
            positive_rows=positive_rows_for_training,
            explicit_paths=args.context_pattern_rows,
        )
    )
    context_pattern_store = context_patterns.write_pattern_store(
        context_pattern_rows,
        profile=args.profile,
        output=context_pattern_output,
        min_support=max(int(args.context_pattern_min_support or 0), 1),
        max_patterns=max(int(args.context_pattern_max_patterns or 0), 0),
        cluster_similarity=float(args.context_pattern_cluster_similarity),
        semantic_cluster_similarity=float(args.context_pattern_semantic_cluster_similarity),
    )

    materialize_summary = materialize_proposal_training_debug.materialize_proposal_debug(
        profile=args.profile,
        cross_platform_rows=positive_rows_for_training,
        storage_dir=materialized_storage_dir,
        vod_cache_root=args.vod_cache_root,
        max_vods=max(int(args.materialize_max_vods or 0), 0),
        durations=materialize_proposal_training_debug._parse_float_list(args.durations_sec),
        stride_sec=args.stride_sec,
        max_candidates_per_vod=max(int(args.max_candidates_per_vod or 0), 0),
        refresh=bool(args.materialize_refresh),
    )

    build_proposal_training_rows.main(
        [
            "--profile",
            args.profile,
            "--cross-platform-rows",
            str(positive_rows_for_training),
            "--holdout-policy-rows",
            str(args.holdout_policy_rows or ""),
            "--storage-dir",
            str(materialized_storage_dir),
            "--output",
            str(training_rows),
            "--summary-output",
            str(DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_training_summary.json"),
            "--durations-sec",
            args.durations_sec,
            "--stride-sec",
            str(args.stride_sec),
            "--max-candidates-per-vod",
            str(args.max_candidates_per_vod),
            "--positive-overlap-ratio",
            str(args.positive_overlap_ratio),
            "--near-miss-sec",
            str(args.near_miss_sec),
            "--background-negatives-per-positive",
            str(args.background_negatives_per_positive),
            "--max-background-negatives-per-vod",
            str(args.max_background_negatives_per_vod),
        ]
    )

    debug_rows = train_proposal_scorer.collect_training_rows(args.storage_dir, positive_rows_path=positive_rows_for_training)
    debug_rows_before_holdout_filter = len(debug_rows)
    debug_rows = [row for row in debug_rows if str(row.get("vod_id") or "").strip() not in holdout_vod_ids]
    false_negative_rows = [
        row
        for row in debug_rows
        if str(row.get("label_source") or "") == "cross_platform_proposal_false_negative"
    ]
    write_jsonl(false_negative_output, false_negative_rows)
    rows = list(debug_rows)
    rows.extend(train_proposal_scorer.collect_training_rows_from_files([training_rows]))
    artifact = train_proposal_scorer.train_logistic_regression_model(
        rows,
        profile=args.profile,
        min_rows=args.min_rows,
        l2=args.l2,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    audit_dir = output.parent
    audit_output = audit_dir / f"{args.profile}_clipfinding_artifact_audit.json"
    summary = {
        "profile": args.profile,
        "positive_rows": str(positive_rows),
        "positive_rows_for_training": str(positive_rows_for_training),
        "holdout_policy_rows": str(args.holdout_policy_rows or ""),
        "holdout_filter_summary": {
            **positive_filter_summary,
            "holdout_vod_ids": sorted(holdout_vod_ids),
            "debug_rows_before_holdout_filter": debug_rows_before_holdout_filter,
            "debug_rows_after_holdout_filter": len(debug_rows),
        },
        "storage_dir": str(args.storage_dir),
        "materialized_storage_dir": str(materialized_storage_dir),
        "training_rows": str(training_rows),
        "false_negative_rows": str(false_negative_output),
        "false_negative_training_rows": len(false_negative_rows),
        "context_patterns": str(context_pattern_output),
        "context_pattern_summary": {
            "patterns": len(context_pattern_store.get("patterns") or []),
            "usable_rows": context_pattern_store.get("usable_rows"),
            "source_rows": context_pattern_store.get("source_rows"),
            "semantic_matching_enabled": context_pattern_store.get("semantic_matching_enabled"),
            "embedding_model": context_pattern_store.get("embedding_model"),
            "input_sources": context_pattern_input_summary,
            "labeling_note": "Context rows are used only for pattern discovery; Proposal labels still come from positive/debug training rows.",
        },
        "output": str(output),
        "materialize_summary": materialize_summary,
        "training_summary": artifact.get("training_summary") or {},
        "artifact_audit_output": str(audit_output),
    }
    write_json(summary_output, summary)
    artifact_audit = audit_clipfinding_artifacts.build_audit(
        profile=args.profile,
        artifact_dir=audit_dir,
    )
    audit_output.write_text(json.dumps(artifact_audit, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    summary["artifact_audit"] = artifact_audit.get("summary") or {}
    write_json(summary_output, summary)
    print(
        json.dumps(
            {
                "event": "proposal_training_pipeline_done",
                "summary_output": str(summary_output),
                **summary["training_summary"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _context_pattern_input_paths(
    *,
    profile: str,
    artifact_dir: pathlib.Path,
    positive_rows: pathlib.Path,
    explicit_paths: list[pathlib.Path] | None,
) -> list[pathlib.Path]:
    if explicit_paths:
        return [pathlib.Path(path) for path in explicit_paths]
    return [
        positive_rows,
        artifact_dir / f"{profile}_cross_platform_reposts.jsonl",
        artifact_dir / f"{profile}_cross_platform_fingerprint_diagnostics.jsonl",
    ]


def _load_context_pattern_rows(paths: list[pathlib.Path]) -> tuple[list[dict], dict[str, object]]:
    merged_by_key: dict[str, dict] = {}
    summary_sources: list[dict[str, object]] = []
    for path in paths:
        expanded = pathlib.Path(path).expanduser()
        rows = read_jsonl(expanded) if expanded.exists() else []
        loaded = 0
        for row in rows:
            normalized = _normalize_context_pattern_row(row, source_path=expanded)
            key = _context_pattern_row_key(normalized)
            if not key:
                continue
            existing = merged_by_key.setdefault(key, {})
            _merge_context_pattern_row(existing, normalized)
            loaded += 1
        summary_sources.append(
            {
                "path": str(expanded),
                "exists": expanded.exists(),
                "rows": len(rows),
                "loaded_rows": loaded,
            }
        )
    return list(merged_by_key.values()), {
        "sources": summary_sources,
        "deduped_rows": len(merged_by_key),
    }


def _holdout_vod_ids(path: pathlib.Path | None) -> set[str]:
    if path is None or not pathlib.Path(path).exists():
        return set()
    vod_ids: set[str] = set()
    for row in read_jsonl(pathlib.Path(path)):
        split = str(row.get("gold_holdout_split") or row.get("outcome_split") or row.get("dataset_role") or "").lower()
        source = str(row.get("dataset_source") or row.get("supervision_tier") or "").lower()
        if "holdout" not in split and "holdout" not in source:
            continue
        vod_id = str(row.get("vod_id") or "").strip()
        if vod_id:
            vod_ids.add(vod_id)
    return vod_ids


def _write_holdout_filtered_positive_rows(
    source: pathlib.Path,
    output: pathlib.Path,
    holdout_vod_ids: set[str],
) -> tuple[pathlib.Path, dict[str, object]]:
    rows = read_jsonl(source)
    if not holdout_vod_ids:
        return source, {
            "positive_rows_before_holdout_filter": len(rows),
            "positive_rows_after_holdout_filter": len(rows),
            "positive_rows_removed_by_holdout": 0,
            "filtered_positive_rows": str(source),
        }
    filtered = [row for row in rows if str(row.get("vod_id") or "").strip() not in holdout_vod_ids]
    write_jsonl(output, filtered)
    return output, {
        "positive_rows_before_holdout_filter": len(rows),
        "positive_rows_after_holdout_filter": len(filtered),
        "positive_rows_removed_by_holdout": len(rows) - len(filtered),
        "filtered_positive_rows": str(output),
    }


def _normalize_context_pattern_row(row: dict, *, source_path: pathlib.Path) -> dict:
    normalized = dict(row)
    source_url = str(
        normalized.get("source_url")
        or normalized.get("repost_source_url")
        or normalized.get("url")
        or ""
    ).strip()
    if source_url:
        normalized["source_url"] = source_url
    title = str(
        normalized.get("repost_title")
        or normalized.get("title")
        or normalized.get("video_title")
        or normalized.get("short_title")
        or ""
    ).strip()
    if title:
        normalized["repost_title"] = title
    views = normalized.get("repost_views")
    if views is None:
        views = normalized.get("views")
    if views is not None:
        normalized["repost_views"] = views
    normalized.setdefault("context_pattern_source_path", str(source_path))
    normalized.setdefault("context_pattern_only", True)
    return normalized


def _context_pattern_row_key(row: dict) -> str:
    source_url = str(row.get("source_url") or row.get("repost_source_url") or row.get("url") or "").strip()
    if source_url:
        return f"url:{source_url}"
    text = context_patterns.text_from_row(row).strip()
    if not text:
        return ""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"text:{digest}"


def _merge_context_pattern_row(target: dict, source: dict) -> None:
    for key, value in source.items():
        if value in (None, ""):
            continue
        if key not in target or target.get(key) in (None, ""):
            target[key] = value
            continue
        if key in {"transcript_repost_excerpt", "description", "repost_description"}:
            existing = str(target.get(key) or "")
            incoming = str(value or "")
            if incoming and incoming not in existing:
                target[key] = f"{existing} {incoming}".strip()


if __name__ == "__main__":
    raise SystemExit(main())
