from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_proposal_training_rows, materialize_proposal_training_debug, train_proposal_scorer
from ml.row_identity import write_json

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
    parser.add_argument("--training-rows", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--summary-output", type=pathlib.Path, default=None)
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
    args = parser.parse_args(argv)

    positive_rows = args.positive_rows or DEFAULT_ARTIFACT_DIR / f"{args.profile}_cross_platform_fingerprint_rows.jsonl"
    training_rows = args.training_rows or DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_training_rows.jsonl"
    output = args.output or DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_scorer.json"
    summary_output = args.summary_output or DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_training_pipeline_summary.json"
    materialized_storage_dir = args.materialized_storage_dir or args.storage_dir / "_proposal_training_materialized"

    materialize_summary = materialize_proposal_training_debug.materialize_proposal_debug(
        profile=args.profile,
        cross_platform_rows=positive_rows,
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
            str(positive_rows),
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

    rows = train_proposal_scorer.collect_training_rows(args.storage_dir, positive_rows_path=positive_rows)
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

    summary = {
        "profile": args.profile,
        "positive_rows": str(positive_rows),
        "storage_dir": str(args.storage_dir),
        "materialized_storage_dir": str(materialized_storage_dir),
        "training_rows": str(training_rows),
        "output": str(output),
        "materialize_summary": materialize_summary,
        "training_summary": artifact.get("training_summary") or {},
    }
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


if __name__ == "__main__":
    raise SystemExit(main())
