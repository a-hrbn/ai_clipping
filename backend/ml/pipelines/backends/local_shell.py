from __future__ import annotations

import argparse
import json
import pathlib

from ml.pipelines.components._subprocess import run_python_module


def run_local_pipeline(
    *,
    streamer: str,
    profile: str,
    date_from: str,
    date_to: str,
    pipeline_root: str,
    storage_dir: str = "tmp/storage",
) -> dict[str, str]:
    root = pathlib.Path(pipeline_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    labels_uri = str(root / "datasets" / f"{streamer}_twitch_labels.jsonl")
    progress_uri = str(root / "state" / f"{streamer}_twitch_progress.json")
    training_rows_uri = str(root / "datasets" / f"{streamer}_training_rows.jsonl")
    proxy_eval_dataset_uri = str(root / "datasets" / f"{streamer}_proxy_eval_rows.jsonl")
    proxy_eval_summary_uri = str(root / "metrics" / f"{streamer}_proxy_eval_dataset_summary.json")
    gold_eval_rows_uri = str(root / "datasets" / f"{streamer}_gold_eval_rows.jsonl")
    gold_eval_dataset_uri = str(root / "datasets" / f"{streamer}_gold_eval_dataset.jsonl")
    gold_eval_summary_uri = str(root / "metrics" / f"{streamer}_gold_eval_dataset_summary.json")
    artifact_bundle_uri = str(root / "artifacts" / f"{streamer}_ranker_bundle")
    training_stats_uri = str(root / "metrics" / f"{streamer}_training_stats.json")
    proxy_eval_report_uri = str(root / "metrics" / f"{streamer}_proxy_eval.json")
    gold_eval_report_uri = str(root / "metrics" / f"{streamer}_gold_eval.json")
    pipeline_eval_report_uri = str(root / "metrics" / f"{streamer}_pipeline_eval.json")
    registry_report_uri = str(root / "metrics" / f"{streamer}_registry_report.json")

    run_python_module(
        "ml.pipelines.components.collect_twitch_clips_op",
        [
            "--streamer",
            streamer,
            "--date-from",
            date_from,
            "--date-to",
            date_to,
            "--output-uri",
            labels_uri,
            "--progress-uri",
            progress_uri,
            "--merge-output",
            "--skip-covered",
        ],
    )
    run_python_module(
        "ml.pipelines.components.build_training_dataset_op",
        [
            "--labels-uri",
            labels_uri,
            "--output-uri",
            training_rows_uri,
            "--profile",
            profile,
        ],
    )
    run_python_module(
        "ml.pipelines.components.train_ranker_op",
        [
            "--training-rows-uri",
            training_rows_uri,
            "--artifact-bundle-uri",
            artifact_bundle_uri,
            "--training-stats-uri",
            training_stats_uri,
            "--profile",
            profile,
        ],
    )
    run_python_module(
        "ml.pipelines.components.build_eval_dataset_op",
        [
            "--training-rows-uri",
            training_rows_uri,
            "--proxy-eval-dataset-uri",
            proxy_eval_dataset_uri,
            "--proxy-eval-summary-uri",
            proxy_eval_summary_uri,
            "--gold-eval-rows-uri",
            gold_eval_rows_uri,
            "--gold-eval-dataset-uri",
            gold_eval_dataset_uri,
            "--gold-eval-summary-uri",
            gold_eval_summary_uri,
            "--streamer",
            streamer,
            "--storage-dir",
            storage_dir,
        ],
    )
    run_python_module(
        "ml.pipelines.components.evaluate_ranker_op",
        [
            "--artifact-bundle-uri",
            artifact_bundle_uri,
            "--proxy-eval-dataset-uri",
            proxy_eval_dataset_uri,
            "--gold-eval-rows-uri",
            gold_eval_rows_uri,
            "--proxy-eval-report-uri",
            proxy_eval_report_uri,
            "--gold-eval-report-uri",
            gold_eval_report_uri,
            "--pipeline-eval-report-uri",
            pipeline_eval_report_uri,
            "--profile",
            profile,
        ],
    )
    run_python_module(
        "ml.pipelines.components.promote_op",
        [
            "--artifact-bundle-uri",
            artifact_bundle_uri,
            "--proxy-eval-report-uri",
            proxy_eval_report_uri,
            "--gold-eval-report-uri",
            gold_eval_report_uri,
            "--pipeline-eval-report-uri",
            pipeline_eval_report_uri,
            "--output-uri",
            registry_report_uri,
            "--profile",
            profile,
        ],
    )
    return {
        "labels_uri": labels_uri,
        "training_rows_uri": training_rows_uri,
        "artifact_bundle_uri": artifact_bundle_uri,
        "proxy_eval_report_uri": proxy_eval_report_uri,
        "gold_eval_report_uri": gold_eval_report_uri,
        "pipeline_eval_report_uri": pipeline_eval_report_uri,
        "registry_report_uri": registry_report_uri,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the portable training pipeline sequentially via local subprocess calls.")
    parser.add_argument("--streamer", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--storage-dir", default="tmp/storage")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_local_pipeline(
        streamer=args.streamer,
        profile=args.profile,
        date_from=args.date_from,
        date_to=args.date_to,
        pipeline_root=args.pipeline_root,
        storage_dir=args.storage_dir,
    )
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
