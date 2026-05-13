import argparse

from ml.pipelines import io
from ml.pipelines.components._subprocess import run_python_module


def run(
    *,
    artifact_bundle_uri: str,
    proxy_eval_dataset_uri: str,
    gold_eval_rows_uri: str,
    proxy_eval_report_uri: str,
    gold_eval_report_uri: str,
    pipeline_eval_report_uri: str,
    profile: str,
) -> None:
    with io.temporary_work_dir("evaluate-ranker-") as work_dir:
        artifact_dir = io.materialize_directory(artifact_bundle_uri, work_dir / "artifact_bundle")
        proxy_eval_path = io.materialize_file(proxy_eval_dataset_uri, work_dir / "proxy_eval_rows.jsonl")
        gold_rows_path = io.materialize_file(gold_eval_rows_uri, work_dir / "gold_eval_rows.jsonl")
        proxy_report_path = work_dir / "proxy_eval.json"
        gold_report_path = work_dir / "gold_eval.json"
        pipeline_report_path = work_dir / "pipeline_eval.json"

        run_python_module(
            "ml.evaluate_ranker",
            [
                "--dataset",
                str(proxy_eval_path),
                "--profile",
                profile,
                "--artifact-dir",
                str(artifact_dir),
                "--report-out",
                str(proxy_report_path),
            ],
        )
        run_python_module(
            "ml.evaluate_ranker",
            [
                "--dataset",
                str(gold_rows_path),
                "--profile",
                profile,
                "--artifact-dir",
                str(artifact_dir),
                "--report-out",
                str(gold_report_path),
            ],
        )
        from ml import evaluate_pipeline, train_ranker

        rows = train_ranker._read_jsonl(proxy_eval_path)
        io.write_json(
            str(pipeline_report_path),
            evaluate_pipeline.evaluate_pipeline_rows(
                rows,
                profile=profile,
                artifact_dir=artifact_dir,
                evaluation_mode="trained_product",
            ),
        )

        io.copy_file(proxy_report_path, proxy_eval_report_uri)
        io.copy_file(gold_report_path, gold_eval_report_uri)
        io.copy_file(pipeline_report_path, pipeline_eval_report_uri)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline component: evaluate trained ranker.")
    parser.add_argument("--artifact-bundle-uri", required=True)
    parser.add_argument("--proxy-eval-dataset-uri", required=True)
    parser.add_argument("--gold-eval-rows-uri", required=True)
    parser.add_argument("--proxy-eval-report-uri", required=True)
    parser.add_argument("--gold-eval-report-uri", required=True)
    parser.add_argument("--pipeline-eval-report-uri", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run(
        artifact_bundle_uri=args.artifact_bundle_uri,
        proxy_eval_dataset_uri=args.proxy_eval_dataset_uri,
        gold_eval_rows_uri=args.gold_eval_rows_uri,
        proxy_eval_report_uri=args.proxy_eval_report_uri,
        gold_eval_report_uri=args.gold_eval_report_uri,
        pipeline_eval_report_uri=args.pipeline_eval_report_uri,
        profile=args.profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
