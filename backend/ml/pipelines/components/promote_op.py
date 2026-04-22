import argparse

from ml import mlflow_tracking
from ml.pipelines import io
from ml.pipelines import runtime_env


def run(
    *,
    artifact_bundle_uri: str,
    proxy_eval_report_uri: str,
    gold_eval_report_uri: str,
    pipeline_eval_report_uri: str,
    output_uri: str,
    profile: str,
    mlflow_tracking_uri_secret: str = "",
) -> None:
    with io.temporary_work_dir("promote-ranker-") as work_dir:
        artifact_dir = io.materialize_directory(artifact_bundle_uri, work_dir / "artifact_bundle")
        proxy_eval_path = io.materialize_file(proxy_eval_report_uri, work_dir / "proxy_eval.json")
        gold_eval_path = io.materialize_file(gold_eval_report_uri, work_dir / "gold_eval.json")
        pipeline_eval_path = io.materialize_file(pipeline_eval_report_uri, work_dir / "pipeline_eval.json")
        artifact_path = artifact_dir / f"{profile}_ranker.json"
        artifact_health_path = artifact_dir / f"{profile}_artifact_health.json"

        with runtime_env.apply_env(
            secret_bindings={
                "MLFLOW_TRACKING_URI": mlflow_tracking_uri_secret,
            }
        ):
            io.write_json(
                output_uri,
                mlflow_tracking.register_existing_artifact_bundle(
                    artifact_path=artifact_path,
                    artifact_health_path=artifact_health_path if artifact_health_path.exists() else None,
                    proxy_eval_path=proxy_eval_path,
                    gold_eval_path=gold_eval_path,
                    pipeline_eval_path=pipeline_eval_path,
                ),
            )

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline component: register/promote ranker bundle in MLflow.")
    parser.add_argument("--artifact-bundle-uri", required=True)
    parser.add_argument("--proxy-eval-report-uri", required=True)
    parser.add_argument("--gold-eval-report-uri", required=True)
    parser.add_argument("--pipeline-eval-report-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mlflow-tracking-uri-secret", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run(
        artifact_bundle_uri=args.artifact_bundle_uri,
        proxy_eval_report_uri=args.proxy_eval_report_uri,
        gold_eval_report_uri=args.gold_eval_report_uri,
        pipeline_eval_report_uri=args.pipeline_eval_report_uri,
        output_uri=args.output_uri,
        profile=args.profile,
        mlflow_tracking_uri_secret=args.mlflow_tracking_uri_secret,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
