import argparse
import json
import os

from ml.pipelines import io
from ml.pipelines.components._subprocess import run_python_module
from ml.pipelines import runtime_env


def run(
    *,
    training_rows_uri: str,
    artifact_bundle_uri: str,
    training_stats_uri: str,
    profile: str,
    streamer: str = "",
    date_from: str = "",
    date_to: str = "",
    mlflow_tracking_uri_secret: str = "",
    register_on_train: bool = False,
) -> None:
    with io.temporary_work_dir("train-ranker-") as work_dir:
        dataset_path = io.materialize_file(training_rows_uri, work_dir / "training_rows.jsonl")
        artifact_dir = work_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        report_path = work_dir / "training_report.json"

        env = runtime_env.build_env(
            secret_bindings={
                "MLFLOW_TRACKING_URI": mlflow_tracking_uri_secret,
            },
            value_bindings={
                "TRAINING_STREAMER": streamer,
                "DATE_FROM": date_from,
                "DATE_TO": date_to,
            },
            base_env=os.environ,
        )
        env["MLFLOW_REGISTER_ON_TRAIN"] = "1" if register_on_train else "0"
        run_python_module(
            "ml.train_ranker",
            [
                "--dataset",
                str(dataset_path),
                "--profile",
                profile,
                "--artifact-dir",
                str(artifact_dir),
                "--report-out",
                str(report_path),
            ],
            env=env,
        )

        artifact_path = artifact_dir / f"{profile}_ranker.json"
        artifact_health_path = artifact_dir / f"{profile}_artifact_health.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        metrics_payload = {
            "selected_model": artifact.get("selected_model", ""),
            "validation": artifact.get("validation", {}),
            "training_stats": artifact.get("training_stats", {}),
            "mlflow": artifact.get("mlflow", {}),
        }
        io.copy_directory(artifact_dir, artifact_bundle_uri)
        io.write_json(training_stats_uri, metrics_payload)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline component: train ranker.")
    parser.add_argument("--training-rows-uri", required=True)
    parser.add_argument("--artifact-bundle-uri", required=True)
    parser.add_argument("--training-stats-uri", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--streamer", default="")
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--mlflow-tracking-uri-secret", default="")
    parser.add_argument("--register-on-train", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run(
        training_rows_uri=args.training_rows_uri,
        artifact_bundle_uri=args.artifact_bundle_uri,
        training_stats_uri=args.training_stats_uri,
        profile=args.profile,
        streamer=args.streamer,
        date_from=args.date_from,
        date_to=args.date_to,
        mlflow_tracking_uri_secret=args.mlflow_tracking_uri_secret,
        register_on_train=args.register_on_train,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
