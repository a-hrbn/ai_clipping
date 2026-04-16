from __future__ import annotations

import argparse
import json
from datetime import datetime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the Jynxzi VOD ranker training container as a Vertex AI CustomJob.",
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--staging-bucket", required=True)
    parser.add_argument("--container-uri", required=True)
    parser.add_argument("--artifacts-gcs-uri", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--display-name", default="")
    parser.add_argument("--streamer", default="jynxzi")
    parser.add_argument("--machine-type", default="n1-standard-8")
    parser.add_argument("--boot-disk-type", default="pd-ssd")
    parser.add_argument("--boot-disk-size-gb", type=int, default=200)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--dataset-llm-provider", default="auto")
    parser.add_argument("--storage-dir", default="tmp/storage")
    parser.add_argument("--train-tmpdir", default="tmp/vertex_training_runs")
    parser.add_argument("--twitch-client-id-secret", required=True)
    parser.add_argument("--twitch-client-secret-secret", required=True)
    parser.add_argument("--groq-api-key-secret", default="")
    parser.add_argument("--gemini-api-key-secret", default="")
    parser.add_argument("--ollama-host", default="")
    parser.add_argument("--network", default="")
    parser.add_argument("--async-job", action="store_true")
    return parser.parse_args()


def _job_display_name(args: argparse.Namespace) -> str:
    if args.display_name.strip():
        return args.display_name.strip()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{args.streamer}-ranker-train-{timestamp}"


def _container_args(args: argparse.Namespace) -> list[str]:
    container_args = [
        "--streamer",
        args.streamer,
        "--date-from",
        args.date_from,
        "--date-to",
        args.date_to,
        "--artifacts-gcs-uri",
        args.artifacts_gcs_uri,
        "--storage-dir",
        args.storage_dir,
        "--train-tmpdir",
        args.train_tmpdir,
        "--whisper-model",
        args.whisper_model,
        "--dataset-llm-provider",
        args.dataset_llm_provider,
        "--twitch-client-id-secret",
        args.twitch_client_id_secret,
        "--twitch-client-secret-secret",
        args.twitch_client_secret_secret,
    ]
    optional_pairs = (
        ("--groq-api-key-secret", args.groq_api_key_secret),
        ("--gemini-api-key-secret", args.gemini_api_key_secret),
        ("--ollama-host", args.ollama_host),
    )
    for flag, value in optional_pairs:
        if value.strip():
            container_args.extend([flag, value.strip()])
    return container_args


def main() -> int:
    args = _parse_args()

    from google.cloud import aiplatform

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
            },
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": args.boot_disk_type,
                "boot_disk_size_gb": args.boot_disk_size_gb,
            },
            "container_spec": {
                "image_uri": args.container_uri,
                "args": _container_args(args),
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=_job_display_name(args),
        worker_pool_specs=worker_pool_specs,
        staging_bucket=args.staging_bucket,
    )
    job.run(
        service_account=args.service_account,
        network=args.network.strip() or None,
        sync=not args.async_job,
    )

    print(
        json.dumps(
            {
                "display_name": job.display_name,
                "resource_name": getattr(job, "resource_name", ""),
                "state": str(getattr(job, "state", "")),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
