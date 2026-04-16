from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_SUMMARY_PATH = ARTIFACT_DIR / "vertex_job_summary.json"
DEFAULT_STORAGE_DIR = "tmp/storage"
DEFAULT_TRAIN_TMPDIR = "tmp/vertex_training_runs"


@dataclass(frozen=True)
class GcsLocation:
    bucket: str
    prefix: str

    @property
    def prefix_root(self) -> str:
        return f"{self.prefix}/" if self.prefix else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing Jynxzi VOD ranker training flow inside Vertex AI.",
    )
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--streamer", default="jynxzi")
    parser.add_argument("--artifacts-gcs-uri", required=True)
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--train-tmpdir", default=DEFAULT_TRAIN_TMPDIR)
    parser.add_argument("--whisper-model", default=os.environ.get("WHISPER_MODEL", "base"))
    parser.add_argument(
        "--dataset-llm-provider",
        default=os.environ.get("TRAINING_DATASET_LLM_PROVIDER", "auto"),
        help="Provider used during feature-building semantic scoring: auto, groq, or ollama.",
    )
    parser.add_argument("--twitch-client-id-secret", default="")
    parser.add_argument("--twitch-client-secret-secret", default="")
    parser.add_argument("--groq-api-key-secret", default="")
    parser.add_argument("--gemini-api-key-secret", default="")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", ""))
    parser.add_argument("--skip-covered", dest="skip_covered", action="store_true", default=True)
    parser.add_argument("--no-skip-covered", dest="skip_covered", action="store_false")
    parser.add_argument("--merge-output", dest="merge_output", action="store_true", default=True)
    parser.add_argument("--no-merge-output", dest="merge_output", action="store_false")
    parser.add_argument("--no-download-existing-artifacts", action="store_true")
    parser.add_argument("--no-upload-artifacts", action="store_true")
    return parser.parse_args()


def _parse_gcs_uri(uri: str) -> GcsLocation:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    return GcsLocation(
        bucket=parsed.netloc,
        prefix=parsed.path.lstrip("/").rstrip("/"),
    )


def _storage_client():
    from google.cloud import storage

    return storage.Client()


def _secret_client():
    from google.cloud import secretmanager

    return secretmanager.SecretManagerServiceClient()


def _download_artifacts(gcs_uri: str, target_dir: Path) -> int:
    location = _parse_gcs_uri(gcs_uri)
    client = _storage_client()
    bucket = client.bucket(location.bucket)
    blob_count = 0
    for blob in client.list_blobs(bucket, prefix=location.prefix_root):
        if blob.name.endswith("/"):
            continue
        relative_name = blob.name[len(location.prefix_root):] if location.prefix_root else blob.name
        if not relative_name:
            continue
        destination = target_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(destination)
        blob_count += 1
    return blob_count


def _upload_artifacts(source_dir: Path, gcs_uri: str) -> int:
    location = _parse_gcs_uri(gcs_uri)
    client = _storage_client()
    bucket = client.bucket(location.bucket)
    uploaded = 0
    if not source_dir.exists():
        return uploaded
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_name = path.relative_to(source_dir).as_posix()
        blob_name = f"{location.prefix_root}{relative_name}" if location.prefix_root else relative_name
        bucket.blob(blob_name).upload_from_filename(path)
        uploaded += 1
    return uploaded


def _resolve_secret(secret_ref: str) -> str:
    payload = secret_ref.strip()
    if not payload:
        return ""
    if not payload.startswith("projects/"):
        return payload
    response = _secret_client().access_secret_version(request={"name": payload})
    return response.payload.data.decode("utf-8").strip()


def _configure_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    secret_bindings = (
        ("TWITCH_CLIENT_ID", args.twitch_client_id_secret),
        ("TWITCH_CLIENT_SECRET", args.twitch_client_secret_secret),
        ("GROQ_API_KEY", args.groq_api_key_secret),
        ("GEMINI_API_KEY", args.gemini_api_key_secret),
    )
    for env_name, secret_ref in secret_bindings:
        if secret_ref.strip():
            env[env_name] = _resolve_secret(secret_ref)
    env["WHISPER_MODEL"] = args.whisper_model
    env["STORAGE_DIR"] = args.storage_dir
    env["AI_CLIPPING_TRAIN_TMPDIR"] = args.train_tmpdir
    env["TRAINING_DATASET_LLM_PROVIDER"] = args.dataset_llm_provider
    if args.ollama_host.strip():
        env["OLLAMA_HOST"] = args.ollama_host.strip()
    _validate_runtime_env(env, args.dataset_llm_provider, args.ollama_host.strip())
    return env


def _validate_runtime_env(
    env: dict[str, str],
    dataset_llm_provider: str,
    explicit_ollama_host: str,
) -> None:
    missing = [
        name
        for name in ("TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET")
        if not env.get(name, "").strip()
    ]
    if missing:
        raise SystemExit(
            "Missing required credentials for Vertex training: " + ", ".join(missing)
        )

    provider = dataset_llm_provider.strip().lower() or "auto"
    has_groq = bool(env.get("GROQ_API_KEY", "").strip())
    has_remote_ollama = bool(explicit_ollama_host)
    if provider == "groq" and not has_groq:
        raise SystemExit(
            "TRAINING_DATASET_LLM_PROVIDER=groq requires GROQ_API_KEY or --groq-api-key-secret."
        )
    if provider == "auto" and not has_groq and not has_remote_ollama:
        raise SystemExit(
            "TRAINING_DATASET_LLM_PROVIDER=auto in Vertex requires GROQ_API_KEY or an explicit --ollama-host."
        )
    if provider == "ollama" and not has_remote_ollama:
        raise SystemExit(
            "TRAINING_DATASET_LLM_PROVIDER=ollama in Vertex requires --ollama-host."
        )


def _train_command(args: argparse.Namespace) -> list[str]:
    command = [
        "/bin/zsh",
        "ml/train_jynxzi_vod.sh",
        args.date_from,
        args.date_to,
    ]
    command.append("--skip-covered" if args.skip_covered else "--no-skip-covered")
    command.append("--merge-output" if args.merge_output else "--no-merge-output")
    return command


def _positive_cluster_key(row: dict) -> str:
    clip_ids = tuple(sorted(str(item) for item in row.get("clip_ids") or []))
    return json.dumps(
        {
            "vod_id": str(row.get("vod_id", "")),
            "clip_ids": clip_ids,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _build_summary(args: argparse.Namespace, training_return_code: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "streamer": args.streamer,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "training_return_code": int(training_return_code),
        "artifacts_gcs_uri": args.artifacts_gcs_uri,
    }

    dataset_path = ARTIFACT_DIR / f"{args.streamer}_vod_training_rows.jsonl"
    ranker_eval_path = ARTIFACT_DIR / f"{args.streamer}_ranker_eval.json"
    pipeline_eval_path = ARTIFACT_DIR / f"{args.streamer}_pipeline_eval.json"

    if dataset_path.exists():
        rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary["dataset"] = {
            "rows": len(rows),
            "positive_rows": sum(int(row.get("label", 0)) == 1 for row in rows),
            "unique_vods": len({str(row.get("vod_id", "")) for row in rows if row.get("vod_id")}),
            "unique_positive_clusters": len(
                {
                    _positive_cluster_key(row)
                    for row in rows
                    if int(row.get("label", 0)) == 1
                }
            ),
        }

    if ranker_eval_path.exists():
        summary["ranker_eval"] = json.loads(ranker_eval_path.read_text(encoding="utf-8"))
    if pipeline_eval_path.exists():
        summary["pipeline_eval"] = json.loads(pipeline_eval_path.read_text(encoding="utf-8"))
    return summary


def main() -> int:
    args = _parse_args()
    if args.streamer != "jynxzi":
        raise SystemExit("This first Vertex scaffold currently wraps only the jynxzi training flow.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    env = _configure_runtime_env(args)

    if not args.no_download_existing_artifacts:
        downloaded = _download_artifacts(args.artifacts_gcs_uri, ARTIFACT_DIR)
        print(
            json.dumps(
                {
                    "event": "vertex_artifacts_downloaded",
                    "count": downloaded,
                    "artifacts_gcs_uri": args.artifacts_gcs_uri,
                },
                ensure_ascii=True,
            )
        )

    command = _train_command(args)
    print(json.dumps({"event": "vertex_training_start", "command": command}, ensure_ascii=True))
    completed = subprocess.run(
        command,
        cwd=BACKEND_ROOT,
        env=env,
        check=False,
    )

    summary = _build_summary(args, completed.returncode)
    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = BACKEND_ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    if not args.no_upload_artifacts:
        uploaded = _upload_artifacts(ARTIFACT_DIR, args.artifacts_gcs_uri)
        print(
            json.dumps(
                {
                    "event": "vertex_artifacts_uploaded",
                    "count": uploaded,
                    "artifacts_gcs_uri": args.artifacts_gcs_uri,
                },
                ensure_ascii=True,
            )
        )

    print(json.dumps(summary, ensure_ascii=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
