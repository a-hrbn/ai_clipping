import argparse

from ml.pipelines import io
from ml.pipelines.components._subprocess import run_python_module
from ml.pipelines import runtime_env


def run(
    *,
    labels_uri: str,
    output_uri: str,
    profile: str,
    dataset_llm_provider: str = "",
    groq_api_key_secret: str = "",
    gemini_api_key_secret: str = "",
    ollama_host: str = "",
    include_short_form_uri: str = "",
) -> None:
    with io.temporary_work_dir("build-training-dataset-") as work_dir:
        labels_path = io.materialize_file(labels_uri, work_dir / "labels.jsonl")
        output_path = work_dir / "training_rows.jsonl"
        short_form_path = work_dir / "short_form_rows.jsonl"

        args = [
            "--labels",
            str(labels_path),
            "--output",
            str(output_path),
            "--profile",
            profile,
            "--work-dir",
            str(work_dir / "dataset_work"),
        ]
        if include_short_form_uri and io.path_exists(include_short_form_uri):
            io.materialize_file(include_short_form_uri, short_form_path)
            args.extend(["--include-short-form", str(short_form_path)])

        env = runtime_env.build_env(
            secret_bindings={
                "GROQ_API_KEY": groq_api_key_secret,
                "GEMINI_API_KEY": gemini_api_key_secret,
            },
            value_bindings={
                "TRAINING_DATASET_LLM_PROVIDER": dataset_llm_provider,
                "OLLAMA_HOST": ollama_host,
            },
        )
        run_python_module("ml.build_twitch_training_dataset", args, env=env)
        io.copy_file(output_path, output_uri)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline component: build training dataset.")
    parser.add_argument("--labels-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset-llm-provider", default="")
    parser.add_argument("--groq-api-key-secret", default="")
    parser.add_argument("--gemini-api-key-secret", default="")
    parser.add_argument("--ollama-host", default="")
    parser.add_argument("--include-short-form-uri", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run(
        labels_uri=args.labels_uri,
        output_uri=args.output_uri,
        profile=args.profile,
        dataset_llm_provider=args.dataset_llm_provider,
        groq_api_key_secret=args.groq_api_key_secret,
        gemini_api_key_secret=args.gemini_api_key_secret,
        ollama_host=args.ollama_host,
        include_short_form_uri=args.include_short_form_uri,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
