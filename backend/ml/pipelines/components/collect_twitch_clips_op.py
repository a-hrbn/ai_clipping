import argparse

from ml.pipelines import io
from ml.pipelines.components._subprocess import run_python_module
from ml.pipelines import runtime_env


def run(
    *,
    streamer: str,
    date_from: str,
    date_to: str,
    output_uri: str,
    twitch_client_id_secret: str = "",
    twitch_client_secret_secret: str = "",
    progress_uri: str = "",
    sources_uri: str = "",
    merge_output: bool = True,
    skip_covered: bool = True,
) -> None:
    with io.temporary_work_dir("collect-twitch-clips-") as work_dir:
        output_path = work_dir / "twitch_clip_labels.jsonl"
        progress_path = work_dir / "twitch_clip_label_progress.json"
        sources_path = work_dir / "streamer_sources.csv"

        if io.path_exists(output_uri):
            io.materialize_file(output_uri, output_path)
        if progress_uri and io.path_exists(progress_uri):
            io.materialize_file(progress_uri, progress_path)
        if sources_uri and io.path_exists(sources_uri):
            io.materialize_file(sources_uri, sources_path)

        args = [
            "--broadcaster",
            streamer,
            "--date-from",
            date_from,
            "--date-to",
            date_to,
            "--output",
            str(output_path),
            "--progress-file",
            str(progress_path),
        ]
        if sources_uri:
            args.extend(["--sources", str(sources_path)])
        if merge_output:
            args.append("--merge-output")
        if skip_covered:
            args.append("--skip-covered")

        env = runtime_env.build_env(
            secret_bindings={
                "TWITCH_CLIENT_ID": twitch_client_id_secret,
                "TWITCH_CLIENT_SECRET": twitch_client_secret_secret,
            }
        )
        run_python_module("ml.collect_twitch_clip_labels", args, env=env)
        io.copy_file(output_path, output_uri)
        if progress_uri:
            io.copy_file(progress_path, progress_uri)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline component: collect Twitch clip labels.")
    parser.add_argument("--streamer", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--twitch-client-id-secret", default="")
    parser.add_argument("--twitch-client-secret-secret", default="")
    parser.add_argument("--progress-uri", default="")
    parser.add_argument("--sources-uri", default="")
    parser.add_argument("--merge-output", action="store_true")
    parser.add_argument("--skip-covered", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run(
        streamer=args.streamer,
        date_from=args.date_from,
        date_to=args.date_to,
        output_uri=args.output_uri,
        twitch_client_id_secret=args.twitch_client_id_secret,
        twitch_client_secret_secret=args.twitch_client_secret_secret,
        progress_uri=args.progress_uri,
        sources_uri=args.sources_uri,
        merge_output=args.merge_output,
        skip_covered=args.skip_covered,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
