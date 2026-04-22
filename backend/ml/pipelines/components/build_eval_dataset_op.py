import argparse
import pathlib

from ml.pipelines import io
from ml import build_eval_dataset, export_frontend_gold_eval


def run(
    *,
    training_rows_uri: str,
    proxy_eval_dataset_uri: str,
    proxy_eval_summary_uri: str,
    gold_eval_rows_uri: str,
    gold_eval_dataset_uri: str,
    gold_eval_summary_uri: str,
    streamer: str,
    storage_dir: str = "tmp/storage",
) -> None:
    with io.temporary_work_dir("build-eval-dataset-") as work_dir:
        training_rows_path = io.materialize_file(training_rows_uri, work_dir / "training_rows.jsonl")
        proxy_eval_path = work_dir / "proxy_eval_rows.jsonl"
        proxy_rows = build_eval_dataset.build_eval_dataset(build_eval_dataset._read_jsonl(training_rows_path))
        io.write_jsonl(str(proxy_eval_path), proxy_rows)
        io.write_json(proxy_eval_summary_uri, build_eval_dataset.eval_dataset_summary(proxy_rows))

        gold_payload = export_frontend_gold_eval.export_frontend_gold_eval(
            pathlib.Path(storage_dir),
            creator_id=streamer,
        )

        io.copy_file(proxy_eval_path, proxy_eval_dataset_uri)
        io.write_jsonl(gold_eval_rows_uri, list(gold_payload["rows"]))
        io.write_jsonl(gold_eval_dataset_uri, list(gold_payload["eval_rows"]))
        io.write_json(gold_eval_summary_uri, gold_payload["summary"])

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline component: build proxy and gold eval datasets.")
    parser.add_argument("--training-rows-uri", required=True)
    parser.add_argument("--proxy-eval-dataset-uri", required=True)
    parser.add_argument("--proxy-eval-summary-uri", required=True)
    parser.add_argument("--gold-eval-rows-uri", required=True)
    parser.add_argument("--gold-eval-dataset-uri", required=True)
    parser.add_argument("--gold-eval-summary-uri", required=True)
    parser.add_argument("--streamer", required=True)
    parser.add_argument("--storage-dir", default="tmp/storage")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run(
        training_rows_uri=args.training_rows_uri,
        proxy_eval_dataset_uri=args.proxy_eval_dataset_uri,
        proxy_eval_summary_uri=args.proxy_eval_summary_uri,
        gold_eval_rows_uri=args.gold_eval_rows_uri,
        gold_eval_dataset_uri=args.gold_eval_dataset_uri,
        gold_eval_summary_uri=args.gold_eval_summary_uri,
        streamer=args.streamer,
        storage_dir=args.storage_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
