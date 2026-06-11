from __future__ import annotations

import argparse
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import annotation_store
from ml.row_identity import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge completed manual annotations back into a training dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_rows = read_jsonl(pathlib.Path(args.dataset))
    annotations = annotation_store.read_annotations(pathlib.Path(args.annotations))
    merged = annotation_store.merge_annotations(dataset_rows, annotations)
    output_path = pathlib.Path(args.output)
    write_jsonl(output_path, merged)
    print(f"Wrote {len(merged)} merged rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
