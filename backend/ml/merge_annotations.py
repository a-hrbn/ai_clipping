from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import annotation_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge completed manual annotations back into a training dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_rows = _read_jsonl(pathlib.Path(args.dataset))
    annotations = annotation_store.read_annotations(pathlib.Path(args.annotations))
    merged = annotation_store.merge_annotations(dataset_rows, annotations)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in merged), encoding="utf-8")
    print(f"Wrote {len(merged)} merged rows to {output_path}")
    return 0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
