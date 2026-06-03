#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tmp"
    / "prewarmed_sentence_transformers_local"
    / DEFAULT_MODEL.replace("/", "--")
)

REQUIRED_FILES = (
    "modules.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
    "config.json",
    "tokenizer_config.json",
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and save a complete local SentenceTransformer snapshot."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="Delete the existing output directory first.")
    args = parser.parse_args()

    if args.refresh and args.output.exists():
        shutil.rmtree(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    model.save(str(args.output))

    summary = validate_snapshot(args.output)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    if not summary["complete"]:
        return 1
    return 0


def validate_snapshot(path: Path) -> dict[str, object]:
    missing_required = [name for name in REQUIRED_FILES if not (path / name).exists()]
    weight_files = [name for name in WEIGHT_FILES if (path / name).exists()]
    tokenizer_files = [name for name in TOKENIZER_FILES if (path / name).exists()]
    return {
        "path": str(path),
        "complete": not missing_required and bool(weight_files) and bool(tokenizer_files),
        "missing_required": missing_required,
        "weight_files": weight_files,
        "tokenizer_files": tokenizer_files,
        "size_mb": round(_directory_size(path) / 1024 / 1024, 2),
    }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
