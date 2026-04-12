from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.semantic_candidates import _get_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed vector reference clips into a local similarity index.")
    parser.add_argument("--manifest", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vector_manifest.jsonl"))
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vector_index.npz"))
    args = parser.parse_args()

    rows = _read_jsonl(pathlib.Path(args.manifest))
    if not rows:
        print(json.dumps({"rows": 0, "output": str(args.output)}, ensure_ascii=True))
        return 0

    model = _get_model()
    if model is None:
        print("Embedding model unavailable.")
        return 1

    full_texts = [str(row.get("full_text") or " ") for row in rows]
    hook_texts = [str(row.get("hook_text") or row.get("full_text") or " ") for row in rows]
    payoff_texts = [str(row.get("payoff_text") or row.get("full_text") or " ") for row in rows]

    full_embeddings = model.encode(full_texts, normalize_embeddings=True, show_progress_bar=False)
    hook_embeddings = model.encode(hook_texts, normalize_embeddings=True, show_progress_bar=False)
    payoff_embeddings = model.encode(payoff_texts, normalize_embeddings=True, show_progress_bar=False)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        full_embeddings=np.asarray(full_embeddings, dtype=np.float32),
        hook_embeddings=np.asarray(hook_embeddings, dtype=np.float32),
        payoff_embeddings=np.asarray(payoff_embeddings, dtype=np.float32),
    )
    print(json.dumps({"rows": len(rows), "output": str(output_path)}, ensure_ascii=True))
    return 0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
