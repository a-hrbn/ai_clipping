from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a vector reference manifest from labeled clip rows.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vector_manifest.jsonl"))
    parser.add_argument("--profile", default="jynxzi")
    args = parser.parse_args()

    dataset_path = _default_dataset_path(args.dataset)
    rows = _read_jsonl(dataset_path)
    manifest = build_reference_manifest(rows, args.profile)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in manifest),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(manifest), "output": str(output_path), "dataset": str(dataset_path)}, ensure_ascii=True))
    return 0


def build_reference_manifest(rows: list[dict], profile: str) -> list[dict]:
    manifest: list[dict] = []
    for index, row in enumerate(rows):
        text = _full_text(row)
        if not text:
            continue
        label = _row_label(row)
        if label is None:
            continue
        manifest.append(
            {
                "_index": len(manifest),
                "clip_id": str(row.get("candidate_id") or f"{profile}_{index}"),
                "profile": profile,
                "label": label,
                "full_text": text,
                "hook_text": _hook_text(row, text),
                "payoff_text": _payoff_text(row, text),
                "quality_score": _quality_score(row),
                "view_count": _as_int(row.get("view_count")),
                "like_count": _as_int(row.get("like_count")),
                "comment_count": _as_int(row.get("comment_count")),
                "publish_timestamp": _as_int(row.get("publish_timestamp")),
                "tags": row.get("tags") or [],
                "source_url": str(row.get("source_url") or ""),
            }
        )
    return manifest


def _row_label(row: dict) -> str | None:
    label = row.get("label")
    if label == 1:
        return "winner"
    if label == 0:
        return "loser"
    return None


def _full_text(row: dict) -> str:
    for key in ("transcript", "full_text", "text"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    title = str(row.get("title") or "").strip()
    notes = str(row.get("notes") or "").strip()
    tags = row.get("tags") or []
    if isinstance(tags, str):
        tags_text = tags
    else:
        tags_text = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
    synthetic = " ".join(part for part in [title, notes, tags_text] if part)
    return synthetic.strip()


def _hook_text(row: dict, full_text: str) -> str:
    value = str(row.get("hook_text") or "").strip()
    if value:
        return value
    return " ".join(full_text.split()[:24])


def _payoff_text(row: dict, full_text: str) -> str:
    value = str(row.get("payoff_text") or "").strip()
    if value:
        return value
    words = full_text.split()
    return " ".join(words[-24:]) if words else ""


def _quality_score(row: dict) -> float:
    views = _as_int(row.get("view_count"))
    likes = _as_int(row.get("like_count"))
    comments = _as_int(row.get("comment_count"))
    if views <= 0:
        return 0.0
    return round(min((likes / max(views, 1)) * 8.0 + (comments / max(views, 1)) * 20.0, 2.0), 3)


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _default_dataset_path(configured: str) -> pathlib.Path:
    if configured:
        return pathlib.Path(configured)
    enriched = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vector_manifest_enriched.jsonl"
    if enriched.exists():
        return enriched
    return BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_training_rows.jsonl"


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
