from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import re
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import read_jsonl


DEFAULT_INPUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vod_training_rows.jsonl"
DEFAULT_OUTPUT = BACKEND_ROOT.parent / "tasks" / "background_negative_spotcheck.csv"


def main() -> int:
    return main_with_args_for_test(None)


def main_with_args_for_test(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Export a spot-check CSV for sampled background negatives.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args(argv)

    rows = read_jsonl(pathlib.Path(args.input))
    background_rows = [
        row
        for row in rows
        if str(row.get("supervision_tier") or "").strip().lower() == "background_negative"
        or str(row.get("label_class") or "").strip().lower() == "background_negative"
    ]
    sampled = _sample_rows(background_rows, sample_size=max(int(args.sample_size), 0), seed=int(args.seed))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, sampled)
    print(json.dumps({"input_rows": len(rows), "background_negative_rows": len(background_rows), "exported_rows": len(sampled), "output": str(output)}, ensure_ascii=True))
    return 0


def _sample_rows(rows: list[dict], *, sample_size: int, seed: int) -> list[dict]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return list(rows)
    rng = random.Random(seed)
    return sorted(rng.sample(rows, sample_size), key=lambda row: (str(row.get("vod_id") or ""), float(row.get("window_start") or 0.0)))


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = [
        "vod_id",
        "window_start",
        "window_end",
        "transcript_first_30_words",
        "chat_rate",
        "max_hype",
        "face_peak",
        "twitch_clip_link",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            features = dict(row.get("features") or {})
            vod_id = str(row.get("vod_id") or "")
            start = float(row.get("window_start") or 0.0)
            writer.writerow({
                "vod_id": vod_id,
                "window_start": round(start, 2),
                "window_end": round(float(row.get("window_end") or start), 2),
                "transcript_first_30_words": _first_words(row),
                "chat_rate": features.get("chat_message_rate_mean", 0.0),
                "max_hype": features.get("max_hype", 0.0),
                "face_peak": features.get("face_reaction_peak", 0.0),
                "twitch_clip_link": _twitch_link(vod_id, start),
            })


def _first_words(row: dict, limit: int = 30) -> str:
    text = str(row.get("transcript") or row.get("text") or row.get("notes") or "")
    words = re.findall(r"\S+", text)
    return " ".join(words[:limit])


def _twitch_link(vod_id: str, start_seconds: float) -> str:
    if not vod_id:
        return ""
    return f"https://www.twitch.tv/videos/{vod_id}?t={max(int(start_seconds), 0)}s"


if __name__ == "__main__":
    raise SystemExit(main())
