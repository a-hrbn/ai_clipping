"""Generate a labeling queue for a gold evaluation set.

Reads a manifest of (job_id, vod_url, ...) rows, pulls candidates + features +
transcript snippets from each job's debug/ dir, and writes two files:

  {out}.csv         — human labeling queue (open in Google Sheets, fill
                      `gold_grade` column 0-3, re-export as CSV)
  {out}.features.jsonl — sidecar with the raw feature dict per candidate,
                      used by import_gold_eval_labels.py to reassemble the
                      eval dataset after labeling. Do not edit.

Candidates are shuffled within a VOD (deterministic with --seed) to avoid
position bias during labeling. The model's heuristic / trained score is
intentionally omitted from the CSV so the annotator labels blind.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
from urllib.parse import urlparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV with columns: job_id,vod_url[,creator_id][,platform][,vod_id]",
    )
    parser.add_argument(
        "--storage-dir",
        default="tmp/storage",
        help="Pipeline storage directory (contains one subdir per job_id).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path prefix; writes {out}.csv and {out}.features.jsonl.",
    )
    parser.add_argument("--snippet-chars", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_rows = _read_manifest(pathlib.Path(args.manifest))
    if not manifest_rows:
        print("Manifest is empty.", file=sys.stderr)
        return 1

    storage_dir = pathlib.Path(args.storage_dir)
    out_base = pathlib.Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    queue_path = out_base.with_suffix(".csv") if out_base.suffix != ".csv" else out_base
    features_path = queue_path.with_suffix(".features.jsonl")

    rng = random.Random(args.seed)
    queue_rows: list[dict] = []
    features_rows: list[dict] = []

    for entry in manifest_rows:
        job_id = entry["job_id"]
        job_debug = storage_dir / job_id / "debug"
        if not job_debug.is_dir():
            print(f"[skip] {job_id}: no debug dir at {job_debug}", file=sys.stderr)
            continue

        candidates = _read_jsonl(job_debug / "candidates_normalized.jsonl")
        features = _read_jsonl(job_debug / "candidate_features.jsonl")
        segments = _read_jsonl(job_debug / "segments.jsonl")
        ranked_rows = _read_jsonl(job_debug / "ranked_candidates_pre_judge.jsonl")
        if not candidates:
            print(f"[skip] {job_id}: no normalized candidates", file=sys.stderr)
            continue

        features_by_id = {row.get("candidate_id"): row for row in features if row.get("candidate_id")}
        ranked_by_id = {}
        for row in ranked_rows:
            candidate = row.get("candidate") or {}
            candidate_id = str(candidate.get("id") or row.get("candidate_id") or "").strip()
            if candidate_id:
                ranked_by_id[candidate_id] = row
        vod_id = entry.get("vod_id") or _vod_id_from_url(entry["vod_url"]) or job_id
        platform = entry.get("platform") or _platform_from_url(entry["vod_url"])
        creator_id = entry.get("creator_id", "")

        shuffled = list(candidates)
        rng.shuffle(shuffled)

        for candidate in shuffled:
            candidate_id = candidate.get("id") or ""
            start = float(candidate.get("start") or 0.0)
            end = float(candidate.get("end") or 0.0)
            feature_dict = features_by_id.get(candidate_id) or {}
            ranked = ranked_by_id.get(candidate_id) or {}
            snippet = _snippet_for_window(segments, start, end, args.snippet_chars)
            queue_id = _queue_id(job_id, candidate_id)

            queue_rows.append(
                {
                    "queue_id": queue_id,
                    "job_id": job_id,
                    "candidate_id": candidate_id,
                    "vod_id": vod_id,
                    "creator_id": creator_id,
                    "platform": platform,
                    "start_mmss": _format_mmss(start),
                    "end_mmss": _format_mmss(end),
                    "duration_sec": round(end - start, 1),
                    "preview_url": _preview_url(entry["vod_url"], start, platform),
                    "source_url": entry["vod_url"],
                    "snippet": snippet,
                    "gold_grade": "",
                    "notes": "",
                }
            )
            features_rows.append(
                {
                    "queue_id": queue_id,
                    "job_id": job_id,
                    "candidate_id": candidate_id,
                    "vod_id": vod_id,
                    "creator_id": creator_id,
                    "platform": platform,
                    "source_url": entry["vod_url"],
                    "window_start": start,
                    "window_end": end,
                    "heuristic_rank_score": ranked.get("heuristic_rank_score"),
                    "features": feature_dict,
                }
            )

    if not queue_rows:
        print("No candidates collected; nothing to write.", file=sys.stderr)
        return 1

    _write_csv(queue_path, queue_rows)
    _write_jsonl(features_path, features_rows)
    print(
        f"Wrote {len(queue_rows)} labeling rows to {queue_path} "
        f"(features sidecar: {features_path})"
    )
    return 0


def _read_manifest(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            job_id = (row.get("job_id") or "").strip()
            vod_url = (row.get("vod_url") or "").strip()
            if not job_id or not vod_url:
                continue
            rows.append(
                {
                    "job_id": job_id,
                    "vod_url": vod_url,
                    "creator_id": (row.get("creator_id") or "").strip(),
                    "platform": (row.get("platform") or "").strip(),
                    "vod_id": (row.get("vod_id") or "").strip(),
                }
            )
        return rows


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _snippet_for_window(segments: list[dict], start: float, end: float, max_chars: int) -> str:
    pieces: list[str] = []
    for seg in segments:
        seg_start = float(seg.get("start") or 0.0)
        seg_end = float(seg.get("end") or 0.0)
        if seg_end < start or seg_start > end:
            continue
        text = str(seg.get("text") or "").strip()
        if text:
            pieces.append(text)
    joined = " ".join(pieces)
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 1].rstrip() + "…"


def _queue_id(job_id: str, candidate_id: str) -> str:
    return f"{job_id}:{candidate_id}".strip(":")


def _format_mmss(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _preview_url(base_url: str, start: float, platform: str) -> str:
    ts = max(int(start), 0)
    platform = (platform or "").lower()
    if platform == "twitch" or "twitch.tv" in base_url:
        h, rem = divmod(ts, 3600)
        m, s = divmod(rem, 60)
        fragments = []
        if h:
            fragments.append(f"{h}h")
        fragments.append(f"{m}m")
        fragments.append(f"{s}s")
        t_value = "".join(fragments)
    else:
        t_value = f"{ts}s"
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}t={t_value}"


def _platform_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "twitch.tv" in host:
        return "twitch"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "tiktok.com" in host:
        return "tiktok"
    return "other"


def _vod_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.strip("/")
    if "twitch.tv" in host and path.startswith("videos/"):
        return path.split("/", 1)[1].split("/")[0]
    if "youtu.be" in host:
        return path.split("/")[0]
    if "youtube.com" in host:
        from urllib.parse import parse_qs

        qs = parse_qs(parsed.query)
        if qs.get("v"):
            return qs["v"][0]
    return ""


if __name__ == "__main__":
    sys.exit(main())
