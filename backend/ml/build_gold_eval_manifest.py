from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_gold_eval_queue
from ml.row_identity import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a starter gold-eval manifest from existing job folders."
    )
    parser.add_argument(
        "--storage-dir",
        default=str(BACKEND_ROOT / "tmp" / "storage"),
        help="Pipeline storage directory containing one subdir per job.",
    )
    parser.add_argument(
        "--output",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gold_eval_manifest.csv"),
        help="CSV output for resolved jobs.",
    )
    parser.add_argument(
        "--unresolved-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gold_eval_manifest_unresolved.csv"),
        help="CSV output for jobs that were found but are missing a recoverable vod_url.",
    )
    parser.add_argument(
        "--job-map",
        default="",
        help="Optional CSV with columns job_id,vod_url[,creator_id][,platform][,vod_id] to backfill missing metadata.",
    )
    parser.add_argument(
        "--min-selected",
        type=int,
        default=1,
        help="Only include jobs with at least this many selected feedback rows.",
    )
    args = parser.parse_args()

    storage_dir = pathlib.Path(args.storage_dir)
    output = pathlib.Path(args.output)
    unresolved_out = pathlib.Path(args.unresolved_out)
    job_map_path = pathlib.Path(args.job_map) if args.job_map else None

    payload = build_manifest(
        storage_dir,
        job_map_path=job_map_path,
        min_selected=max(int(args.min_selected), 0),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, payload["resolved"])
    unresolved_out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(unresolved_out, payload["unresolved"])

    print(
        json.dumps(
            {
                "resolved_jobs": len(payload["resolved"]),
                "unresolved_jobs": len(payload["unresolved"]),
                "manifest": str(output),
                "unresolved_manifest": str(unresolved_out),
            },
            ensure_ascii=True,
        )
    )
    return 0


def build_manifest(
    storage_dir: pathlib.Path,
    *,
    job_map_path: pathlib.Path | None = None,
    min_selected: int = 1,
) -> dict[str, list[dict]]:
    job_map = _read_job_map(job_map_path)
    resolved: list[dict] = []
    unresolved: list[dict] = []

    for job_dir in sorted(path for path in storage_dir.iterdir() if path.is_dir()):
        debug_dir = job_dir / "debug"
        if not debug_dir.is_dir():
            continue

        summary = _job_summary(job_dir)
        if summary["selected_count"] < min_selected:
            continue

        mapped = dict(job_map.get(job_dir.name) or {})
        vod_url = (
            str(mapped.get("vod_url") or "").strip()
            or _recover_vod_url(job_dir)
        )
        creator_id = str(mapped.get("creator_id") or "").strip()
        platform = str(mapped.get("platform") or "").strip().lower()
        vod_id = str(mapped.get("vod_id") or "").strip()

        if vod_url:
            platform = platform or build_gold_eval_queue._platform_from_url(vod_url)
            vod_id = vod_id or build_gold_eval_queue._vod_id_from_url(vod_url)
            resolved.append(
                {
                    "job_id": job_dir.name,
                    "vod_url": vod_url,
                    "creator_id": creator_id,
                    "platform": platform,
                    "vod_id": vod_id,
                    "selected_count": summary["selected_count"],
                    "feedback_rows": summary["feedback_rows"],
                    "candidate_count": summary["candidate_count"],
                }
            )
            continue

        unresolved.append(
            {
                "job_id": job_dir.name,
                "creator_id": creator_id,
                "platform": platform,
                "vod_id": vod_id,
                "selected_count": summary["selected_count"],
                "feedback_rows": summary["feedback_rows"],
                "candidate_count": summary["candidate_count"],
                "reason": "missing_vod_url",
            }
        )

    return {"resolved": resolved, "unresolved": unresolved}


def _job_summary(job_dir: pathlib.Path) -> dict[str, int]:
    feedback_rows = read_jsonl(job_dir / "debug" / "feedback.jsonl", strict=False)
    candidate_rows = read_jsonl(job_dir / "debug" / "candidates_normalized.jsonl", strict=False)
    return {
        "selected_count": sum(1 for row in feedback_rows if bool(row.get("selected"))),
        "feedback_rows": len(feedback_rows),
        "candidate_count": len(candidate_rows),
    }


def _recover_vod_url(job_dir: pathlib.Path) -> str:
    candidates = [
        job_dir / "source_url.txt",
        job_dir / "request.json",
        job_dir / "metadata.json",
        job_dir / "job.json",
    ]
    for path in candidates:
        url = _url_from_path(path)
        if url:
            return url
    return ""


def _url_from_path(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    if path.suffix == ".txt":
        return str(path.read_text(encoding="utf-8").strip())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return _url_from_payload(payload)


def _url_from_payload(payload) -> str:
    if isinstance(payload, dict):
        for key in ("vod_url", "source_url", "url", "video_url", "input_url"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        for value in payload.values():
            nested = _url_from_payload(value)
            if nested:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _url_from_payload(item)
            if nested:
                return nested
    return ""


def _read_job_map(path: pathlib.Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            job_id = str(row.get("job_id") or "").strip()
            if not job_id:
                continue
            rows[job_id] = {
                "vod_url": str(row.get("vod_url") or "").strip(),
                "creator_id": str(row.get("creator_id") or "").strip(),
                "platform": str(row.get("platform") or "").strip(),
                "vod_id": str(row.get("vod_id") or "").strip(),
            }
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
