from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml import evaluate_candidate_funnel


DEFAULT_PROFILE = "jynxzi"
DEFAULT_STAGES = [
    "peak",
    "semantic",
    "dense",
    "dense_inventory",
    "merged",
    "normalized",
    "proposal_input",
    "proposal_kept",
    "ranked_pre_judge",
    "judge_scope",
    "clip_judge",
    "quality_gate",
    "judged_quality_kept",
    "final_ranked",
    "final_moments",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate oracle-free serving recall for existing frontend/debug jobs "
            "against cross-platform positive windows."
        )
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--storage-dir", type=Path, default=_default_storage_dir())
    parser.add_argument("--positive-rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--vod-id", action="append", default=[])
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--positive-overlap-ratio", type=float, default=0.45)
    parser.add_argument(
        "--include-repeat-vod-runs",
        action="store_true",
        help="Count multiple debug jobs for the same VOD instead of only the newest valid job per VOD.",
    )
    parser.add_argument(
        "--allow-unknown-oracle-state",
        action="store_true",
        help="Include jobs that lack the newer diagnostic_only oracle-state marker.",
    )
    args = parser.parse_args(argv)

    positive_rows = args.positive_rows or _default_positive_rows(args.profile)
    output = args.output or _default_output(args.profile)
    report = evaluate_holdout(
        storage_dir=args.storage_dir,
        positive_rows_path=positive_rows,
        overlap_ratio=args.positive_overlap_ratio,
        vod_ids=set(args.vod_id or []),
        max_jobs=max(int(args.max_jobs or 0), 0),
        latest_per_vod=not args.include_repeat_vod_runs,
        require_known_oracle_free=not args.allow_unknown_oracle_state,
    )

    text = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    return 0


def evaluate_holdout(
    *,
    storage_dir: Path,
    positive_rows_path: Path | None,
    overlap_ratio: float = 0.45,
    vod_ids: set[str] | None = None,
    max_jobs: int = 0,
    latest_per_vod: bool = True,
    require_known_oracle_free: bool = True,
) -> dict[str, Any]:
    vod_filter = {str(vod_id).strip() for vod_id in (vod_ids or set()) if str(vod_id).strip()}
    job_reports: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    evaluated_vods: set[str] = set()

    for debug_dir in _debug_dirs(storage_dir):
        request = _read_json(debug_dir.parent / "request.json")
        vod_id = _vod_id_from_request(request)
        if vod_filter and vod_id not in vod_filter:
            skipped["vod_filter"] += 1
            continue
        if latest_per_vod and vod_id and vod_id in evaluated_vods:
            skipped["duplicate_vod_run"] += 1
            continue

        job_report = _evaluate_job(
            debug_dir,
            positive_rows_path=positive_rows_path,
            overlap_ratio=overlap_ratio,
            require_known_oracle_free=require_known_oracle_free,
        )
        if not job_report.get("included"):
            skipped[str(job_report.get("skip_reason") or "unknown")] += 1
            continue

        job_reports.append(job_report)
        if vod_id:
            evaluated_vods.add(vod_id)
        if max_jobs and len(job_reports) >= max_jobs:
            break

    aggregate = _aggregate_jobs(job_reports)
    return {
        "event": "serving_recall_holdout_eval",
        "storage_dir": str(storage_dir),
        "positive_rows_path": str(positive_rows_path) if positive_rows_path else "",
        "jobs_evaluated": len(job_reports),
        "jobs_skipped": dict(sorted(skipped.items())),
        "oracle_policy": {
            "require_known_oracle_free": require_known_oracle_free,
            "definition": (
                "diagnostic positives loaded with serving recall disabled and no "
                "cross_platform_recall_target/proposal_recall_target markers in candidate_trace"
            ),
        },
        "dedupe_policy": {
            "latest_per_vod": latest_per_vod,
        },
        "aggregate": aggregate,
        "jobs": job_reports,
    }


def _evaluate_job(
    debug_dir: Path,
    *,
    positive_rows_path: Path | None,
    overlap_ratio: float,
    require_known_oracle_free: bool,
) -> dict[str, Any]:
    job_dir = debug_dir.parent
    request = _read_json(job_dir / "request.json")
    report = _read_json(debug_dir / "candidate_recall_report.json")
    funnel = _read_json(debug_dir / "candidate_funnel_summary.json")
    trace_rows = _read_jsonl(debug_dir / "candidate_trace.jsonl")
    vod_id = _vod_id_from_request(request)

    if not trace_rows:
        return _skipped(debug_dir, request, "missing_candidate_trace")

    oracle_state = _oracle_state(funnel, trace_rows)
    if not oracle_state["oracle_free"]:
        return {
            **_job_base(debug_dir, request),
            "included": False,
            "skip_reason": "oracle_injection_detected",
            "oracle_state": oracle_state,
        }
    if require_known_oracle_free and not oracle_state["known"]:
        return {
            **_job_base(debug_dir, request),
            "included": False,
            "skip_reason": "unknown_oracle_state",
            "oracle_state": oracle_state,
        }

    embedded_recall = report.get("positive_window_recall") if isinstance(report, dict) else {}
    embedded_recall = embedded_recall if isinstance(embedded_recall, dict) else {}
    fallback_recall = {}
    if positive_rows_path and positive_rows_path.exists():
        fallback_recall = evaluate_candidate_funnel.positive_window_recall(
            debug_dir,
            positive_rows_path,
            overlap_ratio=overlap_ratio,
        )

    recall = embedded_recall if int(embedded_recall.get("positive_windows") or 0) > 0 else fallback_recall
    positive_windows = int(recall.get("positive_windows") or 0)
    if positive_windows <= 0:
        return {
            **_job_base(debug_dir, request),
            "included": False,
            "skip_reason": "no_positive_windows",
            "oracle_state": oracle_state,
        }

    top_k_recall = fallback_recall.get("top_k_recall") if isinstance(fallback_recall, dict) else {}
    return {
        **_job_base(debug_dir, request),
        "included": True,
        "vod_id": vod_id,
        "oracle_state": oracle_state,
        "positive_windows": positive_windows,
        "recall_by_stage": _normalize_stage_recall(recall.get("recall_by_stage") or {}),
        "top_k_recall": top_k_recall or {},
        "positive_details": _positive_details(recall.get("details") or []),
    }


def _oracle_state(funnel: dict[str, Any], trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = funnel.get("cross_platform_positive_diagnostics") or {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    diagnostic_only = diagnostics.get("diagnostic_only") is True
    serving_enabled = diagnostics.get("enabled") is True
    has_target_markers = any(_has_oracle_marker(row) for row in trace_rows)
    known = bool(diagnostics)
    return {
        "known": known,
        "diagnostic_only": diagnostic_only,
        "serving_enabled": serving_enabled,
        "has_recall_target_markers": has_target_markers,
        "oracle_free": (not serving_enabled) and (not has_target_markers) and (diagnostic_only or not known),
        "diagnostics_status": diagnostics.get("status") or "",
        "diagnostic_positive_windows": diagnostics.get("matched_rows"),
    }


def _has_oracle_marker(row: dict[str, Any]) -> bool:
    if row.get("cross_platform_recall_target") or row.get("proposal_recall_target"):
        return True
    rationale = row.get("rationale")
    if isinstance(rationale, list) and any(str(item) in {"cross_platform_recall_target", "proposal_recall_target"} for item in rationale):
        return True
    stages = row.get("stages")
    if isinstance(stages, dict):
        for stage_row in stages.values():
            if isinstance(stage_row, dict) and (
                stage_row.get("cross_platform_recall_target")
                or stage_row.get("proposal_recall_target")
            ):
                return True
    return False


def _aggregate_jobs(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    top_k_totals: dict[str, dict[str, int]] = {}
    positive_windows = 0
    for job in jobs:
        positive_windows += int(job.get("positive_windows") or 0)
        for stage, row in (job.get("recall_by_stage") or {}).items():
            total = int(row.get("total") or job.get("positive_windows") or 0)
            matched = int(row.get("matched") or 0)
            bucket = totals.setdefault(stage, {"matched": 0, "total": 0})
            bucket["matched"] += matched
            bucket["total"] += total
        for key, row in (job.get("top_k_recall") or {}).items():
            total = int(job.get("positive_windows") or 0)
            matched = int(row.get("matched_positive_windows") or 0)
            bucket = top_k_totals.setdefault(key, {"matched": 0, "total": 0})
            bucket["matched"] += matched
            bucket["total"] += total

    return {
        "positive_windows": positive_windows,
        "recall_by_stage": {
            stage: {
                **row,
                "ratio": _ratio(row["matched"], row["total"]),
            }
            for stage, row in sorted(totals.items(), key=lambda item: _stage_sort_key(item[0]))
        },
        "top_k_recall": {
            key: {
                **row,
                "ratio": _ratio(row["matched"], row["total"]),
            }
            for key, row in sorted(top_k_totals.items())
        },
    }


def _normalize_stage_recall(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for stage, row in raw.items():
        if not isinstance(row, dict):
            continue
        matched = row.get("matched", row.get("matched_positive_windows"))
        total = row.get("total")
        if total is None and row.get("recall") is not None and matched is not None:
            total = None
        normalized[stage] = {
            "matched": int(matched or 0),
            "total": int(total or 0),
            "ratio": row.get("ratio", row.get("recall")),
            "candidate_rows": row.get("candidate_rows"),
        }
    return normalized


def _positive_details(details: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        result.append(
            {
                "positive_id": detail.get("positive_id"),
                "source_url": detail.get("source_url"),
                "title": detail.get("title"),
                "start": detail.get("start"),
                "end": detail.get("end"),
                "views": detail.get("views"),
                "similarity": detail.get("similarity"),
                "last_matched_stage": detail.get("last_matched_stage"),
                "first_missed_stage_after_last_match": detail.get("first_missed_stage_after_last_match"),
            }
        )
    return result


def _debug_dirs(storage_dir: Path) -> list[Path]:
    if not storage_dir.exists():
        return []
    dirs = [
        path / "debug"
        for path in storage_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "debug").is_dir()
    ]
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return dirs


def _job_base(debug_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": debug_dir.parent.name,
        "debug_dir": str(debug_dir),
        "url": str(request.get("url") or ""),
        "vod_id": _vod_id_from_request(request),
    }


def _skipped(debug_dir: Path, request: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **_job_base(debug_dir, request),
        "included": False,
        "skip_reason": reason,
    }


def _vod_id_from_request(request: dict[str, Any]) -> str:
    url = str(request.get("url") or "")
    marker = "twitch.tv/videos/"
    if marker not in url:
        return ""
    return url.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]


def _default_storage_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tmp" / "storage"


def _default_positive_rows(profile: str) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / f"{profile}_cross_platform_fingerprint_rows.jsonl"


def _default_output(profile: str) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / f"{profile}_serving_recall_holdout_eval.json"


def _stage_sort_key(stage: str) -> tuple[int, str]:
    try:
        return (DEFAULT_STAGES.index(stage), stage)
    except ValueError:
        return (len(DEFAULT_STAGES), stage)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
