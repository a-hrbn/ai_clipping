from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize or compare serving candidate funnel quality from debug artifacts."
    )
    parser.add_argument("--debug-dir", type=Path, required=True, help="Debug directory for the current run.")
    parser.add_argument("--baseline-debug-dir", type=Path, default=None, help="Optional previous debug directory.")
    parser.add_argument("--positive-windows", type=Path, default=None, help="Optional JSONL with positive window_start/window_end rows.")
    parser.add_argument("--positive-overlap-ratio", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    current = summarize_debug_dir(args.debug_dir)
    if args.positive_windows is not None:
        current["positive_window_recall"] = positive_window_recall(
            args.debug_dir,
            args.positive_windows,
            overlap_ratio=args.positive_overlap_ratio,
        )
    payload: dict[str, Any] = {"current": current}
    if args.baseline_debug_dir is not None:
        baseline = summarize_debug_dir(args.baseline_debug_dir)
        payload["baseline"] = baseline
        payload["delta"] = _delta_summary(current, baseline)

    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


def summarize_debug_dir(debug_dir: Path) -> dict[str, Any]:
    trace_rows = _read_jsonl(debug_dir / "candidate_trace.jsonl")
    funnel = _read_json(debug_dir / "candidate_funnel_summary.json")
    final_rows = [row for row in trace_rows if row.get("final_selected")]
    high_judge_rows = [row for row in trace_rows if (_as_float(row.get("judge_score")) or 0.0) >= 7.0]
    judged_rows = [row for row in trace_rows if row.get("judge_score") not in (None, "")]
    return {
        "debug_dir": str(debug_dir),
        "trace_rows": len(trace_rows),
        "final_selected_rows": len(final_rows),
        "judged_rows": len(judged_rows),
        "high_judge_rows": len(high_judge_rows),
        "stage_counts": funnel.get("stage_counts") or {},
        "quality_breakdown": funnel.get("quality_breakdown") or {},
        "proposal_lift": funnel.get("proposal_lift") or {},
        "ranker_lift": funnel.get("ranker_lift") or {},
        "final_selected": _candidate_summary(final_rows),
        "high_judge_not_final": _candidate_summary(
            [row for row in high_judge_rows if not row.get("final_selected")][:10]
        ),
        "score_summary": {
            "final_proposal_mean": _mean(row.get("proposal_score") for row in final_rows),
            "final_pre_judge_rank_mean": _mean(row.get("pre_judge_rank") for row in final_rows),
            "final_judge_score_mean": _mean(row.get("judge_score") for row in final_rows),
            "high_judge_pre_judge_top20_ratio": (funnel.get("ranker_lift") or {}).get(
                "high_judge_pre_judge_top20_ratio"
            ),
        },
    }


def positive_window_recall(debug_dir: Path, positive_rows_path: Path, *, overlap_ratio: float = 0.45) -> dict[str, Any]:
    positives = _positive_windows(positive_rows_path, debug_dir)
    trace_rows = _read_jsonl(debug_dir / "candidate_trace.jsonl")
    stages = [
        "proposal_input",
        "proposal_kept",
        "ranked_pre_judge",
        "judge_scope",
        "clip_judge",
        "quality_gate",
        "final_ranked",
        "final_moments",
    ]
    recall_by_stage = {
        stage: _stage_recall(positives, trace_rows, stage, overlap_ratio=overlap_ratio)
        for stage in stages
    }
    top_k_recall = {
        "pre_judge_top20": _rank_recall(positives, trace_rows, rank_field="pre_judge_rank", limit=20, overlap_ratio=overlap_ratio),
        "pre_judge_top40": _rank_recall(positives, trace_rows, rank_field="pre_judge_rank", limit=40, overlap_ratio=overlap_ratio),
        "pre_judge_top80": _rank_recall(positives, trace_rows, rank_field="pre_judge_rank", limit=80, overlap_ratio=overlap_ratio),
    }
    return {
        "positive_windows": len(positives),
        "overlap_ratio": overlap_ratio,
        "recall_by_stage": recall_by_stage,
        "top_k_recall": top_k_recall,
    }


def _delta_summary(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current_scores = current.get("score_summary") or {}
    baseline_scores = baseline.get("score_summary") or {}
    keys = sorted(set(current_scores) | set(baseline_scores))
    return {
        "score_summary_delta": {
            key: _numeric_delta(current_scores.get(key), baseline_scores.get(key))
            for key in keys
        },
        "final_selected_delta": int(current.get("final_selected_rows") or 0)
        - int(baseline.get("final_selected_rows") or 0),
        "judged_rows_delta": int(current.get("judged_rows") or 0) - int(baseline.get("judged_rows") or 0),
        "high_judge_rows_delta": int(current.get("high_judge_rows") or 0) - int(baseline.get("high_judge_rows") or 0),
    }


def _candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": row.get("candidate_id"),
            "source": row.get("source"),
            "start": row.get("start"),
            "end": row.get("end"),
            "proposal_score": row.get("proposal_score"),
            "pre_judge_rank": row.get("pre_judge_rank"),
            "pre_judge_rank_score": row.get("pre_judge_rank_score"),
            "judge_score": row.get("judge_score"),
            "final_rank": row.get("final_rank"),
            "drop_reason": row.get("drop_reason"),
        }
        for row in rows
    ]


def _positive_windows(path: Path, debug_dir: Path) -> list[dict[str, float]]:
    rows = _read_jsonl(path)
    job_vod_id = _job_vod_id(debug_dir)
    windows: list[dict[str, float]] = []
    for row in rows:
        if job_vod_id:
            vod_id = str(row.get("vod_id") or "").strip()
            if vod_id and vod_id != job_vod_id:
                continue
        start = _as_float(row.get("window_start") or row.get("start"))
        end = _as_float(row.get("window_end") or row.get("end"))
        if start is None or end is None or end <= start:
            continue
        windows.append({"start": start, "end": end})
    return windows


def _stage_recall(
    positives: list[dict[str, float]],
    trace_rows: list[dict[str, Any]],
    stage: str,
    *,
    overlap_ratio: float,
) -> dict[str, Any]:
    rows = [row for row in trace_rows if _row_has_effective_stage(row, stage)]
    matched = _matched_positive_count(positives, rows, overlap_ratio=overlap_ratio)
    return {
        "stage": stage,
        "candidate_rows": len(rows),
        "matched_positive_windows": matched,
        "recall": _ratio(matched, len(positives)),
    }


def _rank_recall(
    positives: list[dict[str, float]],
    trace_rows: list[dict[str, Any]],
    *,
    rank_field: str,
    limit: int,
    overlap_ratio: float,
) -> dict[str, Any]:
    rows = [
        row for row in trace_rows
        if (_as_float(row.get(rank_field)) or 10**9) <= limit
    ]
    matched = _matched_positive_count(positives, rows, overlap_ratio=overlap_ratio)
    return {
        "limit": limit,
        "candidate_rows": len(rows),
        "matched_positive_windows": matched,
        "recall": _ratio(matched, len(positives)),
    }


def _matched_positive_count(
    positives: list[dict[str, float]],
    rows: list[dict[str, Any]],
    *,
    overlap_ratio: float,
) -> int:
    matched = 0
    for positive in positives:
        if any(_window_overlap_ratio(positive["start"], positive["end"], row.get("start"), row.get("end")) >= overlap_ratio for row in rows):
            matched += 1
    return matched


def _row_has_stage(row: dict[str, Any], stage: str) -> bool:
    stages = row.get("stages")
    return isinstance(stages, dict) and stage in stages


def _row_has_effective_stage(row: dict[str, Any], stage: str) -> bool:
    if not _row_has_stage(row, stage):
        return False
    stages = row.get("stages") if isinstance(row.get("stages"), dict) else {}
    stage_row = stages.get(stage)
    status = str(stage_row.get("status") or "").strip().lower() if isinstance(stage_row, dict) else ""
    if stage in {"proposal_input", "ranked_pre_judge"}:
        return True
    if stage == "proposal_kept":
        return status == "kept" or bool(row.get("proposal_accepted"))
    if stage == "judge_scope":
        return status == "included"
    if stage == "clip_judge":
        return status == "judged" and row.get("judge_score") not in (None, "")
    if stage == "quality_gate":
        return status == "kept" and bool(row.get("quality_has_judge_result"))
    if stage == "final_ranked":
        return status == "selected" or row.get("final_rank") not in (None, "")
    if stage == "final_moments":
        return status == "selected" or bool(row.get("final_selected"))
    return status not in {"rejected", "excluded", "missing_result"}


def _window_overlap_ratio(pos_start: float, pos_end: float, row_start: Any, row_end: Any) -> float:
    start = _as_float(row_start)
    end = _as_float(row_end)
    if start is None or end is None or end <= start or pos_end <= pos_start:
        return 0.0
    overlap = max(0.0, min(pos_end, end) - max(pos_start, start))
    if overlap <= 0.0:
        return 0.0
    return max(overlap / max(pos_end - pos_start, 1e-6), overlap / max(end - start, 1e-6))


def _job_vod_id(debug_dir: Path) -> str:
    request_path = debug_dir.parent / "request.json"
    request = _read_json(request_path)
    url = str(request.get("url") or "")
    marker = "twitch.tv/videos/"
    if marker not in url:
        return ""
    return url.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]


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


def _mean(values) -> float | None:
    numeric = [_as_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def _numeric_delta(current: Any, baseline: Any) -> float | None:
    current_value = _as_float(current)
    baseline_value = _as_float(baseline)
    if current_value is None or baseline_value is None:
        return None
    return round(current_value - baseline_value, 6)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
