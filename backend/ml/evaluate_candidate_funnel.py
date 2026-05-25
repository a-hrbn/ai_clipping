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
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    current = summarize_debug_dir(args.debug_dir)
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


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
