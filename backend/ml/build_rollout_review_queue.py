from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Iterable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_gold_eval_queue
from services.ranker import Ranker
from services.timeline_types import CandidateFeatures


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a manual-review rollout queue from scored VOD candidates.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vod_eval_dataset.jsonl"),
        help="JSONL rows with candidate features to score and review.",
    )
    parser.add_argument(
        "--existing-gold-rows",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gold_eval_rows.jsonl"),
        help="Existing gold rows to exclude from the rollout queue.",
    )
    parser.add_argument(
        "--holdout-rows",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gold_eval_rows.jsonl"),
        help="Gold rows containing holdout split metadata; holdout VODs are excluded.",
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument(
        "--out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_rollout_review_queue"),
        help="Output path prefix; writes {out}.csv, {out}.features.jsonl, and {out}.summary.json.",
    )
    parser.add_argument("--top-k-per-vod", type=int, default=10)
    parser.add_argument("--max-total", type=int, default=200)
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=20.0,
        help="Minimum start-time gap between selected candidates from the same VOD.",
    )
    args = parser.parse_args()

    report = build_queue(
        dataset_path=pathlib.Path(args.dataset),
        existing_gold_rows_path=pathlib.Path(args.existing_gold_rows) if args.existing_gold_rows else None,
        holdout_rows_path=pathlib.Path(args.holdout_rows) if args.holdout_rows else None,
        profile=args.profile,
        artifact_dir=pathlib.Path(args.artifact_dir),
        out_base=pathlib.Path(args.out),
        top_k_per_vod=max(int(args.top_k_per_vod), 1),
        max_total=max(int(args.max_total), 1),
        min_gap_sec=max(float(args.min_gap_sec), 0.0),
    )
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report.get("queue_rows", 0) else 1


def build_queue(
    *,
    dataset_path: pathlib.Path,
    existing_gold_rows_path: pathlib.Path | None,
    holdout_rows_path: pathlib.Path | None,
    profile: str,
    artifact_dir: pathlib.Path,
    out_base: pathlib.Path,
    top_k_per_vod: int = 10,
    max_total: int = 200,
    min_gap_sec: float = 20.0,
) -> dict[str, object]:
    rows = _read_jsonl(dataset_path, require_features=True)
    existing_gold_rows = _read_jsonl(existing_gold_rows_path, require_features=False)
    holdout_rows = _read_jsonl(holdout_rows_path, require_features=False)
    existing_keys = _existing_gold_keys(existing_gold_rows)
    holdout_vods = _gold_holdout_vods(holdout_rows)

    candidates: list[dict] = []
    excluded_existing_gold = 0
    excluded_holdout_vod = 0
    for row in rows:
        if _vod_id(row) in holdout_vods:
            excluded_holdout_vod += 1
            continue
        if _candidate_key(row) in existing_keys or _window_key(row) in existing_keys or _queue_key(row) in existing_keys:
            excluded_existing_gold += 1
            continue
        candidates.append(row)

    scored = _score_rows(candidates, profile=profile, artifact_dir=artifact_dir)
    selected = _select_top_by_vod(scored, top_k_per_vod=top_k_per_vod, max_total=max_total, min_gap_sec=min_gap_sec)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    queue_path = out_base.with_suffix(".csv")
    features_path = queue_path.with_suffix(".features.jsonl")
    summary_path = queue_path.with_suffix(".summary.json")
    queue_rows = [_queue_row(item) for item in selected]
    feature_rows = [_feature_row(item) for item in selected]
    _write_csv(queue_path, queue_rows)
    _write_jsonl(features_path, feature_rows)

    summary = {
        "dataset": str(dataset_path),
        "existing_gold_rows": str(existing_gold_rows_path or ""),
        "holdout_rows": str(holdout_rows_path or ""),
        "profile": profile,
        "artifact_dir": str(artifact_dir),
        "input_rows": len(rows),
        "excluded_existing_gold_rows": excluded_existing_gold,
        "excluded_gold_holdout_vod_rows": excluded_holdout_vod,
        "gold_holdout_vods": sorted(holdout_vods),
        "candidate_rows_after_exclusion": len(candidates),
        "scored_rows": len(scored),
        "queue_rows": len(queue_rows),
        "top_k_per_vod": int(top_k_per_vod),
        "max_total": int(max_total),
        "min_gap_sec": float(min_gap_sec),
        "serving_threshold": _serving_threshold(scored),
        "review_required": True,
        "auto_publish_allowed": False,
        "recommended_action": "manual_review_before_publish",
        "by_vod_id": dict(Counter(str(item["row"].get("vod_id") or "") for item in selected)),
        "by_candidate_source": dict(Counter(str(item["row"].get("candidate_source") or "") for item in selected)),
        "queue_csv": str(queue_path),
        "features_jsonl": str(features_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def _score_rows(rows: list[dict], *, profile: str, artifact_dir: pathlib.Path) -> list[dict[str, object]]:
    if not rows:
        return []
    baseline = Ranker(artifact_dir=artifact_dir, enable_trained_model=False)
    trained = Ranker(artifact_dir=artifact_dir, enable_trained_model=True)
    features = []
    score_ids = []
    for index, row in enumerate(rows):
        score_id = f"{_queue_id(row)}::{index}"
        score_ids.append(score_id)
        features.append(CandidateFeatures.from_dict({**dict(row.get("features") or {}), "candidate_id": score_id}))

    baseline_scores = dict(baseline.score(features, profile))
    trained_details = {str(item["candidate_id"]): item for item in trained.score_details(features, profile)}
    threshold = _artifact_threshold(trained, profile)
    scored: list[dict[str, object]] = []
    for row, score_id in zip(rows, score_ids):
        detail = trained_details.get(score_id, {})
        rank_score = _safe_float(detail.get("score"), _safe_float(baseline_scores.get(score_id), 0.0))
        heuristic_score = _safe_float(detail.get("heuristic_score"), _safe_float(baseline_scores.get(score_id), 0.0))
        trained_component = detail.get("trained_score")
        scored.append(
            {
                "row": row,
                "rank_score": rank_score,
                "heuristic_score": heuristic_score,
                "trained_score": None if trained_component is None else _safe_float(trained_component, 0.0),
                "blend": _safe_float(detail.get("blend"), 0.0),
                "serving_threshold": threshold,
                "score_margin": rank_score - threshold,
                "model_positive": rank_score >= threshold,
                "rationale": list(detail.get("rationale") or []),
            }
        )
    scored.sort(
        key=lambda item: (
            float(item["rank_score"]),
            float(item["heuristic_score"]),
            str((item["row"]).get("candidate_id") or ""),
        ),
        reverse=True,
    )
    return scored


def _select_top_by_vod(
    scored: list[dict[str, object]],
    *,
    top_k_per_vod: int,
    max_total: int,
    min_gap_sec: float,
) -> list[dict[str, object]]:
    by_vod: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in scored:
        by_vod[_vod_id(item["row"])].append(item)

    selected: list[dict[str, object]] = []
    for vod_id in sorted(by_vod):
        picked_for_vod: list[dict[str, object]] = []
        for item in sorted(by_vod[vod_id], key=lambda value: float(value["rank_score"]), reverse=True):
            start = _start_sec(item["row"])
            if any(abs(start - _start_sec(existing["row"])) < min_gap_sec for existing in picked_for_vod):
                continue
            item = dict(item)
            item["rank_within_vod"] = len(picked_for_vod) + 1
            picked_for_vod.append(item)
            if len(picked_for_vod) >= top_k_per_vod:
                break
        selected.extend(picked_for_vod)

    selected.sort(
        key=lambda item: (
            float(item["rank_score"]),
            str((item["row"]).get("vod_id") or ""),
            int(item.get("rank_within_vod") or 0),
        ),
        reverse=True,
    )
    return selected[:max_total]


def _queue_row(item: dict[str, object]) -> dict[str, object]:
    row = item["row"]
    start = _start_sec(row)
    end = _end_sec(row, start)
    source_url = str(row.get("source_url") or "")
    platform = str(row.get("platform") or row.get("publish_platform") or "twitch").lower()
    return {
        "queue_id": _queue_id(row),
        "review_status": "",
        "gold_grade": "",
        "review_notes": "",
        "review_required": "true",
        "auto_publish_allowed": "false",
        "rollout_bucket": "top_candidate_manual_review",
        "rank_within_vod": int(item.get("rank_within_vod") or 0),
        "job_id": str(row.get("job_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "vod_id": str(row.get("vod_id") or ""),
        "creator_id": str(row.get("creator_id") or row.get("streamer") or ""),
        "platform": platform,
        "candidate_source": str(row.get("candidate_source") or (row.get("features") or {}).get("source") or ""),
        "model_positive": "true" if item["model_positive"] else "false",
        "rank_score": round(float(item["rank_score"]), 3),
        "trained_score": "" if item["trained_score"] is None else round(float(item["trained_score"]), 3),
        "heuristic_score": round(float(item["heuristic_score"]), 3),
        "serving_threshold": round(float(item["serving_threshold"]), 3),
        "score_margin": round(float(item["score_margin"]), 3),
        "start_mmss": build_gold_eval_queue._format_mmss(start),
        "end_mmss": build_gold_eval_queue._format_mmss(end),
        "duration_sec": round(max(end - start, 0.0), 1),
        "preview_url": build_gold_eval_queue._preview_url(source_url, start, platform) if source_url else "",
        "source_url": source_url,
        "rationale": "|".join(str(reason) for reason in item.get("rationale", [])),
    }


def _feature_row(item: dict[str, object]) -> dict[str, object]:
    row = item["row"]
    start = _start_sec(row)
    end = _end_sec(row, start)
    return {
        "queue_id": _queue_id(row),
        "review_required": True,
        "auto_publish_allowed": False,
        "rollout_bucket": "top_candidate_manual_review",
        "rank_within_vod": int(item.get("rank_within_vod") or 0),
        "job_id": str(row.get("job_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "vod_id": str(row.get("vod_id") or ""),
        "creator_id": str(row.get("creator_id") or row.get("streamer") or ""),
        "platform": str(row.get("platform") or row.get("publish_platform") or "twitch").lower(),
        "source_url": str(row.get("source_url") or ""),
        "window_start": start,
        "window_end": end,
        "rank_score": float(item["rank_score"]),
        "trained_score": item["trained_score"],
        "heuristic_score": float(item["heuristic_score"]),
        "serving_threshold": float(item["serving_threshold"]),
        "score_margin": float(item["score_margin"]),
        "model_positive": bool(item["model_positive"]),
        "rationale": list(item.get("rationale") or []),
        "features": dict(row.get("features") or {}),
    }


def _existing_gold_keys(rows: Iterable[dict]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        keys.add(_candidate_key(row))
        keys.add(_window_key(row))
        keys.add(_queue_key(row))
    return {key for key in keys if key}


def _gold_holdout_vods(rows: Iterable[dict]) -> set[str]:
    return {
        _vod_id(row)
        for row in rows
        if str(row.get("gold_holdout_split") or "").strip().lower() == "holdout" and _vod_id(row)
    }


def _artifact_threshold(ranker: Ranker, profile: str) -> float:
    inspection = ranker.inspect_artifact(profile)
    artifact = inspection.get("artifact") if isinstance(inspection.get("artifact"), dict) else {}
    return _safe_float((artifact or {}).get("serving_threshold"), 0.0)


def _serving_threshold(scored: list[dict[str, object]]) -> float:
    if not scored:
        return 0.0
    return round(float(scored[0].get("serving_threshold", 0.0)), 6)


def _candidate_key(row: dict) -> str:
    return f"candidate:{row.get('job_id') or ''}:{row.get('candidate_id') or ''}"


def _window_key(row: dict) -> str:
    vod_id = _vod_id(row) or str(row.get("source_url") or "").strip()
    start = round(_start_sec(row), 1)
    end = round(_end_sec(row, start), 1)
    return f"window:{vod_id}:{start:.1f}:{end:.1f}"


def _queue_key(row: dict) -> str:
    queue_id = str(row.get("queue_id") or "").strip()
    if queue_id:
        return f"queue:{queue_id}"
    return f"queue:{_queue_id(row)}"


def _queue_id(row: dict) -> str:
    existing = str(row.get("queue_id") or "").strip()
    if existing:
        return existing
    return build_gold_eval_queue._queue_id(str(row.get("job_id") or ""), str(row.get("candidate_id") or ""))


def _vod_id(row: dict) -> str:
    return str(row.get("vod_id") or "").strip()


def _start_sec(row: dict) -> float:
    return _safe_float(row.get("window_start", row.get("final_start")), 0.0)


def _end_sec(row: dict, default: float) -> float:
    return _safe_float(row.get("window_end", row.get("final_end")), default)


def _read_jsonl(path: pathlib.Path | None, *, require_features: bool) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and (not require_features or isinstance(row.get("features"), dict)):
            rows.append(row)
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
