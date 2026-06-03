from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.gold_eval_policy import GOLD_HOLDOUT_SPLIT_FIELD


DEFAULT_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_serving_hard_negative_train_rows.jsonl"
DEFAULT_SUMMARY = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_serving_hard_negative_train_summary.json"
DEFAULT_POSITIVES = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_STORAGE = BACKEND_ROOT / "tmp" / "storage"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export high-ranked organic serving candidates as hard negatives when they miss known positives."
    )
    parser.add_argument("--storage-dir", type=pathlib.Path, default=DEFAULT_STORAGE)
    parser.add_argument("--positive-rows", type=pathlib.Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--rows-out", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=pathlib.Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--pre-judge-top-k", type=int, default=80)
    parser.add_argument("--max-per-vod", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--positive-exclusion-overlap", type=float, default=0.25)
    parser.add_argument("--min-rank-score", type=float, default=-999.0)
    args = parser.parse_args(argv)

    payload = export_serving_hard_negatives(
        storage_dir=args.storage_dir,
        positive_rows_path=args.positive_rows,
        profile=args.profile,
        pre_judge_top_k=max(int(args.pre_judge_top_k), 0),
        max_per_vod=max(int(args.max_per_vod), 0),
        max_rows=max(int(args.max_rows), 0),
        positive_exclusion_overlap=max(min(float(args.positive_exclusion_overlap), 1.0), 0.0),
        min_rank_score=float(args.min_rank_score),
    )
    _write_jsonl(args.rows_out, payload["rows"])
    _write_json(args.summary_out, payload["summary"])
    print(json.dumps(payload["summary"], ensure_ascii=True, sort_keys=True))
    return 0


def export_serving_hard_negatives(
    *,
    storage_dir: pathlib.Path,
    positive_rows_path: pathlib.Path,
    profile: str = "jynxzi",
    pre_judge_top_k: int = 80,
    max_per_vod: int = 20,
    max_rows: int = 300,
    positive_exclusion_overlap: float = 0.25,
    min_rank_score: float = -999.0,
) -> dict[str, Any]:
    positives_by_vod = _positive_windows_by_vod(_read_jsonl(positive_rows_path))
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    jobs_seen = 0
    latest_seen_vods: set[str] = set()

    for debug_dir in _debug_dirs(storage_dir):
        job_dir = debug_dir.parent
        request = _read_json(job_dir / "request.json")
        vod_id = _vod_id_from_request(request)
        if not vod_id:
            skipped["missing_vod_id"] += 1
            continue
        if vod_id in latest_seen_vods:
            skipped["duplicate_vod_run"] += 1
            continue
        latest_seen_vods.add(vod_id)
        positives = positives_by_vod.get(vod_id) or []
        if not positives:
            skipped["no_positive_windows_for_vod"] += 1
            continue
        ranked_rows = _read_jsonl(debug_dir / "ranked_candidates_pre_judge.jsonl")
        if not ranked_rows:
            skipped["missing_pre_judge_rows"] += 1
            continue
        serving_outcomes = _serving_outcomes_by_candidate_id(debug_dir)
        jobs_seen += 1
        per_vod_rows = _hard_negative_rows_for_job(
            job_id=job_dir.name,
            vod_id=vod_id,
            source_url=str(request.get("url") or f"https://www.twitch.tv/videos/{vod_id}"),
            ranked_rows=ranked_rows,
            serving_outcomes=serving_outcomes,
            positives=positives,
            profile=profile,
            pre_judge_top_k=pre_judge_top_k,
            max_per_vod=max_per_vod,
            positive_exclusion_overlap=positive_exclusion_overlap,
            min_rank_score=min_rank_score,
            skipped=skipped,
        )
        selected.extend(per_vod_rows)

    selected.sort(
        key=lambda row: (
            int(row.get("pre_judge_rank") or 10**9),
            -float(row.get("pre_judge_rank_score") or 0.0),
            str(row.get("vod_id") or ""),
            str(row.get("candidate_id") or ""),
        )
    )
    if max_rows:
        selected = selected[:max_rows]

    summary = {
        "event": "serving_hard_negative_exported",
        "profile": profile,
        "storage_dir": str(storage_dir),
        "positive_rows": str(positive_rows_path),
        "jobs_seen": jobs_seen,
        "positive_vods": len(positives_by_vod),
        "exported_rows": len(selected),
        "unique_vods": len({str(row.get("vod_id") or "") for row in selected if row.get("vod_id")}),
        "pre_judge_top_k": int(pre_judge_top_k),
        "max_per_vod": int(max_per_vod),
        "max_rows": int(max_rows),
        "positive_exclusion_overlap": float(positive_exclusion_overlap),
        "min_rank_score": float(min_rank_score),
        "skipped": dict(sorted(skipped.items())),
        "by_candidate_source": dict(Counter(str(row.get("candidate_source") or "") for row in selected)),
    }
    return {"rows": selected, "summary": summary}


def _hard_negative_rows_for_job(
    *,
    job_id: str,
    vod_id: str,
    source_url: str,
    ranked_rows: list[dict[str, Any]],
    serving_outcomes: dict[str, dict[str, Any]],
    positives: list[dict[str, float]],
    profile: str,
    pre_judge_top_k: int,
    max_per_vod: int,
    positive_exclusion_overlap: float,
    min_rank_score: float,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, ranked in enumerate(ranked_rows, start=1):
        if pre_judge_top_k and rank > pre_judge_top_k:
            break
        rank_score = _as_float(ranked.get("rank_score"))
        if rank_score < min_rank_score:
            skipped["below_min_rank_score"] += 1
            continue
        candidate = ranked.get("candidate") if isinstance(ranked.get("candidate"), dict) else {}
        features = ranked.get("features") if isinstance(ranked.get("features"), dict) else {}
        candidate_id = str(candidate.get("id") or features.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in seen:
            skipped["duplicate_or_missing_candidate_id"] += 1
            continue
        seen.add(candidate_id)
        rationale = {str(item) for item in (ranked.get("rationale") or [])}
        if {"cross_platform_recall_target", "proposal_recall_target"} & rationale:
            skipped["oracle_target_marker"] += 1
            continue
        outcome = serving_outcomes.get(candidate_id) or {}
        if _serving_selected_or_judged_good(outcome):
            skipped["selected_or_judged_good"] += 1
            continue
        start = _as_float(candidate.get("start"))
        end = _as_float(candidate.get("end"))
        if end <= start:
            skipped["invalid_window"] += 1
            continue
        best_overlap = max((_window_overlap_ratio(start, end, pos["start"], pos["end"]) for pos in positives), default=0.0)
        if best_overlap >= positive_exclusion_overlap:
            skipped["positive_window_overlap"] += 1
            continue
        if _ranked_selected_or_judged_good(ranked):
            skipped["selected_or_judged_good"] += 1
            continue
        rows.append(
            {
                "queue_id": f"serving_hard_negative:{job_id}:{candidate_id}",
                "job_id": job_id,
                "vod_id": vod_id,
                "source_url": source_url,
                "vod_url": source_url,
                "candidate_id": candidate_id,
                "candidate_source": str(candidate.get("source") or features.get("source") or "serving_candidate"),
                "window_start": round(start, 2),
                "window_end": round(max(end, start + 1.0), 2),
                "label": 0,
                "selected": False,
                "exported": False,
                "dataset_source": "serving_trace_hard_negative_train",
                "label_source": "serving_trace",
                "label_class": "serving_trace_hard_negative",
                "supervision_tier": "serving_hard_negative",
                "supervision_reason": "high_ranked_serving_candidate_without_cross_platform_outcome",
                "sample_weight": 1.0,
                "ordinal_label": 0,
                GOLD_HOLDOUT_SPLIT_FIELD: "train",
                "pre_judge_rank": rank,
                "pre_judge_rank_score": round(rank_score, 6),
                "heuristic_rank_score": _as_float(ranked.get("heuristic_rank_score")),
                "trained_rank_score": _as_float(ranked.get("trained_rank_score")),
                "model_blend": _as_float(ranked.get("model_blend")),
                "best_positive_overlap_ratio": round(best_overlap, 6),
                "features": dict(features),
                "profile_hint": profile,
                "hard_negative_source": "serving_trace",
            }
        )
        if max_per_vod and len(rows) >= max_per_vod:
            break
    return rows


def _ranked_selected_or_judged_good(row: dict[str, Any]) -> bool:
    if bool(row.get("selected_after_diversity")):
        return True
    judge = row.get("judge_result")
    if not isinstance(judge, dict):
        return False
    return _as_float(judge.get("judge_score")) >= 7.0 and _as_float(judge.get("judge_confidence")) >= 0.45


def _serving_selected_or_judged_good(row: dict[str, Any]) -> bool:
    if not row:
        return False
    if bool(row.get("selected_after_diversity") or row.get("selected")):
        return True
    judge = row.get("judge_result")
    judge_score = _as_float(row.get("judged_rank_score") or row.get("judge_score"))
    judge_confidence = _as_float(row.get("judge_confidence"))
    if isinstance(judge, dict):
        judge_score = max(judge_score, _as_float(judge.get("judge_score")))
        judge_confidence = max(judge_confidence, _as_float(judge.get("judge_confidence")))
    return judge_score >= 7.0 and judge_confidence >= 0.45


def _serving_outcomes_by_candidate_id(debug_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for filename in ("ranked_candidates_v2.jsonl", "ranked_candidates.jsonl", "final_moments.jsonl"):
        for row in _read_jsonl(debug_dir / filename):
            candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
            candidate_id = str(row.get("candidate_id") or candidate.get("id") or "").strip()
            if not candidate_id:
                continue
            existing = outcomes.get(candidate_id)
            if existing and _serving_selected_or_judged_good(existing):
                continue
            outcomes[candidate_id] = row
    return outcomes


def _positive_windows_by_vod(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    by_vod: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if int(row.get("label", 1) or 1) != 1:
            continue
        vod_id = str(row.get("vod_id") or "").strip()
        start = _as_float(row.get("window_start") or row.get("start"))
        end = _as_float(row.get("window_end") or row.get("end"))
        if vod_id and end > start:
            by_vod[vod_id].append({"start": start, "end": end})
    return dict(by_vod)


def _window_overlap_ratio(start: float, end: float, pos_start: float, pos_end: float) -> float:
    overlap = max(0.0, min(end, pos_end) - max(start, pos_start))
    if overlap <= 0.0 or end <= start or pos_end <= pos_start:
        return 0.0
    return max(overlap / max(end - start, 1e-6), overlap / max(pos_end - pos_start, 1e-6))


def _debug_dirs(storage_dir: pathlib.Path) -> list[pathlib.Path]:
    if not storage_dir.exists():
        return []
    dirs = [
        path / "debug"
        for path in storage_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "debug").is_dir()
    ]
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return dirs


def _vod_id_from_request(request: dict[str, Any]) -> str:
    url = str(request.get("url") or "")
    marker = "twitch.tv/videos/"
    if marker not in url:
        return ""
    return url.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
