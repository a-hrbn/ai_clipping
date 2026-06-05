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
DEFAULT_HOLDOUT_POLICY = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_training_holdout_policy_rows.jsonl"
DEFAULT_STORAGE = BACKEND_ROOT / "tmp" / "storage"
_LOOKALIKE_FEATURES: tuple[str, ...] = (
    "chat_peak",
    "chat_mean",
    "chat_velocity_score",
    "chat_rate_derivative_score",
    "chat_ema_crossing_score",
    "chat_message_rate_peak",
    "chat_activity_coverage",
    "max_hype",
    "mean_hype",
    "hook_strength",
    "payoff_score",
    "payoff_clarity_score",
    "surprise_score",
    "quoteability_score",
    "standalone_score",
    "context_completeness_score",
    "context_strength_score",
    "chapter_coherence_score",
    "chapter_local_density",
    "audio_pop_score",
    "streamer_shout_score",
    "voice_energy_burst_score",
    "face_reaction_peak",
    "boundary_variant_score",
    "multimodal_proxy_score",
    "jynxzi_chat_shock_score",
    "ranked_tension_score",
    "context_pattern_score",
)
_LOOKALIKE_FEATURE_SCALES: dict[str, float] = {
    "chat_message_rate_peak": 25.0,
    "max_hype": 10.0,
    "mean_hype": 10.0,
    "payoff_score": 10.0,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export high-ranked organic serving candidates as hard negatives when they miss known positives."
    )
    parser.add_argument("--storage-dir", type=pathlib.Path, default=DEFAULT_STORAGE)
    parser.add_argument("--positive-rows", type=pathlib.Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--holdout-policy-rows", type=pathlib.Path, default=DEFAULT_HOLDOUT_POLICY)
    parser.add_argument("--include-holdout-vods", action="store_true")
    parser.add_argument("--rows-out", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=pathlib.Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument(
        "--vod-ids",
        default="",
        help="Optional comma-separated VOD IDs to include. Empty scans all eligible storage jobs.",
    )
    parser.add_argument("--pre-judge-top-k", type=int, default=500)
    parser.add_argument("--max-per-vod", type=int, default=20)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Global export cap. 0 means no global cap; per-VOD cap still applies.",
    )
    parser.add_argument("--positive-exclusion-overlap", type=float, default=0.25)
    parser.add_argument("--min-rank-score", type=float, default=-999.0)
    args = parser.parse_args(argv)

    payload = export_serving_hard_negatives(
        storage_dir=args.storage_dir,
        positive_rows_path=args.positive_rows,
        holdout_policy_rows_path=args.holdout_policy_rows,
        include_holdout_vods=bool(args.include_holdout_vods),
        vod_ids=_parse_vod_ids(args.vod_ids),
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
    holdout_policy_rows_path: pathlib.Path | None = DEFAULT_HOLDOUT_POLICY,
    include_holdout_vods: bool = False,
    vod_ids: set[str] | None = None,
    profile: str = "jynxzi",
    pre_judge_top_k: int = 500,
    max_per_vod: int = 20,
    max_rows: int = 300,
    positive_exclusion_overlap: float = 0.25,
    min_rank_score: float = -999.0,
) -> dict[str, Any]:
    positives_by_vod = _positive_windows_by_vod(_read_jsonl(positive_rows_path))
    holdout_vod_ids = set()
    if holdout_policy_rows_path is not None and pathlib.Path(holdout_policy_rows_path).exists():
        holdout_vod_ids = _holdout_vod_ids(_read_jsonl(pathlib.Path(holdout_policy_rows_path)))
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    jobs_seen = 0
    latest_seen_vods: set[str] = set()
    include_vod_ids = {str(vod_id).strip() for vod_id in (vod_ids or set()) if str(vod_id).strip()}

    for debug_dir in _debug_dirs(storage_dir):
        job_dir = debug_dir.parent
        request = _read_json(job_dir / "request.json")
        vod_id = _vod_id_from_request(request)
        if not vod_id:
            skipped["missing_vod_id"] += 1
            continue
        if include_vod_ids and vod_id not in include_vod_ids:
            skipped["outside_vod_filter"] += 1
            continue
        if vod_id in holdout_vod_ids and not include_holdout_vods:
            skipped["holdout_vod_id"] += 1
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

    if max_rows:
        selected = _fair_cap_rows_by_vod(selected, max_rows=max_rows)

    positive_vods_with_jobs = {
        str(row.get("vod_id") or "")
        for row in selected
        if str(row.get("vod_id") or "")
    }
    under_target_vods = {
        vod_id: count
        for vod_id in sorted(positive_vods_with_jobs)
        if (count := sum(1 for row in selected if str(row.get("vod_id") or "") == vod_id)) < max_per_vod
    }

    summary = {
        "event": "serving_hard_negative_exported",
        "selection_strategy": "same_vod_positive_lookalike",
        "profile": profile,
        "storage_dir": str(storage_dir),
        "positive_rows": str(positive_rows_path),
        "holdout_policy_rows": str(holdout_policy_rows_path) if holdout_policy_rows_path else "",
        "holdout_vods": len(holdout_vod_ids),
        "include_holdout_vods": bool(include_holdout_vods),
        "vod_filter_size": len(include_vod_ids),
        "vod_filter_ids": sorted(include_vod_ids),
        "jobs_seen": jobs_seen,
        "positive_vods": len(positives_by_vod),
        "exported_rows": len(selected),
        "unique_vods": len({str(row.get("vod_id") or "") for row in selected if row.get("vod_id")}),
        "target_rows_per_positive_vod": int(max_per_vod),
        "under_target_vods": under_target_vods,
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
    positive_refs = _positive_reference_rows(
        ranked_rows,
        positives,
        min_overlap=max(float(positive_exclusion_overlap), 0.45),
    )
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
        if "cross_platform_recall_target" in rationale:
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
        lookalike = _positive_lookalike_score(features, str(candidate.get("source") or features.get("source") or ""), positive_refs)
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
                "supervision_reason": "same_vod_lookalike_serving_candidate_without_cross_platform_outcome",
                "sample_weight": 1.0,
                "ordinal_label": 0,
                GOLD_HOLDOUT_SPLIT_FIELD: "train",
                "pre_judge_rank": rank,
                "pre_judge_rank_score": round(rank_score, 6),
                "heuristic_rank_score": _as_float(ranked.get("heuristic_rank_score")),
                "trained_rank_score": _as_float(ranked.get("trained_rank_score")),
                "model_blend": _as_float(ranked.get("model_blend")),
                "best_positive_overlap_ratio": round(best_overlap, 6),
                "same_vod_positive_lookalike_score": round(float(lookalike), 6),
                "same_vod_positive_reference_count": len(positive_refs),
                "features": dict(features),
                "profile_hint": profile,
                "hard_negative_source": "serving_trace_same_vod_lookalike",
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("same_vod_positive_lookalike_score") or 0.0),
            int(row.get("pre_judge_rank") or 10**9),
            -float(row.get("pre_judge_rank_score") or 0.0),
            str(row.get("candidate_id") or ""),
        )
    )
    if max_per_vod:
        rows = rows[:max_per_vod]
    return rows


def _positive_reference_rows(
    ranked_rows: list[dict[str, Any]],
    positives: list[dict[str, float]],
    *,
    min_overlap: float,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ranked in ranked_rows:
        candidate = ranked.get("candidate") if isinstance(ranked.get("candidate"), dict) else {}
        features = ranked.get("features") if isinstance(ranked.get("features"), dict) else {}
        candidate_id = str(candidate.get("id") or features.get("candidate_id") or "").strip()
        if candidate_id in seen:
            continue
        start = _as_float(candidate.get("start"))
        end = _as_float(candidate.get("end"))
        if end <= start:
            continue
        best_overlap = max((_window_overlap_ratio(start, end, pos["start"], pos["end"]) for pos in positives), default=0.0)
        if best_overlap < min_overlap:
            continue
        seen.add(candidate_id)
        refs.append(
            {
                "candidate_source": str(candidate.get("source") or features.get("source") or ""),
                "features": dict(features),
            }
        )
    return refs


def _positive_lookalike_score(features: dict[str, Any], candidate_source: str, positive_refs: list[dict[str, Any]]) -> float:
    if not positive_refs:
        return 0.0
    return max(_feature_similarity(features, candidate_source, ref) for ref in positive_refs)


def _feature_similarity(features: dict[str, Any], candidate_source: str, positive_ref: dict[str, Any]) -> float:
    ref_features = positive_ref.get("features") if isinstance(positive_ref.get("features"), dict) else {}
    weights_total = 0.0
    score_total = 0.0
    for name in _LOOKALIKE_FEATURES:
        if name not in features or name not in ref_features:
            continue
        left = _as_float(features.get(name))
        right = _as_float(ref_features.get(name))
        scale = _LOOKALIKE_FEATURE_SCALES.get(name, 1.0)
        similarity = max(0.0, 1.0 - abs(left - right) / max(scale, 1e-6))
        weight = 1.5 if name.startswith("chat_") else 1.0
        weights_total += weight
        score_total += similarity * weight
    if weights_total <= 0.0:
        return 0.0
    source_bonus = 0.05 if candidate_source and candidate_source == str(positive_ref.get("candidate_source") or "") else 0.0
    return min(score_total / weights_total + source_bonus, 1.0)


def _fair_cap_rows_by_vod(rows: list[dict[str, Any]], *, max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("vod_id") or "")].append(row)
    for bucket in grouped.values():
        bucket.sort(
            key=lambda row: (
                -float(row.get("same_vod_positive_lookalike_score") or 0.0),
                int(row.get("pre_judge_rank") or 10**9),
                -float(row.get("pre_judge_rank_score") or 0.0),
                str(row.get("candidate_id") or ""),
            )
        )
    capped: list[dict[str, Any]] = []
    while len(capped) < max_rows:
        added = False
        for vod_id in sorted(grouped):
            bucket = grouped[vod_id]
            if not bucket:
                continue
            capped.append(bucket.pop(0))
            added = True
            if len(capped) >= max_rows:
                break
        if not added:
            break
    return capped


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


def _holdout_vod_ids(rows: list[dict[str, Any]]) -> set[str]:
    vod_ids: set[str] = set()
    for row in rows:
        vod_id = str(row.get("vod_id") or row.get("twitch_vod_id") or "").strip()
        if vod_id:
            vod_ids.add(vod_id)
    return vod_ids


def _parse_vod_ids(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


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
