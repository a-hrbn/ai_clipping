from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import sys
from collections import Counter

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_gold_eval_queue
from services import content_buckets
from services.ranker import Ranker
from services.timeline_types import CandidateFeatures


_DEFAULT_BUCKET_WEIGHTS = {
    "false_negative_proxy": 0.20,
    "false_positive_proxy": 0.20,
    "proxy_positive_review": 0.25,
    "near_threshold": 0.20,
    "model_heuristic_disagreement": 0.15,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an active-learning gold eval labeling queue from scored candidate rows.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vod_eval_dataset.jsonl"),
        help="JSONL rows with candidate features. Usually the latest VOD eval dataset.",
    )
    parser.add_argument(
        "--existing-gold-rows",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gold_eval_rows.jsonl"),
        help="Existing gold rows to avoid relabeling exact candidates/windows.",
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument(
        "--out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_active_gold_eval_queue"),
        help="Output path prefix; writes {out}.csv, {out}.features.jsonl, and {out}.summary.json.",
    )
    parser.add_argument("--target-size", type=int, default=50)
    parser.add_argument("--per-vod-limit", type=int, default=5)
    parser.add_argument("--near-threshold-window", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build_queue(
        dataset_path=pathlib.Path(args.dataset),
        existing_gold_rows_path=pathlib.Path(args.existing_gold_rows) if args.existing_gold_rows else None,
        profile=args.profile,
        artifact_dir=pathlib.Path(args.artifact_dir),
        out_base=pathlib.Path(args.out),
        target_size=max(int(args.target_size), 1),
        per_vod_limit=max(int(args.per_vod_limit), 1),
        near_threshold_window=max(float(args.near_threshold_window), 0.0),
        seed=int(args.seed),
    )
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report.get("queue_rows", 0) else 1


def build_queue(
    *,
    dataset_path: pathlib.Path,
    existing_gold_rows_path: pathlib.Path | None,
    profile: str,
    artifact_dir: pathlib.Path,
    out_base: pathlib.Path,
    target_size: int = 120,
    per_vod_limit: int = 12,
    near_threshold_window: float = 1.25,
    seed: int = 42,
) -> dict[str, object]:
    rows = _read_jsonl(dataset_path)
    existing_keys = _existing_gold_keys(_read_jsonl(existing_gold_rows_path) if existing_gold_rows_path else [])
    candidates = [row for row in rows if _candidate_key(row) not in existing_keys and _window_key(row) not in existing_keys]
    scored = _score_rows(candidates, profile=profile, artifact_dir=artifact_dir, near_threshold_window=near_threshold_window)
    selected = _select_active_rows(scored, target_size=target_size, per_vod_limit=per_vod_limit, seed=seed)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    queue_path = out_base.with_suffix(".csv") if out_base.suffix == ".csv" else out_base.with_suffix(".csv")
    features_path = queue_path.with_suffix(".features.jsonl")
    summary_path = queue_path.with_suffix(".summary.json")
    queue_rows = [_queue_row(item) for item in selected]
    feature_rows = [_feature_row(item) for item in selected]
    _write_csv(queue_path, queue_rows)
    _write_jsonl(features_path, feature_rows)

    summary = {
        "dataset": str(dataset_path),
        "existing_gold_rows": str(existing_gold_rows_path or ""),
        "profile": profile,
        "artifact_dir": str(artifact_dir),
        "input_rows": len(rows),
        "candidate_rows_after_exclusion": len(candidates),
        "scored_rows": len(scored),
        "queue_rows": len(queue_rows),
        "target_size": int(target_size),
        "per_vod_limit": int(per_vod_limit),
        "near_threshold_window": float(near_threshold_window),
        "serving_threshold": _serving_threshold(scored),
        "by_active_bucket": dict(Counter(item["active_bucket"] for item in selected)),
        "by_proxy_label": dict(Counter(str(item["row"].get("label")) for item in selected)),
        "by_vod_id": dict(Counter(str(item["row"].get("vod_id") or "") for item in selected)),
        "queue_csv": str(queue_path),
        "features_jsonl": str(features_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def _score_rows(
    rows: list[dict],
    *,
    profile: str,
    artifact_dir: pathlib.Path,
    near_threshold_window: float,
) -> list[dict[str, object]]:
    if not rows:
        return []
    baseline = Ranker(artifact_dir=artifact_dir, enable_trained_model=False)
    trained = Ranker(artifact_dir=artifact_dir, enable_trained_model=True)
    features = [CandidateFeatures.from_dict({"candidate_id": str(row.get("candidate_id") or "unknown"), **dict(row.get("features") or {})}) for row in rows]
    bucket_counts = Counter(content_buckets.bucket_from_feature(feature) for feature in features)
    baseline_scores = dict(baseline.score(features, profile))
    trained_details = {str(item["candidate_id"]): item for item in trained.score_details(features, profile)}
    inspection = trained.inspect_artifact(profile)
    artifact = inspection.get("artifact") if isinstance(inspection.get("artifact"), dict) else {}
    threshold = _safe_float((artifact or {}).get("serving_threshold"), 0.0)

    scored: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        feature = features[index]
        candidate_id = str(row.get("candidate_id") or "")
        detail = trained_details.get(candidate_id, {})
        trained_score = _safe_float(detail.get("score"), 0.0)
        heuristic_score = _safe_float(baseline_scores.get(candidate_id), 0.0)
        label = int(row.get("label", 0) or 0)
        model_positive = trained_score >= threshold
        heuristic_positive = heuristic_score >= 0.0
        bucket = _active_bucket(
            label=label,
            trained_score=trained_score,
            heuristic_score=heuristic_score,
            threshold=threshold,
            near_threshold_window=near_threshold_window,
        )
        content_bucket = content_buckets.bucket_from_feature(feature)
        priority = _priority_score(
            bucket=bucket,
            label=label,
            trained_score=trained_score,
            heuristic_score=heuristic_score,
            threshold=threshold,
            train_bucket_rows=bucket_counts[content_bucket],
        )
        scored.append(
            {
                "row": row,
                "active_bucket": bucket,
                "active_reason": _active_reason(bucket, label, model_positive, heuristic_positive),
                "trained_score": trained_score,
                "heuristic_score": heuristic_score,
                "serving_threshold": threshold,
                "score_margin": trained_score - threshold,
                "model_positive": model_positive,
                "heuristic_positive": heuristic_positive,
                "priority_score": priority,
                "content_bucket": content_bucket,
            }
        )
    return scored


def _active_bucket(
    *,
    label: int,
    trained_score: float,
    heuristic_score: float,
    threshold: float,
    near_threshold_window: float,
) -> str:
    model_positive = trained_score >= threshold
    heuristic_positive = heuristic_score >= 0.0
    if label == 1 and not model_positive:
        return "false_negative_proxy"
    if label == 0 and model_positive:
        return "false_positive_proxy"
    if abs(trained_score - threshold) <= near_threshold_window:
        return "near_threshold"
    if label == 1:
        return "proxy_positive_review"
    if model_positive != heuristic_positive:
        return "model_heuristic_disagreement"
    return "diverse_context"


def _priority_score(
    *,
    bucket: str,
    label: int,
    trained_score: float,
    heuristic_score: float,
    threshold: float,
    train_bucket_rows: int,
) -> float:
    del bucket, label
    denominator = max(abs(threshold), 1.0)
    uncertainty = max(0.0, 1.0 - abs(trained_score - threshold) / denominator)
    disagreement = 1.0 if (trained_score >= threshold) != (heuristic_score >= 0.0) else 0.0
    bucket_undersampled = 1.0 if int(train_bucket_rows) < 50 else 0.0
    return 0.5 * uncertainty + 0.3 * disagreement + 0.2 * bucket_undersampled


def _select_active_rows(
    scored: list[dict[str, object]],
    *,
    target_size: int,
    per_vod_limit: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    shuffled = list(scored)
    rng.shuffle(shuffled)
    by_bucket: dict[str, list[dict[str, object]]] = {}
    for item in shuffled:
        by_bucket.setdefault(str(item["active_bucket"]), []).append(item)
    for values in by_bucket.values():
        values.sort(key=lambda item: (float(item["priority_score"]), str((item["row"]).get("candidate_id") or "")), reverse=True)

    selected: list[dict[str, object]] = []
    selected_keys: set[str] = set()
    vod_counts: Counter[str] = Counter()
    random_reserve = max(1, int(round(target_size * 0.20)))

    def add_item(item: dict[str, object]) -> bool:
        row = item["row"]
        key = _candidate_key(row)
        vod_id = str(row.get("vod_id") or row.get("source_url") or "")
        if key in selected_keys or vod_counts[vod_id] >= per_vod_limit:
            return False
        selected.append(item)
        selected_keys.add(key)
        vod_counts[vod_id] += 1
        return True

    random_fill = [item for item in shuffled if str(item["active_bucket"]) == "diverse_context"]
    for item in random_fill:
        if len(selected) >= random_reserve:
            break
        add_item(item)

    quota_target = max(target_size - random_reserve, 1)
    for bucket, weight in _DEFAULT_BUCKET_WEIGHTS.items():
        quota = max(1, int(round(quota_target * weight)))
        added = 0
        for item in by_bucket.get(bucket, []):
            if len(selected) >= target_size or added >= quota:
                break
            if add_item(item):
                added += 1

    remaining = sorted(
        scored,
        key=lambda item: (float(item["priority_score"]), str((item["row"]).get("candidate_id") or "")),
        reverse=True,
    )
    for item in remaining:
        if len(selected) >= target_size:
            break
        add_item(item)
    return selected


def _queue_row(item: dict[str, object]) -> dict[str, object]:
    row = item["row"]
    start = _safe_float(row.get("window_start", row.get("final_start")), 0.0)
    end = _safe_float(row.get("window_end", row.get("final_end")), start)
    source_url = str(row.get("source_url") or "")
    platform = str(row.get("platform") or row.get("publish_platform") or "twitch").lower()
    return {
        "queue_id": _queue_id(row),
        "job_id": str(row.get("job_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "vod_id": str(row.get("vod_id") or ""),
        "creator_id": str(row.get("creator_id") or row.get("streamer") or ""),
        "platform": platform,
        "active_bucket": str(item["active_bucket"]),
        "active_reason": str(item["active_reason"]),
        "proxy_label": int(row.get("label", 0) or 0),
        "proxy_label_class": str(row.get("label_class") or ""),
        "candidate_source": str(row.get("candidate_source") or (row.get("features") or {}).get("source") or ""),
        "content_bucket": str(item.get("content_bucket") or ""),
        "trained_score": round(float(item["trained_score"]), 3),
        "heuristic_score": round(float(item["heuristic_score"]), 3),
        "serving_threshold": round(float(item["serving_threshold"]), 3),
        "score_margin": round(float(item["score_margin"]), 3),
        "start_mmss": build_gold_eval_queue._format_mmss(start),
        "end_mmss": build_gold_eval_queue._format_mmss(end),
        "duration_sec": round(max(end - start, 0.0), 1),
        "preview_url": build_gold_eval_queue._preview_url(source_url, start, platform) if source_url else "",
        "source_url": source_url,
        "gold_grade": "",
        "notes": "",
    }


def _feature_row(item: dict[str, object]) -> dict[str, object]:
    row = item["row"]
    start = _safe_float(row.get("window_start", row.get("final_start")), 0.0)
    end = _safe_float(row.get("window_end", row.get("final_end")), start)
    feature_row = {
        "queue_id": _queue_id(row),
        "job_id": str(row.get("job_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "vod_id": str(row.get("vod_id") or ""),
        "creator_id": str(row.get("creator_id") or row.get("streamer") or ""),
        "platform": str(row.get("platform") or row.get("publish_platform") or "twitch").lower(),
        "source_url": str(row.get("source_url") or ""),
        "window_start": start,
        "window_end": end,
        "heuristic_rank_score": float(item["heuristic_score"]),
        "trained_rank_score": float(item["trained_score"]),
        "active_bucket": str(item["active_bucket"]),
        "active_reason": str(item["active_reason"]),
        "content_bucket": str(item.get("content_bucket") or ""),
        "serving_threshold": float(item["serving_threshold"]),
        "score_margin": float(item["score_margin"]),
        "proxy_label": int(row.get("label", 0) or 0),
        "proxy_label_class": str(row.get("label_class") or ""),
        "features": dict(row.get("features") or {}),
    }
    return feature_row


def _active_reason(bucket: str, label: int, model_positive: bool, heuristic_positive: bool) -> str:
    if bucket == "false_negative_proxy":
        return "proxy_positive_but_model_below_threshold"
    if bucket == "false_positive_proxy":
        return "proxy_negative_but_model_above_threshold"
    if bucket == "near_threshold":
        return "model_score_near_serving_threshold"
    if bucket == "proxy_positive_review":
        return "proxy_positive_for_gold_confirmation"
    if bucket == "model_heuristic_disagreement":
        return f"model_positive={int(model_positive)} heuristic_positive={int(heuristic_positive)}"
    return f"diversity_fill_proxy_label={label}"


def _serving_threshold(scored: list[dict[str, object]]) -> float:
    if not scored:
        return 0.0
    return round(float(scored[0].get("serving_threshold", 0.0)), 6)


def _existing_gold_keys(rows: list[dict]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        keys.add(_candidate_key(row))
        keys.add(_window_key(row))
        queue_id = str(row.get("queue_id") or "").strip()
        if queue_id:
            keys.add(f"queue:{queue_id}")
    return {key for key in keys if key}


def _candidate_key(row: dict) -> str:
    return f"candidate:{row.get('job_id') or ''}:{row.get('candidate_id') or ''}"


def _window_key(row: dict) -> str:
    vod_id = str(row.get("vod_id") or row.get("source_url") or "").strip()
    start = round(_safe_float(row.get("window_start", row.get("final_start")), 0.0), 1)
    end = round(_safe_float(row.get("window_end", row.get("final_end")), 0.0), 1)
    return f"window:{vod_id}:{start:.1f}:{end:.1f}"


def _queue_id(row: dict) -> str:
    existing = str(row.get("queue_id") or "").strip()
    if existing:
        return existing
    return build_gold_eval_queue._queue_id(str(row.get("job_id") or ""), str(row.get("candidate_id") or ""))


def _read_jsonl(path: pathlib.Path | None) -> list[dict]:
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
        if isinstance(row, dict) and isinstance(row.get("features"), dict):
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
