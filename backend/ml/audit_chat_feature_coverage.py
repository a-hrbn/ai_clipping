from __future__ import annotations

import argparse
from collections import Counter
import json
import pathlib
import statistics
import sys
from typing import Iterable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings

ARTIFACTS_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_ROWS_PATH = ARTIFACTS_DIR / "jynxzi_vod_training_rows.jsonl"
DEFAULT_GOLD_ROWS_PATH = ARTIFACTS_DIR / "jynxzi_gold_eval_rows.jsonl"

CHAT_FEATURE_NAMES = (
    "chat_peak",
    "chat_mean",
    "chat_early_peak",
    "chat_post_hook_peak",
    "chat_delta",
    "chat_spike_coverage",
    "chat_hype_score",
    "chat_laugh_score",
    "chat_shock_score",
    "chat_sadge_score",
)
CATEGORY_FEATURE_NAMES = (
    "chat_hype_score",
    "chat_laugh_score",
    "chat_shock_score",
    "chat_sadge_score",
)


def main() -> int:
    args = _parse_args()
    row_paths = [args.rows]
    if args.gold_rows.exists():
        row_paths.append(args.gold_rows)

    rows = _read_rows(row_paths)
    summary = build_summary(
        rows,
        storage_dir=args.storage_dir,
        vod_feature_cache_dir=args.vod_feature_cache_dir,
        top_vods=args.top_vods,
    )
    payload = json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True)
    print(payload)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(payload + "\n", encoding="utf-8")
    return 0


def build_summary(
    rows: list[dict],
    *,
    storage_dir: pathlib.Path,
    vod_feature_cache_dir: pathlib.Path,
    top_vods: int = 10,
) -> dict[str, object]:
    feature_rows = [row for row in rows if isinstance(_features(row), dict)]
    rows_with_chat = [row for row in feature_rows if _has_chat_signal(_features(row))]
    vod_ids = sorted({str(row.get("vod_id") or "") for row in rows if str(row.get("vod_id") or "")})

    by_source = Counter(str(row.get("dataset_source") or "unknown") for row in rows)
    chat_by_source = Counter(str(row.get("dataset_source") or "unknown") for row in rows_with_chat)
    by_split = Counter(str(row.get("gold_holdout_split") or "unspecified") for row in rows)
    chat_by_split = Counter(str(row.get("gold_holdout_split") or "unspecified") for row in rows_with_chat)

    vod_chat_counts = Counter(str(row.get("vod_id") or "") for row in rows_with_chat if row.get("vod_id"))
    cache_summary = _cache_summary(vod_ids, storage_dir=storage_dir, vod_feature_cache_dir=vod_feature_cache_dir)

    return {
        "rows": len(rows),
        "feature_rows": len(feature_rows),
        "rows_with_chat_features": len(rows_with_chat),
        "chat_feature_coverage": _ratio(len(rows_with_chat), len(feature_rows)),
        "vods": len(vod_ids),
        "vods_with_chat_feature_rows": len(vod_chat_counts),
        "dataset_sources": dict(sorted(by_source.items())),
        "chat_rows_by_dataset_source": dict(sorted(chat_by_source.items())),
        "gold_splits": dict(sorted(by_split.items())),
        "chat_rows_by_gold_split": dict(sorted(chat_by_split.items())),
        "chat_feature_stats": {
            name: _stats(_feature_values(feature_rows, name))
            for name in CHAT_FEATURE_NAMES
        },
        "chat_category_nonzero_rows": {
            name: sum(1 for row in feature_rows if abs(_float(_features(row).get(name))) > 1e-9)
            for name in CATEGORY_FEATURE_NAMES
        },
        "top_vods_by_chat_rows": vod_chat_counts.most_common(max(int(top_vods), 0)),
        "cache": cache_summary,
    }


def _parse_args() -> argparse.Namespace:
    default_storage_dir = pathlib.Path(settings.storage_dir)
    default_cache_dir = default_storage_dir.resolve().parent / "vod_feature_cache"
    parser = argparse.ArgumentParser(description="Audit Twitch chat feature coverage in training/eval artifacts.")
    parser.add_argument("--rows", type=pathlib.Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--gold-rows", type=pathlib.Path, default=DEFAULT_GOLD_ROWS_PATH)
    parser.add_argument("--storage-dir", type=pathlib.Path, default=default_storage_dir)
    parser.add_argument("--vod-feature-cache-dir", type=pathlib.Path, default=default_cache_dir)
    parser.add_argument("--summary-out", type=pathlib.Path)
    parser.add_argument("--top-vods", type=int, default=10)
    return parser.parse_args()


def _read_rows(paths: Iterable[pathlib.Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[pathlib.Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _features(row: dict) -> dict:
    features = row.get("features")
    return features if isinstance(features, dict) else row


def _has_chat_signal(features: dict) -> bool:
    if bool(features.get("has_chat_features")):
        return True
    return any(abs(_float(features.get(name))) > 1e-9 for name in CHAT_FEATURE_NAMES)


def _feature_values(rows: list[dict], name: str) -> list[float]:
    return [_float(_features(row).get(name)) for row in rows]


def _float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 6),
        "p50": round(_percentile(ordered, 0.50), 6),
        "p90": round(_percentile(ordered, 0.90), 6),
        "p95": round(_percentile(ordered, 0.95), 6),
        "max": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
    }


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _cache_summary(
    vod_ids: list[str],
    *,
    storage_dir: pathlib.Path,
    vod_feature_cache_dir: pathlib.Path,
) -> dict[str, object]:
    cache_vods = []
    chat_cached_vods = []
    for vod_id in vod_ids:
        vod_cache_dir = vod_feature_cache_dir / vod_id
        if vod_cache_dir.exists():
            cache_vods.append(vod_id)
        if list((vod_cache_dir / "chat").glob("chat*.json*")):
            chat_cached_vods.append(vod_id)

    loose_chat_files = []
    if storage_dir.exists():
        loose_chat_files = [
            path
            for path in storage_dir.glob("twitch_vod_*/chat*.json*")
            if path.is_file()
        ]

    return {
        "vod_feature_cache_dir": str(vod_feature_cache_dir),
        "cached_vods": len(cache_vods),
        "chat_cached_vods": len(chat_cached_vods),
        "chat_cached_vod_ids": chat_cached_vods[:25],
        "storage_dir": str(storage_dir),
        "loose_training_chat_files": len(loose_chat_files),
    }


if __name__ == "__main__":
    raise SystemExit(main())
