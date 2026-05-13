from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_eval_dataset
from ml.row_identity import row_identity_keys
from services.gold_eval_policy import GOLD_HOLDOUT_SPLIT_FIELD

OUTCOME_TIER_PREFIX = "outcome"
CROSS_PLATFORM_OUTCOME_SOURCE_PREFIX = "cross_platform_outcome"
AUTO_NEGATIVE_OUTCOME_SOURCE_PREFIX = "outcome_auto_negative"
DEFAULT_CROSS_PLATFORM_HOLDOUT_RATIO = 0.2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a label-light outcome dataset from cross-platform positives and auto-outcome negatives."
    )
    parser.add_argument(
        "--positive-rows",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_rows.jsonl"),
        help="JSONL rows with positive cross-platform outcome labels.",
    )
    parser.add_argument(
        "--top-viewed-rows",
        default="",
        help="Deprecated. Twitch top-viewed rows are ignored by the outcome dataset builder.",
    )
    parser.add_argument(
        "--auto-negative-rows",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_auto_outcome_negative_holdout_rows.jsonl"),
    )
    parser.add_argument(
        "--feature-source-rows",
        action="append",
        default=[],
        help=(
            "Optional JSONL dataset rows with built features. Cross-platform "
            "positive rows that only contain VOD/window labels are enriched "
            "from matching identities before missing-feature filtering."
        ),
    )
    parser.add_argument("--split", choices=("train", "holdout"), default="holdout")
    parser.add_argument(
        "--output",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_outcome_holdout_dataset.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_outcome_holdout_summary.json"),
    )
    parser.add_argument(
        "--target-positive-rows",
        type=int,
        default=0,
        help="Optional positive-row target to record in the summary for holdout-size readiness.",
    )
    parser.add_argument(
        "--cross-platform-holdout-ratio",
        type=float,
        default=DEFAULT_CROSS_PLATFORM_HOLDOUT_RATIO,
    )
    args = parser.parse_args()

    payload = build_outcome_holdout_dataset(
        positive_rows=_read_jsonl(pathlib.Path(args.positive_rows)),
        auto_negative_rows=_read_jsonl(pathlib.Path(args.auto_negative_rows)),
        feature_source_rows=[
            row
            for feature_source_path in args.feature_source_rows
            for row in _read_jsonl(pathlib.Path(feature_source_path))
        ],
        split=args.split,
        target_positive_rows=max(int(args.target_positive_rows), 0),
        cross_platform_holdout_ratio=float(args.cross_platform_holdout_ratio),
    )
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in payload["rows"])
        + ("\n" if payload["rows"] else ""),
        encoding="utf-8",
    )
    summary_path = pathlib.Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload["summary"], ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=True))
    return 0


def build_outcome_holdout_dataset(
    *,
    positive_rows: list[dict] | None = None,
    top_viewed_rows: list[dict] | None = None,
    auto_negative_rows: list[dict],
    feature_source_rows: list[dict] | None = None,
    split: str = "holdout",
    target_positive_rows: int = 0,
    cross_platform_holdout_ratio: float = DEFAULT_CROSS_PLATFORM_HOLDOUT_RATIO,
) -> dict[str, object]:
    split = _normalize_split(split)
    tier = f"{OUTCOME_TIER_PREFIX}_{split}"
    cross_platform_source = f"{CROSS_PLATFORM_OUTCOME_SOURCE_PREFIX}_{split}"
    auto_negative_source = f"{AUTO_NEGATIVE_OUTCOME_SOURCE_PREFIX}_{split}"
    input_positive_rows = list(positive_rows or [])
    feature_replay = _feature_replay_index(feature_source_rows or [])
    selected_rows: list[dict] = []
    seen_identities: set[str] = set()
    skipped: Counter[str] = Counter()
    if top_viewed_rows:
        skipped["top_viewed_rows_ignored"] += len(top_viewed_rows)
    selected_rows.extend(
        _prepare_rows(
            input_positive_rows,
            label=1,
            source=cross_platform_source,
            label_source=None,
            label_class="cross_platform_outcome_positive",
            supervision_tier=tier,
            supervision_reason=f"cross_platform_shortform_{split}_positive",
            split=split,
            seen_identities=seen_identities,
            skipped=skipped,
            feature_replay=feature_replay,
            predicate=lambda row, selected_split: _is_cross_platform_positive_split(
                row,
                selected_split,
                holdout_ratio=cross_platform_holdout_ratio,
            ),
        )
    )
    selected_rows.extend(
        _prepare_rows(
            auto_negative_rows,
            label=0,
            source=auto_negative_source,
            label_source="auto_outcome_negative",
            label_class="outcome_auto_negative",
            supervision_tier=tier,
            supervision_reason="candidate_without_positive_short_form_outcome",
            split=split,
            seen_identities=seen_identities,
            skipped=skipped,
            feature_replay=None,
            predicate=_is_auto_negative_split,
        )
    )
    eval_rows = build_eval_dataset.build_eval_dataset(selected_rows)
    summary = build_eval_dataset.eval_dataset_summary(eval_rows)
    positive_count = sum(1 for row in eval_rows if int(row.get("label", 0)) == 1)
    negative_count = sum(1 for row in eval_rows if int(row.get("label", 0)) == 0)
    feature_replay_applied = int(skipped.pop("_feature_replay_applied", 0))
    summary.update(
        {
            "outcome_holdout_version": 1,
            "outcome_split": split,
            "outcome_supervision_tier": tier,
            "input_cross_platform_rows": len(input_positive_rows),
            "input_feature_source_rows": len(feature_source_rows or []),
            "input_top_viewed_rows": len(top_viewed_rows or []),
            "input_auto_negative_rows": len(auto_negative_rows),
            "cross_platform_positive_rows": positive_count,
            "top_viewed_positive_rows": 0,
            "auto_negative_rows": negative_count,
            "cross_platform_holdout_ratio": float(cross_platform_holdout_ratio),
            "unique_vods": len({str(row.get("vod_id") or "") for row in eval_rows if str(row.get("vod_id") or "")}),
            "skipped": dict(sorted(skipped.items())),
            "feature_replay": {
                "indexed_rows": feature_replay["indexed_rows"],
                "indexed_identities": len(feature_replay["by_identity"]),
                "applied_rows": feature_replay_applied,
            },
        }
    )
    if target_positive_rows > 0:
        summary.update(
            {
                "target_positive_rows": target_positive_rows,
                "target_positive_rows_met": positive_count >= target_positive_rows,
                "target_positive_rows_needed": max(target_positive_rows - positive_count, 0),
            }
        )
    return {"rows": eval_rows, "summary": summary}


def _feature_replay_index(rows: list[dict]) -> dict[str, object]:
    by_identity: dict[str, dict] = {}
    indexed_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        features = dict(row.get("features") or {})
        if not features:
            continue
        identities = row_identity_keys(row)
        if not identities:
            continue
        indexed_rows += 1
        snapshot = {
            "features": features,
            "candidate_source": row.get("candidate_source"),
            "content_bucket": row.get("content_bucket"),
            "quality_bucket": row.get("quality_bucket"),
            "platform": row.get("platform"),
            "platform_source": row.get("platform_source"),
        }
        for identity in identities:
            by_identity.setdefault(identity, snapshot)
    return {"indexed_rows": indexed_rows, "by_identity": by_identity}


def _replay_snapshot(row: dict, feature_replay: dict[str, object] | None) -> dict:
    if not feature_replay:
        return {}
    by_identity = feature_replay.get("by_identity")
    if not isinstance(by_identity, dict):
        return {}
    for identity in sorted(row_identity_keys(row)):
        snapshot = by_identity.get(identity)
        if not isinstance(snapshot, dict):
            continue
        if dict(snapshot.get("features") or {}):
            return snapshot
    return {}


def _replay_features(row: dict, feature_replay: dict[str, object] | None) -> dict:
    snapshot = _replay_snapshot(row, feature_replay)
    return dict(snapshot.get("features") or {})


def _prepare_rows(
    rows: list[dict],
    *,
    label: int,
    source: str,
    label_source: str | None,
    label_class: str,
    supervision_tier: str,
    supervision_reason: str,
    split: str,
    seen_identities: set[str],
    skipped: Counter[str],
    feature_replay: dict[str, object] | None,
    predicate,
) -> list[dict]:
    prepared: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            skipped["invalid_row"] += 1
            continue
        if not predicate(row, split):
            skipped[f"not_{source}"] += 1
            continue
        features = dict(row.get("features") or {})
        replayed_snapshot = _replay_snapshot(row, feature_replay)
        replayed = dict(replayed_snapshot.get("features") or {})
        if replayed:
            features.update(replayed)
            skipped["_feature_replay_applied"] += 1
        if not features:
            skipped[f"{source}_missing_features"] += 1
            continue
        identities = row_identity_keys(row)
        if not identities:
            skipped[f"{source}_missing_identity"] += 1
            continue
        if identities & seen_identities:
            skipped[f"{source}_duplicate_identity"] += 1
            continue
        copied = dict(row)
        copied["features"] = features
        copied["label"] = int(label)
        copied["selected"] = bool(label)
        copied["exported"] = bool(label)
        copied["dataset_source"] = source
        copied["original_dataset_source"] = str(row.get("dataset_source") or "")
        copied["label_source"] = str(label_source or row.get("label_source") or row.get("dataset_source") or "")
        copied["supervision_tier"] = supervision_tier
        copied["supervision_reason"] = supervision_reason
        copied["label_class"] = label_class
        copied["ordinal_label"] = 3 if label else 0
        copied[GOLD_HOLDOUT_SPLIT_FIELD] = split
        copied["outcome_holdout_source"] = source
        copied["outcome_holdout_version"] = 1
        copied["candidate_id"] = str(copied.get("candidate_id") or _fallback_candidate_id(copied, source))
        for metadata_key in (
            "candidate_source",
            "content_bucket",
            "quality_bucket",
            "platform",
            "platform_source",
        ):
            if not copied.get(metadata_key) and replayed_snapshot.get(metadata_key):
                copied[metadata_key] = replayed_snapshot.get(metadata_key)
        if not copied.get("candidate_source"):
            copied["candidate_source"] = str(features.get("source") or source)
        prepared.append(copied)
        seen_identities.update(identities)
    return prepared


def _is_cross_platform_positive_split(row: dict, split: str, *, holdout_ratio: float) -> bool:
    existing_split = str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower()
    if existing_split and existing_split != split:
        return False
    if not existing_split and _stable_positive_split(row, holdout_ratio=holdout_ratio) != split:
        return False
    if int(row.get("label", -1)) != 1:
        return False
    source = str(row.get("dataset_source") or "").strip()
    label_source = str(row.get("label_source") or "").strip()
    tier = str(row.get("supervision_tier") or "").strip()
    return (
        source in {"cross_platform_audio_fingerprint", "cross_platform_transcript_match"}
        or label_source in {"cross_platform_audio_fingerprint", "cross_platform_transcript_match"}
        or tier == "cross_platform_outcome"
    )


def _stable_positive_split(row: dict, *, holdout_ratio: float) -> str:
    ratio = min(max(float(holdout_ratio), 0.0), 0.9)
    if ratio <= 0:
        return "train"
    vod_id = str(row.get("vod_id") or "").strip()
    key = f"vod:{vod_id}" if vod_id else _fallback_candidate_id(row, "cross_platform_outcome")
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "holdout" if bucket < ratio else "train"


def _is_auto_negative_split(row: dict, split: str) -> bool:
    if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() != split:
        return False
    if int(row.get("label") or 0) != 0:
        return False
    tier = str(row.get("supervision_tier") or "").strip().lower()
    source = str(row.get("dataset_source") or "").strip().lower()
    return tier == "auto_outcome_negative" or source == f"auto_outcome_negative_{split}"


def _normalize_split(split: str) -> str:
    normalized = str(split or "").strip().lower()
    if normalized not in {"train", "holdout"}:
        raise ValueError(f"Unsupported outcome split: {split}")
    return normalized


def _fallback_candidate_id(row: dict, source: str) -> str:
    vod_id = str(row.get("vod_id") or row.get("source_url") or "unknown").strip()
    start = _safe_float(row.get("window_start") or row.get("proxy_window_start"))
    end = _safe_float(row.get("window_end") or row.get("proxy_window_end"))
    return f"{source}:{vod_id}:{start:.2f}:{end:.2f}"


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
