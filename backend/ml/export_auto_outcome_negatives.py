from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Iterable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.gold_eval_policy import (
    GOLD_HOLDOUT_KEY_FIELD,
    GOLD_HOLDOUT_POLICY_FIELD,
    GOLD_HOLDOUT_SPLIT_FIELD,
)


AUTO_OUTCOME_NEGATIVE_POLICY = "auto_outcome_negative"
AUTO_OUTCOME_NEGATIVE_SOURCE = "auto_outcome_negative"
AUTO_OUTCOME_NEGATIVE_VERSION = 1
HARD_NEGATIVE_CLASS = "auto_outcome_hard_negative"
BACKGROUND_NEGATIVE_CLASS = "auto_outcome_background_negative"
POSITIVE_LABEL_CLASSES = {"positive", "weak_positive", "viral_positive", "gold_positive"}
DEFAULT_HARD_SCORE_THRESHOLD = 7.5


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build train/holdout negatives from old candidates with no positive short-form outcome."
    )
    parser.add_argument(
        "--candidate-rows",
        action="append",
        default=[],
        help="JSONL files containing feature-backed candidate rows. Can be passed multiple times.",
    )
    parser.add_argument(
        "--positive-rows",
        action="append",
        default=[],
        help="JSONL files containing known positive rows used for dedupe and proximity exclusion.",
    )
    parser.add_argument(
        "--existing-rows",
        action="append",
        default=[],
        help="Optional JSONL files whose identities should not be exported again.",
    )
    parser.add_argument(
        "--rows-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_auto_outcome_negative_rows.jsonl"),
    )
    parser.add_argument(
        "--train-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_auto_outcome_negative_train_rows.jsonl"),
    )
    parser.add_argument(
        "--holdout-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_auto_outcome_negative_holdout_rows.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_auto_outcome_negative_summary.json"),
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--min-age-days", type=int, default=0)
    parser.add_argument("--holdout-ratio", type=float, default=0.20)
    parser.add_argument("--positive-exclusion-sec", type=float, default=180.0)
    parser.add_argument("--hard-score-threshold", type=float, default=DEFAULT_HARD_SCORE_THRESHOLD)
    parser.add_argument("--hard-sample-weight", type=float, default=1.0)
    parser.add_argument("--background-sample-weight", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument(
        "--skip-undated",
        action="store_true",
        help="Skip candidates without a parseable date when --min-age-days is greater than zero.",
    )
    args = parser.parse_args()

    candidate_paths = [pathlib.Path(path) for path in args.candidate_rows]
    positive_paths = [pathlib.Path(path) for path in args.positive_rows]
    existing_paths = [pathlib.Path(path) for path in args.existing_rows]
    candidate_rows = _read_jsonl_many(candidate_paths)
    positive_rows = _read_jsonl_many(positive_paths)
    existing_rows = _read_jsonl_many(existing_paths)
    payload = build_auto_outcome_negative_rows(
        candidate_rows=candidate_rows,
        positive_rows=positive_rows,
        existing_rows=existing_rows,
        profile=args.profile,
        as_of_date=_parse_date(args.as_of_date) or date.today(),
        min_age_days=max(int(args.min_age_days), 0),
        holdout_ratio=_clamp_ratio(args.holdout_ratio),
        positive_exclusion_sec=max(float(args.positive_exclusion_sec), 0.0),
        hard_score_threshold=float(args.hard_score_threshold),
        hard_sample_weight=max(float(args.hard_sample_weight), 0.0),
        background_sample_weight=max(float(args.background_sample_weight), 0.0),
        max_rows=max(int(args.max_rows), 0),
        allow_undated=not args.skip_undated,
        candidate_paths=candidate_paths,
        positive_paths=positive_paths,
        existing_paths=existing_paths,
    )
    rows = payload["rows"]
    train_rows = [row for row in rows if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "") == "train"]
    holdout_rows = [row for row in rows if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "") == "holdout"]
    _write_jsonl(pathlib.Path(args.rows_out), rows)
    _write_jsonl(pathlib.Path(args.train_out), train_rows)
    _write_jsonl(pathlib.Path(args.holdout_out), holdout_rows)
    _write_json(pathlib.Path(args.summary_out), payload["summary"])
    print(json.dumps(payload["summary"], ensure_ascii=True))
    return 0


def build_auto_outcome_negative_rows(
    *,
    candidate_rows: list[dict],
    positive_rows: list[dict],
    existing_rows: list[dict] | None = None,
    profile: str = "jynxzi",
    as_of_date: date | None = None,
    min_age_days: int = 0,
    holdout_ratio: float = 0.20,
    positive_exclusion_sec: float = 180.0,
    hard_score_threshold: float = DEFAULT_HARD_SCORE_THRESHOLD,
    hard_sample_weight: float = 1.0,
    background_sample_weight: float = 0.5,
    max_rows: int = 2000,
    allow_undated: bool = True,
    candidate_paths: list[pathlib.Path] | None = None,
    positive_paths: list[pathlib.Path] | None = None,
    existing_paths: list[pathlib.Path] | None = None,
) -> dict[str, object]:
    existing_rows = existing_rows or []
    as_of = as_of_date or date.today()
    holdout_ratio = _clamp_ratio(holdout_ratio)
    positive_index = _positive_window_index(positive_rows)
    blocked_identity_keys = _positive_identity_key_set(positive_rows) | _identity_key_set(existing_rows)
    seen_identity_keys: set[str] = set()
    skipped: Counter[str] = Counter()
    selected: list[dict] = []

    for row in candidate_rows:
        if not isinstance(row, dict):
            skipped["invalid_row"] += 1
            continue
        if _is_positive_row(row):
            skipped["positive_candidate"] += 1
            continue
        if not isinstance(row.get("features"), dict) or not row.get("features"):
            skipped["missing_features"] += 1
            continue
        identity_keys = _identity_keys(row)
        if not identity_keys:
            skipped["missing_identity"] += 1
            continue
        if identity_keys & blocked_identity_keys:
            skipped["existing_or_positive_identity"] += 1
            continue
        if identity_keys & seen_identity_keys:
            skipped["duplicate_candidate_identity"] += 1
            continue
        row_date = _row_date(row)
        if min_age_days > 0:
            if row_date is None:
                if not allow_undated:
                    skipped["undated_candidate"] += 1
                    continue
            elif (as_of - row_date).days < min_age_days:
                skipped["candidate_too_recent"] += 1
                continue
        if _near_positive_window(row, positive_index, positive_exclusion_sec):
            skipped["positive_proximity"] += 1
            continue
        prepared = _auto_negative_row(
            row,
            profile=profile,
            identity_key=_primary_identity_key(row),
            split=_stable_split(_primary_identity_key(row), holdout_ratio=holdout_ratio),
            hard_score_threshold=hard_score_threshold,
            hard_sample_weight=hard_sample_weight,
            background_sample_weight=background_sample_weight,
        )
        selected.append(prepared)
        seen_identity_keys.update(identity_keys)

    selected.sort(
        key=lambda row: (
            str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "") != "train",
            str(row.get("auto_outcome_negative_tier") or "") != "hard_negative",
            -float(row.get("auto_outcome_negative_score") or 0.0),
            str(row.get(GOLD_HOLDOUT_KEY_FIELD) or ""),
        )
    )
    if max_rows:
        selected = selected[:max_rows]

    summary = {
        "profile": profile,
        "version": AUTO_OUTCOME_NEGATIVE_VERSION,
        "candidate_paths": [str(path) for path in candidate_paths or []],
        "positive_paths": [str(path) for path in positive_paths or []],
        "existing_paths": [str(path) for path in existing_paths or []],
        "candidate_rows": len(candidate_rows),
        "positive_rows": len(positive_rows),
        "existing_rows": len(existing_rows),
        "exported_rows": len(selected),
        "train_rows": sum(1 for row in selected if row.get(GOLD_HOLDOUT_SPLIT_FIELD) == "train"),
        "holdout_rows": sum(1 for row in selected if row.get(GOLD_HOLDOUT_SPLIT_FIELD) == "holdout"),
        "by_tier": dict(Counter(str(row.get("auto_outcome_negative_tier") or "") for row in selected)),
        "by_label_class": dict(Counter(str(row.get("label_class") or "") for row in selected)),
        "by_dataset_source": dict(Counter(str(row.get("dataset_source") or "") for row in selected)),
        "unique_vods": len({str(row.get("vod_id") or "") for row in selected if str(row.get("vod_id") or "")}),
        "skipped": dict(skipped),
        "holdout_ratio": holdout_ratio,
        "positive_exclusion_sec": float(positive_exclusion_sec),
        "hard_score_threshold": float(hard_score_threshold),
        "min_age_days": int(min_age_days),
        "allow_undated": bool(allow_undated),
        "as_of_date": as_of.isoformat(),
    }
    return {"rows": selected, "summary": summary}


def _auto_negative_row(
    row: dict,
    *,
    profile: str,
    identity_key: str,
    split: str,
    hard_score_threshold: float,
    hard_sample_weight: float,
    background_sample_weight: float,
) -> dict:
    copied = dict(row)
    copied["features"] = dict(row.get("features") or {})
    score = _candidate_score(row)
    tier = "hard_negative" if score >= float(hard_score_threshold) else "background_negative"
    copied["candidate_id"] = str(copied.get("candidate_id") or f"auto_outcome_negative:{_stable_hash(identity_key)}")
    copied["label"] = 0
    copied["selected"] = False
    copied["exported"] = False
    copied["dataset_source"] = f"{AUTO_OUTCOME_NEGATIVE_SOURCE}_{split}"
    copied["label_source"] = AUTO_OUTCOME_NEGATIVE_SOURCE
    copied["supervision_tier"] = AUTO_OUTCOME_NEGATIVE_SOURCE
    copied["supervision_reason"] = "candidate_without_positive_short_form_outcome"
    copied["label_class"] = HARD_NEGATIVE_CLASS if tier == "hard_negative" else BACKGROUND_NEGATIVE_CLASS
    copied["ordinal_label"] = 0
    copied[GOLD_HOLDOUT_SPLIT_FIELD] = split
    copied[GOLD_HOLDOUT_POLICY_FIELD] = AUTO_OUTCOME_NEGATIVE_POLICY
    copied[GOLD_HOLDOUT_KEY_FIELD] = identity_key
    copied["auto_outcome_negative_tier"] = tier
    copied["auto_outcome_negative_score"] = round(score, 6)
    copied["auto_outcome_negative_source"] = AUTO_OUTCOME_NEGATIVE_SOURCE
    copied["auto_outcome_negative_version"] = AUTO_OUTCOME_NEGATIVE_VERSION
    copied["sample_weight"] = 1.0
    copied["auto_outcome_negative_recommended_sample_weight"] = round(
        hard_sample_weight if tier == "hard_negative" else background_sample_weight,
        3,
    )
    creator = str(copied.get("creator_id") or copied.get("streamer") or copied.get("profile_hint") or "").strip().lower()
    copied["profile_hint"] = creator or profile
    copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
    copied["feature_provenance"]["auto_outcome_negative"] = True
    return copied


def _positive_window_index(rows: Iterable[dict]) -> dict[str, list[tuple[float, float]]]:
    index: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if not _is_positive_row(row):
            continue
        window = _row_window(row)
        if window is None:
            continue
        vod_key = _vod_key(row)
        if vod_key:
            index[vod_key].append(window)
    return dict(index)


def _near_positive_window(
    row: dict,
    positive_index: dict[str, list[tuple[float, float]]],
    exclusion_sec: float,
) -> bool:
    vod_key = _vod_key(row)
    window = _row_window(row)
    if not vod_key or window is None:
        return False
    for positive_window in positive_index.get(vod_key, []):
        expanded = (positive_window[0] - exclusion_sec, positive_window[1] + exclusion_sec)
        if _windows_overlap(window, expanded):
            return True
    return False


def _is_positive_row(row: dict) -> bool:
    try:
        if int(row.get("label", -1)) == 1:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if int(row.get("gold_grade", -1)) >= 2:
            return True
    except (TypeError, ValueError):
        pass
    return str(row.get("label_class") or "").strip().lower() in POSITIVE_LABEL_CLASSES


def _identity_key_set(rows: Iterable[dict]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(_identity_keys(row))
    return keys


def _positive_identity_key_set(rows: Iterable[dict]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        if _is_positive_row(row):
            keys.update(_identity_keys(row))
    return keys


def _identity_keys(row: dict) -> set[str]:
    keys: set[str] = set()
    clip_ids = _clip_ids(row)
    if clip_ids:
        keys.add("clips:" + ",".join(clip_ids))
        keys.update(f"clip:{clip_id}" for clip_id in clip_ids)
    for field in ("queue_id",):
        value = str(row.get(field) or "").strip()
        if value:
            keys.add(f"{field}:{value}")
    candidate_id = str(row.get("candidate_id") or "").strip()
    job_id = str(row.get("job_id") or "").strip()
    vod_id = str(row.get("vod_id") or "").strip()
    if candidate_id and job_id:
        keys.add(f"candidate:{job_id}:{candidate_id}")
    elif candidate_id:
        keys.add(f"candidate:{candidate_id}")
    window = _row_window(row)
    source_url = str(row.get("source_url") or row.get("vod_url") or row.get("review_url") or "").strip()
    if window is not None:
        start, end = window
        if vod_id:
            keys.add(f"vod_window:{vod_id}:{start:.2f}:{end:.2f}")
        if source_url:
            keys.add(f"url_window:{source_url}:{start:.2f}:{end:.2f}")
    return keys


def _primary_identity_key(row: dict) -> str:
    keys = sorted(_identity_keys(row))
    if keys:
        return keys[0]
    return _stable_hash(json.dumps(row, sort_keys=True, default=str))


def _clip_ids(row: dict) -> tuple[str, ...]:
    raw = row.get("clip_ids") or []
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _row_window(row: dict) -> tuple[float, float] | None:
    start = _optional_float(row.get("window_start"))
    end = _optional_float(row.get("window_end"))
    if start is None:
        start = _optional_float(row.get("proxy_window_start"))
    if end is None:
        end = _optional_float(row.get("proxy_window_end"))
    if start is None:
        start = _optional_float(row.get("final_start"))
    if end is None:
        end = _optional_float(row.get("final_end"))
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


def _vod_key(row: dict) -> str:
    vod_id = str(row.get("vod_id") or "").strip()
    if vod_id:
        return f"vod:{vod_id}"
    source_url = str(row.get("source_url") or row.get("vod_url") or row.get("review_url") or "").strip()
    if source_url:
        return f"url:{source_url}"
    return ""


def _windows_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0])


def _row_date(row: dict) -> date | None:
    for key in (
        "vod_created_at",
        "stream_started_at",
        "broadcast_started_at",
        "started_at",
        "published_at",
        "cluster_created_at",
        "created_at",
        "upload_date",
    ):
        parsed = _parse_date(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None


def _candidate_score(row: dict) -> float:
    for key in (
        "forced_score",
        "heuristic_rank_score",
        "rank_score",
        "score",
        "trained_score",
        "heuristic_score",
    ):
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    for key in ("rank_score", "heuristic_rank_score", "hook_strength", "moment_score"):
        value = _optional_float(features.get(key))
        if value is not None:
            return value
    return 0.0


def _stable_split(identity_key: str, *, holdout_ratio: float) -> str:
    digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "holdout" if bucket < holdout_ratio else "train"


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _clamp_ratio(value: object) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(raw, 1.0))


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_jsonl_many(paths: Iterable[pathlib.Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    return rows


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    _replace_text(path, payload + ("\n" if payload else ""))


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_text(path, json.dumps(payload, ensure_ascii=True, indent=2))


def _replace_text(path: pathlib.Path, payload: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
