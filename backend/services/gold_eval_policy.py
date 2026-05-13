from __future__ import annotations

import hashlib


GOLD_HOLDOUT_SPLIT_FIELD = "gold_holdout_split"
GOLD_HOLDOUT_KEY_FIELD = "gold_holdout_key"
GOLD_HOLDOUT_POLICY_FIELD = "gold_holdout_policy"
GOLD_HOLDOUT_POLICY_TRAIN_ONLY = "train_only"
GOLD_HOLDOUT_POLICY_TOP_VIEWED_AUTO = "top_viewed_auto"
GOLD_HOLDOUT_POLICY_AUTO_OUTCOME_NEGATIVE = "auto_outcome_negative"
GOLD_HOLDOUT_RATIO = 0.30
TOP_VIEWED_GOLD_HOLDOUT_RATIO = 0.20
GOLD_HOLDOUT_VERSION = 1
HIGH_HEURISTIC_SCORE_FOR_NEGATIVE_NOTES = 8.0


def validate_gold_notes(
    gold_grade: int,
    notes: str | None,
    *,
    heuristic_rank_score: float | None = None,
) -> None:
    normalized_notes = str(notes or "").strip()
    if gold_grade in {2, 3} and not normalized_notes:
        raise ValueError("gold_notes are required for gold grades 2 and 3.")
    if (
        gold_grade == 0
        and heuristic_rank_score is not None
        and float(heuristic_rank_score) >= HIGH_HEURISTIC_SCORE_FOR_NEGATIVE_NOTES
        and not normalized_notes
    ):
        raise ValueError(
            "gold_notes are required for grade 0 when the heuristic rank score is high."
        )


def annotate_gold_row(row: dict) -> dict:
    annotated = dict(row)
    split_key = _gold_holdout_key(annotated)
    annotated[GOLD_HOLDOUT_KEY_FIELD] = split_key
    if _gold_holdout_policy(annotated) == GOLD_HOLDOUT_POLICY_TRAIN_ONLY:
        annotated[GOLD_HOLDOUT_SPLIT_FIELD] = "train"
        annotated["gold_holdout_ratio"] = 0.0
    elif _gold_holdout_policy(annotated) == GOLD_HOLDOUT_POLICY_TOP_VIEWED_AUTO:
        existing_split = str(annotated.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower()
        annotated[GOLD_HOLDOUT_SPLIT_FIELD] = existing_split if existing_split in {"train", "holdout"} else (
            _gold_holdout_bucket(split_key, ratio=TOP_VIEWED_GOLD_HOLDOUT_RATIO)
        )
        annotated["gold_holdout_ratio"] = TOP_VIEWED_GOLD_HOLDOUT_RATIO
    else:
        annotated[GOLD_HOLDOUT_SPLIT_FIELD] = _gold_holdout_bucket(split_key, ratio=GOLD_HOLDOUT_RATIO)
        annotated["gold_holdout_ratio"] = GOLD_HOLDOUT_RATIO
    annotated["gold_holdout_version"] = GOLD_HOLDOUT_VERSION
    return annotated


def annotate_gold_rows(rows: list[dict]) -> list[dict]:
    return [annotate_gold_row(row) for row in rows]


def _gold_holdout_key(row: dict) -> str:
    for key in ("queue_id", "candidate_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    source_url = str(row.get("source_url") or "").strip()
    window_start = round(float(row.get("window_start") or 0.0), 2)
    window_end = round(float(row.get("window_end") or 0.0), 2)
    return f"{source_url}|{window_start:.2f}|{window_end:.2f}"


def _gold_holdout_policy(row: dict) -> str:
    return str(row.get(GOLD_HOLDOUT_POLICY_FIELD) or "").strip().lower()


def _gold_holdout_bucket(split_key: str, *, ratio: float) -> str:
    digest = hashlib.sha256(split_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "holdout" if bucket < ratio else "train"
