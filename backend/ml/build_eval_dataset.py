from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import train_ranker
from services.gold_eval_policy import (
    GOLD_HOLDOUT_SPLIT_FIELD,
    annotate_gold_row,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a reproducible eval dataset from exported training rows.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "tmp" / "storage" / "feedback" / "training_rows.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "eval_dataset.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "eval_dataset_summary.json"),
    )
    args = parser.parse_args()

    rows = train_ranker._read_jsonl(pathlib.Path(args.dataset))
    if not rows:
        print("No dataset rows found.")
        return 1

    eval_rows = build_eval_dataset(rows)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in eval_rows),
        encoding="utf-8",
    )

    summary = eval_dataset_summary(eval_rows)
    summary_path = pathlib.Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))
    return 0


def build_eval_dataset(rows: list[dict]) -> list[dict]:
    eval_rows: list[dict] = []
    for row in rows:
        eval_row = _annotated_eval_row(row)
        features = dict(row.get("features") or {})
        eval_row["eval_group"] = _eval_group(row)
        eval_row["platform_bucket"] = _platform_bucket(row)
        eval_row["variant_bucket"] = _variant_bucket(row)
        eval_row["source_bucket"] = _source_bucket(row, features)
        eval_row["content_bucket"] = _content_bucket(row, features)
        eval_row["has_outcome_feedback"] = bool(_has_outcome_feedback(row))
        eval_row["rank_target"] = int(row.get("ordinal_label") or row.get("label") or 0)
        eval_row["dataset_role"] = _dataset_role(row)
        eval_rows.append(eval_row)
    return eval_rows


def eval_dataset_summary(rows: list[dict]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "positive_rows": sum(1 for row in rows if int(row.get("label", 0)) == 1),
        "negative_rows": sum(1 for row in rows if int(row.get("label", 0)) == 0),
        "eval_groups": len({row["eval_group"] for row in rows if row.get("eval_group")}),
        "dataset_roles": _count_values(rows, "dataset_role"),
        "by_label_class": _count_values(rows, "label_class"),
        "by_supervision_tier": _count_values(rows, "supervision_tier"),
        "by_platform": _count_values(rows, "platform_bucket"),
        "by_variant": _count_values(rows, "variant_bucket"),
        "by_source": _count_values(rows, "source_bucket"),
        "by_content_bucket": _count_values(rows, "content_bucket"),
        "gold_holdout_splits": _count_values(rows, GOLD_HOLDOUT_SPLIT_FIELD),
    }


def _eval_group(row: dict) -> str:
    for key in ("vod_id", "source_url", "job_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(row.get("candidate_id") or "unknown")


def _platform_bucket(row: dict) -> str:
    value = str(row.get("publish_platform") or "").strip().lower()
    if value:
        return value
    return "general"


def _variant_bucket(row: dict) -> str:
    value = str(row.get("final_variant_key") or "").strip().lower()
    if value:
        return value
    return "balanced"


def _source_bucket(row: dict, features: dict) -> str:
    value = str(row.get("candidate_source") or features.get("source") or "").strip().lower()
    return value or "unknown"


def _content_bucket(row: dict, features: dict) -> str:
    preset = str(row.get("content_bucket") or "").strip().lower()
    if preset:
        return preset
    source = _source_bucket(row, features)
    if source in {"quote", "hot_take", "dialogue", "controversy"}:
        return "quote_driven"
    if source in {"reaction", "clean_reaction", "chat_sync"}:
        return "reaction"
    if source in {"visual_reveal", "audio_burst", "hype_peak"}:
        return "visual_action"
    if source in {"story_payoff", "narrative_reveal", "dense_window", "tension_build"}:
        return "story"
    return "general"


def _has_outcome_feedback(row: dict) -> bool:
    for key in (
        "impressions",
        "avg_watch_time",
        "retention_3s",
        "retention_50pct",
        "completion_rate",
        "views_1h",
        "views_24h",
        "share_count",
        "save_count",
        "baseline_view_rate",
        "observed_view_rate",
    ):
        if row.get(key) not in (None, ""):
            return True
    return False


def _dataset_role(row: dict) -> str:
    if str(row.get("supervision_tier") or "").strip().lower() == "gold_eval":
        return "gold_eval"
    if str(row.get("label_source") or "").strip().lower() == "gold_eval":
        return "gold_eval"
    if _has_outcome_feedback(row):
        return "outcome_eval"
    if bool(row.get("selected")):
        return "pipeline_positive_eval"
    return "pipeline_negative_eval"


def _annotated_eval_row(row: dict) -> dict:
    if _is_gold_eval_row(row):
        return annotate_gold_row(row)
    return dict(row)


def _is_gold_eval_row(row: dict) -> bool:
    if str(row.get("supervision_tier") or "").strip().lower() == "gold_eval":
        return True
    if str(row.get("label_source") or "").strip().lower() == "gold_eval":
        return True
    return False


def _count_values(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
