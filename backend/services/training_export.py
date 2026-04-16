from __future__ import annotations

import json
import pathlib


def export_training_rows(storage_dir: pathlib.Path, output_path: pathlib.Path) -> int:
    rows: list[dict] = []
    for job_dir in sorted(p for p in storage_dir.iterdir() if p.is_dir()):
        debug_dir = job_dir / "debug"
        features_path = debug_dir / "candidate_features.jsonl"
        feedback_path = debug_dir / "feedback.jsonl"
        if not features_path.exists() or not feedback_path.exists():
            continue

        feature_rows = _read_jsonl(features_path)
        feedback_rows = {row["candidate_id"]: row for row in _read_jsonl(feedback_path)}
        for feature in feature_rows:
            feedback = feedback_rows.get(feature["candidate_id"])
            if feedback is None:
                continue
            if not _should_export_feedback(feedback):
                continue
            (
                label,
                label_source,
                label_score,
                sample_weight,
                label_class,
                supervision_tier,
                ordinal_label,
            ) = _resolved_feedback_label(feedback)
            row = {
                "job_id": feedback["job_id"],
                "candidate_id": feature["candidate_id"],
                "dataset_source": _dataset_source(feedback),
                "label": label,
                "selected": feedback["selected"],
                "exported": feedback["exported"],
                "proposed_start": feedback.get("proposed_start"),
                "proposed_end": feedback.get("proposed_end"),
                "final_start": feedback["final_start"],
                "final_end": feedback["final_end"],
                "rejected_reason": feedback["rejected_reason"],
                "rejection_source": feedback.get("rejection_source"),
                "label_source": label_source,
                "label_score": label_score,
                "label_class": label_class,
                "sample_weight": sample_weight,
                "supervision_tier": supervision_tier,
                "supervision_reason": feedback.get("rejection_source") or label_source,
                "ordinal_label": ordinal_label,
                "feedback_rating": feedback.get("feedback_rating"),
                "feedback_notes": feedback.get("feedback_notes"),
                "publish_platform": feedback.get("publish_platform"),
                "final_variant_key": feedback.get("final_variant_key"),
                "candidate_source": feature.get("source"),
                "content_bucket": _content_bucket(feature),
                "feature_provenance": _feature_provenance(feature),
                "features": feature,
            }
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _should_export_feedback(feedback: dict) -> bool:
    if _has_real_outcome_feedback(feedback):
        return True
    if bool(feedback.get("selected")):
        return True
    return feedback.get("rejection_source") == "low_rank"


def _has_real_outcome_feedback(feedback: dict) -> bool:
    return str(feedback.get("feedback_rating") or "").strip().lower() in {"good", "medium", "bad"}


def _resolved_feedback_label(feedback: dict) -> tuple[int, str, float, float, str, str, int]:
    rating = str(feedback.get("feedback_rating") or "").strip().lower()
    if rating == "good":
        return 1, "feedback_rating", 1.0, 2.2, "outcome_positive", "outcome_feedback", 3
    if rating == "bad":
        return 0, "feedback_rating", -1.0, 1.8, "outcome_negative", "outcome_feedback", 0
    if rating == "medium":
        fallback_label = 1 if bool(feedback.get("selected")) else 0
        label_class = "pipeline_positive" if fallback_label == 1 else "pipeline_negative"
        return fallback_label, "feedback_rating_neutral", 0.0, 0.7, label_class, "feedback_rating_neutral", 1

    fallback_label = 1 if bool(feedback.get("selected")) else 0
    label_class = "pipeline_positive" if fallback_label == 1 else "pipeline_negative"
    ordinal_label = 2 if fallback_label == 1 else 0
    return fallback_label, "pipeline_feedback", 0.0, 1.0, label_class, "pipeline_feedback", ordinal_label


def _feature_provenance(feature: dict) -> dict[str, object]:
    return {
        "has_llm_semantics": bool(feature.get("has_llm_semantics", False)),
        "has_chat_features": bool(feature.get("has_chat_features", False)),
        "has_vector_similarity": bool(feature.get("has_vector_similarity", False)),
        "has_vod_context": bool(feature.get("vod_progress", 0.0) or feature.get("clip_overlap_count", 0.0)),
        "has_vod_memory": bool(
            feature.get("retrieved_memory_span_count", 0.0)
            or feature.get("setup_depth_score", 0.0)
            or feature.get("callback_strength", 0.0)
        ),
        "feature_build_version": int(feature.get("feature_build_version", 1) or 1),
        "has_dynamic_trend": bool(feature.get("dynamic_trend_score", 0.0)),
        "has_variant_projection": bool(feature.get("projected_retention_score", 0.0)),
    }


def _dataset_source(feedback: dict) -> str:
    if _has_real_outcome_feedback(feedback):
        return "feedback_outcome_export"
    if bool(feedback.get("selected")):
        return "feedback_selected_export"
    return "feedback_low_rank_export"


def _content_bucket(feature: dict) -> str:
    source = str(feature.get("source") or "").strip().lower()
    if source in {"quote", "hot_take", "dialogue", "controversy"}:
        return "quote_driven"
    if source in {"reaction", "clean_reaction", "chat_sync"}:
        return "reaction"
    if source in {"visual_reveal", "audio_burst", "hype_peak"}:
        return "visual_action"
    if source in {"story_payoff", "narrative_reveal", "dense_window", "tension_build"}:
        return "story"
    return "general"
