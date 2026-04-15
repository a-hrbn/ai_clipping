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
                "user_feedback_tags": list(feedback.get("user_feedback_tags") or []),
                "feedback_notes": feedback.get("feedback_notes"),
                "publish_platform": feedback.get("publish_platform"),
                "final_variant_key": feedback.get("final_variant_key"),
                "candidate_source": feature.get("source"),
                "content_bucket": _content_bucket(feature),
                "view_count": feedback.get("view_count"),
                "like_count": feedback.get("like_count"),
                "comment_count": feedback.get("comment_count"),
                "share_count": feedback.get("share_count"),
                "save_count": feedback.get("save_count"),
                "impressions": feedback.get("impressions"),
                "avg_watch_time": feedback.get("avg_watch_time"),
                "retention_3s": feedback.get("retention_3s"),
                "retention_50pct": feedback.get("retention_50pct"),
                "completion_rate": feedback.get("completion_rate"),
                "views_1h": feedback.get("views_1h"),
                "views_24h": feedback.get("views_24h"),
                "baseline_view_rate": feedback.get("baseline_view_rate"),
                "observed_view_rate": feedback.get("observed_view_rate"),
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
    outcome_keys = {
        "view_count",
        "like_count",
        "comment_count",
        "share_count",
        "save_count",
        "impressions",
        "avg_watch_time",
        "retention_3s",
        "retention_50pct",
        "completion_rate",
        "views_1h",
        "views_24h",
        "baseline_view_rate",
        "observed_view_rate",
    }
    if any(feedback.get(key) not in (None, "") for key in outcome_keys):
        return True
    return bool(feedback.get("user_feedback_tags"))


def _resolved_feedback_label(feedback: dict) -> tuple[int, str, float, float, str, str, int]:
    score = 0.0
    tags = set(feedback.get("user_feedback_tags") or [])
    if "strong_hook" in tags:
        score += 0.35
    if "too_much_context" in tags:
        score -= 0.35
    if "missing_payoff" in tags:
        score -= 0.4
    if "too_long" in tags:
        score -= 0.25

    baseline_rate = _as_float(feedback.get("baseline_view_rate"))
    observed_rate = _as_float(feedback.get("observed_view_rate"))
    if baseline_rate > 0 and observed_rate > 0:
        ratio = observed_rate / baseline_rate
        if ratio >= 1.15:
            score += 0.9
        elif ratio <= 0.85:
            score -= 0.9

    views_1h = _as_float(feedback.get("views_1h"))
    views_24h = _as_float(feedback.get("views_24h"))
    if views_1h > 0 and views_24h > 0:
        growth = views_24h / views_1h
        if growth >= 5.0:
            score += 0.45
        elif growth <= 1.5:
            score -= 0.35

    shares = _as_float(feedback.get("share_count"))
    saves = _as_float(feedback.get("save_count"))
    impressions = max(_as_float(feedback.get("impressions")), 1.0)
    views = max(_as_float(feedback.get("view_count")), 1.0)
    if shares > 0:
        score += min(shares / views, 0.05) * 20.0
    if saves > 0:
        score += min(saves / views, 0.05) * 18.0
    retention_3s = _as_float(feedback.get("retention_3s"))
    retention_50pct = _as_float(feedback.get("retention_50pct"))
    completion_rate = _as_float(feedback.get("completion_rate"))
    avg_watch_time = _as_float(feedback.get("avg_watch_time"))
    if retention_3s >= 0.75:
        score += 0.55
    elif 0.0 < retention_3s <= 0.45:
        score -= 0.45
    if retention_50pct >= 0.4:
        score += 0.35
    elif 0.0 < retention_50pct <= 0.18:
        score -= 0.35
    if completion_rate >= 0.5:
        score += 0.4
    elif 0.0 < completion_rate <= 0.2:
        score -= 0.35
    if avg_watch_time > 0 and impressions > 0 and avg_watch_time / max(impressions, 1.0) > 0.02:
        score += 0.1

    if abs(score) >= 0.6:
        label = 1 if score > 0 else 0
        label_class = "outcome_positive" if label == 1 else "outcome_negative"
        return label, "outcome_feedback", round(score, 3), 2.4, label_class, "outcome_feedback", 3 if label == 1 else 0

    fallback_label = 1 if bool(feedback.get("selected")) else 0
    label_class = "pipeline_positive" if fallback_label == 1 else "pipeline_negative"
    ordinal_label = 2 if fallback_label == 1 else 0
    return fallback_label, "pipeline_feedback", round(score, 3), 1.0, label_class, "pipeline_feedback", ordinal_label


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _feature_provenance(feature: dict) -> dict[str, object]:
    return {
        "has_llm_semantics": bool(feature.get("has_llm_semantics", False)),
        "has_chat_features": bool(feature.get("has_chat_features", False)),
        "has_vector_similarity": bool(feature.get("has_vector_similarity", False)),
        "has_vod_context": bool(feature.get("vod_progress", 0.0) or feature.get("clip_overlap_count", 0.0)),
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
