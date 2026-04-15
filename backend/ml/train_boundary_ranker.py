from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import train_ranker
from services.boundary_scorer import (
    END_MODEL_FEATURE_NAMES,
    PAIR_MODEL_FEATURE_NAMES,
    START_MODEL_FEATURE_NAMES,
)

_DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
_START_POSITIVE_THRESHOLD = 0.25
_START_NEGATIVE_THRESHOLD = 0.75
_END_POSITIVE_THRESHOLD = 0.25
_END_NEGATIVE_THRESHOLD = 0.75
_PAIR_POSITIVE_THRESHOLD = 0.35
_PAIR_NEGATIVE_THRESHOLD = 1.25


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a boundary refinement ranker from boundary-labeled training rows.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", default="general")
    parser.add_argument("--artifact-dir", default=str(_DEFAULT_ARTIFACT_DIR))
    args = parser.parse_args()

    rows = _read_jsonl(pathlib.Path(args.dataset))
    return train_from_rows(rows, args.profile, pathlib.Path(args.artifact_dir))


def train_from_rows(rows: list[dict], profile: str, artifact_dir: pathlib.Path) -> int:
    source_rows = [row for row in rows if _has_boundary_reference(row)]
    if not source_rows:
        print("No boundary reference rows found.")
        return 1

    start_rows = _build_start_examples(source_rows)
    end_rows = _build_end_examples(source_rows)
    pair_rows = _build_pair_examples(source_rows)
    if not start_rows or not end_rows or not pair_rows:
        print("Need start/end/pair examples with both positive and negative labels.")
        return 1

    start_model = _train_logistic_model(start_rows, START_MODEL_FEATURE_NAMES)
    end_model = _train_logistic_model(end_rows, END_MODEL_FEATURE_NAMES)
    pair_model = _train_logistic_model(pair_rows, PAIR_MODEL_FEATURE_NAMES)
    if start_model is None or end_model is None or pair_model is None:
        print("Could not fit one or more boundary models.")
        return 1

    artifact = {
        "version": 1,
        "profile": profile,
        "model_type": "boundary_logistic",
        "blend": {
            "start": 0.35,
            "end": 0.35,
            "pair": 0.40,
        },
        "start_model": start_model,
        "end_model": end_model,
        "pair_model": pair_model,
        "training_stats": {
            "rows": len(rows),
            "boundary_rows": len(source_rows),
            "start_examples": len(start_rows),
            "end_examples": len(end_rows),
            "pair_examples": len(pair_rows),
            "by_label_class": _count_values(source_rows, "label_class"),
            "by_perturbation_kind": _count_values(source_rows, "boundary_perturbation_kind"),
        },
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{profile}_boundary_ranker.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote artifact to {artifact_path}")
    print(
        json.dumps(
            {
                "start_validation": start_model["validation"],
                "end_validation": end_model["validation"],
                "pair_validation": pair_model["validation"],
                "training_stats": artifact["training_stats"],
            },
            ensure_ascii=True,
        )
    )
    return 0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if row.get("label") in {0, 1}:
                rows.append(row)
    return rows


def _has_boundary_reference(row: dict) -> bool:
    return row.get("boundary_reference_start") not in (None, "") and row.get("boundary_reference_end") not in (None, "")


def _build_start_examples(rows: list[dict]) -> list[dict]:
    examples: list[dict] = []
    for row in rows:
        error = float(row.get("boundary_abs_start_error_sec") or 0.0)
        label = _binary_label(error, _START_POSITIVE_THRESHOLD, _START_NEGATIVE_THRESHOLD)
        if label is None:
            continue
        examples.append(
            {
                "candidate_id": f"{row['candidate_id']}:start",
                "label": label,
                "job_id": row.get("job_id", ""),
                "creator_id": row.get("creator_id", ""),
                "sample_weight": float(row.get("sample_weight") or 1.0),
                "label_class": str(row.get("label_class") or ""),
                "features": _start_feature_row(row),
            }
        )
    return examples


def _build_end_examples(rows: list[dict]) -> list[dict]:
    examples: list[dict] = []
    for row in rows:
        error = float(row.get("boundary_abs_end_error_sec") or 0.0)
        label = _binary_label(error, _END_POSITIVE_THRESHOLD, _END_NEGATIVE_THRESHOLD)
        if label is None:
            continue
        examples.append(
            {
                "candidate_id": f"{row['candidate_id']}:end",
                "label": label,
                "job_id": row.get("job_id", ""),
                "creator_id": row.get("creator_id", ""),
                "sample_weight": float(row.get("sample_weight") or 1.0),
                "label_class": str(row.get("label_class") or ""),
                "features": _end_feature_row(row),
            }
        )
    return examples


def _build_pair_examples(rows: list[dict]) -> list[dict]:
    examples: list[dict] = []
    for row in rows:
        error = float(row.get("boundary_abs_total_error_sec") or 0.0)
        label = _binary_label(error, _PAIR_POSITIVE_THRESHOLD, _PAIR_NEGATIVE_THRESHOLD)
        if label is None:
            continue
        examples.append(
            {
                "candidate_id": f"{row['candidate_id']}:pair",
                "label": label,
                "job_id": row.get("job_id", ""),
                "creator_id": row.get("creator_id", ""),
                "sample_weight": float(row.get("sample_weight") or 1.0),
                "label_class": str(row.get("label_class") or ""),
                "features": _pair_feature_row(row),
            }
        )
    return examples


def _binary_label(error: float, positive_threshold: float, negative_threshold: float) -> int | None:
    if error <= positive_threshold:
        return 1
    if error >= negative_threshold:
        return 0
    return None


def _start_feature_row(row: dict) -> dict[str, float]:
    features = row.get("features") or {}
    hook_strength = _clamp01(features.get("hook_strength", 0.0))
    standalone = _clamp01(features.get("standalone_score", 0.0))
    context_efficiency = _clamp01(features.get("context_efficiency", 0.0))
    slow_open = _clamp01(features.get("slow_open_penalty", 0.0))
    context_dependency = _clamp01(features.get("context_dependency_score", 0.0))
    hook_delay = float(features.get("hook_delay_sec", 0.0) or 0.0)
    return {
        "clean_open_score": _clamp01(0.45 * hook_strength + 0.30 * standalone + 0.25 * context_efficiency),
        "mid_thought_risk": slow_open,
        "context_dependency_risk": context_dependency,
        "hook_proximity_score": hook_strength,
        "duration_prior_score": _duration_prior(hook_delay),
        "speaker_transition_score": _clamp01(float(features.get("speaker_turns", 0.0) or 0.0) / 3.0),
        "scene_transition_score": _clamp01(max(float(features.get("motion_peak", 0.0) or 0.0), float(features.get("topic_shift_strength", 0.0) or 0.0))),
        "pause_strength_score": _clamp01(float(features.get("boundary_left_context_sec", 0.0) or 0.0) / 3.0),
        "lexical_break_score": _clamp01(float(features.get("quoteability_score", 0.0) or 0.0)),
    }


def _end_feature_row(row: dict) -> dict[str, float]:
    features = row.get("features") or {}
    payoff_score = _clamp01(float(features.get("payoff_score", 0.0) or 0.0) / 8.0)
    payoff_clarity = _clamp01(features.get("payoff_clarity_score", 0.0))
    weak_resolution = _clamp01(features.get("weak_resolution_penalty", 0.0))
    post_hook_hype = _clamp01(float(features.get("post_hook_hype", 0.0) or 0.0) / 6.0)
    duration = _duration(row)
    hook_delay = float(features.get("hook_delay_sec", 0.0) or 0.0)
    return {
        "closure_score": _clamp01(0.45 * payoff_clarity + 0.35 * (1.0 - weak_resolution) + 0.20 * _clamp01(features.get("standalone_score", 0.0))),
        "mid_thought_risk": weak_resolution * 0.4,
        "context_dependency_risk": _clamp01(features.get("context_dependency_score", 0.0)) * 0.5,
        "payoff_capture_score": payoff_score,
        "reaction_tail_score": _clamp01(max(post_hook_hype, float(features.get("chat_peak", 0.0) or 0.0), float(features.get("face_reaction_peak", 0.0) or 0.0))),
        "duration_prior_score": _duration_prior(max(duration - hook_delay, 0.0)),
        "scene_transition_score": _clamp01(float(features.get("motion_peak", 0.0) or 0.0)),
        "pause_strength_score": _clamp01(float(features.get("boundary_right_context_sec", 0.0) or 0.0) / 3.0),
        "lexical_break_score": payoff_clarity,
    }


def _pair_feature_row(row: dict) -> dict[str, float]:
    features = row.get("features") or {}
    duration = _duration(row)
    hook_delay = float(features.get("hook_delay_sec", 0.0) or 0.0)
    setup_before_anchor = hook_delay
    payoff_after_anchor = max(duration - hook_delay, 0.0)
    start_row = _start_feature_row(row)
    end_row = _end_feature_row(row)
    start_score = _heuristic_start_from_row(start_row)
    end_score = _heuristic_end_from_row(end_row)
    penalty_count = 0.0
    if _clamp01(features.get("slow_open_penalty", 0.0)) >= 0.45:
        penalty_count += 1.0
    if _clamp01(features.get("context_dependency_score", 0.0)) >= 0.5:
        penalty_count += 1.0
    if _clamp01(features.get("weak_resolution_penalty", 0.0)) >= 0.5:
        penalty_count += 1.0
    if duration > 40.0:
        penalty_count += 1.0

    return {
        "start_score": start_score,
        "end_score": end_score,
        "hook_score": _clamp01(features.get("hook_strength", 0.0)),
        "payoff_score": _clamp01(0.65 * (float(features.get("payoff_score", 0.0) or 0.0) / 8.0) + 0.35 * float(features.get("payoff_clarity_score", 0.0) or 0.0)),
        "narrative_score": _clamp01(
            0.35 * float(features.get("standalone_score", 0.0) or 0.0)
            + 0.35 * float(features.get("context_efficiency", 0.0) or 0.0)
            + 0.30 * min(float(features.get("arc_score", 0.0) or 0.0) / 5.0, 1.0)
        ),
        "duration": duration,
        "setup_before_anchor": setup_before_anchor,
        "payoff_after_anchor": payoff_after_anchor,
        "duration_fit": max(1.0 - min(abs(duration - 22.0) / 22.0, 1.0), 0.0),
        "penalty_count": penalty_count,
        "start_sentence_start": 1.0 if float(features.get("standalone_score", 0.0) or 0.0) >= 0.55 else 0.0,
        "start_scene_cut": 1.0 if float(features.get("topic_shift_strength", 0.0) or 0.0) >= 0.45 else 0.0,
        "start_speaker_turn": _clamp01(float(features.get("speaker_turns", 0.0) or 0.0) / 2.0),
        "start_pause_break": 1.0 if float(features.get("boundary_left_context_sec", 0.0) or 0.0) >= 1.2 else 0.0,
        "end_sentence_end": 1.0 if float(features.get("payoff_clarity_score", 0.0) or 0.0) >= 0.55 else 0.0,
        "end_pause_break": 1.0 if float(features.get("boundary_right_context_sec", 0.0) or 0.0) >= 1.2 else 0.0,
    }


def _train_logistic_model(rows: list[dict], feature_names: tuple[str, ...]) -> dict | None:
    positive = [row for row in rows if int(row["label"]) == 1]
    negative = [row for row in rows if int(row["label"]) == 0]
    if not positive or not negative:
        return None

    split = train_ranker._split_rows(rows)
    train_rows = split["train"]
    validation_rows = split["validation"]
    if not train_rows or not validation_rows:
        return None

    scales = train_ranker._compute_scales(train_rows, list(feature_names))
    train_matrix = train_ranker._feature_matrix(train_rows, list(feature_names), scales)
    validation_matrix = train_ranker._feature_matrix(validation_rows, list(feature_names), scales)
    train_labels = np.array([int(row["label"]) for row in train_rows], dtype=int)
    validation_labels = np.array([int(row["label"]) for row in validation_rows], dtype=int)
    train_weights = np.array([float(row.get("sample_weight") or 1.0) for row in train_rows], dtype=float)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    model.fit(train_matrix, train_labels, sample_weight=train_weights)
    validation_scores = model.predict_proba(validation_matrix)[:, 1]
    metrics = train_ranker._evaluate_scores(validation_labels, validation_scores, score_kind="probability")
    return {
        "model_type": "logistic",
        "feature_names": list(feature_names),
        "weights": {
            name: float(value)
            for name, value in zip(feature_names, model.coef_[0], strict=False)
        },
        "intercept": float(model.intercept_[0]),
        "scales": scales,
        "validation": metrics,
        "training_rows": len(rows),
        "positive_rows": len(positive),
        "negative_rows": len(negative),
        "split_strategy": split["strategy"],
        "split_unit": split.get("unit", ""),
    }


def _heuristic_start_from_row(row: dict[str, float]) -> float:
    return _clamp01(
        row["clean_open_score"] * 0.34
        + row["hook_proximity_score"] * 0.24
        + row["duration_prior_score"] * 0.14
        + row["speaker_transition_score"] * 0.08
        + row["scene_transition_score"] * 0.06
        + row["pause_strength_score"] * 0.06
        + row["lexical_break_score"] * 0.08
        - row["mid_thought_risk"] * 0.26
        - row["context_dependency_risk"] * 0.18
    )


def _heuristic_end_from_row(row: dict[str, float]) -> float:
    return _clamp01(
        row["closure_score"] * 0.34
        + row["payoff_capture_score"] * 0.26
        + row["reaction_tail_score"] * 0.14
        + row["duration_prior_score"] * 0.12
        + row["scene_transition_score"] * 0.05
        + row["pause_strength_score"] * 0.05
        + row["lexical_break_score"] * 0.04
        - row["mid_thought_risk"] * 0.20
        - row["context_dependency_risk"] * 0.10
    )


def _duration_prior(delta_sec: float) -> float:
    delta_sec = max(float(delta_sec), 0.0)
    if 1.5 <= delta_sec <= 8.0:
        return 1.0
    if delta_sec < 1.5:
        return max(delta_sec / 1.5, 0.0)
    if delta_sec <= 14.0:
        return max(1.0 - ((delta_sec - 8.0) / 8.0), 0.0)
    return 0.0


def _duration(row: dict) -> float:
    features = row.get("features") or {}
    duration = float(features.get("duration", 0.0) or 0.0)
    if duration > 0:
        return duration
    return max(float(row.get("window_end") or 0.0) - float(row.get("window_start") or 0.0), 0.0)


def _count_values(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
