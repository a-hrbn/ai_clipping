from __future__ import annotations

import json
import math
import pathlib

from config import settings
from services.timeline_types import BoundaryFeatures, BoundaryPair, BoundaryPoint, Candidate
from services.transcriber import RichSegment

_DEFAULT_ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "ml" / "artifacts"
_ARTIFACT_CACHE: dict[str, dict | None] = {}
_MIN_DURATION_SEC = 8.0
_MAX_DURATION_SEC = 90.0
_TOP_BOUNDARIES_PER_ROLE = 6
_MIN_CONTEXT_BEFORE_ANCHOR = 1.2
_MIN_PAYOFF_AFTER_ANCHOR = 2.0
_MIN_BOUNDARY_MODEL_ROWS = 8
_MIN_BOUNDARY_MODEL_ACCURACY = 0.55
_MIN_BOUNDARY_MODEL_F1 = 0.5

START_MODEL_FEATURE_NAMES = (
    "clean_open_score",
    "mid_thought_risk",
    "context_dependency_risk",
    "hook_proximity_score",
    "duration_prior_score",
    "speaker_transition_score",
    "scene_transition_score",
    "pause_strength_score",
    "lexical_break_score",
)
END_MODEL_FEATURE_NAMES = (
    "closure_score",
    "mid_thought_risk",
    "context_dependency_risk",
    "payoff_capture_score",
    "reaction_tail_score",
    "duration_prior_score",
    "scene_transition_score",
    "pause_strength_score",
    "lexical_break_score",
)
PAIR_MODEL_FEATURE_NAMES = (
    "start_score",
    "end_score",
    "hook_score",
    "payoff_score",
    "narrative_score",
    "duration",
    "setup_before_anchor",
    "payoff_after_anchor",
    "duration_fit",
    "penalty_count",
    "start_sentence_start",
    "start_scene_cut",
    "start_speaker_turn",
    "start_pause_break",
    "end_sentence_end",
    "end_pause_break",
)


def score_start_boundaries(
    features: list[BoundaryFeatures],
    profile: str,
) -> dict[str, float]:
    heuristic = _heuristic_start_scores(features)
    artifact = _load_boundary_artifact(profile)
    if not False or artifact is None:
        return heuristic

    model = artifact.get("start_model") or {}
    ml_scores = _boundary_model_scores(features, model, START_MODEL_FEATURE_NAMES, role="start")
    return _blend_score_maps(heuristic, ml_scores, float((artifact.get("blend") or {}).get("start", 0.0)))


def score_end_boundaries(
    features: list[BoundaryFeatures],
    profile: str,
) -> dict[str, float]:
    heuristic = _heuristic_end_scores(features)
    artifact = _load_boundary_artifact(profile)
    if not False or artifact is None:
        return heuristic

    model = artifact.get("end_model") or {}
    ml_scores = _boundary_model_scores(features, model, END_MODEL_FEATURE_NAMES, role="end")
    return _blend_score_maps(heuristic, ml_scores, float((artifact.get("blend") or {}).get("end", 0.0)))


def score_boundary_pairs(
    candidate: Candidate,
    segments: list[RichSegment],
    boundaries: list[BoundaryPoint],
    start_scores: dict[str, float],
    end_scores: dict[str, float],
    profile: str,
) -> list[BoundaryPair]:
    boundary_map = {boundary.id: boundary for boundary in boundaries}
    top_start_ids = [
        boundary_id
        for boundary_id, _score in sorted(start_scores.items(), key=lambda item: item[1], reverse=True)[:_TOP_BOUNDARIES_PER_ROLE]
    ]
    top_end_ids = [
        boundary_id
        for boundary_id, _score in sorted(end_scores.items(), key=lambda item: item[1], reverse=True)[:_TOP_BOUNDARIES_PER_ROLE]
    ]

    pairs: list[BoundaryPair] = []
    pair_index = 0
    for start_id in top_start_ids:
        start = boundary_map.get(start_id)
        if start is None:
            continue
        if not (candidate.start <= start.time < candidate.anchor_time):
            continue
        for end_id in top_end_ids:
            end = boundary_map.get(end_id)
            if end is None:
                continue
            if not (candidate.anchor_time < end.time <= candidate.end):
                continue

            duration = end.time - start.time
            if duration < _MIN_DURATION_SEC or duration > _MAX_DURATION_SEC:
                continue

            setup_before_anchor = candidate.anchor_time - start.time
            payoff_after_anchor = end.time - candidate.anchor_time
            window = [seg for seg in segments if seg.end > start.time and seg.start < end.time]
            early_window = [seg for seg in window if seg.end > start.time and seg.start < start.time + 3.0]
            tail_window = [seg for seg in window if seg.end > max(candidate.anchor_time, end.time - 3.0) and seg.start < end.time]
            post_anchor_window = [seg for seg in window if seg.end > candidate.anchor_time and seg.start < end.time]

            hook_score = _hook_score(early_window)
            payoff_score = _payoff_score(post_anchor_window, tail_window)
            narrative_score = 0.0
            if setup_before_anchor >= _MIN_CONTEXT_BEFORE_ANCHOR:
                narrative_score += 0.35
            if payoff_after_anchor >= _MIN_PAYOFF_AFTER_ANCHOR:
                narrative_score += 0.35
            if start.is_sentence_start or start.is_speaker_turn or start.is_scene_cut:
                narrative_score += 0.15
            if end.is_sentence_end or end.is_pause_break:
                narrative_score += 0.15
            narrative_score = min(narrative_score, 1.0)

            penalties: list[str] = []
            penalty_score = 0.0
            if setup_before_anchor < _MIN_CONTEXT_BEFORE_ANCHOR:
                penalties.append("thin_setup")
                penalty_score += 0.1
            if payoff_after_anchor < _MIN_PAYOFF_AFTER_ANCHOR:
                penalties.append("truncated_payoff")
                penalty_score += 0.12
            if start.text_right and start.text_right[:1].islower():
                penalties.append("lowercase_open")
                penalty_score += 0.05
            if not end.is_sentence_end and payoff_score < 0.35:
                penalties.append("weak_closure")
                penalty_score += 0.08
            if duration > 40.0:
                penalties.append("overlong_context")
                penalty_score += 0.05

            raw_score = (
                start_scores.get(start_id, 0.0) * 0.30
                + end_scores.get(end_id, 0.0) * 0.30
                + hook_score * 0.15
                + payoff_score * 0.15
                + narrative_score * 0.10
                - penalty_score
            )
            pair_score = max(raw_score, 0.0)
            reason = _reason_for_pair(start, end, penalties)
            pairs.append(
                BoundaryPair(
                    id=f"pair_{pair_index}",
                    candidate_id=candidate.id,
                    start_boundary_id=start_id,
                    end_boundary_id=end_id,
                    start_time=round(start.time, 2),
                    end_time=round(end.time, 2),
                    duration=round(duration, 2),
                    pair_score=round(pair_score, 4),
                    start_score=round(start_scores.get(start_id, 0.0), 4),
                    end_score=round(end_scores.get(end_id, 0.0), 4),
                    hook_score=round(hook_score, 4),
                    payoff_score=round(payoff_score, 4),
                    narrative_score=round(narrative_score, 4),
                    penalties=penalties,
                    reason=reason,
                )
            )
            pair_index += 1

    artifact = _load_boundary_artifact(profile)
    if False and artifact is not None and pairs:
        model = artifact.get("pair_model") or {}
        ml_scores = _pair_model_scores(pairs, boundary_map, candidate, model)
        blend = float((artifact.get("blend") or {}).get("pair", 0.0))
        for pair in pairs:
            pair.pair_score = round(_blend_value(pair.pair_score, ml_scores.get(pair.id, pair.pair_score), blend), 4)

    return sorted(
        pairs,
        key=lambda pair: (
            pair.pair_score,
            pair.payoff_score,
            pair.start_score,
            pair.end_score,
            -pair.duration,
        ),
        reverse=True,
    )


def _heuristic_start_scores(features: list[BoundaryFeatures]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in features:
        if item.role != "start":
            continue
        score = (
            item.clean_open_score * 0.34
            + item.hook_proximity_score * 0.24
            + item.duration_prior_score * 0.14
            + item.speaker_transition_score * 0.08
            + item.scene_transition_score * 0.06
            + item.pause_strength_score * 0.06
            + item.lexical_break_score * 0.08
            - item.mid_thought_risk * 0.26
            - item.context_dependency_risk * 0.18
        )
        scores[item.boundary_id] = round(max(score, 0.0), 4)
    return scores


def _heuristic_end_scores(features: list[BoundaryFeatures]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in features:
        if item.role != "end":
            continue
        score = (
            item.closure_score * 0.34
            + item.payoff_capture_score * 0.26
            + item.reaction_tail_score * 0.14
            + item.duration_prior_score * 0.12
            + item.scene_transition_score * 0.05
            + item.pause_strength_score * 0.05
            + item.lexical_break_score * 0.04
            - item.mid_thought_risk * 0.2
            - item.context_dependency_risk * 0.1
        )
        scores[item.boundary_id] = round(max(score, 0.0), 4)
    return scores


def _boundary_model_scores(
    features: list[BoundaryFeatures],
    model: dict,
    feature_names: tuple[str, ...],
    *,
    role: str,
) -> dict[str, float]:
    if not model or model.get("model_type") != "logistic":
        return {}

    scores: dict[str, float] = {}
    weights = model.get("weights") or {}
    scales = model.get("scales") or {}
    intercept = float(model.get("intercept", 0.0))
    for item in features:
        if item.role != role:
            continue
        row = _boundary_feature_row(item)
        margin = intercept
        for name in feature_names:
            margin += _normalized_feature_value(row, name, scales) * float(weights.get(name, 0.0))
        scores[item.boundary_id] = round(_sigmoid(margin), 4)
    return scores


def _pair_model_scores(
    pairs: list[BoundaryPair],
    boundary_map: dict[str, BoundaryPoint],
    candidate: Candidate,
    model: dict,
) -> dict[str, float]:
    if not model or model.get("model_type") != "logistic":
        return {}

    scores: dict[str, float] = {}
    weights = model.get("weights") or {}
    scales = model.get("scales") or {}
    intercept = float(model.get("intercept", 0.0))
    for pair in pairs:
        start = boundary_map.get(pair.start_boundary_id)
        end = boundary_map.get(pair.end_boundary_id)
        if start is None or end is None:
            continue
        row = pair_feature_row(pair, candidate, start, end)
        margin = intercept
        for name in PAIR_MODEL_FEATURE_NAMES:
            margin += _normalized_feature_value(row, name, scales) * float(weights.get(name, 0.0))
        scores[pair.id] = round(_sigmoid(margin), 4)
    return scores


def _blend_score_maps(
    heuristic: dict[str, float],
    ml_scores: dict[str, float],
    blend: float,
) -> dict[str, float]:
    if not ml_scores or blend <= 0.0:
        return heuristic
    merged = dict(heuristic)
    for key, base in heuristic.items():
        merged[key] = round(_blend_value(base, ml_scores.get(key, base), blend), 4)
    return merged


def _blend_value(base: float, learned: float, blend: float) -> float:
    blend = max(0.0, min(blend, 1.0))
    return base * (1.0 - blend) + learned * blend


def _load_boundary_artifact(profile: str) -> dict | None:
    if not False:
        return None
    key = str(profile or "general")
    if key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[key]

    candidates = [key]
    if key != "general":
        candidates.append("general")
    for name in candidates:
        path = _DEFAULT_ARTIFACT_DIR / f"{name}_boundary_ranker.json"
        if not path.exists():
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _boundary_artifact_is_usable(artifact, requested_profile=key, artifact_name=name):
            continue
        _ARTIFACT_CACHE[key] = artifact
        return artifact

    _ARTIFACT_CACHE[key] = None
    return None


def _boundary_artifact_is_usable(
    artifact: dict,
    *,
    requested_profile: str,
    artifact_name: str,
) -> bool:
    if int(artifact.get("version", 0)) < 1:
        return False
    if str(artifact.get("model_type") or "").strip().lower() != "boundary_logistic":
        return False

    artifact_profile = str(artifact.get("profile") or "").strip()
    if artifact_profile != artifact_name:
        return False
    if artifact_name not in {requested_profile, "general"}:
        return False

    return (
        _boundary_model_is_usable(artifact.get("start_model") or {}, START_MODEL_FEATURE_NAMES)
        and _boundary_model_is_usable(artifact.get("end_model") or {}, END_MODEL_FEATURE_NAMES)
        and _boundary_model_is_usable(artifact.get("pair_model") or {}, PAIR_MODEL_FEATURE_NAMES)
    )


def _boundary_model_is_usable(model: dict, expected_features: tuple[str, ...]) -> bool:
    if str(model.get("model_type") or "").strip().lower() != "logistic":
        return False

    feature_names = tuple(model.get("feature_names") or ())
    if feature_names != expected_features:
        return False
    if not model.get("weights") or not model.get("scales"):
        return False
    if "intercept" not in model:
        return False

    if int(model.get("training_rows", 0)) < _MIN_BOUNDARY_MODEL_ROWS:
        return False
    validation = model.get("validation") or {}
    accuracy = float(validation.get("accuracy", 0.0))
    f1 = float(validation.get("f1", 0.0))
    return accuracy >= _MIN_BOUNDARY_MODEL_ACCURACY and f1 >= _MIN_BOUNDARY_MODEL_F1


def _boundary_feature_row(item: BoundaryFeatures) -> dict[str, float]:
    return {
        "clean_open_score": float(item.clean_open_score),
        "closure_score": float(item.closure_score),
        "mid_thought_risk": float(item.mid_thought_risk),
        "context_dependency_risk": float(item.context_dependency_risk),
        "hook_proximity_score": float(item.hook_proximity_score),
        "payoff_capture_score": float(item.payoff_capture_score),
        "speaker_transition_score": float(item.speaker_transition_score),
        "scene_transition_score": float(item.scene_transition_score),
        "pause_strength_score": float(item.pause_strength_score),
        "lexical_break_score": float(item.lexical_break_score),
        "reaction_tail_score": float(item.reaction_tail_score),
        "duration_prior_score": float(item.duration_prior_score),
    }


def pair_feature_row(
    pair: BoundaryPair,
    candidate: Candidate,
    start: BoundaryPoint,
    end: BoundaryPoint,
) -> dict[str, float]:
    setup_before_anchor = max(candidate.anchor_time - pair.start_time, 0.0)
    payoff_after_anchor = max(pair.end_time - candidate.anchor_time, 0.0)
    duration_fit = 1.0 - min(abs(pair.duration - 22.0) / 22.0, 1.0)
    return {
        "start_score": float(pair.start_score),
        "end_score": float(pair.end_score),
        "hook_score": float(pair.hook_score),
        "payoff_score": float(pair.payoff_score),
        "narrative_score": float(pair.narrative_score),
        "duration": float(pair.duration),
        "setup_before_anchor": float(setup_before_anchor),
        "payoff_after_anchor": float(payoff_after_anchor),
        "duration_fit": float(duration_fit),
        "penalty_count": float(len(pair.penalties)),
        "start_sentence_start": 1.0 if start.is_sentence_start else 0.0,
        "start_scene_cut": 1.0 if start.is_scene_cut else 0.0,
        "start_speaker_turn": 1.0 if start.is_speaker_turn else 0.0,
        "start_pause_break": 1.0 if start.is_pause_break else 0.0,
        "end_sentence_end": 1.0 if end.is_sentence_end else 0.0,
        "end_pause_break": 1.0 if end.is_pause_break else 0.0,
    }


def _normalized_feature_value(features: dict[str, float], name: str, scales: dict[str, dict[str, float]]) -> float:
    raw_value = float(features.get(name, 0.0) or 0.0)
    scale = scales.get(name, {"mean": 0.0, "std": 1.0})
    std = float(scale.get("std", 1.0)) or 1.0
    return max(-4.0, min((raw_value - float(scale.get("mean", 0.0))) / std, 4.0))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _hook_score(segments: list[RichSegment]) -> float:
    if not segments:
        return 0.0
    score = min(max((seg.hype_score for seg in segments), default=0.0) / 6.0, 0.45)
    if any("question" in seg.emotions or "exclamation" in seg.emotions for seg in segments):
        score += 0.2
    if any(seg.chat_spike_score >= 0.35 for seg in segments):
        score += 0.15
    if any(seg.max_face_reaction >= 0.35 for seg in segments):
        score += 0.15
    if any(seg.text.strip().endswith(("?", "!")) for seg in segments):
        score += 0.05
    return min(score, 1.0)


def _payoff_score(post_anchor_window: list[RichSegment], tail_window: list[RichSegment]) -> float:
    if not post_anchor_window:
        return 0.0
    score = min(max((seg.hype_score for seg in post_anchor_window), default=0.0) / 7.0, 0.3)
    if any({"laughter", "audio_laughter", "face_reaction", "exclamation"} & set(seg.emotions) for seg in post_anchor_window):
        score += 0.35
    if any(seg.chat_spike_score >= 0.35 for seg in post_anchor_window):
        score += 0.15
    if any(seg.max_face_reaction >= 0.3 for seg in post_anchor_window):
        score += 0.1
    if any(seg.text.strip().endswith((".", "!", "?")) for seg in tail_window or post_anchor_window):
        score += 0.1
    return min(score, 1.0)


def _reason_for_pair(start: BoundaryPoint, end: BoundaryPoint, penalties: list[str]) -> str:
    start_reason = "clean sentence start"
    if start.is_scene_cut:
        start_reason = "scene-cut opening"
    elif start.is_speaker_turn:
        start_reason = "speaker-turn opening"
    elif start.is_pause_break:
        start_reason = "pause-reset opening"

    end_reason = "complete payoff"
    if end.is_sentence_end:
        end_reason = "sentence closure"
    elif end.is_pause_break:
        end_reason = "natural pause ending"

    if penalties:
        return f"{start_reason}; {end_reason}; penalties: {', '.join(penalties)}"
    return f"{start_reason}; {end_reason}"
