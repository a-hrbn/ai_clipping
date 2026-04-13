from __future__ import annotations

import json
import math
import pathlib

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - optional when only heuristic/linear artifacts are used
    joblib = None

from services import profile_config
from services.timeline_types import CandidateFeatures


_DEFAULT_ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "ml" / "artifacts"
_MIN_TRAINING_ROWS = 50
_MIN_POSITIVE_ROWS = 15
_MIN_NEGATIVE_ROWS = 15
_MIN_VALIDATION_ACCURACY = 0.60
_MIN_VALIDATION_NDCG_AT_5 = 0.70
_MIN_VALIDATION_PRECISION_AT_5 = 0.60
_MAX_NORMALIZED_ABS = 4.0
_MAX_ML_ABS_SCORE = 12.0
_MIN_BLEND = 0.15
_MAX_BLEND = 0.65
OFFLINE_ONLY_FEATURE_NAMES = {
    "vod_progress",
    "boundary_left_context_sec",
    "boundary_right_context_sec",
    "neighbor_positive_density",
    "clip_overlap_count",
    "has_llm_semantics",
    "has_chat_features",
    "has_vector_similarity",
    "feature_build_version",
}


class Ranker:
    def __init__(self, artifact_dir: pathlib.Path | None = None):
        self.artifact_dir = artifact_dir or _DEFAULT_ARTIFACT_DIR

    def score(self, features: list[CandidateFeatures], profile: str) -> list[tuple[str, float]]:
        scored = self.score_with_rationale(features, profile)
        return [(candidate_id, score) for candidate_id, score, _ in scored]

    def score_with_rationale(
        self,
        features: list[CandidateFeatures],
        profile: str,
    ) -> list[tuple[str, float, list[str]]]:
        heuristic = self._heuristic_scores(features, profile)
        artifact = self._load_artifact(profile)
        if artifact is None:
            return [
                (feature.candidate_id, score, rationale)
                for feature, (score, rationale) in zip(features, heuristic)
            ]

        trained_scores = self._trained_scores(features, artifact)
        blend = self._safe_blend(artifact)
        merged: list[tuple[str, float, list[str]]] = []
        for feature, (heur_score, rationale), trained_score in zip(features, heuristic, trained_scores):
            score = round(self._clamp_score(heur_score * (1.0 - blend) + trained_score * blend), 3)
            enriched = list(rationale)
            enriched.append("ml_ranked")
            merged.append((feature.candidate_id, score, enriched))
        return merged

    def _heuristic_scores(
        self,
        features: list[CandidateFeatures],
        profile: str,
    ) -> list[tuple[float, list[str]]]:
        profile_weights = profile_config.get_profile(profile)
        hw = profile_config.get_effective_heuristic_weights(profile)
        scored: list[tuple[float, list[str]]] = []
        for feature in features:
            hook_score = feature.hook_strength * 10.0
            visual_reaction = max(feature.motion_peak, feature.face_reaction_peak) * 10.0
            dialogue_dynamism = min(feature.speaker_turns * 2.0, 10.0)
            chat_alignment = feature.chat_peak * 10.0
            chapter_coherence = feature.chapter_coherence_score * 10.0
            topic_shift = feature.topic_shift_strength * 10.0
            chapter_density = feature.chapter_local_density * 10.0
            trend_prior = feature.trend_prior_score * 10.0
            overlap_risk = min(feature.clip_overlap_count / 3.0, 1.0) * 10.0
            semantic_score = (
                feature.quoteability_score
                + feature.surprise_score
                + feature.contradiction_score
                + feature.standalone_score
                + feature.semantic_density
            ) / 5.0 * 10.0
            payoff_clarity = feature.payoff_clarity_score * 10.0
            profile_fit = feature.profile_fit_score * 10.0
            context_efficiency = feature.context_efficiency * 10.0
            slow_open_penalty = feature.slow_open_penalty * 10.0
            context_dependency_penalty = feature.context_dependency_score * 10.0
            weak_resolution_penalty = feature.weak_resolution_penalty * 10.0

            rank_score = (
                hw["hook"]               * hook_score * profile_weights["hook_weight"]
                + hw["payoff"]           * feature.payoff_score
                + hw.get("payoff_clarity", 0.0) * payoff_clarity
                + hw["hype"]             * feature.max_hype
                + hw["arc"]              * feature.arc_score
                + hw["chat"]             * chat_alignment * profile_weights["chat_weight"]
                + hw["visual"]           * visual_reaction * profile_weights["visual_weight"]
                + hw["dialogue"]         * dialogue_dynamism * profile_weights["dialogue_weight"]
                + hw["semantic"]         * semantic_score * profile_weights.get("semantic_weight", 1.0)
                + hw["context_efficiency"] * context_efficiency
                + hw["profile_fit"]      * profile_fit
                + hw.get("chapter_coherence", 0.0) * chapter_coherence
                + hw.get("topic_shift", 0.0) * topic_shift
                + hw.get("chapter_density", 0.0) * chapter_density
                + hw.get("trend_prior", 0.0) * trend_prior
                + hw.get("overlap_penalty", 0.0) * overlap_risk
                + hw.get("slow_open_penalty", 0.0) * slow_open_penalty
                + hw.get("context_dependency_penalty", 0.0) * context_dependency_penalty
                + hw.get("weak_resolution_penalty", 0.0) * weak_resolution_penalty
                + hw["context_cost"]     * (feature.context_cost * 10.0)
            )

            rationale: list[str] = []
            if hook_score >= 7.0:
                rationale.append("strong_hook")
            if feature.payoff_score >= 4.0:
                rationale.append("clear_payoff")
            if feature.chat_peak >= 0.5:
                rationale.append("chat_confirmed")
            if visual_reaction >= 4.0:
                rationale.append("visual_reaction")
            if feature.question_open:
                rationale.append("question_open")
            if feature.quoteability_score >= 0.6:
                rationale.append("quoteable")
            if feature.surprise_score >= 0.5:
                rationale.append("surprising")
            if feature.standalone_score >= 0.6:
                rationale.append("standalone")
            if feature.payoff_clarity_score >= 0.65:
                rationale.append("clear_resolution")
            if feature.profile_fit_score >= 0.55:
                rationale.append(f"profile_fit:{profile}")
            if feature.chapter_coherence_score >= 0.6:
                rationale.append("chapter_coherent")
            if feature.topic_shift_strength >= 0.45:
                rationale.append("topic_shift_hook")
            if feature.chapter_local_density >= 0.5:
                rationale.append("chapter_dense")
            if feature.trend_prior_score >= 0.55:
                rationale.append("trend_aligned")
            if feature.clip_overlap_count >= 2.0:
                rationale.append("overlap_risk")
            if feature.slow_open_penalty >= 0.45:
                rationale.append("slow_open_risk")
            if feature.context_dependency_score >= 0.5:
                rationale.append("context_dependent")
            if feature.weak_resolution_penalty >= 0.5:
                rationale.append("weak_resolution_risk")

            scored.append((round(rank_score, 3), rationale))
        return scored

    def _load_artifact(self, profile: str) -> dict | None:
        # Level 1: exact artifact (streamer-specific or named profile)
        artifact = self._try_load_artifact_file(profile)
        if artifact is not None:
            return artifact
        # Level 2: content-type fallback via STREAMER_CONTENT_TYPE_MAP
        from services.profile_config import get_content_type_for_streamer
        content_type = get_content_type_for_streamer(profile)
        if content_type and content_type != profile:
            artifact = self._try_load_artifact_file(content_type)
            if artifact is not None:
                return artifact
        # Level 3: heuristic only
        return None

    def _try_load_artifact_file(self, profile: str) -> dict | None:
        path = self.artifact_dir / f"{profile}_ranker.json"
        if not path.exists():
            return None
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return artifact if self._artifact_is_usable(artifact, profile) else None

    def _trained_scores(self, features: list[CandidateFeatures], artifact: dict) -> list[float]:
        feature_names: list[str] = artifact.get("feature_names", [])
        scales: dict[str, dict[str, float]] = artifact.get("scales", {})
        model_type = str(artifact.get("model_type") or "linear").strip().lower()
        if model_type in {"gradient_boosting", "lightgbm_ranker"}:
            return self._joblib_model_scores(features, artifact, feature_names, scales, model_type)
        weights: dict[str, float] = artifact.get("weights", {})
        intercept = float(artifact.get("intercept", 0.0))
        scores: list[float] = []
        for feature in features:
            score = intercept
            for name in feature_names:
                if name in OFFLINE_ONLY_FEATURE_NAMES:
                    continue
                raw_value = getattr(feature, name, 0.0)
                value = float(raw_value)
                scale = scales.get(name, {"mean": 0.0, "std": 1.0})
                std = float(scale.get("std", 1.0)) or 1.0
                normalized = (value - float(scale.get("mean", 0.0))) / std
                normalized = max(-_MAX_NORMALIZED_ABS, min(_MAX_NORMALIZED_ABS, normalized))
                score += normalized * float(weights.get(name, 0.0))
            scores.append(self._normalize_trained_score(score, model_type))
        return scores

    def _joblib_model_scores(
        self,
        features: list[CandidateFeatures],
        artifact: dict,
        feature_names: list[str],
        scales: dict[str, dict[str, float]],
        model_type: str,
    ) -> list[float]:
        model_path = str(artifact.get("model_path") or "").strip()
        if not model_path or joblib is None:
            return [0.0 for _ in features]
        try:
            model = joblib.load(self.artifact_dir / model_path)
        except Exception:
            return [0.0 for _ in features]

        matrix: list[list[float]] = []
        for feature in features:
            row: list[float] = []
            for name in feature_names:
                if name in OFFLINE_ONLY_FEATURE_NAMES:
                    continue
                raw_value = getattr(feature, name, 0.0)
                value = float(raw_value)
                scale = scales.get(name, {"mean": 0.0, "std": 1.0})
                std = float(scale.get("std", 1.0)) or 1.0
                normalized = (value - float(scale.get("mean", 0.0))) / std
                row.append(max(-_MAX_NORMALIZED_ABS, min(_MAX_NORMALIZED_ABS, normalized)))
            matrix.append(row)
        if model_type == "gradient_boosting":
            probabilities = model.predict_proba(matrix)[:, 1]
            return [self._normalize_trained_score(float(probability), "gradient_boosting") for probability in probabilities]
        scores = model.predict(matrix)
        return [self._normalize_trained_score(float(score), "lightgbm_ranker") for score in scores]

    def _artifact_is_usable(self, artifact: dict, profile: str) -> bool:
        if artifact.get("profile") != profile:
            return False
        if int(artifact.get("version", 0)) < 1:
            return False
        model_type = str(artifact.get("model_type") or "linear").strip().lower()
        has_linear_params = bool(artifact.get("weights"))
        has_joblib_model = model_type in {"gradient_boosting", "lightgbm_ranker"} and bool(artifact.get("model_path"))
        if not artifact.get("feature_names") or not artifact.get("scales") or not (has_linear_params or has_joblib_model):
            return False

        training_stats = artifact.get("training_stats", {})
        validation = artifact.get("validation", {})
        if int(training_stats.get("rows", 0)) < _MIN_TRAINING_ROWS:
            return False
        if int(training_stats.get("positive_rows", 0)) < _MIN_POSITIVE_ROWS:
            return False
        if int(training_stats.get("negative_rows", 0)) < _MIN_NEGATIVE_ROWS:
            return False
        if not self._validation_passes(validation):
            return False
        return True

    def _validation_passes(self, validation: dict) -> bool:
        accuracy = float(validation.get("accuracy", 0.0))
        ndcg = float(validation.get("ndcg_at_5", 0.0))
        precision_at_5 = float(validation.get("precision_at_5", 0.0))
        legacy_precision = float(validation.get("precision", 0.0))

        ranking_ok = (
            ndcg >= _MIN_VALIDATION_NDCG_AT_5
            or precision_at_5 >= _MIN_VALIDATION_PRECISION_AT_5
            or legacy_precision >= 0.70
        )
        return accuracy >= _MIN_VALIDATION_ACCURACY and ranking_ok

    def _safe_blend(self, artifact: dict) -> float:
        requested = float(artifact.get("blend", _MAX_BLEND))
        return max(_MIN_BLEND, min(_MAX_BLEND, requested))

    def _normalize_trained_score(self, raw_score: float, model_type: str = "linear") -> float:
        if model_type == "gradient_boosting":
            logistic = max(0.0, min(1.0, raw_score))
        elif model_type == "lightgbm_ranker":
            logistic = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, raw_score))))
        elif model_type == "logistic":
            logistic = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, raw_score))))
        else:
            logistic = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, raw_score))))
        centered = (logistic - 0.5) * 2.0
        return centered * _MAX_ML_ABS_SCORE

    def _clamp_score(self, score: float) -> float:
        return max(-20.0, min(20.0, score))


def score_candidates(
    features: list[CandidateFeatures],
    profile: str = "gaming",
) -> dict[str, tuple[float, list[str]]]:
    scorer = Ranker()
    return {
        candidate_id: (score, rationale)
        for candidate_id, score, rationale in scorer.score_with_rationale(features, profile)
    }
