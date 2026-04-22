from __future__ import annotations

import json
import logging
import math
import pathlib

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - optional when only heuristic/linear artifacts are used
    joblib = None

from config import settings
from services import clip_strategy, profile_config, semantic_service
from services.timeline_types import CandidateFeatures


_DEFAULT_ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "ml" / "artifacts"
_MIN_TRAINING_ROWS = 50
_MIN_POSITIVE_ROWS = 15
_MIN_NEGATIVE_ROWS = 15
_MIN_VALIDATION_ACCURACY = 0.60
_MIN_VALIDATION_NDCG_AT_5 = 0.70
_MIN_VALIDATION_PRECISION_AT_5 = 0.60
_MIN_VALIDATION_PRIMARY_RANKING_METRIC = 0.60
_MAX_NORMALIZED_ABS = 4.0
_MAX_ML_ABS_SCORE = 12.0
_MIN_BLEND = 0.0
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
logger = logging.getLogger(__name__)


class Ranker:
    def __init__(
        self,
        artifact_dir: pathlib.Path | None = None,
        *,
        enable_trained_model: bool | None = None,
    ):
        self.artifact_dir = artifact_dir or _DEFAULT_ARTIFACT_DIR
        self.enable_trained_model = settings.enable_trained_ranker_v2 if enable_trained_model is None else bool(enable_trained_model)

    def score(self, features: list[CandidateFeatures], profile: str, strategy: dict | None = None) -> list[tuple[str, float]]:
        details = self.score_details(features, profile, strategy=strategy)
        return [(item["candidate_id"], item["score"]) for item in details]

    def score_with_rationale(
        self,
        features: list[CandidateFeatures],
        profile: str,
        strategy: dict | None = None,
    ) -> list[tuple[str, float, list[str]]]:
        details = self.score_details(features, profile, strategy=strategy)
        return [(item["candidate_id"], item["score"], list(item["rationale"])) for item in details]

    def score_details(
        self,
        features: list[CandidateFeatures],
        profile: str,
        strategy: dict | None = None,
    ) -> list[dict[str, object]]:
        heuristic = self._heuristic_scores(features, profile, strategy=strategy)
        artifact = self._load_artifact(profile)
        if artifact is None:
            return [
                {
                    "candidate_id": feature.candidate_id,
                    "score": score,
                    "heuristic_score": score,
                    "trained_score": None,
                    "blend": 0.0,
                    "trained_model_used": False,
                    "rationale": list(rationale),
                }
                for feature, (score, rationale) in zip(features, heuristic)
            ]

        # Expose the heuristic score as an input feature so trained models that
        # were fit with heuristic_score in their feature vector can consume it.
        for feature, (heur_score, _rationale) in zip(features, heuristic):
            feature.heuristic_score = float(heur_score)
        trained_scores = self._trained_scores(features, artifact)
        blend = self._safe_blend(artifact)
        merged: list[dict[str, object]] = []
        for feature, (heur_score, rationale), trained_score in zip(features, heuristic, trained_scores):
            score = round(self._clamp_score(heur_score * (1.0 - blend) + trained_score * blend), 3)
            enriched = list(rationale)
            enriched.append("ml_ranked")
            merged.append(
                {
                    "candidate_id": feature.candidate_id,
                    "score": score,
                    "heuristic_score": round(float(heur_score), 3),
                    "trained_score": round(float(trained_score), 3),
                    "blend": round(float(blend), 3),
                    "trained_model_used": True,
                    "rationale": enriched,
                }
            )
        return merged

    def _heuristic_scores(
        self,
        features: list[CandidateFeatures],
        profile: str,
        strategy: dict | None = None,
    ) -> list[tuple[float, list[str]]]:
        profile_weights = profile_config.get_profile(profile)
        hw = profile_config.get_effective_heuristic_weights(profile)
        normalized_strategy = clip_strategy.normalize_strategy(strategy)
        platform_bias = clip_strategy.platform_biases(normalized_strategy)
        scored: list[tuple[float, list[str]]] = []
        for feature in features:
            hook_score = feature.hook_strength * 10.0
            visual_reaction = max(feature.motion_peak, feature.face_reaction_peak) * 10.0
            visual_hook_score = feature.visual_hook_score * 10.0
            audio_pop_score = feature.audio_pop_score * 10.0
            event_surprise_score = feature.event_surprise_score * 10.0
            nonverbal_payoff = feature.nonverbal_payoff_score * 10.0
            dialogue_dynamism = min(feature.speaker_turns * 2.0, 10.0)
            chat_alignment = feature.chat_peak * 10.0
            chapter_coherence = feature.chapter_coherence_score * 10.0
            topic_shift = feature.topic_shift_strength * 10.0
            chapter_density = feature.chapter_local_density * 10.0
            trend_prior = feature.trend_prior_score * 10.0
            dynamic_trend = feature.dynamic_trend_score * 10.0
            outcome_prior = feature.outcome_prior_score * 10.0
            goal_relevance = feature.goal_relevance_score * 10.0
            platform_relevance = feature.platform_relevance_score * 10.0
            projected_hook = feature.projected_hook_score * 10.0
            projected_retention = feature.projected_retention_score * 10.0
            overlap_risk = min(feature.clip_overlap_count / 3.0, 1.0) * 10.0
            semantic_score = (
                feature.quoteability_score
                + feature.surprise_score
                + feature.contradiction_score
                + feature.standalone_score
                + feature.semantic_density
            ) * 2.0
            payoff_clarity = feature.payoff_clarity_score * 10.0
            profile_fit = feature.profile_fit_score * 10.0
            context_efficiency = feature.context_efficiency * 10.0
            slow_open_penalty = feature.slow_open_penalty * 10.0
            context_dependency_penalty = feature.context_dependency_score * 10.0
            weak_resolution_penalty = feature.weak_resolution_penalty * 10.0

            rank_score = (
                hw["hook"]               * hook_score * profile_weights["hook_weight"] * platform_bias.get("hook", 1.0)
                + hw["payoff"]           * feature.payoff_score
                + hw.get("payoff_clarity", 0.0) * payoff_clarity
                + hw["hype"]             * feature.max_hype
                + hw["arc"]              * feature.arc_score
                + hw["chat"]             * chat_alignment * profile_weights["chat_weight"]
                + hw["visual"]           * visual_reaction * profile_weights["visual_weight"] * platform_bias.get("visual", 1.0)
                + hw["dialogue"]         * dialogue_dynamism * profile_weights["dialogue_weight"]
                + hw["semantic"]         * semantic_score * profile_weights.get("semantic_weight", 1.0)
                + hw["context_efficiency"] * context_efficiency
                # profile_fit_score is already profile-weighted upstream in
                # feature_builder._profile_fit_score(); do not multiply it again.
                + hw["profile_fit"]      * profile_fit
                + hw.get("chapter_coherence", 0.0) * chapter_coherence
                + hw.get("topic_shift", 0.0) * topic_shift
                + hw.get("chapter_density", 0.0) * chapter_density
                + hw.get("trend_prior", 0.0) * trend_prior
                + visual_hook_score * 0.05
                + audio_pop_score * 0.05
                + event_surprise_score * 0.05
                + nonverbal_payoff * 0.05
                + dynamic_trend * 0.04
                + outcome_prior * 0.05
                + goal_relevance * 0.06
                + platform_relevance * 0.05
                + projected_hook * 0.05
                + projected_retention * 0.06
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
            if feature.dynamic_trend_score >= 0.45:
                rationale.append("trend_feedback_aligned")
            if feature.goal_relevance_score >= 0.45:
                rationale.append("goal_aligned")
            if feature.platform_relevance_score >= 0.5:
                rationale.append(f"platform_fit:{normalized_strategy['platform']}")
            if feature.projected_hook_score >= 0.55:
                rationale.append("projected_strong_open")
            if feature.projected_retention_score >= 0.55:
                rationale.append("projected_retention")
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
        inspection = self.inspect_artifact(profile)
        artifact = inspection.get("artifact")
        return artifact if isinstance(artifact, dict) and inspection.get("status") == "usable" else None

    def inspect_artifact(self, profile: str) -> dict[str, object]:
        if not self.enable_trained_model:
            return {
                "status": "disabled",
                "reason": "trained_model_disabled",
                "requested_profile": profile,
                "resolved_profile": "",
                "source": "",
                "artifact": None,
            }

        exact = self._inspect_artifact_file(profile, expected_profile=profile)
        if exact["status"] != "usable":
            registry_exact = self._inspect_registry_artifact(profile, expected_profile=profile)
            if registry_exact["status"] == "usable":
                return registry_exact
            if exact["status"] == "missing" and registry_exact["status"] != "missing":
                return registry_exact
        else:
            return exact

        from services.profile_config import get_content_type_for_streamer
        content_type = get_content_type_for_streamer(profile)
        if content_type and content_type != profile:
            fallback = self._inspect_artifact_file(content_type, expected_profile=content_type)
            if fallback["status"] == "usable":
                fallback["requested_profile"] = profile
                fallback["source"] = "content_type_fallback"
                return fallback
            if fallback["status"] == "missing":
                registry_fallback = self._inspect_registry_artifact(content_type, expected_profile=content_type)
                if registry_fallback["status"] != "missing":
                    registry_fallback["requested_profile"] = profile
                    registry_fallback["source"] = "content_type_registry_fallback"
                    return registry_fallback
            if exact["status"] == "missing" and fallback["status"] != "missing":
                fallback["requested_profile"] = profile
                if fallback["status"] == "rejected":
                    fallback["source"] = "content_type_fallback"
                return fallback
        return exact

    def _inspect_artifact_file(self, profile: str, *, expected_profile: str) -> dict[str, object]:
        path = self.artifact_dir / f"{profile}_ranker.json"
        if not path.exists():
            return {
                "status": "missing",
                "reason": "artifact_file_missing",
                "requested_profile": expected_profile,
                "resolved_profile": profile,
                "source": "exact",
                "artifact": None,
            }
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            return {
                "status": "rejected",
                "reason": "artifact_file_unreadable",
                "requested_profile": expected_profile,
                "resolved_profile": profile,
                "source": "exact",
                "artifact": None,
            }
        except json.JSONDecodeError:
            return {
                "status": "rejected",
                "reason": "artifact_invalid_json",
                "requested_profile": expected_profile,
                "resolved_profile": profile,
                "source": "exact",
                "artifact": None,
            }

        usable, reason = self._artifact_is_usable(artifact, profile)
        return {
            "status": "usable" if usable else "rejected",
            "reason": reason,
            "requested_profile": expected_profile,
            "resolved_profile": profile,
            "source": "exact",
            "artifact": artifact,
        }

    def _inspect_registry_artifact(self, profile: str, *, expected_profile: str) -> dict[str, object]:
        try:
            from ml import mlflow_tracking
            artifact = mlflow_tracking.load_registered_ranker_artifact(profile=profile, stage="Production")
        except Exception:
            artifact = None
        if artifact is None:
            return {
                "status": "missing",
                "reason": "registry_artifact_missing",
                "requested_profile": expected_profile,
                "resolved_profile": profile,
                "source": "mlflow_registry",
                "artifact": None,
            }

        usable, reason = self._artifact_is_usable(artifact, profile)
        return {
            "status": "usable" if usable else "rejected",
            "reason": reason,
            "requested_profile": expected_profile,
            "resolved_profile": profile,
            "source": "mlflow_registry",
            "artifact": artifact,
        }

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
        artifact_base_dir = pathlib.Path(str(artifact.get("_artifact_base_dir") or self.artifact_dir))
        model_path = str(artifact.get("model_path") or "").strip()
        if not model_path or joblib is None:
            logger.warning(
                "artifact_model_load_failure profile=%s model_type=%s model_path=%s reason=%s",
                artifact.get("profile", ""),
                model_type,
                model_path,
                "joblib_unavailable" if joblib is None else "model_path_missing",
            )
            return [0.0 for _ in features]
        try:
            model = joblib.load(artifact_base_dir / model_path)
        except Exception as exc:
            logger.warning(
                "artifact_model_load_failure profile=%s model_type=%s model_path=%s reason=%s",
                artifact.get("profile", ""),
                model_type,
                model_path,
                exc,
            )
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

    def _artifact_is_usable(self, artifact: dict, profile: str) -> tuple[bool, str]:
        artifact_base_dir = pathlib.Path(str(artifact.get("_artifact_base_dir") or self.artifact_dir))
        if artifact.get("profile") != profile:
            return False, "profile_mismatch"
        if int(artifact.get("version", 0)) < 1:
            return False, "artifact_version_too_old"
        if int(artifact.get("version", 0)) >= 4:
            if artifact.get("semantic_score_config") != semantic_service.semantic_score_config():
                return False, "semantic_score_config_mismatch"
        feature_names = [str(name) for name in artifact.get("feature_names") or []]
        offline_names = sorted({name for name in feature_names if name in OFFLINE_ONLY_FEATURE_NAMES})
        if offline_names:
            logger.warning(
                "artifact_rejected_offline_only_features profile=%s offline_only_features=%s",
                profile,
                ",".join(offline_names),
            )
            return False, "artifact_references_offline_only_features"
        model_type = str(artifact.get("model_type") or "linear").strip().lower()
        has_linear_params = bool(artifact.get("weights"))
        has_joblib_model = model_type in {"gradient_boosting", "lightgbm_ranker"} and bool(artifact.get("model_path"))
        if not feature_names or not artifact.get("scales") or not (has_linear_params or has_joblib_model):
            return False, "artifact_schema_incomplete"
        if model_type in {"gradient_boosting", "lightgbm_ranker"}:
            model_path = str(artifact.get("model_path") or "").strip()
            if joblib is None:
                logger.warning(
                    "artifact_model_load_failure profile=%s model_type=%s model_path=%s reason=%s",
                    profile,
                    model_type,
                    model_path,
                    "joblib_unavailable",
                )
                return False, "artifact_model_loader_unavailable"
            try:
                joblib.load(artifact_base_dir / model_path)
            except Exception as exc:
                logger.warning(
                    "artifact_model_load_failure profile=%s model_type=%s model_path=%s reason=%s",
                    profile,
                    model_type,
                    model_path,
                    exc,
                )
                return False, "artifact_model_unloadable"

        training_stats = artifact.get("training_stats", {})
        validation = artifact.get("validation", {})
        if int(training_stats.get("rows", 0)) < _MIN_TRAINING_ROWS:
            return False, "training_rows_below_minimum"
        if int(training_stats.get("positive_rows", 0)) < _MIN_POSITIVE_ROWS:
            return False, "positive_rows_below_minimum"
        if int(training_stats.get("negative_rows", 0)) < _MIN_NEGATIVE_ROWS:
            return False, "negative_rows_below_minimum"
        validation_ok, validation_reason = self._validation_passes(validation)
        if not validation_ok:
            return False, validation_reason
        return True, "usable"

    def _validation_passes(self, validation: dict) -> tuple[bool, str]:
        accuracy = float(validation.get("accuracy", 0.0))
        ndcg = float(validation.get("ndcg_at_5", 0.0))
        ranking_metric_value = float(validation.get("ranking_metric_value", ndcg))
        precision_at_5 = float(validation.get("precision_at_5", 0.0))
        legacy_precision = float(validation.get("precision", 0.0))
        ranking_groups = int(validation.get("ranking_groups", 0) or 0)
        ranking_metric_min = float(validation.get("ranking_metric_min", validation.get("ndcg_at_5_min", ranking_metric_value)))
        ranking_metric_p25 = float(validation.get("ranking_metric_p25", validation.get("ndcg_at_5_p25", ranking_metric_value)))

        if ranking_groups and ranking_groups < 2:
            return False, "validation_ranking_groups_below_minimum"
        if ranking_groups >= 2:
            if ranking_metric_min < 0.2:
                return False, "validation_group_ranking_metric_min_below_minimum"
            if ranking_metric_p25 < 0.45:
                return False, "validation_group_ranking_metric_p25_below_minimum"

        ranking_ok = (
            ranking_metric_value >= _MIN_VALIDATION_PRIMARY_RANKING_METRIC
            or ndcg >= _MIN_VALIDATION_NDCG_AT_5
            or precision_at_5 >= _MIN_VALIDATION_PRECISION_AT_5
            or legacy_precision >= 0.70
        )
        if accuracy < _MIN_VALIDATION_ACCURACY:
            return False, "validation_accuracy_below_minimum"
        if not ranking_ok:
            return False, "validation_ranking_metrics_below_minimum"
        return True, "usable"

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
    strategy: dict | None = None,
) -> dict[str, tuple[float, list[str]]]:
    scorer = Ranker()
    return {
        candidate_id: (score, rationale)
        for candidate_id, score, rationale in scorer.score_with_rationale(features, profile, strategy=strategy)
    }
