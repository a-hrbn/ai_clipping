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
from services import clip_strategy, content_buckets, context_quality, profile_config, semantic_service
from services.timeline_types import CandidateFeatures


_DEFAULT_ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "ml" / "artifacts"
_MIN_TRAINING_ROWS = 50
_MIN_POSITIVE_ROWS = 15
_MIN_NEGATIVE_ROWS = 15
_MIN_VALIDATION_ACCURACY_FLOOR = 0.55
_MIN_VALIDATION_NDCG_AT_5 = 0.70
_MIN_VALIDATION_PRECISION_AT_5 = 0.60
_MIN_VALIDATION_PRIMARY_RANKING_METRIC = 0.60
_MIN_FILTER_PRECISION = 0.50
_MIN_FILTER_RECALL = 0.65
_MIN_FILTER_F1 = 0.55
_MAX_FILTER_SELECTED_RATE = 0.75
_MAX_NORMALIZED_ABS = 4.0
_MAX_ML_ABS_SCORE = 12.0
_MIN_BLEND = 0.0
_MAX_BLEND = 0.65
_STANDALONE_HARD_FILTER_MAX_SCORE = 3.0
_STANDALONE_HARD_FILTER_MIN_STANDALONE = 0.25
_STANDALONE_HARD_FILTER_CONTEXT_DEPENDENCY = 0.75
OFFLINE_ONLY_FEATURE_NAMES = {
    "vod_progress",
    "boundary_left_context_sec",
    "boundary_right_context_sec",
    "neighbor_positive_density",
    "clip_overlap_count",
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
                self._apply_serving_reranker_stages(
                    feature,
                    {
                    "candidate_id": feature.candidate_id,
                    "score": score,
                    "heuristic_score": score,
                    "trained_score": None,
                    "blend": 0.0,
                    "trained_model_used": False,
                    "rationale": list(rationale),
                    },
                )
                for feature, (score, rationale) in zip(features, heuristic)
            ]

        # Expose the heuristic score as an input feature so trained models that
        # were fit with heuristic_score in their feature vector can consume it.
        for feature, (heur_score, _rationale) in zip(features, heuristic):
            feature.heuristic_score = float(heur_score)
        trained_scores = self._trained_scores(features, artifact)
        blend = self._safe_blend(artifact)
        score_mode = str(artifact.get("score_mode") or "absolute").strip().lower()
        merged: list[dict[str, object]] = []
        for feature, (heur_score, rationale), trained_score in zip(features, heuristic, trained_scores):
            bucket = content_buckets.bucket_from_feature(feature)
            bucket_model = self._bucket_model(artifact, bucket)
            serving_threshold = self._safe_serving_threshold(artifact, bucket=bucket)
            if bucket_model == "fallback_heuristic":
                raw_score = float(heur_score)
                effective_blend = 0.0
                trained_used = False
            elif score_mode == "pairwise_median":
                raw_score = float(trained_score)
                effective_blend = 0.0
                trained_used = True
            else:
                raw_score = float(heur_score) * (1.0 - blend) + float(trained_score) * blend
                effective_blend = blend
                trained_used = True
            score = round(self._clamp_score(raw_score), 3)
            enriched = list(rationale)
            enriched.append("heuristic_bucket_fallback" if bucket_model == "fallback_heuristic" else "ml_ranked")
            reranked_item = self._apply_serving_reranker_stages(
                feature,
                {
                    "candidate_id": feature.candidate_id,
                    "score": score,
                    "content_bucket": bucket,
                    "bucket_model": bucket_model,
                    "heuristic_score": round(float(heur_score), 3),
                    "trained_score": round(float(trained_score), 3),
                    "blend": round(float(effective_blend), 3),
                    "serving_threshold": round(float(serving_threshold), 6),
                    "trained_model_used": trained_used,
                    "rationale": enriched,
                },
            )
            reranked_item["passes_serving_threshold"] = float(reranked_item["score"]) >= serving_threshold
            merged.append(reranked_item)
        return merged

    def _apply_hook_reranker_stage(
        self,
        feature: CandidateFeatures,
        item: dict[str, object],
    ) -> dict[str, object]:
        if not feature.has_hook_rewrite:
            return item
        hook_score = max(float(feature.hook_rewrite_score), float(feature.hook_curiosity_gap_score))
        score = float(item["score"])
        rationale = list(item.get("rationale") or [])
        if hook_score >= 0.75:
            score += 0.65
            rationale.append("hook_reranker_boost")
        elif hook_score <= 0.25 and feature.hook_delay_sec > 2.0:
            score -= 0.75
            rationale.append("hook_reranker_penalty")
        item["score"] = round(self._clamp_score(score), 3)
        item["hook_reranker_score"] = round(hook_score, 3)
        item["rationale"] = rationale
        return item

    def _apply_boundary_variant_set_stage(
        self,
        feature: CandidateFeatures,
        item: dict[str, object],
    ) -> dict[str, object]:
        variant_score = max(0.0, min(float(feature.boundary_variant_score), 1.0))
        if variant_score <= 0.0:
            return item
        margin_score = max(0.0, min(float(feature.boundary_variant_margin_score), 1.0))
        count = max(0.0, float(feature.boundary_variant_count))
        set_depth_bonus = min(count / 24.0, 1.0) * 0.15
        score_delta = variant_score * 0.55 + margin_score * 0.35 + set_depth_bonus
        score = float(item["score"]) + score_delta
        rationale = list(item.get("rationale") or [])
        rationale.append("boundary_variant_set_max")
        item["score"] = round(self._clamp_score(score), 3)
        item["boundary_variant_set_score"] = round(variant_score, 3)
        item["boundary_variant_set_margin"] = round(margin_score, 3)
        item["rationale"] = rationale
        return item

    def _apply_serving_reranker_stages(
        self,
        feature: CandidateFeatures,
        item: dict[str, object],
    ) -> dict[str, object]:
        item = self._apply_context_primary_stage(feature, item)
        item = self._apply_boundary_variant_set_stage(feature, item)
        return self._apply_hook_reranker_stage(feature, item)

    def _apply_context_primary_stage(
        self,
        feature: CandidateFeatures,
        item: dict[str, object],
    ) -> dict[str, object]:
        context_primary = context_quality.primary_score(
            completeness=feature.context_completeness_score,
            strength=feature.context_strength_score,
        )
        score = float(item["score"])
        rationale = list(item.get("rationale") or [])
        if context_primary >= 0.65:
            score += 1.35
            rationale.append("context_primary_boost")
        elif context_primary >= 0.52:
            score += 0.85
            rationale.append("context_primary_signal")
        elif context_primary < 0.22 and feature.max_hype >= 3.5:
            score -= 0.65
            rationale.append("context_missing_penalty")
        item["score"] = round(self._clamp_score(score), 3)
        item["context_primary_score"] = round(context_primary, 3)
        item["rationale"] = rationale
        return item

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
            streamer_shout = feature.streamer_shout_score * 10.0
            multimodal_proxy = feature.multimodal_proxy_score * 10.0
            context_pattern = feature.context_pattern_confidence * 10.0
            jynxzi_type_signal = max(
                feature.rage_score,
                feature.clutch_score,
                feature.funny_reaction_score,
                feature.trash_talk_payoff_score,
                feature.jynxzi_chat_shock_score,
                feature.ranked_tension_score,
                feature.failed_setup_score,
                feature.streamer_persona_beat_score,
            ) * 10.0
            event_surprise_score = feature.event_surprise_score * 10.0
            nonverbal_payoff = feature.nonverbal_payoff_score * 10.0
            game_event = feature.game_event_score * 10.0
            hook_rerank = max(feature.hook_rewrite_score, feature.hook_curiosity_gap_score) * 10.0
            vision_language = feature.vl_prompt_max_score * 10.0
            dialogue_dynamism = min(feature.speaker_turns * 2.0, 10.0)
            chat_alignment = feature.chat_peak * 10.0
            chat_velocity = max(
                feature.chat_velocity_score,
                feature.chat_rate_derivative_score,
                feature.chat_ema_crossing_score,
            ) * 10.0
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
            cross_platform_recall = (
                feature.cross_platform_positive_overlap_ratio
                * max(feature.cross_platform_positive_similarity, 0.55)
                * 10.0
                if feature.cross_platform_recall_target
                else 0.0
            )
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
            context_dependency_penalty = feature.context_dependency_score * 10.0
            weak_resolution_penalty = feature.weak_resolution_penalty * 10.0
            context_primary = context_quality.primary_score(
                completeness=feature.context_completeness_score,
                strength=feature.context_strength_score,
            ) * 10.0
            context_completeness = feature.context_completeness_score * 10.0
            context_strength = feature.context_strength_score * 10.0
            setup_present = feature.setup_present_score * 10.0
            payoff_present = feature.payoff_present_score * 10.0

            rank_score = (
                context_primary * 0.46
                + context_completeness * 0.28
                + context_strength * 0.18
                + setup_present * 0.12
                + payoff_present * 0.18
                + hw["hook"]               * hook_score * profile_weights["hook_weight"] * platform_bias.get("hook", 1.0)
                + hw["payoff"]           * feature.payoff_score
                + hw.get("payoff_clarity", 0.0) * payoff_clarity
                + hw["hype"]             * feature.max_hype
                + hw["arc"]              * feature.arc_score
                + hw["chat"]             * chat_alignment * profile_weights["chat_weight"]
                + chat_velocity * 0.05 * profile_weights["chat_weight"]
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
                + streamer_shout * 0.05
                + multimodal_proxy * 0.05
                + context_pattern * 0.18
                + jynxzi_type_signal * 0.04
                + event_surprise_score * 0.05
                + nonverbal_payoff * 0.05
                + game_event * 0.04
                + hook_rerank * 0.04
                + vision_language * 0.04
                + dynamic_trend * 0.04
                + outcome_prior * 0.05
                + cross_platform_recall * 0.36
                + goal_relevance * 0.06
                + platform_relevance * 0.05
                + projected_hook * 0.05
                + projected_retention * 0.06
                + hw.get("overlap_penalty", 0.0) * overlap_risk
                + hw.get("context_dependency_penalty", 0.0) * context_dependency_penalty
                + hw.get("weak_resolution_penalty", 0.0) * weak_resolution_penalty
                + hw["context_cost"]     * (feature.context_cost * 10.0)
            )

            standalone_hard_filtered = (
                feature.standalone_score < _STANDALONE_HARD_FILTER_MIN_STANDALONE
                and feature.context_dependency_score >= _STANDALONE_HARD_FILTER_CONTEXT_DEPENDENCY
            )
            if standalone_hard_filtered:
                rank_score = min(rank_score, _STANDALONE_HARD_FILTER_MAX_SCORE)

            rationale: list[str] = []
            if context_primary >= 6.5:
                rationale.append("context_primary")
            if feature.context_completeness_score >= 0.55:
                rationale.append("context_complete")
            if feature.context_strength_score >= 0.55:
                rationale.append("context_strong")
            if hook_score >= 7.0:
                rationale.append("strong_hook")
            if feature.payoff_score >= 4.0:
                rationale.append("clear_payoff")
            if feature.cross_platform_recall_target:
                rationale.append("cross_platform_recall_target")
            if feature.chat_peak >= 0.5:
                rationale.append("chat_confirmed")
            if max(feature.chat_rate_derivative_score, feature.chat_ema_crossing_score) >= 0.5:
                rationale.append("chat_velocity_spike")
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
            if feature.streamer_shout_score >= 0.55:
                rationale.append("streamer_shout")
            if feature.multimodal_proxy_score >= 0.5:
                rationale.append("multimodal_proxy_signal")
            if feature.context_pattern_confidence >= 0.4:
                label = feature.context_pattern_label or feature.context_pattern_id
                rationale.append(f"context_pattern:{label}")
            if feature.clip_type and feature.clip_type != "general" and jynxzi_type_signal >= 4.5:
                rationale.append(f"clip_type:{feature.clip_type}")
            if feature.game_event_score >= 0.5:
                rationale.append("game_event")
            if feature.vl_prompt_max_score >= 0.5:
                rationale.append("vision_language_match")
            if feature.has_hook_rewrite and max(feature.hook_rewrite_score, feature.hook_curiosity_gap_score) >= 0.55:
                rationale.append("hook_reranked")
            if feature.clip_overlap_count >= 2.0:
                rationale.append("overlap_risk")
            if feature.context_dependency_score >= 0.5:
                rationale.append("context_dependent")
            if feature.weak_resolution_penalty >= 0.5:
                rationale.append("weak_resolution_risk")
            if standalone_hard_filtered:
                rationale.append("standalone_hard_filter")

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
        if model_type == "pairwise_logistic" or str(artifact.get("score_mode") or "") == "pairwise_median":
            return self._pairwise_median_scores(features, artifact, feature_names, scales)
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
                bucket = content_buckets.bucket_from_feature(feature)
                value = content_buckets.feature_value(name, feature, bucket)
                scale = scales.get(name, {"mean": 0.0, "std": 1.0})
                std = float(scale.get("std", 1.0)) or 1.0
                normalized = (value - float(scale.get("mean", 0.0))) / std
                normalized = max(-_MAX_NORMALIZED_ABS, min(_MAX_NORMALIZED_ABS, normalized))
                score += normalized * float(weights.get(name, 0.0))
            scores.append(self._normalize_trained_score(score, model_type))
        return scores

    def _pairwise_median_scores(
        self,
        features: list[CandidateFeatures],
        artifact: dict,
        feature_names: list[str],
        scales: dict[str, dict[str, float]],
    ) -> list[float]:
        if not features:
            return []
        rows: list[list[float]] = []
        for feature in features:
            bucket = content_buckets.bucket_from_feature(feature)
            rows.append([
                self._normalized_feature_value(feature, name, scales, bucket)
                for name in feature_names
            ])
        columns = list(zip(*rows, strict=False)) if rows else []
        medians = [sorted(column)[len(column) // 2] if column else 0.0 for column in columns]
        weights: dict[str, float] = artifact.get("weights", {})
        intercept = float(artifact.get("intercept", 0.0))
        scores: list[float] = []
        for row in rows:
            raw = intercept + sum(
                (value - median) * float(weights.get(name, 0.0))
                for value, median, name in zip(row, medians, feature_names, strict=False)
            )
            scores.append(self._normalize_trained_score(raw, "logistic"))
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
                bucket = content_buckets.bucket_from_feature(feature)
                row.append(self._normalized_feature_value(feature, name, scales, bucket))
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
            valid_for = artifact.get("valid_for")
            if isinstance(valid_for, list) and valid_for:
                normalized_valid_for = {str(item).strip().lower() for item in valid_for if str(item).strip()}
                if profile.strip().lower() not in normalized_valid_for:
                    return False, "profile_not_in_artifact_valid_for"
        if str(artifact.get("selected_model") or "").strip().lower() == "fallback_heuristic":
            return False, "fallback_heuristic_selected"
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
        validation_ok, validation_reason = self._validation_passes(validation, artifact)
        if not validation_ok:
            return False, validation_reason
        return True, "usable"

    def _validation_passes(self, validation: dict, artifact: dict | None = None) -> tuple[bool, str]:
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
        filter_ok, filter_reason = self._filter_gate_passes(artifact)
        if filter_ok is False:
            return False, filter_reason
        if not ranking_ok:
            return False, "validation_ranking_metrics_below_minimum"
        return True, "usable"

    def _filter_gate_passes(self, artifact: dict | None) -> tuple[bool | None, str]:
        if not isinstance(artifact, dict):
            return None, "filter_gate_missing"

        gate = artifact.get("filter_gate")
        if isinstance(gate, dict) and gate:
            if isinstance(gate.get("bucket_gates"), dict) and bool(gate.get("passed")):
                return True, "usable"
            metrics = gate
        else:
            calibration = artifact.get("serving_threshold_calibration")
            if not isinstance(calibration, dict):
                return None, "filter_gate_missing"
            metrics = calibration.get("metrics")
            if not isinstance(metrics, dict):
                return None, "filter_gate_missing"

        try:
            precision = float(metrics.get("precision", 0.0))
            recall = float(metrics.get("recall", 0.0))
            f1 = float(metrics.get("f1", 0.0))
            selected_rate = self._filter_selected_rate(metrics)
        except (TypeError, ValueError):
            return False, "validation_filter_gate_invalid"

        if precision < _MIN_FILTER_PRECISION:
            return False, "validation_filter_precision_below_minimum"
        if recall < _MIN_FILTER_RECALL:
            return False, "validation_filter_recall_below_minimum"
        if f1 < _MIN_FILTER_F1:
            return False, "validation_filter_f1_below_minimum"
        if selected_rate > _MAX_FILTER_SELECTED_RATE:
            return False, "validation_filter_selected_rate_above_maximum"
        return True, "usable"

    def _filter_selected_rate(self, metrics: dict) -> float:
        if "selected_rate" in metrics:
            return float(metrics.get("selected_rate", 0.0))
        selected_count = int(
            metrics.get(
                "selected_count",
                int(metrics.get("true_positive", 0) or 0) + int(metrics.get("false_positive", 0) or 0),
            )
            or 0
        )
        rows = int(metrics.get("rows") or 0)
        if rows <= 0:
            return 0.0
        return float(selected_count / rows)

    def _safe_blend(self, artifact: dict) -> float:
        requested = float(artifact.get("blend", _MAX_BLEND))
        return max(_MIN_BLEND, min(_MAX_BLEND, requested))

    def _safe_serving_threshold(self, artifact: dict, *, bucket: str | None = None) -> float:
        if bucket:
            gate = artifact.get("filter_gate")
            if isinstance(gate, dict):
                per_bucket = gate.get("serving_threshold_per_bucket")
                if isinstance(per_bucket, dict) and bucket in per_bucket:
                    try:
                        return self._clamp_score(float(per_bucket[bucket]))
                    except (TypeError, ValueError):
                        pass
        try:
            requested = float(artifact.get("serving_threshold", 0.0))
        except (TypeError, ValueError):
            requested = 0.0
        if not math.isfinite(requested):
            return 0.0
        return self._clamp_score(requested)

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

    def _bucket_model(self, artifact: dict, bucket: str) -> str:
        gate = artifact.get("filter_gate")
        if isinstance(gate, dict):
            mapping = gate.get("selected_model_per_bucket")
            if isinstance(mapping, dict):
                return str(mapping.get(bucket) or "trained")
        return "trained"

    def _normalized_feature_value(
        self,
        feature: CandidateFeatures,
        name: str,
        scales: dict[str, dict[str, float]],
        bucket: str,
    ) -> float:
        value = content_buckets.feature_value(name, feature, bucket)
        scale = scales.get(name, {"mean": 0.0, "std": 1.0})
        std = float(scale.get("std", 1.0)) or 1.0
        normalized = (value - float(scale.get("mean", 0.0))) / std
        return max(-_MAX_NORMALIZED_ABS, min(_MAX_NORMALIZED_ABS, normalized))


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
