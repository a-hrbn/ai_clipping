from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from config import settings
from services import feature_builder


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def build_jynxzi_workflow_state(
    *,
    date_from: str,
    date_to: str,
    skip_covered: bool,
    merge_output: bool,
) -> dict[str, object]:
    use_user_labels = (os.environ.get("VOD_RANKER_USE_USER_LABELS") or "0").strip() or "0"
    return {
        "workflow": "jynxzi_vod_training",
        "date_from": date_from,
        "date_to": date_to,
        "skip_covered": bool(skip_covered),
        "merge_output": bool(merge_output),
        "whisper_model": settings.whisper_model,
        "dataset_llm_provider": (os.environ.get("TRAINING_DATASET_LLM_PROVIDER") or "").strip() or "ollama",
        "enable_clip_judge": str(bool(settings.enable_clip_judge)).lower(),
        "clip_judge_fail_fast": str(bool(settings.clip_judge_fail_fast)).lower(),
        "judge_provider_fallback_enabled": str(bool(settings.judge_provider_fallback_enabled)).lower(),
        "ranker_short_form_mode": (os.environ.get("VOD_RANKER_SHORT_FORM_MODE") or "off").strip().lower() or "off",
        "ranker_gold_train_weight": (os.environ.get("VOD_RANKER_GOLD_TRAIN_WEIGHT") or "2.0").strip() or "2.0",
        "ranker_use_gold_train": (
            (os.environ.get("VOD_RANKER_USE_GOLD_TRAIN") or "0").strip() or "0"
            if use_user_labels == "1"
            else "0"
        ),
        "ranker_hard_negative_weight": (os.environ.get("VOD_RANKER_HARD_NEGATIVE_WEIGHT") or "4.0").strip() or "4.0",
        "ranker_use_gold_hard_negatives": (
            (os.environ.get("VOD_RANKER_USE_GOLD_HARD_NEGATIVES") or "0").strip() or "0"
            if use_user_labels == "1"
            else "0"
        ),
        "ranker_use_user_labels": use_user_labels,
        "ranker_rollout_review_negative_weight": (
            os.environ.get("VOD_RANKER_ROLLOUT_REVIEW_NEGATIVE_WEIGHT") or "4.0"
        ).strip() or "4.0",
        "ranker_outcome_positive_weight": (
            os.environ.get("VOD_RANKER_OUTCOME_POSITIVE_WEIGHT") or "2.0"
        ).strip() or "2.0",
        "ranker_use_mm_judge_proxy": (
            os.environ.get("VOD_RANKER_USE_MM_JUDGE_PROXY") or "1"
        ).strip() or "1",
        "ranker_mm_judge_weight": (
            os.environ.get("VOD_RANKER_MM_JUDGE_WEIGHT") or "0.4"
        ).strip() or "0.4",
        "ranker_max_features": (os.environ.get("RANKER_MAX_FEATURES") or "40").strip() or "40",
        "ranker_prefer_lightgbm": (os.environ.get("RANKER_PREFER_LIGHTGBM") or "1").strip() or "1",
        "ranker_lightgbm_selection_tolerance": (
            os.environ.get("RANKER_LIGHTGBM_SELECTION_TOLERANCE") or "0.02"
        ).strip() or "0.02",
        "ranker_rollout_threshold_min_recall": (
            os.environ.get("VOD_RANKER_ROLLOUT_THRESHOLD_MIN_RECALL") or "0.85"
        ).strip() or "0.85",
        "ranker_rollout_threshold_max_selected_rate": (
            os.environ.get("VOD_RANKER_ROLLOUT_THRESHOLD_MAX_SELECTED_RATE") or "0.75"
        ).strip() or "0.75",
        "ranker_rollout_threshold_target_precision": (
            os.environ.get("VOD_RANKER_ROLLOUT_THRESHOLD_TARGET_PRECISION") or "0.85"
        ).strip() or "0.85",
        "ranker_primary_label_policy": (
            os.environ.get("VOD_RANKER_PRIMARY_LABEL_POLICY") or "outcome_only"
        ).strip() or "outcome_only",
        "ranker_require_cross_platform_outcome": (
            os.environ.get("VOD_RANKER_REQUIRE_CROSS_PLATFORM_OUTCOME") or "1"
        ).strip() or "1",
        "ranker_min_cross_platform_train_rows": (
            os.environ.get("VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS") or "70"
        ).strip() or "70",
        "outcome_holdout_target_positive_rows": (
            os.environ.get("OUTCOME_HOLDOUT_TARGET_POSITIVE_ROWS") or "300"
        ).strip() or "300",
        "rollout_readiness_target_new_vod_groups": (
            os.environ.get("ROLLOUT_READINESS_TARGET_NEW_VOD_GROUPS") or "20"
        ).strip() or "20",
        "rollout_readiness_target_available_groups": (
            os.environ.get("ROLLOUT_READINESS_TARGET_AVAILABLE_GROUPS") or "40"
        ).strip() or "40",
        "rollout_readiness_max_candidates_per_vod": (
            os.environ.get("ROLLOUT_READINESS_MAX_CANDIDATES_PER_VOD") or "3"
        ).strip() or "3",
        "targeted_backfill_require_all_vods": (
            os.environ.get("TARGETED_BACKFILL_REQUIRE_ALL_VODS") or "1"
        ).strip() or "1",
        "cross_platform_fingerprint_enabled": (
            os.environ.get("CROSS_PLATFORM_FINGERPRINT_ENABLED") or "1"
        ).strip() or "1",
        "cross_platform_fingerprint_min_confidence": (
            os.environ.get("CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE") or "0.50"
        ).strip() or "0.50",
        "cross_platform_vod_manifest_path": (
            os.environ.get("CROSS_PLATFORM_VOD_MANIFEST_PATH")
            or str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_vods.jsonl")
        ).strip(),
        "cross_platform_repost_manifest_path": (
            os.environ.get("CROSS_PLATFORM_REPOST_MANIFEST_PATH")
            or str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_reposts.jsonl")
        ).strip(),
        "cross_platform_rows_path": (
            os.environ.get("CROSS_PLATFORM_FINGERPRINT_ROWS_PATH")
            or str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_rows.jsonl")
        ).strip(),
        "cross_platform_diagnostics_path": (
            os.environ.get("CROSS_PLATFORM_FINGERPRINT_DIAGNOSTICS_PATH")
            or str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_diagnostics.jsonl")
        ).strip(),
        "cross_platform_source_manifest_path": (
            os.environ.get("CROSS_PLATFORM_SOURCE_MANIFEST_PATH")
            or str(BACKEND_ROOT / "ml" / "cross_platform_sources.csv")
        ).strip(),
        "cross_platform_url_manifest_path": (
            os.environ.get("CROSS_PLATFORM_URL_MANIFEST_PATH")
            or str(BACKEND_ROOT / "ml" / "viral_clips.txt")
        ).strip(),
        "cross_platform_max_reposts": (os.environ.get("CROSS_PLATFORM_MAX_REPOSTS") or "3600").strip() or "3600",
        "cross_platform_min_reposts": (os.environ.get("CROSS_PLATFORM_MIN_REPOSTS") or "0").strip() or "0",
        "cross_platform_min_matched_rows": (
            os.environ.get("CROSS_PLATFORM_MIN_MATCHED_ROWS") or "70"
        ).strip() or "70",
        "cross_platform_seed_min_matched_rows": (
            os.environ.get("CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS") or "70"
        ).strip() or "70",
        "cross_platform_min_matched_vods": (
            os.environ.get("CROSS_PLATFORM_MIN_MATCHED_VODS") or "20"
        ).strip() or "20",
        "cross_platform_refresh_vods": (
            os.environ.get("CROSS_PLATFORM_REFRESH_VODS") or "0"
        ).strip() or "0",
        "cross_platform_second_pass_after_vod_build": (
            os.environ.get("CROSS_PLATFORM_SECOND_PASS_AFTER_VOD_BUILD") or "1"
        ).strip() or "1",
        "cross_platform_refresh_reposts": (
            os.environ.get("CROSS_PLATFORM_REFRESH_REPOSTS") or "0"
        ).strip() or "0",
        "cross_platform_reuse_existing_rows": (
            os.environ.get("CROSS_PLATFORM_REUSE_EXISTING_ROWS") or "0"
        ).strip() or "0",
        "cross_platform_holdout_ratio": (
            os.environ.get("CROSS_PLATFORM_HOLDOUT_RATIO") or "0.20"
        ).strip() or "0.20",
        "cross_platform_discover_reposts": (
            os.environ.get("CROSS_PLATFORM_DISCOVER_REPOSTS") or "1"
        ).strip() or "1",
        "cross_platform_download_reposts": (
            os.environ.get("CROSS_PLATFORM_DOWNLOAD_REPOSTS") or "1"
        ).strip() or "1",
        "cross_platform_full_vod_audio_fingerprints": (
            os.environ.get("CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS") or "1"
        ).strip() or "1",
        "cross_platform_full_vod_audio_max_vods": (
            os.environ.get("CROSS_PLATFORM_FULL_VOD_AUDIO_MAX_VODS") or "0"
        ).strip() or "0",
        "cross_platform_full_vod_audio_workers": (
            os.environ.get("CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS") or "2"
        ).strip() or "2",
        "cross_platform_keep_full_vod_audio": (
            os.environ.get("CROSS_PLATFORM_KEEP_FULL_VOD_AUDIO") or "0"
        ).strip() or "0",
        "cross_platform_coarse_vod_tiling": (
            os.environ.get("CROSS_PLATFORM_COARSE_VOD_TILING") or "0"
        ).strip() or "0",
        "cross_platform_coarse_vod_tile_sec": (
            os.environ.get("CROSS_PLATFORM_COARSE_VOD_TILE_SEC") or "360"
        ).strip() or "360",
        "cross_platform_coarse_vod_stride_sec": (
            os.environ.get("CROSS_PLATFORM_COARSE_VOD_STRIDE_SEC") or "180"
        ).strip() or "180",
        "cross_platform_coarse_vod_max_tiles_per_vod": (
            os.environ.get("CROSS_PLATFORM_COARSE_VOD_MAX_TILES_PER_VOD") or "0"
        ).strip() or "0",
        "cross_platform_coarse_vod_max_vods": (
            os.environ.get("CROSS_PLATFORM_COARSE_VOD_MAX_VODS") or "0"
        ).strip() or "0",
        "cross_platform_coarse_vod_horizon_padding_sec": (
            os.environ.get("CROSS_PLATFORM_COARSE_VOD_HORIZON_PADDING_SEC") or "900"
        ).strip() or "900",
        "storage_dir": settings.storage_dir,
        "feature_build_version": int(getattr(feature_builder, "FEATURE_BUILD_VERSION", 1)),
        "enable_multimodal_embeddings": bool(getattr(settings, "enable_multimodal_embeddings", False)),
        "code_hash": _code_hash(),
    }


def should_retrain(
    *,
    labels_path: Path,
    dataset_path: Path,
    artifact_path: Path,
    state_path: Path,
    current_state: dict[str, object],
    dependency_paths: list[Path] | None = None,
) -> bool:
    if not labels_path.exists() or not dataset_path.exists() or not artifact_path.exists():
        return True
    stored_state = _read_state(state_path)
    if stored_state != current_state:
        return True
    upstream_paths = [labels_path]
    for path in dependency_paths or []:
        if path.exists():
            upstream_paths.append(path)
    labels_mtime = max(path.stat().st_mtime for path in upstream_paths)
    dataset_mtime = dataset_path.stat().st_mtime
    artifact_mtime = artifact_path.stat().st_mtime
    return labels_mtime > min(dataset_mtime, artifact_mtime)


def should_reevaluate(
    *,
    dataset_path: Path,
    artifact_path: Path,
    output_paths: list[Path],
    dependency_paths: list[Path] | None = None,
    state_path: Path,
    current_state: dict[str, object],
) -> bool:
    if not dataset_path.exists() or not artifact_path.exists():
        return False
    stored_state = _read_state(state_path)
    if stored_state != current_state:
        return True
    if any(not path.exists() for path in output_paths):
        return True
    upstream_inputs = [dataset_path, artifact_path]
    for path in dependency_paths or []:
        if path.exists():
            upstream_inputs.append(path)
    upstream_mtime = max(path.stat().st_mtime for path in upstream_inputs)
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output < upstream_mtime


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_state(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in _tracked_files():
        relative = path.relative_to(BACKEND_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tracked_files() -> list[Path]:
    tracked: list[Path] = []
    for relative in ("config.py", "pyproject.toml"):
        path = BACKEND_ROOT / relative
        if path.exists():
            tracked.append(path)
    for folder in ("ml", "services"):
        root = BACKEND_ROOT / folder
        if not root.exists():
            continue
        tracked.extend(sorted(path for path in root.rglob("*.py") if path.is_file()))
    return tracked
