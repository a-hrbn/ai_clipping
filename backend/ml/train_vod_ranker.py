from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import (
    analyze_ranker_feature_confounds,
    build_twitch_training_dataset,
    outcome_view_weighting,
    train_ranker,
    training_workflow_state,
)
from services import vision_language_features
from services.gold_eval_policy import (
    GOLD_HOLDOUT_POLICY_FIELD,
    GOLD_HOLDOUT_POLICY_TOP_VIEWED_AUTO,
    GOLD_HOLDOUT_SPLIT_FIELD,
)
from ml.row_identity import POSITIVE_LABEL_CLASSES, primary_identity_key, row_identity_keys

_MIN_PRIMARY_POSITIVE_ROWS = 6
_MIN_PRIMARY_UNIQUE_VODS = 2
_MIN_PRIMARY_POSITIVE_CLUSTERS = 3
_MIN_PRIMARY_RANKING_GROUPS = 2
_MIN_PRIMARY_MIXED_RANKING_GROUPS = 2
_TARGET_PRIMARY_MIXED_RANKING_GROUPS = 20
_TARGET_NEGATIVE_ROWS_PER_POSITIVE_MAX = 8
_MAX_CANDIDATE_SOURCE_SHARE = 0.40
_SHORT_FORM_MODES = {"off", "supplemental"}
_DEFAULT_HARD_NEGATIVE_WEIGHT = 4.0
_DEFAULT_ROLLOUT_REVIEW_NEGATIVE_WEIGHT = _DEFAULT_HARD_NEGATIVE_WEIGHT
_DEFAULT_OUTCOME_POSITIVE_WEIGHT = 0.55
_DEFAULT_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT = 1.0
_DEFAULT_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT = 0.25
_DEFAULT_SERVING_HARD_NEGATIVE_WEIGHT = 2.0
_DEFAULT_CROSS_PLATFORM_WEIGHT = 1.0
_DEFAULT_CROSS_PLATFORM_PATH = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_rows.jsonl"
_PRIMARY_LABEL_POLICIES = {"all", "outcome_only"}
_JUDGE_TEACHER_FEATURE_NAMES = (
    "judge_score",
    "judge_confidence",
    "judge_scroll_stop_power",
    "judge_standalone_clarity",
    "judge_payoff_strength",
    "judge_narrative_completeness",
    "judge_platform_fit_shortform",
    "has_judge_score",
    "judge_llm_used",
)
_NEW_MODALITY_FEATURE_NAMES = [
    "boundary_variant_count",
    "boundary_variant_margin_score",
    "boundary_variant_score",
    "chat_ema_crossing_score",
    "chat_rate_derivative_score",
    "chat_velocity_score",
    "f0_sigma_score",
    "game_event_score",
    "has_hook_rewrite",
    "hook_curiosity_gap_score",
    "hook_rewrite_score",
    "is_clutch_window",
    "is_match_endpoint",
    "kill_density_in_window",
    "pitch_variation_score",
    "pre_roll_hook_score",
    "shout_onset_score",
    "streamer_shout_score",
    "context_pattern_score",
    "context_pattern_support",
    "context_pattern_views_prior",
    "context_pattern_confidence",
    "context_pattern_similarity_margin",
    "vad_voice_ratio",
    "vl_clutch_score",
    "vl_comedy_score",
    "vl_prompt_max_score",
    "vl_reaction_face_score",
    "vl_scoreboard_score",
    "voice_energy_burst_score",
]
_TWITCH_VOD_ID_PATTERN = re.compile(r"(?:twitch\.tv/videos/|/videos/)(\d+)")
_MAX_HOLDOUT_EXCLUSION_SAMPLES = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _load_workflow_state(raw: str | None) -> dict[str, object] | None:
    if not raw:
        return None
    payload = raw.strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --workflow-state-json payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--workflow-state-json must decode to a JSON object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a VOD-first training dataset from Twitch clip labels, optionally "
            "blend in short-form rows as low-weight supplemental data, and train the ranker."
        )
    )
    parser.add_argument(
        "--labels",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_clip_labels.jsonl"),
    )
    parser.add_argument(
        "--dataset-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "twitch_vod_training_rows.jsonl"),
    )
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--work-dir", default=str(BACKEND_ROOT / "tmp" / "training_runs"))
    parser.add_argument(
        "--include-short-form",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "real_short_form_training_rows.jsonl"),
        help="Optional supplemental short-form JSONL dataset. Missing files are ignored.",
    )
    parser.add_argument(
        "--short-form-weight",
        type=float,
        default=0.35,
        help="Relative weight multiplier for supplemental short-form rows.",
    )
    parser.add_argument(
        "--short-form-mode",
        choices=sorted(_SHORT_FORM_MODES),
        default="off",
        help=(
            "Use 'off' for VOD-only contrast training. Use 'supplemental' to append "
            "short-form rows as low-weight auxiliary data."
        ),
    )
    parser.add_argument(
        "--include-gold-train",
        default="",
        help=(
            "Optional gold-eval JSONL rows. Only rows with gold_holdout_split=train "
            "are merged into the ranker training dataset; holdout rows are ignored."
        ),
    )
    parser.add_argument(
        "--holdout-policy-rows",
        default="",
        help=(
            "Optional JSONL rows used only to exclude holdout VODs/windows from training. "
            "Rows from this file are not merged as labels."
        ),
    )
    parser.add_argument(
        "--gold-train-weight",
        type=float,
        default=2.0,
        help="Sample-weight multiplier for gold-train rows merged into ranker training.",
    )
    parser.add_argument(
        "--include-hard-negative-train",
        default="",
        help=(
            "Optional hard-negative JSONL rows, usually exported from ranker holdout diagnosis. "
            "Rows are forced into ranker training with --hard-negative-weight."
        ),
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=_DEFAULT_HARD_NEGATIVE_WEIGHT,
        help="Sample-weight multiplier for diagnosis-mined hard-negative rows.",
    )
    parser.add_argument(
        "--include-serving-hard-negative-train",
        default="",
        help=(
            "Optional serving-trace hard-negative JSONL rows exported from recent "
            "frontend/backend jobs. Rows are high-ranked organic candidates that "
            "miss known cross-platform outcome windows."
        ),
    )
    parser.add_argument(
        "--serving-hard-negative-weight",
        type=float,
        default=_DEFAULT_SERVING_HARD_NEGATIVE_WEIGHT,
        help="Sample-weight multiplier for serving-trace hard-negative rows.",
    )
    parser.add_argument(
        "--include-outcome-positive-train",
        default="",
        help=(
            "Deprecated optional outcome-train JSONL rows. Twitch top-viewed rows "
            "are ignored by the default outcome-only Jynxzi workflow."
        ),
    )
    parser.add_argument(
        "--outcome-positive-weight",
        type=float,
        default=_DEFAULT_OUTCOME_POSITIVE_WEIGHT,
        help="Sample-weight multiplier for outcome-positive rows.",
    )
    parser.add_argument(
        "--rollout-review-negative-weight",
        type=float,
        default=_DEFAULT_ROLLOUT_REVIEW_NEGATIVE_WEIGHT,
        help=(
            "Sample-weight multiplier for gold-train rows created from bad rollout "
            "auto-accept reviews."
        ),
    )
    parser.add_argument(
        "--include-auto-outcome-negative-train",
        default="",
        help=(
            "Optional auto-outcome negative JSONL rows. Only train-split rows are merged; "
            "holdout rows are excluded from training."
        ),
    )
    parser.add_argument(
        "--auto-outcome-hard-negative-weight",
        type=float,
        default=_DEFAULT_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT,
        help="Sample-weight multiplier for hard auto-outcome negatives.",
    )
    parser.add_argument(
        "--auto-outcome-background-negative-weight",
        type=float,
        default=_DEFAULT_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT,
        help="Sample-weight multiplier for background auto-outcome negatives.",
    )
    parser.add_argument(
        "--include-cross-platform-outcome-train",
        default=str(_DEFAULT_CROSS_PLATFORM_PATH),
        help=(
            "Optional cross-platform audio-fingerprint outcome JSONL rows. "
            "The default artifact path is auto-merged when present; missing files are ignored."
        ),
    )
    parser.add_argument(
        "--primary-label-policy",
        choices=sorted(_PRIMARY_LABEL_POLICIES),
        default="all",
        help=(
            "Controls whether Twitch clip-label rows are used as primary training "
            "labels. Use outcome_only to train only from cross-platform outcome "
            "labels plus explicitly merged outcome negatives/proxies."
        ),
    )
    parser.add_argument(
        "--require-cross-platform-outcome",
        action="store_true",
        help="Fail training when no cross-platform outcome positives are available.",
    )
    parser.add_argument(
        "--min-cross-platform-train-rows",
        type=int,
        default=0,
        help="Fail training when fewer cross-platform outcome positive rows are available.",
    )
    parser.add_argument(
        "--targeted-backfill-labels",
        action="append",
        default=[],
        help=(
            "Optional JSONL label queue for targeted VOD feature backfill. Rows are "
            "normalized as positive labels and merged before the VOD "
            "feature builder runs. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--cross-platform-outcome-weight",
        type=float,
        default=_DEFAULT_CROSS_PLATFORM_WEIGHT,
        help="Sample-weight multiplier for audio-fingerprint cross-platform outcome positives.",
    )
    parser.add_argument(
        "--include-mm-judge-train",
        default="",
        help="Optional multimodal judge proxy JSONL rows imported from persisted judge snapshots.",
    )
    parser.add_argument(
        "--mm-judge-weight",
        type=float,
        default=0.4,
        help="Sample-weight multiplier for multimodal judge proxy rows.",
    )
    parser.add_argument(
        "--per-streamer",
        action="store_true",
        help="Train one artifact per broadcaster using that broadcaster's VOD windows as the primary data.",
    )
    parser.add_argument(
        "--workflow-state-path",
        default=None,
        help=(
            "Path to the workflow-state fingerprint JSON. When combined with --workflow-state-json, "
            "training is skipped if the stored fingerprint matches the current state and all artifacts "
            "already exist. After a successful training run the fingerprint is rewritten to this path."
        ),
    )
    parser.add_argument(
        "--workflow-state-json",
        default=None,
        help=(
            "JSON-encoded current workflow state. When omitted, fingerprint gating is disabled. "
            "When provided alongside --workflow-state-path, training uses "
            "training_workflow_state.should_retrain to decide whether to skip."
        ),
    )
    args = parser.parse_args()

    labels_path = pathlib.Path(args.labels)
    labels = _read_jsonl(labels_path)
    if not labels:
        print(f"No Twitch label rows found in {labels_path}; continuing with outcome/cross-platform rows only.")
    targeted_backfill_paths = [
        pathlib.Path(path)
        for path in args.targeted_backfill_labels
        if str(path or "").strip()
    ]
    targeted_backfill_rows = _read_targeted_backfill_labels(targeted_backfill_paths, args.profile)
    if targeted_backfill_rows:
        labels = _merge_label_rows(labels, targeted_backfill_rows)
        logger.info(
            "training_targeted_backfill_labels_added rows=%s paths=%s",
            len(targeted_backfill_rows),
            ",".join(str(path) for path in targeted_backfill_paths),
        )

    short_form_path = pathlib.Path(args.include_short_form) if args.include_short_form else None
    short_form_rows = _read_jsonl(short_form_path) if short_form_path and short_form_path.exists() else []
    gold_train_path = pathlib.Path(args.include_gold_train) if args.include_gold_train else None
    gold_train_rows = _read_jsonl(gold_train_path) if gold_train_path and gold_train_path.exists() else []
    holdout_policy_path = pathlib.Path(args.holdout_policy_rows) if args.holdout_policy_rows else None
    holdout_policy_rows = _read_jsonl(holdout_policy_path) if holdout_policy_path and holdout_policy_path.exists() else []
    hard_negative_path = pathlib.Path(args.include_hard_negative_train) if args.include_hard_negative_train else None
    hard_negative_rows = _read_jsonl(hard_negative_path) if hard_negative_path and hard_negative_path.exists() else []
    serving_hard_negative_path = (
        pathlib.Path(args.include_serving_hard_negative_train)
        if args.include_serving_hard_negative_train
        else None
    )
    serving_hard_negative_rows = (
        _read_jsonl(serving_hard_negative_path)
        if serving_hard_negative_path and serving_hard_negative_path.exists()
        else []
    )
    outcome_positive_path = (
        pathlib.Path(args.include_outcome_positive_train)
        if args.include_outcome_positive_train
        else None
    )
    outcome_positive_rows = (
        _read_jsonl(outcome_positive_path)
        if outcome_positive_path and outcome_positive_path.exists()
        else []
    )
    auto_outcome_negative_path = (
        pathlib.Path(args.include_auto_outcome_negative_train)
        if args.include_auto_outcome_negative_train
        else None
    )
    auto_outcome_negative_rows = (
        _read_jsonl(auto_outcome_negative_path)
        if auto_outcome_negative_path and auto_outcome_negative_path.exists()
        else []
    )
    cross_platform_path = (
        pathlib.Path(args.include_cross_platform_outcome_train)
        if args.include_cross_platform_outcome_train
        else None
    )
    cross_platform_rows = (
        _read_jsonl(cross_platform_path)
        if cross_platform_path and cross_platform_path.exists()
        else []
    )
    mm_judge_path = pathlib.Path(args.include_mm_judge_train) if args.include_mm_judge_train else None
    mm_judge_rows = _read_jsonl(mm_judge_path) if mm_judge_path and mm_judge_path.exists() else []
    cross_platform_input = _cross_platform_input_summary(cross_platform_path, cross_platform_rows)
    if not cross_platform_rows:
        logger.warning(
            "training_cross_platform_input_empty path=%s exists=%s",
            cross_platform_input["path"],
            cross_platform_input["exists"],
        )
    cross_platform_positive_rows = _cross_platform_positive_count(cross_platform_rows)
    min_cross_platform_rows = max(int(args.min_cross_platform_train_rows), 0)
    if args.require_cross_platform_outcome and cross_platform_positive_rows <= 0:
        raise SystemExit("Cross-platform outcome training is required, but no positive rows were loaded.")
    if min_cross_platform_rows > 0 and cross_platform_positive_rows < min_cross_platform_rows:
        raise SystemExit(
            "Cross-platform outcome training rows below minimum: "
            f"{cross_platform_positive_rows} < {min_cross_platform_rows}"
        )

    if args.per_streamer:
        return _train_per_streamer(
            labels,
            short_form_rows,
            artifact_dir=pathlib.Path(args.artifact_dir),
            work_dir=pathlib.Path(args.work_dir),
            short_form_weight=args.short_form_weight,
            short_form_mode=args.short_form_mode,
            gold_train_rows=gold_train_rows,
            gold_train_weight=args.gold_train_weight,
            holdout_policy_rows=holdout_policy_rows,
            rollout_review_negative_weight=args.rollout_review_negative_weight,
            hard_negative_rows=hard_negative_rows,
            hard_negative_weight=args.hard_negative_weight,
            serving_hard_negative_rows=serving_hard_negative_rows,
            serving_hard_negative_weight=args.serving_hard_negative_weight,
            outcome_positive_rows=outcome_positive_rows,
            outcome_positive_weight=args.outcome_positive_weight,
            auto_outcome_negative_rows=auto_outcome_negative_rows,
            auto_outcome_hard_negative_weight=args.auto_outcome_hard_negative_weight,
            auto_outcome_background_negative_weight=args.auto_outcome_background_negative_weight,
            cross_platform_rows=cross_platform_rows,
            cross_platform_weight=args.cross_platform_outcome_weight,
            primary_label_policy=args.primary_label_policy,
            mm_judge_rows=mm_judge_rows,
            mm_judge_weight=args.mm_judge_weight,
        )

    artifact_dir = pathlib.Path(args.artifact_dir)
    dataset_out = pathlib.Path(args.dataset_out)
    artifact_path = artifact_dir / f"{args.profile}_ranker.json"

    workflow_state = _load_workflow_state(args.workflow_state_json)
    state_path = pathlib.Path(args.workflow_state_path) if args.workflow_state_path else None
    if workflow_state is not None and state_path is not None:
        if not training_workflow_state.should_retrain(
            labels_path=labels_path,
            dataset_path=dataset_out,
            artifact_path=artifact_path,
            state_path=state_path,
            current_state=workflow_state,
            dependency_paths=[
                path
                for path in (
                    gold_train_path,
                    holdout_policy_path,
                    hard_negative_path,
                    serving_hard_negative_path,
                    outcome_positive_path,
                    auto_outcome_negative_path,
                    cross_platform_path,
                    mm_judge_path,
                    *targeted_backfill_paths,
                )
                if path is not None and path.exists()
            ],
        ):
            print(
                json.dumps(
                    {
                        "event": "train_vod_ranker_skipped",
                        "reason": "workflow_state_unchanged",
                        "profile": args.profile,
                        "state_path": str(state_path),
                    },
                    ensure_ascii=True,
                )
            )
            return 0

    rows, holdout_exclusion = build_composite_dataset_with_report(
        labels,
        profile=args.profile,
        work_dir=pathlib.Path(args.work_dir),
        supplemental_rows=short_form_rows,
        supplemental_weight=args.short_form_weight,
        short_form_mode=args.short_form_mode,
        gold_train_rows=gold_train_rows,
        gold_train_weight=args.gold_train_weight,
        holdout_policy_rows=holdout_policy_rows,
        rollout_review_negative_weight=args.rollout_review_negative_weight,
        hard_negative_rows=hard_negative_rows,
        hard_negative_weight=args.hard_negative_weight,
        serving_hard_negative_rows=serving_hard_negative_rows,
        serving_hard_negative_weight=args.serving_hard_negative_weight,
        outcome_positive_rows=outcome_positive_rows,
        outcome_positive_weight=args.outcome_positive_weight,
        auto_outcome_negative_rows=auto_outcome_negative_rows,
        auto_outcome_hard_negative_weight=args.auto_outcome_hard_negative_weight,
        auto_outcome_background_negative_weight=args.auto_outcome_background_negative_weight,
        cross_platform_rows=cross_platform_rows,
        cross_platform_weight=args.cross_platform_outcome_weight,
        primary_label_policy=args.primary_label_policy,
        mm_judge_rows=mm_judge_rows,
        mm_judge_weight=args.mm_judge_weight,
    )
    _write_jsonl(dataset_out, rows)
    summary = dataset_summary(rows)
    summary["holdout_training_exclusion"] = holdout_exclusion
    summary["cross_platform_input"] = cross_platform_input
    summary["primary_label_policy"] = args.primary_label_policy
    summary["targeted_backfill_input"] = _targeted_backfill_input_summary(
        targeted_backfill_paths,
        targeted_backfill_rows,
    )
    summary["mm_judge_input"] = _mm_judge_input_summary(mm_judge_path, mm_judge_rows)
    print(json.dumps(summary, ensure_ascii=True))
    code = _train_models_for_dataset(
        dataset_out,
        args.profile,
        artifact_dir,
    )
    if code == 0 and workflow_state is not None and state_path is not None:
        training_workflow_state.write_state(state_path, workflow_state)
    return code


def build_composite_dataset(
    labels: list[dict],
    profile: str,
    work_dir: pathlib.Path,
    supplemental_rows: list[dict] | None = None,
    supplemental_weight: float = 0.35,
    short_form_mode: str = "off",
    gold_train_rows: list[dict] | None = None,
    gold_train_weight: float = 2.0,
    holdout_policy_rows: list[dict] | None = None,
    rollout_review_negative_weight: float = _DEFAULT_ROLLOUT_REVIEW_NEGATIVE_WEIGHT,
    hard_negative_rows: list[dict] | None = None,
    hard_negative_weight: float = _DEFAULT_HARD_NEGATIVE_WEIGHT,
    serving_hard_negative_rows: list[dict] | None = None,
    serving_hard_negative_weight: float = _DEFAULT_SERVING_HARD_NEGATIVE_WEIGHT,
    outcome_positive_rows: list[dict] | None = None,
    outcome_positive_weight: float = _DEFAULT_OUTCOME_POSITIVE_WEIGHT,
    auto_outcome_negative_rows: list[dict] | None = None,
    auto_outcome_hard_negative_weight: float = _DEFAULT_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT,
    auto_outcome_background_negative_weight: float = _DEFAULT_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT,
    cross_platform_rows: list[dict] | None = None,
    cross_platform_weight: float = _DEFAULT_CROSS_PLATFORM_WEIGHT,
    primary_label_policy: str = "all",
    mm_judge_rows: list[dict] | None = None,
    mm_judge_weight: float = 0.4,
) -> list[dict]:
    rows, _report = build_composite_dataset_with_report(
        labels,
        profile=profile,
        work_dir=work_dir,
        supplemental_rows=supplemental_rows,
        supplemental_weight=supplemental_weight,
        short_form_mode=short_form_mode,
        gold_train_rows=gold_train_rows,
        gold_train_weight=gold_train_weight,
        holdout_policy_rows=holdout_policy_rows,
        rollout_review_negative_weight=rollout_review_negative_weight,
        hard_negative_rows=hard_negative_rows,
        hard_negative_weight=hard_negative_weight,
        serving_hard_negative_rows=serving_hard_negative_rows,
        serving_hard_negative_weight=serving_hard_negative_weight,
        outcome_positive_rows=outcome_positive_rows,
        outcome_positive_weight=outcome_positive_weight,
        auto_outcome_negative_rows=auto_outcome_negative_rows,
        auto_outcome_hard_negative_weight=auto_outcome_hard_negative_weight,
        auto_outcome_background_negative_weight=auto_outcome_background_negative_weight,
        cross_platform_rows=cross_platform_rows,
        cross_platform_weight=cross_platform_weight,
        primary_label_policy=primary_label_policy,
        mm_judge_rows=mm_judge_rows,
        mm_judge_weight=mm_judge_weight,
    )
    return rows


def build_composite_dataset_with_report(
    labels: list[dict],
    profile: str,
    work_dir: pathlib.Path,
    supplemental_rows: list[dict] | None = None,
    supplemental_weight: float = 0.35,
    short_form_mode: str = "off",
    gold_train_rows: list[dict] | None = None,
    gold_train_weight: float = 2.0,
    holdout_policy_rows: list[dict] | None = None,
    rollout_review_negative_weight: float = _DEFAULT_ROLLOUT_REVIEW_NEGATIVE_WEIGHT,
    hard_negative_rows: list[dict] | None = None,
    hard_negative_weight: float = _DEFAULT_HARD_NEGATIVE_WEIGHT,
    serving_hard_negative_rows: list[dict] | None = None,
    serving_hard_negative_weight: float = _DEFAULT_SERVING_HARD_NEGATIVE_WEIGHT,
    outcome_positive_rows: list[dict] | None = None,
    outcome_positive_weight: float = _DEFAULT_OUTCOME_POSITIVE_WEIGHT,
    auto_outcome_negative_rows: list[dict] | None = None,
    auto_outcome_hard_negative_weight: float = _DEFAULT_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT,
    auto_outcome_background_negative_weight: float = _DEFAULT_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT,
    cross_platform_rows: list[dict] | None = None,
    cross_platform_weight: float = _DEFAULT_CROSS_PLATFORM_WEIGHT,
    primary_label_policy: str = "all",
    mm_judge_rows: list[dict] | None = None,
    mm_judge_weight: float = 0.4,
) -> tuple[list[dict], dict[str, object]]:
    policy_rows = holdout_policy_rows if holdout_policy_rows is not None else (gold_train_rows or [])
    holdout_policy = _build_holdout_training_exclusion_policy(policy_rows or [])
    exclusion_report = _new_holdout_exclusion_report(holdout_policy)
    primary_label_policy = _normalize_primary_label_policy(primary_label_policy)
    if primary_label_policy == "outcome_only":
        training_labels = [
            row for row in labels if bool(row.get("targeted_feature_backfill"))
        ]
        exclusion_report["twitch_label_rows_ignored_by_policy"] = len(labels)
        exclusion_report["targeted_feature_backfill_label_rows"] = len(training_labels)
    else:
        training_labels = _filter_holdout_training_rows(
            labels,
            holdout_policy,
            stage="twitch_label_input",
            report=exclusion_report,
        )
    if cross_platform_rows:
        cross_platform_labels = _filter_holdout_training_rows(
            _prepare_cross_platform_vod_labels(
                cross_platform_rows,
                profile,
                cross_platform_weight,
            ),
            holdout_policy,
            stage="cross_platform_label_input",
            report=exclusion_report,
        )
        if cross_platform_labels:
            logger.info("training_cross_platform_labels_added rows=%s", len(cross_platform_labels))
            training_labels = [*training_labels, *cross_platform_labels]
    _log_training_vod_usage_overview(
        labels,
        training_labels,
        policy_rows or [],
        holdout_policy,
    )
    vod_rows = build_twitch_training_dataset.build_dataset(training_labels, profile, work_dir)
    targeted_feature_rows = [row for row in vod_rows if bool(row.get("targeted_feature_backfill"))]
    if targeted_feature_rows:
        exclusion_report["targeted_feature_backfill_rows_built"] = len(targeted_feature_rows)
        exclusion_report["targeted_feature_backfill_vods_built"] = len(
            {
                str(row.get("vod_id") or "").strip()
                for row in targeted_feature_rows
                if str(row.get("vod_id") or "").strip()
            }
        )
        vod_rows = [row for row in vod_rows if not bool(row.get("targeted_feature_backfill"))]
    vod_rows = _filter_holdout_training_rows(
        vod_rows,
        holdout_policy,
        stage="twitch_vod_rows",
        report=exclusion_report,
    )
    merged = list(vod_rows)
    if _normalize_short_form_mode(short_form_mode) != "off" and supplemental_rows:
        prepared_short_form = _filter_holdout_training_rows(
            _prepare_short_form_rows(supplemental_rows, profile, supplemental_weight),
            holdout_policy,
            stage="short_form_rows",
            report=exclusion_report,
        )
        merged.extend(prepared_short_form)
    if gold_train_rows:
        merged = _merge_gold_train_rows(
            merged,
            _filter_holdout_training_rows(
                _prepare_gold_train_rows(
                    gold_train_rows,
                    profile,
                    gold_train_weight,
                    rollout_review_negative_weight=rollout_review_negative_weight,
                ),
                holdout_policy,
                stage="gold_train_rows",
                report=exclusion_report,
            ),
        )
    if hard_negative_rows:
        merged = _merge_gold_train_rows(
            merged,
            _filter_holdout_training_rows(
                _prepare_hard_negative_rows(hard_negative_rows, profile, hard_negative_weight),
                holdout_policy,
                stage="hard_negative_rows",
                report=exclusion_report,
            ),
        )
    if serving_hard_negative_rows:
        merged = _merge_gold_train_rows(
            merged,
            _filter_holdout_training_rows(
                _prepare_serving_hard_negative_rows(
                    serving_hard_negative_rows,
                    profile,
                    serving_hard_negative_weight,
                ),
                holdout_policy,
                stage="serving_hard_negative_rows",
                report=exclusion_report,
            ),
        )
    if outcome_positive_rows:
        merged = _merge_gold_train_rows(
            merged,
            _filter_holdout_training_rows(
                _prepare_outcome_positive_train_rows(
                    outcome_positive_rows,
                    profile,
                    outcome_positive_weight,
                ),
                holdout_policy,
                stage="outcome_positive_rows",
                report=exclusion_report,
            ),
        )
    if auto_outcome_negative_rows:
        merged = _merge_gold_train_rows(
            merged,
            _filter_holdout_training_rows(
                _prepare_auto_outcome_negative_rows(
                    auto_outcome_negative_rows,
                    profile,
                    hard_negative_weight=auto_outcome_hard_negative_weight,
                    background_negative_weight=auto_outcome_background_negative_weight,
                ),
                holdout_policy,
                stage="auto_outcome_negative_rows",
                report=exclusion_report,
            ),
        )
    if mm_judge_rows:
        merged = _merge_gold_train_rows(
            merged,
            _filter_holdout_training_rows(
                _prepare_mm_judge_rows(mm_judge_rows, profile, mm_judge_weight),
                holdout_policy,
                stage="mm_judge_rows",
                report=exclusion_report,
            ),
        )
        exclusion_report["mm_judge_feature_replay"] = _apply_mm_judge_feature_replay(merged, mm_judge_rows)
    merged = _filter_holdout_training_rows(
        merged,
        holdout_policy,
        stage="final_merged_rows",
        report=exclusion_report,
    )
    return merged, _finalize_holdout_exclusion_report(exclusion_report)


def _build_holdout_training_exclusion_policy(gold_rows: list[dict]) -> dict[str, object]:
    holdout_vod_ids: set[str] = set()
    holdout_source_keys: set[str] = set()
    holdout_windows_by_vod: dict[str, list[tuple[float, float]]] = defaultdict(list)
    holdout_rows = 0
    window_only_holdout_rows = 0
    for row in gold_rows:
        if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() != "holdout":
            continue
        holdout_rows += 1
        window_only_holdout = _is_top_viewed_auto_holdout_row(row)
        vod_id = _row_vod_id(row)
        source_key = _row_source_key(row)
        if window_only_holdout:
            window_only_holdout_rows += 1
            window = _row_window(row)
            if vod_id and window is not None:
                holdout_windows_by_vod[vod_id].append(window)
        elif vod_id:
            holdout_vod_ids.add(vod_id)
            window = _row_window(row)
            if window is not None:
                holdout_windows_by_vod[vod_id].append(window)
        elif source_key:
            holdout_source_keys.add(source_key)

    return {
        "enabled": bool(holdout_vod_ids or holdout_source_keys),
        "holdout_rows": holdout_rows,
        "window_only_holdout_rows": window_only_holdout_rows,
        "holdout_vod_ids": sorted(holdout_vod_ids),
        "holdout_source_keys": sorted(holdout_source_keys),
        "holdout_windows_by_vod": {
            vod_id: sorted(windows)
            for vod_id, windows in sorted(holdout_windows_by_vod.items())
        },
    }


def _new_holdout_exclusion_report(policy: dict[str, object]) -> dict[str, object]:
    return {
        "enabled": bool(policy.get("enabled")),
        "strict_vod_level": True,
        "holdout_rows": int(policy.get("holdout_rows") or 0),
        "window_only_holdout_rows": int(policy.get("window_only_holdout_rows") or 0),
        "holdout_vod_count": len(policy.get("holdout_vod_ids") or []),
        "holdout_source_count": len(policy.get("holdout_source_keys") or []),
        "filtered_rows": 0,
        "filtered_by_stage": defaultdict(int),
        "filtered_by_reason": defaultdict(int),
        "sample_filtered_rows": [],
    }


def _log_training_vod_usage_overview(
    labels: list[dict],
    training_labels: list[dict],
    gold_rows: list[dict],
    policy: dict[str, object],
) -> None:
    raw_by_vod = _group_rows_by_vod(labels, skip_ambiguous=True)
    kept_by_vod = _group_rows_by_vod(training_labels, skip_ambiguous=True)
    gold_by_vod = _group_rows_by_vod(gold_rows, skip_ambiguous=False)
    current_vod_ids = set(raw_by_vod)
    strict_holdout_vod_ids = {
        str(vod_id)
        for vod_id in (policy.get("holdout_vod_ids") or [])
        if str(vod_id).strip()
    }
    window_holdout_vod_ids = {
        str(vod_id)
        for vod_id, windows in (policy.get("holdout_windows_by_vod") or {}).items()
        if str(vod_id).strip() and windows
    }

    complete_blocked_ids = sorted(
        current_vod_ids & strict_holdout_vod_ids,
        key=lambda vod_id: _vod_sort_key(vod_id, raw_by_vod, gold_by_vod),
    )
    partially_blocked_ids = sorted(
        (current_vod_ids & window_holdout_vod_ids) - strict_holdout_vod_ids,
        key=lambda vod_id: _vod_sort_key(vod_id, raw_by_vod, gold_by_vod),
    )
    fully_used_ids = sorted(
        set(kept_by_vod) - strict_holdout_vod_ids - window_holdout_vod_ids,
        key=lambda vod_id: _vod_sort_key(vod_id, raw_by_vod, gold_by_vod),
    )
    strict_not_in_current_labels = len(strict_holdout_vod_ids - current_vod_ids)

    logger.info(
        "training_vod_usage_overview_summary current_label_vods=%s complete_blocked=%s partially_blocked=%s fully_used=%s strict_gold_blocked_not_in_current_labels=%s",
        len(current_vod_ids),
        len(complete_blocked_ids),
        len(partially_blocked_ids),
        len(fully_used_ids),
        strict_not_in_current_labels,
    )
    _log_vod_usage_entries(
        "complete_blocked",
        complete_blocked_ids,
        raw_by_vod,
        kept_by_vod,
        gold_by_vod,
        policy,
    )
    _log_vod_usage_entries(
        "partially_blocked",
        partially_blocked_ids,
        raw_by_vod,
        kept_by_vod,
        gold_by_vod,
        policy,
    )
    _log_vod_usage_entries(
        "fully_used",
        fully_used_ids,
        raw_by_vod,
        kept_by_vod,
        gold_by_vod,
        policy,
    )


def _log_vod_usage_entries(
    category: str,
    vod_ids: list[str],
    raw_by_vod: dict[str, list[dict]],
    kept_by_vod: dict[str, list[dict]],
    gold_by_vod: dict[str, list[dict]],
    policy: dict[str, object],
) -> None:
    if not vod_ids:
        logger.info("training_vod_usage_overview_%s none", category)
        return

    date_counts: dict[str, int] = defaultdict(int)
    for vod_id in vod_ids:
        vod_date, date_source = _vod_display_date(vod_id, raw_by_vod, gold_by_vod)
        date_counts[vod_date] += 1
        windows = list((policy.get("holdout_windows_by_vod") or {}).get(vod_id, []))
        logger.info(
            "training_vod_usage_overview_%s vod_date=%s date_instance=%s date_source=%s label_rows=%s kept_label_rows=%s gold_rows=%s blocked_windows=%s",
            category,
            vod_date,
            date_counts[vod_date],
            date_source,
            len(raw_by_vod.get(vod_id, [])),
            len(kept_by_vod.get(vod_id, [])),
            len(gold_by_vod.get(vod_id, [])),
            len(windows),
        )


def _group_rows_by_vod(rows: list[dict], *, skip_ambiguous: bool) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if skip_ambiguous and row.get("label_class") == "ambiguous":
            continue
        vod_id = _row_vod_id(row)
        if not vod_id:
            continue
        grouped[vod_id].append(row)
    return dict(grouped)


def _vod_sort_key(
    vod_id: str,
    raw_by_vod: dict[str, list[dict]],
    gold_by_vod: dict[str, list[dict]],
) -> tuple[str, str]:
    vod_date, _source = _vod_display_date(vod_id, raw_by_vod, gold_by_vod)
    return (vod_date, vod_id)


def _vod_display_date(
    vod_id: str,
    raw_by_vod: dict[str, list[dict]],
    gold_by_vod: dict[str, list[dict]],
) -> tuple[str, str]:
    for source, rows in (
        ("label_timestamp", raw_by_vod.get(vod_id, [])),
        ("gold_timestamp", gold_by_vod.get(vod_id, [])),
    ):
        dates = [
            parsed
            for row in rows
            if (parsed := _row_vod_date(row)) is not None
        ]
        if dates:
            return (min(dates), source)
    return ("unknown", "missing")


def _row_vod_date(row: dict) -> str | None:
    for key in (
        "vod_created_at",
        "stream_started_at",
        "broadcast_started_at",
        "started_at",
        "published_at",
        "cluster_created_at",
        "created_at",
    ):
        parsed = _parse_iso_date(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_iso_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _filter_holdout_training_rows(
    rows: list[dict],
    policy: dict[str, object],
    *,
    stage: str,
    report: dict[str, object],
) -> list[dict]:
    if not rows:
        return list(rows)

    kept: list[dict] = []
    for row in rows:
        reason = _holdout_training_exclusion_reason(row, policy)
        if reason:
            _record_holdout_exclusion(report, row, stage=stage, reason=reason)
            continue
        kept.append(row)
    return kept


def _normalize_primary_label_policy(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in _PRIMARY_LABEL_POLICIES:
        raise ValueError(f"Unsupported primary label policy: {value}")
    return normalized


def _cross_platform_positive_count(rows: list[dict]) -> int:
    return len(_prepare_cross_platform_vod_labels(rows, profile="jynxzi", cross_platform_weight=1.0))


def _holdout_training_exclusion_reason(row: dict, policy: dict[str, object]) -> str:
    if _is_promoted_hard_negative_train_row(row):
        return ""
    split = str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower()
    if split == "holdout":
        return "gold_holdout_row"
    if str(row.get("hard_negative_original_split") or "").strip().lower() == "holdout":
        return "promoted_holdout_hard_negative"

    holdout_vod_ids = set(str(item) for item in (policy.get("holdout_vod_ids") or []))
    vod_id = _row_vod_id(row)
    if vod_id and vod_id in holdout_vod_ids:
        return "holdout_vod_id"

    holdout_source_keys = set(str(item) for item in (policy.get("holdout_source_keys") or []))
    source_key = _row_source_key(row)
    if source_key and source_key in holdout_source_keys:
        return "holdout_source_url"

    # This catches rows where a VOD id was recoverable but future policies relax
    # strict VOD-level blocking; keep it here so overlap rules remain centralized.
    window = _row_window(row)
    if vod_id and window is not None:
        for holdout_window in (policy.get("holdout_windows_by_vod") or {}).get(vod_id, []):
            if _windows_overlap(window, holdout_window):
                return "holdout_window_overlap"
    return ""


def _is_top_viewed_auto_holdout_row(row: dict) -> bool:
    return str(row.get(GOLD_HOLDOUT_POLICY_FIELD) or "").strip().lower() == GOLD_HOLDOUT_POLICY_TOP_VIEWED_AUTO


def _is_promoted_hard_negative_train_row(row: dict) -> bool:
    if int(row.get("label", -1)) != 0:
        return False
    if str(row.get("dataset_source") or "").strip() != "gold_eval_hard_negative_train":
        return False
    return str(row.get("hard_negative_source") or "").strip() == "ranker_holdout_diagnosis"


def _record_holdout_exclusion(report: dict[str, object], row: dict, *, stage: str, reason: str) -> None:
    report["filtered_rows"] = int(report.get("filtered_rows") or 0) + 1
    report["filtered_by_stage"][stage] += 1
    report["filtered_by_reason"][reason] += 1
    samples = report["sample_filtered_rows"]
    if len(samples) >= _MAX_HOLDOUT_EXCLUSION_SAMPLES:
        return
    samples.append(
        {
            "stage": stage,
            "reason": reason,
            "vod_id": _row_vod_id(row),
            "queue_id": str(row.get("queue_id") or ""),
            "candidate_id": str(row.get("candidate_id") or ""),
            "dataset_source": str(row.get("dataset_source") or ""),
            "label_class": str(row.get("label_class") or ""),
            "window_start": _nullable_float(row.get("window_start")),
            "window_end": _nullable_float(row.get("window_end")),
            GOLD_HOLDOUT_SPLIT_FIELD: str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or ""),
        }
    )


def _finalize_holdout_exclusion_report(report: dict[str, object]) -> dict[str, object]:
    finalized = dict(report)
    finalized["filtered_by_stage"] = dict(sorted(report["filtered_by_stage"].items()))
    finalized["filtered_by_reason"] = dict(sorted(report["filtered_by_reason"].items()))
    finalized["sample_filtered_rows"] = list(report["sample_filtered_rows"])
    return finalized


def dataset_summary(rows: list[dict]) -> dict:
    summary = build_twitch_training_dataset.dataset_summary(rows)
    by_source: dict[str, int] = defaultdict(int)
    by_label_class: dict[str, int] = defaultdict(int)
    by_supervision_tier: dict[str, int] = defaultdict(int)
    provenance_coverage: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[str(row.get("dataset_source") or "unknown")] += 1
        label_class = str(row.get("label_class") or "")
        if label_class:
            by_label_class[label_class] += 1
        supervision_tier = str(row.get("supervision_tier") or "")
        if supervision_tier:
            by_supervision_tier[supervision_tier] += 1
        provenance = row.get("feature_provenance") or {}
        for key in ("has_llm_semantics", "has_chat_features", "has_vector_similarity", "has_vod_context"):
            if provenance.get(key):
                provenance_coverage[key] += 1
    summary.update(
        {
            "dataset_sources": dict(sorted(by_source.items())),
            "label_classes": dict(sorted(by_label_class.items())),
            "supervision_tiers": dict(sorted(by_supervision_tier.items())),
            "feature_provenance_coverage": dict(sorted(provenance_coverage.items())),
            "feature_signal_preflight": _feature_signal_preflight(rows),
            "vision_language_model": vision_language_features.model_availability(),
        }
    )
    return summary


def _feature_signal_preflight(rows: list[dict]) -> dict[str, object]:
    counts: dict[str, int] = {}
    nonzero: dict[str, int] = {}
    for name in _NEW_MODALITY_FEATURE_NAMES:
        values = [_row_feature_float(row, name) for row in rows]
        present = [value for value in values if value is not None]
        counts[name] = len(present)
        nonzero[name] = sum(1 for value in present if abs(float(value)) > 1e-9)
    dark_features = [
        name
        for name in _NEW_MODALITY_FEATURE_NAMES
        if rows and nonzero.get(name, 0) == 0
    ]
    if dark_features:
        logger.warning(
            "training_feature_signal_preflight_dark_features rows=%s dark_features=%s",
            len(rows),
            ",".join(dark_features),
        )
    return {
        "rows": len(rows),
        "present_counts": counts,
        "nonzero_counts": nonzero,
        "dark_features": dark_features,
        "ok": not dark_features,
    }


def _row_feature_float(row: dict, name: str) -> float | None:
    features = row.get("features")
    if isinstance(features, dict) and name in features:
        return _nullable_float(features.get(name))
    if name in row:
        return _nullable_float(row.get(name))
    return None


def ranker_dataset_quality(rows: list[dict]) -> dict[str, object]:
    primary_rows = [row for row in rows if str(row.get("dataset_source") or "") != "short_form_supplemental"]
    evaluated_rows = primary_rows or list(rows)
    positive_rows = [row for row in evaluated_rows if int(row.get("label", 0)) == 1]
    negative_rows = [row for row in evaluated_rows if int(row.get("label", 0)) == 0]
    unique_vods = {
        str(row.get("vod_id") or row.get("source_url") or "").strip()
        for row in evaluated_rows
        if str(row.get("vod_id") or row.get("source_url") or "").strip()
    }
    positive_clusters = {
        _positive_cluster_key(row)
        for row in positive_rows
        if _positive_cluster_key(row) is not None
    }
    ranking_groups = {
        train_ranker._ranking_group_key(row, "vod_id")
        for row in evaluated_rows
        if str(train_ranker._ranking_group_key(row, "vod_id")).strip()
    }
    mixed_ranking_groups = _mixed_ranking_groups(evaluated_rows, "vod_id")
    contrast_balance = _vod_contrast_balance(evaluated_rows)
    source_mix = _candidate_source_mix(evaluated_rows)

    violations: list[str] = []
    if len(positive_rows) < _MIN_PRIMARY_POSITIVE_ROWS:
        violations.append("primary_positive_rows_below_minimum")
    if len(negative_rows) == 0:
        violations.append("primary_negative_rows_missing")
    if len(unique_vods) < _MIN_PRIMARY_UNIQUE_VODS:
        violations.append("primary_unique_vods_below_minimum")
    if len(positive_clusters) < _MIN_PRIMARY_POSITIVE_CLUSTERS:
        violations.append("primary_positive_clusters_below_minimum")
    if len(ranking_groups) < _MIN_PRIMARY_RANKING_GROUPS:
        violations.append("primary_ranking_groups_below_minimum")
    if len(mixed_ranking_groups) < _MIN_PRIMARY_MIXED_RANKING_GROUPS:
        violations.append("primary_mixed_ranking_groups_below_minimum")
    if source_mix["peak_share"] > _MAX_CANDIDATE_SOURCE_SHARE:
        violations.append("candidate_source_peak_share_above_maximum")

    return {
        "ok": not violations,
        "evaluated_rows": len(evaluated_rows),
        "primary_rows": len(primary_rows),
        "positive_rows": len(positive_rows),
        "negative_rows": len(negative_rows),
        "unique_vods": len(unique_vods),
        "unique_positive_clusters": len(positive_clusters),
        "ranking_groups": len(ranking_groups),
        "mixed_ranking_groups": len(mixed_ranking_groups),
        "thresholds": {
            "positive_rows": _MIN_PRIMARY_POSITIVE_ROWS,
            "unique_vods": _MIN_PRIMARY_UNIQUE_VODS,
            "unique_positive_clusters": _MIN_PRIMARY_POSITIVE_CLUSTERS,
            "ranking_groups": _MIN_PRIMARY_RANKING_GROUPS,
            "mixed_ranking_groups": _MIN_PRIMARY_MIXED_RANKING_GROUPS,
            "candidate_source_peak_share": _MAX_CANDIDATE_SOURCE_SHARE,
        },
        "targets": {
            "mixed_ranking_groups": _TARGET_PRIMARY_MIXED_RANKING_GROUPS,
            "negative_rows_per_positive_max": _TARGET_NEGATIVE_ROWS_PER_POSITIVE_MAX,
        },
        "vod_contrast_balance": contrast_balance,
        "candidate_source_mix": source_mix,
        "violations": violations,
    }


def _candidate_source_mix(rows: list[dict]) -> dict[str, object]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        source = (
            str(row.get("candidate_source") or features.get("source") or row.get("dataset_source") or "")
            .strip()
            .lower()
            or "unknown"
        )
        counts[source] += 1
    total = sum(counts.values()) or 1
    share_by_source = {
        source: round(count / total, 3)
        for source, count in sorted(counts.items())
    }
    peak_source = max(share_by_source, key=share_by_source.get) if share_by_source else ""
    return {
        "rows": total if rows else 0,
        "distinct_sources": len(counts),
        "peak_source": peak_source,
        "peak_share": float(share_by_source.get(peak_source, 0.0)) if peak_source else 0.0,
        "by_source": dict(sorted(counts.items())),
        "share_by_source": share_by_source,
    }


def _vod_contrast_balance(rows: list[dict]) -> dict[str, object]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(train_ranker._ranking_group_key(row, "vod_id") or "").strip()
        if key:
            groups[key].append(row)

    mixed_groups = 0
    worst_ratio = 0.0
    over_limit_groups: list[dict[str, object]] = []
    for key, group_rows in groups.items():
        positive_count = sum(1 for row in group_rows if int(row.get("label", 0)) == 1)
        negative_count = sum(1 for row in group_rows if int(row.get("label", 0)) == 0)
        if positive_count <= 0 or negative_count <= 0:
            continue
        mixed_groups += 1
        ratio = negative_count / positive_count
        worst_ratio = max(worst_ratio, ratio)
        if ratio > _TARGET_NEGATIVE_ROWS_PER_POSITIVE_MAX:
            over_limit_groups.append(
                {
                    "group": key,
                    "positive_rows": positive_count,
                    "negative_rows": negative_count,
                    "negative_rows_per_positive": round(ratio, 3),
                }
            )

    over_limit_groups.sort(
        key=lambda item: (
            -float(item["negative_rows_per_positive"]),
            str(item["group"]),
        )
    )
    return {
        "mixed_ranking_groups": mixed_groups,
        "target_mixed_ranking_groups": _TARGET_PRIMARY_MIXED_RANKING_GROUPS,
        "target_negative_rows_per_positive_max": _TARGET_NEGATIVE_ROWS_PER_POSITIVE_MAX,
        "worst_negative_rows_per_positive": round(worst_ratio, 3),
        "over_limit_group_count": len(over_limit_groups),
        "over_limit_groups": over_limit_groups[:10],
    }


def _train_per_streamer(
    labels: list[dict],
    short_form_rows: list[dict],
    *,
    artifact_dir: pathlib.Path,
    work_dir: pathlib.Path,
    short_form_weight: float,
    short_form_mode: str,
    gold_train_rows: list[dict] | None = None,
    gold_train_weight: float = 2.0,
    holdout_policy_rows: list[dict] | None = None,
    rollout_review_negative_weight: float = _DEFAULT_ROLLOUT_REVIEW_NEGATIVE_WEIGHT,
    hard_negative_rows: list[dict] | None = None,
    hard_negative_weight: float = _DEFAULT_HARD_NEGATIVE_WEIGHT,
    serving_hard_negative_rows: list[dict] | None = None,
    serving_hard_negative_weight: float = _DEFAULT_SERVING_HARD_NEGATIVE_WEIGHT,
    outcome_positive_rows: list[dict] | None = None,
    outcome_positive_weight: float = 2.0,
    auto_outcome_negative_rows: list[dict] | None = None,
    auto_outcome_hard_negative_weight: float = _DEFAULT_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT,
    auto_outcome_background_negative_weight: float = _DEFAULT_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT,
    cross_platform_rows: list[dict] | None = None,
    cross_platform_weight: float = _DEFAULT_CROSS_PLATFORM_WEIGHT,
    primary_label_policy: str = "all",
    mm_judge_rows: list[dict] | None = None,
    mm_judge_weight: float = 0.4,
) -> int:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        broadcaster = str(row.get("broadcaster") or "").strip().lower()
        if broadcaster:
            grouped[broadcaster].append(row)

    if not grouped:
        print("No broadcaster values found in Twitch labels.")
        return 1

    artifact_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for broadcaster, broadcaster_labels in sorted(grouped.items()):
        dataset_rows = build_composite_dataset(
            broadcaster_labels,
            profile=broadcaster,
            work_dir=work_dir / broadcaster,
            supplemental_rows=_filter_short_form_rows(short_form_rows, broadcaster),
            supplemental_weight=short_form_weight,
            short_form_mode=short_form_mode,
            gold_train_rows=_filter_gold_train_rows(gold_train_rows or [], broadcaster),
            gold_train_weight=gold_train_weight,
            holdout_policy_rows=_filter_gold_train_rows(holdout_policy_rows or [], broadcaster),
            rollout_review_negative_weight=rollout_review_negative_weight,
            hard_negative_rows=_filter_gold_train_rows(hard_negative_rows or [], broadcaster),
            hard_negative_weight=hard_negative_weight,
            serving_hard_negative_rows=_filter_gold_train_rows(serving_hard_negative_rows or [], broadcaster),
            serving_hard_negative_weight=serving_hard_negative_weight,
            outcome_positive_rows=_filter_gold_train_rows(outcome_positive_rows or [], broadcaster),
            outcome_positive_weight=outcome_positive_weight,
            auto_outcome_negative_rows=_filter_gold_train_rows(auto_outcome_negative_rows or [], broadcaster),
            auto_outcome_hard_negative_weight=auto_outcome_hard_negative_weight,
            auto_outcome_background_negative_weight=auto_outcome_background_negative_weight,
            cross_platform_rows=_filter_gold_train_rows(cross_platform_rows or [], broadcaster),
            cross_platform_weight=cross_platform_weight,
            primary_label_policy=primary_label_policy,
            mm_judge_rows=_filter_gold_train_rows(mm_judge_rows or [], broadcaster),
            mm_judge_weight=mm_judge_weight,
        )
        dataset_out = artifact_dir / f"{broadcaster}_vod_training_rows.jsonl"
        _write_jsonl(dataset_out, dataset_rows)
        print(
            json.dumps(
                {
                    "profile": broadcaster,
                    "dataset": str(dataset_out),
                    "summary": dataset_summary(dataset_rows),
                },
                ensure_ascii=True,
            )
        )
        code = _train_models_for_dataset(dataset_out, broadcaster, artifact_dir)
        if code != 0:
            failures += 1
    return 0 if failures == 0 else 1


def _train_models_for_dataset(
    dataset_path: pathlib.Path,
    profile: str,
    artifact_dir: pathlib.Path,
) -> int:
    evaluate_boundary_refinement, train_boundary_ranker = _boundary_training_modules()
    rows = _read_jsonl(dataset_path)
    quality = ranker_dataset_quality(rows)
    if not quality["ok"]:
        print(
            json.dumps(
                {
                    "profile": profile,
                    "ranker": {
                        "trained": False,
                        "reason": "insufficient_dataset_quality",
                        "quality": quality,
                    },
                },
                ensure_ascii=True,
            )
        )
        return 1

    audit = analyze_ranker_feature_confounds.analyze_feature_correlation_audit(
        dataset_path=dataset_path,
        fail_threshold=0.85,
    )
    audit_path = artifact_dir / f"{profile}_feature_correlation_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8")
    if audit.get("failed"):
        print(
            json.dumps(
                {
                    "profile": profile,
                    "ranker": {
                        "trained": False,
                        "reason": "feature_correlation_audit_failed",
                        "audit": str(audit_path),
                    },
                },
                ensure_ascii=True,
            )
        )
        return 1

    code = train_ranker.train_from_dataset(dataset_path, profile, artifact_dir)
    if code != 0:
        return code

    boundary_rows = [row for row in rows if evaluate_boundary_refinement._has_boundary_reference(row)]
    if not boundary_rows:
        print(
            json.dumps(
                {
                    "profile": profile,
                    "boundary_ranker": {
                        "trained": False,
                        "reason": "no_boundary_reference_rows",
                    },
                },
                ensure_ascii=True,
            )
        )
        return 0

    print(
        json.dumps(
            {
                "profile": profile,
                "boundary_dataset": {
                    "rows": len(rows),
                    "boundary_rows": len(boundary_rows),
                    "metrics": evaluate_boundary_refinement._boundary_metrics(boundary_rows),
                },
            },
            ensure_ascii=True,
        )
    )
    return train_boundary_ranker.train_from_rows(rows, profile, artifact_dir)


def _boundary_training_modules():
    from ml import evaluate_boundary_refinement, train_boundary_ranker

    return evaluate_boundary_refinement, train_boundary_ranker


def _prepare_short_form_rows(
    rows: list[dict],
    profile: str,
    supplemental_weight: float,
) -> list[dict]:
    weight = max(0.0, min(float(supplemental_weight), 1.0))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", 0)) not in {0, 1}:
            continue
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        copied["dataset_source"] = "short_form_supplemental"
        if not copied.get("label_class"):
            copied["label_class"] = "short_form_positive" if int(copied["label"]) == 1 else "short_form_negative"
        copied["supervision_tier"] = str(copied.get("supervision_tier") or "short_form_discovered")
        copied["supervision_reason"] = str(copied.get("supervision_reason") or "manifest_label")
        copied["ordinal_label"] = int(copied.get("ordinal_label", 3 if int(copied["label"]) == 1 else 0))
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        creator = str(copied.get("creator_id") or copied.get("streamer") or "").strip().lower()
        copied["profile_hint"] = creator or profile
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        prepared.append(copied)
    return prepared


def _prepare_gold_train_rows(
    rows: list[dict],
    profile: str,
    gold_train_weight: float,
    *,
    rollout_review_negative_weight: float = _DEFAULT_ROLLOUT_REVIEW_NEGATIVE_WEIGHT,
) -> list[dict]:
    weight = max(0.0, float(gold_train_weight))
    rollout_negative_weight = max(0.0, float(rollout_review_negative_weight))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", -1)) not in {0, 1}:
            continue
        if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() != "train":
            continue
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        rollout_review_negative = _is_rollout_review_negative(row)
        copied["dataset_source"] = (
            "gold_eval_rollout_review_hard_negative_train"
            if rollout_review_negative
            else "gold_eval_train"
        )
        copied["label_source"] = "gold_eval"
        copied["supervision_tier"] = "gold_eval"
        copied["supervision_reason"] = (
            "manual_rollout_auto_accept_false_positive"
            if rollout_review_negative
            else str(copied.get("supervision_reason") or "manual_gold_eval_train")
        )
        if rollout_review_negative:
            copied["label_class"] = "gold_rollout_review_hard_negative"
        elif not copied.get("label_class"):
            copied["label_class"] = "gold_positive" if int(copied["label"]) == 1 else "gold_negative"
        copied["ordinal_label"] = int(copied.get("ordinal_label", copied.get("gold_grade", 3 if int(copied["label"]) == 1 else 0)))
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        multiplier = rollout_negative_weight if rollout_review_negative else weight
        copied["sample_weight"] = round(base_weight * multiplier, 3)
        creator = str(copied.get("creator_id") or copied.get("streamer") or "").strip().lower()
        copied["profile_hint"] = creator or profile
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        prepared.append(copied)
    return prepared


def _prepare_outcome_positive_train_rows(
    rows: list[dict],
    profile: str,
    outcome_positive_weight: float,
) -> list[dict]:
    weight = max(0.0, float(outcome_positive_weight))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", -1)) != 1:
            continue
        if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() != "train":
            continue
        source = str(row.get("dataset_source") or "").strip().lower()
        tier = str(row.get("supervision_tier") or "").strip().lower()
        label_source = str(row.get("label_source") or "").strip().lower()
        if (
            source != "cross_platform_outcome_train"
            and label_source != "cross_platform_audio_fingerprint"
            and label_source != "cross_platform_transcript_match"
            and tier != "cross_platform_outcome"
        ):
            continue
        preserved_label_source = str(
            row.get("label_source")
            or row.get("original_label_source")
            or row.get("dataset_source")
            or "cross_platform_transcript_match"
        )
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        copied["label"] = 1
        copied["dataset_source"] = "cross_platform_outcome_train"
        copied["label_source"] = preserved_label_source
        copied["supervision_tier"] = "cross_platform_outcome"
        copied["supervision_reason"] = str(
            copied.get("supervision_reason") or "cross_platform_shortform_train_positive"
        )
        copied["label_class"] = "cross_platform_outcome_positive"
        copied.pop("weak_positive_reason", None)
        copied.pop("top_viewed_weak_positive", None)
        if "creator_baseline_views" not in copied:
            copied["creator_baseline_views"] = float(copied.get("baseline_view_count") or 0.0)
        if "exposure_corrected_view_ratio" not in copied:
            baseline = _as_float(copied.get("creator_baseline_views")) or 0.0
            views = _as_float(copied.get("view_count")) or 0.0
            copied["exposure_corrected_view_ratio"] = round(views / baseline, 4) if baseline > 0 else 0.0
        copied["ordinal_label"] = int(copied.get("ordinal_label", 3) or 3)
        copied[GOLD_HOLDOUT_SPLIT_FIELD] = "train"
        copied = outcome_view_weighting.annotate_cross_platform_view_weight(copied)
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        creator = str(copied.get("creator_id") or copied.get("streamer") or "").strip().lower()
        copied["profile_hint"] = creator or profile
        if not copied.get("candidate_source"):
            copied["candidate_source"] = str(copied["features"].get("source") or "cross_platform_outcome")
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        copied["feature_provenance"]["cross_platform_outcome"] = True
        prepared.append(copied)
    return prepared


def _prepare_mm_judge_rows(
    rows: list[dict],
    profile: str,
    mm_judge_weight: float,
) -> list[dict]:
    weight = max(0.0, float(mm_judge_weight))
    prepared: list[dict] = []
    for row in rows:
        try:
            label = int(row.get("label", -1))
        except (TypeError, ValueError):
            continue
        if label not in {0, 1}:
            continue
        features = dict(row.get("features") or {})
        if not features:
            continue
        copied = dict(row)
        copied["features"] = features
        copied["label"] = label
        copied["selected"] = label == 1
        copied["dataset_source"] = "mm_judge_proxy_train"
        copied["label_source"] = "mm_judge_proxy"
        copied["supervision_tier"] = "mm_judge_proxy"
        copied["supervision_reason"] = str(copied.get("supervision_reason") or "multimodal_judge_proxy")
        copied["label_class"] = "mm_judge_positive" if label == 1 else "mm_judge_negative"
        copied["ordinal_label"] = 3 if label == 1 else 0
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        copied["profile_hint"] = str(copied.get("profile") or copied.get("profile_hint") or profile)
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        copied["feature_provenance"]["mm_judge_proxy"] = True
        prepared.append(copied)
    return prepared


def _apply_mm_judge_feature_replay(rows: list[dict], mm_judge_rows: list[dict]) -> dict[str, object]:
    snapshots = _mm_judge_feature_snapshots(mm_judge_rows)
    applied_rows = 0
    for row in rows:
        candidate_id = str(row.get("candidate_id") or row.get("queue_id") or "").strip()
        if not candidate_id:
            continue
        snapshot = snapshots.get(candidate_id)
        if not snapshot:
            continue
        features = dict(row.get("features") or {})
        features.update(snapshot)
        row["features"] = features
        provenance = dict(row.get("feature_provenance") or {})
        provenance["mm_judge_feature_replay"] = True
        row["feature_provenance"] = provenance
        applied_rows += 1
    return {
        "snapshot_rows": len(mm_judge_rows),
        "replayable_snapshots": len(snapshots),
        "applied_rows": applied_rows,
        "feature_names": list(_JUDGE_TEACHER_FEATURE_NAMES),
    }


def _mm_judge_feature_snapshots(rows: list[dict]) -> dict[str, dict[str, float | bool]]:
    snapshots: dict[str, dict[str, float | bool]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or row.get("queue_id") or "").strip()
        if not candidate_id:
            continue
        features = dict(row.get("features") or {})
        snapshot: dict[str, float | bool] = {}
        for name in _JUDGE_TEACHER_FEATURE_NAMES:
            if name in {"has_judge_score", "judge_llm_used"}:
                if name in features:
                    snapshot[name] = bool(features.get(name))
                elif name == "has_judge_score" and row.get("judge_score") is not None:
                    snapshot[name] = True
                elif name == "judge_llm_used" and row.get("judge_llm_used") is not None:
                    snapshot[name] = bool(row.get("judge_llm_used"))
                continue
            value = features.get(name, row.get(name))
            if value is None:
                continue
            snapshot[name] = round(_as_float(value), 6)
        if "judge_score" not in snapshot:
            continue
        snapshot.setdefault("has_judge_score", True)
        snapshot.setdefault("judge_llm_used", bool(row.get("judge_llm_used")))
        snapshots[candidate_id] = snapshot
    return snapshots


def _is_rollout_review_negative(row: dict) -> bool:
    if int(row.get("label", -1)) != 0:
        return False
    dataset_source = str(row.get("dataset_source") or "").strip()
    supervision_reason = str(row.get("supervision_reason") or "").strip()
    rollout_bucket = str(row.get("rollout_bucket") or "").strip()
    return (
        dataset_source == "gold_eval_rollout_review"
        or supervision_reason == "manual_rollout_auto_accept_review"
        or rollout_bucket == "auto_accept_candidate"
    )


def _prepare_hard_negative_rows(
    rows: list[dict],
    profile: str,
    hard_negative_weight: float,
) -> list[dict]:
    weight = max(0.0, float(hard_negative_weight))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", -1)) != 0:
            continue
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        copied["label"] = 0
        copied["dataset_source"] = "gold_eval_hard_negative_train"
        copied["label_source"] = "gold_eval"
        copied["supervision_tier"] = "gold_eval"
        copied["supervision_reason"] = str(copied.get("supervision_reason") or "ranker_diagnosis_false_positive")
        copied["label_class"] = "gold_hard_negative"
        copied["ordinal_label"] = int(copied.get("ordinal_label", copied.get("gold_grade", 0)) or 0)
        copied[GOLD_HOLDOUT_SPLIT_FIELD] = "train"
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        creator = str(copied.get("creator_id") or copied.get("streamer") or "").strip().lower()
        copied["profile_hint"] = creator or profile
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        prepared.append(copied)
    return prepared


def _prepare_serving_hard_negative_rows(
    rows: list[dict],
    profile: str,
    hard_negative_weight: float,
) -> list[dict]:
    weight = max(0.0, float(hard_negative_weight))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", -1)) != 0:
            continue
        if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() not in {"", "train"}:
            continue
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        if not copied["features"]:
            continue
        copied["label"] = 0
        copied["selected"] = False
        copied["dataset_source"] = "serving_trace_hard_negative_train"
        copied["label_source"] = "serving_trace"
        copied["supervision_tier"] = "serving_hard_negative"
        copied["supervision_reason"] = str(
            copied.get("supervision_reason")
            or "high_ranked_serving_candidate_without_cross_platform_outcome"
        )
        copied["label_class"] = "serving_trace_hard_negative"
        copied["ordinal_label"] = 0
        copied[GOLD_HOLDOUT_SPLIT_FIELD] = "train"
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        creator = str(
            copied.get("creator_id")
            or copied.get("streamer")
            or copied.get("profile_hint")
            or ""
        ).strip().lower()
        copied["profile_hint"] = creator or profile
        if not copied.get("candidate_source"):
            copied["candidate_source"] = str(copied["features"].get("source") or "serving_trace")
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        copied["feature_provenance"]["serving_trace_hard_negative"] = True
        prepared.append(copied)
    return prepared


def _prepare_auto_outcome_negative_rows(
    rows: list[dict],
    profile: str,
    *,
    hard_negative_weight: float,
    background_negative_weight: float,
) -> list[dict]:
    hard_weight = max(0.0, float(hard_negative_weight))
    background_weight = max(0.0, float(background_negative_weight))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", -1)) != 0:
            continue
        if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() != "train":
            continue
        copied = dict(row)
        copied["features"] = dict(row.get("features") or {})
        copied["label"] = 0
        copied["dataset_source"] = "auto_outcome_negative_train"
        copied["label_source"] = "auto_outcome_negative"
        copied["supervision_tier"] = "auto_outcome_negative"
        copied["supervision_reason"] = str(
            copied.get("supervision_reason") or "candidate_without_positive_short_form_outcome"
        )
        tier = str(copied.get("auto_outcome_negative_tier") or "").strip().lower()
        if tier == "hard_negative":
            weight = hard_weight
            copied["label_class"] = "auto_outcome_hard_negative"
        else:
            weight = background_weight
            copied["auto_outcome_negative_tier"] = tier or "background_negative"
            copied["label_class"] = "auto_outcome_background_negative"
        copied["ordinal_label"] = int(copied.get("ordinal_label", 0) or 0)
        copied[GOLD_HOLDOUT_SPLIT_FIELD] = "train"
        base_weight = _as_float(copied.get("sample_weight")) or 1.0
        copied["sample_weight"] = round(base_weight * weight, 3)
        creator = str(copied.get("creator_id") or copied.get("streamer") or copied.get("profile_hint") or "").strip().lower()
        copied["profile_hint"] = creator or profile
        if not copied.get("candidate_source"):
            copied["candidate_source"] = str(copied["features"].get("source") or "auto_outcome_negative")
        copied["feature_provenance"] = dict(copied.get("feature_provenance") or {})
        copied["feature_provenance"]["auto_outcome_negative"] = True
        prepared.append(copied)
    return prepared


def _prepare_cross_platform_vod_labels(
    rows: list[dict],
    profile: str,
    cross_platform_weight: float,
) -> list[dict]:
    weight = max(0.0, float(cross_platform_weight))
    prepared: list[dict] = []
    for row in rows:
        if int(row.get("label", -1)) != 1:
            continue
        source = str(row.get("dataset_source") or "").strip().lower()
        tier = str(row.get("supervision_tier") or "").strip().lower()
        label_source = str(row.get("label_source") or "").strip().lower()
        if (
            source not in {"cross_platform_audio_fingerprint", "cross_platform_transcript_match"}
            and label_source not in {"cross_platform_audio_fingerprint", "cross_platform_transcript_match"}
            and tier != "cross_platform_outcome"
        ):
            continue
        vod_id = str(row.get("vod_id") or "").strip()
        vod_url = str(row.get("vod_url") or row.get("source_vod_url") or "").strip()
        if not vod_id or not vod_url:
            continue
        window = _row_window(row)
        if window is None:
            continue
        start, end = window
        copied = dict(row)
        copied["vod_id"] = vod_id
        copied["vod_url"] = vod_url
        copied["window_start"] = round(start, 2)
        copied["window_end"] = round(max(end, start + 1.0), 2)
        copied["broadcaster"] = str(copied.get("broadcaster") or copied.get("creator_id") or copied.get("streamer") or profile)
        copied["label_class"] = "cross_platform_outcome_positive"
        preserved_label_source = str(copied.get("label_source") or copied.get("dataset_source") or "cross_platform_transcript_match")
        copied["label_source"] = preserved_label_source
        copied["platform_source"] = str(copied.get("platform_source") or "cross_platform_shortform")
        copied["quality_bucket"] = str(copied.get("quality_bucket") or "cross_platform_outcome")
        copied["label_score"] = max(
            _as_float(copied.get("transcript_similarity")),
            _as_float(copied.get("fingerprint_confidence")),
            _as_float(copied.get("label_score")),
        )
        copied["clip_ids"] = list(copied.get("clip_ids") or [str(copied.get("candidate_id") or "")])
        copied["training_dataset_source"] = (
            "cross_platform_transcript_match_train"
            if preserved_label_source == "cross_platform_transcript_match"
            else "cross_platform_audio_fingerprint_train"
        )
        copied["training_supervision_tier"] = "cross_platform_outcome"
        copied["training_supervision_reason"] = (
            ":transcript_matched_shortform_outcome"
            if preserved_label_source == "cross_platform_transcript_match"
            else ":audio_fingerprint_matched_shortform_outcome"
        )
        copied = outcome_view_weighting.annotate_cross_platform_view_weight(copied)
        copied["training_sample_weight_multiplier"] = weight
        copied["cross_platform_recall_target"] = True
        prepared.append(copied)
    return prepared


def _merge_gold_train_rows(base_rows: list[dict], gold_rows: list[dict]) -> list[dict]:
    if not gold_rows:
        return list(base_rows)
    merged: list[dict] = []
    index_by_key: dict[tuple[str, ...], int] = {}
    for row in base_rows:
        key = _training_row_key(row)
        if key:
            index_by_key[key] = len(merged)
        merged.append(row)
    for row in gold_rows:
        key = _training_row_key(row)
        if key and key in index_by_key:
            merged[index_by_key[key]] = row
            continue
        if key:
            index_by_key[key] = len(merged)
        merged.append(row)
    return merged


def _filter_short_form_rows(rows: list[dict], broadcaster: str) -> list[dict]:
    if not rows:
        return []
    normalized = broadcaster.strip().lower()
    filtered = [
        row for row in rows
        if str(row.get("creator_id") or row.get("streamer") or "").strip().lower() == normalized
    ]
    return filtered


def _read_targeted_backfill_labels(paths: list[pathlib.Path], profile: str) -> list[dict]:
    prepared: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for row in _read_jsonl(path):
            label = _prepare_targeted_backfill_label(row, profile)
            if not label:
                continue
            key = primary_identity_key(label)
            if key in seen:
                continue
            seen.add(key)
            prepared.append(label)
    return prepared


def _prepare_targeted_backfill_label(row: dict, profile: str) -> dict | None:
    label_class = str(row.get("label_class") or "").strip()
    if label_class and label_class not in POSITIVE_LABEL_CLASSES:
        return None
    try:
        numeric_label = int(row.get("label", 1))
    except (TypeError, ValueError):
        numeric_label = 1
    if numeric_label != 1:
        return None

    vod_id = str(row.get("vod_id") or "").strip() or _row_vod_id(row)
    if not vod_id:
        return None
    window = _row_window(row)
    if window is None:
        return None
    start, end = window
    vod_url = str(row.get("vod_url") or row.get("source_url") or "").strip()
    if not vod_url or "twitch.tv/videos/" not in vod_url:
        vod_url = f"https://www.twitch.tv/videos/{vod_id}"

    copied = dict(row)
    copied["vod_id"] = vod_id
    copied["vod_url"] = vod_url
    copied["window_start"] = round(start, 2)
    copied["window_end"] = round(max(end, start + 1.0), 2)
    copied["label"] = 1
    copied["label_class"] = label_class or "positive"
    copied["broadcaster"] = str(copied.get("broadcaster") or copied.get("creator_id") or copied.get("streamer") or profile)
    copied["manual_gold_review"] = True
    copied["manual_gold_review_source"] = str(copied.get("manual_gold_review_source") or "targeted_backfill")
    copied["gold_candidate_status"] = "needs_feature_build"
    copied["label_source"] = str(copied.get("label_source") or "targeted_backfill")
    copied["platform_source"] = str(copied.get("platform_source") or "twitch_clip")
    copied["quality_bucket"] = str(copied.get("quality_bucket") or "targeted_outcome")
    copied["targeted_feature_backfill"] = True
    copied["targeted_feature_backfill_source"] = str(
        copied.get("targeted_feature_backfill_source") or "rollout_readiness_collection"
    )
    copied["training_dataset_source"] = "targeted_feature_backfill"
    copied["training_supervision_tier"] = "feature_backfill_only"
    copied["training_supervision_reason"] = ":feature_cache_backfill_only"
    if not copied.get("clip_ids"):
        fallback_clip_id = str(copied.get("clip_id") or copied.get("candidate_id") or primary_identity_key(copied)).strip()
        copied["clip_ids"] = [fallback_clip_id] if fallback_clip_id else []
    return copied


def _merge_label_rows(labels: list[dict], extra_labels: list[dict]) -> list[dict]:
    merged = list(labels)
    seen_keys: set[str] = set()
    for row in labels:
        keys = row_identity_keys(row)
        if keys:
            seen_keys.update(keys)
        else:
            seen_keys.add(primary_identity_key(row))
    for row in extra_labels:
        keys = row_identity_keys(row)
        candidate_keys = keys or {primary_identity_key(row)}
        if candidate_keys & seen_keys:
            continue
        seen_keys.update(candidate_keys)
        merged.append(row)
    return merged


def _filter_gold_train_rows(rows: list[dict], broadcaster: str) -> list[dict]:
    if not rows:
        return []
    normalized = broadcaster.strip().lower()
    return [
        row
        for row in rows
        if str(row.get("creator_id") or row.get("streamer") or row.get("profile_hint") or "").strip().lower()
        in {"", normalized}
    ]


def _training_row_key(row: dict) -> tuple[str, ...] | None:
    queue_id = str(row.get("queue_id") or "").strip()
    if queue_id:
        return ("queue_id", queue_id)

    job_id = str(row.get("job_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or "").strip()
    if job_id and candidate_id:
        return ("job_candidate", job_id, candidate_id)

    vod_id = str(row.get("vod_id") or "").strip()
    window_start = _stable_time_value(row.get("window_start"))
    window_end = _stable_time_value(row.get("window_end"))
    if vod_id and candidate_id:
        return ("vod_candidate_window", vod_id, candidate_id, window_start, window_end)

    source_url = str(row.get("source_url") or "").strip()
    if source_url and candidate_id:
        return ("source_candidate_window", source_url, candidate_id, window_start, window_end)
    return None


def _row_vod_id(row: dict) -> str:
    explicit = str(row.get("vod_id") or "").strip()
    if explicit:
        return explicit
    for key in ("source_url", "review_url", "url"):
        match = _TWITCH_VOD_ID_PATTERN.search(str(row.get(key) or ""))
        if match:
            return match.group(1)
    return ""


def _row_source_key(row: dict) -> str:
    for key in ("source_url", "review_url", "url"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _row_window(row: dict) -> tuple[float, float] | None:
    start = _nullable_float(row.get("window_start"))
    end = _nullable_float(row.get("window_end"))
    if start is None:
        start = _nullable_float(row.get("final_start"))
    if end is None:
        end = _nullable_float(row.get("final_end"))
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


def _windows_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    if overlap <= 0.0:
        return False
    if overlap >= 5.0:
        return True
    left_duration = max(left[1] - left[0], 1e-9)
    right_duration = max(right[1] - right[0], 1e-9)
    return overlap / min(left_duration, right_duration) >= 0.30


def _stable_time_value(value) -> str:
    try:
        return f"{float(value or 0.0):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _normalize_short_form_mode(mode: str) -> str:
    normalized = str(mode or "off").strip().lower()
    if normalized not in _SHORT_FORM_MODES:
        raise ValueError(f"Unsupported short-form mode: {mode!r}")
    return normalized


def _mixed_ranking_groups(rows: list[dict], split_unit: str) -> set[str]:
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        key = train_ranker._ranking_group_key(row, split_unit)
        if not str(key).strip():
            continue
        labels_by_group[key].add(int(row.get("label", 0)))
    return {
        group
        for group, labels in labels_by_group.items()
        if 0 in labels and 1 in labels
    }


def _positive_cluster_key(row: dict) -> tuple[str, str] | None:
    if int(row.get("label", 0)) != 1:
        return None
    vod_id = str(row.get("vod_id") or row.get("source_url") or "").strip()
    if not vod_id:
        return None
    clip_ids = sorted(str(item) for item in (row.get("clip_ids") or []) if str(item).strip())
    if clip_ids:
        return vod_id, ",".join(clip_ids)
    candidate_id = str(row.get("candidate_id") or "").strip()
    if not candidate_id:
        return None
    return vod_id, f"candidate:{candidate_id}"


def _read_jsonl(path: pathlib.Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _targeted_backfill_input_summary(paths: list[pathlib.Path], rows: list[dict]) -> dict[str, object]:
    return {
        "paths": [str(path) for path in paths],
        "existing_paths": [str(path) for path in paths if path.exists()],
        "rows": len(rows),
        "positive_rows": sum(1 for row in rows if int(row.get("label", 0)) == 1),
        "distinct_vod_groups": len({str(row.get("vod_id") or "").strip() for row in rows if str(row.get("vod_id") or "").strip()}),
        "gold_candidate_statuses": dict(
            sorted(Counter(str(row.get("gold_candidate_status") or "unknown") for row in rows).items())
        ),
        "sources": dict(
            sorted(Counter(str(row.get("targeted_feature_backfill_source") or "unknown") for row in rows).items())
        ),
    }


def _mm_judge_input_summary(path: pathlib.Path | None, rows: list[dict]) -> dict[str, object]:
    return {
        "path": str(path or ""),
        "exists": bool(path and path.exists()),
        "rows": len(rows),
        "positive_rows": sum(1 for row in rows if _optional_int(row.get("label"), -1) == 1),
        "negative_rows": sum(1 for row in rows if _optional_int(row.get("label"), -1) == 0),
        "with_features": sum(1 for row in rows if isinstance(row.get("features"), dict) and bool(row.get("features"))),
    }


def _cross_platform_input_summary(path: pathlib.Path | None, rows: list[dict]) -> dict[str, object]:
    summary = {
        "path": str(path or ""),
        "exists": bool(path and path.exists()),
        "rows": len(rows),
        "positive_rows": sum(1 for row in rows if int(row.get("label", 0)) == 1),
        "dataset_sources": dict(sorted(Counter(str(row.get("dataset_source") or "unknown") for row in rows).items())),
    }
    prepare_summary_path = path.with_name("jynxzi_cross_platform_fingerprint_summary.json") if path else None
    prepare_summary = _read_json(prepare_summary_path)
    if prepare_summary:
        summary["prepare_summary_path"] = str(prepare_summary_path)
        summary["prepare_status"] = str(prepare_summary.get("status") or "")
        summary["prepare_vod_rows"] = int(prepare_summary.get("vod_rows") or 0)
        summary["prepare_repost_rows"] = int(prepare_summary.get("repost_rows") or 0)
        summary["prepare_matched_rows"] = int(prepare_summary.get("matched_rows") or 0)
        summary["prepare_diagnostic_rows"] = int(prepare_summary.get("diagnostic_rows") or 0)
        summary["prepare_preserved_existing_output"] = bool(prepare_summary.get("preserved_existing_output"))
        match_diagnostics = prepare_summary.get("match_diagnostics")
        if isinstance(match_diagnostics, dict):
            recommendation = match_diagnostics.get("recommendation")
            distribution = match_diagnostics.get("confidence_distribution")
            ladder = match_diagnostics.get("threshold_ladder")
            if isinstance(recommendation, dict):
                summary["prepare_match_recommendation"] = dict(recommendation)
            if isinstance(distribution, dict):
                summary["prepare_confidence_distribution"] = dict(distribution)
            if isinstance(ladder, dict):
                summary["prepare_threshold_ladder"] = dict(ladder)
        if prepare_summary.get("missing_inputs"):
            summary["prepare_missing_inputs"] = list(prepare_summary.get("missing_inputs") or [])
        if prepare_summary.get("error"):
            summary["prepare_error"] = str(prepare_summary.get("error") or "")
    return summary


def _read_json(path: pathlib.Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows), encoding="utf-8")
    tmp_path.replace(path)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _nullable_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
