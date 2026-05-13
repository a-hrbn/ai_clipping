from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Candidate:
    id: str
    start: float
    end: float
    source: str
    anchor_time: float
    score_hint: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chapter:
    id: str
    start: float
    end: float
    index: int
    label: str = "mixed"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemorySpan:
    id: str
    kind: str
    start: float
    end: float
    confidence: float = 0.0
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    related_span_ids: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentScore:
    segment_index: int
    start: float
    end: float
    virality_score: float
    hook_score: float = 0.0
    payoff_score: float = 0.0
    audience_score: float = 0.0
    semantic_score: float = 0.0
    smoothed_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectedPeak:
    peak_time: float
    peak_score: float
    left_base_time: float
    right_base_time: float
    prominence: float
    window_start: float
    window_end: float
    segment_index: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BoundaryPoint:
    id: str
    time: float
    kind: str
    segment_index: int
    speaker: str = "UNKNOWN"
    text_left: str = ""
    text_right: str = ""
    is_sentence_end: bool = False
    is_sentence_start: bool = False
    is_speaker_turn: bool = False
    is_scene_cut: bool = False
    is_pause_break: bool = False
    pause_before_sec: float = 0.0
    pause_after_sec: float = 0.0
    chat_delta: float = 0.0
    hype_delta: float = 0.0
    visual_delta: float = 0.0
    face_reaction_delta: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BoundaryFeatures:
    boundary_id: str
    candidate_id: str
    role: str
    anchor_distance_sec: float = 0.0
    candidate_left_distance_sec: float = 0.0
    candidate_right_distance_sec: float = 0.0
    clean_open_score: float = 0.0
    closure_score: float = 0.0
    mid_thought_risk: float = 0.0
    context_dependency_risk: float = 0.0
    hook_proximity_score: float = 0.0
    payoff_capture_score: float = 0.0
    speaker_transition_score: float = 0.0
    scene_transition_score: float = 0.0
    pause_strength_score: float = 0.0
    lexical_break_score: float = 0.0
    reaction_tail_score: float = 0.0
    duration_prior_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BoundaryPair:
    id: str
    candidate_id: str
    start_boundary_id: str
    end_boundary_id: str
    start_time: float
    end_time: float
    duration: float
    pair_score: float
    start_score: float
    end_score: float
    hook_score: float
    payoff_score: float
    narrative_score: float
    penalties: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidateFeatures:
    candidate_id: str
    duration: float = 0.0
    hook_delay_sec: float = 0.0
    hook_strength: float = 0.0
    max_hype: float = 0.0
    mean_hype: float = 0.0
    arc_score: float = 0.0
    chat_peak: float = 0.0
    chat_mean: float = 0.0
    chat_early_peak: float = 0.0
    chat_post_hook_peak: float = 0.0
    chat_delta: float = 0.0
    chat_spike_coverage: float = 0.0
    chat_activity_coverage: float = 0.0
    chat_message_rate_peak: float = 0.0
    chat_message_rate_mean: float = 0.0
    chat_density_peak: float = 0.0
    chat_density_mean: float = 0.0
    chat_hype_score: float = 0.0
    chat_laugh_score: float = 0.0
    chat_shock_score: float = 0.0
    chat_sadge_score: float = 0.0
    chat_velocity_score: float = 0.0
    chat_rate_derivative_score: float = 0.0
    chat_ema_crossing_score: float = 0.0
    motion_peak: float = 0.0
    face_reaction_peak: float = 0.0
    speaker_turns: int = 0
    exclamation_count: int = 0
    laughter_count: int = 0
    question_open: bool = False
    context_cost: float = 0.0
    payoff_score: float = 0.0
    context_efficiency: float = 0.0
    quoteability_score: float = 0.0
    surprise_score: float = 0.0
    contradiction_score: float = 0.0
    standalone_score: float = 0.0
    semantic_density: float = 0.0
    payoff_clarity_score: float = 0.0
    slow_open_penalty: float = 0.0
    context_dependency_score: float = 0.0
    weak_resolution_penalty: float = 0.0
    profile_fit_score: float = 0.0
    pre_hook_hype: float = 0.0
    post_hook_hype: float = 0.0
    hype_delta: float = 0.0
    local_peak_density: float = 0.0
    winner_similarity_max: float = 0.0
    winner_similarity_mean: float = 0.0
    loser_similarity_max: float = 0.0
    similarity_margin: float = 0.0
    neighbor_quality_score: float = 0.0
    vod_progress: float = 0.0
    boundary_left_context_sec: float = 0.0
    boundary_right_context_sec: float = 0.0
    neighbor_positive_density: float = 0.0
    clip_overlap_count: float = 0.0
    chapter_position: float = 0.0
    topic_shift_strength: float = 0.0
    chapter_local_density: float = 0.0
    chapter_boundary_distance: float = 0.0
    chapter_coherence_score: float = 0.0
    setup_depth_score: float = 0.0
    callback_strength: float = 0.0
    arc_payoff_match: float = 0.0
    topic_continuity_score: float = 0.0
    speaker_context_clarity: float = 0.0
    memory_dependency_score: float = 0.0
    story_stage_score: float = 0.0
    required_setup_lead_sec: float = 0.0
    retrieved_memory_span_count: float = 0.0
    trend_hook_score: float = 0.0
    trend_topic_score: float = 0.0
    trend_prior_score: float = 0.0
    dynamic_trend_score: float = 0.0
    outcome_prior_score: float = 0.0
    visual_hook_score: float = 0.0
    audio_pop_score: float = 0.0
    streamer_shout_score: float = 0.0
    voice_energy_burst_score: float = 0.0
    pitch_variation_score: float = 0.0
    f0_sigma_score: float = 0.0
    vad_voice_ratio: float = 0.0
    shout_onset_score: float = 0.0
    event_surprise_score: float = 0.0
    nonverbal_payoff_score: float = 0.0
    goal_relevance_score: float = 0.0
    platform_relevance_score: float = 0.0
    projected_hook_score: float = 0.0
    projected_retention_score: float = 0.0
    vl_prompt_max_score: float = 0.0
    vl_clutch_score: float = 0.0
    vl_comedy_score: float = 0.0
    vl_reaction_face_score: float = 0.0
    vl_scoreboard_score: float = 0.0
    game_event_score: float = 0.0
    kill_density_in_window: float = 0.0
    is_clutch_window: bool = False
    is_match_endpoint: bool = False
    boundary_variant_score: float = 0.0
    boundary_variant_margin_score: float = 0.0
    boundary_variant_count: float = 0.0
    hook_rewrite_score: float = 0.0
    hook_curiosity_gap_score: float = 0.0
    has_hook_rewrite: bool = False
    pre_roll_hook_score: float = 1.0
    judge_score: float = 0.0
    judge_confidence: float = 0.0
    judge_scroll_stop_power: float = 0.0
    judge_standalone_clarity: float = 0.0
    judge_payoff_strength: float = 0.0
    judge_narrative_completeness: float = 0.0
    judge_platform_fit_shortform: float = 0.0
    has_judge_score: bool = False
    judge_llm_used: bool = False
    has_llm_semantics: bool = False
    has_chat_features: bool = False
    has_vector_similarity: bool = False
    feature_build_version: int = 1
    source: str = ""
    # Rule-based heuristic score used as an input feature for trained rankers.
    # Populated at scoring time (services/ranker.py) from _heuristic_scores so the
    # trained model can learn how to adjust it rather than being blended post-hoc.
    heuristic_score: float = 0.0

    @classmethod
    def create(cls, **kwargs) -> "CandidateFeatures":
        known_names = {item.name for item in fields(cls)}
        accepted = {key: value for key, value in kwargs.items() if key in known_names}
        extra = {key: value for key, value in kwargs.items() if key not in known_names}
        instance = cls(**accepted)
        for key, value in extra.items():
            setattr(instance, key, value)
        return instance

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateFeatures":
        return cls.create(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RankedCandidate:
    candidate: Candidate
    features: CandidateFeatures | None
    rank_score: float
    rationale: list[str] = field(default_factory=list)
    base_rank_score: float | None = None
    heuristic_rank_score: float | None = None
    trained_rank_score: float | None = None
    model_blend: float | None = None
    trained_model_used: bool = False
    judge_result: "JudgeResult | None" = None

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "features": self.features.to_dict() if self.features is not None else None,
            "rank_score": self.rank_score,
            "rationale": list(self.rationale),
            "base_rank_score": self.base_rank_score,
            "heuristic_rank_score": self.heuristic_rank_score,
            "trained_rank_score": self.trained_rank_score,
            "model_blend": self.model_blend,
            "trained_model_used": self.trained_model_used,
            "judge_result": self.judge_result.to_dict() if self.judge_result is not None else None,
        }


@dataclass
class JudgeResult:
    judge_score: float
    scroll_stop_power: float
    standalone_clarity: float
    payoff_strength: float
    narrative_completeness: float
    platform_fit_shortform: float
    judge_reason: str
    judge_confidence: float
    provider: str = "heuristic"
    llm_used: bool = False
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
