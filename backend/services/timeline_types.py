from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
class CandidateFeatures:
    candidate_id: str
    duration: float = 0.0
    hook_delay_sec: float = 0.0
    hook_strength: float = 0.0
    max_hype: float = 0.0
    mean_hype: float = 0.0
    arc_score: float = 0.0
    chat_peak: float = 0.0
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
    has_llm_semantics: bool = False
    has_chat_features: bool = False
    has_vector_similarity: bool = False
    feature_build_version: int = 1
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RankedCandidate:
    candidate: Candidate
    features: CandidateFeatures | None
    rank_score: float
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "features": self.features.to_dict() if self.features is not None else None,
            "rank_score": self.rank_score,
            "rationale": list(self.rationale),
        }
