from __future__ import annotations

from services import boundary_candidates, boundary_features, boundary_scorer
from services.timeline_types import BoundaryPair, Candidate, RankedCandidate
from services.transcriber import RichSegment


def build_boundary_pairs(
    candidate: Candidate,
    segments: list[RichSegment],
    profile: str = "general",
) -> list[BoundaryPair]:
    boundaries = boundary_candidates.build_boundary_points(segments, candidate)
    if not boundaries:
        return []
    features = boundary_features.build_boundary_features(candidate, segments, boundaries)
    start_scores = boundary_scorer.score_start_boundaries(features, profile)
    end_scores = boundary_scorer.score_end_boundaries(features, profile)
    return boundary_scorer.score_boundary_pairs(
        candidate,
        segments,
        boundaries,
        start_scores,
        end_scores,
        profile,
    )


def refine_candidate_boundaries(
    ranked: RankedCandidate,
    segments: list[RichSegment],
    profile: str = "general",
) -> tuple[BoundaryPair | None, list[BoundaryPair]]:
    pairs = build_boundary_pairs(ranked.candidate, segments, profile=profile)
    if not pairs:
        return None, []
    return pairs[0], pairs
