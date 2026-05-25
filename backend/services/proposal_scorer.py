from __future__ import annotations

import json
import math
import pathlib
import re
from dataclasses import asdict, dataclass

from config import BACKEND_ROOT
from services import clip_strategy
from services.timeline_types import Candidate
from services.transcriber import RichSegment

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "who",
    "when",
    "where",
    "wait",
    "was",
    "warum",
    "wie",
    "wer",
    "wann",
}
_HOOK_WORDS = {
    "wait",
    "bro",
    "listen",
    "look",
    "watch",
    "chat",
    "no",
    "what",
    "guys",
    "yo",
    "digga",
}
_PAYOFF_WORDS = {
    "insane",
    "crazy",
    "wild",
    "no way",
    "what",
    "oh my god",
    "fuck",
    "finally",
    "won",
    "lost",
    "dead",
    "clutch",
}


@dataclass
class ProposalFeatures:
    candidate_id: str
    source: str
    duration: float = 0.0
    segment_count: int = 0
    token_count: int = 0
    max_hype: float = 0.0
    mean_hype: float = 0.0
    arc_peak: float = 0.0
    chat_peak: float = 0.0
    motion_peak: float = 0.0
    face_peak: float = 0.0
    laughter_count: int = 0
    exclamation_count: int = 0
    question_count: int = 0
    speaker_turns: int = 0
    hook_strength: float = 0.0
    payoff_strength: float = 0.0
    hype_delta: float = 0.0
    early_hook_score: float = 0.0
    late_payoff_score: float = 0.0
    standalone_text_score: float = 0.0
    source_prior: float = 0.0
    heuristic_hit_count: int = 0
    strategy_relevance: float = 0.0
    dense_inventory: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def numeric_values(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
            if key not in {"candidate_id", "source"}
            if isinstance(value, int | float)
        }


@dataclass
class ProposalScore:
    candidate: Candidate
    features: ProposalFeatures
    score: float
    model_score: float | None
    heuristic_score: float
    rationale: list[str]

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "features": self.features.to_dict(),
            "score": self.score,
            "model_score": self.model_score,
            "heuristic_score": self.heuristic_score,
            "rationale": list(self.rationale),
        }


@dataclass
class ProposalResult:
    kept: list[Candidate]
    scored: list[ProposalScore]
    audit_rows: list[dict]
    summary: dict[str, object]


def score_candidates(
    candidates: list[Candidate],
    segments: list[RichSegment],
    profile: str,
    strategy: dict | None = None,
    *,
    artifact_dir: pathlib.Path | str | None = None,
) -> list[ProposalScore]:
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    artifact = _load_artifact(profile, artifact_dir=artifact_dir)
    scored: list[ProposalScore] = []
    for candidate in candidates:
        features = build_proposal_features(candidate, segments, normalized_strategy)
        heuristic_score, rationale = _heuristic_score(features)
        model_score = _model_score(features, artifact) if artifact is not None else None
        score = model_score if model_score is not None else heuristic_score
        scored.append(
            ProposalScore(
                candidate=candidate,
                features=features,
                score=round(float(score), 6),
                model_score=None if model_score is None else round(float(model_score), 6),
                heuristic_score=round(float(heuristic_score), 6),
                rationale=rationale,
            )
        )
    scored.sort(key=lambda item: (-item.score, item.candidate.start, item.candidate.end, item.candidate.id))
    return scored


def select_candidates(
    candidates: list[Candidate],
    segments: list[RichSegment],
    profile: str,
    strategy: dict | None = None,
    *,
    top_k: int = 240,
    min_score: float = 0.0,
    artifact_dir: pathlib.Path | str | None = None,
) -> ProposalResult:
    scored = score_candidates(candidates, segments, profile, strategy, artifact_dir=artifact_dir)
    limit = max(int(top_k or 0), 0)
    threshold = float(min_score or 0.0)
    kept_scores = [item for item in scored if item.score >= threshold]
    if limit > 0:
        kept_scores = kept_scores[:limit]
    kept = [item.candidate for item in kept_scores]
    kept_ids = {candidate.id for candidate in kept}
    audit_rows = [
        {
            "candidate_id": item.candidate.id,
            "source": item.candidate.source,
            "start": item.candidate.start,
            "end": item.candidate.end,
            "score": item.score,
            "model_score": item.model_score,
            "heuristic_score": item.heuristic_score,
            "accepted": item.candidate.id in kept_ids,
            "rationale": list(item.rationale),
            "features": item.features.to_dict(),
        }
        for item in scored
    ]
    summary = {
        "input_candidates": len(candidates),
        "scored_candidates": len(scored),
        "kept_candidates": len(kept),
        "top_k": limit,
        "min_score": threshold,
        "model_artifact_loaded": any(item.model_score is not None for item in scored),
        "sources": _source_counts(candidates),
        "kept_sources": _source_counts(kept),
    }
    return ProposalResult(kept=kept, scored=scored, audit_rows=audit_rows, summary=summary)


def build_proposal_features(
    candidate: Candidate,
    segments: list[RichSegment],
    strategy: dict,
) -> ProposalFeatures:
    window = [seg for seg in segments if seg.end > candidate.start and seg.start < candidate.end]
    duration = max(candidate.end - candidate.start, 0.0)
    text = " ".join(seg.text.strip() for seg in window if seg.text.strip())
    lower_text = text.lower()
    tokens = _TOKEN_RE.findall(lower_text)
    speakers = [seg.speaker for seg in window if seg.speaker != "UNKNOWN"]
    early, late = _split_window(window, candidate)
    source_prior = source_prior_for_candidate(candidate.source)
    strategy_relevance = clip_strategy.text_relevance_score(text, strategy) if text else 0.0
    max_hype = max((seg.hype_score for seg in window), default=0.0)
    mean_hype = sum(seg.hype_score for seg in window) / len(window) if window else 0.0
    arc_peak = max((seg.arc_score for seg in window), default=0.0)
    chat_peak = max((seg.chat_spike_score for seg in window), default=0.0)
    motion_peak = max((seg.max_motion_score for seg in window), default=0.0)
    face_peak = max((seg.max_face_reaction for seg in window), default=0.0)
    laughter_count = sum(
        1 for seg in window if "audio_laughter" in seg.emotions or "laughter" in seg.emotions
    )
    exclamation_count = sum(1 for seg in window if "exclamation" in seg.emotions or "!" in seg.text)
    question_count = sum(1 for seg in window if "question" in seg.emotions or "?" in seg.text)
    speaker_turns = sum(
        1
        for left, right in zip(speakers, speakers[1:])
        if left != right
    )
    early_hook_score = _segment_signal(early, hook=True)
    late_payoff_score = _segment_signal(late, hook=False)
    hook_strength = max(
        early_hook_score,
        min(sum(1 for token in tokens[:12] if token in _HOOK_WORDS) / 3.0, 1.0),
        0.4 if any(word in _QUESTION_WORDS for word in tokens[:10]) else 0.0,
    )
    payoff_strength = max(
        late_payoff_score,
        min(sum(1 for phrase in _PAYOFF_WORDS if phrase in lower_text) / 3.0, 1.0),
        min((laughter_count + exclamation_count) / 4.0, 1.0),
    )
    standalone_text_score = _standalone_text_score(tokens, text)
    hype_delta = (window[-1].hype_score - window[0].hype_score) if len(window) >= 2 else 0.0
    return ProposalFeatures(
        candidate_id=candidate.id,
        source=candidate.source,
        duration=round(duration, 3),
        segment_count=len(window),
        token_count=len(tokens),
        max_hype=round(max_hype, 6),
        mean_hype=round(mean_hype, 6),
        arc_peak=round(arc_peak, 6),
        chat_peak=round(chat_peak, 6),
        motion_peak=round(motion_peak, 6),
        face_peak=round(face_peak, 6),
        laughter_count=laughter_count,
        exclamation_count=exclamation_count,
        question_count=question_count,
        speaker_turns=speaker_turns,
        hook_strength=round(_clamp01(hook_strength), 6),
        payoff_strength=round(_clamp01(payoff_strength), 6),
        hype_delta=round(hype_delta, 6),
        early_hook_score=round(_clamp01(early_hook_score), 6),
        late_payoff_score=round(_clamp01(late_payoff_score), 6),
        standalone_text_score=round(_clamp01(standalone_text_score), 6),
        source_prior=round(source_prior, 6),
        heuristic_hit_count=0 if candidate.source == "dense_inventory" else 1,
        strategy_relevance=round(strategy_relevance, 6),
        dense_inventory=1.0 if candidate.source == "dense_inventory" else 0.0,
    )


def _heuristic_score(features: ProposalFeatures) -> tuple[float, list[str]]:
    duration_prior = _duration_prior(features.duration)
    signal = (
        min(features.max_hype / 6.0, 1.0) * 0.17
        + min(features.mean_hype / 4.0, 1.0) * 0.08
        + min(features.arc_peak / 5.0, 1.0) * 0.12
        + min(features.chat_peak, 1.0) * 0.11
        + max(min(features.motion_peak, 1.0), min(features.face_peak, 1.0)) * 0.10
        + features.hook_strength * 0.12
        + features.payoff_strength * 0.15
        + features.standalone_text_score * 0.07
        + duration_prior * 0.05
        + min(features.strategy_relevance, 1.0) * 0.05
        + features.source_prior * 0.08
    )
    if features.segment_count <= 0:
        signal -= 0.25
    if features.token_count < 3 and features.max_hype < 3.5:
        signal -= 0.12
    score = _clamp01(signal)
    rationale = []
    if features.hook_strength >= 0.55:
        rationale.append("hook_signal")
    if features.payoff_strength >= 0.55:
        rationale.append("payoff_signal")
    if features.chat_peak >= 0.45:
        rationale.append("chat_spike")
    if features.max_hype >= 4.0:
        rationale.append("hype_peak")
    if features.source_prior >= 0.55:
        rationale.append("heuristic_source")
    if not rationale:
        rationale.append("dense_low_cost_features")
    return score, rationale


def _model_score(features: ProposalFeatures, artifact: dict) -> float:
    weights = artifact.get("weights") or {}
    bias = float(artifact.get("bias", 0.0) or 0.0)
    values = features.numeric_values()
    means = artifact.get("feature_means") or {}
    scales = artifact.get("feature_scales") or {}
    raw = bias
    for name in weights:
        value = values.get(name, 0.0)
        if name in means or name in scales:
            value = (value - float(means.get(name, 0.0) or 0.0)) / max(
                float(scales.get(name, 1.0) or 1.0),
                1e-6,
            )
        raw += float(weights.get(name, 0.0) or 0.0) * value
    link = str(artifact.get("link", "sigmoid") or "sigmoid").lower()
    if link == "identity":
        return _clamp01(raw)
    return 1.0 / (1.0 + math.exp(-max(min(raw, 40.0), -40.0)))


def _load_artifact(profile: str, *, artifact_dir: pathlib.Path | str | None = None) -> dict | None:
    artifact_root = pathlib.Path(artifact_dir) if artifact_dir is not None else BACKEND_ROOT / "ml" / "artifacts"
    candidates = [
        artifact_root / f"{profile}_proposal_scorer.json",
        artifact_root / "general_proposal_scorer.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("training_summary") if isinstance(payload, dict) else {}
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("weights"), dict)
            and bool(payload.get("weights"))
            and str((summary or {}).get("status") or "trained") == "trained"
        ):
            return payload
    return None


def _split_window(
    window: list[RichSegment],
    candidate: Candidate,
) -> tuple[list[RichSegment], list[RichSegment]]:
    if not window:
        return [], []
    duration = max(candidate.end - candidate.start, 0.001)
    early_cut = candidate.start + duration * 0.35
    late_cut = candidate.start + duration * 0.60
    early = [seg for seg in window if seg.start < early_cut]
    late = [seg for seg in window if seg.end > late_cut]
    return early or window[:1], late or window[-1:]


def _segment_signal(segments: list[RichSegment], *, hook: bool) -> float:
    if not segments:
        return 0.0
    signal = 0.0
    for seg in segments:
        text = seg.text.lower()
        words = _TOKEN_RE.findall(text)
        lexical = 0.0
        if hook:
            lexical += min(sum(1 for word in words[:10] if word in _HOOK_WORDS) / 3.0, 1.0)
            lexical += 0.35 if "?" in text or "question" in seg.emotions else 0.0
        else:
            lexical += min(sum(1 for phrase in _PAYOFF_WORDS if phrase in text) / 3.0, 1.0)
            lexical += 0.35 if "!" in text or "exclamation" in seg.emotions else 0.0
            lexical += 0.35 if "laughter" in seg.emotions or "audio_laughter" in seg.emotions else 0.0
        signal = max(
            signal,
            min(seg.hype_score / 6.0, 1.0) * 0.35
            + min(seg.chat_spike_score, 1.0) * 0.25
            + max(min(seg.max_face_reaction, 1.0), min(seg.max_motion_score, 1.0)) * 0.20
            + min(lexical, 1.0) * 0.20,
        )
    return _clamp01(signal)


def _standalone_text_score(tokens: list[str], text: str) -> float:
    if not tokens:
        return 0.0
    length_score = min(len(tokens) / 24.0, 1.0)
    sentence_score = 0.2 if text.strip().endswith((".", "!", "?")) else 0.0
    question_or_answer = 0.2 if any(token in _QUESTION_WORDS for token in tokens[:12]) else 0.0
    repetition_penalty = 0.25 if len(set(tokens)) <= max(2, len(tokens) // 3) else 0.0
    return _clamp01(length_score * 0.65 + sentence_score + question_or_answer - repetition_penalty)


def _duration_prior(duration: float) -> float:
    if duration <= 0:
        return 0.0
    if 12.0 <= duration <= 45.0:
        return 1.0
    if 8.0 <= duration < 12.0 or 45.0 < duration <= 70.0:
        return 0.65
    return 0.25


def source_prior_for_candidate(source: str) -> float:
    priors = {
        "peak": 0.95,
        "hype_peak": 0.85,
        "story_payoff": 0.85,
        "tension_build": 0.78,
        "quote": 0.76,
        "reaction": 0.72,
        "chat_sync": 0.72,
        "goal_prompt": 0.70,
        "clean_reaction": 0.68,
        "controversy": 0.67,
        "narrative_reveal": 0.65,
        "dense_window": 0.48,
        "dense_inventory": 0.38,
    }
    return priors.get(source, 0.45)


def _source_counts(candidates: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.source] = counts.get(candidate.source, 0) + 1
    return counts


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
