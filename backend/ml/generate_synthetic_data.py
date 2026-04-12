"""
Generate synthetic pre-training data for the ML ranker.

Produces 80 training rows (40 positive, 40 negative) based on domain knowledge
about viral clip characteristics. Each row has realistic noise so the model
learns general patterns rather than memorising exact values.

Usage:
    python3 ml/generate_synthetic_data.py [--profile gaming] [--seed 42]

Output:
    ml/artifacts/synthetic_training_rows.jsonl  (training data)
    ml/artifacts/<profile>_ranker.json          (trained artifact)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ---------------------------------------------------------------------------
# Feature profiles
# ---------------------------------------------------------------------------

def _uniform(lo: float, hi: float, rng: random.Random) -> float:
    return round(rng.uniform(lo, hi), 4)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _make_positive(idx: int, rng: random.Random) -> dict:
    """Strong viral clip: early hook, high hype, clear arc, audience confirmed."""
    hook_strength = _clamp(_uniform(0.55, 0.95, rng))
    hook_delay = _uniform(0.5, 4.0, rng)
    max_hype = _uniform(3.5, 9.0, rng)
    mean_hype = _clamp(_uniform(max_hype * 0.35, max_hype * 0.75, rng), 0.0, 14.0)
    arc_score = _uniform(3.0, 8.0, rng)
    payoff = _uniform(3.5, 8.5, rng)
    chat_peak = _uniform(0.3, 1.0, rng)
    duration = _uniform(15.0, 50.0, rng)
    context_cost = _clamp(_uniform(0.0, 0.4, rng))
    return {
        "candidate_id": f"syn_pos_{idx:03d}",
        "duration": round(duration, 2),
        "hook_delay_sec": round(hook_delay, 2),
        "hook_strength": hook_strength,
        "max_hype": round(max_hype, 3),
        "mean_hype": round(mean_hype, 3),
        "arc_score": round(arc_score, 3),
        "chat_peak": round(chat_peak, 3),
        "motion_peak": round(_uniform(0.3, 1.0, rng), 3),
        "face_reaction_peak": round(_uniform(0.25, 0.9, rng), 3),
        "speaker_turns": rng.randint(1, 6),
        "exclamation_count": rng.randint(1, 5),
        "laughter_count": rng.randint(0, 4),
        "question_open": rng.random() > 0.5,
        "context_cost": round(context_cost, 3),
        "payoff_score": round(payoff, 3),
        "context_efficiency": round(_clamp(_uniform(0.5, 1.0, rng)), 3),
        "quoteability_score": round(_clamp(_uniform(0.4, 1.0, rng)), 3),
        "surprise_score": round(_clamp(_uniform(0.3, 1.0, rng)), 3),
        "contradiction_score": round(_clamp(_uniform(0.2, 0.8, rng)), 3),
        "standalone_score": round(_clamp(_uniform(0.5, 1.0, rng)), 3),
        "semantic_density": round(_clamp(_uniform(0.3, 0.9, rng)), 3),
        "profile_fit_score": round(_clamp(_uniform(0.45, 0.95, rng)), 3),
        "source": rng.choice(["hype_peak", "chat_sync", "reaction", "quote", "tension_build"]),
    }


def _make_negative(idx: int, rng: random.Random) -> dict:
    """Weak clip: late or absent hook, low energy, no arc, no audience signal."""
    hook_strength = _clamp(_uniform(0.0, 0.32, rng))
    hook_delay = _uniform(8.0, 25.0, rng)
    max_hype = _uniform(0.0, 2.4, rng)
    mean_hype = _clamp(_uniform(0.0, max_hype * 0.8, rng), 0.0, 14.0)
    arc_score = _uniform(0.0, 2.4, rng)
    payoff = _uniform(0.0, 2.4, rng)
    chat_peak = _uniform(0.0, 0.28, rng)
    duration = _uniform(55.0, 90.0, rng)
    context_cost = _clamp(_uniform(0.5, 1.0, rng))
    return {
        "candidate_id": f"syn_neg_{idx:03d}",
        "duration": round(duration, 2),
        "hook_delay_sec": round(hook_delay, 2),
        "hook_strength": hook_strength,
        "max_hype": round(max_hype, 3),
        "mean_hype": round(mean_hype, 3),
        "arc_score": round(arc_score, 3),
        "chat_peak": round(chat_peak, 3),
        "motion_peak": round(_uniform(0.0, 0.3, rng), 3),
        "face_reaction_peak": round(_uniform(0.0, 0.25, rng), 3),
        "speaker_turns": rng.randint(0, 2),
        "exclamation_count": rng.randint(0, 1),
        "laughter_count": 0,
        "question_open": rng.random() > 0.8,
        "context_cost": round(context_cost, 3),
        "payoff_score": round(payoff, 3),
        "context_efficiency": round(_clamp(_uniform(0.0, 0.4, rng)), 3),
        "quoteability_score": round(_clamp(_uniform(0.0, 0.3, rng)), 3),
        "surprise_score": round(_clamp(_uniform(0.0, 0.3, rng)), 3),
        "contradiction_score": round(_clamp(_uniform(0.0, 0.3, rng)), 3),
        "standalone_score": round(_clamp(_uniform(0.0, 0.4, rng)), 3),
        "semantic_density": round(_clamp(_uniform(0.0, 0.3, rng)), 3),
        "profile_fit_score": round(_clamp(_uniform(0.0, 0.35, rng)), 3),
        "source": rng.choice(["hype_peak", "dialogue", "quote"]),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-positive", type=int, default=40)
    parser.add_argument("--n-negative", type=int, default=40)
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    artifact_dir = pathlib.Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / "synthetic_training_rows.jsonl"

    rows: list[dict] = []
    for i in range(args.n_positive):
        features = _make_positive(i, rng)
        rows.append({
            "job_id": "synthetic",
            "candidate_id": features["candidate_id"],
            "label": 1,
            "label_class": "synthetic_positive",
            "selected": True,
            "exported": False,
            "final_start": None,
            "final_end": None,
            "rejected_reason": None,
            "supervision_tier": "synthetic",
            "supervision_reason": "generated_bootstrap",
            "ordinal_label": 2,
            "feature_provenance": {
                "has_llm_semantics": False,
                "has_chat_features": False,
                "has_vector_similarity": False,
                "has_vod_context": False,
                "feature_build_version": 1,
            },
            "features": features,
        })

    for i in range(args.n_negative):
        features = _make_negative(i, rng)
        rows.append({
            "job_id": "synthetic",
            "candidate_id": features["candidate_id"],
            "label": 0,
            "label_class": "synthetic_negative",
            "selected": False,
            "exported": False,
            "final_start": None,
            "final_end": None,
            "rejected_reason": "low_rank",
            "supervision_tier": "synthetic",
            "supervision_reason": "generated_bootstrap",
            "ordinal_label": 0,
            "feature_provenance": {
                "has_llm_semantics": False,
                "has_chat_features": False,
                "has_vector_similarity": False,
                "has_vod_context": False,
                "feature_build_version": 1,
            },
            "features": features,
        })

    rng.shuffle(rows)
    output_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows ({args.n_positive} positive, {args.n_negative} negative) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
