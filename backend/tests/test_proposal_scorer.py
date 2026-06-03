import json
import sys
import types

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import proposal_scorer
from services.timeline_types import Candidate
from services.transcriber import RichSegment


def test_heuristic_proposal_score_prefers_hook_and_payoff_window(tmp_path) -> None:
    weak = Candidate(
        id="dense_inventory:20:0:0:0",
        start=0.0,
        end=20.0,
        source="dense_inventory",
        anchor_time=10.0,
    )
    strong = Candidate(
        id="dense_inventory:20:0:1:2000",
        start=20.0,
        end=40.0,
        source="dense_inventory",
        anchor_time=30.0,
    )
    segments = [
        RichSegment(start=0.0, end=4.0, text="quiet menu", hype_score=0.5),
        RichSegment(start=20.0, end=24.0, text="wait chat watch this?", hype_score=4.0, chat_spike_score=0.5),
        RichSegment(
            start=34.0,
            end=38.0,
            text="no way that was insane!",
            hype_score=5.2,
            arc_score=4.0,
            emotions=["exclamation", "laughter"],
        ),
    ]

    scored = proposal_scorer.score_candidates([weak, strong], segments, "jynxzi", artifact_dir=tmp_path)

    assert scored[0].candidate.id == strong.id
    assert scored[0].score > scored[1].score
    assert "payoff_signal" in scored[0].rationale
    assert scored[0].features.clip_type != "general"
    assert scored[0].features.multimodal_proxy_score > 0.0


def test_json_artifact_can_train_proposal_ranking(tmp_path) -> None:
    artifact = {
        "model_type": "linear_proposal_scorer",
        "link": "identity",
        "bias": 0.0,
        "weights": {
            "payoff_strength": 0.6,
            "hook_strength": 0.3,
            "source_prior": 0.1,
        },
    }
    (tmp_path / "jynxzi_proposal_scorer.json").write_text(json.dumps(artifact), encoding="utf-8")
    candidates = [
        Candidate(id="dense_inventory:20:0:0:0", start=0.0, end=20.0, source="dense_inventory", anchor_time=10.0),
        Candidate(id="quote:0:2000", start=20.0, end=40.0, source="quote", anchor_time=30.0),
    ]
    segments = [
        RichSegment(start=0.0, end=5.0, text="plain setup", hype_score=0.2),
        RichSegment(start=20.0, end=24.0, text="wait listen", hype_score=3.8),
        RichSegment(start=34.0, end=38.0, text="that's crazy!", hype_score=4.5, emotions=["exclamation"]),
    ]

    result = proposal_scorer.select_candidates(
        candidates,
        segments,
        "jynxzi",
        top_k=1,
        artifact_dir=tmp_path,
    )

    assert [candidate.id for candidate in result.kept] == ["quote:0:2000"]
    assert result.summary["model_artifact_loaded"] is True
    assert len(result.audit_rows) == 2


def test_untrained_artifact_does_not_override_heuristic_scores(tmp_path) -> None:
    artifact = {
        "model_type": "logistic_regression_proposal_scorer",
        "link": "sigmoid",
        "training_summary": {"status": "insufficient_rows"},
        "bias": 0.0,
        "weights": {},
    }
    (tmp_path / "jynxzi_proposal_scorer.json").write_text(json.dumps(artifact), encoding="utf-8")
    candidate = Candidate(
        id="dense_inventory:20:0:0:0",
        start=0.0,
        end=20.0,
        source="dense_inventory",
        anchor_time=10.0,
    )

    result = proposal_scorer.select_candidates([candidate], [], "jynxzi", artifact_dir=tmp_path)

    assert result.summary["model_artifact_loaded"] is False
    assert result.audit_rows[0]["model_score"] is None


def test_low_model_score_keeps_recall_floor_for_strong_dense_candidate(tmp_path) -> None:
    artifact = {
        "model_type": "logistic_regression_proposal_scorer",
        "link": "sigmoid",
        "training_summary": {"status": "trained"},
        "bias": -20.0,
        "weights": {"duration": 0.0},
    }
    (tmp_path / "jynxzi_proposal_scorer.json").write_text(json.dumps(artifact), encoding="utf-8")
    candidate = Candidate(
        id="dense_inventory:20:0:0:0",
        start=0.0,
        end=20.0,
        source="dense_inventory",
        anchor_time=10.0,
    )
    segments = [
        RichSegment(start=0.0, end=5.0, text="wait chat watch this?", hype_score=4.2),
        RichSegment(
            start=14.0,
            end=19.0,
            text="no way that was insane!",
            hype_score=5.0,
            emotions=["exclamation"],
        ),
    ]

    scored = proposal_scorer.score_candidates([candidate], segments, "jynxzi", artifact_dir=tmp_path)

    assert scored[0].model_score is not None
    assert scored[0].model_score < 0.001
    assert scored[0].score > scored[0].model_score
    assert scored[0].score >= scored[0].heuristic_score * 0.34


def test_context_complete_window_beats_context_poor_hype_window(tmp_path) -> None:
    empty_artifact_dir = tmp_path
    hype_only = Candidate(
        id="dense_inventory:20:0:0:0",
        start=0.0,
        end=20.0,
        source="dense_inventory",
        anchor_time=10.0,
    )
    contextual = Candidate(
        id="quote:0:2000",
        start=20.0,
        end=44.0,
        source="quote",
        anchor_time=34.0,
    )
    segments = [
        RichSegment(start=0.0, end=6.0, text="rrrr rrrr rrrr", hype_score=6.0),
        RichSegment(start=20.0, end=26.0, text="wait chat he sent fanmail from Marlon", hype_score=1.0),
        RichSegment(start=32.0, end=40.0, text="no way it's a Darth Maul Funko pop that's crazy!", hype_score=1.2),
    ]

    scored = proposal_scorer.score_candidates([hype_only, contextual], segments, "jynxzi", artifact_dir=empty_artifact_dir)

    assert scored[0].candidate.id == contextual.id
    assert scored[0].features.context_completeness_score > scored[1].features.context_completeness_score
    assert "context_primary" in scored[0].rationale or "context_complete" in scored[0].rationale


def test_source_prior_is_shared_for_training_and_serving() -> None:
    assert proposal_scorer.source_prior_for_candidate("quote") == 0.76
    assert proposal_scorer.source_prior_for_candidate("dense_inventory") == 0.38
