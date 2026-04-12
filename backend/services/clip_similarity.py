from __future__ import annotations

import json
import logging
import pathlib
import threading

import numpy as np

from services.semantic_candidates import _get_model
from services.transcriber import RichSegment

logger = logging.getLogger(__name__)

_ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "ml" / "artifacts"
_INDEX_SUFFIX = "_vector_index.npz"
_MANIFEST_SUFFIX = "_vector_manifest.jsonl"
_cache_lock = threading.Lock()
_artifact_cache: dict[str, dict | None] = {}


def score_candidate_window(
    window: list[RichSegment],
    anchor_time: float,
    profile: str,
) -> dict[str, float]:
    if not window:
        return _empty_similarity_features()
    artifact = _load_artifact(profile)
    model = _get_model()
    if artifact is None or model is None:
        return _empty_similarity_features()

    full_text, hook_text, payoff_text = _window_texts(window, anchor_time)
    texts = [full_text or " ", hook_text or full_text or " ", payoff_text or full_text or " "]
    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.warning("Vector similarity embedding failed: %s", exc)
        return _empty_similarity_features()

    winner_rows = [row for row in artifact["rows"] if row.get("label") == "winner"]
    loser_rows = [row for row in artifact["rows"] if row.get("label") == "loser"]
    if not winner_rows:
        return _empty_similarity_features()

    winner_scores = _neighbor_scores(embeddings, artifact["full"], artifact["hook"], artifact["payoff"], winner_rows)
    loser_scores = _neighbor_scores(embeddings, artifact["full"], artifact["hook"], artifact["payoff"], loser_rows) if loser_rows else []

    winner_top = winner_scores[:3]
    loser_top = loser_scores[:3]
    winner_similarity_max = winner_top[0]["similarity"] if winner_top else 0.0
    winner_similarity_mean = sum(item["similarity"] for item in winner_top) / len(winner_top) if winner_top else 0.0
    loser_similarity_max = loser_top[0]["similarity"] if loser_top else 0.0
    neighbor_quality_score = sum(item["quality_score"] for item in winner_top) / len(winner_top) if winner_top else 0.0
    return {
        "winner_similarity_max": round(winner_similarity_max, 3),
        "winner_similarity_mean": round(winner_similarity_mean, 3),
        "loser_similarity_max": round(loser_similarity_max, 3),
        "similarity_margin": round(winner_similarity_max - loser_similarity_max, 3),
        "neighbor_quality_score": round(neighbor_quality_score, 3),
        "vector_reference_available": True,
    }


def _load_artifact(profile: str) -> dict | None:
    cache_key = profile.strip().lower() or "general"
    with _cache_lock:
        if cache_key in _artifact_cache:
            return _artifact_cache[cache_key]

        manifest_path = _ARTIFACT_DIR / f"{cache_key}{_MANIFEST_SUFFIX}"
        index_path = _ARTIFACT_DIR / f"{cache_key}{_INDEX_SUFFIX}"
        if not manifest_path.exists() or not index_path.exists():
            _artifact_cache[cache_key] = None
            return None

        rows = _read_jsonl(manifest_path)
        if not rows:
            _artifact_cache[cache_key] = None
            return None
        try:
            embeddings = np.load(index_path)
            artifact = {
                "rows": rows,
                "full": embeddings["full_embeddings"],
                "hook": embeddings["hook_embeddings"],
                "payoff": embeddings["payoff_embeddings"],
            }
        except Exception as exc:
            logger.warning("Could not load vector index for %s: %s", cache_key, exc)
            artifact = None
        _artifact_cache[cache_key] = artifact
        return artifact


def _window_texts(window: list[RichSegment], anchor_time: float) -> tuple[str, str, str]:
    full_text = " ".join(seg.text.strip() for seg in window if seg.text.strip())
    hook_segments = [seg for seg in window if seg.end <= anchor_time][:2]
    payoff_segments = [seg for seg in window if seg.start >= anchor_time][:2]
    if not hook_segments:
        hook_segments = window[:2]
    if not payoff_segments:
        payoff_segments = window[-2:]
    hook_text = " ".join(seg.text.strip() for seg in hook_segments if seg.text.strip())
    payoff_text = " ".join(seg.text.strip() for seg in payoff_segments if seg.text.strip())
    return full_text, hook_text, payoff_text


def _neighbor_scores(
    query_embeddings: np.ndarray,
    full_embeddings: np.ndarray,
    hook_embeddings: np.ndarray,
    payoff_embeddings: np.ndarray,
    rows: list[dict],
) -> list[dict]:
    if not rows:
        return []
    indices = [int(row["_index"]) for row in rows]
    sims: list[dict] = []
    for row, index in zip(rows, indices, strict=False):
        score = (
            float(query_embeddings[0] @ full_embeddings[index]) * 0.5
            + float(query_embeddings[1] @ hook_embeddings[index]) * 0.25
            + float(query_embeddings[2] @ payoff_embeddings[index]) * 0.25
        )
        sims.append(
            {
                "clip_id": row.get("clip_id", ""),
                "similarity": score,
                "quality_score": float(row.get("quality_score", 0.0)),
            }
        )
    sims.sort(key=lambda item: item["similarity"], reverse=True)
    return sims


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _empty_similarity_features() -> dict[str, float]:
    return {
        "winner_similarity_max": 0.0,
        "winner_similarity_mean": 0.0,
        "loser_similarity_max": 0.0,
        "similarity_margin": 0.0,
        "neighbor_quality_score": 0.0,
        "vector_reference_available": False,
    }
