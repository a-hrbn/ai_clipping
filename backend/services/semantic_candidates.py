"""
Semantic candidate generation using sentence embeddings.

Uses sentence-transformers to find transcript segments that are semantically
similar to known viral patterns — independent of whether audio/chat signals
already flagged them as "hype peaks".

Signal types (content-type agnostic):
  - tension_build   : Rising tension before an explosion. Viewer feels "something is about to happen".
  - narrative_reveal: A twist, reveal, or information drop that reframes the context.
  - controversy     : Disagreement, confrontation, hot-take. Distinct from mere "exclamation".
  - personal_moment : Emotional, vulnerable, or heartfelt moment — common in Just Chatting / IRL.
  - insight         : Insight, lesson, or perspective shift — common in podcast / interview content.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - Supports DE + EN natively (content is often mixed)
  - 471 MB, loaded once at process start and cached
  - ~15ms per batch of 50 segments on CPU
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import threading
from typing import TYPE_CHECKING

import numpy as np
from config import DEFAULT_CACHE_DIR

if TYPE_CHECKING:
    from services.transcriber import RichSegment

logger = logging.getLogger(__name__)

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_REPO_ID = f"sentence-transformers/{_MODEL_NAME}"
_model = None
_model_lock = threading.Lock()
_model_runtime_logged = False

# ---------------------------------------------------------------------------
# Viral pattern templates — bilingual (DE + EN)
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, list[str]] = {
    "tension_build": [
        # EN — gaming, IRL, drama
        "this is getting intense", "something big is about to happen",
        "you can feel the tension building", "wait for it, here it comes",
        "I have a bad feeling about this", "this is going to be insane",
        "I can barely watch", "hold on, this is getting serious",
        "it's all leading up to this moment",
        # EN — Just Chatting / story build-up
        "and then it got really weird", "things started to escalate",
        "that's when everything changed", "I didn't know what was about to happen",
        "the situation was getting out of hand",
        # DE
        "gleich passiert es", "das wird gleich krass", "ich kann kaum hinschauen",
        "gleich knallt es", "das wird nicht gut ausgehen", "ich hab ein schlechtes Gefühl",
        "das steigert sich gerade", "hier kommt gleich was",
        "dann ist es eskaliert", "da hat sich alles verändert",
    ],
    "narrative_reveal": [
        # EN — gaming, drama, storytelling
        "I can't believe this just happened", "nobody saw that coming",
        "turns out the whole time", "the truth is actually",
        "this changes everything we thought we knew",
        "wait, that's not what I expected at all",
        "the real reason behind this is", "plot twist",
        "I didn't realize until now", "this reveals everything",
        # EN — Just Chatting / IRL / personal story
        "so it turned out", "and that's when I found out",
        "the crazy part is", "what I didn't know was",
        "I never told anyone this before", "here's the thing nobody knows",
        # DE
        "ich kann es nicht glauben", "das hätte niemand erwartet",
        "stellt sich raus", "die Wahrheit ist eigentlich",
        "das verändert alles", "das hätte ich nie gedacht",
        "das ist der eigentliche Grund", "der wahre Grund",
        "jetzt ergibt alles einen Sinn", "das erklärt alles",
        "und dann hab ich rausgefunden", "das wusste keiner",
    ],
    "controversy": [
        # EN — debates, hot takes, confrontations
        "I completely disagree with that", "that is objectively wrong",
        "controversial opinion incoming", "hot take but hear me out",
        "unpopular opinion but I'll say it", "fight me on this",
        "nobody wants to hear this but", "I'm going to upset people by saying this",
        "this is going to be divisive", "let's argue about this",
        "I hate to say it but", "this is the part where people get mad",
        # EN — Just Chatting / IRL confrontation
        "I called them out on it", "that's not okay and I said it",
        "we had a huge argument about this", "people are going to disagree with me",
        "I stood my ground", "that's where I drew the line",
        # DE
        "ich bin komplett anderer Meinung", "das stimmt einfach nicht",
        "kontroverse Meinung aber", "ich sage es mal so",
        "das will keiner hören aber", "streitet mich an dafür",
        "unpopuläre Meinung", "das wird Leute aufregen",
        "hier kommen Gegenargumente", "ich widerspreche hier komplett",
        "da hab ich widersprochen", "ich hab das direkt angesprochen",
    ],
    "quote": [
        # EN — advice, insight, memorable lines
        "remember this one thing", "listen very carefully",
        "here is the most important lesson", "this is the truth nobody talks about",
        "I am going to say something real right now",
        "write this down", "the secret that changed everything for me",
        "here's what nobody tells you", "this is the advice I wish I had",
        # EN — Just Chatting / personal wisdom
        "and I realized that", "what I've learned is",
        "the thing that really matters is", "if I could go back",
        "the best decision I ever made", "what changed my perspective was",
        # DE
        "merk dir das genau", "hör genau zu jetzt",
        "das ist die wichtigste Lektion", "das ist die Wahrheit die keiner ausspricht",
        "ich sag jetzt mal was Ehrliches", "das ist das Geheimnis",
        "schreib das auf", "das hat alles verändert für mich",
        "und ich hab gemerkt dass", "was ich gelernt habe ist",
        "das Wichtigste im Leben ist", "wenn ich zurückgehen könnte",
    ],
    # --- New: personal / emotional moments (Just Chatting, IRL, storytelling) ---
    "personal_moment": [
        # EN
        "I'm going to be honest with you", "this is something I don't usually talk about",
        "I actually cried when", "that was one of the hardest moments of my life",
        "I'm getting emotional just thinking about it",
        "this changed me as a person", "I was at my lowest point",
        "that's when I knew I had to change", "nobody knew what I was going through",
        "I just need to be real for a second", "I've never felt so",
        "that hit different", "I can't explain how much that meant to me",
        # DE
        "ich bin mal ehrlich mit euch", "darüber rede ich normalerweise nicht",
        "ich hab damals wirklich geweint", "das war einer der schwersten Momente",
        "ich werde gerade emotional", "das hat mich als Person verändert",
        "ich war auf dem Tiefpunkt", "da wusste ich es muss sich was ändern",
        "das hat mich wirklich getroffen", "so hab ich mich noch nie gefühlt",
    ],
    # --- New: insight / perspective shift (podcast, interview, Just Chatting) ---
    "insight": [
        # EN
        "the problem with most people is", "here's why that doesn't work",
        "what most people get wrong about", "the thing that actually matters is",
        "and that's the key insight", "it completely changed how I see",
        "the reason this happens is", "think about it this way",
        "most people don't realize this but", "the counterintuitive thing is",
        "what I've noticed is", "the pattern I keep seeing is",
        "this is why I believe", "and that's the uncomfortable truth",
        # DE
        "das Problem mit den meisten ist", "und das ist der entscheidende Punkt",
        "was die meisten falsch verstehen", "darauf kommt es wirklich an",
        "das hat meine Sichtweise komplett verändert", "der Grund dafür ist",
        "stell dir das mal so vor", "die meisten merken das gar nicht aber",
        "was ich immer wieder beobachte ist", "und das ist die unbequeme Wahrheit",
    ],
}

# Cosine similarity thresholds per category.
# Calibrated empirically: midpoint between noise floor and true-positive scores.
#   tension_build:    noise≈0.67  target≈0.87  → 0.75
#   narrative_reveal: noise≈0.57  target≈0.70  → 0.64
#   controversy:      noise≈0.64  target≈0.92  → 0.74
#   quote:            noise≈0.62  target≈0.97  → 0.74
#   personal_moment:  noise≈0.60  target≈0.80  → 0.68  (conversational, broader match)
#   insight:          noise≈0.60  target≈0.78  → 0.67  (analytical, broader match)
_THRESHOLDS: dict[str, float] = {
    "tension_build":    0.75,
    "narrative_reveal": 0.64,
    "controversy":      0.74,
    "quote":            0.74,
    "personal_moment":  0.68,
    "insight":          0.67,
}

# Maximum candidates per category
_MAX_PER_CATEGORY: dict[str, int] = {
    "tension_build":    8,
    "narrative_reveal": 8,
    "controversy":      8,
    "quote":            6,
    "personal_moment":  8,
    "insight":          8,
}

# Cached pattern embeddings (computed once after model load)
_pattern_embeddings: dict[str, np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _sentence_transformer_cache_folder() -> Path:
    configured = (os.environ.get("HF_HOME") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_CACHE_DIR / "huggingface"


def _local_sentence_transformer_model_dir(model_repo_id: str) -> Path:
    return DEFAULT_CACHE_DIR / "sentence_transformers_local" / model_repo_id.replace("/", "--")


def _snapshot_dir(cache_namespace: str, model_repo_id: str) -> Path | None:
    repo_dir_name = f"models--{model_repo_id.replace('/', '--')}"
    cache_root = _sentence_transformer_cache_folder() / cache_namespace / repo_dir_name
    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_dir = cache_root / "snapshots" / revision
            if snapshot_dir.exists():
                return snapshot_dir

    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshot_dirs = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshot_dirs[0] if snapshot_dirs else None


def _has_sentence_transformer_metadata(snapshot_dir: Path | None) -> bool:
    if snapshot_dir is None:
        return False
    return (
        (snapshot_dir / "modules.json").exists()
        and (snapshot_dir / "config_sentence_transformers.json").exists()
        and (snapshot_dir / "1_Pooling" / "config.json").exists()
    )


def _has_transformer_weights(snapshot_dir: Path | None) -> bool:
    if snapshot_dir is None:
        return False
    has_model = any(
        (snapshot_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    has_tokenizer = any(
        (snapshot_dir / name).exists()
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.txt",
        )
    )
    return (
        (snapshot_dir / "config.json").exists()
        and (snapshot_dir / "tokenizer_config.json").exists()
        and has_model
        and has_tokenizer
    )


def _materialized_model_complete(model_dir: Path) -> bool:
    has_model = any(
        (model_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    has_tokenizer = any(
        (model_dir / name).exists()
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.txt",
        )
    )
    return (
        (model_dir / "modules.json").exists()
        and (model_dir / "config_sentence_transformers.json").exists()
        and (model_dir / "1_Pooling" / "config.json").exists()
        and (model_dir / "config.json").exists()
        and (model_dir / "tokenizer_config.json").exists()
        and has_model
        and has_tokenizer
    )


def _link_snapshot_contents(source_dir: Path, target_dir: Path) -> None:
    for source_path in sorted(source_dir.rglob("*")):
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        if target_path.exists():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            target_path.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def _materialize_local_sentence_transformer_model(model_repo_id: str) -> Path | None:
    hub_snapshot = _snapshot_dir("hub", model_repo_id)
    transformer_snapshot = _snapshot_dir("transformers", model_repo_id)
    if not _has_sentence_transformer_metadata(hub_snapshot):
        return None
    if not _has_transformer_weights(transformer_snapshot):
        return None

    model_dir = _local_sentence_transformer_model_dir(model_repo_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    _link_snapshot_contents(hub_snapshot, model_dir)
    _link_snapshot_contents(transformer_snapshot, model_dir)
    if not _materialized_model_complete(model_dir):
        return None
    return model_dir


def _get_model():
    global _model, _model_runtime_logged
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            cache_folder = str(_sentence_transformer_cache_folder())
            local_model_dir = _materialize_local_sentence_transformer_model(_MODEL_REPO_ID)
            source = "local_snapshot" if local_model_dir is not None else "huggingface"
            logger.info("semantic_candidates_init model=%s source=%s", _MODEL_NAME, source)
            if local_model_dir is not None:
                _model = SentenceTransformer(
                    str(local_model_dir),
                    local_files_only=True,
                )
            else:
                _model = SentenceTransformer(
                    _MODEL_NAME,
                    cache_folder=cache_folder,
                    local_files_only=False,
                )
            if not _model_runtime_logged:
                logger.info("semantic_candidates_runtime model=%s source=%s", _MODEL_NAME, source)
                _model_runtime_logged = True
        except Exception as exc:
            logger.warning("Could not load sentence-transformers model: %s", exc)
            _model = None
    return _model


def _get_pattern_embeddings(model) -> dict[str, np.ndarray]:
    global _pattern_embeddings
    if _pattern_embeddings is not None:
        return _pattern_embeddings
    result = {}
    for category, texts in _PATTERNS.items():
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        result[category] = emb  # shape: (n_patterns, dim)
    _pattern_embeddings = result
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_semantic_candidates(
    segments: list[RichSegment],
) -> dict[str, list[int]]:
    """
    Return a dict mapping category → list of segment indices that exceed the
    similarity threshold for that category.

    Returns empty dict if the model is unavailable (graceful degradation).
    """
    model = _get_model()
    if model is None or not segments:
        return {}

    texts = [seg.text.strip() or " " for seg in segments]

    try:
        seg_embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )  # shape: (n_segments, dim)
    except Exception as exc:
        logger.warning("Semantic embedding failed: %s", exc)
        return {}

    pattern_embs = _get_pattern_embeddings(model)
    results: dict[str, list[int]] = {}

    for category, pat_emb in pattern_embs.items():
        threshold = _THRESHOLDS[category]
        max_n = _MAX_PER_CATEGORY[category]

        # cosine similarity: (n_segments, n_patterns)
        sim = seg_embeddings @ pat_emb.T
        # max similarity across all patterns for each segment
        max_sim = sim.max(axis=1)  # shape: (n_segments,)

        # Rank by similarity, keep top candidates above threshold
        ranked = np.argsort(max_sim)[::-1]
        hits = [int(i) for i in ranked if max_sim[i] >= threshold][:max_n]
        if hits:
            results[category] = hits

    return results
