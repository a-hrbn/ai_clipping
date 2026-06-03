from __future__ import annotations

import hashlib
import json
import logging
import math
import pathlib
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence

from config import BACKEND_ROOT

DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
PATTERN_STORE_VERSION = 2
SEMANTIC_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_CLUSTER_SIMILARITY = 0.72
SEMANTIC_SCORE_WEIGHT = 0.78
LEXICAL_SCORE_WEIGHT = 0.22
logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_TEXT_FIELDS = (
    "repost_title",
    "short_title",
    "video_title",
    "title",
    "matched_cross_platform_title",
    "transcript_repost_excerpt",
    "transcript_vod_excerpt",
    "description",
    "repost_description",
)
_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "all",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "bro",
    "but",
    "can",
    "chat",
    "did",
    "does",
    "dont",
    "for",
    "from",
    "get",
    "got",
    "gonna",
    "going",
    "had",
    "has",
    "have",
    "here",
    "hes",
    "how",
    "im",
    "into",
    "its",
    "ive",
    "jynxzi",
    "just",
    "know",
    "like",
    "look",
    "not",
    "now",
    "off",
    "one",
    "out",
    "that",
    "thats",
    "the",
    "them",
    "there",
    "they",
    "think",
    "this",
    "was",
    "wait",
    "what",
    "when",
    "where",
    "with",
    "would",
    "yeah",
    "you",
    "your",
}


@dataclass(frozen=True)
class PatternMatch:
    pattern_id: str = ""
    label: str = ""
    score: float = 0.0
    support: float = 0.0
    views_prior: float = 0.0
    confidence: float = 0.0
    similarity_margin: float = 0.0


def artifact_path(profile: str, artifact_dir: pathlib.Path | str | None = None) -> pathlib.Path:
    root = pathlib.Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACT_DIR
    return root / f"{profile}_context_patterns.json"


def text_from_row(row: Mapping[str, object]) -> str:
    parts = [
        str(row.get(field) or "").strip()
        for field in _TEXT_FIELDS
        if str(row.get(field) or "").strip()
    ]
    return " ".join(dict.fromkeys(parts))


def build_pattern_store(
    rows: Iterable[Mapping[str, object]],
    *,
    profile: str,
    min_support: int = 2,
    max_patterns: int = 48,
    cluster_similarity: float = 0.22,
    semantic_cluster_similarity: float = SEMANTIC_CLUSTER_SIMILARITY,
    embedding_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
) -> dict[str, object]:
    rows_list = list(rows)
    row_items = _row_items(rows_list, embedding_fn=embedding_fn)
    embedding_enabled = any(item.get("embedding") for item in row_items)
    clusters: list[dict[str, object]] = []
    usable_rows = 0
    for item in row_items:
        row = item["row"]
        text = str(item["text"])
        tokens = list(item["tokens"])
        usable_rows += 1
        token_set = set(tokens)
        embedding = item.get("embedding")
        best_index = -1
        best_similarity = 0.0
        for index, cluster in enumerate(clusters):
            if embedding_enabled and embedding and cluster.get("embedding_centroid"):
                similarity = _cosine(embedding, cluster["embedding_centroid"])
            else:
                cluster_tokens = set(cluster["tokens"])
                similarity = _cluster_similarity(token_set, cluster_tokens)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        threshold = semantic_cluster_similarity if embedding_enabled else cluster_similarity
        if best_index >= 0 and best_similarity >= threshold:
            _add_to_cluster(clusters[best_index], row, text, token_set, embedding=embedding)
        else:
            clusters.append(_new_cluster(row, text, token_set, embedding=embedding))

    patterns = [_pattern_from_cluster(cluster, embedding_enabled=embedding_enabled) for cluster in clusters]
    patterns = [
        pattern
        for pattern in patterns
        if int(pattern["support_count"]) >= int(min_support)
    ]
    patterns.sort(
        key=lambda pattern: (
            -float(pattern["views_prior"]),
            -int(pattern["support_count"]),
            str(pattern["label"]),
        )
    )
    if max_patterns > 0:
        patterns = patterns[:max_patterns]
    return {
        "version": PATTERN_STORE_VERSION,
        "profile": profile,
        "source_rows": len(rows_list),
        "usable_rows": usable_rows,
        "min_support": int(min_support),
        "max_patterns": int(max_patterns),
        "cluster_similarity": float(cluster_similarity),
        "semantic_cluster_similarity": float(semantic_cluster_similarity),
        "embedding_model": SEMANTIC_EMBEDDING_MODEL if embedding_enabled else "",
        "semantic_matching_enabled": bool(embedding_enabled),
        "score_weights": {
            "semantic": SEMANTIC_SCORE_WEIGHT,
            "lexical": LEXICAL_SCORE_WEIGHT,
        },
        "patterns": patterns,
    }


def write_pattern_store(
    rows: Iterable[Mapping[str, object]],
    *,
    profile: str,
    output: pathlib.Path,
    min_support: int = 2,
    max_patterns: int = 48,
    cluster_similarity: float = 0.22,
    semantic_cluster_similarity: float = SEMANTIC_CLUSTER_SIMILARITY,
    embedding_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
) -> dict[str, object]:
    rows_list = list(rows)
    payload = build_pattern_store(
        rows_list,
        profile=profile,
        min_support=min_support,
        max_patterns=max_patterns,
        cluster_similarity=cluster_similarity,
        semantic_cluster_similarity=semantic_cluster_similarity,
        embedding_fn=embedding_fn,
    )
    payload["source_rows"] = len(rows_list)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    _load_pattern_store_cached.cache_clear()
    return payload


def score_text(
    text: str,
    *,
    profile: str,
    artifact_dir: pathlib.Path | str | None = None,
    embedding_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
) -> PatternMatch:
    store = load_pattern_store(profile, artifact_dir=artifact_dir)
    query_embedding = _query_embedding(text, store, embedding_fn=embedding_fn)
    return score_text_with_store(text, store, query_embedding=query_embedding)


def score_segments(
    segments,
    *,
    profile: str,
    artifact_dir: pathlib.Path | str | None = None,
) -> PatternMatch:
    text = " ".join(str(getattr(segment, "text", "") or "").strip() for segment in segments)
    return score_text(text, profile=profile, artifact_dir=artifact_dir)


def score_text_with_store(
    text: str,
    store: Mapping[str, object] | None,
    *,
    query_embedding: Sequence[float] | None = None,
) -> PatternMatch:
    patterns = list((store or {}).get("patterns") or [])
    if not patterns:
        return PatternMatch()
    token_counts = Counter(_tokens(text))
    if not token_counts and query_embedding is None:
        return PatternMatch()
    scored = []
    for pattern in patterns:
        lexical_score = _score_pattern(token_counts, pattern) if token_counts else 0.0
        semantic_score = _semantic_score(query_embedding, pattern)
        score = _combined_pattern_score(semantic_score, lexical_score)
        scored.append((score, semantic_score, lexical_score, pattern))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_semantic_score, best_lexical_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score <= 0.0:
        return PatternMatch()
    support = min(float(best.get("support_count") or 0.0) / 5.0, 1.0)
    views_prior = float(best.get("views_prior") or 0.0)
    confidence = max(0.0, min(1.0, best_score * 0.72 + support * 0.16 + views_prior * 0.12))
    return PatternMatch(
        pattern_id=str(best.get("pattern_id") or ""),
        label=str(best.get("label") or ""),
        score=round(best_score, 6),
        support=round(support, 6),
        views_prior=round(views_prior, 6),
        confidence=round(confidence, 6),
        similarity_margin=round(max(0.0, best_score - second_score), 6),
    )


def load_pattern_store(profile: str, artifact_dir: pathlib.Path | str | None = None) -> dict[str, object] | None:
    path = artifact_path(profile, artifact_dir=artifact_dir)
    return _load_pattern_store_cached(str(path))


@lru_cache(maxsize=32)
def _load_pattern_store_cached(path: str) -> dict[str, object] | None:
    file_path = pathlib.Path(path)
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(payload.get("version") or 0) != PATTERN_STORE_VERSION:
        return None
    return payload


def _row_items(
    rows: Sequence[Mapping[str, object]],
    *,
    embedding_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None,
) -> list[dict[str, object]]:
    base_items: list[dict[str, object]] = []
    for row in rows:
        text = text_from_row(row)
        tokens = _tokens(text)
        if len(tokens) < 3:
            continue
        base_items.append({"row": row, "text": text, "tokens": tokens})
    if not base_items:
        return []

    embeddings = _embed_texts([str(item["text"]) for item in base_items], embedding_fn=embedding_fn)
    if embeddings is None:
        return base_items
    for item, embedding in zip(base_items, embeddings):
        item["embedding"] = _normalize_vector(embedding)
    return [item for item in base_items if item.get("embedding")]


def _embed_texts(
    texts: Sequence[str],
    *,
    embedding_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
) -> list[list[float]] | None:
    if not texts:
        return []
    try:
        if embedding_fn is not None:
            return [_normalize_vector(vector) for vector in embedding_fn(texts)]
        from services import semantic_candidates

        model = semantic_candidates._get_model()
        if model is None:
            return None
        encoded = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=semantic_candidates._semantic_batch_size(),
        )
        return [_normalize_vector(vector) for vector in encoded]
    except Exception as exc:
        logger.warning("context_pattern_embedding_failed reason=%s", exc)
        return None


def _query_embedding(
    text: str,
    store: Mapping[str, object] | None,
    *,
    embedding_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
) -> list[float] | None:
    if not text or not bool((store or {}).get("semantic_matching_enabled")):
        return None
    patterns = (store or {}).get("patterns") or []
    if not any(pattern.get("embedding_centroid") for pattern in patterns):
        return None
    embeddings = _embed_texts([text], embedding_fn=embedding_fn)
    if not embeddings:
        return None
    return embeddings[0]


def _new_cluster(
    row: Mapping[str, object],
    text: str,
    tokens: set[str],
    *,
    embedding: Sequence[float] | None,
) -> dict[str, object]:
    cluster = {
        "tokens": set(),
        "counter": Counter(),
        "support_count": 0,
        "view_count_total": 0.0,
        "view_count_max": 0.0,
        "embedding_sum": [],
        "embedding_centroid": [],
        "example_titles": [],
        "example_urls": [],
    }
    _add_to_cluster(cluster, row, text, tokens, embedding=embedding)
    return cluster


def _add_to_cluster(
    cluster: dict[str, object],
    row: Mapping[str, object],
    text: str,
    tokens: set[str],
    *,
    embedding: Sequence[float] | None,
) -> None:
    cluster["tokens"] = set(cluster["tokens"]) | set(tokens)
    cluster["counter"] = Counter(cluster["counter"]) + Counter(tokens)
    cluster["support_count"] = int(cluster["support_count"]) + 1
    if embedding:
        existing = list(cluster.get("embedding_sum") or [])
        vector = list(embedding)
        if not existing:
            existing = [0.0] * len(vector)
        if len(existing) == len(vector):
            cluster["embedding_sum"] = [left + right for left, right in zip(existing, vector)]
            cluster["embedding_centroid"] = _normalize_vector(cluster["embedding_sum"])
    views = _views(row)
    cluster["view_count_total"] = float(cluster["view_count_total"]) + views
    cluster["view_count_max"] = max(float(cluster["view_count_max"]), views)
    title = _first_text(row, ("repost_title", "short_title", "video_title", "title"))
    if title:
        _append_unique(cluster["example_titles"], title, limit=5)
    url = _first_text(row, ("source_url", "repost_source_url", "url"))
    if url:
        _append_unique(cluster["example_urls"], url, limit=5)
    if not title:
        _append_unique(cluster["example_titles"], text[:90], limit=5)


def _pattern_from_cluster(cluster: Mapping[str, object], *, embedding_enabled: bool) -> dict[str, object]:
    counter = Counter(cluster["counter"])
    keywords = counter.most_common(14)
    keyword_terms = [term for term, _count in keywords[:4]]
    label = "_".join(keyword_terms[:3]) or "context_pattern"
    digest_source = "|".join(term for term, _count in keywords[:8])
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    pattern = {
        "pattern_id": f"ctx_{digest}",
        "label": label,
        "support_count": int(cluster["support_count"]),
        "view_count_total": round(float(cluster["view_count_total"]), 3),
        "view_count_max": round(float(cluster["view_count_max"]), 3),
        "views_prior": round(_views_prior(float(cluster["view_count_max"])), 6),
        "keywords": [
            {"term": term, "weight": round(count / max(int(cluster["support_count"]), 1), 6)}
            for term, count in keywords
        ],
        "example_titles": list(cluster["example_titles"]),
        "example_urls": list(cluster["example_urls"]),
    }
    if embedding_enabled and cluster.get("embedding_centroid"):
        pattern["embedding_model"] = SEMANTIC_EMBEDDING_MODEL
        pattern["embedding_centroid"] = [round(float(value), 6) for value in cluster["embedding_centroid"]]
    return pattern


def _score_pattern(token_counts: Counter[str], pattern: Mapping[str, object]) -> float:
    keywords = pattern.get("keywords") or []
    if not keywords:
        return 0.0
    total_weight = 0.0
    hit_weight = 0.0
    hits = 0
    for item in keywords:
        term = str(item.get("term") or "")
        weight = max(float(item.get("weight") or 0.0), 0.1)
        total_weight += weight
        if token_counts.get(term, 0) > 0:
            hits += 1
            hit_weight += weight
    if total_weight <= 0:
        return 0.0
    weighted = hit_weight / total_weight
    coverage = hits / min(len(keywords), 8)
    return round(max(weighted, coverage) * min(1.0, math.log1p(len(token_counts)) / 3.2), 6)


def _semantic_score(query_embedding: Sequence[float] | None, pattern: Mapping[str, object]) -> float:
    centroid = pattern.get("embedding_centroid") or []
    if query_embedding is None or not centroid:
        return 0.0
    # SentenceTransformer cosine scores are in [-1, 1]. Compress the useful
    # positive band into [0, 1] so the feature remains compatible with the
    # previous lexical score scale.
    cosine = _cosine(query_embedding, centroid)
    return round(max(0.0, min(1.0, (cosine + 1.0) / 2.0)), 6)


def _combined_pattern_score(semantic_score: float, lexical_score: float) -> float:
    if semantic_score > 0.0:
        return round(semantic_score * SEMANTIC_SCORE_WEIGHT + lexical_score * LEXICAL_SCORE_WEIGHT, 6)
    return round(lexical_score, 6)


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return []
    return [value / norm for value in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left_values, right_values))


def _tokens(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(str(text or "").lower()):
        normalized = token.replace("'", "")
        if len(normalized) < 3 and not normalized.isdigit():
            continue
        if normalized in _STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _cluster_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    containment = intersection / min(len(left), len(right))
    return max(_jaccard(left, right), containment * 0.7)


def _views(row: Mapping[str, object]) -> float:
    for key in ("repost_views", "views", "view_count"):
        try:
            return max(float(row.get(key) or 0.0), 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _views_prior(views: float) -> float:
    if views <= 0:
        return 0.0
    return max(0.0, min(1.0, math.log10(views + 1.0) / 7.0))


def _first_text(row: Mapping[str, object], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def _append_unique(items: list, value: str, *, limit: int) -> None:
    if value in items:
        return
    if len(items) >= limit:
        return
    items.append(value)
