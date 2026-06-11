from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import json
import math
import pathlib
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import match_cross_platform_audio_fingerprints as audio_matcher
from ml import outcome_view_weighting, row_identity
from services import audio_extractor, transcriber
from services.transcriber import RichSegment


def _serialize_rich_segment(segment) -> dict:
    fields = ("start", "end", "text", "words_per_second", "hype_score", "speaker_id", "word_count")
    return {f: getattr(segment, f, None) for f in fields if getattr(segment, f, None) is not None}


def _deserialize_rich_segment(item: dict):
    try:
        return RichSegment(**{k: v for k, v in item.items() if k in RichSegment.__dataclass_fields__})
    except Exception:
        import types
        return types.SimpleNamespace(**item)


_VOD_TRANSCRIPT_CACHE_VERSION = 1


def _load_compatible_transcript_cache(cache_dir: pathlib.Path, vod_id: str) -> tuple[pathlib.Path, list] | None:
    for p in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        meta = payload.get("metadata") or {}
        if int(meta.get("version") or 0) != _VOD_TRANSCRIPT_CACHE_VERSION:
            continue
        raw = payload.get("transcript")
        if not isinstance(raw, list):
            continue
        try:
            segments = [_deserialize_rich_segment(item) for item in raw]
        except Exception:
            continue
        return p, segments
    return None


DEFAULT_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"
DEFAULT_SHORT_TRANSCRIPT_CACHE_ROOT = pathlib.Path(settings.storage_dir) / "_download_cache" / "short_form_transcripts"
DEFAULT_SHORT_FINGERPRINT_CACHE_ROOT = pathlib.Path(settings.storage_dir) / "_download_cache" / "short_form_fingerprints"
TRANSCRIPT_MATCH_SOURCE = "cross_platform_transcript_match"
_SHORT_TRANSCRIPT_CACHE_VERSION = 1
_MULTI_WINDOW_NEAR_DUPLICATE_MERGE_GAP_SEC = 30.0
_CROSS_VOD_WEAK_TOKEN_RECALL_THRESHOLD = 0.55
# A merged multi-window span may not exceed the repost length by more than
# this factor/padding — [min(starts), max(ends)] merges were observed to
# produce 84s GT windows for <=60s Shorts.
_MULTI_WINDOW_MERGE_SPAN_FACTOR = 1.2
_MULTI_WINDOW_MERGE_SPAN_PAD_SEC = 5.0
# If the best window's score beats the best alternative position (different
# VOD or >60s away) by less than this margin, the placement is ambiguous —
# Jynxzi repeats catchphrases, so near-ties are routed to review instead of
# being silently accepted.
_POSITION_AMBIGUITY_MARGIN = 0.08
_POSITION_AMBIGUITY_DISTANCE_SEC = 60.0
# Audio-fallback acceptance additionally requires the best alignment peak to
# beat the best peak elsewhere in the VOD by this margin. Measured on the
# 5-anchor set: noise-only margins have median 0.007 / max 0.033, real
# scoped confirmations sit at 0.04+; 0.02 keeps only plausible candidates.
_AUDIO_FALLBACK_MIN_PEAK_MARGIN = 0.02
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "like",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "you",
    "yeah",
}
_WORKER_VOD_TRANSCRIPTS: list["VodTranscript"] = []


@dataclass(frozen=True)
class TranscriptMatch:
    score: float
    token_recall: float
    bigram_recall: float
    sequence_ratio: float
    start: float
    end: float
    vod: dict
    vod_excerpt: str
    repost_excerpt: str
    match_kind: str = "single_window"
    acceptance_reason: str = ""
    acceptance_threshold: float = 0.0
    source_score: float | None = None
    chunk_index: int | None = None
    chunk_count: int = 1
    audio_confidence: float | None = None
    rank_score: float | None = None
    position_ambiguous: bool = False


@dataclass(frozen=True)
class RepostTranscriptChunk:
    index: int
    count: int
    text: str
    tokens: list[str]
    duration: float


@dataclass(frozen=True)
class RepostTranscriptMatchResult:
    best: TranscriptMatch | None
    chunk_matches: tuple[TranscriptMatch, ...] = ()
    repost_duration: float = 0.0


@dataclass(frozen=True)
class MatchDecision:
    accepted: bool
    reason: str
    threshold: float


@dataclass
class VodTranscript:
    row: dict
    segments: list
    segment_tokens: list[list[str]]
    token_to_indices: dict[str, list[int]]
    token_idf: dict[str, float] = field(default_factory=dict)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match local TikTok/Shorts reposts back to VOD timestamps with transcript similarity."
    )
    parser.add_argument("--vods", required=True, help="JSONL rows with vod_id and cached VOD transcript state.")
    parser.add_argument("--reposts", required=True, help="JSONL rows with video_path, platform, views, optional clip_id.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--vod-cache-root", default=str(DEFAULT_VOD_CACHE_ROOT))
    parser.add_argument("--min-similarity", type=float, default=0.60)
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for per-repost matching.")
    args = parser.parse_args()

    rows, _diagnostics = match_reposts_to_vods_with_diagnostics(
        row_identity.read_jsonl(pathlib.Path(args.vods), strict=False),
        row_identity.read_jsonl(pathlib.Path(args.reposts), strict=False),
        min_similarity=args.min_similarity,
        vod_cache_root=pathlib.Path(args.vod_cache_root),
        log_progress=True,
        transcript_match_workers=args.workers,
    )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, ensure_ascii=True))
    return 0


def match_reposts_to_vods(
    vod_rows: list[dict],
    repost_rows: list[dict],
    *,
    min_similarity: float = 0.60,
    vod_cache_root: pathlib.Path = DEFAULT_VOD_CACHE_ROOT,
    transcript_match_workers: int = 1,
) -> list[dict]:
    rows, _diagnostics = match_reposts_to_vods_with_diagnostics(
        vod_rows,
        repost_rows,
        min_similarity=min_similarity,
        vod_cache_root=vod_cache_root,
        transcript_match_workers=transcript_match_workers,
    )
    return rows


def match_reposts_to_vods_with_diagnostics(
    vod_rows: list[dict],
    repost_rows: list[dict],
    *,
    min_similarity: float = 0.60,
    vod_cache_root: pathlib.Path = DEFAULT_VOD_CACHE_ROOT,
    short_transcript_cache_root: pathlib.Path = DEFAULT_SHORT_TRANSCRIPT_CACHE_ROOT,
    short_fingerprint_cache_root: pathlib.Path = DEFAULT_SHORT_FINGERPRINT_CACHE_ROOT,
    diagnostic_top_k: int = 25,
    log_progress: bool = False,
    vod_progress_interval: int = 25,
    repost_progress_interval: int = 5,
    transcript_match_workers: int = 1,
) -> tuple[list[dict], dict[str, object]]:
    started_at = time.monotonic()
    threshold = max(float(min_similarity), 0.0)
    requested_workers = max(1, int(transcript_match_workers or 1))
    if log_progress:
        _log_event(
            "cross_platform_transcript_match_start",
            vod_rows=len(vod_rows),
            repost_rows=len(repost_rows),
            min_similarity=round(threshold, 6),
            diagnostic_top_k=int(diagnostic_top_k),
            transcript_match_workers=requested_workers,
        )

    vod_transcripts: list[VodTranscript] = []
    skipped_vod_transcript = 0
    for index, vod in enumerate(vod_rows, start=1):
        loaded = _load_vod_transcript(vod, vod_cache_root=vod_cache_root)
        if loaded is None:
            skipped_vod_transcript += 1
            if log_progress and skipped_vod_transcript <= 5:
                _log_event(
                    "cross_platform_transcript_vod_skip",
                    reason="missing_transcript_cache",
                    vod_index=index,
                    vod_rows=len(vod_rows),
                    vod_id=str(vod.get("vod_id") or ""),
                )
        else:
            vod_transcripts.append(loaded)
        if log_progress and _should_log_progress(index, len(vod_rows), vod_progress_interval):
            _log_event(
                "cross_platform_transcript_vod_progress",
                processed_vods=index,
                total_vods=len(vod_rows),
                vod_transcripts_loaded=len(vod_transcripts),
                skipped_vod_transcript=skipped_vod_transcript,
                elapsed_sec=round(time.monotonic() - started_at, 1),
            )

    output_rows: list[dict] = []
    best_match_rows: list[dict] = []
    skipped_repost_media = 0
    skipped_repost_transcript = 0
    repost_transcript_cache_hits = 0
    repost_transcript_cache_misses = 0
    if log_progress:
        _log_event(
            "cross_platform_transcript_vods_ready",
            vod_rows=len(vod_rows),
            vod_transcripts_loaded=len(vod_transcripts),
            skipped_vod_transcript=skipped_vod_transcript,
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )

    best_similarity_seen = 0.0
    match_inputs: list[tuple[int, dict, str, list[str], float, tuple[RepostTranscriptChunk, ...]]] = []
    for index, repost in enumerate(repost_rows, start=1):
        repost_path = audio_matcher._resolve_media_path(str(repost.get("video_path") or ""))
        if repost_path is None:
            skipped_repost_media += 1
            if log_progress and skipped_repost_media <= 5:
                _log_event(
                    "cross_platform_transcript_repost_skip",
                    reason="missing_media",
                    repost_index=index,
                    repost_rows=len(repost_rows),
                    source_url=str(repost.get("source_url") or ""),
                    video_path=str(repost.get("video_path") or ""),
                )
            if log_progress and _should_log_progress(index, len(repost_rows), repost_progress_interval):
                _log_repost_progress(
                    index=index,
                    total_reposts=len(repost_rows),
                    output_rows=output_rows,
                    best_match_rows=best_match_rows,
                    skipped_repost_media=skipped_repost_media,
                    skipped_repost_transcript=skipped_repost_transcript,
                    repost_transcript_cache_hits=repost_transcript_cache_hits,
                    repost_transcript_cache_misses=repost_transcript_cache_misses,
                    best_similarity_seen=best_similarity_seen,
                    started_at=started_at,
                )
            continue
        try:
            repost_transcript, cache_hit = _load_or_build_repost_transcript(
                repost,
                repost_path,
                cache_root=short_transcript_cache_root,
            )
        except Exception as exc:
            skipped_repost_transcript += 1
            if log_progress and skipped_repost_transcript <= 5:
                _log_event(
                    "cross_platform_transcript_repost_skip",
                    reason="transcript_failed",
                    repost_index=index,
                    repost_rows=len(repost_rows),
                    source_url=str(repost.get("source_url") or ""),
                    video_path=str(repost_path),
                    error=_short_error(exc),
                )
            if log_progress and _should_log_progress(index, len(repost_rows), repost_progress_interval):
                _log_repost_progress(
                    index=index,
                    total_reposts=len(repost_rows),
                    output_rows=output_rows,
                    best_match_rows=best_match_rows,
                    skipped_repost_media=skipped_repost_media,
                    skipped_repost_transcript=skipped_repost_transcript,
                    repost_transcript_cache_hits=repost_transcript_cache_hits,
                    repost_transcript_cache_misses=repost_transcript_cache_misses,
                    best_similarity_seen=best_similarity_seen,
                    started_at=started_at,
                )
            continue
        if cache_hit:
            repost_transcript_cache_hits += 1
        else:
            repost_transcript_cache_misses += 1

        repost_text = _segments_text(repost_transcript)
        repost_tokens = _tokens(repost_text)
        if not repost_tokens:
            skipped_repost_transcript += 1
            continue
        repost_duration = _transcript_duration(repost_transcript, fallback=_safe_float(repost.get("duration_sec")))
        repost_chunks = _repost_transcript_chunks(repost_transcript, total_duration=repost_duration)
        match_inputs.append((index, repost, repost_text, repost_tokens, repost_duration, repost_chunks))

    effective_workers = min(requested_workers, max(1, len(match_inputs)))
    if log_progress and match_inputs:
        _log_event(
            "cross_platform_transcript_repost_match_workers",
            requested_workers=requested_workers,
            effective_workers=effective_workers,
            match_inputs=len(match_inputs),
        )

    audio_vod_fps: list[tuple[dict, audio_matcher.Fingerprint]] | None = None
    audio_fallback_attempts = 0
    audio_fallback_matches = 0
    view_weighted_accepts = 0
    multi_window_chunk_matches = 0
    multi_window_cross_vod_filtered = 0
    multi_window_near_duplicate_merges = 0
    accepted_rows_before_cleanup = 0
    for index, repost, match_result in _iter_repost_match_results(
        match_inputs,
        vod_transcripts=vod_transcripts,
        workers=effective_workers,
    ):
        best = match_result.best
        if best is None:
            if log_progress and _should_log_progress(index, len(repost_rows), repost_progress_interval):
                _log_repost_progress(
                    index=index,
                    total_reposts=len(repost_rows),
                    output_rows=output_rows,
                    best_match_rows=best_match_rows,
                    skipped_repost_media=skipped_repost_media,
                    skipped_repost_transcript=skipped_repost_transcript,
                    repost_transcript_cache_hits=repost_transcript_cache_hits,
                    repost_transcript_cache_misses=repost_transcript_cache_misses,
                    best_similarity_seen=best_similarity_seen,
                    started_at=started_at,
                )
            continue
        best_similarity_seen = max(best_similarity_seen, best.score)
        accepted_matches: list[TranscriptMatch] = []
        single_decision = _acceptance_decision(repost, best, threshold)
        best_for_diagnostic = _with_acceptance(best, single_decision)

        chunk_accepts = 0
        for chunk_match in match_result.chunk_matches:
            chunk_decision = _acceptance_decision(repost, chunk_match, threshold)
            if not chunk_decision.accepted:
                continue
            accepted_chunk = _with_acceptance(chunk_match, chunk_decision)
            if _append_unique_match(accepted_matches, accepted_chunk):
                chunk_accepts += 1
                if chunk_decision.reason == "view_weighted_transcript_threshold":
                    view_weighted_accepts += 1
        if single_decision.accepted and (
            chunk_accepts <= 0 or single_decision.reason != "view_weighted_transcript_threshold"
        ):
            if _append_unique_match(accepted_matches, best_for_diagnostic):
                if single_decision.reason == "view_weighted_transcript_threshold":
                    view_weighted_accepts += 1
        multi_window_chunk_matches += chunk_accepts

        # Audio fallback only when the transcript produced no accepted window
        # at all: accepted chunk windows are high-precision anchors, and a
        # whole-short audio alignment added next to them is more likely a
        # misplaced duplicate than a real extra segment.
        if not accepted_matches and _should_try_audio_fallback(repost, best, threshold):
            audio_fallback_attempts += 1
            if audio_vod_fps is None:
                audio_vod_fps = _load_audio_vod_fingerprints(vod_rows)
            audio_match = _audio_fingerprint_fallback(repost, best, vod_fps=audio_vod_fps, fp_cache_root=short_fingerprint_cache_root)
            if audio_match is not None:
                accepted_audio = _with_acceptance(
                    audio_match,
                    MatchDecision(
                        accepted=True,
                        reason="audio_fingerprint_fallback",
                        threshold=_audio_fallback_confidence_threshold(repost, best),
                    ),
                )
                if _append_unique_match(accepted_matches, accepted_audio):
                    audio_fallback_matches += 1

        accepted_rows_before_cleanup += len(accepted_matches)
        accepted_matches, cleanup_stats = _clean_accepted_matches(
            accepted_matches,
            repost_duration=match_result.repost_duration or _safe_float(repost.get("duration")),
        )
        multi_window_cross_vod_filtered += cleanup_stats["multi_window_cross_vod_filtered"]
        multi_window_near_duplicate_merges += cleanup_stats["multi_window_near_duplicate_merges"]
        best_match_rows.append(
            _best_match_diagnostic_row(
                repost,
                best_for_diagnostic,
                accepted=bool(accepted_matches),
                accepted_rows=len(accepted_matches),
                accepted_match_kinds=[match.match_kind for match in accepted_matches],
                accepted_reasons=[match.acceptance_reason for match in accepted_matches],
            )
        )
        for match in accepted_matches:
            output_rows.append(_matched_row(repost, match))
        if log_progress and _should_log_progress(index, len(repost_rows), repost_progress_interval):
            _log_repost_progress(
                index=index,
                total_reposts=len(repost_rows),
                output_rows=output_rows,
                best_match_rows=best_match_rows,
                skipped_repost_media=skipped_repost_media,
                skipped_repost_transcript=skipped_repost_transcript,
                repost_transcript_cache_hits=repost_transcript_cache_hits,
                repost_transcript_cache_misses=repost_transcript_cache_misses,
                best_similarity_seen=best_similarity_seen,
                started_at=started_at,
            )

    diagnostics = _match_diagnostics(
        vod_rows=vod_rows,
        repost_rows=repost_rows,
        vod_transcripts=vod_transcripts,
        best_match_rows=best_match_rows,
        matched_rows=output_rows,
        min_similarity=threshold,
        diagnostic_top_k=diagnostic_top_k,
        skipped_vod_transcript=skipped_vod_transcript,
        skipped_repost_media=skipped_repost_media,
        skipped_repost_transcript=skipped_repost_transcript,
        repost_transcript_cache_hits=repost_transcript_cache_hits,
        repost_transcript_cache_misses=repost_transcript_cache_misses,
        transcript_match_workers=effective_workers,
        audio_fallback_attempts=audio_fallback_attempts,
        audio_fallback_matches=audio_fallback_matches,
        view_weighted_accepts=view_weighted_accepts,
        multi_window_chunk_matches=multi_window_chunk_matches,
        multi_window_cross_vod_filtered=multi_window_cross_vod_filtered,
        multi_window_near_duplicate_merges=multi_window_near_duplicate_merges,
        accepted_rows_before_cleanup=accepted_rows_before_cleanup,
    )
    if log_progress:
        _log_event(
            "cross_platform_transcript_match_done",
            repost_rows=len(repost_rows),
            vod_rows=len(vod_rows),
            vod_transcripts_loaded=len(vod_transcripts),
            matched_rows=len(output_rows),
            repost_best_match_rows=len(best_match_rows),
            skipped_vod_transcript=skipped_vod_transcript,
            skipped_repost_media=skipped_repost_media,
            skipped_repost_transcript=skipped_repost_transcript,
            repost_transcript_cache_hits=repost_transcript_cache_hits,
            repost_transcript_cache_misses=repost_transcript_cache_misses,
            best_similarity_seen=round(best_similarity_seen, 6),
            elapsed_sec=round(time.monotonic() - started_at, 1),
            transcript_match_workers=effective_workers,
            audio_fallback_attempts=audio_fallback_attempts,
            audio_fallback_matches=audio_fallback_matches,
            view_weighted_accepts=view_weighted_accepts,
            multi_window_chunk_matches=multi_window_chunk_matches,
            multi_window_cross_vod_filtered=multi_window_cross_vod_filtered,
            multi_window_near_duplicate_merges=multi_window_near_duplicate_merges,
            accepted_rows_before_cleanup=accepted_rows_before_cleanup,
        )
    return output_rows, diagnostics


def _iter_repost_match_results(
    match_inputs: list[tuple[int, dict, str, list[str], float, tuple[RepostTranscriptChunk, ...]]],
    *,
    vod_transcripts: list[VodTranscript],
    workers: int,
):
    if not match_inputs:
        return
    if workers <= 1 or len(match_inputs) <= 1:
        for match_input in match_inputs:
            yield _match_repost_input(match_input, vod_transcripts=vod_transcripts)
        return
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_match_worker,
        initargs=(vod_transcripts,),
    ) as executor:
        yield from executor.map(_match_repost_input_in_worker, match_inputs, chunksize=1)


def _init_match_worker(vod_transcripts: list[VodTranscript]) -> None:
    global _WORKER_VOD_TRANSCRIPTS
    _WORKER_VOD_TRANSCRIPTS = vod_transcripts


def _match_repost_input_in_worker(
    match_input: tuple[int, dict, str, list[str], float, tuple[RepostTranscriptChunk, ...]]
) -> tuple[int, dict, RepostTranscriptMatchResult]:
    return _match_repost_input(match_input, vod_transcripts=_WORKER_VOD_TRANSCRIPTS)


def _match_repost_input(
    match_input: tuple[int, dict, str, list[str], float, tuple[RepostTranscriptChunk, ...]],
    *,
    vod_transcripts: list[VodTranscript],
) -> tuple[int, dict, RepostTranscriptMatchResult]:
    index, repost, repost_text, repost_tokens, repost_duration, repost_chunks = match_input
    # If candidate_vod_ids is set (GT pipeline: date-window filtered), only compare
    # this repost against its candidate VODs instead of the full VOD list.
    candidate_vod_ids = repost.get("candidate_vod_ids")
    if candidate_vod_ids:
        candidate_set = {str(v) for v in candidate_vod_ids}
        active_vods = [vt for vt in vod_transcripts if str(vt.row.get("vod_id") or "") in candidate_set]
    else:
        active_vods = vod_transcripts
    best = _best_transcript_match(
        repost,
        repost_text=repost_text,
        repost_tokens=repost_tokens,
        repost_duration=repost_duration,
        vod_transcripts=active_vods,
    )
    chunk_matches: list[TranscriptMatch] = []
    for chunk in repost_chunks:
        chunk_best = _best_transcript_match(
            repost,
            repost_text=chunk.text,
            repost_tokens=chunk.tokens,
            repost_duration=chunk.duration,
            vod_transcripts=active_vods,
            match_kind="multi_window_chunk",
            chunk_index=chunk.index,
            chunk_count=chunk.count,
        )
        if chunk_best is not None:
            chunk_matches.append(chunk_best)
    return index, repost, RepostTranscriptMatchResult(
        best=best,
        chunk_matches=tuple(chunk_matches),
        repost_duration=float(repost_duration or 0.0),
    )


def _load_vod_transcript(vod: dict, *, vod_cache_root: pathlib.Path) -> VodTranscript | None:
    vod_id = str(vod.get("vod_id") or "").strip()
    if not vod_id:
        return None
    loaded = _load_compatible_transcript_cache(vod_cache_root / vod_id, vod_id)
    if loaded is None:
        return None
    _path, transcript = loaded
    segments = [segment for segment in transcript if str(getattr(segment, "text", "") or "").strip()]
    if not segments:
        return None
    return _build_vod_transcript(vod, segments)


def _build_vod_transcript(vod: dict, segments: list) -> VodTranscript:
    segment_tokens = [_tokens(str(getattr(segment, "text", "") or "")) for segment in segments]
    token_to_indices: dict[str, list[int]] = {}
    for index, tokens in enumerate(segment_tokens):
        for token in set(tokens):
            token_to_indices.setdefault(token, []).append(index)
    n_segments = max(len(segments), 1)
    token_idf = {
        token: math.log(1.0 + n_segments / len(indices))
        for token, indices in token_to_indices.items()
    }
    return VodTranscript(
        row=vod,
        segments=segments,
        segment_tokens=segment_tokens,
        token_to_indices=token_to_indices,
        token_idf=token_idf,
    )


def _load_or_build_repost_transcript(
    repost: dict,
    video_path: pathlib.Path,
    *,
    cache_root: pathlib.Path,
) -> tuple[list, bool]:
    cache_path = _repost_transcript_cache_path(repost, video_path, cache_root=cache_root)
    cached = _load_repost_transcript_cache(cache_path)
    if cached is not None:
        return cached, True
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path = cache_path.with_suffix(".wav")
    try:
        audio_extractor.extract(
            video_path,
            f"cross_platform_transcript:{cache_path.stem[:12]}",
            output_path=audio_path,
        )
        transcript = transcriber.transcribe(audio_path, enable_diarization=False)
    finally:
        try:
            audio_path.unlink()
        except OSError:
            pass
    payload = {
        "metadata": _repost_transcript_cache_payload(repost, video_path),
        "transcript": [_serialize_rich_segment(segment) for segment in transcript],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return list(transcript), False


def _load_repost_transcript_cache(path: pathlib.Path) -> list | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if int(metadata.get("version") or 0) != _SHORT_TRANSCRIPT_CACHE_VERSION:
        return None
    if str(metadata.get("whisper_model") or "") != str(settings.whisper_model):
        return None
    if str(metadata.get("whisper_language") or "").strip().lower() != str(settings.whisper_language or "").strip().lower():
        return None
    try:
        return [_deserialize_rich_segment(item) for item in list(payload.get("transcript") or [])]
    except (TypeError, ValueError):
        return None


def _repost_transcript_cache_path(repost: dict, video_path: pathlib.Path, *, cache_root: pathlib.Path) -> pathlib.Path:
    payload = _repost_transcript_cache_payload(repost, video_path)
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return cache_root / f"{digest}.json"


def _repost_transcript_cache_payload(repost: dict, video_path: pathlib.Path) -> dict[str, object]:
    try:
        stat = video_path.stat()
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        size = 0
        mtime_ns = 0
    return {
        "version": _SHORT_TRANSCRIPT_CACHE_VERSION,
        "kind": "short_form_transcript",
        "source_url": str(repost.get("source_url") or ""),
        "video_path": str(video_path),
        "video_size": size,
        "video_mtime_ns": mtime_ns,
        "whisper_model": str(settings.whisper_model),
        "whisper_language": str(settings.whisper_language or "").strip().lower(),
    }


def _repost_fingerprint_cache_path(repost: dict, video_path: pathlib.Path, *, cache_root: pathlib.Path) -> pathlib.Path:
    try:
        stat = video_path.stat()
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        size = 0
        mtime_ns = 0
    payload = {
        "kind": "short_form_fingerprint",
        "source_url": str(repost.get("source_url") or ""),
        "video_path": str(video_path),
        "video_size": size,
        "video_mtime_ns": mtime_ns,
    }
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return cache_root / f"{digest}.json"


def _load_or_build_repost_fingerprint(
    repost: dict,
    video_path: pathlib.Path,
    fpcalc: str,
    *,
    fp_cache_root: pathlib.Path,
) -> audio_matcher.Fingerprint:
    cache_path = _repost_fingerprint_cache_path(repost, video_path, cache_root=fp_cache_root)
    if cache_path.exists():
        try:
            fp = audio_matcher._read_fingerprint_file(cache_path)
            if fp.values:
                return fp
        except Exception:
            pass
    fp = audio_matcher._fingerprint(video_path, fpcalc)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"duration": float(fp.duration), "fingerprint": fp.values}, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass
    return fp


def _repost_transcript_chunks(transcript: list, *, total_duration: float) -> tuple[RepostTranscriptChunk, ...]:
    segments = [segment for segment in transcript if _tokens(str(getattr(segment, "text", "") or ""))]
    if len(segments) < 2:
        return ()
    all_text = _segments_text(segments)
    all_tokens = _tokens(all_text)
    duration = max(float(total_duration or 0.0), _transcript_duration(segments, fallback=0.0))
    if len(all_tokens) < 14 or duration < 8.0:
        return ()

    raw_chunks: list[tuple[str, list[str], float]] = []
    current: list = []
    current_start: float | None = None
    for segment in segments:
        if current_start is None:
            current_start = float(getattr(segment, "start", 0.0) or 0.0)
        current.append(segment)
        text = _segments_text(current)
        tokens = _tokens(text)
        chunk_end = float(getattr(segment, "end", current_start + 1.0) or current_start + 1.0)
        chunk_duration = max(chunk_end - float(current_start), 1.0)
        should_close = (
            (len(tokens) >= 8 and chunk_duration >= 4.0)
            or len(tokens) >= 28
            or chunk_duration >= 22.0
        )
        if should_close:
            raw_chunks.append((text, tokens, chunk_duration))
            current = []
            current_start = None
    if current:
        text = _segments_text(current)
        tokens = _tokens(text)
        if len(tokens) >= 6:
            first_start = float(getattr(current[0], "start", 0.0) or 0.0)
            last_end = float(getattr(current[-1], "end", first_start + 1.0) or first_start + 1.0)
            raw_chunks.append((text, tokens, max(last_end - first_start, 1.0)))

    filtered = [(text, tokens, chunk_duration) for text, tokens, chunk_duration in raw_chunks if len(tokens) >= 6]
    if len(filtered) < 2:
        return ()
    count = len(filtered)
    return tuple(
        RepostTranscriptChunk(
            index=index,
            count=count,
            text=text,
            tokens=tokens,
            duration=chunk_duration,
        )
        for index, (text, tokens, chunk_duration) in enumerate(filtered, start=1)
    )


def _best_transcript_match(
    repost: dict,
    *,
    repost_text: str,
    repost_tokens: list[str],
    repost_duration: float,
    vod_transcripts: list[VodTranscript],
    match_kind: str = "single_window",
    chunk_index: int | None = None,
    chunk_count: int = 1,
) -> TranscriptMatch | None:
    best: TranscriptMatch | None = None
    top_candidates: list[TranscriptMatch] = []
    target_duration = max(float(repost_duration or 0.0), _safe_float(repost.get("duration")), 8.0)
    for vod_transcript in vod_transcripts:
        for start_index in _candidate_start_indices(vod_transcript, repost_tokens):
            candidate = _score_window(
                vod_transcript,
                start_index,
                repost_text=repost_text,
                repost_tokens=repost_tokens,
                target_duration=target_duration,
                match_kind=match_kind,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
            )
            if candidate is None:
                continue
            if best is None or _ranking_score(candidate) > _ranking_score(best):
                best = candidate
            top_candidates.append(candidate)
            if len(top_candidates) > 24:
                top_candidates.sort(key=_ranking_score, reverse=True)
                del top_candidates[12:]
    if best is None:
        return None
    return _flag_ambiguous_position(best, top_candidates)


def _ranking_score(match: TranscriptMatch) -> float:
    return float(match.rank_score if match.rank_score is not None else match.score)


def _flag_ambiguous_position(best: TranscriptMatch, candidates: list[TranscriptMatch]) -> TranscriptMatch:
    """Mark the best window ambiguous when a competing position (different VOD
    or far away in the same VOD) scores almost as high."""
    best_vod = str(best.vod.get("vod_id") or "")
    second_score: float | None = None
    for candidate in candidates:
        if candidate is best:
            continue
        same_vod = str(candidate.vod.get("vod_id") or "") == best_vod
        if same_vod and abs(float(candidate.start) - float(best.start)) <= _POSITION_AMBIGUITY_DISTANCE_SEC:
            continue
        if second_score is None or float(candidate.score) > second_score:
            second_score = float(candidate.score)
    if second_score is None or float(best.score) - second_score >= _POSITION_AMBIGUITY_MARGIN:
        return best
    return _copy_match(best, position_ambiguous=True)


def _copy_match(match: TranscriptMatch, **overrides: Any) -> TranscriptMatch:
    return dataclass_replace(match, **overrides)


def _candidate_start_indices(vod_transcript: VodTranscript, repost_tokens: list[str]) -> list[int]:
    counts: Counter[int] = Counter()
    for token in set(repost_tokens):
        if token in _STOPWORDS:
            continue
        for index in vod_transcript.token_to_indices.get(token, []):
            counts[index] += 1
    if not counts:
        return []
    starts: set[int] = set()
    for index, _count in counts.most_common(500):
        for delta in (-3, -2, -1, 0, 1, 2, 3):
            candidate = index + delta
            if 0 <= candidate < len(vod_transcript.segments):
                starts.add(candidate)
    return sorted(starts)


def _score_window(
    vod_transcript: VodTranscript,
    start_index: int,
    *,
    repost_text: str,
    repost_tokens: list[str],
    target_duration: float,
    match_kind: str = "single_window",
    chunk_index: int | None = None,
    chunk_count: int = 1,
) -> TranscriptMatch | None:
    segments = vod_transcript.segments
    if start_index >= len(segments):
        return None
    window_start = max(float(getattr(segments[start_index], "start", 0.0) or 0.0), 0.0)
    max_duration = max(target_duration * 1.6, target_duration + 15.0, 20.0)
    selected: list = []
    for segment in segments[start_index:]:
        segment_end = float(getattr(segment, "end", 0.0) or 0.0)
        if segment_end - window_start > max_duration and selected:
            break
        selected.append(segment)
        if segment_end - window_start >= target_duration:
            break
    if not selected:
        return None
    window_end = max(float(getattr(selected[-1], "end", window_start + 1.0) or window_start + 1.0), window_start + 1.0)
    window_text = _segments_text(selected)
    window_tokens = _tokens(window_text)
    if not window_tokens:
        return None
    repost_counter = Counter(repost_tokens)
    window_counter = Counter(window_tokens)
    token_recall = _counter_recall(repost_counter, window_counter)
    bigram_recall = _counter_recall(Counter(_bigrams(repost_tokens)), Counter(_bigrams(window_tokens)))
    sequence_ratio = 0.0
    if token_recall >= 0.15 or bigram_recall >= 0.10:
        sequence_ratio = difflib.SequenceMatcher(None, _compact_text(repost_text), _compact_text(window_text)).ratio()
    score = (0.55 * token_recall) + (0.35 * bigram_recall) + (0.10 * sequence_ratio)
    # Rank candidates with rare-token emphasis so catchphrase-heavy windows
    # lose against content-specific ones; acceptance uses the plain score.
    idf_recall = _idf_weighted_recall(repost_counter, window_counter, vod_transcript.token_idf)
    rank_score = (0.55 * idf_recall) + (0.35 * bigram_recall) + (0.10 * sequence_ratio)
    return TranscriptMatch(
        score=round(max(0.0, min(1.0, score)), 6),
        rank_score=round(max(0.0, min(1.0, rank_score)), 6),
        token_recall=round(token_recall, 6),
        bigram_recall=round(bigram_recall, 6),
        sequence_ratio=round(sequence_ratio, 6),
        start=window_start,
        end=window_end,
        vod=vod_transcript.row,
        vod_excerpt=_short_text(window_text, 260),
        repost_excerpt=_short_text(repost_text, 260),
        match_kind=match_kind,
        source_score=round(max(0.0, min(1.0, score)), 6),
        chunk_index=chunk_index,
        chunk_count=max(1, int(chunk_count or 1)),
    )


def _acceptance_decision(repost: dict, match: TranscriptMatch, base_threshold: float) -> MatchDecision:
    threshold = max(float(base_threshold), 0.0)
    if match.score >= threshold:
        return MatchDecision(True, "transcript_threshold", threshold)
    view_threshold = _view_weighted_similarity_threshold(repost, match, threshold)
    if view_threshold < threshold and match.score >= view_threshold:
        return MatchDecision(True, "view_weighted_transcript_threshold", view_threshold)
    return MatchDecision(False, "below_threshold", threshold)


def _view_weighted_similarity_threshold(repost: dict, match: TranscriptMatch, base_threshold: float) -> float:
    views = _safe_float(repost.get("views") or repost.get("view_count"))
    threshold = max(float(base_threshold), 0.0)
    if views >= 1_000_000 and match.token_recall >= 0.40:
        return min(threshold, 0.42)
    if views >= 500_000 and match.token_recall >= 0.42:
        return min(threshold, 0.45)
    if views >= 100_000 and match.token_recall >= 0.45:
        return min(threshold, 0.48)
    return threshold


def _with_acceptance(match: TranscriptMatch, decision: MatchDecision) -> TranscriptMatch:
    return dataclass_replace(
        match,
        acceptance_reason=decision.reason,
        acceptance_threshold=round(float(decision.threshold), 6),
        source_score=match.source_score if match.source_score is not None else match.score,
    )


def _append_unique_match(matches: list[TranscriptMatch], candidate: TranscriptMatch) -> bool:
    for existing in matches:
        if str(existing.vod.get("vod_id") or "") != str(candidate.vod.get("vod_id") or ""):
            continue
        if _window_overlap_ratio(existing.start, existing.end, candidate.start, candidate.end) >= 0.70:
            return False
    matches.append(candidate)
    return True


def _clean_accepted_matches(
    matches: list[TranscriptMatch],
    *,
    repost_duration: float = 0.0,
) -> tuple[list[TranscriptMatch], dict[str, int]]:
    stats = {
        "multi_window_cross_vod_filtered": 0,
        "multi_window_near_duplicate_merges": 0,
        "multi_window_merge_span_capped": 0,
    }
    if len(matches) <= 1:
        return list(matches), stats

    filtered = _filter_multi_window_matches_to_top_vod(matches, stats)
    cleaned = _merge_near_duplicate_accepted_matches(filtered, stats, repost_duration=repost_duration)
    return cleaned, stats


def _filter_multi_window_matches_to_top_vod(
    matches: list[TranscriptMatch],
    stats: dict[str, int],
) -> list[TranscriptMatch]:
    vod_ids = {_match_vod_id(match) for match in matches if _match_vod_id(match)}
    if len(vod_ids) <= 1:
        return list(matches)

    vod_scores: dict[str, tuple[float, float, float, float, int]] = {}
    for match in matches:
        vod_id = _match_vod_id(match)
        if not vod_id:
            continue
        strong_evidence, max_token_recall, max_score, total_score, count = vod_scores.get(
            vod_id,
            (0.0, 0.0, 0.0, 0.0, 0),
        )
        source_score = float(match.source_score if match.source_score is not None else match.score)
        token_recall = float(match.token_recall)
        score = float(match.score)
        strong_weight = _dominant_vod_strong_match_weight(match)
        vod_scores[vod_id] = (
            strong_evidence + strong_weight,
            max(max_token_recall, token_recall),
            max(max_score, score),
            total_score + max(source_score, 0.0),
            count + 1,
        )
    if not vod_scores:
        return list(matches)

    top_vod_id = max(vod_scores.items(), key=lambda item: (*item[1], item[0]))[0]
    filtered = [
        match
        for match in matches
        if _match_vod_id(match) == top_vod_id or _is_strong_cross_vod_secondary_match(match)
    ]
    stats["multi_window_cross_vod_filtered"] += max(0, len(matches) - len(filtered))
    return filtered


def _is_strong_cross_vod_secondary_match(match: TranscriptMatch) -> bool:
    return float(match.token_recall) >= _CROSS_VOD_WEAK_TOKEN_RECALL_THRESHOLD


def _dominant_vod_strong_match_weight(match: TranscriptMatch) -> float:
    token_recall = float(match.token_recall)
    score = float(match.score)
    if token_recall < _CROSS_VOD_WEAK_TOKEN_RECALL_THRESHOLD and score < _CROSS_VOD_WEAK_TOKEN_RECALL_THRESHOLD:
        return 0.0
    kind = str(match.match_kind or "")
    if kind in {"single_window", "audio_fingerprint_fallback"}:
        return 1.5
    return 1.0


def _merge_near_duplicate_accepted_matches(
    matches: list[TranscriptMatch],
    stats: dict[str, int],
    *,
    repost_duration: float = 0.0,
) -> list[TranscriptMatch]:
    if len(matches) <= 1:
        return list(matches)

    cleaned: list[TranscriptMatch] = []
    for _vod_id, group in _accepted_matches_by_vod(matches).items():
        ordered = sorted(group, key=lambda match: (float(match.start), float(match.end), -float(match.score)))
        cluster: list[TranscriptMatch] = []
        cluster_end = 0.0
        for match in ordered:
            if not cluster:
                cluster = [match]
                cluster_end = float(match.end)
                continue
            gap = float(match.start) - float(cluster_end)
            if gap < _MULTI_WINDOW_NEAR_DUPLICATE_MERGE_GAP_SEC:
                cluster.append(match)
                cluster_end = max(cluster_end, float(match.end))
                continue
            cleaned.append(_merge_match_cluster(cluster, stats, repost_duration=repost_duration))
            cluster = [match]
            cluster_end = float(match.end)
        if cluster:
            cleaned.append(_merge_match_cluster(cluster, stats, repost_duration=repost_duration))
    return sorted(cleaned, key=lambda match: (_match_vod_id(match), float(match.start), float(match.end)))


def _accepted_matches_by_vod(matches: list[TranscriptMatch]) -> dict[str, list[TranscriptMatch]]:
    grouped: dict[str, list[TranscriptMatch]] = {}
    for match in matches:
        grouped.setdefault(_match_vod_id(match), []).append(match)
    return grouped


def _merge_match_cluster(
    matches: list[TranscriptMatch],
    stats: dict[str, int],
    *,
    repost_duration: float = 0.0,
) -> TranscriptMatch:
    if len(matches) == 1:
        return matches[0]
    stats["multi_window_near_duplicate_merges"] += len(matches) - 1
    best = max(matches, key=lambda match: float(match.score))
    start = min(float(match.start) for match in matches)
    end = max(float(match.end) for match in matches)
    if repost_duration > 0:
        # The merged GT window may not exceed the repost length by much —
        # uncapped [min(starts), max(ends)] merges produced 84s windows for
        # <=60s Shorts, which poisons the coverage-based eval.
        span_cap = max(
            repost_duration * _MULTI_WINDOW_MERGE_SPAN_FACTOR + _MULTI_WINDOW_MERGE_SPAN_PAD_SEC,
            float(best.end) - float(best.start),
        )
        if end - start > span_cap:
            stats["multi_window_merge_span_capped"] = stats.get("multi_window_merge_span_capped", 0) + 1
            slack = span_cap - (float(best.end) - float(best.start))
            start = max(start, float(best.start) - slack / 2)
            end = min(end, start + span_cap)
    return TranscriptMatch(
        score=max(float(match.score) for match in matches),
        token_recall=max(float(match.token_recall) for match in matches),
        bigram_recall=max(float(match.bigram_recall) for match in matches),
        sequence_ratio=max(float(match.sequence_ratio) for match in matches),
        start=start,
        end=end,
        vod=best.vod,
        vod_excerpt=_short_text(" | ".join(match.vod_excerpt for match in matches if match.vod_excerpt), 260),
        repost_excerpt=_short_text(" | ".join(match.repost_excerpt for match in matches if match.repost_excerpt), 260),
        match_kind="multi_window_merged",
        acceptance_reason="multi_window_near_duplicate_merge",
        acceptance_threshold=min(float(match.acceptance_threshold or 0.0) for match in matches),
        source_score=max(float(match.source_score if match.source_score is not None else match.score) for match in matches),
        chunk_index=None,
        chunk_count=max(int(match.chunk_count or 1) for match in matches),
        audio_confidence=max(
            (float(match.audio_confidence) for match in matches if match.audio_confidence is not None),
            default=None,
        ),
        rank_score=max(
            (float(match.rank_score) for match in matches if match.rank_score is not None),
            default=None,
        ),
        position_ambiguous=any(match.position_ambiguous for match in matches),
    )


def _match_vod_id(match: TranscriptMatch) -> str:
    return str(match.vod.get("vod_id") or match.vod.get("id") or match.vod.get("vod_url") or "").strip()


def _window_overlap_ratio(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    # Bewusst NICHT an row_identity.window_overlap_ratio delegiert: der GT-Matcher
    # nudged entartete Fenster (end = max(end, start + 0.001)) zero-length-tolerant,
    # was für solche Fenster ein anderes Ergebnis liefert als die strikte
    # row_identity-Geometrie (zero-length -> 0.0). Dieser Kern prägt Gold und bleibt
    # darum eigenständig.
    left_end = max(float(left_end), float(left_start) + 0.001)
    right_end = max(float(right_end), float(right_start) + 0.001)
    overlap = max(0.0, min(left_end, right_end) - max(float(left_start), float(right_start)))
    shortest = max(0.001, min(left_end - float(left_start), right_end - float(right_start)))
    return overlap / shortest


def _should_try_audio_fallback(repost: dict, best: TranscriptMatch, base_threshold: float) -> bool:
    # Audio fingerprinting needs no speech, so the fallback must not be gated
    # on transcript quality: low-speech clips (screams, mod trolls, visual
    # gags) score near zero on transcripts and are exactly the ones the
    # fallback exists for. Precision is guarded by
    # _audio_fallback_confidence_threshold instead.
    return best.score < float(base_threshold)


def _audio_fallback_confidence_threshold(repost: dict, best: TranscriptMatch) -> float:
    views = _safe_float(repost.get("views") or repost.get("view_count"))
    if views >= 1_000_000 and best.token_recall >= 0.40:
        return 0.52
    if views >= 500_000 and best.token_recall >= 0.35:
        return 0.54
    return 0.56


def _load_audio_vod_fingerprints(vod_rows: list[dict]) -> list[tuple[dict, audio_matcher.Fingerprint]]:
    fpcalc = shutil.which("fpcalc")
    if fpcalc is None:
        return []
    vod_fps: list[tuple[dict, audio_matcher.Fingerprint]] = []
    for vod in vod_rows:
        try:
            vod_fps.append((vod, audio_matcher._fingerprint_from_row(vod, fpcalc)))
        except Exception:
            continue
    return vod_fps


def _audio_fingerprint_fallback(
    repost: dict,
    transcript_best: TranscriptMatch,
    *,
    vod_fps: list[tuple[dict, audio_matcher.Fingerprint]],
    fp_cache_root: pathlib.Path | None = None,
) -> TranscriptMatch | None:
    if not vod_fps:
        return None
    fpcalc = shutil.which("fpcalc")
    if fpcalc is None:
        return None
    repost_path = audio_matcher._resolve_media_path(str(repost.get("video_path") or ""))
    if repost_path is None:
        return None
    try:
        if fp_cache_root is not None:
            repost_fp = _load_or_build_repost_fingerprint(repost, repost_path, fpcalc, fp_cache_root=fp_cache_root)
        else:
            repost_fp = audio_matcher._fingerprint(repost_path, fpcalc)
    except Exception:
        return None
    best_audio: tuple[float, int, dict, audio_matcher.Fingerprint] | None = None
    for vod, vod_fp in vod_fps:
        confidence, offset_index = audio_matcher.best_fingerprint_alignment(repost_fp.values, vod_fp.values)
        if best_audio is None or confidence > best_audio[0]:
            best_audio = (confidence, offset_index, vod, vod_fp)
    if best_audio is None:
        return None
    confidence, offset_index, vod, vod_fp = best_audio
    if confidence < _audio_fallback_confidence_threshold(repost, transcript_best):
        return None
    peak_margin = _audio_peak_margin(repost_fp, vod_fp, confidence, offset_index)
    if peak_margin < _AUDIO_FALLBACK_MIN_PEAK_MARGIN:
        # The best alignment barely beats the second-best peak elsewhere in
        # the VOD — over a multi-hour fingerprint the max of pure noise sits
        # at this level (measured 2026-06: noise margins median 0.007), so
        # this "match" is most likely a random peak, not the repost.
        return None
    seconds_per_item = vod_fp.duration / max(len(vod_fp.values), 1)
    scoped_start = max(offset_index * seconds_per_item, 0.0)
    source_offset = _safe_float(
        vod.get("source_start_offset")
        or vod.get("expanded_start")
        or vod.get("window_start_offset")
    )
    start = max(source_offset + scoped_start, 0.0)
    end = start + max(repost_fp.duration, 1.0)
    return TranscriptMatch(
        score=transcript_best.score,
        token_recall=transcript_best.token_recall,
        bigram_recall=transcript_best.bigram_recall,
        sequence_ratio=transcript_best.sequence_ratio,
        start=start,
        end=end,
        vod=vod,
        vod_excerpt=transcript_best.vod_excerpt,
        repost_excerpt=transcript_best.repost_excerpt,
        match_kind="audio_fingerprint_fallback",
        source_score=transcript_best.score,
        audio_confidence=round(float(confidence), 6),
    )


def _audio_peak_margin(
    repost_fp: "audio_matcher.Fingerprint",
    vod_fp: "audio_matcher.Fingerprint",
    best_confidence: float,
    best_index: int,
) -> float:
    """Peak sharpness: best alignment vs the best peak outside ±60s of it."""
    values = vod_fp.values
    if not values or vod_fp.duration <= 0:
        return 0.0
    rate = len(values) / vod_fp.duration
    exclusion = int(60.0 * rate)
    second = 0.0
    for part in (values[: max(best_index - exclusion, 0)], values[best_index + exclusion:]):
        if len(part) >= len(repost_fp.values):
            confidence, _index = audio_matcher.best_fingerprint_alignment(repost_fp.values, part)
            second = max(second, confidence)
    return float(best_confidence) - second


def _matched_row(repost: dict, match: TranscriptMatch) -> dict:
    views = _safe_float(repost.get("views") or repost.get("view_count"))
    vod = match.vod
    title = _repost_title(repost)
    row = {
        "vod_id": str(vod.get("vod_id") or ""),
        "vod_url": str(vod.get("vod_url") or ""),
        "candidate_id": str(repost.get("clip_id") or repost.get("candidate_id") or f"cross_platform:{vod.get('vod_id')}:{match.start:.1f}"),
        "window_start": round(match.start, 2),
        "window_end": round(max(match.end, match.start + 1.0), 2),
        "label": 1,
        "selected": True,
        "label_class": "cross_platform_outcome_positive",
        "dataset_source": TRANSCRIPT_MATCH_SOURCE,
        "label_source": TRANSCRIPT_MATCH_SOURCE,
        "supervision_tier": "cross_platform_outcome",
        "sample_weight": 2.0 if views >= 100_000 else 1.5,
        "platform": str(repost.get("platform") or ""),
        "source_url": str(repost.get("source_url") or ""),
        "repost_title": title,
        "views": views,
        "transcript_similarity": round(match.score, 6),
        "transcript_token_recall": round(match.token_recall, 6),
        "transcript_bigram_recall": round(match.bigram_recall, 6),
        "transcript_sequence_ratio": round(match.sequence_ratio, 6),
        "transcript_match_kind": str(match.match_kind or "single_window"),
        "cross_platform_verification": str(match.match_kind or "single_window"),
        "transcript_acceptance_reason": str(match.acceptance_reason or "transcript_threshold"),
        "transcript_acceptance_threshold": round(float(match.acceptance_threshold or match.score), 6),
        "transcript_source_score": round(float(match.source_score if match.source_score is not None else match.score), 6),
        "transcript_repost_excerpt": match.repost_excerpt,
        "transcript_vod_excerpt": match.vod_excerpt,
        "transcript_position_ambiguous": bool(match.position_ambiguous),
        "profile_filter_version": int(_safe_float(repost.get("profile_filter_version"))),
    }
    if match.chunk_index is not None:
        row["transcript_chunk_index"] = int(match.chunk_index)
        row["transcript_chunk_count"] = int(match.chunk_count or 1)
    if match.audio_confidence is not None:
        row["fingerprint_confidence"] = round(float(match.audio_confidence), 6)
        row["audio_fingerprint_confidence"] = round(float(match.audio_confidence), 6)
    return outcome_view_weighting.annotate_cross_platform_view_weight(row)


def _best_match_diagnostic_row(
    repost: dict,
    match: TranscriptMatch,
    *,
    accepted: bool,
    accepted_rows: int = 1,
    accepted_match_kinds: list[str] | None = None,
    accepted_reasons: list[str] | None = None,
) -> dict:
    title = _repost_title(repost)
    row = {
        "accepted": bool(accepted),
        "accepted_rows": int(accepted_rows),
        "accepted_match_kinds": list(accepted_match_kinds or []),
        "accepted_reasons": list(accepted_reasons or []),
        "confidence": round(match.score, 6),
        "transcript_similarity": round(match.score, 6),
        "transcript_token_recall": round(match.token_recall, 6),
        "transcript_bigram_recall": round(match.bigram_recall, 6),
        "transcript_sequence_ratio": round(match.sequence_ratio, 6),
        "transcript_match_kind": str(match.match_kind or "single_window"),
        "transcript_acceptance_reason": str(match.acceptance_reason or ("transcript_threshold" if accepted else "below_threshold")),
        "transcript_acceptance_threshold": round(float(match.acceptance_threshold or match.score), 6),
        "transcript_source_score": round(float(match.source_score if match.source_score is not None else match.score), 6),
        "vod_id": str(match.vod.get("vod_id") or ""),
        "vod_url": str(match.vod.get("vod_url") or ""),
        "window_start": round(match.start, 2),
        "window_end": round(max(match.end, match.start + 1.0), 2),
        "repost_clip_id": str(repost.get("clip_id") or repost.get("candidate_id") or ""),
        "repost_source_url": str(repost.get("source_url") or ""),
        "repost_title": title,
        "repost_platform": str(repost.get("platform") or ""),
        "repost_views": _safe_float(repost.get("views") or repost.get("view_count")),
        "transcript_repost_excerpt": match.repost_excerpt,
        "transcript_vod_excerpt": match.vod_excerpt,
    }
    if match.chunk_index is not None:
        row["transcript_chunk_index"] = int(match.chunk_index)
        row["transcript_chunk_count"] = int(match.chunk_count or 1)
    if match.audio_confidence is not None:
        row["fingerprint_confidence"] = round(float(match.audio_confidence), 6)
        row["audio_fingerprint_confidence"] = round(float(match.audio_confidence), 6)
    return row


def _repost_title(repost: dict) -> str:
    for key in ("title", "repost_title", "short_title", "video_title"):
        value = str(repost.get(key) or "").strip()
        if value:
            return value
    return ""


def _match_diagnostics(
    *,
    vod_rows: list[dict],
    repost_rows: list[dict],
    vod_transcripts: list[VodTranscript],
    best_match_rows: list[dict],
    matched_rows: list[dict],
    min_similarity: float,
    diagnostic_top_k: int,
    skipped_vod_transcript: int,
    skipped_repost_media: int,
    skipped_repost_transcript: int,
    repost_transcript_cache_hits: int,
    repost_transcript_cache_misses: int,
    transcript_match_workers: int,
    audio_fallback_attempts: int = 0,
    audio_fallback_matches: int = 0,
    view_weighted_accepts: int = 0,
    multi_window_chunk_matches: int = 0,
    multi_window_cross_vod_filtered: int = 0,
    multi_window_near_duplicate_merges: int = 0,
    accepted_rows_before_cleanup: int = 0,
) -> dict[str, object]:
    scores = sorted(float(row.get("transcript_similarity") or row.get("confidence") or 0.0) for row in best_match_rows)
    top_rows = sorted(
        best_match_rows,
        key=lambda row: float(row.get("transcript_similarity") or row.get("confidence") or 0.0),
        reverse=True,
    )
    return {
        "verification_mode": "transcript",
        "min_confidence": round(float(min_similarity), 6),
        "min_similarity": round(float(min_similarity), 6),
        "vod_rows_input": len(vod_rows),
        "repost_rows_input": len(repost_rows),
        "vod_transcripts_loaded": len(vod_transcripts),
        "transcript_match_workers": max(1, int(transcript_match_workers or 1)),
        "repost_best_match_rows": len(best_match_rows),
        "matched_rows": len(matched_rows),
        "skipped_vod_transcript_rows": int(skipped_vod_transcript),
        "skipped_repost_media_rows": int(skipped_repost_media),
        "skipped_repost_transcript_rows": int(skipped_repost_transcript),
        "repost_transcript_cache_hits": int(repost_transcript_cache_hits),
        "repost_transcript_cache_misses": int(repost_transcript_cache_misses),
        "audio_fallback_attempts": int(audio_fallback_attempts),
        "audio_fallback_matches": int(audio_fallback_matches),
        "view_weighted_accepts": int(view_weighted_accepts),
        "multi_window_chunk_matches": int(multi_window_chunk_matches),
        "multi_window_cross_vod_filtered": int(multi_window_cross_vod_filtered),
        "multi_window_near_duplicate_merges": int(multi_window_near_duplicate_merges),
        "accepted_rows_before_cleanup": int(accepted_rows_before_cleanup),
        "confidence_distribution": audio_matcher._score_distribution(scores),
        "threshold_ladder": audio_matcher._threshold_ladder(scores),
        "best_match_rows": best_match_rows,
        "top_best_matches": top_rows[: max(0, int(diagnostic_top_k))],
        "recommendation": _match_recommendation(scores, min_similarity),
    }


def _match_recommendation(scores: list[float], min_similarity: float) -> dict[str, str]:
    if not scores:
        return {"status": "no_transcript_overlap", "message": "No transcript best-match rows were produced."}
    if max(scores) < float(min_similarity):
        return {
            "status": "threshold_too_high_or_no_text_match",
            "message": "Transcript matching produced candidates, but none reached the current threshold.",
        }
    return {
        "status": "threshold_produces_matches",
        "message": "Current transcript threshold produced cross-platform rows.",
    }


def _segments_text(segments: list) -> str:
    return " ".join(str(getattr(segment, "text", "") or "").strip() for segment in segments).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(str(text or "").lower()) if token and token not in _STOPWORDS]


def _bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:], strict=False))


def _counter_recall(needle: Counter, haystack: Counter) -> float:
    total = sum(needle.values())
    if total <= 0:
        return 0.0
    overlap = sum(min(count, haystack.get(token, 0)) for token, count in needle.items())
    return max(0.0, min(1.0, overlap / total))


def _idf_weighted_recall(needle: Counter, haystack: Counter, token_idf: dict[str, float]) -> float:
    """Token recall with rare-token emphasis: ubiquitous catchphrase tokens
    contribute almost nothing, content-specific tokens dominate. Used only to
    RANK candidate windows — acceptance thresholds stay on the unweighted
    score so they keep their calibration."""
    total = 0.0
    overlap = 0.0
    for token, count in needle.items():
        weight = token_idf.get(token, 1.0)
        total += count * weight
        overlap += min(count, haystack.get(token, 0)) * weight
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, overlap / total))


def _compact_text(text: str) -> str:
    return " ".join(_tokens(text))[:1200]


def _transcript_duration(transcript: list, *, fallback: float = 0.0) -> float:
    ends = [float(getattr(segment, "end", 0.0) or 0.0) for segment in transcript]
    return max(ends or [0.0, float(fallback or 0.0)])


def _log_repost_progress(
    *,
    index: int,
    total_reposts: int,
    output_rows: list[dict],
    best_match_rows: list[dict],
    skipped_repost_media: int,
    skipped_repost_transcript: int,
    repost_transcript_cache_hits: int,
    repost_transcript_cache_misses: int,
    best_similarity_seen: float,
    started_at: float,
) -> None:
    _log_event(
        "cross_platform_transcript_repost_progress",
        processed_reposts=index,
        total_reposts=total_reposts,
        matched_rows=len(output_rows),
        repost_best_match_rows=len(best_match_rows),
        skipped_repost_media=skipped_repost_media,
        skipped_repost_transcript=skipped_repost_transcript,
        repost_transcript_cache_hits=repost_transcript_cache_hits,
        repost_transcript_cache_misses=repost_transcript_cache_misses,
        best_similarity_seen=round(best_similarity_seen, 6),
        elapsed_sec=round(time.monotonic() - started_at, 1),
    )


def _should_log_progress(index: int, total: int, interval: int) -> bool:
    if total <= 0:
        return False
    if index == 1 or index >= total:
        return True
    return int(interval) > 0 and index % int(interval) == 0


def _log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=True), flush=True)


def _short_error(exc: BaseException, *, limit: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return _short_text(text, limit)


def _short_text(text: object, limit: int = 240) -> str:
    value = str(text or "")
    value = " ".join(value.split())
    return value if len(value) <= limit else f"{value[:limit - 3]}..."


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
