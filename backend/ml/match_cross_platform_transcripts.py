from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import json
import pathlib
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import build_twitch_training_dataset as vod_training
from ml import match_cross_platform_audio_fingerprints as audio_matcher
from services import audio_extractor, transcriber


DEFAULT_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"
DEFAULT_SHORT_TRANSCRIPT_CACHE_ROOT = pathlib.Path(settings.storage_dir) / "_download_cache" / "short_form_transcripts"
TRANSCRIPT_MATCH_SOURCE = "cross_platform_transcript_match"
_SHORT_TRANSCRIPT_CACHE_VERSION = 1
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


@dataclass
class VodTranscript:
    row: dict
    segments: list
    segment_tokens: list[list[str]]
    token_to_indices: dict[str, list[int]]


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
        audio_matcher._read_jsonl(pathlib.Path(args.vods)),
        audio_matcher._read_jsonl(pathlib.Path(args.reposts)),
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
    match_inputs: list[tuple[int, dict, str, list[str], float]] = []
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
        match_inputs.append((index, repost, repost_text, repost_tokens, repost_duration))

    effective_workers = min(requested_workers, max(1, len(match_inputs)))
    if log_progress and match_inputs:
        _log_event(
            "cross_platform_transcript_repost_match_workers",
            requested_workers=requested_workers,
            effective_workers=effective_workers,
            match_inputs=len(match_inputs),
        )

    for index, repost, best in _iter_repost_match_results(
        match_inputs,
        vod_transcripts=vod_transcripts,
        workers=effective_workers,
    ):
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
        accepted = best.score >= threshold
        best_match_rows.append(_best_match_diagnostic_row(repost, best, accepted=accepted))
        if accepted:
            output_rows.append(_matched_row(repost, best))
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
        )
    return output_rows, diagnostics


def _iter_repost_match_results(
    match_inputs: list[tuple[int, dict, str, list[str], float]],
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


def _match_repost_input_in_worker(match_input: tuple[int, dict, str, list[str], float]) -> tuple[int, dict, TranscriptMatch | None]:
    return _match_repost_input(match_input, vod_transcripts=_WORKER_VOD_TRANSCRIPTS)


def _match_repost_input(
    match_input: tuple[int, dict, str, list[str], float],
    *,
    vod_transcripts: list[VodTranscript],
) -> tuple[int, dict, TranscriptMatch | None]:
    index, repost, repost_text, repost_tokens, repost_duration = match_input
    best = _best_transcript_match(
        repost,
        repost_text=repost_text,
        repost_tokens=repost_tokens,
        repost_duration=repost_duration,
        vod_transcripts=vod_transcripts,
    )
    return index, repost, best


def _load_vod_transcript(vod: dict, *, vod_cache_root: pathlib.Path) -> VodTranscript | None:
    vod_id = str(vod.get("vod_id") or "").strip()
    if not vod_id:
        return None
    loaded = vod_training._load_compatible_transcript_cache(vod_cache_root / vod_id, vod_id)
    if loaded is None:
        return None
    _path, transcript = loaded
    segments = [segment for segment in transcript if str(getattr(segment, "text", "") or "").strip()]
    if not segments:
        return None
    segment_tokens = [_tokens(str(getattr(segment, "text", "") or "")) for segment in segments]
    token_to_indices: dict[str, list[int]] = {}
    for index, tokens in enumerate(segment_tokens):
        for token in set(tokens):
            token_to_indices.setdefault(token, []).append(index)
    return VodTranscript(row=vod, segments=segments, segment_tokens=segment_tokens, token_to_indices=token_to_indices)


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
        "transcript": [vod_training._serialize_rich_segment(segment) for segment in transcript],
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
        return [vod_training._deserialize_rich_segment(item) for item in list(payload.get("transcript") or [])]
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


def _best_transcript_match(
    repost: dict,
    *,
    repost_text: str,
    repost_tokens: list[str],
    repost_duration: float,
    vod_transcripts: list[VodTranscript],
) -> TranscriptMatch | None:
    best: TranscriptMatch | None = None
    target_duration = max(float(repost_duration or 0.0), _safe_float(repost.get("duration")), 8.0)
    for vod_transcript in vod_transcripts:
        for start_index in _candidate_start_indices(vod_transcript, repost_tokens):
            candidate = _score_window(
                vod_transcript,
                start_index,
                repost_text=repost_text,
                repost_tokens=repost_tokens,
                target_duration=target_duration,
            )
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
    return best


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
    token_recall = _counter_recall(Counter(repost_tokens), Counter(window_tokens))
    bigram_recall = _counter_recall(Counter(_bigrams(repost_tokens)), Counter(_bigrams(window_tokens)))
    sequence_ratio = 0.0
    if token_recall >= 0.15 or bigram_recall >= 0.10:
        sequence_ratio = difflib.SequenceMatcher(None, _compact_text(repost_text), _compact_text(window_text)).ratio()
    score = (0.55 * token_recall) + (0.35 * bigram_recall) + (0.10 * sequence_ratio)
    return TranscriptMatch(
        score=round(max(0.0, min(1.0, score)), 6),
        token_recall=round(token_recall, 6),
        bigram_recall=round(bigram_recall, 6),
        sequence_ratio=round(sequence_ratio, 6),
        start=window_start,
        end=window_end,
        vod=vod_transcript.row,
        vod_excerpt=_short_text(window_text, 260),
        repost_excerpt=_short_text(repost_text, 260),
    )


def _matched_row(repost: dict, match: TranscriptMatch) -> dict:
    views = _safe_float(repost.get("views") or repost.get("view_count"))
    vod = match.vod
    return {
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
        "views": views,
        "transcript_similarity": round(match.score, 6),
        "transcript_token_recall": round(match.token_recall, 6),
        "transcript_bigram_recall": round(match.bigram_recall, 6),
        "transcript_sequence_ratio": round(match.sequence_ratio, 6),
        "transcript_repost_excerpt": match.repost_excerpt,
        "transcript_vod_excerpt": match.vod_excerpt,
        "profile_filter_version": int(_safe_float(repost.get("profile_filter_version"))),
    }


def _best_match_diagnostic_row(repost: dict, match: TranscriptMatch, *, accepted: bool) -> dict:
    return {
        "accepted": bool(accepted),
        "confidence": round(match.score, 6),
        "transcript_similarity": round(match.score, 6),
        "transcript_token_recall": round(match.token_recall, 6),
        "transcript_bigram_recall": round(match.bigram_recall, 6),
        "transcript_sequence_ratio": round(match.sequence_ratio, 6),
        "vod_id": str(match.vod.get("vod_id") or ""),
        "vod_url": str(match.vod.get("vod_url") or ""),
        "window_start": round(match.start, 2),
        "window_end": round(max(match.end, match.start + 1.0), 2),
        "repost_clip_id": str(repost.get("clip_id") or repost.get("candidate_id") or ""),
        "repost_source_url": str(repost.get("source_url") or ""),
        "repost_platform": str(repost.get("platform") or ""),
        "repost_views": _safe_float(repost.get("views") or repost.get("view_count")),
        "transcript_repost_excerpt": match.repost_excerpt,
        "transcript_vod_excerpt": match.vod_excerpt,
    }


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
