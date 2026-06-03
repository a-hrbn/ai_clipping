from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import outcome_view_weighting


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match local TikTok/Shorts repost files back to VOD timestamps with Chromaprint/fpcalc."
    )
    parser.add_argument("--vods", required=True, help="JSONL rows with vod_id and video_path.")
    parser.add_argument("--reposts", required=True, help="JSONL rows with video_path, platform, views, optional clip_id.")
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_rows.jsonl"))
    parser.add_argument("--min-confidence", type=float, default=0.60)
    args = parser.parse_args()

    rows, _diagnostics = match_reposts_to_vods_with_diagnostics(
        _read_jsonl(pathlib.Path(args.vods)),
        _read_jsonl(pathlib.Path(args.reposts)),
        min_confidence=args.min_confidence,
        log_progress=True,
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
    min_confidence: float = 0.60,
) -> list[dict]:
    rows, _diagnostics = match_reposts_to_vods_with_diagnostics(
        vod_rows,
        repost_rows,
        min_confidence=min_confidence,
    )
    return rows


def match_reposts_to_vods_with_diagnostics(
    vod_rows: list[dict],
    repost_rows: list[dict],
    *,
    min_confidence: float = 0.60,
    diagnostic_top_k: int = 25,
    log_progress: bool = False,
    vod_progress_interval: int = 25,
    repost_progress_interval: int = 5,
) -> tuple[list[dict], dict[str, object]]:
    started_at = time.monotonic()
    if log_progress:
        _log_event(
            "cross_platform_fingerprint_match_start",
            vod_rows=len(vod_rows),
            repost_rows=len(repost_rows),
            min_confidence=round(float(min_confidence), 6),
            diagnostic_top_k=int(diagnostic_top_k),
        )
    fpcalc = shutil.which("fpcalc")
    if fpcalc is None:
        if log_progress:
            _log_event("cross_platform_fingerprint_blocked", reason="missing_fpcalc")
        raise RuntimeError("fpcalc not found. Install chromaprint-tools to enable audio-fingerprint matching.")
    vod_fps: list[tuple[dict, Fingerprint]] = []
    skipped_vod_media = 0
    skipped_vod_fingerprint = 0
    total_vods = len(vod_rows)
    for index, vod in enumerate(vod_rows, start=1):
        try:
            vod_fingerprint = _fingerprint_from_row(vod, fpcalc)
        except FileNotFoundError:
            skipped_vod_media += 1
            if log_progress and skipped_vod_media <= 5:
                _log_event(
                    "cross_platform_fingerprint_vod_skip",
                    reason="missing_media",
                    vod_index=index,
                    vod_rows=total_vods,
                    vod_id=str(vod.get("vod_id") or ""),
                    video_path=str(vod.get("video_path") or ""),
                    fingerprint_path=str(vod.get("fingerprint_path") or ""),
                )
            if log_progress and _should_log_progress(index, total_vods, vod_progress_interval):
                _log_event(
                    "cross_platform_fingerprint_vod_progress",
                    processed_vods=index,
                    total_vods=total_vods,
                    vod_fingerprints_built=len(vod_fps),
                    skipped_vod_media=skipped_vod_media,
                    skipped_vod_fingerprint=skipped_vod_fingerprint,
                    elapsed_sec=round(time.monotonic() - started_at, 1),
                )
            continue
        except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
            skipped_vod_fingerprint += 1
            if log_progress and skipped_vod_fingerprint <= 5:
                _log_event(
                    "cross_platform_fingerprint_vod_skip",
                    reason="fingerprint_failed",
                    vod_index=index,
                    vod_rows=total_vods,
                    vod_id=str(vod.get("vod_id") or ""),
                    video_path=str(vod.get("video_path") or ""),
                    fingerprint_path=str(vod.get("fingerprint_path") or ""),
                    error=_short_error(exc),
                )
            if log_progress and _should_log_progress(index, total_vods, vod_progress_interval):
                _log_event(
                    "cross_platform_fingerprint_vod_progress",
                    processed_vods=index,
                    total_vods=total_vods,
                    vod_fingerprints_built=len(vod_fps),
                    skipped_vod_media=skipped_vod_media,
                    skipped_vod_fingerprint=skipped_vod_fingerprint,
                    elapsed_sec=round(time.monotonic() - started_at, 1),
                )
            continue
        vod_fps.append((vod, vod_fingerprint))
        if log_progress and _should_log_progress(index, total_vods, vod_progress_interval):
            _log_event(
                "cross_platform_fingerprint_vod_progress",
                processed_vods=index,
                total_vods=total_vods,
                vod_fingerprints_built=len(vod_fps),
                skipped_vod_media=skipped_vod_media,
                skipped_vod_fingerprint=skipped_vod_fingerprint,
                elapsed_sec=round(time.monotonic() - started_at, 1),
            )
    output_rows: list[dict] = []
    best_match_rows: list[dict] = []
    skipped_repost_media = 0
    skipped_repost_fingerprint = 0
    if log_progress:
        _log_event(
            "cross_platform_fingerprint_vods_ready",
            vod_rows=total_vods,
            vod_fingerprints_built=len(vod_fps),
            skipped_vod_media=skipped_vod_media,
            skipped_vod_fingerprint=skipped_vod_fingerprint,
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
    total_reposts = len(repost_rows)
    best_confidence_seen = 0.0
    for index, repost in enumerate(repost_rows, start=1):
        repost_path = _resolve_media_path(str(repost.get("video_path") or ""))
        if repost_path is None:
            skipped_repost_media += 1
            if log_progress and skipped_repost_media <= 5:
                _log_event(
                    "cross_platform_fingerprint_repost_skip",
                    reason="missing_media",
                    repost_index=index,
                    repost_rows=total_reposts,
                    source_url=str(repost.get("source_url") or ""),
                    video_path=str(repost.get("video_path") or ""),
                )
            if log_progress and _should_log_progress(index, total_reposts, repost_progress_interval):
                _log_repost_progress(
                    index=index,
                    total_reposts=total_reposts,
                    output_rows=output_rows,
                    best_match_rows=best_match_rows,
                    skipped_repost_media=skipped_repost_media,
                    skipped_repost_fingerprint=skipped_repost_fingerprint,
                    best_confidence_seen=best_confidence_seen,
                    started_at=started_at,
                )
            continue
        try:
            repost_fp = _fingerprint(repost_path, fpcalc)
        except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
            skipped_repost_fingerprint += 1
            if log_progress and skipped_repost_fingerprint <= 5:
                _log_event(
                    "cross_platform_fingerprint_repost_skip",
                    reason="fingerprint_failed",
                    repost_index=index,
                    repost_rows=total_reposts,
                    source_url=str(repost.get("source_url") or ""),
                    video_path=str(repost_path),
                    error=_short_error(exc),
                )
            if log_progress and _should_log_progress(index, total_reposts, repost_progress_interval):
                _log_repost_progress(
                    index=index,
                    total_reposts=total_reposts,
                    output_rows=output_rows,
                    best_match_rows=best_match_rows,
                    skipped_repost_media=skipped_repost_media,
                    skipped_repost_fingerprint=skipped_repost_fingerprint,
                    best_confidence_seen=best_confidence_seen,
                    started_at=started_at,
                )
            continue
        best: tuple[float, int, dict, Fingerprint] | None = None
        for vod, vod_fp in vod_fps:
            confidence, offset_index = best_fingerprint_alignment(repost_fp.values, vod_fp.values)
            if best is None or confidence > best[0]:
                best = (confidence, offset_index, vod, vod_fp)
        if best is None:
            if log_progress and _should_log_progress(index, total_reposts, repost_progress_interval):
                _log_repost_progress(
                    index=index,
                    total_reposts=total_reposts,
                    output_rows=output_rows,
                    best_match_rows=best_match_rows,
                    skipped_repost_media=skipped_repost_media,
                    skipped_repost_fingerprint=skipped_repost_fingerprint,
                    best_confidence_seen=best_confidence_seen,
                    started_at=started_at,
                )
            continue
        confidence, offset_index, vod, vod_fp = best
        best_confidence_seen = max(best_confidence_seen, float(confidence))
        seconds_per_item = vod_fp.duration / max(len(vod_fp.values), 1)
        scoped_start = max(offset_index * seconds_per_item, 0.0)
        source_offset = _safe_float(
            vod.get("source_start_offset")
            or vod.get("expanded_start")
            or vod.get("window_start_offset")
        )
        start = max(source_offset + scoped_start, 0.0)
        end = start + repost_fp.duration
        best_match = _best_match_diagnostic_row(
            repost,
            vod,
            start,
            end,
            confidence,
            accepted=confidence >= min_confidence,
        )
        best_match_rows.append(best_match)
        if confidence >= min_confidence:
            output_rows.append(_matched_row(repost, vod, start, end, confidence))
        if log_progress and _should_log_progress(index, total_reposts, repost_progress_interval):
            _log_repost_progress(
                index=index,
                total_reposts=total_reposts,
                output_rows=output_rows,
                best_match_rows=best_match_rows,
                skipped_repost_media=skipped_repost_media,
                skipped_repost_fingerprint=skipped_repost_fingerprint,
                best_confidence_seen=best_confidence_seen,
                started_at=started_at,
            )
    diagnostics = _match_diagnostics(
        vod_rows=vod_rows,
        repost_rows=repost_rows,
        vod_fps=vod_fps,
        best_match_rows=best_match_rows,
        matched_rows=output_rows,
        min_confidence=min_confidence,
        diagnostic_top_k=diagnostic_top_k,
        skipped_vod_media=skipped_vod_media,
        skipped_vod_fingerprint=skipped_vod_fingerprint,
        skipped_repost_media=skipped_repost_media,
        skipped_repost_fingerprint=skipped_repost_fingerprint,
    )
    if log_progress:
        _log_event(
            "cross_platform_fingerprint_match_done",
            vod_rows=len(vod_rows),
            repost_rows=len(repost_rows),
            vod_fingerprints_built=len(vod_fps),
            repost_best_match_rows=len(best_match_rows),
            matched_rows=len(output_rows),
            skipped_vod_media=skipped_vod_media,
            skipped_vod_fingerprint=skipped_vod_fingerprint,
            skipped_repost_media=skipped_repost_media,
            skipped_repost_fingerprint=skipped_repost_fingerprint,
            best_confidence_seen=round(best_confidence_seen, 6),
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
    return output_rows, diagnostics


def _should_log_progress(done: int, total: int, interval: int) -> bool:
    if total <= 0:
        return False
    interval = max(1, int(interval))
    return done == 1 or done == total or done % interval == 0


def _log_repost_progress(
    *,
    index: int,
    total_reposts: int,
    output_rows: list[dict],
    best_match_rows: list[dict],
    skipped_repost_media: int,
    skipped_repost_fingerprint: int,
    best_confidence_seen: float,
    started_at: float,
) -> None:
    _log_event(
        "cross_platform_fingerprint_repost_progress",
        processed_reposts=index,
        total_reposts=total_reposts,
        repost_best_match_rows=len(best_match_rows),
        matched_rows=len(output_rows),
        skipped_repost_media=skipped_repost_media,
        skipped_repost_fingerprint=skipped_repost_fingerprint,
        best_confidence_seen=round(float(best_confidence_seen), 6),
        elapsed_sec=round(time.monotonic() - started_at, 1),
    )


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)


def _short_error(exc: BaseException, *, limit: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


class Fingerprint:
    def __init__(self, duration: float, values: list[int]) -> None:
        self.duration = duration
        self.values = values


def _fingerprint(path: pathlib.Path, fpcalc: str) -> Fingerprint:
    if not path.exists():
        raise FileNotFoundError(path)
    completed = subprocess.run(
        [fpcalc, "-raw", "-json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    payload = json.loads(completed.stdout)
    raw = payload.get("fingerprint") or []
    values = [int(item) for item in raw]
    return Fingerprint(duration=float(payload.get("duration") or 0.0), values=values)


def fingerprint_media_to_file(media_path: pathlib.Path, output_path: pathlib.Path, fpcalc: str) -> Fingerprint:
    fingerprint = _fingerprint(media_path, fpcalc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "duration": float(fingerprint.duration),
                "fingerprint": fingerprint.values,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return fingerprint


def _fingerprint_from_row(row: dict, fpcalc: str) -> Fingerprint:
    raw_fingerprint_path = str(row.get("fingerprint_path") or "").strip()
    if raw_fingerprint_path:
        fingerprint_path = _resolve_media_path(raw_fingerprint_path)
        if fingerprint_path is not None:
            return _read_fingerprint_file(fingerprint_path)
    media_path = _resolve_media_path(str(row.get("video_path") or ""))
    if media_path is None:
        raise FileNotFoundError(str(row.get("video_path") or row.get("fingerprint_path") or ""))
    return _fingerprint(media_path, fpcalc)


def _read_fingerprint_file(path: pathlib.Path) -> Fingerprint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("fingerprint") or []
    values = [int(item) for item in raw]
    return Fingerprint(duration=float(payload.get("duration") or 0.0), values=values)


def _resolve_media_path(raw_path: str) -> pathlib.Path | None:
    value = str(raw_path or "").strip()
    if not value:
        return None
    path = pathlib.Path(value).expanduser()
    if path.exists():
        return path
    remapped = _remap_backend_path(path)
    if remapped.exists():
        return remapped
    return None


def _remap_backend_path(path: pathlib.Path) -> pathlib.Path:
    raw = str(path)
    marker = "/backend/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1]
        return BACKEND_ROOT / suffix
    parts = path.parts
    if "backend" in parts:
        index = parts.index("backend")
        return BACKEND_ROOT.joinpath(*parts[index + 1:])
    return path


def best_fingerprint_alignment(needle: list[int], haystack: list[int]) -> tuple[float, int]:
    if not needle or not haystack or len(needle) > len(haystack):
        return 0.0, 0
    best_score = 0.0
    best_index = 0
    step = max(1, min(len(needle) // 8, 12))
    for start in range(0, len(haystack) - len(needle) + 1, step):
        score = _fingerprint_similarity(needle, haystack[start:start + len(needle)])
        if score > best_score:
            best_score = score
            best_index = start
    # Refine around the coarse winner.
    for start in range(max(0, best_index - step), min(len(haystack) - len(needle), best_index + step) + 1):
        score = _fingerprint_similarity(needle, haystack[start:start + len(needle)])
        if score > best_score:
            best_score = score
            best_index = start
    return round(best_score, 6), best_index


def _fingerprint_similarity(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    distances = [(int(a) ^ int(b)).bit_count() / 32.0 for a, b in zip(left, right, strict=False)]
    return max(0.0, min(1.0, 1.0 - (sum(distances) / len(distances))))


def _matched_row(repost: dict, vod: dict, start: float, end: float, confidence: float) -> dict:
    views = _safe_float(repost.get("views") or repost.get("view_count"))
    title = _repost_title(repost)
    return outcome_view_weighting.annotate_cross_platform_view_weight({
        "vod_id": str(vod.get("vod_id") or ""),
        "vod_url": str(vod.get("vod_url") or ""),
        "candidate_id": str(repost.get("clip_id") or repost.get("candidate_id") or f"cross_platform:{vod.get('vod_id')}:{start:.1f}"),
        "window_start": round(start, 2),
        "window_end": round(max(end, start + 1.0), 2),
        "label": 1,
        "selected": True,
        "label_class": "cross_platform_outcome_positive",
        "dataset_source": "cross_platform_audio_fingerprint",
        "label_source": "cross_platform_audio_fingerprint",
        "supervision_tier": "cross_platform_outcome",
        "sample_weight": 2.0 if views >= 100_000 else 1.5,
        "platform": str(repost.get("platform") or ""),
        "source_url": str(repost.get("source_url") or ""),
        "repost_title": title,
        "views": views,
        "fingerprint_confidence": round(confidence, 6),
        "profile_filter_version": int(_safe_float(repost.get("profile_filter_version"))),
    })


def _best_match_diagnostic_row(
    repost: dict,
    vod: dict,
    start: float,
    end: float,
    confidence: float,
    *,
    accepted: bool,
) -> dict:
    title = _repost_title(repost)
    return {
        "accepted": bool(accepted),
        "confidence": round(float(confidence), 6),
        "vod_id": str(vod.get("vod_id") or ""),
        "vod_url": str(vod.get("vod_url") or ""),
        "window_start": round(float(start), 2),
        "window_end": round(max(float(end), float(start) + 1.0), 2),
        "repost_clip_id": str(repost.get("clip_id") or repost.get("candidate_id") or ""),
        "repost_source_url": str(repost.get("source_url") or ""),
        "repost_title": title,
        "repost_platform": str(repost.get("platform") or ""),
        "repost_views": _safe_float(repost.get("views") or repost.get("view_count")),
    }


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
    vod_fps: list[tuple[dict, Fingerprint]],
    best_match_rows: list[dict],
    matched_rows: list[dict],
    min_confidence: float,
    diagnostic_top_k: int,
    skipped_vod_media: int,
    skipped_vod_fingerprint: int,
    skipped_repost_media: int,
    skipped_repost_fingerprint: int,
) -> dict[str, object]:
    scores = sorted(float(row.get("confidence") or 0.0) for row in best_match_rows)
    top_rows = sorted(best_match_rows, key=lambda row: float(row.get("confidence") or 0.0), reverse=True)
    ladder = _threshold_ladder(scores)
    return {
        "min_confidence": round(float(min_confidence), 6),
        "vod_rows_input": len(vod_rows),
        "repost_rows_input": len(repost_rows),
        "vod_fingerprints_built": len(vod_fps),
        "repost_best_match_rows": len(best_match_rows),
        "matched_rows": len(matched_rows),
        "skipped_vod_media_rows": int(skipped_vod_media),
        "skipped_vod_fingerprint_rows": int(skipped_vod_fingerprint),
        "skipped_repost_media_rows": int(skipped_repost_media),
        "skipped_repost_fingerprint_rows": int(skipped_repost_fingerprint),
        "confidence_distribution": _score_distribution(scores),
        "threshold_ladder": ladder,
        "best_match_rows": best_match_rows,
        "top_best_matches": top_rows[: max(0, int(diagnostic_top_k))],
        "recommendation": _matching_recommendation(scores, len(matched_rows), float(min_confidence), ladder),
    }


def _score_distribution(scores: list[float]) -> dict[str, object]:
    if not scores:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(scores),
        "min": round(scores[0], 6),
        "p10": round(_quantile(scores, 0.10), 6),
        "p25": round(_quantile(scores, 0.25), 6),
        "median": round(_quantile(scores, 0.50), 6),
        "p75": round(_quantile(scores, 0.75), 6),
        "p90": round(_quantile(scores, 0.90), 6),
        "p95": round(_quantile(scores, 0.95), 6),
        "max": round(scores[-1], 6),
        "mean": round(sum(scores) / len(scores), 6),
    }


def _quantile(sorted_scores: list[float], q: float) -> float:
    if not sorted_scores:
        return 0.0
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    position = max(0.0, min(1.0, q)) * (len(sorted_scores) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_scores) - 1)
    fraction = position - lower
    return sorted_scores[lower] + (sorted_scores[upper] - sorted_scores[lower]) * fraction


def _threshold_ladder(scores: list[float]) -> dict[str, int]:
    thresholds = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80)
    return {f"{threshold:.2f}": sum(1 for score in scores if score >= threshold) for threshold in thresholds}


def _matching_recommendation(
    scores: list[float],
    matched_count: int,
    min_confidence: float,
    ladder: dict[str, int],
) -> dict[str, object]:
    if not scores:
        return {
            "status": "no_comparable_fingerprints",
            "message": "No repost could be fingerprinted against a VOD. Check local media availability and fpcalc output.",
        }
    distribution = _score_distribution(scores)
    median = float(distribution.get("median") or 0.0)
    p95 = float(distribution.get("p95") or 0.0)
    max_score = float(distribution.get("max") or 0.0)
    if matched_count > 0:
        return {
            "status": "threshold_produces_matches",
            "message": "Current Chromaprint threshold produced cross-platform rows.",
        }
    candidate_thresholds = [
        float(raw_threshold)
        for raw_threshold, count in ladder.items()
        if count > 0 and float(raw_threshold) < min_confidence
    ]
    if p95 >= 0.55:
        suggested = max(candidate_thresholds) if candidate_thresholds else max_score
        return {
            "status": "consider_lowering_threshold",
            "suggested_min_confidence": round(float(suggested), 2),
            "message": (
                "Best-match scores cluster below the current threshold but are not near zero. "
                "Review the diagnostic rows before lowering the production threshold."
            ),
        }
    if median >= 0.30 or max_score >= 0.45:
        return {
            "status": "chromaprint_signal_weak",
            "message": (
                "Chromaprint sees weak partial similarity. Voice-over, edits or music replacement may be damaging hashes; "
                "prefer semantic audio embeddings or transcript/visual matching over simply lowering the threshold."
            ),
        }
    return {
        "status": "chromaprint_signal_absent",
        "message": (
            "Best-match scores are very low. Lowering the threshold is likely unsafe; use an embedding-based or "
            "multimodal matcher for these reposts."
        ),
    }


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
