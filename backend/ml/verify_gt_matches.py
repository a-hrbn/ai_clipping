"""Verify GT manifest moments with scoped audio-fingerprint confirmation.

The eval manifest claims that each gt_moment's [start, end] window in a Twitch
VOD is the source of an Iboclipss repost. The transcript matcher that produced
those claims can anchor on repeated catchphrases, so this script cross-checks
every moment with a second modality. For each gt_moment it:

1. resolves the repost media (cached short-form download),
2. obtains VOD audio evidence around the claimed window:
   - anchor VODs: a slice of the cached full-VOD chromaprint fingerprint,
   - other VODs: a ranged Twitch download of [start - margin, end + margin],
3. aligns the repost fingerprint against the scoped window AND against a
   deliberately wrong "null" window of the same length,
4. confirms the moment only when the scoped confidence clears the null
   confidence by a margin (self-calibrating against the chromaprint noise
   floor and music-overlay repost edits),
5. writes one verification row per moment to gt_match_verification.jsonl.

Tier policy (--apply writes gt_tier into the manifest):
  gold          audio-confirmed at the claimed position
  silver        no audio confirmation but strong transcript evidence
  review        weak transcript evidence and no audio confirmation
  unverifiable  repost or VOD media unavailable (kept; decide via manual review)

--apply-boundaries additionally replaces start/end of gold moments on anchor
VODs with the audio-refined window (~0.12s granularity); the transcript window
is preserved in transcript_start/transcript_end.

Usage (inside the Docker container):
  python -m ml.verify_gt_matches [--limit N] [--vod-id ID ...]
  python -m ml.verify_gt_matches --apply [--apply-boundaries]
  python -m ml.verify_gt_matches --export-review-csv PATH
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import match_cross_platform_audio_fingerprints as audio_matcher

DEFAULT_MANIFEST = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_eval_manifest.jsonl"
DEFAULT_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "gt_match_verification.jsonl"
VOD_FEATURE_CACHE = BACKEND_ROOT / "tmp" / "vod_feature_cache"
GT_VOD_STORAGE = BACKEND_ROOT / "tmp" / "storage"

# Scoped window padding around the claimed GT window. Generous enough to
# absorb segment-granular transcript boundaries, small enough that a match
# inside the scope still confirms the claimed position.
SCOPE_MARGIN_SEC = 45.0
# Confirm rule, calibrated on the anchor-VOD verification set (2026-06):
# repost edits (music overlays, captions, re-encoding) dampen true-position
# confidence to ~0.6-0.7 while the chromaprint noise floor sits at ~0.55-0.6,
# so confirmation combines two independent signals — margin over the null
# window AND the best alignment landing where the transcript claims.
CONFIRM_FLOOR = 0.55
CONFIRM_RULES = (
    # (min margin over null, max |refined_offset_sec|)
    (0.04, 10.0),
    (0.02, 5.0),
)
# Ranged Twitch downloads cut at HLS segment boundaries before the requested
# start, biasing the measured offset positive by up to ~12s.
RANGE_OFFSET_BIAS_SEC = 12.0
# Distance of the deliberately wrong null window from the claimed window.
NULL_OFFSET_SEC = 1800.0
SILVER_MIN_SIMILARITY = 0.65
REVIEW_CSV_SAMPLE_SIZE = 15

GOLD = "gold"
SILVER = "silver"
REVIEW = "review"
UNVERIFIABLE = "unverifiable"


def moment_key(vod_id: str, moment: dict) -> str:
    return f"{vod_id}:{float(moment.get('start') or 0.0):.2f}:{float(moment.get('end') or 0.0):.2f}"


def fingerprint_rate(fp: audio_matcher.Fingerprint) -> float:
    if fp.duration <= 0 or not fp.values:
        return 0.0
    return len(fp.values) / fp.duration


def slice_fingerprint_values(fp: audio_matcher.Fingerprint, start_sec: float, end_sec: float) -> tuple[list[int], float]:
    """Slice [start_sec, end_sec] out of a full fingerprint.

    Returns (values, actual_start_sec) — the start is clamped to the media.
    """
    rate = fingerprint_rate(fp)
    if rate <= 0:
        return [], 0.0
    start = max(float(start_sec), 0.0)
    end = min(max(float(end_sec), start), fp.duration)
    lo = int(start * rate)
    hi = int(end * rate)
    return fp.values[lo:hi], lo / rate


def null_window_start(scope_start: float, scope_end: float, media_duration: float, seed_text: str) -> float:
    """Deterministic start of a wrong window far away from the claimed scope."""
    scope_len = max(scope_end - scope_start, 1.0)
    usable = max(media_duration - scope_len, 1.0)
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    for attempt in range(8):
        candidate = (seed / 0xFFFFFFFF * usable + attempt * NULL_OFFSET_SEC) % usable
        if candidate + scope_len <= scope_start or candidate >= scope_end:
            return candidate
    return 0.0


def is_confirmed(
    confidence: float,
    null_confidence: float,
    offset_sec: float,
    *,
    boundary_basis: str = "full_fingerprint",
) -> bool:
    if confidence < CONFIRM_FLOOR:
        return False
    margin = confidence - null_confidence
    bias = RANGE_OFFSET_BIAS_SEC if boundary_basis == "range_approx" else 0.0
    for min_margin, max_offset in CONFIRM_RULES:
        if margin >= min_margin and -max_offset <= offset_sec <= max_offset + bias:
            return True
    return False


def assign_tier(
    *,
    available: bool,
    confirmed: bool,
    match_similarity: float,
    position_ambiguous: bool = False,
) -> str:
    if not available:
        return UNVERIFIABLE
    if confirmed:
        return GOLD
    if match_similarity >= SILVER_MIN_SIMILARITY and not position_ambiguous:
        return SILVER
    return REVIEW


def judge_row(row: dict) -> dict:
    """(Re)compute verdict + gt_tier from a row's stored alignment evidence.

    Kept separate from evidence collection so the confirm rule can be
    recalibrated (--rejudge) without re-downloading any media.
    """
    if str(row.get("verdict") or "") in ("repost_unavailable", "vod_unavailable", "scope_too_short"):
        return row
    confirmed = is_confirmed(
        float(row.get("confirm_confidence") or 0.0),
        float(row.get("null_confidence") or 0.0),
        float(row.get("refined_offset_sec") or 0.0),
        boundary_basis=str(row.get("boundary_basis") or "full_fingerprint"),
    )
    row["verdict"] = "confirmed" if confirmed else "not_confirmed"
    row["gt_tier"] = assign_tier(
        available=True,
        confirmed=confirmed,
        match_similarity=float(row.get("match_similarity") or 0.0),
        position_ambiguous=bool(row.get("position_ambiguous")),
    )
    return row


def _anchor_vod_media(vod_id: str) -> pathlib.Path | None:
    media = GT_VOD_STORAGE / f"gt_vod_{vod_id}" / "video.mp4"
    return media if media.exists() else None


def _load_full_vod_fingerprint(vod_id: str, fpcalc: str) -> audio_matcher.Fingerprint | None:
    cache_path = VOD_FEATURE_CACHE / vod_id / "fingerprint.json"
    if cache_path.exists():
        try:
            return audio_matcher._read_fingerprint_file(cache_path)
        except ValueError:
            pass  # stale prefix-only cache from before the -length fix
    media = _anchor_vod_media(vod_id)
    if media is None:
        return None
    return audio_matcher.fingerprint_media_to_file(media, cache_path, fpcalc)


def _repost_fingerprint(source_url: str, fpcalc: str) -> audio_matcher.Fingerprint:
    from ml import match_cross_platform_transcripts as transcript_matcher
    from services.downloader import ensure_cached_download

    video_path = ensure_cached_download(source_url, cache_namespace="short_form", cache_ttl_sec=-1)
    return transcript_matcher._load_or_build_repost_fingerprint(
        {"source_url": source_url},
        video_path,
        fpcalc,
        fp_cache_root=transcript_matcher.DEFAULT_SHORT_FINGERPRINT_CACHE_ROOT,
    )


def _scoped_remote_fingerprint(vod_id: str, start_sec: float, end_sec: float, fpcalc: str) -> audio_matcher.Fingerprint:
    from config import settings
    from services.downloader import download_range

    job_id = f"_gt_verify_{vod_id}_{int(start_sec)}"
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    try:
        media = download_range(
            f"https://www.twitch.tv/videos/{vod_id}",
            job_id,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        return audio_matcher._fingerprint(media, fpcalc)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _base_row(vod_id: str, moment: dict, *, vod_media: str) -> dict:
    return {
        "key": moment_key(vod_id, moment),
        "vod_id": vod_id,
        "start": float(moment.get("start") or 0.0),
        "end": float(moment.get("end") or 0.0),
        "source_url": str(moment.get("source_url") or ""),
        "match_kind": str(moment.get("match_kind") or ""),
        "match_similarity": float(moment.get("match_similarity") or 0.0),
        "views": float(moment.get("views") or 0.0),
        "position_ambiguous": bool(moment.get("position_ambiguous")),
        "vod_media": vod_media,
    }


def verify_moment(
    vod_id: str,
    moment: dict,
    *,
    fpcalc: str,
    full_vod_fp: audio_matcher.Fingerprint | None,
    null_source_fp: audio_matcher.Fingerprint | None,
) -> dict:
    row = _base_row(vod_id, moment, vod_media="local" if full_vod_fp is not None else "range")
    start = row["start"]
    end = row["end"]
    source_url = row["source_url"]

    try:
        repost_fp = _repost_fingerprint(source_url, fpcalc)
    except Exception as exc:  # deleted short, network failure, fpcalc error
        row.update({"verdict": "repost_unavailable", "error": _short_error(exc), "gt_tier": UNVERIFIABLE})
        return row
    if not repost_fp.values:
        row.update({"verdict": "repost_unavailable", "error": "empty repost fingerprint", "gt_tier": UNVERIFIABLE})
        return row
    repost_duration = float(repost_fp.duration)
    row["repost_duration"] = round(repost_duration, 2)

    scope_start = max(start - SCOPE_MARGIN_SEC, 0.0)
    scope_end = start + max(end - start, repost_duration) + SCOPE_MARGIN_SEC

    if full_vod_fp is not None:
        scoped_values, actual_scope_start = slice_fingerprint_values(full_vod_fp, scope_start, scope_end)
        scoped_rate = fingerprint_rate(full_vod_fp)
        scoped_duration = len(scoped_values) / scoped_rate if scoped_rate else 0.0
    else:
        try:
            scoped_fp = _scoped_remote_fingerprint(vod_id, scope_start, scope_end, fpcalc)
        except Exception as exc:  # expired VOD, network failure
            row.update({"verdict": "vod_unavailable", "error": _short_error(exc), "gt_tier": UNVERIFIABLE})
            return row
        scoped_values = scoped_fp.values
        scoped_duration = float(scoped_fp.duration)
        scoped_rate = fingerprint_rate(scoped_fp)
        actual_scope_start = scope_start  # ranged downloads cut at HLS segment boundaries (~±10s)
    if not scoped_values or len(repost_fp.values) > len(scoped_values):
        row.update({"verdict": "scope_too_short", "gt_tier": REVIEW})
        return row

    confidence, offset_index = audio_matcher.best_fingerprint_alignment(repost_fp.values, scoped_values)

    null_fp = full_vod_fp or null_source_fp
    null_confidence = 0.0
    if null_fp is not None:
        null_start = null_window_start(
            scope_start if full_vod_fp is not None else 0.0,
            scope_end if full_vod_fp is not None else 0.0,
            null_fp.duration,
            row["key"],
        )
        null_values, _ = slice_fingerprint_values(null_fp, null_start, null_start + max(scoped_duration, repost_duration + 1.0))
        if null_values and len(repost_fp.values) <= len(null_values):
            null_confidence, _ = audio_matcher.best_fingerprint_alignment(repost_fp.values, null_values)

    refined_start = actual_scope_start + (offset_index / scoped_rate if scoped_rate else 0.0)
    row.update({
        "confirm_confidence": round(float(confidence), 6),
        "null_confidence": round(float(null_confidence), 6),
        "refined_start": round(refined_start, 2),
        "refined_end": round(refined_start + repost_duration, 2),
        "refined_offset_sec": round(refined_start - start, 2),
        "boundary_basis": "full_fingerprint" if full_vod_fp is not None else "range_approx",
    })
    return judge_row(row)


def _short_error(exc: BaseException, *, limit: int = 200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(rows: list[dict]) -> dict:
    by_tier: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    offsets: list[float] = []
    for row in rows:
        tier = str(row.get("gt_tier") or "")
        by_tier[tier] = by_tier.get(tier, 0) + 1
        kind = str(row.get("match_kind") or "")
        kind_stats = by_kind.setdefault(kind, {"total": 0, "confirmed": 0})
        kind_stats["total"] += 1
        if row.get("verdict") == "confirmed":
            kind_stats["confirmed"] += 1
            offsets.append(abs(float(row.get("refined_offset_sec") or 0.0)))
    offsets.sort()
    return {
        "rows": len(rows),
        "by_tier": by_tier,
        "by_match_kind": by_kind,
        "confirmed_offset_abs_sec": {
            "count": len(offsets),
            "median": round(offsets[len(offsets) // 2], 2) if offsets else None,
            "p90": round(offsets[int(len(offsets) * 0.9)], 2) if offsets else None,
            "max": round(offsets[-1], 2) if offsets else None,
        },
    }


def apply_tiers(
    manifest_path: pathlib.Path,
    verification_rows: list[dict],
    *,
    apply_boundaries: bool,
) -> dict:
    by_key = {str(row.get("key")): row for row in verification_rows}
    manifest_rows = _read_jsonl(manifest_path)
    tiers_applied = 0
    boundaries_applied = 0
    for vod_row in manifest_rows:
        vod_id = str(vod_row.get("vod_id") or "")
        for moment in vod_row.get("gt_moments") or []:
            verification = by_key.get(moment_key(vod_id, moment))
            if verification is None:
                continue
            moment["gt_tier"] = verification.get("gt_tier")
            tiers_applied += 1
            if (
                apply_boundaries
                and not moment.get("manual_edit")
                and verification.get("gt_tier") == GOLD
                and verification.get("boundary_basis") == "full_fingerprint"
            ):
                moment["transcript_start"] = moment["start"]
                moment["transcript_end"] = moment["end"]
                moment["start"] = float(verification["refined_start"])
                moment["end"] = float(verification["refined_end"])
                moment["boundary_source"] = "audio_refined"
                boundaries_applied += 1
    payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in manifest_rows)
    manifest_path.write_text(payload + "\n", encoding="utf-8")
    return {"tiers_applied": tiers_applied, "boundaries_applied": boundaries_applied}


def _twitch_deep_link(vod_id: str, start_sec: float) -> str:
    total = max(int(start_sec), 0)
    return f"https://www.twitch.tv/videos/{vod_id}?t={total // 3600}h{(total % 3600) // 60}m{total % 60}s"


def export_review_csv(path: pathlib.Path, verification_rows: list[dict]) -> int:
    needs_review = [r for r in verification_rows if r.get("gt_tier") in (REVIEW, UNVERIFIABLE)]
    spot_check = sorted(
        (r for r in verification_rows if r.get("gt_tier") in (GOLD, SILVER)),
        key=lambda r: hashlib.sha256(str(r.get("key")).encode("utf-8")).hexdigest(),
    )[:REVIEW_CSV_SAMPLE_SIZE]
    rows = needs_review + spot_check
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "key", "gt_tier", "verdict", "vod_deep_link", "start", "end",
            "repost_url", "views", "match_kind", "match_similarity",
            "confirm_confidence", "null_confidence",
            "human_verdict", "corrected_start", "corrected_end", "notes",
        ])
        for row in rows:
            writer.writerow([
                row.get("key"), row.get("gt_tier"), row.get("verdict"),
                _twitch_deep_link(str(row.get("vod_id")), float(row.get("start") or 0.0)),
                row.get("start"), row.get("end"),
                row.get("source_url"), row.get("views"),
                row.get("match_kind"), row.get("match_similarity"),
                row.get("confirm_confidence"), row.get("null_confidence"),
                "", "", "", "",
            ])
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vod-id", action="append", default=None, help="Only verify these VOD ids (repeatable)")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N newly verified moments")
    parser.add_argument("--apply", action="store_true", help="Write gt_tier into the manifest")
    parser.add_argument("--apply-boundaries", action="store_true",
                        help="With --apply: replace start/end of gold anchor moments with audio-refined boundaries")
    parser.add_argument("--export-review-csv", type=pathlib.Path, default=None)
    parser.add_argument("--rejudge", action="store_true",
                        help="Recompute verdict/gt_tier of stored rows under the current confirm rule (no downloads)")
    args = parser.parse_args(argv)

    manifest_rows = _read_jsonl(args.manifest)
    if not manifest_rows:
        print(f"no manifest rows at {args.manifest}", file=sys.stderr)
        return 1

    existing = {str(row.get("key")): row for row in _read_jsonl(args.output)}
    if args.rejudge and existing:
        rejudged = [judge_row(dict(row)) for row in existing.values()]
        existing = {str(row.get("key")): row for row in rejudged}
        payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in rejudged)
        args.output.write_text(payload + "\n", encoding="utf-8")

    fpcalc = shutil.which("fpcalc")
    if fpcalc is None:
        raise RuntimeError("fpcalc not found. Install chromaprint-tools.")

    null_source_fp: audio_matcher.Fingerprint | None = None
    verified_new = 0
    for vod_row in manifest_rows:
        vod_id = str(vod_row.get("vod_id") or "")
        if args.vod_id and vod_id not in args.vod_id:
            continue
        moments = list(vod_row.get("gt_moments") or [])
        pending = [m for m in moments if moment_key(vod_id, m) not in existing]
        if not pending:
            continue
        full_vod_fp = _load_full_vod_fingerprint(vod_id, fpcalc)
        if full_vod_fp is not None and null_source_fp is None:
            null_source_fp = full_vod_fp
        if null_source_fp is None:
            for anchor_dir in sorted(GT_VOD_STORAGE.glob("gt_vod_*")):
                null_source_fp = _load_full_vod_fingerprint(anchor_dir.name.removeprefix("gt_vod_"), fpcalc)
                if null_source_fp is not None:
                    break
        vod_dead_error: str | None = None
        for moment in pending:
            if args.limit and verified_new >= args.limit:
                break
            if vod_dead_error is not None:
                row = _base_row(vod_id, moment, vod_media="range")
                row.update({"verdict": "vod_unavailable", "error": vod_dead_error, "gt_tier": UNVERIFIABLE})
            else:
                row = verify_moment(
                    vod_id,
                    moment,
                    fpcalc=fpcalc,
                    full_vod_fp=full_vod_fp,
                    null_source_fp=null_source_fp,
                )
                if row.get("verdict") == "vod_unavailable" and "does not exist" in str(row.get("error") or ""):
                    vod_dead_error = str(row.get("error"))
            existing[str(row["key"])] = row
            _append_jsonl(args.output, row)
            verified_new += 1
            print(json.dumps({k: row.get(k) for k in ("key", "verdict", "gt_tier", "confirm_confidence", "null_confidence", "refined_offset_sec")}, ensure_ascii=True), flush=True)
        if args.limit and verified_new >= args.limit:
            break

    rows = list(existing.values())
    summary = summarize(rows)
    if args.apply:
        summary["apply"] = apply_tiers(args.manifest, rows, apply_boundaries=args.apply_boundaries)
    if args.export_review_csv:
        summary["review_csv_rows"] = export_review_csv(args.export_review_csv, rows)
        summary["review_csv"] = str(args.export_review_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
