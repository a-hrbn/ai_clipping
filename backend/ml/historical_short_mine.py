"""Historical-short-mine — read-only retention ingest + per-short report (Step 0).

PRD #75 / issue #76 (Slice 1). The cheapest possible first look at the
first-party-outcome pivot (ADR docs/adr/0001-first-party-outcome-flywheel.md):
turn a locally exported YT audience-retention curve for the 52 already-posted
@donni_clips shorts into a per-short report, *before* the forward loop is built.

This slice is the ingest skeleton only. A tolerant parser derives per-short
``retention_3s``/``retention_50pct``/``completion`` from the YT-*relative*
audience-retention curve (``elapsedVideoTimeRatio`` -> ``audienceWatchRatio``)
plus ``averageViewPercentage`` as the degraded fallback. Retention is read
**views-decoupled** (views are a secondary field, never a positive-gate) per the
ADR's cold-start win. The report is keyed by ``row_identity`` so re-runs are
idempotent and join-able with a later VOD attribution. Facet columns
(goal_family / length_bucket / transcript_excerpt) stay empty until Slice 2.

Offline only: no network, no paid call, no serving touch, no new pipeline stage.
Built against the public YT-Analytics schema; reconciliation against the real
export format is deferred to Slice 4 (HITL).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import tempfile

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import outcome_view_weighting, row_identity
from services import clip_strategy


# Elapsed-time offset (seconds) at which early-retention is read.
_EARLY_RETENTION_S = 3.0

# Length-bucket boundary (seconds): below -> "kurz", at/above -> "lang".
_LENGTH_BUCKET_BOUNDARY_S = 25.0

# Max characters of transcript kept in the per-short report for auditing.
_TRANSCRIPT_EXCERPT_CHARS = 280

# Facet columns. Empty until a ``transcribe`` seam is injected (Slice 1 shape);
# filled from the short's own transcript once it is (Slice 2).
_EMPTY_FACETS = {"goal_family": None, "length_bucket": None, "transcript_excerpt": None}


def build_short_report(raw_rows: list[dict], *, transcribe=None) -> dict:
    """Compose the per-short report from raw YT export rows, views-decoupled.

    Each short carries its derived retention fields, a stable ``short_id`` (via
    ``row_identity``) and ``views`` as a *secondary* field — there is no views
    gate and no label drop, so a strong-retention/low-views short stays in.

    When a ``transcribe`` seam is injected (``Callable[[dict], str | None]``,
    a row -> transcript text), each kept short is additionally classified into
    its facet — goal-family (``infer_goal_family``, transcript as topic) ×
    length-bucket — and carries a transcript excerpt. Chat-emotion is *not* a
    facet here: that signal lives in the VOD, not in the posted short.
    """
    shorts: list[dict] = []
    drops: list[dict] = []
    for row in raw_rows:
        derived = derive_retention(row)
        short_id = _short_identity(row)
        if derived["retention_source"] == "missing":
            drops.append({"short_id": short_id, "reason": "missing_retention_row"})
            continue
        facets = dict(_EMPTY_FACETS)
        if transcribe is not None:
            facets = derive_facets(row, transcribe)
            if facets is None:
                # Un-transcribable / not classifiable: surfaced as a logged drop,
                # never emitted as a bogus "general" short (honest denominator).
                drops.append({"short_id": short_id, "reason": "untranscribable"})
                continue
        short = {
            "short_id": short_id,
            "retention_3s": derived["retention_3s"],
            "retention_50pct": derived["retention_50pct"],
            "completion": derived["completion"],
            "retention_source": derived["retention_source"],
            "averageViewPercentage": derived.get("averageViewPercentage"),
            "duration_s": _as_float(row.get("duration_s")),
            "views": outcome_view_weighting.view_count(row),
            **facets,
        }
        shorts.append(short)
    # Sort by short_id so the report is input-order independent — a re-run over
    # the same export writes byte-identical output.
    shorts.sort(key=lambda s: s["short_id"])
    drops.sort(key=lambda d: d["short_id"])
    return {"shorts": shorts, "drops": drops}


def derive_facets(row: dict, transcribe) -> dict | None:
    """Classify a short into its facet from its *own* transcript.

    ``transcribe(row)`` -> transcript text. The goal-family is read off the
    transcript via ``clip_strategy.infer_goal_family`` (transcript as the
    ``topic``). Chat-emotion is deliberately excluded — it is a VOD-only signal,
    not available in the posted short. Returns ``None`` when the short cannot be
    transcribed (missing media / blank transcript) so the caller logs a drop
    instead of emitting an un-classifiable short.
    """
    transcript = transcribe(row)
    if not transcript or not transcript.strip():
        return None
    return {
        "goal_family": clip_strategy.infer_goal_family("", "", topic=transcript),
        "length_bucket": _length_bucket(_as_float(row.get("duration_s"))),
        "transcript_excerpt": _excerpt(transcript),
    }


def transcribe_short(media_path, *, extract=None, transcribe=None) -> str | None:
    """Transcribe one local short: mp4 -> wav -> text.

    Composes the two existing leaf seams — ``services.audio_extractor.extract``
    (mp4 -> 16 kHz wav) then ``services.transcriber.transcribe`` (wav -> rich
    segments) — and joins the segment texts. Returns ``None`` when the media is
    missing or yields no speech, so the report logs an untranscribable drop.

    Both leaf seams are injectable so tests run offline (no ffmpeg / Whisper /
    network); the real seams are imported lazily only when not injected.
    """
    media_path = pathlib.Path(media_path)
    if not media_path.exists():
        return None
    if extract is None or transcribe is None:
        from services import audio_extractor, transcriber
        extract = extract or audio_extractor.extract
        transcribe = transcribe or transcriber.transcribe
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = extract(
            media_path, media_path.stem, output_path=pathlib.Path(tmp_dir) / "audio.wav")
        segments = transcribe(wav_path)
    text = " ".join(str(getattr(seg, "text", "") or "") for seg in segments)
    text = " ".join(text.split())
    return text or None


def _length_bucket(duration_s: float) -> str:
    """Coarse length stratum: "kurz" below the boundary, "lang" at/above it."""
    return "kurz" if duration_s < _LENGTH_BUCKET_BOUNDARY_S else "lang"


def _excerpt(transcript: str | None) -> str:
    """A short, whitespace-collapsed slice of the transcript for auditing."""
    text = " ".join(str(transcript or "").split())
    return text[:_TRANSCRIPT_EXCERPT_CHARS]


def _short_identity(row: dict) -> str:
    """Stable per-short identity via ``row_identity`` (``clips:<video_id>``)."""
    video_id = str(row.get("video_id") or row.get("short_id") or "").strip()
    return row_identity.primary_identity_key({"clip_ids": [video_id]})


def derive_retention(row: dict) -> dict:
    """Derive the three retention fields from a raw YT export row.

    ``retention_3s`` / ``retention_50pct`` / ``completion`` are read off the
    YT-relative audience-retention curve by linear interpolation at the elapsed
    ratios ``3s/duration``, ``0.5`` and ``1.0``. ``retention_source`` is
    ``"curve"`` when the curve drives the fields.
    """
    avg_view_percentage = row.get("averageViewPercentage")
    curve = _read_curve(row)
    if not curve:
        # No curve. ``degraded_avp_only`` keeps the short (flagged) when the
        # scalar averageViewPercentage survived; ``missing`` marks a truly-empty
        # retention row that build_short_report will drop+log. Never abort.
        source = "degraded_avp_only" if avg_view_percentage is not None else "missing"
        return {
            "retention_3s": None,
            "retention_50pct": None,
            "completion": None,
            "retention_source": source,
            "averageViewPercentage": avg_view_percentage,
        }
    duration = _as_float(row.get("duration_s"))
    early_ratio = _EARLY_RETENTION_S / duration if duration > 0 else 0.0
    return {
        "retention_3s": _interpolate(curve, early_ratio),
        "retention_50pct": _interpolate(curve, 0.5),
        "completion": _interpolate(curve, 1.0),
        "retention_source": "curve",
        "averageViewPercentage": avg_view_percentage,
    }


def _read_curve(row: dict) -> list[tuple[float, float]]:
    raw = row.get("audience_retention") or row.get("audienceRetention") or []
    points: list[tuple[float, float]] = []
    for point in raw:
        ratio = _as_float(point.get("elapsedVideoTimeRatio"))
        watch = _as_float(point.get("audienceWatchRatio"))
        points.append((ratio, watch))
    return sorted(points)


def _interpolate(curve: list[tuple[float, float]], query: float) -> float:
    """Linear interpolation of the watch-ratio at ``query`` elapsed ratio."""
    if query <= curve[0][0]:
        return curve[0][1]
    if query >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= query <= x1:
            if x1 == x0:
                return y0
            return y0 + (query - x0) / (x1 - x0) * (y1 - y0)
    return curve[-1][1]


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "historical_short_mine_per_short.jsonl"


def main(argv: list[str] | None = None, *, transcribe=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--metrics", required=True,
        help="Local YT audience-retention export (JSONL); read offline, no network.")
    parser.add_argument(
        "--media-dir", type=pathlib.Path, default=None,
        help="Local directory of short media (<video_id>.mp4); transcribed offline "
             "for the goal-family/length facet. Omit to leave facets empty.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                        help="Per-short report JSONL destination.")
    parser.add_argument("--json", dest="output_json", action="store_true",
                        help="Print the full report (shorts + drops) as JSON.")
    args = parser.parse_args(argv)

    if transcribe is None and args.media_dir is not None:
        transcribe = _media_dir_transcribe(args.media_dir)

    raw_rows = row_identity.read_jsonl(pathlib.Path(args.metrics), strict=False)
    report = build_short_report(raw_rows, transcribe=transcribe)

    row_identity.write_jsonl(args.output, report["shorts"])

    if report["drops"]:
        # No silent truncation: surface every dropped short (and why) so the
        # denominator stays honest (mirrors gap_decomposition's gap logging).
        reasons = collections.Counter(d["reason"] for d in report["drops"])
        print(
            f"historical_short_mine: dropped {len(report['drops'])}/{len(raw_rows)} "
            f"shorts ({_format_reasons(reasons)}): "
            f"{', '.join(d['short_id'] for d in report['drops'])}",
            file=sys.stderr)

    if args.output_json:
        print(json.dumps(report, ensure_ascii=True))
    else:
        print(_format_summary(report, output=args.output))
    return 0


def _media_dir_transcribe(media_dir: pathlib.Path):
    """A row -> transcript seam that reads each short's mp4 from ``media_dir``.

    The short's ``video_id`` names the file (``<video_id>.mp4``); a missing file
    transcribes to ``None`` (logged as an untranscribable drop), never aborting.
    """
    media_dir = pathlib.Path(media_dir)

    def transcribe(row: dict) -> str | None:
        video_id = str(row.get("video_id") or row.get("short_id") or "").strip()
        return transcribe_short(media_dir / f"{video_id}.mp4")

    return transcribe


def _format_reasons(reasons: collections.Counter) -> str:
    return ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))


def _format_summary(report: dict, *, output: pathlib.Path) -> str:
    sources = collections.Counter(s["retention_source"] for s in report["shorts"])
    reasons = collections.Counter(d["reason"] for d in report["drops"])
    lines = [
        f"historical_short_mine: {len(report['shorts'])} shorts -> {output}",
        "  retention_source: " + ", ".join(f"{k}={v}" for k, v in sorted(sources.items())),
        f"  drops: {len(report['drops'])}"
        + (f" ({_format_reasons(reasons)})" if reasons else ""),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
