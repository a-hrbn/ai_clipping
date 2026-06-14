"""Historical-short-mine — read-only retention ingest + per-short report + aggregate.

PRD #75 (Step 0). The cheapest possible first look at the first-party-outcome
pivot (ADR docs/adr/0001-first-party-outcome-flywheel.md): turn a locally exported
YT audience-retention curve for the 52 already-posted @donni_clips shorts into a
per-short report *and* a directional aggregate, *before* the forward loop is built.

Ingest (Slice 1, #76): a tolerant parser derives per-short
``retention_3s``/``retention_50pct``/``completion`` from the YT-*relative*
audience-retention curve (``elapsedVideoTimeRatio`` -> ``audienceWatchRatio``)
plus ``averageViewPercentage`` as the degraded fallback. Retention is read
**views-decoupled** (views are a secondary field, never a positive-gate) per the
ADR's cold-start win. The report is keyed by ``row_identity`` so re-runs are
idempotent and join-able with a later VOD attribution.

Facet (Slice 2, #77): each short is transcribed (mp4 -> wav -> text) and classified
into its facet — goal-family (``infer_goal_family``, transcript as topic) ×
length-bucket — from the short's *own* content (no VOD attribution, no chat-emotion).

Aggregate (Slice 3, #78): the diagnostic roll-up — overall retention spread, median
retention per goal-family×length (with n), a τ-baseline percentile suggestion, and
an honest drop summary, under an explicit caveat header (n / packaging / dormant).
It **reports, it does not route**: no pre-fixed gate, no verdict — n=52 +
packaging-confounding + dormancy permit only a directional read.

Offline only: no network, no paid call, no serving touch, no new pipeline stage.
Built against the public YT-Analytics schema; reconciliation against the real
export format is deferred to Slice 4 (HITL).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
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
# Aggregate — the diagnostic roll-up (reports, routes not)
#
# The directional pre-look over the per-short rows (PRD #75 Slice 3, issue #78):
# overall retention spread, median retention per goal-family×length (with n), a
# τ-baseline percentile suggestion, the caveat header and an honest drop summary.
# Deliberately NO gate / verdict / route — n=52 + packaging-confounding + dormancy
# allow only a directional read, never a decision baked into code.
# ─────────────────────────────────────────────────────────────────────────────

# The three retention fields every aggregate ranges over (same order as the report).
RETENTION_FIELDS = ("retention_3s", "retention_50pct", "completion")

# The forward loop gates on early-retention and completion — the τ-baseline suggests
# thresholds for *those* fields only (retention_50pct is not a gate knob).
TAU_FIELDS = ("retention_3s", "completion")

# The percentile cut points the τ-baseline reports as candidate thresholds.
TAU_PERCENTILES = (0.25, 0.50, 0.75)

# The caveat header, surfaced verbatim so the directional read is never over-read as
# a decision: the sample is thin (~8 per goal-family), packaging is uncontrolled
# (confounded), and the channel has been dormant since 10.04. (lukewarm signal).
CAVEATS = (
    "n=52 dünn → ~8/Goal-Family direktional",
    "packaging-konfundiert → suggestiv",
    "Kanal seit 10.04. dormant → lauwarm",
)


def aggregate(report: dict) -> dict:
    """The diagnostic roll-up over the per-short report — reports, routes not.

    Composes the overall retention spread, the per-facet median retention (with n),
    the τ-baseline percentile suggestion and an honest drop summary, under the
    explicit caveat header (n / packaging / dormant). There is deliberately **no**
    gate, verdict or route: n=52 + packaging-confounding + dormancy permit only a
    directional read, so the decision stays with the reader, never baked into code.
    """
    shorts = report["shorts"]
    return {
        "caveats": list(CAVEATS),
        "n_shorts": len(shorts),
        "retention_spread": retention_spread(shorts),
        "facet_medians": facet_medians(shorts),
        "tau_baseline": tau_baseline(shorts),
        "drop_summary": drop_summary(report),
    }


def facet_medians(shorts: list[dict]) -> list[dict]:
    """Median retention per facet cell (goal-family × length-bucket), with n.

    Groups the kept shorts by their (goal_family, length_bucket) facet and reports,
    for each cell, its short count ``n`` and the median of each retention field over
    the shorts in the cell that carry it (degraded avp-only shorts contribute None
    and are skipped per field — never silently counted as a 0). Cells are sorted by
    facet so the aggregate is input-order independent.
    """
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for short in shorts:
        groups[(short.get("goal_family"), short.get("length_bucket"))].append(short)
    cells = [
        {
            "goal_family": goal_family,
            "length_bucket": length_bucket,
            "n": len(members),
            "median": {f: _median([m.get(f) for m in members]) for f in RETENTION_FIELDS},
        }
        for (goal_family, length_bucket), members in groups.items()
    ]
    cells.sort(key=lambda c: (str(c["goal_family"]), str(c["length_bucket"])))
    return cells


def _median(values: list) -> float | None:
    """Median over the non-None values, or None when the cell has none present."""
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def retention_spread(shorts: list[dict]) -> dict:
    """Overall retention variation per field — the "does it vary meaningfully?" read.

    For each retention field reports n, min/median/max, the inter-quartile range
    (p75 − p25) and the sample stdev over all kept shorts that carry the field.
    Degraded (avp-only) shorts carry None and are skipped per field — never folded
    in as a 0 that would understate or inflate the spread.
    """
    return {f: _field_spread([s.get(f) for s in shorts]) for f in RETENTION_FIELDS}


def _field_spread(values: list) -> dict:
    present = sorted(v for v in values if v is not None)
    n = len(present)
    if n == 0:
        return {k: None for k in ("min", "p25", "median", "p75", "max", "stdev", "iqr")} | {"n": 0}
    p25 = _percentile(present, 0.25)
    p75 = _percentile(present, 0.75)
    return {
        "n": n,
        "min": present[0],
        "p25": p25,
        "median": _percentile(present, 0.5),
        "p75": p75,
        "max": present[-1],
        "stdev": statistics.stdev(present) if n >= 2 else None,
        "iqr": p75 - p25,
    }


def tau_baseline(shorts: list[dict]) -> dict:
    """A τ-baseline threshold suggestion from real retention percentiles.

    For each gate field (retention_3s / completion) reports the p25/p50/p75 cut
    points so the forward loop's thresholds are set from real numbers rather than a
    gut guess. A suggestion, not a gate — the module reports, it does not route.
    """
    return {
        field: {
            f"p{int(q * 100)}": _percentile([s.get(field) for s in shorts], q)
            for q in TAU_PERCENTILES
        }
        for field in TAU_FIELDS
    }


def drop_summary(report: dict) -> dict:
    """Per-bottleneck drop breakdown with an honest denominator.

    ``n_input`` counts kept + dropped shorts (the real total, not just the
    survivors) so a silent cap can never read as "all 52 covered". ``by_reason``
    surfaces every bottleneck that fired — ``missing_retention_row`` (no retention)
    and ``untranscribable`` (no transcript, hence no classifiable facet either).
    """
    drops = report["drops"]
    n_kept = len(report["shorts"])
    n_dropped = len(drops)
    return {
        "n_input": n_kept + n_dropped,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "by_reason": dict(collections.Counter(d["reason"] for d in drops)),
    }


def _percentile(values: list, q: float) -> float | None:
    """Linear-interpolation percentile (q in [0,1]) — numpy/`inclusive`-style.

    Hand-computable: the query lands at rank ``q*(n-1)`` in the sorted values and
    interpolates between the bracketing ranks. None-safe; None when no value present.
    """
    present = sorted(v for v in values if v is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    pos = q * (len(present) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(present) - 1)
    return present[lo] + (pos - lo) * (present[hi] - present[lo])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "historical_short_mine_per_short.jsonl"
DEFAULT_AGGREGATE_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "historical_short_mine_aggregate.json"


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
    parser.add_argument("--aggregate-output", type=pathlib.Path, default=DEFAULT_AGGREGATE_OUTPUT,
                        help="Aggregate report JSON destination (spread, facet medians, "
                             "τ-baseline, caveats, drop summary).")
    parser.add_argument("--json", dest="output_json", action="store_true",
                        help="Print the full report (shorts + drops + aggregate) as JSON.")
    args = parser.parse_args(argv)

    if transcribe is None and args.media_dir is not None:
        transcribe = _media_dir_transcribe(args.media_dir)

    raw_rows = row_identity.read_jsonl(pathlib.Path(args.metrics), strict=False)
    report = build_short_report(raw_rows, transcribe=transcribe)
    report_aggregate = aggregate(report)

    row_identity.write_jsonl(args.output, report["shorts"])
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_output.write_text(
        json.dumps(report_aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
        print(json.dumps({**report, "aggregate": report_aggregate}, ensure_ascii=True))
    else:
        print(_format_summary(report, output=args.output))
        print(format_aggregate(report_aggregate))
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


def _fmt_num(value) -> str:
    if value is None:
        return "  —  "
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def format_aggregate(report_aggregate: dict) -> str:
    """The directional aggregate as a legible block — reports, never routes.

    Surfaces the caveat header verbatim, the overall retention spread, the per-facet
    median retention (with n), the τ-baseline percentiles and the honest drop
    summary. No verdict line: the read stays with the operator.
    """
    drop = report_aggregate["drop_summary"]
    lines = [
        "",
        "  AGGREGATE — directional pre-look (berichtet, routet nicht):",
        f"  ⚠  Caveats: {' | '.join(report_aggregate['caveats'])}",
        f"  n_shorts={report_aggregate['n_shorts']}  "
        f"(kept {drop['n_kept']}/{drop['n_input']}, dropped {drop['n_dropped']}"
        + (f": {_format_reasons(collections.Counter(drop['by_reason']))}" if drop["by_reason"] else "")
        + ")",
        "  RETENTION SPREAD (does it vary meaningfully?):",
    ]
    for field in RETENTION_FIELDS:
        s = report_aggregate["retention_spread"][field]
        lines.append(
            f"    {field:<15} n={s['n']:>2}  min {_fmt_num(s['min'])}  median {_fmt_num(s['median'])}  "
            f"max {_fmt_num(s['max'])}  iqr {_fmt_num(s['iqr'])}  stdev {_fmt_num(s['stdev'])}")
    lines.append("  MEDIAN RETENTION per goal-family×length (with n):")
    for cell in report_aggregate["facet_medians"]:
        med = cell["median"]
        lines.append(
            f"    {str(cell['goal_family']):<12} {str(cell['length_bucket']):<5} n={cell['n']:>2}  "
            f"r3s {_fmt_num(med['retention_3s'])}  r50 {_fmt_num(med['retention_50pct'])}  "
            f"compl {_fmt_num(med['completion'])}")
    lines.append("  τ-BASELINE (percentile threshold suggestions; not a gate):")
    for field, cuts in report_aggregate["tau_baseline"].items():
        lines.append(
            f"    {field:<15} p25 {_fmt_num(cuts['p25'])}  p50 {_fmt_num(cuts['p50'])}  "
            f"p75 {_fmt_num(cuts['p75'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
