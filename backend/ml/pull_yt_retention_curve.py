"""Curve-driven Mine — read-only pull of the YT audience-retention curve behind a seam.

PRD #80 (Step 0.5), issue #81 (Slice 1). #75 read the 52 already-posted shorts
*completion-only* (the curve-less Studio CSV); the forward loop's hard-negatives,
however, gate on the *early-hook* fields ``retention_3s``/``retention_50pct`` —
exactly what completion-only never measured. This module closes the end-to-end path
"curve -> retention fields" behind **one injectable fetcher seam**, fully offline
against a fake (no network, no creds in this slice).

It composes only existing seams: ``read_metrics`` reads the curve-less Studio rows,
the injected ``fetch_curve`` returns each short's ``audienceWatchRatio`` curve, the
curve is merged onto the row by ``video_id`` as ``audience_retention``, and the
**unchanged** ``historical_short_mine`` derives ``retention_3s/50pct/completion`` off
it (instead of completion-only). Loop-inflated curves (values > 1.0) are passed
through, never treated as an error. A short the fetcher returns no curve for falls to
the existing degraded path (``studio_endwatch`` / ``degraded_avp_only``), is flagged
and logged, and is kept — never an aborted run. Short identity is stable via
``row_identity`` (``clips:<video_id>``) so re-runs are idempotent.

The real read-only ``youtubeAnalytics.reports.query`` adapter is a later slice — the
default fetcher is wired but defers to it, so a no-fetcher run refuses clearly instead
of silently fetching nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Callable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import historical_short_mine as hsm, row_identity

# The fetcher seam: a short's ``video_id`` -> its ``audienceWatchRatio`` curve as the
# API yields it (``[{elapsedVideoTimeRatio, audienceWatchRatio}, ...]``), or an empty
# list / None when no curve exists for that short. Injected in tests (canned curves,
# no network); the default wiring defers to the real read-only adapter (later slice).
Fetcher = Callable[[str], "list[dict] | None"]


def merge_curves(rows: list[dict], fetch_curve: Fetcher) -> list[dict]:
    """Attach each Studio row's fetched ``audienceWatchRatio`` curve by ``video_id``.

    Returns new rows (inputs are never mutated): a row the fetcher delivers a curve
    for carries it as ``audience_retention`` (the key ``derive_retention`` already
    reads); a row with no curve is passed through unchanged so the unchanged mine
    degrades it honestly. Loop-inflated curves are merged verbatim, never validated.
    """
    return [_with_curve(row, fetch_curve) for row in rows]


def _with_curve(row: dict, fetch_curve: Fetcher) -> dict:
    video_id = _video_id(row)
    curve = fetch_curve(video_id) if video_id else None
    if not curve:
        return dict(row)
    return {**row, "audience_retention": curve}


def pull_report(rows: list[dict], *, fetch_curve: Fetcher) -> dict:
    """Merge fetched curves onto the Studio rows and run the UNCHANGED mine over them.

    The per-short report is exactly what ``historical_short_mine.build_short_report``
    emits — only now its retention fields are curve-driven for every short the fetcher
    delivered a curve for, and the degraded read for the rest.
    """
    return hsm.build_short_report(merge_curves(rows, fetch_curve))


def _video_id(row: dict) -> str:
    return str(row.get("video_id") or row.get("short_id") or "").strip()


def default_fetcher(video_id: str) -> list[dict]:
    """The wired default — defers to the real read-only YT-Analytics adapter.

    The seam ships with a default so production needs no injection, but the read-only
    ``youtubeAnalytics.reports.query`` adapter (Scope ``yt-analytics.readonly``) is a
    later slice. Until it lands the default refuses *clearly* rather than silently
    returning no curve (which would mis-read as "all shorts degraded") or hitting the
    network from a slice that is meant to stay offline. Tests inject a fake fetcher.
    """
    raise NotImplementedError(
        "pull_yt_retention_curve: the real read-only youtubeAnalytics.reports.query "
        "adapter (yt-analytics.readonly) is a later slice — inject a fetcher to run "
        f"offline (video_id={video_id!r}).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "pull_yt_retention_curve_per_short.jsonl"
DEFAULT_AGGREGATE_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "pull_yt_retention_curve_aggregate.json"


def main(argv: list[str] | None = None, *, fetch_curve: Fetcher | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--metrics", required=True,
        help="Curve-less Studio export (CSV) of the already-posted shorts; read offline.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                        help="Per-short report JSONL destination.")
    parser.add_argument("--aggregate-output", type=pathlib.Path, default=DEFAULT_AGGREGATE_OUTPUT,
                        help="Aggregate report JSON destination (spread, facet medians, "
                             "τ-baseline, caveats, drop summary).")
    parser.add_argument("--json", dest="output_json", action="store_true",
                        help="Print the full report (shorts + drops + aggregate) as JSON.")
    args = parser.parse_args(argv)

    if fetch_curve is None:
        fetch_curve = default_fetcher

    raw_rows = hsm.read_metrics(args.metrics)
    report = pull_report(raw_rows, fetch_curve=fetch_curve)
    report_aggregate = hsm.aggregate(report)

    row_identity.write_jsonl(args.output, report["shorts"])
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_output.write_text(
        json.dumps(report_aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _log_degraded(report["shorts"])
    if report["drops"]:
        reasons = collections.Counter(d["reason"] for d in report["drops"])
        print(
            f"pull_yt_retention_curve: dropped {len(report['drops'])}/{len(raw_rows)} "
            f"shorts ({_format_counter(reasons)}): "
            f"{', '.join(d['short_id'] for d in report['drops'])}",
            file=sys.stderr)

    if args.output_json:
        print(json.dumps({**report, "aggregate": report_aggregate}, ensure_ascii=True))
    else:
        sources = collections.Counter(s["retention_source"] for s in report["shorts"])
        print(f"pull_yt_retention_curve: {len(report['shorts'])} shorts -> {args.output}")
        print("  retention_source: " + _format_counter(sources))
        print(hsm.format_aggregate(report_aggregate))
    return 0


def _log_degraded(shorts: list[dict]) -> None:
    """Surface every short the fetcher returned no curve for — no silent caps.

    A short kept on the degraded Studio read (``retention_source`` not ``"curve"``)
    had no API curve; it is logged here so a curve-less short never reads as "curve
    covered all 52". Mirrors ``historical_short_mine``'s drop logging.
    """
    degraded = [s for s in shorts if s["retention_source"] != "curve"]
    if not degraded:
        return
    sources = collections.Counter(s["retention_source"] for s in degraded)
    print(
        f"pull_yt_retention_curve: no API curve for {len(degraded)}/{len(shorts)} "
        f"shorts ({_format_counter(sources)}): "
        f"{', '.join(s['short_id'] for s in degraded)}",
        file=sys.stderr)


def _format_counter(counter: collections.Counter) -> str:
    """A ``Counter`` as a stable ``key=count, key=count`` string (sorted for re-runs)."""
    return ", ".join(f"{key}={count}" for key, count in sorted(counter.items()))


if __name__ == "__main__":
    raise SystemExit(main())
