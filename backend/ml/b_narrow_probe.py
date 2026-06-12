"""B-narrow Regen-Probe — data foundation (PRD #64, Phase 1).

The free, offline foundation of the B-narrow probe: it selects the *recoverable
subset* — the llm_skipped golds a stronger model (#60 strong-model window-probe,
gemini-2.5-pro) lifted clearly above the delivered competition — so a later
in-context re-gen (#66) and product-recall@20 gate (#67) operate on exactly that
subset. It does NOT add a new measurement layer; it **composes** existing seams:

  * `strong_model_window_probe` (#60/#63) — the persisted per-window scores and the
    within-set ``analyze_separation`` (the probability-of-superiority percentile +
    the pre-fixed 0.66 cut), consumed read-only, and
  * (via that) the #56 ``llm_skipped`` origin + #49 ``is_hit`` tier filtering.

The recoverable subset is the golds with within-set percentile **>= 0.66** — the
same pre-fixed #60 cut, read mechanically off the persisted scores, never
hand-picked. The distinct subset VODs + golds + the thin-n size are surfaced.
No paid/network call (the paid re-gen is #66).

Usage (inside the Docker container):
  python -m ml.b_narrow_probe
  python -m ml.b_narrow_probe --json
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import separability_probe as sp, strong_model_window_probe as swp


def select_recoverable(rating: dict, *, threshold: float = sp.SEPARATION_THRESHOLD) -> dict:
    """The recoverable subset: llm_skipped golds with within-set percentile >= cut.

    Composes the #60 ``analyze_separation`` over a ``rating`` (the per-VOD gold +
    control scores) and keeps the golds it marks ``separated`` (percentile >=
    ``threshold``, the pre-fixed 0.66 cut). The selection is mechanical — the cut
    decides, nothing is hand-picked — and surfaces the distinct subset VODs, the
    gold windows themselves, and the thin-n size (subset of all golds considered).
    """
    separation = swp.analyze_separation(rating, threshold=threshold)
    recoverable = [w for w in separation["windows"] if w["separated"]]
    return {
        "threshold": threshold,
        "recoverable": recoverable,
        "vods": sorted({w["vod_id"] for w in recoverable}),
        "count": len(recoverable),
        "headline_count": sum(1 for w in recoverable if w["headline"]),
        "total_golds": len(separation["windows"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting + CLI
# ─────────────────────────────────────────────────────────────────────────────

def format_report(result: dict) -> str:
    lines = [
        "=" * 78,
        "B-narrow recoverable subset — llm_skipped golds the strong model separated",
        "=" * 78,
        f"Source: #60 strong-model window-probe scores   cut: within-set pct >= "
        f"{result['threshold']}   (mechanical, not hand-picked)",
        f"Recoverable: {result['count']}/{result['total_golds']} golds "
        f"({result['headline_count']} headline) across {len(result['vods'])} VODs — "
        f"thin n, read as an existence set, not a statistic.",
    ]
    if result["recoverable"]:
        lines.append("")
        lines.append("  RECOVERABLE GOLDS (the #66 re-gen target subset):")
        for w in sorted(result["recoverable"], key=lambda x: -(x["percentile"] or 0.0)):
            tier = "headline" if w["headline"] else "floor"
            lines.append(
                f"    {w['vod_id']}  [{tier:^8}] pct {w['percentile']:.3f} "
                f"score {w['score']:.2f}  {str(w.get('desc') or '')[:54]}")
        lines.append("")
        lines.append(f"  Distinct VODs: {', '.join(result['vods'])}")
    else:
        lines.append("")
        lines.append("  No recoverable golds at this cut (or no #60 scores found).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", default=str(swp.DEFAULT_ARTIFACTS_DIR),
                        help="Where the #60 window-probe scores (*.scores.json) live")
    parser.add_argument("--threshold", type=float, default=sp.SEPARATION_THRESHOLD,
                        help="Within-set percentile cut (the pre-fixed #60 0.66 bar)")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    rating = swp.load_persisted_rating(pathlib.Path(args.artifacts_dir))
    result = select_recoverable(rating, threshold=args.threshold)
    result["computed_at"] = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.output_json:
        print(payload)
    else:
        print(format_report(result))
        print(f"\ncomputed_at: {result['computed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
