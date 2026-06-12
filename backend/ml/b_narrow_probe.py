"""B-narrow Regen-Probe — recoverable subset (#65) + in-context re-gen (#66), PRD #64.

The B-narrow probe: it selects the *recoverable subset* — the llm_skipped golds a
stronger model (#60 window-probe, gemini-2.5-pro) lifted clearly above the
delivered competition — then re-generates those VODs in-context with that same
strong model, so the #67 product-recall@20 gate measures whether it surfaces them
into the shipped top-20. It does NOT add a new measurement layer; it **composes**
existing seams:

  * `strong_model_window_probe` (#60/#63) — the persisted per-window scores and the
    within-set ``analyze_separation`` (percentile + the pre-fixed 0.66 cut),
  * (via that) the #56 ``llm_skipped`` origin + #49 ``is_hit`` tier filtering,
  * `replay_harness.load_replay_inputs` (#47, #65-fixed loader) — the subset VODs'
    segments, parity-preserving so the re-gen prompt matches production, and
  * `services.moment_finder.find_moments` — the generation seam, model swapped to
    gemini-2.5-pro (only the model changes; production chunking/max_moments/tokens).

Selection (#65): the golds with within-set percentile **>= 0.66**, mechanical, never
hand-picked. Re-gen (#66): the single paid step — ``--regen`` generates + persists
the new raw pool per VOD (re-evaluable), ``--replay`` re-reads it net-free; a fake
generator keeps every unit test off the network. The postprocess + gate is #67.

Usage (inside the Docker container):
  python -m ml.b_narrow_probe                    # free recoverable-subset selection (#65)
  python -m ml.b_narrow_probe --regen            # PAID: re-gen the subset, persist pools (#66)
  python -m ml.b_narrow_probe --replay [--json]  # free re-read of the persisted pools
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys
from typing import Callable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import (eval_clip_finder, replay_harness, separability_probe as sp,
                strong_model_window_probe as swp)

logger = logging.getLogger(__name__)

# The in-context re-gen swaps ONLY the generation model — the strong frontier
# model that separated in the #60 probe — keeping production chunking / max_moments
# / max_tokens. Where the new raw pools persist (re-evaluable without re-pay).
REGEN_PROVIDER = "gemini"
REGEN_MODEL = "gemini-2.5-pro"
DEFAULT_PROFILE = "jynxzi"
DEFAULT_REGEN_DIR = BACKEND_ROOT / "ml" / "artifacts" / "b_narrow_regen"


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
# In-context re-gen — the single paid step (generate raw pool, persist, replay)
# ─────────────────────────────────────────────────────────────────────────────
# (segments, vod_id) -> the re-generated raw moment rows. The injected generation
# adapter: real paid gemini-2.5-pro find_moments, or a fake/canned one in tests
# (the two-adapter discipline — the network never touches a unit test).
GenerateFn = Callable[[list, str], list[dict]]
# (vod_id, segments) -> raw rows: paid generate+persist, or replay-from-disk.
MakeRegenFn = Callable[[str, list], list[dict]]


def _regen_path(regen_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return regen_dir / f"{vod_id}.regen_raw.jsonl"


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def gemini_pro_generator(
    *,
    find_moments_fn: Callable | None = None,
    profile: str = DEFAULT_PROFILE,
    model: str = REGEN_MODEL,
) -> GenerateFn:
    """The real paid adapter: re-generate raw moments with the strong model.

    Wraps the serving ``find_moments`` generation seam, swapping ONLY the model
    (``provider="gemini"`` + ``gemini-2.5-pro``) and running with ``eval_recall``
    so the raw pool is produced. It deliberately passes no chunking / max_moments /
    max_tokens, so ``find_moments`` keeps its production defaults — the comparison
    isolates the generation model. ``find_moments_fn`` is injected in tests.
    """
    if find_moments_fn is None:
        from services.moment_finder import find_moments as find_moments_fn

    def _generate(segments: list, vod_id: str) -> list[dict]:
        result = find_moments_fn(
            segments, profile=profile, vod_id=vod_id,
            provider=REGEN_PROVIDER, model=model, eval_recall=True)
        return list(result.raw_moments)

    return _generate


def paid_make_regen(regen_dir: pathlib.Path, *, generate: GenerateFn) -> MakeRegenFn:
    """Per-VOD rater bound to the paid generator, persisting the raw pool to disk.

    Generates the new raw moments via ``generate`` (real gemini-2.5-pro in prod, a
    fake in tests) and persists them as ``<vod_id>.regen_raw.jsonl`` — the
    re-evaluable artifact a later ``--replay`` re-reads without paying again.
    """
    def _make(vod_id: str, segments: list) -> list[dict]:
        rows = generate(segments, vod_id)
        _write_jsonl(_regen_path(regen_dir, vod_id), rows)
        return rows

    return _make


def replay_make_regen(regen_dir: pathlib.Path) -> MakeRegenFn:
    """Per-VOD rater bound to the persisted raw pool — re-reads it, never re-pays."""
    def _make(vod_id: str, _segments: list) -> list[dict]:
        return _read_jsonl(_regen_path(regen_dir, vod_id))

    return _make


def run_regen(vod_ids: list[str], segments_by_vod: dict[str, list], *,
              make_regen: MakeRegenFn) -> dict:
    """Re-generate the raw pool for each recoverable-subset VOD, surfacing caps.

    Iterates the subset's distinct VODs and, per VOD, generates (or replays) the
    new raw moments via ``make_regen``. Every cap is surfaced, never silent: a VOD
    with no loadable segments is named (``no_segments``), and a VOD whose
    generation came back empty is named too (``empty_generation``) — so a fail
    later distinguishes "not generated" from "generated but buried" (#67).
    """
    vods: dict[str, dict] = {}
    no_segments: list[str] = []
    empty_generation: list[str] = []
    for vod_id in vod_ids:
        segments = segments_by_vod.get(vod_id)
        if not segments:
            no_segments.append(vod_id)
            logger.info(json.dumps(
                {"event": "b_narrow_regen_no_segments", "vod_id": vod_id}, sort_keys=True))
            continue
        rows = make_regen(vod_id, segments)
        vods[vod_id] = {"raw_moments": rows, "count": len(rows)}
        if not rows:
            empty_generation.append(vod_id)
        logger.info(json.dumps(
            {"event": "b_narrow_regen", "vod_id": vod_id, "raw_moments": len(rows)},
            sort_keys=True))
    return {
        "vods": vods,
        "counts": {"vods": len(vods), "raw_moments": sum(v["count"] for v in vods.values())},
        "caps": {"no_segments": no_segments, "empty_generation": empty_generation},
        "model": REGEN_MODEL,
    }


def build_regen_inputs(vod_ids: list[str], storage: pathlib.Path) -> dict[str, list]:
    """Load each subset VOD's segments via the parity-preserving #65 loader.

    Composes ``replay_harness.load_replay_inputs`` (the #65-fixed ``_load_segments``,
    so the re-gen prompt is identical to production) — no re-download / re-transcribe,
    only the later generation call costs. A VOD with no loadable artifacts is simply
    absent from the result, surfaced downstream by ``run_regen`` as ``no_segments``.
    """
    segments_by_vod: dict[str, list] = {}
    for vod_id in vod_ids:
        loaded = replay_harness.load_replay_inputs(vod_id, storage)
        if loaded is None:
            continue
        _raw_rows, segments = loaded
        segments_by_vod[vod_id] = segments
    return segments_by_vod


# ─────────────────────────────────────────────────────────────────────────────
# Reporting + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _format_regen(regen: dict) -> str:
    caps = regen["caps"]
    lines = [
        "",
        "=" * 78,
        f"IN-CONTEXT RE-GEN ({regen['model']}): new raw pool per recoverable-subset VOD",
        "=" * 78,
        f"Generated: {regen['counts']['vods']} VODs, "
        f"{regen['counts']['raw_moments']} raw moments total",
    ]
    if caps["no_segments"]:
        lines.append(f"⚠  No loadable segments (skipped): {', '.join(caps['no_segments'])}")
    if caps["empty_generation"]:
        lines.append(f"⚠  Empty generation (not buried — not generated): "
                     f"{', '.join(caps['empty_generation'])}")
    for vod_id, entry in regen["vods"].items():
        lines.append(f"    {vod_id}: {entry['count']} raw moments")
    return "\n".join(lines)

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
    if result.get("regen"):
        lines.append(_format_regen(result["regen"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", default=str(swp.DEFAULT_ARTIFACTS_DIR),
                        help="Where the #60 window-probe scores (*.scores.json) live")
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR),
                        help="pipeline_debug_* dirs to load the subset VODs' segments from")
    parser.add_argument("--regen-dir", default=str(DEFAULT_REGEN_DIR),
                        help="Where the re-generated raw pools persist (re-evaluable)")
    parser.add_argument("--threshold", type=float, default=sp.SEPARATION_THRESHOLD,
                        help="Within-set percentile cut (the pre-fixed #60 0.66 bar)")
    parser.add_argument("--regen", action="store_true",
                        help="PAID: re-generate the subset with gemini-2.5-pro, persist the pool")
    parser.add_argument("--replay", action="store_true",
                        help="Re-read the persisted re-gen pools — NO paid call")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    rating = swp.load_persisted_rating(pathlib.Path(args.artifacts_dir))
    result = select_recoverable(rating, threshold=args.threshold)

    # Optional paid step (#66): re-generate the recoverable subset in-context.
    # --regen is the single paid gemini-2.5-pro call/VOD; --replay re-reads the
    # persisted pools net-free. Default (neither) is the free subset selection.
    if args.regen or args.replay:
        segments_by_vod = build_regen_inputs(result["vods"], pathlib.Path(args.predictions_dir))
        regen_dir = pathlib.Path(args.regen_dir)
        if args.replay:
            make_regen = replay_make_regen(regen_dir)
            mode = "replay"
        else:
            make_regen = paid_make_regen(regen_dir, generate=gemini_pro_generator())
            mode = "paid"
        print(f"b_narrow_probe: re-gen mode={mode}  regen_dir={regen_dir}", file=sys.stderr)
        result["regen"] = run_regen(result["vods"], segments_by_vod, make_regen=make_regen)
        result["regen_mode"] = mode

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
