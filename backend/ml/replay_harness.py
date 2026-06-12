"""Offline-Replay-Harness — measure a ranking proposal against existing artifacts.

Replays the *real* serving post-processing on a VOD's already-captured finder
output, with no paid LLM call and no fresh VOD run. For each anchor VOD it
resolves the newest ``pipeline_debug_*`` job dir that holds both the raw dict-rows
(``llm_moment_finder_moments_raw.jsonl`` form) and ``segments.jsonl``, runs the
serving sequence

    snap_boundaries -> dedupe_moments_by_overlap -> enforce_diversity -> top_k

under a pluggable ``rank_score_fn`` (default identity = the current composite
order), and feeds the result straight into ``directional_eval.run``.

The harness *calls* the serving functions, it does not reimplement them — that is
exactly what makes the offline number equal the serving number (no eval/serving
skew). ``rank_score_fn`` is the single variation point: it receives the deduped
rows and returns rows with a possibly-overridden ``composite_score`` (which the
downstream diversity + ranking honour). Missing artifacts are surfaced as a
coverage gap (the VOD is left out of the scored set), never silently counted as a
miss.

Usage (inside the Docker container):
  python -m ml.replay_harness
  python -m ml.replay_harness --json
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Callable, Sequence

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import directional_eval, eval_clip_finder
from ml import llm_moment_finder as d0
from services.moment_finder import _post_processor, _rank_post_processed_rows
from services.transcriber import RichSegment

# The product's clip_top_k. Replaying the same K keeps the offline final set the
# same size as the served one.
DEFAULT_TOP_K = 20

# The two debug artifacts a job dir must hold to be replayable: the raw finder
# rows and the segment timeline snap_boundaries aligns against.
RAW_MOMENTS_ARTIFACT = "llm_moment_finder_moments_raw.jsonl"
SEGMENTS_ARTIFACT = "segments.jsonl"

# The generation-stage debug artifacts a root-cause decomposition reads on top of
# the two replayable ones: the chunked LLM response (truncation proof), the
# per-chunk token usage (the truncation pre-filter) and the stage summary (the
# chunk geometry).
RESPONSE_ARTIFACT = "llm_moment_finder_response.txt"
USAGE_ARTIFACT = "llm_moment_finder_usage.json"
STAGE_ARTIFACT = "llm_moment_finder_stage.json"

# The product's delivered top-20 (post dedupe + enforce_diversity + top_k), in
# product-rank order — the control set the window-probe rates the gold against.
FINAL_MOMENTS_ARTIFACT = "final_moments.json"

# rows -> rows. The single variation point; default identity keeps the current
# composite order. A fn may overwrite each row's composite_score and/or reorder.
RankScoreFn = Callable[[list[dict]], list[dict]]


def _identity(rows: list[dict]) -> list[dict]:
    return rows


def _row_to_moment(row: dict) -> d0.LlmMoment:
    """Rebuild the LlmMoment the dedupe step consumes from a post-processed row.

    The raw artifact rows are the dict form produced by
    ``llm_moments_to_post_processor_rows``; this is its faithful inverse for the
    fields ``dedupe_moments_by_overlap`` reads (window + virality), so the harness
    calls the real dedupe instead of copying its overlap logic.
    """
    return d0.LlmMoment(
        start=float(row.get("start_time") or 0.0),
        end=float(row.get("end_time") or 0.0),
        hook=float(row.get("hook") or 0.0),
        flow=float(row.get("flow") or 0.0),
        value=float(row.get("value") or 0.0),
        trend=float(row.get("trend") or 0.0),
        virality=float(row.get("llm_virality") or 0.0),
        rationale=str(row.get("reason") or ""),
    )


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    """Drop overlap-duplicate rows via the real dedupe, preserving input order.

    Reconstructs one LlmMoment per row, asks ``dedupe_moments_by_overlap`` which
    survive, and keeps the surviving rows in their incoming (snap) order — the
    order the downstream diversity/ranking expect. Survivors are matched back by
    object identity, so two distinct rows that snapped to the same window are
    deduped correctly (a window-key match would keep both).
    """
    moments = [_row_to_moment(row) for row in rows]
    kept, _ = d0.dedupe_moments_by_overlap(moments)
    kept_ids = {id(moment) for moment in kept}
    return [row for row, moment in zip(rows, moments) if id(moment) in kept_ids]


def _resolve_replay_job_dir(storage: pathlib.Path, vod_id: str) -> pathlib.Path | None:
    """Newest ``pipeline_debug_*`` job dir for the VOD that holds *both* artifacts.

    Unlike a newest-outright pick, a newer run missing ``segments.jsonl`` (e.g. a
    serving run with ``eval_recall`` off) is skipped in favour of the newest run
    that is actually replayable. ``None`` means the VOD is a coverage gap.
    """
    candidates = sorted(
        storage.glob(f"pipeline_debug_*_{vod_id}"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for job_dir in candidates:
        debug = job_dir / "debug"
        if (debug / RAW_MOMENTS_ARTIFACT).exists() and (debug / SEGMENTS_ARTIFACT).exists():
            return job_dir
    return None


def _load_segments(path: pathlib.Path) -> list[RichSegment]:
    """Parse ``segments.jsonl`` into the RichSegment timeline snap_boundaries reads."""
    segments: list[RichSegment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        segments.append(
            RichSegment(
                start=float(obj.get("start") or 0.0),
                end=float(obj.get("end") or 0.0),
                text=str(obj.get("text") or ""),
                chat_spike_score=float(obj.get("chat_spike_score") or 0.0),
                voice_energy_burst_score=float(obj.get("voice_energy_burst_score") or 0.0),
            )
        )
    return segments


def load_replay_inputs(
    vod_id: str,
    storage: pathlib.Path,
) -> tuple[list[dict], list[RichSegment]] | None:
    """Load (raw rows, segments) for a VOD's newest replayable job dir.

    Returns ``None`` when no job dir holds both artifacts — the caller surfaces
    that as a coverage gap rather than scoring it as a miss.
    """
    job_dir = _resolve_replay_job_dir(storage, vod_id)
    if job_dir is None:
        return None
    raw_rows = eval_clip_finder._read_moments_jsonl(job_dir / "debug" / RAW_MOMENTS_ARTIFACT)
    segments = _load_segments(job_dir / "debug" / SEGMENTS_ARTIFACT)
    return raw_rows, segments


def _read_json_artifact(path: pathlib.Path) -> dict:
    """A debug JSON artifact as a dict, or ``{}`` when absent/unparseable."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_generation_artifacts(
    vod_id: str,
    storage: pathlib.Path,
) -> tuple[list[RichSegment], str, dict, dict] | None:
    """Load (segments, response_text, usage, stage) for a VOD's replayable job dir.

    Resolves the same newest replayable job dir as ``load_replay_inputs`` and
    reads the generation-stage debug artifacts a root-cause decomposition needs
    on top of the raw rows — ``response.txt`` (the chunked LLM output whose
    mid-JSON abort proves truncation), ``usage.json`` (the per-chunk token
    pre-filter) and ``stage.json`` (the chunk geometry). Returns ``None`` when the
    VOD has no replayable job dir (a coverage gap); ``response_text`` is ``""`` and
    ``usage``/``stage`` are ``{}`` when an individual artifact is missing.
    """
    job_dir = _resolve_replay_job_dir(storage, vod_id)
    if job_dir is None:
        return None
    debug = job_dir / "debug"
    segments = _load_segments(debug / SEGMENTS_ARTIFACT)
    response_path = debug / RESPONSE_ARTIFACT
    response_text = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
    usage = _read_json_artifact(debug / USAGE_ARTIFACT)
    stage = _read_json_artifact(debug / STAGE_ARTIFACT)
    return segments, response_text, usage, stage


def load_final_moments(vod_id: str, storage: pathlib.Path) -> list[dict] | None:
    """The VOD's delivered top-20 (``final_moments.json``) in product-rank order.

    Resolves the same newest replayable job dir as ``load_replay_inputs`` and reads
    the delivered final set via ``eval_clip_finder._read_final_moments_json`` (no
    reimplemented I/O). ``None`` means the VOD has no replayable job dir (a coverage
    gap); an empty list means the job dir holds no/empty ``final_moments.json``.
    """
    job_dir = _resolve_replay_job_dir(storage, vod_id)
    if job_dir is None:
        return None
    return eval_clip_finder._read_final_moments_json(job_dir / "debug" / FINAL_MOMENTS_ARTIFACT)


def _snap_dedupe_rescore(
    raw_rows: Sequence[dict],
    segments: Sequence[RichSegment],
    rank_score_fn: RankScoreFn | None,
) -> list[dict]:
    """snap_boundaries -> dedupe -> rank_score_fn: the shared pre-ranking base.

    The point where ``replay_ranked_pool`` (rank only) and ``replay_postprocess``
    (enforce_diversity + rank + top_k) diverge — every step a call into the real
    serving function, so both stay free of eval/serving skew.
    """
    snapped = _post_processor.snap_boundaries(list(raw_rows), list(segments))
    deduped = _dedupe_rows(snapped)
    return (rank_score_fn or _identity)(deduped)


def replay_ranked_pool(
    raw_rows: Sequence[dict],
    segments: Sequence[RichSegment],
    *,
    rank_score_fn: RankScoreFn | None = None,
) -> list[dict]:
    """The deduped pool, rescored and rank-ordered — no diversity, no top_k.

    snap_boundaries -> dedupe -> rank_score_fn -> rank. This is the candidate
    currency a re-rank/judge operates on (every candidate that survives dedupe,
    in rank order), the seam where the buried-real gold actually sit (median
    rank ~141). ``replay_postprocess`` is enforce_diversity + top_k layered on
    top of this pool.
    """
    return _rank_post_processed_rows(_snap_dedupe_rescore(raw_rows, segments, rank_score_fn))


def replay_postprocess(
    raw_rows: Sequence[dict],
    segments: Sequence[RichSegment],
    *,
    rank_score_fn: RankScoreFn | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Replay the serving post-processing over raw finder rows.

    snap_boundaries -> dedupe -> rank_score_fn -> enforce_diversity -> rank ->
    top_k, every step a call into the real serving function. Under the identity
    ``rank_score_fn`` (the default) the output is bit-equal to the serving path on
    the same input. ``top_k > 0`` truncates to the K highest-ranked rows.
    """
    rescored = _snap_dedupe_rescore(raw_rows, segments, rank_score_fn)
    diverse = _post_processor.enforce_diversity(rescored)
    ranked = _rank_post_processed_rows(diverse)
    return ranked[:top_k] if top_k > 0 else ranked


def run_replay(
    anchor_ids: Sequence[str],
    storage: pathlib.Path,
    manifest: dict[str, list[dict]],
    *,
    rank_score_fn: RankScoreFn | None = None,
    top_k: int = DEFAULT_TOP_K,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: Sequence[str] = (directional_eval.GOLD_TIER,),
) -> dict:
    """Replay every anchor VOD and score the result through ``directional_eval.run``.

    For each VOD it loads the raw rows + segments, replays the serving
    post-processing under ``rank_score_fn``, and feeds both the replayed final set
    and the raw rows to ``directional_eval`` (so the same pass yields product
    recall and the generation ceiling). VODs without replayable artifacts are
    surfaced as a coverage gap — left out of the scored set and logged, never
    counted as silent misses.
    """
    final_by_vod: dict[str, list[dict]] = {}
    raw_by_vod: dict[str, list[dict]] = {}
    gaps: list[str] = []
    for vod_id in anchor_ids:
        loaded = load_replay_inputs(vod_id, storage)
        if loaded is None:
            gaps.append(vod_id)
            continue
        raw_rows, segments = loaded
        raw_by_vod[vod_id] = raw_rows
        final_by_vod[vod_id] = replay_postprocess(
            raw_rows, segments, rank_score_fn=rank_score_fn, top_k=top_k
        )
    if gaps:
        # No silent truncation: the gap is logged here AND surfaced in the result
        # by directional_eval (missing_prediction_vods).
        print(
            f"replay_harness: {len(gaps)}/{len(anchor_ids)} anchor VODs have no "
            f"replayable artifacts — coverage gap, excluded from recall: {', '.join(gaps)}",
            file=sys.stderr,
        )
    return directional_eval.run(
        manifest,
        final_by_vod,
        raw_by_vod,
        anchor_vod_ids=list(anchor_ids),
        ks=ks,
        headline_min_views=headline_min_views,
        gold_tiers=tuple(gold_tiers),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — identity replay over the anchor set, scored through directional_eval.
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--ks", default="5,10,20,50", help="Recall@K comma-separated")
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
                        help="Min views for the headline tier (interim 100000)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Final-set size the replay truncates to (the product clip_top_k)")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Write the full result JSON here")
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))

    result = run_replay(
        anchors,
        pathlib.Path(args.predictions_dir),
        manifest,
        top_k=args.top_k,
        ks=ks,
        headline_min_views=args.headline_min_views,
    )
    result["computed_at"] = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.output_json:
        print(payload)
    else:
        print(directional_eval.format_report(result))
        print(f"\ncomputed_at: {result['computed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
