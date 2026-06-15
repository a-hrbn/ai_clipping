"""WALL-Re-Test re-run driver — re-run the banter/tic subset with the cheap levers.

The production half of the WALL re-test (PRD #90, Slice 2 of #92): pick the
verbal-tic-/banter-heavy subset, then re-run the *real* pipeline over it with the
two cheap levers on — the Jynxzi **glossary** biased into Whisper
(``initial_prompt``) and **diarization** with the speaker rendered into the finder
transcript — writing fresh prediction dirs into their *own* directory so the
``base`` dir (the WALL status quo) is never touched. The model stays
``large-v3-turbo`` — glossary + diarization are the only changed inputs.

The subset is the VODs whose ``gt_moment.desc`` names a glossary entity (the
banter/tic signal) intersected with the VODs that have a replayable ``base`` dir,
so Slice 1's ``wall_retest_compare`` diffs the same VODs base-vs-neu.

Usage (inside the Docker container):
  python -m ml.wall_retest_rerun --predictions-dir tmp/storage_neu
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
from typing import Callable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import replay_harness

# vod_id, predictions_dir -> the fresh job dir, or None when the re-run failed.
RunVodFn = Callable[[str, pathlib.Path], "pathlib.Path | None"]


@dataclasses.dataclass(frozen=True)
class RerunResult:
    predictions_dir: pathlib.Path
    completed: list[str]   # VODs that produced a fresh neu prediction dir
    drops: list[str]       # VODs whose re-run failed — logged, never silent

# The Jynxzi glossary, in two shapes from one source of truth: the comma-joined
# string biased into Whisper via initial_prompt, and the per-entity term list the
# subset selector matches against gt_moment.desc to find banter/tic-heavy VODs.
GLOSSARY_TERMS: tuple[str, ...] = (
    "Balright",
    "Darth Maura",
    "Me ho",
    "Hivise",
    "StableRonaldo",
    "Dantes",
    "Tyler1",
    "Caseoh",
)
JYNXZI_GLOSSARY = ", ".join(GLOSSARY_TERMS)


def rerun_subset(
    subset: list[str],
    predictions_dir: pathlib.Path,
    *,
    run_vod: RunVodFn,
) -> RerunResult:
    """Re-run every subset VOD via ``run_vod`` into ``predictions_dir`` (the neu dir).

    ``run_vod`` is the injected leaf that actually re-runs the pipeline for one VOD
    (glossary + diarization on) and returns its fresh job dir, or ``None`` on
    failure. The base dir is never passed in here, so it stays untouched. Each VOD
    is independent: one failure becomes a logged drop, the rest keep going.
    """
    predictions_dir = pathlib.Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    drops: list[str] = []
    for vod_id in subset:
        try:
            job_dir = run_vod(vod_id, predictions_dir)
        except Exception as exc:  # one VOD's failure must not abort the batch
            job_dir = None
            print(
                f"wall_retest_rerun: drop vod={vod_id} reason=run_failed "
                f"error_type={type(exc).__name__} error={_compact(str(exc))}",
                file=sys.stderr,
            )
        if job_dir is None:
            drops.append(vod_id)
            print(f"wall_retest_rerun: drop vod={vod_id} (no neu prediction dir)", file=sys.stderr)
        else:
            completed.append(vod_id)
    if drops:
        print(
            f"wall_retest_rerun: {len(drops)}/{len(subset)} VOD(s) dropped from the "
            f"neu re-run: {', '.join(drops)}",
            file=sys.stderr,
        )
    return RerunResult(predictions_dir=predictions_dir, completed=completed, drops=drops)


def _compact(message: str, limit: int = 200) -> str:
    flattened = " ".join(str(message or "").split())
    return flattened if len(flattened) <= limit else flattened[: limit - 3] + "..."


def build_rerun_env(
    predictions_dir: pathlib.Path, *, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    """The subprocess env for a neu re-run: the two cheap levers + output redirect.

    Overlays onto ``base_env`` — the glossary biased into Whisper
    (``WHISPER_INITIAL_PROMPT``), diarization on (``PIPELINE_ENABLE_DIARIZATION``),
    the neu output dir (``STORAGE_DIR``, so base stays untouched), and eval-recall
    on (``MOMENT_FINDER_EVAL_RECALL``). The last one is parity, not a lever: the
    neu run goes through the serving worker, where the raw-recall branch and the
    ``moments_raw.jsonl`` dump are off — but the compare harness replays that raw
    artifact and base was an eval-capture run with it on, so without it every neu
    dir is unreplayable and drops. ``WHISPER_MODEL`` is deliberately *not* set, so
    it inherits large-v3-turbo from the base env / .env — no model flip (AC3).
    """
    env = dict(base_env) if base_env is not None else {}
    env["WHISPER_INITIAL_PROMPT"] = JYNXZI_GLOSSARY
    env["PIPELINE_ENABLE_DIARIZATION"] = "true"
    env["MOMENT_FINDER_EVAL_RECALL"] = "true"
    env["STORAGE_DIR"] = str(predictions_dir)
    return env


def _mentions_glossary(moments: list[dict], terms: tuple[str, ...]) -> bool:
    """True if any gt_moment's desc names a glossary entity (case-insensitive)."""
    lowered_terms = [t.lower() for t in terms]
    for moment in moments:
        desc = str((moment or {}).get("desc") or "").lower()
        if any(term in desc for term in lowered_terms):
            return True
    return False


def select_subset(
    manifest: dict[str, list[dict]],
    predictions_dir: pathlib.Path,
    *,
    terms: tuple[str, ...] = GLOSSARY_TERMS,
) -> list[str]:
    """The banter/tic re-test subset: glossary-desc VODs ∩ replayable base dirs.

    A VOD qualifies when (a) any of its ``gt_moments`` carries a ``desc`` naming a
    glossary entity (the verbal-tic/banter signal) and (b) it has a replayable
    ``base`` job dir under ``predictions_dir`` (so base-vs-neu compares the same
    VODs). Deterministic: returned sorted by VOD id.
    """
    storage = pathlib.Path(predictions_dir)
    selected: list[str] = []
    for vod_id, moments in manifest.items():
        if not _mentions_glossary(list(moments or []), terms):
            continue
        if replay_harness._resolve_replay_job_dir(storage, vod_id) is None:
            continue
        selected.append(str(vod_id))
    return sorted(selected)


# ─────────────────────────────────────────────────────────────────────────────
# The real run_vod leaf — spawn the pipeline for one VOD with the levers on.
# ─────────────────────────────────────────────────────────────────────────────

def _is_replayable(job_dir: pathlib.Path) -> bool:
    """True once a neu job dir carries both debug artifacts the compare harness
    replays — ``segments.jsonl`` (re-derived chat=/vol=) and the raw moments dump.
    The single source of truth for "this VOD is done", used both to skip finished
    VODs on resume and to judge a fresh run's output."""
    debug = pathlib.Path(job_dir) / "debug"
    return (debug / replay_harness.SEGMENTS_ARTIFACT).exists() and (
        debug / replay_harness.RAW_MOMENTS_ARTIFACT
    ).exists()


def pipeline_run_vod(
    vod_id: str,
    predictions_dir: pathlib.Path,
    *,
    workflow_id: str,
    profile: str = "jynxzi",
    timeout_sec: float | None = None,
) -> pathlib.Path | None:
    """Re-run the real pipeline for one VOD into ``predictions_dir`` (the neu dir).

    Spawns ``pipeline_worker.py`` (the same per-job worker serving uses) under the
    re-run env — the glossary biased into Whisper, diarization on and
    ``STORAGE_DIR`` pointed at the neu dir — under a ``pipeline_debug_*`` job id so
    the worker writes the debug artifacts (``segments.jsonl`` re-derives chat=/vol=,
    ``llm_moment_finder_moments_raw.jsonl``) that Slice 1's compare harness replays.
    Returns the fresh job dir, or ``None`` when the run failed or wrote no replayable
    artifacts — the driver records that as a logged drop. The model stays
    large-v3-turbo (``build_rerun_env`` never sets ``WHISPER_MODEL``).

    This is the heavy, network/GPU leaf the real Slice-3 run (#93) drives; the
    driver's own tests inject a fake in its place.

    Resumable: a VOD whose neu dir is already replayable is returned as-is without
    re-spawning, so a re-launched driver skips the VODs it already finished. Robust:
    ``timeout_sec`` bounds the per-VOD wall-clock — a VOD that thrashes past it (the
    long-VOD swap-thrash failure mode) is dropped, not left to hang the whole run.
    """
    job_id = f"pipeline_debug_{workflow_id}_{vod_id}"
    job_dir = pathlib.Path(predictions_dir) / job_id
    if _is_replayable(job_dir):
        print(f"wall_retest_rerun: vod={vod_id} already replayable — skip (resume)", file=sys.stderr)
        return job_dir

    from pipeline_runner import build_subprocess_command, build_subprocess_env

    url = f"https://www.twitch.tv/videos/{vod_id}"
    env = build_rerun_env(predictions_dir, base_env=build_subprocess_env())
    command = build_subprocess_command(
        job_id=job_id, url=url, video_path=None, profile=profile,
        strategy={"platform": "twitch"},
    )
    try:
        completed = subprocess.run(
            command, cwd=str(BACKEND_ROOT), env=env,
            capture_output=True, text=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        print(
            f"wall_retest_rerun: vod={vod_id} TIMED OUT after {timeout_sec}s — drop "
            f"(likely memory-thrash on a long VOD)",
            file=sys.stderr,
        )
        return None
    replayable = _is_replayable(job_dir)
    if completed.returncode != 0 or not replayable:
        print(
            f"wall_retest_rerun: vod={vod_id} run returncode={completed.returncode} "
            f"replayable={replayable}",
            file=sys.stderr,
        )
        return None
    return job_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI — select the subset, re-run it into its own dir, report subset + drops.
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    from ml import eval_clip_finder

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True,
                        help="The base prediction dir (untouched) — selects the replayable subset")
    parser.add_argument("--predictions-dir", required=True,
                        help="The neu output dir for the re-run (base stays untouched)")
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--workflow-id", default="wallretest",
                        help="Job-id stem for the neu pipeline_debug_* dirs")
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--vod-timeout-sec", type=float, default=5400.0,
                        help="Per-VOD wall-clock cap (default 90 min); a VOD that "
                             "thrashes past it is dropped, not left to hang the run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only select and print the subset; do not re-run the pipeline")
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    subset = select_subset(manifest, pathlib.Path(args.base_dir))

    result: dict = {
        "subset": subset,
        "subset_count": len(subset),
        "base_dir": args.base_dir,
        "predictions_dir": args.predictions_dir,
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        def _run_vod(vod_id: str, predictions_dir: pathlib.Path) -> pathlib.Path | None:
            return pipeline_run_vod(
                vod_id, predictions_dir,
                workflow_id=args.workflow_id, profile=args.profile,
                timeout_sec=args.vod_timeout_sec,
            )

        rerun = rerun_subset(subset, pathlib.Path(args.predictions_dir), run_vod=_run_vod)
        result["completed"] = rerun.completed
        result["drops"] = rerun.drops

    if args.output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"WALL-Re-Test re-run — subset of {len(subset)} VOD(s): {', '.join(subset) or '(none)'}")
        if not args.dry_run:
            print(f"  completed: {', '.join(result['completed']) or '(none)'}")
            print(f"  drops:     {', '.join(result['drops']) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
