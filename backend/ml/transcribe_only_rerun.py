"""Transcribe-only re-transcription driver + transcript-consistency report.

The WALL-Re-Test (PRD #95) leans on the *transcript* every downstream judge
reads, but the existing prediction dirs were transcribed across a drift of models
and settings. This driver re-transcribes the **anchor VODs that have a local
``video.mp4``** with *one* uniform model (``large-v3-turbo``, **plain** — no
glossary ``initial_prompt``, no diarization) into fresh, schema-uniform
``segments.jsonl`` files, **without** touching the candidate pool: the
``llm_moment_finder_moments_raw.jsonl`` dirs (and the buried-gold set derived from
them) are never opened for write. It is *not* a full pipeline re-run.

Each VOD's fresh transcript lands in its own dir
(``<predictions_dir>/transcribe_only_<vod>/debug/segments.jsonl``), so the WALL
status-quo ``pipeline_debug_*`` dirs stay byte-for-byte untouched and the run is
resumable (a VOD whose fresh segments already exist is skipped).

The 6 anchor VODs without a local video are surfaced as **excluded**, never
silently dropped.

On top of the fresh transcripts a **consistency report** flags drift fail-closed:
schema-uniformity, coverage-span (late-start / truncation vs the VOD length),
words/segment density outliers (a proxy for model drift, since no model field is
persisted) and replayability — per VOD pass/flag.

The heavy Whisper transcriber is injected, so the unit tests run with a net-free
fake (no GPU, no ffmpeg, no network).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import statistics
import sys
from typing import Callable

from services.transcriber import RichSegment, format_jsonl

from ml import vod_download_preflight

# The fixed artifact names and own-dir layout — the single place the fresh
# transcribe-only artifacts are spelled, so writer and report-reader never drift.
SEGMENTS_ARTIFACT = "segments.jsonl"
META_ARTIFACT = "transcribe_only_meta.json"

# The canonical record schema IS whatever the writer (``format_jsonl``) emits, so
# the report's notion of "uniform" can never drift from how the transcripts are
# written: a record passes iff its key set equals this set.
EXPECTED_SCHEMA_KEYS: frozenset[str] = frozenset(
    json.loads(format_jsonl([RichSegment(start=0.0, end=1.0, text="x")]).splitlines()[0]).keys()
)


@dataclasses.dataclass(frozen=True)
class TranscribeOutput:
    """What the injected transcribe leaf returns for one VOD.

    ``segments`` is the fresh plain transcript (no diarization, no glossary);
    ``duration_sec`` is the VOD's full length, carried so the consistency report
    can flag truncation against it without a second media pass.
    """
    segments: list[RichSegment]
    duration_sec: float


# vod_id, video_path -> the fresh plain transcript + VOD length.
TranscribeVodFn = Callable[[str, pathlib.Path], TranscribeOutput]


@dataclasses.dataclass(frozen=True)
class RetranscribeResult:
    predictions_dir: pathlib.Path
    completed: list[str]   # fresh segments written (or already fresh)
    excluded: list[str]    # anchor VODs with no local video — surfaced, not dropped
    drops: list[str]       # VODs whose transcribe failed — logged, never silent


def fresh_segments_path(predictions_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    """Where the fresh transcribe-only ``segments.jsonl`` for a VOD lives.

    Its own dir (``transcribe_only_<vod>/debug/``) — never a ``pipeline_debug_*``
    pool dir — so the WALL status-quo dirs stay byte-for-byte untouched.
    """
    return (
        pathlib.Path(predictions_dir)
        / f"transcribe_only_{vod_id}"
        / "debug"
        / SEGMENTS_ARTIFACT
    )


def fresh_meta_path(predictions_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    """The duration-meta sidecar next to a VOD's fresh ``segments.jsonl``.

    Persists the VOD length so the consistency report can flag truncation on a
    resumed / report-only run without a second media pass.
    """
    return fresh_segments_path(predictions_dir, vod_id).with_name(META_ARTIFACT)


def _is_fresh(path: pathlib.Path) -> bool:
    """True once a VOD's fresh ``segments.jsonl`` exists and is non-empty.

    The single source of truth for "this VOD is done", used to resume-skip a
    re-launched driver. An empty file (a half-written crash) is not fresh.
    """
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def retranscribe_vod(
    vod_id: str,
    video_path: pathlib.Path,
    predictions_dir: pathlib.Path,
    *,
    transcribe_vod: TranscribeVodFn,
) -> pathlib.Path | None:
    """Re-transcribe one VOD transcribe-only into its own fresh ``segments.jsonl``.

    Calls the injected ``transcribe_vod`` leaf (real = extract audio + plain
    Whisper; fake in tests) and writes the result via ``format_jsonl`` into the
    driver's own dir. The candidate pool is never opened — this is transcribe-only,
    not a pipeline re-run. Returns the fresh segments path.

    Resumable: a VOD whose fresh segments already exist is returned as-is without
    re-transcribing, so a re-launched driver skips the VODs it already finished.
    """
    path = fresh_segments_path(predictions_dir, vod_id)
    if _is_fresh(path):
        return path
    output = transcribe_vod(str(vod_id), pathlib.Path(video_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Meta first, then segments: segments.jsonl is the freshness marker (written
    # last), so a resumed run that sees it is guaranteed to find the meta too.
    fresh_meta_path(predictions_dir, vod_id).write_text(
        json.dumps({
            "vod_id": str(vod_id),
            "duration_sec": float(output.duration_sec),
            "segment_count": len(output.segments),
        }, ensure_ascii=True),
        encoding="utf-8",
    )
    path.write_text(format_jsonl(output.segments) + "\n", encoding="utf-8")
    return path


def _compact(message: str, limit: int = 200) -> str:
    flattened = " ".join(str(message or "").split())
    return flattened if len(flattened) <= limit else flattened[: limit - 3] + "..."


def load_report_input(
    predictions_dir: pathlib.Path, vod_id: str
) -> tuple[list[dict], float]:
    """Load a VOD's fresh ``(records, duration_sec)`` from disk for the report.

    Parses the fresh ``segments.jsonl`` into raw records (blank/unparseable lines
    skipped, like the replay loader) and reads the VOD length from the meta
    sidecar (``0.0`` when the sidecar is absent — a legacy run with no meta;
    truncation simply isn't checked for it).
    """
    seg_path = fresh_segments_path(predictions_dir, vod_id)
    records: list[dict] = []
    if seg_path.exists():
        for line in seg_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)

    duration_sec = 0.0
    meta_path = fresh_meta_path(predictions_dir, vod_id)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            duration_sec = float(meta.get("duration_sec") or 0.0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            duration_sec = 0.0
    return records, duration_sec


def load_report_inputs(
    predictions_dir: pathlib.Path, vod_ids: list[str]
) -> dict[str, tuple[list[dict], float]]:
    """Load ``(records, duration_sec)`` for every VOD, in the given order."""
    return {vod_id: load_report_input(predictions_dir, vod_id) for vod_id in vod_ids}


# ─────────────────────────────────────────────────────────────────────────────
# The real transcribe leaf — extract audio + plain Whisper + VOD length.
# This is the heavy, GPU/ffmpeg leaf the real Slice run drives; the driver's own
# tests inject a net-free fake in its place.
# ─────────────────────────────────────────────────────────────────────────────

def _default_extract_audio(
    video_path: pathlib.Path, job_id: str, output_path: pathlib.Path
) -> pathlib.Path:
    from services import audio_extractor

    return audio_extractor.extract(video_path, job_id, output_path=output_path)


def _default_probe_duration(media_path: pathlib.Path) -> float:
    """The media's full length in seconds via ffprobe (the truncation yardstick)."""
    import json as _json
    import subprocess

    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(media_path)],
        capture_output=True, text=True, check=True,
    )
    return float(_json.loads(completed.stdout)["format"]["duration"])


def _default_transcribe_vod(
    vod_id: str,
    video_path: pathlib.Path,
    *,
    extract_audio: Callable[..., pathlib.Path] = _default_extract_audio,
    transcribe: Callable[..., list[RichSegment]] | None = None,
    probe_duration: Callable[[pathlib.Path], float] = _default_probe_duration,
    work_dir: pathlib.Path | None = None,
) -> TranscribeOutput:
    """Extract audio + transcribe one VOD *plain*, returning segments + length.

    PLAIN is the whole point (AC1): ``enable_diarization=False`` and an explicit
    empty ``initial_prompt`` (overriding any ambient glossary setting), with
    ``cache_key=None`` so a previously glossary-biased chunk cache cannot
    contaminate the fresh uniform transcript. The in-process path
    (``isolate_chunks=False``) is used so the empty prompt is honoured here rather
    than re-read from the env in a chunk subprocess. The model is left untouched —
    the adapter never names one, so it inherits large-v3-turbo from the env/.env.
    """
    video_path = pathlib.Path(video_path)
    job_id = f"transcribe_only_{vod_id}"
    work = pathlib.Path(work_dir) if work_dir is not None else video_path.parent
    work.mkdir(parents=True, exist_ok=True)
    audio_path = work / f"{job_id}.wav"

    extract_audio(video_path, job_id, audio_path)

    if transcribe is None:
        from services import transcriber as _t

        transcribe = _t.transcribe_with_progress

    segments = transcribe(
        audio_path,
        enable_diarization=False,
        initial_prompt="",
        isolate_chunks=False,
        cache_key=None,
    )
    duration = float(probe_duration(video_path))
    return TranscribeOutput(segments=list(segments), duration_sec=duration)


def retranscribe_anchors(
    anchor_ids: list[str],
    predictions_dir: pathlib.Path,
    video_source_root: pathlib.Path,
    *,
    transcribe_vod: TranscribeVodFn,
) -> RetranscribeResult:
    """Re-transcribe every anchor VOD with a local video; surface the rest.

    Splits the anchors into with-/without-local-video, re-transcribes each
    with-video VOD transcribe-only into its own fresh dir, and returns the
    partition: ``completed`` (fresh segments written or already fresh),
    ``excluded`` (no local video — surfaced, never silently dropped) and ``drops``
    (a transcribe failure — logged, with the batch continuing). Each VOD is
    independent: one failure becomes a logged drop, the rest keep going.
    """
    predictions_dir = pathlib.Path(predictions_dir)
    with_video, without_video = split_by_local_video(anchor_ids, video_source_root)

    completed: list[str] = []
    drops: list[str] = []
    for vod_id, video_path in with_video:
        try:
            path = retranscribe_vod(
                vod_id, video_path, predictions_dir, transcribe_vod=transcribe_vod)
        except Exception as exc:  # one VOD's failure must not abort the batch
            path = None
            print(
                f"transcribe_only_rerun: drop vod={vod_id} reason=transcribe_failed "
                f"error_type={type(exc).__name__} error={_compact(str(exc))}",
                file=sys.stderr,
            )
        if path is None:
            drops.append(vod_id)
        else:
            completed.append(vod_id)

    if without_video:
        print(
            f"transcribe_only_rerun: {len(without_video)} anchor VOD(s) excluded — "
            f"no local video.mp4 (not re-transcribed, surfaced): {', '.join(without_video)}",
            file=sys.stderr,
        )
    if drops:
        print(
            f"transcribe_only_rerun: {len(drops)}/{len(with_video)} with-video VOD(s) "
            f"dropped from the re-transcribe: {', '.join(drops)}",
            file=sys.stderr,
        )
    return RetranscribeResult(
        predictions_dir=predictions_dir,
        completed=completed,
        excluded=without_video,
        drops=drops,
    )


def split_by_local_video(
    anchor_ids: list[str],
    video_source_root: pathlib.Path,
) -> tuple[list[tuple[str, pathlib.Path]], list[str]]:
    """Split anchors into those with a local ``video.mp4`` and those without.

    Resolution reuses ``vod_download_preflight.source_video_path`` (the existing
    ``vod_<id>.mp4`` / ``<id>.mp4`` lookup). Returns ``(with_video, without_video)``
    where ``with_video`` pairs each VOD id with its resolved path (input order
    preserved) and ``without_video`` is the surfaced exclusion set (AC5).
    """
    root = pathlib.Path(video_source_root)
    with_video: list[tuple[str, pathlib.Path]] = []
    without_video: list[str] = []
    for vod_id in anchor_ids:
        path = vod_download_preflight.source_video_path(root, str(vod_id))
        if path is None:
            without_video.append(str(vod_id))
        else:
            with_video.append((str(vod_id), path))
    return with_video, without_video


# ─────────────────────────────────────────────────────────────────────────────
# Consistency report — flag transcript drift fail-closed (schema / coverage /
# density / replayability), per VOD. A pure function over the fresh records, so
# it is fully testable without GPU/ffmpeg/network.
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class VodReport:
    vod_id: str
    passed: bool
    flags: list[str]


@dataclasses.dataclass(frozen=True)
class ConsistencyReport:
    clean: bool
    vod_reports: list[VodReport]
    excluded: list[str]


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _schema_mismatch(records: list[dict]) -> bool:
    """True if any record's key set differs from the canonical writer schema."""
    return any(frozenset(rec.keys()) != EXPECTED_SCHEMA_KEYS for rec in records)


def _span(records: list[dict]) -> tuple[float, float]:
    """The transcript's (first start, last end), or (0, 0) when empty."""
    if not records:
        return (0.0, 0.0)
    starts = [_safe_float(rec.get("start")) for rec in records]
    ends = [_safe_float(rec.get("end")) for rec in records]
    return (min(starts), max(ends))


def _coverage_flags(
    records: list[dict],
    duration_sec: float,
    *,
    late_start_max_sec: float,
    coverage_floor: float,
    coverage_ceiling: float,
) -> list[str]:
    """Flag a late start, a truncated transcript, or one running past the VOD end.

    ``over_run`` is the upper-bound counterpart to ``truncated``: a transcript whose
    last segment ends well *past* the VOD length means another recording's audio was
    spliced onto this timeline (e.g. stale leftover chunks of a longer VOD) — the
    contamination class this report must surface, not silently pass.
    """
    if not records:
        return []
    span_start, span_end = _span(records)
    flags: list[str] = []
    if span_start > late_start_max_sec:
        flags.append("late_start")
    if duration_sec > 0 and span_end < duration_sec * coverage_floor:
        flags.append("truncated")
    if duration_sec > 0 and span_end > duration_sec * coverage_ceiling:
        flags.append("over_run")
    return flags


def _unreplayable(records: list[dict]) -> bool:
    """True when the records yield no usable timeline for the replay loader.

    Empty transcripts and ones whose span has no positive length (nothing for
    ``snap_boundaries`` to align against) can't be replayed against the candidate
    pool — the whole point of keeping the pool was to replay the fresh transcript.
    """
    if not records:
        return True
    span_start, span_end = _span(records)
    return span_end <= span_start


def _vod_density(records: list[dict], duration_sec: float) -> tuple[float, float]:
    """A VOD's (mean words/sec, segments/hour) — the two density signals."""
    wps_values = [_safe_float(rec.get("words_per_second")) for rec in records]
    mean_wps = statistics.fmean(wps_values) if wps_values else 0.0
    seg_per_hour = (len(records) / (duration_sec / 3600.0)) if duration_sec > 0 else 0.0
    return mean_wps, seg_per_hour


def _density_outlier_ids(
    per_vod: dict[str, tuple[list[dict], float]],
    *,
    band_ratio: float,
    min_vods: int,
) -> set[str]:
    """VODs whose words/sec or segments/hour falls outside the cross-VOD band.

    The band is ``[median / ratio, median * ratio]`` per metric — interpretable as
    "more than ``ratio``× off the median density". Needs ``min_vods`` VODs for the
    band to be meaningful (with fewer, an outlier can't be defined). A metric whose
    median is non-positive is skipped (no band can be formed).
    """
    if len(per_vod) < min_vods:
        return set()
    wps_by: dict[str, float] = {}
    sph_by: dict[str, float] = {}
    for vod_id, (records, duration_sec) in per_vod.items():
        wps_by[vod_id], sph_by[vod_id] = _vod_density(records, float(duration_sec))

    outliers: set[str] = set()
    for metric in (wps_by, sph_by):
        median = statistics.median(metric.values())
        if median <= 0:
            continue
        lo, hi = median / band_ratio, median * band_ratio
        outliers.update(vod_id for vod_id, value in metric.items() if value < lo or value > hi)
    return outliers


def build_consistency_report(
    per_vod: dict[str, tuple[list[dict], float]],
    *,
    excluded: list[str] | None = None,
    late_start_max_sec: float | None = None,
    chunk_seconds: float = 180.0,
    late_start_max_chunks: float = 4.0,
    coverage_floor: float = 0.95,
    coverage_ceiling: float = 1.05,
    density_band_ratio: float = 2.5,
    min_vods_for_density: int = 3,
) -> ConsistencyReport:
    """Build the per-VOD consistency report over the fresh transcripts.

    ``per_vod`` maps each VOD id to ``(records, duration_sec)`` — the parsed
    ``segments.jsonl`` rows and the VOD's full length. Each VOD is checked for
    transcript drift and gets a pass/flag verdict. ``excluded`` (anchors with no
    local video) is carried through so it is surfaced, never silently dropped.
    """
    excluded = list(excluded or [])
    # Late-start threshold defaults to a few chunks, not a flat sub-chunk value: a
    # speechless intro/standby chunk lands the first segment at ~chunk_seconds, so a
    # threshold below the chunk granularity would flag every VOD with a brief intro.
    # An explicit late_start_max_sec still overrides this.
    if late_start_max_sec is None:
        late_start_max_sec = late_start_max_chunks * chunk_seconds
    density_outliers = _density_outlier_ids(
        per_vod, band_ratio=density_band_ratio, min_vods=min_vods_for_density)

    vod_reports: list[VodReport] = []
    for vod_id, (records, duration_sec) in per_vod.items():
        flags: list[str] = []
        if _schema_mismatch(records):
            flags.append("schema_mismatch")
        flags.extend(_coverage_flags(
            records, float(duration_sec),
            late_start_max_sec=late_start_max_sec, coverage_floor=coverage_floor,
            coverage_ceiling=coverage_ceiling,
        ))
        if vod_id in density_outliers:
            flags.append("density_outlier")
        if _unreplayable(records):
            flags.append("unreplayable")
        vod_reports.append(VodReport(vod_id=vod_id, passed=not flags, flags=flags))

    clean = not excluded and all(r.passed for r in vod_reports)
    return ConsistencyReport(clean=clean, vod_reports=vod_reports, excluded=excluded)


def report_to_dict(result: RetranscribeResult, report: ConsistencyReport) -> dict:
    """The run + report as a JSON-serialisable dict (the CLI's machine output)."""
    return {
        "predictions_dir": str(result.predictions_dir),
        "completed": result.completed,
        "excluded": report.excluded,
        "drops": result.drops,
        "clean": report.clean,
        "vod_reports": [
            {"vod_id": r.vod_id, "passed": r.passed, "flags": r.flags}
            for r in report.vod_reports
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI — re-transcribe the anchors transcribe-only, then report fail-closed.
# Exit code is non-zero when the report is not clean (a flagged VOD or a surfaced
# exclusion), so a CI/operator run can't let drift pass silently.
# ─────────────────────────────────────────────────────────────────────────────

def main(
    argv: list[str] | None = None,
    *,
    transcribe_vod: TranscribeVodFn = _default_transcribe_vod,
) -> int:
    import argparse

    from ml import eval_clip_finder

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", required=True,
                        help="The output dir for the fresh transcribe-only transcripts")
    parser.add_argument("--video-source-root",
                        default=str(vod_download_preflight.DEFAULT_VIDEO_SOURCE_ROOT),
                        help="Where the local vod_<id>.mp4 sources live")
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    predictions_dir = pathlib.Path(args.predictions_dir)
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))

    result = retranscribe_anchors(
        anchors, predictions_dir, pathlib.Path(args.video_source_root),
        transcribe_vod=transcribe_vod,
    )
    per_vod = load_report_inputs(predictions_dir, result.completed)
    report = build_consistency_report(per_vod, excluded=result.excluded)
    payload = report_to_dict(result, report)

    if args.output_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"transcribe-only re-transcribe — {len(result.completed)} completed, "
            f"{len(result.excluded)} excluded (no video), {len(result.drops)} dropped"
        )
        for r in report.vod_reports:
            verdict = "PASS" if r.passed else "FLAG " + ",".join(r.flags)
            print(f"  vod={r.vod_id} {verdict}")
        if result.excluded:
            print(f"  excluded (no local video): {', '.join(result.excluded)}")
        print(f"clean: {report.clean}")

    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
