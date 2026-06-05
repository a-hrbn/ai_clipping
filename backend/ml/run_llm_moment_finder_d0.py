from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import vod_download_preflight


DEFAULT_PROFILE = "jynxzi"
DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_POSITIVES = DEFAULT_ARTIFACT_DIR / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_RECENT_HOLDOUT_POLICY = DEFAULT_ARTIFACT_DIR / "jynxzi_recent_vod_holdout_policy.jsonl"
DEFAULT_TRAINING_HOLDOUT_POLICY = DEFAULT_ARTIFACT_DIR / "jynxzi_training_holdout_policy_rows.jsonl"
DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"
DEFAULT_VIDEO_SOURCE_ROOT = vod_download_preflight.DEFAULT_VIDEO_SOURCE_ROOT
DEFAULT_OUTPUT_PREFIX = DEFAULT_ARTIFACT_DIR / "jynxzi_llm_moment_finder_d0"
DEFAULT_RUN_ROOT = DEFAULT_ARTIFACT_DIR / "jynxzi_llm_moment_finder_runs"
DEFAULT_OVERLAP_RATIO = 0.45
DEFAULT_TOP_N = 180
DEFAULT_MAX_MOMENTS = 180
DEFAULT_VOD_LIMIT = 1
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT_SEC = 900.0
DEFAULT_API_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SEC = 20.0
DEFAULT_MIN_DURATION_SEC = 8.0
DEFAULT_MAX_DURATION_SEC = 90.0
DEFAULT_PROVIDER = "gemini"
PROVIDERS = {"anthropic", "gemini"}
DEFAULT_GEMINI_MODEL = str(getattr(settings, "gemini_model", "") or "gemini-2.5-flash-lite")
DEFAULT_GEMINI_RESCORER_MODEL = "gemini-2.5-pro"
DEFAULT_ANTHROPIC_MODEL = str(getattr(settings, "anthropic_model_topk", "") or "claude-sonnet-4-6")
DEFAULT_MODEL = DEFAULT_GEMINI_MODEL
DEFAULT_GATE_MIN_RECALL = 0.15
DEFAULT_MATCH_MODE = "center_or_overlap"
MATCH_MODES = {"overlap", "center", "center_or_overlap"}
DEFAULT_RESCORER_OPERATING_POINTS = (12, 20, 80, 180)
DEFAULT_RESCORER_FEW_SHOTS = 12
DEFAULT_RESCORER_MAX_TOKENS = 12000
DEFAULT_RESCORER_BATCH_SIZE = 40
DEFAULT_RESCORER_MAX_DURATION_SEC = 120.0
_FILLER_TOKENS = {
    "you",
    "yeah",
    "uh",
    "um",
    "huh",
    "oh",
    "ah",
    "okay",
    "ok",
    "music",
    "[music]",
}
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class PromptTranscript:
    text: str
    input_segments: int
    kept_segments: int
    dropped_segments: int
    input_chars: int
    kept_chars: int
    dropped_reasons: dict[str, int]


@dataclass(frozen=True)
class PositiveWindow:
    positive_id: str
    vod_id: str
    start: float
    end: float
    title: str = ""
    source_url: str = ""
    views: float | None = None


@dataclass(frozen=True)
class LlmMoment:
    start: float
    end: float
    hook: float
    flow: float
    value: float
    trend: float
    virality: float
    rationale: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def main(argv: list[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    output_prefix_explicit = cli_option_present(argv_list, "--output-prefix")
    parser = argparse.ArgumentParser(
        description=(
            "D0 spike: send one or more non-holdout VOD full transcripts to an "
            "LLM moment finder and evaluate overlap against known cross-platform positives."
        )
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--vod-id", default="", help="Optional explicit train VOD ID. Holdout is rejected unless --allow-holdout is set.")
    parser.add_argument("--positive-rows", type=pathlib.Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--recent-holdout-policy", type=pathlib.Path, default=DEFAULT_RECENT_HOLDOUT_POLICY)
    parser.add_argument("--training-holdout-policy", type=pathlib.Path, default=DEFAULT_TRAINING_HOLDOUT_POLICY)
    parser.add_argument("--vod-cache-root", type=pathlib.Path, default=DEFAULT_VOD_CACHE_ROOT)
    parser.add_argument("--video-source-root", type=pathlib.Path, default=DEFAULT_VIDEO_SOURCE_ROOT)
    parser.add_argument("--output-prefix", type=pathlib.Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default="", help="Optional deterministic artifact run directory name.")
    parser.add_argument("--run-dir", type=pathlib.Path, default=None, help="Explicit artifact run directory to write/read.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default="", help="Override provider model. Defaults to Gemini Flash Lite for Gemini.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-moments", type=int, default=DEFAULT_MAX_MOMENTS)
    parser.add_argument(
        "--vod-limit",
        type=int,
        default=DEFAULT_VOD_LIMIT,
        help="Number of eligible train VODs to evaluate when --vod-id is not set. Use 3 for the D0.5 aggregate re-run.",
    )
    parser.add_argument("--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    parser.add_argument(
        "--match-mode",
        choices=sorted(MATCH_MODES),
        default=DEFAULT_MATCH_MODE,
        help="D0 eval match rule. center_or_overlap counts a moment containing the positive center as a hit.",
    )
    parser.add_argument("--min-duration-sec", type=float, default=DEFAULT_MIN_DURATION_SEC)
    parser.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--api-retries", type=int, default=DEFAULT_API_RETRIES)
    parser.add_argument("--retry-backoff-sec", type=float, default=DEFAULT_RETRY_BACKOFF_SEC)
    parser.add_argument("--download-cache-ttl-sec", type=int, default=-1)
    parser.add_argument(
        "--skip-vod-download-preflight",
        action="store_true",
        help="Disable the default full-VOD download preflight before real provider calls.",
    )
    parser.add_argument(
        "--gate-min-recall",
        type=float,
        default=DEFAULT_GATE_MIN_RECALL,
        help="Minimum Recall@Top-N for a D0 Go decision. Default requires at least a clear lift over 8%%.",
    )
    parser.add_argument("--no-filter-transcript-noise", action="store_true")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write prompt/metadata and exit without calling Anthropic.")
    parser.add_argument(
        "--reuse-responses",
        action="store_true",
        help="Read existing per-VOD *.response.txt artifacts instead of calling the provider.",
    )
    parser.add_argument("--mock-response", type=pathlib.Path, default=None, help="Read an Anthropic-style JSON response text from disk instead of calling the API.")
    parser.add_argument(
        "--rescore-existing",
        action="store_true",
        help="Re-score existing *.moments.json artifacts instead of generating new moments.",
    )
    parser.add_argument(
        "--rescore-model",
        default=DEFAULT_GEMINI_RESCORER_MODEL,
        help="Gemini model for --rescore-existing. Defaults to Gemini 2.5 Pro.",
    )
    parser.add_argument("--rescore-max-tokens", type=int, default=DEFAULT_RESCORER_MAX_TOKENS)
    parser.add_argument("--rescore-batch-size", type=int, default=DEFAULT_RESCORER_BATCH_SIZE)
    parser.add_argument("--rescore-max-duration-sec", type=float, default=DEFAULT_RESCORER_MAX_DURATION_SEC)
    parser.add_argument("--few-shot-limit", type=int, default=DEFAULT_RESCORER_FEW_SHOTS)
    parser.add_argument(
        "--reuse-rescore-responses",
        action="store_true",
        help="Read existing per-VOD *.rescore_response.txt artifacts instead of calling Gemini.",
    )
    parser.add_argument(
        "--mock-rescore-response",
        type=pathlib.Path,
        default=None,
        help="Read a rescore JSON response from disk instead of calling Gemini.",
    )
    args = parser.parse_args(argv)
    output_prefix, run_dir, run_id = resolve_cli_output_prefix(
        output_prefix=args.output_prefix,
        run_root=args.run_root,
        run_id=str(args.run_id or ""),
        run_dir=args.run_dir,
        output_prefix_explicit=output_prefix_explicit,
        rescore_existing=bool(args.rescore_existing),
    )

    if args.rescore_existing:
        report = run_d0_rescore_existing(
            profile=args.profile,
            vod_id=args.vod_id.strip(),
            positive_rows_path=args.positive_rows,
            recent_holdout_policy_path=args.recent_holdout_policy,
            training_holdout_policy_path=args.training_holdout_policy,
            vod_cache_root=args.vod_cache_root,
            video_source_root=args.video_source_root,
            output_prefix=output_prefix,
            provider="gemini",
            model=str(args.rescore_model or DEFAULT_GEMINI_RESCORER_MODEL),
            vod_limit=max(int(args.vod_limit or 0), 1),
            overlap_ratio=float(args.overlap_ratio),
            match_mode=str(args.match_mode),
            timeout_sec=float(args.timeout_sec),
            api_retries=max(int(args.api_retries or 0), 0),
            retry_backoff_sec=max(float(args.retry_backoff_sec or 0.0), 0.0),
            temperature=float(args.temperature),
            max_tokens=max(int(args.rescore_max_tokens or 0), 512),
            rescore_batch_size=max(int(args.rescore_batch_size or 0), 1),
            rescore_max_duration_sec=max(float(args.rescore_max_duration_sec or 0.0), 1.0),
            few_shot_limit=max(int(args.few_shot_limit or 0), 0),
            allow_holdout=bool(args.allow_holdout),
            dry_run=bool(args.dry_run),
            reuse_responses=bool(args.reuse_rescore_responses),
            mock_response_path=args.mock_rescore_response,
            download_vod_preflight=not bool(args.skip_vod_download_preflight),
            download_cache_ttl_sec=int(args.download_cache_ttl_sec),
        )
        report = attach_cli_run_manifest(
            report,
            run_dir=run_dir,
            run_id=run_id,
            output_prefix=output_prefix,
            mode="rescore_existing",
            argv=argv_list,
        )
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 1 if report_failed(report) else 0

    report = run_d0(
        profile=args.profile,
        vod_id=args.vod_id.strip(),
        positive_rows_path=args.positive_rows,
        recent_holdout_policy_path=args.recent_holdout_policy,
        training_holdout_policy_path=args.training_holdout_policy,
        vod_cache_root=args.vod_cache_root,
        video_source_root=args.video_source_root,
        output_prefix=output_prefix,
        provider=str(args.provider),
        model=resolve_provider_model(str(args.provider), str(args.model or "")),
        top_n=max(int(args.top_n or 0), 1),
        max_moments=max(int(args.max_moments or 0), 1),
        vod_limit=max(int(args.vod_limit or 0), 1),
        overlap_ratio=float(args.overlap_ratio),
        match_mode=str(args.match_mode),
        min_duration_sec=float(args.min_duration_sec),
        max_duration_sec=float(args.max_duration_sec),
        temperature=float(args.temperature),
        max_tokens=max(int(args.max_tokens or 0), 512),
        timeout_sec=float(args.timeout_sec),
        api_retries=max(int(args.api_retries or 0), 0),
        retry_backoff_sec=max(float(args.retry_backoff_sec or 0.0), 0.0),
        gate_min_recall=float(args.gate_min_recall),
        filter_transcript_noise=not bool(args.no_filter_transcript_noise),
        allow_holdout=bool(args.allow_holdout),
        dry_run=bool(args.dry_run),
        reuse_responses=bool(args.reuse_responses),
        mock_response_path=args.mock_response,
        download_vod_preflight=not bool(args.skip_vod_download_preflight),
        download_cache_ttl_sec=int(args.download_cache_ttl_sec),
    )
    report = attach_cli_run_manifest(
        report,
        run_dir=run_dir,
        run_id=run_id,
        output_prefix=output_prefix,
        mode="moment_finder",
        argv=argv_list,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 1 if report_failed(report) else 0


def run_d0(
    *,
    profile: str,
    vod_id: str,
    positive_rows_path: pathlib.Path,
    recent_holdout_policy_path: pathlib.Path,
    training_holdout_policy_path: pathlib.Path,
    vod_cache_root: pathlib.Path,
    video_source_root: pathlib.Path,
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    top_n: int,
    max_moments: int,
    vod_limit: int,
    overlap_ratio: float,
    match_mode: str,
    min_duration_sec: float,
    max_duration_sec: float,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    api_retries: int,
    retry_backoff_sec: float,
    gate_min_recall: float,
    filter_transcript_noise: bool,
    allow_holdout: bool,
    dry_run: bool,
    reuse_responses: bool = False,
    mock_response_path: pathlib.Path | None = None,
    download_vod_preflight: bool = True,
    download_cache_ttl_sec: int = -1,
) -> dict[str, Any]:
    positive_rows = read_jsonl(positive_rows_path)
    positives_by_vod = positive_windows_by_vod(positive_rows)
    holdout_vod_ids = load_holdout_vod_ids(
        recent_holdout_policy_path=recent_holdout_policy_path,
        training_holdout_policy_path=training_holdout_policy_path,
        positive_rows=positive_rows,
    )
    selected_vod_ids = select_spike_vods(
        positives_by_vod,
        vod_cache_root=vod_cache_root,
        requested_vod_id=vod_id,
        holdout_vod_ids=holdout_vod_ids,
        allow_holdout=allow_holdout,
        limit=vod_limit,
    )
    try:
        download_preflight_summary = maybe_preflight_vod_downloads(
            selected_vod_ids,
            video_source_root=video_source_root,
            enabled=download_vod_preflight,
            dry_run=dry_run,
            reuse_responses=reuse_responses,
            mock_response_path=mock_response_path,
            cache_ttl_sec=download_cache_ttl_sec,
        )
    except vod_download_preflight.VodDownloadPreflightError as exc:
        return write_d0_preflight_error_report(
            profile=profile,
            selected_vod_ids=selected_vod_ids,
            positives_by_vod=positives_by_vod,
            output_prefix=output_prefix,
            provider=provider,
            model=model,
            top_n=top_n,
            max_moments=max_moments,
            overlap_ratio=overlap_ratio,
            match_mode=match_mode,
            gate_min_recall=gate_min_recall,
            preflight_summary=exc.summary,
        )
    if len(selected_vod_ids) == 1:
        try:
            report = run_d0_for_vod(
                profile=profile,
                selected_vod_id=selected_vod_ids[0],
                positives_by_vod=positives_by_vod,
                holdout_vod_ids=holdout_vod_ids,
                vod_cache_root=vod_cache_root,
                output_prefix=output_prefix,
                provider=provider,
                model=model,
                top_n=top_n,
                max_moments=max_moments,
                overlap_ratio=overlap_ratio,
                match_mode=match_mode,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                api_retries=api_retries,
                retry_backoff_sec=retry_backoff_sec,
                gate_min_recall=gate_min_recall,
                filter_transcript_noise=filter_transcript_noise,
                allow_holdout=allow_holdout,
                dry_run=dry_run,
                reuse_responses=reuse_responses,
                mock_response_path=mock_response_path,
            )
            if download_preflight_summary is not None:
                report["download_preflight"] = download_preflight_summary
                write_json(pathlib.Path(str(report["eval_path"])), report)
            return report
        except Exception as exc:
            return write_d0_error_report(
                profile=profile,
                selected_vod_id=selected_vod_ids[0],
                positives_by_vod=positives_by_vod,
                output_prefix=output_prefix,
                provider=provider,
                model=model,
                top_n=top_n,
                max_moments=max_moments,
                overlap_ratio=overlap_ratio,
                match_mode=match_mode,
                gate_min_recall=gate_min_recall,
                error=exc,
            )

    reports: list[dict[str, Any]] = []
    for selected_vod_id in selected_vod_ids:
        vod_output_prefix = output_prefix.parent / f"{output_prefix.name}_{selected_vod_id}"
        try:
            reports.append(
                run_d0_for_vod(
                    profile=profile,
                    selected_vod_id=selected_vod_id,
                    positives_by_vod=positives_by_vod,
                    holdout_vod_ids=holdout_vod_ids,
                    vod_cache_root=vod_cache_root,
                    output_prefix=vod_output_prefix,
                    provider=provider,
                    model=model,
                    top_n=top_n,
                    max_moments=max_moments,
                    overlap_ratio=overlap_ratio,
                    match_mode=match_mode,
                    min_duration_sec=min_duration_sec,
                    max_duration_sec=max_duration_sec,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_sec=timeout_sec,
                    api_retries=api_retries,
                    retry_backoff_sec=retry_backoff_sec,
                    gate_min_recall=gate_min_recall,
                    filter_transcript_noise=filter_transcript_noise,
                    allow_holdout=allow_holdout,
                    dry_run=dry_run,
                    reuse_responses=reuse_responses,
                    mock_response_path=mock_response_path,
                )
            )
        except Exception as exc:
            reports.append(
                write_d0_error_report(
                    profile=profile,
                    selected_vod_id=selected_vod_id,
                    positives_by_vod=positives_by_vod,
                    output_prefix=vod_output_prefix,
                    provider=provider,
                    model=model,
                    top_n=top_n,
                    max_moments=max_moments,
                    overlap_ratio=overlap_ratio,
                    match_mode=match_mode,
                    gate_min_recall=gate_min_recall,
                    error=exc,
                )
            )
    batch_report = aggregate_d0_reports(
        reports,
        output_prefix=output_prefix,
        provider=provider,
        model=model,
        top_n=top_n,
        max_moments=max_moments,
        overlap_ratio=overlap_ratio,
        match_mode=match_mode,
        gate_min_recall=gate_min_recall,
        dry_run=dry_run,
    )
    if download_preflight_summary is not None:
        batch_report["download_preflight"] = download_preflight_summary
    write_json(output_prefix.with_suffix(".eval.json"), batch_report)
    return batch_report


def run_d0_rescore_existing(
    *,
    profile: str,
    vod_id: str,
    positive_rows_path: pathlib.Path,
    recent_holdout_policy_path: pathlib.Path,
    training_holdout_policy_path: pathlib.Path,
    vod_cache_root: pathlib.Path,
    video_source_root: pathlib.Path,
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    vod_limit: int,
    overlap_ratio: float,
    match_mode: str,
    timeout_sec: float,
    api_retries: int,
    retry_backoff_sec: float,
    temperature: float,
    max_tokens: int,
    rescore_batch_size: int,
    rescore_max_duration_sec: float,
    few_shot_limit: int,
    allow_holdout: bool,
    dry_run: bool,
    reuse_responses: bool = False,
    mock_response_path: pathlib.Path | None = None,
    download_vod_preflight: bool = True,
    download_cache_ttl_sec: int = -1,
) -> dict[str, Any]:
    positive_rows = read_jsonl(positive_rows_path)
    positives_by_vod = positive_windows_by_vod(positive_rows)
    holdout_vod_ids = load_holdout_vod_ids(
        recent_holdout_policy_path=recent_holdout_policy_path,
        training_holdout_policy_path=training_holdout_policy_path,
        positive_rows=positive_rows,
    )
    selected_vod_ids = select_spike_vods(
        positives_by_vod,
        vod_cache_root=vod_cache_root,
        requested_vod_id=vod_id,
        holdout_vod_ids=holdout_vod_ids,
        allow_holdout=allow_holdout,
        limit=vod_limit,
    )
    try:
        download_preflight_summary = maybe_preflight_vod_downloads(
            selected_vod_ids,
            video_source_root=video_source_root,
            enabled=download_vod_preflight,
            dry_run=dry_run,
            reuse_responses=reuse_responses,
            mock_response_path=mock_response_path,
            cache_ttl_sec=download_cache_ttl_sec,
        )
    except vod_download_preflight.VodDownloadPreflightError as exc:
        return write_rescore_preflight_error_report(
            selected_vod_ids=selected_vod_ids,
            output_prefix=output_prefix,
            provider=provider,
            model=model,
            overlap_ratio=overlap_ratio,
            match_mode=match_mode,
            preflight_summary=exc.summary,
        )
    reports: list[dict[str, Any]] = []
    for selected_vod_id in selected_vod_ids:
        artifact_prefix = rescore_artifact_prefix(output_prefix, selected_vod_id, len(selected_vod_ids))
        try:
            reports.append(
                run_d0_rescore_for_vod(
                    profile=profile,
                    selected_vod_id=selected_vod_id,
                    positives_by_vod=positives_by_vod,
                    holdout_vod_ids=holdout_vod_ids,
                    vod_cache_root=vod_cache_root,
                    output_prefix=artifact_prefix,
                    provider=provider,
                    model=model,
                    overlap_ratio=overlap_ratio,
                    match_mode=match_mode,
                    timeout_sec=timeout_sec,
                    api_retries=api_retries,
                    retry_backoff_sec=retry_backoff_sec,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    rescore_batch_size=rescore_batch_size,
                    rescore_max_duration_sec=rescore_max_duration_sec,
                    few_shot_limit=few_shot_limit,
                    allow_holdout=allow_holdout,
                    dry_run=dry_run,
                    reuse_responses=reuse_responses,
                    mock_response_path=mock_response_path,
                )
            )
        except Exception as exc:
            reports.append(
                write_rescore_error_report(
                    selected_vod_id=selected_vod_id,
                    positives_by_vod=positives_by_vod,
                    output_prefix=artifact_prefix,
                    provider=provider,
                    model=model,
                    overlap_ratio=overlap_ratio,
                    match_mode=match_mode,
                    error=exc,
                )
            )
    aggregate = aggregate_rescore_reports(
        reports,
        output_prefix=output_prefix,
        provider=provider,
        model=model,
        overlap_ratio=overlap_ratio,
        match_mode=match_mode,
        dry_run=dry_run,
    )
    if download_preflight_summary is not None:
        aggregate["download_preflight"] = download_preflight_summary
    write_json(output_prefix.with_suffix(".rescore_eval.json"), aggregate)
    return aggregate


def run_d0_rescore_for_vod(
    *,
    profile: str,
    selected_vod_id: str,
    positives_by_vod: Mapping[str, Sequence[PositiveWindow]],
    holdout_vod_ids: set[str],
    vod_cache_root: pathlib.Path,
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    overlap_ratio: float,
    match_mode: str,
    timeout_sec: float,
    api_retries: int,
    retry_backoff_sec: float,
    temperature: float,
    max_tokens: int,
    rescore_batch_size: int,
    rescore_max_duration_sec: float,
    few_shot_limit: int,
    allow_holdout: bool,
    dry_run: bool,
    reuse_responses: bool = False,
    mock_response_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    transcript_path = latest_transcript_path(vod_cache_root, selected_vod_id)
    if transcript_path is None:
        raise FileNotFoundError(f"No transcript cache found for VOD {selected_vod_id} under {vod_cache_root}")
    transcript_segments = load_transcript_segments(transcript_path)
    raw_moment_rows = load_moment_rows(output_prefix.with_suffix(".moments.json"))
    moment_rows, dropped_long_rows = filter_rescore_moment_rows(
        raw_moment_rows,
        max_duration_sec=rescore_max_duration_sec,
    )
    if not moment_rows:
        raise ValueError(
            f"No moments remain for re-scoring after duration filter <= {rescore_max_duration_sec:.3f}s."
        )
    positives = positives_by_vod.get(selected_vod_id, [])
    few_shots = build_rescore_few_shots(
        target_vod_id=selected_vod_id,
        positives_by_vod=positives_by_vod,
        holdout_vod_ids=holdout_vod_ids,
        vod_cache_root=vod_cache_root,
        limit=few_shot_limit,
        allow_holdout=allow_holdout,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = output_prefix.with_suffix(".rescore_prompt.txt")
    response_path = output_prefix.with_suffix(".rescore_response.txt")
    rescored_moments_path = output_prefix.with_suffix(".rescored_moments.json")
    eval_path = output_prefix.with_suffix(".rescore_eval.json")
    batches = chunk_sequence(moment_rows, max(int(rescore_batch_size or 0), 1))
    prompt_paths = rescore_batch_paths(output_prefix, ".rescore_prompt.txt", len(batches))
    response_paths = rescore_batch_paths(output_prefix, ".rescore_response.txt", len(batches))
    prompts = [
        build_rescore_prompt(
            profile=profile,
            vod_id=selected_vod_id,
            moment_rows=batch,
            transcript_segments=transcript_segments,
            few_shots=few_shots,
        )
        for batch in batches
    ]
    for path, prompt in zip(prompt_paths, prompts):
        path.write_text(prompt, encoding="utf-8")
    metadata: dict[str, Any] = {
        "event": "llm_moment_finder_d0_rescore_started",
        "vod_id": selected_vod_id,
        "provider": normalize_provider(provider),
        "model": model,
        "moment_count": len(moment_rows),
        "input_moment_count": len(raw_moment_rows),
        "dropped_long_moment_count": len(dropped_long_rows),
        "rescore_max_duration_sec": round(float(rescore_max_duration_sec), 3),
        "rescore_batch_size": max(int(rescore_batch_size or 0), 1),
        "rescore_batch_count": len(batches),
        "positive_windows": len(positives),
        "few_shot_count": len(few_shots),
        "few_shot_vod_ids": sorted({str(row.get("vod_id") or "") for row in few_shots if row.get("vod_id")}),
        "prompt_path": str(prompt_paths[0]),
        "response_path": str(response_paths[0]),
        "prompt_paths": [str(path) for path in prompt_paths],
        "response_paths": [str(path) for path in response_paths],
        "moments_path": str(output_prefix.with_suffix(".moments.json")),
        "rescored_moments_path": str(rescored_moments_path),
        "eval_path": str(eval_path),
        "overlap_ratio": bounded_ratio(overlap_ratio),
        "match_mode": normalize_match_mode(match_mode),
        "allow_holdout": bool(allow_holdout),
        "is_holdout_vod": selected_vod_id in holdout_vod_ids,
    }
    if dry_run:
        report = {**metadata, "event": "llm_moment_finder_d0_rescore_dry_run", "evaluation_by_top_n": {}}
        write_json(eval_path, report)
        return report

    started_at = time.monotonic()
    usage: dict[str, Any] = {"batches": []}
    scores: dict[str, dict[str, Any]] = {}
    for batch_index, (prompt, batch, batch_response_path) in enumerate(
        zip(prompts, batches, response_paths),
        start=1,
    ):
        if reuse_responses:
            if not batch_response_path.exists():
                raise FileNotFoundError(f"Cannot reuse missing rescore response artifact: {batch_response_path}")
            response_text = batch_response_path.read_text(encoding="utf-8")
            batch_usage: dict[str, Any] = {"reused_response": str(batch_response_path)}
        elif mock_response_path is not None:
            response_text = mock_response_path.read_text(encoding="utf-8")
            batch_usage = {"mock_response": str(mock_response_path)}
        else:
            response_text, batch_usage = call_moment_finder_provider(
                prompt,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                retries=api_retries,
                retry_backoff_sec=retry_backoff_sec,
            )
        batch_response_path.write_text(response_text, encoding="utf-8")
        batch_scores = parse_rescore_response(response_text)
        validate_rescore_scores_complete(batch, batch_scores, batch_index=batch_index)
        scores.update(batch_scores)
        usage["batches"].append(
            {
                "batch_index": batch_index,
                "candidate_count": len(batch),
                "response_path": str(batch_response_path),
                "usage": batch_usage,
            }
        )
    rescored_rows = apply_rescore_scores(moment_rows, scores, require_all_scores=True)
    rescored_moments = [moment_from_row(row) for row in rescored_rows]
    evaluation_by_top_n = evaluate_operating_points(
        rescored_moments,
        positives,
        overlap_ratio=overlap_ratio,
        match_mode=match_mode,
    )
    report = {
        **metadata,
        "event": "llm_moment_finder_d0_rescore_done",
        "elapsed_sec": round(time.monotonic() - started_at, 3),
        "usage": usage,
        "missing_score_count": sum(1 for row in rescored_rows if row.get("rescore_source") == "fallback_virality"),
        "evaluation_by_top_n": evaluation_by_top_n,
        "gate": {
            "top_n": 12,
            "gate_min_recall": 0.25,
            "recall_at_top_n": evaluation_by_top_n.get("12", {}).get("recall_at_top_n"),
            "passes": bool((evaluation_by_top_n.get("12", {}).get("recall_at_top_n") or 0.0) >= 0.25),
            "decision_rule": "D0.7 passes only if strong-model re-scoring lifts Recall@Top-12 to at least 25%.",
        },
    }
    write_json(rescored_moments_path, {"moments": rescored_rows})
    write_json(eval_path, report)
    return report


def run_d0_for_vod(
    *,
    profile: str,
    selected_vod_id: str,
    positives_by_vod: Mapping[str, Sequence[PositiveWindow]],
    holdout_vod_ids: set[str],
    vod_cache_root: pathlib.Path,
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    top_n: int,
    max_moments: int,
    overlap_ratio: float,
    match_mode: str,
    min_duration_sec: float,
    max_duration_sec: float,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    api_retries: int,
    retry_backoff_sec: float,
    gate_min_recall: float,
    filter_transcript_noise: bool,
    allow_holdout: bool,
    dry_run: bool,
    reuse_responses: bool = False,
    mock_response_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    transcript_path = latest_transcript_path(vod_cache_root, selected_vod_id)
    if transcript_path is None:
        raise SystemExit(f"No transcript cache found for VOD {selected_vod_id} under {vod_cache_root}")
    transcript_segments = load_transcript_segments(transcript_path)
    prompt_transcript = format_transcript_for_prompt(
        transcript_segments,
        filter_noise=filter_transcript_noise,
    )
    positives = positives_by_vod.get(selected_vod_id, [])
    prompt = build_moment_finder_prompt(
        profile=profile,
        vod_id=selected_vod_id,
        transcript_text=prompt_transcript.text,
        max_moments=max_moments,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = output_prefix.with_suffix(".prompt.txt")
    response_path = output_prefix.with_suffix(".response.txt")
    prompt_path.write_text(prompt, encoding="utf-8")

    metadata: dict[str, Any] = {
        "event": "llm_moment_finder_d0_started",
        "profile": profile,
        "vod_id": selected_vod_id,
        "provider": normalize_provider(provider),
        "model": model,
        "positive_windows": len(positives),
        "transcript_path": str(transcript_path),
        "transcript_segments": len(transcript_segments),
        "transcript_chars": prompt_transcript.kept_chars,
        "transcript_input_chars": prompt_transcript.input_chars,
        "transcript_prompt_segments": prompt_transcript.kept_segments,
        "transcript_dropped_segments": prompt_transcript.dropped_segments,
        "transcript_filter_enabled": bool(filter_transcript_noise),
        "transcript_filter_reasons": prompt_transcript.dropped_reasons,
        "prompt_path": str(prompt_path),
        "response_path": str(response_path),
        "top_n": top_n,
        "max_moments": max_moments,
        "overlap_ratio": bounded_ratio(overlap_ratio),
        "match_mode": normalize_match_mode(match_mode),
        "allow_holdout": bool(allow_holdout),
        "is_holdout_vod": selected_vod_id in holdout_vod_ids,
    }

    if dry_run:
        report = {
            **metadata,
            "event": "llm_moment_finder_d0_dry_run",
            "moments_path": str(output_prefix.with_suffix(".moments.json")),
            "eval_path": str(output_prefix.with_suffix(".eval.json")),
            "response_path": str(response_path),
            "moments": [],
            "evaluation": None,
        }
        write_json(output_prefix.with_suffix(".eval.json"), report)
        return report

    started_at = time.monotonic()
    if reuse_responses:
        if not response_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing response artifact: {response_path}")
        response_text = response_path.read_text(encoding="utf-8")
        usage = {"reused_response": str(response_path)}
    elif mock_response_path is not None:
        response_text = mock_response_path.read_text(encoding="utf-8")
        usage: dict[str, Any] = {"mock_response": str(mock_response_path)}
    else:
        response_text, usage = call_moment_finder_provider(
            prompt,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            retries=api_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
    response_path.write_text(response_text, encoding="utf-8")
    moments = parse_moments(response_text)
    ranked_moments = sorted(
        (moment for moment in moments if moment.duration > 0.0),
        key=lambda moment: moment.virality,
        reverse=True,
    )
    moments_payload = [moment_to_row(moment, rank=index) for index, moment in enumerate(ranked_moments, start=1)]
    evaluation = evaluate_moments(
        ranked_moments,
        positives,
        top_n=top_n,
        overlap_ratio=overlap_ratio,
        match_mode=match_mode,
    )
    recall_at_top_n = evaluation["recall_at_top_n"]
    gate_min_recall = bounded_ratio(gate_min_recall)
    report = {
        **metadata,
        "event": "llm_moment_finder_d0_done",
        "elapsed_sec": round(time.monotonic() - started_at, 3),
        "usage": usage,
        "moments_path": str(output_prefix.with_suffix(".moments.json")),
        "eval_path": str(output_prefix.with_suffix(".eval.json")),
        "response_path": str(response_path),
        "moments": moments_payload,
        "evaluation": evaluation,
        "gate": {
            "baseline_pre_judge_top80_ratio": 0.08,
            "gate_min_recall": gate_min_recall,
            "recall_at_top_n": recall_at_top_n,
            "passes": bool((recall_at_top_n or 0.0) >= gate_min_recall),
            "decision_rule": "D0 passes only if Recall@Top-N is clearly above the 8% Phase-B baseline.",
        },
    }
    write_json(output_prefix.with_suffix(".moments.json"), {"moments": moments_payload})
    write_json(output_prefix.with_suffix(".eval.json"), report)
    return report


def write_d0_error_report(
    *,
    profile: str,
    selected_vod_id: str,
    positives_by_vod: Mapping[str, Sequence[PositiveWindow]],
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    top_n: int,
    max_moments: int,
    overlap_ratio: float,
    match_mode: str,
    gate_min_recall: float,
    error: Exception,
) -> dict[str, Any]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "event": "llm_moment_finder_d0_failed",
        "profile": profile,
        "vod_id": selected_vod_id,
        "provider": normalize_provider(provider),
        "model": model,
        "positive_windows": len(positives_by_vod.get(selected_vod_id, [])),
        "top_n": top_n,
        "max_moments": max_moments,
        "overlap_ratio": bounded_ratio(overlap_ratio),
        "match_mode": normalize_match_mode(match_mode),
        "prompt_path": str(output_prefix.with_suffix(".prompt.txt")),
        "response_path": str(output_prefix.with_suffix(".response.txt")),
        "moments_path": str(output_prefix.with_suffix(".moments.json")),
        "eval_path": str(output_prefix.with_suffix(".eval.json")),
        "error_type": type(error).__name__,
        "error": str(error),
        "evaluation": None,
        "gate": {
            "baseline_pre_judge_top80_ratio": 0.08,
            "gate_min_recall": bounded_ratio(gate_min_recall),
            "recall_at_top_n": None,
            "passes": False,
            "blocked": True,
            "decision_rule": "D0 cannot pass while a selected VOD failed.",
        },
    }
    write_json(output_prefix.with_suffix(".eval.json"), report)
    return report


def write_d0_preflight_error_report(
    *,
    profile: str,
    selected_vod_ids: Sequence[str],
    positives_by_vod: Mapping[str, Sequence[PositiveWindow]],
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    top_n: int,
    max_moments: int,
    overlap_ratio: float,
    match_mode: str,
    gate_min_recall: float,
    preflight_summary: Mapping[str, Any],
) -> dict[str, Any]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    positive_windows = sum(len(positives_by_vod.get(vod_id, [])) for vod_id in selected_vod_ids)
    report = {
        "event": "llm_moment_finder_d0_preflight_failed",
        "profile": profile,
        "vod_ids": list(selected_vod_ids),
        "failed_vod_ids": list(preflight_summary.get("failed_vod_ids") or []),
        "provider": normalize_provider(provider),
        "model": model,
        "positive_windows": positive_windows,
        "top_n": top_n,
        "max_moments": max_moments,
        "overlap_ratio": bounded_ratio(overlap_ratio),
        "match_mode": normalize_match_mode(match_mode),
        "eval_path": str(output_prefix.with_suffix(".eval.json")),
        "error_type": "VodDownloadPreflightError",
        "error": "Full-VOD download preflight failed before provider call.",
        "download_preflight": preflight_summary,
        "evaluation": None,
        "gate": {
            "baseline_pre_judge_top80_ratio": 0.08,
            "gate_min_recall": bounded_ratio(gate_min_recall),
            "recall_at_top_n": None,
            "passes": False,
            "blocked": True,
            "decision_rule": "D0 cannot call the provider unless every selected VOD has a local full-video source.",
        },
    }
    write_json(output_prefix.with_suffix(".eval.json"), report)
    return report


def aggregate_d0_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    output_prefix: pathlib.Path,
    top_n: int,
    max_moments: int,
    overlap_ratio: float,
    match_mode: str,
    gate_min_recall: float,
    dry_run: bool,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    positive_windows = 0
    matched_positive_windows = 0
    failed_reports: list[Mapping[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for report in reports:
        evaluation = report.get("evaluation") if isinstance(report.get("evaluation"), Mapping) else {}
        if report_failed(report):
            failed_reports.append(report)
        positive_count = int(evaluation.get("positive_windows") or report.get("positive_windows") or 0)
        matched_count = int(evaluation.get("matched_positive_windows") or 0)
        positive_windows += positive_count
        matched_positive_windows += matched_count
        summaries.append(
            {
                "vod_id": report.get("vod_id"),
                "event": report.get("event"),
                "positive_windows": positive_count,
                "matched_positive_windows": matched_count,
                "recall_at_top_n": evaluation.get("recall_at_top_n"),
                "moment_count": evaluation.get("moment_count") if evaluation else len(report.get("moments") or []),
                "prompt_path": report.get("prompt_path"),
                "response_path": report.get("response_path"),
                "moments_path": report.get("moments_path"),
                "eval_path": report.get("eval_path"),
                "transcript_input_chars": report.get("transcript_input_chars"),
                "transcript_chars": report.get("transcript_chars"),
                "transcript_dropped_segments": report.get("transcript_dropped_segments"),
                "error_type": report.get("error_type"),
                "error": report.get("error"),
            }
        )
    recall_at_top_n = round(matched_positive_windows / positive_windows, 6) if positive_windows else None
    gate_min_recall = bounded_ratio(gate_min_recall)
    failed_vod_count = len(failed_reports)
    if dry_run:
        event = "llm_moment_finder_d0_batch_dry_run"
    elif failed_vod_count == len(reports):
        event = "llm_moment_finder_d0_batch_failed"
    elif failed_vod_count:
        event = "llm_moment_finder_d0_batch_partial_failed"
    else:
        event = "llm_moment_finder_d0_batch_done"
    return {
        "event": event,
        "vod_count": len(reports),
        "vod_ids": [str(report.get("vod_id") or "") for report in reports],
        "failed_vod_count": failed_vod_count,
        "failed_vod_ids": [str(report.get("vod_id") or "") for report in failed_reports],
        "provider": normalize_provider(provider),
        "model": model,
        "top_n": top_n,
        "max_moments": max_moments,
        "overlap_ratio": bounded_ratio(overlap_ratio),
        "match_mode": normalize_match_mode(match_mode),
        "positive_windows": positive_windows,
        "matched_positive_windows": matched_positive_windows,
        "recall_at_top_n": recall_at_top_n,
        "per_vod": summaries,
        "eval_path": str(output_prefix.with_suffix(".eval.json")),
        "gate": {
            "baseline_pre_judge_top80_ratio": 0.08,
            "gate_min_recall": gate_min_recall,
            "recall_at_top_n": recall_at_top_n,
            "passes": bool(failed_vod_count == 0 and (recall_at_top_n or 0.0) >= gate_min_recall),
            "blocked": bool(failed_vod_count > 0),
            "decision_rule": "D0.5 aggregate passes only if Recall@Top-N is clearly above the 8% Phase-B baseline.",
        },
    }


def aggregate_rescore_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    overlap_ratio: float,
    match_mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    failed_reports = [report for report in reports if report_failed(report)]
    operating_points: dict[str, dict[str, Any]] = {}
    for top_n in DEFAULT_RESCORER_OPERATING_POINTS:
        key = str(top_n)
        positive_windows = 0
        matched_positive_windows = 0
        for report in reports:
            evaluation_by_top_n = report.get("evaluation_by_top_n")
            if not isinstance(evaluation_by_top_n, Mapping):
                continue
            evaluation = evaluation_by_top_n.get(key)
            if not isinstance(evaluation, Mapping):
                continue
            positive_windows += int(evaluation.get("positive_windows") or 0)
            matched_positive_windows += int(evaluation.get("matched_positive_windows") or 0)
        operating_points[key] = {
            "top_n": top_n,
            "positive_windows": positive_windows,
            "matched_positive_windows": matched_positive_windows,
            "recall_at_top_n": round(matched_positive_windows / positive_windows, 6) if positive_windows else None,
        }
    if dry_run:
        event = "llm_moment_finder_d0_rescore_batch_dry_run"
    elif len(failed_reports) == len(reports):
        event = "llm_moment_finder_d0_rescore_batch_failed"
    elif failed_reports:
        event = "llm_moment_finder_d0_rescore_batch_partial_failed"
    else:
        event = "llm_moment_finder_d0_rescore_batch_done"
    top12_recall = operating_points.get("12", {}).get("recall_at_top_n")
    return {
        "event": event,
        "vod_count": len(reports),
        "vod_ids": [str(report.get("vod_id") or "") for report in reports],
        "failed_vod_count": len(failed_reports),
        "failed_vod_ids": [str(report.get("vod_id") or "") for report in failed_reports],
        "provider": normalize_provider(provider),
        "model": model,
        "overlap_ratio": bounded_ratio(overlap_ratio),
        "match_mode": normalize_match_mode(match_mode),
        "operating_points": operating_points,
        "per_vod": [
            {
                "vod_id": report.get("vod_id"),
                "event": report.get("event"),
                "moment_count": report.get("moment_count"),
                "input_moment_count": report.get("input_moment_count"),
                "dropped_long_moment_count": report.get("dropped_long_moment_count"),
                "rescore_max_duration_sec": report.get("rescore_max_duration_sec"),
                "rescore_batch_size": report.get("rescore_batch_size"),
                "rescore_batch_count": report.get("rescore_batch_count"),
                "positive_windows": report.get("positive_windows"),
                "missing_score_count": report.get("missing_score_count"),
                "few_shot_count": report.get("few_shot_count"),
                "few_shot_vod_ids": report.get("few_shot_vod_ids"),
                "prompt_path": report.get("prompt_path"),
                "response_path": report.get("response_path"),
                "prompt_paths": report.get("prompt_paths"),
                "response_paths": report.get("response_paths"),
                "rescored_moments_path": report.get("rescored_moments_path"),
                "eval_path": report.get("eval_path"),
                "evaluation_by_top_n": report.get("evaluation_by_top_n"),
                "error_type": report.get("error_type"),
                "error": report.get("error"),
            }
            for report in reports
        ],
        "eval_path": str(output_prefix.with_suffix(".rescore_eval.json")),
        "gate": {
            "top_n": 12,
            "gate_min_recall": 0.25,
            "recall_at_top_n": top12_recall,
            "passes": bool(len(failed_reports) == 0 and (top12_recall or 0.0) >= 0.25),
            "blocked": bool(failed_reports),
            "decision_rule": "D0.7 passes only if strong-model re-scoring lifts Recall@Top-12 to at least 25%.",
        },
    }


def write_rescore_error_report(
    *,
    selected_vod_id: str,
    positives_by_vod: Mapping[str, Sequence[PositiveWindow]],
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    overlap_ratio: float,
    match_mode: str,
    error: Exception,
) -> dict[str, Any]:
    report = {
        "event": "llm_moment_finder_d0_rescore_failed",
        "vod_id": selected_vod_id,
        "provider": normalize_provider(provider),
        "model": model,
        "positive_windows": len(positives_by_vod.get(selected_vod_id, [])),
        "overlap_ratio": bounded_ratio(overlap_ratio),
        "match_mode": normalize_match_mode(match_mode),
        "prompt_path": str(output_prefix.with_suffix(".rescore_prompt.txt")),
        "response_path": str(output_prefix.with_suffix(".rescore_response.txt")),
        "rescored_moments_path": str(output_prefix.with_suffix(".rescored_moments.json")),
        "eval_path": str(output_prefix.with_suffix(".rescore_eval.json")),
        "error_type": type(error).__name__,
        "error": str(error),
        "evaluation_by_top_n": {},
        "gate": {"blocked": True, "passes": False},
    }
    write_json(output_prefix.with_suffix(".rescore_eval.json"), report)
    return report


def write_rescore_preflight_error_report(
    *,
    selected_vod_ids: Sequence[str],
    output_prefix: pathlib.Path,
    provider: str,
    model: str,
    overlap_ratio: float,
    match_mode: str,
    preflight_summary: Mapping[str, Any],
) -> dict[str, Any]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "event": "llm_moment_finder_d0_rescore_preflight_failed",
        "vod_ids": list(selected_vod_ids),
        "failed_vod_ids": list(preflight_summary.get("failed_vod_ids") or []),
        "provider": normalize_provider(provider),
        "model": model,
        "overlap_ratio": bounded_ratio(overlap_ratio),
        "match_mode": normalize_match_mode(match_mode),
        "eval_path": str(output_prefix.with_suffix(".rescore_eval.json")),
        "error_type": "VodDownloadPreflightError",
        "error": "Full-VOD download preflight failed before provider call.",
        "download_preflight": preflight_summary,
        "evaluation_by_top_n": {},
        "gate": {
            "blocked": True,
            "passes": False,
            "decision_rule": "D0.7 cannot call the provider unless every selected VOD has a local full-video source.",
        },
    }
    write_json(output_prefix.with_suffix(".rescore_eval.json"), report)
    return report


def report_failed(report: Mapping[str, Any]) -> bool:
    event = str(report.get("event") or "")
    if event.endswith("_failed") or event.endswith("_partial_failed"):
        return True
    gate = report.get("gate")
    return bool(isinstance(gate, Mapping) and gate.get("blocked"))


def maybe_preflight_vod_downloads(
    vod_ids: Sequence[str],
    *,
    video_source_root: pathlib.Path,
    enabled: bool,
    dry_run: bool,
    reuse_responses: bool,
    mock_response_path: pathlib.Path | None,
    cache_ttl_sec: int,
) -> dict[str, Any] | None:
    if not enabled or dry_run or reuse_responses or mock_response_path is not None:
        return None
    return vod_download_preflight.ensure_vod_sources(
        vod_ids,
        video_source_root=video_source_root,
        download_missing=True,
        require_all=True,
        cache_ttl_sec=cache_ttl_sec,
    )


def cli_option_present(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def resolve_cli_output_prefix(
    *,
    output_prefix: pathlib.Path,
    run_root: pathlib.Path,
    run_id: str,
    run_dir: pathlib.Path | None,
    output_prefix_explicit: bool,
    rescore_existing: bool,
) -> tuple[pathlib.Path, pathlib.Path | None, str]:
    use_run_dir = run_dir is not None or bool(str(run_id or "").strip()) or (not output_prefix_explicit and not rescore_existing)
    if not use_run_dir:
        return output_prefix, None, ""
    resolved_run_dir = run_dir if run_dir is not None else run_root / resolve_artifact_run_id(run_id)
    resolved_run_id = safe_artifact_name(str(run_id or "")) or resolved_run_dir.name
    return resolved_run_dir / output_prefix.name, resolved_run_dir, resolved_run_id


def resolve_artifact_run_id(value: str = "") -> str:
    explicit = safe_artifact_name(value)
    if explicit:
        return explicit
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def safe_artifact_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned.strip("._-")


def attach_cli_run_manifest(
    report: dict[str, Any],
    *,
    run_dir: pathlib.Path | None,
    run_id: str,
    output_prefix: pathlib.Path,
    mode: str,
    argv: Sequence[str],
) -> dict[str, Any]:
    if run_dir is None:
        return report
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    report["run_id"] = run_id
    report["run_dir"] = str(run_dir)
    report["manifest_path"] = str(manifest_path)
    report["output_prefix"] = str(output_prefix)
    manifest = {
        "event": "llm_moment_finder_run_manifest",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": mode,
        "argv": list(argv),
        "output_prefix": str(output_prefix),
        "moment_prefix": str(output_prefix),
        "eval_path": report.get("eval_path"),
        "provider": report.get("provider"),
        "model": report.get("model"),
        "vod_ids": report.get("vod_ids") or ([report.get("vod_id")] if report.get("vod_id") else []),
        "report_event": report.get("event"),
        "report": report,
    }
    write_json(manifest_path, manifest)
    eval_path = report.get("eval_path")
    if eval_path:
        write_json(pathlib.Path(str(eval_path)), report)
    return report


def normalize_provider(provider: str) -> str:
    normalized = str(provider or DEFAULT_PROVIDER).strip().lower()
    if normalized not in PROVIDERS:
        return DEFAULT_PROVIDER
    return normalized


def resolve_provider_model(provider: str, model: str = "") -> str:
    explicit_model = str(model or "").strip()
    if explicit_model:
        return explicit_model
    if normalize_provider(provider) == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    return DEFAULT_GEMINI_MODEL


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def positive_windows_by_vod(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[PositiveWindow]]:
    grouped: dict[str, list[PositiveWindow]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        if int_or_none(row.get("label")) == 0:
            continue
        vod_id = str(row.get("vod_id") or "").strip()
        start = float_or_none(row.get("window_start"))
        end = float_or_none(row.get("window_end"))
        if not vod_id or start is None or end is None or end <= start:
            continue
        grouped[vod_id].append(
            PositiveWindow(
                positive_id=str(row.get("candidate_id") or row.get("source_url") or f"positive:{index}"),
                vod_id=vod_id,
                start=float(start),
                end=float(end),
                title=str(row.get("repost_title") or row.get("title") or ""),
                source_url=str(row.get("source_url") or row.get("repost_source_url") or row.get("vod_url") or ""),
                views=float_or_none(row.get("views") or row.get("repost_views")),
            )
        )
    return {vod: sorted(windows, key=lambda item: (item.start, item.end)) for vod, windows in grouped.items()}


def load_holdout_vod_ids(
    *,
    recent_holdout_policy_path: pathlib.Path,
    training_holdout_policy_path: pathlib.Path,
    positive_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    holdout_ids: set[str] = set()
    for row in read_jsonl(recent_holdout_policy_path):
        vod_id = str(row.get("vod_id") or "").strip()
        if vod_id:
            holdout_ids.add(vod_id)
    for row in [*read_jsonl(training_holdout_policy_path), *positive_rows]:
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id:
            continue
        split = str(row.get("gold_holdout_split") or "").strip().lower()
        role = str(row.get("dataset_role") or "").strip().lower()
        tier = str(row.get("supervision_tier") or "").strip().lower()
        source = str(row.get("dataset_source") or "").strip().lower()
        if split == "holdout" or role == "outcome_holdout" or tier == "outcome_holdout" or source.endswith("_holdout"):
            holdout_ids.add(vod_id)
    return holdout_ids


def select_spike_vod(
    positives_by_vod: Mapping[str, Sequence[PositiveWindow]],
    *,
    vod_cache_root: pathlib.Path,
    requested_vod_id: str,
    holdout_vod_ids: set[str],
    allow_holdout: bool,
) -> str:
    return select_spike_vods(
        positives_by_vod,
        vod_cache_root=vod_cache_root,
        requested_vod_id=requested_vod_id,
        holdout_vod_ids=holdout_vod_ids,
        allow_holdout=allow_holdout,
        limit=1,
    )[0]


def select_spike_vods(
    positives_by_vod: Mapping[str, Sequence[PositiveWindow]],
    *,
    vod_cache_root: pathlib.Path,
    requested_vod_id: str,
    holdout_vod_ids: set[str],
    allow_holdout: bool,
    limit: int,
) -> list[str]:
    requested_vod_ids = split_requested_vod_ids(requested_vod_id)
    if requested_vod_ids:
        selected: list[str] = []
        for vod_id in requested_vod_ids:
            if vod_id in holdout_vod_ids and not allow_holdout:
                raise SystemExit(f"Requested VOD {vod_id} is holdout; use --allow-holdout only for diagnostics.")
            if vod_id not in positives_by_vod:
                raise SystemExit(f"Requested VOD {vod_id} has no positive rows.")
            if latest_transcript_path(vod_cache_root, vod_id) is None:
                raise SystemExit(f"Requested VOD {vod_id} has no transcript cache.")
            if vod_id not in selected:
                selected.append(vod_id)
        return selected

    candidates: list[tuple[int, float, str]] = []
    for vod_id, positives in positives_by_vod.items():
        if vod_id in holdout_vod_ids and not allow_holdout:
            continue
        if latest_transcript_path(vod_cache_root, vod_id) is None:
            continue
        total_views = sum(float(window.views or 0.0) for window in positives)
        candidates.append((len(positives), total_views, vod_id))
    if not candidates:
        raise SystemExit("No eligible non-holdout VOD has both positives and a transcript cache.")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [vod_id for _, _, vod_id in candidates[: max(int(limit or 0), 1)]]


def split_requested_vod_ids(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def rescore_artifact_prefix(output_prefix: pathlib.Path, vod_id: str, selected_count: int) -> pathlib.Path:
    direct_moments = output_prefix.with_suffix(".moments.json")
    if selected_count == 1 and direct_moments.exists():
        return output_prefix
    return output_prefix.parent / f"{output_prefix.name}_{vod_id}"


def latest_transcript_path(vod_cache_root: pathlib.Path, vod_id: str) -> pathlib.Path | None:
    vod_dir = vod_cache_root / str(vod_id)
    paths = [path for path in vod_dir.glob("transcript_*.json") if path.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def load_transcript_segments(path: pathlib.Path) -> list[TranscriptSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = payload.get("transcript") if isinstance(payload, dict) else payload
    if not isinstance(raw_segments, list):
        raise ValueError(f"Transcript payload is not a list: {path}")
    segments: list[TranscriptSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            continue
        start = float_or_none(raw.get("start"))
        end = float_or_none(raw.get("end"))
        text = str(raw.get("text") or "").strip()
        if start is None or end is None or end <= start or not text:
            continue
        segments.append(TranscriptSegment(start=float(start), end=float(end), text=text))
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def format_transcript_for_prompt(
    segments: Sequence[TranscriptSegment],
    *,
    filter_noise: bool = True,
) -> PromptTranscript:
    lines: list[str] = []
    input_chars = 0
    kept_chars = 0
    dropped_reasons: Counter[str] = Counter()
    for segment in segments:
        text = " ".join(segment.text.split())
        input_chars += len(text)
        if not text:
            dropped_reasons["empty"] += 1
            continue
        drop_reason = transcript_noise_reason(segment, text) if filter_noise else None
        if drop_reason:
            dropped_reasons[drop_reason] += 1
            continue
        lines.append(f"[{format_time(segment.start)} - {format_time(segment.end)}] {text}")
        kept_chars += len(text)
    return PromptTranscript(
        text="\n".join(lines),
        input_segments=len(segments),
        kept_segments=len(lines),
        dropped_segments=max(len(segments) - len(lines), 0),
        input_chars=input_chars,
        kept_chars=kept_chars,
        dropped_reasons=dict(sorted(dropped_reasons.items())),
    )


def transcript_noise_reason(segment: TranscriptSegment, text: str) -> str | None:
    normalized = normalize_transcript_text(text)
    if not normalized:
        return "empty_after_normalize"
    tokens = normalized.split()
    duration = max(float(segment.end) - float(segment.start), 0.0)
    if len(tokens) == 1 and tokens[0] in _FILLER_TOKENS and duration <= 3.5:
        return "single_filler_token"
    if len(tokens) <= 3 and len(set(tokens)) == 1 and tokens[0] in _FILLER_TOKENS:
        return "repeated_filler_token"
    if len(tokens) <= 2 and normalized in {"thank you", "thanks", "bye", "subtitles"} and duration <= 3.5:
        return "short_boilerplate"
    return None


def normalize_transcript_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[\[\]\(\){}]", " ", lowered)
    lowered = re.sub(r"[^a-z0-9']+", " ", lowered)
    return " ".join(lowered.split())


def build_moment_finder_prompt(
    *,
    profile: str,
    vod_id: str,
    transcript_text: str,
    max_moments: int,
    min_duration_sec: float,
    max_duration_sec: float,
) -> str:
    return f"""You are an expert short-form clip producer for Twitch creator {profile}.

Read the entire VOD transcript and find the strongest viral clip moments. This is a moment-finder pass, not a judge for preselected clips. Look for the moments an OpusClip-style system would surface: immediate hook, clear reaction or conflict, payoff, standalone value, short-form trend fit, replay value.

Return raw JSON only, no markdown. Use this exact shape:
{{
  "moments": [
    {{
      "start": <seconds>,
      "end": <seconds>,
      "hook": <0.0-1.0>,
      "flow": <0.0-1.0>,
      "value": <0.0-1.0>,
      "trend": <0.0-1.0>,
      "virality": <0.0-1.0>,
      "rationale": "<one short sentence>"
    }}
  ]
}}

Rules:
- Return at most {max_moments} moments.
- Each moment should be between {min_duration_sec:.0f} and {max_duration_sec:.0f} seconds unless a tighter boundary is clearly better.
- Use exact transcript bracket timestamps. Start on the first transcript segment needed for the hook/setup; end on the final payoff/reaction segment.
- Do not use rounded minute boundaries like 3600, 4980, or 8100 unless the transcript itself has that exact boundary.
- Prefer many localized moments over a few broad summaries. A good 12-30s exact window is better than a generic 60s region.
- Optimize recall for viral Jynxzi-style clips: rage, clutch, absurd chat reactions, fanmail/unboxing reveals, strong arguments, quotable hot takes, streamer shock, and clear payoff beats.
- Penalize generic talking, long setup with no payoff, and moments that need too much missing context.

VOD ID: {vod_id}

Transcript:
{transcript_text}
"""


def build_rescore_few_shots(
    *,
    target_vod_id: str,
    positives_by_vod: Mapping[str, Sequence[PositiveWindow]],
    holdout_vod_ids: set[str],
    vod_cache_root: pathlib.Path,
    limit: int,
    allow_holdout: bool,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    examples: list[tuple[float, str, PositiveWindow]] = []
    for vod_id, windows in positives_by_vod.items():
        if vod_id == target_vod_id:
            continue
        if vod_id in holdout_vod_ids and not allow_holdout:
            continue
        if latest_transcript_path(vod_cache_root, vod_id) is None:
            continue
        for window in windows:
            examples.append((float(window.views or 0.0), vod_id, window))
    examples.sort(key=lambda item: (item[0], item[1], item[2].start), reverse=True)
    transcript_cache: dict[str, list[TranscriptSegment]] = {}
    rows: list[dict[str, Any]] = []
    for _, vod_id, window in examples[:limit]:
        if vod_id not in transcript_cache:
            transcript_path = latest_transcript_path(vod_cache_root, vod_id)
            if transcript_path is None:
                continue
            transcript_cache[vod_id] = load_transcript_segments(transcript_path)
        rows.append(
            {
                "vod_id": vod_id,
                "start": round(window.start, 3),
                "end": round(window.end, 3),
                "title": window.title,
                "views": window.views,
                "excerpt": transcript_excerpt(transcript_cache[vod_id], window.start, window.end, pad_sec=4.0, max_chars=600),
            }
        )
    return rows


def build_rescore_prompt(
    *,
    profile: str,
    vod_id: str,
    moment_rows: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[TranscriptSegment],
    few_shots: Sequence[Mapping[str, Any]],
) -> str:
    examples_text = "\n".join(
        (
            f"- VOD {row.get('vod_id')} [{row.get('start')} - {row.get('end')}], "
            f"views={row.get('views')}: title={json.dumps(str(row.get('title') or ''))}; "
            f"excerpt={json.dumps(str(row.get('excerpt') or ''))}"
        )
        for row in few_shots
    )
    if not examples_text:
        examples_text = "- No few-shot examples available; rank from the candidate evidence only."
    candidate_blocks: list[str] = []
    for row in moment_rows:
        start = float_or_none(row.get("start"))
        end = float_or_none(row.get("end"))
        if start is None or end is None:
            continue
        base_rank = int_or_none(row.get("rank")) or len(candidate_blocks) + 1
        excerpt = transcript_excerpt(transcript_segments, float(start), float(end), pad_sec=3.0, max_chars=700)
        candidate_blocks.append(
            "\n".join(
                [
                    f"id: m{base_rank}",
                    f"base_rank: {base_rank}",
                    f"window: {round(float(start), 3)} - {round(float(end), 3)}",
                    f"duration: {round(float(end) - float(start), 3)}",
                    f"flash_lite_scores: hook={row.get('hook')}, flow={row.get('flow')}, value={row.get('value')}, trend={row.get('trend')}, virality={row.get('virality')}",
                    f"flash_lite_rationale: {json.dumps(str(row.get('rationale') or ''))}",
                    f"transcript_excerpt: {json.dumps(excerpt)}",
                ]
            )
        )
    candidates_text = "\n\n".join(candidate_blocks)
    return f"""You are re-ranking candidate Twitch clip moments for creator {profile}.

This is not a moment generation task. You must only score the provided candidate IDs.
Your job is to separate true short-form Clipper picks from merely plausible moments.

Calibration examples are real reposted Train-VOD positives. They intentionally exclude target VOD {vod_id}.
Look for the shared pattern: immediate hook, quotable conflict/reaction, clear payoff, standalone context, and short-form replay value.

Few-shot Clipper picks:
{examples_text}

Return raw JSON only:
{{
  "scores": [
    {{"id": "m1", "score": <0.0-1.0>}}
  ]
}}

Rules:
- Score every provided candidate ID exactly once.
- Use score as the final rank signal; 1.0 means closest to a real Clipper pick.
- Keep output compact. Do not include rationale, explanation, markdown, or extra fields.
- Do not reward broad coverage alone. Prefer exact, self-contained, high-payoff moments.
- Penalize generic conversation, long setup, weak payoff, and clips whose hook depends on missing context.
- A candidate can score high even if Flash Lite ranked it low.

Target VOD: {vod_id}

Candidates:
{candidates_text}
"""


def transcript_excerpt(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
    *,
    pad_sec: float,
    max_chars: int,
) -> str:
    lower = float(start) - float(pad_sec)
    upper = float(end) + float(pad_sec)
    lines: list[str] = []
    for segment in segments:
        if segment.end < lower:
            continue
        if segment.start > upper:
            break
        lines.append(f"[{format_time(segment.start)} - {format_time(segment.end)}] {segment.text}")
    text = " ".join(" ".join(lines).split())
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 3, 0)].rstrip() + "..."


def call_moment_finder_provider(
    prompt: str,
    *,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    retries: int = DEFAULT_API_RETRIES,
    retry_backoff_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
) -> tuple[str, dict[str, Any]]:
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "gemini":
        return call_gemini_moment_finder(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
    return call_anthropic_moment_finder(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
    )


def call_gemini_moment_finder(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    retries: int = DEFAULT_API_RETRIES,
    retry_backoff_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
) -> tuple[str, dict[str, Any]]:
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is required for D0 unless --dry-run or --mock-response is used.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for Gemini's OpenAI-compatible endpoint.") from exc

    client = OpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_BASE_URL, timeout=timeout_sec if timeout_sec > 0 else None)
    attempts = max(int(retries or 0), 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You return strict JSON for offline clip-finding evaluation."},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                response_format={"type": "json_object"},
            )
            content = str(response.choices[0].message.content or "").strip()
            usage = openai_usage_to_dict(getattr(response, "usage", None))
            return content, usage
        except Exception as exc:
            if not is_retryable_llm_error(exc) or attempt >= attempts:
                raise RuntimeError(f"Gemini D0 call failed after {attempt} attempt(s): {type(exc).__name__}: {exc}") from exc
            delay = float(retry_backoff_sec) * attempt
            print_retry_event(
                provider="gemini",
                attempt=attempt,
                attempts=attempts,
                timeout_sec=timeout_sec,
                retry_after_sec=delay,
                error=exc,
            )
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("Gemini D0 call failed before receiving a response.")


def call_anthropic_moment_finder(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    retries: int = DEFAULT_API_RETRIES,
    retry_backoff_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
) -> tuple[str, dict[str, Any]]:
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is required for D0 unless --dry-run or --mock-response is used.")
    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system": [
            {
                "type": "text",
                "text": "You return strict JSON for offline clip-finding evaluation.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    request = urllib.request.Request(
        ANTHROPIC_MESSAGES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": str(settings.anthropic_api_key),
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    attempts = max(int(retries or 0), 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec if timeout_sec > 0 else None) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"Anthropic D0 call failed: status={exc.code} body={body[:500]}") from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"Anthropic D0 call failed after {attempts} attempts: {type(exc).__name__}: {exc}"
                ) from exc
            delay = float(retry_backoff_sec) * attempt
            print_retry_event(
                provider="anthropic",
                attempt=attempt,
                attempts=attempts,
                timeout_sec=timeout_sec,
                retry_after_sec=delay,
                error=exc,
            )
            if delay > 0:
                time.sleep(delay)
    else:
        raise RuntimeError("Anthropic D0 call failed before receiving a response.")
    text = anthropic_text_content(response_payload)
    usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
    return text, dict(usage or {})


def print_retry_event(
    *,
    provider: str,
    attempt: int,
    attempts: int,
    timeout_sec: float,
    retry_after_sec: float,
    error: Exception,
) -> None:
    print(
        json.dumps(
            {
                "event": "llm_moment_finder_d0_retry",
                "provider": normalize_provider(provider),
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "attempts": attempts,
                "timeout_sec": timeout_sec,
                "retry_after_sec": retry_after_sec,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout, urllib.error.URLError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ["timeout", "rate limit", "rate_limit", "temporarily unavailable", "503"])


def openai_usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())
    payload: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            payload[key] = value
    return payload


def anthropic_text_content(response_payload: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    for block in response_payload.get("content") or []:
        if isinstance(block, Mapping) and block.get("type") == "text":
            chunks.append(str(block.get("text") or ""))
    return "\n".join(chunks).strip()


def parse_moments(text: str) -> list[LlmMoment]:
    payload = parse_json_payload(text)
    raw_moments = payload.get("moments") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_moments, list):
        raise ValueError("LLM response does not contain a moments list.")
    moments: list[LlmMoment] = []
    for raw in raw_moments:
        if not isinstance(raw, Mapping):
            continue
        start = float_or_none(raw.get("start"))
        end = float_or_none(raw.get("end"))
        if start is None or end is None or end <= start:
            continue
        moments.append(
            LlmMoment(
                start=round(float(start), 3),
                end=round(float(end), 3),
                hook=score01(raw.get("hook")),
                flow=score01(raw.get("flow")),
                value=score01(raw.get("value")),
                trend=score01(raw.get("trend")),
                virality=score01(raw.get("virality")),
                rationale=str(raw.get("rationale") or "").strip(),
            )
        )
    return moments


def load_moment_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing moment artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("moments") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_rows, list):
        raise ValueError(f"Moment artifact does not contain a moments list: {path}")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row.setdefault("rank", index)
        if float_or_none(row.get("start")) is None or float_or_none(row.get("end")) is None:
            continue
        rows.append(row)
    return rows


def filter_rescore_moment_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_duration_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        duration = moment_row_duration(row)
        if duration is None or duration <= 0.0:
            dropped.append(row)
            continue
        if duration > float(max_duration_sec):
            dropped.append(row)
            continue
        row["duration"] = round(duration, 3)
        kept.append(row)
    return kept, dropped


def moment_row_duration(row: Mapping[str, Any]) -> float | None:
    duration = float_or_none(row.get("duration"))
    if duration is not None:
        return float(duration)
    start = float_or_none(row.get("start"))
    end = float_or_none(row.get("end"))
    if start is None or end is None:
        return None
    return max(float(end) - float(start), 0.0)


def chunk_sequence(rows: Sequence[Mapping[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(int(batch_size or 0), 1)
    return [list(dict(row) for row in rows[index : index + size]) for index in range(0, len(rows), size)]


def rescore_batch_paths(output_prefix: pathlib.Path, suffix: str, batch_count: int) -> list[pathlib.Path]:
    if batch_count <= 1:
        return [output_prefix.with_suffix(suffix)]
    return [
        output_prefix.parent / f"{output_prefix.name}{suffix.removesuffix('.txt')}_{index:03d}.txt"
        for index in range(1, batch_count + 1)
    ]


def parse_rescore_response(text: str) -> dict[str, dict[str, Any]]:
    try:
        payload = parse_json_payload(text)
    except json.JSONDecodeError:
        payload = extract_partial_array_payload(text, "scores")
        if payload is None:
            raise
    raw_scores = payload.get("scores") if isinstance(payload, Mapping) else payload
    if raw_scores is None and isinstance(payload, Mapping):
        raw_scores = payload.get("moments")
    if not isinstance(raw_scores, list):
        raise ValueError("Rescore response does not contain a scores list.")
    scores: dict[str, dict[str, Any]] = {}
    for raw in raw_scores:
        if not isinstance(raw, Mapping):
            continue
        score = score01(raw.get("score") if "score" in raw else raw.get("virality"))
        raw_id = str(raw.get("id") or "").strip()
        rank = int_or_none(raw.get("rank") or raw.get("base_rank"))
        key = raw_id or (f"m{rank}" if rank is not None else "")
        if not key:
            continue
        scores[key] = {
            "score": score,
            "rationale": str(raw.get("rationale") or raw.get("reason") or "").strip(),
        }
    return scores


def apply_rescore_scores(
    moment_rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    *,
    require_all_scores: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, raw in enumerate(moment_rows, start=1):
        row = dict(raw)
        base_rank = int_or_none(row.get("rank")) or index
        score_payload = scores.get(f"m{base_rank}") or scores.get(str(base_rank)) or {}
        score = float_or_none(score_payload.get("score")) if isinstance(score_payload, Mapping) else None
        if score is None:
            missing.append(f"m{base_rank}")
            if require_all_scores:
                continue
            score = score01(row.get("virality"))
            row["rescore_source"] = "fallback_virality"
            row["rescore_rationale"] = ""
        else:
            row["rescore_source"] = "llm"
            row["rescore_rationale"] = str(score_payload.get("rationale") or "")
        row["base_rank"] = base_rank
        row["base_virality"] = score01(row.get("virality"))
        row["rescore"] = score01(score)
        rows.append(row)
    if missing and require_all_scores:
        preview = ", ".join(missing[:20])
        suffix = "..." if len(missing) > 20 else ""
        raise ValueError(f"Missing {len(missing)} rescore score(s): {preview}{suffix}")
    rows.sort(key=lambda item: (float(item.get("rescore") or 0.0), -int(item.get("base_rank") or 0)), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rescore_rank"] = rank
    return rows


def validate_rescore_scores_complete(
    moment_rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    *,
    batch_index: int,
) -> None:
    missing: list[str] = []
    for index, row in enumerate(moment_rows, start=1):
        base_rank = int_or_none(row.get("rank")) or index
        if f"m{base_rank}" not in scores and str(base_rank) not in scores:
            missing.append(f"m{base_rank}")
    if not missing:
        return
    preview = ", ".join(missing[:20])
    suffix = "..." if len(missing) > 20 else ""
    raise ValueError(f"Batch {batch_index} missing {len(missing)} rescore score(s): {preview}{suffix}")


def moment_from_row(row: Mapping[str, Any]) -> LlmMoment:
    start = float_or_none(row.get("start"))
    end = float_or_none(row.get("end"))
    if start is None or end is None:
        raise ValueError("Moment row requires start/end.")
    return LlmMoment(
        start=round(float(start), 3),
        end=round(float(end), 3),
        hook=score01(row.get("hook")),
        flow=score01(row.get("flow")),
        value=score01(row.get("value")),
        trend=score01(row.get("trend")),
        virality=score01(row.get("rescore") if "rescore" in row else row.get("virality")),
        rationale=str(row.get("rescore_rationale") or row.get("rationale") or ""),
    )


def parse_json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    last_decode_error: json.JSONDecodeError | None = None

    def try_parse(candidate: str) -> Any | None:
        nonlocal last_decode_error
        try:
            return parse_json_with_repair(candidate)
        except json.JSONDecodeError as exc:
            last_decode_error = exc
            return None

    parsed = try_parse(stripped)
    if parsed is not None:
        return parsed
    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        parsed = try_parse(object_match.group(0))
        if parsed is not None:
            return parsed
    array_match = re.search(r"\[.*\]", stripped, flags=re.DOTALL)
    if array_match:
        parsed = try_parse(array_match.group(0))
        if parsed is not None:
            return parsed
    partial_payload = extract_partial_array_payload(stripped, "moments")
    if partial_payload is not None:
        return partial_payload
    if last_decode_error is not None:
        raise last_decode_error
    raise ValueError("No JSON object or array found in LLM response.")


def parse_json_with_repair(text: str) -> Any | None:
    candidates = [text, repair_common_json_breaks(text)]
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return None


def repair_common_json_breaks(text: str) -> str:
    repaired = text.strip()
    # Gemini occasionally emits adjacent object/array fields without commas in
    # long JSON arrays. Keep this intentionally narrow to avoid inventing data.
    repaired = re.sub(r"(\})(\s*)(\{)", r"\1,\2\3", repaired)
    repaired = re.sub(r"(\])(\s*)(\")", r"\1,\2\3", repaired)
    repaired = re.sub(r"(\})(\s*)(\")", r"\1,\2\3", repaired)
    repaired = re.sub(r"(\")(\s*\n\s*)(\")", r"\1,\2\3", repaired)
    return repaired


def extract_partial_array_payload(text: str, key: str) -> dict[str, list[dict[str, Any]]] | None:
    list_start = find_json_array_start(text, key)
    if list_start is None:
        return None
    raw_objects = balanced_json_objects_from_list(text, list_start)
    rows: list[dict[str, Any]] = []
    for raw_object in raw_objects:
        try:
            parsed = parse_json_with_repair(raw_object)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    if not rows:
        return None
    return {key: rows}


def find_json_array_start(text: str, key: str) -> int | None:
    escaped_key = re.escape(str(key))
    match = re.search(rf'"{escaped_key}"\s*:\s*\[', text)
    if not match:
        return None
    return match.end()


def balanced_json_objects_from_list(text: str, list_start: int) -> list[str]:
    objects: list[str] = []
    depth = 0
    start_index: int | None = None
    in_string = False
    escape_next = False
    for index in range(list_start, len(text)):
        char = text[index]
        if start_index is None:
            if char == "]":
                break
            if char == "{":
                start_index = index
                depth = 1
                in_string = False
                escape_next = False
            continue
        if in_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start_index : index + 1])
                start_index = None
    return objects


def evaluate_moments(
    moments: Sequence[LlmMoment],
    positives: Sequence[PositiveWindow],
    *,
    top_n: int,
    overlap_ratio: float,
    match_mode: str = DEFAULT_MATCH_MODE,
) -> dict[str, Any]:
    threshold = bounded_ratio(overlap_ratio)
    match_mode = normalize_match_mode(match_mode)
    top_moments = list(moments[: max(int(top_n), 1)])
    details: list[dict[str, Any]] = []
    matched = 0
    for positive in positives:
        best = best_moment_match(positive, top_moments)
        is_match = bool(best and moment_match_passes(best, threshold=threshold, match_mode=match_mode))
        if is_match:
            matched += 1
        details.append(
            {
                "positive_id": positive.positive_id,
                "start": round(positive.start, 3),
                "end": round(positive.end, 3),
                "title": positive.title,
                "source_url": positive.source_url,
                "matched": is_match,
                "best_moment": best,
            }
        )
    total = len(details)
    return {
        "positive_windows": total,
        "top_n": len(top_moments),
        "overlap_ratio": threshold,
        "match_mode": match_mode,
        "matched_positive_windows": matched,
        "recall_at_top_n": round(matched / total, 6) if total else None,
        "moment_count": len(moments),
        "details": details,
    }


def evaluate_operating_points(
    moments: Sequence[LlmMoment],
    positives: Sequence[PositiveWindow],
    *,
    overlap_ratio: float,
    match_mode: str,
    operating_points: Sequence[int] = DEFAULT_RESCORER_OPERATING_POINTS,
) -> dict[str, dict[str, Any]]:
    return {
        str(int(top_n)): evaluate_moments(
            moments,
            positives,
            top_n=int(top_n),
            overlap_ratio=overlap_ratio,
            match_mode=match_mode,
        )
        for top_n in operating_points
    }


def best_moment_match(positive: PositiveWindow, moments: Sequence[LlmMoment]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    for rank, moment in enumerate(moments, start=1):
        ratio = window_overlap_ratio(moment.start, moment.end, positive.start, positive.end)
        center_delta = positive_center_delta_sec(positive, moment)
        center_inside = center_delta <= 0.0
        score = max(ratio, 1.0 if center_inside else 0.0)
        if score < best_score or (score == best_score and center_delta >= float(best.get("positive_center_delta_sec", 1e9) if best else 1e9)):
            continue
        best_score = score
        best = {
            "rank": rank,
            "start": round(moment.start, 3),
            "end": round(moment.end, 3),
            "overlap_ratio": round(ratio, 6),
            "positive_center_inside": bool(center_inside),
            "positive_center_delta_sec": round(center_delta, 3),
            "virality": round(moment.virality, 6),
            "rationale": moment.rationale,
        }
    return best


def moment_match_passes(best: Mapping[str, Any], *, threshold: float, match_mode: str) -> bool:
    match_mode = normalize_match_mode(match_mode)
    overlap_passes = float(best.get("overlap_ratio") or 0.0) >= threshold
    center_passes = bool(best.get("positive_center_inside"))
    if match_mode == "overlap":
        return overlap_passes
    if match_mode == "center":
        return center_passes
    return overlap_passes or center_passes


def normalize_match_mode(match_mode: str) -> str:
    normalized = str(match_mode or DEFAULT_MATCH_MODE).strip().lower()
    if normalized not in MATCH_MODES:
        return DEFAULT_MATCH_MODE
    return normalized


def positive_center_delta_sec(positive: PositiveWindow, moment: LlmMoment) -> float:
    center = (float(positive.start) + float(positive.end)) / 2.0
    if float(moment.start) <= center <= float(moment.end):
        return 0.0
    return min(abs(center - float(moment.start)), abs(center - float(moment.end)))


def window_overlap_ratio(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    left_end = max(float(left_end), float(left_start) + 0.001)
    right_end = max(float(right_end), float(right_start) + 0.001)
    overlap = max(0.0, min(left_end, right_end) - max(float(left_start), float(right_start)))
    shortest = max(0.001, min(left_end - float(left_start), right_end - float(right_start)))
    return overlap / shortest


def moment_to_row(moment: LlmMoment, *, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "start": round(moment.start, 3),
        "end": round(moment.end, 3),
        "duration": round(moment.duration, 3),
        "hook": round(moment.hook, 6),
        "flow": round(moment.flow, 6),
        "value": round(moment.value, 6),
        "trend": round(moment.trend, 6),
        "virality": round(moment.virality, 6),
        "rationale": moment.rationale,
    }


def format_time(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def score01(value: Any) -> float:
    number = float_or_none(value)
    if number is None:
        return 0.0
    if number > 1.0 and number <= 10.0:
        number = number / 10.0
    return bounded_ratio(number)


def bounded_ratio(value: Any) -> float:
    number = float_or_none(value)
    if number is None:
        return 0.0
    return max(0.0, min(float(number), 1.0))


def float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def vod_positive_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        vod_id = str(row.get("vod_id") or "").strip()
        if vod_id and int_or_none(row.get("label")) != 0:
            counts[vod_id] += 1
    return dict(counts)


if __name__ == "__main__":
    raise SystemExit(main())
