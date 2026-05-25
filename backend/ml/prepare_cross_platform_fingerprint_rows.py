from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import random
import shutil
import subprocess
import sys
import time
import urllib.parse
from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import auto_train_short_form
from ml import build_twitch_training_dataset
from ml import collect_viral_clips
from ml import match_cross_platform_audio_fingerprints as matcher
from ml import match_cross_platform_transcripts as transcript_matcher
from config import settings
from services.yt_dlp_auth import apply_yt_dlp_auth_options


DEFAULT_VOD_MANIFEST = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_vods.jsonl"
DEFAULT_REPOST_MANIFEST = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_reposts.jsonl"
DEFAULT_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_SUMMARY = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_summary.json"
DEFAULT_DIAGNOSTIC_OUTPUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_cross_platform_fingerprint_diagnostics.jsonl"
DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"
DEFAULT_REPOST_SOURCE_MANIFEST = BACKEND_ROOT / "ml" / "cross_platform_sources.csv"
DEFAULT_REPOST_URL_MANIFEST = BACKEND_ROOT / "ml" / "viral_clips.txt"
DEFAULT_TWITCH_LABELS = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_twitch_clip_labels.jsonl"
DEFAULT_VOD_INVENTORY = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_twitch_vod_inventory.jsonl"
DEFAULT_TARGETED_VODS = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_rollout_readiness_collection_candidates.jsonl"
REPOST_PROFILE_FILTER_VERSION = 8
MATCH_INPUT_SIGNATURE_VERSION = 1
YOUTUBE_RATE_LIMIT_ERROR_MARKERS = (
    "this content isn't available, try again later",
    "this content is not available, try again later",
    "rate-limited by youtube",
    "rate limited by youtube",
    "exceeding the rate limit",
    "add a delay between video requests",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare cross-platform outcome rows by matching local Shorts/TikTok "
            "repost media against local VOD media."
        )
    )
    parser.add_argument("--vods", default=str(DEFAULT_VOD_MANIFEST), help="JSONL rows with vod_id, vod_url and video_path.")
    parser.add_argument(
        "--reposts",
        default=str(DEFAULT_REPOST_MANIFEST),
        help="JSONL rows with local video_path plus platform/views/source_url metadata.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--diagnostic-output", default=str(DEFAULT_DIAGNOSTIC_OUTPUT))
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument(
        "--verification-mode",
        choices=("transcript", "audio_fingerprint"),
        default="transcript",
        help="Primary repost-to-VOD verification signal. Transcript matching is the production default.",
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--vod-cache-root", default=str(DEFAULT_VOD_CACHE_ROOT))
    parser.add_argument("--repost-source-manifest", default=str(DEFAULT_REPOST_SOURCE_MANIFEST))
    parser.add_argument("--repost-url-manifest", default=str(DEFAULT_REPOST_URL_MANIFEST))
    parser.add_argument("--max-reposts", type=int, default=400)
    parser.add_argument("--min-reposts", type=int, default=0)
    parser.add_argument("--min-matched-rows", type=int, default=0)
    parser.add_argument(
        "--transcript-match-workers",
        type=int,
        default=1,
        help="Parallel worker processes for transcript repost-to-VOD matching.",
    )
    parser.add_argument(
        "--min-matched-vods",
        type=int,
        default=0,
        help="Require cross-platform matches to cover at least this many distinct VOD IDs.",
    )
    parser.add_argument(
        "--refresh-vods",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild the VOD manifest from the current scoped-window cache before matching.",
    )
    parser.add_argument("--refresh-reposts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--discover-reposts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-reposts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--full-vod-audio-fingerprints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build full-VOD audio fingerprints for selected VODs and match against those full timelines.",
    )
    parser.add_argument("--full-vod-audio-selected-vods", default=str(DEFAULT_TARGETED_VODS))
    parser.add_argument("--full-vod-audio-labels", default=str(DEFAULT_TWITCH_LABELS))
    parser.add_argument("--full-vod-audio-vod-inventory", default=str(DEFAULT_VOD_INVENTORY))
    parser.add_argument("--full-vod-audio-max-vods", type=int, default=0)
    parser.add_argument("--full-vod-audio-workers", type=int, default=1)
    parser.add_argument(
        "--filter-reposts-to-vod-date-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep reposts only when their publish date falls near the actually usable VOD fingerprint date window.",
    )
    parser.add_argument("--repost-vod-date-lookback-days", type=int, default=1)
    parser.add_argument("--repost-vod-date-lookahead-days", type=int, default=30)
    parser.add_argument(
        "--keep-full-vod-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep extracted full-VOD audio media after writing the fingerprint JSON.",
    )
    parser.add_argument(
        "--coarse-vod-tiling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build coarse VOD audio/video tiles for selected VODs before refreshing the VOD manifest.",
    )
    parser.add_argument("--coarse-vod-labels", default=str(DEFAULT_TWITCH_LABELS))
    parser.add_argument("--coarse-vod-selected-vods", default=str(DEFAULT_TARGETED_VODS))
    parser.add_argument("--coarse-vod-tile-sec", type=float, default=360.0)
    parser.add_argument("--coarse-vod-stride-sec", type=float, default=180.0)
    parser.add_argument("--coarse-vod-max-tiles-per-vod", type=int, default=0)
    parser.add_argument("--coarse-vod-max-vods", type=int, default=0)
    parser.add_argument("--coarse-vod-horizon-padding-sec", type=float, default=900.0)
    parser.add_argument(
        "--reuse-existing-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse an existing output JSONL when it already contains enough "
            "cross-platform rows for the current threshold. Disable together "
            "with --refresh-reposts to force rematching."
        ),
    )
    parser.add_argument("--auto-build-manifests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--overwrite-on-missing-inputs",
        action="store_true",
        help="Write an empty output file even when one or both manifests are missing.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return non-zero when inputs exist but fpcalc/matching is unavailable.",
    )
    parser.add_argument(
        "--fail-on-insufficient",
        action="store_true",
        help="Return non-zero when repost or matched-row coverage is below the configured minimum.",
    )
    args = parser.parse_args()

    summary = prepare_cross_platform_fingerprint_rows(
        vod_manifest=pathlib.Path(args.vods),
        repost_manifest=pathlib.Path(args.reposts),
        output_path=pathlib.Path(args.output),
        summary_path=pathlib.Path(args.summary),
        diagnostic_output_path=pathlib.Path(args.diagnostic_output),
        min_confidence=args.min_confidence,
        verification_mode=args.verification_mode,
        preserve_existing_on_missing_inputs=not args.overwrite_on_missing_inputs,
        auto_build_manifests=args.auto_build_manifests,
        vod_cache_root=pathlib.Path(args.vod_cache_root),
        repost_source_manifest=pathlib.Path(args.repost_source_manifest),
        repost_url_manifest=pathlib.Path(args.repost_url_manifest),
        profile=args.profile,
        max_reposts=args.max_reposts,
        min_reposts=args.min_reposts,
        min_matched_rows=args.min_matched_rows,
        transcript_match_workers=args.transcript_match_workers,
        min_matched_vods=args.min_matched_vods,
        refresh_vods=args.refresh_vods,
        refresh_reposts=args.refresh_reposts,
        discover_reposts=args.discover_reposts,
        download_reposts=args.download_reposts,
        reuse_existing_output=args.reuse_existing_output,
        full_vod_audio_fingerprints=args.full_vod_audio_fingerprints,
        full_vod_audio_labels=pathlib.Path(args.full_vod_audio_labels),
        full_vod_audio_vod_inventory=pathlib.Path(args.full_vod_audio_vod_inventory),
        full_vod_audio_selected_vods=pathlib.Path(args.full_vod_audio_selected_vods),
        full_vod_audio_max_vods=args.full_vod_audio_max_vods,
        full_vod_audio_workers=args.full_vod_audio_workers,
        filter_reposts_to_vod_date_window=args.filter_reposts_to_vod_date_window,
        repost_vod_date_lookback_days=args.repost_vod_date_lookback_days,
        repost_vod_date_lookahead_days=args.repost_vod_date_lookahead_days,
        keep_full_vod_audio=args.keep_full_vod_audio,
        coarse_vod_tiling=args.coarse_vod_tiling,
        coarse_vod_labels=pathlib.Path(args.coarse_vod_labels),
        coarse_vod_selected_vods=pathlib.Path(args.coarse_vod_selected_vods),
        coarse_vod_tile_sec=args.coarse_vod_tile_sec,
        coarse_vod_stride_sec=args.coarse_vod_stride_sec,
        coarse_vod_max_tiles_per_vod=args.coarse_vod_max_tiles_per_vod,
        coarse_vod_max_vods=args.coarse_vod_max_vods,
        coarse_vod_horizon_padding_sec=args.coarse_vod_horizon_padding_sec,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    if args.fail_on_blocked and str(summary.get("status") or "").startswith("blocked_"):
        return 1
    if args.fail_on_insufficient and str(summary.get("status") or "").startswith("insufficient_"):
        return 1
    return 0


def prepare_cross_platform_fingerprint_rows(
    *,
    vod_manifest: pathlib.Path,
    repost_manifest: pathlib.Path,
    output_path: pathlib.Path,
    summary_path: pathlib.Path,
    diagnostic_output_path: pathlib.Path | None = None,
    min_confidence: float = 0.50,
    verification_mode: str = "transcript",
    preserve_existing_on_missing_inputs: bool = True,
    auto_build_manifests: bool = True,
    vod_cache_root: pathlib.Path = DEFAULT_VOD_CACHE_ROOT,
    repost_source_manifest: pathlib.Path = DEFAULT_REPOST_SOURCE_MANIFEST,
    repost_url_manifest: pathlib.Path = DEFAULT_REPOST_URL_MANIFEST,
    profile: str = "jynxzi",
    max_reposts: int = 400,
    min_reposts: int = 0,
    min_matched_rows: int = 0,
    transcript_match_workers: int = 1,
    min_matched_vods: int = 0,
    refresh_vods: bool = False,
    refresh_reposts: bool = False,
    discover_reposts: bool = True,
    download_reposts: bool = True,
    reuse_existing_output: bool = True,
    full_vod_audio_fingerprints: bool = True,
    full_vod_audio_labels: pathlib.Path = DEFAULT_TWITCH_LABELS,
    full_vod_audio_vod_inventory: pathlib.Path = DEFAULT_VOD_INVENTORY,
    full_vod_audio_selected_vods: pathlib.Path = DEFAULT_TARGETED_VODS,
    full_vod_audio_max_vods: int = 0,
    full_vod_audio_workers: int = 1,
    filter_reposts_to_vod_date_window: bool = True,
    repost_vod_date_lookback_days: int = 1,
    repost_vod_date_lookahead_days: int = 30,
    keep_full_vod_audio: bool = False,
    coarse_vod_tiling: bool = False,
    coarse_vod_labels: pathlib.Path = DEFAULT_TWITCH_LABELS,
    coarse_vod_selected_vods: pathlib.Path = DEFAULT_TARGETED_VODS,
    coarse_vod_tile_sec: float = 360.0,
    coarse_vod_stride_sec: float = 180.0,
    coarse_vod_max_tiles_per_vod: int = 0,
    coarse_vod_max_vods: int = 0,
    coarse_vod_horizon_padding_sec: float = 900.0,
) -> dict[str, object]:
    vod_manifest = vod_manifest.expanduser()
    repost_manifest = repost_manifest.expanduser()
    output_path = output_path.expanduser()
    summary_path = summary_path.expanduser()
    if diagnostic_output_path is None:
        diagnostic_output_path = output_path.with_name(f"{output_path.stem}_diagnostics.jsonl")
    diagnostic_output_path = diagnostic_output_path.expanduser()
    vod_cache_root = vod_cache_root.expanduser()
    repost_source_manifest = repost_source_manifest.expanduser()
    repost_url_manifest = repost_url_manifest.expanduser()
    full_vod_audio_labels = full_vod_audio_labels.expanduser()
    full_vod_audio_vod_inventory = full_vod_audio_vod_inventory.expanduser()
    full_vod_audio_selected_vods = full_vod_audio_selected_vods.expanduser()
    coarse_vod_labels = coarse_vod_labels.expanduser()
    coarse_vod_selected_vods = coarse_vod_selected_vods.expanduser()
    verification_mode = _normalize_verification_mode(verification_mode)

    existing_rows = matcher._read_jsonl(output_path)
    started_at = time.monotonic()
    reusable_existing_rows = _reusable_existing_cross_platform_rows(
        existing_rows,
        min_confidence=min_confidence,
        verification_mode=verification_mode,
    )
    auto_manifest_summary: dict[str, object] = {}
    summary: dict[str, object] = {
        "status": "unknown",
        "vod_manifest_path": str(vod_manifest),
        "repost_manifest_path": str(repost_manifest),
        "output_path": str(output_path),
        "diagnostic_output_path": str(diagnostic_output_path),
        "min_confidence": float(min_confidence),
        "verification_mode": verification_mode,
        "min_reposts": int(min_reposts),
        "min_matched_rows": int(min_matched_rows),
        "min_matched_vods": int(min_matched_vods),
        "refresh_vods": bool(refresh_vods),
        "refresh_reposts": bool(refresh_reposts),
        "reuse_existing_output": bool(reuse_existing_output),
        "vod_manifest_exists": vod_manifest.exists(),
        "repost_manifest_exists": repost_manifest.exists(),
        "existing_output_rows": len(existing_rows),
        "existing_reusable_output_rows": len(reusable_existing_rows),
        "existing_output_dataset_sources": _dataset_sources(existing_rows),
        "preserved_existing_output": False,
        "auto_build_manifests": bool(auto_build_manifests),
        "auto_manifest_summary": auto_manifest_summary,
        "full_vod_audio_fingerprints": bool(full_vod_audio_fingerprints),
        "coarse_vod_tiling": bool(coarse_vod_tiling),
    }
    _log_event(
        "cross_platform_fingerprint_prepare_start",
        vod_manifest=str(vod_manifest),
        repost_manifest=str(repost_manifest),
        output_path=str(output_path),
        min_confidence=round(float(min_confidence), 6),
        verification_mode=verification_mode,
        min_reposts=int(min_reposts),
        min_matched_rows=int(min_matched_rows),
        min_matched_vods=int(min_matched_vods),
        refresh_vods=bool(refresh_vods),
        existing_output_rows=len(existing_rows),
        existing_reusable_output_rows=len(reusable_existing_rows),
        existing_reusable_output_vods=_matched_vod_count(reusable_existing_rows),
        reuse_existing_output=bool(reuse_existing_output),
        refresh_reposts=bool(refresh_reposts),
        auto_build_manifests=bool(auto_build_manifests),
    )
    if (
        reuse_existing_output
        and verification_mode == "audio_fingerprint"
        and not refresh_reposts
        and existing_rows
        and reusable_existing_rows
        and len(reusable_existing_rows) >= max(int(min_matched_rows), 0)
        and _matched_vod_count(reusable_existing_rows) >= max(int(min_matched_vods), 0)
    ):
        reusable_vods = _matched_vod_count(reusable_existing_rows)
        summary.update(
            {
                "status": "reused_existing_output",
                "matched_rows": len(reusable_existing_rows),
                "matched_vods": reusable_vods,
                "output_rows": len(existing_rows),
                "output_dataset_sources": _dataset_sources(existing_rows),
                "matched_rows_needed": 0,
                "matched_vods_needed": 0,
                "preserved_existing_output": True,
                "reuse_reason": "existing_output_meets_min_matched_rows_and_vods",
            }
        )
        _write_summary(summary_path, summary)
        _log_event(
            "cross_platform_fingerprint_prepare_done",
            status=str(summary["status"]),
            output_rows=int(summary["output_rows"]),
            reusable_output_rows=len(reusable_existing_rows),
            reusable_output_vods=reusable_vods,
            preserved_existing_output=True,
            output_path=str(output_path),
            summary_path=str(summary_path),
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
        return summary

    if auto_build_manifests:
        auto_manifest_summary = _build_missing_manifests(
            vod_manifest=vod_manifest,
            repost_manifest=repost_manifest,
            vod_cache_root=vod_cache_root,
            repost_source_manifest=repost_source_manifest,
            repost_url_manifest=repost_url_manifest,
            profile=profile,
            max_reposts=max_reposts,
            min_reposts=min_reposts,
            refresh_vods=refresh_vods,
            refresh_reposts=refresh_reposts,
            discover_reposts=discover_reposts,
            download_reposts=download_reposts,
            full_vod_audio_fingerprints=full_vod_audio_fingerprints,
            full_vod_audio_labels=full_vod_audio_labels,
            full_vod_audio_vod_inventory=full_vod_audio_vod_inventory,
            full_vod_audio_selected_vods=full_vod_audio_selected_vods,
            full_vod_audio_max_vods=full_vod_audio_max_vods,
            full_vod_audio_workers=full_vod_audio_workers,
            filter_reposts_to_vod_date_window=filter_reposts_to_vod_date_window,
            repost_vod_date_lookback_days=repost_vod_date_lookback_days,
            repost_vod_date_lookahead_days=repost_vod_date_lookahead_days,
            keep_full_vod_audio=keep_full_vod_audio,
            coarse_vod_tiling=coarse_vod_tiling,
            coarse_vod_labels=coarse_vod_labels,
            coarse_vod_selected_vods=coarse_vod_selected_vods,
            coarse_vod_tile_sec=coarse_vod_tile_sec,
            coarse_vod_stride_sec=coarse_vod_stride_sec,
            coarse_vod_max_tiles_per_vod=coarse_vod_max_tiles_per_vod,
            coarse_vod_max_vods=coarse_vod_max_vods,
            coarse_vod_horizon_padding_sec=coarse_vod_horizon_padding_sec,
        )
        summary["auto_manifest_summary"] = auto_manifest_summary
        summary["vod_manifest_exists"] = vod_manifest.exists()
        summary["repost_manifest_exists"] = repost_manifest.exists()
        profile_filter_rebuild = (
            str(auto_manifest_summary.get("repost_manifest_build_reason") or "") == "profile_filter_policy"
        )
        full_audio_block = (
            _full_vod_audio_block_reason(auto_manifest_summary)
            if verification_mode == "audio_fingerprint"
            else ""
        )
        if full_audio_block:
            summary.update(
                {
                    "status": "blocked_full_vod_audio_fingerprints",
                    "blocked_reason": full_audio_block,
                    "vod_rows": 0,
                    "repost_rows": 0,
                    "matched_rows": 0,
                    "matched_vods": 0,
                    "output_rows": len(existing_rows),
                }
            )
            if existing_rows and preserve_existing_on_missing_inputs:
                summary["preserved_existing_output"] = True
            else:
                _write_jsonl(output_path, [])
                _write_jsonl(diagnostic_output_path, [])
                summary["output_rows"] = 0
            _write_summary(summary_path, summary)
            _log_event(
                "cross_platform_fingerprint_prepare_done",
                status=str(summary["status"]),
                blocked_reason=full_audio_block,
                matched_rows=0,
                output_rows=int(summary["output_rows"]),
                preserved_existing_output=bool(summary["preserved_existing_output"]),
                elapsed_sec=round(time.monotonic() - started_at, 1),
            )
            return summary
    else:
        profile_filter_rebuild = False

    missing_inputs = [
        label
        for label, path in (("vod_manifest", vod_manifest), ("repost_manifest", repost_manifest))
        if not path.exists()
    ]
    if missing_inputs:
        summary.update(
            {
                "status": "missing_inputs",
                "missing_inputs": missing_inputs,
                "vod_rows": 0,
                "repost_rows": 0,
                "matched_rows": 0,
                "matched_vods": 0,
                "output_rows": len(existing_rows),
            }
        )
        if existing_rows and preserve_existing_on_missing_inputs:
            summary["preserved_existing_output"] = True
        else:
            _write_jsonl(output_path, [])
            _write_jsonl(diagnostic_output_path, [])
            summary["output_rows"] = 0
        _write_summary(summary_path, summary)
        _log_event(
            "cross_platform_fingerprint_prepare_done",
            status=str(summary["status"]),
            missing_inputs=missing_inputs,
            output_rows=int(summary["output_rows"]),
            preserved_existing_output=bool(summary["preserved_existing_output"]),
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
        return summary

    vod_rows = matcher._read_jsonl(vod_manifest)
    repost_rows = matcher._read_jsonl(repost_manifest)
    summary.update(
        {
            "vod_rows": len(vod_rows),
            "repost_rows": len(repost_rows),
            "vod_rows_with_video_path": _rows_with_path(vod_rows),
            "repost_rows_with_video_path": _rows_with_path(repost_rows),
        }
    )
    if min_reposts > 0 and len(repost_rows) < min_reposts:
        output_rows = len(existing_rows)
        if existing_rows and not profile_filter_rebuild:
            summary["preserved_existing_output"] = True
        else:
            _write_jsonl(output_path, [])
            _write_jsonl(diagnostic_output_path, [])
            output_rows = 0
        summary.update(
            {
                "status": "insufficient_repost_rows",
                "matched_rows": 0,
                "matched_vods": 0,
                "output_rows": output_rows,
                "repost_rows_needed": max(int(min_reposts) - len(repost_rows), 0),
                "matched_vods_needed": max(int(min_matched_vods), 0),
            }
        )
        _write_summary(summary_path, summary)
        _log_event(
            "cross_platform_fingerprint_prepare_done",
            status=str(summary["status"]),
            repost_rows=len(repost_rows),
            repost_rows_needed=int(summary["repost_rows_needed"]),
            matched_rows=0,
            output_rows=output_rows,
            preserved_existing_output=bool(summary["preserved_existing_output"]),
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
        return summary

    if not vod_rows or not repost_rows:
        if existing_rows and preserve_existing_on_missing_inputs and not profile_filter_rebuild:
            summary["preserved_existing_output"] = True
            output_rows = len(existing_rows)
        else:
            _write_jsonl(output_path, [])
            _write_jsonl(diagnostic_output_path, [])
            output_rows = 0
        summary.update({"status": "empty_inputs", "matched_rows": 0, "matched_vods": 0, "output_rows": output_rows})
        _write_summary(summary_path, summary)
        _log_event(
            "cross_platform_fingerprint_prepare_done",
            status=str(summary["status"]),
            vod_rows=len(vod_rows),
            repost_rows=len(repost_rows),
            matched_rows=0,
            output_rows=output_rows,
            preserved_existing_output=bool(summary["preserved_existing_output"]),
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
        return summary

    match_input_signature = _cross_platform_match_input_signature(
        vod_rows=vod_rows,
        repost_rows=repost_rows,
        verification_mode=verification_mode,
        min_confidence=min_confidence,
        vod_cache_root=vod_cache_root,
    )
    summary["match_input_signature"] = match_input_signature
    previous_summary = _read_json(summary_path)
    if _can_reuse_previous_match_result(
        previous_summary=previous_summary,
        match_input_signature=match_input_signature,
        output_path=output_path,
        diagnostic_output_path=diagnostic_output_path,
    ):
        diagnostic_rows = matcher._read_jsonl(diagnostic_output_path)
        reused_status = str(previous_summary.get("status") or "reused_existing_match")
        summary.update(
            {
                "status": reused_status,
                "matched_rows": int(_safe_float(previous_summary.get("matched_rows"))),
                "matched_vods": int(_safe_float(previous_summary.get("matched_vods"))),
                "output_rows": len(existing_rows),
                "output_dataset_sources": _dataset_sources(existing_rows),
                "match_diagnostics": previous_summary.get("match_diagnostics") or {},
                "diagnostic_rows": len(diagnostic_rows),
                "matched_rows_needed": int(_safe_float(previous_summary.get("matched_rows_needed"))),
                "matched_vods_needed": int(_safe_float(previous_summary.get("matched_vods_needed"))),
                "insufficient_reasons": list(previous_summary.get("insufficient_reasons") or []),
                "preserved_existing_output": True,
                "reused_match_result": True,
                "reuse_reason": "match_input_signature_unchanged",
            }
        )
        _write_summary(summary_path, summary)
        _log_event(
            "cross_platform_match_reused",
            status=reused_status,
            verification_mode=verification_mode,
            output_rows=len(existing_rows),
            diagnostic_rows=len(diagnostic_rows),
            match_input_signature=match_input_signature,
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
        _log_event(
            "cross_platform_fingerprint_prepare_done",
            status=str(summary["status"]),
            matched_rows=int(summary["matched_rows"]),
            matched_vods=int(summary["matched_vods"]),
            output_rows=int(summary["output_rows"]),
            diagnostic_rows=int(summary["diagnostic_rows"]),
            matched_rows_needed=int(summary["matched_rows_needed"]),
            matched_vods_needed=int(summary["matched_vods_needed"]),
            preserved_existing_output=True,
            output_path=str(output_path),
            diagnostic_output_path=str(diagnostic_output_path),
            summary_path=str(summary_path),
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
        return summary

    if verification_mode == "transcript":
        transcript_summary = build_full_vod_transcript_caches(
            vod_rows=vod_rows,
            vod_cache_root=vod_cache_root,
            keep_audio=keep_full_vod_audio,
        )
        summary["full_vod_transcript_summary"] = transcript_summary

    try:
        if verification_mode == "transcript":
            _log_event(
                "cross_platform_transcript_matching_call",
                vod_rows=len(vod_rows),
                repost_rows=len(repost_rows),
                vod_rows_with_video_path=int(summary["vod_rows_with_video_path"]),
                repost_rows_with_video_path=int(summary["repost_rows_with_video_path"]),
                min_confidence=round(float(min_confidence), 6),
                transcript_match_workers=max(1, int(transcript_match_workers or 1)),
            )
            matched_rows, match_diagnostics = transcript_matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_similarity=min_confidence,
                vod_cache_root=vod_cache_root,
                log_progress=True,
                transcript_match_workers=transcript_match_workers,
            )
        else:
            _log_event(
                "cross_platform_fingerprint_matching_call",
                vod_rows=len(vod_rows),
                repost_rows=len(repost_rows),
                vod_rows_with_video_path=int(summary["vod_rows_with_video_path"]),
                repost_rows_with_video_path=int(summary["repost_rows_with_video_path"]),
                min_confidence=round(float(min_confidence), 6),
            )
            matched_rows, match_diagnostics = matcher.match_reposts_to_vods_with_diagnostics(
                vod_rows,
                repost_rows,
                min_confidence=min_confidence,
                log_progress=True,
            )
    except RuntimeError as exc:
        status = "blocked_missing_fpcalc" if "fpcalc not found" in str(exc) else "blocked_matching_error"
        summary.update(
            {
                "status": status,
                "error": str(exc),
                "matched_rows": 0,
                "output_rows": len(existing_rows),
            }
        )
        if existing_rows:
            summary["preserved_existing_output"] = True
        else:
            _write_jsonl(output_path, [])
            _write_jsonl(diagnostic_output_path, [])
            summary["output_rows"] = 0
        _write_summary(summary_path, summary)
        _log_event(
            "cross_platform_fingerprint_prepare_done",
            status=str(summary["status"]),
            error=str(summary["error"]),
            matched_rows=0,
            output_rows=int(summary["output_rows"]),
            preserved_existing_output=bool(summary["preserved_existing_output"]),
            elapsed_sec=round(time.monotonic() - started_at, 1),
        )
        return summary

    matched_rows = _dedupe_rows(matched_rows)
    matched_vods = _matched_vod_count(matched_rows)
    diagnostic_rows = list(match_diagnostics.get("best_match_rows") or [])
    summary_diagnostics = dict(match_diagnostics)
    summary_diagnostics.pop("best_match_rows", None)
    _write_jsonl(diagnostic_output_path, diagnostic_rows)
    status = "matched" if matched_rows else "no_matches_below_threshold"
    insufficient_reasons: list[str] = []
    if min_matched_rows > 0 and len(matched_rows) < min_matched_rows:
        insufficient_reasons.append("matched_rows")
    if min_matched_vods > 0 and matched_vods < min_matched_vods:
        insufficient_reasons.append("matched_vods")
    if insufficient_reasons:
        status = "insufficient_" + "_and_".join(insufficient_reasons)
    can_preserve_existing_output = bool(
        existing_rows
        and preserve_existing_on_missing_inputs
        and not profile_filter_rebuild
        and _existing_output_compatible_with_verification_mode(
            existing_rows,
            reusable_existing_rows,
            verification_mode=verification_mode,
        )
    )
    if (
        (not matched_rows or insufficient_reasons)
        and can_preserve_existing_output
    ):
        summary["preserved_existing_output"] = True
        output_rows = len(existing_rows)
    else:
        _write_jsonl(output_path, matched_rows)
        output_rows = len(matched_rows)
    summary.update(
        {
            "status": status,
            "matched_rows": len(matched_rows),
            "matched_vods": matched_vods,
            "output_rows": output_rows,
            "output_dataset_sources": _dataset_sources(matched_rows),
            "match_diagnostics": summary_diagnostics,
            "diagnostic_rows": len(diagnostic_rows),
            "matched_rows_needed": max(int(min_matched_rows) - len(matched_rows), 0),
            "matched_vods_needed": max(int(min_matched_vods) - matched_vods, 0),
            "insufficient_reasons": insufficient_reasons,
        }
    )
    _write_summary(summary_path, summary)
    _log_event(
        "cross_platform_fingerprint_prepare_done",
        status=str(summary["status"]),
        matched_rows=len(matched_rows),
        matched_vods=matched_vods,
        output_rows=output_rows,
        diagnostic_rows=len(diagnostic_rows),
        matched_rows_needed=int(summary["matched_rows_needed"]),
        matched_vods_needed=int(summary["matched_vods_needed"]),
        preserved_existing_output=bool(summary["preserved_existing_output"]),
        output_path=str(output_path),
        diagnostic_output_path=str(diagnostic_output_path),
        summary_path=str(summary_path),
        elapsed_sec=round(time.monotonic() - started_at, 1),
    )
    return summary


def _build_missing_manifests(
    *,
    vod_manifest: pathlib.Path,
    repost_manifest: pathlib.Path,
    vod_cache_root: pathlib.Path,
    repost_source_manifest: pathlib.Path,
    repost_url_manifest: pathlib.Path,
    profile: str,
    max_reposts: int,
    min_reposts: int,
    refresh_vods: bool,
    refresh_reposts: bool,
    discover_reposts: bool,
    download_reposts: bool,
    full_vod_audio_fingerprints: bool,
    full_vod_audio_labels: pathlib.Path,
    full_vod_audio_vod_inventory: pathlib.Path,
    full_vod_audio_selected_vods: pathlib.Path,
    full_vod_audio_max_vods: int,
    full_vod_audio_workers: int,
    filter_reposts_to_vod_date_window: bool,
    repost_vod_date_lookback_days: int,
    repost_vod_date_lookahead_days: int,
    keep_full_vod_audio: bool,
    coarse_vod_tiling: bool,
    coarse_vod_labels: pathlib.Path,
    coarse_vod_selected_vods: pathlib.Path,
    coarse_vod_tile_sec: float,
    coarse_vod_stride_sec: float,
    coarse_vod_max_tiles_per_vod: int,
    coarse_vod_max_vods: int,
    coarse_vod_horizon_padding_sec: float,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "vod_cache_root": str(vod_cache_root),
        "repost_source_manifest": str(repost_source_manifest),
        "repost_url_manifest": str(repost_url_manifest),
        "profile": profile,
        "coarse_vod_tiling": bool(coarse_vod_tiling),
        "discover_reposts": bool(discover_reposts),
        "download_reposts": bool(download_reposts),
        "max_reposts": int(max_reposts),
        "min_reposts": int(min_reposts),
        "refresh_vods": bool(refresh_vods),
        "refresh_reposts": bool(refresh_reposts),
        "full_vod_audio_fingerprints": bool(full_vod_audio_fingerprints),
    }
    if full_vod_audio_fingerprints:
        summary["full_vod_audio_summary"] = build_full_vod_audio_fingerprints(
            labels_path=full_vod_audio_labels,
            vod_inventory_path=full_vod_audio_vod_inventory,
            selected_vods_path=full_vod_audio_selected_vods,
            vod_cache_root=vod_cache_root,
            profile=profile,
            max_vods=full_vod_audio_max_vods,
            workers=full_vod_audio_workers,
            keep_audio=keep_full_vod_audio,
        )
    if coarse_vod_tiling:
        summary["coarse_vod_tile_summary"] = build_coarse_vod_tiles(
            labels_path=coarse_vod_labels,
            selected_vods_path=coarse_vod_selected_vods,
            vod_cache_root=vod_cache_root,
            profile=profile,
            tile_sec=coarse_vod_tile_sec,
            stride_sec=coarse_vod_stride_sec,
            max_tiles_per_vod=coarse_vod_max_tiles_per_vod,
            max_vods=coarse_vod_max_vods,
            horizon_padding_sec=coarse_vod_horizon_padding_sec,
        )
    existing_vod_rows = matcher._read_jsonl(vod_manifest)
    repaired_vod_rows, vod_path_summary = _repair_manifest_media_paths(existing_vod_rows)
    summary["vod_manifest_existing_rows"] = len(existing_vod_rows)
    summary["vod_manifest_path_summary"] = vod_path_summary
    audio_block_reason = _full_vod_audio_block_reason(summary)
    full_audio_manifest_ready = bool(full_vod_audio_fingerprints) and not audio_block_reason
    if full_audio_manifest_ready or refresh_vods or not repaired_vod_rows or int(vod_path_summary["missing_media_rows"]) > 0:
        vod_rows = build_vod_manifest_from_cache(
            vod_cache_root,
            include_scoped_windows=not bool(full_vod_audio_fingerprints),
        )
        if full_audio_manifest_ready:
            reason = "full_vod_audio_fingerprints"
        elif refresh_vods:
            reason = "refresh_requested"
        elif not repaired_vod_rows:
            reason = "missing_or_empty_or_unusable"
        else:
            reason = "stale_or_missing_media_paths"
        if full_vod_audio_fingerprints and audio_block_reason and not vod_rows:
            summary["vod_manifest_built"] = False
            summary["vod_manifest_build_reason"] = "blocked_" + audio_block_reason
            summary["vod_manifest_rows"] = len(repaired_vod_rows)
            summary["vod_manifest_preserved_existing_rows"] = len(repaired_vod_rows)
        else:
            _write_jsonl(vod_manifest, vod_rows)
            summary["vod_manifest_built"] = True
            summary["vod_manifest_build_reason"] = reason
            summary["vod_manifest_rows"] = len(vod_rows)
    else:
        if int(vod_path_summary["remapped_media_rows"]) > 0:
            _write_jsonl(vod_manifest, repaired_vod_rows)
        summary["vod_manifest_built"] = False
        summary["vod_manifest_rows"] = len(repaired_vod_rows)
        vod_rows = repaired_vod_rows

    existing_repost_rows = matcher._read_jsonl(repost_manifest)
    repaired_repost_rows, repost_path_summary = _repair_manifest_media_paths(existing_repost_rows)
    summary["repost_manifest_existing_rows"] = len(existing_repost_rows)
    summary["repost_manifest_path_summary"] = repost_path_summary
    has_stale_profile_filter = _repost_manifest_needs_profile_filter_rebuild(
        repaired_repost_rows,
        source_manifest=repost_source_manifest,
        discover_reposts=discover_reposts,
    )
    has_curated_source_freshness_refresh = _repost_manifest_needs_curated_source_freshness_refresh(
        source_manifest=repost_source_manifest,
        profile=profile,
        discover_reposts=discover_reposts,
    )
    has_too_few_reposts = int(min_reposts) > 0 and len(repaired_repost_rows) < int(min_reposts)
    has_less_than_requested_reposts = int(max_reposts) > 0 and len(repaired_repost_rows) < int(max_reposts)
    summary["repost_manifest_profile_filter_rebuild_required"] = bool(has_stale_profile_filter)
    summary["repost_manifest_curated_source_freshness_required"] = bool(has_curated_source_freshness_refresh)
    summary["repost_manifest_below_max_reposts"] = bool(has_less_than_requested_reposts)
    if (
        not repaired_repost_rows
        or refresh_reposts
        or has_too_few_reposts
        or has_stale_profile_filter
        or has_curated_source_freshness_refresh
    ):
        repost_date_window = None
        if filter_reposts_to_vod_date_window:
            repost_date_window = _vod_repost_date_window(
                vod_rows=vod_rows,
                vod_inventory_rows=_read_jsonl(full_vod_audio_vod_inventory),
                lookback_days=repost_vod_date_lookback_days,
                lookahead_days=repost_vod_date_lookahead_days,
            )
        repost_rows, repost_summary = build_repost_manifest_from_sources(
            source_manifest=repost_source_manifest,
            url_manifest=repost_url_manifest,
            profile=profile,
            max_reposts=max_reposts,
            discover_reposts=discover_reposts,
            download_reposts=download_reposts,
            repost_date_window=repost_date_window,
            existing_repost_rows=repaired_repost_rows,
        )
        _write_jsonl(repost_manifest, repost_rows)
        summary["repost_manifest_built"] = True
        if refresh_reposts:
            reason = "refresh_requested"
        elif has_stale_profile_filter:
            reason = "profile_filter_policy"
        elif has_too_few_reposts:
            reason = "below_min_reposts"
        elif has_curated_source_freshness_refresh:
            reason = "curated_source_freshness"
        else:
            reason = "missing_or_empty"
        summary["repost_manifest_build_reason"] = reason
        summary["repost_manifest_rows"] = len(repost_rows)
        summary["repost_manifest_detail"] = repost_summary
    else:
        if int(repost_path_summary["remapped_media_rows"]) > 0 or int(repost_path_summary["missing_media_rows"]) > 0:
            _write_jsonl(repost_manifest, repaired_repost_rows)
        summary["repost_manifest_built"] = False
        summary["repost_manifest_rows"] = len(repaired_repost_rows)
    return summary


def _full_vod_audio_block_reason(auto_manifest_summary: dict[str, object]) -> str | None:
    if not bool(auto_manifest_summary.get("full_vod_audio_fingerprints")):
        return None
    audio_summary = auto_manifest_summary.get("full_vod_audio_summary")
    if not isinstance(audio_summary, dict):
        return None
    status = str(audio_summary.get("status") or "").strip()
    if status.startswith("blocked_") or status.startswith("failed_"):
        return status
    selected_vods = int(_safe_float(audio_summary.get("selected_vods")))
    processed_vods = int(_safe_float(audio_summary.get("processed_vods")))
    if selected_vods > 0 and processed_vods <= 0:
        return "no_full_vod_audio_fingerprints"
    return None


def build_full_vod_transcript_caches(
    *,
    vod_rows: list[dict],
    vod_cache_root: pathlib.Path,
    keep_audio: bool,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "event": "full_vod_transcript_cache_build_done",
        "vod_cache_root": str(vod_cache_root),
        "keep_audio": bool(keep_audio),
        "vod_rows": len(vod_rows),
        "selected_vods": 0,
        "processed_vods": 0,
        "built_transcripts": 0,
        "cached_transcripts": 0,
        "failed_vods": 0,
        "unavailable_vods": 0,
        "skipped_vods": 0,
        "download_workers": 1,
        "transcribe_workers": 1,
        "pipeline_mode": "download_queue_ahead",
        "status": "passed",
    }
    seen_vods: set[str] = set()
    unique_vod_rows: list[dict] = []
    for row in vod_rows:
        vod_id = _row_vod_id(row)
        if not vod_id or vod_id in seen_vods:
            continue
        seen_vods.add(vod_id)
        unique_vod_rows.append(row)
    summary["selected_vods"] = len(unique_vod_rows)
    if not unique_vod_rows:
        summary["status"] = "no_selected_vods"
        done_summary = dict(summary)
        done_summary.pop("event", None)
        _log_event("full_vod_transcript_cache_build_done", **done_summary)
        return summary

    classified_vods: list[dict[str, object]] = []
    missing_transcripts = 0
    for vod_index, row in enumerate(unique_vod_rows, start=1):
        vod_id = _row_vod_id(row)
        cache_dir = vod_cache_root / vod_id
        compatible = build_twitch_training_dataset._load_compatible_transcript_cache(cache_dir, vod_id)
        if compatible is not None:
            transcript_path, transcript = compatible
            classified_vods.append(
                {
                    "kind": "cached",
                    "vod_id": vod_id,
                    "vod_index": vod_index,
                    "transcript_path": transcript_path,
                    "transcript": transcript,
                }
            )
            continue
        vod_url = _vod_url_for_rows(vod_id, [row])
        if not vod_url:
            classified_vods.append({"kind": "skip", "vod_id": vod_id, "vod_index": vod_index})
            continue
        audio_path = cache_dir / "full_vod_audio" / "source.m4a"
        classified_vods.append(
            {
                "kind": "missing",
                "vod_id": vod_id,
                "vod_index": vod_index,
                "vod_url": vod_url,
                "cache_dir": cache_dir,
                "audio_path": audio_path,
            }
        )
        missing_transcripts += 1
    _log_event(
        "full_vod_transcript_cache_build_start",
        vod_rows=len(vod_rows),
        selected_vods=len(unique_vod_rows),
        missing_transcripts=missing_transcripts,
        cached_transcripts=len(unique_vod_rows) - missing_transcripts,
        keep_audio=bool(keep_audio),
    )

    from services import transcriber

    missing_vods: list[dict[str, object]] = []
    for item in classified_vods:
        kind = str(item.get("kind") or "")
        vod_id = str(item.get("vod_id") or "")
        vod_index = int(item.get("vod_index") or 0)
        if kind == "cached":
            transcript_path = item["transcript_path"]
            transcript = list(item.get("transcript") or [])
            summary["cached_transcripts"] = int(summary["cached_transcripts"]) + 1
            summary["processed_vods"] = int(summary["processed_vods"]) + 1
            _log_event(
                "full_vod_transcript_cache_vod_cached",
                vod_id=vod_id,
                vod_index=vod_index,
                selected_vods=len(unique_vod_rows),
                transcript_path=str(transcript_path),
                transcript_segments=len(transcript),
            )
            continue
        if kind == "skip":
            summary["skipped_vods"] = int(summary["skipped_vods"]) + 1
            _log_event(
                "full_vod_transcript_cache_vod_skip",
                vod_id=vod_id,
                vod_index=vod_index,
                selected_vods=len(unique_vod_rows),
                reason="missing_vod_url",
            )
            continue
        if kind == "missing":
            missing_vods.append(item)

    def _ensure_vod_audio(item: dict[str, object]) -> dict[str, object]:
        vod_id = str(item.get("vod_id") or "")
        vod_index = int(item.get("vod_index") or 0)
        vod_url = str(item.get("vod_url") or "")
        audio_path = pathlib.Path(item["audio_path"])
        downloaded_audio = False
        try:
            audio_cached = _usable_media(audio_path)
            _log_event(
                "full_vod_transcript_cache_vod_start",
                vod_id=vod_id,
                vod_index=vod_index,
                selected_vods=len(unique_vod_rows),
                vod_url=vod_url,
                audio_cached=audio_cached,
                pipeline_stage="audio_prefetch",
            )
            if not audio_cached:
                _download_full_vod_audio(vod_url, audio_path)
                downloaded_audio = True
            return {
                "ok": True,
                "vod_id": vod_id,
                "vod_index": vod_index,
                "vod_url": vod_url,
                "cache_dir": item["cache_dir"],
                "audio_path": audio_path,
                "downloaded_audio": downloaded_audio,
            }
        except Exception as exc:
            return {
                "ok": False,
                "vod_id": vod_id,
                "vod_index": vod_index,
                "error": exc,
            }

    if missing_vods:
        _log_event(
            "full_vod_transcript_cache_pipeline_start",
            missing_vods=len(missing_vods),
            download_workers=1,
            transcribe_workers=1,
            pipeline_mode="download_queue_ahead",
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as download_executor:
        audio_futures = [download_executor.submit(_ensure_vod_audio, item) for item in missing_vods]
        for missing_index, audio_future in enumerate(audio_futures):
            audio_result = audio_future.result()
            vod_id = str(audio_result.get("vod_id") or "")
            vod_index = int(audio_result.get("vod_index") or 0)
            if not bool(audio_result.get("ok")):
                exc = audio_result.get("error")
                if not isinstance(exc, Exception):
                    exc = RuntimeError("unknown VOD audio prefetch error")
                is_unavailable_vod = _is_permanent_vod_download_error(exc)
                summary["failed_vods"] = int(summary["failed_vods"]) + 1
                summary["unavailable_vods"] = int(summary["unavailable_vods"]) + (1 if is_unavailable_vod else 0)
                _log_event(
                    "full_vod_transcript_cache_vod_failed",
                    vod_id=vod_id,
                    vod_index=vod_index,
                    selected_vods=len(unique_vod_rows),
                    error=_short_error(exc),
                    **_exception_log_fields(exc),
                    permanent_vod_unavailable=is_unavailable_vod,
                    pipeline_stage="audio_prefetch",
                )
                continue
            audio_path = pathlib.Path(audio_result["audio_path"])
            cache_dir = pathlib.Path(audio_result["cache_dir"])
            downloaded_audio = bool(audio_result.get("downloaded_audio"))
            try:
                _log_event(
                    "full_vod_transcript_cache_vod_transcribe_start",
                    vod_id=vod_id,
                    vod_index=vod_index,
                    selected_vods=len(unique_vod_rows),
                    audio_path=str(audio_path),
                    queued_audio_downloads_remaining=len(audio_futures) - missing_index - 1,
                )
                transcript = transcriber.transcribe(audio_path, enable_diarization=False)
                transcript_payload = build_twitch_training_dataset._transcript_cache_payload(vod_id)
                transcript_path = build_twitch_training_dataset._transcript_cache_path(cache_dir, transcript_payload)
                build_twitch_training_dataset._write_transcript_cache(transcript_path, transcript_payload, transcript)
                build_twitch_training_dataset._prune_compatible_transcript_caches(
                    cache_dir,
                    keep_path=transcript_path,
                    vod_id=vod_id,
                )
                if not keep_audio:
                    try:
                        audio_path.unlink()
                    except OSError:
                        pass
            except Exception as exc:
                summary["failed_vods"] = int(summary["failed_vods"]) + 1
                _log_event(
                    "full_vod_transcript_cache_vod_failed",
                    vod_id=vod_id,
                    vod_index=vod_index,
                    selected_vods=len(unique_vod_rows),
                    error=_short_error(exc),
                    **_exception_log_fields(exc),
                    permanent_vod_unavailable=False,
                    pipeline_stage="transcribe",
                )
                continue

            summary["built_transcripts"] = int(summary["built_transcripts"]) + 1
            summary["processed_vods"] = int(summary["processed_vods"]) + 1
            _log_event(
                "full_vod_transcript_cache_vod_done",
                vod_id=vod_id,
                vod_index=vod_index,
                selected_vods=len(unique_vod_rows),
                transcript_path=str(transcript_path),
                transcript_segments=len(transcript),
                downloaded_audio=downloaded_audio,
                kept_audio=bool(keep_audio and _usable_media(audio_path)),
            )

    if int(summary["processed_vods"]) <= 0:
        summary["status"] = "failed_no_vod_transcripts"
    elif int(summary["failed_vods"]) > 0:
        summary["status"] = "partial"
    done_summary = dict(summary)
    done_summary.pop("event", None)
    _log_event("full_vod_transcript_cache_build_done", **done_summary)
    return summary


def build_full_vod_audio_fingerprints(
    *,
    labels_path: pathlib.Path,
    vod_inventory_path: pathlib.Path | None = None,
    selected_vods_path: pathlib.Path,
    vod_cache_root: pathlib.Path,
    profile: str,
    max_vods: int,
    keep_audio: bool,
    workers: int = 1,
) -> dict[str, object]:
    inventory_rows = _read_jsonl(vod_inventory_path) if vod_inventory_path is not None else []
    labels = [] if inventory_rows else _read_jsonl(labels_path)
    selected_rows = _read_jsonl(selected_vods_path)
    selected_vods = _ordered_vod_ids(selected_rows)
    if max_vods > 0:
        selected_vods = selected_vods[:max_vods]
    vod_rows = [*inventory_rows, *labels]
    rows_by_vod: dict[str, list[dict]] = {}
    for row in vod_rows:
        if str(row.get("broadcaster") or row.get("creator_id") or row.get("streamer") or profile).strip().lower() != profile.strip().lower():
            continue
        vod_id = _row_vod_id(row)
        if vod_id:
            rows_by_vod.setdefault(vod_id, []).append(row)
    if not selected_vods:
        selected_vods = sorted(rows_by_vod, key=lambda vod_id: (-len(rows_by_vod[vod_id]), vod_id))
        if max_vods > 0:
            selected_vods = selected_vods[:max_vods]
    input_selected_vods = len(selected_vods)
    skipped_not_in_inventory = 0
    skipped_cache_only_without_full_audio_fingerprint = 0
    if inventory_rows:
        filtered_vods: list[str] = []
        for vod_id in selected_vods:
            vod_metadata_rows = rows_by_vod.get(vod_id) or []
            if not vod_metadata_rows:
                skipped_not_in_inventory += 1
                continue
            has_api_source = any(_row_has_inventory_source(row, "twitch_vod_api") for row in vod_metadata_rows)
            if not has_api_source and not _cached_full_vod_audio_fingerprint_is_usable(vod_cache_root, vod_id, vod_metadata_rows):
                skipped_cache_only_without_full_audio_fingerprint += 1
                continue
            filtered_vods.append(vod_id)
        selected_vods = filtered_vods

    summary: dict[str, object] = {
        "event": "full_vod_audio_fingerprint_build_done",
        "labels_path": str(labels_path),
        "vod_inventory_path": str(vod_inventory_path) if vod_inventory_path is not None else "",
        "selected_vods_path": str(selected_vods_path),
        "vod_cache_root": str(vod_cache_root),
        "profile": profile,
        "input_label_rows": len(labels),
        "input_vod_inventory_rows": len(inventory_rows),
        "max_vods": int(max_vods),
        "keep_audio": bool(keep_audio),
        "workers": max(int(workers), 1),
        "input_selected_vods": input_selected_vods,
        "selected_vods": len(selected_vods),
        "skipped_not_in_vod_inventory": skipped_not_in_inventory,
        "skipped_cache_only_without_full_audio_fingerprint": skipped_cache_only_without_full_audio_fingerprint,
        "processed_vods": 0,
        "built_fingerprints": 0,
        "cached_fingerprints": 0,
        "failed_vods": 0,
        "unavailable_vods": 0,
        "skipped_vods": 0,
        "total_duration_sec": 0.0,
        "status": "passed",
    }
    if not selected_vods:
        summary["status"] = "no_selected_vods"
        done_summary = dict(summary)
        done_summary.pop("event", None)
        _log_event("full_vod_audio_fingerprint_build_done", **done_summary)
        return summary
    fpcalc = shutil.which("fpcalc")
    if fpcalc is None:
        summary["status"] = "blocked_missing_fpcalc"
        _log_event("full_vod_audio_fingerprint_blocked", reason="missing_fpcalc")
        return summary

    _log_event(
        "full_vod_audio_fingerprint_build_start",
        labels_path=str(labels_path),
        selected_vods_path=str(selected_vods_path),
        input_selected_vods=input_selected_vods,
        selected_vods=len(selected_vods),
        skipped_not_in_vod_inventory=skipped_not_in_inventory,
        skipped_cache_only_without_full_audio_fingerprint=skipped_cache_only_without_full_audio_fingerprint,
        workers=max(int(workers), 1),
        keep_audio=bool(keep_audio),
    )

    def _process_vod(vod_index: int, vod_id: str) -> dict[str, object]:
        result: dict[str, object] = {
            "processed_vods": 0,
            "built_fingerprints": 0,
            "cached_fingerprints": 0,
            "failed_vods": 0,
            "unavailable_vods": 0,
            "skipped_vods": 0,
            "duration_sec": 0.0,
        }
        vod_metadata_rows = rows_by_vod.get(vod_id) or []
        vod_url = _vod_url_for_rows(vod_id, [*selected_rows, *vod_metadata_rows])
        if not vod_url:
            _log_event("full_vod_audio_fingerprint_vod_skip", vod_id=vod_id, reason="missing_vod_url")
            result["skipped_vods"] = 1
            return result
        horizon, horizon_source = _vod_horizon(vod_metadata_rows, padding_sec=0)
        audio_dir = vod_cache_root / vod_id / "full_vod_audio"
        audio_path = audio_dir / "source.m4a"
        fingerprint_path = audio_dir / "source.fpcalc.json"
        metadata_path = audio_dir / "source.metadata.json"
        metadata_payload = {
            "vod_id": vod_id,
            "expanded_start": 0.0,
            "expanded_end": round(float(horizon), 3) if horizon > 0 else 0.0,
            "source_kind": "full_vod_audio_fingerprint",
            "source": {
                "kind": "full_vod_audio_fingerprint",
                "vod_url": vod_url,
                "horizon_source": horizon_source,
            },
        }
        if _fingerprint_cache_is_usable(fingerprint_path, metadata_path, metadata_payload):
            _log_event(
                "full_vod_audio_fingerprint_vod_cached",
                vod_id=vod_id,
                vod_index=vod_index,
                selected_vods=len(selected_vods),
                fingerprint_path=str(fingerprint_path),
            )
            result["cached_fingerprints"] = 1
            result["processed_vods"] = 1
            result["duration_sec"] = max(horizon, 0.0)
            return result
        try:
            _log_event(
                "full_vod_audio_fingerprint_vod_start",
                vod_id=vod_id,
                vod_index=vod_index,
                selected_vods=len(selected_vods),
                vod_url=vod_url,
                horizon=round(horizon, 3),
                horizon_source=horizon_source,
            )
            _download_full_vod_audio(vod_url, audio_path)
            fingerprint = matcher.fingerprint_media_to_file(audio_path, fingerprint_path, fpcalc)
            metadata_payload["expanded_end"] = round(max(float(horizon), float(fingerprint.duration)), 3)
            metadata_payload["fingerprint_duration"] = round(float(fingerprint.duration), 3)
            _write_json(metadata_path, metadata_payload)
            if not keep_audio:
                try:
                    audio_path.unlink()
                except OSError:
                    pass
        except Exception as exc:
            is_unavailable_vod = _is_permanent_vod_download_error(exc)
            _log_event(
                "full_vod_audio_fingerprint_vod_failed",
                vod_id=vod_id,
                vod_index=vod_index,
                error=_short_error(exc),
                **_exception_log_fields(exc),
                permanent_vod_unavailable=is_unavailable_vod,
            )
            result["failed_vods"] = 1
            result["unavailable_vods"] = 1 if is_unavailable_vod else 0
            return result
        _log_event(
            "full_vod_audio_fingerprint_vod_done",
            vod_id=vod_id,
            vod_index=vod_index,
            fingerprint_path=str(fingerprint_path),
            duration_sec=float(metadata_payload["expanded_end"]),
            kept_audio=bool(keep_audio and _usable_media(audio_path)),
        )
        result["processed_vods"] = 1
        result["built_fingerprints"] = 1
        result["duration_sec"] = float(metadata_payload["expanded_end"])
        return result

    worker_count = min(max(int(workers), 1), len(selected_vods))
    results: list[dict[str, object]] = []
    if worker_count == 1:
        results = [_process_vod(vod_index, vod_id) for vod_index, vod_id in enumerate(selected_vods, start=1)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_process_vod, vod_index, vod_id)
                for vod_index, vod_id in enumerate(selected_vods, start=1)
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    for result in results:
        for key in (
            "processed_vods",
            "built_fingerprints",
            "cached_fingerprints",
            "failed_vods",
            "unavailable_vods",
            "skipped_vods",
        ):
            summary[key] = int(summary[key]) + int(result.get(key) or 0)
        summary["total_duration_sec"] = round(
            float(summary["total_duration_sec"]) + float(result.get("duration_sec") or 0.0),
            3,
        )
    usable_fingerprints = int(summary["processed_vods"])
    if selected_vods and usable_fingerprints <= 0:
        summary["status"] = "failed_no_full_vod_audio_fingerprints"
    elif int(summary["failed_vods"]) > 0:
        summary["status"] = "partial"
    done_summary = dict(summary)
    done_summary.pop("event", None)
    _log_event("full_vod_audio_fingerprint_build_done", **done_summary)
    return summary


def build_coarse_vod_tiles(
    *,
    labels_path: pathlib.Path,
    selected_vods_path: pathlib.Path,
    vod_cache_root: pathlib.Path,
    profile: str,
    tile_sec: float,
    stride_sec: float,
    max_tiles_per_vod: int,
    max_vods: int,
    horizon_padding_sec: float,
) -> dict[str, object]:
    labels = _read_jsonl(labels_path)
    selected_rows = _read_jsonl(selected_vods_path)
    selected_vods = _ordered_vod_ids(selected_rows)
    if max_vods > 0:
        selected_vods = selected_vods[:max_vods]
    labels_by_vod: dict[str, list[dict]] = {}
    for row in labels:
        if str(row.get("broadcaster") or row.get("creator_id") or row.get("streamer") or profile).strip().lower() != profile.strip().lower():
            continue
        vod_id = _row_vod_id(row)
        if vod_id:
            labels_by_vod.setdefault(vod_id, []).append(row)
    if not selected_vods:
        selected_vods = sorted(labels_by_vod, key=lambda vod_id: (-len(labels_by_vod[vod_id]), vod_id))
        if max_vods > 0:
            selected_vods = selected_vods[:max_vods]

    summary: dict[str, object] = {
        "event": "coarse_vod_tile_build_done",
        "labels_path": str(labels_path),
        "selected_vods_path": str(selected_vods_path),
        "vod_cache_root": str(vod_cache_root),
        "profile": profile,
        "tile_sec": float(tile_sec),
        "stride_sec": float(stride_sec),
        "max_tiles_per_vod": int(max_tiles_per_vod),
        "max_vods": int(max_vods),
        "coverage_mode": _coarse_vod_coverage_mode(
            tile_sec=tile_sec,
            stride_sec=stride_sec,
            max_tiles_per_vod=max_tiles_per_vod,
        ),
        "selected_vods": len(selected_vods),
        "built_tiles": 0,
        "cached_tiles": 0,
        "failed_tiles": 0,
        "skipped_vods": 0,
        "unavailable_vods": 0,
        "processed_vods": 0,
        "full_duration_vods": 0,
        "label_horizon_vods": 0,
        "total_tiles": 0,
        "total_covered_sec": 0.0,
        "status": "passed",
    }
    _log_event(
        "coarse_vod_tile_build_start",
        labels_path=str(labels_path),
        selected_vods_path=str(selected_vods_path),
        selected_vods=len(selected_vods),
        tile_sec=float(tile_sec),
        stride_sec=float(stride_sec),
        max_tiles_per_vod=int(max_tiles_per_vod),
        coverage_mode=str(summary["coverage_mode"]),
    )
    for vod_index, vod_id in enumerate(selected_vods, start=1):
        vod_labels = labels_by_vod.get(vod_id) or []
        vod_url = _vod_url_for_rows(vod_id, [*selected_rows, *vod_labels])
        if not vod_url:
            summary["skipped_vods"] = int(summary["skipped_vods"]) + 1
            _log_event("coarse_vod_tile_vod_skip", vod_id=vod_id, reason="missing_vod_url")
            continue
        horizon, horizon_source = _vod_horizon(vod_labels, padding_sec=horizon_padding_sec)
        if horizon <= 0:
            summary["skipped_vods"] = int(summary["skipped_vods"]) + 1
            _log_event("coarse_vod_tile_vod_skip", vod_id=vod_id, reason="missing_label_horizon")
            continue
        tile_ranges = _coarse_tile_ranges(
            horizon=horizon,
            tile_sec=tile_sec,
            stride_sec=stride_sec,
            max_tiles=max_tiles_per_vod,
        )
        summary["processed_vods"] = int(summary["processed_vods"]) + 1
        if horizon_source == "vod_duration":
            summary["full_duration_vods"] = int(summary["full_duration_vods"]) + 1
        else:
            summary["label_horizon_vods"] = int(summary["label_horizon_vods"]) + 1
        summary["total_tiles"] = int(summary["total_tiles"]) + len(tile_ranges)
        summary["total_covered_sec"] = round(float(summary["total_covered_sec"]) + float(horizon), 3)
        _log_event(
            "coarse_vod_tile_vod_start",
            vod_id=vod_id,
            vod_index=vod_index,
            selected_vods=len(selected_vods),
            tiles=len(tile_ranges),
            horizon=round(horizon, 3),
            horizon_source=horizon_source,
            coverage_mode=str(summary["coverage_mode"]),
        )
        for tile_index, (start, end) in enumerate(tile_ranges, start=1):
            source_signature = {
                "kind": "coarse_vod_audio_tile",
                "vod_url": vod_url,
                "tile_index": tile_index,
                "tile_start": round(start, 3),
                "tile_end": round(end, 3),
                "tile_sec": round(float(tile_sec), 3),
                "stride_sec": round(float(stride_sec), 3),
            }
            digest = hashlib.sha256(
                json.dumps(source_signature, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
            tile_dir = vod_cache_root / vod_id / "scoped_windows" / f"coarse_{digest}"
            source_path = tile_dir / "source.mp4"
            metadata_path = tile_dir / "source.metadata.json"
            metadata_payload = build_twitch_training_dataset._scoped_source_metadata(
                vod_id,
                expanded_start=start,
                expanded_end=end,
                source_kind="coarse_vod_audio_tile",
                source_signature=source_signature,
            )
            was_cached = build_twitch_training_dataset._scoped_source_is_usable(
                source_path,
                metadata_path,
                metadata_payload,
            )
            try:
                build_twitch_training_dataset._load_or_download_window_scoped_vod_range(
                    vod_id,
                    {"vod_url": vod_url},
                    source_path,
                    metadata_path,
                    metadata_payload=metadata_payload,
                    job_id=f"coarse_vod_tile_{vod_id}_{tile_index}",
                    expanded_start=start,
                    expanded_end=end,
                )
            except Exception as exc:
                summary["failed_tiles"] = int(summary["failed_tiles"]) + 1
                is_unavailable_vod = _is_permanent_vod_download_error(exc)
                _log_event(
                    "coarse_vod_tile_failed",
                    vod_id=vod_id,
                    tile_index=tile_index,
                    start=round(start, 3),
                    end=round(end, 3),
                    error=_short_error(exc),
                    permanent_vod_unavailable=is_unavailable_vod,
                )
                if is_unavailable_vod:
                    summary["unavailable_vods"] = int(summary["unavailable_vods"]) + 1
                    _log_event(
                        "coarse_vod_tile_vod_abort",
                        vod_id=vod_id,
                        reason="permanent_vod_unavailable",
                        failed_tile_index=tile_index,
                        skipped_remaining_tiles=max(len(tile_ranges) - tile_index, 0),
                        error=_short_error(exc),
                    )
                    break
                continue
            if was_cached:
                summary["cached_tiles"] = int(summary["cached_tiles"]) + 1
            else:
                summary["built_tiles"] = int(summary["built_tiles"]) + 1
        _log_event(
            "coarse_vod_tile_vod_done",
            vod_id=vod_id,
            built_tiles=int(summary["built_tiles"]),
            cached_tiles=int(summary["cached_tiles"]),
            failed_tiles=int(summary["failed_tiles"]),
            unavailable_vods=int(summary["unavailable_vods"]),
        )
    done_summary = dict(summary)
    done_summary.pop("event", None)
    _log_event("coarse_vod_tile_build_done", **done_summary)
    return summary


def build_vod_manifest_from_cache(
    vod_cache_root: pathlib.Path,
    *,
    include_scoped_windows: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    full_audio_vods: set[str] = set()
    if not vod_cache_root.exists():
        return rows
    for metadata_path in sorted(vod_cache_root.glob("*/full_vod_audio/source.metadata.json")):
        fingerprint_path = metadata_path.with_name("source.fpcalc.json")
        if not _usable_fingerprint(fingerprint_path):
            continue
        metadata = _read_json(metadata_path)
        vod_id = str(metadata.get("vod_id") or metadata_path.parents[1].name).strip()
        source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
        vod_url = str(source.get("vod_url") or f"https://www.twitch.tv/videos/{vod_id}").strip()
        expanded_start = _safe_float(metadata.get("expanded_start"))
        expanded_end = _safe_float(metadata.get("expanded_end") or metadata.get("fingerprint_duration"))
        full_audio_vods.add(vod_id)
        rows.append(
            {
                "vod_id": vod_id,
                "vod_url": vod_url,
                "fingerprint_path": str(fingerprint_path),
                "source_start_offset": round(expanded_start, 3),
                "source_end_offset": round(expanded_end, 3),
                "manifest_source": "vod_feature_cache_full_vod_audio_fingerprint",
            }
        )
    if not include_scoped_windows:
        return _dedupe_manifest_rows(rows, keys=("fingerprint_path", "video_path"))
    for metadata_path in sorted(vod_cache_root.glob("*/scoped_windows/*/source.metadata.json")):
        source_path = metadata_path.with_name("source.mp4")
        if not _usable_media(source_path):
            continue
        metadata = _read_json(metadata_path)
        vod_id = str(metadata.get("vod_id") or metadata_path.parents[2].name).strip()
        if vod_id in full_audio_vods:
            continue
        source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
        vod_url = str(source.get("vod_url") or f"https://www.twitch.tv/videos/{vod_id}").strip()
        expanded_start = _safe_float(metadata.get("expanded_start"))
        expanded_end = _safe_float(metadata.get("expanded_end"))
        rows.append(
            {
                "vod_id": vod_id,
                "vod_url": vod_url,
                "video_path": str(source_path),
                "source_start_offset": round(expanded_start, 3),
                "source_end_offset": round(expanded_end, 3),
                "source_window_start": _safe_float(source.get("window_start")),
                "source_window_end": _safe_float(source.get("window_end")),
                "scoped_window_id": metadata_path.parent.name,
                "manifest_source": "vod_feature_cache_scoped_window",
            }
        )
    return _dedupe_manifest_rows(rows, keys=("fingerprint_path", "video_path"))


def build_repost_manifest_from_sources(
    *,
    source_manifest: pathlib.Path,
    url_manifest: pathlib.Path,
    profile: str,
    max_reposts: int,
    discover_reposts: bool,
    download_reposts: bool,
    repost_date_window: tuple[datetime, datetime] | None = None,
    existing_repost_rows: list[dict] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    entries: list[dict[str, str]] = []
    detail: dict[str, object] = {
        "source_manifest_exists": source_manifest.exists(),
        "url_manifest_exists": url_manifest.exists(),
        "source_manifest_rows": 0,
        "url_manifest_rows": 0,
        "download_failures": 0,
        "date_window_filter_enabled": bool(repost_date_window),
        "date_window_start": repost_date_window[0].isoformat() if repost_date_window else "",
        "date_window_end": repost_date_window[1].isoformat() if repost_date_window else "",
    }
    if discover_reposts and source_manifest.exists():
        seeds = auto_train_short_form._read_source_manifest(source_manifest)
        seeds = auto_train_short_form._filter_seeds_by_streamers(seeds, {profile.strip().lower()})
        detail["source_manifest_rows"] = len(seeds)
        args = SimpleNamespace(
            min_duration=8,
            max_duration=180,
            top_fraction=1.0,
            bottom_fraction=0.0,
            proxy="",
            skip_platforms=set(),
        )
        entries.extend(auto_train_short_form._discover_seeds(seeds, args))
    if url_manifest.exists():
        url_entries = collect_viral_clips._read_entries(url_manifest)
        detail["url_manifest_rows"] = len(url_entries)
        entries.extend(url_entries)

    entries = _dedupe_entries(entries)
    detail["candidate_entries"] = len(entries)

    eligible_entries: list[dict] = []
    skipped_non_profile_reposts = 0
    skipped_non_profile_samples: list[dict[str, str]] = []
    skipped_outside_vod_date_window = 0
    skipped_missing_publish_date = 0
    skipped_date_samples: list[dict[str, str]] = []
    for entry in entries:
        url = str(entry.get("url") or entry.get("source_url") or "").strip()
        if not url:
            continue
        if not _repost_entry_matches_profile(entry, url, profile):
            skipped_non_profile_reposts += 1
            if len(skipped_non_profile_samples) < 10:
                skipped_non_profile_samples.append(
                    {
                        "url": url,
                        "title": str(entry.get("title") or "")[:160],
                        "uploader": str(entry.get("uploader") or entry.get("channel") or "")[:120],
                    }
                )
            continue
        if repost_date_window:
            published_at = _entry_publish_datetime(entry)
            if published_at is None:
                skipped_missing_publish_date += 1
                if len(skipped_date_samples) < 10:
                    skipped_date_samples.append(
                        {
                            "reason": "missing_publish_date",
                            "url": url,
                            "title": str(entry.get("title") or "")[:160],
                            "uploader": str(entry.get("uploader") or entry.get("channel") or "")[:120],
                        }
                    )
                continue
            if not (repost_date_window[0] <= published_at <= repost_date_window[1]):
                skipped_outside_vod_date_window += 1
                if len(skipped_date_samples) < 10:
                    skipped_date_samples.append(
                        {
                            "reason": "outside_vod_date_window",
                            "published_at": published_at.isoformat(),
                            "url": url,
                            "title": str(entry.get("title") or "")[:160],
                        }
                    )
                continue
        eligible_entries.append(entry)

    eligible_profile_date_rows = len(eligible_entries)
    eligible_entries.sort(key=_repost_candidate_sort_key, reverse=True)
    if max_reposts > 0:
        eligible_entries = eligible_entries[:max_reposts]

    rows, download_detail = _materialize_repost_rows(
        eligible_entries,
        profile=profile,
        download_reposts=download_reposts,
        existing_repost_rows=existing_repost_rows or [],
    )
    detail.update(download_detail)
    detail["skipped_non_profile_reposts"] = skipped_non_profile_reposts
    detail["skipped_non_profile_samples"] = skipped_non_profile_samples
    detail["skipped_outside_vod_date_window"] = skipped_outside_vod_date_window
    detail["skipped_missing_publish_date"] = skipped_missing_publish_date
    detail["skipped_date_samples"] = skipped_date_samples
    detail["eligible_profile_date_rows"] = eligible_profile_date_rows
    detail["selected_repost_candidates"] = len(eligible_entries)
    detail["downloaded_or_cached_rows"] = len(rows)
    return _dedupe_manifest_rows(rows, keys=("source_url", "video_path")), detail


def _materialize_repost_rows(
    entries: list[dict],
    *,
    profile: str,
    download_reposts: bool,
    existing_repost_rows: list[dict] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    reusable_media_by_key = _existing_repost_media_by_key(existing_repost_rows or [])
    prepared: list[tuple[int, dict, str, pathlib.Path | None]] = []
    to_download: list[tuple[int, str]] = []
    reused_existing_media_rows = 0
    for index, entry in enumerate(entries):
        url = str(entry.get("url") or entry.get("source_url") or "").strip()
        if not url:
            continue
        video_path = pathlib.Path(str(entry.get("video_path") or "")).expanduser()
        if _usable_media(video_path):
            prepared.append((index, entry, url, video_path))
            continue
        existing_media_path = reusable_media_by_key.get(_repost_media_reuse_key(entry, url))
        if existing_media_path is not None:
            prepared.append((index, entry, url, existing_media_path))
            reused_existing_media_rows += 1
            continue
        if download_reposts:
            prepared.append((index, entry, url, None))
            to_download.append((index, url))

    download_failures = 0
    download_error_cooldowns = 0
    download_success_cooldowns = 0
    download_request_cooldowns = 0
    rate_limit_cooldowns = 0
    rate_limit_cooldown_sec = _short_form_rate_limit_cooldown_sec()
    downloaded: dict[int, pathlib.Path] = {}
    for index, url in to_download:
        outcome = "success"
        is_rate_limited = False
        try:
            downloaded[index] = collect_viral_clips._ensure_short_form_media_cached(url)
        except Exception as exc:
            outcome = "failed"
            is_rate_limited = _is_youtube_rate_limited_error(exc)
            download_failures += 1
            download_error_cooldowns += 1
            if is_rate_limited:
                rate_limit_cooldowns += 1
            _log_event(
                "cross_platform_repost_download_failed_cooldown",
                cooldown_sec=round(rate_limit_cooldown_sec, 3),
                error=_short_error(exc, limit=500),
                rate_limited=is_rate_limited,
                url=url,
            )
        finally:
            if outcome == "success":
                download_success_cooldowns += 1
            download_request_cooldowns += 1
            _log_event(
                "cross_platform_repost_download_request_cooldown",
                cooldown_sec=round(rate_limit_cooldown_sec, 3),
                outcome=outcome,
                rate_limited=is_rate_limited,
                url=url,
            )
            if rate_limit_cooldown_sec > 0.0:
                time.sleep(rate_limit_cooldown_sec)

    rows: list[dict] = []
    for index, entry, url, existing_path in prepared:
        video_path = existing_path or downloaded.get(index)
        if video_path is None or not _usable_media(video_path):
            continue
        rows.append(_repost_row(entry, url, video_path, profile))

    return rows, {
        "download_failures": download_failures,
        "repost_download_mode": "serial",
        "repost_download_request_cooldown_sec": rate_limit_cooldown_sec,
        "repost_download_submitted_rows": len(to_download),
        "repost_download_error_cooldowns": download_error_cooldowns,
        "repost_download_request_cooldowns": download_request_cooldowns,
        "repost_download_rate_limit_cooldown_sec": rate_limit_cooldown_sec,
        "repost_download_rate_limit_cooldowns": rate_limit_cooldowns,
        "repost_download_success_cooldowns": download_success_cooldowns,
        "repost_existing_media_reused_rows": reused_existing_media_rows,
    }


def _existing_repost_media_by_key(rows: list[dict]) -> dict[tuple[str, str], pathlib.Path]:
    media_by_key: dict[tuple[str, str], pathlib.Path] = {}
    for row in rows:
        raw_path = str(row.get("video_path") or "").strip()
        if not raw_path:
            continue
        media_path = pathlib.Path(raw_path).expanduser()
        if not _usable_media(media_path):
            continue
        url = str(row.get("source_url") or row.get("url") or row.get("repost_source_url") or "").strip()
        key = _repost_media_reuse_key(row, url)
        if key[1]:
            media_by_key.setdefault(key, media_path)
    return media_by_key


def _repost_media_reuse_key(row: dict, url: str) -> tuple[str, str]:
    platform = str(row.get("platform") or collect_viral_clips._infer_platform(url)).strip().lower()
    if platform == "youtube":
        youtube_id = _youtube_video_id(url) or str(row.get("clip_id") or row.get("id") or "").strip()
        if youtube_id:
            return ("youtube", youtube_id)
    return (platform or "unknown", url.strip())


def _youtube_video_id(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/", 1)[0]
    if "youtube.com" not in host:
        return ""
    query_id = urllib.parse.parse_qs(parsed.query).get("v", [""])[0].strip()
    if query_id:
        return query_id
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0].lower() in {"shorts", "embed", "live"}:
        return parts[1].strip()
    return ""


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("vod_id") or ""),
            str(row.get("candidate_id") or ""),
            _stable_time(row.get("window_start")),
            _stable_time(row.get("window_end")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _repair_manifest_media_paths(rows: list[dict]) -> tuple[list[dict], dict[str, object]]:
    repaired: list[dict] = []
    remapped = 0
    missing = 0
    missing_samples: list[str] = []
    for row in rows:
        path_key = "fingerprint_path" if str(row.get("fingerprint_path") or "").strip() else "video_path"
        raw_path = str(row.get(path_key) or "").strip()
        if not raw_path:
            missing += 1
            continue
        resolved = matcher._resolve_media_path(raw_path)
        if resolved is None:
            missing += 1
            if len(missing_samples) < 5:
                missing_samples.append(raw_path)
            continue
        copied = dict(row)
        resolved_text = str(resolved)
        if resolved_text != raw_path:
            remapped += 1
            copied[f"original_{path_key}"] = raw_path
            copied[path_key] = resolved_text
        repaired.append(copied)
    return repaired, {
        "input_rows": len(rows),
        "usable_media_rows": len(repaired),
        "remapped_media_rows": remapped,
        "missing_media_rows": missing,
        "missing_media_samples": missing_samples,
    }


def _dedupe_manifest_rows(rows: list[dict], *, keys: tuple[str, ...]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row.get(item) or "").strip() for item in keys)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _dedupe_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        url = str(entry.get("url") or entry.get("source_url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(entry)
    return deduped


def _repost_manifest_needs_profile_filter_rebuild(
    rows: list[dict],
    *,
    source_manifest: pathlib.Path,
    discover_reposts: bool,
) -> bool:
    if not rows or not discover_reposts or not source_manifest.exists():
        return False
    return any(
        int(_safe_float(row.get("profile_filter_version"))) < REPOST_PROFILE_FILTER_VERSION
        for row in rows
    )


def _repost_manifest_needs_curated_source_freshness_refresh(
    *,
    source_manifest: pathlib.Path,
    profile: str,
    discover_reposts: bool,
) -> bool:
    if not discover_reposts or not source_manifest.exists():
        return False
    try:
        seeds = auto_train_short_form._read_source_manifest(source_manifest)
    except Exception:
        logger.debug("Failed to read repost source manifest for freshness check: %s", source_manifest, exc_info=True)
        return False
    normalized_profile = profile.strip().lower()
    if normalized_profile:
        seeds = auto_train_short_form._filter_seeds_by_streamers(seeds, {normalized_profile})
    return any(_is_freshness_checked_curated_source(seed) for seed in seeds)


def _is_freshness_checked_curated_source(seed: dict) -> bool:
    if not _is_curated_repost_source(seed):
        return False
    source_url = str(seed.get("source_url") or seed.get("url") or "").strip()
    if not source_url:
        return False
    return auto_train_short_form._is_youtube_channel_shorts_source(source_url)


def _vod_repost_date_window(
    *,
    vod_rows: list[dict],
    vod_inventory_rows: list[dict],
    lookback_days: int,
    lookahead_days: int,
) -> tuple[datetime, datetime] | None:
    vod_ids = {str(row.get("vod_id") or "").strip() for row in vod_rows if str(row.get("vod_id") or "").strip()}
    if not vod_ids:
        return None
    created_dates = [
        created_at
        for row in vod_inventory_rows
        if str(row.get("vod_id") or "").strip() in vod_ids
        for created_at in [_parse_datetime_value(row.get("created_at"))]
        if created_at is not None
    ]
    if not created_dates:
        return None
    start = min(created_dates) - timedelta(days=max(int(lookback_days), 0))
    end = max(created_dates) + timedelta(days=max(int(lookahead_days), 0))
    return start, end


def _entry_publish_datetime(entry: dict) -> datetime | None:
    timestamp = _safe_float(entry.get("publish_timestamp"))
    if timestamp > 0:
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            pass
    return _parse_datetime_value(entry.get("upload_date"))


def _repost_candidate_sort_key(entry: dict) -> tuple[float, float, float, str]:
    published_at = _entry_publish_datetime(entry)
    publish_timestamp = published_at.timestamp() if published_at is not None else 0.0
    return (
        _safe_float(entry.get("views") or entry.get("view_count")),
        _safe_float(entry.get("viral_score")),
        publish_timestamp,
        str(entry.get("url") or entry.get("source_url") or ""),
    )


def _parse_datetime_value(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) == 8 and raw.isdigit():
        try:
            return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _repost_entry_matches_profile(entry: dict, url: str, profile: str) -> bool:
    profile_token = profile.strip().lower()
    if not profile_token:
        return True
    if _is_curated_repost_source(entry):
        return True
    discovery_source = str(entry.get("discovery_source_url") or "").strip().lower()
    if (
        discovery_source
        and not discovery_source.startswith(("ytsearch", "tiktoksearch"))
        and profile_token in discovery_source
    ):
        return True
    if _is_manual_repost_entry(entry):
        return True
    haystack = " ".join(
        str(entry.get(key) or "")
        for key in (
            "title",
            "uploader",
            "channel",
            "channel_id",
            "description",
        )
    ).lower()
    aliases = {profile_token}
    if profile_token == "jynxzi":
        aliases.add("junko")
    return any(alias and alias in haystack for alias in aliases)


def _is_curated_repost_source(entry: dict) -> bool:
    tags = {
        token.strip().lower()
        for token in str(entry.get("tags") or "").replace(",", ";").split(";")
        if token.strip()
    }
    return bool(tags & {"curated_repost_source", "trusted_repost_source"})


def _is_manual_repost_entry(entry: dict) -> bool:
    if entry.get("discovery_source_url"):
        return False
    metadata_keys = ("title", "uploader", "channel", "description")
    return not any(str(entry.get(key) or "").strip() for key in metadata_keys)


def _repost_row(entry: dict[str, str], url: str, video_path: pathlib.Path, profile: str) -> dict:
    return {
        "video_path": str(video_path),
        "platform": str(entry.get("platform") or collect_viral_clips._infer_platform(url)),
        "source_url": url,
        "clip_id": str(entry.get("clip_id") or entry.get("candidate_id") or ""),
        "streamer": str(entry.get("streamer") or profile),
        "creator_id": str(entry.get("streamer") or profile),
        "views": _safe_float(entry.get("views") or entry.get("view_count")),
        "like_count": _safe_float(entry.get("like_count")),
        "comment_count": _safe_float(entry.get("comment_count")),
        "upload_date": str(entry.get("upload_date") or ""),
        "publish_timestamp": str(entry.get("publish_timestamp") or ""),
        "title": str(entry.get("title") or ""),
        "uploader": str(entry.get("uploader") or ""),
        "channel": str(entry.get("channel") or ""),
        "channel_id": str(entry.get("channel_id") or ""),
        "discovery_source_url": str(entry.get("discovery_source_url") or ""),
        "profile_filter_version": REPOST_PROFILE_FILTER_VERSION,
        "quality_bucket": str(entry.get("quality_bucket") or ""),
        "label_reason": str(entry.get("label_reason") or "cross_platform_repost_source"),
    }


def _usable_media(path: pathlib.Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _usable_fingerprint(path: pathlib.Path) -> bool:
    if not _usable_media(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("fingerprint"))


def _fingerprint_cache_is_usable(
    fingerprint_path: pathlib.Path,
    metadata_path: pathlib.Path,
    expected_metadata: dict[str, object],
) -> bool:
    if not _usable_fingerprint(fingerprint_path) or not metadata_path.exists():
        return False
    metadata = _read_json(metadata_path)
    if str(metadata.get("vod_id") or "") != str(expected_metadata.get("vod_id") or ""):
        return False
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    expected_source = expected_metadata.get("source") if isinstance(expected_metadata.get("source"), dict) else {}
    return (
        str(metadata.get("source_kind") or "") == str(expected_metadata.get("source_kind") or "")
        and str(source.get("vod_url") or "") == str(expected_source.get("vod_url") or "")
    )


def _download_full_vod_audio(vod_url: str, output_path: pathlib.Path) -> pathlib.Path:
    from yt_dlp import YoutubeDL

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _full_vod_audio_temp_path(output_path)
    for path in (temp_path, output_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    apply_yt_dlp_auth_options(ydl_opts)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(vod_url, download=False)
    media_url, headers, format_summary = _select_vod_audio_input(info if isinstance(info, dict) else {})
    _log_event(
        "full_vod_audio_download_input_selected",
        vod_url=vod_url,
        output_path=str(output_path),
        **format_summary,
    )
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
    ]
    header_text = _ffmpeg_header_text(headers)
    if header_text:
        command.extend(["-headers", header_text])
    command.extend(
        [
            "-i",
            media_url,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "11025",
            "-c:a",
            "aac",
            "-b:a",
            "48k",
            str(temp_path),
        ]
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60 * 60 * 8)
    except subprocess.CalledProcessError as exc:
        raise FullVodAudioDownloadError(
            "ffmpeg failed while extracting full VOD audio",
            returncode=exc.returncode,
            stderr=exc.stderr,
            stdout=exc.stdout,
            format_summary=format_summary,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FullVodAudioDownloadError(
            "ffmpeg timed out while extracting full VOD audio",
            returncode=None,
            stderr=exc.stderr,
            stdout=exc.stdout,
            format_summary=format_summary,
        ) from exc
    temp_path.replace(output_path)
    return output_path


def _full_vod_audio_temp_path(output_path: pathlib.Path) -> pathlib.Path:
    suffix = output_path.suffix or ".m4a"
    return output_path.with_name(f"{output_path.stem}.part{suffix}")


class FullVodAudioDownloadError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        returncode: int | None,
        stderr: object,
        stdout: object,
        format_summary: dict[str, object],
    ) -> None:
        self.returncode = returncode
        self.stderr = _decode_process_text(stderr)
        self.stdout = _decode_process_text(stdout)
        self.format_summary = dict(format_summary)
        detail = _short_text(self.stderr or self.stdout or "no ffmpeg output", 1000)
        super().__init__(f"{message}: returncode={returncode} detail={detail}")


def _select_vod_audio_input(info: dict[str, object]) -> tuple[str, dict[str, str], dict[str, object]]:
    formats = info.get("formats") if isinstance(info.get("formats"), list) else []
    candidates: list[dict] = []
    for item in formats:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        if not url:
            continue
        acodec = str(item.get("acodec") or "")
        if acodec == "none":
            continue
        candidates.append(item)
    if not candidates:
        url = str(info.get("url") or "")
        if not url:
            raise RuntimeError("No playable VOD media URL found for audio fingerprint extraction")
        return url, _string_dict(info.get("http_headers")), {
            "format_id": str(info.get("format_id") or ""),
            "format_note": str(info.get("format_note") or ""),
            "protocol": str(info.get("protocol") or ""),
            "ext": str(info.get("ext") or ""),
            "acodec": str(info.get("acodec") or ""),
            "vcodec": str(info.get("vcodec") or ""),
            "height": _safe_float(info.get("height")),
            "tbr": _safe_float(info.get("tbr")),
            "abr": _safe_float(info.get("abr")),
            "source": "info_url",
            "media_url_prefix": url[:120],
        }
    audio_only = [item for item in candidates if str(item.get("vcodec") or "") == "none"]
    if audio_only:
        selected = max(audio_only, key=lambda item: _safe_float(item.get("abr") or item.get("tbr")))
    else:
        selected = min(
            candidates,
            key=lambda item: (
                _safe_float(item.get("height")) or 10_000.0,
                _safe_float(item.get("tbr")) or 10_000.0,
            ),
        )
    selected_url = str(selected["url"])
    return selected_url, _string_dict(selected.get("http_headers") or info.get("http_headers")), {
        "format_id": str(selected.get("format_id") or ""),
        "format_note": str(selected.get("format_note") or ""),
        "protocol": str(selected.get("protocol") or ""),
        "ext": str(selected.get("ext") or ""),
        "acodec": str(selected.get("acodec") or ""),
        "vcodec": str(selected.get("vcodec") or ""),
        "height": _safe_float(selected.get("height")),
        "tbr": _safe_float(selected.get("tbr")),
        "abr": _safe_float(selected.get("abr")),
        "source": "formats",
        "media_url_prefix": selected_url[:120],
    }


def _string_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


def _ffmpeg_header_text(headers: dict[str, str]) -> str:
    return "".join(f"{key}: {value}\r\n" for key, value in headers.items() if key and value)


def _read_json(path: pathlib.Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _ordered_vod_ids(rows: list[dict]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        vod_id = _row_vod_id(row)
        if not vod_id or vod_id in seen:
            continue
        seen.add(vod_id)
        ordered.append(vod_id)
    return ordered


def _row_has_inventory_source(row: dict, expected_source: str) -> bool:
    expected = expected_source.strip().lower()
    raw_sources = row.get("inventory_sources")
    sources: list[str] = []
    if isinstance(raw_sources, list):
        sources.extend(str(source) for source in raw_sources)
    sources.append(str(row.get("inventory_source") or ""))
    return any(source.strip().lower() == expected for source in sources)


def _cached_full_vod_audio_fingerprint_is_usable(
    vod_cache_root: pathlib.Path,
    vod_id: str,
    rows: list[dict],
) -> bool:
    vod_url = _vod_url_for_rows(vod_id, rows)
    horizon, horizon_source = _vod_horizon(rows, padding_sec=0)
    audio_dir = vod_cache_root / vod_id / "full_vod_audio"
    metadata_payload = {
        "vod_id": vod_id,
        "expanded_start": 0.0,
        "expanded_end": round(float(horizon), 3) if horizon > 0 else 0.0,
        "source_kind": "full_vod_audio_fingerprint",
        "source": {
            "kind": "full_vod_audio_fingerprint",
            "vod_url": vod_url,
            "horizon_source": horizon_source,
        },
    }
    return _fingerprint_cache_is_usable(
        audio_dir / "source.fpcalc.json",
        audio_dir / "source.metadata.json",
        metadata_payload,
    )


def _row_vod_id(row: dict) -> str:
    explicit = str(row.get("vod_id") or row.get("matched_vod_id") or row.get("source_vod_id") or "").strip()
    if explicit:
        return explicit
    for key in ("vod_url", "source_url", "review_url", "url"):
        text = str(row.get(key) or "")
        marker = "twitch.tv/videos/"
        if marker in text:
            return text.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]
    return ""


def _vod_url_for_rows(vod_id: str, rows: list[dict]) -> str:
    for row in rows:
        if _row_vod_id(row) != vod_id:
            continue
        for key in ("vod_url", "source_url", "review_url", "url"):
            value = str(row.get(key) or "").strip()
            if "twitch.tv/videos/" in value:
                return value
    return f"https://www.twitch.tv/videos/{vod_id}" if vod_id else ""


def _vod_horizon(rows: list[dict], *, padding_sec: float) -> tuple[float, str]:
    vod_duration = 0.0
    for row in rows:
        for key in (
            "vod_duration_seconds",
            "duration_seconds",
            "stream_duration_seconds",
            "video_duration_seconds",
            "vod_duration",
        ):
            value = _safe_float(row.get(key))
            if value > vod_duration:
                vod_duration = value
    if vod_duration > 0:
        return vod_duration, "vod_duration"

    max_end = 0.0
    for row in rows:
        for key in ("window_end", "final_end", "end"):
            value = _safe_float(row.get(key))
            if value > max_end:
                max_end = value
    if max_end > 0:
        return max_end + max(float(padding_sec), 0.0), "label_horizon"
    return 0.0, "missing"


def _coarse_vod_coverage_mode(
    *,
    tile_sec: float,
    stride_sec: float,
    max_tiles_per_vod: int,
) -> str:
    tile = max(float(tile_sec), 1.0)
    stride = max(float(stride_sec), 1.0)
    if int(max_tiles_per_vod) <= 0 and stride < tile:
        return "full_vod_overlap"
    if int(max_tiles_per_vod) <= 0 and stride <= tile:
        return "full_vod_contiguous"
    return "sampled"


def _coarse_tile_ranges(
    *,
    horizon: float,
    tile_sec: float,
    stride_sec: float,
    max_tiles: int,
) -> list[tuple[float, float]]:
    tile = max(float(tile_sec), 1.0)
    stride = max(float(stride_sec), 1.0)
    limit = max(int(max_tiles), 0)
    starts: list[float] = []
    current = 0.0
    while current < horizon and (limit <= 0 or len(starts) < limit):
        starts.append(current)
        current += stride
    return [(round(start, 3), round(min(start + tile, max(horizon, start + 1.0)), 3)) for start in starts]


def _rows_with_path(rows: list[dict]) -> int:
    return sum(1 for row in rows if str(row.get("video_path") or row.get("fingerprint_path") or "").strip())


def _dataset_sources(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("dataset_source") or "unknown") for row in rows).items()))


def _matched_vod_count(rows: list[dict]) -> int:
    return len({str(row.get("vod_id") or "").strip() for row in rows if str(row.get("vod_id") or "").strip()})


def _reusable_existing_cross_platform_rows(
    rows: list[dict],
    *,
    min_confidence: float,
    verification_mode: str = "transcript",
) -> list[dict]:
    reusable: list[dict] = []
    threshold = max(float(min_confidence), 0.0)
    verification_mode = _normalize_verification_mode(verification_mode)
    for row in rows:
        source = str(row.get("dataset_source") or "")
        label_source = str(row.get("label_source") or "")
        tier = str(row.get("supervision_tier") or "")
        if verification_mode == "transcript":
            if source != "cross_platform_transcript_match" and label_source != "cross_platform_transcript_match":
                continue
        else:
            if (
                source != "cross_platform_audio_fingerprint"
                and label_source != "cross_platform_audio_fingerprint"
                and tier != "cross_platform_outcome"
            ):
                continue
        if int(_safe_float(row.get("profile_filter_version"))) < REPOST_PROFILE_FILTER_VERSION:
            continue
        try:
            if verification_mode == "transcript":
                confidence = float(row.get("transcript_similarity") or row.get("label_score") or 0.0)
            else:
                confidence = float(row.get("fingerprint_confidence") or row.get("label_score") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < threshold:
            continue
        reusable.append(row)
    return reusable


def _existing_output_compatible_with_verification_mode(
    existing_rows: list[dict],
    reusable_existing_rows: list[dict],
    *,
    verification_mode: str,
) -> bool:
    if not existing_rows:
        return False
    if _normalize_verification_mode(verification_mode) == "audio_fingerprint":
        return True
    if not reusable_existing_rows:
        return False
    return len(reusable_existing_rows) == len(existing_rows)


def _cross_platform_match_input_signature(
    *,
    vod_rows: list[dict],
    repost_rows: list[dict],
    verification_mode: str,
    min_confidence: float,
    vod_cache_root: pathlib.Path,
) -> str:
    payload = {
        "version": MATCH_INPUT_SIGNATURE_VERSION,
        "profile_filter_version": REPOST_PROFILE_FILTER_VERSION,
        "verification_mode": _normalize_verification_mode(verification_mode),
        "min_confidence": round(float(min_confidence), 6),
        "vod_cache_root": str(vod_cache_root),
        "transcript_match_cache_version": getattr(transcript_matcher, "_SHORT_TRANSCRIPT_CACHE_VERSION", 0),
        "vod_transcript_cache_version": getattr(build_twitch_training_dataset, "_VOD_FEATURE_CACHE_VERSION", 0),
        "vod_rows": _rows_for_match_signature(vod_rows),
        "repost_rows": _rows_for_match_signature(repost_rows),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rows_for_match_signature(rows: list[dict]) -> list[dict[str, object]]:
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        normalized_rows.append(
            {
                "row": row,
                "media": _row_media_state(row),
            }
        )
    return sorted(
        normalized_rows,
        key=lambda item: json.dumps(item.get("row") or {}, ensure_ascii=True, sort_keys=True, default=str),
    )


def _row_media_state(row: dict) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    seen: set[str] = set()
    for key in ("video_path", "fingerprint_path"):
        raw_path = str(row.get(key) or "").strip()
        if not raw_path or raw_path in seen:
            continue
        seen.add(raw_path)
        path = pathlib.Path(raw_path)
        try:
            stat = path.stat()
        except OSError:
            states.append({"key": key, "path": raw_path, "exists": False})
            continue
        states.append(
            {
                "key": key,
                "path": raw_path,
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return states


def _can_reuse_previous_match_result(
    *,
    previous_summary: dict,
    match_input_signature: str,
    output_path: pathlib.Path,
    diagnostic_output_path: pathlib.Path,
) -> bool:
    if not previous_summary:
        return False
    if str(previous_summary.get("match_input_signature") or "") != match_input_signature:
        return False
    if not output_path.exists() or not diagnostic_output_path.exists():
        return False
    status = str(previous_summary.get("status") or "").strip()
    if not status:
        return False
    return status in {
        "matched",
        "no_matches_below_threshold",
        "insufficient_matched_rows",
        "insufficient_matched_vods",
        "insufficient_matched_rows_and_matched_vods",
    }


def _normalize_verification_mode(value: str) -> str:
    mode = str(value or "transcript").strip().lower().replace("-", "_")
    if mode in {"audio", "fingerprint", "audio_fingerprint"}:
        return "audio_fingerprint"
    return "transcript"


def _stable_time(value) -> str:
    try:
        return f"{float(value or 0.0):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _short_error(exc: BaseException, *, limit: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return _short_text(text, limit)


def _short_text(text: object, limit: int = 240) -> str:
    value = str(text or "")
    return value if len(value) <= limit else f"{value[:limit - 3]}..."


def _decode_process_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _exception_log_fields(exc: BaseException) -> dict[str, object]:
    fields: dict[str, object] = {"error_type": type(exc).__name__}
    if isinstance(exc, FullVodAudioDownloadError):
        fields["ffmpeg_returncode"] = exc.returncode
        fields["ffmpeg_stderr"] = _short_text(exc.stderr, 2000)
        fields["ffmpeg_stdout"] = _short_text(exc.stdout, 1000)
        fields["ffmpeg_format"] = exc.format_summary
    return fields


def _is_permanent_vod_download_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    permanent_markers = (
        " does not exist",
        "video is unavailable",
        "not available",
        "has been deleted",
        "this video is no longer available",
    )
    return (
        ("twitch:vod" in text or "twitch.tv/videos/" in text)
        and any(marker in text for marker in permanent_markers)
    )


def _is_youtube_rate_limited_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "youtube" in text and any(marker in text for marker in YOUTUBE_RATE_LIMIT_ERROR_MARKERS)


def _short_form_rate_limit_cooldown_sec() -> float:
    return _short_form_sleep_range_sample()


def _short_form_sleep_range_sample() -> float:
    try:
        min_sleep = float(getattr(settings, "yt_dlp_short_form_min_sleep_interval_sec", 5.0) or 5.0)
    except (TypeError, ValueError):
        min_sleep = 5.0
    try:
        max_sleep = float(getattr(settings, "yt_dlp_short_form_max_sleep_interval_sec", 10.0) or 10.0)
    except (TypeError, ValueError):
        max_sleep = 10.0
    min_sleep = max(min_sleep, 0.0)
    max_sleep = max(max_sleep, min_sleep)
    if max_sleep <= min_sleep:
        return min_sleep
    return random.uniform(min_sleep, max_sleep)


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    _write_text_if_changed(path, text)


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_if_changed(path, json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _write_summary(path: pathlib.Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_if_changed(path, json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def _write_text_if_changed(path: pathlib.Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
