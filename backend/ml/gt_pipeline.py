#!/usr/bin/env python3
"""
Jynxzi Ground-Truth Pipeline
==============================
Reads backend/ml/jynxzi_gt_vods.csv, discovers @Iboclipss clips for each VOD's
date window, downloads them, transcribes the VODs, runs transcript+audio matching,
and produces jynxzi_eval_manifest.jsonl as ground truth.

State is saved after every step — interrupted runs resume from where they left off.

Run via the shell wrapper (from repo root):
  ./scripts/run_gt_pipeline.sh

Or directly inside the container:
  docker exec -w /app/backend ai-clipping-backend python ml/gt_pipeline.py

With a custom CSV:
  docker exec -w /app/backend ai-clipping-backend python ml/gt_pipeline.py --vods ml/my_vods.csv

Force restart from scratch:
  ./scripts/run_gt_pipeline.sh --reset
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import shutil
import sys
import time
from datetime import datetime

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_LOG_FMT = "%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
logger = logging.getLogger("gt_pipeline")


def _attach_file_logger(log_path: pathlib.Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    logging.getLogger().addHandler(fh)
    logger.info("Log file: %s", log_path)


from ml import match_cross_platform_transcripts as _matcher  # noqa: E402
from ml.collect_twitch_clip_labels import TwitchClient  # noqa: E402
from ml.discover_iboclipss_clips import (  # noqa: E402
    _build_date_windows,
    _list_channel_entries,
    _enrich_with_upload_dates,
    _filter_to_windows,
)
from ml.build_gt_vod_transcripts import _process_vod  # noqa: E402
from ml.build_eval_manifest import dedup_moments  # noqa: E402
from ml.gt_match_verify import filter_gold_candidates  # noqa: E402
from ml.row_identity import read_jsonl, write_jsonl  # noqa: E402

_DEFAULT_VOD_CSV = BACKEND_ROOT / "ml" / "jynxzi_gt_vods.csv"
_DEFAULT_CHANNEL = "https://www.youtube.com/@Iboclipss/shorts"
_DEFAULT_WINDOW_DAYS = 2
_DEFAULT_MAX_CLIPS = 250
_DEFAULT_MIN_SIMILARITY = 0.58
_DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"
_DEFAULT_RUNS_ROOT = BACKEND_ROOT / "tmp" / "gt_runs"


# ══════════════════════════════════════════════════════════════════════════════
# CSV
# ══════════════════════════════════════════════════════════════════════════════

def _read_vod_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        logger.error("VOD CSV not found: %s", path)
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        non_comment = (line for line in f if not line.strip().startswith("#"))
        reader = csv.DictReader(non_comment)
        for row in reader:
            vod_id = (row.get("vod_id") or "").strip()
            if not vod_id:
                continue
            rows.append({
                "vod_id": vod_id,
                "vod_url": f"https://www.twitch.tv/videos/{vod_id}",
                "vod_date": (row.get("vod_date") or "").strip(),  # may be empty → resolved via Twitch API
                "note": (row.get("note") or "").strip(),
            })
    return rows


def _resolve_vod_dates(vod_rows: list[dict], state: dict) -> list[dict]:
    """Fill in missing vod_date fields by querying the Twitch /videos API."""
    missing = [r for r in vod_rows if not r.get("vod_date")]
    if not missing:
        return vod_rows

    cached: dict = state.setdefault("vod_dates", {})
    still_missing = [r for r in missing if r["vod_id"] not in cached]

    if still_missing:
        from config import settings as _settings
        if not _settings.twitch_client_id or not _settings.twitch_client_secret:
            raise RuntimeError(
                "vod_date missing for some VODs and Twitch credentials not set. "
                "Either add vod_date to the CSV or set TWITCH_CLIENT_ID/SECRET in backend/.env."
            )
        client = TwitchClient(_settings.twitch_client_id, _settings.twitch_client_secret)
        logger.info("Resolving vod_date for %d VODs via Twitch API...", len(still_missing))
        for row in still_missing:
            vod_id = row["vod_id"]
            try:
                data = client.get_json("/videos", id=vod_id)
                created_at = (data.get("data") or [{}])[0].get("created_at", "")
                if created_at:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    cached[vod_id] = dt.strftime("%Y-%m-%d")
                    logger.info("  VOD %s → %s", vod_id, cached[vod_id])
                else:
                    logger.warning("  VOD %s: no created_at in Twitch response — check VOD ID", vod_id)
            except Exception:
                logger.warning("  VOD %s: Twitch API lookup failed", vod_id, exc_info=True)

    resolved = []
    for r in vod_rows:
        if not r.get("vod_date"):
            date_str = cached.get(r["vod_id"], "")
            if not date_str:
                logger.warning("VOD %s: could not resolve vod_date — skipping", r["vod_id"])
                continue
            r = {**r, "vod_date": date_str}
        resolved.append(r)
    return resolved


# ══════════════════════════════════════════════════════════════════════════════
# State helpers
# ══════════════════════════════════════════════════════════════════════════════

def _state_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "state.json"


def _load_state(run_dir: pathlib.Path) -> dict:
    p = _state_path(run_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read state file (%s) — starting fresh", exc)
        return {}


def _save_state(run_dir: pathlib.Path, state: dict) -> None:
    p = _state_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    tmp.replace(p)


def _is_done(state: dict, step: str) -> bool:
    return bool(state.get("steps_done", {}).get(step))


def _mark_done(state: dict, step: str) -> None:
    state.setdefault("steps_done", {})[step] = True


def _inputs_complete(state: dict) -> bool:
    """Matching/eval_manifest may only be cached once every repost is downloaded
    and every VOD transcribed — otherwise the next run must re-match with the
    data that arrived in between."""
    return _is_done(state, "iboclipss_downloaded") and _is_done(state, "vod_transcripts_all")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — List @Iboclipss clips within VOD date windows
# ══════════════════════════════════════════════════════════════════════════════

def _channel_cache_path(runs_root: pathlib.Path) -> pathlib.Path:
    return runs_root / "_iboclipss_channel_cache.jsonl"


def _load_channel_cache(runs_root: pathlib.Path) -> list[dict] | None:
    """Return cached enriched channel entries (permanent — pass --reset to re-fetch)."""
    p = _channel_cache_path(runs_root)
    if not p.exists():
        return None
    entries = read_jsonl(p, strict=False)
    if not entries:
        return None
    age_h = (time.time() - p.stat().st_mtime) / 3600
    logger.info("      Channel cache HIT: %d entries (%.0fh old)", len(entries), age_h)
    return entries


def _save_channel_cache(runs_root: pathlib.Path, entries: list[dict]) -> None:
    write_jsonl(_channel_cache_path(runs_root), entries)


def step_list_iboclipss(
    run_dir: pathlib.Path,
    state: dict,
    *,
    vod_rows: list[dict],
    window_days: int,
    max_clips: int,
    channel: str,
    runs_root: pathlib.Path,
) -> list[dict]:
    discovered_path = run_dir / "iboclipss_discovered.jsonl"
    if _is_done(state, "iboclipss_listed") and discovered_path.exists():
        rows = read_jsonl(discovered_path, strict=False)
        logger.info("[1/5] iboclipss_listed [CACHED]: %d clips", len(rows))
        for r in rows:
            logger.info("      %s  %5dk views  %s  → VODs %s",
                        r.get("upload_date", "?"), int(r.get("view_count", 0)) // 1000,
                        r.get("title", "")[:55], r.get("candidate_vod_ids", []))
        return rows

    date_windows = _build_date_windows(vod_rows, window_days=window_days)
    logger.info("[1/5] iboclipss_listed: %d date windows across %d VODs", len(date_windows), len(vod_rows))

    # Channel listing is cached globally across runs.
    entries = _load_channel_cache(runs_root)
    if entries is None:
        logger.info("      Fetching channel %s  max=%d  window_days=%d", channel, max_clips, window_days)
        entries = _list_channel_entries(channel, max_clips)
        if not entries:
            raise RuntimeError(
                f"No entries returned from channel {channel}. "
                "Check that YTDLP_COOKIEFILE is set and the channel URL is correct."
            )
        logger.info("      Fetched %d raw channel entries — saving before enrichment", len(entries))
        _save_channel_cache(runs_root, entries)  # crash-safe: persist raw listing now

    # Enrich any entries that still lack upload dates (no-op if cache is already complete;
    # resumes partial work if a previous run was killed mid-enrichment).
    needs_dates = sum(
        1 for e in entries
        if not e.get("upload_date") and not e.get("timestamp")
    )
    if needs_dates:
        logger.info("      Enriching dates for %d/%d entries...", needs_dates, len(entries))
        entries = _enrich_with_upload_dates(entries, channel)
        _save_channel_cache(runs_root, entries)
        logger.info("      Channel cache updated (%d entries)", len(entries))

    matched = _filter_to_windows(entries, date_windows)

    if not matched:
        logger.warning(
            "      0 clips matched any VOD date window. "
            "Verify that vod_dates are correct and Iboclipss posted within %d days of those VODs.",
            window_days,
        )
    else:
        for row in matched:
            logger.info("      %s  %5dk views  %s  → VODs %s",
                        row.get("upload_date", "?"), int(row.get("view_count", 0)) // 1000,
                        row.get("title", "")[:55], row.get("candidate_vod_ids", []))

    write_jsonl(discovered_path, matched)
    _mark_done(state, "iboclipss_listed")
    logger.info("[1/5] iboclipss_listed DONE: %d clips → %s", len(matched), discovered_path.name)
    return matched


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Download Iboclipss clips
# ══════════════════════════════════════════════════════════════════════════════

def step_download_iboclipss(
    run_dir: pathlib.Path,
    state: dict,
    *,
    discovered: list[dict],
) -> list[dict]:
    from services.downloader import ensure_cached_download

    reposts_path = run_dir / "iboclipss_reposts.jsonl"
    existing = read_jsonl(reposts_path, strict=False) if reposts_path.exists() else []
    done_ids = {str(r.get("clip_id") or "") for r in existing if r.get("clip_id")}
    pending = [d for d in discovered if str(d.get("clip_id") or "") not in done_ids]

    if not pending:
        logger.info("[2/5] iboclipss_downloaded [CACHED]: all %d clips already downloaded", len(existing))
        return existing

    logger.info("[2/5] iboclipss_downloaded: %d pending / %d total", len(pending), len(discovered))
    if existing:
        logger.info("      Resuming — %d done, %d still pending", len(existing), len(pending))

    new_reposts: list[dict] = []
    for i, row in enumerate(pending, 1):
        url = (row.get("source_url") or "").strip()
        clip_id = row.get("clip_id") or f"clip_{i}"
        if not url:
            logger.warning("      [%d/%d] SKIP %s: no source_url", i, len(pending), clip_id)
            continue

        logger.info("      [%d/%d] %s  (%s)...", i, len(pending), clip_id, url)
        t0 = time.monotonic()
        try:
            video_path = ensure_cached_download(url, cache_namespace="short_form", cache_ttl_sec=-1)
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = "rate-limit" in exc_str or "rate_limit" in exc_str or "isn't available" in exc_str
            is_forbidden = "403" in exc_str or "forbidden" in exc_str
            if is_rate_limit:
                logger.warning(
                    "      [%d/%d] RATE-LIMITED on %s — sleeping 90s before continuing",
                    i, len(pending), clip_id,
                )
                time.sleep(90)
            elif is_forbidden:
                logger.warning(
                    "      [%d/%d] SKIP %s — HTTP 403 (age restriction or unavailable)",
                    i, len(pending), clip_id,
                )
            else:
                logger.warning("      [%d/%d] FAILED %s", i, len(pending), url, exc_info=True)
            continue

        elapsed = time.monotonic() - t0
        size_mb = video_path.stat().st_size / 1_048_576 if video_path.exists() else 0.0
        logger.info("      [%d/%d] OK  %.1fs  %.1f MB  → %s", i, len(pending), elapsed, size_mb, video_path.name)
        # Small inter-clip pause to stay under YouTube's rate limit — but only
        # after a real network download. Cache hits return in well under a
        # second and need no backoff (saves minutes on resume runs).
        if i < len(pending) and elapsed > 1.0:
            time.sleep(3)

        new_reposts.append({
            "source_url": url,
            "video_path": str(video_path),
            "platform": "youtube",
            "views": row.get("view_count", 0),
            "title": row.get("title", ""),
            "clip_id": clip_id,
            "upload_date": row.get("upload_date", ""),
            "candidate_vod_ids": row.get("candidate_vod_ids", []),
        })
        write_jsonl(reposts_path, existing + new_reposts)

    all_reposts = existing + new_reposts
    write_jsonl(reposts_path, all_reposts)

    if len(all_reposts) >= len(discovered):
        _mark_done(state, "iboclipss_downloaded")
    else:
        logger.warning(
            "      %d/%d clips downloaded — will retry remaining on next run",
            len(all_reposts), len(discovered),
        )
    logger.info("[2/5] iboclipss_downloaded: %d/%d → %s",
                len(all_reposts), len(discovered), reposts_path.name)
    return all_reposts


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Download VODs, transcribe, pre-compute fingerprints
# ══════════════════════════════════════════════════════════════════════════════

def step_build_vod_transcripts(
    run_dir: pathlib.Path,
    state: dict,
    *,
    vod_rows: list[dict],
    vod_cache_root: pathlib.Path,
) -> list[dict]:
    vod_manifest_path = run_dir / "vod_manifest.jsonl"
    fpcalc = shutil.which("fpcalc")

    existing_manifest = read_jsonl(vod_manifest_path, strict=False) if vod_manifest_path.exists() else []
    done_vod_ids = {str(r.get("vod_id") or "") for r in existing_manifest}
    per_vod_state: dict = state.setdefault("steps_done", {}).setdefault("vod_transcripts", {})

    pending = [v for v in vod_rows if str(v.get("vod_id") or "") not in done_vod_ids]
    if not pending:
        logger.info("[3/5] vod_transcripts [CACHED]: all %d VODs already transcribed", len(existing_manifest))
        return existing_manifest

    logger.info("[3/5] vod_transcripts: %d pending / %d total", len(pending), len(vod_rows))
    if existing_manifest:
        logger.info("      Resuming — %d done, %d still pending", len(existing_manifest), len(pending))
    logger.info("      fpcalc: %s", fpcalc or "NOT FOUND — fingerprints disabled, transcript-only matching")
    logger.info("      NOTE: slow step (~1h per VOD on GPU). Ctrl+C is safe — resumes here next run.")

    all_manifest = list(existing_manifest)
    all_ok = True
    for i, vod in enumerate(pending, 1):
        vod_id = str(vod.get("vod_id") or "").strip()
        vod_url = vod.get("vod_url") or f"https://www.twitch.tv/videos/{vod_id}"
        cache_dir = vod_cache_root / vod_id
        transcript_path = cache_dir / "transcript.json"
        fingerprint_path = cache_dir / "fingerprint.json"
        label = vod.get("note") or vod_id

        if transcript_path.exists():
            logger.info("      [%d/%d] VOD %s: transcript cache exists [CACHED]", i, len(pending), vod_id)
        else:
            logger.info("      [%d/%d] VOD %s: downloading + transcribing  (%s)...",
                        i, len(pending), vod_id, label)
            t0 = time.monotonic()
            try:
                _process_vod(vod_id, vod_url, cache_dir, transcript_path, fpcalc)
            except Exception:
                logger.exception("      [%d/%d] VOD %s: FAILED — skipping, will retry next run",
                                 i, len(pending), vod_id)
                all_ok = False
                continue
            elapsed = time.monotonic() - t0
            logger.info("      [%d/%d] VOD %s: done in %.0fs (%.1f min)",
                        i, len(pending), vod_id, elapsed, elapsed / 60)

        per_vod_state[vod_id] = True
        manifest_row: dict = {"vod_id": vod_id, "vod_url": vod_url}
        if fingerprint_path.exists():
            manifest_row["fingerprint_path"] = str(fingerprint_path)
        all_manifest.append(manifest_row)
        done_vod_ids.add(vod_id)

        write_jsonl(vod_manifest_path, all_manifest)
        _save_state(run_dir, state)
        logger.info("      [%d/%d] VOD %s: checkpoint saved", i, len(pending), vod_id)

    if all_ok:
        _mark_done(state, "vod_transcripts_all")
    logger.info("[3/5] vod_transcripts: %d/%d done → %s",
                len(all_manifest), len(vod_rows), vod_manifest_path.name)
    return all_manifest


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Run transcript + audio matcher
# ══════════════════════════════════════════════════════════════════════════════

def _default_match_workers() -> int:
    """CPU scoring workers for the matcher. Repost transcription and the audio
    fallback run in the parent process, so workers never touch the GPU."""
    import os
    return max(1, min(16, (os.cpu_count() or 4) - 2))


def step_run_matcher(
    run_dir: pathlib.Path,
    state: dict,
    *,
    vod_manifest: list[dict],
    reposts: list[dict],
    min_similarity: float,
    vod_cache_root: pathlib.Path,
    match_workers: int = 0,
) -> list[dict]:
    matches_path = run_dir / "matches.jsonl"
    if _is_done(state, "matching") and matches_path.exists():
        rows = read_jsonl(matches_path, strict=False)
        logger.info("[4/5] matching [CACHED]: %d accepted matches", len(rows))
        _log_match_summary(rows)
        return rows

    # Log how many repost→VOD pairs will actually be compared after candidate_vod_ids filtering.
    total_pairs = len(vod_manifest) * len(reposts)
    filtered_pairs = sum(
        len(r.get("candidate_vod_ids") or []) or len(vod_manifest) for r in reposts
    )
    logger.info(
        "[4/5] matching: %d VODs × %d reposts  min_similarity=%.2f  "
        "pairs: %d → %d after date-window filter (%.0f%% reduction)",
        len(vod_manifest), len(reposts), min_similarity,
        total_pairs, filtered_pairs,
        100 * (1 - filtered_pairs / total_pairs) if total_pairs else 0,
    )
    t0 = time.monotonic()

    workers = int(match_workers) if int(match_workers) > 0 else _default_match_workers()
    output_rows, diag = _matcher.match_reposts_to_vods_with_diagnostics(
        vod_manifest,
        reposts,
        min_similarity=min_similarity,
        vod_cache_root=vod_cache_root,
        log_progress=True,
        transcript_match_workers=workers,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "[4/5] matching DONE in %.0fs:  accepted=%d  audio_fallback=%d/%d  "
        "skip_vod_transcript=%d  skip_repost_media=%d",
        elapsed,
        len(output_rows),
        int(diag.get("audio_fallback_matches", 0)),
        int(diag.get("audio_fallback_attempts", 0)),
        int(diag.get("skipped_vod_transcript_rows", 0)),
        int(diag.get("skipped_repost_media_rows", 0)),
    )

    if diag.get("recommendation"):
        rec = diag["recommendation"]
        logger.info("      Matcher recommendation: action=%s  %s",
                    rec.get("action", ""), rec.get("message", ""))

    _write_diagnostics(run_dir / "match_diagnostics.json", diag)
    write_jsonl(matches_path, output_rows)
    if _inputs_complete(state):
        _mark_done(state, "matching")
    else:
        logger.warning("      matching not cached — downloads/transcripts incomplete, will re-match next run")
    _log_match_summary(output_rows)
    return output_rows


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Build eval manifest
# ══════════════════════════════════════════════════════════════════════════════

def step_build_eval_manifest(
    run_dir: pathlib.Path,
    state: dict,
    *,
    matches: list[dict],
    min_similarity: float,
) -> None:
    manifest_path = run_dir / "eval_manifest.jsonl"
    if _is_done(state, "eval_manifest") and manifest_path.exists():
        rows = read_jsonl(manifest_path, strict=False)
        logger.info("[5/5] eval_manifest [CACHED]: %s", manifest_path.name)
        _log_eval_summary(rows)
        return

    filtered = filter_gold_candidates(matches, min_similarity=min_similarity)
    logger.info("[5/5] eval_manifest: %d/%d matches pass the gold filter (audio fallback or sim >= %.2f)",
                len(filtered), len(matches), min_similarity)

    by_vod: dict[str, list[dict]] = {}
    for row in filtered:
        vod_id = str(row.get("vod_id") or "").strip()
        if vod_id:
            by_vod.setdefault(vod_id, []).append(row)

    output_rows: list[dict] = []
    total_moments = 0
    for vod_id, vod_rows in sorted(by_vod.items()):
        gt_moments = dedup_moments(vod_rows)
        total_moments += len(gt_moments)
        output_rows.append({"vod_id": vod_id, "gt_moments": gt_moments})
        logger.info("      VOD %s: %d raw → %d moments after dedup", vod_id, len(vod_rows), len(gt_moments))

    write_jsonl(manifest_path, output_rows)
    if _inputs_complete(state):
        _mark_done(state, "eval_manifest")
    logger.info("[5/5] eval_manifest DONE: %d VODs  %d gt_moments → %s",
                len(output_rows), total_moments, manifest_path.name)

    # MERGE into the canonical manifest instead of overwriting it: the canonical
    # file can contain VODs from other runs/sources (e.g. the training-positive
    # expansion) and manually curated windows (manual_edit) that a blind copy
    # would silently destroy.
    canonical = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_eval_manifest.jsonl"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = read_jsonl(canonical, strict=False) if canonical.exists() else []
    if existing_rows:
        backup = canonical.with_name(
            f"jynxzi_eval_manifest.pre_{datetime.now().strftime('%Y%m%dT%H%M%S')}.bak.jsonl"
        )
        shutil.copy2(str(canonical), str(backup))
        logger.info("      Backup of canonical manifest → %s", backup.name)
    merged: dict[str, dict] = {str(r.get("vod_id")): r for r in existing_rows if r.get("vod_id")}
    replaced = 0
    for row in output_rows:
        vod_id = str(row.get("vod_id"))
        previous = merged.get(vod_id)
        if previous and any(m.get("manual_edit") for m in previous.get("gt_moments") or []):
            logger.warning(
                "      VOD %s: keeping existing manually curated gt_moments — "
                "run output for this VOD is in %s, merge by hand if wanted.",
                vod_id, manifest_path.name,
            )
            continue
        if previous:
            _carry_over_verification(previous.get("gt_moments") or [], row.get("gt_moments") or [])
        merged[vod_id] = row
        replaced += 1
    merged_rows = [merged[k] for k in sorted(merged)]
    write_jsonl(canonical, merged_rows)
    logger.info(
        "      Merged → %s (%d VODs total, %d written by this run, %d untouched)",
        canonical.name, len(merged_rows), replaced, len(merged_rows) - replaced,
    )

    _log_eval_summary(output_rows)


def _carry_over_verification(previous_moments: list[dict], new_moments: list[dict]) -> None:
    """Preserve ml.verify_gt_matches results when a re-run replaces a VOD row.

    A new run re-derives transcript windows, so verified tiers and
    audio-refined boundaries would silently reset to unverified. Carry them
    over when the new moment is the same repost (same source_url) at an
    overlapping position; everything else stays unverified and goes through
    verification again.
    """
    by_url: dict[str, list[dict]] = {}
    for moment in previous_moments:
        url = str(moment.get("source_url") or "")
        if url and moment.get("gt_tier"):
            by_url.setdefault(url, []).append(moment)
    for moment in new_moments:
        candidates = by_url.get(str(moment.get("source_url") or ""), [])
        for previous in candidates:
            prev_start = float(previous.get("transcript_start") or previous.get("start") or 0.0)
            prev_end = float(previous.get("transcript_end") or previous.get("end") or 0.0)
            start = float(moment.get("start") or 0.0)
            end = float(moment.get("end") or 0.0)
            overlap = min(end, prev_end) - max(start, prev_start)
            if overlap <= 0:
                continue
            moment["gt_tier"] = previous["gt_tier"]
            if previous.get("boundary_source") == "audio_refined":
                moment["transcript_start"] = start
                moment["transcript_end"] = end
                moment["start"] = float(previous["start"])
                moment["end"] = float(previous["end"])
                moment["boundary_source"] = "audio_refined"
            break


# ══════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ══════════════════════════════════════════════════════════════════════════════

def _log_match_summary(matches: list[dict]) -> None:
    if not matches:
        logger.warning("      No accepted matches.")
        return
    by_vod: dict[str, list[dict]] = {}
    for m in matches:
        by_vod.setdefault(str(m.get("vod_id") or ""), []).append(m)
    for vod_id, rows in sorted(by_vod.items()):
        logger.info("      VOD %s: %d matches", vod_id, len(rows))
        for m in sorted(rows, key=lambda x: float(x.get("window_start") or 0)):
            logger.info(
                "        t=%6.0f–%6.0f  sim=%.3f  %-22s  %5dk  %s",
                float(m.get("window_start") or 0),
                float(m.get("window_end") or 0),
                float(m.get("transcript_similarity") or 0),
                str(m.get("transcript_match_kind") or "")[:22],
                int(float(m.get("views") or 0)) // 1000,
                str(m.get("repost_title") or "")[:50],
            )


def _log_eval_summary(rows: list[dict]) -> None:
    total = sum(len(r.get("gt_moments") or []) for r in rows)
    logger.info("      ┌─ Eval manifest: %d VODs  %d gt_moments", len(rows), total)
    for row in rows:
        moments = row.get("gt_moments") or []
        logger.info("      │  VOD %-12s  %d moments", row.get("vod_id", "?"), len(moments))
        for m in moments:
            logger.info(
                "      │    t=%6.0f–%6.0f  sim=%.3f  %5dk  %s",
                float(m.get("start") or 0),
                float(m.get("end") or 0),
                float(m.get("match_similarity") or 0),
                int(float(m.get("views") or 0)) // 1000,
                str(m.get("desc") or "")[:50],
            )
    logger.info("      └─ done")


def _write_diagnostics(path: pathlib.Path, diag: dict) -> None:
    serializable = {}
    for k, v in diag.items():
        if k in ("best_match_rows", "top_best_matches"):
            continue
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vods", default=str(_DEFAULT_VOD_CSV),
        help=f"Path to VOD CSV (default: {_DEFAULT_VOD_CSV.relative_to(BACKEND_ROOT)})",
    )
    parser.add_argument("--channel", default=_DEFAULT_CHANNEL,
                        help="YouTube channel URL for Iboclipss clips")
    parser.add_argument("--window-days", type=int, default=_DEFAULT_WINDOW_DAYS,
                        help=f"Days after each VOD date to look for clips (default: {_DEFAULT_WINDOW_DAYS})")
    parser.add_argument("--max-clips", type=int, default=_DEFAULT_MAX_CLIPS,
                        help=f"Max clips to fetch from channel listing (default: {_DEFAULT_MAX_CLIPS})")
    parser.add_argument("--min-similarity", type=float, default=_DEFAULT_MIN_SIMILARITY,
                        help=f"Transcript similarity threshold (default: {_DEFAULT_MIN_SIMILARITY})")
    parser.add_argument("--match-workers", type=int, default=0,
                        help="Parallel CPU workers for repost matching (0 = auto: min(8, cores-2)).")
    parser.add_argument("--vod-cache-root", default=str(_DEFAULT_VOD_CACHE_ROOT))
    parser.add_argument("--runs-root", default=str(_DEFAULT_RUNS_ROOT))
    parser.add_argument("--reset", action="store_true",
                        help="Ignore existing state and restart from scratch")
    args = parser.parse_args()

    vod_csv = pathlib.Path(args.vods)
    vod_rows = _read_vod_csv(vod_csv)
    if not vod_rows:
        logger.error("No VODs in %s — add entries and re-run.", vod_csv)
        return 1

    vod_cache_root = pathlib.Path(args.vod_cache_root)

    # Run ID is stable as long as the same set of VOD IDs is in the CSV.
    vod_ids_key = "_".join(sorted(r["vod_id"] for r in vod_rows))
    if len(vod_ids_key) > 80:
        import hashlib
        vod_ids_key = hashlib.sha1(vod_ids_key.encode()).hexdigest()[:12]
    run_id = f"gt_{vod_ids_key}"
    run_dir = pathlib.Path(args.runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _attach_file_logger(run_dir / "pipeline.log")

    # Load state early so Twitch date lookups are cached across runs.
    state: dict = {}
    if not args.reset:
        state = _load_state(run_dir)
        done_steps = [k for k, v in state.get("steps_done", {}).items() if v is True]
        if done_steps:
            logger.info("Resuming — steps already done: %s", done_steps)
    else:
        logger.info("--reset: ignoring existing state")

    # Resolve missing vod_dates via Twitch API (results cached in state["vod_dates"]).
    try:
        vod_rows = _resolve_vod_dates(vod_rows, state)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    if not vod_rows:
        logger.error("No VODs with a resolvable vod_date — cannot build date windows.")
        return 1
    _save_state(run_dir, state)

    logger.info("=" * 64)
    logger.info("Jynxzi Ground-Truth Pipeline  run_id=%s", run_id)
    logger.info("  vods_csv       : %s  (%d VODs)", vod_csv.name, len(vod_rows))
    for r in vod_rows:
        logger.info("    VOD %-12s  %s  %s", r["vod_id"], r["vod_date"], r.get("note", ""))
    logger.info("  channel        : %s", args.channel)
    logger.info("  window_days    : %d", args.window_days)
    logger.info("  max_clips      : %d", args.max_clips)
    logger.info("  min_similarity : %.2f", args.min_similarity)
    logger.info("  run_dir        : %s", run_dir)
    logger.info("=" * 64)

    t_pipeline = time.monotonic()

    try:
        discovered = step_list_iboclipss(
            run_dir, state, vod_rows=vod_rows, window_days=args.window_days,
            max_clips=args.max_clips, channel=args.channel,
            runs_root=pathlib.Path(args.runs_root),
        )
        _save_state(run_dir, state)
    except Exception:
        logger.exception("Step 1 (iboclipss_listed) FAILED")
        return 1
    if not discovered:
        logger.error(
            "No Iboclipss clips found for %d VOD date windows (window_days=%d). "
            "Check vod_dates in %s or increase --max-clips.",
            len(vod_rows), args.window_days, vod_csv.name,
        )
        return 1

    try:
        reposts = step_download_iboclipss(run_dir, state, discovered=discovered)
        _save_state(run_dir, state)
    except Exception:
        logger.exception("Step 2 (iboclipss_downloaded) FAILED")
        return 1
    if not reposts:
        logger.error("No clips downloaded. Check YTDLP_COOKIEFILE in backend/.env.")
        return 1

    try:
        vod_manifest = step_build_vod_transcripts(
            run_dir, state, vod_rows=vod_rows, vod_cache_root=vod_cache_root,
        )
        _save_state(run_dir, state)
    except Exception:
        logger.exception("Step 3 (vod_transcripts) FAILED")
        return 1
    if not vod_manifest:
        logger.error("No VOD transcripts built. Check Twitch download config and VOD availability.")
        return 1

    try:
        matches = step_run_matcher(
            run_dir, state, vod_manifest=vod_manifest, reposts=reposts,
            min_similarity=args.min_similarity, vod_cache_root=vod_cache_root,
            match_workers=args.match_workers,
        )
        _save_state(run_dir, state)
    except Exception:
        logger.exception("Step 4 (matching) FAILED")
        return 1

    try:
        step_build_eval_manifest(run_dir, state, matches=matches, min_similarity=args.min_similarity)
        _save_state(run_dir, state)
    except Exception:
        logger.exception("Step 5 (eval_manifest) FAILED")
        return 1

    total_elapsed = time.monotonic() - t_pipeline
    eval_rows = read_jsonl(run_dir / "eval_manifest.jsonl", strict=False)
    total_moments = sum(len(r.get("gt_moments") or []) for r in eval_rows)
    logger.info("=" * 64)
    logger.info("Pipeline complete in %.0fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("  iboclipss_discovered : %d clips", len(discovered))
    logger.info("  iboclipss_reposts    : %d downloaded", len(reposts))
    logger.info("  vod_manifest         : %d transcribed", len(vod_manifest))
    logger.info("  matches              : %d accepted", len(matches))
    logger.info("  eval_manifest        : %d gt_moments across %d VODs", total_moments, len(eval_rows))
    logger.info("  Run dir              : %s", run_dir)
    logger.info("  Log                  : %s", run_dir / "pipeline.log")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
