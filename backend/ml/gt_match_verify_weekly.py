"""GT-flywheel weekly match+verify job — the gold-minting wrapper (issue #8 / PRD #3).

The lag-tolerant, weekly half of the flywheel. It closes the loop the daily
capture job (:mod:`ml.gt_capture_daily` / #6) opened:

  1. Re-scan @Iboclipss for shorts, reusing the existing yt-dlp cookie-auth and
     request-pacing wiring (:func:`services.yt_dlp_auth.apply_yt_dlp_auth_options`
     and ``services.downloader._apply_short_form_sleep_opts``) so bot-protection
     ("try again later") cannot *silently zero out* discovery — a zeroed scan is
     surfaced and refused, never read as "0 new reposts".
  2. Diff against the already-matched set via a **pure new-repost-selection**
     function (mirrors :func:`ml.gt_capture_daily.select_new_vods`) so the whole
     @Iboclipss back-catalogue isn't re-matched every week.
  3. Run the match+verify core (:func:`ml.gt_match_verify.match_and_verify` / #7)
     over the VODs captured by the daily job, auto-minting gold/silver/review.
  4. Merge the new verdicts into the accumulating manifest and **rebuild the
     gold-filtered eval manifest end-to-end**, so the finder eval always scores
     against the freshest confirmed gold.
  5. Emit a compact summary: new VODs captured, new reposts matched,
     gold/silver/review counts, and gold-accrual vs the target (~150–200 gold
     over ≥25 VODs).

State (already-matched reposts + VODs seen at the last run) is persisted so the
job is idempotent and resumable. Scheduling lives outside this module (a weekly
cron — see ``scripts/run_gt_match_verify_weekly.sh``), since reposts lag the VOD
by days. Heavy collaborators (the matcher, which pulls Whisper) are imported
lazily inside the core, so the pure decision seams stay testable without the ML
stack or network.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from datetime import datetime, timedelta, timezone

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import gt_capture_daily
from ml import gt_match_verify
from ml.row_identity import read_jsonl, write_jsonl  # noqa: E402

logger = logging.getLogger(__name__)

STATE_FILENAME = "match_verify_state.json"
MANIFEST_FILENAME = "gt_matched_manifest.jsonl"
GOLD_MANIFEST_FILENAME = "gt_gold_eval_manifest.jsonl"
CAPTURE_STATE_FILENAME = gt_capture_daily.STATE_FILENAME

DEFAULT_CHANNEL = "https://www.youtube.com/@Iboclipss/shorts"
DEFAULT_MAX_CANDIDATES = 250

GOLD = gt_match_verify.GOLD
SILVER = gt_match_verify.SILVER

# Target that makes "grow the GT" a finite job, not a forever-project: ~150–200
# confirmed-gold moments over ≥25 VODs (PRD #3 success metric). We track against
# the midpoint so accrual reads as "% of the way there".
TARGET_GOLD = 175
TARGET_VODS = 25

# A repost without a match row stays open (re-matched against newly captured
# VODs each run) until it is older than this, then it is done for good.
OPEN_REPOST_DAYS = 14


# ---------------------------------------------------------------------------
# Pure decision seam: new-repost selection
# ---------------------------------------------------------------------------
def _repost_id(repost) -> str:
    """Normalize a repost record (or bare id) to a stable string id.

    Prefers the discovery ``clip_id`` (``ibo_<videoId>``); falls back to the
    ``source_url`` so reposts that lost their clip_id are still de-duplicated.
    """
    if isinstance(repost, dict):
        raw = repost.get("clip_id") or repost.get("source_url")
    else:
        raw = repost
    return str(raw).strip() if raw is not None else ""


def select_new_reposts(reposts, already_matched) -> list:
    """Pure new-repost-selection diff.

    Given scanned @Iboclipss repost records and the set of already-matched repost
    ids, return the records not yet matched — order-preserving, de-duplicated
    (first occurrence wins), records without a usable id skipped. The mirror of
    :func:`ml.gt_capture_daily.select_new_vods` on the repost side.
    """
    seen = {str(rid).strip() for rid in already_matched}
    out: list = []
    emitted: set[str] = set()
    for repost in reposts:
        rid = _repost_id(repost)
        if not rid or rid in seen or rid in emitted:
            continue
        emitted.add(rid)
        out.append(repost)
    return out


# ---------------------------------------------------------------------------
# Pure decision seam: accumulating manifest merge across weekly runs
# ---------------------------------------------------------------------------
def _moment_key(moment: dict) -> str:
    """Identity of a matched moment within a VOD: its repost URL, else its span."""
    url = str(moment.get("source_url") or "").strip()
    if url:
        return url
    return f"{moment.get('start')}-{moment.get('end')}"


def merge_manifests(existing: list[dict], new: list[dict]) -> list[dict]:
    """Union a fresh batch's manifest into the accumulated one, per VOD.

    Moments are keyed by repost URL within a VOD; a later (``new``) verdict for
    the same repost overwrites the earlier one, so a re-verified moment upgrades
    in place. VODs are emitted sorted (deterministic, matching the core), moments
    sorted by start. This is what makes the gold manifest *accumulate* week over
    week instead of being clobbered by each run's small new batch.
    """
    by_vod: dict[str, dict[str, dict]] = {}
    for row in list(existing) + list(new):
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id:
            continue
        bucket = by_vod.setdefault(vod_id, {})
        for moment in row.get("gt_moments") or []:
            bucket[_moment_key(moment)] = moment
    return [
        {"vod_id": vod_id, "gt_moments": sorted(bucket.values(), key=lambda m: float(m.get("start") or 0))}
        for vod_id, bucket in sorted(by_vod.items())
    ]


# ---------------------------------------------------------------------------
# Pure decision seam: gold-accrual vs target (the headline progress metric)
# ---------------------------------------------------------------------------
def gold_accrual(gold_manifest: list[dict], *, target_gold: int = TARGET_GOLD, target_vods: int = TARGET_VODS) -> dict:
    """Confirmed-gold accrual across the manifest, as progress toward the target.

    Counts gold moments and the VODs carrying at least one (silver does not count
    toward the gold target). ``target_reached`` is true only when both the gold
    and VOD thresholds are met — "robust enough to trust the headline metric".
    """
    gold = sum(
        1
        for row in gold_manifest
        for m in (row.get("gt_moments") or [])
        if str(m.get("gt_tier") or "") == GOLD
    )
    vods = sum(
        1
        for row in gold_manifest
        if any(str(m.get("gt_tier") or "") == GOLD for m in (row.get("gt_moments") or []))
    )
    return {
        "gold": gold,
        "vods": vods,
        "target_gold": target_gold,
        "target_vods": target_vods,
        "gold_pct": round(100.0 * gold / target_gold, 1) if target_gold else 0.0,
        "vods_pct": round(100.0 * vods / target_vods, 1) if target_vods else 0.0,
        "target_reached": gold >= target_gold and vods >= target_vods,
    }


# ---------------------------------------------------------------------------
# Already-matched state (resumable / idempotent)
# ---------------------------------------------------------------------------
def matched_ids(state: dict) -> set[str]:
    """The set of repost ids recorded as already matched in ``state``."""
    return {str(rid) for rid in (state.get("matched") or {})}


def load_match_state(path: pathlib.Path) -> dict:
    """Load the already-matched state. Missing/unreadable → empty (safe restart)."""
    path = pathlib.Path(path)
    if not path.exists():
        return {"matched": {}, "vods_seen": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("gt_match_verify_weekly: unreadable state %s (%s) — starting fresh", path, exc)
        return {"matched": {}, "vods_seen": []}
    if not isinstance(data, dict):
        return {"matched": {}, "vods_seen": []}
    data.setdefault("matched", {})
    data.setdefault("vods_seen", [])
    return data


def save_match_state(path: pathlib.Path, state: dict) -> None:
    """Atomically persist the already-matched state (tmp file + replace)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# @Iboclipss re-scan (cookie-auth + pacing; lazy import keeps seams net-free)
# ---------------------------------------------------------------------------
class RepostDiscoveryZeroed(RuntimeError):
    """Raised when the @Iboclipss scan returns nothing / errors.

    Bot-protection ("try again later") can make yt-dlp return an empty listing.
    Treating that as "no new reposts" would silently freeze the flywheel, so the
    job refuses a zeroed scan instead of advancing state on a phantom zero.
    """


def list_iboclipss_reposts(channel: str = DEFAULT_CHANNEL, *, max_candidates: int = DEFAULT_MAX_CANDIDATES) -> list[dict]:
    """Re-scan @Iboclipss shorts, reusing the cookie-auth + pacing wiring.

    Lists the channel with the same auth (cookiefile/browser) and request pacing
    (``sleep_interval_requests`` / ``sleep_interval``) the short-form download
    path uses, so the scan looks like the established, un-blocked traffic. A raw
    listing that comes back empty or errors raises :class:`RepostDiscoveryZeroed`
    rather than masquerading as "no new reposts".
    """
    import yt_dlp

    from services.downloader import _apply_short_form_sleep_opts
    from services.yt_dlp_auth import apply_yt_dlp_auth_options

    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max(int(max_candidates), 1),
        "skip_download": True,
    }
    apply_yt_dlp_auth_options(opts)
    _apply_short_form_sleep_opts(opts)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel, download=False)
    except Exception as exc:  # noqa: BLE001 — any listing failure is a zeroed scan
        raise RepostDiscoveryZeroed(f"channel listing failed for {channel}: {exc}") from exc

    entries = [e for e in ((info or {}).get("entries") or []) if e]
    if not entries:
        raise RepostDiscoveryZeroed(f"channel listing for {channel} returned 0 entries (bot-protection?)")

    reposts: list[dict] = []
    for entry in entries:
        video_id = str(entry.get("id") or "").strip()
        source_url = (
            str(entry.get("webpage_url") or "").strip()
            or str(entry.get("url") or "").strip()
            or (f"https://www.youtube.com/shorts/{video_id}" if video_id else "")
        )
        if not source_url:
            continue
        reposts.append({
            "clip_id": f"ibo_{video_id}" if video_id else "",
            "source_url": source_url,
            "platform": "youtube",
            "title": str(entry.get("title") or "").strip(),
            "views": int(entry.get("view_count") or 0),
            "duration": int(entry.get("duration") or 0),
        })
    return reposts


def download_short_form(url: str) -> pathlib.Path:
    """Fetch one repost through the short-form download cache (bot-protection sleeps included).

    Same call the ``gt_pipeline`` download step uses; a cache hit returns at once.
    """
    from services.downloader import ensure_cached_download

    return ensure_cached_download(url, cache_namespace="short_form", cache_ttl_sec=-1)


def attach_media(reposts: list[dict], download_repost) -> tuple[list[dict], list[dict]]:
    """Download each repost; return ``(with_video_path, failed)``.

    The matcher skips a repost without ``video_path`` (``missing_media``), so
    only the downloaded ones are worth matching. A failed download is logged and
    handed back so the caller leaves it un-done and retries it next run.
    """
    downloaded: list[dict] = []
    failed: list[dict] = []
    for repost in reposts:
        url = str(repost.get("source_url") or "").strip()
        try:
            if not url:
                raise ValueError("no source_url")
            video_path = download_repost(url)
        except Exception as exc:  # noqa: BLE001 — any failure just means "retry next run"
            logger.warning("gt_match_verify_weekly: download failed for %s (%s) — will retry next run",
                           _repost_id(repost) or url, exc)
            failed.append(repost)
            continue
        downloaded.append(dict(repost, video_path=str(video_path)))
    return downloaded, failed


def load_captured_vod_ids(capture_state_path: pathlib.Path) -> list[str]:
    """The VOD ids captured by the daily job — the corpus to match reposts against."""
    state = gt_capture_daily.load_captured_state(capture_state_path)
    return sorted(gt_capture_daily.captured_ids(state))


# ---------------------------------------------------------------------------
# Weekly match+verify orchestrator (thin shell over the pure seams + #7 core)
# ---------------------------------------------------------------------------
def run_weekly_match_verify(
    *,
    scan_reposts,
    captured_vod_ids,
    state_path: pathlib.Path,
    cache_root: pathlib.Path,
    manifest_path: pathlib.Path,
    gold_manifest_path: pathlib.Path,
    match_and_verify=None,
    download_repost=None,
    matched_at: str | None = None,
    target_gold: int = TARGET_GOLD,
    target_vods: int = TARGET_VODS,
    limit: int | None = None,
) -> dict:
    """Re-scan reposts, match+verify the new ones, rebuild the gold manifest.

    ``scan_reposts`` is a zero-arg callable returning repost records (it raises
    :class:`RepostDiscoveryZeroed` on a bot-protection-zeroed listing);
    ``match_and_verify`` defaults to the real #7 core and ``download_repost``
    (url -> local path, raises on failure) to :func:`download_short_form`; both
    are injectable for tests. Each new repost is downloaded before matching; a
    repost whose download fails is not marked done, so the next run retries it.
    A repost is marked done only once it produced a match row or has been open
    longer than ``OPEN_REPOST_DAYS``; until then each run re-matches it.
    State and both manifests are persisted; a zeroed scan returns a summary with
    ``scan_zeroed=True`` and mutates nothing, so next week simply retries.
    """
    state_path = pathlib.Path(state_path)
    manifest_path = pathlib.Path(manifest_path)
    gold_manifest_path = pathlib.Path(gold_manifest_path)
    if match_and_verify is None:
        match_and_verify = gt_match_verify.match_and_verify
    if download_repost is None:
        download_repost = download_short_form
    if matched_at is None:
        matched_at = datetime.now(timezone.utc).isoformat()

    captured = sorted({str(v).strip() for v in captured_vod_ids if str(v).strip()})
    state = load_match_state(state_path)
    seen_vods = {str(v) for v in (state.get("vods_seen") or [])}
    new_vods = [v for v in captured if v not in seen_vods]

    try:
        reposts = list(scan_reposts())
    except RepostDiscoveryZeroed as exc:
        logger.warning("gt_match_verify_weekly: discovery zeroed (%s) — skipping, will retry next run", exc)
        return {
            "scan_zeroed": True,
            "reposts_scanned": 0,
            "new_reposts": 0,
            "download_failed": 0,
            "new_vods": new_vods,
            "total_captured_vods": len(captured),
            "run_counts": {},
            "accrual": gold_accrual(read_jsonl(gold_manifest_path), target_gold=target_gold, target_vods=target_vods),
        }

    already = matched_ids(state)
    new_reposts = select_new_reposts(reposts, already)
    if limit is not None:
        new_reposts = new_reposts[:limit]

    logger.info(
        "gt_match_verify_weekly: %d reposts scanned, %d new, %d already matched; %d captured VODs (%d new)",
        len(reposts), len(new_reposts), len(already), len(captured), len(new_vods),
    )

    downloaded, failed = attach_media(new_reposts, download_repost)
    result = match_and_verify(downloaded, captured, cache_root=cache_root)

    merged = merge_manifests(read_jsonl(manifest_path), result.manifest)
    write_jsonl(manifest_path, merged)
    gold_manifest = gt_match_verify.filter_gold_manifest(merged)
    write_jsonl(gold_manifest_path, gold_manifest)

    # A repost is done once it produced a match row, or after OPEN_REPOST_DAYS
    # without one (its VOD may only be captured after this run). Failed
    # downloads never reached the matcher, so they never count.
    hit_urls = {
        str(m.get("source_url") or "").strip()
        for row in result.manifest
        for m in (row.get("gt_moments") or [])
    }
    pending = state.setdefault("pending", {})
    now = datetime.fromisoformat(matched_at)
    for repost in downloaded:
        rid = _repost_id(repost)
        if not rid:
            continue
        first_tried = pending.setdefault(rid, matched_at)
        expired = now - datetime.fromisoformat(first_tried) > timedelta(days=OPEN_REPOST_DAYS)
        if str(repost.get("source_url") or "").strip() in hit_urls or expired:
            pending.pop(rid)
            state.setdefault("matched", {})[rid] = {
                "clip_id": repost.get("clip_id", ""),
                "source_url": repost.get("source_url", ""),
                "matched_at": matched_at,
            }
    state["vods_seen"] = captured
    save_match_state(state_path, state)

    accrual = gold_accrual(gold_manifest, target_gold=target_gold, target_vods=target_vods)
    summary = {
        "scan_zeroed": False,
        "reposts_scanned": len(reposts),
        "new_reposts": len(new_reposts),
        "download_failed": len(failed),
        "matched_moments": len(result.verifications),
        "new_vods": new_vods,
        "total_captured_vods": len(captured),
        "run_counts": result.counts,
        "accrual": accrual,
    }
    logger.info(
        "gt_match_verify_weekly: done — %d new reposts → tiers=%s; gold-accrual %d/%d gold over %d/%d VODs (%.1f%%)",
        len(new_reposts), result.counts, accrual["gold"], accrual["target_gold"],
        accrual["vods"], accrual["target_vods"], accrual["gold_pct"],
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entrypoint (driven by the weekly cron wrapper)
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="GT flywheel weekly match+verify job — re-scan @Iboclipss, mint gold, rebuild the eval manifest."
    )
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument(
        "--cache-root",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gt_captures"),
        help="Root of the per-VOD capture artifacts (must match the daily job's --cache-root).",
    )
    parser.add_argument("--state", default="", help=f"Already-matched state file (default: <cache-root>/{STATE_FILENAME}).")
    parser.add_argument("--capture-state", default="", help=f"Daily-capture state file (default: <cache-root>/{CAPTURE_STATE_FILENAME}).")
    parser.add_argument("--manifest", default="", help=f"Accumulated all-tier manifest (default: <cache-root>/{MANIFEST_FILENAME}).")
    parser.add_argument("--gold-manifest", default="", help=f"Gold-filtered eval manifest (default: <cache-root>/{GOLD_MANIFEST_FILENAME}).")
    parser.add_argument("--target-gold", type=int, default=TARGET_GOLD)
    parser.add_argument("--target-vods", type=int, default=TARGET_VODS)
    parser.add_argument("--limit", type=int, default=None, help="Cap how many new reposts to match this run.")
    parser.add_argument("--json", action="store_true", help="Print the run summary as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")

    cache_root = pathlib.Path(args.cache_root)
    state_path = pathlib.Path(args.state) if args.state else cache_root / STATE_FILENAME
    capture_state = pathlib.Path(args.capture_state) if args.capture_state else cache_root / CAPTURE_STATE_FILENAME
    manifest_path = pathlib.Path(args.manifest) if args.manifest else cache_root / MANIFEST_FILENAME
    gold_path = pathlib.Path(args.gold_manifest) if args.gold_manifest else cache_root / GOLD_MANIFEST_FILENAME

    summary = run_weekly_match_verify(
        scan_reposts=lambda: list_iboclipss_reposts(args.channel, max_candidates=args.max_candidates),
        captured_vod_ids=load_captured_vod_ids(capture_state),
        state_path=state_path,
        cache_root=cache_root,
        manifest_path=manifest_path,
        gold_manifest_path=gold_path,
        target_gold=args.target_gold,
        target_vods=args.target_vods,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=True))
    elif summary["scan_zeroed"]:
        print("gt_match_verify_weekly: @Iboclipss scan was zeroed (bot-protection?) — skipped, will retry next run.")
    else:
        acc = summary["accrual"]
        print(
            f"Matched {summary['new_reposts'] - summary['download_failed']} new repost(s) → {summary['run_counts']} "
            f"({summary['download_failed']} download(s) failed, retry next run); "
            f"new VODs: {summary['new_vods']}; gold-accrual {acc['gold']}/{acc['target_gold']} "
            f"over {acc['vods']}/{acc['target_vods']} VODs ({acc['gold_pct']}%)"
            + ("  — TARGET REACHED" if acc["target_reached"] else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
