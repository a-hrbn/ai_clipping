"""GT-flywheel daily capture job — the standing, idempotent wrapper around the
capture core (issue #6 / PRD #3).

Polls Twitch Helix for the streamer's archive VODs, diffs them against the
already-captured set via a *pure* new-VOD-selection function, runs the capture
core (:func:`ml.gt_capture.capture_vod`) on each genuinely-new VOD, and persists
the already-captured state so re-runs are safe and resumable.

Decoupling capture (time-critical, daily) from match+verify (lag-tolerant,
weekly) is the central architectural move of the flywheel: the coupled
``gt_pipeline`` is what produced ``vod_unavailable`` by fingerprinting a VOD only
after reposts already existed. Because the capture core stores fingerprint +
transcript artifacts and deletes the media, VODs captured today stay matchable
later even after Twitch deletes the original VOD.

State is persisted **after every captured VOD**, so a crash mid-batch resumes
without re-capturing what already succeeded.

Scheduling lives outside this module (a daily cron after stream-end — see
``scripts/run_gt_capture_daily.sh``); this is the reusable, dependency-injectable
orchestrator. The Helix client is imported lazily so the pure new-VOD-selection
seam stays testable without network access or the ML stack installed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import pathlib
import sys
from datetime import datetime, timezone

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import gt_capture

logger = logging.getLogger(__name__)

STATE_FILENAME = "capture_state.json"
LOCK_SUFFIX = ".lock"
DEFAULT_BROADCASTER = "jynxzi"


# ---------------------------------------------------------------------------
# Pure decision seam: new-VOD selection
# ---------------------------------------------------------------------------
def _vod_id(vod) -> str:
    """Normalize a Helix VOD record (or bare id) to a string id."""
    raw = vod.get("id") if isinstance(vod, dict) else vod
    return str(raw).strip() if raw is not None else ""


def select_new_vods(vods, already_captured) -> list:
    """Pure new-VOD-selection diff.

    Given Helix archive-VOD records (dicts with an ``id``) and the set of
    already-captured VOD ids, return the records whose ids are not yet captured —
    order-preserving and de-duplicated (first occurrence wins). Records without a
    usable id are skipped. Ids are normalized to strings so int/str mismatches
    between Helix and persisted state still match.
    """
    seen = {str(vid).strip() for vid in already_captured}
    out: list = []
    emitted: set[str] = set()
    for vod in vods:
        vid = _vod_id(vod)
        if not vid or vid in seen or vid in emitted:
            continue
        emitted.add(vid)
        out.append(vod)
    return out


# ---------------------------------------------------------------------------
# Already-captured state (resumable / idempotent)
# ---------------------------------------------------------------------------
def captured_ids(state: dict) -> set[str]:
    """The set of VOD ids recorded as captured in ``state``."""
    return {str(vid) for vid in (state.get("captured") or {})}


def load_captured_state(path: pathlib.Path) -> dict:
    """Load the already-captured state. Missing or unreadable file → empty state
    (so a corrupt file degrades to a fresh, safe start rather than crashing)."""
    path = pathlib.Path(path)
    if not path.exists():
        return {"captured": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("gt_capture_daily: unreadable state %s (%s) — starting fresh", path, exc)
        return {"captured": {}}
    if not isinstance(data, dict):
        return {"captured": {}}
    data.setdefault("captured", {})
    return data


def save_captured_state(path: pathlib.Path, state: dict) -> None:
    """Atomically persist the already-captured state (tmp file + replace)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Single-run lock — stops overlapping runs from double-downloading the same VODs
# ---------------------------------------------------------------------------
class CaptureAlreadyRunning(RuntimeError):
    """Raised when another gt_capture_daily run already holds the lock."""


def _read_lock_pid(lock_path: pathlib.Path) -> int | None:
    try:
        return int(pathlib.Path(lock_path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists (signal 0 probes without killing it)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by another user
    return True


@contextlib.contextmanager
def capture_lock(lock_path, *, pid: int | None = None, is_alive=_pid_alive):
    """Hold an exclusive single-run lock for the whole capture run.

    A PID file (not ``flock``), so it works on WSL/drvfs and bind-mounted
    container volumes where advisory locks are unreliable. A lock held by a
    *live* PID rejects the run; a stale lock (dead PID — a crashed or ``kill -9``'d
    run) is taken over. The lock is removed on exit only if we still own it.
    """
    lock_path = pathlib.Path(lock_path)
    pid = pid if pid is not None else os.getpid()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    holder = _read_lock_pid(lock_path)
    if holder is not None and holder != pid and is_alive(holder):
        raise CaptureAlreadyRunning(
            f"another gt_capture_daily run is active (pid {holder}); lock: {lock_path}"
        )

    lock_path.write_text(str(pid), encoding="utf-8")
    try:
        yield
    finally:
        if _read_lock_pid(lock_path) == pid:
            lock_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helix archive-VOD listing (lazy import keeps the pure seams net/ML-free)
# ---------------------------------------------------------------------------
def list_archive_vods(client, broadcaster: str, *, max_vods: int = 100) -> list[dict]:
    """List the streamer's most-recent archive VODs via the existing Helix client.

    Helix ``/videos?type=archive`` returns newest-first; one page (≤100) covers
    far more than any daily/weekly gap, and the pure new-VOD diff filters the
    already-captured ones. Annotates ``duration_seconds`` like the existing
    ``collect_twitch_clip_labels._get_vods`` for downstream consistency.
    """
    from ml.collect_twitch_clip_labels import _get_user, _parse_twitch_duration

    user = _get_user(client, broadcaster)
    payload = client.get_json(
        "/videos",
        user_id=str(user["id"]),
        type="archive",
        first=min(int(max_vods), 100),
    )
    vods: list[dict] = []
    for vod in payload.get("data") or []:
        row = dict(vod)
        row["duration_seconds"] = _parse_twitch_duration(str(vod.get("duration") or ""))
        vods.append(row)
    return vods


def _vod_date(vod) -> str:
    """Best-available date string for provenance (Helix ``created_at``)."""
    if isinstance(vod, dict):
        return str(vod.get("created_at") or vod.get("published_at") or "")
    return ""


# ---------------------------------------------------------------------------
# Daily capture orchestrator (thin shell over the pure seams + capture core)
# ---------------------------------------------------------------------------
def run_daily_capture(
    *,
    list_vods,
    state_path: pathlib.Path,
    cache_root: pathlib.Path,
    capture=gt_capture.capture_vod,
    captured_at: str | None = None,
    limit: int | None = None,
    lock_path=None,
    is_alive=_pid_alive,
) -> dict:
    """Capture every not-yet-captured archive VOD, persisting state per VOD.

    ``list_vods`` is a zero-arg callable returning Helix archive-VOD records;
    ``capture`` defaults to the real capture core but is injectable for tests.
    The whole run is guarded by a single-run lock so overlapping invocations
    (e.g. a manual re-run while the cron is mid-download) can't double-download
    the same VODs — a second run raises ``CaptureAlreadyRunning``.
    Returns a compact summary: captured ids, skipped count, and failed ids.
    """
    state_path = pathlib.Path(state_path)
    cache_root = pathlib.Path(cache_root)
    if lock_path is None:
        lock_path = state_path.parent / (state_path.name + LOCK_SUFFIX)
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).isoformat()

    with capture_lock(lock_path, is_alive=is_alive):
        state = load_captured_state(state_path)
        already = captured_ids(state)

        vods = list(list_vods())
        new_vods = select_new_vods(vods, already)
        if limit is not None:
            new_vods = new_vods[:limit]

        skipped = len(vods) - len(new_vods)
        captured: list[str] = []
        failed: list[str] = []

        logger.info(
            "gt_capture_daily: %d archive VODs listed, %d new, %d already captured",
            len(vods), len(new_vods), len(already),
        )

        for vod in new_vods:
            vid = _vod_id(vod)
            try:
                result = capture(
                    vid,
                    cache_dir=cache_root / vid,
                    vod_date=_vod_date(vod),
                    captured_at=captured_at,
                )
            except Exception:  # noqa: BLE001 — one bad VOD must not abort the batch
                logger.warning("gt_capture_daily: capture failed for VOD %s", vid, exc_info=True)
                failed.append(vid)
                continue

            # Commit each success immediately so a later crash is resumable.
            state.setdefault("captured", {})[vid] = getattr(result, "provenance", {"vod_id": vid})
            save_captured_state(state_path, state)
            captured.append(vid)
            logger.info("gt_capture_daily: captured VOD %s", vid)

        summary = {
            "broadcaster_vods": len(vods),
            "captured": captured,
            "skipped": skipped,
            "failed": failed,
            "total_captured": len(captured_ids(state)),
        }
        logger.info(
            "gt_capture_daily: done — captured %d, skipped %d, failed %d (total gold VODs: %d)",
            len(captured), skipped, len(failed), summary["total_captured"],
        )
        return summary


# ---------------------------------------------------------------------------
# CLI entrypoint (driven by the daily cron wrapper)
# ---------------------------------------------------------------------------
def main(argv=None, *, list_vods=None, capture=None) -> int:
    """CLI entrypoint. ``list_vods``/``capture`` are injectable for tests; by
    default VODs come from Helix and capture is the real core. Returns 1 when
    any VOD failed so cron/monitoring sees it (successes are still committed)."""
    parser = argparse.ArgumentParser(
        description="GT flywheel daily capture job — poll Helix archives and capture new VODs."
    )
    parser.add_argument("--broadcaster", default=DEFAULT_BROADCASTER)
    parser.add_argument(
        "--cache-root",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gt_captures"),
        help="Root directory for per-VOD fingerprint/transcript/provenance artifacts.",
    )
    parser.add_argument(
        "--state",
        default="",
        help=f"Path to the already-captured state file (default: <cache-root>/{STATE_FILENAME}).",
    )
    parser.add_argument("--max-vods", type=int, default=100, help="Helix archive page size to scan.")
    parser.add_argument("--limit", type=int, default=None, help="Cap how many new VODs to capture this run.")
    parser.add_argument("--json", action="store_true", help="Print the run summary as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")

    if list_vods is None:
        from config import settings
        from ml.collect_twitch_clip_labels import TwitchClient

        if not settings.twitch_client_id or not settings.twitch_client_secret:
            print("TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be configured.")
            return 1
        client = TwitchClient(settings.twitch_client_id, settings.twitch_client_secret)
        list_vods = lambda: list_archive_vods(client, args.broadcaster, max_vods=args.max_vods)  # noqa: E731

    cache_root = pathlib.Path(args.cache_root)
    state_path = pathlib.Path(args.state) if args.state else cache_root / STATE_FILENAME

    try:
        summary = run_daily_capture(
            list_vods=list_vods,
            state_path=state_path,
            cache_root=cache_root,
            capture=capture or gt_capture.capture_vod,
            limit=args.limit,
        )
    except CaptureAlreadyRunning as exc:
        # A cron firing while a long run is still going is expected, not an error.
        print(f"gt_capture_daily: skipped — {exc}")
        return 0

    if args.json:
        print(json.dumps(summary, ensure_ascii=True))
    else:
        print(
            f"Captured {len(summary['captured'])} new VOD(s): {summary['captured']}; "
            f"skipped {summary['skipped']}, failed {len(summary['failed'])}; "
            f"total captured VODs: {summary['total_captured']}"
        )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
