"""GT-flywheel match+verify core — mint gold from reposts (issue #7 / PRD #3).

The lag-tolerant half of the GT flywheel: the *gold-minting* core. Given a batch
of @Iboclipss reposts and the set of VOD ids captured by the daily capture job
(:mod:`ml.gt_capture` / #5), this:

  1. Matches each repost against the captured VODs (transcript primary, audio
     fallback) — reusing :mod:`ml.match_cross_platform_transcripts` unchanged.
  2. Builds a per-VOD, dedup'd manifest of matched moments — reusing
     :func:`ml.build_eval_manifest._dedup_moments`.
  3. Auto-verifies every moment with the confirm-margin rule — reusing
     :func:`ml.verify_gt_matches.verify_moment` (``is_confirmed`` /
     ``assign_tier``): gold when the margin clears the floor, silver for
     strong-transcript-but-unconfirmed, review for borderline, unverifiable when
     the media is gone.
  4. Rebuilds a **gold-filtered** eval manifest containing only gold (+ silver)
     targets — ``not_confirmed`` (review) and ``vod_unavailable`` (unverifiable)
     moments are excluded so phantom/mispositioned targets never poison Recall@K.

Matching and verification both read the *stored* fingerprint + transcript
artifacts (``cache_root/<vod_id>/{fingerprint,transcript}.json``), so a VOD whose
Twitch media is already gone still confirms — which is the whole point of
decoupling capture from match+verify.

This module is the reusable, dependency-injectable core. Scheduling, @Iboclipss
new-repost selection, persisted state, and the run summary live in the weekly
match+verify job (#8). Heavy collaborators (the transcript matcher, which pulls
Whisper) are imported lazily inside the default seams so the pure decision
functions stay testable without the ML stack installed.
"""
from __future__ import annotations

import logging
import pathlib
import shutil
from dataclasses import dataclass

from ml import gt_capture
from ml import match_cross_platform_audio_fingerprints as audio_matcher
from ml import verify_gt_matches as verifier
from ml.build_eval_manifest import _dedup_moments

logger = logging.getLogger(__name__)

FINGERPRINT_FILENAME = gt_capture.FINGERPRINT_FILENAME

# Tiers that earn a slot in the gold-filtered eval manifest. Everything else
# (review = weak not_confirmed, unverifiable = vod/repost media gone) is dropped.
GOLD = verifier.GOLD
SILVER = verifier.SILVER

# Transcript-similarity floor for accepting a candidate window, matching the
# coupled gt_pipeline default; verification is the precision gate downstream.
DEFAULT_MIN_SIMILARITY = 0.58


@dataclass
class MatchVerifyResult:
    """Outcome of one match+verify pass over a batch of reposts.

    ``manifest`` carries every matched moment with its ``gt_tier``;
    ``gold_manifest`` is the gold(+silver)-filtered subset that becomes the eval
    ground truth. ``verifications`` is one verdict row per moment, and ``counts``
    aggregates tiers for the run summary.
    """

    manifest: list[dict]
    gold_manifest: list[dict]
    verifications: list[dict]
    counts: dict


# ---------------------------------------------------------------------------
# Pure decision seams
# ---------------------------------------------------------------------------
def build_captured_vod_rows(captured_vod_ids, cache_root: pathlib.Path) -> list[dict]:
    """Turn captured VOD ids into matcher ``vod_rows`` pointing at stored artifacts.

    Each row carries ``vod_id``, the Twitch ``vod_url`` and a ``fingerprint_path``
    into the capture artifact dir, so the matcher's audio fallback reads the
    stored full-VOD fingerprint (``_fingerprint_from_row``) and the transcript
    matcher reads ``cache_root/<vod_id>/transcript.json`` — both work with the
    media already deleted. Order-preserving, de-duplicated, blank ids skipped.
    """
    cache_root = pathlib.Path(cache_root)
    rows: list[dict] = []
    emitted: set[str] = set()
    for raw in captured_vod_ids:
        vod_id = str(raw).strip() if raw is not None else ""
        if not vod_id or vod_id in emitted:
            continue
        emitted.add(vod_id)
        rows.append({
            "vod_id": vod_id,
            "vod_url": f"https://www.twitch.tv/videos/{vod_id}",
            "fingerprint_path": str(cache_root / vod_id / FINGERPRINT_FILENAME),
        })
    return rows


def build_manifest_from_matches(match_rows: list[dict]) -> list[dict]:
    """Group accepted match rows by VOD and dedup near-overlapping windows.

    Reuses :func:`ml.build_eval_manifest._dedup_moments` (highest-views wins a
    slot) so the manifest shape is identical to the canonical eval manifest;
    moments carry the fields :func:`ml.verify_gt_matches.verify_moment` reads
    (``start``/``end``/``source_url``/``match_similarity``/...).
    """
    by_vod: dict[str, list[dict]] = {}
    for row in match_rows:
        vod_id = str(row.get("vod_id") or "").strip()
        if vod_id:
            by_vod.setdefault(vod_id, []).append(row)
    return [
        {"vod_id": vod_id, "gt_moments": _dedup_moments(rows)}
        for vod_id, rows in sorted(by_vod.items())
    ]


def filter_gold_manifest(manifest_rows: list[dict], *, keep_silver: bool = True) -> list[dict]:
    """Rebuild the eval manifest keeping only gold (+ silver) moments.

    ``review`` (weak ``not_confirmed``) and ``unverifiable`` (``vod_unavailable``)
    moments are excluded so they never poison Recall@K; a VOD left with no kept
    moment is dropped entirely.
    """
    keep = {GOLD, SILVER} if keep_silver else {GOLD}
    filtered: list[dict] = []
    for row in manifest_rows:
        kept = [m for m in (row.get("gt_moments") or []) if str(m.get("gt_tier") or "") in keep]
        if kept:
            filtered.append({**row, "gt_moments": kept})
    return filtered


def verdict_counts(verification_rows: list[dict]) -> dict:
    """Count verification rows by ``gt_tier`` for the run summary."""
    counts: dict[str, int] = {}
    for row in verification_rows:
        tier = str(row.get("gt_tier") or "")
        counts[tier] = counts.get(tier, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Default collaborators (lazy heavy imports keep this module torch-free)
# ---------------------------------------------------------------------------
def _make_default_match(cache_root: pathlib.Path, min_similarity: float, match_workers: int):
    def _match(vod_rows: list[dict], reposts: list[dict]) -> list[dict]:
        # Lazy import: the transcript matcher pulls Whisper/torch.
        from ml import match_cross_platform_transcripts as transcript_matcher

        return transcript_matcher.match_reposts_to_vods(
            vod_rows,
            reposts,
            min_similarity=min_similarity,
            vod_cache_root=cache_root,
            transcript_match_workers=max(1, int(match_workers or 1)),
        )

    return _match


def _make_default_load_fingerprint(cache_root: pathlib.Path):
    def _load(vod_id: str) -> audio_matcher.Fingerprint | None:
        path = cache_root / str(vod_id) / FINGERPRINT_FILENAME
        if not path.exists():
            return None
        try:
            return audio_matcher._read_fingerprint_file(path)
        except (ValueError, OSError):
            # Prefix-only/corrupt cache → fall back to the range-download path
            # inside verify_moment by reporting "no stored fingerprint".
            return None

    return _load


def _make_default_verify(fpcalc: str):
    def _verify(vod_id, moment, *, full_vod_fp, null_source_fp):
        return verifier.verify_moment(
            vod_id,
            moment,
            fpcalc=fpcalc,
            full_vod_fp=full_vod_fp,
            null_source_fp=null_source_fp,
        )

    return _verify


# ---------------------------------------------------------------------------
# Match+verify core (thin shell over the pure seams + reused collaborators)
# ---------------------------------------------------------------------------
def match_and_verify(
    reposts: list[dict],
    captured_vod_ids,
    *,
    cache_root: pathlib.Path,
    fpcalc: str | None = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    match_workers: int = 0,
    keep_silver: bool = True,
    match_reposts=None,
    load_vod_fingerprint=None,
    verify_moment=None,
) -> MatchVerifyResult:
    """Match a batch of reposts against captured VODs and auto-mint gold.

    Collaborators are injectable for testing; the defaults wire the real
    transcript+audio matcher, the stored-fingerprint loader, and the
    confirm-margin verifier. Returns the tiered manifest, the gold-filtered
    eval manifest, the per-moment verdicts and a tier-count summary. Persisting
    these (append verdicts, merge manifest, state, scheduling) is the weekly
    job's responsibility (#8).
    """
    cache_root = pathlib.Path(cache_root)

    if match_reposts is None:
        match_reposts = _make_default_match(cache_root, min_similarity, match_workers)
    if load_vod_fingerprint is None:
        load_vod_fingerprint = _make_default_load_fingerprint(cache_root)
    if verify_moment is None:
        if fpcalc is None:
            fpcalc = shutil.which("fpcalc")
            if fpcalc is None:
                raise RuntimeError(
                    "fpcalc not found. Install chromaprint-tools to enable match verification."
                )
        verify_moment = _make_default_verify(fpcalc)

    vod_rows = build_captured_vod_rows(captured_vod_ids, cache_root)
    match_rows = match_reposts(vod_rows, reposts)
    manifest = build_manifest_from_matches(match_rows)

    verifications: list[dict] = []
    null_source_fp: audio_matcher.Fingerprint | None = None
    for vod_row in manifest:
        vod_id = str(vod_row.get("vod_id") or "")
        full_vod_fp = load_vod_fingerprint(vod_id)
        # First captured fingerprint seen doubles as the noise-floor null source
        # for VODs whose own media is gone (mirrors verify_gt_matches).
        if full_vod_fp is not None and null_source_fp is None:
            null_source_fp = full_vod_fp
        for moment in vod_row.get("gt_moments") or []:
            row = verify_moment(vod_id, moment, full_vod_fp=full_vod_fp, null_source_fp=null_source_fp)
            moment["gt_tier"] = row.get("gt_tier")
            verifications.append(row)

    gold_manifest = filter_gold_manifest(manifest, keep_silver=keep_silver)
    counts = verdict_counts(verifications)
    logger.info(
        "gt_match_verify: %d reposts → %d matched moments across %d VODs; tiers=%s; "
        "gold manifest: %d VODs / %d moments",
        len(reposts),
        len(verifications),
        len(manifest),
        counts,
        len(gold_manifest),
        sum(len(r.get("gt_moments") or []) for r in gold_manifest),
    )
    return MatchVerifyResult(
        manifest=manifest,
        gold_manifest=gold_manifest,
        verifications=verifications,
        counts=counts,
    )
