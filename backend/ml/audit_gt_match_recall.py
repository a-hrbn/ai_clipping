"""Audit GT match recall: which discovered Iboclipss reposts never made it into GT.

Joins ``iboclipss_discovered.jsonl`` against ``matches.jsonl`` of a gt_pipeline
run directory and reports clip-level and view-weighted match recall, recall per
view tier, and a content classification of the unmatched clips. The point is to
detect whether match failures are random or systematically hit low-speech
moments (screams, rage, visual gags) that transcript matching cannot anchor.

Usage:
    python -m ml.audit_gt_match_recall [--run-dir PATH] [--top-n 20]

Defaults to the newest run under backend/tmp/gt_runs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_RUNS_ROOT = BACKEND_ROOT / "tmp" / "gt_runs"

# Title markers for clips whose payoff is mostly non-verbal (screams, rage,
# physical/visual comedy). Transcript matching is expected to struggle here.
# Deliberately word-based: Iboclipss decorates nearly every title with the
# same emojis, so emojis carry no signal.
_LOW_SPEECH_PATTERN = re.compile(
    r"rage|loses? (it|his mind)|freak(s|ing)? out|melt ?down|screams?|yell|"
    r"breaks?|smash|destroy|throws?|slams?|cries|crash(es)? out|insane|"
    r"trap|tricks?|prank|sabotage|falls? for|caught|trolls?",
    re.IGNORECASE,
)
_TALK_PATTERN = re.compile(
    r"explains?|reacts? to|talks? about|story|tells|says|asks?|reads?|"
    r"interview|opinion|speech|calls? out",
    re.IGNORECASE,
)


def _video_id(url: str) -> str:
    """Normalize YouTube URL forms (shorts/<id>, watch?v=<id>, youtu.be/<id>)."""
    m = re.search(r"(?:shorts/|watch\?v=|youtu\.be/)([A-Za-z0-9_-]{6,})", url or "")
    return m.group(1) if m else (url or "")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _classify_title(title: str) -> str:
    low_speech = bool(_LOW_SPEECH_PATTERN.search(title))
    talk = bool(_TALK_PATTERN.search(title))
    if low_speech and not talk:
        return "low_speech"
    if talk and not low_speech:
        return "talky"
    if low_speech and talk:
        return "mixed"
    return "unclear"


def _newest_run_dir(runs_root: pathlib.Path) -> pathlib.Path:
    candidates = [
        d for d in runs_root.iterdir()
        if d.is_dir() and (d / "iboclipss_discovered.jsonl").exists() and (d / "matches.jsonl").exists()
    ]
    if not candidates:
        raise SystemExit(f"No usable run dirs under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _tier_stats(clips: list[dict], matched_ids: set[str]) -> list[dict]:
    tiers = [(500_000, ">=500k"), (100_000, "100k-500k"), (50_000, "50k-100k"), (0, "<50k")]
    rows = []
    upper = float("inf")
    for floor, label in tiers:
        bucket = [c for c in clips if floor <= (c.get("view_count") or 0) < upper]
        hit = sum(1 for c in bucket if _video_id(c["source_url"]) in matched_ids)
        rows.append({"tier": label, "clips": len(bucket), "matched": hit,
                     "recall": round(hit / len(bucket), 3) if bucket else None})
        upper = floor
    return rows


def build_audit(run_dir: pathlib.Path, top_n: int = 20, matches_name: str = "matches.jsonl") -> dict:
    discovered = _read_jsonl(run_dir / "iboclipss_discovered.jsonl")
    matches = _read_jsonl(run_dir / matches_name)

    # Audio-fallback rows place the repost via chromaprint alignment only and
    # historically carried meaningless positions (pre -length fix they always
    # did). They are dropped from the GT manifest, so counting them as
    # "matched" inflates recall — recall here means transcript-matched.
    transcript_rows = [r for r in matches if r.get("transcript_match_kind") != "audio_fingerprint_fallback"]
    matched_ids = {_video_id(r.get("source_url", "")) for r in transcript_rows}
    audio_only_ids = {
        _video_id(r.get("source_url", ""))
        for r in matches
        if r.get("transcript_match_kind") == "audio_fingerprint_fallback"
    } - matched_ids
    kind_counts: dict[str, int] = {}
    for r in matches:
        kind = str(r.get("transcript_match_kind") or "unknown")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    unmatched = [c for c in discovered if _video_id(c["source_url"]) not in matched_ids]
    matched = [c for c in discovered if _video_id(c["source_url"]) in matched_ids]

    total_views = sum(c.get("view_count") or 0 for c in discovered)
    matched_views = sum(c.get("view_count") or 0 for c in matched)

    def class_counts(clips: list[dict]) -> dict:
        counts: dict[str, int] = {}
        for c in clips:
            counts[_classify_title(c.get("title", ""))] = counts.get(_classify_title(c.get("title", "")), 0) + 1
        return counts

    unmatched_sorted = sorted(unmatched, key=lambda c: -(c.get("view_count") or 0))
    return {
        "run_dir": str(run_dir),
        "discovered_clips": len(discovered),
        "matched_clips": len(matched),
        "clip_match_recall": round(len(matched) / len(discovered), 3) if discovered else None,
        "match_rows_by_kind": kind_counts,
        "audio_fallback_only_clips": len(audio_only_ids),
        "view_weighted_recall": round(matched_views / total_views, 3) if total_views else None,
        "total_views": total_views,
        "matched_views": matched_views,
        "recall_by_view_tier": _tier_stats(discovered, matched_ids),
        "title_classes_matched": class_counts(matched),
        "title_classes_unmatched": class_counts(unmatched),
        "top_unmatched_by_views": [
            {
                "views": c.get("view_count"),
                "title": c.get("title"),
                "url": c.get("source_url"),
                "title_class": _classify_title(c.get("title", "")),
            }
            for c in unmatched_sorted[:top_n]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, default=None,
                        help="gt_pipeline run dir (default: newest under tmp/gt_runs)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="How many unmatched clips to list (default: 20)")
    parser.add_argument("--matches-name", default="matches.jsonl",
                        help="Matches file inside the run dir (default: matches.jsonl)")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Optional path to write the audit JSON")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or _newest_run_dir(_DEFAULT_RUNS_ROOT)
    audit = build_audit(run_dir, top_n=args.top_n, matches_name=args.matches_name)
    payload = json.dumps(audit, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
