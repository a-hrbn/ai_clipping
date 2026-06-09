"""Generate a Jynxzi clip profile from @Iboclipss YouTube Shorts.

Fetches the top clips from the last N days, calls an LLM to extract clip
archetypes and selection rules, and writes jynxzi_profile.json.

Usage:
  python ml/generate_jynxzi_profile.py --dry-run --top-n 80
  python ml/generate_jynxzi_profile.py --provider gemini --window-days 90
"""
from __future__ import annotations

import argparse
import datetime
import glob as _glob
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.discover_iboclipss_clips import (
    DEFAULT_CHANNEL,
    _entry_date,
    _enrich_with_upload_dates,
    _list_channel_entries,
)
from ml import llm_moment_finder as d0

DEFAULT_WINDOW_DAYS = 90
DEFAULT_MIN_VIEWS = 0
DEFAULT_DISCOVERED_GLOB = str(BACKEND_ROOT / "tmp" / "**" / "iboclipss_discovered.jsonl")
DEFAULT_LIVE_CACHE = BACKEND_ROOT / "ml" / "artifacts" / "iboclipss_live_cache.jsonl"
DEFAULT_OUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_profile.json"

_ANALYSIS_PROMPT = """\
You are a viral-clip expert analyzing a YouTube Shorts channel (@Iboclipss) that reposts the best clips from Twitch streamer Jynxzi (variety streamer).

Below are clips from the last {window_days} days (most recent first):

{clip_list}

Based on these titles, analyze what **situations and narrative contexts** make a Jynxzi clip go viral.
Focus on the game state, emotional arc, and circumstances — not on specific words or phrases.
Return raw JSON only (no markdown), using EXACTLY this schema:

{{
  "summary": "<2-3 sentences describing the situational patterns behind Jynxzi's viral clips>",
  "archetypes": [
    {{
      "name": "<UPPERCASE_SLUG>",
      "description": "<1 sentence: what situation/context defines this moment type>",
      "transcript_signals": [
        "<contextual indicator in the transcript: game state, pressure level, buildup, emotional arc, what is happening — not a specific quote>"
      ],
      "weight": <0.0-1.0, how commonly this archetype appears>
    }}
  ],
  "selection_rules": [
    "<imperative rule for an LLM reading a transcript: what situational context to look for, how the tension builds, where the payoff lands, how to set start/end to capture the full arc>"
  ],
  "avoid": [
    "<situation or context that should NOT be clipped>"
  ]
}}

Requirements:
- Provide 5-8 archetypes covering distinct situational clip types visible in the titles
- Provide 8-12 selection_rules focused on narrative arc, tension, and payoff — not keyword matching
- Provide 4-6 avoid rules describing low-signal contexts
- transcript_signals must describe the **situation or game context** (e.g. "solo vs multiple enemies, stack pressure rising", "comeback from a near-death state"), not exact quotes
- Weights must sum to approximately 1.0
"""


def _load_cached_rows(discovered_glob: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path_str in _glob.glob(discovered_glob, recursive=True):
        try:
            for line in pathlib.Path(path_str).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("source_url") or row.get("clip_id") or "")
                if key and key not in seen:
                    seen.add(key)
                    rows.append(row)
        except OSError:
            continue
    return rows


def _filter_to_window(rows: list[dict], window_days: int) -> list[dict]:
    cutoff = datetime.date.today() - datetime.timedelta(days=window_days)
    out: list[dict] = []
    for row in rows:
        date_str = str(row.get("upload_date") or "").strip()
        if not date_str:
            continue
        try:
            row_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        if row_date >= cutoff:
            out.append(row)
    return out


def _load_date_cache(path: pathlib.Path) -> dict[str, str]:
    """Return video_id -> upload_date (YYYYMMDD) from a previously saved live cache."""
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        clip_id = str(row.get("clip_id") or "").strip()
        video_id = clip_id[4:] if clip_id.startswith("ibo_") else clip_id
        upload_date = str(row.get("upload_date") or "").strip().replace("-", "")
        if video_id and len(upload_date) == 8 and upload_date.isdigit():
            cache[video_id] = upload_date
    return cache


def _save_live_cache(rows: list[dict], path: pathlib.Path) -> None:
    """Persist live-fetched rows so publish dates are not re-fetched next run."""
    if not rows:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"WARNING: Could not write live cache: {exc}", file=sys.stderr)


def _live_fetch(channel: str, window_days: int, date_cache: dict[str, str] | None = None) -> list[dict]:
    try:
        entries = _list_channel_entries(channel, 500)
        if date_cache:
            seeded = 0
            for entry in entries:
                video_id = str(entry.get("id") or "").strip()
                if video_id and video_id in date_cache:
                    cached_date = date_cache[video_id]
                    if not entry.get("upload_date"):
                        entry["upload_date"] = cached_date
                        seeded += 1
            if seeded:
                print(f"  Seeded {seeded} entries from date cache", file=sys.stderr)
        entries = _enrich_with_upload_dates(entries, channel)
    except Exception as exc:
        print(f"WARNING: Live channel fetch failed: {exc}", file=sys.stderr)
        return []

    rows: list[dict] = []
    for entry in entries:
        entry_date_val = _entry_date(entry)
        if entry_date_val is None:
            continue
        video_id = str(entry.get("id") or "").strip()
        source_url = (
            str(entry.get("webpage_url") or "").strip()
            or str(entry.get("url") or "").strip()
            or (f"https://www.youtube.com/shorts/{video_id}" if video_id else "")
        )
        if not source_url:
            continue
        rows.append({
            "source_url": source_url,
            "clip_id": f"ibo_{video_id}" if video_id else "",
            "title": str(entry.get("title") or "").strip(),
            "upload_date": entry_date_val.isoformat(),
            "view_count": int(entry.get("view_count") or 0),
            "duration": int(entry.get("duration") or 0),
        })
    return rows


def _build_clip_list(rows: list[dict]) -> str:
    lines: list[str] = []
    for i, row in enumerate(rows, start=1):
        title = str(row.get("title") or "").strip()
        upload_date = str(row.get("upload_date") or "").strip()
        duration = int(row.get("duration") or 0)
        lines.append(f"{i:3d}. {upload_date} · {duration:>3}s · {title}")
    return "\n".join(lines)


def _validate_profile(profile: dict) -> dict:
    if not isinstance(profile.get("summary"), str) or not profile["summary"].strip():
        raise ValueError("Profile missing summary")
    archetypes = profile.get("archetypes")
    if not isinstance(archetypes, list) or not archetypes:
        raise ValueError("Profile missing archetypes")
    cleaned: list[dict] = []
    for arch in archetypes:
        if not isinstance(arch, dict):
            continue
        name = str(arch.get("name") or "").strip().upper()
        description = str(arch.get("description") or "").strip()
        if not name or not description:
            continue
        cleaned.append({
            "name": name,
            "description": description,
            "transcript_signals": [str(s) for s in (arch.get("transcript_signals") or []) if s],
            "weight": round(d0.bounded_ratio(arch.get("weight")), 3),
        })
    if not cleaned:
        raise ValueError("No valid archetypes after cleaning")
    return {
        **profile,
        "archetypes": cleaned,
        "selection_rules": [str(r) for r in (profile.get("selection_rules") or []) if str(r).strip()],
        "avoid": [str(r) for r in (profile.get("avoid") or []) if str(r).strip()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Jynxzi clip profile from @Iboclipss.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--min-views", type=int, default=DEFAULT_MIN_VIEWS, help="Min view count (default: %(default)s)")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Lookback window in days (default: %(default)s)")
    parser.add_argument("--discovered-glob", default=DEFAULT_DISCOVERED_GLOB)
    parser.add_argument("--live-cache", default=str(DEFAULT_LIVE_CACHE), help="Path to live fetch cache (default: %(default)s)")
    parser.add_argument("--provider", default="anthropic", help="LLM provider (gemini/anthropic)")
    parser.add_argument("--model", default="claude-opus-4-8", help="LLM model override")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true", help="Print JSON to stdout, do not write")
    args = parser.parse_args()

    from config import settings

    print("Loading cached iboclipss rows…", file=sys.stderr)
    cached_rows = _load_cached_rows(args.discovered_glob)
    print(f"  Cached: {len(cached_rows)} rows", file=sys.stderr)

    date_cache = _load_date_cache(pathlib.Path(args.live_cache))
    print(f"  Date cache: {len(date_cache)} known entries ({args.live_cache})", file=sys.stderr)

    print(f"Fetching live channel entries from {args.channel}…", file=sys.stderr)
    live_rows = _live_fetch(args.channel, args.window_days, date_cache=date_cache)
    print(f"  Live: {len(live_rows)} rows", file=sys.stderr)

    if live_rows:
        _save_live_cache(live_rows, pathlib.Path(args.live_cache))
        print(f"  Live cache updated ({len(live_rows)} rows) → {args.live_cache}", file=sys.stderr)

    seen: set[str] = set()
    merged: list[dict] = []
    for row in live_rows + cached_rows:
        key = str(row.get("source_url") or row.get("clip_id") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(row)

    windowed = _filter_to_window(merged, args.window_days)
    if args.min_views > 0:
        windowed_filtered = [r for r in windowed if int(r.get("view_count") or 0) >= args.min_views]
        if not windowed_filtered:
            print(f"WARNING: No rows passed min-views={args.min_views}, ignoring filter", file=sys.stderr)
        else:
            windowed = windowed_filtered

    if not windowed:
        print(f"ERROR: No clips within the last {args.window_days} days. Cannot generate profile.", file=sys.stderr)
        return 1

    windowed.sort(key=lambda r: str(r.get("upload_date") or ""), reverse=True)
    print(
        f"Analyzing {len(windowed)} clips "
        f"(window={args.window_days}d, sorted by date, merged={len(merged)})",
        file=sys.stderr,
    )

    clip_list = _build_clip_list(windowed)
    prompt = _ANALYSIS_PROMPT.format(window_days=args.window_days, clip_list=clip_list)

    provider = args.provider or str(getattr(settings, "moment_finder_provider", "gemini") or "gemini")
    model = args.model or str(getattr(settings, "moment_finder_model", "") or "")
    resolved_provider = d0.normalize_provider(provider)
    resolved_model = d0.resolve_provider_model(resolved_provider, model)
    print(f"Calling {resolved_provider}/{resolved_model}…", file=sys.stderr)

    try:
        response_text, usage = d0.call_moment_finder_provider(
            prompt,
            provider=resolved_provider,
            model=resolved_model,
            temperature=0.0,
            max_tokens=args.max_tokens,
            timeout_sec=120.0,
            retries=1,
            retry_backoff_sec=10.0,
        )
    except Exception as exc:
        print(f"ERROR: LLM call failed: {exc}", file=sys.stderr)
        return 1

    try:
        raw_profile = d0.parse_json_payload(response_text)
        if not isinstance(raw_profile, dict):
            raise ValueError(f"LLM returned non-object JSON: {type(raw_profile)}")
        profile = _validate_profile(raw_profile)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: Profile validation failed: {exc}", file=sys.stderr)
        print(f"LLM response:\n{response_text[:2000]}", file=sys.stderr)
        return 1

    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    profile_out = {
        "version": 1,
        "streamer": "jynxzi",
        "generated_at": generated_at,
        "source": {
            "channel": args.channel,
            "clips_analyzed": len(windowed),
            "window_days": args.window_days,
            "provider": resolved_provider,
            "model": resolved_model,
        },
        "summary": profile["summary"],
        "archetypes": profile["archetypes"],
        "selection_rules": profile["selection_rules"],
        "avoid": profile["avoid"],
    }

    json_out = json.dumps(profile_out, ensure_ascii=False, indent=2)
    if args.dry_run:
        print(json_out)
        print(f"\n--- DRY RUN: not written to disk ---", file=sys.stderr)
        print(f"LLM usage: {usage}", file=sys.stderr)
    else:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_out + "\n", encoding="utf-8")
        print(f"Written → {out_path}", file=sys.stderr)
        print(f"LLM usage: {usage}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
