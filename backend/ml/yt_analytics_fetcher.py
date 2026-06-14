"""Real read-only YT-Analytics fetcher adapter — fulfils the #81 ``Fetcher`` seam.

PRD #80 (Step 0.5), issue #83 (Slice 3). #81 closed the curve-driven Mine behind ONE
injectable seam — ``Fetcher = video_id -> [{elapsedVideoTimeRatio, audienceWatchRatio}]``
— and ran it offline against a fake. This module supplies the *real* adapter that is
drop-in for that fake: it calls ``youtubeAnalytics.reports.query`` read-only
(scope ``yt-analytics.readonly``), consuming an operator-provided ``client_secret.json``,
and maps the ~100-point ``audienceWatchRatio`` curve onto the seam point shape.

``relativeRetentionPerformance`` is **not** requested — it is unavailable for Shorts.
Loop-inflated values (``audienceWatchRatio`` > 1.0, the Shorts replay) are mapped
verbatim, never clamped. A short the API returns no rows for maps to ``[]`` — the seam's
falsy "no curve" signal that the unchanged Mine degrades honestly.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Callable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.pull_yt_retention_curve import Fetcher  # the #81 seam this adapter fulfils

# Read-only YouTube Analytics scope — the adapter only ever queries, never writes.
READONLY_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
# Cached OAuth token, gitignored via the `yt_*token*.json` pattern (see .gitignore).
DEFAULT_TOKEN = BACKEND_ROOT / "ml" / "yt_analytics_token.json"

# The query seam: a ``youtubeAnalytics.reports.query``-shaped callable. Injected as a
# fake in tests (canned result table, no network); ``authorized_query`` supplies the
# real read-only one. Returns the API's result-table dict (columnHeaders + rows).
QueryFn = Callable[..., dict]

# The lifetime window: retention is cumulative over the video's life, so the default
# start predates YouTube and the end is resolved per call to "today".
_LIFETIME_START = "2005-01-01"


def map_response_to_curve(response: dict) -> list[dict]:
    """Map a ``reports.query`` result table onto the #81 seam point shape.

    Locates the ``elapsedVideoTimeRatio`` / ``audienceWatchRatio`` columns via the
    response's ``columnHeaders`` (never a positional assumption) and emits one
    ``{elapsedVideoTimeRatio, audienceWatchRatio}`` point per row, values verbatim.
    """
    headers = [h["name"] for h in response.get("columnHeaders", [])]
    if "elapsedVideoTimeRatio" not in headers or "audienceWatchRatio" not in headers:
        return []  # no-data response → the seam's falsy "no curve", never an abort
    e_idx = headers.index("elapsedVideoTimeRatio")
    a_idx = headers.index("audienceWatchRatio")
    return [
        {"elapsedVideoTimeRatio": row[e_idx], "audienceWatchRatio": row[a_idx]}
        for row in (response.get("rows") or [])
    ]


def build_fetcher(query: QueryFn, *, start_date: str = _LIFETIME_START,
                  end_date: str | None = None) -> Fetcher:
    """Wrap a ``reports.query`` callable into the #81 ``Fetcher`` seam — drop-in for the fake.

    The returned ``fetch_curve(video_id)`` issues the read-only audience-retention query
    for that one video (``ids=channel==MINE``, ``dimensions=elapsedVideoTimeRatio``,
    ``metrics=audienceWatchRatio``, ``filters=video==<id>``) over the lifetime window and
    maps the result table onto the seam point shape. ``end_date`` defaults to today.
    """
    def fetch_curve(video_id: str) -> list[dict]:
        if not (video_id or "").strip():
            return []  # no video_id → no query, no malformed "video==" filter
        end = end_date or datetime.date.today().isoformat()
        response = query(
            ids="channel==MINE",
            dimensions="elapsedVideoTimeRatio",
            metrics="audienceWatchRatio",
            filters=f"video=={video_id}",
            startDate=start_date,
            endDate=end,
        )
        return map_response_to_curve(response)

    return fetch_curve


# ─────────────────────────────────────────────────────────────────────────────
# OAuth transport — the real read-only query (HITL: consumes operator credentials)
# ─────────────────────────────────────────────────────────────────────────────


def authorized_query(client_secret: str | pathlib.Path,
                     token: str | pathlib.Path | None = DEFAULT_TOKEN,
                     *, scopes: list[str] | None = None) -> QueryFn:
    """Build the real read-only ``youtubeAnalytics.reports.query`` callable.

    Consumes the operator-provided ``client_secret.json`` (the GCP project + enabled
    YT-Analytics API + OAuth desktop client + consent are the manual HITL precondition,
    not this code's job), runs the installed-app OAuth flow once, and caches the token at
    ``token`` (gitignored) for re-use. The google client libs are imported lazily so this
    module — and the offline unit tests — import without them; install via the ``yt``
    optional group. Returns a ``QueryFn`` suitable for :func:`build_fetcher`.
    """
    # Lazy: keeps the module offline-importable; only the spike path needs these.
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    scopes = scopes or [READONLY_SCOPE]
    token_path = pathlib.Path(token) if token else None

    creds = None
    if token_path and token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), scopes)
            creds = flow.run_local_server(port=0)
        if token_path:
            token_path.write_text(creds.to_json(), encoding="utf-8")

    service = build("youtubeAnalytics", "v2", credentials=creds)

    def query(**params: object) -> dict:
        return service.reports().query(**params).execute()

    return query


# ─────────────────────────────────────────────────────────────────────────────
# Spike-gate CLI — run once against ONE real Short to confirm the API serves the curve
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None, *, query: QueryFn | None = None) -> int:
    """Spike-gate runner: fetch ONE Short's retention curve via the real adapter.

    Acceptance #5: run against the highest-views Short → the ~100-point curve comes back
    and matches the Studio UI (the parity eyeball is the operator's manual step). An empty
    curve means the API serves no Shorts retention → the A-branch is dead (back to B).
    ``query`` is injectable for offline tests; the CLI builds the real one from the creds.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--client-secret", required=True,
                        help="Operator-provided OAuth desktop client_secret.json (gitignored).")
    parser.add_argument("--video-id", required=True,
                        help="The Short to spike against (use the highest-views one).")
    parser.add_argument("--token", type=pathlib.Path, default=DEFAULT_TOKEN,
                        help="Cached OAuth token path (gitignored). Reused across runs.")
    parser.add_argument("--start-date", default=_LIFETIME_START)
    parser.add_argument("--end-date", default=None, help="Defaults to today.")
    parser.add_argument("--json", dest="output_json", action="store_true",
                        help="Print the full mapped curve as JSON.")
    args = parser.parse_args(argv)

    if query is None:
        query = authorized_query(args.client_secret, args.token)
    fetch_curve = build_fetcher(query, start_date=args.start_date, end_date=args.end_date)

    curve = fetch_curve(args.video_id)

    if args.output_json:
        print(json.dumps(curve, ensure_ascii=True))
        return 0 if curve else 1

    if not curve:
        print(f"yt_analytics_fetcher: SPIKE FAILED — no curve for video {args.video_id!r}. "
              "The API serves no Shorts retention → A-branch dead, fall back to B.",
              file=sys.stderr)
        return 1

    ratios = [p["audienceWatchRatio"] for p in curve]
    print(f"yt_analytics_fetcher: SPIKE OK — {len(curve)} points for video {args.video_id!r}")
    print(f"  audienceWatchRatio: min={min(ratios):.3f} max={max(ratios):.3f} "
          f"(>1.0 = loop-inflated, expected for Shorts)")
    print(f"  head: {curve[0]}")
    print(f"  tail: {curve[-1]}")
    print("  → eyeball parity against the Studio UI retention graph for this Short.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
