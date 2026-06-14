"""Tests for the real read-only YT-Analytics fetcher adapter — ml/yt_analytics_fetcher.py.

PRD #80 (Step 0.5), issue #83 (Slice 3). #81 closed the curve-driven Mine behind ONE
injectable ``Fetcher`` seam (``video_id -> [{elapsedVideoTimeRatio, audienceWatchRatio}]``)
and ran it offline against a fake. This slice supplies the *real* read-only adapter that
satisfies that seam contract drop-in: it maps a ``youtubeAnalytics.reports.query`` result
table onto the seam point shape and wraps an injected query callable into a ``Fetcher``.

The tests pin the adapter through its public interface against a CANNED API response — no
network, no creds. The OAuth/transport (``authorized_query``) is the HITL spike and is not
exercised here; the query callable is injected as a fake so the mapping + wiring are tested
in isolation. Prior art: tests/test_pull_yt_retention_curve.py.
"""
from ml import pull_yt_retention_curve as pull
from ml import yt_analytics_fetcher as fetcher


def _result_table(points):
    """A youtubeAnalytics.reports.query result table for the given (ratio, watch) points."""
    return {
        "kind": "youtubeAnalytics#resultTable",
        "columnHeaders": [
            {"name": "elapsedVideoTimeRatio", "columnType": "DIMENSION", "dataType": "FLOAT"},
            {"name": "audienceWatchRatio", "columnType": "METRIC", "dataType": "FLOAT"},
        ],
        "rows": [[ratio, watch] for ratio, watch in points],
    }


def test_map_response_to_curve_maps_result_table_to_seam_points() -> None:
    # The tracer bullet: a ~100-point reports.query result table (elapsedVideoTimeRatio
    # dimension + audienceWatchRatio metric) is mapped onto the exact seam point shape
    # the #81 fetcher contract expects — {elapsedVideoTimeRatio, audienceWatchRatio}.
    points = [(i / 100.0, 1.0 - i / 200.0) for i in range(101)]  # 101 pts, 0.00..1.00

    curve = fetcher.map_response_to_curve(_result_table(points))

    assert len(curve) == 101
    assert curve[0] == {"elapsedVideoTimeRatio": 0.0, "audienceWatchRatio": 1.0}
    assert curve[10] == {"elapsedVideoTimeRatio": 0.1, "audienceWatchRatio": 0.95}
    assert curve[-1] == {"elapsedVideoTimeRatio": 1.0, "audienceWatchRatio": 0.5}


def test_map_passes_loop_inflated_watch_ratio_through_verbatim() -> None:
    # Shorts replay, so audienceWatchRatio runs > 1.0 early on. The adapter maps those
    # values straight through — never clamped to 1.0, never raised — exactly as the #81
    # seam contract passes loop-inflated curves verbatim.
    points = [(0.0, 1.92), (0.1, 1.40), (0.5, 1.05), (1.0, 0.83)]

    curve = fetcher.map_response_to_curve(_result_table(points))

    assert [p["audienceWatchRatio"] for p in curve] == [1.92, 1.40, 1.05, 0.83]


def test_map_returns_empty_for_a_short_the_api_has_no_curve_for() -> None:
    # A short with no retention data (too few views, or not this channel's video) must
    # degrade to [] — the seam's falsy "no curve" signal the unchanged Mine handles —
    # not raise and abort the whole pull. Covers both the empty-rows table and a
    # bare/no-data response that omits the expected columns altogether.
    assert fetcher.map_response_to_curve(_result_table([])) == []
    assert fetcher.map_response_to_curve({"kind": "youtubeAnalytics#resultTable"}) == []
    assert fetcher.map_response_to_curve({}) == []


def _recording_query(response):
    """A fake reports.query that records its kwargs and returns a canned result table."""
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        return response

    return query, calls


def test_build_fetcher_queries_read_only_for_the_video_and_returns_the_mapped_curve() -> None:
    # build_fetcher turns an injected reports.query callable into the #81 Fetcher seam:
    # fetch_curve(video_id) issues the read-only retention query for that one video
    # (elapsedVideoTimeRatio dimension, audienceWatchRatio metric, channel==MINE,
    # video==<id>) and returns the curve mapped onto the seam point shape.
    query, calls = _recording_query(_result_table([(0.0, 1.0), (0.5, 0.6), (1.0, 0.4)]))
    fetch_curve = fetcher.build_fetcher(query, start_date="2024-01-01", end_date="2026-06-14")

    curve = fetch_curve("vid123")

    assert curve == [
        {"elapsedVideoTimeRatio": 0.0, "audienceWatchRatio": 1.0},
        {"elapsedVideoTimeRatio": 0.5, "audienceWatchRatio": 0.6},
        {"elapsedVideoTimeRatio": 1.0, "audienceWatchRatio": 0.4},
    ]
    assert len(calls) == 1
    kw = calls[0]
    assert kw["ids"] == "channel==MINE"
    assert kw["dimensions"] == "elapsedVideoTimeRatio"
    assert kw["metrics"] == "audienceWatchRatio"        # NOT relativeRetentionPerformance
    assert kw["filters"] == "video==vid123"
    assert kw["startDate"] == "2024-01-01"
    assert kw["endDate"] == "2026-06-14"


def test_build_fetcher_skips_the_query_for_a_blank_video_id() -> None:
    # A row with no video_id must not issue a malformed "video==" query; it short-circuits
    # to [] (the seam's no-curve signal) without touching the network at all.
    query, calls = _recording_query(_result_table([(0.0, 1.0)]))
    fetch_curve = fetcher.build_fetcher(query)

    assert fetch_curve("") == []
    assert calls == []


def test_build_fetcher_is_drop_in_for_the_fake_in_the_curve_driven_mine() -> None:
    # Acceptance #1: the real adapter is drop-in for the #81 fake. Injected straight into
    # pull.pull_report it drives curve-source retention for the matched video and degrades
    # the rest — same contract the fake satisfied, now off a reports.query result table.
    def query(*, filters, **_):
        if filters == "video==abc":
            return _result_table([(0.0, 1.0), (0.1, 0.80), (0.5, 0.55), (1.0, 0.40)])
        return _result_table([])  # the API has no curve for the other short

    studio_rows = [
        {"video_id": "abc", "duration_s": 30.0, "views": 1000, "watched_to_end_pct": 50.0},
        {"video_id": "nope", "duration_s": 20.0, "views": 50, "watched_to_end_pct": 62.0},
    ]

    report = pull.pull_report(studio_rows, fetch_curve=fetcher.build_fetcher(query))

    by_id = {s["short_id"]: s for s in report["shorts"]}
    assert by_id["clips:abc"]["retention_source"] == "curve"
    assert by_id["clips:abc"]["retention_3s"] == 0.80   # 3s/30s = 0.1 -> 0.80, off the curve
    assert by_id["clips:abc"]["completion"] == 0.40
    assert by_id["clips:nope"]["retention_source"] == "studio_endwatch"  # degraded, kept
    assert report["drops"] == []


def test_spike_gate_passes_when_the_api_returns_a_curve(capsys) -> None:
    # Acceptance #5 (pass branch): the spike CLI, given an injected query that yields a
    # curve for the one Short, exits 0 and reports the point count + loop-inflation hint
    # so the operator can eyeball parity with Studio. No network — query is injected.
    table = _result_table([(0.0, 1.90), (0.5, 0.9), (1.0, 0.6)])  # loop-inflated head

    rc = fetcher.main(["--client-secret", "cs.json", "--video-id", "topShort"],
                      query=lambda **_: table)

    assert rc == 0
    out = capsys.readouterr().out
    assert "3 points" in out and "topShort" in out
    assert "max=1.900" in out  # loop-inflated value surfaced, not clamped


def test_spike_gate_fails_signalling_a_dead_when_the_api_serves_no_curve(capsys) -> None:
    # Acceptance #5 (fail branch): if the API serves no Shorts retention curve, the spike
    # exits non-zero and says the A-branch is dead (fall back to B) — never silently "ok".
    rc = fetcher.main(["--client-secret", "cs.json", "--video-id", "topShort"],
                      query=lambda **_: _result_table([]))

    assert rc == 1
    assert "A-branch dead" in capsys.readouterr().err
