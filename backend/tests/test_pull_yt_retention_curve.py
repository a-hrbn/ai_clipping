"""Tests for the curve-driven Mine — ml/pull_yt_retention_curve.py (PRD #80, issue #81).

Slice 1 closes the end-to-end path "curve -> retention fields" behind ONE injectable
fetcher seam, fully offline against a fake (no network, no creds). The module reads the
existing curve-less Studio-CSV rows, calls the fetcher per row, merges the returned
``audienceWatchRatio`` curve onto the row by ``video_id``, and runs the UNCHANGED
``historical_short_mine`` over it. The result is per-short rows whose
``retention_3s/50pct/completion`` are interpolated from the real curve (instead of the
completion-only read of #75).

The tests pin external behaviour through the public interface: canned Studio rows + canned
curves (fetcher injected/fake) in, exact retention fields / flags / drops out. Prior art:
tests/test_historical_short_mine.py.
"""
import pytest

from ml import pull_yt_retention_curve as pull


# A canned YT-relative audience-retention curve. Points land exactly on the query
# ratios (3s/30s = 0.1, 0.5, 1.0) so the expected retention fields are read off by hand.
_CURVE = [
    {"elapsedVideoTimeRatio": 0.0, "audienceWatchRatio": 1.00},
    {"elapsedVideoTimeRatio": 0.1, "audienceWatchRatio": 0.80},
    {"elapsedVideoTimeRatio": 0.5, "audienceWatchRatio": 0.55},
    {"elapsedVideoTimeRatio": 1.0, "audienceWatchRatio": 0.40},
]


def test_pull_report_merges_fetched_curve_and_drives_retention_from_it() -> None:
    # The tracer bullet: a curve-less Studio row + a fetched curve, merged by video_id,
    # makes the unchanged mine read retention off the CURVE (retention_source "curve",
    # 3s/50pct/completion interpolated) rather than the completion-only studio read.
    studio_rows = [
        {"video_id": "abc", "duration_s": 30.0, "views": 1000, "watched_to_end_pct": 50.0},
    ]

    report = pull.pull_report(studio_rows, fetch_curve=lambda video_id: _CURVE)

    short = report["shorts"][0]
    assert short["short_id"] == "clips:abc"
    assert short["retention_source"] == "curve"
    assert short["retention_3s"] == 0.80   # 3s/30s = 0.1 -> 0.80
    assert short["retention_50pct"] == 0.55
    assert short["completion"] == 0.40


# A loop-inflated curve: Shorts replay, so the watch-ratio runs > 1.0 early on. The
# points land on the same query ratios (0.1, 0.5, 1.0) for a hand-read expectation.
_LOOP_CURVE = [
    {"elapsedVideoTimeRatio": 0.0, "audienceWatchRatio": 1.90},
    {"elapsedVideoTimeRatio": 0.1, "audienceWatchRatio": 1.50},
    {"elapsedVideoTimeRatio": 0.5, "audienceWatchRatio": 1.10},
    {"elapsedVideoTimeRatio": 1.0, "audienceWatchRatio": 0.80},
]


def test_loop_inflated_curve_passes_through_and_interpolates_retention_3s() -> None:
    # A Shorts curve loops past 100% (values > 1.0). It is merged verbatim and the
    # unchanged mine interpolates retention_3s straight off it — never clamped to 1.0,
    # never raised as an error. 3s/30s = 0.1 -> the curve's 1.50 point.
    studio_rows = [{"video_id": "loop", "duration_s": 30.0, "views": 5000}]

    report = pull.pull_report(studio_rows, fetch_curve=lambda video_id: _LOOP_CURVE)

    short = report["shorts"][0]
    assert short["retention_source"] == "curve"
    assert short["retention_3s"] == 1.50            # loop-inflated, passed through
    assert short["completion"] == 0.80


def test_short_without_a_curve_falls_to_the_degraded_path_flagged_not_aborted() -> None:
    # The fetcher returns no curve for one short. It must NOT abort the run: that short
    # falls to the existing degraded Studio read (watched-to-end -> studio_endwatch,
    # or avp-only -> degraded_avp_only), is flagged via retention_source, and is kept.
    studio_rows = [
        {"video_id": "has", "duration_s": 30.0, "views": 100, "watched_to_end_pct": 50.0},
        {"video_id": "endwatch", "duration_s": 20.0, "views": 50,
         "watched_to_end_pct": 62.0, "averageViewPercentage": 80.0},
        {"video_id": "avp_only", "duration_s": 15.0, "views": 7, "averageViewPercentage": 44.0},
    ]
    curves = {"has": _CURVE}  # the other two get no curve from the fetcher

    report = pull.pull_report(studio_rows, fetch_curve=lambda video_id: curves.get(video_id))

    by_id = {s["short_id"]: s for s in report["shorts"]}
    assert by_id["clips:has"]["retention_source"] == "curve"
    assert by_id["clips:endwatch"]["retention_source"] == "studio_endwatch"  # degraded, flagged
    assert by_id["clips:endwatch"]["completion"] == 0.62                      # kept, not dropped
    assert by_id["clips:avp_only"]["retention_source"] == "degraded_avp_only"
    assert by_id["clips:avp_only"]["averageViewPercentage"] == 44.0
    assert report["drops"] == []  # nothing aborted, nothing silently discarded


def test_pull_report_is_deterministic_and_idempotent_via_row_identity() -> None:
    # Short identity is the row_identity primary key (clips:<video_id>), and the report
    # is ordered by it — so a re-run over the same rows in any order yields an identical
    # report. A deterministic fetcher therefore makes the whole pull idempotent.
    studio_rows = [
        {"video_id": "zzz", "duration_s": 30.0, "views": 1},
        {"video_id": "aaa", "duration_s": 30.0, "views": 2},
    ]
    fetch = lambda video_id: _CURVE

    forward = pull.pull_report(studio_rows, fetch_curve=fetch)
    reversed_ = pull.pull_report(list(reversed(studio_rows)), fetch_curve=fetch)

    assert [s["short_id"] for s in forward["shorts"]] == ["clips:aaa", "clips:zzz"]
    assert forward["shorts"] == reversed_["shorts"]


def _studio_csv(tmp_path):
    # The real curve-less German Studio export shape (read_studio_csv): a "Gesamt"
    # totals row plus two shorts carrying watched-to-end + avp but NO curve.
    export = tmp_path / "Tabellendaten.csv"
    export.write_text(
        "Videos,Dauer,Bis zum Ende angesehen (%),"
        "Durchschnittliche Wiedergabedauer in Prozent (%),Aufrufe\n"
        "Gesamt,,65.0,90.0,5000\n"
        "withcurve,30,60.0,80.0,1000\n"
        "nocurve,20,70.0,120.0,2000\n",
        encoding="utf-8")
    return export


def test_main_runs_offline_against_a_fake_fetcher_and_prints_json(tmp_path, capsys) -> None:
    # End-to-end through the CLI, no network: a curve-less Studio CSV is read, the fake
    # fetcher supplies a curve for one short and nothing for the other. The written
    # report has the curve short curve-driven and the other degraded+flagged; the
    # degraded short is logged (stderr), never silently dropped; --json carries the
    # aggregate. A re-run over the same inputs is byte-identical.
    import json

    export = _studio_csv(tmp_path)
    out = tmp_path / "report.jsonl"
    agg_out = tmp_path / "aggregate.json"
    curves = {"withcurve": _CURVE}

    rc = pull.main(
        ["--metrics", str(export), "--output", str(out),
         "--aggregate-output", str(agg_out), "--json"],
        fetch_curve=lambda video_id: curves.get(video_id))

    assert rc == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {s["short_id"]: s for s in written}
    assert set(by_id) == {"clips:withcurve", "clips:nocurve"}
    assert by_id["clips:withcurve"]["retention_source"] == "curve"
    assert by_id["clips:withcurve"]["retention_3s"] == 0.80
    assert by_id["clips:nocurve"]["retention_source"] == "studio_endwatch"  # degraded

    captured = capsys.readouterr()
    assert "nocurve" in captured.err  # the no-curve short is surfaced, not silent
    payload = json.loads(captured.out)
    assert payload["aggregate"]["n_shorts"] == 2
    assert payload["aggregate"]["retention_spread"]["retention_3s"]["n"] == 1  # only the curve short

    first = out.read_bytes()
    pull.main(
        ["--metrics", str(export), "--output", str(out), "--aggregate-output", str(agg_out)],
        fetch_curve=lambda video_id: curves.get(video_id))
    assert out.read_bytes() == first  # idempotent re-run


def test_default_fetcher_is_wired_into_main_but_defers_the_network_adapter(tmp_path) -> None:
    # The seam ships with a default so production needs no injection, but the real
    # read-only YT-Analytics adapter is a later slice. A run with NO injected fetcher
    # must therefore refuse *clearly* (NotImplementedError) rather than hit the network
    # or silently return no curve — proving the wiring is present and offline-safe.
    export = _studio_csv(tmp_path)

    with pytest.raises(NotImplementedError):
        pull.main(["--metrics", str(export), "--output", str(tmp_path / "o.jsonl"),
                   "--aggregate-output", str(tmp_path / "a.json")])
