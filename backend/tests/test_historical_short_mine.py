"""Tests for the historical-short-mine retention ingest + per-short report
(ml/historical_short_mine.py, PRD #75 Slice 1, issue #76).

This slice builds the read-only Mine skeleton end-to-end: a tolerant parser turns
a locally exported YT audience-retention curve into per-short retention fields
(retention_3s/retention_50pct/completion), reads them views-decoupled, and emits
a per-short report keyed by row_identity. Facet columns stay empty until Slice 2.

The tests pin external behaviour through the public interface: small hand-built
fixtures in, exact retention fields / drops / report rows out. Prior art:
tests/test_outcome_view_weighting.py, tests/test_gap_decomposition.py.
"""
import json

from ml import historical_short_mine as hsm
from ml import row_identity


# A canned YT-relative audience-retention curve. Points land exactly on the query
# ratios (3s/30s = 0.1, 0.5, 1.0) so the expected fields are read off by hand.
_CURVE = [
    {"elapsedVideoTimeRatio": 0.0, "audienceWatchRatio": 1.00},
    {"elapsedVideoTimeRatio": 0.1, "audienceWatchRatio": 0.80},
    {"elapsedVideoTimeRatio": 0.5, "audienceWatchRatio": 0.55},
    {"elapsedVideoTimeRatio": 1.0, "audienceWatchRatio": 0.40},
]


def _short(**overrides):
    row = {
        "video_id": "abc123",
        "duration_s": 30.0,
        "averageViewPercentage": 60.0,
        "views": 1000,
        "audience_retention": [dict(p) for p in _CURVE],
    }
    row.update(overrides)
    return row


def test_derive_retention_reads_three_fields_off_the_curve() -> None:
    # 3s into a 30s short is elapsed ratio 0.1 (audienceWatchRatio 0.80); the 50%
    # point is 0.55; completion is the curve's last point (1.0 -> 0.40).
    derived = hsm.derive_retention(_short())

    assert derived["retention_3s"] == 0.80
    assert derived["retention_50pct"] == 0.55
    assert derived["completion"] == 0.40
    assert derived["retention_source"] == "curve"


def test_derive_retention_interpolates_between_curve_points() -> None:
    # 3s into a 60s short is elapsed ratio 0.05 — halfway between the 0.0 (1.00)
    # and 0.1 (0.80) curve points, so retention_3s is the linear midpoint 0.90.
    derived = hsm.derive_retention(_short(duration_s=60.0))

    assert derived["retention_3s"] == 0.90


def test_derive_retention_degrades_when_only_avg_view_percentage_present() -> None:
    # No curve, only averageViewPercentage: the three curve points cannot be
    # derived, so they stay null and the row is flagged degraded — never raised.
    derived = hsm.derive_retention(_short(audience_retention=[], averageViewPercentage=42.0))

    assert derived["retention_source"] == "degraded_avp_only"
    assert derived["retention_3s"] is None
    assert derived["retention_50pct"] is None
    assert derived["completion"] is None
    assert derived["averageViewPercentage"] == 42.0


# A strong-retention curve (completion 0.70) so the views-decoupling case is
# unambiguous: this short would clear any retention bar yet has tiny views.
_STRONG_CURVE = [
    {"elapsedVideoTimeRatio": 0.0, "audienceWatchRatio": 1.00},
    {"elapsedVideoTimeRatio": 0.1, "audienceWatchRatio": 0.95},
    {"elapsedVideoTimeRatio": 0.5, "audienceWatchRatio": 0.85},
    {"elapsedVideoTimeRatio": 1.0, "audienceWatchRatio": 0.70},
]


def test_build_short_report_keeps_strong_retention_below_baseline_views() -> None:
    # Views-decoupled (ADR 0001 cold-start win): a short with strong retention but
    # views far below baseline must stay in the report — no views gate, no label
    # drop. Views ride along only as a secondary field.
    row = _short(video_id="weak_views", audience_retention=_STRONG_CURVE,
                 views=10, baseline_views=100_000)

    report = hsm.build_short_report([row])

    assert report["drops"] == []
    assert len(report["shorts"]) == 1
    short = report["shorts"][0]
    assert short["completion"] == 0.70
    assert short["views"] == 10.0
    # Facet columns are deferred to Slice 2 — present but empty here.
    assert short["goal_family"] is None
    assert short["length_bucket"] is None
    assert short["transcript_excerpt"] is None


def test_build_short_report_records_missing_retention_row_as_drop() -> None:
    # A short with no retention signal at all (no curve, no averageViewPercentage)
    # is a missing retention row: surfaced as a logged drop, never silently
    # discarded and never emitted as a bogus short. Degraded (avp-only) still
    # counts as present — the drop is only the truly-empty row.
    kept = _short(video_id="has_curve")
    missing = _short(video_id="no_retention", audience_retention=[], averageViewPercentage=None)

    report = hsm.build_short_report([kept, missing])

    assert [s["short_id"] for s in report["shorts"]] == ["clips:has_curve"]
    assert report["drops"] == [{"short_id": "clips:no_retention", "reason": "missing_retention_row"}]


def test_report_is_keyed_by_row_identity_and_ordered_for_idempotent_reruns() -> None:
    # short_id is the row_identity primary key (stable + join-able with a later
    # VOD attribution, not an incidental string). Ordering by short_id is
    # input-order independent, so a re-run over the same export is idempotent.
    a = _short(video_id="zzz")
    b = _short(video_id="aaa")

    forward = hsm.build_short_report([a, b])
    reversed_ = hsm.build_short_report([b, a])

    assert forward["shorts"][0]["short_id"] == row_identity.primary_identity_key({"clip_ids": ["aaa"]})
    assert [s["short_id"] for s in forward["shorts"]] == ["clips:aaa", "clips:zzz"]
    assert forward["shorts"] == reversed_["shorts"]


def test_main_runs_offline_writes_jsonl_and_prints_json(tmp_path, capsys) -> None:
    # End-to-end: read a local export (no network, no paid call), write the
    # per-short report as JSONL and print it with --json. A degraded short is
    # kept, a missing-retention short is dropped+surfaced.
    export = tmp_path / "retention_export.jsonl"
    rows = [
        _short(video_id="good"),
        {"video_id": "deg", "duration_s": 20.0, "averageViewPercentage": 40.0, "views": 5},
        {"video_id": "gone", "duration_s": 20.0},
    ]
    export.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "report.jsonl"

    rc = hsm.main(["--metrics", str(export), "--output", str(out), "--json"])

    assert rc == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [s["short_id"] for s in written] == ["clips:deg", "clips:good"]  # sorted; missing dropped
    payload = json.loads(capsys.readouterr().out)
    assert [d["short_id"] for d in payload["drops"]] == ["clips:gone"]

    # Idempotent re-run over the same export -> byte-identical output.
    first = out.read_bytes()
    hsm.main(["--metrics", str(export), "--output", str(out)])
    assert out.read_bytes() == first
