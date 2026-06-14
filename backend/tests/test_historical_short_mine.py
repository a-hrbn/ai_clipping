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
import statistics
import types

import pytest

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


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 (issue #77): transcription + goal-family + length-bucket per short.
# The facet is derived from the short's *own* transcript (no VOD attribution).
# Transcription is injected/faked here — these tests never touch the network or
# the real Whisper/ffmpeg stack.
# ─────────────────────────────────────────────────────────────────────────────


def test_short_report_classifies_goal_family_from_injected_transcript() -> None:
    # The facet's goal-family comes from the short's own transcript via
    # clip_strategy.infer_goal_family (transcript as topic): a clip whose
    # transcript reads as a reaction is classified "reaction".
    row = _short(video_id="reax")

    report = hsm.build_short_report(
        [row], transcribe=lambda r: "his reaction was absolutely priceless")

    assert report["shorts"][0]["goal_family"] == "reaction"


def test_length_bucket_splits_at_the_25_second_boundary() -> None:
    # Length-bucket is "kurz" below 25s and "lang" at/above it. The boundary is
    # exact: 24.9s -> kurz, 25.0s -> lang.
    kurz = _short(video_id="kurz", duration_s=24.9)
    lang = _short(video_id="lang", duration_s=25.0)

    report = hsm.build_short_report(
        [kurz, lang], transcribe=lambda r: "some spoken words")

    by_id = {s["short_id"]: s for s in report["shorts"]}
    assert by_id["clips:kurz"]["length_bucket"] == "kurz"
    assert by_id["clips:lang"]["length_bucket"] == "lang"


def test_report_carries_transcript_excerpt_and_no_chat_emotion_facet() -> None:
    # The report surfaces a transcript excerpt so each facet is auditable. Chat-
    # emotion is deliberately NOT a facet: the chat signal lives in the VOD, not
    # in the posted short, so the Mine must not fake a facet it cannot compute.
    row = _short(video_id="exc")

    report = hsm.build_short_report(
        [row], transcribe=lambda r: "the streamer breaks down the whole play")

    short = report["shorts"][0]
    assert "the streamer breaks down" in short["transcript_excerpt"]
    assert not any("chat" in key for key in short)


def test_untranscribable_short_is_dropped_and_surfaced_not_silently_emitted() -> None:
    # A short whose media cannot be transcribed (transcribe -> None) or yields a
    # blank transcript cannot be classified — it must be surfaced as a logged
    # drop, never emitted as a bogus "general" short. Both retention-present
    # shorts otherwise clear Slice 1, so only the facet stage drops them.
    ok = _short(video_id="ok")
    missing_media = _short(video_id="no_media")
    blank = _short(video_id="silent")

    def transcribe(row):
        return {"ok": "his honest take on the meta", "no_media": None, "silent": "   "}[
            row["video_id"]]

    report = hsm.build_short_report([ok, missing_media, blank], transcribe=transcribe)

    assert [s["short_id"] for s in report["shorts"]] == ["clips:ok"]
    assert {"short_id": "clips:no_media", "reason": "untranscribable"} in report["drops"]
    assert {"short_id": "clips:silent", "reason": "untranscribable"} in report["drops"]


def test_transcribe_short_composes_audio_extractor_then_transcriber(tmp_path) -> None:
    # The production transcription seam is mp4 -> wav (audio_extractor) -> text
    # (transcriber.transcribe), joining the segment texts. Both leaf seams are
    # injected here, so no ffmpeg / Whisper / network is touched.
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake mp4")
    seen = {}

    def fake_extract(video_path, job_id, *, output_path):
        seen["extract_src"] = video_path
        output_path.write_bytes(b"fake wav")
        return output_path

    def fake_transcribe(wav_path):
        # Read the wav now: transcribe_short cleans up its temp wav on return.
        seen["transcribe_bytes"] = wav_path.read_bytes()
        return [types.SimpleNamespace(text="hello"), types.SimpleNamespace(text="world")]

    text = hsm.transcribe_short(media, extract=fake_extract, transcribe=fake_transcribe)

    assert text == "hello world"
    assert seen["extract_src"] == media
    assert seen["transcribe_bytes"] == b"fake wav"  # transcriber reads the extracted wav


def test_transcribe_short_returns_none_for_missing_media(tmp_path) -> None:
    # A short whose media file is absent transcribes to None — build_short_report
    # then logs it as an untranscribable drop rather than aborting the run.
    absent = tmp_path / "absent.mp4"

    text = hsm.transcribe_short(
        absent, extract=_unreached, transcribe=_unreached)

    assert text is None


def _unreached(*_args, **_kwargs):
    raise AssertionError("leaf seam must not run for missing media")


def test_main_writes_facets_with_injected_transcription(tmp_path, capsys) -> None:
    # End-to-end through the CLI with transcription injected (no network): the
    # written per-short report carries goal-family, length-bucket and a
    # transcript excerpt, and an un-transcribable short surfaces as a drop.
    export = tmp_path / "retention_export.jsonl"
    rows = [_short(video_id="reax", duration_s=30.0), _short(video_id="mute")]
    export.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "report.jsonl"

    transcripts = {"reax": "his reaction was hilarious", "mute": None}
    rc = hsm.main(
        ["--metrics", str(export), "--output", str(out), "--json"],
        transcribe=lambda row: transcripts[row["video_id"]])

    assert rc == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [s["short_id"] for s in written] == ["clips:reax"]  # mute dropped
    assert written[0]["goal_family"] == "reaction"
    assert written[0]["length_bucket"] == "lang"
    assert "his reaction" in written[0]["transcript_excerpt"]
    payload = json.loads(capsys.readouterr().out)
    assert {"short_id": "clips:mute", "reason": "untranscribable"} in payload["drops"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 (issue #78): aggregate + τ-baseline + caveat-header (reports, routes not).
# The diagnostic roll-up over the per-short rows — overall retention spread, median
# retention per goal-family×length (with n), a τ-baseline percentile suggestion, the
# caveat header (n / packaging / dormant) and an honest drop summary. Deliberately
# NO gate/verdict/route: n=52 + packaging-confounding + dormancy allow only a
# directional read, not a decision. Hand-computed fixtures pin the exact aggregates.
# ─────────────────────────────────────────────────────────────────────────────


def _short_row(goal_family, length_bucket, **fields):
    """A per-short report row as build_short_report emits — facet + retention fields."""
    row = {
        "short_id": fields.pop("short_id", f"clips:{goal_family}_{length_bucket}"),
        "retention_3s": None,
        "retention_50pct": None,
        "completion": None,
        "goal_family": goal_family,
        "length_bucket": length_bucket,
    }
    row.update(fields)
    return row


def test_facet_medians_group_by_goal_family_and_length_with_correct_n() -> None:
    # Median retention is reported per (goal-family × length) cell with the cell's
    # n. reaction×kurz has three shorts -> median completion of [0.40,0.50,0.60] is
    # 0.50, n=3; quote×lang has one -> its own value, n=1. Hand-computed.
    shorts = [
        _short_row("reaction", "kurz", short_id="clips:a", completion=0.40),
        _short_row("reaction", "kurz", short_id="clips:b", completion=0.60),
        _short_row("reaction", "kurz", short_id="clips:c", completion=0.50),
        _short_row("quote", "lang", short_id="clips:d", completion=0.20),
    ]

    cells = hsm.facet_medians(shorts)

    by_facet = {(c["goal_family"], c["length_bucket"]): c for c in cells}
    assert by_facet[("reaction", "kurz")]["n"] == 3
    assert by_facet[("reaction", "kurz")]["median"]["completion"] == 0.50
    assert by_facet[("quote", "lang")]["n"] == 1
    assert by_facet[("quote", "lang")]["median"]["completion"] == 0.20


def test_retention_spread_reports_overall_variation_views_decoupled() -> None:
    # The headline "does retention vary meaningfully?" read: per field min / median /
    # max / iqr / stdev over all kept shorts. completion = [0.20,0.40,0.60,0.80] ->
    # median 0.50, iqr 0.30 (p75 0.65 − p25 0.35), stdev hand-checked. A degraded
    # (avp-only) short carries completion None and is skipped — never folded as a 0.
    shorts = [
        _short_row("reaction", "kurz", short_id="clips:a", completion=0.20),
        _short_row("reaction", "kurz", short_id="clips:b", completion=0.40),
        _short_row("quote", "lang", short_id="clips:c", completion=0.60),
        _short_row("quote", "lang", short_id="clips:d", completion=0.80),
        _short_row("story", "kurz", short_id="clips:e", completion=None),
    ]

    comp = hsm.retention_spread(shorts)["completion"]

    assert comp["n"] == 4  # the None-completion short is excluded, not counted as 0
    assert comp["min"] == 0.20
    assert comp["max"] == 0.80
    assert comp["median"] == 0.50
    assert comp["iqr"] == pytest.approx(0.30)
    assert comp["stdev"] == pytest.approx(statistics.stdev([0.20, 0.40, 0.60, 0.80]))


def test_tau_baseline_suggests_retention_percentiles_for_gate_fields() -> None:
    # The τ-baseline turns real retention into threshold suggestions for the forward
    # loop's gate. retention_3s = [0.50,0.70,0.80,0.90] -> p25 0.65, p50 0.75, p75
    # 0.825 (linear interpolation, hand-computed). It covers the gate fields
    # (retention_3s / completion), not retention_50pct — those are the loop's knobs.
    shorts = [
        _short_row("a", "kurz", short_id="clips:a", retention_3s=0.50, completion=0.20),
        _short_row("a", "kurz", short_id="clips:b", retention_3s=0.70, completion=0.40),
        _short_row("a", "kurz", short_id="clips:c", retention_3s=0.80, completion=0.60),
        _short_row("a", "kurz", short_id="clips:d", retention_3s=0.90, completion=0.80),
    ]

    tau = hsm.tau_baseline(shorts)

    assert tau["retention_3s"]["p25"] == pytest.approx(0.65)
    assert tau["retention_3s"]["p50"] == 0.75
    assert tau["retention_3s"]["p75"] == pytest.approx(0.825)
    assert set(tau) == {"retention_3s", "completion"}


def test_drop_summary_breaks_down_each_bottleneck_with_honest_denominator() -> None:
    # Every bottleneck is surfaced with an honest denominator: n_input is kept +
    # dropped (not just the survivors), and each drop reason — missing retention vs
    # un-transcribable (= no transcript, so no facet either) — is counted, so a
    # silent cap can never read as "all covered".
    report = {
        "shorts": [_short_row("reaction", "kurz", short_id="clips:keep")],
        "drops": [
            {"short_id": "clips:no_ret", "reason": "missing_retention_row"},
            {"short_id": "clips:mute", "reason": "untranscribable"},
            {"short_id": "clips:mute2", "reason": "untranscribable"},
        ],
    }

    summary = hsm.drop_summary(report)

    assert summary["n_input"] == 4  # 1 kept + 3 dropped — the honest denominator
    assert summary["n_kept"] == 1
    assert summary["n_dropped"] == 3
    assert summary["by_reason"]["missing_retention_row"] == 1
    assert summary["by_reason"]["untranscribable"] == 2


def test_aggregate_carries_caveat_header_and_composes_subaggregates() -> None:
    # The aggregate header carries the three caveats explicitly (n / packaging /
    # dormant) so "a facet leads" can never be over-read, and it composes the
    # spread, facet medians, τ-baseline and drop summary into one report.
    report = {
        "shorts": [
            _short_row("reaction", "kurz", short_id="clips:a", retention_3s=0.80, completion=0.40),
            _short_row("reaction", "kurz", short_id="clips:b", retention_3s=0.90, completion=0.60),
        ],
        "drops": [{"short_id": "clips:x", "reason": "missing_retention_row"}],
    }

    agg = hsm.aggregate(report)

    blob = " ".join(agg["caveats"]).lower()
    assert "n=52" in blob          # thin sample → directional
    assert "packaging" in blob     # packaging-confounded → suggestive
    assert "dormant" in blob       # dormant channel → lukewarm
    assert agg["n_shorts"] == 2
    assert agg["retention_spread"]["completion"]["n"] == 2
    assert agg["facet_medians"][0]["goal_family"] == "reaction"
    assert "p50" in agg["tau_baseline"]["retention_3s"]
    assert agg["drop_summary"]["n_dropped"] == 1


def test_aggregate_reports_but_does_not_route_no_gate_or_verdict() -> None:
    # The module reports, it does not route: unlike b_narrow_probe /
    # strong_model_window_probe there is NO pre-fixed gate, verdict or route in the
    # output — n=52 + packaging-confounding + dormancy allow only a directional read.
    report = {
        "shorts": [_short_row("reaction", "kurz", short_id="clips:a", completion=0.5)],
        "drops": [],
    }

    agg = hsm.aggregate(report)

    def _all_keys(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield key
                yield from _all_keys(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from _all_keys(value)

    keys = [k.lower() for k in _all_keys(agg)]
    assert not any("verdict" in k or "route" in k for k in keys)


def test_main_writes_aggregate_artifact_and_includes_it_in_json(tmp_path, capsys) -> None:
    # End-to-end through the CLI (no network): the aggregate is written as its own
    # JSON artifact and rides along in the --json payload. Both shorts are 30s
    # (lang) reactions; completion [0.40, 0.70] -> facet-cell median 0.55, n=2.
    export = tmp_path / "retention_export.jsonl"
    rows = [
        _short(video_id="strong", duration_s=30.0, audience_retention=_STRONG_CURVE),
        _short(video_id="plain", duration_s=30.0),
    ]
    export.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "report.jsonl"
    agg_out = tmp_path / "aggregate.json"

    rc = hsm.main(
        ["--metrics", str(export), "--output", str(out),
         "--aggregate-output", str(agg_out), "--json"],
        transcribe=lambda row: "his reaction was hilarious")

    assert rc == 0
    aggregate = json.loads(agg_out.read_text(encoding="utf-8"))
    assert any("n=52" in caveat for caveat in aggregate["caveats"])
    cell = aggregate["facet_medians"][0]
    assert (cell["goal_family"], cell["length_bucket"], cell["n"]) == ("reaction", "lang", 2)
    assert cell["median"]["completion"] == pytest.approx(0.55)
    payload = json.loads(capsys.readouterr().out)
    assert payload["aggregate"]["n_shorts"] == 2
    assert payload["aggregate"]["drop_summary"]["n_input"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 (issue #79): reconcile the REAL YouTube-Studio export against the parser.
# The operator's export is a German Studio "Tabellendaten.csv": German headers, a
# leading "Gesamt" totals row, and titles quoted because they contain commas. The
# reconciliation surprise is that the default column set carries NO retention curve
# and no avp column. The format-fix reads the real CSV into the row shape
# derive_retention already consumes, locating columns by NAME so an added avp column
# can shift positions without breaking the map. The header fixture mirrors the real
# export byte-for-byte (tmp/.../Tabellendaten.csv).
# ─────────────────────────────────────────────────────────────────────────────

_STUDIO_HEADER = (
    "Videos,Videotitel,Veröffentlichungszeitpunkt des Videos,Dauer,Aufrufe,"
    "Wiedergabezeit (Stunden),Abonnenten,Impressionen,Klickrate der Impressionen (%)"
)


def test_read_studio_csv_maps_real_columns_and_skips_totals_row(tmp_path) -> None:
    # The real Studio export leads with a "Gesamt" totals row and quotes titles that
    # contain commas. read_studio_csv drops the totals row and maps the German columns
    # onto the row shape the mine consumes (video_id / duration_s / views).
    csv_path = tmp_path / "Tabellendaten.csv"
    csv_path.write_text(
        _STUDIO_HEADER + "\n"
        "Gesamt,,,,32757,86.4206,6,11754,6.3\n"
        'Ws649UG936E,"Jynxzi Crashes Out, In Chess #shorts","Mar 29, 2026",10,2396,4.3651,0,1104,17.66\n',
        encoding="utf-8")

    rows = hsm.read_studio_csv(csv_path)

    assert len(rows) == 1  # the "Gesamt" totals row is dropped, not parsed as a short
    row = rows[0]
    assert row["video_id"] == "Ws649UG936E"
    assert row["duration_s"] == 10.0
    assert row["views"] == 2396.0


def test_read_studio_csv_reads_endwatch_completion_and_avp_columns(tmp_path) -> None:
    # The rich re-export adds "Bis zum Ende angesehen (%)" (watched-to-end, a real
    # completion signal) and "Durchschnittliche Wiedergabedauer in Prozent (%)" (avp,
    # loop-inflatable past 100%). read_studio_csv surfaces both by EXACT column name —
    # avp's column must not be confused with the bare "Durchschnittliche
    # Wiedergabedauer" (a h:mm:ss duration sitting right beside it).
    csv_path = tmp_path / "Tabellendaten.csv"
    csv_path.write_text(
        "Videos,Dauer,Bis zum Ende angesehen (%),Durchschnittliche Wiedergabedauer,"
        "Durchschnittliche Wiedergabedauer in Prozent (%),Aufrufe\n"
        "Gesamt,,62.74,0:00:17,76.61,32757\n"
        "Ws649UG936E,10,72.45,0:00:14,149.22,2396\n",
        encoding="utf-8")

    rows = hsm.read_studio_csv(csv_path)

    assert len(rows) == 1
    assert rows[0]["watched_to_end_pct"] == 72.45
    assert rows[0]["averageViewPercentage"] == 149.22  # avp, loop-inflated past 100%


def test_derive_retention_reads_completion_from_studio_end_watch_without_a_curve() -> None:
    # The rich Studio export has no curve but carries watched-to-end completion + avp.
    # derive_retention reads completion off the end-watch field (62% -> 0.62), leaves
    # the curve-only fields (3s / 50pct) None, carries avp, and flags the source as the
    # studio export — never "curve" (there is none) and never "degraded_avp_only"
    # (completion IS present, so it is not avp-only).
    row = {"video_id": "x", "duration_s": 23.0,
           "watched_to_end_pct": 62.0, "averageViewPercentage": 85.5}

    derived = hsm.derive_retention(row)

    assert derived["completion"] == 0.62
    assert derived["retention_3s"] is None
    assert derived["retention_50pct"] is None
    assert derived["averageViewPercentage"] == 85.5
    assert derived["retention_source"] == "studio_endwatch"


def test_main_runs_on_a_studio_csv_export_and_spreads_completion_and_avp(tmp_path, capsys) -> None:
    # End-to-end on the REAL export shape: a ``.csv`` metrics file routes through the
    # Studio reader (no curve), each short carries watched-to-end completion + avp, and
    # the aggregate spreads both. Two shorts: completion [0.60, 0.70] -> median 0.65;
    # avp present for both. Offline — a plain local CSV read, no network.
    export = tmp_path / "Tabellendaten.csv"
    export.write_text(
        "Videos,Dauer,Bis zum Ende angesehen (%),"
        "Durchschnittliche Wiedergabedauer in Prozent (%),Aufrufe\n"
        "Gesamt,,65.0,90.0,5000\n"
        "aaa,20,60.0,80.0,1000\n"
        "bbb,30,70.0,120.0,2000\n",
        encoding="utf-8")
    out = tmp_path / "report.jsonl"
    agg_out = tmp_path / "aggregate.json"

    rc = hsm.main(["--metrics", str(export), "--output", str(out),
                   "--aggregate-output", str(agg_out), "--json"])

    assert rc == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [s["short_id"] for s in written] == ["clips:aaa", "clips:bbb"]
    assert [s["completion"] for s in written] == [0.60, 0.70]
    assert [s["retention_source"] for s in written] == ["studio_endwatch", "studio_endwatch"]
    aggregate = json.loads(agg_out.read_text(encoding="utf-8"))
    assert aggregate["retention_spread"]["completion"]["median"] == pytest.approx(0.65)
    assert aggregate["retention_spread"]["averageViewPercentage"]["n"] == 2


def test_read_studio_csv_keeps_the_video_title_for_title_based_faceting(tmp_path) -> None:
    # The Videotitel column is kept verbatim so the facet's goal-family can be read off
    # the title (packaging) without transcribing media — the cheap facet path for the
    # confounded pre-look (#79). The "Gesamt" totals row carries no title and is dropped.
    csv_path = tmp_path / "Tabellendaten.csv"
    csv_path.write_text(
        "Videos,Videotitel,Dauer,Aufrufe\n"
        "Gesamt,,,32757\n"
        'abc,"Jynxzi Crashes Out In Chess 💀 #funny #rage",30,1000\n',
        encoding="utf-8")

    rows = hsm.read_studio_csv(csv_path)

    assert rows[0]["title"] == "Jynxzi Crashes Out In Chess 💀 #funny #rage"


def test_main_facets_from_titles_on_a_studio_csv_without_media(tmp_path, capsys) -> None:
    # --facet-from-title reads the goal-family off the title (packaging) — no media, no
    # Whisper. A "#funny #rage" short classifies "reaction"; length from Dauer (30s ->
    # lang); the title rides along as the auditable excerpt. The cheap facet for #79.
    export = tmp_path / "Tabellendaten.csv"
    export.write_text(
        "Videos,Videotitel,Dauer,Bis zum Ende angesehen (%),Aufrufe\n"
        "Gesamt,,,62.0,5000\n"
        'reax,"Jynxzi Crashes Out In Chess 💀 #funny #rage",30,70.0,1000\n',
        encoding="utf-8")
    out = tmp_path / "report.jsonl"

    rc = hsm.main(["--metrics", str(export), "--output", str(out),
                   "--aggregate-output", str(tmp_path / "agg.json"),
                   "--facet-from-title", "--json"])

    assert rc == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert written[0]["goal_family"] == "reaction"
    assert written[0]["length_bucket"] == "lang"  # 30s >= 25s boundary
    assert "Crashes Out" in written[0]["transcript_excerpt"]


def test_aggregate_spreads_avp_so_a_curveless_run_surfaces_signal() -> None:
    # The real Studio export has NO retention curve — only averageViewPercentage
    # survives, so the three curve fields are None and the aggregate must still
    # surface a spread over avp (the only live retention signal). avp = [30,45,55,70]
    # -> median 50, min 30, max 70; the curve-field spreads stay empty (n=0), never
    # faked as a 0 that would invent variation.
    shorts = [
        _short_row("reaction", "kurz", short_id="clips:a", averageViewPercentage=30.0),
        _short_row("reaction", "kurz", short_id="clips:b", averageViewPercentage=45.0),
        _short_row("quote", "lang", short_id="clips:c", averageViewPercentage=55.0),
        _short_row("quote", "lang", short_id="clips:d", averageViewPercentage=70.0),
    ]

    spread = hsm.retention_spread(shorts)

    avp = spread["averageViewPercentage"]
    assert avp["n"] == 4
    assert avp["min"] == 30.0
    assert avp["median"] == 50.0
    assert avp["max"] == 70.0
    assert spread["completion"]["n"] == 0  # no curve → curve field empty, not faked


def test_facet_medians_include_avp_for_the_curveless_facet_tendency() -> None:
    # #75's question is goal-family×length, so the facet cells must read off avp in the
    # curve-less export. reaction×kurz avp [40,60] -> median 50; quote×lang avp [20] ->
    # 20. Each cell's median dict carries avp alongside the (here-empty) curve fields.
    shorts = [
        _short_row("reaction", "kurz", short_id="clips:a", averageViewPercentage=40.0),
        _short_row("reaction", "kurz", short_id="clips:b", averageViewPercentage=60.0),
        _short_row("quote", "lang", short_id="clips:c", averageViewPercentage=20.0),
    ]

    cells = hsm.facet_medians(shorts)

    by_facet = {(c["goal_family"], c["length_bucket"]): c for c in cells}
    assert by_facet[("reaction", "kurz")]["median"]["averageViewPercentage"] == 50.0
    assert by_facet[("quote", "lang")]["median"]["averageViewPercentage"] == 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Slice 5 (issue #82): within-bucket early-hook spread + double-report + Stage-2
# trigger + forward-loop reference. The aggregate's core Stage-1 read: does
# retention_3s spread WITHIN a length-bucket (kurz <25s / lang >=25s) at all, or
# cluster as tightly as completion? Loop-inflated Shorts curves are length-biased,
# so the comparison is *within* buckets. Reported twice (all shorts + the >=50-views
# subset), with an explicit Stage-2 trigger flag. Reports, routes not. Hand-computed.
# ─────────────────────────────────────────────────────────────────────────────


def test_early_hook_spread_reports_retention_3s_within_each_length_bucket() -> None:
    # retention_3s spread is read WITHIN each length-bucket. kurz r3s
    # [0.50,0.70,1.10,1.50] (one loop-inflated past 1.0) -> n=4, median 0.90, max
    # 1.50 (passed through, never clamped), iqr 0.55 (p75 1.20 - p25 0.65). lang r3s
    # [0.80,0.90] -> n=2, median 0.85. Hand-computed.
    # The bucket is derived from duration_s (kurz <25s / lang >=25s), NOT the facet
    # length_bucket column — that column is transcription-gated and None in the
    # curve-only Stage-1 read this slice serves.
    shorts = [
        _short_row("a", None, short_id="clips:a", retention_3s=0.50, views=1000, duration_s=10.0),
        _short_row("a", None, short_id="clips:b", retention_3s=0.70, views=1000, duration_s=10.0),
        _short_row("a", None, short_id="clips:c", retention_3s=1.10, views=1000, duration_s=10.0),
        _short_row("a", None, short_id="clips:d", retention_3s=1.50, views=1000, duration_s=10.0),
        _short_row("a", None, short_id="clips:e", retention_3s=0.80, views=1000, duration_s=30.0),
        _short_row("a", None, short_id="clips:f", retention_3s=0.90, views=1000, duration_s=30.0),
    ]

    spread = hsm.early_hook_spread(shorts)

    by_bucket = {b["length_bucket"]: b for b in spread["buckets"]}
    assert by_bucket["kurz"]["all"]["n"] == 4
    assert by_bucket["kurz"]["all"]["median"] == pytest.approx(0.90)
    assert by_bucket["kurz"]["all"]["max"] == 1.50  # loop-inflated, passed through
    assert by_bucket["kurz"]["all"]["iqr"] == pytest.approx(0.55)
    assert by_bucket["lang"]["all"]["n"] == 2
    assert by_bucket["lang"]["all"]["median"] == pytest.approx(0.85)


def test_early_hook_spread_double_reports_all_vs_min_views_subset() -> None:
    # Double report: the spread is read once over ALL shorts (honest denominator) and
    # once over the >=50-views subset (reach-starved shorts are retention-noise). A
    # short with <50 views drops from the filtered read but stays in the all read —
    # never silently dropped. kurz: A(views 100, r3s 0.80) + B(views 10, r3s 0.50).
    shorts = [
        _short_row("a", None, short_id="clips:a", retention_3s=0.80, views=100, duration_s=10.0),
        _short_row("a", None, short_id="clips:b", retention_3s=0.50, views=10, duration_s=10.0),
    ]

    spread = hsm.early_hook_spread(shorts)

    kurz = {b["length_bucket"]: b for b in spread["buckets"]}["kurz"]
    assert kurz["all"]["n"] == 2              # both kept in the honest denominator
    assert kurz["all"]["median"] == pytest.approx(0.65)
    assert kurz["min_views"]["n"] == 1        # the <50-views short drops from the filtered read
    assert kurz["min_views"]["median"] == 0.80
    assert spread["view_floor"] == 50         # the floor is surfaced, not hidden


def test_stage2_trigger_is_eng_when_retention_3s_clusters_tightly() -> None:
    # When retention_3s clusters as tightly as completion WITHIN every length-bucket
    # (iqr below the band), the early-hook is no lever: the flag reads "eng" and #75's
    # "-> Auto-Packaging" routing stands. kurz r3s [0.80,0.82,0.84,0.86] -> iqr 0.03 <
    # 0.10 band. Reports, it does not route.
    shorts = [
        _short_row("a", None, short_id=f"clips:{i}", retention_3s=r, views=1000, duration_s=10.0)
        for i, r in enumerate([0.80, 0.82, 0.84, 0.86])
    ]

    spread = hsm.early_hook_spread(shorts)

    assert spread["stage2_trigger"] == "eng"
    assert {b["length_bucket"]: b["spreads"] for b in spread["buckets"]} == {"kurz": False}


def test_stage2_trigger_is_spreizt_when_retention_3s_spreads_within_a_bucket() -> None:
    # When retention_3s spreads within a length-bucket past the band, Stage 2 is worth
    # it: the flag reads "spreizt". kurz r3s [0.50,0.70,1.10,1.50] -> iqr 0.55 > 0.10
    # band (one value loop-inflated past 1.0, passed through). Reports, it does not route.
    shorts = [
        _short_row("a", None, short_id=f"clips:{i}", retention_3s=r, views=1000, duration_s=10.0)
        for i, r in enumerate([0.50, 0.70, 1.10, 1.50])
    ]

    spread = hsm.early_hook_spread(shorts)

    assert spread["stage2_trigger"] == "spreizt"
    assert {b["length_bucket"]: b["spreads"] for b in spread["buckets"]} == {"kurz": True}
    assert spread["spread_band"] == pytest.approx(0.10)


def test_aggregate_carries_forward_loop_thresholds_as_loop_uncalibrated_reference() -> None:
    # The forward loop's hard-coded gate (ml/import_yt_retention.py: 3s<0.5 / 50pct<0.4 /
    # completion>0.6) is surfaced as a DOCUMENTED REFERENCE only — never applied here, and
    # marked loop-uncalibrated (Shorts curves are loop-inflated, so the fixed 0-1 thresholds
    # don't transfer; the percentile-τ within-bucket read is operative).
    report = {
        "shorts": [_short_row("a", "kurz", short_id="clips:a", retention_3s=0.8, views=100)],
        "drops": [],
    }

    ref = hsm.aggregate(report)["forward_loop_reference"]

    assert ref["thresholds"]["retention_3s_lt"] == 0.5
    assert ref["thresholds"]["retention_50pct_lt"] == 0.4
    assert ref["thresholds"]["completion_gt"] == 0.6
    assert ref["calibration"] == "loop-uncalibrated"
    assert "import_yt_retention" in ref["source"]


def test_aggregate_composes_early_hook_spread_and_surfaces_the_view_floor_caveat() -> None:
    # The aggregate composes the within-bucket early-hook spread (with its Stage-2
    # trigger) and surfaces the view-floor in the caveat header, so the double-report's
    # filtered half can never be over-read as a silent drop. Two reactions, kurz, r3s
    # [0.80,0.90] both >=50 views -> the early_hook_spread rides along under the caveats.
    report = {
        "shorts": [
            _short_row("reaction", None, short_id="clips:a", retention_3s=0.80, views=1000, duration_s=10.0),
            _short_row("reaction", None, short_id="clips:b", retention_3s=0.90, views=1000, duration_s=10.0),
        ],
        "drops": [],
    }

    agg = hsm.aggregate(report)

    assert "early_hook_spread" in agg
    assert agg["early_hook_spread"]["stage2_trigger"] in {"eng", "spreizt"}
    assert agg["early_hook_spread"]["buckets"][0]["length_bucket"] == "kurz"
    blob = " ".join(agg["caveats"]).lower()
    assert "view-floor" in blob   # the view floor is surfaced as a caveat
    assert "50" in blob


def test_aggregate_with_early_hook_spread_still_reports_no_gate_or_verdict() -> None:
    # Slice 2 adds the Stage-2 *trigger* flag, but the module still reports — it does
    # not route. No key in the extended aggregate names a verdict or a route; the flag
    # is "stage2_trigger" / "spreads", never a baked decision.
    report = {
        "shorts": [_short_row("reaction", "kurz", short_id="clips:a", retention_3s=0.8, views=100)],
        "drops": [],
    }

    agg = hsm.aggregate(report)

    def _all_keys(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield key
                yield from _all_keys(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from _all_keys(value)

    keys = [k.lower() for k in _all_keys(agg)]
    assert not any("verdict" in k or "route" in k for k in keys)
