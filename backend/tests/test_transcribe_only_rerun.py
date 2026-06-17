from __future__ import annotations

import json
import pathlib

from services.transcriber import RichSegment, format_jsonl

from ml import transcribe_only_rerun as tor


def _records(*segments: RichSegment) -> list[dict]:
    """Canonical segments.jsonl records, produced by the real writer."""
    return [json.loads(line) for line in format_jsonl(list(segments)).splitlines()]


def test_split_by_local_video_separates_anchors_with_and_without_video(tmp_path) -> None:
    # AC5: the anchors split into those with a local vod_<id>.mp4 (re-transcribable)
    # and those without — the latter surfaced as excluded, never silently dropped.
    root = tmp_path / "twitch"
    root.mkdir()
    (root / "vod_111.mp4").write_bytes(b"x")
    (root / "vod_333.mp4").write_bytes(b"x")
    # 222 has no local video file

    with_video, without_video = tor.split_by_local_video(["111", "222", "333"], root)

    assert [vod for vod, _ in with_video] == ["111", "333"]
    assert [path.name for _, path in with_video] == ["vod_111.mp4", "vod_333.mp4"]
    assert without_video == ["222"]


def test_retranscribe_vod_writes_fresh_segments_without_touching_pool(tmp_path) -> None:
    # AC1: transcribe-only re-run writes a fresh segments.jsonl into the driver's
    # OWN dir; the candidate pool (moments_raw) and the base dir's own segments
    # are never opened for write.
    predictions = tmp_path / "neu"
    pool_debug = predictions / "pipeline_debug_BASE_111" / "debug"
    pool_debug.mkdir(parents=True)
    (pool_debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        '{"start_time":1}\n', encoding="utf-8")
    (pool_debug / "segments.jsonl").write_text(
        '{"start":0,"end":1,"text":"stale"}\n', encoding="utf-8")
    pool_before = (pool_debug / "llm_moment_finder_moments_raw.jsonl").read_bytes()
    base_segments_before = (pool_debug / "segments.jsonl").read_bytes()

    canned = tor.TranscribeOutput(
        segments=[
            RichSegment(start=10.0, end=20.0, text="fresh one"),
            RichSegment(start=20.0, end=35.0, text="fresh two"),
        ],
        duration_sec=3600.0,
    )

    def fake_transcribe(vod_id: str, video_path: pathlib.Path) -> tor.TranscribeOutput:
        assert vod_id == "111"
        return canned

    path = tor.retranscribe_vod(
        "111", tmp_path / "vod_111.mp4", predictions, transcribe_vod=fake_transcribe)

    assert path == tor.fresh_segments_path(predictions, "111")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [(r["start"], r["end"], r["text"]) for r in records] == [
        (10.0, 20.0, "fresh one"),
        (20.0, 35.0, "fresh two"),
    ]
    # Candidate pool + base segments are byte-for-byte untouched.
    assert (pool_debug / "llm_moment_finder_moments_raw.jsonl").read_bytes() == pool_before
    assert (pool_debug / "segments.jsonl").read_bytes() == base_segments_before


def test_retranscribe_vod_persists_duration_meta(tmp_path) -> None:
    # The VOD length must outlive the in-memory run so the consistency report can
    # flag truncation on a resumed / report-only invocation (no second media pass).
    predictions = tmp_path / "neu"
    canned = tor.TranscribeOutput(
        segments=[RichSegment(start=0.0, end=10.0, text="hi")], duration_sec=3600.0)

    tor.retranscribe_vod(
        "111", tmp_path / "vod_111.mp4", predictions, transcribe_vod=lambda v, p: canned)

    meta = json.loads(tor.fresh_meta_path(predictions, "111").read_text(encoding="utf-8"))
    assert meta["vod_id"] == "111"
    assert meta["duration_sec"] == 3600.0
    assert meta["segment_count"] == 1


def test_load_report_input_reads_records_and_duration_from_disk(tmp_path) -> None:
    # The report runs off disk (resume / report-only): the loader round-trips the
    # writer's segments.jsonl back into canonical records plus the persisted length.
    predictions = tmp_path / "neu"
    canned = tor.TranscribeOutput(
        segments=[
            RichSegment(start=0.0, end=5.0, text="a", words_per_second=2.0),
            RichSegment(start=5.0, end=9.0, text="b", words_per_second=2.0),
        ],
        duration_sec=1800.0,
    )
    tor.retranscribe_vod(
        "111", tmp_path / "vod_111.mp4", predictions, transcribe_vod=lambda v, p: canned)

    records, duration = tor.load_report_input(predictions, "111")

    assert duration == 1800.0
    assert [(r["start"], r["end"], r["text"]) for r in records] == [(0.0, 5.0, "a"), (5.0, 9.0, "b")]
    assert all(frozenset(r.keys()) == tor.EXPECTED_SCHEMA_KEYS for r in records)


def test_retranscribe_vod_skips_already_fresh_vod(tmp_path) -> None:
    # AC2: resume — a VOD whose fresh segments already exist is skipped, not
    # re-transcribed. A transcribe leaf that explodes proves it is never called.
    predictions = tmp_path / "neu"
    path = tor.fresh_segments_path(predictions, "111")
    path.parent.mkdir(parents=True)
    path.write_text('{"start":0,"end":1,"text":"already fresh"}\n', encoding="utf-8")

    def boom(vod_id: str, video_path: pathlib.Path) -> tor.TranscribeOutput:
        raise AssertionError("must not re-transcribe an already-fresh VOD")

    result = tor.retranscribe_vod(
        "111", tmp_path / "vod_111.mp4", predictions, transcribe_vod=boom)

    assert result == path
    assert path.read_text(encoding="utf-8") == '{"start":0,"end":1,"text":"already fresh"}\n'


def test_retranscribe_anchors_completes_with_video_excludes_without_logs_drops(tmp_path, capsys) -> None:
    # AC4/AC5: the with-video anchors are re-transcribed (completed) or, on a
    # transcribe failure, recorded as a logged drop (the rest keep going); the
    # no-video anchors are surfaced as excluded, never silently dropped.
    root = tmp_path / "twitch"
    root.mkdir()
    (root / "vod_ok.mp4").write_bytes(b"x")
    (root / "vod_boom.mp4").write_bytes(b"x")
    # "novideo" has no local file -> excluded
    predictions = tmp_path / "neu"

    def fake_transcribe(vod_id: str, video_path: pathlib.Path) -> tor.TranscribeOutput:
        if vod_id == "boom":
            raise RuntimeError("kapow")
        return tor.TranscribeOutput(
            segments=[RichSegment(start=0.0, end=1.0, text="hi")], duration_sec=10.0)

    result = tor.retranscribe_anchors(
        ["ok", "novideo", "boom"], predictions, root, transcribe_vod=fake_transcribe)

    assert result.completed == ["ok"]
    assert result.excluded == ["novideo"]
    assert result.drops == ["boom"]
    assert tor.fresh_segments_path(predictions, "ok").exists()
    assert not tor.fresh_segments_path(predictions, "boom").exists()
    err = capsys.readouterr().err
    assert "novideo" in err  # exclusion surfaced
    assert "boom" in err     # drop logged


def test_default_transcribe_vod_is_plain_no_glossary_no_diarization(tmp_path) -> None:
    # AC1/AC6: the real adapter transcribes PLAIN — diarization off, no glossary
    # initial_prompt, and no stale chunk cache (so a previously glossary-biased
    # cache cannot contaminate the fresh uniform transcript). The model is left at
    # the inherited large-v3-turbo (the adapter never names a model).
    calls: dict = {}

    def recording_transcribe(audio_path, **kwargs):
        calls["audio_path"] = audio_path
        calls["kwargs"] = kwargs
        return [RichSegment(start=0.0, end=2.0, text="plain text")]

    def fake_extract(video_path, job_id, output_path):
        return output_path

    output = tor._default_transcribe_vod(
        "111",
        tmp_path / "vod_111.mp4",
        extract_audio=fake_extract,
        transcribe=recording_transcribe,
        probe_duration=lambda _p: 1234.0,
        work_dir=tmp_path,
    )

    assert calls["kwargs"]["enable_diarization"] is False
    assert calls["kwargs"]["initial_prompt"] == ""
    assert calls["kwargs"]["cache_key"] is None
    assert output.duration_sec == 1234.0
    assert [s.text for s in output.segments] == ["plain text"]


def test_report_flags_schema_mismatch_passes_uniform(tmp_path) -> None:
    # AC3: schema-uniformity — a VOD whose records drift from the canonical
    # segments.jsonl key set is flagged; a uniform VOD passes that check.
    good = _records(
        RichSegment(start=0.0, end=5.0, text="hi", words_per_second=2.0),
        RichSegment(start=5.0, end=10.0, text="yo", words_per_second=2.0),
    )
    drifted = _records(RichSegment(start=0.0, end=5.0, text="hi", words_per_second=2.0))
    drifted[0].pop("voice_energy_burst_score")  # a missing field => schema drift

    report = tor.build_consistency_report({"good": (good, 10.0), "bad": (drifted, 10.0)})

    by_id = {r.vod_id: r for r in report.vod_reports}
    assert "schema_mismatch" not in by_id["good"].flags
    assert "schema_mismatch" in by_id["bad"].flags


def test_report_flags_late_start_and_truncation_against_duration() -> None:
    # AC3: coverage-span — a transcript whose first segment starts well after 0
    # (late start) or whose last segment ends well before the VOD length
    # (truncation) is flagged; one covering ~the whole VOD passes.
    full = _records(
        RichSegment(start=0.0, end=5.0, text="a", words_per_second=2.0),
        RichSegment(start=5.0, end=3590.0, text="b", words_per_second=2.0),
    )
    late = _records(
        RichSegment(start=300.0, end=305.0, text="a", words_per_second=2.0),
        RichSegment(start=305.0, end=3590.0, text="b", words_per_second=2.0),
    )
    truncated = _records(
        RichSegment(start=0.0, end=5.0, text="a", words_per_second=2.0),
        RichSegment(start=5.0, end=1000.0, text="b", words_per_second=2.0),
    )

    report = tor.build_consistency_report(
        {"full": (full, 3600.0), "late": (late, 3600.0), "trunc": (truncated, 3600.0)},
        late_start_max_sec=120.0,
        coverage_floor=0.95,
    )

    by = {r.vod_id: r.flags for r in report.vod_reports}
    assert "late_start" not in by["full"]
    assert "truncated" not in by["full"]
    assert "late_start" in by["late"]
    assert "truncated" in by["trunc"]


def test_report_flags_over_run_past_vod_length() -> None:
    # AC3 (coverage-span, upper bound): a transcript whose last segment ends far
    # PAST the VOD length — e.g. stale leftover chunks of a longer VOD spliced onto
    # this timeline — is flagged over_run; one ending at ~the real length passes.
    # This is the contamination class the report previously let through silently.
    clean = _records(
        RichSegment(start=0.0, end=3580.0, text="a", words_per_second=2.0),
    )
    over = _records(
        RichSegment(start=0.0, end=3580.0, text="a", words_per_second=2.0),
        RichSegment(start=7000.0, end=9000.0, text="leftover-other-stream", words_per_second=2.0),
    )

    report = tor.build_consistency_report(
        {"clean": (clean, 3600.0), "over": (over, 3600.0)},
        min_vods_for_density=99,  # isolate the coverage check from density
    )

    by = {r.vod_id: r.flags for r in report.vod_reports}
    assert "over_run" not in by["clean"]
    assert "over_run" in by["over"]


def test_late_start_default_is_chunk_granularity_aware() -> None:
    # A speechless intro/standby chunk lands the first segment ~one chunk in
    # (chunk_seconds), so a flat sub-chunk threshold would flag every VOD with a
    # brief intro. By default the late-start threshold sits above the chunk
    # granularity: a brief multi-chunk intro must NOT flag, while a transcript
    # starting many chunks in (likely dropped head audio) still does.
    intro = _records(  # first speech ~one 180s intro chunk in — benign
        RichSegment(start=181.0, end=185.0, text="a", words_per_second=2.0),
        RichSegment(start=185.0, end=3590.0, text="b", words_per_second=2.0),
    )
    dropped_head = _records(  # first speech ~10 chunks in — suspicious
        RichSegment(start=1800.0, end=1805.0, text="a", words_per_second=2.0),
        RichSegment(start=1805.0, end=3590.0, text="b", words_per_second=2.0),
    )

    report = tor.build_consistency_report(
        {"intro": (intro, 3600.0), "dropped": (dropped_head, 3600.0)},
        min_vods_for_density=99,
    )

    by = {r.vod_id: r.flags for r in report.vod_reports}
    assert "late_start" not in by["intro"]
    assert "late_start" in by["dropped"]


def _evenly_spread_vod(n_segments: int, wps: float, duration: float = 3600.0) -> tuple[list[dict], float]:
    """A VOD whose segments span ~the full duration (no coverage confound)."""
    step = duration / n_segments
    segs = [
        RichSegment(start=round(i * step, 2), end=round(i * step + 0.5, 2),
                    text="w", words_per_second=wps)
        for i in range(n_segments)
    ]
    return (_records(*segs), duration)


def test_report_flags_density_outlier_against_cross_vod_band() -> None:
    # AC3: density-outlier — words/sec and segments/hour should sit in a consistent
    # band across VODs; a VOD whose density is far off the cross-VOD median (a proxy
    # for model/quality drift, since no model field is persisted) is flagged, while
    # the in-band VODs pass.
    per_vod = {
        "n1": _evenly_spread_vod(60, 2.0),
        "n2": _evenly_spread_vod(58, 2.1),
        "n3": _evenly_spread_vod(62, 1.9),
        "drift": _evenly_spread_vod(600, 2.0),  # ~10x denser → outlier
    }

    report = tor.build_consistency_report(per_vod, density_band_ratio=2.5)

    by = {r.vod_id: r.flags for r in report.vod_reports}
    assert "density_outlier" in by["drift"]
    assert all("density_outlier" not in by[v] for v in ("n1", "n2", "n3"))


def test_report_flags_unreplayable_empty_transcript() -> None:
    # AC3: replayability — a transcript that yields no usable timeline (empty, or a
    # zero/negative span the replay loader can't snap against) is flagged; a real
    # transcript passes.
    good = _records(
        RichSegment(start=0.0, end=5.0, text="a", words_per_second=2.0),
        RichSegment(start=5.0, end=3500.0, text="b", words_per_second=2.0),
    )

    report = tor.build_consistency_report(
        {"good": (good, 3600.0), "empty": ([], 3600.0)},
        min_vods_for_density=99,  # disable density so it can't confound this check
    )

    by = {r.vod_id: r.flags for r in report.vod_reports}
    assert "unreplayable" not in by["good"]
    assert "unreplayable" in by["empty"]


def test_report_is_fail_closed_on_any_flag_or_exclusion() -> None:
    # AC4: fail-closed — one flagged VOD, or any surfaced exclusion, makes the whole
    # report NOT clean; the individual flags are surfaced (a single VOD can carry
    # several), and exclusions are carried through, never folded into a silent pass.
    good = _records(
        RichSegment(start=0.0, end=5.0, text="a", words_per_second=2.0),
        RichSegment(start=5.0, end=3500.0, text="b", words_per_second=2.0),
    )

    all_good = tor.build_consistency_report(
        {"a": (good, 3600.0), "b": (good, 3600.0)}, min_vods_for_density=99)
    assert all_good.clean is True

    # A VOD that is both late-starting and truncated surfaces BOTH flags.
    late_and_truncated = _records(
        RichSegment(start=300.0, end=305.0, text="a", words_per_second=2.0),
        RichSegment(start=305.0, end=1000.0, text="b", words_per_second=2.0),
    )
    flagged = tor.build_consistency_report(
        {"a": (good, 3600.0), "bad": (late_and_truncated, 3600.0)},
        late_start_max_sec=120.0, min_vods_for_density=99)
    assert flagged.clean is False
    bad = next(r for r in flagged.vod_reports if r.vod_id == "bad")
    assert "late_start" in bad.flags and "truncated" in bad.flags

    # Every transcript passes, but excluded anchors keep the report NOT clean.
    with_excluded = tor.build_consistency_report(
        {"a": (good, 3600.0)}, excluded=["nov1", "nov2"], min_vods_for_density=99)
    assert with_excluded.clean is False
    assert with_excluded.excluded == ["nov1", "nov2"]


def test_cli_retranscribes_then_reports_and_exits_fail_closed(tmp_path, capsys) -> None:
    # AC4/AC5/AC6 end-to-end (net-free): the CLI re-transcribes the with-video
    # anchors, reports over the fresh transcripts, surfaces the no-video anchor as
    # excluded, and returns a non-zero exit so a non-clean report can't pass silently.
    root = tmp_path / "twitch"
    root.mkdir()
    (root / "vod_111.mp4").write_bytes(b"x")
    (root / "vod_222.mp4").write_bytes(b"x")
    # "333" has no local video -> excluded -> report not clean -> rc != 0
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id,vod_date,note\n111,,\n222,,\n333,,\n", encoding="utf-8")
    predictions = tmp_path / "neu"

    def fake_transcribe(vod_id: str, video_path: pathlib.Path) -> tor.TranscribeOutput:
        segs = [
            RichSegment(start=float(i) * 180.0, end=float(i) * 180.0 + 170.0,
                        text="hi there", words_per_second=2.0)
            for i in range(10)  # spans ~0..1790 of 1800 -> full coverage
        ]
        return tor.TranscribeOutput(segments=segs, duration_sec=1800.0)

    rc = tor.main(
        [
            "--predictions-dir", str(predictions),
            "--video-source-root", str(root),
            "--anchor-csv", str(anchor_csv),
            "--json",
        ],
        transcribe_vod=fake_transcribe,
    )

    out = json.loads(capsys.readouterr().out)
    assert sorted(out["completed"]) == ["111", "222"]
    assert out["excluded"] == ["333"]
    assert out["clean"] is False  # excluded anchor => fail-closed
    assert rc == 1
    assert tor.fresh_segments_path(predictions, "111").exists()
