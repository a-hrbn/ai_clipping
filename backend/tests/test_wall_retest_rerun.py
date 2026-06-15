from __future__ import annotations

import json
import pathlib

from ml import wall_retest_rerun as rr


def _make_replayable(predictions_dir: pathlib.Path, vod_id: str) -> None:
    """Lay down the two debug artifacts that mark a job dir replayable."""
    debug = predictions_dir / f"pipeline_debug_W_{vod_id}" / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text("{}\n", encoding="utf-8")
    (debug / "segments.jsonl").write_text("{}\n", encoding="utf-8")


def test_select_subset_keeps_banter_vods_with_replayable_dirs(tmp_path) -> None:
    # WALL-Re-Test (#92): the re-test subset is the verbal-tic-/banter-heavy VODs
    # — those whose gt_moment.desc names a glossary entity — intersected with VODs
    # that have a replayable base dir (so base-vs-neu compares the same VODs).
    manifest = {
        "vodA": [{"desc": "Jynxzi vs Hivise banter in chess"}],   # banter + replayable -> in
        "vodB": [{"desc": "Jynxzi plays ranked siege"}],          # no glossary term -> out
        "vodC": [{"desc": "StableRonaldo 1v1 vs Jynxzi"}],        # banter but no dir -> out
        "vodD": [{"desc": "Caseoh reacts to Jynxzi"}],            # banter + replayable -> in
    }
    for vod_id in ("vodA", "vodB", "vodD"):
        _make_replayable(tmp_path, vod_id)

    subset = rr.select_subset(manifest, tmp_path)

    assert subset == ["vodA", "vodD"]


def test_select_subset_is_deterministic_sorted(tmp_path) -> None:
    manifest = {
        "vod_3": [{"desc": "Tyler1 rages at Jynxzi"}],
        "vod_1": [{"desc": "Hivise trolls Jynxzi"}],
        "vod_2": [{"desc": "Dantes and Jynxzi banter"}],
    }
    for vod_id in manifest:
        _make_replayable(tmp_path, vod_id)

    assert rr.select_subset(manifest, tmp_path) == ["vod_1", "vod_2", "vod_3"]


def test_rerun_subset_runs_each_vod_into_predictions_dir_leaving_base_untouched(tmp_path) -> None:
    base_dir = tmp_path / "base"
    neu_dir = tmp_path / "neu"
    # base holds the WALL status-quo dirs; the re-run must never touch them.
    _make_replayable(base_dir, "vodA")
    _make_replayable(base_dir, "vodD")
    base_before = sorted(str(p.relative_to(base_dir)) for p in base_dir.rglob("*"))

    calls: list[tuple[str, pathlib.Path]] = []

    def fake_run_vod(vod_id: str, predictions_dir: pathlib.Path) -> pathlib.Path:
        job_dir = predictions_dir / f"pipeline_debug_NEU_{vod_id}"
        (job_dir / "debug").mkdir(parents=True, exist_ok=True)
        calls.append((vod_id, predictions_dir))
        return job_dir

    result = rr.rerun_subset(["vodA", "vodD"], neu_dir, run_vod=fake_run_vod)

    assert result.completed == ["vodA", "vodD"]
    assert result.drops == []
    assert [v for v, _ in calls] == ["vodA", "vodD"]
    assert all(d == neu_dir for _, d in calls)
    assert (neu_dir / "pipeline_debug_NEU_vodA" / "debug").exists()
    # base dir is byte-for-byte untouched
    assert sorted(str(p.relative_to(base_dir)) for p in base_dir.rglob("*")) == base_before


def test_rerun_subset_logs_drops_and_keeps_going(tmp_path, capsys) -> None:
    # A VOD whose re-run returns None (no artifacts) or raises is a logged drop,
    # not a crash — the rest of the subset still runs.
    def fake_run_vod(vod_id: str, predictions_dir: pathlib.Path):
        if vod_id == "gap":
            return None
        if vod_id == "boom":
            raise RuntimeError("kapow")
        job_dir = predictions_dir / f"pipeline_debug_NEU_{vod_id}"
        (job_dir / "debug").mkdir(parents=True, exist_ok=True)
        return job_dir

    result = rr.rerun_subset(["ok", "gap", "boom"], tmp_path, run_vod=fake_run_vod)

    assert result.completed == ["ok"]
    assert result.drops == ["gap", "boom"]
    err = capsys.readouterr().err
    assert "gap" in err
    assert "boom" in err


def test_build_rerun_env_sets_levers_without_flipping_model(tmp_path) -> None:
    # AC3 guard: the re-run sets only the two cheap levers (glossary + diarization)
    # and redirects output; WHISPER_MODEL passes through untouched (stays whatever
    # the base env / .env has — large-v3-turbo). No model flip.
    base_env = {"WHISPER_MODEL": "large-v3-turbo", "FOO": "bar"}

    env = rr.build_rerun_env(tmp_path / "neu", base_env=base_env)

    assert env["WHISPER_INITIAL_PROMPT"] == rr.JYNXZI_GLOSSARY
    assert env["PIPELINE_ENABLE_DIARIZATION"] == "true"
    assert env["STORAGE_DIR"] == str(tmp_path / "neu")
    assert env["WHISPER_MODEL"] == "large-v3-turbo"  # not flipped
    assert env["FOO"] == "bar"  # base env preserved


def test_build_rerun_env_enables_eval_recall_so_neu_dirs_are_replayable(tmp_path) -> None:
    # The neu run goes through the serving worker, where the raw-recall branch +
    # the moments_raw.jsonl dump are off (serving doesn't pay for them). But the
    # compare harness replays that raw artifact — and base was an eval-capture run
    # with it on. So the re-run must opt into eval_recall for base-vs-neu parity;
    # without it the neu dirs are not replayable and every VOD drops.
    env = rr.build_rerun_env(tmp_path / "neu", base_env={})

    assert env["MOMENT_FINDER_EVAL_RECALL"] == "true"


def test_cli_dry_run_prints_selected_subset(tmp_path, capsys) -> None:
    # The CLI composes manifest-load + select_subset against the base dir; --dry-run
    # surfaces the subset without re-running anything.
    base = tmp_path / "base"
    neu = tmp_path / "neu"
    _make_replayable(base, "111")
    _make_replayable(base, "222")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"vod_id": "111", "gt_moments": [{"desc": "Hivise trolls Jynxzi"}]}) + "\n"
        + json.dumps({"vod_id": "222", "gt_moments": [{"desc": "normal ranked game"}]}) + "\n",
        encoding="utf-8",
    )

    rc = rr.main([
        "--base-dir", str(base),
        "--predictions-dir", str(neu),
        "--manifest", str(manifest_path),
        "--dry-run", "--json",
    ])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["subset"] == ["111"]
    assert out["dry_run"] is True


def test_is_replayable_requires_both_debug_artifacts(tmp_path) -> None:
    job = tmp_path / "pipeline_debug_WF_vodX"
    debug = job / "debug"
    debug.mkdir(parents=True)
    assert rr._is_replayable(job) is False                       # neither artifact
    (debug / "segments.jsonl").write_text("{}\n", encoding="utf-8")
    assert rr._is_replayable(job) is False                       # only segments
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text("{}\n", encoding="utf-8")
    assert rr._is_replayable(job) is True                        # both -> done


def test_pipeline_run_vod_skips_already_replayable_vod_on_resume(tmp_path, capsys) -> None:
    # Resumability: a re-launched driver must NOT re-spawn the heavy pipeline for a
    # VOD it already finished. With the replayable artifacts in place, pipeline_run_vod
    # returns the dir without ever importing pipeline_runner or spawning a subprocess —
    # so a fake subprocess.run that explodes proves the spawn path was never reached.
    neu = tmp_path / "neu"
    debug = neu / "pipeline_debug_WF_vodX" / "debug"
    debug.mkdir(parents=True)
    (debug / "segments.jsonl").write_text("{}\n", encoding="utf-8")
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text("{}\n", encoding="utf-8")

    def _boom(*_a, **_k):
        raise AssertionError("pipeline must not be spawned for an already-replayable VOD")

    original = rr.subprocess.run
    rr.subprocess.run = _boom
    try:
        result = rr.pipeline_run_vod("vodX", neu, workflow_id="WF")
    finally:
        rr.subprocess.run = original

    assert result == neu / "pipeline_debug_WF_vodX"
    assert "skip (resume)" in capsys.readouterr().err


def test_glossary_terms_are_the_initial_prompt_entities() -> None:
    # One source of truth: the terms matched in desc are the glossary entities
    # also biased into Whisper via initial_prompt.
    assert "Hivise" in rr.GLOSSARY_TERMS
    assert "StableRonaldo" in rr.GLOSSARY_TERMS
    for term in rr.GLOSSARY_TERMS:
        assert term in rr.JYNXZI_GLOSSARY
