"""Slice 3 (#105) — the Archetype-Judge go-live flip: default-ON, per-job opt-out.

Slice 2 (#104) wired the judge into serving behind ``moment_finder_archetype_judge_enabled``,
default OFF. Slice 3 flips production to default-ON for archetype-carrying profiles and adds
a per-job opt-out through the existing ``strategy`` dict the Conductor already threads into
``run_pipeline`` — so an operator can turn the judge off for one job (cost/latency) without a
code change. These tests pin the go-live behaviour at its public seams: the config default,
the Conductor's ``MomentParams`` snapshot, and the normalized per-job strategy.
"""
from __future__ import annotations

import pytest

import config
import pipeline
from services import clip_strategy


def test_config_default_enables_the_archetype_judge() -> None:
    # Go-Live (#105 AC): production defaults the judge ON. A profile-gate inside
    # find_moments still keeps non-archetype profiles on the composite, so this is
    # default-on *for archetype-carrying profiles*.
    assert config.Settings().moment_finder_archetype_judge_enabled is True


def test_moment_params_snapshot_is_default_on() -> None:
    # The Conductor's settings snapshot carries the default-on judge flag into the
    # MomentParams the MomentFinding Producer is handed — no per-job override given.
    params = pipeline._moment_params()
    assert params.archetype_judge is True


def test_per_job_opt_out_disables_the_judge_without_a_code_change() -> None:
    # Per-job opt-out (#105 AC): a job whose normalized strategy opts the judge out
    # snapshots archetype_judge=False, even though the global default is ON.
    strategy = clip_strategy.normalize_strategy({"archetype_judge": False})
    params = pipeline._moment_params(strategy)
    assert params.archetype_judge is False


def test_strategy_without_opt_out_inherits_the_global_default() -> None:
    # A job that says nothing about the judge inherits the global default (ON) — the
    # opt-out is opt-in, it never silently flips a job that didn't ask.
    strategy = clip_strategy.normalize_strategy({"platform": "tiktok"})
    params = pipeline._moment_params(strategy)
    assert params.archetype_judge is True


def test_normalize_strategy_carries_the_opt_out_as_a_tristate() -> None:
    # The opt-out is a first-class normalized per-job field: absent -> None (inherit),
    # explicit false -> False (off), explicit true -> True (force on).
    assert clip_strategy.normalize_strategy({})["archetype_judge"] is None
    assert clip_strategy.normalize_strategy({"archetype_judge": False})["archetype_judge"] is False
    assert clip_strategy.normalize_strategy({"archetype_judge": True})["archetype_judge"] is True


def test_job_request_exposes_the_opt_out_field() -> None:
    # Operator surface (#105 AC): the opt-out is reachable from the job request without a
    # code change — absent on the model defaults to None (inherit the global default).
    from models import JobRequest

    url = "https://www.twitch.tv/videos/123456789"
    assert JobRequest(url=url).archetype_judge is None
    assert JobRequest(url=url, archetype_judge=False).archetype_judge is False


def test_request_strategy_threads_the_opt_out_into_the_strategy() -> None:
    # The request builder carries the opt-out into the normalized strategy the Conductor
    # reads, so a job submitted with the flag off snapshots archetype_judge=False.
    pytest.importorskip("fastapi")
    from routers import jobs

    assert jobs._request_strategy()["archetype_judge"] is None
    assert jobs._request_strategy(archetype_judge=False)["archetype_judge"] is False
