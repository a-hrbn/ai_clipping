from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import HTTPException

from routers import llm_moment_review as router


def test_start_workflow_runs_moment_finder_and_precision_review(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(router, "WORKFLOW_ROOT", tmp_path / "workflows")
    monkeypatch.setattr(router, "workflow_id_now", lambda: "workflow-a")
    monkeypatch.setattr(router.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(router.d0, "run_d0", fake_run_d0)
    monkeypatch.setattr(router.precision_review, "build_review_queue", fake_build_review_queue)

    response = asyncio.run(
        router.start_llm_moment_review(
            router.LlmMomentReviewStartRequest(
                vod_ids=["vod-a"],
                cut_clips=True,
                dry_run=True,
            )
        )
    )

    assert response["workflow_id"] == "workflow-a"
    assert response["status"] == "done"
    assert response["stage"] == "done"
    assert response["progress_pct"] == 100
    assert response["row_count"] == 1
    assert response["summary"]["queue_rows"] == 1
    assert response["summary"]["exclude_known_reposts"] is False
    assert response["label_summary"]["non_repost_rows"] == 1
    assert response["phase2_gate"]["blocking_reasons"] == ["recall_floor_missing", "unlabeled_non_repost_rows:1"]
    row = response["rows"][0]
    assert row["review_id"] == "llm_precision:vod-a:rank01"
    assert row["clip_url"] == (
        "/api/llm-moment-review/workflows/workflow-a/clips/"
        "clips/workflow-a/vod-a_rank01_000010_000020.mp4"
    )
    assert (tmp_path / "workflows" / "workflow-a" / "workflow.json").exists()


def test_save_label_merges_into_workflow_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(router, "WORKFLOW_ROOT", tmp_path / "workflows")
    root = router.workflow_dir("workflow-a")
    (root / "review").mkdir(parents=True)
    router.write_json(
        router.status_path(root),
        {
            "workflow_id": "workflow-a",
            "status": "done",
            "stage": "done",
            "progress_pct": 100,
            "message": "ready",
            "created_at": router.utc_now(),
            "updated_at": router.utc_now(),
        },
    )
    (root / "review" / "precision_review.jsonl").write_text(
        json.dumps(
            {
                "review_id": "llm_precision:vod-a:rank01",
                "vod_id": "vod-a",
                "review_rank": 1,
                "clip_rel_path": "clips/workflow-a/vod-a.mp4",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        router.save_llm_moment_review_label(
            "workflow-a",
            router.LlmMomentReviewLabelRequest(
                review_id="llm_precision:vod-a:rank01",
                human_viral_label="yes",
                human_boundary_label="good",
                human_notes="strong hook",
            ),
        )
    )
    response = asyncio.run(router.get_llm_moment_review_workflow("workflow-a"))

    assert result["label"]["human_viral_label"] == "yes"
    assert response["rows"][0]["human_viral_label"] == "yes"
    assert response["rows"][0]["human_boundary_label"] == "good"
    assert response["rows"][0]["human_notes"] == "strong hook"
    assert response["label_summary"]["non_repost_yes"] == 1
    assert response["label_summary"]["non_repost_yes_no_precision"] == 1.0
    assert response["phase2_gate"]["non_repost_precision_passes"] is True
    assert response["phase2_gate"]["blocking_reasons"] == ["recall_floor_missing"]
    assert (root / "review" / "review_label_events.jsonl").exists()


def test_clip_serving_blocks_path_traversal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(router, "WORKFLOW_ROOT", tmp_path / "workflows")
    root = router.workflow_dir("workflow-a")
    (root / "review" / "clips" / "workflow-a").mkdir(parents=True)
    (root / "review" / "clips" / "workflow-a" / "clip.mp4").write_bytes(b"clip")

    response = router.serve_llm_moment_review_clip("workflow-a", "clips/workflow-a/clip.mp4")
    assert response.path.endswith("clip.mp4")

    try:
        router.serve_llm_moment_review_clip("workflow-a", "../workflow.json")
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("path traversal should be rejected")


def test_attach_latest_recall_floor_requires_llm_holdout_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(router, "WORKFLOW_ROOT", tmp_path / "workflows")
    summary_path = tmp_path / "holdout_summary.json"
    monkeypatch.setattr(router.holdout, "DEFAULT_RUN_SUMMARY", summary_path)
    root = router.workflow_dir("workflow-a")
    router.write_json(
        router.status_path(root),
        {
            "workflow_id": "workflow-a",
            "status": "done",
            "stage": "done",
            "progress_pct": 100,
            "message": "ready",
            "request": {"phase2_min_non_repost_precision": 0.5},
            "created_at": router.utc_now(),
            "updated_at": router.utc_now(),
        },
    )
    (root / "review").mkdir(parents=True)
    (root / "review" / "precision_review.jsonl").write_text(
        json.dumps(
            {
                "review_id": "llm_precision:vod-a:rank01",
                "vod_id": "vod-a",
                "review_rank": 1,
                "known_repost_match": False,
                "human_viral_label": "",
                "clip_rel_path": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "run_id": "phase2",
                "llm_moment_finder": {"enabled": True, "model": "gemini-2.5-flash-lite"},
                "eval_aggregate": {
                    "product_selection": {
                        "final_moments@k12": {
                            "per_vod_recall": 0.2,
                            "matched_positive_windows": 2,
                            "positive_windows": 10,
                            "selected_candidate_rows": 12,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    response = asyncio.run(router.attach_latest_recall_floor("workflow-a"))

    assert response["phase2_gate"]["recall_floor"]["available"] is True
    assert response["phase2_gate"]["recall_floor"]["passes"] is True
    assert response["phase2_gate"]["blocking_reasons"] == ["unlabeled_non_repost_rows:1"]


class ImmediateThread:
    def __init__(self, *, target, args, name=None, daemon=None):
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        self.target(*self.args)


def fake_run_d0(**kwargs):
    output_prefix: Path = kwargs["output_prefix"]
    moment_path = output_prefix.parent / f"{output_prefix.name}_vod-a.moments.json"
    moment_path.parent.mkdir(parents=True, exist_ok=True)
    moment_path.write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "start": 10.0,
                    "end": 20.0,
                    "virality": 0.9,
                    "hook": 0.8,
                    "flow": 0.7,
                    "value": 0.9,
                    "trend": 0.6,
                    "rationale": "best",
                }
            ]
        ),
        encoding="utf-8",
    )
    return {
        "event": "llm_moment_finder_d0_batch_done",
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite",
        "vod_ids": ["vod-a"],
        "vod_count": 1,
        "eval_path": str(output_prefix.with_suffix(".eval.json")),
    }


def fake_build_review_queue(**kwargs):
    assert kwargs["exclude_known_reposts"] is False
    out_base: Path = kwargs["out_base"]
    clip_output_dir: Path = kwargs["clip_output_dir"]
    clip_run_id = str(kwargs["clip_run_id"])
    review_dir = out_base.parent
    clip_dir = clip_output_dir / clip_run_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip = clip_dir / "vod-a_rank01_000010_000020.mp4"
    clip.write_bytes(b"clip")
    row = {
        "review_id": "llm_precision:vod-a:rank01",
        "profile": "jynxzi",
        "vod_id": "vod-a",
        "review_rank": 1,
        "source_rank": 1,
        "start": 10.0,
        "end": 20.0,
        "duration": 10.0,
        "virality": 0.9,
        "hook": 0.8,
        "flow": 0.7,
        "value": 0.9,
        "trend": 0.6,
        "rationale": "best",
        "transcript_excerpt": "setup payoff",
        "clip_status": "ready",
        "clip_path": str(clip),
        "clip_rel_path": "clips/workflow-a/vod-a_rank01_000010_000020.mp4",
        "known_repost_match": False,
        "human_viral_label": "",
        "human_boundary_label": "",
        "human_notes": "",
    }
    review_dir.mkdir(parents=True, exist_ok=True)
    (out_base.with_suffix(".jsonl")).write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = {
        "event": "llm_moment_precision_review_built",
        "queue_rows": 1,
        "exclude_known_reposts": False,
        "known_repost_matches_in_queue": 0,
        "non_repost_rows_in_queue": 1,
        "clip_summary": {"all_clips_ready": True, "ready_clip_count": 1},
        "jsonl_path": str(out_base.with_suffix(".jsonl")),
        "summary_path": str(out_base.with_suffix(".summary.json")),
    }
    (out_base.with_suffix(".summary.json")).write_text(json.dumps(summary), encoding="utf-8")
    return summary
