from __future__ import annotations

import csv
import json
from pathlib import Path

from ml import build_llm_moment_precision_review as review


def test_build_review_queue_writes_human_precision_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    cache_root = tmp_path / "vod_feature_cache"
    moment_prefix = artifacts / "jynxzi_llm_moment_finder_d0"
    positives = artifacts / "positives.jsonl"
    out = artifacts / "precision_review"
    _write_moments(
        moment_prefix.parent / "jynxzi_llm_moment_finder_d0_vod-a.moments.json",
        [
            {"rank": 1, "start": 100.0, "end": 120.0, "virality": 0.9, "hook": 0.8, "flow": 0.7, "value": 0.9, "trend": 0.6, "rationale": "best"},
            {"rank": 2, "start": 200.0, "end": 350.0, "virality": 0.95, "rationale": "too long"},
            {"rank": 3, "start": 130.0, "end": 150.0, "virality": 0.8, "rationale": "second kept"},
        ],
    )
    _write_transcript(cache_root, "vod-a")
    _write_jsonl(
        positives,
        [
            {
                "vod_id": "vod-a",
                "candidate_id": "known",
                "window_start": 105.0,
                "window_end": 115.0,
                "label": 1,
                "repost_title": "Known viral",
                "source_url": "https://example.test/known",
            }
        ],
    )

    summary = review.build_review_queue(
        profile="jynxzi",
        vod_ids=["vod-a"],
        moment_prefix=moment_prefix,
        positive_rows_path=positives,
        vod_cache_root=cache_root,
        out_base=out,
        top_n_per_vod=2,
        max_duration_sec=120.0,
        excerpt_pad_sec=1.0,
        excerpt_max_chars=400,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
    )

    csv_rows = list(csv.DictReader(out.with_suffix(".csv").open("r", encoding="utf-8", newline="")))
    jsonl_rows = [json.loads(line) for line in out.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["queue_rows"] == 2
    assert summary["clip_summary"]["requested"] is False
    assert summary["by_vod"]["vod-a"]["input_moment_count"] == 3
    assert summary["by_vod"]["vod-a"]["dropped_long_moment_count"] == 1
    assert summary["known_repost_matches_in_queue"] == 1
    assert [row["source_rank"] for row in csv_rows] == ["1", "3"]
    assert csv_rows[0]["known_repost_match"] == "True"
    assert csv_rows[0]["human_viral_label"] == ""
    assert "setup" in jsonl_rows[0]["transcript_excerpt"]
    assert out.with_suffix(".html").exists()
    assert out.with_suffix(".summary.json").exists()


def test_cut_review_clips_writes_ready_clip_metadata(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source = source_root / "vod_vod-a.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fake video")
    output_dir = tmp_path / "artifacts" / "clips"
    html_dir = tmp_path / "artifacts"
    rows = [
        {
            "vod_id": "vod-a",
            "review_rank": 1,
            "start": 10.0,
            "end": 22.0,
        }
    ]
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    summary = review.cut_review_clips(
        rows,
        video_source_root=source_root,
        output_dir=output_dir,
        html_dir=html_dir,
        pad_sec=1.0,
        ffmpeg_bin=Path("/fake/ffmpeg"),
        runner=fake_runner,
    )

    assert summary["all_clips_ready"] is True
    assert summary["ready_clip_count"] == 1
    assert rows[0]["clip_status"] == "ready"
    assert rows[0]["clip_rel_path"] == "clips/vod-a_rank01_000010_000022.mp4"
    assert Path(rows[0]["clip_path"]).read_bytes() == b"clip"
    assert "-ss" in calls[0]


def test_cut_review_clips_marks_missing_source(tmp_path: Path) -> None:
    rows = [{"vod_id": "missing-vod", "review_rank": 1, "start": 0.0, "end": 12.0}]

    summary = review.cut_review_clips(
        rows,
        video_source_root=tmp_path / "sources",
        output_dir=tmp_path / "clips",
        html_dir=tmp_path,
        pad_sec=0.0,
        ffmpeg_bin=Path("/fake/ffmpeg"),
    )

    assert summary["all_clips_ready"] is False
    assert summary["missing_source_count"] == 1
    assert summary["missing_source_vods"] == ["missing-vod"]
    assert rows[0]["clip_status"] == "missing_source"
    assert "vod_missing-vod.mp4" in rows[0]["clip_error"]


def test_build_review_queue_downloads_before_cutting_clips(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    cache_root = tmp_path / "vod_feature_cache"
    source_root = tmp_path / "sources"
    source = source_root / "vod_vod-a.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    moment_prefix = artifacts / "jynxzi_llm_moment_finder_d0"
    positives = artifacts / "positives.jsonl"
    out = artifacts / "precision_review"
    _write_moments(
        moment_prefix.parent / "jynxzi_llm_moment_finder_d0_vod-a.moments.json",
        [{"rank": 1, "start": 100.0, "end": 120.0, "virality": 0.9, "rationale": "best"}],
    )
    _write_transcript(cache_root, "vod-a")
    _write_jsonl(positives, [])

    def fake_preflight(vod_ids, **kwargs):
        assert list(vod_ids) == ["vod-a"]
        assert kwargs["download_missing"] is True
        assert kwargs["require_all"] is True
        return {
            "all_available": True,
            "failed_vod_ids": [],
            "by_vod": {
                "vod-a": {
                    "status": "downloaded",
                    "source_video_path": str(source),
                    "error": "",
                }
            },
        }

    def fake_cut(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"clip")

    monkeypatch.setattr(review.vod_download_preflight, "ensure_vod_sources", fake_preflight)
    monkeypatch.setattr(review, "run_ffmpeg_cut", fake_cut)

    summary = review.build_review_queue(
        profile="jynxzi",
        vod_ids=["vod-a"],
        moment_prefix=moment_prefix,
        positive_rows_path=positives,
        vod_cache_root=cache_root,
        out_base=out,
        top_n_per_vod=1,
        max_duration_sec=120.0,
        excerpt_pad_sec=1.0,
        excerpt_max_chars=400,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        cut_clips=True,
        download_missing_vods=True,
        video_source_root=source_root,
        clip_output_dir=artifacts / "clips",
        clip_run_id="run-a",
        require_clips=True,
    )

    assert summary["clip_summary"]["all_clips_ready"] is True
    assert summary["clip_summary"]["download_preflight"]["all_available"] is True
    assert summary["clip_run_id"] == "run-a"
    assert summary["clip_output_dir"] == str(artifacts / "clips" / "run-a")
    csv_rows = list(csv.DictReader(out.with_suffix(".csv").open("r", encoding="utf-8", newline="")))
    assert csv_rows[0]["clip_status"] == "ready"
    assert csv_rows[0]["clip_rel_path"] == "clips/run-a/vod-a_rank01_000100_000120.mp4"


def test_review_builder_can_use_moment_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "job-a"
    artifacts = tmp_path / "artifacts"
    cache_root = tmp_path / "vod_feature_cache"
    positives = artifacts / "positives.jsonl"
    out = artifacts / "precision_review"
    moment_prefix = run_dir / "jynxzi_llm_moment_finder_d0"
    _write_moments(
        run_dir / "jynxzi_llm_moment_finder_d0_vod-a.moments.json",
        [{"rank": 1, "start": 100.0, "end": 120.0, "virality": 0.9, "rationale": "best"}],
    )
    run_dir.joinpath("manifest.json").write_text(
        json.dumps({"moment_prefix": str(moment_prefix), "vod_ids": ["vod-a"]}),
        encoding="utf-8",
    )
    _write_transcript(cache_root, "vod-a")
    _write_jsonl(positives, [])

    summary = review.build_review_queue(
        profile="jynxzi",
        vod_ids=[],
        moment_prefix=review.resolve_moment_prefix(Path("unused"), run_dir),
        moment_run_dir=run_dir,
        positive_rows_path=positives,
        vod_cache_root=cache_root,
        out_base=out,
        top_n_per_vod=1,
        max_duration_sec=120.0,
        excerpt_pad_sec=1.0,
        excerpt_max_chars=400,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
    )

    assert summary["queue_rows"] == 1
    assert summary["vod_ids"] == ["vod-a"]
    assert summary["moment_run_dir"] == str(run_dir)
    assert summary["moment_prefix"] == str(moment_prefix)


def test_discover_vod_ids_from_moment_artifacts(tmp_path: Path) -> None:
    prefix = tmp_path / "artifacts" / "jynxzi_llm_moment_finder_d0"
    _write_moments(prefix.parent / "jynxzi_llm_moment_finder_d0_b.moments.json", [])
    _write_moments(prefix.parent / "jynxzi_llm_moment_finder_d0_a.moments.json", [])

    assert review.discover_vod_ids(prefix) == ["a", "b"]


def test_resolve_clip_run_id_sanitizes_explicit_value() -> None:
    assert review.resolve_clip_run_id("review run/1") == "review_run_1"


def test_resolve_clip_run_id_defaults_to_timestamp_not_item() -> None:
    run_id = review.resolve_clip_run_id("")

    assert run_id != "item"
    assert run_id.endswith("Z")
    assert "T" in run_id


def _write_moments(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"moments": rows}), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_transcript(root: Path, vod_id: str) -> None:
    path = root / vod_id / "transcript_fixture.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "transcript": [
                    {"start": 99.0, "end": 101.0, "text": "setup"},
                    {"start": 110.0, "end": 112.0, "text": "payoff"},
                    {"start": 140.0, "end": 142.0, "text": "another moment"},
                ]
            }
        ),
        encoding="utf-8",
    )
