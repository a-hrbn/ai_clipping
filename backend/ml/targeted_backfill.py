"""Utilities for feature-only targeted VOD backfill before fingerprint rematch."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable

from ml import build_twitch_training_dataset, prepare_cross_platform_fingerprint_rows, train_vod_ranker


def build_feature_cache(
    *,
    queue_path: pathlib.Path,
    output_path: pathlib.Path,
    summary_path: pathlib.Path,
    work_dir: pathlib.Path,
    profile: str,
    require_all_vods: bool,
) -> int:
    labels = train_vod_ranker._read_targeted_backfill_labels([queue_path], profile)
    requested_vods = _vod_ids(labels)
    rows: list[dict] = []
    status = "no_targeted_rows"
    error = ""
    if labels:
        try:
            rows = build_twitch_training_dataset.build_dataset(labels, profile, work_dir)
            status = "passed"
        except Exception as exc:
            status = "failed"
            error = _compact_error(exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, rows)
    built_vods = _vod_ids(rows)
    missing_vods = sorted(requested_vods - built_vods)
    if require_all_vods and missing_vods and status == "passed":
        status = "missing_targeted_vods"

    summary = {
        "event": "targeted_backfill_feature_cache_done",
        "queue_path": str(queue_path),
        "output_path": str(output_path),
        "work_dir": str(work_dir),
        "profile": profile,
        "require_all_vods": bool(require_all_vods),
        "input_rows": len(labels),
        "requested_vods": len(requested_vods),
        "built_rows": len(rows),
        "built_vods": len(built_vods),
        "missing_vods": missing_vods,
        "status": status,
    }
    if error:
        summary["error"] = error
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 1 if status in {"failed", "missing_targeted_vods"} else 0


def refresh_vod_manifest(
    *,
    vod_cache_root: pathlib.Path,
    manifest_path: pathlib.Path,
    summary_path: pathlib.Path,
) -> int:
    rows = prepare_cross_platform_fingerprint_rows.build_vod_manifest_from_cache(vod_cache_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest_path, rows)
    summary = {
        "event": "targeted_backfill_vod_manifest_refreshed",
        "vod_cache_root": str(vod_cache_root),
        "manifest_path": str(manifest_path),
        "manifest_rows": len(rows),
        "manifest_vods": len(_vod_ids(rows)),
        "status": "passed",
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


def guard_pre_rematch(
    *,
    queue_path: pathlib.Path,
    feature_rows_path: pathlib.Path,
    vod_manifest_path: pathlib.Path,
    summary_path: pathlib.Path,
    profile: str,
    require_all_vods: bool,
) -> int:
    labels = train_vod_ranker._read_targeted_backfill_labels([queue_path], profile)
    feature_rows = _read_jsonl(feature_rows_path)
    manifest_rows = _read_jsonl(vod_manifest_path)
    selected_vods = _vod_ids(labels)
    feature_vods = _vod_ids(feature_rows)
    manifest_vods = _vod_ids(manifest_rows)
    missing_feature_vods = sorted(selected_vods - feature_vods)
    missing_manifest_vods = sorted(selected_vods - manifest_vods)
    passed = not require_all_vods or (not missing_feature_vods and not missing_manifest_vods)
    status = "passed" if passed else "missing_targeted_vod_materialization"
    summary = {
        "event": "targeted_backfill_pre_rematch_guard",
        "queue_path": str(queue_path),
        "feature_rows_path": str(feature_rows_path),
        "vod_manifest_path": str(vod_manifest_path),
        "profile": profile,
        "require_all_vods": bool(require_all_vods),
        "selected_vods": len(selected_vods),
        "feature_vods": len(feature_vods),
        "manifest_vods": len(manifest_vods),
        "missing_feature_vods": missing_feature_vods,
        "missing_manifest_vods": missing_manifest_vods,
        "status": status,
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0 if passed else 1


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _vod_ids(rows: Iterable[dict]) -> set[str]:
    vods: set[str] = set()
    for row in rows:
        vod_id = str(row.get("vod_id") or row.get("matched_vod_id") or row.get("source_vod_id") or "").strip()
        if vod_id:
            vods.add(vod_id)
    return vods


def _compact_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:500]


def _bool_arg(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted feature-backfill helpers for Jynxzi training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-cache")
    build_parser.add_argument("--queue", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--summary", required=True)
    build_parser.add_argument("--work-dir", required=True)
    build_parser.add_argument("--profile", default="jynxzi")
    build_parser.add_argument("--require-all-vods", default="true")

    refresh_parser = subparsers.add_parser("refresh-vod-manifest")
    refresh_parser.add_argument("--vod-cache-root", required=True)
    refresh_parser.add_argument("--manifest", required=True)
    refresh_parser.add_argument("--summary", required=True)

    guard_parser = subparsers.add_parser("guard-rematch")
    guard_parser.add_argument("--queue", required=True)
    guard_parser.add_argument("--feature-rows", required=True)
    guard_parser.add_argument("--vod-manifest", required=True)
    guard_parser.add_argument("--summary", required=True)
    guard_parser.add_argument("--profile", default="jynxzi")
    guard_parser.add_argument("--require-all-vods", default="true")

    args = parser.parse_args()
    if args.command == "build-cache":
        return build_feature_cache(
            queue_path=pathlib.Path(args.queue),
            output_path=pathlib.Path(args.output),
            summary_path=pathlib.Path(args.summary),
            work_dir=pathlib.Path(args.work_dir),
            profile=args.profile,
            require_all_vods=_bool_arg(args.require_all_vods),
        )
    if args.command == "refresh-vod-manifest":
        return refresh_vod_manifest(
            vod_cache_root=pathlib.Path(args.vod_cache_root),
            manifest_path=pathlib.Path(args.manifest),
            summary_path=pathlib.Path(args.summary),
        )
    return guard_pre_rematch(
        queue_path=pathlib.Path(args.queue),
        feature_rows_path=pathlib.Path(args.feature_rows),
        vod_manifest_path=pathlib.Path(args.vod_manifest),
        summary_path=pathlib.Path(args.summary),
        profile=args.profile,
        require_all_vods=_bool_arg(args.require_all_vods),
    )


if __name__ == "__main__":
    raise SystemExit(main())
