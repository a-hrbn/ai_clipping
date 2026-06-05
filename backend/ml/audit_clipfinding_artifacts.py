from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a versioned audit of clipfinding model/data artifacts.")
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = build_audit(profile=args.profile, artifact_dir=args.artifact_dir)
    output = args.output or args.artifact_dir / f"{args.profile}_clipfinding_artifact_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "clipfinding_artifact_audit_written", "output": str(output), **payload["summary"]}, ensure_ascii=True, sort_keys=True))
    return 0


def build_audit(*, profile: str, artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    paths = {
        "ranker": artifact_dir / f"{profile}_ranker.json",
        "boundary_ranker": artifact_dir / f"{profile}_boundary_ranker.json",
        "cross_platform_rows": artifact_dir / f"{profile}_cross_platform_fingerprint_rows.jsonl",
        "ranker_eval": artifact_dir / f"{profile}_ranker_eval_summary.json",
        "rollout_readiness": artifact_dir / f"{profile}_rollout_readiness_summary.json",
    }
    artifacts = {name: _artifact_snapshot(path) for name, path in paths.items()}
    warnings = _warnings(artifacts)
    return {
        "profile": profile,
        "artifact_dir": str(artifact_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "warnings": warnings,
        "summary": {
            "profile": profile,
            "artifact_count": len(artifacts),
            "existing_artifacts": sum(1 for row in artifacts.values() if row["exists"]),
            "warnings": len(warnings),
            "cross_platform_rows": artifacts["cross_platform_rows"].get("rows"),
            "ranker_status": artifacts["ranker"].get("training_status"),
        },
    }


def _artifact_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    snapshot: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256(path),
    }
    if path.suffix == ".jsonl":
        snapshot["rows"] = _count_jsonl_rows(path)
    elif path.suffix == ".json":
        payload = _read_json(path)
        summary = payload.get("training_summary") if isinstance(payload, dict) else None
        if isinstance(summary, dict):
            snapshot["training_status"] = summary.get("status")
            snapshot["training_rows"] = summary.get("rows")
            snapshot["positive_rows"] = summary.get("positive_rows")
            snapshot["negative_rows"] = summary.get("negative_rows")
            snapshot["metrics"] = summary.get("metrics")
        if isinstance(payload, dict) and "rows" in payload:
            snapshot["rows"] = payload.get("rows")
    return snapshot


def _warnings(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if artifacts.get("cross_platform_rows", {}).get("rows") in (0, None):
        warnings.append({"code": "cross_platform_rows_missing", "message": "No cross-platform positive rows were found."})
    return warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
