from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import ranker

_SCALE_TARGET_ROWS = 5_000
_SCALE_TARGET_POSITIVE_ROWS = 800
_SCALE_TARGET_NEGATIVE_ROWS = 2_400


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ranker artifacts against deployment and scale gates.")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    args = parser.parse_args()

    artifact_dir = pathlib.Path(args.artifact_dir)
    payload = audit_artifacts(artifact_dir)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


def audit_artifacts(artifact_dir: pathlib.Path) -> dict[str, object]:
    profiles: list[dict[str, object]] = []
    for path in sorted(artifact_dir.glob("*_ranker.json")):
        if path.name.endswith("_boundary_ranker.json"):
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        stats = artifact.get("training_stats") or {}
        rows = int(stats.get("rows", 0) or 0)
        positives = int(stats.get("positive_rows", 0) or 0)
        negatives = int(stats.get("negative_rows", 0) or 0)
        profiles.append(
            {
                "profile": str(artifact.get("profile") or path.stem.removesuffix("_ranker")),
                "artifact": str(path),
                "selected_model": str(artifact.get("selected_model") or ""),
                "training_stats": {
                    "rows": rows,
                    "positive_rows": positives,
                    "negative_rows": negatives,
                },
                "deploy_gate": {
                    "rows_min": ranker._MIN_TRAINING_ROWS,
                    "positive_rows_min": ranker._MIN_POSITIVE_ROWS,
                    "negative_rows_min": ranker._MIN_NEGATIVE_ROWS,
                    "ready": (
                        rows >= ranker._MIN_TRAINING_ROWS
                        and positives >= ranker._MIN_POSITIVE_ROWS
                        and negatives >= ranker._MIN_NEGATIVE_ROWS
                    ),
                },
                "deploy_gap": {
                    "rows": max(ranker._MIN_TRAINING_ROWS - rows, 0),
                    "positive_rows": max(ranker._MIN_POSITIVE_ROWS - positives, 0),
                    "negative_rows": max(ranker._MIN_NEGATIVE_ROWS - negatives, 0),
                },
                "scale_target": {
                    "rows_min": _SCALE_TARGET_ROWS,
                    "positive_rows_min": _SCALE_TARGET_POSITIVE_ROWS,
                    "negative_rows_min": _SCALE_TARGET_NEGATIVE_ROWS,
                    "ready": (
                        rows >= _SCALE_TARGET_ROWS
                        and positives >= _SCALE_TARGET_POSITIVE_ROWS
                        and negatives >= _SCALE_TARGET_NEGATIVE_ROWS
                    ),
                },
                "scale_gap": {
                    "rows": max(_SCALE_TARGET_ROWS - rows, 0),
                    "positive_rows": max(_SCALE_TARGET_POSITIVE_ROWS - positives, 0),
                    "negative_rows": max(_SCALE_TARGET_NEGATIVE_ROWS - negatives, 0),
                },
            }
        )
    return {
        "artifact_dir": str(artifact_dir),
        "profiles": profiles,
    }


if __name__ == "__main__":
    raise SystemExit(main())
