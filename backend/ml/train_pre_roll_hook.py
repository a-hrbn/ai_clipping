from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sklearn.linear_model import LogisticRegression

_FEATURES = (
    "pre_hook_hype",
    "hook_strength",
    "hook_delay_sec",
    "visual_hook_score",
    "audio_pop_score",
    "hook_rewrite_score",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a small pre-roll hook model from outcome rows.")
    parser.add_argument("--rows", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vod_training_rows.jsonl"))
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_pre_roll_hook.json"))
    args = parser.parse_args()
    artifact = train(_read_jsonl(pathlib.Path(args.rows)))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": artifact["training_rows"], "trained": artifact["trained"]}, ensure_ascii=True))
    return 0 if artifact["trained"] else 1


def train(rows: list[dict]) -> dict:
    usable = [
        row
        for row in rows
        if row.get("retention_3s") not in (None, "")
        and int(row.get("label", 0) or 0) in {0, 1}
    ]
    if len(usable) < 20 or len({int(_target(row)) for row in usable}) < 2:
        return {"trained": False, "reason": "insufficient_retention_rows", "training_rows": len(usable), "feature_names": list(_FEATURES)}
    matrix = [[_feature(row, name) for name in _FEATURES] for row in usable]
    labels = [int(_target(row)) for row in usable]
    model = LogisticRegression(max_iter=500, solver="liblinear")
    model.fit(matrix, labels)
    return {
        "trained": True,
        "model_type": "logistic_regression",
        "feature_names": list(_FEATURES),
        "weights": {name: float(value) for name, value in zip(_FEATURES, model.coef_[0], strict=False)},
        "intercept": float(model.intercept_[0]),
        "training_rows": len(usable),
        "multiplier_range": [0.7, 1.3],
    }


def _target(row: dict) -> int:
    return int(int(row.get("label", 0) or 0) == 1 and _safe_float(row.get("retention_3s")) >= 0.85)


def _feature(row: dict, name: str) -> float:
    return _safe_float((row.get("features") or {}).get(name, row.get(name)))


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
