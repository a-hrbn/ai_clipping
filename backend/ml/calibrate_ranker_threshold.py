from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import train_ranker
from services.ranker import Ranker
from services.timeline_types import CandidateFeatures


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a ranker artifact serving threshold on labeled eval rows.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--split-field", default="")
    parser.add_argument("--split-value", default="")
    parser.add_argument("--source", default="eval_calibration")
    parser.add_argument("--min-recall", type=float, default=train_ranker._SERVING_THRESHOLD_MIN_RECALL)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--min-positive-rows", type=int, default=3)
    parser.add_argument("--report-out", default="")
    parser.add_argument("--write-artifact", action="store_true")
    args = parser.parse_args()

    report = calibrate_threshold(
        dataset_path=pathlib.Path(args.dataset),
        profile=args.profile,
        artifact_dir=pathlib.Path(args.artifact_dir),
        split_field=args.split_field,
        split_value=args.split_value,
        source=args.source,
        min_recall=args.min_recall,
        min_rows=args.min_rows,
        min_positive_rows=args.min_positive_rows,
        write_artifact=args.write_artifact,
    )
    payload = json.dumps(report, ensure_ascii=True)
    if args.report_out:
        report_path = pathlib.Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def calibrate_threshold(
    *,
    dataset_path: pathlib.Path,
    profile: str,
    artifact_dir: pathlib.Path,
    split_field: str = "",
    split_value: str = "",
    source: str = "eval_calibration",
    min_recall: float = train_ranker._SERVING_THRESHOLD_MIN_RECALL,
    min_rows: int = 10,
    min_positive_rows: int = 3,
    write_artifact: bool = False,
) -> dict[str, object]:
    rows = _read_jsonl(dataset_path)
    calibration_rows = _calibration_rows(rows, split_field, split_value)
    positive_rows = sum(1 for row in calibration_rows if int(row.get("label", 0)) == 1)
    negative_rows = sum(1 for row in calibration_rows if int(row.get("label", 0)) == 0)
    base_report: dict[str, object] = {
        "status": "skipped",
        "reason": "",
        "profile": profile,
        "dataset": str(dataset_path),
        "split_field": split_field,
        "split_value": split_value,
        "source": source,
        "rows": len(rows),
        "calibration_rows": len(calibration_rows),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "write_artifact": bool(write_artifact),
    }
    if len(calibration_rows) < int(min_rows):
        return {**base_report, "reason": "calibration_rows_below_minimum"}
    if positive_rows < int(min_positive_rows) or negative_rows < 1:
        return {**base_report, "reason": "calibration_class_rows_below_minimum"}

    ranker = Ranker(artifact_dir=artifact_dir, enable_trained_model=True)
    inspection = ranker.inspect_artifact(profile)
    artifact = inspection.get("artifact") if isinstance(inspection.get("artifact"), dict) else None
    if inspection.get("status") != "usable" or artifact is None:
        return {
            **base_report,
            "reason": str(inspection.get("reason") or "artifact_not_usable"),
            "artifact_status": str(inspection.get("status") or "missing"),
        }

    features = [CandidateFeatures.from_dict(row["features"]) for row in calibration_rows]
    score_map = {candidate_id: score for candidate_id, score in ranker.score(features, profile)}
    labels = train_ranker.np.array([int(row["label"]) for row in calibration_rows], dtype=int)
    scores = train_ranker.np.array(
        [float(score_map.get(row["candidate_id"], 0.0)) for row in calibration_rows],
        dtype=float,
    )
    split = train_ranker._split_rows(calibration_rows)
    groups = train_ranker._ranking_group_labels(calibration_rows, split.get("unit", ""))
    calibration = train_ranker._calibrate_serving_threshold(
        labels,
        scores,
        groups=groups,
        source=source,
        min_recall=float(min_recall),
    )
    calibration["calibration_rows"] = len(calibration_rows)
    calibration["positive_rows"] = positive_rows
    calibration["negative_rows"] = negative_rows
    calibration["split_field"] = split_field
    calibration["split_value"] = split_value
    calibration["split_strategy"] = split.get("strategy", "")
    calibration["split_unit"] = split.get("unit", "")

    report = {
        **base_report,
        "status": "updated" if write_artifact else "computed",
        "reason": "ok",
        "artifact_status": inspection.get("status", ""),
        "old_threshold": artifact.get("serving_threshold"),
        "new_threshold": calibration["threshold"],
        "calibration": calibration,
    }
    if write_artifact:
        artifact["serving_threshold"] = float(calibration["threshold"])
        artifact["serving_threshold_calibration"] = calibration
        _write_artifact(artifact_dir, profile, artifact)
        _write_artifact_health(artifact_dir, profile, artifact)
    return report


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("label") in {0, 1} and isinstance(row.get("features"), dict):
            rows.append(row)
    return rows


def _calibration_rows(rows: list[dict], split_field: str, split_value: str) -> list[dict]:
    field = str(split_field or "").strip()
    value = str(split_value or "").strip()
    if not field or not value:
        return list(rows)
    return [row for row in rows if str(row.get(field) or "").strip() == value]


def _write_artifact(artifact_dir: pathlib.Path, profile: str, artifact: dict) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{profile}_ranker.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_artifact_health(artifact_dir: pathlib.Path, profile: str, artifact: dict) -> None:
    health = train_ranker._artifact_health_summary(artifact)
    path = artifact_dir / f"{profile}_artifact_health.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(health, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
