from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import diagnose_ranker_holdout, export_ranker_hard_negatives


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh holdout diagnosis and the derived hard-negative training artifact."
    )
    parser.add_argument(
        "--gold-rows",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gold_eval_rows.jsonl"),
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument(
        "--diagnosis-json-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_holdout_diagnosis.json"),
    )
    parser.add_argument(
        "--diagnosis-csv-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_holdout_diagnosis.csv"),
    )
    parser.add_argument(
        "--hard-negative-rows-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_hard_negative_train_rows.jsonl"),
    )
    parser.add_argument(
        "--hard-negative-summary-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_hard_negative_train_summary.json"),
    )
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--min-forced-score", type=float, default=0.0)
    args = parser.parse_args()

    artifact_dir = pathlib.Path(args.artifact_dir)
    diagnosis_csv = pathlib.Path(args.diagnosis_csv_out)
    diagnosis_report = diagnose_ranker_holdout.diagnose_holdout(
        dataset_path=pathlib.Path(args.gold_rows),
        profile=args.profile,
        artifact_dir=artifact_dir,
        csv_out=diagnosis_csv,
    )
    _write_json(pathlib.Path(args.diagnosis_json_out), diagnosis_report)
    hard_negative_payload = export_ranker_hard_negatives.export_hard_negatives(
        diagnosis_csv=diagnosis_csv,
        gold_rows_path=pathlib.Path(args.gold_rows),
        existing_rows_path=pathlib.Path(args.hard_negative_rows_out),
        profile=args.profile,
        min_forced_score=float(args.min_forced_score),
        max_rows=int(args.max_rows),
    )
    export_ranker_hard_negatives._write_jsonl(pathlib.Path(args.hard_negative_rows_out), hard_negative_payload["rows"])
    export_ranker_hard_negatives._write_json(pathlib.Path(args.hard_negative_summary_out), hard_negative_payload["summary"])
    summary = {
        "diagnosis": {
            "status": diagnosis_report.get("status"),
            "selected_rows": diagnosis_report.get("selected_rows"),
            "csv_rows": diagnosis_report.get("csv_rows"),
            "csv_out": str(diagnosis_csv),
        },
        "hard_negatives": hard_negative_payload["summary"],
    }
    print(json.dumps(summary, ensure_ascii=True))
    return 0


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
