import csv
import json
import tempfile
import unittest
from pathlib import Path

from ml import build_gold_holdout_annotation_manifest


class BuildGoldHoldoutAnnotationManifestTests(unittest.TestCase):
    def test_build_manifest_prioritizes_holdout_vods_with_diagnosis_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_rows = root / "gold.jsonl"
            diagnosis = root / "diagnosis.csv"
            _write_jsonl(
                gold_rows,
                [
                    _gold("vod-a", "train", 1),
                    _gold("vod-b", "holdout", 1),
                    _gold("vod-b", "holdout", 0),
                    _gold("vod-c", "holdout", 0),
                ],
            )
            with diagnosis.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["group", "error_type"])
                writer.writeheader()
                writer.writerow({"group": "vod-c", "error_type": "false_positive"})

            rows = build_gold_holdout_annotation_manifest.build_manifest(
                gold_rows_path=gold_rows,
                diagnosis_csv=diagnosis,
                limit=10,
            )

        self.assertEqual([row["vod_id"] for row in rows], ["vod-c", "vod-b"])
        self.assertEqual(rows[0]["diagnosis_error_rows"], 1)
        self.assertEqual(rows[1]["current_gold_rows"], 2)


def _gold(vod_id: str, split: str, label: int) -> dict:
    return {
        "job_id": f"twitch_vod_{vod_id}",
        "vod_id": vod_id,
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "creator_id": "jynxzi",
        "platform": "twitch",
        "gold_holdout_split": split,
        "label": label,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
