import csv
import json
import tempfile
import unittest
from pathlib import Path

from ml import inspect_background_negatives


class InspectBackgroundNegativesTests(unittest.TestCase):
    def test_exports_spotcheck_csv_for_background_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rows.jsonl"
            output_path = root / "spotcheck.csv"
            rows = [
                {
                    "vod_id": "123",
                    "window_start": 61.5,
                    "window_end": 91.5,
                    "label_class": "background_negative",
                    "supervision_tier": "background_negative",
                    "notes": "one two three",
                    "features": {
                        "chat_message_rate_mean": 0.2,
                        "max_hype": 1.5,
                        "face_reaction_peak": 0.1,
                    },
                },
                {
                    "vod_id": "123",
                    "window_start": 10.0,
                    "window_end": 20.0,
                    "label_class": "positive",
                    "supervision_tier": "twitch_clip_proxy",
                    "features": {},
                },
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            rc = inspect_background_negatives.main_with_args_for_test([
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--sample-size",
                "30",
            ]) if hasattr(inspect_background_negatives, "main_with_args_for_test") else _run_script(input_path, output_path)

            self.assertEqual(rc, 0)
            with output_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(csv_rows[0]["vod_id"], "123")
        self.assertEqual(csv_rows[0]["chat_rate"], "0.2")
        self.assertEqual(csv_rows[0]["twitch_clip_link"], "https://www.twitch.tv/videos/123?t=61s")


def _run_script(input_path: Path, output_path: Path) -> int:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "ml" / "inspect_background_negatives.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--sample-size",
            "30",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode


if __name__ == "__main__":
    unittest.main()
