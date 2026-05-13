import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import targeted_backfill


class TargetedBackfillTests(unittest.TestCase):
    def test_build_feature_cache_materializes_all_targeted_vods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue.jsonl"
            output = root / "features.jsonl"
            summary = root / "summary.json"
            queue.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "candidate_id": "a",
                            "label": 1,
                            "label_class": "positive",
                            "vod_id": "111",
                            "vod_url": "https://www.twitch.tv/videos/111",
                            "window_start": 10,
                            "window_end": 30,
                            "targeted_feature_backfill": True,
                        },
                        {
                            "candidate_id": "b",
                            "label": 1,
                            "label_class": "positive",
                            "vod_id": "222",
                            "vod_url": "https://www.twitch.tv/videos/222",
                            "window_start": 20,
                            "window_end": 40,
                            "targeted_feature_backfill": True,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "ml.targeted_backfill.build_twitch_training_dataset.build_dataset",
                return_value=[
                    {"vod_id": "111", "targeted_feature_backfill": True, "features": {"hook_strength": 1.0}},
                    {"vod_id": "222", "targeted_feature_backfill": True, "features": {"hook_strength": 0.8}},
                ],
            ) as build_mock:
                code = targeted_backfill.build_feature_cache(
                    queue_path=queue,
                    output_path=output,
                    summary_path=summary,
                    work_dir=root / "work",
                    profile="jynxzi",
                    require_all_vods=True,
                )
                payload = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        build_mock.assert_called_once()
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["requested_vods"], 2)
        self.assertEqual(payload["built_vods"], 2)
        self.assertEqual(payload["missing_vods"], [])

    def test_build_feature_cache_fails_when_selected_vod_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue.jsonl"
            output = root / "features.jsonl"
            summary = root / "summary.json"
            queue.write_text(
                json.dumps(
                    {
                        "candidate_id": "a",
                        "label": 1,
                        "vod_id": "111",
                        "vod_url": "https://www.twitch.tv/videos/111",
                        "window_start": 10,
                        "window_end": 30,
                        "targeted_feature_backfill": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("ml.targeted_backfill.build_twitch_training_dataset.build_dataset", return_value=[]):
                code = targeted_backfill.build_feature_cache(
                    queue_path=queue,
                    output_path=output,
                    summary_path=summary,
                    work_dir=root / "work",
                    profile="jynxzi",
                    require_all_vods=True,
                )
                payload = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "missing_targeted_vods")
        self.assertEqual(payload["missing_vods"], ["111"])

    def test_guard_pre_rematch_fails_when_manifest_is_missing_selected_vod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue.jsonl"
            features = root / "features.jsonl"
            manifest = root / "vods.jsonl"
            summary = root / "guard.json"
            queue.write_text(
                json.dumps(
                    {
                        "candidate_id": "a",
                        "label": 1,
                        "vod_id": "111",
                        "vod_url": "https://www.twitch.tv/videos/111",
                        "window_start": 10,
                        "window_end": 30,
                        "targeted_feature_backfill": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            features.write_text(json.dumps({"vod_id": "111", "features": {"hook_strength": 1}}) + "\n", encoding="utf-8")
            manifest.write_text(json.dumps({"vod_id": "222", "video_path": "/tmp/source.mp4"}) + "\n", encoding="utf-8")

            code = targeted_backfill.guard_pre_rematch(
                queue_path=queue,
                feature_rows_path=features,
                vod_manifest_path=manifest,
                summary_path=summary,
                profile="jynxzi",
                require_all_vods=True,
            )
            payload = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "missing_targeted_vod_materialization")
        self.assertEqual(payload["missing_manifest_vods"], ["111"])


if __name__ == "__main__":
    unittest.main()
