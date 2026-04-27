import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import build_rollout_review_queue


class BuildRolloutReviewQueueTests(unittest.TestCase):
    def test_build_queue_excludes_gold_and_holdout_vods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "rows.jsonl"
            gold_rows = root / "gold.jsonl"
            out = root / "rollout_queue"
            _write_artifact(artifact_dir)
            rows = [
                _row("vod-a", "top-a", 3.0, start=10.0),
                _row("vod-a", "near-a", 2.7, start=20.0),
                _row("vod-a", "second-a", 2.0, start=90.0),
                _row("vod-b", "top-b", 4.0, start=30.0),
                _row("vod-holdout", "holdout-candidate", 9.0, start=40.0),
                _row("vod-a", "already-gold", 8.0, start=160.0),
            ]
            _write_jsonl(dataset, rows)
            _write_jsonl(
                gold_rows,
                [
                    {**rows[-1], "queue_id": "existing-gold"},
                    {"vod_id": "vod-holdout", "gold_holdout_split": "holdout"},
                ],
            )

            summary = build_rollout_review_queue.build_queue(
                dataset_path=dataset,
                existing_gold_rows_path=gold_rows,
                holdout_rows_path=gold_rows,
                profile="gaming",
                artifact_dir=artifact_dir,
                out_base=out,
                top_k_per_vod=2,
                max_total=10,
                min_gap_sec=20.0,
            )

            with (root / "rollout_queue.csv").open("r", encoding="utf-8", newline="") as handle:
                queue_rows = list(csv.DictReader(handle))
            sidecar_rows = [json.loads(line) for line in (root / "rollout_queue.features.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["review_required"], True)
        self.assertEqual(summary["auto_publish_allowed"], False)
        self.assertEqual(summary["excluded_gold_holdout_vod_rows"], 1)
        self.assertEqual(summary["excluded_existing_gold_rows"], 1)
        self.assertEqual(summary["queue_rows"], 3)
        self.assertEqual(len(queue_rows), 3)
        self.assertEqual(len(sidecar_rows), 3)
        self.assertEqual({row["queue_id"] for row in queue_rows}, {row["queue_id"] for row in sidecar_rows})
        self.assertNotIn("holdout-candidate", {row["candidate_id"] for row in queue_rows})
        self.assertNotIn("already-gold", {row["candidate_id"] for row in queue_rows})
        self.assertEqual([row["candidate_id"] for row in queue_rows if row["vod_id"] == "vod-a"], ["top-a", "second-a"])
        self.assertTrue(all(row["review_required"] == "true" for row in queue_rows))
        self.assertTrue(all(row["auto_publish_allowed"] == "false" for row in queue_rows))


def _write_artifact(artifact_dir: Path) -> None:
    (artifact_dir / "gaming_ranker.json").write_text(
        json.dumps(
            {
                "version": 3,
                "profile": "gaming",
                "model_type": "linear",
                "feature_names": ["hook_strength"],
                "weights": {"hook_strength": 1.0},
                "scales": {"hook_strength": {"mean": 0.0, "std": 1.0}},
                "training_stats": {
                    "rows": 100,
                    "positive_rows": 40,
                    "negative_rows": 60,
                    "train_rows": 80,
                    "validation_rows": 20,
                },
                "validation": {
                    "accuracy": 0.75,
                    "precision": 0.7,
                    "recall": 0.8,
                },
                "intercept": 0.0,
                "blend": 1.0,
                "serving_threshold": 0.0,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _row(vod_id: str, candidate_id: str, hook_strength: float, *, start: float) -> dict:
    return {
        "job_id": f"job-{vod_id}",
        "candidate_id": candidate_id,
        "vod_id": vod_id,
        "creator_id": "jynxzi",
        "platform": "twitch",
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "window_start": start,
        "window_end": start + 25.0,
        "label": 0,
        "label_class": "review_candidate",
        "candidate_source": "test_source",
        "features": {
            "candidate_id": candidate_id,
            "hook_strength": hook_strength,
            "source": "test_source",
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
