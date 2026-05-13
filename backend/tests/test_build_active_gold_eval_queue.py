import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import build_active_gold_eval_queue


class BuildActiveGoldEvalQueueTests(unittest.TestCase):
    def test_build_queue_selects_error_and_threshold_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "rows.jsonl"
            existing = root / "existing_gold.jsonl"
            out = root / "active_queue"
            _write_artifact(artifact_dir)
            rows = [
                _row("vod-a", "positive-missed", 1, -1.0),
                _row("vod-a", "negative-high", 0, 1.0),
                _row("vod-b", "near-threshold", 1, 0.01),
                _row("vod-b", "positive-review", 1, 0.8),
                _row("vod-b", "diverse", 0, -0.5),
                _row("vod-c", "already-gold", 0, 1.2),
            ]
            _write_jsonl(dataset, rows)
            _write_jsonl(existing, [rows[-1]])

            summary = build_active_gold_eval_queue.build_queue(
                dataset_path=dataset,
                existing_gold_rows_path=existing,
                profile="gaming",
                artifact_dir=artifact_dir,
                out_base=out,
                target_size=5,
                per_vod_limit=3,
                near_threshold_window=1.0,
                seed=1,
            )

            with (root / "active_queue.csv").open("r", encoding="utf-8", newline="") as handle:
                queue_rows = list(csv.DictReader(handle))
            sidecar_rows = [json.loads(line) for line in (root / "active_queue.features.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["queue_rows"], 5)
        self.assertEqual(len(queue_rows), 5)
        self.assertEqual(len(sidecar_rows), 5)
        self.assertNotIn("already-gold", {row["candidate_id"] for row in queue_rows})
        self.assertIn("false_negative_proxy", {row["active_bucket"] for row in queue_rows})
        self.assertIn("false_positive_proxy", {row["active_bucket"] for row in queue_rows})
        self.assertIn("near_threshold", {row["active_bucket"] for row in queue_rows})
        self.assertIn("proxy_positive_review", {row["active_bucket"] for row in queue_rows})
        self.assertEqual({row["queue_id"] for row in queue_rows}, {row["queue_id"] for row in sidecar_rows})
        self.assertTrue(all("transcript" not in row for row in sidecar_rows))
        self.assertTrue(all("prior_context" not in row for row in sidecar_rows))

    def test_active_bucket_detects_model_heuristic_disagreement(self) -> None:
        bucket = build_active_gold_eval_queue._active_bucket(
            label=0,
            trained_score=-1.0,
            heuristic_score=1.0,
            threshold=0.0,
            near_threshold_window=0.2,
        )

        self.assertEqual(bucket, "model_heuristic_disagreement")


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


def _row(vod_id: str, candidate_id: str, label: int, hook_strength: float) -> dict:
    return {
        "job_id": f"job-{vod_id}",
        "candidate_id": candidate_id,
        "vod_id": vod_id,
        "creator_id": "jynxzi",
        "platform": "twitch",
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "window_start": 60.0,
        "window_end": 90.0,
        "label": label,
        "label_class": "positive" if label else "hard_negative",
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
