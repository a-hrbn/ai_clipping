import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import hpo_ranker


class _FakeTrial:
    def __init__(self, params: dict[str, float]) -> None:
        self._params = params

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return int(self._params[name])

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        return float(self._params[name])


class _FakeStudy:
    def __init__(self, param_sets: list[dict[str, float]]) -> None:
        self._param_sets = param_sets
        self.best_params: dict[str, float] = {}
        self.best_value = 0.0

    def optimize(self, objective, n_trials: int) -> None:
        best_value = float("-inf")
        best_params: dict[str, float] = {}
        for params in self._param_sets[:n_trials]:
            value = float(objective(_FakeTrial(params)))
            if value > best_value:
                best_value = value
                best_params = dict(params)
        self.best_value = best_value
        self.best_params = best_params


class HpoRankerTests(unittest.TestCase):
    def test_run_hpo_writes_best_params_artifact(self) -> None:
        rows = [
            {
                "vod_id": f"vod-{i // 2}",
                "candidate_id": f"c{i}",
                "label": 1 if i % 2 == 0 else 0,
                "ordinal_label": 3 if i % 2 == 0 else 0,
                "features": {"signal": 5.0 if i % 2 == 0 else -5.0},
            }
            for i in range(8)
        ]
        fake_optuna = types.SimpleNamespace(
            samplers=types.SimpleNamespace(TPESampler=lambda seed: ("sampler", seed)),
            create_study=lambda direction, sampler: _FakeStudy(
                [
                    {
                        "num_leaves": 15,
                        "learning_rate": 0.09,
                        "min_child_samples": 6,
                        "feature_fraction": 0.9,
                        "bagging_fraction": 0.9,
                        "n_estimators": 60,
                    },
                    {
                        "num_leaves": 31,
                        "learning_rate": 0.05,
                        "min_child_samples": 7,
                        "feature_fraction": 0.8,
                        "bagging_fraction": 0.75,
                        "n_estimators": 120,
                    },
                ]
            ),
        )

        def fake_fit(train_rows, validation_rows, feature_names, scales, split_unit, *, hpo_params=None, random_state=0):
            score = 0.82 if int(hpo_params["num_leaves"]) == 31 else 0.64
            return {
                "artifact_model_type": "lightgbm_ranker",
                "metrics": {"ndcg_at_5": score, "precision_at_5": score - 0.05, "accuracy": 0.7, "f1": 0.7},
            }

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(hpo_ranker, "optuna", fake_optuna), \
             patch("ml.hpo_ranker.train_ranker.LGBMRanker", object()), \
             patch("ml.hpo_ranker.train_ranker._fit_lightgbm_ranker", side_effect=fake_fit):
            artifact_dir = Path(tmp)
            rc = hpo_ranker.run_hpo(rows, "gaming", artifact_dir, trials=2, seed=13)
            self.assertEqual(rc, 0)
            payload = json.loads((artifact_dir / "gaming_ranker_hpo.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["best_params"]["num_leaves"], 31)
        self.assertEqual(payload["selection_metric"], "composite_validation_score")
        self.assertEqual(payload["best_value"], 0.813)
        self.assertEqual(payload["seed"], 13)
        self.assertEqual(payload["trials"], 2)

    def test_run_hpo_fails_without_optuna(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(hpo_ranker, "optuna", None):
            rc = hpo_ranker.run_hpo([], "gaming", Path(tmp), trials=2, seed=0)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
