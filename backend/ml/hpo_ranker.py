from __future__ import annotations

import argparse
import json
import pathlib
import sys

try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import train_ranker


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune the LightGBM ranker with Optuna.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--trials", type=int, default=int(settings.ranker_hpo_trials))
    parser.add_argument("--seed", type=int, default=int(settings.ranker_hpo_seed))
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    rows = train_ranker._read_jsonl(pathlib.Path(args.dataset))
    report_path = pathlib.Path(args.report_out) if args.report_out else None
    return run_hpo(
        rows,
        args.profile,
        pathlib.Path(args.artifact_dir),
        trials=int(args.trials),
        seed=int(args.seed),
        report_path=report_path,
    )


def run_hpo(
    rows: list[dict],
    profile: str,
    artifact_dir: pathlib.Path,
    *,
    trials: int,
    seed: int,
    report_path: pathlib.Path | None = None,
) -> int:
    if optuna is None:
        print("Optuna is not installed. Install the backend ml extras before running HPO.")
        return 1
    if train_ranker.LGBMRanker is None:
        print("LightGBM is not installed. Install the backend ml extras before running HPO.")
        return 1
    context = _build_context(rows)
    if context is None:
        print("Need both train and validation rows with positive and negative examples for HPO.")
        return 1

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_objective(context, seed), n_trials=max(1, int(trials)))

    best_params = dict(study.best_params)
    best_candidate = train_ranker._fit_lightgbm_ranker(
        context["train_rows"],
        context["validation_rows"],
        context["feature_names"],
        context["scales"],
        context["split_unit"],
        hpo_params=best_params,
        random_state=seed,
    )
    if best_candidate is None:
        print("Best HPO parameters did not produce a usable LightGBM candidate.")
        return 1

    artifact = {
        "version": 1,
        "artifact_family": "ranker_hpo",
        "profile": profile,
        "model_type": "lightgbm_ranker",
        "selection_metric": "validation_ndcg_at_5",
        "best_value": round(float(study.best_value), 3),
        "best_params": best_params,
        "best_metrics": best_candidate["metrics"],
        "seed": int(seed),
        "trials": max(1, int(trials)),
        "training_stats": {
            "rows": len(rows),
            "train_rows": len(context["train_rows"]),
            "validation_rows": len(context["validation_rows"]),
            "split_strategy": context["split"]["strategy"],
            "split_unit": context["split"]["unit"],
        },
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{profile}_ranker_hpo.json"
    tmp_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(artifact_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote HPO artifact to {artifact_path}")
    return 0


def _build_context(rows: list[dict]) -> dict[str, object] | None:
    if not rows:
        return None
    positive = [row for row in rows if row.get("label") == 1]
    negative = [row for row in rows if row.get("label") == 0]
    if not positive or not negative:
        return None
    split = train_ranker._split_rows(rows)
    train_rows = list(split["train"])
    validation_rows = list(split["validation"])
    if not train_rows or not validation_rows:
        return None
    train_positive = [row for row in train_rows if row["label"] == 1]
    train_negative = [row for row in train_rows if row["label"] == 0]
    if not train_positive or not train_negative:
        return None
    feature_names = train_ranker._numeric_feature_names(train_positive[0]["features"])
    scales = train_ranker._compute_scales(train_rows, feature_names)
    return {
        "split": split,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "feature_names": feature_names,
        "scales": scales,
        "split_unit": str(split.get("unit") or ""),
    }


def _objective(context: dict[str, object], seed: int):
    train_rows = list(context["train_rows"])
    validation_rows = list(context["validation_rows"])
    feature_names = list(context["feature_names"])
    scales = dict(context["scales"])
    split_unit = str(context["split_unit"])

    def inner(trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 40),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 40, 240),
        }
        candidate = train_ranker._fit_lightgbm_ranker(
            train_rows,
            validation_rows,
            feature_names,
            scales,
            split_unit,
            hpo_params=params,
            random_state=seed,
        )
        if candidate is None:
            return 0.0
        return float(candidate["metrics"].get("ndcg_at_5", 0.0))

    return inner


if __name__ == "__main__":
    raise SystemExit(main())
