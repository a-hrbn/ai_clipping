from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_EXPERIMENT_NAME = "ai-clipping"
_DEFAULT_PROMOTION_STAGE = "Staging"
_DEFAULT_PROMOTION_NDCG_FLOOR = 0.70
_MAX_CANDIDATE_SOURCE_SHARE = 0.40


def _import_mlflow():
    try:
        import mlflow
        from mlflow import MlflowClient
    except Exception:
        return None, None
    return mlflow, MlflowClient


def tracking_uri() -> str:
    return str(os.environ.get("MLFLOW_TRACKING_URI") or "").strip()


def experiment_name() -> str:
    value = str(os.environ.get("MLFLOW_EXPERIMENT_NAME") or "").strip()
    return value or _DEFAULT_EXPERIMENT_NAME


def is_configured() -> bool:
    return bool(tracking_uri())


def registry_model_name(profile: str) -> str:
    return f"{profile}_ranker"


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    return text or None


def _clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        coerced = _coerce_scalar(value)
        if coerced is None:
            continue
        cleaned[str(key)] = coerced
    return cleaned


def _metric_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def training_params_from_context(profile: str, rows: list[dict]) -> dict[str, Any]:
    date_from = str(os.environ.get("DATE_FROM") or "").strip()
    date_to = str(os.environ.get("DATE_TO") or "").strip()
    window_size = str(os.environ.get("TRAINING_WINDOW_SIZE") or "").strip()
    blend_floor = str(os.environ.get("TRAINING_BLEND_FLOOR") or "").strip()
    streamer = str(
        os.environ.get("TRAINING_STREAMER")
        or os.environ.get("STREAMER")
        or profile
    ).strip()
    return _clean_mapping(
        {
            "streamer": streamer or profile,
            "profile": profile,
            "date_range": f"{date_from}..{date_to}" if date_from and date_to else "",
            "window_size": window_size,
            "blend_floor": blend_floor,
            "dataset_rows": len(rows),
            "positive_rows": sum(int(row.get("label", 0)) == 1 for row in rows),
        }
    )


def training_tags_from_context(profile: str, rows: list[dict]) -> dict[str, str]:
    image_tag = str(
        os.environ.get("LOCAL_DOCKER_IMAGE")
        or os.environ.get("IMAGE_TAG")
        or os.environ.get("K_REVISION")
        or ""
    ).strip()
    run_id = str(
        os.environ.get("LOCAL_TRAIN_RUN_ID")
        or ""
    ).strip()
    return {
        key: str(value)
        for key, value in _clean_mapping(
            {
                "profile": profile,
                "git_sha": _git_sha(),
                "run_id": run_id,
                "image_tag": image_tag,
                "dataset_rows": len(rows),
                "positive_rows": sum(int(row.get("label", 0)) == 1 for row in rows),
            }
        ).items()
    }


@dataclass
class MlflowTrainingSession:
    enabled: bool
    mlflow: Any = None
    client: Any = None
    run_id: str = ""
    artifact_uri: str = ""

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        cleaned = _clean_mapping(params)
        if cleaned:
            self.mlflow.log_params(cleaned)

    def log_metrics(self, metrics: dict[str, Any], *, prefix: str = "") -> None:
        if not self.enabled:
            return
        cleaned: dict[str, float] = {}
        for key, value in metrics.items():
            metric_value = _metric_value(value)
            if metric_value is None:
                continue
            cleaned[f"{prefix}{key}" if prefix else str(key)] = float(metric_value)
        if cleaned:
            self.mlflow.log_metrics(cleaned)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if not self.enabled:
            return
        cleaned = {key: str(value) for key, value in _clean_mapping(tags).items()}
        if cleaned:
            self.mlflow.set_tags(cleaned)

    def log_artifact(self, path: pathlib.Path, *, artifact_path: str | None = None) -> None:
        if not self.enabled or not path.exists():
            return
        self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def register_ranker_bundle(
        self,
        *,
        profile: str,
        artifact_path: pathlib.Path,
        artifact_health_path: pathlib.Path,
        model_path: pathlib.Path | None,
        proxy_eval_path: pathlib.Path | None = None,
        gold_eval_path: pathlib.Path | None = None,
        pipeline_eval_path: pathlib.Path | None = None,
        ndcg_floor: float = _DEFAULT_PROMOTION_NDCG_FLOOR,
        positive_rows: int = 0,
        candidate_source_peak_share: float = 1.0,
        validation_ndcg_at_5: float = 0.0,
    ) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            return {"enabled": False}

        artifacts: dict[str, str] = {
            artifact_path.name: str(artifact_path),
            artifact_health_path.name: str(artifact_health_path),
        }
        if model_path is not None and model_path.exists():
            artifacts[model_path.name] = str(model_path)
        for optional_path in (proxy_eval_path, gold_eval_path, pipeline_eval_path):
            if optional_path is not None and optional_path.exists():
                artifacts[optional_path.name] = str(optional_path)

        mlflow = self.mlflow
        model_name = registry_model_name(profile)

        class _RankerBundleModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input):  # pragma: no cover - registry wrapper only
                return model_input

        model_info = mlflow.pyfunc.log_model(
            name="ranker_bundle",
            python_model=_RankerBundleModel(),
            artifacts=artifacts,
            input_example=[{}],
        )
        model_uri = getattr(model_info, "model_uri", "") or f"runs:/{self.run_id}/ranker_bundle"
        registered = mlflow.register_model(model_uri=model_uri, name=model_name)

        source_mix_ok = candidate_source_peak_share <= _MAX_CANDIDATE_SOURCE_SHARE
        promotion_ok = (
            positive_rows >= 15
            and validation_ndcg_at_5 >= ndcg_floor
            and source_mix_ok
        )
        stage = ""
        if promotion_ok:
            try:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=str(registered.version),
                    stage=_DEFAULT_PROMOTION_STAGE,
                )
                stage = _DEFAULT_PROMOTION_STAGE
            except Exception as exc:  # pragma: no cover - network/service dependent
                logger.warning(
                    "mlflow_stage_transition_failed model=%s version=%s reason=%s",
                    model_name,
                    getattr(registered, "version", ""),
                    exc,
                )

        result = {
            "enabled": True,
            "model_name": model_name,
            "model_uri": model_uri,
            "version": str(getattr(registered, "version", "")),
            "stage": stage,
            "promotion_ok": promotion_ok,
            "promotion_gates": {
                "positive_rows": int(positive_rows),
                "ndcg_at_5": float(validation_ndcg_at_5),
                "ndcg_floor": float(ndcg_floor),
                "source_mix_ok": bool(source_mix_ok),
                "candidate_source_peak_share": float(candidate_source_peak_share),
            },
        }
        self.set_tags(
            {
                "mlflow_registered_model": model_name,
                "mlflow_registered_version": result["version"],
                "mlflow_registered_stage": stage,
            }
        )
        return result


@contextlib.contextmanager
def start_training_session(
    *,
    profile: str,
    rows: list[dict],
    run_name: str | None = None,
) -> MlflowTrainingSession:
    mlflow, mlflow_client_cls = _import_mlflow()
    if mlflow is None or mlflow_client_cls is None or not is_configured():
        yield MlflowTrainingSession(enabled=False)
        return

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment_name())
    with mlflow.start_run(run_name=run_name or f"{profile}-ranker-train") as run:
        session = MlflowTrainingSession(
            enabled=True,
            mlflow=mlflow,
            client=mlflow_client_cls(),
            run_id=run.info.run_id,
            artifact_uri=str(getattr(run.info, "artifact_uri", "") or ""),
        )
        session.log_params(training_params_from_context(profile, rows))
        session.set_tags(training_tags_from_context(profile, rows))
        yield session


def finalize_training_run(
    *,
    artifact_path: pathlib.Path,
    artifact_health_path: pathlib.Path | None = None,
    proxy_eval_path: pathlib.Path | None = None,
    gold_eval_path: pathlib.Path | None = None,
    pipeline_eval_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    mlflow, _mlflow_client_cls = _import_mlflow()
    if mlflow is None or not is_configured() or not artifact_path.exists():
        return {"enabled": False}

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    run_id = str(
        ((artifact.get("mlflow") or {}).get("run_id"))
        or artifact.get("mlflow_run_id")
        or ""
    ).strip()
    if not run_id:
        return {"enabled": False, "reason": "missing_run_id"}

    mlflow.set_tracking_uri(tracking_uri())
    with mlflow.start_run(run_id=run_id):
        session = MlflowTrainingSession(
            enabled=True,
            mlflow=mlflow,
            run_id=run_id,
        )
        for path in (artifact_path, artifact_health_path, proxy_eval_path, gold_eval_path, pipeline_eval_path):
            if path is not None and path.exists():
                session.log_artifact(path)

        if proxy_eval_path is not None and proxy_eval_path.exists():
            proxy_eval = json.loads(proxy_eval_path.read_text(encoding="utf-8"))
            runtime_metrics = (((proxy_eval.get("metrics") or {}).get("trained_runtime")) or {})
            session.log_metrics(runtime_metrics, prefix="proxy_eval_")

        if gold_eval_path is not None and gold_eval_path.exists():
            gold_eval = json.loads(gold_eval_path.read_text(encoding="utf-8"))
            runtime_metrics = (((gold_eval.get("metrics") or {}).get("trained_runtime")) or {})
            session.log_metrics(runtime_metrics, prefix="gold_eval_")
            gold_ndcg = _metric_value(runtime_metrics.get("ndcg_at_5"))
            if gold_ndcg is not None:
                session.log_metrics({"gold_eval_ndcg": gold_ndcg})

        if pipeline_eval_path is not None and pipeline_eval_path.exists():
            pipeline_eval = json.loads(pipeline_eval_path.read_text(encoding="utf-8"))
            classification = (((pipeline_eval.get("metrics") or {}).get("classification")) or {})
            session.log_metrics(classification, prefix="pipeline_eval_")
            session.log_metrics(
                {
                    "pipeline_top1_positive_rate": (pipeline_eval.get("metrics") or {}).get("top1_positive_rate"),
                    "pipeline_recall_at_10": (pipeline_eval.get("metrics") or {}).get("recall_at_10"),
                    "pipeline_mrr": (pipeline_eval.get("metrics") or {}).get("mrr"),
                }
            )
    return {"enabled": True, "run_id": run_id}


def register_existing_artifact_bundle(
    *,
    artifact_path: pathlib.Path,
    artifact_health_path: pathlib.Path | None = None,
    proxy_eval_path: pathlib.Path | None = None,
    gold_eval_path: pathlib.Path | None = None,
    pipeline_eval_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    mlflow, mlflow_client_cls = _import_mlflow()
    if mlflow is None or mlflow_client_cls is None or not is_configured() or not artifact_path.exists():
        return {"enabled": False}

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    profile = str(artifact.get("profile") or "").strip()
    run_id = str(
        ((artifact.get("mlflow") or {}).get("run_id"))
        or artifact.get("mlflow_run_id")
        or ""
    ).strip()
    if not profile or not run_id:
        return {"enabled": False, "reason": "missing_profile_or_run_id"}

    model_path = str(artifact.get("model_path") or "").strip()
    resolved_model_path = artifact_path.parent / model_path if model_path else None

    mlflow.set_tracking_uri(tracking_uri())
    with mlflow.start_run(run_id=run_id):
        session = MlflowTrainingSession(
            enabled=True,
            mlflow=mlflow,
            client=mlflow_client_cls(),
            run_id=run_id,
        )
        return session.register_ranker_bundle(
            profile=profile,
            artifact_path=artifact_path,
            artifact_health_path=artifact_health_path or artifact_path.parent / f"{profile}_artifact_health.json",
            model_path=resolved_model_path if resolved_model_path and resolved_model_path.exists() else None,
            proxy_eval_path=proxy_eval_path,
            gold_eval_path=gold_eval_path,
            pipeline_eval_path=pipeline_eval_path,
            positive_rows=int(((artifact.get("training_stats") or {}).get("positive_rows")) or 0),
            candidate_source_peak_share=float(((artifact.get("training_stats") or {}).get("candidate_source_peak_share")) or 1.0),
            validation_ndcg_at_5=float(((artifact.get("validation") or {}).get("ndcg_at_5")) or 0.0),
        )


def load_registered_ranker_artifact(
    *,
    profile: str,
    stage: str = "Production",
) -> dict[str, Any] | None:
    mlflow, _mlflow_client_cls = _import_mlflow()
    if mlflow is None or not is_configured():
        return None

    mlflow.set_tracking_uri(tracking_uri())
    model_name = registry_model_name(profile)
    try:
        download_dir = pathlib.Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{model_name}/{stage}",
            )
        )
    except Exception:
        return None

    artifact_candidates = sorted(download_dir.rglob(f"{profile}_ranker.json"))
    if not artifact_candidates:
        artifact_candidates = sorted(download_dir.rglob("*_ranker.json"))
    if not artifact_candidates:
        return None

    artifact_path = artifact_candidates[0]
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    artifact["_artifact_base_dir"] = str(artifact_path.parent)
    artifact["_artifact_source"] = "mlflow_registry"
    artifact["_artifact_registry_model_name"] = model_name
    artifact["_artifact_registry_stage"] = stage
    return artifact


def promote_registered_model(*, profile: str, version: str, stage: str) -> dict[str, Any]:
    mlflow, mlflow_client_cls = _import_mlflow()
    if mlflow is None or mlflow_client_cls is None or not is_configured():
        return {"enabled": False}
    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow_client_cls()
    model_name = registry_model_name(profile)
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
    )
    return {
        "enabled": True,
        "model_name": model_name,
        "version": str(version),
        "stage": stage,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utilities for MLflow tracking around the AI clipping training flow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    finalize_parser = subparsers.add_parser("finalize-run")
    finalize_parser.add_argument("--artifact-path", required=True)
    finalize_parser.add_argument("--artifact-health-path", default="")
    finalize_parser.add_argument("--proxy-eval-path", default="")
    finalize_parser.add_argument("--gold-eval-path", default="")
    finalize_parser.add_argument("--pipeline-eval-path", default="")

    promote_parser = subparsers.add_parser("promote-model")
    promote_parser.add_argument("--profile", required=True)
    promote_parser.add_argument("--version", required=True)
    promote_parser.add_argument("--stage", required=True)

    register_parser = subparsers.add_parser("register-bundle")
    register_parser.add_argument("--artifact-path", required=True)
    register_parser.add_argument("--artifact-health-path", default="")
    register_parser.add_argument("--proxy-eval-path", default="")
    register_parser.add_argument("--gold-eval-path", default="")
    register_parser.add_argument("--pipeline-eval-path", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "finalize-run":
        payload = finalize_training_run(
            artifact_path=pathlib.Path(args.artifact_path),
            artifact_health_path=pathlib.Path(args.artifact_health_path) if args.artifact_health_path else None,
            proxy_eval_path=pathlib.Path(args.proxy_eval_path) if args.proxy_eval_path else None,
            gold_eval_path=pathlib.Path(args.gold_eval_path) if args.gold_eval_path else None,
            pipeline_eval_path=pathlib.Path(args.pipeline_eval_path) if args.pipeline_eval_path else None,
        )
        print(json.dumps(payload, ensure_ascii=True))
        return 0

    if args.command == "register-bundle":
        payload = register_existing_artifact_bundle(
            artifact_path=pathlib.Path(args.artifact_path),
            artifact_health_path=pathlib.Path(args.artifact_health_path) if args.artifact_health_path else None,
            proxy_eval_path=pathlib.Path(args.proxy_eval_path) if args.proxy_eval_path else None,
            gold_eval_path=pathlib.Path(args.gold_eval_path) if args.gold_eval_path else None,
            pipeline_eval_path=pathlib.Path(args.pipeline_eval_path) if args.pipeline_eval_path else None,
        )
        print(json.dumps(payload, ensure_ascii=True))
        return 0

    payload = promote_registered_model(
        profile=args.profile,
        version=args.version,
        stage=args.stage,
    )
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
