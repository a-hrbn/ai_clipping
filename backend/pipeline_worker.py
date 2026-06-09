from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys

from job_manager import update_job


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one clipping pipeline job in an isolated subprocess."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--url", default="")
    parser.add_argument("--video-path", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--strategy-json", default="{}")
    return parser.parse_args(argv)


def _load_strategy(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid strategy JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("strategy JSON must decode to an object")
    return parsed


def _preload_semantic_candidates_before_pipeline_import() -> bool:
    """Load SentenceTransformer before pipeline imports TensorFlow/DeepFace.

    The serving container can segfault when the semantic candidate model is
    lazily imported later from an executor thread after the CV stack has already
    initialized native libraries. The worker owns this import ordering because
    it runs in the isolated subprocess.
    """

    try:
        from services import semantic_candidates

        preload = getattr(semantic_candidates, "preload_model", None)
        if callable(preload):
            return bool(preload())
    except Exception as exc:
        print(f"semantic_candidate_preload_failed error={exc}", file=sys.stderr)
    return False


async def _run(args: argparse.Namespace) -> None:
    # Import inside the worker process so native CV stacks cannot crash FastAPI.
    _preload_semantic_candidates_before_pipeline_import()
    from pipeline import run_pipeline

    video_path = pathlib.Path(args.video_path) if args.video_path else None
    await run_pipeline(
        args.job_id,
        url=args.url or None,
        video_path=video_path,
        profile=args.profile or None,
        strategy=_load_strategy(args.strategy_json),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from log_setup import configure_job_logging
    configure_job_logging()
    logging.getLogger("pipeline_worker").info(
        "pipeline_worker_start job_id=%s url=%s profile=%s",
        args.job_id, args.url or "(none)", args.profile or "(none)",
    )
    try:
        asyncio.run(_run(args))
    except Exception as exc:
        logging.getLogger("pipeline_worker").exception("pipeline_worker_failed job_id=%s", args.job_id)
        update_job(args.job_id, status="failed", error=str(exc))
        return 1
    logging.getLogger("pipeline_worker").info("pipeline_worker_done job_id=%s", args.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
