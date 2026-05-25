from __future__ import annotations

import argparse
import asyncio
import json
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


async def _run(args: argparse.Namespace) -> None:
    # Import inside the worker process so native CV stacks cannot crash FastAPI.
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
    try:
        asyncio.run(_run(args))
    except Exception as exc:
        update_job(args.job_id, status="failed", error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
