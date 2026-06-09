from __future__ import annotations

import logging
import os

# Third-party loggers that are noisy at INFO/DEBUG and drown out pipeline logs.
_NOISY_LOGGERS = (
    "transformers",
    "sentence_transformers",
    "httpx",
    "httpcore",
    "urllib3",
    "asyncio",
    "PIL",
    "matplotlib",
    "numba",
    "filelock",
    "huggingface_hub",
)

_CONFIGURED = False


def configure_job_logging(*, level: int | None = None) -> None:
    """Configure root logging for a detached pipeline subprocess.

    The orchestrator and worker run outside uvicorn, which means nothing sets up
    logging for them — so all of the pipeline's ``logger.info(...)`` progress
    lines were silently dropped (root logger defaults to WARNING). This installs
    a single stderr handler with timestamps so those logs land in
    ``orchestrator.log`` and are actually useful for debugging.

    Idempotent and safe to call more than once. Level can be overridden via the
    ``AI_CLIPPING_LOG_LEVEL`` env var (e.g. ``DEBUG``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level_name = str(os.environ.get("AI_CLIPPING_LOG_LEVEL", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()  # stderr — inherited into orchestrator.log
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any pre-existing handlers so we don't double-log.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
