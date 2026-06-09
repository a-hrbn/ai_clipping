from __future__ import annotations

import argparse
import os
import pathlib
import sys

_BACKEND_ROOT = pathlib.Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detached pipeline-debug workflow orchestrator. Runs independently of the "
        "uvicorn server so --reload restarts / server crashes do not abort an in-flight run."
    )
    parser.add_argument("--workflow-id", required=True)
    args = parser.parse_args()

    from log_setup import configure_job_logging
    configure_job_logging()

    # Imported after sys.path is prepared so the module resolves standalone.
    from routers.pipeline_debug import (
        _read_json,
        _run_workflow_sync,
        _status_path,
        _update_status,
        _workflow_dir,
        _write_json,
    )

    root = _workflow_dir(args.workflow_id)
    status = _read_json(_status_path(root))
    request = status.get("request") if isinstance(status, dict) else None
    if not isinstance(request, dict):
        print(f"No request payload found for workflow {args.workflow_id}", file=sys.stderr)
        return 2

    # Record our own pid so the run is observable/controllable. _run_workflow's
    # _update_status preserves existing keys via {**current}, so this survives.
    status["orchestrator_pid"] = os.getpid()
    status["orchestrator_log"] = str(root / "orchestrator.log")
    _write_json(_status_path(root), status)

    try:
        _run_workflow_sync(args.workflow_id, request)
    except Exception as exc:  # pragma: no cover - defensive, mirrors thread behaviour
        _update_status(
            args.workflow_id,
            status="failed",
            stage="failed",
            message=f"Orchestrator crashed: {exc}",
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
