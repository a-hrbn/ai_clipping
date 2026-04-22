from __future__ import annotations

import pathlib
import subprocess
import sys


def run_python_module(module: str, args: list[str], *, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> None:
    command = [sys.executable, "-m", module, *args]
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")
