from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from config import settings


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def build_jynxzi_workflow_state(
    *,
    date_from: str,
    date_to: str,
    skip_covered: bool,
    merge_output: bool,
) -> dict[str, object]:
    return {
        "workflow": "jynxzi_vod_training",
        "date_from": date_from,
        "date_to": date_to,
        "skip_covered": bool(skip_covered),
        "merge_output": bool(merge_output),
        "whisper_model": settings.whisper_model,
        "dataset_llm_provider": (os.environ.get("TRAINING_DATASET_LLM_PROVIDER") or "").strip() or "ollama",
        "storage_dir": settings.storage_dir,
        "code_hash": _code_hash(),
    }


def should_retrain(
    *,
    labels_path: Path,
    dataset_path: Path,
    artifact_path: Path,
    state_path: Path,
    current_state: dict[str, object],
) -> bool:
    if not labels_path.exists() or not dataset_path.exists() or not artifact_path.exists():
        return True
    stored_state = _read_state(state_path)
    if stored_state != current_state:
        return True
    labels_mtime = labels_path.stat().st_mtime
    dataset_mtime = dataset_path.stat().st_mtime
    artifact_mtime = artifact_path.stat().st_mtime
    return labels_mtime > min(dataset_mtime, artifact_mtime)


def should_reevaluate(
    *,
    dataset_path: Path,
    artifact_path: Path,
    output_paths: list[Path],
    state_path: Path,
    current_state: dict[str, object],
) -> bool:
    if not dataset_path.exists() or not artifact_path.exists():
        return False
    stored_state = _read_state(state_path)
    if stored_state != current_state:
        return True
    if any(not path.exists() for path in output_paths):
        return True
    upstream_mtime = max(dataset_path.stat().st_mtime, artifact_path.stat().st_mtime)
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output < upstream_mtime


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_state(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in _tracked_files():
        relative = path.relative_to(BACKEND_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tracked_files() -> list[Path]:
    tracked: list[Path] = []
    for relative in ("config.py", "pyproject.toml"):
        path = BACKEND_ROOT / relative
        if path.exists():
            tracked.append(path)
    for folder in ("ml", "services"):
        root = BACKEND_ROOT / folder
        if not root.exists():
            continue
        tracked.extend(sorted(path for path in root.rglob("*.py") if path.is_file()))
    return tracked
