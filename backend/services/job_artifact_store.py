"""Job-Artifact-Store — the single owner of the job-artifact *names* and their
json/jsonl/text read+write I/O (PRD #39, tracer #42).

A job's on-disk layout (the ``llm_moment_finder_*`` debug dumps, ``segments.jsonl``,
``final_moments.json``, ``request.json``, ``progress.json``, …) used to be spelled
out as string literals in five modules — producers wrote the names, routers read
them, each independently. Renaming a file on the writer side could silently strand
a reader, and the json/jsonl decode helpers were duplicated three times.

This module concentrates that knowledge. Callers pass only a ``job_id`` and ask for
an artifact *role* (``read_stage``, ``write_segments``, …); they never build a path
and never spell a filename. The base class owns the names and the codecs; the two
adapters own only where the bytes live:

* :class:`RealFsJobArtifactStore` — files under ``storage_dir/<job_id>/...`` (prod).
* :class:`InMemoryJobArtifactStore` — a dict (the "DebugSink as a port" fake parked
  in ``docs/design/clip-pipeline-producers.md``), so artifact-contract tests assert
  in memory instead of reading a tmp directory.

This is slice 1 of 3 (foundation). The reader migration (routers) and the writer
migration (producers) follow in their own tracers. The store knows only paths + I/O;
it deliberately does NOT touch the ``jobs.json`` persistence/lock/reload mechanics in
``job_manager`` — it only exposes that registry's path via :meth:`jobs_registry_path`.
"""
from __future__ import annotations

import abc
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

# --- Artifact role → relative path. The ONE place these names are spelled. ---
# Debug-level artifacts live under ``<job_id>/debug/``; the request/progress
# side-files live at the job root. Reader and writer cross the same names here.
_DEBUG = "debug"
_ROLE_PATHS: dict[str, str] = {
    "stage": f"{_DEBUG}/llm_moment_finder_stage.json",
    "moments": f"{_DEBUG}/llm_moment_finder_moments.jsonl",
    "raw_moments": f"{_DEBUG}/llm_moment_finder_moments_raw.jsonl",
    "usage": f"{_DEBUG}/llm_moment_finder_usage.json",
    "prompt": f"{_DEBUG}/llm_moment_finder_prompt.txt",
    "response": f"{_DEBUG}/llm_moment_finder_response.txt",
    "segments": f"{_DEBUG}/segments.jsonl",
    "final_moments": f"{_DEBUG}/final_moments.json",
    "feedback": f"{_DEBUG}/feedback.jsonl",
    "review_candidates": f"{_DEBUG}/review_candidates.jsonl",
    "request_metadata": "request.json",
    "progress": "progress.json",
}
_JOBS_REGISTRY = "jobs.json"


class JobArtifactStore(abc.ABC):
    """Owner of the artifact names and the json/jsonl/text codecs.

    Subclasses implement only the raw byte storage (:meth:`_read_text` /
    :meth:`_write_text`) plus the two debug-directory queries; everything a caller
    sees — the named per-role accessors and the encode/decode semantics — lives
    here, so the two adapters cannot drift on names or formats.
    """

    # --- raw storage primitives: the only thing an adapter must implement ---

    @abc.abstractmethod
    def _read_text(self, job_id: str, relpath: str) -> str | None:
        """Return the artifact's raw text, or ``None`` if it is absent/unreadable."""

    @abc.abstractmethod
    def _write_text(self, job_id: str, relpath: str, text: str) -> None:
        """Persist the artifact's raw text, creating any parent namespace."""

    @abc.abstractmethod
    def debug_artifacts_present(self, job_id: str) -> bool:
        """Whether the job has any debug-level artifact namespace yet."""

    @abc.abstractmethod
    def debug_last_modified(self, job_id: str) -> str | None:
        """ISO-8601 stamp of the most recently touched debug artifact, or ``None``."""

    @abc.abstractmethod
    def jobs_registry_path(self) -> pathlib.Path:
        """Filesystem path of the ``jobs.json`` registry (paths only — the store
        never reads/writes or locks it; that stays with ``job_manager``)."""

    # --- codecs (owner of the formerly-triplicated json/jsonl helpers) ---
    # JSON: missing/undecodable → None. JSONL: missing → []; malformed or
    # non-dict lines are skipped. Both mirror the helpers they replace.

    def _read_json(self, job_id: str, relpath: str) -> Any | None:
        text = self._read_text(job_id, relpath)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _write_json(self, job_id: str, relpath: str, payload: Any) -> None:
        self._write_text(job_id, relpath, json.dumps(payload, ensure_ascii=True, indent=2))

    def _read_jsonl(self, job_id: str, relpath: str) -> list[dict]:
        text = self._read_text(job_id, relpath)
        if text is None:
            return []
        rows: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def _write_jsonl(self, job_id: str, relpath: str, rows: list[dict]) -> None:
        self._write_text(
            job_id, relpath, "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
        )

    # --- named per-role accessors (callers use only these) ---

    def read_stage(self, job_id: str) -> Any | None:
        return self._read_json(job_id, _ROLE_PATHS["stage"])

    def write_stage(self, job_id: str, payload: Any) -> None:
        self._write_json(job_id, _ROLE_PATHS["stage"], payload)

    def read_moments(self, job_id: str) -> list[dict]:
        return self._read_jsonl(job_id, _ROLE_PATHS["moments"])

    def write_moments(self, job_id: str, rows: list[dict]) -> None:
        self._write_jsonl(job_id, _ROLE_PATHS["moments"], rows)

    def read_raw_moments(self, job_id: str) -> list[dict]:
        return self._read_jsonl(job_id, _ROLE_PATHS["raw_moments"])

    def write_raw_moments(self, job_id: str, rows: list[dict]) -> None:
        self._write_jsonl(job_id, _ROLE_PATHS["raw_moments"], rows)

    def read_usage(self, job_id: str) -> Any | None:
        return self._read_json(job_id, _ROLE_PATHS["usage"])

    def write_usage(self, job_id: str, payload: Any) -> None:
        self._write_json(job_id, _ROLE_PATHS["usage"], payload)

    def read_prompt(self, job_id: str) -> str | None:
        return self._read_text(job_id, _ROLE_PATHS["prompt"])

    def write_prompt(self, job_id: str, text: str) -> None:
        self._write_text(job_id, _ROLE_PATHS["prompt"], text)

    def read_response(self, job_id: str) -> str | None:
        return self._read_text(job_id, _ROLE_PATHS["response"])

    def write_response(self, job_id: str, text: str) -> None:
        self._write_text(job_id, _ROLE_PATHS["response"], text)

    def read_segments(self, job_id: str) -> list[dict]:
        return self._read_jsonl(job_id, _ROLE_PATHS["segments"])

    def write_segments(self, job_id: str, rows: list[dict]) -> None:
        self._write_jsonl(job_id, _ROLE_PATHS["segments"], rows)

    def read_final_moments(self, job_id: str) -> Any | None:
        return self._read_json(job_id, _ROLE_PATHS["final_moments"])

    def write_final_moments(self, job_id: str, payload: Any) -> None:
        self._write_json(job_id, _ROLE_PATHS["final_moments"], payload)

    def read_feedback(self, job_id: str) -> list[dict]:
        return self._read_jsonl(job_id, _ROLE_PATHS["feedback"])

    def write_feedback(self, job_id: str, rows: list[dict]) -> None:
        self._write_jsonl(job_id, _ROLE_PATHS["feedback"], rows)

    def read_review_candidates(self, job_id: str) -> list[dict]:
        return self._read_jsonl(job_id, _ROLE_PATHS["review_candidates"])

    def write_review_candidates(self, job_id: str, rows: list[dict]) -> None:
        self._write_jsonl(job_id, _ROLE_PATHS["review_candidates"], rows)

    def read_request_metadata(self, job_id: str) -> Any | None:
        return self._read_json(job_id, _ROLE_PATHS["request_metadata"])

    def write_request_metadata(self, job_id: str, payload: Any) -> None:
        self._write_json(job_id, _ROLE_PATHS["request_metadata"], payload)

    def read_progress(self, job_id: str) -> Any | None:
        return self._read_json(job_id, _ROLE_PATHS["progress"])

    def write_progress(self, job_id: str, payload: Any) -> None:
        self._write_json(job_id, _ROLE_PATHS["progress"], payload)


class RealFsJobArtifactStore(JobArtifactStore):
    """Filesystem adapter: artifacts live under ``storage_dir/<job_id>/<relpath>``."""

    def __init__(self, storage_dir: str | os.PathLike[str]) -> None:
        self._storage_dir = pathlib.Path(storage_dir)

    def _path(self, job_id: str, relpath: str) -> pathlib.Path:
        return self._storage_dir / job_id / relpath

    def _read_text(self, job_id: str, relpath: str) -> str | None:
        path = self._path(job_id, relpath)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_text(self, job_id: str, relpath: str, text: str) -> None:
        path = self._path(job_id, relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def debug_artifacts_present(self, job_id: str) -> bool:
        return (self._storage_dir / job_id / _DEBUG).exists()

    def debug_last_modified(self, job_id: str) -> str | None:
        debug_dir = self._storage_dir / job_id / _DEBUG
        mtimes: list[float] = []
        try:
            for path in debug_dir.iterdir():
                if path.is_file():
                    mtimes.append(path.stat().st_mtime)
        except OSError:
            return None
        if not mtimes:
            return None
        return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()

    def jobs_registry_path(self) -> pathlib.Path:
        return self._storage_dir / _JOBS_REGISTRY


class InMemoryJobArtifactStore(JobArtifactStore):
    """In-memory adapter (the parked "DebugSink as a port"): artifact-contract
    tests seed/assert against a dict, no tmp directory to create or clean up."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def _read_text(self, job_id: str, relpath: str) -> str | None:
        return self._store.get((job_id, relpath))

    def _write_text(self, job_id: str, relpath: str, text: str) -> None:
        self._store[(job_id, relpath)] = text

    def debug_artifacts_present(self, job_id: str) -> bool:
        prefix = f"{_DEBUG}/"
        return any(jid == job_id and rel.startswith(prefix) for (jid, rel) in self._store)

    def debug_last_modified(self, job_id: str) -> str | None:
        # No wall-clock mtimes in memory; the contract allows None and the only
        # consumer (the diagnostics reader) treats None as "unknown".
        return None

    def jobs_registry_path(self) -> pathlib.Path:
        raise NotImplementedError(
            "InMemoryJobArtifactStore has no on-disk jobs.json registry"
        )
