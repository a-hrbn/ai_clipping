"""MomentFinding — turns a transcript into ranked viral moments via the LLM finder.

Owns the ``stage`` running/ok/failed contract the job router reads, plus the
prompt / response / usage / moments(.raw) debug dumps. Pure: the ~17
``moment_finder_*`` settings are bound into an injected :class:`MomentParams`
snapshot, ``find`` is injected, and the artifacts are written through the injected
:class:`~services.job_artifact_store.JobArtifactStore` (the same injection slot
``debug_dir`` used to fill) — so it never reads ``config.settings``, touches job
state, or spells an artifact filename. Empty ``post_processed_moments`` (the
serving currency) is a failure — it writes the failed marker, then raises the
injected ``MomentFinderError`` (which is also injected rather than imported, since
the real ``moment_finder`` service pulls torch via ``RichSegment``). The eval-only
raw-moments artifact is written only when the snapshot's ``eval_recall`` is on.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING

from .seam import FindMomentsFn, Progress

if TYPE_CHECKING:
    from services.job_artifact_store import JobArtifactStore
    from services.moment_finder import MomentFinderResult
    from services.transcriber import RichSegment

logger = logging.getLogger(__name__)


def _noop_progress(_fraction: float) -> None:
    pass


@dataclasses.dataclass(frozen=True)
class MomentParams:
    """Frozen snapshot of the ~17 ``moment_finder_*`` settings the Conductor reads
    once and injects, so the Producer never touches ``config.settings``."""

    provider: str
    model: str
    max_moments: int
    min_duration_sec: float
    max_duration_sec: float
    max_tokens: int
    chunk_duration_sec: float
    chunk_overlap_sec: float
    timeout_sec: float
    api_retries: int
    retry_backoff_sec: float
    filter_transcript_noise: bool
    temperature: float = 0.0
    # Off for serving: gates the raw-recall branch and the typed RankedCandidate in
    # find_moments (and the moments_raw.jsonl dump here). Eval/capture runs opt in.
    eval_recall: bool = False


class MomentFinding:
    def __init__(
        self,
        *,
        profile: str,
        vod_id: str,
        job_id: str,
        store: "JobArtifactStore",
        params: MomentParams,
        find: FindMomentsFn,
        error: type[Exception],
    ) -> None:
        self._profile = profile
        self._vod_id = vod_id
        self._job_id = job_id
        self._store = store
        self._params = params
        self._find = find
        self._error = error

    def find(
        self, transcript: list["RichSegment"], *, progress: Progress = _noop_progress
    ) -> "MomentFinderResult":
        """Run the LLM moment finder, owning its stage.json + debug-artifact contract.

        Writes ``stage.json="running"`` on entry, ``="ok"`` merged with the
        result summary on success, and ``="failed"`` with the error in the except
        BEFORE re-raising. Empty ``ranked_candidates`` raises the injected error
        after the failed marker is written. ``find_moments`` exposes no sub-step
        progress, so ``progress`` is emitted once at the band start.
        """
        progress(0.0)
        p = self._params
        stage: dict[str, object] = {
            "attempted": True,
            "status": "running",
            "provider": p.provider,
            "model": p.model,
            "max_moments": p.max_moments,
        }
        self._store.write_stage(self._job_id, stage)
        try:
            result = self._find(
                transcript,
                profile=self._profile,
                vod_id=self._vod_id,
                provider=p.provider,
                model=p.model,
                max_moments=p.max_moments,
                min_duration_sec=p.min_duration_sec,
                max_duration_sec=p.max_duration_sec,
                max_tokens=p.max_tokens,
                chunk_duration_sec=p.chunk_duration_sec,
                chunk_overlap_sec=p.chunk_overlap_sec,
                temperature=p.temperature,
                timeout_sec=p.timeout_sec,
                api_retries=p.api_retries,
                retry_backoff_sec=p.retry_backoff_sec,
                filter_transcript_noise=p.filter_transcript_noise,
                eval_recall=p.eval_recall,
            )
            if not result.post_processed_moments:
                raise self._error("LLM moment finder returned no moments.")
            stage.update({"status": "ok", **result.summary})
            self._store.write_prompt(self._job_id, result.prompt)
            self._store.write_response(self._job_id, result.response_text)
            self._store.write_stage(self._job_id, stage)
            self._store.write_usage(self._job_id, result.usage)
            self._store.write_moments(self._job_id, result.post_processed_moments)
            # Eval-only artifact: skipped on serving runs (no raw branch computed),
            # so the debug dir holds only what was actually produced.
            if p.eval_recall:
                self._store.write_raw_moments(self._job_id, result.raw_moments)
            return result
        except self._error as exc:
            stage.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            self._store.write_stage(self._job_id, stage)
            logger.error(
                json.dumps(
                    {
                        "event": "llm_moment_finder_failed",
                        "vod_id": self._vod_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            raise
