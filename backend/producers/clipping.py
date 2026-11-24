"""Clipping — turns ranked moments into cut clips.

Post-processes the moment-finder rows into clip candidates, selects the top-K,
writes the ``final_moments`` artifact, then cuts the clips — mapping the cutter's
``(so_far, done, total)`` callback onto a 0..1 progress fraction. Pure:
``job_id`` / ``top_k`` / ``platform``, the injected
:class:`~services.job_artifact_store.JobArtifactStore` (the same injection slot
``debug_dir`` used to fill), and the ``cut`` leaf op are injected, so it never
reads ``config.settings`` or spells an artifact filename. The partial-clips
streaming (``update_job(clips=so_far)``) stays a Conductor concern — the Conductor
injects a ``cut`` that wraps the real cutter with that streaming.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import TYPE_CHECKING

from services.feedback_store import initial_feedback_rows

from .seam import CutClipsFn, Progress

if TYPE_CHECKING:
    from models import ClipResult
    from services.job_artifact_store import JobArtifactStore
    from services.moment_finder import MomentFinderResult

logger = logging.getLogger(__name__)


def _noop_progress(_fraction: float) -> None:
    pass


def moments_from_post_processed(rows: list[dict], platform: str) -> list[dict]:
    moments = []
    for i, row in enumerate(rows, start=1):
        start = float(row["start_time"])
        end = float(row["end_time"])
        moments.append({
            "start_time": start,
            "end_time": end,
            "viral_score": float(row.get("viral_score") or 1.0),
            "composite_score": float(row.get("composite_score") or row.get("viral_score") or 1.0),
            "reason": str(row.get("reason") or ""),
            "candidate_id": f"llm_moment_{i:02d}",
            "ranking_mode": "llm_moment_finder",
            "platform": platform,
            "projected_hook_score": row.get("hook"),
            "projected_retention_score": row.get("flow"),
            "value_score": row.get("value"),
            "trend_score": row.get("trend"),
        })
    return moments


class Clipping:
    def __init__(
        self,
        *,
        job_id: str,
        store: "JobArtifactStore",
        top_k: int,
        platform: str,
        cut: CutClipsFn,
    ) -> None:
        self._job_id = job_id
        self._store = store
        self._top_k = top_k
        self._platform = platform
        self._cut = cut

    def run(
        self,
        video_path: pathlib.Path,
        result: "MomentFinderResult",
        *,
        progress: Progress = _noop_progress,
    ) -> "list[ClipResult]":
        """Post-process → top-K select → write the final_moments artifact → cut.

        ``top_k > 0`` truncates to the K highest-ranked moments; otherwise all are
        kept (the full candidate set stays in the moment-finder debug JSONLs). The
        cutter's ``(so_far, done, total)`` callback is remapped to a 0..1 fraction
        on ``progress`` (the Conductor maps it onto the clipping band); ``so_far``
        streaming is handled inside the injected ``cut``.
        """
        progress(0.0)
        candidates = moments_from_post_processed(result.post_processed_moments, self._platform)
        moments = candidates[: self._top_k] if self._top_k > 0 else candidates
        self._store.write_final_moments(
            self._job_id, {"moments": moments, "count": len(moments)}
        )
        logger.info(
            json.dumps(
                {
                    "event": "pipeline_moments_ready",
                    "job_id": self._job_id,
                    "candidate_count": len(candidates),
                    "clipped_count": len(moments),
                    "clip_top_k": self._top_k,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )

        def on_cut(_so_far: list, done: int, total: int) -> None:
            if total > 0:
                progress(min(max(done / total, 0.0), 1.0))

        clips = self._cut(video_path, moments, self._job_id, progress_cb=on_cut)
        self._store.write_feedback(self._job_id, initial_feedback_rows(self._job_id, moments, clips))
        return clips
