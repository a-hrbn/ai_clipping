import asyncio
import json
import logging
import pathlib
import re
import shutil
import subprocess
from config import settings
from job_manager import is_job_cancelled, update_job
from services import (
    downloader,
    audio_extractor,
    transcriber,
    analyzer,
    clipper,
    audio_features,
    video_features,
    scorer,
    post_processor,
    chat_analyzer,
    candidate_generators,
    candidate_normalizer,
    feature_builder,
    proposal_scorer,
    selector,
    feedback_store,
    peak_detector,
    clip_judge,
    keyframe_sampler,
    chapter_segmenter,
    clip_strategy,
    candidate_recall_report,
)

logger = logging.getLogger(__name__)


class PipelineCancelledError(RuntimeError):
    pass


def _raise_if_cancelled(job_id: str) -> None:
    if is_job_cancelled(job_id):
        raise PipelineCancelledError(f"Job {job_id} cancelled by client")


def _frontend_cache_enabled_for_job(job_id: str) -> bool:
    if bool(getattr(settings, "frontend_job_cache_enabled", False)):
        return True
    job_id = str(job_id or "")
    return job_id.startswith(("twitch_vod", "proposal_training_", "judge_backfill_"))


def _path_is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _cleanup_audio_artifacts(audio_path: pathlib.Path | None) -> None:
    if audio_path is None:
        return
    try:
        audio_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("pipeline_audio_cleanup_failed path=%s", audio_path, exc_info=True)
    chunks_dir = audio_path.parent / "chunks"
    if chunks_dir.exists():
        try:
            shutil.rmtree(chunks_dir)
        except OSError:
            logger.warning("pipeline_audio_chunks_cleanup_failed path=%s", chunks_dir, exc_info=True)


def _cleanup_serving_source_media(
    *,
    job_id: str,
    video_path: pathlib.Path | None,
    audio_path: pathlib.Path | None,
) -> None:
    if _frontend_cache_enabled_for_job(job_id):
        return

    job_dir = pathlib.Path(settings.storage_dir) / job_id
    _cleanup_audio_artifacts(audio_path)

    if video_path is None or not _path_is_relative_to(video_path, job_dir):
        return

    try:
        video_path.unlink(missing_ok=True)
        logger.info("pipeline_source_video_deleted job_id=%s path=%s", job_id, video_path)
    except OSError:
        logger.warning(
            "pipeline_source_video_cleanup_failed job_id=%s path=%s",
            job_id,
            video_path,
            exc_info=True,
        )


def _transcript_cache_source(url: str | None, video_path: pathlib.Path | None) -> str | None:
    if url:
        twitch_vod = re.search(r"twitch\.tv/videos/(\d+)", url.strip().lower())
        if twitch_vod:
            return f"twitch_vod:{twitch_vod.group(1)}"
        return f"url:{url.strip()}"
    if video_path is not None:
        try:
            return f"path:{video_path.resolve()}"
        except OSError:
            return f"path:{video_path}"
    return None


def _probe_media_duration_sec(video_path: pathlib.Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def _load_complete_cached_transcript(
    *,
    cache_source: str | None,
    video_path: pathlib.Path,
    progress_cb,
) -> list[transcriber.RichSegment] | None:
    duration_sec = _probe_media_duration_sec(video_path)
    if duration_sec is None:
        return None
    return transcriber.load_cached_transcript(
        cache_source,
        expected_duration_sec=duration_sec,
        progress_cb=progress_cb,
    )


def _should_extract_full_visual_features(video_path: pathlib.Path) -> bool:
    max_duration_sec = float(settings.pipeline_full_visual_feature_max_duration_sec or 0.0)
    if max_duration_sec <= 0:
        return True
    duration_sec = _probe_media_duration_sec(video_path)
    if duration_sec is None or duration_sec <= max_duration_sec:
        return True
    logger.info(
        "pipeline_full_visual_features_skipped path=%s duration_sec=%.3f max_duration_sec=%.3f",
        video_path,
        duration_sec,
        max_duration_sec,
    )
    return False


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


async def _await_with_optional_timeout(awaitable, timeout_sec: float | None):
    if timeout_sec is None or timeout_sec <= 0:
        return await asyncio.shield(awaitable)
    return await asyncio.wait_for(
        asyncio.shield(awaitable),
        timeout=timeout_sec,
    )


async def _await_clip_judge_or_fallback(
    awaitable,
    *,
    timeout_sec: float | None,
    fallback_candidates: list,
    job_id: str,
) -> tuple[list, bool]:
    try:
        return await _await_with_optional_timeout(awaitable, timeout_sec), False
    except asyncio.TimeoutError:
        fallback = list(fallback_candidates)
        logger.warning(
            json.dumps(
                {
                    "event": "clip_judge_stage_degraded",
                    "job_id": job_id,
                    "reason": "timeout",
                    "timeout_sec": float(timeout_sec or 0.0),
                    "fail_fast": bool(getattr(settings, "clip_judge_fail_fast", False)),
                    "fallback": "pre_judge_ranked_candidates",
                    "fallback_candidates": len(fallback),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return fallback, True


async def run_pipeline(
    job_id: str,
    url: str | None = None,
    video_path: pathlib.Path | None = None,
    profile: str | None = None,
    strategy: dict | None = None,
) -> None:
    effective_profile = profile or settings.clip_profile
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    loop = asyncio.get_running_loop()
    audio_path: pathlib.Path | None = None
    try:
        _raise_if_cancelled(job_id)
        if video_path is None:
            update_job(job_id, status="downloading", progress_pct=5)
            video_path = await loop.run_in_executor(None, downloader.download, url, job_id)
        else:
            update_job(job_id, status="downloading", progress_pct=19)

        def _transcription_progress(done_chunks: int, total_chunks: int, _chunk_path) -> None:
            # Reserve 30-55% for ASR progress.
            pct = 30 + int((done_chunks / max(total_chunks, 1)) * 25)
            update_job(job_id, status="transcribing", progress_pct=min(max(pct, 30), 55))

        _raise_if_cancelled(job_id)
        transcript_cache_source = (
            _transcript_cache_source(url, video_path)
            if _frontend_cache_enabled_for_job(job_id)
            else None
        )
        chat_path = None
        if settings.enable_chat_analysis:
            update_job(job_id, status="downloading_chat", progress_pct=28)
            chat_path = await loop.run_in_executor(
                None,
                chat_analyzer.download_chat,
                url,
                job_id,
                pathlib.Path(settings.storage_dir) / job_id,
            )

        update_job(job_id, status="transcribing", progress_pct=30)
        transcript = await loop.run_in_executor(
            None,
            lambda: (
                _load_complete_cached_transcript(
                    cache_source=transcript_cache_source,
                    video_path=video_path,
                    progress_cb=_transcription_progress,
                )
                if transcript_cache_source
                else None
            ),
        )
        if transcript is not None:
            logger.info(
                "pipeline_transcript_cache_hit job_id=%s source=%s segments=%s",
                job_id,
                transcript_cache_source,
                len(transcript),
            )
        else:
            update_job(job_id, status="extracting_audio", progress_pct=20)
            audio_path = await loop.run_in_executor(None, audio_extractor.extract, video_path, job_id)

            # Start non-transcription tasks in the background, but keep transcription
            # as its own visible phase so long uploads do not appear stuck.
            update_job(job_id, status="transcribing", progress_pct=30)
            transcript = await loop.run_in_executor(
                None,
                lambda: transcriber.transcribe_with_progress(
                    audio_path,
                    _transcription_progress,
                    cache_key=transcript_cache_source,
                    isolate_chunks=bool(settings.pipeline_isolate_transcription_chunks),
                    enable_diarization=bool(settings.pipeline_enable_diarization),
                ),
            )

        overlap_feature_extraction = bool(settings.pipeline_overlap_feature_extraction)
        extract_full_visual_features = _should_extract_full_visual_features(video_path)
        audio_features_task = None
        video_features_task = None
        if overlap_feature_extraction:
            if audio_path is not None:
                audio_features_task = loop.run_in_executor(None, audio_features.extract, audio_path)
            if extract_full_visual_features:
                video_features_task = loop.run_in_executor(None, video_features.extract, video_path)

        _raise_if_cancelled(job_id)
        update_job(job_id, status="extracting_features", progress_pct=56)
        if overlap_feature_extraction:
            visual_frames = await video_features_task if video_features_task is not None else []
            audio_frames = await audio_features_task if audio_features_task is not None else []
        else:
            audio_frames = []
            if audio_path is not None:
                audio_frames = await loop.run_in_executor(None, audio_features.extract, audio_path)
            _raise_if_cancelled(job_id)
            visual_frames = []
            if extract_full_visual_features:
                visual_frames = await loop.run_in_executor(None, video_features.extract, video_path)

        # Merge audio features, then visual features into transcript segments
        transcript = audio_features.enrich_segments(transcript, audio_frames)
        transcript = video_features.enrich_segments(transcript, visual_frames)

        # Speaker diarization — MFCC clustering (runs when pyannote is unavailable)
        # pyannote already ran inside transcriber.transcribe() when hf_token is set
        if audio_path is not None and settings.pipeline_enable_diarization and not settings.hf_token:
            update_job(job_id, status="diarizing", progress_pct=65)
            transcript = await loop.run_in_executor(
                None, audio_features.diarize_speakers, transcript, audio_path
            )

        # Chat-replay enrichment (best-effort — no-op when chat_path is None)
        transcript = chat_analyzer.enrich_segments(transcript, chat_path)

        # Attach composite hype scores before saving or analyzing
        _raise_if_cancelled(job_id)
        transcript = scorer.score_segments(transcript)
        chapters = await loop.run_in_executor(
            None,
            chapter_segmenter.segment_chapters,
            transcript,
        )
        score_timeline = scorer.build_score_timeline(transcript)
        detected_peaks = await loop.run_in_executor(None, peak_detector.detect_peaks, score_timeline)
        peak_candidates = await loop.run_in_executor(
            None,
            peak_detector.build_peak_candidates,
            detected_peaks,
            transcript,
        )

        update_job(job_id, status="analyzing", progress_pct=72)
        semantic_candidates = await loop.run_in_executor(
            None,
            candidate_generators.generate_semantic_candidates,
            transcript,
            chapters,
            normalized_strategy,
        )
        dense_candidates = []
        if settings.enable_whole_vod_dense_candidates:
            dense_candidates = await loop.run_in_executor(
                None,
                candidate_generators.generate_dense_window_candidates,
                transcript,
                chapters,
                normalized_strategy,
            )
        fallback_candidates = []
        if not peak_candidates:
            fallback_candidates = await loop.run_in_executor(
                None,
                candidate_generators.generate_fallback_candidates,
                transcript,
                chapters,
                normalized_strategy,
            )
        heuristic_candidates = await loop.run_in_executor(
            None,
            lambda: candidate_generators.merge_prioritized_candidates(
                peak_candidates or fallback_candidates,
                [*semantic_candidates, *dense_candidates],
                strategy=normalized_strategy,
                source_cap_overrides={
                    "dense_window": max(int(settings.whole_vod_dense_source_cap or 0), 0),
                },
            ),
        )
        dense_inventory_candidates = []
        proposal_result: proposal_scorer.ProposalResult | None = None
        candidates = heuristic_candidates
        if settings.enable_trainable_candidate_proposal:
            dense_inventory_candidates = await loop.run_in_executor(
                None,
                lambda: candidate_generators.generate_dense_candidate_inventory(
                    transcript,
                    chapters,
                    normalized_strategy,
                    durations=_parse_float_list(settings.proposal_dense_durations_sec),
                    stride_sec=float(settings.proposal_dense_stride_sec),
                    max_candidates=max(int(settings.proposal_dense_max_candidates or 0), 0),
                ),
            )
            proposal_inventory_cap = max(int(settings.proposal_dense_max_candidates or 0), 0)
            proposal_inventory_merge_cap = proposal_inventory_cap or len(dense_inventory_candidates)
            proposal_input_candidates = await loop.run_in_executor(
                None,
                lambda: candidate_generators.merge_prioritized_candidates(
                    heuristic_candidates,
                    dense_inventory_candidates,
                    supplemental_source_cap=proposal_inventory_merge_cap,
                    source_cap_overrides={
                        "dense_inventory": proposal_inventory_merge_cap,
                    },
                    strategy=normalized_strategy,
                ),
            )
            proposal_input_candidates = await loop.run_in_executor(
                None,
                candidate_normalizer.normalize_candidates,
                proposal_input_candidates,
                transcript,
                settings.max_clip_duration,
            )
            proposal_result = await loop.run_in_executor(
                None,
                lambda: proposal_scorer.select_candidates(
                    proposal_input_candidates,
                    transcript,
                    effective_profile,
                    normalized_strategy,
                    top_k=max(int(settings.proposal_shortlist_top_k or 0), 0),
                    min_score=float(settings.proposal_min_score or 0.0),
                ),
            )
            proposal_shortlist_cap = max(int(settings.proposal_shortlist_top_k or 0), 0)
            proposal_shortlist_merge_cap = proposal_shortlist_cap or len(proposal_result.kept)
            if settings.proposal_preserve_heuristic_candidates:
                candidates = await loop.run_in_executor(
                    None,
                    lambda: candidate_generators.merge_prioritized_candidates(
                        heuristic_candidates,
                        proposal_result.kept,
                        supplemental_source_cap=proposal_shortlist_merge_cap,
                        source_cap_overrides={
                            "dense_inventory": proposal_shortlist_merge_cap,
                            "dense_window": proposal_shortlist_merge_cap,
                        },
                        strategy=normalized_strategy,
                    ),
                )
            else:
                candidates = proposal_result.kept
            logger.info(
                json.dumps(
                    {
                        "event": "candidate_proposal_stage_done",
                        "job_id": job_id,
                        "heuristic_candidates": len(heuristic_candidates),
                        "dense_inventory_candidates": len(dense_inventory_candidates),
                        "merged_candidates": len(candidates),
                        **proposal_result.summary,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            proposal_debug_dir = pathlib.Path(settings.storage_dir) / job_id / "debug"
            proposal_debug_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(
                proposal_debug_dir / "candidates_dense_inventory.jsonl",
                [candidate.to_dict() for candidate in dense_inventory_candidates],
            )
            _write_jsonl(proposal_debug_dir / "candidate_proposals.jsonl", proposal_result.audit_rows)
            (proposal_debug_dir / "candidate_proposal_summary.json").write_text(
                json.dumps(proposal_result.summary, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        normalized_candidates = await loop.run_in_executor(
            None,
            candidate_normalizer.normalize_candidates,
            candidates,
            transcript,
            settings.max_clip_duration,
        )
        candidate_features = await loop.run_in_executor(
            None,
            lambda: feature_builder.build_features(
                normalized_candidates,
                transcript,
                effective_profile,
                False,
                None,
                None,
                chapters,
                normalized_strategy,
                video_path=video_path,
            ),
        )
        ranked_candidates = await loop.run_in_executor(
            None,
            selector.rank_candidates,
            normalized_candidates,
            candidate_features,
            effective_profile,
            normalized_strategy,
        )
        shortlist_top_k = min(max(settings.shortlist_face_reaction_top_k, 0), len(ranked_candidates))
        llm_top_k = min(max(settings.shortlist_llm_top_k, 0), len(ranked_candidates))
        if shortlist_top_k > 0:
            transcript = await loop.run_in_executor(
                None,
                video_features.refine_face_reactions,
                transcript,
                video_path,
                [ranked.candidate for ranked in ranked_candidates[:shortlist_top_k]],
                settings.shortlist_face_reaction_sample_sec,
            )
        if shortlist_top_k > 0 or llm_top_k > 0:
            llm_candidate_ids = {
                ranked.candidate.id for ranked in ranked_candidates[:llm_top_k]
            } if llm_top_k > 0 else set()
            candidate_features = await loop.run_in_executor(
                None,
                lambda: feature_builder.build_features(
                    normalized_candidates,
                    transcript,
                    effective_profile,
                    True,
                    llm_candidate_ids,
                    None,
                    chapters,
                    normalized_strategy,
                    video_path=video_path,
                ),
            )
            ranked_candidates = await loop.run_in_executor(
                None,
                selector.rank_candidates,
                normalized_candidates,
                candidate_features,
                effective_profile,
                normalized_strategy,
            )
        ranked_candidates_pre_judge = list(ranked_candidates)
        judged_ranked_candidates = list(ranked_candidates)
        judge_top_k = (
            len(ranked_candidates)
            if settings.judge_all_candidates
            else min(max(settings.shortlist_judge_top_k, 0), len(ranked_candidates))
        )
        job_dir = pathlib.Path(settings.storage_dir) / job_id
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        sampled_keyframes: dict[str, list[keyframe_sampler.SampledKeyframe]] = {}
        judge_provider = (getattr(settings, "judge_provider", None) or "auto").strip().lower()
        clip_judge_stage_status = {
            "enabled": bool(settings.enable_clip_judge),
            "attempted": False,
            "status": "pending" if settings.enable_clip_judge and judge_top_k > 0 else "disabled",
            "reason": None,
            "error_type": None,
            "error": None,
            "candidates": len(ranked_candidates),
            "judge_top_k": judge_top_k,
            "judge_provider": judge_provider,
            "stage_timeout_sec": 0.0,
            "provider_request_timeout_sec": float(settings.clip_judge_timeout_sec or 0.0),
            "fail_fast": bool(getattr(settings, "clip_judge_fail_fast", False)),
            "provider_fallback_enabled": bool(getattr(settings, "judge_provider_fallback_enabled", True)),
            "fallback": None,
            "fallback_candidates": 0,
            "processed_candidates": 0,
            "attempted_candidates": 0,
            "recall_filtered_candidates": 0,
            "judged_candidates": 0,
            "elapsed_sec": 0.0,
        }

        def _clip_judge_progress(progress: dict[str, object]) -> None:
            processed = int(progress.get("processed_candidates") or 0)
            total = max(int(progress.get("total_candidates") or judge_top_k or 0), 1)
            clip_judge_stage_status.update(
                {
                    "status": "running",
                    "processed_candidates": processed,
                    "attempted_candidates": int(progress.get("attempted_candidates") or 0),
                    "judged_candidates": int(progress.get("judged_candidates") or 0),
                    "recall_filtered_candidates": int(progress.get("recall_filtered_candidates") or 0),
                    "current_candidate_id": progress.get("candidate_id"),
                    "current_candidate_start": progress.get("candidate_start"),
                    "current_candidate_end": progress.get("candidate_end"),
                    "elapsed_sec": float(progress.get("elapsed_sec") or 0.0),
                }
            )
            _write_json(debug_dir / "clip_judge_stage.json", clip_judge_stage_status)
            progress_fraction = min(max(processed / total, 0.0), 1.0)
            update_job(
                job_id,
                status="judging",
                progress_pct=76 + int(progress_fraction * 10),
            )

        if not settings.enable_clip_judge:
            clip_judge_stage_status.update(
                {
                    "status": "disabled",
                    "reason": "enable_clip_judge_false",
                }
            )
            logger.info(
                json.dumps(
                    {
                        "event": "clip_judge_stage_disabled",
                        "job_id": job_id,
                        "reason": "enable_clip_judge_false",
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        if settings.enable_clip_judge and judge_top_k > 0:
            update_job(job_id, status="judging", progress_pct=76)
            use_anthropic_two_tier = (
                bool(settings.judge_two_tier_enabled)
                and bool(settings.anthropic_api_key)
                and bool(settings.enable_multimodal_clip_judge)
                and int(settings.judge_topk_multimodal_k) > 0
                and judge_provider in {"", "auto", "anthropic", "anthropic_text", "anthropic_multimodal"}
            )
            clip_judge_stage_status.update(
                {
                    "attempted": True,
                    "status": "running",
                    "anthropic_two_tier": use_anthropic_two_tier,
                }
            )
            logger.info(
                json.dumps(
                    {
                        "event": "clip_judge_stage_start",
                        "job_id": job_id,
                        "candidates": len(ranked_candidates),
                        "judge_top_k": judge_top_k,
                        "judge_provider": judge_provider,
                        "anthropic_two_tier": use_anthropic_two_tier,
                        "fail_fast": bool(getattr(settings, "clip_judge_fail_fast", False)),
                        "provider_fallback_enabled": bool(getattr(settings, "judge_provider_fallback_enabled", True)),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            try:
                if use_anthropic_two_tier:
                    text_judge_future = loop.run_in_executor(
                        None,
                        lambda: clip_judge.rerank_candidates(
                            ranked_candidates,
                            transcript,
                            effective_profile,
                            judge_top_k,
                            "anthropic",
                            None,
                            progress_cb=_clip_judge_progress,
                        ),
                    )
                    judged_ranked_candidates = await _await_with_optional_timeout(text_judge_future, None)
                    judge_degraded = False
                    if judge_degraded:
                        clip_judge_stage_status.update(
                            {
                                "status": "degraded",
                                "reason": "timeout",
                                "fallback": "pre_judge_ranked_candidates",
                                "fallback_candidates": len(ranked_candidates_pre_judge),
                            }
                        )
                    if not judge_degraded:
                        multimodal_top_k = min(
                            max(int(settings.judge_topk_multimodal_k), 0),
                            len(judged_ranked_candidates),
                        )
                        if multimodal_top_k > 0:
                            sampled_keyframes = await loop.run_in_executor(
                                None,
                                keyframe_sampler.sample_candidate_keyframes,
                                video_path,
                                judged_ranked_candidates[:multimodal_top_k],
                                debug_dir,
                                settings.judge_keyframes_per_candidate,
                            )
                            multimodal_judge_future = loop.run_in_executor(
                                None,
                                lambda: clip_judge.rerank_candidates(
                                    judged_ranked_candidates,
                                    transcript,
                                    effective_profile,
                                    multimodal_top_k,
                                    "anthropic_multimodal",
                                    sampled_keyframes,
                                    progress_cb=_clip_judge_progress,
                                ),
                            )
                            judged_ranked_candidates = await _await_with_optional_timeout(
                                multimodal_judge_future,
                                None,
                            )
                            _judge_degraded = False
                            if _judge_degraded:
                                clip_judge_stage_status.update(
                                    {
                                        "status": "degraded",
                                        "reason": "timeout",
                                        "fallback": "text_judged_candidates",
                                        "fallback_candidates": len(judged_ranked_candidates),
                                    }
                                )
                else:
                    multimodal_provider_allowed = judge_provider in {
                        "",
                        "auto",
                        "anthropic",
                        "anthropic_multimodal",
                        "gemini",
                    }
                    multimodal_provider_available = bool(
                        multimodal_provider_allowed and (settings.gemini_api_key or settings.anthropic_api_key)
                    )
                    if settings.enable_multimodal_clip_judge and multimodal_provider_available:
                        sampled_keyframes = await loop.run_in_executor(
                            None,
                            keyframe_sampler.sample_candidate_keyframes,
                            video_path,
                            ranked_candidates[:judge_top_k],
                            debug_dir,
                            settings.judge_keyframes_per_candidate,
                        )
                    judge_future = loop.run_in_executor(
                        None,
                        lambda: clip_judge.rerank_candidates(
                            ranked_candidates,
                            transcript,
                            effective_profile,
                            judge_top_k,
                            judge_provider,
                            sampled_keyframes,
                            progress_cb=_clip_judge_progress,
                        ),
                    )
                    judged_ranked_candidates = await _await_with_optional_timeout(judge_future, None)
                    _judge_degraded = False
                    if _judge_degraded:
                        clip_judge_stage_status.update(
                            {
                                "status": "degraded",
                                "reason": "timeout",
                                "fallback": "pre_judge_ranked_candidates",
                                "fallback_candidates": len(ranked_candidates_pre_judge),
                            }
                        )
            except clip_judge.ClipJudgeError as exc:
                clip_judge_stage_status.update(
                    {
                        "status": "failed" if bool(getattr(settings, "clip_judge_fail_fast", False)) else "degraded",
                        "reason": "llm_judge_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "fallback": "pre_judge_ranked_candidates",
                        "fallback_candidates": len(ranked_candidates_pre_judge),
                    }
                )
                logger.error(
                    json.dumps(
                        {
                            "event": "clip_judge_stage_failed",
                            "job_id": job_id,
                            "reason": "llm_judge_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "fail_fast": bool(getattr(settings, "clip_judge_fail_fast", False)),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
                if bool(getattr(settings, "clip_judge_fail_fast", False)):
                    raise
            else:
                judged_count = sum(
                    1 for item in judged_ranked_candidates if item.judge_result is not None
                )
                clip_judge_stage_status["judged_candidates"] = judged_count
                if clip_judge_stage_status.get("status") == "running":
                    clip_judge_stage_status["status"] = "ok" if judged_count > 0 else "degraded"
                    if judged_count <= 0:
                        clip_judge_stage_status.update(
                            {
                                "reason": "empty_judge_result",
                                "fallback": "pre_judge_ranked_candidates",
                                "fallback_candidates": len(ranked_candidates_pre_judge),
                            }
                        )
                logger.info(
                    json.dumps(
                        {
                            "event": "clip_judge_stage_done",
                            "job_id": job_id,
                            "judged_candidates": judged_count,
                            "candidates": len(judged_ranked_candidates),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
        _write_json(debug_dir / "clip_judge_stage.json", clip_judge_stage_status)
        quality_gate_result = await loop.run_in_executor(
            None,
            selector.apply_serving_quality_gate,
            judged_ranked_candidates,
        )
        logger.info(
            json.dumps(
                {
                    "event": "serving_quality_gate_done",
                    "job_id": job_id,
                    **quality_gate_result.summary,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        ranked_candidates = await loop.run_in_executor(
            None,
            lambda: selector.select_diverse_ranked_candidates(
                quality_gate_result.kept,
                12,
                strategy=normalized_strategy,
            ),
        )
        refine_future = loop.run_in_executor(
            None,
            analyzer.refine_candidates,
            ranked_candidates,
            transcript,
            5,
            effective_profile,
            normalized_strategy,
        )
        try:
            moments = await _await_with_optional_timeout(
                refine_future,
                settings.analyzer_refine_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "analyzer.refine_candidates timed out after %.1fs; no moments will be produced for this job",
                settings.analyzer_refine_timeout_sec,
            )
            moments = []

        transcripts_dir = pathlib.Path(__file__).parents[1] / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        (transcripts_dir / f"{job_id}.txt").write_text(
            transcriber.format_transcript(transcript), encoding="utf-8"
        )
        (transcripts_dir / f"{job_id}.jsonl").write_text(
            transcriber.format_jsonl(transcript), encoding="utf-8"
        )
        _write_jsonl(debug_dir / "segments.jsonl", _segment_records(transcript))
        _write_jsonl(debug_dir / "chapters.jsonl", [chapter.to_dict() for chapter in chapters])
        _write_jsonl(debug_dir / "segment_scores.jsonl", _score_records(score_timeline))
        _write_jsonl(debug_dir / "detected_peaks.jsonl", [peak.to_dict() for peak in detected_peaks])
        _write_jsonl(debug_dir / "candidates_peak.jsonl", [candidate.to_dict() for candidate in peak_candidates])
        _write_jsonl(debug_dir / "candidates_semantic.jsonl", [candidate.to_dict() for candidate in semantic_candidates])
        _write_jsonl(debug_dir / "candidates_dense.jsonl", [candidate.to_dict() for candidate in dense_candidates])
        _write_jsonl(
            debug_dir / "candidates_dense_inventory.jsonl",
            [candidate.to_dict() for candidate in dense_inventory_candidates],
        )
        _write_jsonl(
            debug_dir / "candidate_proposals.jsonl",
            proposal_result.audit_rows if proposal_result is not None else [],
        )
        (debug_dir / "candidate_proposal_summary.json").write_text(
            json.dumps(proposal_result.summary if proposal_result is not None else {}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        _write_jsonl(debug_dir / "candidates_fallback.jsonl", [candidate.to_dict() for candidate in fallback_candidates])
        _write_jsonl(debug_dir / "candidates_merged.jsonl", [candidate.to_dict() for candidate in candidates])
        _write_jsonl(
            debug_dir / "candidates_normalized.jsonl",
            [candidate.to_dict() for candidate in normalized_candidates],
        )
        _write_jsonl(
            debug_dir / "candidate_features.jsonl",
            [feature.to_dict() for feature in candidate_features],
        )
        _write_jsonl(
            debug_dir / "ranked_candidates_pre_judge.jsonl",
            [ranked.to_dict() for ranked in ranked_candidates_pre_judge],
        )
        _write_jsonl(
            debug_dir / "judged_candidates.jsonl",
            [
                ranked.to_dict()
                for ranked in judged_ranked_candidates
                if ranked.judge_result is not None
            ],
        )
        _write_jsonl(
            debug_dir / "serving_quality_gate.jsonl",
            quality_gate_result.audit_rows,
        )
        _write_jsonl(
            debug_dir / "judge_keyframes.jsonl",
            [
                frame.to_dict()
                for frames in sampled_keyframes.values()
                for frame in frames
            ],
        )
        _write_jsonl(
            debug_dir / "ranked_candidates.jsonl",
            [ranked.to_dict() for ranked in ranked_candidates],
        )
        _write_jsonl(
            debug_dir / "ranked_candidates_v2.jsonl",
            _ranked_score_records(
                ranked_candidates_pre_judge,
                judged_ranked_candidates,
                ranked_candidates,
            ),
        )
        # Post-process: align to sentence boundaries, then enforce clip diversity
        moments = post_processor.snap_boundaries(moments, transcript)
        moments = post_processor.enforce_diversity(moments)
        moments = await loop.run_in_executor(
            None,
            analyzer.expand_variants,
            moments,
            ranked_candidates,
            transcript,
            normalized_strategy,
        )
        feedback_events = feedback_store.build_pipeline_feedback(
            job_id,
            ranked_candidates,
            moments,
            quality_gate_result.audit_rows,
        )
        feedback_store.write_job_feedback(debug_dir, feedback_events)
        feedback_store.append_feedback_events(pathlib.Path(settings.storage_dir), feedback_events)
        (debug_dir / "final_moments.json").write_text(
            json.dumps(moments, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        recall_report = candidate_recall_report.build_candidate_recall_report(
            stage_candidates={
                "peak": peak_candidates,
                "semantic": semantic_candidates,
                "dense": dense_candidates,
                "dense_inventory": dense_inventory_candidates,
                "fallback": fallback_candidates,
                "merged": candidates,
                "normalized": normalized_candidates,
            },
            proposal_audit_rows=proposal_result.audit_rows if proposal_result is not None else [],
            pre_judge_ranked=ranked_candidates_pre_judge,
            judged_ranked=judged_ranked_candidates,
            quality_gate_rows=quality_gate_result.audit_rows,
            final_ranked=ranked_candidates,
            final_moments=moments,
            judge_top_k=judge_top_k,
        )
        candidate_trace_rows = candidate_recall_report.build_candidate_trace(
            stage_candidates={
                "peak": peak_candidates,
                "semantic": semantic_candidates,
                "dense": dense_candidates,
                "dense_inventory": dense_inventory_candidates,
                "fallback": fallback_candidates,
                "merged": candidates,
                "normalized": normalized_candidates,
            },
            proposal_audit_rows=proposal_result.audit_rows if proposal_result is not None else [],
            pre_judge_ranked=ranked_candidates_pre_judge,
            judged_ranked=judged_ranked_candidates,
            quality_gate_rows=quality_gate_result.audit_rows,
            final_ranked=ranked_candidates,
            final_moments=moments,
            judge_top_k=judge_top_k,
        )
        candidate_funnel_summary = candidate_recall_report.build_candidate_funnel_summary(
            recall_report,
            candidate_trace_rows,
        )
        _write_json(debug_dir / "candidate_recall_report.json", recall_report)
        _write_jsonl(debug_dir / "candidate_trace.jsonl", candidate_trace_rows)
        _write_json(debug_dir / "candidate_funnel_summary.json", candidate_funnel_summary)
        logger.info(
            json.dumps(
                {
                    "event": "candidate_recall_report_built",
                    "job_id": job_id,
                    "stage_counts": recall_report.get("stage_counts", {}),
                    "trace_rows": len(candidate_trace_rows),
                    "drop_stage_counts": candidate_funnel_summary.get("drop_stage_counts", {}),
                    "largest_losses": recall_report.get("largest_losses", [])[:3],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )

        update_job(job_id, status="clipping", progress_pct=88)
        _raise_if_cancelled(job_id)
        clips = await loop.run_in_executor(None, clipper.cut_clips, video_path, moments, job_id)

        _cleanup_audio_artifacts(audio_path)
        audio_path = None

        update_job(job_id, status="done", progress_pct=100, clips=clips)

        # Schedule cleanup of entire job directory after 1 hour
        asyncio.create_task(_cleanup_later(job_id))

    except (PipelineCancelledError, downloader.DownloadCancelledError):
        logger.info("Pipeline cancelled for job %s", job_id)
    except Exception as e:
        logger.exception("Pipeline failed for job %s", job_id)
        update_job(job_id, status="failed", error=str(e))
    finally:
        _cleanup_serving_source_media(
            job_id=job_id,
            video_path=video_path,
            audio_path=audio_path,
        )


async def _cleanup_later(job_id: str) -> None:
    await asyncio.sleep(3600)
    import pathlib
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )


def _parse_float_list(value: str | list[float] | tuple[float, ...] | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    parsed: list[float] = []
    for item in raw_items:
        try:
            parsed_value = float(str(item).strip())
        except ValueError:
            continue
        if parsed_value > 0:
            parsed.append(parsed_value)
    return tuple(parsed)


def _segment_records(transcript) -> list[dict]:
    lines = transcriber.format_jsonl(transcript).splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _score_records(score_timeline) -> list[dict]:
    return [item.to_dict() for item in score_timeline]


def _ranked_score_records(
    pre_judge: list,
    judged: list,
    final_ranked: list,
) -> list[dict]:
    judged_map = {item.candidate.id: item for item in judged}
    final_ids = {item.candidate.id for item in final_ranked}
    rows: list[dict] = []
    for item in pre_judge:
        judged_item = judged_map.get(item.candidate.id, item)
        rows.append(
            {
                "candidate_id": item.candidate.id,
                "source": item.candidate.source,
                "heuristic_rank_score": item.heuristic_rank_score,
                "trained_rank_score": item.trained_rank_score,
                "model_blend": item.model_blend,
                "pre_judge_rank_score": item.rank_score,
                "judged_rank_score": judged_item.rank_score,
                "trained_model_used": item.trained_model_used,
                "judge_used": judged_item.judge_result is not None,
                "selected_after_diversity": item.candidate.id in final_ids,
                "rationale": list(judged_item.rationale),
            }
        )
    rows.sort(key=lambda row: (float(row.get("judged_rank_score") or 0.0), row["candidate_id"]), reverse=True)
    return rows
