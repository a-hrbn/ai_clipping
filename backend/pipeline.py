import asyncio
import json
import logging
import pathlib
import shutil
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
    selector,
    feedback_store,
    peak_detector,
    clip_judge,
    keyframe_sampler,
    chapter_segmenter,
    clip_strategy,
)

logger = logging.getLogger(__name__)


class PipelineCancelledError(RuntimeError):
    pass


def _raise_if_cancelled(job_id: str) -> None:
    if is_job_cancelled(job_id):
        raise PipelineCancelledError(f"Job {job_id} cancelled by client")


async def _await_with_optional_timeout(awaitable, timeout_sec: float | None):
    if timeout_sec is None or timeout_sec <= 0:
        return await asyncio.shield(awaitable)
    return await asyncio.wait_for(
        asyncio.shield(awaitable),
        timeout=timeout_sec,
    )


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
    try:
        _raise_if_cancelled(job_id)
        if video_path is None:
            update_job(job_id, status="downloading", progress_pct=5)
            video_path = await loop.run_in_executor(None, downloader.download, url, job_id)
        else:
            update_job(job_id, status="downloading", progress_pct=19)

        _raise_if_cancelled(job_id)
        update_job(job_id, status="extracting_audio", progress_pct=20)
        audio_path = await loop.run_in_executor(None, audio_extractor.extract, video_path, job_id)

        # Start non-transcription tasks in the background, but keep transcription
        # as its own visible phase so long uploads do not appear stuck.
        update_job(job_id, status="transcribing", progress_pct=30)
        # Chat download is currently disabled. Keep the enrichment call below as
        # a stable no-op path so downstream feature extraction remains unchanged.
        chat_path = None

        audio_features_task = loop.run_in_executor(None, audio_features.extract, audio_path)
        video_features_task = loop.run_in_executor(None, video_features.extract, video_path)

        def _transcription_progress(done_chunks: int, total_chunks: int, _chunk_path) -> None:
            # Reserve 30-55% for ASR progress.
            pct = 30 + int((done_chunks / max(total_chunks, 1)) * 25)
            update_job(job_id, status="transcribing", progress_pct=min(max(pct, 30), 55))

        transcript = await loop.run_in_executor(
            None,
            transcriber.transcribe_with_progress,
            audio_path,
            _transcription_progress,
        )

        _raise_if_cancelled(job_id)
        update_job(job_id, status="extracting_features", progress_pct=56)
        audio_frames, visual_frames = await asyncio.gather(
            audio_features_task,
            video_features_task,
        )

        # Merge audio features, then visual features into transcript segments
        transcript = audio_features.enrich_segments(transcript, audio_frames)
        transcript = video_features.enrich_segments(transcript, visual_frames)

        # Speaker diarization — MFCC clustering (runs when pyannote is unavailable)
        # pyannote already ran inside transcriber.transcribe() when hf_token is set
        if not settings.hf_token:
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
        fallback_candidates = []
        if not peak_candidates:
            fallback_candidates = await loop.run_in_executor(
                None,
                candidate_generators.generate_fallback_candidates,
                transcript,
                chapters,
                normalized_strategy,
            )
        candidates = await loop.run_in_executor(
            None,
            lambda: candidate_generators.merge_prioritized_candidates(
                peak_candidates or fallback_candidates,
                semantic_candidates,
                strategy=normalized_strategy,
            ),
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
            feature_builder.build_features,
            normalized_candidates,
            transcript,
            effective_profile,
            False,
            None,
            None,
            chapters,
            normalized_strategy,
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
                feature_builder.build_features,
                normalized_candidates,
                transcript,
                effective_profile,
                True,
                llm_candidate_ids,
                None,
                chapters,
                normalized_strategy,
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
        judge_top_k = min(max(settings.shortlist_judge_top_k, 0), len(ranked_candidates))
        job_dir = pathlib.Path(settings.storage_dir) / job_id
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        sampled_keyframes: dict[str, list[keyframe_sampler.SampledKeyframe]] = {}
        if (
            settings.enable_clip_judge
            and settings.enable_multimodal_clip_judge
            and settings.gemini_api_key
            and judge_top_k > 0
        ):
            sampled_keyframes = await loop.run_in_executor(
                None,
                keyframe_sampler.sample_candidate_keyframes,
                video_path,
                ranked_candidates[:judge_top_k],
                debug_dir,
                settings.judge_keyframes_per_candidate,
            )
        if settings.enable_clip_judge and judge_top_k > 0:
            judge_future = loop.run_in_executor(
                None,
                clip_judge.rerank_candidates,
                ranked_candidates,
                transcript,
                effective_profile,
                judge_top_k,
                None,
                sampled_keyframes,
            )
            try:
                judged_ranked_candidates = await asyncio.wait_for(
                    asyncio.shield(judge_future),
                    timeout=settings.clip_judge_timeout_sec,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "clip_judge.rerank_candidates timed out after %.1fs; skipping judge stage",
                    settings.clip_judge_timeout_sec,
                )
                judged_ranked_candidates = list(ranked_candidates)
        ranked_candidates = await loop.run_in_executor(
            None,
            lambda: selector.select_diverse_ranked_candidates(
                judged_ranked_candidates,
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
        feedback_events = feedback_store.build_pipeline_feedback(job_id, ranked_candidates, moments)
        feedback_store.write_job_feedback(debug_dir, feedback_events)
        feedback_store.append_feedback_events(pathlib.Path(settings.storage_dir), feedback_events)
        (debug_dir / "final_moments.json").write_text(
            json.dumps(moments, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        update_job(job_id, status="clipping", progress_pct=88)
        _raise_if_cancelled(job_id)
        clips = await loop.run_in_executor(None, clipper.cut_clips, video_path, moments, job_id)

        # Clean up intermediate files to save disk space
        audio_path.unlink(missing_ok=True)
        chunks_dir = audio_path.parent / "chunks"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)

        update_job(job_id, status="done", progress_pct=100, clips=clips)

        # Schedule cleanup of entire job directory after 1 hour
        asyncio.create_task(_cleanup_later(job_id))

    except (PipelineCancelledError, downloader.DownloadCancelledError):
        logger.info("Pipeline cancelled for job %s", job_id)
    except Exception as e:
        logger.exception("Pipeline failed for job %s", job_id)
        update_job(job_id, status="failed", error=str(e))


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
