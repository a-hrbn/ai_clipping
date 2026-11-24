import asyncio
import concurrent.futures
import json
import logging
import pathlib
import re
import subprocess
import threading
from config import settings
from job_manager import is_job_cancelled, is_self_managed_job, update_job
from producers.adapters import FsAudioCache
from producers.audio import AudioResource
from producers.chat import ChatEnrichment
from producers.clipping import Clipping
from producers.energy import VoiceEnergy
from producers.moments import MomentFinding, MomentParams
from producers.transcript import TranscriptResource
from services.job_artifact_store import RealFsJobArtifactStore
from services import (
    downloader,
    audio_extractor,
    transcriber,
    clipper,
    clip_strategy,
    moment_finder,
    chat_signal,
    audio_energy,
)

logger = logging.getLogger(__name__)

_TWITCH_VOD_RE = re.compile(r"twitch\.tv/videos/(\d+)")


class PipelineCancelledError(RuntimeError):
    pass


def _raise_if_cancelled(job_id: str) -> None:
    if is_job_cancelled(job_id):
        raise PipelineCancelledError(f"Job {job_id} cancelled by client")


def _transcript_cache_enabled_for_job(job_id: str, url: str | None) -> bool:
    """Whether this job's transcript goes into the shared transcript cache.

    Always for self-managed workflows and when the frontend job cache is on;
    otherwise only for Twitch-VOD serving jobs, unless the transcript cache is
    switched off. Independent of whether the source *media* is kept.
    """
    if is_self_managed_job(job_id) or bool(getattr(settings, "frontend_job_cache_enabled", False)):
        return True
    if not bool(getattr(settings, "frontend_transcript_cache_enabled", True)):
        return False
    return bool(_source_twitch_vod_id(url))



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



def _source_twitch_vod_id(url: str | None) -> str:
    if not url:
        return ""
    match = _TWITCH_VOD_RE.search(str(url).strip().lower())
    return match.group(1) if match else ""



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


def _resolve_archetype_judge(strategy: dict | None) -> bool:
    """The effective Archetype-Judge flag for one job: the global default unless the
    job's normalized ``strategy`` opts out (or in). ``moment_finder_archetype_judge_enabled``
    is the master switch (default ON since Go-Live, #105); a per-job ``archetype_judge``
    of ``True``/``False`` overrides it, ``None`` (absent) inherits — so an operator can
    turn the judge off for a single job's cost/latency without a code change."""
    global_default = bool(getattr(settings, "moment_finder_archetype_judge_enabled", False))
    override = (strategy or {}).get("archetype_judge")
    return global_default if override is None else bool(override)


def _moment_params(strategy: dict | None = None) -> MomentParams:
    """Snapshot the ~17 ``moment_finder_*`` settings once, so the MomentFinding
    Producer is handed plain values and never reads ``config.settings``. The
    per-job normalized ``strategy`` resolves the Archetype-Judge opt-out (#105)."""
    return MomentParams(
        provider=str(getattr(settings, "moment_finder_provider", "gemini") or "gemini"),
        model=str(getattr(settings, "moment_finder_model", "") or ""),
        max_moments=int(getattr(settings, "moment_finder_max_moments", 180) or 180),
        min_duration_sec=float(getattr(settings, "moment_finder_min_duration_sec", 8.0) or 8.0),
        max_duration_sec=float(getattr(settings, "moment_finder_max_duration_sec", 90.0) or 90.0),
        max_tokens=int(getattr(settings, "moment_finder_max_tokens", 64000) or 64000),
        chunk_duration_sec=float(getattr(settings, "moment_finder_chunk_duration_sec", 2700.0) or 0.0),
        chunk_overlap_sec=float(getattr(settings, "moment_finder_chunk_overlap_sec", 120.0) or 0.0),
        timeout_sec=float(getattr(settings, "moment_finder_timeout_sec", 900.0) or 900.0),
        api_retries=int(getattr(settings, "moment_finder_api_retries", 2) or 0),
        retry_backoff_sec=float(getattr(settings, "moment_finder_retry_backoff_sec", 20.0) or 20.0),
        filter_transcript_noise=bool(getattr(settings, "moment_finder_filter_transcript_noise", True)),
        temperature=0.0,
        eval_recall=bool(getattr(settings, "moment_finder_eval_recall", False)),
        archetype_judge=_resolve_archetype_judge(strategy),
        judge_model=str(getattr(settings, "moment_finder_archetype_judge_model", "") or ""),
        judge_chunk_size=int(getattr(settings, "moment_finder_archetype_judge_chunk_size", 0) or 0),
        judge_max_retries=int(getattr(settings, "moment_finder_archetype_judge_max_retries", 3) or 0),
        judge_pad_sec=float(getattr(settings, "moment_finder_archetype_judge_pad_sec", 12.0) or 12.0),
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
    audio: AudioResource | None = None
    try:
        _raise_if_cancelled(job_id)

        async def run(call, *, status, band):
            # Map a Producer's 0..1 progress onto an absolute % band and check
            # cancellation BETWEEN Producers (cancel → no `failed` status). The
            # Producer runs on the executor; status/progress stay a Conductor job.
            _raise_if_cancelled(job_id)
            lo, hi = band

            def on_progress(fraction: float) -> None:
                pct = lo + int(min(max(fraction, 0.0), 1.0) * (hi - lo))
                update_job(job_id, status=status, progress_pct=pct)

            on_progress(0.0)
            return await loop.run_in_executor(None, lambda: call(progress=on_progress))

        transcript_cache_source = (
            _transcript_cache_source(url, video_path)
            if _transcript_cache_enabled_for_job(job_id, url)
            else None
        )

        # Prefetch Twitch chat in parallel with download/transcription — it only
        # needs the URL and otherwise costs 5-15 serial minutes on cold VODs.
        # Runs on a daemon thread (not the asyncio executor): if the pipeline
        # fails mid-run, loop close would otherwise block worker exit for up to
        # 300s joining the still-downloading prefetch thread.
        chat_prefetch: concurrent.futures.Future | None = None
        if settings.enable_chat_analysis and url and not transcriber.has_chat_enrichment(transcript_cache_source):
            prefetch_debug_dir = pathlib.Path(settings.storage_dir) / job_id / "debug"
            prefetch_debug_dir.mkdir(parents=True, exist_ok=True)
            chat_prefetch = concurrent.futures.Future()

            def _prefetch_chat(future: concurrent.futures.Future = chat_prefetch) -> None:
                try:
                    future.set_result(chat_signal.download_chat(
                        url, job_id, prefetch_debug_dir,
                        timeout_sec=settings.chat_download_timeout_sec,
                    ))
                except BaseException as exc:  # surfaces at the await site
                    future.set_exception(exc)

            threading.Thread(target=_prefetch_chat, name=f"chat-prefetch-{job_id}", daemon=True).start()

        if video_path is None:
            update_job(job_id, status="downloading", progress_pct=5)
            video_path = await loop.run_in_executor(None, downloader.download, url, job_id)
        else:
            update_job(job_id, status="downloading", progress_pct=19)

        # The keystone Resource: transcription and voice-energy both pull it, but
        # audio is extracted at most once per job. The Conductor tears it down.
        audio = AudioResource(
            video_path=video_path,
            job_id=job_id,
            cache_source=transcript_cache_source,
            extract=audio_extractor.extract,
            cache=FsAudioCache(
                settings.storage_dir,
                cache_key=transcriber.stable_cache_key_for_source,
                probe=_probe_media_duration_sec,
            ),
        )

        # TranscriptResource folds the duration probe + cache hit/miss decision in:
        # a hit returns without pulling audio (zero extraction); a miss pulls the
        # shared AudioResource then transcribes. Audio extraction on a miss now
        # shows under the `transcribing` band (the lazy pull lives in the Producer).
        transcript_resource = TranscriptResource(
            audio=audio,
            cache_source=transcript_cache_source,
            video_path=video_path,
            isolate_chunks=bool(settings.pipeline_isolate_transcription_chunks),
            enable_diarization=bool(settings.pipeline_enable_diarization),
            load_cached=transcriber.load_cached_transcript,
            transcribe=transcriber.transcribe_with_progress,
            probe=_probe_media_duration_sec,
        )
        transcript = await run(
            lambda *, progress: transcript_resource.segments(progress=progress),
            status="transcribing",
            band=(30, 55),
        )
        if transcript_resource.from_cache():
            logger.info(
                "pipeline_transcript_cache_hit job_id=%s source=%s segments=%s",
                job_id,
                transcript_cache_source,
                len(transcript),
            )

        job_dir = pathlib.Path(settings.storage_dir) / job_id
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # The single owner of the job-artifact names + I/O. Injected into the
        # writer Producers (same slot debug_dir used to fill) and used for the
        # Conductor's own segments artifact, so a rename lives in one place and the
        # router readers (which already go through the store) cannot drift. The raw
        # chat download below still takes debug_dir — its files are not store roles.
        store = RealFsJobArtifactStore(settings.storage_dir)

        # Voice energy burst: load from the transcript cache when possible; else
        # compute it — pulling the SAME AudioResource transcription used, so audio
        # is extracted at most once — and persist it so cache-hit runs keep the
        # signal. The Producer is pure; load/save/compute are injected, and the
        # run() helper maps its work onto the `extracting_audio` band.
        voice_energy = VoiceEnergy(
            audio=audio,
            cache_source=transcript_cache_source,
            load=transcriber.load_voice_energy,
            save=transcriber.save_voice_energy,
            compute=audio_energy.compute_segment_energy,
        )
        transcript = await run(
            lambda *, progress: voice_energy.apply(transcript),
            status="extracting_audio",
            band=(56, 60),
        )

        if settings.enable_chat_analysis and url:
            # The prefetch (daemon thread) stays a Conductor concern; the Producer
            # only sees a chat_source callable that blocks on the prefetched result
            # OR downloads — no thread/future knowledge. apply_cached + enrich +
            # save are injected; the run() helper maps its work onto the
            # `downloading_chat` band.
            def chat_source() -> pathlib.Path | None:
                if chat_prefetch is not None:
                    return chat_prefetch.result()
                return chat_signal.download_chat(
                    url, job_id, debug_dir,
                    timeout_sec=settings.chat_download_timeout_sec,
                )

            chat_enrichment = ChatEnrichment(
                cache_source=transcript_cache_source,
                chat_source=chat_source,
                apply_cached=transcriber.apply_chat_enrichment,
                enrich=chat_signal.enrich_segments,
                save=transcriber.save_chat_enrichment,
            )
            transcript = await run(
                lambda *, progress: chat_enrichment.apply(transcript),
                status="downloading_chat",
                band=(60, 72),
            )
            if chat_prefetch is not None:
                # Best-effort: a no-op once the future was consumed by chat_source;
                # on a cache hit (chat_source untouched) it frees the prefetch.
                chat_prefetch.cancel()

        # MomentFinding owns the llm_moment_finder_stage.json running/ok/failed
        # contract and the prompt/response/usage/moments(.raw).jsonl debug dumps.
        # It is pure: the ~17 moment_finder_* settings are snapshotted into
        # MomentParams and find/error are injected (the real service pulls torch,
        # so the error class is injected rather than imported). The run() helper
        # maps its work onto the `analyzing` 73–87 band.
        moment_finding = MomentFinding(
            profile=effective_profile,
            vod_id=_source_twitch_vod_id(url) or job_id,
            job_id=job_id,
            store=store,
            params=_moment_params(normalized_strategy),
            find=moment_finder.find_moments,
            error=moment_finder.MomentFinderError,
        )
        llm_result = await run(
            lambda *, progress: moment_finding.find(transcript, progress=progress),
            status="analyzing",
            band=(73, 87),
        )

        store.write_segments(job_id, _segment_records(transcript))

        # Clipping post-processes the moment rows, selects top-K, writes
        # final_moments.json, then cuts the clips. Pure: top_k/platform/debug_dir/
        # job_id + cut are injected; the run() helper maps cut progress onto the
        # `clipping` 88–99 band. Partial-clips streaming stays a Conductor concern
        # via a cut wrapper that update_job(clips=so_far) before delegating to the
        # producer's band-progress callback.
        def cut_with_clip_streaming(video_path_, moments, job_id_, *, progress_cb):
            def on_cut(so_far: list, done: int, total: int) -> None:
                try:
                    update_job(job_id_, status="clipping", clips=so_far)
                except Exception:
                    pass
                progress_cb(so_far, done, total)

            return clipper.cut_clips(video_path_, moments, job_id_, progress_cb=on_cut)

        clipping = Clipping(
            job_id=job_id,
            store=store,
            top_k=int(getattr(settings, "clip_top_k", 0) or 0),
            platform=str(normalized_strategy.get("platform") or "general"),
            cut=cut_with_clip_streaming,
        )
        clips = await run(
            lambda *, progress: clipping.run(video_path, llm_result, progress=progress),
            status="clipping",
            band=(88, 99),
        )

        audio.teardown()
        audio = None

        update_job(job_id, status="done", progress_pct=100, clips=clips)
        # The job folder is deleted when the registry evicts the job (#117).

    except (PipelineCancelledError, downloader.DownloadCancelledError):
        logger.info("Pipeline cancelled for job %s", job_id)
    except Exception as e:
        logger.exception("Pipeline failed for job %s", job_id)
        update_job(job_id, status="failed", error=str(e))
    finally:
        # Source media (video/audio/chunks) is cleaned up by the jobs router
        # after the worker subprocess exits, so it also runs after an OOM kill.
        if audio is not None:
            audio.teardown()


def _segment_records(transcript) -> list[dict]:
    lines = transcriber.format_jsonl(transcript).splitlines()
    return [json.loads(line) for line in lines if line.strip()]
