from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import math
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterator

from config import settings

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:  # pragma: no cover - optional runtime acceleration
    FasterWhisperModel = None


def __getattr__(name: str):
    # Lazily expose the heavy ML deps as module attributes. Importing this
    # module (e.g. for RichSegment) must not pull in torch/whisper, but callers
    # and tests still reference `services.transcriber.torch` / `.whisper`.
    if name in ("torch", "whisper"):
        import importlib

        return importlib.import_module(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = logging.getLogger(__name__)

CHUNK_SECONDS = 180  # legacy export; runtime uses _chunk_seconds()
_TRANSCRIPT_CHUNK_CACHE_VERSION = 1

# Module-level Whisper singleton — loaded once on first use, reused across jobs.
_whisper_model: object | None = None
_whisper_model_backend: str | None = None
_whisper_runtime_logged = False
_whisper_lock = threading.Lock()
_whisper_inference_lock = threading.Lock()


def _chunk_seconds() -> int:
    value = int(getattr(settings, "transcription_chunk_seconds", CHUNK_SECONDS) or CHUNK_SECONDS)
    return max(value, 30)


def _worker_batch_size() -> int:
    value = int(getattr(settings, "transcription_worker_batch_size", 12) or 12)
    return max(value, 1)


def _word_timestamps_enabled() -> bool:
    return bool(getattr(settings, "whisper_word_timestamps", False))


def _vad_filter_enabled() -> bool:
    return bool(getattr(settings, "whisper_vad_filter", True))


def _resolved_whisper_backend() -> str:
    requested = str(getattr(settings, "whisper_backend", "auto") or "auto").strip().lower()
    requested = requested.replace("_", "-")
    if requested in {"faster", "faster-whisper", "ctranslate2"}:
        if FasterWhisperModel is None:
            logger.warning("transcriber_faster_whisper_unavailable_fallback resolved_backend=openai")
            return "openai"
        return "faster-whisper"
    if requested == "openai":
        return "openai"
    if requested == "auto" and FasterWhisperModel is not None:
        return "faster-whisper"
    return "openai"


def _resolved_whisper_device() -> str:
    import torch  # deferred: heavy import, only needed when a model is actually used

    requested = str(settings.whisper_device or "auto").strip().lower() or "auto"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("transcriber_cuda_unavailable_fallback requested=cuda resolved=cpu")
        return "cpu"
    return requested


def _resolved_precision(device: str) -> str:
    return "fp16" if device == "cuda" else "fp32"


def _resolved_compute_type(device: str) -> str:
    configured = str(getattr(settings, "whisper_compute_type", "") or "").strip().lower()
    if configured:
        return configured
    return "float16" if device == "cuda" else "float32"


def _model_device_name(model: whisper.Whisper) -> str:
    try:
        parameter = next(model.parameters())
        return str(parameter.device)
    except Exception:
        device = getattr(model, "device", None)
        return str(device or "unknown")


def runtime_summary() -> dict[str, object]:
    configured_language = str(settings.whisper_language or "").strip().lower()
    backend = _resolved_whisper_backend() if _whisper_model_backend is None else _whisper_model_backend
    if _whisper_model is None:
        device = _resolved_whisper_device()
        return {
            "backend": backend,
            "whisper_model": settings.whisper_model,
            "device": device,
            "precision": _resolved_compute_type(device) if backend == "faster-whisper" else _resolved_precision(device),
            "language": configured_language,
            "chunk_seconds": _chunk_seconds(),
            "worker_batch_size": _worker_batch_size(),
            "word_timestamps": _word_timestamps_enabled(),
        }
    device = _resolved_whisper_device() if backend == "faster-whisper" else _model_device_name(_whisper_model)
    return {
        "backend": backend,
        "whisper_model": settings.whisper_model,
        "device": device,
        "precision": _resolved_compute_type(device) if backend == "faster-whisper" else _resolved_precision(device),
        "language": configured_language,
        "chunk_seconds": _chunk_seconds(),
        "worker_batch_size": _worker_batch_size(),
        "word_timestamps": _word_timestamps_enabled(),
    }


def _get_model() -> object:
    global _whisper_model, _whisper_model_backend, _whisper_runtime_logged
    # Double-checked locking: the load is a multi-second disk op; without
    # the lock two concurrent first-transcriptions would each download and load
    # the model into memory, then silently race to bind _whisper_model.
    resolved_backend = _resolved_whisper_backend()
    if _whisper_model is None or _whisper_model_backend != resolved_backend:
        with _whisper_lock:
            if _whisper_model is None or _whisper_model_backend != resolved_backend:
                logger.info(
                    "transcriber_init backend=%s whisper_model=%s",
                    resolved_backend,
                    settings.whisper_model,
                )
                whisper_cache_dir = pathlib.Path(settings.storage_dir).parent / "cache" / "whisper"
                whisper_cache_dir.mkdir(parents=True, exist_ok=True)
                resolved_device = _resolved_whisper_device()
                if resolved_backend == "faster-whisper":
                    if FasterWhisperModel is None:
                        raise RuntimeError("faster-whisper backend requested but package is not installed")
                    _whisper_model = FasterWhisperModel(
                        settings.whisper_model,
                        device=resolved_device,
                        compute_type=_resolved_compute_type(resolved_device),
                        download_root=str(whisper_cache_dir),
                    )
                else:
                    import whisper  # deferred: heavy import, only the openai backend needs it

                    _whisper_model = whisper.load_model(
                        settings.whisper_model,
                        download_root=str(whisper_cache_dir),
                        device=resolved_device,
                    )
                _whisper_model_backend = resolved_backend
                _whisper_runtime_logged = False
    if not _whisper_runtime_logged:
        summary = runtime_summary()
        logger.info(
            "transcriber_runtime backend=%s whisper_model=%s device=%s precision=%s language=%s chunk_seconds=%s worker_batch_size=%s word_timestamps=%s",
            summary["backend"],
            summary["whisper_model"],
            summary["device"],
            summary["precision"],
            summary["language"] or "auto",
            summary["chunk_seconds"],
            summary["worker_batch_size"],
            summary["word_timestamps"],
        )
        _whisper_runtime_logged = True
    return _whisper_model

# Legacy type alias for backward compatibility
Segment = tuple[float, float, str]  # (start, end, text)

# Emotion detection patterns
_LAUGHTER_RE = re.compile(r"\b(ha(ha)+|hehe|lol|lmao|xd|kekw|lulw)\b", re.IGNORECASE)
_EXCLAMATION_RE = re.compile(r"[!]{1,}$|(!)\s")
_QUESTION_RE = re.compile(r"\?")

_EMPTY_TRANSCRIBE_ERROR_SNIPPET = "cannot reshape tensor of 0 elements"
_MIN_SEGMENT_DURATION_SECONDS = 0.01
_WORKER_RESULT_PREFIX = "TRANSCRIBER_CHUNK_RESULT "

# Container-local path (NOT the bind mount — flock on drvfs/9p is unreliable).
# Slot 0 keeps the historical name so old and new worker processes contend on
# the same file across a mid-run code rollout.
_TRANSCRIPTION_LOCK_PATH = pathlib.Path(tempfile.gettempdir()) / "ai_clipping_transcription.lock"


def transcription_slots() -> int:
    """How many Whisper instances the GPU can hold concurrently.

    Auto-derived from VRAM (~2.5GB per large-v3-turbo int8 instance plus
    headroom => one slot per ~5GB). Override via settings.transcription_gpu_slots.
    CPU-only setups get 1 (serialization avoids RAM thrash there too).
    """
    override = int(getattr(settings, "transcription_gpu_slots", 0) or 0)
    if override > 0:
        return override
    try:
        import torch  # deferred: heavy import, optional at runtime

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return max(1, int(vram_gb // 5))
    except Exception:
        pass
    return 1


def _slot_lock_path(slot: int) -> pathlib.Path:
    if slot == 0:
        return _TRANSCRIPTION_LOCK_PATH
    return _TRANSCRIPTION_LOCK_PATH.with_name(f"ai_clipping_transcription.slot{slot}.lock")


@contextlib.contextmanager
def _transcription_lock():
    """Bound concurrent Whisper inference across worker processes.

    Acquires one of N slot locks (N = transcription_slots()): on a 6GB consumer
    GPU exactly one model lives in VRAM while other workers overlap their
    download/ffmpeg/chat stages; bigger GPUs transcribe N VODs concurrently.
    """
    slots = transcription_slots()
    handles = [(_slot_lock_path(i)).open("a") for i in range(slots)]
    acquired: int | None = None
    try:
        if slots == 1:
            fcntl.flock(handles[0], fcntl.LOCK_EX)  # blocking: instant handover
            acquired = 0
        else:
            while acquired is None:
                for i, handle in enumerate(handles):
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = i
                        break
                    except OSError:
                        continue
                if acquired is None:
                    time.sleep(5.0)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(handles[acquired], fcntl.LOCK_UN)
        for handle in handles:
            handle.close()


@dataclass
class RichSegment:
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"
    emotions: list[str] = field(default_factory=list)  # "exclamation", "question", "laughter"
    words_per_second: float = 0.0
    # Chat-replay signal (populated by chat_signal.enrich_segments)
    chat_spike_score: float = 0.0   # normalised message-rate spike vs. local baseline (0–1)
    chat_message_count: int = 0
    # Emote-class scores: fraction of classified emotes in window that are hype/funny/shock
    chat_hype_score: float = 0.0
    chat_funny_score: float = 0.0
    chat_shock_score: float = 0.0
    # Audio signal (populated by audio_energy.compute_segment_energy before WAV cleanup)
    voice_energy_burst_score: float = 0.0  # relative RMS burst vs. local baseline (0–1)

    def to_tuple(self) -> Segment:
        return (self.start, self.end, self.text)


def transcribe(audio_path: pathlib.Path, *, enable_diarization: bool = True) -> list[RichSegment]:
    return transcribe_with_progress(audio_path, enable_diarization=enable_diarization)


def transcribe_with_progress(
    audio_path: pathlib.Path,
    progress_cb=None,
    *,
    enable_diarization: bool = True,
    cache_key: str | None = None,
    isolate_chunks: bool = False,
) -> list[RichSegment]:
    chunks_dir = audio_path.parent / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    chunk_cache_dir = _chunk_cache_dir(cache_key)

    # Split audio into fixed, non-overlapping chunks via ffmpeg CLI.
    # Use explicit PCM re-encoding (not -c copy): the segment muxer does not
    # write correct WAV headers when stream-copying, which produces truncated
    # or malformed chunks that cause "Broken pipe" when Whisper reads them.
    chunk_pattern = str(chunks_dir / "chunk_%03d.wav")
    chunk_seconds = _chunk_seconds()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(chunk_seconds),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_pattern,
        ],
        check=True,
        capture_output=True,
    )

    chunk_files = _filter_transcribable_chunks(sorted(chunks_dir.glob("chunk_*.wav")))
    model: whisper.Whisper | None = None

    total_chunks = len(chunk_files)
    segments_by_chunk: dict[int, list[RichSegment]] = {}
    pending_chunks: list[tuple[int, pathlib.Path]] = []
    progress_index_by_chunk: dict[int, int] = {}

    for progress_idx, (chunk_idx, chunk_file) in enumerate(chunk_files, start=1):
        progress_index_by_chunk[chunk_idx] = progress_idx
        cached_segments = _load_chunk_cache(chunk_cache_dir, chunk_idx)
        if cached_segments is not None:
            segments_by_chunk[chunk_idx] = cached_segments
            if progress_cb is not None and total_chunks > 0:
                progress_cb(progress_idx, total_chunks, chunk_file)
            continue
        pending_chunks.append((chunk_idx, chunk_file))

    if isolate_chunks:
        for chunk_batch in _chunk_batches(pending_chunks, _worker_batch_size()):
            chunk_by_idx = {int(chunk_idx): chunk_file for chunk_idx, chunk_file in chunk_batch}
            with _transcription_lock():
                for chunk_idx, result in _transcribe_chunks_isolated(chunk_batch):
                    chunk_file = chunk_by_idx.get(int(chunk_idx))
                    if chunk_file is None:
                        raise RuntimeError(f"isolated Whisper worker returned unknown chunk index {chunk_idx}")
                    chunk_segments = _rich_segments_from_result(result, chunk_idx=chunk_idx, chunk_file=chunk_file)
                    segments_by_chunk[chunk_idx] = chunk_segments
                    _write_chunk_cache(chunk_cache_dir, chunk_idx, chunk_segments)

                    if progress_cb is not None and total_chunks > 0:
                        progress_cb(progress_index_by_chunk[chunk_idx], total_chunks, chunk_file)
    elif pending_chunks:
        # Non-isolated path keeps the model in this process; the lock spans the
        # whole loop so a second worker never loads a model alongside it.
        with _transcription_lock():
            for chunk_idx, chunk_file in pending_chunks:
                try:
                    if model is None:
                        model = _get_model()
                    result = _transcribe_chunk(model, chunk_file)
                except RuntimeError as exc:
                    if _EMPTY_TRANSCRIBE_ERROR_SNIPPET not in str(exc):
                        raise
                    logger.warning(
                        "transcriber_skip_chunk chunk=%s reason=empty_tensor_runtime_error",
                        chunk_file,
                    )
                    segments_by_chunk[chunk_idx] = []
                    if progress_cb is not None and total_chunks > 0:
                        progress_cb(progress_index_by_chunk[chunk_idx], total_chunks, chunk_file)
                    _write_chunk_cache(chunk_cache_dir, chunk_idx, [])
                    continue

                chunk_segments = _rich_segments_from_result(result, chunk_idx=chunk_idx, chunk_file=chunk_file)
                segments_by_chunk[chunk_idx] = chunk_segments
                _write_chunk_cache(chunk_cache_dir, chunk_idx, chunk_segments)

                if progress_cb is not None and total_chunks > 0:
                    progress_cb(progress_index_by_chunk[chunk_idx], total_chunks, chunk_file)

    raw_segments: list[RichSegment] = []
    for chunk_idx, _chunk_file in chunk_files:
        raw_segments.extend(segments_by_chunk.get(chunk_idx, []))

    deduped = _deduplicate(raw_segments)

    # Speaker diarization (optional — requires HF token)
    if enable_diarization and settings.hf_token:
        deduped = _diarize(audio_path, deduped)

    return deduped


def load_cached_transcript(
    cache_key: str | None,
    *,
    expected_duration_sec: float | None,
    progress_cb=None,
) -> list[RichSegment] | None:
    if expected_duration_sec is None or expected_duration_sec <= 0:
        return None
    expected_chunks = max(int(math.ceil(float(expected_duration_sec) / _chunk_seconds())), 1)
    cache_dir = _chunk_cache_dir(cache_key)
    if cache_dir is None:
        return None

    raw_segments: list[RichSegment] = []
    for chunk_idx in range(expected_chunks):
        cached_segments = _load_chunk_cache(cache_dir, chunk_idx)
        if cached_segments is None:
            return None
        raw_segments.extend(cached_segments)
        if progress_cb is not None:
            progress_cb(chunk_idx + 1, expected_chunks, _chunk_cache_path(cache_dir, chunk_idx))

    return _deduplicate(raw_segments)


def _chunk_file_for_index(
    chunk_files: list[tuple[int, pathlib.Path]],
    chunk_idx: int,
) -> pathlib.Path:
    for candidate_idx, chunk_file in chunk_files:
        if int(candidate_idx) == int(chunk_idx):
            return chunk_file
    raise RuntimeError(f"isolated Whisper worker returned unknown chunk index {chunk_idx}")


def _chunk_batches(
    chunk_files: list[tuple[int, pathlib.Path]],
    batch_size: int,
) -> Iterator[list[tuple[int, pathlib.Path]]]:
    size = max(int(batch_size), 1)
    for start in range(0, len(chunk_files), size):
        yield chunk_files[start : start + size]


def _rich_segments_from_result(
    result: dict,
    *,
    chunk_idx: int,
    chunk_file: pathlib.Path,
) -> list[RichSegment]:
    time_offset = int(chunk_idx) * _chunk_seconds()
    chunk_segments: list[RichSegment] = []
    raw_segments_payload = _validated_segments_payload(result, chunk_file=chunk_file)
    for seg_idx, raw_seg in enumerate(raw_segments_payload):
        seg = _normalize_segment(raw_seg, chunk_file=chunk_file, segment_index=seg_idx)

        start = seg["start"] + time_offset
        end = seg["end"] + time_offset
        text = seg["text"].strip()
        if not text:
            continue

        wps = _calc_tempo(seg, time_offset)
        emotions = _tag_emotions(text)

        chunk_segments.append(RichSegment(
            start=start,
            end=end,
            text=text,
            emotions=emotions,
            words_per_second=wps,
        ))
    return chunk_segments


def _transcribe_chunk_isolated(chunk_file: pathlib.Path) -> dict:
    output_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{chunk_file.stem}.",
            suffix=".transcript.json",
            dir=str(chunk_file.parent),
            delete=False,
        ) as handle:
            output_path = pathlib.Path(handle.name)
        command = [
            sys.executable,
            "-m",
            "services.transcriber",
            "--transcribe-chunk",
            str(chunk_file),
            str(output_path),
        ]
        logger.info("transcriber_chunk_subprocess_start chunk=%s", chunk_file)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = _compact_subprocess_output(completed.stderr or completed.stdout)
            raise RuntimeError(
                f"isolated Whisper chunk failed for {chunk_file} "
                f"with exit code {completed.returncode}: {message}"
            )
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"isolated Whisper chunk wrote invalid output for {chunk_file}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"isolated Whisper chunk output for {chunk_file} was not a mapping")
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise RuntimeError(f"isolated Whisper chunk output for {chunk_file} did not contain segments")
        logger.info(
            "transcriber_chunk_subprocess_done chunk=%s segments=%s",
            chunk_file,
            len(segments),
        )
        return {"segments": segments}
    finally:
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass


def _transcribe_chunks_isolated(chunk_files: list[tuple[int, pathlib.Path]]) -> Iterator[tuple[int, dict]]:
    if not chunk_files:
        return

    manifest_path: pathlib.Path | None = None
    common_dir = pathlib.Path(chunk_files[0][1]).parent
    try:
        with tempfile.NamedTemporaryFile(
            prefix="transcriber_chunks.",
            suffix=".manifest.json",
            dir=str(common_dir),
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as handle:
            manifest_path = pathlib.Path(handle.name)
            json.dump(
                [
                    {
                        "chunk_index": int(chunk_idx),
                        "chunk_path": str(chunk_file),
                    }
                    for chunk_idx, chunk_file in chunk_files
                ],
                handle,
                ensure_ascii=True,
            )

        command = [
            sys.executable,
            "-m",
            "services.transcriber",
            "--transcribe-chunks",
            str(manifest_path),
        ]
        logger.info(
            "transcriber_chunk_worker_start chunks=%s first_chunk=%s",
            len(chunk_files),
            chunk_files[0][1],
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stderr_lines: list[str] = []

        def _read_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                line = line.strip()
                if line:
                    stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        seen_chunks: set[int] = set()
        result_count = 0
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if not line.startswith(_WORKER_RESULT_PREFIX):
                stderr_lines.append(line)
                continue
            payload = json.loads(line[len(_WORKER_RESULT_PREFIX):])
            if not isinstance(payload, dict):
                raise RuntimeError("isolated Whisper worker emitted a non-object payload")
            chunk_idx = int(payload["chunk_index"])
            segments = payload.get("segments")
            if not isinstance(segments, list):
                raise RuntimeError(f"isolated Whisper worker omitted segments for chunk {chunk_idx}")
            seen_chunks.add(chunk_idx)
            result_count += 1
            logger.info(
                "transcriber_chunk_worker_result chunk_index=%s segments=%s",
                chunk_idx,
                len(segments),
            )
            yield chunk_idx, {"segments": segments}

        returncode = process.wait()
        stderr_thread.join(timeout=1.0)
        if returncode != 0:
            message = _compact_subprocess_output("\n".join(stderr_lines))
            raise RuntimeError(
                f"isolated Whisper chunk worker failed with exit code {returncode}: {message}"
            )

        expected_chunks = {int(chunk_idx) for chunk_idx, _chunk_file in chunk_files}
        missing_chunks = sorted(expected_chunks - seen_chunks)
        if missing_chunks:
            raise RuntimeError(
                f"isolated Whisper chunk worker completed without results for chunks {missing_chunks}"
            )

        logger.info("transcriber_chunk_worker_done chunks=%s", result_count)
    finally:
        if manifest_path is not None:
            try:
                manifest_path.unlink(missing_ok=True)
            except OSError:
                pass


def _transcribe_chunk_worker_main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "--transcribe-chunks":
        return _transcribe_chunks_worker_main(pathlib.Path(args[1]))
    if len(args) != 3 or args[0] != "--transcribe-chunk":
        print(
            "usage: python -m services.transcriber "
            "--transcribe-chunk <chunk.wav> <output.json> | --transcribe-chunks <manifest.json>",
            file=sys.stderr,
        )
        return 2
    chunk_file = pathlib.Path(args[1])
    output_path = pathlib.Path(args[2])
    try:
        model = _get_model()
        result = _transcribe_chunk(model, chunk_file)
        output_path.write_text(
            json.dumps(
                {
                    "segments": _validated_segments_payload(result, chunk_file=chunk_file),
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        return 0
    except RuntimeError as exc:
        if _EMPTY_TRANSCRIBE_ERROR_SNIPPET in str(exc):
            output_path.write_text(
                json.dumps({"segments": []}, ensure_ascii=True),
                encoding="utf-8",
            )
            return 0
        raise


def _transcribe_chunks_worker_main(manifest_path: pathlib.Path) -> int:
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid chunk manifest {manifest_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(raw_manifest, list):
        print(f"invalid chunk manifest {manifest_path}: expected list", file=sys.stderr)
        return 2

    try:
        model = _get_model()
        for raw_item in raw_manifest:
            if not isinstance(raw_item, dict):
                raise RuntimeError("chunk manifest item must be an object")
            chunk_idx = int(raw_item["chunk_index"])
            chunk_file = pathlib.Path(str(raw_item["chunk_path"]))
            try:
                result = _transcribe_chunk(model, chunk_file)
                segments = _validated_segments_payload(result, chunk_file=chunk_file)
            except RuntimeError as exc:
                if _EMPTY_TRANSCRIBE_ERROR_SNIPPET not in str(exc):
                    raise
                segments = []
            print(
                _WORKER_RESULT_PREFIX
                + json.dumps(
                    {
                        "chunk_index": chunk_idx,
                        "segments": segments,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        return 0
    except Exception as exc:
        print(
            f"isolated Whisper chunk worker failed: {type(exc).__name__}: {_compact_exception_message(exc)}",
            file=sys.stderr,
        )
        return 1


def _compact_subprocess_output(output: str, limit: int = 1200) -> str:
    message = " ".join(str(output or "").split())
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def stable_cache_key_for_source(source: str | pathlib.Path | None) -> str | None:
    value = str(source or "").strip()
    if not value:
        return None
    return sha256(value.encode("utf-8")).hexdigest()


def _chunk_cache_dir(cache_key: str | None) -> pathlib.Path | None:
    digest = stable_cache_key_for_source(cache_key)
    if not digest:
        return None
    cache_dir = pathlib.Path(settings.storage_dir) / "_transcript_cache" / digest / "chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _chunk_cache_path(cache_dir: pathlib.Path | None, chunk_idx: int) -> pathlib.Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"chunk_{chunk_idx:03d}.json"


def _chunk_cache_metadata(chunk_idx: int) -> dict[str, object]:
    return {
        "version": _TRANSCRIPT_CHUNK_CACHE_VERSION,
        "chunk_index": int(chunk_idx),
        "chunk_seconds": _chunk_seconds(),
        "backend": _resolved_whisper_backend(),
        "whisper_model": str(settings.whisper_model or ""),
        "language": str(settings.whisper_language or "").strip().lower(),
        "word_timestamps": _word_timestamps_enabled(),
    }


def _load_chunk_cache(cache_dir: pathlib.Path | None, chunk_idx: int) -> list[RichSegment] | None:
    cache_path = _chunk_cache_path(cache_dir, chunk_idx)
    if cache_path is None or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("transcriber_chunk_cache_invalid path=%s reason=read_error", cache_path)
        return None
    if not isinstance(payload, dict):
        logger.warning("transcriber_chunk_cache_invalid path=%s reason=payload_type", cache_path)
        return None
    if payload.get("metadata") != _chunk_cache_metadata(chunk_idx):
        return None
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        logger.warning("transcriber_chunk_cache_invalid path=%s reason=segments_type", cache_path)
        return None
    try:
        return [_segment_from_cache_record(record) for record in raw_segments]
    except (TypeError, ValueError):
        logger.warning("transcriber_chunk_cache_invalid path=%s reason=segment_payload", cache_path)
        return None


def _write_chunk_cache(
    cache_dir: pathlib.Path | None,
    chunk_idx: int,
    segments: list[RichSegment],
) -> None:
    cache_path = _chunk_cache_path(cache_dir, chunk_idx)
    if cache_path is None:
        return
    payload = {
        "metadata": _chunk_cache_metadata(chunk_idx),
        "segments": [_segment_cache_record(segment) for segment in segments],
    }
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        tmp_path.replace(cache_path)
    except OSError as exc:
        logger.warning(
            "transcriber_chunk_cache_write_failed path=%s error_type=%s error=%s",
            cache_path,
            type(exc).__name__,
            _compact_exception_message(exc),
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _segment_cache_record(segment: RichSegment) -> dict[str, object]:
    return {
        "start": float(segment.start),
        "end": float(segment.end),
        "text": segment.text,
        "speaker": segment.speaker,
        "emotions": list(segment.emotions),
        "words_per_second": float(segment.words_per_second),
    }


def _segment_from_cache_record(record: object) -> RichSegment:
    if not isinstance(record, dict):
        raise TypeError("cached segment must be a mapping")
    emotions = record.get("emotions") or []
    if not isinstance(emotions, list):
        raise TypeError("cached segment emotions must be a list")
    return RichSegment(
        start=float(record["start"]),
        end=float(record["end"]),
        text=str(record.get("text") or "").strip(),
        speaker=str(record.get("speaker") or "UNKNOWN"),
        emotions=[str(value) for value in emotions],
        words_per_second=float(record.get("words_per_second") or 0.0),
    )



# Bump when the enrichment schema changes: caches written by older code are
# treated as a miss and re-enriched. Version 2 added the emote class scores —
# legacy caches (a bare list, no version field) silently zeroed hype/funny/shock.
_CHAT_ENRICHMENT_VERSION = 2


def _chat_enrichment_path(cache_key: str | None) -> pathlib.Path | None:
    digest = stable_cache_key_for_source(cache_key)
    if not digest:
        return None
    return pathlib.Path(settings.storage_dir) / "_transcript_cache" / digest / "chat_enriched.json"


def save_chat_enrichment(cache_key: str | None, segments: list[RichSegment]) -> None:
    """Persist chat fields from enriched segments alongside the transcript chunk cache."""
    path = _chat_enrichment_path(cache_key)
    if path is None:
        return
    records = [
        {
            "start": round(seg.start, 3),
            "chat_spike_score": round(seg.chat_spike_score, 4),
            "chat_message_count": seg.chat_message_count,
            "chat_hype_score":  round(seg.chat_hype_score, 4),
            "chat_funny_score": round(seg.chat_funny_score, 4),
            "chat_shock_score": round(seg.chat_shock_score, 4),
            "voice_energy_burst_score": round(seg.voice_energy_burst_score, 4),
        }
        for seg in segments
        if seg.chat_spike_score > 0 or seg.chat_message_count > 0
        or seg.voice_energy_burst_score > 0
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"version": _CHAT_ENRICHMENT_VERSION, "records": records}, ensure_ascii=True),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        pass


def has_chat_enrichment(cache_key: str | None) -> bool:
    """True if a current-version chat enrichment cache exists for cache_key."""
    path = _chat_enrichment_path(cache_key)
    if path is None or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and int(payload.get("version") or 0) == _CHAT_ENRICHMENT_VERSION


def apply_chat_enrichment(
    cache_key: str | None,
    segments: list[RichSegment],
) -> list[RichSegment] | None:
    """Load cached chat enrichment and apply it to segments. Returns None on cache miss."""
    path = _chat_enrichment_path(cache_key)
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or int(payload.get("version") or 0) != _CHAT_ENRICHMENT_VERSION:
        return None  # legacy/unknown schema — treat as miss so enrichment reruns
    records = payload.get("records")
    if not isinstance(records, list):
        return None
    lookup: dict[float, dict] = {round(r["start"], 3): r for r in records if isinstance(r, dict)}
    result: list[RichSegment] = []
    for seg in segments:
        rec = lookup.get(round(seg.start, 3))
        if rec:
            result.append(RichSegment(
                start=seg.start, end=seg.end, text=seg.text,
                speaker=seg.speaker, emotions=seg.emotions,
                words_per_second=seg.words_per_second,
                chat_spike_score=float(rec.get("chat_spike_score", 0.0)),
                chat_message_count=int(rec.get("chat_message_count", 0)),
                chat_hype_score=float(rec.get("chat_hype_score", 0.0)),
                chat_funny_score=float(rec.get("chat_funny_score", 0.0)),
                chat_shock_score=float(rec.get("chat_shock_score", 0.0)),
                # Keep the segment's freshly computed energy when the cached
                # record predates it — the energy stage runs before this one.
                voice_energy_burst_score=float(
                    rec.get("voice_energy_burst_score") or seg.voice_energy_burst_score or 0.0
                ),
            ))
        else:
            result.append(seg)
    return result


_VOICE_ENERGY_VERSION = 1


def _voice_energy_path(cache_key: str | None) -> pathlib.Path | None:
    digest = stable_cache_key_for_source(cache_key)
    if not digest:
        return None
    return pathlib.Path(settings.storage_dir) / "_transcript_cache" / digest / "voice_energy.json"


def save_voice_energy(cache_key: str | None, segments: list[RichSegment], scores: list[float]) -> None:
    """Persist per-segment voice energy so transcript-cache hits keep the signal."""
    path = _voice_energy_path(cache_key)
    if path is None or len(scores) != len(segments):
        return
    records = [
        {"start": round(seg.start, 3), "score": round(float(score), 4)}
        for seg, score in zip(segments, scores)
        if float(score) > 0
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"version": _VOICE_ENERGY_VERSION, "records": records}, ensure_ascii=True),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        pass


def load_voice_energy(cache_key: str | None, segments: list[RichSegment]) -> list[float] | None:
    """Return per-segment energy scores aligned to `segments`, or None on cache miss."""
    path = _voice_energy_path(cache_key)
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or int(payload.get("version") or 0) != _VOICE_ENERGY_VERSION:
        return None
    records = payload.get("records")
    if not isinstance(records, list):
        return None
    lookup = {
        round(float(r["start"]), 3): float(r.get("score") or 0.0)
        for r in records
        if isinstance(r, dict) and "start" in r
    }
    return [lookup.get(round(seg.start, 3), 0.0) for seg in segments]


def _filter_transcribable_chunks(
    chunk_files: list[pathlib.Path],
) -> list[tuple[int, pathlib.Path]]:
    valid_chunks: list[tuple[int, pathlib.Path]] = []
    for chunk_idx, chunk_file in enumerate(chunk_files):
        if _chunk_frame_count(chunk_file) > 0:
            valid_chunks.append((chunk_idx, chunk_file))
            continue
        logger.warning(
            "transcriber_skip_chunk chunk=%s reason=empty_or_invalid_wav",
            chunk_file,
        )
    return valid_chunks


def _chunk_frame_count(chunk_file: pathlib.Path) -> int:
    try:
        with wave.open(str(chunk_file), "rb") as wav_file:
            return int(wav_file.getnframes())
    except (FileNotFoundError, wave.Error, EOFError):
        return 0


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------

def _calc_tempo(seg: dict, time_offset: float) -> float:
    """Words per second using the full segment duration as the denominator."""
    words = seg.get("words", [])
    segment_duration = seg["end"] - seg["start"]
    if words:
        word_count = len(words)
    else:
        word_count = len(seg["text"].split())

    # Sub-500 ms segments are too noisy for tempo scoring and frequently
    # represent single-word interjections rather than sustained excited speech.
    if segment_duration < 0.5:
        return 0.0

    return round(word_count / segment_duration, 2) if segment_duration > 0 else 0.0


def _transcribe_chunk(model: object, chunk_file: pathlib.Path) -> dict:
    if _whisper_model_backend == "faster-whisper":
        return _transcribe_chunk_faster(model, chunk_file)
    return _transcribe_chunk_openai(model, chunk_file)


def _transcribe_chunk_openai(model: whisper.Whisper, chunk_file: pathlib.Path) -> dict:
    with _whisper_inference_lock:
        use_fp16 = str(_model_device_name(model)).startswith("cuda")
        transcribe_kwargs: dict[str, object] = {
            "word_timestamps": _word_timestamps_enabled(),
            "verbose": False,
            "fp16": use_fp16,
        }
        configured_language = str(settings.whisper_language or "").strip().lower()
        if configured_language:
            transcribe_kwargs["language"] = configured_language
        return model.transcribe(
            str(chunk_file),
            **transcribe_kwargs,
        )


def _transcribe_chunk_faster(model: object, chunk_file: pathlib.Path) -> dict:
    with _whisper_inference_lock:
        configured_language = str(settings.whisper_language or "").strip().lower()
        transcribe_kwargs: dict[str, object] = {
            "word_timestamps": _word_timestamps_enabled(),
            "vad_filter": _vad_filter_enabled(),
        }
        if configured_language:
            transcribe_kwargs["language"] = configured_language
        segments_iter, _info = model.transcribe(str(chunk_file), **transcribe_kwargs)
        segments = []
        for segment in segments_iter:
            payload: dict[str, object] = {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text or "").strip(),
            }
            words = getattr(segment, "words", None)
            if words:
                payload["words"] = [
                    {
                        "start": float(getattr(word, "start", 0.0) or 0.0),
                        "end": float(getattr(word, "end", 0.0) or 0.0),
                        "word": str(getattr(word, "word", "") or ""),
                    }
                    for word in words
                ]
            segments.append(payload)
        return {"segments": segments}


def _validated_segments_payload(
    result: dict,
    *,
    chunk_file: pathlib.Path,
) -> list[object]:
    payload = result.get("segments") or []
    if isinstance(payload, list):
        return payload
    raise ValueError(
        f"invalid Whisper segments payload for {chunk_file}: expected list, got {type(payload).__name__}"
    )


def _normalize_segment(
    raw_seg: object,
    *,
    chunk_file: pathlib.Path,
    segment_index: int,
) -> dict:
    if not isinstance(raw_seg, dict):
        raise ValueError(
            "invalid Whisper segment payload "
            f"for {chunk_file} segment {segment_index}: expected mapping, got {type(raw_seg).__name__}"
        )

    try:
        start = float(raw_seg["start"])
        end = float(raw_seg["end"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"invalid Whisper timestamps for {chunk_file} segment {segment_index}"
        )

    text = str(raw_seg.get("text") or "").strip()
    if end < start:
        raise ValueError(
            "non-positive Whisper segment duration "
            f"for {chunk_file} segment {segment_index}: start={start:.3f} end={end:.3f}"
        )
    if end == start:
        adjusted_end = start + _MIN_SEGMENT_DURATION_SECONDS
        logger.warning(
            "transcriber_clamp_zero_duration_segment chunk=%s segment=%s start=%.3f end=%.3f adjusted_end=%.3f",
            chunk_file,
            segment_index,
            start,
            end,
            adjusted_end,
        )
        end = adjusted_end

    normalized = dict(raw_seg)
    normalized["start"] = start
    normalized["end"] = end
    normalized["text"] = text
    return normalized


# ---------------------------------------------------------------------------
# Emotional markers
# ---------------------------------------------------------------------------

def _tag_emotions(text: str) -> list[str]:
    tags: list[str] = []
    if _LAUGHTER_RE.search(text):
        tags.append("laughter")
    if _EXCLAMATION_RE.search(text):
        tags.append("exclamation")
    if _QUESTION_RE.search(text):
        tags.append("question")
    return tags


# ---------------------------------------------------------------------------
# Speaker diarization via pyannote
# ---------------------------------------------------------------------------

def _diarize(audio_path: pathlib.Path, segments: list[RichSegment]) -> list[RichSegment]:
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return segments  # pyannote not installed — skip silently

    try:
        pipeline = _load_pyannote_pipeline(Pipeline, settings.hf_token)
        if pipeline is None:
            logger.warning(
                "transcriber_diarization_unavailable audio=%s reason=%s",
                audio_path,
                "pipeline_not_loaded",
            )
            return segments
        diarization = pipeline(str(audio_path))
    except Exception as exc:
        logger.warning(
            "transcriber_diarization_unavailable audio=%s reason=%s error_type=%s error=%s",
            audio_path,
            _diarization_failure_reason(exc),
            type(exc).__name__,
            _compact_exception_message(exc),
        )
        return segments

    # Build speaker timeline: list of (start, end, speaker_label)
    speaker_timeline: list[tuple[float, float, str]] = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    # Determine which pyannote label maps to the streamer
    # Heuristic: the speaker with the most total speaking time is the streamer
    speaker_duration: dict[str, float] = {}
    for s, e, spk in speaker_timeline:
        speaker_duration[spk] = speaker_duration.get(spk, 0.0) + (e - s)
    streamer_label = max(speaker_duration, key=lambda k: speaker_duration[k]) if speaker_duration else None

    for seg in segments:
        dominant = _dominant_speaker(seg.start, seg.end, speaker_timeline)
        if dominant is None:
            seg.speaker = "UNKNOWN"
        elif dominant == streamer_label:
            seg.speaker = "STREAMER"
        else:
            # Stable guest numbering: GUEST_1, GUEST_2, ...
            guests = sorted(k for k in speaker_duration if k != streamer_label)
            idx = guests.index(dominant) + 1 if dominant in guests else 0
            seg.speaker = f"GUEST_{idx}"

    return segments


def _load_pyannote_pipeline(Pipeline, hf_token: str | None):
    model_id = "pyannote/speaker-diarization-3.1"
    token = str(hf_token or "").strip()
    if not token:
        return Pipeline.from_pretrained(model_id)
    try:
        return Pipeline.from_pretrained(model_id, token=token)
    except TypeError as exc:
        message = str(exc)
        if "token" not in message or "unexpected keyword" not in message:
            raise
        return Pipeline.from_pretrained(model_id, use_auth_token=token)


def _diarization_failure_reason(exc: Exception) -> str:
    message = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    if (
        "gatedrepo" in exc_name
        or "gated repo" in message
        or "403" in message
        or "forbidden" in message
        or "restricted" in message
        or "authorized list" in message
    ):
        return "hf_gated_model_access"
    if "401" in message or "unauthorized" in message or "authentication" in message:
        return "hf_auth"
    if "could not download pipeline" in message or "hf_hub_download" in message:
        return "pipeline_download"
    return "pipeline_error"


def _compact_exception_message(exc: Exception, limit: int = 300) -> str:
    message = " ".join(str(exc).split())
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _dominant_speaker(
    seg_start: float,
    seg_end: float,
    timeline: list[tuple[float, float, str]],
) -> str | None:
    """Return the speaker with the most overlap with [seg_start, seg_end]."""
    overlap: dict[str, float] = {}
    for t_start, t_end, spk in timeline:
        o = min(seg_end, t_end) - max(seg_start, t_start)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0.0) + o
    return max(overlap, key=lambda k: overlap[k]) if overlap else None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(segments: list[RichSegment]) -> list[RichSegment]:
    if not segments:
        return []
    result = [segments[0]]
    for seg in segments[1:]:
        prev_end = result[-1].end
        if seg.start < prev_end - 1.0:
            continue
        result.append(seg)
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_transcript(
    segments: list[RichSegment] | list[Segment],
    use_seconds: bool = False,
) -> str:
    """Format segments as a transcript string.

    Args:
        segments: List of RichSegment or legacy (start, end, text) tuples.
        use_seconds: When True, emit timestamps as raw float seconds so the
            LLM can copy them directly into JSON without any conversion.
            When False (default), emit HH:MM:SS for human readability.
    """
    lines = []
    for seg in segments:
        if isinstance(seg, RichSegment):
            if use_seconds:
                ts = f"[{seg.start:.2f} - {seg.end:.2f}]"
            else:
                ts = f"[{_fmt(seg.start)} - {_fmt(seg.end)}]"
            parts = [ts]
            if seg.speaker != "UNKNOWN":
                parts.append(f"[{seg.speaker}]")
            if seg.emotions:
                parts.append(f"[{','.join(seg.emotions).upper()}]")
            if seg.words_per_second:
                parts.append(f"[{seg.words_per_second:.1f}w/s]")
            if seg.chat_spike_score > 0:
                parts.append(f"[CHAT:{seg.chat_spike_score:.2f}]")
            parts.append(seg.text)
            lines.append(" ".join(parts))
        else:
            # Legacy tuple support
            start, end, text = seg
            if use_seconds:
                ts = f"[{start:.2f} - {end:.2f}]"
            else:
                ts = f"[{_fmt(start)} - {_fmt(end)}]"
            lines.append(f"{ts} {text}")
    return "\n".join(lines)


def format_jsonl(segments: list[RichSegment]) -> str:
    """Return one JSON object per line, preserving all numerical features."""
    lines = []
    for seg in segments:
        if not isinstance(seg, RichSegment):
            continue
        lines.append(json.dumps({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "speaker": seg.speaker,
            "text": seg.text,
            "tags": seg.emotions,
            "words_per_second": seg.words_per_second,
            "chat_spike_score": seg.chat_spike_score,
            "chat_message_count": seg.chat_message_count,
            "chat_hype_score": seg.chat_hype_score,
            "chat_funny_score": seg.chat_funny_score,
            "chat_shock_score": seg.chat_shock_score,
            "voice_energy_burst_score": seg.voice_energy_burst_score,
        }))
    return "\n".join(lines)


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    raise SystemExit(_transcribe_chunk_worker_main())
