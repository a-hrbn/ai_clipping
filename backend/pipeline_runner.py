from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
import sys
import tempfile

from config import settings
from job_lifecycle import is_terminal
from job_manager import get_job, is_self_managed_job, update_job

_BACKEND_ROOT = pathlib.Path(__file__).resolve().parent


def build_subprocess_command(
    *,
    job_id: str,
    url: str | None,
    video_path: pathlib.Path | None,
    profile: str,
    strategy: dict,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_BACKEND_ROOT / "pipeline_worker.py"),
        "--job-id",
        job_id,
        "--profile",
        profile,
        "--strategy-json",
        json.dumps(_json_safe(strategy or {}), ensure_ascii=True),
    ]
    if video_path is not None:
        cmd.extend(["--video-path", str(video_path)])
    elif url:
        cmd.extend(["--url", url])
    return cmd


def build_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(_BACKEND_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast")
    env.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env["MPLCONFIGDIR"] = str(writable_cache_dir("matplotlib"))
    env.setdefault("WHISPER_LANGUAGE", "en")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    if (env.get("WHISPER_DEVICE") or "").strip().lower() == "cuda":
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault(
        "PIPELINE_ENABLE_DIARIZATION",
        "true" if settings.pipeline_enable_diarization else "false",
    )
    env.setdefault("AI_CLIPPING_DISABLE_FACE_DETECTION", "0")
    return env


def keeps_source_media(job_id: str) -> bool:
    """True if a job's source media (video, audio, chunks) outlives its run.

    Debug/training/backfill workflows manage their own media; serving jobs keep
    it only with ``frontend_job_cache_enabled``.
    """
    if bool(getattr(settings, "frontend_job_cache_enabled", False)):
        return True
    return is_self_managed_job(job_id)


async def run_job_subprocess(
    *,
    job_id: str,
    url: str | None = None,
    video_path: pathlib.Path | None = None,
    profile: str,
    strategy: dict,
) -> int:
    cmd = build_subprocess_command(
        job_id=job_id,
        url=url,
        video_path=video_path,
        profile=profile,
        strategy=strategy,
    )
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_BACKEND_ROOT),
        env=build_subprocess_env(),
    )
    returncode = await process.wait()
    if returncode != 0:
        job = get_job(job_id)
        if job is not None and not is_terminal(job.status):
            update_job(job_id, status="failed", error=_subprocess_error(returncode))
    return returncode


def _subprocess_error(returncode: int) -> str:
    if returncode < 0:
        signal = -returncode
        if signal == 9:
            return "Pipeline subprocess exited from signal 9 (SIGKILL; likely OOM or external kill)"
        return f"Pipeline subprocess exited from signal {signal}"
    return f"Pipeline subprocess exited with code {returncode}"


def writable_cache_dir(name: str) -> pathlib.Path:
    """Resolve a writable cache directory, preferring a sibling of the storage dir.

    Shared by the subprocess env builder and the status router's face-detection
    probe, so both adapters resolve ``MPLCONFIGDIR`` through one seam.
    """
    preferred = pathlib.Path(settings.storage_dir).parent / name
    fallback = pathlib.Path(tempfile.gettempdir()) / f"ai_clipping_{name}"
    for candidate in (preferred, fallback):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return fallback


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return value
