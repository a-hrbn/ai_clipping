from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_SUMMARY_PATH = ARTIFACT_DIR / "local_train_summary.json"
DEFAULT_STORAGE_DIR = "tmp/storage"
DEFAULT_TRAIN_TMPDIR = "tmp/training_runs"


@dataclass(frozen=True)
class WorkerPlan:
    vod_workers: int
    label_workers: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VOD ranker training flow locally inside Docker- or workstation-friendly runtime settings.",
    )
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--streamer", default="jynxzi")
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--train-tmpdir", default=DEFAULT_TRAIN_TMPDIR)
    parser.add_argument("--whisper-model", default=os.environ.get("WHISPER_MODEL", "base"))
    parser.add_argument("--whisper-device", default=os.environ.get("WHISPER_DEVICE", "cuda"))
    parser.add_argument("--whisper-language", default=os.environ.get("WHISPER_LANGUAGE", "en"))
    parser.add_argument(
        "--semantic-device",
        default=os.environ.get("SEMANTIC_DEVICE", "cuda"),
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=int(os.environ.get("SEMANTIC_BATCH_SIZE", "0") or "0"),
    )
    parser.add_argument(
        "--dataset-llm-provider",
        default=os.environ.get("TRAINING_DATASET_LLM_PROVIDER", "none"),
        help="Use none for fastest local dataset builds; ollama remains available if explicitly requested.",
    )
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434"))
    parser.add_argument("--training-vod-workers", type=int, default=0)
    parser.add_argument("--training-label-workers", type=int, default=0)
    parser.add_argument(
        "--training-ffmpeg-video-codec",
        default=os.environ.get("TRAINING_FFMPEG_VIDEO_CODEC", "h264_nvenc"),
    )
    parser.add_argument(
        "--training-ffmpeg-audio-codec",
        default=os.environ.get("TRAINING_FFMPEG_AUDIO_CODEC", "aac"),
    )
    parser.add_argument(
        "--training-ffmpeg-preset",
        default=os.environ.get("TRAINING_FFMPEG_PRESET", ""),
    )
    parser.add_argument(
        "--training-ffmpeg-threads",
        type=int,
        default=int(os.environ.get("TRAINING_FFMPEG_THREADS", "0") or "0"),
    )
    parser.add_argument("--video-expression-sample-sec", type=int, default=6)
    parser.add_argument("--artifact-only", action="store_true")
    parser.add_argument("--skip-covered", dest="skip_covered", action="store_true", default=True)
    parser.add_argument("--no-skip-covered", dest="skip_covered", action="store_false")
    parser.add_argument("--merge-output", dest="merge_output", action="store_true", default=True)
    parser.add_argument("--no-merge-output", dest="merge_output", action="store_false")
    return parser.parse_args()


def _recommended_worker_plan(cpu_count: int) -> WorkerPlan:
    if cpu_count >= 12:
        return WorkerPlan(vod_workers=2, label_workers=3)
    if cpu_count >= 8:
        return WorkerPlan(vod_workers=2, label_workers=2)
    return WorkerPlan(vod_workers=1, label_workers=1)


def _configure_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    cpu_count = max(os.cpu_count() or 1, 1)
    worker_plan = _recommended_worker_plan(cpu_count)
    vod_workers = args.training_vod_workers if args.training_vod_workers > 0 else worker_plan.vod_workers
    label_workers = args.training_label_workers if args.training_label_workers > 0 else worker_plan.label_workers
    semantic_batch_size = args.semantic_batch_size if args.semantic_batch_size > 0 else (
        256 if args.semantic_device.strip().lower() == "cuda" else 64
    )

    env.update(
        {
            "WHISPER_MODEL": args.whisper_model,
            "WHISPER_DEVICE": args.whisper_device,
            "WHISPER_LANGUAGE": args.whisper_language,
            "SEMANTIC_DEVICE": args.semantic_device,
            "SEMANTIC_BATCH_SIZE": str(semantic_batch_size),
            "STORAGE_DIR": args.storage_dir,
            "AI_CLIPPING_TRAIN_TMPDIR": args.train_tmpdir,
            "TRAINING_DATASET_LLM_PROVIDER": args.dataset_llm_provider,
            "TRAINING_VOD_WORKERS": str(vod_workers),
            "TRAINING_LABEL_WORKERS": str(label_workers),
            "TRAINING_FFMPEG_VIDEO_CODEC": args.training_ffmpeg_video_codec,
            "TRAINING_FFMPEG_AUDIO_CODEC": args.training_ffmpeg_audio_codec,
            "TRAINING_FFMPEG_PRESET": args.training_ffmpeg_preset,
            "TRAINING_FFMPEG_THREADS": str(args.training_ffmpeg_threads),
            "VIDEO_EXPRESSION_SAMPLE_SEC": str(max(args.video_expression_sample_sec, 1)),
            "SHORTLIST_FACE_REACTION_SAMPLE_SEC": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        }
    )
    if args.ollama_host.strip():
        env["OLLAMA_HOST"] = args.ollama_host.strip()
    if str(args.whisper_device).strip().lower() == "cuda" or str(args.semantic_device).strip().lower() == "cuda":
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _train_command(args: argparse.Namespace) -> list[str]:
    command = [
        "/bin/zsh",
        "ml/train_jynxzi_vod.sh",
        args.date_from,
        args.date_to,
    ]
    if args.artifact_only:
        command.append("--artifact-only")
    command.append("--skip-covered" if args.skip_covered else "--no-skip-covered")
    command.append("--merge-output" if args.merge_output else "--no-merge-output")
    return command


def _build_summary(args: argparse.Namespace, env: dict[str, str], training_return_code: int) -> dict[str, object]:
    cpu_count = max(os.cpu_count() or 1, 1)
    summary = {
        "streamer": args.streamer,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "training_return_code": training_return_code,
        "artifact_only": bool(args.artifact_only),
        "hardware_profile": {
            "cpu_count": cpu_count,
            "whisper_device": env.get("WHISPER_DEVICE", ""),
            "whisper_language": env.get("WHISPER_LANGUAGE", ""),
            "semantic_device": env.get("SEMANTIC_DEVICE", ""),
            "training_vod_workers": env.get("TRAINING_VOD_WORKERS", ""),
            "training_label_workers": env.get("TRAINING_LABEL_WORKERS", ""),
            "training_ffmpeg_video_codec": env.get("TRAINING_FFMPEG_VIDEO_CODEC", ""),
            "training_ffmpeg_audio_codec": env.get("TRAINING_FFMPEG_AUDIO_CODEC", ""),
            "training_ffmpeg_preset": env.get("TRAINING_FFMPEG_PRESET", ""),
            "training_ffmpeg_threads": env.get("TRAINING_FFMPEG_THREADS", ""),
            "semantic_batch_size": env.get("SEMANTIC_BATCH_SIZE", ""),
            "dataset_llm_provider": env.get("TRAINING_DATASET_LLM_PROVIDER", ""),
            "video_expression_sample_sec": env.get("VIDEO_EXPRESSION_SAMPLE_SEC", ""),
        },
        "paths": {
            "storage_dir": env.get("STORAGE_DIR", ""),
            "train_tmpdir": env.get("AI_CLIPPING_TRAIN_TMPDIR", ""),
            "artifact_dir": str(ARTIFACT_DIR),
        },
    }
    return summary


def main() -> int:
    args = _parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    env = _configure_runtime_env(args)
    command = _train_command(args)
    print(json.dumps({"event": "local_training_start", "command": command}, ensure_ascii=True))
    completed = subprocess.run(command, cwd=BACKEND_ROOT, env=env, check=False)
    summary = _build_summary(args, env, completed.returncode)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        print(json.dumps({"event": "local_training_failed", "summary_path": str(summary_path)}, ensure_ascii=True))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
