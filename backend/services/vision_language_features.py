from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import pathlib
import tempfile
from functools import lru_cache

import ffmpeg

from config import settings
from services.timeline_types import Candidate

logger = logging.getLogger(__name__)

_MODEL_ID = "openai/clip-vit-base-patch32"
_PROMPT_BANK = {
    "clutch": [
        "a clutch one tap headshot in a competitive shooter",
        "a player winning an intense clutch round",
        "an ace or multi kill moment in a shooter game",
    ],
    "comedy": [
        "a funny streamer reaction moment",
        "a chaotic funny gaming fail",
        "a streamer laughing at something absurd",
    ],
    "reaction_face": [
        "a shocked streamer face reaction",
        "a streamer yelling in surprise",
        "a close up expressive reaction face",
    ],
    "scoreboard": [
        "a game scoreboard or kill feed visible on screen",
        "a round win or match result screen",
        "a competitive shooter kill feed event",
    ],
}


def empty_scores() -> dict[str, float]:
    return {
        "vl_prompt_max_score": 0.0,
        "vl_clutch_score": 0.0,
        "vl_comedy_score": 0.0,
        "vl_reaction_face_score": 0.0,
        "vl_scoreboard_score": 0.0,
    }


def score_candidates(
    video_path: pathlib.Path | None,
    candidates: list[Candidate],
    *,
    model_id: str = _MODEL_ID,
) -> dict[str, dict[str, float]]:
    if not video_path or not pathlib.Path(video_path).exists() or not candidates:
        return {}
    if not bool(getattr(settings, "enable_multimodal_embeddings", False)):
        return {}
    model_bundle = _load_model(model_id)
    if model_bundle is None:
        return {}
    results: dict[str, dict[str, float]] = {}
    cache_dir = pathlib.Path(settings.storage_dir) / "vision_language_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        cache_path = cache_dir / f"{_cache_key(video_path, candidate, model_id)}.json"
        if cache_path.exists():
            try:
                results[candidate.id] = json.loads(cache_path.read_text(encoding="utf-8"))
                continue
            except (OSError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)
        try:
            scores = _score_candidate(video_path, candidate, model_bundle)
        except Exception as exc:
            logger.warning("vision_language_score_failed candidate_id=%s error=%s", candidate.id, exc)
            continue
        cache_path.write_text(json.dumps(scores, ensure_ascii=True, indent=2), encoding="utf-8")
        results[candidate.id] = scores
    return results


def model_availability(model_id: str = _MODEL_ID) -> dict[str, object]:
    if not bool(getattr(settings, "enable_multimodal_embeddings", False)):
        return {
            "enabled": False,
            "available": False,
            "reason": "multimodal_embeddings_disabled",
            "model_id": model_id,
            "local_files_only": True,
        }
    missing_packages = [
        name
        for name in ("torch", "transformers", "PIL")
        if importlib.util.find_spec(name) is None
    ]
    if missing_packages:
        return {
            "enabled": True,
            "available": False,
            "reason": "missing_python_packages",
            "missing_packages": missing_packages,
            "model_id": model_id,
            "local_files_only": True,
        }
    available = _load_model(model_id) is not None
    return {
        "enabled": True,
        "available": available,
        "reason": "local_model_loaded" if available else "local_model_unavailable",
        "model_id": model_id,
        "local_files_only": True,
    }


def _score_candidate(video_path: pathlib.Path, candidate: Candidate, model_bundle: tuple[object, object, object]) -> dict[str, float]:
    frame_times = _frame_times(candidate)
    if not frame_times:
        return empty_scores()
    with tempfile.TemporaryDirectory(prefix="vl_frames_") as tmp:
        frame_paths: list[pathlib.Path] = []
        tmp_dir = pathlib.Path(tmp)
        for index, time_sec in enumerate(frame_times):
            output = tmp_dir / f"frame_{index}.jpg"
            _extract_frame(video_path, time_sec, output)
            if output.exists() and output.stat().st_size > 0:
                frame_paths.append(output)
        if not frame_paths:
            return empty_scores()
        return _score_images_with_prompts(frame_paths, model_bundle)


def _score_images_with_prompts(frame_paths: list[pathlib.Path], model_bundle: tuple[object, object, object]) -> dict[str, float]:
    processor, model, torch = model_bundle
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in frame_paths]
    prompts = [prompt for prompts in _PROMPT_BANK.values() for prompt in prompts]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1).detach().cpu().numpy()

    scores: dict[str, float] = {}
    cursor = 0
    max_score = 0.0
    for key, bank_prompts in _PROMPT_BANK.items():
        width = len(bank_prompts)
        value = float(probs[:, cursor:cursor + width].max()) if width else 0.0
        scores[f"vl_{key}_score"] = round(min(max(value * len(prompts), 0.0), 1.0), 3)
        max_score = max(max_score, scores[f"vl_{key}_score"])
        cursor += width
    scores["vl_prompt_max_score"] = round(max_score, 3)
    return {**empty_scores(), **scores}


@lru_cache(maxsize=1)
def _load_model(model_id: str) -> tuple[object, object, object] | None:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(model_id, local_files_only=True)
        model.eval()
        return processor, model, torch
    except Exception as exc:
        logger.warning("vision_language_model_unavailable model_id=%s error=%s", model_id, exc)
        return None


def _frame_times(candidate: Candidate) -> list[float]:
    duration = max(float(candidate.end) - float(candidate.start), 0.1)
    return [
        round(min(max(candidate.start + duration * ratio, candidate.start), max(candidate.end - 0.05, candidate.start)), 3)
        for ratio in (0.12, 0.35, 0.65, 0.88)
    ]


def _extract_frame(video_path: pathlib.Path, time_sec: float, output_path: pathlib.Path) -> None:
    width = max(int(getattr(settings, "judge_keyframe_width", 640) or 640), 0)
    stream = ffmpeg.input(str(video_path), ss=max(time_sec, 0.0))
    if width > 0:
        stream = stream.filter("scale", width, -2)
    (
        stream
        .output(str(output_path), vframes=1, format="image2", vcodec="mjpeg", qscale=4)
        .overwrite_output()
        .run(quiet=True)
    )


def _cache_key(video_path: pathlib.Path, candidate: Candidate, model_id: str) -> str:
    try:
        stat = pathlib.Path(video_path).stat()
        video_identity = f"{video_path}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        video_identity = str(video_path)
    payload = {
        "video": video_identity,
        "candidate_id": candidate.id,
        "start": round(float(candidate.start), 3),
        "end": round(float(candidate.end), 3),
        "anchor": round(float(candidate.anchor_time), 3),
        "model_id": model_id,
        "version": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
