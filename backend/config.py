import os
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator


BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_TMP_DIR = BACKEND_ROOT / "tmp"
DEFAULT_STORAGE_DIR = DEFAULT_TMP_DIR / "storage"
DEFAULT_CACHE_DIR = DEFAULT_TMP_DIR / "cache"
DEFAULT_MPLCONFIGDIR = DEFAULT_TMP_DIR / "matplotlib"


def _resolve_local_project_path(
    value: str | None,
    *,
    field_name: str,
    default: Path | None = None,
) -> str | None:
    raw = (value or "").strip()
    if not raw:
        if default is None:
            return None
        candidate = default
    else:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = BACKEND_ROOT / candidate

    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must stay inside the project directory: {PROJECT_ROOT}"
        ) from exc
    return str(candidate)


def _configure_local_cache_env() -> None:
    cache_root = DEFAULT_CACHE_DIR.resolve()
    mpl_dir = DEFAULT_MPLCONFIGDIR.resolve()
    env_paths = {
        "XDG_CACHE_HOME": cache_root,
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "MPLCONFIGDIR": mpl_dir,
    }
    for path in env_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for key, path in env_paths.items():
        os.environ[key] = str(path)


_configure_local_cache_env()


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    whisper_model: str = "base"
    whisper_device: str = "auto"
    # Comma-separated list in env (ALLOWED_ORIGINS="https://app.example.com,https://staging.example.com").
    # Defaults cover the common local dev frontend ports.
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ]
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    storage_dir: str = str(DEFAULT_STORAGE_DIR)
    max_clip_duration: int = 90
    max_upload_size_bytes: int | None = None
    clip_profile: str = "general"
    profiles_path: str | None = None  # Optional path to a JSON file with custom profile overrides
    hf_token: str | None = None  # HuggingFace token for pyannote diarization (optional)
    groq_api_key: str | None = None  # Groq API key for fast semantic scoring (free tier)
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str | None = None  # Google Gemini API key for clip boundary refinement (free tier)
    gemini_model: str = "gemini-2.0-flash"
    twitch_client_id: str | None = None
    twitch_client_secret: str | None = None
    enable_chat_analysis: bool = False
    chat_download_timeout_sec: int = 30
    chat_join_timeout_sec: float = 1.0
    twitch_download_cache_ttl_sec: int = 1800
    twitch_download_concurrent_fragments: int = 8
    twitch_download_use_aria2c: bool = True
    twitch_download_aria2_connections: int = 16
    training_vod_workers: int = 2
    training_label_workers: int = 2
    semantic_device: str = "auto"
    semantic_batch_size: int = 0
    video_expression_sample_sec: int = 3
    shortlist_face_reaction_top_k: int = 12
    shortlist_face_reaction_sample_sec: int = 1
    shortlist_llm_top_k: int = 12
    enable_clip_judge: bool = True
    clip_judge_timeout_sec: float = 90.0
    analyzer_refine_timeout_sec: float | None = None
    shortlist_judge_top_k: int = 24
    judge_blend_weight: float = 0.65
    judge_context_lookback_sec: float = 120.0
    enable_multimodal_clip_judge: bool = True
    judge_keyframes_per_candidate: int = 4
    judge_keyframe_width: int = 640
    enable_hook_subclassifier: bool = False
    enable_multimodal_embeddings: bool = False
    enable_post_publish_labels: bool = False
    enable_trend_prior: bool = True
    # Trained ranker is gated behind quality checks in ranker.py (training rows,
    # positive/negative balance, validation accuracy/NDCG). Enabling by default
    # lets the feedback/training loop actually reach the scoring path — a bad
    # artifact still falls back to the heuristic via those gates.
    enable_trained_ranker_v2: bool = True
    trend_prior_data_path: str | None = None
    enable_boundary_refinement_v2: bool = True
    enable_boundary_llm_pair_rerank: bool = True
    enable_word_level_boundaries: bool = False
    enable_boundary_ranker: bool = False
    enable_legacy_boundary_refinement_llm: bool = False
    ranker_hpo_trials: int = 60
    ranker_hpo_seed: int = 0

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("storage_dir", mode="before")
    @classmethod
    def _validate_storage_dir(cls, value: str | None) -> str:
        return _resolve_local_project_path(
            value,
            field_name="storage_dir",
            default=DEFAULT_STORAGE_DIR,
        ) or str(DEFAULT_STORAGE_DIR.resolve())

settings = Settings()
