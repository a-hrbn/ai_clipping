import os
import sys
import tempfile
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import AliasChoices, ConfigDict, Field, field_validator


BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_TMP_DIR = BACKEND_ROOT / "tmp"
DEFAULT_STORAGE_DIR = DEFAULT_TMP_DIR / "storage"
DEFAULT_CACHE_DIR = DEFAULT_TMP_DIR / "cache"
DEFAULT_MPLCONFIGDIR = DEFAULT_TMP_DIR / "matplotlib"


def _prepare_local_cache_path(path: Path, *, fallback_name: str) -> Path:
    candidates = [
        path,
        DEFAULT_TMP_DIR / "_runtime_cache" / fallback_name,
        Path(tempfile.gettempdir()) / f"ai_clipping_{fallback_name}",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_writable_cache_dir(candidate):
            return candidate
    return candidates[-1].resolve()


def _is_writable_cache_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


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
        "XDG_CACHE_HOME": _prepare_local_cache_path(cache_root, fallback_name="cache"),
        "HF_HOME": _prepare_local_cache_path(cache_root / "huggingface", fallback_name="cache_huggingface"),
        "HUGGINGFACE_HUB_CACHE": _prepare_local_cache_path(
            cache_root / "huggingface" / "hub",
            fallback_name="cache_huggingface_hub",
        ),
        "TRANSFORMERS_CACHE": _prepare_local_cache_path(
            cache_root / "huggingface" / "transformers",
            fallback_name="cache_transformers",
        ),
        "TORCH_HOME": _prepare_local_cache_path(cache_root / "torch", fallback_name="cache_torch"),
        "MPLCONFIGDIR": _prepare_local_cache_path(mpl_dir, fallback_name="matplotlib"),
        "DEEPFACE_HOME": _prepare_local_cache_path(cache_root / "deepface", fallback_name="cache_deepface"),
    }
    for key, path in env_paths.items():
        os.environ[key] = str(path)


def _configure_local_tool_path() -> None:
    venv_bin = Path(sys.prefix).resolve() / "bin"
    if not (venv_bin / "ffmpeg").exists():
        return
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    venv_bin_str = str(venv_bin)
    if venv_bin_str not in parts:
        os.environ["PATH"] = os.pathsep.join([venv_bin_str, *parts])


_configure_local_cache_env()
_configure_local_tool_path()


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=BACKEND_ROOT / ".env", extra="ignore")

    whisper_model: str = "base"
    whisper_device: str = "auto"
    whisper_language: str = ""
    whisper_backend: str = "auto"
    whisper_compute_type: str = ""
    whisper_word_timestamps: bool = False
    transcription_chunk_seconds: int = 180
    transcription_worker_batch_size: int = 12
    # Comma-separated list in env (ALLOWED_ORIGINS="https://app.example.com,https://staging.example.com").
    # Defaults cover the common local dev frontend ports.
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ]
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    storage_dir: str = str(DEFAULT_STORAGE_DIR)
    yt_dlp_cookiefile: str | None = Field(
        default=None,
        validation_alias=AliasChoices("YTDLP_COOKIEFILE", "YT_DLP_COOKIEFILE"),
    )
    yt_dlp_cookies_from_browser: str | None = Field(
        default=None,
        validation_alias=AliasChoices("YTDLP_COOKIES_FROM_BROWSER", "YT_DLP_COOKIES_FROM_BROWSER"),
    )
    yt_dlp_short_form_sleep_requests_sec: float = Field(
        default=5.0,
        validation_alias=AliasChoices("YTDLP_SHORT_FORM_SLEEP_REQUESTS_SEC", "YT_DLP_SHORT_FORM_SLEEP_REQUESTS_SEC"),
    )
    yt_dlp_short_form_min_sleep_interval_sec: float = Field(
        default=5.0,
        validation_alias=AliasChoices("YTDLP_SHORT_FORM_MIN_SLEEP_INTERVAL_SEC", "YT_DLP_SHORT_FORM_MIN_SLEEP_INTERVAL_SEC"),
    )
    yt_dlp_short_form_max_sleep_interval_sec: float = Field(
        default=10.0,
        validation_alias=AliasChoices("YTDLP_SHORT_FORM_MAX_SLEEP_INTERVAL_SEC", "YT_DLP_SHORT_FORM_MAX_SLEEP_INTERVAL_SEC"),
    )
    max_clip_duration: int = 90
    # Only the top-K ranked moments per job are cut into clips (ffmpeg). The full
    # candidate set is still written to the debug JSONL files for review/eval;
    # this only bounds how many video clips are produced. 0 disables the cap.
    clip_top_k: int = 20
    max_upload_size_bytes: int | None = None
    clip_profile: str = "general"
    hf_token: str | None = None  # HuggingFace token for pyannote diarization (optional)
    anthropic_api_key: str | None = None  # Anthropic API key for Claude clip judging (optional)
    anthropic_model_bulk: str = "claude-haiku-4-5-20251001"
    anthropic_model_topk: str = "claude-sonnet-4-6"
    anthropic_prompt_caching: bool = True
    groq_api_key: str | None = None  # Groq API key for fast semantic scoring (free tier)
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str | None = None  # Google Gemini API key for clip boundary refinement (free tier)
    gemini_model: str = "gemini-2.5-flash-lite"
    twitch_client_id: str | None = None
    twitch_client_secret: str | None = None
    enable_chat_analysis: bool = True
    chat_download_timeout_sec: int = 30
    frontend_job_cache_enabled: bool = False
    frontend_transcript_cache_enabled: bool = True
    frontend_feature_cache_enabled: bool = True
    frontend_review_preview_limit: int = Field(
        default=0,
        validation_alias=AliasChoices("FRONTEND_REVIEW_PREVIEW_LIMIT"),
    )
    twitch_download_cache_ttl_sec: int = 86400
    twitch_download_max_height: int = 720
    twitch_download_concurrent_fragments: int = 8
    twitch_download_use_aria2c: bool = True
    twitch_download_aria2_connections: int = 16
    training_vod_prefetch: bool = True
    training_vod_max_height: int = 720
    training_vod_extended_window_scoped: bool = True
    training_vod_extended_context_sec: int = 300
    training_vod_extended_group_max_sec: int = 900
    training_vod_group_source_prefetch: bool = True
    training_vod_workers: int = 1
    training_label_workers: int = 3
    short_form_collection_workers: int = 3
    short_form_prefetch_workers: int = 2
    training_ffmpeg_video_codec: str = "libx264"
    training_ffmpeg_audio_codec: str = "aac"
    training_ffmpeg_preset: str = "veryfast"
    training_ffmpeg_threads: int = 1
    video_feature_ffmpeg_timeout_sec: float = 120.0
    pipeline_isolate_transcription_chunks: bool = True
    pipeline_enable_diarization: bool = False
    semantic_device: str = "auto"
    semantic_batch_size: int = 0
    video_expression_sample_sec: int = 3
    shortlist_face_reaction_top_k: int = 12
    shortlist_face_reaction_sample_sec: int = 1
    shortlist_llm_top_k: int = 12
    judge_provider: str | None = None
    judge_provider_fallback_enabled: bool = True
    analyzer_refine_timeout_sec: float | None = None
    shortlist_judge_top_k: int = 80
    judge_all_candidates: bool = False
    judge_recall_reserve_enabled: bool = True
    judge_recall_reserve_slots: int = 16
    judge_recall_reserve_pool_k: int = 240
    judge_recall_reserve_min_score: float = 0.12
    serving_cross_platform_recall_enabled: bool = False
    serving_cross_platform_feature_boosts_enabled: bool = False
    serving_cross_platform_positive_rows_path: str | None = None
    serving_cross_platform_recall_overlap_ratio: float = 0.45
    serving_cross_platform_judge_score_floor: float = 7.2
    serving_cross_platform_quality_min_judge_score: float = 3.0
    judge_blend_weight: float = 0.65
    judge_llm_first: bool = False
    judge_llm_first_min_confidence: float = 0.35
    judge_llm_first_tiebreaker_weight: float = 0.01
    judge_recall_filter_enabled: bool = True
    judge_recall_filter_min_score: float = 0.0
    judge_context_lookback_sec: float = 120.0
    serving_quality_gate_enabled: bool = True
    serving_quality_min_judge_score: float = 4.4
    serving_quality_min_standalone_clarity: float = 0.35
    serving_quality_min_payoff_strength: float = 0.35
    serving_quality_min_narrative_completeness: float = 0.45
    serving_quality_min_judge_confidence: float = 0.45
    serving_quality_subscore_gate_mode: str = "soft"
    serving_quality_hard_min_judge_score: float = 3.0
    serving_quality_soft_single_subscore_miss_enabled: bool = True
    serving_quality_soft_single_subscore_min_judge_score: float = 6.0
    serving_quality_soft_single_subscore_min_axis_mean: float = 0.55
    moment_finder_provider: str = "gemini"
    moment_finder_model: str = "gemini-2.5-flash-lite"
    moment_finder_max_moments: int = 180
    moment_finder_min_duration_sec: float = 8.0
    moment_finder_max_duration_sec: float = 90.0
    moment_finder_max_tokens: int = 32000
    moment_finder_timeout_sec: float = 900.0
    moment_finder_api_retries: int = 2
    moment_finder_retry_backoff_sec: float = 20.0
    moment_finder_filter_transcript_noise: bool = True
    moment_finder_multimodal_confirm_top_n: int = 12

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
