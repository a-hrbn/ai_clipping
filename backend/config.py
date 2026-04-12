from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    whisper_model: str = "base"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    storage_dir: str = "/tmp/ai_clipping"
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
    enable_chat_analysis: bool = True
    chat_download_timeout_sec: int = 30
    chat_join_timeout_sec: float = 1.0
    twitch_download_cache_ttl_sec: int = 1800
    training_vod_cache_ttl_sec: int = 2_592_000
    video_expression_sample_sec: int = 3
    shortlist_face_reaction_top_k: int = 12
    shortlist_face_reaction_sample_sec: int = 1
    shortlist_llm_top_k: int = 12
    enable_clip_judge: bool = True
    shortlist_judge_top_k: int = 24
    judge_blend_weight: float = 0.65
    judge_context_lookback_sec: float = 120.0
    enable_multimodal_clip_judge: bool = True
    judge_keyframes_per_candidate: int = 4
    judge_keyframe_width: int = 640
    enable_trend_prior: bool = True
    trend_prior_data_path: str | None = None


settings = Settings()
