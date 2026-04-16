import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


JobState = Literal[
    "queued",
    "downloading",
    "extracting_audio",
    "transcribing",
    "extracting_features",
    "diarizing",
    "analyzing",
    "clipping",
    "done",
    "cancelled",
    "failed",
]

# Supported URL patterns — yt-dlp handles the actual download; this is a
# lightweight pre-check to catch clearly unsupported inputs early.
_SUPPORTED_URL_PATTERNS = [
    re.compile(r"https?://(www\.)?twitch\.tv/videos/\d+"),           # Twitch VOD
    re.compile(r"https?://(www\.)?youtube\.com/watch"),               # YouTube watch
    re.compile(r"https?://youtu\.be/"),                               # YouTube short link
    re.compile(r"https?://(www\.)?youtube\.com/shorts/"),             # YouTube Shorts
    re.compile(r"https?://kick\.com/"),                               # Kick
]

ClipPlatform = Literal["general", "tiktok", "youtube_shorts", "instagram_reels", "linkedin"]
ClipStyle = Literal["balanced", "aggressive_hook", "context_balanced", "reaction_first", "quote_first"]


class TimeRange(BaseModel):
    start_sec: float | None = None
    end_sec: float | None = None

    @model_validator(mode="after")
    def _validate_order(self):
        if self.start_sec is not None and self.end_sec is not None and self.end_sec <= self.start_sec:
            raise ValueError("time_range.end_sec must be greater than time_range.start_sec")
        return self


class JobRequest(BaseModel):
    url: str
    profile: str | None = None   # None → falls back to settings.clip_profile
    session_id: str | None = None
    platform: ClipPlatform | None = None
    goal: str | None = None
    tone: str | None = None
    topic: str | None = None
    must_include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    speaker: str | None = None
    time_range: TimeRange | None = None
    clip_style: ClipStyle | None = None
    variant_count: int | None = Field(default=None, ge=1, le=4)

    @field_validator("url")
    @classmethod
    def must_be_supported_url(cls, v: str) -> str:
        if not any(p.match(v) for p in _SUPPORTED_URL_PATTERNS):
            raise ValueError(
                "URL not supported. Accepted: Twitch VOD, YouTube, Kick."
            )
        return v


class ClipResult(BaseModel):
    candidate_id: str
    filename: str
    download_url: str
    start_time: float
    end_time: float
    viral_score: float
    reason: str
    selection_reasons: list[str] = Field(default_factory=list)
    ranking_mode: Literal["heuristic", "ml_blended"] = "heuristic"
    platform: ClipPlatform = "general"
    variant_key: str = "balanced"
    variant_label: str | None = None
    projected_hook_score: float | None = None
    projected_retention_score: float | None = None


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    progress_pct: int = 0
    profile: str = "general"     # which ranker profile was used; default handles legacy jobs.json entries
    clips: list[ClipResult] = Field(default_factory=list)
    error: str | None = None
    download_speed: float | None = None   # bytes/s
    download_eta: int | None = None       # seconds remaining
    download_cached: bool = False

    @field_validator("download_eta", mode="before")
    @classmethod
    def coerce_download_eta(cls, value):
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return max(int(round(float(value))), 0)
        return value


ClipFeedbackRating = Literal["good", "medium", "bad"]


class ClipFeedbackPayload(BaseModel):
    candidate_id: str
    rating: ClipFeedbackRating
    notes: str | None = None


class ClipFeedbackResponse(BaseModel):
    job_id: str
    candidate_id: str
    rating: ClipFeedbackRating | None = None
    notes: str | None = None
