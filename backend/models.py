import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator


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


class JobRequest(BaseModel):
    url: str
    profile: str | None = None   # None → falls back to settings.clip_profile

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
    ranking_mode: Literal["heuristic", "ml_blended"] = "heuristic"


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


ClipFeedbackTag = Literal[
    "strong_hook",
    "too_much_context",
    "missing_payoff",
    "too_long",
]


class ClipFeedbackPayload(BaseModel):
    candidate_id: str
    tags: list[ClipFeedbackTag] = Field(default_factory=list)
    notes: str | None = None
    view_count: int | None = Field(default=None, ge=0)
    like_count: int | None = Field(default=None, ge=0)
    comment_count: int | None = Field(default=None, ge=0)
    share_count: int | None = Field(default=None, ge=0)
    save_count: int | None = Field(default=None, ge=0)
    views_1h: int | None = Field(default=None, ge=0)
    views_24h: int | None = Field(default=None, ge=0)
    baseline_view_rate: float | None = Field(default=None, ge=0.0)
    observed_view_rate: float | None = Field(default=None, ge=0.0)


class ClipFeedbackResponse(BaseModel):
    job_id: str
    candidate_id: str
    tags: list[ClipFeedbackTag] = Field(default_factory=list)
    notes: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    share_count: int | None = None
    save_count: int | None = None
    views_1h: int | None = None
    views_24h: int | None = None
    baseline_view_rate: float | None = None
    observed_view_rate: float | None = None
