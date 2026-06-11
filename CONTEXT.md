# ai_clipping — backend

The clip-generation backend: a FastAPI pipeline that turns a long-form Twitch VOD into ranked short-form clips, plus the offline ML/eval code that builds training labels for it.

## Language

Deutsch

### Clip pipeline

**Conductor**:
The thin orchestrator of a single job's run (`run_pipeline`). Owns job-status transitions, progress percentages, cancellation checks, the chat prefetch, and resource teardown. Holds no clip-finding logic itself — it sequences Producers.
_Avoid_: orchestrator, manager, runner.

**Producer**:
A deep module that turns its inputs into one pipeline output behind a small interface (transcript, voice-energy annotation, chat enrichment, moments, clips). Pure: it never reads global settings or touches job state, and its leaf operations (ffmpeg, Whisper, the LLM) are injected so it is tested in isolation.
_Avoid_: stage, step, handler.

**Resource**:
A Producer that is lazy and memoized rather than fire-once — it yields its value at most once even when several Producers pull it. `AudioResource` (the 16 kHz wav) is the canonical one: transcription and voice-energy both pull it, audio is extracted at most once, and the Conductor tears it down.
_Avoid_: provider, cache (for this role).

**Cache source**:
The stable identity of a job's media, used as the cache key — e.g. `twitch_vod:123`, `url:…`, `path:…`. `None` means caching is disabled for that job.
_Avoid_: cache key, source id.

**Moment**:
A scored, time-bounded span of a VOD proposed for clipping (start/end + viral/composite scores + reason). Produced by the moment finder; the top-K Moments become clips.
_Avoid_: highlight, segment.

**RichSegment**:
The enriched transcript unit (text + start/end + per-signal scores such as voice-energy burst and chat spike). The shared timeline currency the clip-finding Producers annotate.
_Avoid_: segment (unqualified), chunk.
