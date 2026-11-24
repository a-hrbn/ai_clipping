# Clip pipeline — Producers behind a thin Conductor

Interface sketch for decomposing `run_pipeline` (pipeline.py:376–649) into deep
**Producers** sequenced by a thin **Conductor**. Vocabulary: see `/CONTEXT.md`
(Conductor, Producer, Resource, Cache source, Moment, RichSegment).

**This is a design artifact, not an implementation.** Method bodies are `...`.
Decisions below are locked unless flagged _(reversible)_.

## Locked decisions

1. **Deep Producers, thin Conductor.** Each Producer turns inputs → one output behind a small interface.
2. **Scope: drain `pipeline.py` only.** `transcriber.py` is untouched; its transcript/voice-energy/chat caches stay sealed and are reached through injected callables.
3. **Producers are pure.** No `update_job`, no `job_manager`, never read `config.settings`. Everything injected.
4. **Leaf ops are injected ports** (plain callables/Protocols), placed only where two adapters justify them (real + fake). Hypothetical seams are rejected (see Ports table).
5. **Progress flows out as a fraction** `0.0..1.0`; the Conductor maps it to absolute %.
6. **`AudioResource` is a memoized Resource** — extract at most once across consumers; the Conductor tears it down.
7. **Producers own their debug artifacts** (e.g. MomentFinding writes its own `stage.json` running/ok/failed).
8. **Cancellation is the Conductor's job**, checked _between_ Producers; cancel → no `failed` status.

## Package layout

```
backend/producers/
  __init__.py
  seam.py        # Progress type, ProducerError, leaf-op type aliases
  audio.py       # AudioResource (Resource)
  transcript.py  # TranscriptResource
  energy.py      # VoiceEnergy
  chat.py        # ChatEnrichment
  moments.py     # MomentFinding
  clipping.py    # Clipping
  adapters.py    # build_real_* factories wiring the untouched services
backend/pipeline.py   # becomes the Conductor (run_pipeline + run() helper)
```

---

## Seam currency (`producers/seam.py`)

```python
from typing import Callable, Protocol
import pathlib
from services.transcriber import RichSegment
from services.moment_finder import MomentFinderResult
from models import ClipResult

# Producer-local progress. The Producer reports fraction-of-its-own-work;
# the Conductor maps it onto an absolute % band. Monotonic, optional to call.
Progress = Callable[[float], None]

class ProducerError(RuntimeError):
    """Base for failures a Producer raises. The Conductor maps these to a
    `failed` job status. Cancellation is NOT a ProducerError (see Conductor)."""

# Leaf-op ports are plain callables (no ABCs). Defaults wire the real adapter
# in adapters.py; tests pass fakes positionally/by-kwarg.
ExtractFn      = Callable[[pathlib.Path, str], pathlib.Path | None]          # audio_extractor.extract
TranscribeFn   = Callable[..., list[RichSegment]]                            # transcriber.transcribe_with_progress
LoadCachedFn   = Callable[..., list[RichSegment] | None]                     # transcriber.load_cached_transcript
ProbeFn        = Callable[[pathlib.Path], float | None]                      # ffprobe duration
ComputeEnergyFn= Callable[[pathlib.Path, list[RichSegment]], list[float]]    # audio_energy.compute_segment_energy
ChatSourceFn   = Callable[[], pathlib.Path | None]                           # blocks on prefetch OR downloads
EnrichFn       = Callable[[list[RichSegment], pathlib.Path], list[RichSegment]]  # chat_signal.enrich_segments
FindMomentsFn  = Callable[..., MomentFinderResult]                           # moment_finder.find_moments
CutClipsFn     = Callable[..., list[ClipResult]]                             # clipper.cut_clips

class AudioCache(Protocol):
    """The audio cache drained from pipeline.py:178–341. Two adapters justify
    the port: FsAudioCache (prod) + InMemoryAudioCache/NullAudioCache (tests)."""
    def load(self, cache_source: str | None, *, video_path: pathlib.Path,
             job_id: str) -> pathlib.Path | None: ...
    def store(self, cache_source: str | None, *, video_path: pathlib.Path,
              audio_path: pathlib.Path) -> None: ...
```

---

## Ports table (two-adapter discipline)

| Port | Prod adapter | Test fake | Category | Verdict |
|---|---|---|---|---|
| `ExtractFn` | `audio_extractor.extract` | call-counting `FakeExtract` | local-tool | **keep** — `extract.calls == 1` is the load-bearing assertion |
| `AudioCache` | `FsAudioCache` (:178–341) | `InMemoryAudioCache` / `NullAudioCache` | local-substitutable | **keep** — enables stale-metadata + cache-hit tests, zero disk |
| `TranscribeFn` | `transcriber.transcribe_with_progress` | canned-segments fake | local-tool | **keep** — loading Whisper in unit tests is intolerable |
| `LoadCachedFn` | `transcriber.load_cached_transcript` | `lambda …: SEGS` / `None` | local-substitutable | **keep** — one-line hit/miss toggle |
| `ProbeFn` | ffprobe (`_probe_media_duration_sec`) | `lambda _p: 120.0` | local-tool | **keep, folded** into TranscriptResource (not its own Producer) |
| `ComputeEnergyFn` | `audio_energy.compute_segment_energy` | canned floats | local-tool | **keep** |
| `ChatSourceFn` | prefetch `future.result()` _or_ `chat_signal.download_chat` | `lambda: fixture_path` | true-external | **keep** |
| `FindMomentsFn` | `moment_finder.find_moments` | canned `MomentFinderResult` | true-external | **keep** |
| `CutClipsFn` | `clipper.cut_clips` | canned `ClipResult`s | local-tool | **keep** |
| ~~`SettingsPort`~~ | — | — | — | **rejected** — Conductor reads `settings` once, injects plain values |
| ~~god `CacheStore`~~ | — | — | — | **rejected** — audio/transcript/debug have different keys + lifecycles |
| ~~`ClockPort`/`UuidPort`~~ | — | — | — | **rejected** — buried in `FsAudioCache`, no Producer observes it |

---

## AudioResource (`producers/audio.py`)

Absorbs pipeline.py:178–341 (the entire audio-cache machinery) + the lazy
extract. The keystone — its memoization is what dissolves "audio pulled by two
consumers."

```python
class AudioResource:
    """Lazy, memoized 16 kHz wav for one job. At most ONE extraction across all
    consumers (TranscriptResource + VoiceEnergy share one instance)."""

    def __init__(self, *, video_path: pathlib.Path, job_id: str,
                 cache_source: str | None,
                 extract: ExtractFn, cache: AudioCache) -> None: ...

    def path(self) -> pathlib.Path | None:
        """Resolve the wav, extracting on first call only.

        Ordering:  cache.load (hit → return)  →  extract  →  cache.store.
        Invariants:
          - Idempotent: N calls → at most ONE `extract` call (memoized).
          - Lazy: never called by a transcript cache-hit run → zero extraction.
          - cache_source is None → skip cache, always extract once.
        Errors: `extract` failure propagates. cache.load returns None (never
          raises) on stale/corrupt entries → falls through to extraction."""
        ...

    def teardown(self) -> None:
        """Delete the wav + its chunks/ dir if materialized. Conductor-only;
        Producers never tear down. Idempotent."""
        ...

    def materialized(self) -> pathlib.Path | None:
        """Read-only peek — was audio actually extracted? Lets the Conductor
        clean up without forcing extraction. None == never materialized."""
        ...
```

**Hidden behind the seam:** digest keying, metadata staleness match, the
44-byte empty-wav guard, atomic tmp-replace write, hardlink-or-copy fallback —
all inside `FsAudioCache`; the extract-once memo + the materialized-vs-not
decision inside `AudioResource`.

---

## TranscriptResource (`producers/transcript.py`)

Absorbs pipeline.py:389–469. **Folds the duration probe in** (fixes the
"silent cache no-op" footgun — a missing duration is handled here, not by a
Conductor ordering rule).

```python
class TranscriptResource:
    def __init__(self, *, audio: AudioResource, cache_source: str | None,
                 video_path: pathlib.Path,
                 isolate_chunks: bool, enable_diarization: bool,
                 load_cached: LoadCachedFn, transcribe: TranscribeFn,
                 probe: ProbeFn) -> None: ...

    def segments(self, *, progress: Progress = lambda _f: None) -> list[RichSegment]:
        """Resolve the transcript.

        Ordering (cheapest-first):
          1. cache_source set → probe duration → load_cached(...). Hit returns;
             **AudioResource is never pulled** (so no extraction).
          2. Miss → audio.path() → transcribe(audio_path, cache_key, …, progress).
        Invariants:
          - Idempotent (memoized).
          - Cache hit ⇒ zero audio extraction (the locality win).
          - progress is the chunk callback remapped to 0..1 internally.
        Errors: transcribe failure → raise (wrapped as ProducerError at the
          boundary). load_cached → None means MISS, not error."""
        ...

    def from_cache(self) -> bool: ...
```

The `transcribe`/`load_cached` adapters wrap the untouched
`transcriber.transcribe_with_progress(audio_path, progress_cb, cache_key,
isolate_chunks, enable_diarization)` and
`transcriber.load_cached_transcript(cache_source, expected_duration_sec,
progress_cb)`. The `(done,total,chunk)`→fraction remap lives in the adapter.

---

## VoiceEnergy (`producers/energy.py`)

Absorbs pipeline.py:478–508 (incl. the RichSegment rebuild that stamps
`voice_energy_burst_score`). Pulls the **same** `AudioResource` instance → no
second extraction.

```python
class VoiceEnergy:
    def __init__(self, *, audio: AudioResource, cache_source: str | None,
                 load: LoadVoiceEnergyFn, save: SaveVoiceEnergyFn,
                 compute: ComputeEnergyFn) -> None: ...

    def apply(self, transcript: list[RichSegment]) -> list[RichSegment]:
        """load cache → (miss) audio.path() + compute → save → return a new
        transcript with voice_energy_burst_score stamped per segment.
        Invariant: when load misses but audio is unavailable, return the
        transcript unchanged (matches today's no-audio warning path)."""
        ...
```

---

## ChatEnrichment (`producers/chat.py`)

Absorbs pipeline.py:510–534. **The prefetch stays in the Conductor**; this
Producer receives a `chat_source` callable that either blocks on the prefetched
future or downloads — it never knows a thread/future exists. _(reversible: a
callable was chosen over passing a `Future` so the Producer has no asyncio/
threading knowledge.)_

```python
class ChatEnrichment:
    def __init__(self, *, cache_source: str | None,
                 chat_source: ChatSourceFn,
                 apply_cached: ApplyChatEnrichmentFn,
                 enrich: EnrichFn, save: SaveChatEnrichmentFn) -> None: ...

    def apply(self, transcript: list[RichSegment]) -> list[RichSegment]:
        """1. apply_cached(cache_source) hit → return enriched (and signal the
              Conductor it may cancel the prefetch).
           2. Miss → path = chat_source(); if path → enrich + save.
           Invariant: chat_source() is called at most once. No path → return
           the transcript unchanged."""
        ...
```

---

## MomentFinding (`producers/moments.py`)

Absorbs pipeline.py:536–586. **Owns the `stage.json` running/ok/failed contract**
and the debug dumps (prompt/response/usage/moments.jsonl) the routers read.

```python
class MomentFinding:
    def __init__(self, *, profile: str, vod_id: str, debug_dir: pathlib.Path,
                 params: "MomentParams", find: FindMomentsFn) -> None: ...

    def find(self, transcript: list[RichSegment],
             *, progress: Progress = lambda _f: None) -> MomentFinderResult:
        """Writes stage.json="running" on entry, ="ok" + summary on success,
        ="failed" + error in its except BEFORE re-raising. Also dumps
        prompt/response/usage/moments(.raw).jsonl.
        Errors: empty ranked_candidates → raise MomentFinderError (a
        ProducerError) after the failed marker is written."""
        ...
```

`MomentParams` is a frozen snapshot the Conductor builds from the ~17
`moment_finder_*` settings — so this Producer never reads `config.settings`.

---

## Clipping (`producers/clipping.py`)

Absorbs pipeline.py:353–372 + 594–627 (`_moments_from_post_processed`, top-K,
`final_moments.json`). The `segments.jsonl` write stays in the Conductor (it's
not clip-specific); the repo-level `transcripts/` dumps are gone (#118).

```python
class Clipping:
    def __init__(self, *, job_id: str, debug_dir: pathlib.Path,
                 top_k: int, platform: str, cut: CutClipsFn) -> None: ...

    def run(self, video_path: pathlib.Path, result: MomentFinderResult,
            *, progress: Progress = lambda _f: None) -> list[ClipResult]:
        """post-process → top-K select → write final_moments.json → cut_clips.
        progress maps cut_clips' (so_far, done, total) callback to 0..1.
        Note: today's partial-clips streaming (update_job(clips=so_far)) is the
        Conductor's concern — see Conductor."""
        ...
```

---

## Conductor (`pipeline.py`, slim `run_pipeline`)

The only place that reads `settings`, calls `update_job`, checks cancellation,
runs the chat prefetch, and tears down Resources. The `run()` helper maps a
Producer's `0..1` onto an absolute % band and offloads to the executor.

```python
async def run_pipeline(job_id, url=None, video_path=None, profile=None, strategy=None) -> None:
    cfg = ProducerConfig.from_settings(settings, profile, strategy)  # the ONLY settings read
    loop = asyncio.get_running_loop()
    cache_source = resolve_cache_source(job_id, url)                 # stays in Conductor (tested)
    debug_dir = job_dir(job_id) / "debug"; debug_dir.mkdir(parents=True, exist_ok=True)
    chat = maybe_prefetch_chat(cfg, url, job_id, debug_dir)         # daemon thread, unchanged

    audio = None
    try:
        _raise_if_cancelled(job_id)

        # progress band: producer fraction → absolute %, status owned here
        async def run(call, *, status, band):                      # band = (lo, hi)
            _raise_if_cancelled(job_id)                            # cancel BETWEEN producers
            def on(frac, s=status, b=band):
                update_job(job_id, status=s, progress_pct=b[0] + int(frac*(b[1]-b[0])))
            on(0.0)
            return await loop.run_in_executor(None, lambda: call(progress=on))

        if video_path is None:
            update_job(job_id, status="downloading", progress_pct=5)
            video_path = await loop.run_in_executor(None, downloader.download, url, job_id)
        else:
            update_job(job_id, status="downloading", progress_pct=19)

        audio = AudioResource(video_path=video_path, job_id=job_id, cache_source=cache_source,
                              extract=audio_extractor.extract, cache=FsAudioCache(settings.storage_dir))
        tr = TranscriptResource(audio=audio, cache_source=cache_source, video_path=video_path,
                                isolate_chunks=cfg.isolate_chunks, enable_diarization=cfg.diarization,
                                load_cached=transcriber.load_cached_transcript,
                                transcribe=transcriber.transcribe_with_progress,
                                probe=_probe_media_duration_sec)

        transcript = await run(lambda *, progress: tr.segments(progress=progress),
                               status="transcribing", band=(30, 55))
        transcript = await run(lambda *, progress: VoiceEnergy(audio=audio, cache_source=cache_source, …).apply(transcript),
                               status="extracting_audio", band=(56, 60))
        if cfg.enable_chat_analysis and url:
            transcript = await run(lambda *, progress: ChatEnrichment(cache_source=cache_source, chat_source=chat_src(chat), …).apply(transcript),
                                   status="downloading_chat", band=(60, 72))

        store.write_segments(job_id, _segment_records(transcript)) # segments only, via the Job-Artifact-Store
        result = await run(lambda *, progress: MomentFinding(profile=cfg.profile, vod_id=cfg.vod_id(url, job_id),
                                                             debug_dir=debug_dir, params=cfg.moment_params,
                                                             find=moment_finder.find_moments).find(transcript, progress=progress),
                           status="analyzing", band=(73, 87))

        clips = await run(lambda *, progress: Clipping(job_id=job_id, debug_dir=debug_dir, top_k=cfg.top_k,
                                                       platform=cfg.platform, cut=clipper.cut_clips).run(video_path, result, progress=progress),
                          status="clipping", band=(88, 99))

        audio.teardown(); audio = None
        update_job(job_id, status="done", progress_pct=100, clips=clips)
        # no timed cleanup: the job folder goes when the registry evicts the job (#117)

    except (PipelineCancelledError, downloader.DownloadCancelledError):
        logger.info("Pipeline cancelled for job %s", job_id)        # NO failed status
    except Exception as e:
        logger.exception("Pipeline failed for job %s", job_id)
        update_job(job_id, status="failed", error=str(e))
    finally:
        if audio is not None: audio.teardown()
        # source media (video/audio/chunks) is dropped by the serving router after the run (store.delete_source_media)
```

### Progress-band map (preserves current UX)

| Step | status | band |
|---|---|---|
| download | `downloading` | 5 (url) / 19 (path) |
| transcript | `transcribing` | 30–55 |
| voice energy | `extracting_audio` | 56–60 |
| chat | `downloading_chat` | 60–72 |
| moments | `analyzing` | 73–87 |
| clipping | `clipping` | 88–99 |
| done | `done` | 100 |

_Cosmetic change (accepted):_ audio extraction on a transcript cache-miss now
shows under the `transcribing` band rather than a separate `extracting_audio`
emit, because the extract decision is lazy inside the Producer. _(reversible)_

### Error / status taxonomy at the Conductor seam

- `PipelineCancelledError` / `downloader.DownloadCancelledError` → log only, **no status change**.
- Any other exception (incl. `ProducerError` subclasses like `MomentFinderError`) → `update_job(status="failed", error=…)`.
- A Producer that fails writes its **own** stage marker first (MomentFinding's `stage.json="failed"`) then raises; the Conductor only sets the job status.

---

## Test surface (the interface is the test surface)

**Newly possible unit tests (don't exist today — were only reachable via full `run_pipeline`):**
- `AudioResource`: extract-at-most-once (`assert extract.calls == 1` across two `path()` pulls); cache-hit → `extract.calls == 0`; **stale-metadata** → miss → re-extract.
- `TranscriptResource`: cache hit → audio never pulled (extract tripwire raises if touched); miss → audio + transcribe.
- `MomentFinding`: empty `ranked_candidates` → `MomentFinderError` **and** `stage.json["status"] == "failed"`.
- `Clipping`: top-K truncation + `final_moments.json` shape with a fake `cut`.

**Survive unchanged (stay in the Conductor):**
- `_cleanup_serving_source_media` tests (test_pipeline_integration.py:149–192).
- Cache-source tests (`_transcript_cache_source`, `_frontend_cache_enabled_for_job`, lines 194–224).

**Get thinner / rewritten:**
- The lean-path + debug-job integration tests (42–147) shrink — patch the Producers (or their injected leaf ops) instead of 4 module globals + `update_job`.
- Per DEEPENING.md: the shallow module-global monkeypatch tests are **deleted** once the per-Producer interface tests exist.

## Open branches (deferred, not blocking)

- **DebugSink as a port** (D's idea): inject a writer so artifact-contract tests assert in-memory instead of reading a tmp dir. Nice-to-have; Producers write to `debug_dir` directly for now.
- **AnnotationPass** (B's idea): if signals grow past voice-energy + chat to ~4–5, fold them into a registered annotation list. Not now.
- **Weighted progress spans** vs the fixed bands above. Fixed bands chosen to preserve current UX exactly.
