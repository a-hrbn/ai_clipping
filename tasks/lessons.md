# Lessons

- When rebuilding a repost manifest for source freshness, carry forward usable
  media paths from the previous manifest by canonical repost identity. A fresh
  discovery row usually lacks `video_path`, and exact URL-keyed caches can miss
  equivalent YouTube URLs such as `/shorts/<id>` versus `watch?v=<id>`.
- When overlapping VOD audio download and Whisper transcription, queue the
  entire missing-audio set onto the single download worker up front. A one-item
  lookahead can still idle whenever the current Whisper job runs longer than
  the next audio download.
- For full-VOD transcript backfills, avoid a strictly serial
  download-then-transcribe loop. Keep Whisper/GPU work single-threaded, but
  prefetch the next VOD audio with one download worker while the current VOD is
  being transcribed.
- Transcript-based cross-platform matching must proactively build the VOD-side
  transcript cache for the exact VOD manifest before matching. Do not rely on
  the matcher to skip missing VOD transcripts, because that silently reduces
  coverage and leaves good repost labels unmatched.
- When the user narrows cross-platform verification to one primary signal,
  implement that narrowed path directly. For weak audio-fingerprint matches,
  demote audio to an explicit legacy/diagnostic mode instead of keeping it as
  the default positive-label source.
- When switching a curated repost channel, clear both the short media cache and
  the short metadata cache plus derived cross-platform repost/output artifacts.
  Keep VOD audio fingerprints, because those are source-channel independent.
- If exact date filtering is more important than minimizing YouTube requests,
  restore cached per-Short metadata enrichment instead of relying only on flat
  channel discovery. Cache successful metadata by URL and pace cache misses so
  the next run has exact dates without repeating the same yt-dlp requests.
- Flat YouTube channel/Shorts discovery can return many valid URLs without
  upload dates. If per-Short metadata is disabled, recover dates from the
  channel page's per-video publish-time blocks and still fail closed for rows
  whose date cannot be recovered.
- When the user asks for pacing after every yt-dlp Short download request,
  native yt-dlp request sleeps are not enough. Put an explicit post-attempt
  delay in the serial repost download loop so success and failure outcomes both
  pause before the next Short is attempted.
- When reducing YouTube metadata requests for repost discovery, keep temporal
  safety fail-closed: only use upload/publish dates already present in
  discovery or manifest rows, and skip missing-date rows instead of enriching
  them with extra detail calls.
- Keep yt-dlp error cooldowns aligned with the normal request pacing unless
  the user explicitly asks for a long backoff. A one-hour cooldown can make a
  training run look stuck even when the error handler is working.
- yt-dlp errors in the repost pipeline can happen before final media download,
  especially during source discovery or metadata extraction. Apply requested
  cooldown behavior to every short-form yt-dlp exception path, not only the
  last cache materialization loop.
- For repost Short downloads, a user-facing cooldown requirement means pause
  before moving to the next URL, not retrying the same failed URL. Keep retry
  semantics explicit and only add them when requested.
- yt-dlp native sleep options do not guarantee a pause when YouTube rejects the
  first request with "This content isn't available, try again later"; handle
  that exception in the serial repost download loop with an explicit cooldown
  before continuing to the next URL.
- Search query membership is not label evidence. Cross-platform repost
  discovery must persist title/uploader/channel metadata and require explicit
  streamer evidence for search-discovered rows; tags inherited from the query
  source are not proof that a clip belongs to that streamer.
- After tightening a source-quality filter, recalibrate source-count guards
  against the filtered pool. Do not let an old pre-filter `min_reposts` block
  the matcher before the stronger matched-row/VOD-diversity gates can run.
- Cross-platform repost discovery should be temporally scoped to the actually
  usable VOD fingerprint pool, not just the requested training dates. Preserve
  VOD `created_at` during inventory cache merges and filter reposts by their
  publish/upload date before matching.
- Once repost sources are profile-filtered and date-window scoped, raw
  pre-match repost count is no longer a useful rollout gate. Let matching run
  and gate on matched rows plus matched VOD diversity instead.
- Date-scoped repost discovery needs freshness-aware source ordering and
  metadata fallbacks. Add date-sorted search sources for narrow VOD windows and
  keep search-result RSS rows when detail extraction fails, as long as the row
  still passes explicit profile/date filters downstream.
- For time-scoped repost labels, do not cap by raw discovery order before
  profile/date filtering. Filter to the actual VOD window first, then rank the
  surviving candidates globally by views and score before applying the cap.
- Do not assume yt-dlp supports search pseudo-schemes beyond the installed
  version. If adding source strings such as `ytsearchdateN:<query>`, implement
  an internal handler or verify the extractor before using it in production
  manifests.
- Discovery-code fixes do not affect a run that reuses an existing repost
  manifest. When a fix changes which source rows should be discovered or
  admitted, bump the persisted repost policy version so the next run rebuilds
  the manifest.
- Search APIs can be sparse or unstable per platform. If TikTok RSS discovery
  returns zero rows, add a second extraction route such as HTML result parsing
  before concluding the platform has no labels in the date window.
- Coverage guards for derived label artifacts must fail closed without deleting
  the last valid artifact. A precondition like "too few reposts discovered"
  should preserve existing outcome rows and let later reruns rematch once source
  coverage improves.
- Do not set a pre-match source-count guard above the currently discoverable
  source pool. If `max possible matches <= usable source rows`, the source
  guard must leave room for matching and the stricter quality guard should be
  matched rows / matched VOD diversity.
- Multi-stage training scripts must write fail-prone dataset builds to a
  staging path and promote only after the stage succeeds; failure diagnostics
  should inspect the staged output so canonical artifacts are not replaced by
  partial runs.
- When a workflow builds a derived outcome-train dataset, verify the primary
  trainer actually consumes that dataset and that its split unit matches the
  trainer's holdout exclusion unit; otherwise good labels can be built and then
  silently ignored or filtered away.
- If a long training run fails on a warning/logging path, add module-level
  logger initialization alongside the first `logger.*` use and smoke-test that
  module before relaunching the run.
- When a diagnostic artifact recommends a concrete calibrated threshold and the
  user accepts that recommendation, update the production defaults that drive
  the next training run, not only the diagnostic/reporting code.
- A cross-platform matching pipeline is not fixed when it produces zero rows
  silently; persist below-threshold best-match score distributions so threshold
  problems can be separated from broken signal problems.
- Manifest artifacts that cross host/container boundaries must not store
  environment-specific absolute paths without a repair step; validate media
  paths in the runtime where they are consumed and rebuild stale manifests
  before matching.
- Do not call a data-pipeline bug fixed when the code only reports missing
  input manifests; if the project already has discoverable/cached upstream
  sources, the production workflow should build those manifests automatically.
- Cross-platform supervision is not complete when the trainer can consume a
  JSONL path; the normal training orchestrator must generate or validate that
  JSONL from configured manifests and write a blocked/missing summary when the
  source data is absent.
- When adding new row features, verify the trained artifact accounts for every
  numeric feature as either used or explicitly excluded; deriving the feature
  vector from one sample row can silently drop sparse/new modalities.
- Optional gated ML models must fail open in training paths: log auth/access/download failures and continue with degraded features instead of aborting the full run.
- When adding code that calls optional ML dependencies, verify the installed package's actual function signature before launching long training runs, especially when `pyproject.toml` pins a newer major version than the older API examples.
- When replacing a third-party API path, verify the exact production request shape and log upstream error bodies; a one-VOD happy path is not enough if training sends many offsets through a containerized runtime.
- Do not equate proxy distribution metrics like views or engagement with intrinsic clip quality. Treat them as noisy supervision that must be combined with exposure context, creator baseline, and stronger outcome signals.
- When the user asks for a simpler feedback workflow, optimize for the minimum number of stable inputs instead of preserving analytics-heavy schemas.
- In training collection pipelines, transient network failures must pause and retry the current item instead of being downgraded to skipped/unavailable examples.
- Keep infrastructure extensions reversible and scoped; if the user later rejects an architectural add-on, revert only that slice without disturbing the surrounding workflow.
- When fixing Codex skill installs, verify both discovery and the real user invocation path; a correct `~/.codex/skills/.../SKILL.md` install does not prove the client will accept a manually typed `/trigger` in the UI.
- Do not leave downloaded SDKs or local toolchains inside the project tree; move them to a user-level path before relying on repo-wide tooling or graph scans.
- For Vertex jobs, do not call a `CANCELLED` run manually stopped without checking `startTime`, `endTime`, `error`, and the job timeout first; a timeout-triggered cancellation can look identical at the status level.
- In streamer-specific training workflows, do not pass a profile flag into collection code if that profile expands to a broadcaster pool; use the explicit broadcaster-only path for single-streamer reruns.
- When choosing Vertex machine types, do not keep recommending `highmem` from past incidents once current telemetry shows a CPU bottleneck and low RAM usage; pick a CPU-oriented family that matches the observed bottleneck.
- If the user reports that the `highmem` machine class behaved better in practice for the current training workload, prefer that observed outcome over a theoretical CPU-bottleneck interpretation and switch the run back accordingly.
- When a runtime decoder error persists after an environment-level tweak, do not assume the library honors that setting; verify it against a real run and move to a robust codec-specific fallback instead of iterating on unproven env flags.
- Before submitting a Vertex training job for freshly changed training code, build and push the new container image first and verify the exact Artifact Registry tag; do not reuse the previous tag by assumption.
- When Vertex custom-job logs appear in Google Cloud Console but `gcloud logging read` returns nothing, do not assume there are no logs; first broaden the query and verify the monitored resource type and labels, because runtime logs may be under `resource.type="ml_job"` and `logName=".../workerpool0-0"` rather than `aiplatform.googleapis.com/CustomJob`.
- Before trusting a local Ollama benchmark on this Mac, check for stale benchmark processes and loaded model runners with `ps` and `ollama ps`; do not compare results while an older model run is still active in the background.
- When a user wants the real root cause fixed, do not mask a concurrency bug by silently skipping malformed model outputs; serialize the unsafe shared inference path first, and keep malformed payloads visible via the existing higher-level error logging.
- If the user explicitly asks for a `gcloud` run, do not switch to a local fallback workflow without first proving the `gcloud` path is unavailable and calling that constraint out plainly.
- When promoting a new default local training image, make the replacement image use the standard `ai-clipping-train-local:latest` tag instead of leaving it on a separate variant tag.
- When a multi-stage training workflow produces a useful intermediate artifact before the final quality gates fail, persist that intermediate artifact to the standard artifact directory instead of leaving it only in a temporary side path.
- When migrating cache key schemes, do not only write the new canonical cache entry; also prune compatible superseded entries so the cache does not accumulate duplicates that make later inspection ambiguous.
- Before adding automatic cache cleanup, verify whether legacy work directories contain the only local copy of raw media; keep or migrate those directories instead of treating name-patterned job folders as safely regenerable.
- When the user reviews false negatives and identifies label errors, propagate corrections by exact `queue_id` across gold artifacts, and keep true positives as positives with explicit rationale notes instead of changing all rows from the same VOD.
- For frontend gold-eval corrections, update the append-only `backend/tmp/storage/feedback/gold_eval_updates.jsonl` source snapshots, not only generated `backend/ml/artifacts/*` files; otherwise the next training export will overwrite the correction.
- When the user revises a previous false-negative rationale, treat the latest review as authoritative and append a newer source snapshot even if an earlier correction note said the opposite.
- Do not store local LLM binaries or model weights under `backend/tmp`; Ollama must run as a separate service with model data in a Docker/user-level volume so repo temp cleanup and Graphify scans stay clean.
- If the user rejects an experimental supervision tier, remove the ingest, merge, tests, and generated artifacts for that tier instead of leaving dormant flags that can re-enable it later.
- Extended candidate collection must not write broad proxy labels into the primary training label file; collect to a separate pool and merge only explicitly selected/deduped candidates.
- Treat externally sourced Twitch Top-Viewed VODs as volatile: if a scoped VOD range no longer exists on Twitch, skip that optional candidate/VOD and continue training instead of aborting.
- Do not call a modality improvement implemented if it only adds feature slots or fail-open placeholders. For critique-driven plans, each item needs a real producer, a consumer in serving/training, and a regression test that proves non-empty signal flow.
- A standalone tool is not a completed data-pipeline feature until the main training orchestrator discovers or consumes its output automatically and reports non-empty coverage.
- When a Jynxzi pipeline critique identifies target mismatch, prioritize true cross-platform outcome labels, model/feature-cap defaults, and serving/training feature parity before adding more heuristic features.
- When turning an LLM judge into a teacher signal, verify both sides of the
  loop: runtime feature fields must be persisted and the trainer must replay or
  otherwise populate those fields in normal training rows, not only use the
  judge score as a proxy label.
- Do not imply that the VOD training script calls the serving Clip-Judge unless
  the code path actually invokes `pipeline.run_pipeline` or
  `clip_judge.rerank_candidates`; training can replay existing judge snapshots,
  but snapshot creation needs a separate serving/backfill pass.
- For streamer-specific backfill wrappers, pass an explicit broadcaster and
  make that broadcaster override any profile-based pool expansion. A Jynxzi
  backfill wrapper must not inherit the broader `jynxzi,gaming` Twitch pool.
- Native CV stacks can segfault before Python can catch an exception. For
  headless backfill scripts, set disable/env guards before importing
  `pipeline.py` or `services.video_features`; do not rely on runtime
  try/except around the pipeline call.
- Old date-range VOD IDs recovered from label manifests may point at deleted
  Twitch videos. Before sending them through a serving backfill, resolve
  existing scoped-window media from the local VOD/cache manifests and pass
  `video_path` directly so the run does not depend on Twitch archive retention.
- Outcome holdout feature replay must use a stable feature source that still
  contains holdout candidates. Never rebuild holdout labels from a train-only
  dataset after VOD-level holdout exclusion, or positives can disappear while
  the downstream eval silently falls back to another source.
- When outcome rows wrap replayed candidate features under a new row identity,
  calibration and eval must score by the replayed feature `candidate_id`, not
  only the outer row `candidate_id`; otherwise positives can silently receive
  default scores and threshold selection will optimize the wrong distribution.
- Serving-threshold calibration must use the same runtime scoring path as
  serving/evaluation whenever the artifact is usable. Raw model-only scores can
  diverge from bucket fallback and serving reranker behavior enough to pass a
  gate that the product path still violates.
- A targeted feature-backfill queue is only useful if the workflow proves the
  selected VODs were materialized before dependent stages run. Build a
  feature-only cache artifact for the queue and fail the rematch before audio
  matching when selected VODs are missing from the refreshed VOD manifest.
- Scoped-window cross-platform matching has a Twitch-clip-selection bias. If
  the goal is to discover short-form reposts that were not Twitch top clips,
  add a coarse VOD audio-tiling layer to the VOD manifest before rematching
  instead of only expanding the top Twitch clip windows.
- Sparse coarse VOD tiling is not the same as full-VOD matching. When the user
  asks whether cross-platform outcomes are compared with entire VODs, ensure
  the second-pass tiling covers the full known VOD duration with no tile-count
  cap, or state the remaining sampling gap explicitly.
- Full-VOD tiling must distinguish permanent media unavailability from
  transient tile failures. If Twitch says a VOD does not exist or is
  unavailable, abort the remaining tiles for that VOD and continue with other
  VODs instead of repeatedly downloading doomed ranges.
- Cross-platform full-VOD coverage should not materialize video tiles. For
  audio-fingerprint matching, stream/extract full VOD audio once, persist the
  small fingerprint JSON, and match against that full-VOD fingerprint timeline;
  only materialize video windows after a match exists.
- A failed full-VOD audio fingerprint refresh must fail closed. If no new
  full-audio fingerprints are built or cached, preserve the last valid
  cross-platform outcome artifact and block downstream training instead of
  writing an empty manifest or zero-positive dataset.
- If the product decision is "audio-only full-VOD fingerprints", every
  cross-platform prepare call must run in that mode. Do not enable it only for a
  later refresh pass while an earlier seed pass can still rebuild manifests from
  scoped MP4 cache windows.
- Blocking cross-platform preparation must terminate the training orchestrator
  immediately. Continuing after `blocked_full_vod_audio_fingerprints` creates
  negative-only outcome datasets and hides the real failure behind a later
  trainer error.
- `max_reposts` is a cap, not a freshness requirement. Do not rediscover or
  redownload repost manifests on every run just because cached rows are below
  the cap; rebuild only when rows are missing, below `min_reposts`, or refresh
  is explicitly requested.
- FFmpeg temp output paths must keep a real media extension as the final suffix.
  A path like `source.m4a.part` can make ffmpeg fail muxer inference; use
  `source.part.m4a` or pass an explicit `-f`.
- Cross-platform audio matching must gate on VOD diversity, not only row count.
  One full-VOD fingerprint compared against a global repost pool can create
  hundreds of weak best-match positives at low thresholds.
- When the user narrows a fix explicitly, preserve that scope. For threshold-only
  requests, change the configured threshold and matching tests without adding
  extra gates or workflow behavior.
- Threshold changes do not affect a training run if the workflow reuses a stale
  cross-platform output artifact. Default to rematching for training, and make
  artifact reuse an explicit opt-in when evaluating new match thresholds.
- Docker local-training runs receive `backend/.env` via `--env-file`. When a
  default is changed in `train_jynxzi_vod.sh`, also update the real `.env` and
  `.env.example`; otherwise the env file silently overrides the fixed default.
- Do not trust Twitch label progress independently from the label artifact.
  If `--skip-covered` says a broad date range is covered but the JSONL only has
  labels for one date/VOD, reconcile progress against actual label dates before
  skipping collection.
- For outcome-only Jynxzi training, Twitch clip labels must not be a hidden
  prerequisite. Use a separate VOD inventory for full-VOD audio fingerprints,
  remove clip-label collection from the training workflow, and allow the ranker
  to train from cross-platform rows without non-empty Twitch label rows.
- VOD inventory rows need durable provenance when merged from multiple sources.
  Preserve API provenance over cache provenance, and never re-download
  cache-only VODs unless a usable full-VOD audio fingerprint already exists.
- After building full-VOD audio fingerprints, always rebuild the VOD manifest
  from the full-audio fingerprint cache before matching. A stale but usable
  old manifest can otherwise make matching ignore the newly built fingerprints.
- For Jynxzi outcome-only training, gate cross-platform coverage against the
  actually available VOD/repost universe. If the full available pool yields
  about 74 high-confidence matches over 29 VODs, a 300-row hard minimum blocks
  valid training and should be replaced by a row-plus-VOD diversity gate.
- Treat a zero-row platform discovery result as a source-adapter problem before
  accepting it as real data scarcity. For TikTok, keep a secondary HTML/search
  fallback behind the RSS route so sparse feeds do not collapse the candidate
  pool.
- When a manually trusted repost channel replaces broad search discovery, encode
  that trust explicitly in source tags and bypass only the profile-text filter.
  Keep date-window filtering intact so positives still line up with the VOD
  inventory window.
- YouTube bot-protection failures from yt-dlp need shared auth wiring, not a
  one-off retry. Apply cookiefile/browser-cookie options to every yt-dlp call
  in the repost path: discovery, metadata enrichment, downloads, and any
  metadata extraction used before ffmpeg.
- If the user reverses a diagnostic request, remove the observability surface
  without undoing the underlying functional fix unless they explicitly ask for
  both. For cookie auth, keep yt-dlp cookie options while removing cookie-file
  status/path logs.
- If the user rejects request delays for YouTube rate limits, do not sneak in
  sleep-based throttling. Prefer explicit batch/cache materialization of the
  needed Shorts and make the concurrency knob visible.
- If the user reverses a concurrency decision, remove the concurrency knob too
  when it only exists for that reverted behavior. For repost Shorts, serial
  means one-by-one cache materialization, not default workers set to one.
- YouTube's "try again later" rate limit is about request pacing, not media
  cache correctness. When applying a fix, set yt-dlp's native sleep options on
  the Short-form media cache fill rather than adding an outer manual sleep.
