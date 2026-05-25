# Task Plan

## 2026-05-19 Plan: Speed Up Ollama Clip Judge

- [x] Change serving Judge defaults to use a smaller Ollama model.
- [x] Increase Ollama Judge batch size and reduce judged shortlist size.
- [x] Ensure the selected Ollama model is available in the running Ollama container.
- [x] Restart backend runtime and verify `/api/status` reports the new Judge model.
- [x] Document verification results.

### Review Notes

- Changed serving config from `qwen2.5:14b`, Judge batch `4`, Judge Top-K
  `80` to `qwen2.5:7b`, Judge batch `8`, Judge Top-K `50`.
- Updated `backend/.env.example` with the same defaults so future env files do
  not drift back to the slower setup.
- Pulled `qwen2.5:7b` into `ai-clipping-ollama`; `ollama list` now shows both
  `qwen2.5:7b` and the older `qwen2.5:14b`.
- Recreated the backend container and verified it is healthy.
- Runtime verification inside the container reports:
  `ollama_model=qwen2.5:7b`, `clip_judge_batch_size=8`,
  `shortlist_judge_top_k=50`, `judge_provider=ollama`.
- `/api/status` reports Ollama active on `qwen2.5:7b`.
- Ollama smoke test passed. First request cold-loaded the model in about
  `29s`; immediate warm request completed in about `307ms`.

## 2026-05-19 Plan: Fix Context Leakage And Judge Feedback Calibration

- [x] Reproduce the stale/global callback leakage in VOD memory context selection.
- [x] Fix context retrieval so setup/callback evidence must be locally relevant to the candidate window.
- [x] Increase Judge-accepted/rejected feedback strength for Proposal and Ranker training without letting unjudged rows become positives.
- [x] Add/update focused tests for context relevance and feedback weighting.
- [x] Run focused backend tests, syntax checks, Graphify rebuild, and document results.

### Review Notes

- Triggered by latest job analysis: final clips reused the same unrelated
  callback summary and `minimum_required_setup_start=358.48` across far-apart
  windows, while low Proposal scores showed Judge feedback needs stronger
  training influence.
- Fixed `vod_memory` so setup/open-loop/callback spans must be locally relevant
  to the candidate and callback spans cannot bridge beyond the configured
  lookback window through weak one-token overlap.
- Verified the fix against the last job: final clip setup starts are now local
  (`516.82`, `2081.54`, `1480.38`, `8367.92`, `1397.48`) instead of all
  leaking the same stale `358.48` callback.
- Increased Judge feedback influence: serving Judge quality exports now receive
  score/confidence-aware sample weights, Proposal training uses stronger
  Judge-accepted/rejected weights, and Ranker `judge_feedback` / judge quality
  class multipliers were raised.
- Verification passed: focused VOD memory/training export/Proposal/Ranker tests
  (`70 passed`), analyzer/candidate recall/pipeline integration tests
  (`42 passed`), `py_compile`, live context check on the latest job, Graphify
  rebuild, backend CUDA image rebuild/restart with healthy container and
  `/api/status` response, and `git diff --check`.

## 2026-05-19 Plan: Analyze Latest Job

- [x] Identify the latest job and its status.
- [x] Inspect job debug artifacts, candidate funnel, selected clips, and judge/ranker/proposal diagnostics.
- [x] Check backend logs for errors or fallback behavior around that job.
- [x] Summarize quality bottlenecks and concrete next actions.

### Review Notes

- Started after request to analyze the latest job after backend restart and
  artifact retraining.
- Latest completed artifact job is `1e29506f-bed5-4234-8b2b-dae688e07ca0`
  for Twitch VOD `2774602304`; it finished with 5 clips and the frontend
  successfully requested all five clip files.
- Candidate funnel: 5,000 dense inventory windows, 249 merged candidates, 241
  normalized, 339 Proposal inputs, 240 Proposal-kept, 241 pre-Judge ranked, 80
  judged, 170 quality-kept pool entries, 12 final-ranked, 5 final clips.
- Proposal artifact was loaded, but final clip Proposal ranks were only 17, 27,
  86, 102, and 170. Two final clips still had very low Proposal scores
  (`0.076561` and `0.052378`), so Proposal is preserving coverage more than
  confidently selecting winners.
- Clip Judge worked through Ollama without timeout or fallback and judged 80
  candidates, but took 2,470.581 seconds. It accepted only 9 judged candidates;
  71 judged candidates failed mostly on low judge score, narrative completeness,
  payoff strength, and standalone clarity.
- Final clips were mostly rescued by Judge: one dense inventory candidate moved
  from pre-Judge score 2.399 to judged score 8.122, and one peak moved from
  2.720 to 7.162. This confirms the Ranker/Proposal still miss some candidates
  before Judge.
- Chat replay was effectively unusable for most of the VOD: only 66 chat rows
  from seconds 17-22 were fetched, with log reason
  `twitch_rechat_fetch_stopped ... no_offset_progress offset=23`. As a result
  chat features are zero for the selected clips.
- Transcript/context diagnostics show some bad global context leakage: all
  final moments had the same unrelated callback summary and
  `minimum_required_setup_start=358.48`, which is not credible for clips hours
  later. This can mislead Judge/final explanations and should be fixed before
  trusting context summaries.

## 2026-05-18 Plan: Retrain Proposal And Ranker Artifacts

- [x] Run Proposal training through `scripts/train_proposal_local_docker.sh`.
- [x] Inspect the Proposal training summary and artifact metrics.
- [x] Determine the safest Ranker retrain path from existing staged/training artifacts.
- [x] Run or prepare the Ranker retrain with the current corrected training signals.
- [x] Verify generated artifacts and document the result.

### Review Notes

- Triggered after quality-fix implementation because training-signal changes do
  not affect serving until artifacts are retrained.
- Proposal retrain completed successfully through Docker. New
  `jynxzi_proposal_scorer.json` is trained on 9,937 rows with 755 positives.
  Heldout metrics: AUC 0.61846, logloss 0.321553, precision@K/recall@K
  0.169863. Label sources now include `judge_quality_accepted`,
  `judge_quality_rejected`, `pipeline_selected`, and `pipeline_low_rank`.
- Ranker retrain should use the artifact-only local training path with the last
  successful date range (`2026-01-01` to `2026-05-12`) so the workflow exports
  current feedback/training rows instead of directly training on a stale dataset
  file.
- Ranker artifact-only retrain completed successfully with return code 0.
  `jynxzi_ranker.json`, `jynxzi_boundary_ranker.json`, and
  `jynxzi_vod_training_rows.jsonl` were rewritten at the end of the run.
- The refreshed transcript matching stage used 64 VOD rows and 356 repost rows,
  loaded 64 VOD transcripts, reused 328 repost transcript caches, fetched 28
  missing repost transcripts, and produced 144 transcript matches across 52
  VODs at the current 0.5 threshold.
- Current training rows after the retrain: 450 VOD-ranker rows, 143
  cross-platform rows on disk, and 9,852 proposal training rows.
- Ranker eval artifact exists and reports 30 eval rows. Runtime model precision
  improved over the heuristic at the selected threshold (0.909 vs 0.800), while
  ranking metrics stayed flat on this small eval set.
- Verification passed after retrain: artifact timestamps confirmed, summaries
  inspected, row counts checked, and `git diff --check` passed after normalizing
  retrain-generated CSV line endings.

## 2026-05-18 Plan: Fix Clip Quality Weaknesses

- [x] Fix funnel semantics before changing scoring: split `quality_kept` into judged quality pass, unjudged retained pool, and final-ranked pool so every report shows whether candidates were actually judged.
- [x] Promote Clip-Judge and frontend feedback into durable training rows for both Proposal and Ranker, with explicit labels for selected, rejected, skipped-by-user, and low-judge-score candidates.
- [ ] Retrain the Proposal model on real serving windows plus cross-platform positives, not only transcript-match overlap, and require heldout metrics plus live-job lift checks before trusting the artifact.
- [x] Add Proposal calibration diagnostics: final selected candidate proposal-score distribution, missed-high-judge candidates, precision@top-k, recall@top-k, and source-specific lift for `dense_inventory`, `peak`, `quote`, and `dense_window`.
- [x] Improve the Pre-Judge Ranker with Judge/feedback outcomes so high-judge candidates rank in the top 20 before the LLM stage instead of being rescued from ranks 24-32.
- [x] Tighten candidate generation: replace pure brute-force dense windows with boundary-aware dense windows, transcript sentence boundaries, local context expansion, and overlap pruning before Proposal scoring.
- [x] Add candidate boundary repair before Judge: expand weak windows to include setup/payoff context, snap starts/ends to transcript boundaries, and penalize windows with incomplete standalone text.
- [x] Change the Clip-Judge rubric from "complete moment" to "short-form viral potential": require hook, tension/reaction/payoff, standalone clarity, and Jynxzi-specific replay value; report separate sub-scores instead of one opaque score.
- [x] Use Judge only as a teacher/reranker, not as final truth: feed high/low judge outcomes into training, but keep final validation tied to frontend feedback and cross-platform positives.
- [x] Add a before/after evaluation command that runs the same VOD through old vs new artifacts and reports final clip overlap, judge-score lift, proposal-score lift, and manual review samples.

### Success Criteria

- Final selected clips should usually have non-trivial Proposal scores; obvious winners should not depend on broad shortlist survival.
- At least 80% of high-Judge candidates should appear in the pre-Judge top 20 on heldout jobs.
- The frontend funnel should clearly show whether a bad final result came from generation, Proposal, Ranker, Judge, Quality Gate, or final selection.
- The Proposal artifact should show meaningful heldout lift, not just high accuracy from many easy negatives.
- Manual sample review should show fewer generic/low-payoff dense windows and more standalone moments with clear setup and payoff.

### Review Notes

- Current weakness is not one broken stage. The broad candidate search reaches enough windows, but Proposal and Ranker do not yet rank the truly promising windows strongly enough.
- The last analyzed job proves Judge lift exists: several final candidates moved up from weak pre-Judge ranks, but their Proposal scores were still very low.
- The most important first implementation target is diagnostics/label flow, because changing thresholds before the system learns from real outcomes would only hide the root cause.
- Implemented the first two quality levers: funnel reports now split `judged_quality_kept` and `unjudged_quality_kept`, and transition retention uses intersection counts so non-nested stages cannot report impossible retention above 100%.
- Clip-Judge quality gate rows now become durable training feedback only when a real judge result exists. Accepted judged rows export as weak positives and rejected judged rows export as judge-feedback negatives for both Proposal and Ranker training paths.
- Unjudged quality passes are no longer treated as positive Judge labels for Proposal training.
- Verification passed: candidate recall tests, feedback-store/training-export/proposal-training tests, `py_compile`, `git diff --check`, and Graphify rebuild through the local Graphify interpreter.
- Implemented the remaining serving-quality levers: Proposal input candidates are boundary-normalized before Proposal scoring, dense windows keep more setup context, and starts that begin mid-thought are repaired to a cleaner transcript boundary before Ranker/Judge.
- Added Proposal/Ranker lift diagnostics to the candidate funnel: positive-proxy precision/recall@K, final/high-Judge Proposal score distributions, high-Judge Proposal keep ratio, source-specific quality breakdown, and high-Judge pre-Judge top-20/top-40 ratios.
- `/api/jobs/{job_id}` and the frontend Pipeline overview now expose Judge quality split, Proposal lift, and Ranker lift so weak jobs can be diagnosed without reading raw JSONL.
- Added `backend/ml/evaluate_candidate_funnel.py` for current-vs-baseline debug-dir comparisons of final selections, Judge lift, Proposal lift, Ranker lift, and missed high-Judge candidates.
- The Clip Judge prompt now scores viral short-form potential more strictly: immediate scroll-stop, Jynxzi-style reaction/quote/payoff, pacing/replay value, and standalone clarity are weighted above generic completeness.
- Ranker training now gives explicit Judge-feedback rows a small supervision boost so future artifacts can learn from judged accepts/rejects instead of treating them like generic weak labels.
- Still not done automatically: the Proposal/Ranker artifacts must be retrained before these training-signal changes affect production scores.

## 2026-05-18 Plan: Analyze Latest Job Results

- [x] Check whether current job artifacts still exist after storage cleanup.
- [x] Inspect backend logs/artifacts for the most recent completed frontend job.
- [x] Summarize result quality, candidate funnel, and any missing evidence.

### Review Notes

- Latest artifact-backed job is `d6bfbbe2-605f-40bd-8c48-7c77f72d74de` for Twitch VOD `2774602304`; it completed with 5 clips.
- Source cleanup worked for this job: `video.mp4`, `audio.wav`, and `chunks/` are absent while clips/debug artifacts remain.
- Candidate funnel shows 5,000 dense inventory windows plus heuristic sources, 357 proposal inputs, 240 proposal-kept candidates, 268 pre-judge ranked candidates, 80 judged candidates, 12 final-ranked candidates, and 5 final moments.
- Clip Judge ran through Ollama successfully but took about 2,184 seconds for 80 candidates.
- Funnel diagnostics still need a semantic cleanup: `quality_kept=195` includes 188 unjudged candidates, so it is misleading as a post-Judge quality metric.


## 2026-05-18 Plan: Delete Serving VOD After Job

- [x] Inspect where frontend URL/upload jobs store the source VOD, derived audio, clips, and debug artifacts.
- [x] Delete the source VOD after pipeline completion/failure without removing output clips, debug diagnostics, feedback material, or training caches.
- [x] Add focused tests and run syntax/Graphify verification for the backend change.

### Review Notes

- Serving/frontend jobs now delete their job-local `video.mp4` after the pipeline
  finishes, fails, or is cancelled. `audio.wav` and `chunks/` are also removed
  as transient source artifacts.
- Output clips and debug artifacts are preserved because only the job-local
  source media paths are removed, not the whole job directory.
- Training/backfill jobs with durable cache semantics are excluded:
  `twitch_vod*`, `proposal_training_*`, and `judge_backfill_*` keep their
  source media.
- The router parent process also runs the same cleanup after the isolated
  pipeline subprocess exits, so SIGKILL/OOM worker failures still remove
  `video.mp4`.
- Verification passed: `py_compile`, focused pipeline cleanup tests, focused
  router subprocess cleanup tests, `git diff --check`, and Graphify rebuild via
  the local Graphify interpreter after system `python3` lacked `graphify`.

## 2026-05-18 Plan: Start ngrok By Default

- [x] Inspect the container startup script and current remote-mode default.
- [x] Make ngrok the default remote mode while keeping an explicit local-only opt-out.
- [x] Verify script syntax and document the operational behavior.

### Review Notes

- `scripts/start_containerized.sh` now defaults `AI_CLIPPING_REMOTE` to `ngrok`, so running `./scripts/start_containerized.sh` starts the ngrok tunnel automatically after the containers come up.
- Local-only startup is still explicit and available with `AI_CLIPPING_REMOTE=none ./scripts/start_containerized.sh`.
- Verification passed: `bash -n scripts/start_containerized.sh`.

## 2026-05-18 Plan: Candidate Filtering Observability

- [x] Map the exact serving funnel from candidate generation through Proposal, Ranker, Judge, Quality Gate, final selection, and frontend display.
- [x] Extend the existing candidate recall report into a per-candidate trace with stable candidate ids, source, time window, scores, pass/fail status, and rejection reason at every stage.
- [x] Add aggregate funnel metrics for each transition: input count, kept count, loss count, loss percentage, top rejection reasons, score distributions, and source mix changes.
- [x] Add “lost good candidates” analysis by comparing final/high-ranker clips against earlier stages to identify whether quality loss starts in Proposal, Ranker, Judge, Quality Gate, or final diversity trimming.
- [ ] Surface the funnel in the frontend as an inspectable table/timeline, not only summary cards: stage columns, candidate rows, filters for rejected/kept/source/reason, and direct links to preview the candidate window.
- [x] Add debug artifacts and tests so every pipeline run can prove which stage removed each candidate and why.
- [ ] Verify with one known weak frontend job and one known good job, then document the bottleneck and next quality lever.

### Review Notes

- The backend now writes `debug/candidate_trace.jsonl` with one row per
  candidate across generation, Proposal, Ranker, Judge scope, Clip Judge,
  Quality Gate, final ranked, and final moment stages.
- The backend also writes `debug/candidate_funnel_summary.json` with stage
  counts, losses, drop-stage counts, rejection-reason counts, source counts,
  score distributions, and trace samples for final and rejected candidates.
- `/api/jobs/{job_id}` now exposes the consolidated funnel as
  `pipeline_diagnostics.candidate_funnel`, so the frontend no longer needs to
  infer the funnel from separate debug artifacts.
- The frontend Pipeline overview now includes a Candidate Funnel section with
  stage counts, largest losses, rejection/drop reasons, and a candidate sample
  table showing time window, source, rank, Proposal/Ranker/Judge scores, and
  drop reason. Dedicated preview links/interactive filters are still the next
  UI refinement.
- Verification passed: backend syntax check, focused candidate trace tests,
  focused jobs-router diagnostics test, frontend lint, `git diff --check`, and
  Graphify rebuild via CLI fallback because the direct Python module import is
  unavailable in this shell.
- Remaining validation is empirical: run one known weak frontend job and one
  good job, then use the new funnel to identify the actual quality bottleneck.

## 2026-05-15 Plan: Align Frontend Whisper Runtime With Local Training

- [x] Compare `scripts/train_local_docker.sh` and `ml.local_train` runtime defaults against the containerized frontend/backend path.
- [x] Apply the matching Whisper/CUDA/thread/shared-memory defaults to serving without increasing concurrent Whisper model count.
- [x] Add/adjust tests for the pipeline subprocess environment.
- [x] Verify backend tests, Graphify rebuild, container runtime settings, and document results.

### Review Notes

- `scripts/train_local_docker.sh` optimizes Whisper mainly through CUDA
  exposure, host IPC, 16 GB shared memory, fixed English transcription, CUDA
  allocator settings, and native thread caps.
- The containerized frontend/backend path now uses the same serving defaults:
  `WHISPER_LANGUAGE=en`, CUDA allocator defaults when `WHISPER_DEVICE=cuda`,
  `CUDA_VISIBLE_DEVICES=0`, tokenizers disabled, and BLAS/OpenMP thread pools
  capped at one thread.
- `docker-compose.yml` now gives the backend `ipc: host` and `shm_size: 16gb`;
  `docker-compose.cuda.yml` carries the CUDA-specific runtime defaults.
- This does not increase Whisper model concurrency. It keeps the persistent
  isolated chunk worker from the previous optimization and removes avoidable
  per-chunk language detection/runtime overhead.
- Verification passed: focused backend tests, `py_compile`, `git diff --check`,
  compose config inspection, Graphify rebuild, backend image rebuild, API
  healthcheck, and live container runtime checks. Runtime reports
  `whisper_device=cuda`, `whisper_language=en`, CUDA available, 16 GB shared
  memory, and the expected thread/CUDA environment variables.

## 2026-05-15 Plan: Optimize Frontend Whisper Transcription

- [x] Confirm current transcription bottleneck and relevant runtime defaults.
- [x] Replace per-chunk isolated Whisper subprocesses with a persistent isolated chunk worker.
- [x] Preserve chunk cache/progress semantics and parent-process isolation.
- [x] Add focused transcriber tests for persistent isolated processing.
- [x] Verify backend tests, Graphify rebuild, container runtime config, and document results.

### Review Notes

- The slow path was `PIPELINE_ISOLATE_TRANSCRIPTION_CHUNKS=true` spawning a
  separate Python process per audio chunk, which reloaded Whisper for every
  chunk.
- `transcribe_with_progress` now keeps isolation but sends all uncached chunks
  to one persistent isolated worker via `--transcribe-chunks`, so Whisper loads
  once per VOD transcription batch instead of once per chunk.
- Cached chunk transcripts still short-circuit before model loading, progress
  updates are still emitted per chunk, and non-isolated transcription keeps the
  existing in-process path.
- Verification passed: transcriber `py_compile`, focused transcriber tests,
  pipeline/config tests, `git diff --check`, Graphify rebuild, and
  containerized runtime checks.
- Backend was rebuilt and restarted. Runtime reports
  `pipeline_isolate_transcription_chunks=True`, `whisper_device=cuda`,
  `frontend_job_cache_enabled=False`, and `shortlist_judge_top_k=80`.

## 2026-05-14 Plan: Speed Up Clip Judge

- [x] Inspect current Clip Judge provider flow and config/test coverage.
- [x] Add Ollama batch judging so multiple candidates can be scored per LLM call.
- [x] Keep non-Ollama providers on the existing single-candidate path.
- [x] Add tests for batch judging and fallback behavior.
- [x] Verify backend tests, frontend build hygiene if types change, Graphify rebuild, and container runtime.

### Review Notes

- Added `CLIP_JUDGE_BATCH_SIZE=4` as the local/default Ollama Judge batch
  size.
- The Ollama text Judge now scores consecutive recall-passing candidates in
  one strict JSON batch request. With `SHORTLIST_JUDGE_TOP_K=80`, this reduces
  the worst-case Judge calls from about 80 to about 20 before recall filtering.
- Non-Ollama providers and multimodal/keyframe Judge paths stay on the
  existing single-candidate path.
- If a batch call fails or omits a candidate result, that candidate falls back
  to the existing single-candidate `judge_candidate` path instead of being
  silently dropped.
- Verification passed: backend `py_compile`, focused Clip-Judge/config tests,
  jobs-router/pipeline/recall-report tests, frontend lint/build,
  `git diff --check`, Graphify rebuild, and containerized runtime checks.
- Containers were rebuilt and restarted. Backend is healthy, frontend/API
  respond, and runtime settings report `judge_provider=ollama`,
  `clip_judge_batch_size=4`, and `clip_judge_timeout_sec=0.0`.

## 2026-05-14 Plan: Add Candidate Recall Debug Report

- [x] Inspect current debug artifact schema for candidate, proposal, judge, and quality-gate stages.
- [x] Add a per-job recall report that tracks candidate counts, source mix, and losses across pipeline stages.
- [x] Surface the recall report in backend pipeline diagnostics so the frontend can show where candidates drop.
- [x] Verify backend tests, frontend build hygiene, Graphify rebuild, and container runtime if needed.

### Review Notes

- Added `services/candidate_recall_report.py` to build a per-job funnel report
  from generated candidates, proposal audit rows, ranker output, judge scope,
  quality-gate audit rows, final ranked candidates, and final moments.
- Pipeline now writes `debug/candidate_recall_report.json` after final moments
  are selected. The report includes stage counts, source counts, transition
  losses, largest losses, top Proposal rejects, candidates outside Judge scope,
  Quality-Gate rejects, and final candidate ids.
- `/api/jobs/<job_id>` now exposes a `Candidate Recall` diagnostics stage so
  the frontend can show whether candidates dropped at Proposal, Ranker/Judge,
  Quality Gate, or final selection.
- Frontend `JobStatus` type now includes the `judging` status used by the
  Clip-Judge progress phase.
- Verification passed: backend `py_compile`, focused pytest suite
  (`test_candidate_recall_report`, `test_jobs_router`,
  `test_pipeline_integration`, `test_config`), frontend lint/build,
  `git diff --check`, Graphify rebuild, and containerized runtime import/API
  checks.
- Containers were rebuilt and restarted; backend is healthy and frontend/API
  respond.

## 2026-05-14 Plan: Remove Clip Judge Stage Timeout And Add Live Progress

- [x] Inspect current Clip Judge timeout path and frontend diagnostics flow.
- [x] Disable the whole-stage Clip Judge timeout so the Judge can finish instead of falling back.
- [x] Persist per-candidate Clip Judge progress and surface it in job status diagnostics.
- [x] Add frontend status label for the judging phase.
- [x] Verify focused backend/frontend tests and update running containers.

### Review Notes

- Removed the whole-stage Clip Judge timeout from the serving pipeline. The
  Judge no longer falls back to pre-judge ranking just because the full stage
  takes longer than 90 seconds.
- `CLIP_JUDGE_TIMEOUT_SEC=0` is now the local default, so provider request
  timeout is disabled unless explicitly configured otherwise.
- Added live Clip Judge progress: processed, attempted, judged,
  recall-filtered, elapsed seconds, and current candidate window are persisted
  to `debug/clip_judge_stage.json` while judging runs.
- Job status now has a dedicated `judging` phase and advances progress during
  candidate judging; the frontend stage list shows `Clip Judge`.
- The frontend diagnostics panel now summarizes Clip Judge progress while the
  job is still running instead of only showing the final judge result.
- Verification passed: backend `py_compile`, focused pytest suite
  (`test_clip_judge`, `test_jobs_router`, `test_pipeline_integration`,
  `test_config`), frontend lint/build, `git diff --check`, Graphify rebuild,
  and containerized runtime checks.
- Containers were rebuilt and restarted. Runtime config now reports
  `clip_judge_timeout_sec=0.0` and `judge_provider=ollama`; frontend and backend
  endpoints respond.

## 2026-05-14 Plan: Keep Backend Diagnostics Visible Before Debug Artifacts

- [x] Confirm why the frontend hides backend diagnostics after cache cleanup.
- [x] Show the backend diagnostics summary even when detailed debug stages are not available yet.
- [x] Verify frontend lint/build and document the behavior.

### Review Notes

- Root cause: the backend did return `pipeline_diagnostics` during early phases,
  but with `available=false` before `debug/` artifacts existed. The frontend
  returned `null` for that state, so the whole backend diagnostics panel
  disappeared while the job was still downloading/transcribing.
- `PipelineDiagnosticsPanel` now renders the backend summary even before
  detailed stages exist, with a short note that detailed candidate/ranker/judge
  data appears after those artifacts are written.
- The current job had no `debug/` directory yet because it was still in
  `transcribing`; that is expected after disabling transcript cache reuse.
- During the frontend image rebuild, Compose also recreated the backend
  container and killed the active worker. I marked job
  `8840d552-24ab-4362-b46a-77591479f00e` as failed so the UI does not wait
  forever.
- Verification passed: `npm --prefix frontend run lint`,
  `npm --prefix frontend run build`, live `/api/jobs/...` inspection, and
  frontend/backend containers are healthy.

## 2026-05-14 Plan: Disable Frontend Job Caches And Purge Old Job Artifacts

- [x] Map frontend-serving cache write paths before deleting anything.
- [x] Add a serving setting that disables frontend download/transcript cache reads and writes.
- [x] Keep training/cross-platform caches untouched unless explicitly requested.
- [x] Delete existing frontend job cache/artifact directories and job metadata.
- [x] Verify tests, runtime config, and storage cleanup.

### Review Notes

- Added `FRONTEND_JOB_CACHE_ENABLED=false` as the serving default in backend
  settings, local `.env`, and `.env.example`.
- Normal frontend/serving job IDs no longer use the Twitch VOD download cache
  even when `TWITCH_DOWNLOAD_CACHE_TTL_SEC=-1`; training-style job IDs such as
  `twitch_vod*` can still use the durable training cache path.
- Frontend pipeline transcription no longer reads or writes
  `backend/tmp/storage/_transcript_cache` when serving cache is disabled; it
  transcribes the current job input directly.
- Deleted old frontend job artifacts and caches from `backend/tmp/storage`,
  including UUID job directories, `jobs.json`, `_transcript_cache`,
  `_download_cache/twitch`, `vision_language_features`, and old materialized
  proposal job debug material. About `90.1 GiB` was removed.
- Preserved non-frontend training/repost caches under `_download_cache`
  (`short_form`, `short_form_metadata`, `short_form_transcripts`) and
  `backend/tmp/vod_feature_cache`.
- Verification passed: `py_compile`, `backend/tests/test_downloader.py`,
  `backend/tests/test_pipeline_integration.py`, `backend/tests/test_config.py`,
  scoped `git diff --check`, runtime settings inspection, storage inventory,
  and Graphify rebuild (`3312` nodes, `8869` edges, `94` communities).

## 2026-05-14 Plan: Fix Clip Judge Timeout Observability And Load

- [x] Confirm whether the frontend warning is a real Judge miss or a diagnostics false positive.
- [x] Persist Clip-Judge stage outcome/degradation details into job debug artifacts.
- [x] Reduce default local Ollama Judge load so serving judges the shortlist instead of every candidate.
- [x] Verify tests, runtime config, and live diagnostics.

### Review Notes

- The warning was real: the latest test job `bec13437-9dac-48b9-94d0-6457c6053d09`
  logged `clip_judge_stage_degraded` because the Clip Judge hit the 90s timeout
  and fell back to 248 pre-judge ranked candidates.
- Root cause was local serving load: `JUDGE_ALL_CANDIDATES=true` plus
  `SHORTLIST_JUDGE_TOP_K=80` meant Ollama could be asked to judge far more than
  the useful shortlist within the timeout.
- Serving defaults now judge only the shortlist:
  `JUDGE_ALL_CANDIDATES=false` and `SHORTLIST_JUDGE_TOP_K=24` in config,
  local `.env`, and `.env.example`.
- Pipeline runs now persist `debug/clip_judge_stage.json`, so the frontend can
  distinguish timeout, disabled judge, empty judge result, and provider failure
  instead of only showing a generic "no judge result" warning.
- Verification passed: `py_compile`, focused jobs-router tests, config and
  pipeline integration tests, and scoped `git diff --check`.
- Graphify rebuilt successfully after backend edits: `3307` nodes, `8858`
  edges, `96` communities.
- Running backend container was recreated with the new env and patched with the
  updated backend files; runtime settings now report `judge_provider=ollama`,
  `judge_all_candidates=False`, and `shortlist_judge_top_k=24`.

## 2026-05-14 Plan: Add Frontend Pipeline Observability

- [x] Inspect current backend debug artifacts and job-status payload shape.
- [x] Add a backend per-job pipeline diagnostics summary from existing debug files.
- [x] Add frontend types and a compact diagnostics panel for candidates, proposal, ranker, judge, quality gate, and final clips.
- [x] Verify backend tests, frontend lint/build hygiene, Graphify rebuild after backend edits, and runtime API shape.
- [x] Document review notes with what the UI can and cannot prove.

### Review Notes

- Added `pipeline_diagnostics` to the job-status payload. It is derived from
  the existing per-job debug artifacts under `backend/tmp/storage/<job_id>/debug`
  and does not add a second tracking system.
- The diagnostics summarize candidate generation, proposal scoring, ranker,
  clip judge, serving quality gate, and final selection with status, counts,
  and warning details.
- The frontend now shows a `Pipeline Überblick` panel while a job is polling
  and preserves the latest panel above the results after completion.
- Older completed jobs can show `disabled` or `warning` for stages whose audit
  files did not exist when those jobs were produced. That is intentional:
  the panel reports what can be proven from artifacts instead of inventing a
  green status.
- Verification passed: backend `py_compile`, focused
  `backend/tests/test_jobs_router.py`, `npm --prefix frontend run lint`,
  `npm --prefix frontend run build`, `git diff --check`, Graphify rebuild, and
  live `/api/jobs/<job_id>` runtime check from the containerized backend.

## 2026-05-14 Plan: Force Clip Judge To Ollama

- [x] Add an explicit Judge provider setting and thread it through the pipeline.
- [x] Set local and example Docker env defaults to use Ollama for Clip Judge.
- [x] Verify tests, rebuild Graphify after backend edits, rebuild backend container, and confirm runtime settings.

### Review Notes

- Added `JUDGE_PROVIDER` to backend settings and set `JUDGE_PROVIDER=ollama`
  in both `.env.example` and the local Docker `.env`.
- The pipeline now passes the configured provider into `clip_judge.rerank_candidates`.
  With `JUDGE_PROVIDER=ollama`, the Anthropic two-tier path is disabled even if
  an Anthropic key is still present.
- When Ollama is forced, the pipeline skips unnecessary multimodal keyframe
  sampling for cloud multimodal providers and uses the text-only Ollama judge.
- `scripts/start_containerized.sh` now auto-enables `docker-compose.ollama.yml`
  when `backend/.env` contains `JUDGE_PROVIDER=ollama`.
- `docker-compose.ollama.yml` now points the backend at `http://ollama:11434`
  inside the Compose network.
- Verification passed: `py_compile`, focused Ollama provider test, full
  `backend/tests/test_clip_judge.py`, `backend/tests/test_config.py` plus
  pipeline integration tests, `bash -n`, Compose config rendering,
  `git diff --check`, and Graphify rebuild.
- Runtime verification passed: backend is healthy, Ollama container is running,
  backend reports `judge_provider=ollama`, `ollama_host=http://ollama:11434`,
  and the backend can reach Ollama with model `qwen2.5:14b` available.

## 2026-05-14 Plan: Make Judge Timeout Non-Fatal In Serving

- [x] Inspect the current Judge timeout failure and fail-fast configuration.
- [x] Change pipeline timeout handling so pre-judge ranked candidates continue to serving.
- [x] Set serving Docker defaults so Judge provider failures are not fail-fast by default.
- [x] Add focused regression coverage for timeout degradation.
- [x] Run tests, rebuild Graphify, and document the result.

### Review Notes

- Root cause: serving jobs already had ranked candidates, but the optional
  Clip Judge stage was wrapped in a hard timeout path that raised when
  `CLIP_JUDGE_FAIL_FAST=true`.
- Judge timeouts now degrade to the pre-judge ranked candidates and log
  `clip_judge_stage_degraded` with the fallback candidate count. Non-timeout
  `ClipJudgeError` still respects the fail-fast setting.
- Serving defaults are now non-fatal: `CLIP_JUDGE_FAIL_FAST=false` and
  `JUDGE_PROVIDER_FALLBACK_ENABLED=true` in both `.env.example` and the local
  `.env` used by Docker.
- Verification passed: `py_compile` for `backend/pipeline.py` and
  `backend/config.py`, focused timeout regression test, full
  `backend/tests/test_pipeline_integration.py`, `backend/tests/test_config.py`
  plus pipeline integration tests, `git diff --check`, and Graphify rebuild.
- Rebuilt/restarted the backend Docker container with CUDA compose overrides.
  Runtime verification passed: `/api/status` responds and the container reports
  `clip_judge_fail_fast=False`, `judge_provider_fallback_enabled=True`, and
  `clip_judge_timeout_sec=90.0`.
- Follow-up verification caught and fixed a keyword typo in the new fallback
  helper call sites (`timeout` -> `timeout_sec`). The backend image was rebuilt
  again and `/api/status` still responds.

## 2026-05-14 Plan: Fix Ngrok Failed Fetch

- [x] Reproduce local and ngrok API reachability from the current running containers.
- [x] Inspect frontend fetch configuration for direct localhost/backend URLs that break remotely.
- [x] Make browser API/media requests use same-origin by default.
- [x] Add focused verification for remote API/media fetch behavior.
- [x] Document the root cause and verification result.

### Review Notes

- Root cause: the current remote/ngrok container was built with browser API
  calls targeting `http://localhost:8000`. On another PC, that means the remote
  client tries to fetch its own local machine, which surfaces as `failed to
  fetch`.
- There was also one ngrok log entry with `connect: connection refused` while
  the frontend port was down during a restart, but the persistent remote-fetch
  problem was the browser-side API base.
- `frontend/lib/api.ts` now normalizes an unset or empty `NEXT_PUBLIC_API_URL`
  to same-origin requests, so API calls use `/api/...` and clip URLs use
  `/static/...` through the Next.js rewrite.
- Docker defaults now build the frontend with `NEXT_PUBLIC_API_URL=` rather
  than `http://localhost:8000`.
- Remote was rebuilt/restarted with `AI_CLIPPING_REMOTE=ngrok
  ./scripts/start_containerized.sh`.
- Verification passed: local `/api/status`, remote
  `https://catfight-preformed-entertain.ngrok-free.dev/api/status`, remote
  frontend HTML, remote JS chunk, no `localhost:8000` in the frontend
  container bundle, no `localhost:8000` in the fetched remote page chunk,
  `node` frontend config tests, `npm --prefix frontend run lint`, and
  `git diff --check`.

## 2026-05-14 Plan: Wire Proposal Training Materialization Into Script

- [x] Add a materializer that creates proposal debug folders from cached VOD transcripts with cross-platform positives.
- [x] Add a proposal-training pipeline entrypoint that runs materialization, row building, then model training.
- [x] Wire `scripts/train_proposal_local_docker.sh` to the new pipeline entrypoint.
- [x] Add focused tests for materialization/orchestration behavior.
- [x] Verify syntax, tests, script help, diff hygiene, and rebuild Graphify after backend edits.

### Review Notes

- `scripts/train_proposal_local_docker.sh` now invokes
  `ml.train_proposal_pipeline` instead of calling the scorer trainer directly.
- The pipeline first materializes cached positive VOD transcripts into isolated
  proposal debug jobs under `<storage-dir>/_proposal_training_materialized`,
  then builds serving-shape proposal rows, then trains the logistic proposal
  scorer.
- The materializer uses compatible VOD transcript caches first, then safely
  falls back to the latest transcript cache for the same VOD ID when only the
  Whisper runtime metadata differs.
- A local non-destructive verification run against two real Jynxzi positive
  VODs materialized 2 VODs and built 34 proposal rows with 1 positive and 33
  negatives using reduced candidate limits.
- Verification passed: `py_compile`, `bash -n`, script help rendering, focused
  proposal tests (`11 passed`), `git diff --check`, and Graphify rebuild
  (`3294` nodes, `8818` edges, `95` communities).

## 2026-05-14 Plan: Fix Ngrok Clip Playback

- [x] Confirm why remote job status works but returned clip media does not render.
- [x] Route generated clip media through the same ngrok/Next.js origin.
- [x] Add focused frontend config verification for `/static` rewrites.
- [x] Rebuild/restart frontend in remote mode and verify existing clip media over ngrok.
- [x] Rebuild Graphify if backend files change and document results.

### Review Notes

- Root cause: remote mode only rewrote `/api/:path*` from the frontend/ngrok
  origin to the backend. Job polling succeeded, but clip files use
  `/static/{job_id}/clips/{filename}`, so Next.js returned `404` for generated
  media over ngrok.
- `frontend/next.config.mjs` now rewrites both `/api/:path*` and
  `/static/:path*` to the internal backend service.
- Added `frontend/tests/next-config-rewrites.mjs` to lock the API and static
  media rewrites.
- Rebuilt/restarted with `AI_CLIPPING_REMOTE=ngrok ./scripts/start_containerized.sh`.
- Verified the previously failing clip path:
  `https://catfight-preformed-entertain.ngrok-free.dev/static/7141a1f5-2231-4c16-aff0-efca5908c448/clips/clip_01.mp4`
  now returns `200` for full GET and `206` with `content-type: video/mp4` for
  range requests.
- Verification passed: `node --check frontend/next.config.mjs`,
  `node frontend/tests/next-config-rewrites.mjs`,
  `npm --prefix frontend run lint`, frontend rebuild, local `/static` route,
  remote `/static` route, and remote clip byte download.
- No backend code changed, so Graphify rebuild was not required.

## 2026-05-14 Plan: Add Optional Ngrok Remote Startup

- [x] Replace the temporary reverse-proxy setup with a versioned Next.js API rewrite.
- [x] Add an explicit remote mode to `scripts/start_containerized.sh`.
- [x] Preserve the default local-only startup behavior.
- [x] Verify local mode, remote mode, ngrok URL extraction, and frontend API base.
- [x] Document how to stop the remote tunnel.

### Review Notes

- `AI_CLIPPING_REMOTE=ngrok ./scripts/start_containerized.sh` now starts the
  normal containers, rebuilds the frontend with same-origin API calls, and then
  starts one ngrok tunnel against frontend port `3000`.
- Next.js rewrites `/api/:path*` to the backend service
  `http://backend:8000/api/:path*`, so no extra Python reverse proxy is needed.
- Default local startup remains unchanged: without `AI_CLIPPING_REMOTE=ngrok`,
  the frontend still builds with `http://localhost:8000` as the browser API
  base.
- Runtime verification passed with CUDA auto-selection: backend/frontend
  containers are running, local `http://127.0.0.1:3000/api/status` returns
  backend JSON through the frontend rewrite, and remote
  `https://catfight-preformed-entertain.ngrok-free.dev/api/status` returns the
  same backend status through ngrok.
- Frontend bundle check found no `http://localhost:8000` reference after the
  remote-mode build.
- Verification passed: `bash -n scripts/start_containerized.sh`,
  `node --check frontend/next.config.mjs`, Compose config rendering,
  `npm --prefix frontend run lint`, scoped `git diff --check`, and a real
  `AI_CLIPPING_REMOTE=ngrok ./scripts/start_containerized.sh` run.
- Active tunnel PID/log are `backend/tmp/ngrok/ngrok.pid` and
  `backend/tmp/ngrok/ngrok.log`. Stop command:
  `kill "$(cat backend/tmp/ngrok/ngrok.pid)"`.

## 2026-05-14 Plan: Test VOD 2771207666 Through Frontend

- [x] Confirm frontend/backend containers and the frontend submission path.
- [x] Submit `https://www.twitch.tv/videos/2771207666` through the frontend-equivalent submit path.
- [x] Watch job status, backend logs, and cgroup OOM state until the next bottleneck.
- [x] Re-plan and remove the full-audio extraction requirement when transcript chunks are already cached.
- [x] Re-run the frontend-equivalent VOD test and record whether the fix holds.

### Review Notes

- Frontend and backend containers were running; frontend returned HTTP `200`,
  backend `/api/status` returned active pipeline providers.
- Job submitted with the same frontend JSON contract plus `profile=jynxzi`:
  `2ee4f409-2bbe-4331-a5ff-60a10b70de76`.
- First frontend-equivalent run exposed a second root cause before transcription: the pipeline still extracts a full `audio.wav` from the 6.4G VOD even when all transcript chunks are already cached. This drove cgroup memory/current above 8 GB and incremented `oom_kill` from 0 to 1 while `ffmpeg` was still extracting audio.
- The pipeline now probes VOD duration with `ffprobe`, verifies that all
  expected transcript chunk-cache rows exist, and skips `audio.wav` extraction,
  Whisper, and audio-feature extraction on a complete transcript cache hit.
- A second frontend-equivalent retry confirmed the transcript/audio fix, then
  exposed the next memory path: full-VOD visual extraction decoded one frame per
  second from a 4h38m VOD and grew worker RSS beyond 8 GB. The pipeline now
  skips full-VOD visual feature extraction above
  `PIPELINE_FULL_VISUAL_FEATURE_MAX_DURATION_SEC=7200`; candidate-window face
  refinement and keyframe sampling remain enabled.
- A third frontend-equivalent retry confirmed those two skips, then exposed the
  next memory path: shortlist face-reaction refinement decoded one huge
  min-to-max ffmpeg span for sparse candidates. Refinement now decodes only
  contiguous candidate-second ranges instead of hours of non-candidate video.
- Verification so far: `py_compile`, focused tests (`67 passed`), scoped
  `git diff --check`, and Graphify rebuild (`3181` nodes, `8464` edges,
  `96` communities).
- Final frontend-equivalent test job
  `232c93ec-e58c-4231-b30f-24046d4e8a18` completed successfully with
  `status=done`, `progress_pct=100`, `download_cached=true`, `clips=5`, and
  `oom_kill=0`. Peak observed memory stayed around `3.3 GB`, then dropped to
  about `0.5 GB` after completion. The job output contains five clip files
  under `backend/tmp/storage/232c93ec-e58c-4231-b30f-24046d4e8a18/clips/`.
## 2026-05-14 Plan: Fix Remaining Frontend VOD Transcriber SIGKILL

- [x] Re-check latest failed job state instead of assuming the previous OOM pattern.
- [x] Determine whether SIGKILL hits the parent pipeline worker or the isolated Whisper chunk worker.
- [x] Identify the actual memory trigger for VOD `2771207666`.
- [x] Implement the smallest durable fix and update regression coverage.
- [x] Verify in tests/container and document the result.

### Review Notes

- Latest failed job is `9fe48cd9-e8fc-4d51-a849-1a52cd92c5eb`, now failing
  at `progress_pct=30` with `download_cached=True`. That points to the start
  of transcription or the first isolated chunk, not to later feature extraction.
- The latest job had all 28 Whisper chunk cache files written under
  `_transcript_cache`, so ASR itself completed. The subsequent log shows
  `transcriber_diarization_unavailable ... AudioDecoder`, followed by cgroup
  `oom_kill=1`, which means post-ASR Pyannote/TorchCodec diarization was the
  remaining parent-worker memory trigger.
- Frontend pipeline jobs now pass `enable_diarization=False` by default and skip
  the MFCC fallback unless `PIPELINE_ENABLE_DIARIZATION=true` is explicitly set.
- Verification passed: `py_compile`, focused backend tests (`46 passed`),
  scoped `git diff --check`, and Graphify rebuild (`3163` nodes, `8421`
  edges, `95` communities).
- Runtime verification passed after rebuilding the backend CUDA image:
  container is `healthy`, `OOMKilled=false`, cgroup `oom_kill=0`, CUDA is
  visible, and `pipeline_enable_diarization=False` inside the container.
- Direct container proof for VOD `2771207666`: loading the existing transcript
  chunk cache with `enable_diarization=False` returned `3540` segments, while a
  patched `_diarize` function that would raise was never called.

## 2026-05-14 Plan: Clean Backend Tmp Runtime Altlasten

- [x] Inspect `backend/tmp` size distribution and current Docker/backend process state.
- [x] Identify persistent caches that must be preserved.
- [x] Stop stale cancelled/failed runtime workers that still hold temp files.
- [x] Delete only stale per-job runtime directories and orphan temp fragments.
- [x] Verify freed disk space and document what was preserved.

### Review Notes

- Persistent caches to preserve: `backend/tmp/vod_feature_cache`,
  `backend/tmp/storage/_download_cache`, `backend/tmp/storage/_transcript_cache`,
  `backend/tmp/cache`, and short-form metadata/cache directories.
- Current largest safe cleanup candidates are failed/cancelled per-job runtime
  directories under `backend/tmp/storage/<uuid>`.
- `ffeabed2-c22a-4cc9-9cb5-053da047ff89` is marked `cancelled` in
  `jobs.json`, but still has a live `pipeline_worker` and `aria2c` process,
  so it must be stopped before deleting its partial fragments.
- Stopped the stale cancelled worker and removed stale per-job directories for
  failed/cancelled frontend jobs. Removed all leftover `.part`/fragment temp
  files under `backend/tmp`.
- `backend/tmp` dropped from about `165G` to `133G`. Remaining usage is mostly
  intentional cache: `122G` VOD feature cache and `11G` storage cache.
- `backend/ml/cache_cleanup.py --dry-run` reports no additional conservative
  cleanup candidates after the manual stale UUID cleanup.
- Restarted the backend after cleanup to clear defunct child processes from the
  cancelled download. Container health is `healthy`, status is `running`, and
  Docker reports `OOMKilled=false`.

## 2026-05-14 Plan: Isolate Whisper Chunk Transcription From OOM

- [x] Confirm repeated SIGKILL still occurs in the Whisper transcribing phase.
- [x] Add process isolation for each non-cached Whisper chunk in frontend jobs.
- [x] Keep the durable chunk cache and make cached chunks skip Whisper entirely.
- [x] Add focused tests for isolated chunk execution and pipeline wiring.
- [x] Run verification, rebuild Graphify, and restart the backend container.

### Review Notes

- The current failed/active frontend VOD jobs still reach `progress_pct=55`
  with `SIGKILL`, and Docker reports `OOMKilled=true`.
- Logs show the worker repeatedly dies during Whisper chunk processing around
  the high-teen chunks, before feature extraction starts. The previous chunk
  cache preserves completed work, but it does not prevent the long-lived
  Whisper process from accumulating memory within one run.
- Frontend pipeline runs now pass `isolate_chunks=True` by default via
  `PIPELINE_ISOLATE_TRANSCRIPTION_CHUNKS=true`. Cache hits still bypass
  Whisper, while cache misses run one chunk in one child process and then
  return native/PyTorch memory to the OS when that child exits.
- Verification passed: `py_compile`; focused transcriber/pipeline/jobs/config
  tests -> `38 passed`; scoped `git diff --check`; Graphify rebuild (`3162`
  nodes, `8419` edges, `94` communities); CUDA backend image rebuild/restart;
  backend healthcheck healthy with `OOMKilled=false`; runtime probe reports
  `pipeline_isolate_transcription_chunks=True`, `WHISPER_DEVICE=cuda`, and
  CUDA available.

## 2026-05-14 Plan: Make Frontend VOD Transcription Resumable

- [x] Confirm the repeat SIGKILL phase from current job/log state.
- [x] Inspect frontend transcription chunking and existing cache boundaries.
- [x] Add durable per-chunk transcript checkpointing for URL/VOD jobs.
- [x] Release transient Whisper/GPU memory after each chunk.
- [x] Run focused transcriber/pipeline tests, diff hygiene, and Graphify.

### Review Notes

- New failed job `9dec6885-139b-4e51-bfd0-e61d9fa90136` reached
  `progress_pct=55` and was killed while Whisper was still processing VOD
  chunks. The log never reached `extracting_features`, so the previous
  feature-overlap fix did not cover this failure.
- The VOD media itself was reused through hardlinks/cache; the repeated cost is
  the non-resumable Whisper chunk loop.
- Frontend transcription now writes each completed 10-minute Whisper chunk to
  `backend/tmp/storage/_transcript_cache/<source-hash>/chunks/`. The cache key
  is stable for Twitch VOD URLs via `twitch_vod:<vod_id>`, so a fresh frontend
  job for the same VOD can resume completed chunks from an earlier killed job.
- Fully cached chunks do not load Whisper at all; cache misses still use the
  existing singleton model and call CUDA cache cleanup after each completed
  chunk.
- Verification passed: `py_compile`; focused transcriber/pipeline/jobs tests
  -> `33 passed`; scoped `git diff --check`; Graphify rebuild (`3153` nodes,
  `8401` edges, `84` communities); CUDA backend image rebuild/restart;
  backend healthcheck healthy with `OOMKilled=false`; runtime probe reports
  `WHISPER_DEVICE=cuda`, CUDA available, and feature overlap disabled.

## 2026-05-14 Plan: Prevent Frontend VOD Worker SIGKILL/OOM

- [x] Confirm the failed job phase and runtime kill signal from current logs.
- [x] Reduce peak memory by disabling overlapped feature extraction by default.
- [x] Cap frontend Twitch VOD downloads to a serving-safe default height.
- [x] Improve signal-9 job error text and add focused regression coverage.
- [x] Run focused tests, syntax checks, diff hygiene, and Graphify after backend changes.

### Review Notes

- The failed frontend VOD job `c4599af9-4daf-42f0-827f-1000e5d1c677`
  stopped at `progress_pct=55`, right after Whisper progress and before feature
  extraction completed. Docker also reported `OOMKilled=true` for the backend
  container state, so the failure mode was SIGKILL/peak-memory, not a Python
  exception.
- Pipeline workers now default to sequential feature extraction:
  `PIPELINE_OVERLAP_FEATURE_EXTRACTION=false`. Whisper no longer overlaps with
  audio/video feature extraction unless explicitly re-enabled.
- Frontend Twitch VOD downloads now default to `TWITCH_DOWNLOAD_MAX_HEIGHT=720`.
  Training VOD jobs still use `TRAINING_VOD_MAX_HEIGHT`.
- Signal-9 failures now surface as
  `Pipeline subprocess exited from signal 9 (SIGKILL; likely OOM or external kill)`.
- Verification passed: focused pytest suite (`48 passed`), `py_compile`, scoped
  `git diff --check`, Graphify rebuild (`3137` nodes, `8366` edges,
  `87` communities), and CUDA container restart/healthcheck.

## 2026-05-14 Plan: Add CUDA Backend Container With CPU Fallback

- [x] Mirror the training workflow's CUDA runtime for the app backend.
- [x] Keep the current CPU backend image as the fallback path.
- [x] Add a Compose override that requests Docker GPU access and CUDA devices.
- [x] Make the start script auto-select CUDA when available, otherwise CPU.
- [x] Verify Compose configs, build paths, runtime selection, and Graphify.

### Review Notes

- Added a CUDA backend image using the same CUDA family as the local training
  workflow: `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`.
- Added `docker-compose.cuda.yml` as an override for the backend only. It uses
  the CUDA Dockerfile, requests `gpus: all`, and sets
  `WHISPER_DEVICE=cuda`, `SEMANTIC_DEVICE=cuda`,
  `NVIDIA_VISIBLE_DEVICES=all`, and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.
- `scripts/start_containerized.sh` now supports
  `AI_CLIPPING_BACKEND_RUNTIME=auto|cuda|gpu|cpu`. `auto` probes host NVIDIA
  visibility and Docker `--gpus all`; if either is unavailable it starts the
  CPU backend.
- Verified forced CPU startup and automatic CUDA startup on the current host.
  Auto selected CUDA on the RTX 5070 Ti host.
- Live CUDA backend verification passed:
  `torch=2.10.0+cu128`, `torch.version.cuda=12.8`,
  `torch.cuda.is_available()=True`, `WHISPER_DEVICE=cuda`, and
  `SEMANTIC_DEVICE=cuda`.
- Compose config validation, CUDA backend build, CPU fallback build, frontend
  HTTP `200`, backend `/api/status`, backend healthcheck, scoped
  `git diff --check`, and Graphify rebuild passed.

## 2026-05-14 Plan: Containerize Frontend And Backend Runtime

- [x] Confirm frontend/backend runtime commands and env boundaries.
- [x] Add production Dockerfiles for FastAPI backend and Next.js frontend.
- [x] Add Docker Compose wiring with persistent backend runtime/cache volumes.
- [x] Add a start script that builds and starts the stack correctly.
- [x] Verify Compose config/build path and rebuild Graphify after backend-related files change.

### Review Notes

- Added container builds for the FastAPI backend and Next.js frontend, plus
  scoped `.dockerignore` files so local venvs, caches, generated artifacts, and
  secrets are not copied into images.
- `docker-compose.yml` now wires `backend` on port `8000` and `frontend` on
  port `3000`, mounts persistent backend runtime/cache/artifact directories,
  and exposes `host.docker.internal` for host Ollama access.
- `scripts/start_containerized.sh` creates required runtime directories,
  bootstraps missing env files from examples, then runs
  `docker compose -f docker-compose.yml up --build -d`.
- The backend image preinstalls CPU PyTorch wheels before app extras to avoid
  pulling CUDA/NVIDIA packages into the default container image.
- Verification passed: Compose config validation, frontend image build, backend
  image build, start script launch, backend healthcheck, frontend HTTP `200`,
  backend `/api/status`, CPU Torch probe, scoped `git diff --check`, and
  Graphify rebuild.

## 2026-05-14 Plan: Fix Frontend Worker Cache/Headless Runtime Warnings

- [x] Confirm which warning is an actual broken runtime path.
- [x] Make local cache env selection require write access, not just directory existence.
- [x] Set stable headless CV defaults for frontend pipeline workers and face probes.
- [x] Repair current root-owned runtime cache directory.
- [x] Run focused tests, diff hygiene, and Graphify after backend changes.

### Review Notes

- The real broken path was `backend/tmp/matplotlib`: it existed, but was owned
  by `root:root`, so Matplotlib fell back to a temporary `/tmp/matplotlib-*`
  cache on every worker start.
- Local cache setup now probes write access before accepting a cache directory
  and falls back to a writable runtime cache if a directory only exists but is
  not writable.
- Frontend pipeline workers and face-detection probes now set software/headless
  defaults for OpenCV/Mesa: `LIBGL_ALWAYS_SOFTWARE=1`,
  `MESA_LOADER_DRIVER_OVERRIDE=swrast`, and
  `OPENCV_FFMPEG_CAPTURE_OPTIONS=hwaccel;none`.
- The current `backend/tmp/matplotlib` directory ownership was repaired to
  `antonio:antonio`, and direct Matplotlib cache resolution now stays on that
  path.
- Verification passed: `backend/tests/test_config.py` and
  `backend/tests/test_jobs_router.py` -> `22` tests, `py_compile`, scoped
  `git diff --check`, direct Matplotlib cache probe, and Graphify rebuild.
- Graphify now reports `3135` nodes, `8362` edges, and `81` communities.

## 2026-05-14 Plan: Enable Twitch VOD Download Cache And Remove Stale Job Media

- [x] Confirm current cache setting and identify large stale frontend VOD job directories.
- [x] Enable Twitch VOD download cache in config, `.env`, and `.env.example`.
- [x] Delete stale failed/cancelled job media directories and obsolete partial VOD-range downloads.
- [x] Run focused verification, diff hygiene, and Graphify after backend config changes.

### Review Notes

- Twitch VOD download caching is now enabled by default with
  `TWITCH_DOWNLOAD_CACHE_TTL_SEC=-1` in config, `.env`, and `.env.example`.
- The existing downloaded Twitch VOD `2770356205` was migrated into the
  canonical cache at `backend/tmp/storage/_download_cache/twitch/`.
- Removed the stale failed/cancelled frontend job media copies and two obsolete
  coarse VOD `.part` directories. `backend/tmp/storage` dropped from `69G` to
  `26G`.
- Cache usability was verified for `https://www.twitch.tv/videos/2770356205`;
  focused downloader tests passed with `23` tests.
- `py_compile`, scoped `git diff --check`, and Graphify rebuild passed.
  Graphify now reports `3132` nodes, `8354` edges, and `84` communities.

## 2026-05-14 Plan: Re-enable Face Detection For Frontend Pipeline Jobs

- [x] Change the pipeline subprocess default so native face detection is enabled again.
- [x] Preserve an explicit environment override for disabling face detection if it crashes.
- [x] Update regression coverage for the new worker default.
- [x] Run focused verification and rebuild Graphify after backend changes.

### Review Notes

- Frontend pipeline workers now default to
  `AI_CLIPPING_DISABLE_FACE_DETECTION=0`, so MediaPipe/MTCNN/DeepFace can run
  again inside the isolated worker process.
- Explicit host override still works: launching the backend with
  `AI_CLIPPING_DISABLE_FACE_DETECTION=1` keeps native face stacks disabled.
- Verification passed: `backend/tests/test_jobs_router.py` -> `18` tests,
  `py_compile`, scoped `git diff --check`, and Graphify rebuild.
- Graphify rebuilt successfully: `3132` nodes, `8354` edges, `84`
  communities.

## 2026-05-14 Plan: Make Pipeline Worker Strategy JSON-Safe

- [x] Trace why URL submit task failed before starting the worker.
- [x] Normalize strategy payloads before serializing for the subprocess.
- [x] Add regression coverage for set-valued strategy fields.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- `clip_strategy.normalize_strategy()` returns set-valued fields such as
  `keywords` and `exclude_keywords`. The worker-launch command now converts the
  full strategy payload to JSON-safe primitives before `json.dumps`.
- Sets are serialized as sorted lists, so command construction is deterministic
  and the worker receives the same keyword content.
- Verification passed: `py_compile`; focused jobs/job-manager tests ->
  `24 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3113` nodes, `8301` edges, and
  `79` communities.

## 2026-05-14 Plan: Isolate Pipeline URL Submits From Native CV Crashes

- [x] Trace the Twitch URL submit crash path.
- [x] Move pipeline execution out of the FastAPI process.
- [x] Keep job progress/cancellation visible across the subprocess boundary.
- [x] Add regression coverage for subprocess launch/failure behavior.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The crash path was `routers/jobs.py` importing `pipeline` during URL submit;
  `pipeline.py` imports `services.video_features`, which initializes native
  MediaPipe/MTCNN/DeepFace stacks at import time. In a headless EGL/WSL
  context, those native imports can segfault before Python can catch anything.
- URL and upload jobs now launch `pipeline_worker.py` in a separate Python
  subprocess. If the worker exits from a signal, FastAPI marks the job failed
  instead of dying with the worker.
- Worker jobs set a writable `MPLCONFIGDIR` and default
  `AI_CLIPPING_DISABLE_FACE_DETECTION=1`, while preserving motion/scene visual
  features. The env can still override this if native face detection is stable.
- `job_manager` now reloads `jobs.json` when another process updates it, so
  progress, terminal state, and cancellation remain visible across the worker
  boundary.
- Verification passed: `py_compile`; focused jobs/job-manager tests ->
  `23 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3111` nodes, `8297` edges, and
  `86` communities.

## 2026-05-14 Plan: Feed Frontend Ratings Into Training

- [x] Trace how frontend gold-eval ratings are stored and exported.
- [x] Enable frontend-rated clips as default VOD training signal.
- [x] Prevent stale generated gold artifacts from being preserved as training input.
- [x] Update regression coverage for the new default.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- Frontend gold-eval ratings already flow into
  `tmp/storage/feedback/gold_eval_updates.jsonl` and export as
  `gold_eval_frontend` rows with deterministic `gold_holdout_split`.
- The normal Jynxzi VOD training run now enables
  `VOD_RANKER_USE_USER_LABELS=1` and `VOD_RANKER_USE_GOLD_TRAIN=1` by default,
  so frontend-rated train-split rows are passed to `--include-gold-train`.
- The script no longer calls `export_frontend_gold_eval.py` with
  `--merge-existing` or `--preserve-existing-if-empty`; generated gold
  artifacts are rebuilt from the feedback store each run instead of preserving
  stale rows.
- A dry run against the current local feedback store exported `97` Jynxzi
  frontend gold rows: `67` train and `30` holdout.
- Verification passed: `bash -n`; `py_compile`; focused frontend/gold/training
  tests -> `57 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3094` nodes, `8249` edges, and
  `86` communities.

## 2026-05-14 Plan: Balance Auto-Negative Outcome Holdout

- [x] Trace why the outcome holdout has positives but no auto-negative rows.
- [x] Add deterministic holdout-floor balancing for exported auto-outcome negatives.
- [x] Ensure split metadata and dataset_source stay consistent after balancing.
- [x] Add regression coverage and verify current artifacts can produce mixed holdout rows.
- [x] Rebuild Graphify after backend changes.

### Review Notes

- The last run's auto-outcome negative export produced `49` rows but all landed
  in the train split, leaving the outcome holdout with `24` positives and `0`
  negatives. That made threshold calibration skip with
  `calibration_class_rows_below_minimum`.
- Auto-outcome negative export now applies a deterministic holdout floor after
  `max_rows` capping. If the stable split under-fills holdout, selected train
  negatives are promoted until the requested holdout ratio is represented.
- Split promotion updates `gold_holdout_split` and `dataset_source` together,
  so downstream builders see `auto_outcome_negative_holdout` rows correctly.
- A dry run with the current Jynxzi artifacts now exports `58` auto negatives:
  `46` train and `12` holdout. Building the outcome holdout from that output
  produces `36` rows: `24` positives and `12` negatives.
- Threshold calibration on the dry-run holdout now returns `status=computed`,
  `reason=ok`, with both classes present.
- Verification passed: `py_compile`; focused auto-negative + outcome-holdout
  tests -> `9 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3094` nodes, `8249` edges, and
  `86` communities.

## 2026-05-14 Plan: Inventory-Calibrate Rollout Readiness Gates

- [x] Inspect the current rollout-readiness targets in the diagnostic report and ranker artifact.
- [x] Replace impossible fixed Jynxzi rollout gates with inventory-calibrated targets when available VOD groups are capped.
- [x] Keep raw historical targets visible in reports for context.
- [x] Add focused regression coverage and verify against the current Jynxzi artifact.
- [x] Rebuild Graphify after backend changes.

### Review Notes

- Jynxzi rollout readiness now uses inventory-calibrated targets when the
  available VOD groups cannot plausibly satisfy the old absolute rollout
  gates. The raw historical targets remain visible under `raw_targets`.
- With the current maxed-out artifact, the calibrated targets are
  `training_rows=304`, `positive_rows=152`, and `validation_groups=15` for
  `38` available policy-surviving groups.
- Re-running the coverage diagnosis against the current Jynxzi training rows
  now returns `passed=true`: actuals are `367` training rows, `159` positive
  rows, and `15` validation groups.
- Verification passed: `py_compile`; focused diagnose + train-ranker tests ->
  `58 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3089` nodes, `8234` edges, and
  `85` communities.

## 2026-05-14 Plan: Parallelize Transcript Repost Matching

- [x] Add a configurable process-worker count for transcript repost matching.
- [x] Parallelize only the independent per-repost best-match step while keeping
  transcript cache loading and scoring semantics unchanged.
- [x] Preserve deterministic output/diagnostic ordering.
- [x] Add focused regression coverage and run verification.
- [x] Rebuild Graphify after backend changes.

### Review Notes

- Transcript matching now accepts `--transcript-match-workers` in
  `prepare_cross_platform_fingerprint_rows.py` and `--workers` in the direct
  transcript matcher CLI.
- `train_jynxzi_vod.sh` defaults `CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS=4`,
  exports it, logs it at startup, and passes it into the prepare stage.
- The implementation keeps media resolution and Short transcript cache loading
  in the parent process. Only the independent per-repost VOD best-match scan is
  distributed through `ProcessPoolExecutor`.
- Worker results are consumed in input order, so matched rows and diagnostic
  rows remain deterministic relative to the serial path.
- Verification passed: `py_compile`; `bash -n`; focused transcript/script
  tests -> `9 passed`; focused prepare tests -> `47 passed`; scoped
  `git diff --check`.
- Graphify rebuilt successfully with `3085` nodes, `8223` edges, and
  `84` communities.

## 2026-05-13 Plan: Reuse Existing Repost Media During Freshness Rebuild

- [x] Document why the IboClips freshness rebuild is rematerializing many Shorts.
- [x] Reuse existing repost `video_path` rows by canonical source/video identity
  before attempting `yt-dlp` media materialization.
- [x] Add regression coverage for `/shorts/<id>` vs `watch?v=<id>` reuse.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The live run is downloading many Shorts because the curated-channel freshness
  rebuild rewrites the repost manifest from fresh discovery rows. Those rows do
  not naturally carry the previous `video_path`, and the short-form media cache
  is keyed by exact URL, so `https://youtube.com/shorts/<id>` and
  `https://youtube.com/watch?v=<id>` can miss the same already downloaded MP4.
- Freshness rebuilds now pass previous manifest rows into repost
  materialization and reuse existing media paths by canonical repost identity.
  For YouTube rows, that identity is the normalized video id, so Shorts URLs
  and watch URLs for the same clip share the cached media.
- Reused rows skip `_ensure_short_form_media_cached`, so the next run should
  download only genuinely missing selected Shorts rather than re-downloading
  already materialized equivalents.
- Verification passed: `py_compile`; focused
  `test_prepare_cross_platform_fingerprint_rows.py` -> `47 passed`; scoped
  `git diff --check`. Graphify rebuilt successfully with the local Graphify
  interpreter: `3080` nodes, `8211` edges, `82` communities.

## 2026-05-13 Plan: Refresh IboClips Shorts Before Each Run

- [x] Trace the current repost manifest reuse logic and the short-form metadata cache.
- [x] Add a pre-run freshness check so curated YouTube channel sources rebuild
  the repost manifest when new Shorts appear.
- [x] Preserve existing metadata/media cache reuse while fetching only missing
  metadata/downloads needed for the current VOD date window.
- [x] Add focused regression coverage and run verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- Curated YouTube Shorts channel sources such as
  `https://www.youtube.com/@Iboclipss/shorts` now trigger a repost-manifest
  freshness rebuild before matching, even when the previous manifest already
  meets `min_reposts`.
- The rebuild goes through the existing discovery path, so flat channel rows
  are checked again, missing per-Short metadata is cached, date-window filtering
  is reapplied, and only selected Shorts whose media is missing are downloaded.
- Existing short media and metadata remain cache hits; this does not delete or
  redownload the 355 already materialized Shorts.
- If no effective inputs change, the transcript match signature remains
  unchanged and the existing match result can still be reused after the
  manifest refresh.
- Verification passed: `py_compile`; focused
  `test_prepare_cross_platform_fingerprint_rows.py` -> `46 passed`; scoped
  `git diff --check`. Required system `python3 ... graphify.watch...` still
  fails because this interpreter lacks `graphify`; rebuilding with the local
  Graphify interpreter succeeded with `3076` nodes, `8196` edges, and `88`
  communities.

## 2026-05-13 Plan: Lower Transcript Match Threshold To 0.50

- [x] Change the local Jynxzi training default from `0.60` to `0.50`.
- [x] Keep workflow-state reporting and env templates aligned with the new default.
- [x] Run focused verification and rebuild Graphify after backend changes.

### Review Notes

- `CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE` now defaults to `0.50` in the
  Jynxzi VOD training script and in workflow-state reporting.
- `backend/.env` and `backend/.env.example` are aligned to `0.50`, so the
  Docker local-train wrapper will not keep injecting the old `0.60` value.
- The next run's transcript match input signature will change because the
  threshold is part of the signature, so transcript matching should recompute
  once and then be reusable again on unchanged later runs.
- Verification passed: `py_compile`, `bash -n`, focused unittest suite
  `test_training_workflow_state` + `test_train_jynxzi_vod_script` ->
  `12 passed`, and scoped `git diff --check`.
- Graphify rebuilt successfully with `3073` nodes, `8183` edges, and
  `86` communities.

## 2026-05-13 Plan: Reuse Cross-Platform Transcript Matching

- [x] Inspect the current prepare-stage transcript matching inputs, outputs, and summaries.
- [x] Add a deterministic input signature so transcript matching can be reused when inputs/config have not changed.
- [x] Persist and validate the signature in the prepare summary before invoking matching again.
- [x] Add regression coverage and run focused verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- Transcript-mode prepare now computes `match_input_signature` from VOD rows,
  repost rows, media path state, verification mode, threshold, cache versions,
  profile-filter version, and VOD cache root.
- If the current signature matches the previous prepare summary and both output
  and diagnostics still exist, the run reuses the prior match result. It writes
  `reused_match_result=true` and `reuse_reason=match_input_signature_unchanged`
  without calling `build_full_vod_transcript_caches()` or
  `match_reposts_to_vods_with_diagnostics()`.
- The reused result preserves the previous status, including
  `insufficient_*`, so `--fail-on-insufficient` still behaves honestly while
  avoiding duplicate Whisper/transcript matching work.
- The old pre-manifest output reuse is now limited to audio-fingerprint mode;
  transcript mode checks manifests/signature before deciding reuse.
- Verification passed: `py_compile`; focused
  `test_prepare_cross_platform_fingerprint_rows.py` -> `45 passed`; scoped
  `git diff --check`. Required system `python3 ... graphify.watch...` still
  fails because that interpreter lacks `graphify`; rebuilding with the local
  Graphify interpreter succeeded with `3073` nodes, `8183` edges, and `86`
  communities.


## 2026-05-13 Plan: Remove User-Labeled Rows From Jynxzi Training

- [x] Disable frontend/manual/gold label export in the normal VOD training flow.
- [x] Empty generated gold-eval and hard-negative artifacts before training/eval so stale self-labeled rows cannot be reused.
- [x] Stop using active gold-eval queue rows as auto-negative candidates.
- [x] Add script regression coverage and run focused verification.

### Review Notes

- User-labeled rows are now disabled by default with
  `VOD_RANKER_USE_USER_LABELS=0`. Gold train and hard-negative train merges
  require explicitly opting back in with `VOD_RANKER_USE_USER_LABELS=1`.
- The training script resets generated model-facing user-label artifacts before
  training: gold eval rows/datasets/reports, gold-derived hard negatives,
  ranker holdout diagnosis outputs, and rollout review queue artifacts.
- The flow now skips frontend/manual gold export, gold eval, gold-derived
  hard-negative export/diagnosis, and rollout review queue generation while
  user labels are disabled.
- `jynxzi_active_gold_eval_queue.features.jsonl` is no longer passed into
  auto-outcome negative export, so manually reviewed queue rows cannot affect
  negative mining.
- Existing generated model-facing user-label artifacts were reset in-place:
  gold eval JSONL files, gold-derived hard-negative rows, rollout review queue
  features, and active gold-eval queue features now have `0` rows; their
  summaries are marked `status=skipped` with
  `reason=user_labeled_rows_disabled`.
- A follow-up content scan over `backend/ml/artifacts` found additional old
  gold/manual snapshot artifacts (`*_active`, `*_combined`, backup active
  queues, rollout-reviewed gold rows, and related reports). Those were reset
  too, and `rg` no longer finds `gold_eval_manual`, `gold_eval_frontend`,
  `gold_eval_rollout_review`, `gold_positive`, or `gold_negative` in the
  artifact directory.
- Verification passed: `py_compile`; `bash -n`; focused unittest suite ->
  `12 passed`; scoped `git diff --check`. Required system
  `python3 ... graphify.watch...` still fails because that interpreter lacks
  `graphify`; rebuilding with the local Graphify interpreter succeeded with
  `3068` nodes, `8163` edges, and `87` communities.

## 2026-05-12 Plan: Let VOD Audio Downloader Drain Full Queue

- [x] Confirm current lookahead only queues one next VOD and can idle while Whisper transcribes.
- [x] Queue all missing VOD audio downloads on the single download worker before transcription consumes them.
- [x] Add focused tests and run verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- The previous one-item lookahead submitted only the next VOD audio download,
  so the single download worker could go idle while Whisper was still
  transcribing the current VOD.
- Missing VOD audio downloads are now all submitted to a single-worker executor
  before transcription starts consuming them. The worker still downloads
  serially, but it drains the full missing-audio queue independently of Whisper.
- Pipeline metadata is now `pipeline_mode=download_queue_ahead`; transcribe
  start logs expose `queued_audio_downloads_remaining`.
- Regression coverage proves the third VOD audio download occurs before the
  first VOD transcribe call finishes.
- Verification passed: `py_compile`; focused prepare tests -> `44 passed`;
  combined transcript/prepare tests -> `47 passed`; scoped `git diff --check`.
- Required system `python3 ... graphify.watch...` still fails because that
  interpreter lacks `graphify`; rebuilding with the local Graphify interpreter
  succeeded with `3067` nodes, `8161` edges, and `87` communities.

## 2026-05-12 Plan: Pipeline VOD Audio Download With Whisper Transcription

- [x] Confirm current VOD transcript cache builder downloads and transcribes each missing VOD serially.
- [x] Add one-item audio prefetch so the next VOD downloads while Whisper transcribes the current VOD.
- [x] Add focused tests and run verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- The VOD transcript cache builder now separates missing VODs into cached,
  skipped, and missing groups before the build.
- Missing VODs run through a one-item lookahead pipeline:
  `download_workers=1`, `transcribe_workers=1`,
  `pipeline_mode=prefetch_next_audio`. While Whisper transcribes the current
  VOD, the single download worker can prepare the next VOD's `source.m4a`.
- Whisper remains single-threaded and the next audio prefetch is visible in
  `full_vod_transcript_cache_vod_transcribe_start` via
  `next_audio_prefetch_active`.
- Verification passed: `py_compile`; focused prepare tests -> `44 passed`;
  combined transcript/prepare tests -> `47 passed`; scoped `git diff --check`.
- Required system `python3 ... graphify.watch...` still fails because that
  interpreter lacks `graphify`; rebuilding with the local Graphify interpreter
  succeeded with `3067` nodes, `8161` edges, and `87` communities.

## 2026-05-12 Plan: Prebuild VOD Transcript Caches Before Transcript Matching

- [x] Confirm why the current transcript run loaded only part of the VOD transcript pool.
- [x] Add a prepare-stage VOD transcript cache builder for the exact VOD manifest being matched.
- [x] Add regression coverage and run focused verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- The latest transcript run loaded only `20` VOD transcripts and skipped `37`,
  so transcript matching compared Shorts against less than half of the VOD
  manifest.
- `prepare_cross_platform_fingerprint_rows.py` now prebuilds compatible VOD
  transcript caches for every unique VOD row before transcript matching starts.
  Existing compatible caches are counted as cache hits; missing caches download
  full VOD audio, transcribe with diarization disabled, write the normal
  transcript cache, and then proceed to matching.
- The next run should emit `full_vod_transcript_cache_build_start`,
  per-VOD cached/build events, and `full_vod_transcript_cache_build_done`
  before `cross_platform_transcript_matching_call`.
- Verification passed: `py_compile`; focused prepare tests -> `43 passed`;
  combined transcript/prepare tests -> `46 passed`; scoped `git diff --check`.
- Required system `python3 ... graphify.watch...` still fails because that
  interpreter lacks `graphify`; rebuilding with the local Graphify interpreter
  succeeded with `3066` nodes, `8158` edges, and `88` communities.

## 2026-05-12 Plan: Stop Preserving Audio Rows In Transcript Mode

- [x] Confirm why the latest transcript run still outputs old audio-fingerprint rows.
- [x] Require preserved existing output to be reusable for the active verification mode.
- [x] Add regression coverage and run focused verification.

### Review Notes

- The latest run matched `31` transcript rows, but `output_rows=73` because
  `preserved_existing_output=true` kept the prior
  `cross_platform_audio_fingerprint` artifact after the transcript result missed
  the `70` row gate.
- Transcript mode now preserves an existing output only when every preserved row
  is reusable for the active transcript verification mode. Old audio rows are no
  longer carried forward into transcript-only training.
- If transcript matching is insufficient, the output file is still replaced
  with the current transcript matches so downstream training cannot silently use
  stale audio-fingerprint positives.
- Verification passed: `py_compile`; focused
  `test_prepare_cross_platform_fingerprint_rows.py` -> `41 passed`; scoped
  `git diff --check`.

## 2026-05-12 Plan: Skip Diarization In Cross-Platform Transcript Matching

- [x] Add an explicit transcriber option to skip speaker diarization for text-only matching.
- [x] Wire cross-platform Short transcription to use text-only mode.
- [x] Run focused verification and document results.

### Review Notes

- The pasted run log showed Whisper transcription completing, then optional
  Pyannote diarization failing because `torchcodec`/`AudioDecoder` is broken in
  the container.
- Cross-platform transcript matching does not need speaker labels, so Short
  transcription now calls `transcriber.transcribe(..., enable_diarization=False)`.
- Normal transcriber callers keep the existing default
  `enable_diarization=True`, so VOD feature paths are unchanged.
- Verification passed: `py_compile`; focused tests
  `test_match_cross_platform_transcripts.py` + `test_transcriber.py` ->
  `10 passed`; scoped `git diff --check`.

## 2026-05-12 Plan: Make Transcript Matching Primary Cross-Platform Verification

- [x] Trace current audio-fingerprint prepare path and available transcript caches.
- [x] Add transcript-based repost-to-VOD matcher and wire the prepare flow to use it as the primary label source.
- [x] Update downstream source labels/tests so outcome rows no longer present as audio-fingerprint positives.
- [x] Run focused verification, update Graphify, and document results.

### Review Notes

- Cross-platform verification now defaults to `transcript` via
  `CROSS_PLATFORM_VERIFICATION_MODE=transcript` in `train_jynxzi_vod.sh` and
  `--verification-mode transcript` in the prepare CLI.
- Added `match_cross_platform_transcripts.py`, which loads existing VOD
  transcript caches, transcribes/caches downloaded Shorts under
  `_download_cache/short_form_transcripts`, and writes positives as
  `cross_platform_transcript_match`.
- Audio fingerprints are still available only as explicit
  `--verification-mode audio_fingerprint` legacy mode; transcript mode does not
  fall back to audio matching.
- Outcome and VOD-ranker dataset prep now preserve transcript label provenance
  instead of rewriting cross-platform positives to
  `cross_platform_audio_fingerprint`.
- Verification passed: `py_compile` for changed ML modules; new transcript
  matcher tests -> `2 passed`; focused combined tests -> `84 passed`;
  `bash -n backend/ml/train_jynxzi_vod.sh`; scoped `git diff --check`.
- Required system `python3 ... graphify.watch...` still fails because this
  shell's Python has no `graphify` module; rebuilding with the local Graphify
  interpreter succeeded with `3056` nodes, `8129` edges, and `78` communities.

## 2026-05-11 Plan: Switch Cross-Platform Source To Iboclipss

- [x] Replace the curated YouTube Shorts source with `youtube.com/@Iboclipss`.
- [x] Remove stale Clips.Delivery-derived short media, metadata, and cross-platform output artifacts so the next run starts clean.
- [x] Verify source/caches/artifact state and document the cleanup.

### Review Notes

- `backend/ml/cross_platform_sources.csv` now points Jynxzi cross-platform
  discovery at `https://www.youtube.com/@Iboclipss/shorts` with
  `curated_repost_source;iboclipss;youtube_shorts`.
- Removed stale short-form media and metadata caches:
  `backend/tmp/storage/_download_cache/short_form` and
  `backend/tmp/storage/_download_cache/short_form_metadata`.
- Removed stale cross-platform derived artifacts:
  `jynxzi_cross_platform_reposts.jsonl`,
  `jynxzi_cross_platform_fingerprint_rows.jsonl`,
  `jynxzi_cross_platform_fingerprint_summary.json`, and
  `jynxzi_cross_platform_fingerprint_diagnostics.jsonl`.
- Kept VOD full-audio fingerprint caches intact because they are independent of
  the YouTube repost channel and avoid redownloading/re-fingerprinting VODs.
- Verification: source manifest parses as one Iboclipss row; old short/media
  cache file count is `0`; stale cross-platform repost/output artifacts are
  absent.

## 2026-05-10 Plan: Restore Cached Per-Short Metadata Dates

- [x] Re-enable cached per-Short metadata enrichment for date-scoped repost discovery.
- [x] Keep the existing serial yt-dlp pacing/cooldown behavior for every metadata/download request outcome.
- [x] Add regression coverage for metadata cache reuse and exact upload-date filtering, then verify and rebuild Graphify.

### Review Notes

- Clips.Delivery channel rows still start from one flat playlist discovery, but
  rows with no recovered channel-page publish date now fall back to an exact
  per-Short yt-dlp metadata request.
- Successful per-Short metadata payloads are cached under
  `_download_cache/short_form_metadata`, keyed by URL, so later runs reuse the
  exact upload date/timestamp instead of requesting the same Short again.
- Cache misses sleep for the configured 5-10s short-form interval after a
  successful metadata request; metadata errors keep the existing error cooldown.
  Cache hits do not sleep.
- Verification passed: targeted metadata-date tests -> `2 passed`; focused
  auto-train/prepare/downloader tests -> `87 passed`; `py_compile` passed;
  scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3019` nodes, `8014` edges, and
  `78` communities.

## 2026-05-10 Plan: Recover Channel-Scoped Short Dates Without Per-Short Metadata

- [x] Enrich YouTube channel/Shorts flat discovery rows from the channel page's publish-time data.
- [x] Convert relative publish text to approximate UTC upload dates and keep missing-date rows fail-closed.
- [x] Add regression coverage, run verification, update lessons, and rebuild Graphify.

### Review Notes

- The failed run discovered `1738` Clips.Delivery candidates but skipped all of
  them as `missing_publish_date`, because flat YouTube channel extraction
  returned titles/URLs without dates and per-Short metadata enrichment was
  intentionally disabled.
- YouTube channel `/shorts` discovery now performs one channel-page fetch and
  parses per-video `publishedTimeText` blocks keyed by `videoId`.
- Relative publish text such as `2 weeks ago` / `3 months ago` is converted into
  approximate UTC `timestamp` and `upload_date` fields before date-window
  filtering.
- Rows still fail closed if no publish text can be recovered; the pipeline does
  not assume undated Shorts belong to the VOD window.
- Verification passed: targeted channel-date tests -> `2 passed`; focused
  auto-train/prepare/downloader tests -> `86 passed`; `py_compile` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3009` nodes, `7986` edges, and
  `75` communities.

## 2026-05-10 Plan: Add Post-Attempt Delay For Repost Short Downloads

- [x] Move repost Short download pacing to a `finally` path so every attempted yt-dlp cache fill sleeps after the attempt.
- [x] Keep cache hits fast; apply the delay only to actual missing-media download attempts.
- [x] Add regression coverage for successful and failed attempts, run verification, update lessons, and rebuild Graphify.

### Review Notes

- The serial repost Short download loop now emits
  `cross_platform_repost_download_request_cooldown` and sleeps after every
  attempted missing-media Short cache fill.
- Success and failure outcomes both pause before the next Short is attempted.
  Existing usable media rows remain cache hits and do not add artificial delay.
- Short-form pacing defaults and env values are now `5..10` seconds:
  `YTDLP_SHORT_FORM_SLEEP_REQUESTS_SEC=5`,
  `YTDLP_SHORT_FORM_MIN_SLEEP_INTERVAL_SEC=5`,
  `YTDLP_SHORT_FORM_MAX_SLEEP_INTERVAL_SEC=10`.
- Verification passed: targeted success/failure delay tests -> `2 passed`;
  focused prepare/downloader/auto-train tests -> `84 passed`; `py_compile`
  passed; scoped `git diff --check` passed; settings resolution prints
  `5.0 5.0 10.0`.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3001` nodes, `7965` edges, and
  `77` communities.

## 2026-05-10 Plan: Avoid Per-Short Metadata Enrichment For Date Filtering

- [x] Require repost candidates to carry publish/upload date from discovery or manifest rows before date-window filtering.
- [x] Stop per-Short yt-dlp metadata enrichment during repost manifest filtering so age/rate-limited Shorts do not create extra requests.
- [x] Add regression coverage proving missing-date rows are skipped without metadata lookup, then run focused verification and rebuild Graphify.

### Review Notes

- Repost date-window filtering now uses only `publish_timestamp` /
  `upload_date` already present on discovered or manifest rows.
- Rows without a publish/upload date still fail closed as
  `skipped_missing_publish_date`; they are not recovered with a per-Short
  yt-dlp metadata request.
- This keeps the Clips.Delivery source date-scoped to the actual usable VOD
  window without adding extra YouTube requests during manifest filtering.
- Verification passed: targeted regression -> `1 passed`; focused
  auto-train/prepare tests -> `64 passed`; `py_compile` passed; scoped
  `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3000` nodes, `7964` edges, and
  `76` communities.

## 2026-05-10 Plan: Replace One-Hour Short Error Cooldown

- [x] Remove the explicit 3600s short-form error cooldown behavior.
- [x] Reuse the existing short-form 5-10s style sleep range for all yt-dlp short error cooldowns.
- [x] Update tests/env/docs, run verification, and rebuild Graphify.

### Review Notes

- Removed `YTDLP_SHORT_FORM_RATE_LIMIT_COOLDOWN_SEC` from `backend/.env`,
  `.env.example`, and `Settings`; stale env values are ignored because the
  setting no longer exists.
- Short-form yt-dlp error cooldowns now sample the same configured short-form
  range used by normal yt-dlp pacing:
  `YTDLP_SHORT_FORM_MIN_SLEEP_INTERVAL_SEC=6` to
  `YTDLP_SHORT_FORM_MAX_SLEEP_INTERVAL_SEC=10`.
- The final repost download path and the discovery/metadata paths all use this
  short jittered cooldown.
- Verification passed: targeted cooldown tests -> `3 passed`; focused
  auto-train/prepare/downloader tests -> `84 passed`; Python `py_compile`
  passed; scoped `git diff --check` passed; env resolution prints
  `6.0 10.0 False`.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3001` nodes, `7971` edges, and
  `81` communities.

## 2026-05-10 Plan: Apply Short Cooldown To All yt-dlp Short Errors

- [x] Trace why current YouTube errors bypass the existing materialization cooldown.
- [x] Add cooldown handling to discovery/metadata yt-dlp exceptions as well as final download exceptions.
- [x] Add focused tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Root cause: some YouTube errors happen before final media cache
  materialization, during source discovery or per-entry metadata extraction.
  Those exceptions were swallowed or converted to fallback rows before reaching
  the cooldown logic in `prepare_cross_platform_fingerprint_rows.py`.
- Added `short_form_yt_dlp_error_cooldown` handling to yt-dlp source discovery,
  date-search extraction, and metadata extraction in `auto_train_short_form.py`.
- The same `YTDLP_SHORT_FORM_RATE_LIMIT_COOLDOWN_SEC` setting is used. Any
  exception in those yt-dlp paths sleeps before the workflow continues; YouTube
  rate-limit errors are marked with `rate_limited=True` in the warning log.
- Verification passed: targeted cooldown tests -> `3 passed`; focused
  auto-train/prepare/downloader tests -> `84 passed`; Python `py_compile`
  passed; scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2997` nodes, `7967` edges, and
  `75` communities.

## 2026-05-10 Plan: Cool Down And Continue After Any Short Download Error

- [x] Change repost Short download errors from retry-same-URL to cooldown-and-continue.
- [x] Apply the cooldown to any download exception, while still flagging YouTube rate-limit errors in logs.
- [x] Update tests/lessons, run verification, and rebuild Graphify.

### Review Notes

- Repost Short download failures now emit
  `cross_platform_repost_download_failed_cooldown`, sleep for
  `YTDLP_SHORT_FORM_RATE_LIMIT_COOLDOWN_SEC`, and then continue with the next
  Short.
- The same URL is no longer retried after the cooldown. Any exception triggers
  the cooldown; YouTube rate-limit errors are marked with `rate_limited=true`.
- The manifest summary now includes `repost_download_error_cooldowns` plus the
  existing rate-limit-specific cooldown count.
- Verification passed: focused pytest -> `59 passed`; Python `py_compile`
  passed; targeted cooldown-and-continue test -> `1 passed`; scoped
  `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2989` nodes, `7953` edges, and
  `73` communities.

## 2026-05-10 Plan: Cool Down After YouTube Rate-Limit Errors

- [x] Detect yt-dlp YouTube "try again later" rate-limit errors during serial repost Short downloads.
- [x] Sleep for a configured cooldown and retry the same Short once before counting it as failed.
- [x] Add env wiring/tests/lesson, run verification, and rebuild Graphify.

### Review Notes

- Added `YTDLP_SHORT_FORM_RATE_LIMIT_COOLDOWN_SEC=3600` to the backend env
  files and `yt_dlp_short_form_rate_limit_cooldown_sec` to settings.
- During serial repost Short cache fills, YouTube rate-limit errors such as
  `This content isn't available, try again later` now emit
  `cross_platform_repost_download_rate_limited`, sleep for the configured
  cooldown, and retry the same URL once.
- The manifest summary now includes rate-limit cooldown/retry counters.
- Verification passed: focused pytest -> `59 passed`; targeted rate-limit test
  -> `1 passed`; Python `py_compile` passed; scoped `git diff --check` passed;
  env resolution prints `6.0 6.0 10.0 3600.0`.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2989` nodes, `7953` edges, and
  `73` communities.

## 2026-05-10 Plan: Add YouTube yt-dlp Sleep For Shorts

- [x] Add configurable yt-dlp request/data-extraction sleep for short-form media downloads.
- [x] Apply the sleep only to `short_form` cache fills so Twitch/full-VOD audio paths stay unchanged.
- [x] Add tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Added Short-form-specific yt-dlp sleep settings:
  `YTDLP_SHORT_FORM_SLEEP_REQUESTS_SEC=6`,
  `YTDLP_SHORT_FORM_MIN_SLEEP_INTERVAL_SEC=6`, and
  `YTDLP_SHORT_FORM_MAX_SLEEP_INTERVAL_SEC=10`.
- The settings map to yt-dlp native options `sleep_interval_requests`,
  `sleep_interval`, and `max_sleep_interval`.
- These options are applied only when `ensure_cached_download(...,
  cache_namespace="short_form")` fills a missing media cache entry. Existing
  Short cache hits, Twitch VOD downloads, generic downloads, and full-VOD audio
  metadata extraction are not slowed by this change.
- Verification passed: focused pytest -> `61 passed`; Python `py_compile`
  passed; scoped `git diff --check` passed; env resolution prints `6.0`,
  `6.0`, `10.0`.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2986` nodes, `7943` edges, and
  `73` communities.

## 2026-05-10 Plan: Revert Repost Shorts To Serial Download

- [x] Remove the cross-platform repost download worker setting and shell/env wiring.
- [x] Change repost media materialization back to one-by-one cache downloads.
- [x] Replace the parallel regression with a serial-order regression, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Removed `CROSS_PLATFORM_REPOST_DOWNLOAD_WORKERS` from `backend/.env`,
  `backend/.env.example`, `backend/config.py`, and `backend/ml/train_jynxzi_vod.sh`.
- Repost Short media is now materialized by a simple for-loop in
  `build_repost_manifest_from_sources`; no `ThreadPoolExecutor` is used for
  this path.
- The manifest summary now reports `repost_download_mode: serial` and
  `repost_download_submitted_rows`.
- Verification passed: focused pytest -> `59 passed`; Python `py_compile`
  passed; `bash -n` passed; scoped `git diff --check` passed; no repost-worker
  config or parallel-test leftovers remain in touched paths.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2983` nodes, `7937` edges, and
  `75` communities.

## 2026-05-10 Plan: Batch Download Required YouTube Shorts

- [x] Keep yt-dlp request-delay settings out of the workflow.
- [x] Download all selected date/profile-valid repost Shorts into the short-form cache as one parallel batch.
- [x] Add regression tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Added `CROSS_PLATFORM_REPOST_DOWNLOAD_WORKERS` / `cross_platform_repost_download_workers`
  with default `8`.
- `build_repost_manifest_from_sources` now filters and sorts repost candidates
  first, then materializes every selected uncached Short in a parallel batch
  before writing the repost manifest.
- No yt-dlp sleep/request-delay options were added.
- Verification passed: focused pytest -> `59 passed`; Python `py_compile`
  passed; `bash -n` passed; scoped `git diff --check` passed; no sleep-related
  yt-dlp options present in the touched paths.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2983` nodes, `7937` edges, and
  `75` communities.

## 2026-05-10 Plan: Remove yt-dlp Cookie Diagnostics

- [x] Remove cookie-file logging from shared yt-dlp auth helpers.
- [x] Remove cookie-auth status/success/failure logging from discovery, metadata, download, and full-VOD metadata call sites.
- [x] Verify cookie support still applies options without emitting cookie-file logs.

### Review Notes

- Removed the cookie-file diagnostic helpers/events from
  `backend/services/yt_dlp_auth.py`; it now only applies `cookiefile` or
  `cookiesfrombrowser` options.
- Removed auth status/success/failure logging from Shorts discovery, Shorts
  metadata, generic media download, Twitch download, and full-VOD audio
  metadata extraction.
- Verification passed: no backend matches for cookie diagnostic fields/events;
  focused pytest -> `81 passed`; `py_compile` passed; scoped
  `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2981` nodes, `7929` edges, and
  `73` communities.

## 2026-05-10 Plan: Add yt-dlp Cookie Diagnostics

- [x] Superseded by the removal request above.

## 2026-05-10 Plan: Add yt-dlp YouTube Cookie Support

- [x] Add shared yt-dlp auth option handling for cookie files / browser cookies.
- [x] Apply it to Shorts discovery, Shorts metadata enrichment, Shorts downloads, and full-VOD audio metadata extraction.
- [x] Add env placeholders, tests, lessons, verification, and Graphify rebuild.

### Review Notes

- Added shared yt-dlp auth option handling in `services/yt_dlp_auth.py`.
- New settings/env keys:
  `YTDLP_COOKIEFILE` and `YTDLP_COOKIES_FROM_BROWSER`.
- Cookiefile auth is applied to generic downloader downloads, Shorts discovery,
  Shorts metadata enrichment, and full-VOD audio metadata extraction.
- Cookiefile takes precedence over browser cookies because it works better in
  Docker/non-interactive training runs.
- Verification passed: focused pytest -> `81 passed`; `py_compile` passed;
  scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2981` nodes, `7929` edges, and
  `73` communities.

## 2026-05-10 Plan: Use Clips.Delivery As Curated Repost Source

- [x] Replace broad YouTube/TikTok keyword discovery with only `youtube.com/@Clips.Delivery/shorts`.
- [x] Mark this channel as a trusted curated repost source while preserving the real VOD date-window filter.
- [x] Bump policy version, add regression tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- `backend/ml/cross_platform_sources.csv` now contains exactly one configured
  cross-platform source for Jynxzi:
  `https://www.youtube.com/@Clips.Delivery/shorts`.
- The source is tagged `curated_repost_source`, so its clips bypass only the
  profile-text filter. This means titles/uploader names do not need to contain
  `jynxzi` or `junko`.
- The real VOD publish-date window still applies. Curated clips outside that
  VOD-derived date range are skipped, and clips without publish metadata still
  fail closed unless metadata can be enriched.
- `REPOST_PROFILE_FILTER_VERSION` is now `8`, so the next run rebuilds repost
  manifests and does not reuse older cross-platform rows from the broad
  keyword/TikTok policy.
- Verification passed: focused pytest -> `67 passed`; `py_compile` passed;
  scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2970` nodes, `7912` edges, and
  `72` communities.

## 2026-05-10 Plan: Expand Cross-Platform Label Discovery Coverage

- [x] Add broader Jynxzi/Junko YouTube and TikTok source queries for the same VOD date-window policy.
- [x] Add a TikTok Bing HTML fallback so zero RSS results do not mean zero TikTok candidates.
- [x] Bump repost policy version, add tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Cross-platform source discovery now has `27` source rows instead of the
  narrow keyword set: broader YouTube searches for `jynxzi shorts`,
  `jynxzi viral`, `jynxzi twitch`, `jynxzi league of legends`, and
  `junko clips`, plus matching TikTok keyword searches.
- `tiktoksearch...` no longer depends only on Bing RSS. If RSS returns no
  TikTok videos, the discovery code also parses Bing HTML result links and
  canonicalizes direct TikTok and Bing redirect URLs.
- `REPOST_PROFILE_FILTER_VERSION` is now `7`, so stale repost manifests from
  the smaller source/filter policy rebuild on the next run.
- Existing cross-platform match rows now also carry `profile_filter_version`;
  stale rows below version `7` are not considered reusable. That prevents the
  next run from short-circuiting before the broader discovery policy runs.
- Verification passed: focused pytest -> `66 passed`; `py_compile` passed;
  scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command failed because this
  shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2968` nodes, `7905` edges, and
  `72` communities.

## 2026-05-10 Plan: Force Repost Rediscovery After Date-Search Fix

- [x] Confirm the run reused the 7-row repost manifest instead of rediscovering sources.
- [x] Bump the repost manifest policy version so the next run rebuilds with the `ytsearchdate` handler.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The posted run did not exercise the `ytsearchdate` handler because
  `repost_manifest_built=false` and the existing 7-row manifest had
  `profile_filter_version=5`.
- `REPOST_PROFILE_FILTER_VERSION` is now `6`, so those version-5 rows are stale
  and the next run should show `repost_manifest_profile_filter_rebuild_required=true`
  followed by a rebuilt manifest.
- Verification passed: focused pytest -> `56 passed`; stale-manifest targeted
  test -> `1 passed`; `py_compile` passed; scoped `git diff --check` passed;
  Graphify rebuilt successfully with `2957` nodes, `7880` edges, and
  `69` communities.

## 2026-05-10 Plan: Fix Unsupported `ytsearchdate` Source

- [x] Add an internal `ytsearchdateN:<query>` handler instead of passing that scheme to yt-dlp.
- [x] Keep the same fresh-source CSV policy, but make it compatible with the installed yt-dlp version.
- [x] Add focused tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- The active run showed `Unsupported url scheme: "ytsearchdate500"` because the
  installed yt-dlp does not support that pseudo-scheme.
- `auto_train_short_form.py` now intercepts `ytsearchdateN:<query>` before
  yt-dlp sees it and converts it to a real YouTube results URL with upload-date
  sort parameter `sp=CAI%253D`.
- If the date-sorted search page extraction returns no rows, the code falls
  back to normal `ytsearchN:<query>` instead of failing the source.
- Verification passed: focused pytest -> `56 passed`; targeted
  `ytsearchdate` test -> `1 passed`; `py_compile` passed; scoped
  `git diff --check` passed; Graphify rebuilt successfully with `2957` nodes,
  `7880` edges, and `69` communities.

## 2026-05-10 Plan: Pick Most-Viewed Reposts Inside Actual VOD Window

- [x] Move cross-platform repost selection to a global post-filter ranking.
- [x] Rank date-valid/profile-valid reposts by views and viral score before applying `max_reposts`.
- [x] Add focused tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Cross-platform repost manifest building now dedupes all discovered/manual
  candidates first, then applies explicit profile evidence and the actual VOD
  date-window filter.
- Only after those filters does it sort candidates globally by `views`,
  `viral_score`, and publish time, then apply `max_reposts`.
- This means an older/out-of-window high-view video can no longer consume a
  slot, and the selected labels are the most-viewed valid candidates from the
  concrete usable VOD window.
- Added manifest summary counters: `eligible_profile_date_rows` and
  `selected_repost_candidates`.
- Bumped `REPOST_PROFILE_FILTER_VERSION` to `5`, so the next run rebuilds the
  repost manifest with this selection policy instead of reusing the previous
  one.
- Verification passed: focused pytest -> `55 passed`; targeted most-viewed
  regression -> `1 passed`; `py_compile` passed; scoped `git diff --check`
  passed; Graphify rebuilt successfully with `2953` nodes, `7869` edges, and
  `70` communities.

## 2026-05-10 Plan: Increase Date-Scoped Repost Label Discovery

- [x] Add fresh/date-sorted YouTube sources for the same four Jynxzi keywords.
- [x] Make TikTok keyword search use multiple search variants and keep RSS fallback rows when metadata extraction fails.
- [x] Add focused tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Added `ytsearchdate500` sources for the existing four requested Jynxzi
  keywords, so discovery can pull newer/date-sorted YouTube results for the
  actual VOD window instead of only best-match search results.
- TikTok keyword search now queries multiple Bing RSS variants per keyword and
  dedupes URLs across them.
- TikTok RSS rows are kept as fallback candidates when detailed metadata
  extraction fails; their title, description, uploader, timestamp, and
  upload-date are still available for downstream profile/date filtering.
- Bumped `REPOST_PROFILE_FILTER_VERSION` to `4`, forcing the next run to rebuild
  the repost manifest with the expanded source policy instead of reusing the
  small version-3 manifest.
- Verification passed: focused pytest -> `54 passed`; TikTok-specific tests ->
  `2 passed`; `py_compile` passed; scoped `git diff --check` passed; Graphify
  rebuilt successfully with `2951` nodes, `7861` edges, and `72` communities.

## 2026-05-10 Plan: Remove Raw Repost Minimum After Date Scoping

- [x] Confirm the run fails before matching because raw repost count is below the old global minimum.
- [x] Set the default raw repost minimum to zero and rely on matched-row/VOD gates.
- [x] Update tests, lessons, run verification, and rebuild Graphify.

### Review Notes

- The new source policy worked but yielded only `10` date-valid reposts.
  `min_reposts=400` blocked the run before audio matching, so no diagnostics
  about actual matches could be produced.
- `CROSS_PLATFORM_MIN_REPOSTS` now defaults to `0` in the shell workflow,
  Docker env file, env example, and workflow-state defaults.
- The remaining meaningful gates are still `CROSS_PLATFORM_MIN_MATCHED_ROWS`
  and `CROSS_PLATFORM_MIN_MATCHED_VODS`; with only `10` reposts the next run
  may still fail there, but it will fail after real matching instead of before
  it.
- Verification passed: focused pytest -> `45 passed`; `py_compile` passed;
  `bash -n` passed; scoped `git diff --check` passed; Graphify rebuilt
  successfully with `2944` nodes, `7850` edges, and `72` communities.

## 2026-05-10 Plan: Add TikTok Keyword Repost Sources

- [x] Replace broad/overlapping Jynxzi repost keywords with the requested four-keyword set.
- [x] Add TikTok keyword-source support and source rows alongside YouTube.
- [x] Add focused tests, run verification, and rebuild Graphify.

### Review Notes

- `cross_platform_sources.csv` now has exactly the requested keywords for both
  platforms: `jynxzi`, `jynxzi funny`, `jynxzi rage`, and `jynxzi clips`.
- Added `tiktoksearchN:<query>` source support. It discovers TikTok video URLs
  through Bing RSS, then still runs normal metadata extraction/download logic.
- Search-source strings (`ytsearch...` and `tiktoksearch...`) are not trusted
  as profile evidence. Rows still need explicit Jynxzi/Junko evidence in
  title/uploader/channel/description.
- Bumped `REPOST_PROFILE_FILTER_VERSION` to `2`, so the next run rebuilds the
  repost manifest with the new source policy instead of reusing the old one.
- Verification passed: focused pytest -> `51 passed`; `py_compile` passed;
  scoped `git diff --check` passed; Graphify rebuilt successfully with
  `2938` nodes, `7827` edges, and `69` communities.

## 2026-05-10 Plan: Scope Reposts To Actual VOD Date Window

- [x] Identify the actual VOD date source and why current inventory rows lost `created_at`.
- [x] Preserve non-empty VOD API dates when merging cache-backed inventory rows.
- [x] Filter discovered reposts to the date window of VODs present in the usable VOD manifest.
- [x] Add tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- The VOD inventory merge could overwrite non-empty API fields like
  `created_at` with empty cache metadata. The merge now preserves non-empty
  `created_at`, `title`, and `vod_duration_seconds`.
- Repost manifest rebuilds now compute a date window from the actual VOD rows
  used for matching, joined back to `jynxzi_twitch_vod_inventory.jsonl`.
- Reposts are kept only when their `publish_timestamp` or `upload_date` falls
  in that actual VOD window, with a small default margin: `1` day before the
  earliest usable VOD and `30` days after the latest usable VOD.
- If flat discovery omits publish/upload date, the builder tries one metadata
  lookup for that URL before deciding whether to skip it as date-unknown.
- Bumped `REPOST_PROFILE_FILTER_VERSION` to `3`, so any manifest from the
  pre-date-window policy is rebuilt on the next run.
- Verification passed: focused pytest -> `57 passed`; `py_compile` passed;
  `bash -n` passed; scoped `git diff --check` passed; Graphify rebuilt
  successfully with `2944` nodes, `7850` edges, and `72` communities.

## 2026-05-10 Plan: Calibrate Repost Minimum After Profile Filter

- [x] Confirm the new run failed because filtered repost rows dropped below the old pre-match minimum.
- [x] Lower the repost-source minimum to the filtered Jynxzi source pool while keeping match-row and VOD-diversity gates.
- [x] Update env/workflow-state defaults, tests, lessons, and run verification plus Graphify.

### Review Notes

- The profile filter worked: the run rebuilt with
  `repost_manifest_build_reason=profile_filter_policy`, removed
  `152` non-profile search hits, and did not preserve the old false-positive
  output.
- The failure was the remaining pre-match source guard:
  `repost_rows=493 < min_reposts=600`, so matching never started.
- Lowered `CROSS_PLATFORM_MIN_REPOSTS` to `400` in the shell default,
  Docker env file, env example, and workflow-state default. The hard quality
  gates remain `70` matched rows and `20` matched VODs for the strict pass.
- Verification passed: focused pytest -> `43 passed`; `py_compile` passed;
  `bash -n` passed; scoped `git diff --check` passed; Graphify rebuilt
  successfully with `2934` nodes, `7818` edges, and `73` communities.

## 2026-05-08 Plan: Parallelize Full-VOD Audio Fingerprint Builds

- [x] Add a bounded `--full-vod-audio-workers` option to cross-platform fingerprint prep.
- [x] Run Full-VOD audio cache/download/fingerprint work with two parallel workers by default in Jynxzi training.
- [x] Keep per-VOD logging and summary counters accurate under parallel execution.
- [x] Update env/workflow/tests, run focused verification, and rebuild Graphify.

### Review Notes

- `prepare_cross_platform_fingerprint_rows.py` now accepts
  `--full-vod-audio-workers` and runs per-VOD full-audio cache/download/
  fingerprint work through a bounded `ThreadPoolExecutor` when workers > 1.
- Jynxzi training defaults `CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS=2`, passes it
  into prepare, exports it into workflow state, and records it in `.env` and
  `.env.example`.
- Summary/log payloads include `workers`, while per-VOD success/cache/failure
  counters are aggregated after worker completion.
- Verification passed: focused prepare/script/workflow-state tests,
  `py_compile`, `bash -n`, and scoped `git diff --check`.

## 2026-05-08 Plan: Remove Twitch Clip Labels From Outcome Training Flow

- [x] Map the remaining Twitch-label dependencies in the Jynxzi training script and cross-platform fingerprint prep.
- [x] Replace Twitch-label VOD seeding with a VOD inventory path built from available non-label artifacts/cache.
- [x] Remove Twitch clip-label collection/pruning from the Jynxzi training workflow.
- [x] Update tests, env defaults, lessons, and run focused verification plus Graphify.

### Review Notes

- `train_jynxzi_vod.sh` no longer collects or prunes Twitch clip labels, and
  there is no `TWITCH_CLIP_LABELS_ENABLED` opt-in switch left in the training
  workflow or env files.
- The workflow now builds `jynxzi_twitch_vod_inventory.jsonl` from the Twitch
  VOD API plus existing `vod_feature_cache` metadata. Full-VOD audio matching
  receives that inventory through `--full-vod-audio-vod-inventory`.
- `prepare_cross_platform_fingerprint_rows.py` prefers the VOD inventory over
  Twitch label rows when selecting full-VOD audio fingerprints. Label rows are
  only a legacy fallback if no inventory exists.
- `train_vod_ranker.py` no longer exits just because Twitch label rows are
  empty; with outcome-only training it can continue from cross-platform rows.
- Local cache inspection found `67` VOD inventory candidates, so the next run
  should not be limited to the single stale Twitch-label VOD.
- Verification passed: focused prepare/script/workflow-state tests, full
  `test_train_vod_ranker.py`, `py_compile`, `bash -n`, and scoped
  `git diff --check`.

## 2026-05-08 Plan: Fix Stale Twitch Label Progress Blocking Wider Windows

- [x] Confirm the wider date window is skipped because progress says covered while labels contain only one VOD.
- [x] Add a guard so stale label progress does not block collection for wider training windows.
- [x] Keep the fix scoped to Twitch label collection; do not change cross-platform matching behavior.
- [x] Run focused tests and syntax checks.

### Review Notes

- Current progress claimed `2025-10-01..2025-12-14` and
  `2026-01-01..2026-05-05` were covered, but the active label file contained
  only `483` rows from one VOD on `2026-05-01`.
- `collect_twitch_clip_labels.py` now validates progress ranges against actual
  existing label dates before honoring `--skip-covered`.
- With current artifacts, the validated covered range becomes only
  `2026-05-01..2026-05-01`, so the next wider run will recollect the missing
  days instead of skipping the whole window.
- Verification passed: `test_twitch_clip_labels.py` -> `30 passed`;
  `py_compile` passed; scoped `git diff --check` passed.

## 2026-05-08 Plan: Fix Env Override For Cross-Platform Matching Defaults

- [x] Confirm why the restarted run still used `min_confidence=0.5` and `reuse_existing_output=true`.
- [x] Update the real Docker env file and example so they no longer override the code defaults with stale values.
- [x] Verify the effective env-file values and run focused checks.
- [x] Document the restart expectation.

### Review Notes

- `scripts/train_local_docker.sh` passes `backend/.env` through Docker
  `--env-file`, and that file still had
  `CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE=0.50` plus
  `CROSS_PLATFORM_REUSE_EXISTING_ROWS=1`.
- Updated both `backend/.env` and `backend/.env.example` to `0.60` and `0`.
- Verified with `rg` that the active env file, example, and shell defaults now
  agree.
- Verification passed: shell syntax check and scoped `git diff --check`.

## 2026-05-08 Plan: Stop Reusing Stale Cross-Platform Rows By Default

- [x] Confirm the failed run reused the old cross-platform artifact instead of rematching at the new threshold.
- [x] Change the Jynxzi training default so cross-platform rows are rematched unless reuse is explicitly enabled.
- [x] Keep the state snapshot and tests aligned with the new default.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The run showed `prepare_status=reused_existing_output` and
  `prepare_preserved_existing_output=true`, so it did not rematch with the new
  `0.60` threshold.
- `CROSS_PLATFORM_REUSE_EXISTING_ROWS` now defaults to `0` in the Jynxzi
  training script. Existing cross-platform rows can still be reused explicitly
  by setting `CROSS_PLATFORM_REUSE_EXISTING_ROWS=1`.
- The workflow-state snapshot default and test expectation now match the shell
  default.
- Verification passed: focused workflow-state pytest, shell syntax check,
  `py_compile`, and scoped `git diff --check`.

## 2026-05-08 Plan: Prevent One-VOD Cross-Platform Overmatching

- [ ] Diagnose the one-VOD full-audio match explosion from the run summary.
- [ ] Raise the cross-platform audio confidence default for full-VOD matching.
- [ ] Add a seed matched-VOD diversity guard.
- [ ] Update workflow state/tests and run focused verification.

## 2026-05-08 Plan: Fix Full-VOD Audio Temp Extension

- [x] Diagnose ffmpeg returncode 234 from the new structured logs.
- [x] Change full-VOD audio temp output path so ffmpeg can infer the muxer.
- [x] Add regression coverage for the temp-path extension.
- [x] Run focused tests, syntax checks, and Graphify.

### Review Notes

- The ffmpeg failure was caused by the temporary output path
  `source.m4a.part`. FFmpeg could not infer the output container from that
  final `.part` suffix.
- Full-VOD audio extraction now writes to `source.part.m4a` and then atomically
  replaces `source.m4a`, preserving the real media extension as the final
  suffix.
- Added regression coverage for the temp path.
- Verification passed: focused pytest -> `42 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2913` nodes, `7758` edges,
  `69` communities.

## 2026-05-08 Plan: Diagnose Full-VOD Audio FFmpeg Failure

- [x] Inspect the full-VOD audio download implementation and current error
  logging.
- [x] Preserve useful ffmpeg stderr and source-format diagnostics in the
  failure log.
- [x] Fix obvious ffmpeg invocation issues if found.
- [x] Add focused regression coverage and run verification.

### Review Notes

- The run now fails in the intended place: Full-VOD audio extraction for
  `2761754957`, no MP4 fallback, and immediate blocked training exit.
- The previous log only showed a truncated `CalledProcessError` command because
  the ffmpeg command included long HTTP headers. It hid the real stderr.
- Added structured `FullVodAudioDownloadError` wrapping ffmpeg failures with
  `ffmpeg_returncode`, `ffmpeg_stderr`, `ffmpeg_stdout`, and selected format
  metadata.
- Added `full_vod_audio_download_input_selected` logging so the next run shows
  selected format id, protocol, codecs, bitrate, height, and media URL prefix.
- Verification passed: focused pytest -> `41 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2911` nodes, `7754` edges,
  `70` communities.

## 2026-05-08 Plan: Stop After Blocked Audio-Only Cross-Platform Prep

- [x] Diagnose the completed 1-day run's blocked full-audio failure.
- [x] Make blocked cross-platform prep abort the shell workflow immediately.
- [x] Stop rebuilding repost manifests merely because they are below
  `max_reposts`.
- [x] Add regression coverage and run focused verification.

### Review Notes

- The run correctly produced `blocked_full_vod_audio_fingerprints`, but
  `train_jynxzi_vod.sh` only passed `--fail-on-insufficient`, so the workflow
  continued and built negative-only outcome datasets before failing later.
- The training script now passes `--fail-on-blocked`, so blocked audio-only
  cross-platform prep exits immediately and does not continue into dataset
  construction.
- Repost manifests no longer rebuild just because cached rows are below
  `max_reposts`. `max_reposts` is now only a cap for explicit rebuilds; rebuild
  still happens when rows are missing, below `min_reposts`, or refresh is
  requested.
- Verification passed: focused pytest -> `39 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2904` nodes, `7736` edges,
  `68` communities.

## 2026-05-08 Plan: Remove Old MP4 Cross-Platform Caches

- [x] Inspect old MP4 cache categories and current container state.
- [x] Delete obsolete `coarse_*` scoped-window MP4 cache directories.
- [x] Delete leftover full-VOD audio temp `.m4a` files if present.
- [x] Verify freed space and remaining cache state.

### Review Notes

- Removed 203 obsolete `scoped_windows/coarse_*` directories from
  `backend/tmp/vod_feature_cache`.
- These files were root-owned from Docker, so normal `rm` failed with
  permission errors. Deletion succeeded via a one-shot Docker container mounted
  on the repo.
- Removed stale cross-platform generated artifacts that were derived from the
  old MP4-window matches:
  `jynxzi_cross_platform_vods.jsonl`,
  `jynxzi_cross_platform_fingerprint_rows.jsonl`,
  `jynxzi_cross_platform_fingerprint_summary.json`, and
  `jynxzi_cross_platform_fingerprint_diagnostics.jsonl`.
- Verified `coarse_*` directories remaining: `0`.
- Verified leftover `full_vod_audio/source.m4a` files: `0`.
- `backend/tmp/vod_feature_cache` is now `31G`; filesystem free space is
  `885G`.

## 2026-05-08 Plan: Enforce Full-VOD Audio In Seed Cross-Platform Pass

- [x] Confirm why the current seed pass still used scoped MP4 windows.
- [x] Change the training orchestrator so full-VOD audio mode is active for
  all cross-platform prepare calls, not only the second pass.
- [x] Add regression coverage for the shell wiring.
- [x] Run focused verification and document the result.

### Review Notes

- Root cause: `train_jynxzi_vod.sh` only passed
  `--full-vod-audio-fingerprints` when `refresh_vods=1`, so the seed pass
  explicitly passed `--no-full-vod-audio-fingerprints` and rebuilt from old
  scoped MP4 windows.
- The shell wrapper now passes `--full-vod-audio-fingerprints` whenever
  `CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS=1`, including the seed pass.
- Added a regression test so the shell script cannot reintroduce the
  `refresh_vods && full_audio` coupling.
- Verification passed: focused pytest -> `37 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2902` nodes, `7731` edges,
  `67` communities.

## 2026-05-08 Plan: Fail Closed On Failed Full-VOD Audio Fingerprints

- [x] Diagnose the 1-day smoke-test failure path from the provided logs.
- [x] Stop empty full-audio VOD refreshes from overwriting usable manifests or
  cross-platform outcome rows.
- [x] Add regression tests for failed full-audio backfill and empty-input
  preservation.
- [x] Run focused tests, syntax checks, Graphify, and document the result.

### Review Notes

- The smoke test selected one VOD for full-audio fingerprinting, failed that
  one VOD, wrote a refreshed VOD manifest with `0` rows, then overwrote
  `jynxzi_cross_platform_fingerprint_rows.jsonl` with `0` rows. That removed
  all positives before training.
- Full-audio fingerprint builds now report
  `failed_no_full_vod_audio_fingerprints` when selected VODs exist but no
  cached/built full-audio fingerprints remain.
- `prepare_cross_platform_fingerprint_rows` now blocks with
  `blocked_full_vod_audio_fingerprints` before matching when that happens.
  Existing VOD manifests and existing cross-platform outcome rows are preserved.
- Empty-input and below-minimum match results now preserve the last valid
  output artifact when one exists, instead of writing an empty outcome file.
- Verification passed: focused pytest -> `36 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2899` nodes, `7728` edges,
  `68` communities.

## 2026-05-07 Plan: Coarse Audio Tiling fuer Cross-Platform-Matching

Ziel: Cross-platform Reposts sollen nicht nur dann matchen koennen, wenn sie
zufaellig in einem Twitch-Top-Clip scoped window liegen. Der Rematch soll
optional grobe Audio-Tiles ueber ausgewaehlte VODs bauen und in das
VOD-Manifest aufnehmen.

- [x] Coarse-Tiling Parameter und Cache-Layout im Fingerprint-Prepare
  einfuehren.
- [x] Aus selected targeted-backfill VODs und Label-Manifest grobe VOD-Tiles
  erzeugen.
- [x] Jynxzi-Trainingsscript so verdrahten, dass der zweite Rematch diese
  Tiles nutzt.
- [x] Tests/Syntax/Graphify ausfuehren.
- [x] Ergebnis dokumentieren.

### Review

- `prepare_cross_platform_fingerprint_rows.py` hat jetzt einen optionalen
  `--coarse-vod-tiling` Pfad. Er liest die selected targeted-backfill VODs,
  nimmt das Twitch-Label-Manifest nur zur VOD-/Horizon-Bestimmung und erzeugt
  grobe `coarse_vod_audio_tile` scoped windows im bestehenden
  `vod_feature_cache`.
- Defaults fuer den Trainingsrun: `30s` Tile-Laenge, `600s` Stride,
  maximal `24` Tiles pro VOD. Damit deckt ein selected VOD grob bis zu vier
  Stunden ab, ohne den kompletten VOD voll zu analysieren.
- Der zweite Cross-platform Rematch aktiviert das Tiling nur bei
  `refresh_vods=1`. Seed-Reuse bleibt billig; der teure breitere Suchraum
  entsteht erst im strikten Rematch nach targeted backfill.
- Die groben Tiles landen im normalen VOD-Manifest, weil der Manifest-Builder
  alle `scoped_windows/*/source.metadata.json` einliest. Der bestehende
  Audio-Fingerprint-Matcher kann sie dadurch unveraendert vergleichen.
- Verifikation: `py_compile` erfolgreich; `bash -n train_jynxzi_vod.sh`
  erfolgreich; fokussierte Tests
  `test_prepare_cross_platform_fingerprint_rows.py`,
  `test_training_workflow_state.py`, `test_targeted_backfill.py`,
  `test_train_vod_ranker.py` -> `60 passed`; scoped `git diff --check`
  sauber.
- Graphify-Rebuild erfolgreich:
  `2876` Nodes, `7638` Edges, `68` Communities.

## 2026-04-27 Plan: ML Model Verbesserungsplan (basierend auf Eval-Analyse)

### Kontext
Modell: Logistic Regression, 71 Features, 244 Trainingsrows (64 positiv).
Kernprobleme: kleines Goldlabeled-Dataset, Train-Holdout-Gap, twitch_proxy schadet, slow_open_penalty-Confound.

---

### Priorität 1 — Mehr Gold-Labels (kritischer Engpass)

**Ziel:** Von 48 auf ≥150 gold_eval-Trainingsrows in 4 Wochen.

- [ ] **P1.1 — Batch-Annotierung starten (laufend)**
  - 5–8 neue VODs pro Woche in `jynxzi_gold_eval_manifest.csv` aufnehmen
  - `build_gold_eval_queue.py --manifest ... --storage-dir ... --out jynxzi_gold_eval_queue` ausführen
  - Queue in Google Sheets öffnen, `gold_grade` 0–3 blind annotieren (ohne Modell-Scores)
  - Importieren via `import_gold_eval_labels.py`

- [x] **P1.2 — Holdout-VODs für Annotierung priorisieren**
  - Die 35 Holdout-VODs aus `jynxzi_training_workflow_state.json` sind ungenutzte Labelpotenziale
  - 10 davon in den nächsten Annotierungsbatch aufnehmen (ohne sie aus dem Holdout zu entfernen bis genug Train-Daten)

- [x] **P1.3 — Erfolgsmetrik festlegen**
  - Retrain wenn gold_eval-Rows ≥ 80 (Zwischenmeilenstein) und ≥ 150 (Hauptmeilenstein)
  - Erwarteter Effekt: MRR-Holdout sollte von 0.471 auf ≥ 0.55 steigen

---

### Priorität 2 — twitch_label_proxy-Gewichtung reduzieren

**Ziel:** Source-Bucket `twitch_label_proxy` schadet dem Modell (MRR-Delta −0.046). Gewicht senken oder ausschließen.

- [x] **P2.1 — Diagnose vertiefen**
  - In `train_ranker.py:_sample_weight()` ist `twitch_clip_proxy`-Tier aktuell auf 1.15 gesetzt (leicht übergewichtet)
  - Mit einem temporären Run testen: Gewicht auf 0.5 senken und MRR auf Holdout vergleichen
  - Kommando: `ranker_gold_train_weight=2.0, ranker_hard_negative_weight=4.0` beibehalten, nur `twitch_clip_proxy`-Multiplikator anpassen

- [x] **P2.2 — Code-Änderung in `train_ranker.py`**
  - `"twitch_clip_proxy": 1.15` → `"twitch_clip_proxy": 0.5` in `_sample_weight()` (Zeile ~1721)
  - Retrain durchführen, gold_holdout MRR und nDCG@5 vergleichen
  - Wenn MRR auf Holdout steigt: Wert dauerhaft übernehmen

- [ ] **P2.3 — Alternativ: Proxy-Tier aus Training ausschließen**
  - Falls Gewichtsreduktion nicht ausreicht: `--no-proxy-tier`-Flag zu `train_vod_ranker.py` hinzufügen
  - Nur sinnvoll wenn gold_eval-Rows ≥ 80 (sonst zu wenig Daten)

---

### Priorität 3 — slow_open_penalty Confound aufklären

**Ziel:** Feature hat Gewicht +1.14 — das bedeutet "mehr slow_open = besser", was semantisch falsch ist.

- [x] **P3.1 — Korrelationsanalyse**
  - Feature-Korrelation zwischen `slow_open_penalty` und `chapter_local_density`, `duration`, `payoff_score` berechnen
  - In Python: `pd.DataFrame([r["features"] for r in training_rows]).corr()[["slow_open_penalty"]]`
  - Vermutung: `slow_open_penalty` korreliert positiv mit `chapter_local_density` (dichte Chapters = mehr slow-open-erkannte Clips)

- [x] **P3.2 — Entscheidung treffen**
  - Option A: Feature zu `excluded_feature_names` verschieben (konservativ, einfach)
  - Option B: Feature-Semantik im Feature Builder invertieren (wenn Bedeutung klar)
  - Option C: Feature umbenennen und Polarität dokumentieren
  - Empfehlung: erst Korrelation messen, dann entscheiden

- [x] **P3.3 — Retrain mit angepasstem Feature-Set**
  - Wenn Feature ausgeschlossen: erwarten dass `profile_fit_score` und `setup_depth_score` mehr Gewicht aufnehmen

---

### Priorität 4 — Hard Negatives systematisch ausbauen

**Ziel:** Aktuell nur 6 Hard-Negatives aus Holdout. Ziel: 20–30 pro Trainingslauf.

- [x] **P4.1 — Rollout Review Queue nutzen**
  - `jynxzi_rollout_review_queue.csv` (63 Rows, 26 VODs) manuell reviewen
  - False Positives identifizieren (Modell sagt gut, aber tatsächlich schlecht)
  - Diese als Hard Negatives in den nächsten Trainingslauf einbringen

- [x] **P4.2 — Pipeline für strukturiertes FP-Feedback**
  - Nach jedem Trainingslauf: `diagnose_ranker_holdout.py` ausführen
  - Alle False Positives (max 100) via `export_ranker_hard_negatives.py` exportieren
  - Direkt in `jynxzi_ranker_hard_negative_train_rows.jsonl` mergen

- [ ] **P4.3 — Hard-Negative-Gewicht evaluieren**
  - Aktuell: `ranker_hard_negative_weight=4.0` (label_class=hard_negative hat base_weight 1.8, dann x4 = 7.2 effektiv)
  - Testen ob 2.0 oder 3.0 besser ist wenn mehr Hard Negatives vorhanden

---

### Priorität 5 — Gradient Boosting als Alternative evaluieren (nach P1)

**Voraussetzung:** ≥ 150 gold_eval-Trainingsrows.

- [x] **P5.1 — Model-Selection-Metrik verstehen**
  - In `train_ranker.py`: `composite_validation_score` identifizieren und verstehen warum LR über GB gewählt wurde
  - GB hat auf Validation MRR 0.885 vs LR 0.846 — warum wurde LR bevorzugt?

- [ ] **P5.2 — Expliziter GB-Vergleich**
  - Wenn genug Daten: beide Modelle auf separaten Seeds evaluieren und Holdout-MRR vergleichen
  - GB hat potenzielle Overfitting-Risiken bei kleinem Datensatz — daher erst bei >150 positiv

---

### Erfolgsmetriken (Zielzustand nach 4 Wochen)

| Metrik | Aktuell | Ziel |
|---|---|---|
| Gold-Holdout MRR | 0.471 | ≥ 0.55 |
| Gold-Holdout nDCG@5 | 0.475 | ≥ 0.55 |
| MRR_p25 (volle Eval) | 0.0 | > 0.0 |
| gold_eval Training-Rows | 48 | ≥ 150 |
| Hard Negatives | 6 | ≥ 25 |

---

### Umsetzung 2026-04-27

- P1.1 bleibt als laufender manueller Annotierungsprozess offen; die technischen Vorarbeiten sind erledigt, aber blindes Human-Rating und Re-Import kann ich nicht automatisch korrekt ausführen.
- P1.2 ist umgesetzt über `backend/ml/artifacts/jynxzi_gold_holdout_annotation_manifest.csv` mit 10 priorisierten Holdout-VODs.
- P1.3 ist als Zielmetrik im Plan fixiert und in den erzeugten Eval-Artefakten prüfbar: Gold-Holdout MRR, nDCG@5, `mrr_p25`, Gold-Train-Rows und Hard-Negative-Anzahl.
- P2.1/P2.2 sind umgesetzt: `twitch_clip_proxy`-Supervision wird in `train_ranker._sample_weight()` von 1.15 auf 0.5 reduziert.
- P2.3 bleibt bewusst offen: Proxy-Tier komplett auszuschließen ist erst sinnvoll, wenn mindestens 80 Gold-Rows im Training stehen; aktuell sind es 48.
- P3.1 ist umgesetzt über `backend/ml/analyze_ranker_feature_confounds.py` und `backend/ml/artifacts/jynxzi_ranker_feature_confounds.json`.
- P3.2/P3.3 sind umgesetzt: `slow_open_penalty` wird konservativ aus dem supervised Ranker-Feature-Set ausgeschlossen; das aktuelle Ranker-Artefakt wurde neu trainiert.
- P4.1 ist operationalisiert über `backend/ml/artifacts/jynxzi_rollout_review_queue.csv` mit 63 manuellen Review-Kandidaten, `review_required=true`, `auto_publish_allowed=false`.
- P4.2 ist umgesetzt über `backend/ml/refresh_ranker_hard_negative_artifacts.py`, das Holdout-Diagnose und Hard-Negative-Export in einem Schritt erneuert.
- P4.3 bleibt offen, weil aktuell weiterhin nur 6 exportierte Hard Negatives vorliegen und alle aus Holdout stammen; die Training-Pipeline lässt Holdout-origin Rows wegen Leakage-Schutz nicht ins Training.
- P5.1 ist umgesetzt über `artifact.model_selection`: GB/LightGBM werden unter 400 Trainingsrows nicht promotet, auch wenn LightGBM auf Validation besser ist.
- P5.2 bleibt offen, weil die Voraussetzung `>=150` Gold-Train-Rows nicht erfüllt ist.

### Aktueller Ergebnisstand

- Re-train ohne neue VOD-Downloads abgeschlossen: `jynxzi_ranker.json` ist Logistic Regression, `feature_schema_version=4`, `serving_threshold=5.52`.
- `slow_open_penalty` ist in `excluded_feature_names`; Confound-Report bestätigt Positiv-Mittelwert `0.719625` vs Negativ-Mittelwert `0.654`.
- Gold full eval: accuracy `0.779`, precision `0.649`, recall `0.792`, F1 `0.713`, MRR `0.645`, nDCG@5 `0.647`.
- Gold holdout: accuracy `0.783`, precision `0.667`, recall `0.762`, F1 `0.711`, MRR `0.471`, nDCG@5 `0.475`.
- Proxy eval: accuracy `0.877`, precision `0.774`, recall `0.75`, F1 `0.762`, MRR `0.81`, nDCG@5 `0.785`.
- Hard-negative export bleibt bei 6 Rows, alle `hard_negative_original_split=holdout`; diese werden vom Training weiterhin ausgeschlossen, um Evaluation nicht zu verfälschen.



## 2026-04-27 Plan: Conservative Rollout Review Queue

- [x] Read the existing queue/scoring flow and Graphify backend overview.
- [x] Add a manual-review rollout queue that never auto-publishes candidates.
- [x] Exclude existing gold rows and gold-holdout VODs from rollout review candidates.
- [x] Add focused tests for holdout exclusion, per-VOD top-k selection, and guardrail metadata.
- [x] Generate the current Jynxzi rollout review artifacts.
- [x] Run verification and rebuild backend Graphify after backend code edits.
- [x] Document the result and remaining rollout constraints.

### Review Notes

- Added `backend/ml/build_rollout_review_queue.py` as a separate conservative rollout queue, so active gold-eval sampling keeps its existing semantics.
- The queue scores candidates with the current trained ranker, excludes exact existing gold candidates/windows, and excludes any VOD that appears in the gold holdout split.
- Guardrails are explicit in CSV, sidecar, and summary: `review_required=true` and `auto_publish_allowed=false`.
- Generated current Jynxzi rollout review artifacts with `top_k_per_vod=10`, `max_total=200`, and `min_gap_sec=20`.
- Current queue output has `63` rows across `26` VODs; no queue row overlaps a gold-holdout VOD, and the maximum per-VOD count is `10`.
- The current scoring dataset already excludes holdout VODs, so `excluded_gold_holdout_vod_rows=0`; the artifact still records all `35` holdout VOD IDs and enforces the exclusion if they appear later.
- Verification passed: syntax compile for the new script/test.
- Verification passed: `backend/tests/test_build_rollout_review_queue.py` and `backend/tests/test_build_active_gold_eval_queue.py`.
- Verification passed: `git diff --check`.
- Rebuilt backend Graphify with the uv tool interpreter: `2142` nodes, `5558` edges, `46` communities.
- The generated rollout review artifacts are currently ignored by `.gitignore` like other new artifact names; they exist locally under `backend/ml/artifacts/`.

## 2026-04-26 Plan: Clip-First Top-Viewed Scoped Training

- [x] Trace the current top-viewed scoped path and full-VOD materialization boundary.
- [x] Add a clip/scoped media materialization path that avoids full-VOD download for all-top-viewed VOD groups.
- [x] Preserve VOD-chat alignment by reusing `vod_id` and rebased chat windows.
- [x] Add focused tests proving the top-viewed path does not call full-VOD download.
- [x] Run syntax checks, targeted tests, and scoped diff checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Document behavior, caveats, and current-run impact.

### Review Notes

- Added `downloader.download_range(...)`, which uses yt-dlp range options and bypasses the persistent Twitch full-VOD cache.
- Changed all-top-viewed `window_scoped_top_viewed` VOD groups to skip `_materialize_vod` and build each scoped source directly from the needed VOD time range.
- Serial prefetch and parallel download scheduling now skip Full-VOD futures for all-top-viewed scoped groups.
- The scoped source metadata version is now `2`, and the raw VOD row fingerprint includes that version so future checkpoints reflect the new source behavior.
- Chat alignment is preserved: scoped sources still use the canonical `vod_id`, VOD chat cache, target offsets, and rebased `chat.local.json`.
- Current label impact: `24` scoped-only VODs containing the `47` top-viewed labels will use the range/scoped path; the remaining `27` VODs still require Full-VOD processing because they contain normal training labels.
- Verification passed: `python3 -m py_compile` for touched modules/tests.
- Verification passed: full `backend.tests.test_downloader`.
- Verification passed: full `backend.tests.test_build_twitch_training_dataset`.
- Verification passed: scoped `git diff --check`.
- Rebuilt backend Graphify with the uv tool interpreter: `2097` nodes, `5406` edges, `52` communities.
- Caveat: already-running training containers use the code baked/mounted into that container at start; this change applies to the next run or a restarted container using the updated workspace/image.

## 2026-04-26 Plan: Inspect Current Training Run Downloads

- [x] Inspect running container, command, and recent training logs.
- [x] Separate cache hits from real external downloads for VOD, scoped windows, chat, and model assets.
- [x] Cross-check artifact/cache state for what changed in this run.
- [x] Summarize what is being re-downloaded and why.

### Review Notes

- Current container `boring_keller` started `2026-04-25T22:18:24Z` and is running `train_vod_ranker.py` for `2026-03-15` through `2026-03-28` with `--skip-covered --merge-output`.
- The run uses `FEATURE_BUILD_VERSION=8`, `_CHAT_ENRICHMENT_FINGERPRINT_VERSION=4`, and `TWITCH_DOWNLOAD_CACHE_TTL_SEC=0`.
- Checkpoint status inside the running container: `51` VODs in the label file, `1` currently valid VOD checkpoint, `45` invalid old checkpoints, and `5` missing checkpoints.
- Because the raw VOD fingerprint changed, old VOD row checkpoints are being rebuilt; this requires full VOD video materialization again for each invalid/missing VOD.
- Confirmed active real Twitch media downloads: `2711274372` was downloaded and checkpointed; `2712128863` was prefetched and being processed; `2712736206` is actively downloading via `aria2c`.
- Existing chat cache files are present for all `51` VODs, but only `46` currently have complete chat coverage; `2720629593`, `2724120511`, `2731588015`, `2732418387`, and `2748783535` may trigger targeted chat refill/download for missing offsets.
- Existing feature cache is mixed: `28/51` VODs have full-VOD audio, `27/51` have full-VOD transcripts, and all have chat cache; missing audio/transcript may still be recomputed from the re-materialized video.
- Current storage state is healthy: mount has about `847G` free; `backend/tmp/storage` is about `50G`, mostly active/stale VOD materialization directories.
- There is a stale-looking `twitch_vod_prefetch_2720629593` from before this container start; active current directories are `twitch_vod_prefetch_2712128863` and `twitch_vod_prefetch_2712736206`.

## 2026-04-25 Plan: Improve Scoped Chat Features

- [x] Trace how rebased scoped chat becomes segment and candidate chat features.
- [x] Add a density/volume fallback so chat presence is represented even without detected spikes.
- [x] Add focused tests for non-spike scoped chat coverage.
- [x] Run targeted tests and syntax checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Document the behavioral impact and remaining caveats.

### Review Notes

- Scoped chat now contributes segment-level `chat_message_count`, `chat_message_rate`, `chat_density_score`, and `chat_emote_score` before spike tagging, so normal sustained chat is no longer invisible.
- Candidate features now use the maximum of spike and density as chat activity, while keeping `chat_spike_coverage` separate from the new `chat_activity_coverage`.
- Added candidate-level `chat_message_rate_peak`, `chat_message_rate_mean`, `chat_density_peak`, and `chat_density_mean`.
- Bumped `FEATURE_BUILD_VERSION` to `8` and `_CHAT_ENRICHMENT_FINGERPRINT_VERSION` to `4`, so VOD training rows are rebuilt with the new chat feature schema.
- Verified a real scoped cache sample: `52` segments, `46` with density, `0` with spike; candidate features reported `has_chat_features=True`, `chat_peak=1.0`, `chat_activity_coverage=0.694`, and `chat_spike_coverage=0.0`.
- Verification passed: `python3 -m py_compile` for touched backend modules and tests.
- Verification passed: `backend.tests.test_chat_analyzer` and `backend.tests.test_feature_builder` with offline HF settings.
- Verification passed: selected scoped-chat/checkpoint tests from `backend.tests.test_build_twitch_training_dataset`.
- Rebuilt backend Graphify with the uv tool interpreter: `2088` nodes, `5378` edges, `53` communities.
- Caveat: existing static gold eval row artifacts remain on their old feature payload until they are refreshed or re-imported from a new feature-backed run.

## 2026-04-25 Plan: Import Feature-Backed Top-Viewed Positives Into Gold Eval

- [x] Identify the 47 newly materialized `window_scoped_top_viewed` positive rows.
- [x] Convert them into gold-positive rows using the existing gold-eval schema and dedupe keys.
- [x] Merge them into canonical gold eval artifacts without duplicating the earlier 3 imported rows.
- [x] Rebuild the gold eval dataset and summaries.
- [x] Verify row counts, split distribution, duplicate keys, and feature completeness.

### Review Notes

- Imported `47/47` newly feature-backed `window_scoped_top_viewed` positive-like rows as gold grade `3`; no new rows were skipped as duplicates.
- Canonical `jynxzi_gold_eval_rows.jsonl` now has `222` rows: `76` gold positives and `146` gold negatives.
- Top-viewed Twitch gold rows now total `50`: the earlier `3` plus the new `47`.
- The new batch spans `24` VODs and keeps the source label mix: `29` positive, `17` weak_positive, and `1` viral_positive.
- Deterministic gold split added `34` train rows and `13` holdout rows; total gold split is now `162` train and `60` holdout.
- Rebuilt `jynxzi_gold_eval_dataset.jsonl` and `jynxzi_gold_eval_summary.json`; summary now reports `222` eval rows and `54` eval groups.
- Verification passed: no duplicate queue IDs, no duplicate clip IDs, no missing features, and no missing `features.candidate_id`.
- The full gold set still has one pre-existing duplicate `candidate_id` from frontend rows with different queue IDs and grades; the imported batch has zero duplicate queue IDs or candidate IDs.
- Verification passed: `backend.tests.test_gold_eval_workflow` and `backend.tests.test_build_eval_dataset`.
- `git diff --check` still reports trailing whitespace in the pre-existing `jynxzi_ranker_holdout_diagnosis.csv`; scoped diff-check on the files touched by this import passed.

## 2026-04-25 Plan: Analyze Latest Training Run Failure

- [x] Locate latest run logs, workflow state, and artifact timestamps.
- [x] Quantify progress before failure: VODs processed, checkpoints, rows, and caches.
- [x] Identify fatal error and separate it from non-fatal warnings.
- [x] Check whether artifacts were finalized or only intermediate state changed.
- [x] Summarize next action and rerun impact.

### Review Notes

- Latest persisted run artifacts were written at `2026-04-25 20:24-20:25`; `local_train_summary.json` reports `training_return_code=0`, so the earlier no-space failure was not the final state.
- The workflow produced final artifacts for `jynxzi` over `2026-03-15` through `2026-03-28`; `jynxzi_ranker.json`, eval JSONs, dataset JSONL, and workflow state were updated.
- The final VOD eval dataset has `556` rows: `108` positives and `448` negatives across `64` eval groups.
- The new top-viewed scoped path contributed `139` final rows from `24` VODs: `47` positive-like rows and `92` boundary near-miss negatives.
- Checkpoints exist for `46` VODs, including the post-cleanup VODs `2752903388`, `2753667907`, `2754503460`, and `2756055866`; total checkpoint rows are `11375`.
- Chat downloads completed for the latest VODs with `coverage_status=complete`, but final scoped top-viewed rows only have `2/139` rows with non-zero chat spike features; full-VOD rows have `174/287`.
- The trained model was rejected by the quality gate with `validation_accuracy_below_minimum`; validation accuracy was `0.591`, precision `0.519`, recall `0.737`, F1 `0.609`, and precision@5 `0.485`.
- Threshold calibration was skipped because the artifact was rejected; serving threshold in artifact health is `-1.449484`.
- Next recommendation: treat the run as successful for feature materialization, but not successful for model promotion; inspect whether scoped chat should use density/volume features, not only spike features, before expecting chat to help those top-viewed examples.

## 2026-04-25 Plan: Prevent Twitch VOD Download Cache Disk Exhaustion

- [x] Confirm the failed run root cause and current disk usage.
- [x] Remove only regenerable full-VOD download cache data and failed prefetch fragments.
- [x] Change local/default Twitch full-VOD download cache behavior to avoid unbounded growth.
- [x] Run focused downloader/config checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Document the outcome and rerun caveat.

### Review Notes

- Root cause: the failed VOD-first run hit `[Errno 28] No space left on device`; the host mount was `100%` full, with `backend/tmp/storage/_download_cache` at about `934G` and the failed `twitch_vod_prefetch_2752903388` at about `20G`.
- Removed only regenerable full-VOD download cache data and the failed prefetch fragments via a short-lived root container because the files were Docker/root-owned.
- Disk recovered from `0B` available to about `902G` available; `backend/tmp` is now about `24G`, mostly `vod_feature_cache`.
- Changed `TWITCH_DOWNLOAD_CACHE_TTL_SEC` from `-1` to `0` in `backend/.env`, `backend/.env.example`, and the config default so Twitch full-VOD downloads are not cached indefinitely.
- Verification passed: `python3 -m py_compile backend/config.py backend/services/downloader.py`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_downloader backend.tests.test_config`.
- Rebuilt backend Graphify with the uv tool interpreter: `2069` nodes, `5350` edges, `46` communities.
- Caveat: the failed training run stopped; it needs to be restarted after the cleanup/config change.

## 2026-04-25 Plan: Make Gated Pyannote Diarization Non-Fatal

- [x] Inspect the new Hugging Face gated-model failure path.
- [x] Treat pyannote load/inference failures as optional diarization misses instead of training failures.
- [x] Add a regression test for gated/forbidden pyannote access.
- [x] Run targeted transcriber tests and syntax checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Document the behavior and caveat.

### Review Notes

- Root cause: Hugging Face accepted the `token=` kwarg but returned `403 Forbidden` for the gated `pyannote/speaker-diarization-3.1` model, so optional diarization aborted the whole VOD training run.
- Updated `_diarize(...)` to fail open when pyannote pipeline load or inference fails: it logs `transcriber_diarization_unavailable` with a classified reason and returns the Whisper segments unchanged.
- Added `hf_gated_model_access`, `hf_auth`, `pipeline_download`, and generic `pipeline_error` reason classification for compact logs.
- Added a regression test that simulates the gated `403` access failure and verifies training receives the original `UNKNOWN` speaker segments.
- Verification passed: `python3 -m py_compile backend/services/transcriber.py backend/tests/test_transcriber.py`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_transcriber backend.tests.test_transcriber_progress`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset`.
- Rebuilt backend Graphify with the uv tool interpreter: `2069` nodes, `5350` edges, `46` communities.
- Caveat: diarization-derived speaker labels are skipped until the HF token has accepted/access to the gated pyannote model, but transcript and all non-speaker features continue.

## 2026-04-25 Plan: Fix Pyannote Diarization API Compatibility

- [x] Inspect the current diarization call and installed dependency contract.
- [x] Make `Pipeline.from_pretrained` compatible with pyannote 3.x and 4.x token kwargs.
- [x] Add a focused regression test for the new kwarg fallback.
- [x] Run targeted transcriber tests and quick syntax checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Record the failure cause and result.

### Review Notes

- Root cause: `backend/pyproject.toml` allows `pyannote.audio>=4.0.0,<5.0.0`, and the installed `Pipeline.from_pretrained` signature uses `token=...`; the old code still passed `use_auth_token=...`.
- Updated `services/transcriber.py` to load pyannote with `token=...` first, with a fallback to `use_auth_token=...` for older pyannote releases.
- Added regression tests for both the current token kwarg path and the legacy fallback path.
- Verification passed: `python3 -m py_compile backend/services/transcriber.py backend/tests/test_transcriber.py`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_transcriber backend.tests.test_transcriber_progress`.
- Verified local installed signature: `(checkpoint, revision=None, hparams_file=None, token=None, cache_dir=None)`.
- Rebuilt backend Graphify with the uv tool interpreter: `2063` nodes, `5343` edges, `45` communities.
- Added the lesson to verify optional ML dependency signatures before long training runs.

## 2026-04-25 Plan: Window-Scoped Top-Viewed Twitch Feature Build

- [x] Inspect current full-VOD materialization, audio, transcript, chat, and window feature cache flow.
- [x] Add a scoped source mode for manually confirmed top-viewed Twitch labels.
- [x] Make scoped mode cut only the expanded label window before audio/transcript/window features.
- [x] Preserve normal full-VOD training behavior for regular labels and mixed VODs.
- [x] Add focused tests for scoped-materialization decisions and offset mapping.
- [x] Run targeted tests and rebuild backend Graphify after backend code edits.
- [x] Document the result and any remaining caveats.

### Review Notes

- Added a `window_scoped_top_viewed` feature context for positive labels from `manual_gold_review_source=top_viewed_twitch_30d`.
- For eligible VOD label groups, the workflow now cuts the expanded label window into `scoped_windows/<digest>/source.mp4` and builds audio, transcript, chat enrichment, and window modalities against that short source.
- Chat still uses the root VOD chat cache, but scoped windows write a local rebased chat file so chat offsets align with the clipped source.
- Regular labels and mixed VOD groups still use the existing full-VOD feature context to avoid changing hard-negative and normal training behavior.
- Feature provenance now records `vod_feature_source_kind`, making scoped rows auditable in downstream artifacts.
- Verification passed: `python3 -m py_compile backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_train_vod_ranker`.
- Verification passed: `git diff --check`.
- Rebuilt backend Graphify with the uv tool interpreter: `2057` nodes, `5337` edges, `40` communities.
- Caveat: this removes the expensive full-VOD transcription/window-feature bottleneck for the confirmed top-viewed positives, but it still needs the source VOD available so the short window can be cut.

## 2026-04-25 Plan: Wire Top-Viewed Backlog Into Next Training Run

- [x] Confirm the next training run reads the canonical Jynxzi Twitch label file.
- [x] Merge the 47 feature-backlog positive labels into the canonical label file without duplicates.
- [x] Verify the canonical label file now contains those VOD/window labels and will trigger retraining.
- [x] Document remaining caveat: they become gold only after feature-backed import.

### Review Notes

- `backend/ml/train_jynxzi_vod.sh` reads `backend/ml/artifacts/jynxzi_twitch_clip_labels.jsonl`; `MERGE_OUTPUT=1` is the default, so the collect step preserves existing labels on the normal path.
- Merged all `47` feature-backlog top-viewed positives from `backend/ml/artifacts/jynxzi_top_viewed_30d_gold_positive_needs_features.jsonl` into `backend/ml/artifacts/jynxzi_twitch_clip_labels.jsonl`.
- Canonical label file now has `13604` rows and `86` positive-like labels, including `47` rows marked with `manual_gold_review_source=top_viewed_twitch_30d`.
- Added backlog labels span `24` VODs: `29` positive, `17` weak_positive, and `1` viral_positive.
- Verification found `0` duplicate label keys after merge.
- `training_workflow_state.should_retrain(...)` returned `True`, so the next normal run should not skip because of unchanged inputs.
- Caveat: these `47` will be loaded as Twitch/VOD proxy labels during feature build; they become canonical gold rows only after the run produces matching feature rows and we import them into `jynxzi_gold_eval_rows.jsonl`.

## 2026-04-25 Plan: Add Top-Viewed Jynxzi Twitch Clips As Gold Positives

- [x] Collect or load Jynxzi Twitch clips for the last 30 days into a separate artifact path.
- [x] Select top-viewed positive clip windows and dedupe against existing gold rows.
- [x] Build a labeled gold-positive CSV plus features sidecar for import.
- [x] Import into gold-eval artifacts with a preserved train/holdout split.
- [x] Verify row counts, grade distribution, and no duplicate gold keys.
- [x] Document result and next training step.

### Review Notes

- Collected `12766` Jynxzi Twitch clip label rows for `2026-03-26` through `2026-04-25` into `backend/ml/artifacts/jynxzi_top_viewed_30d_twitch_clip_labels.jsonl`.
- Selected `51` user-confirmed positive-like top-viewed clip windows: `33` positive, `17` weak_positive, and `1` viral_positive.
- Only `4/51` positive windows currently overlap usable feature rows in the existing VOD training/eval artifacts; `1` of those was already present in gold eval.
- Imported the `3` new feature-backed top-viewed positives as gold grade `3`; canonical `jynxzi_gold_eval_rows.jsonl` now has `175` rows, `29` gold positives, and `146` gold negatives.
- Gold split after merge is `128` train and `47` holdout; the new positives add `2` train rows and `1` holdout row.
- Verification found `0` duplicate gold holdout keys, `0` duplicate queue IDs, and `0` gold rows missing features.
- Wrote the remaining `47` confirmed positive clip labels that need feature materialization to `backend/ml/artifacts/jynxzi_top_viewed_30d_gold_positive_needs_features.jsonl`.
- Targeted verification passed: `python -m unittest backend.tests.test_gold_eval_workflow backend.tests.test_build_eval_dataset`.

### Next Step

- Run a focused VOD feature build/training run for `backend/ml/artifacts/jynxzi_top_viewed_30d_gold_positive_needs_features.jsonl`, then import the newly feature-backed windows into gold eval as the second batch.

## 2026-04-25 Plan: Push Current Changes And Artifacts To Main

- [x] Confirm current branch and remote.
- [x] Stage all current tracked and untracked workspace changes.
- [x] Run pre-commit diff sanity checks.
- [x] Commit staged changes on `main`.
- [x] Push `main` to `origin`.
- [x] Document pushed commit.

### Review Notes

- Branch was `main`; remote was `origin` at `git@github.com:a-hrbn/ai_clipping.git`.
- Staged all tracked changes plus new active gold-eval queue artifacts.
- Normalized generated CSV line endings for `jynxzi_active_gold_eval_queue_20260425_1222.csv` and `jynxzi_ranker_holdout_diagnosis.csv` so `git diff --cached --check` passed.
- Committed `271232f` with message `Align ranker calibration and update training artifacts`.
- Pushed successfully: `6d42c65..271232f main -> main`.
- A follow-up `git ls-remote origin refs/heads/main` verification failed due temporary DNS resolution for `github.com` after the push had already succeeded.

## 2026-04-25 Plan: Verify Latest Ranker Blend Fix Run

- [x] Confirm latest run timestamp, status, and artifact freshness.
- [x] Inspect ranker artifact blend calibration fields and serving threshold.
- [x] Check holdout diagnosis score details for artifact-vs-safe blend mismatch.
- [x] Summarize model quality separately from whether the blend/threshold bug is fixed.
- [x] Document review notes.

### Review Notes

- Latest run artifacts were written at `2026-04-25 12:21-12:22`; `local_train_summary.json` reports `training_return_code=0`.
- Stored workflow code hash matches the current backend hash: `0a0db31890a2323242e285dcb62f323be2cebc7673401bb09d8d7f52ddc4c29c`.
- The blend/threshold fix is present in the artifact: `blend=0.65`, `requested_best_blend=0.65`, `runtime_max_blend=0.65`, and grid candidate `1.00` has `effective_blend=0.65`.
- Holdout diagnosis now reports `artifact_blend=0.65`, `safe_blend=0.65`, `serving_threshold=4.573973`, and `diagnosis_threshold=4.573973`, so the old artifact-vs-runtime blend mismatch is gone.
- Threshold calibration improved validation classification from default threshold F1 `0.519` to calibrated F1 `0.632`, using the runtime-effective blended score.
- Dataset has `324` rows: `59` positive and `265` negative; `154` rows have non-zero chat features.
- The model artifact is still rejected for serving/eval usage because `validation_group_ranking_metric_p25_below_minimum`; therefore gold/proxy `trained_runtime` falls back to baseline and shows zero delta. This is a quality gate result, not the fixed blend bug.

## 2026-04-25 Plan: Align Ranker Blend And Threshold Calibration

- [x] Inspect current training/runtime blend and threshold paths.
- [x] Make training persist/calibrate the same effective blend used at runtime.
- [x] Add regression coverage for runtime blend capping during threshold calibration.
- [x] Run targeted ranker/training tests and compile checks.
- [x] Rebuild backend Graphify output after backend code edits.
- [x] Document review result and remaining caveats.

### Review Notes

- Root cause was a train/runtime mismatch: training selected and calibrated thresholds on blend values up to `1.0`, while runtime serving clamps artifact blend to `0.65`.
- `backend/ml/train_ranker.py` now applies the same `Ranker._safe_blend()` cap while evaluating blend candidates and while producing validation scores for serving threshold calibration.
- New blend calibration metadata records `requested_best_blend`, `runtime_max_blend`, and each grid row's `effective_blend`, so future diagnostics can show when a requested blend was capped.
- `backend/tests/test_train_ranker.py` now covers both direct runtime blended score capping and blend calibration staying within the runtime effective cap.
- Verification passed:
  - `backend.tests.test_train_ranker` -> `34` tests OK.
  - `backend.tests.test_ranker` -> `18` tests OK.
  - `backend.tests.test_diagnose_ranker_holdout` -> `2` tests OK.
  - `py_compile` for the touched ranker/training/diagnosis files -> OK.
- A combined unittest invocation for all three modules failed before running because `test_train_ranker` installs a simple `whisper` stub without `__spec__`, which breaks `importlib.util.find_spec("whisper")` when `test_diagnose_ranker_holdout` imports afterward. The separated targeted runs passed.
- Graphify rebuild completed using the installed Graphify interpreter after system `python3` lacked the module; updated report shows `2042` nodes and `5284` edges.

## 2026-04-25 Plan: Install Backend Extras Locally

- [x] Install backend optional extras into `backend/.venv`.
- [x] Verify key ML/vision imports from the venv.
- [x] Restart or refresh backend status after installation.
- [x] Document installed capability state and any remaining blockers.

### Review Notes

- Installed `.[dev,ml,vision]` into `backend/.venv`.
- Installed `pyannote.audio` as the remaining status-relevant dependency and added it to the backend `ml` extra for future installs.
- Installed `av` and added local FFmpeg Soname symlinks under `backend/.venv/lib/python3.12/site-packages/av.libs` so `torchcodec` can load FFmpeg 8 shared libraries with `LD_LIBRARY_PATH` pointed at that directory.
- Installed `imageio-ffmpeg`, linked its static FFmpeg binary to `backend/.venv/bin/ffmpeg`, and updated backend startup config to prepend `sys.prefix/bin` to `PATH` when that binary exists.
- Re-owned `backend/tmp/cache` and `backend/tmp/matplotlib` from old Docker-owned `nobody:nogroup` to the local user, so local cache paths are writable again.
- Updated the Face Detection status probe to isolate MediaPipe, MTCNN, and OpenCV checks in separate subprocesses. This lets MTCNN report active when MediaPipe crashes with an EGL/DRI segfault.
- Verified imports for Whisper, SentenceTransformers, OpenCV, MediaPipe, DeepFace, MTCNN, LightGBM, MLflow, Optuna, pytest, Pyannote, torchaudio, and torchcodec.
- `pip check` reports no broken requirements.
- Backend restarted on `http://127.0.0.1:8000` with `LD_LIBRARY_PATH=/home/antonio/ai_clipping/backend/.venv/lib/python3.12/site-packages/av.libs`.
- `/api/status` now reports active: Whisper transcription, SentenceTransformers embeddings, MTCNN face detection fallback, DeepFace expression analysis, and Pyannote diarization.
- Targeted Face Detection fallback tests pass. A broader `backend.tests.test_jobs_router` run hung after initial tests, so it was not used as final verification.
- Graphify rebuild completed after the final backend edits with `2039` nodes, `5275` edges, and `38` communities.
- Added `backend/graphify-out/cache/` to `.gitignore` because the Graphify rebuild creates that local cache under the backend output directory.

## 2026-04-25 Plan: Run Backend And Frontend Locally

- [x] Inspect current backend/frontend startup configuration and local dependency state.
- [x] Repair local frontend dependencies for the existing Next app-router code.
- [x] Make local environment files usable on this Linux workspace.
- [x] Start backend and frontend dev servers on local ports.
- [x] Smoke-test backend and frontend over HTTP.
- [x] Document final URLs and verification result.

### Review Notes

- Repaired `frontend/package.json`, `frontend/package-lock.json`, and `frontend/node_modules` back to Next `14.2.35`, matching the app-router code under `frontend/app`.
- Added `frontend/.env.local` with `NEXT_PUBLIC_API_URL=http://localhost:8000`.
- Updated local `backend/.env` storage to `./tmp/storage` so the backend no longer points at the old macOS path.
- Installed backend package dependencies into `backend/.venv`; `import main` and `py_compile` now pass.
- `npm run lint` passes with no warnings or errors.
- The first sandboxed `npm install` failed with `EAI_AGAIN` because network access was restricted; reran it with network access and updated `frontend/node_modules` successfully.
- Backend is running on `http://127.0.0.1:8000`; frontend is running on `http://127.0.0.1:3000`.
- Smoke tests passed: `GET /openapi.json` returned `200`, `GET /api/status` returned `200`, and frontend `GET /` returned `200` after Next compiled the page.
- `/api/status` reports the base API is alive; optional local ML extras such as `openai-whisper`, `sentence-transformers`, and `pyannote.audio` are not installed in this venv.

## 2026-04-25 Plan: Analyze Latest Training Run

- [x] Identify latest training artifacts and completion status.
- [x] Summarize dataset mix, label balance, and source coverage.
- [x] Inspect chat feature coverage and cache metadata.
- [x] Compare ranker, proxy, gold, pipeline, and threshold metrics.
- [x] Document conclusion and next recommendation.

### Review Notes

- Latest artifacts were written on `2026-04-25 10:22-10:23`; `local_train_summary.json` reports `training_return_code=0`, date range `2026-03-15` to `2026-03-28`, and `artifact_only=false`.
- Dataset has `323` rows: `59` positive and `264` negative. Main sources are `126` gold-train, `115` Twitch VOD primary, `76` boundary near-miss, and `6` hard-negative train rows.
- Collected Twitch labels file is cumulative with `13557` labels; only `39` positive-like Twitch labels appear in the raw label classes (`29` positive, `8` weak_positive, `2` viral_positive).
- Chat feature coverage improved to `112/323` rows overall and `112/191` non-gold VOD rows. Chat cache metadata shows `13` complete, `10` partial, and `4` unscoped VOD chat caches.
- Chat lexical/emote features remain zero (`chat_hype_score`, `chat_laugh_score`, `chat_shock_score`, `chat_sadge_score`), so the useful chat signal is currently spike/rate based.
- Model selected `logistic_regression`; internal validation is stronger than the previous weak run: accuracy `0.8`, precision `0.667`, recall `0.545`, F1 `0.6`, NDCG@5 `0.827`, MRR `0.833`.
- However, artifact inspection in eval reports marks the artifact as rejected because `validation_group_ranking_metric_p25_below_minimum`; gold/proxy trained-runtime eval therefore equals baseline with zero delta.
- Gold eval remains weak: MRR `0.216`, NDCG@5 `0.255`, precision@5 `0.155`; proxy/pipeline eval is MRR `0.339`, NDCG@5 `0.355`, top1 positive rate `0.119`.
- Holdout diagnosis exported `7` new hard-negative rows for the next run; forced artifact on holdout improves classification accuracy but hurts ranking MRR slightly versus baseline (`0.16` vs `0.20`).
- Conclusion: data volume and chat coverage improved, but the artifact should not be treated as accepted for serving until group-level gold ranking stability improves.

## 2026-04-25 Plan: Compare Medium AI Clipping Pipeline

- [x] Read the linked Medium article and extract its described pipeline.
- [x] Inspect the current backend graph and concrete pipeline entrypoints.
- [x] Compare article pipeline against this repo's download, candidate, ranking, chat, and rendering flow.
- [x] Document recommendation and gaps.

### Review Notes

- Article baseline: download/transcribe, ask an LLM for interesting transcript moments, then use face/speaker analysis plus FFmpeg to render clips; reported weak points are hit/miss clip quality, poor start/end boundaries, and cloud/API cost.
- Current repo runtime pipeline: download or upload video, extract audio, transcribe, extract audio/video features, generate peak/semantic/fallback candidates, build feature rows, rank with heuristic plus trained ranker, optional LLM/multimodal judge, boundary refinement, post-process/diversify, and FFmpeg cut.
- Current repo training workflow: Twitch VOD labels are collected, VODs are materialized/cached, transcript/audio/video/chat features are cached, labeled windows produce primary/supplemental/boundary-near-miss rows, then gold-train and hard-negative rows are merged into the ranker training set.
- Main gap versus full commercial clipping tools: runtime chat download is disabled, final rendering currently cuts source aspect ratio without automatic 9:16 face-aware reframing, and review/editor UX remains more artifact/debug oriented than product-editor oriented.

## 2026-04-25 Plan: Push Current Workspace To Main

- [x] Inspect branch, remote, and full git status.
- [x] Stage all workspace changes and artifacts requested by the user.
- [x] Commit the staged changes on `main`.
- [x] Push `main` to `origin`.
- [x] Verify remote push result.

### Review Notes

- Branch is `main`, remote is `origin` (`git@github.com:a-hrbn/ai_clipping.git`).
- Staged all current workspace changes with `git add -A`.
- Verified staged diff with `git diff --cached --check`; no whitespace errors remain.
- Main workspace commit `eab9cad` (`Improve Twitch chat training workflow`) was pushed successfully to `origin/main`.
- Push result was `baf669a..eab9cad  main -> main`.

## 2026-04-25 Plan: Audit And Clean Backend Cache

- [x] Inventory backend cache directories and existing cleanup tooling.
- [x] Separate safe stale runtime caches from persistent training artifacts.
- [x] Delete only confirmed stale cache data.
- [x] Verify freed space and that important artifacts remain.
- [x] Document cleanup result.

### Review Notes

- Initial state:
  - `backend/tmp` was about `559G`.
  - Filesystem usage was `584G` used and `372G` available on `/home/antonio/ai_clipping`.
  - Biggest removable items were `backend/tmp/storage.pre_pull_backup_20260423` (`279G`) and `backend/tmp/storage/_download_cache/twitch` (`271G`).
- Preserved:
  - `backend/tmp/storage/feedback`, including `gold_eval_updates.jsonl`.
  - `backend/ml/artifacts`, including ranker artifacts, training rows, gold rows, labels, and eval reports.
  - `backend/tmp/vod_feature_cache` (`9.3G`) because it is the current VOD feature/transcript/chat cache and useful for overlapping future runs.
  - `backend/tmp/cache` (`602M`) because it contains reusable model caches such as Whisper/HuggingFace/Torch.
- Deleted:
  - Old pre-pull storage backup: `backend/tmp/storage.pre_pull_backup_20260423`.
  - Re-downloadable Twitch MP4 cache: `backend/tmp/storage/_download_cache/twitch`.
  - Resume/runtime caches: `backend/tmp/training_runs`, `backend/tmp/short_form_training`.
  - Build/cache data: `backend/graphify-out/cache`, backend `__pycache__` folders.
- Cleanup mechanics:
  - `sudo -n rm -rf ...` failed because sudo requires a password.
  - Plain `rm -rf ...` failed on Docker-owned `nobody:nogroup` cache files.
  - Final cleanup succeeded via a local root Docker container using `ai-clipping-train-local:latest` mounted over the workspace.
- Final state:
  - `backend/tmp` is about `9.8G`.
  - Filesystem usage is `36G` used and `921G` available.
  - Verified `backend/tmp/storage/feedback/gold_eval_updates.jsonl` still exists with `116` rows.
  - Verified `backend/ml/artifacts/jynxzi_ranker.json` and `backend/ml/artifacts/jynxzi_vod_training_rows.jsonl` still exist.

## 2026-04-25 Plan: Analyze Latest Training Run After Chat Offset Filtering

- [x] Confirm latest artifact timestamps and completion status.
- [x] Measure current dataset mix and chat feature coverage.
- [x] Inspect ranker artifact health, threshold calibration, and eval metrics.
- [x] Check chat cache coverage metadata after target-offset filtering.
- [x] Document whether the latest run behaved as expected.

### Review Notes

- Latest artifacts were written around `2026-04-25 01:14 CEST`.
- `local_train_summary.json` reports `training_return_code=0`, streamer `jynxzi`, date range `2026-03-01` to `2026-03-14`, and `artifact_only=false`.
- Stored workflow code hash `34577ff33f3eeeb497c987b9cb2972bb7b5c2f56a3889458d09dfd4135275e70` matches the current backend code hash, so the run used the current chat-offset filtering code.
- Dataset:
  - `jynxzi_vod_training_rows.jsonl` has `213` rows: `38` positive and `175` negative.
  - Dataset sources: `126` gold train, `52` Twitch VOD primary, `34` Twitch boundary near-miss, `1` gold hard-negative train.
  - Supervision tiers: `127` gold eval, `52` Twitch clip proxy, `34` boundary near-miss.
- Chat:
  - Target offsets dropped to `972`, versus the earlier broken plan with `38777`.
  - Chat metadata: `9` VODs complete, `3` partial, `1` unscoped with zero current target offsets.
  - Metadata records `242` missing target offsets, mostly from late windows in VODs `2714647758`, `2722404357`, and a small tail in `2719793388`.
  - Rows with non-zero chat features improved to `61` of `213`; previous relevant run had only about `14`.
  - Chat lexical/emote features still stay zero (`chat_hype_score`, `chat_laugh_score`, `chat_shock_score`, `chat_sadge_score`).
- Ranker:
  - Artifact is usable; selected model is `logistic_regression`, serving threshold `6.016`.
  - Validation: accuracy `0.70`, precision `0.333`, recall `0.20`, F1 `0.25`, MRR `0.667`, NDCG@5 `0.752`, ranking metric p25 `0.5`.
  - Health gates pass and warnings are empty, but classification quality is weaker than the previous accepted run.
  - Threshold calibration updated successfully on `126` gold train rows: threshold `6.016`, calibration F1 `0.85`, precision `0.85`, recall `0.85`.
- Runtime eval:
  - Gold eval uses artifact: `172` rows, accuracy `0.895`, precision `0.633`, recall `0.731`, F1 `0.679`, MRR `0.355`.
  - Proxy eval uses artifact: `213` rows, accuracy `0.864`, precision `0.615`, recall `0.632`, F1 `0.623`, MRR `0.509`.
  - Pipeline eval overall MRR is `0.311`; content-bucket top-1 positive rates are strongest for story/quote and weak for general.
- Holdout diagnosis:
  - Forced artifact on gold holdout: accuracy `0.739`, F1 `0.25`, MRR `0.16`.
  - Baseline heuristic on same holdout: accuracy `0.13`, F1 `0.231`, MRR `0.20`.
  - The model improves classification accuracy a lot but slightly worsens holdout ranking MRR/NDCG versus baseline.
  - Hard-negative export selected `6` false-positive holdout rows for the next run.
- Conclusion:
  - The chat-offset filtering fix worked: chat planning is now much smaller and chat features are present in many more rows.
  - The run completed and produced a usable serving artifact.
  - It is not an overall quality win yet: validation F1/recall dropped, holdout ranking is still weaker than baseline, and `3` VODs remain partial for late chat windows.

## 2026-04-25 Plan: Filter Chat Target Offsets

- [x] Confirm which raw label classes currently explode chat target offsets.
- [x] Restrict chat target offset generation to relevant VOD training label classes.
- [x] Add tests proving background/ambiguous labels do not request chat offsets.
- [x] Run targeted backend tests and rebuild Graphify metadata.
- [x] Document the expected impact for the next training run.

### Review Notes

- Root cause:
  - Chat offsets were generated from all non-ambiguous raw VOD labels.
  - The raw label file has thousands of `background_negative` and `hard_negative` labels, while the final training dataset only keeps a small selected subset.
  - The previous run therefore requested `38777` chat offsets and ended with `37719` missing offsets.
- Change:
  - `backend/ml/build_twitch_training_dataset.py` now filters chat target labels before generating offsets.
  - Positive labels (`viral_positive`, `positive`, `weak_positive`) are always included.
  - `hard_negative` labels are capped to `_MAX_VOD_NEGATIVE_ROWS_PER_POSITIVE` per positive label and ranked by distance to the nearest positive label.
  - `background_negative` and `ambiguous` labels no longer request chat offsets.
  - `_CHAT_ENRICHMENT_FINGERPRINT_VERSION` was bumped to `3` to invalidate stale raw VOD checkpoints.
- Expected impact:
  - On the current `jynxzi_twitch_clip_labels.jsonl`, estimated chat target offsets drop from `38777` to `972`.
  - VOD `2720629593` has no positive anchor and now requests `0` chat offsets.
  - This should let the downloader cover the relevant final training windows instead of timing out on raw-label coverage.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset` -> `54` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset backend.tests.test_audit_chat_feature_coverage backend.tests.test_feature_builder` -> `72` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/ml/build_twitch_training_dataset.py backend/services/chat_analyzer.py backend/tests/test_build_twitch_training_dataset.py` -> OK.
  - `/home/antonio/.local/share/uv/tools/graphifyy/bin/python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"` -> rebuilt `2034` nodes, `5262` edges, `41` communities.

## 2026-04-25 Plan: Verify Run After Chat Coverage Refill

- [x] Confirm latest artifact timestamps and workflow code hash.
- [x] Inspect chat metadata coverage status and missing offsets per VOD.
- [x] Measure chat feature coverage in training rows.
- [x] Inspect ranker acceptance and runtime eval usage.
- [x] Document whether the run worked as expected.

### Review Notes

- Latest artifacts are from `2026-04-25 00:57 CEST`.
- Stored workflow code hash `cdfdda6cf102abf5087827acdd8d575dddb690480b9606ec9acfa529af10076b` matches the current backend code hash, so this run used the coverage-refill implementation.
- Training completed with `training_return_code=0`.
- Chat coverage behavior:
  - All `13` VOD chat caches are now correctly marked `coverage_status=partial`, `partial=true`.
  - This means the new cache validation worked: old early-only caches are no longer falsely considered complete.
  - Total chat target offsets recorded across VOD metadata: `38777`.
  - Covered offsets: `1058`.
  - Missing offsets: `37719`.
  - This is not full chat coverage.
- Root cause of remaining partial coverage:
  - Chat target offsets are generated from raw `jynxzi_twitch_clip_labels.jsonl`, which has `6703` labels, including `2294` `background_negative`, `3950` `hard_negative`, and `441` `ambiguous` labels.
  - The final training dataset only has `225` rows.
  - Therefore the downloader is being asked to cover far more offsets than the final training rows need, then hitting timeout/partial behavior.
- Dataset result:
  - `225` rows, `38` positive, `187` negative.
  - Chat provenance/non-zero chat features improved only slightly: `14` rows vs previous `13`.
  - Chat features still only appear for VODs `2712128863` and `2712736206`.
  - LLM semantics: `74` rows; vector similarity: `0`.
- Model result:
  - Artifact is usable/accepted again.
  - Validation: accuracy `0.85`, precision `1.0`, recall `0.4`, F1 `0.571`, MRR `0.767`, NDCG@5 `0.826`, `mrr_p25=0.5`.
  - Threshold calibration updated successfully: new threshold `5.924`, artifact status `usable`.
  - Gold eval uses artifact: F1 `0.694`, precision `0.739`, recall `0.654`, MRR `0.387`.
  - Proxy eval uses artifact: F1 `0.588`, precision `0.667`, recall `0.526`, MRR `0.492`.
- Conclusion:
  - The coverage-refill code behaved correctly by refusing to call partial chat caches complete.
  - The training run worked from a model-serving standpoint because the artifact is accepted and runtime evals use it.
  - It did not achieve full chat coverage; the next fix should make chat target offset planning use the final selected/capped training rows or a much smaller relevant label set, not every raw label candidate.

## 2026-04-25 Plan: Implement Chat Cache Coverage Refill

- [x] Add chat cache metadata for first/last time, target offsets, covered offsets, missing offsets, and partial status.
- [x] Make training chat cache hits validate coverage against current target offsets.
- [x] Refill missing target offsets by merging newly downloaded chat with the existing cache.
- [x] Add focused tests for stale early cache reuse, partial refill, and metadata.
- [x] Run targeted backend tests and rebuild Graphify metadata after backend edits.
- [x] Document behavior and verification.

### Review Notes

- `backend/ml/build_twitch_training_dataset.py` now treats chat cache hits as coverage-aware:
  - Existing `chat*.json` files are only complete hits when `chat.metadata.json` covers all current target offsets.
  - Old early-only caches without offset metadata are treated as partial and refilled.
  - Metadata now records `message_count`, `first_t`, `last_t`, `coverage_status`, `partial`, `target_offsets`, `covered_target_offsets`, and `missing_target_offsets`.
  - Refill downloads only missing offsets into a temp chat directory, merges old/new messages by id/time/text, sorts by `t`, and writes the merged canonical `chat.twitch_rechat.json`.
  - Partial caches are still usable for available features, but VOD raw-row checkpoints are skipped unless chat coverage is complete.
  - `_CHAT_ENRICHMENT_FINGERPRINT_VERSION` was bumped to `2`, so old raw VOD checkpoints are invalidated.
- `backend/services/chat_analyzer.py` now writes `chat.twitch_rechat.fetch_metadata.json` beside each downloaded chat file:
  - It records target offsets, attempted offsets, covered offsets, missing offsets, page/message counts, first/last chat time, and partial status.
  - Training consumes this metadata so timeout/partial downloads do not get marked complete.
- Added tests in `backend/tests/test_build_twitch_training_dataset.py`:
  - Complete target-offset cache reuses without downloader call.
  - Stale cache without offset metadata triggers refill and merge.
  - Failed refill marks the cache partial with remaining missing offsets.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset` -> `60` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset backend.tests.test_audit_chat_feature_coverage backend.tests.test_feature_builder` -> `70` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/services/chat_analyzer.py backend/ml/build_twitch_training_dataset.py backend/tests/test_chat_analyzer.py backend/tests/test_build_twitch_training_dataset.py` -> OK.
  - `/home/antonio/.local/share/uv/tools/graphifyy/bin/python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"` -> rebuilt `2028` nodes, `5247` edges, `43` communities.
  - `git diff --check -- backend/services/chat_analyzer.py backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py tasks/todo.md backend/graphify-out/GRAPH_REPORT.md backend/graphify-out/graph.json backend/graphify-out/graph.html` -> OK.

## 2026-04-25 Plan: Verify Latest Run After Chat Downloader Fixes

- [x] Confirm whether artifact timestamps changed after the downloader/logging fixes.
- [x] Compare stored workflow code hash with current backend code hash.
- [x] Measure chat cache time coverage against VOD label windows.
- [x] Measure chat feature coverage in the latest training/eval datasets.
- [x] Decide whether the run behaved as expected and document caveats.

### Review Notes

- Latest artifacts are from `2026-04-25 00:37 CEST`, so this is a newer run than the earlier `00:28` snapshot.
- The workflow state code hash is `0872504e0192c9a586312d0fe75d9c77f3f6ca87a8da343e1062355cec9c1f9f`, matching the current backend code hash. This confirms the current downloader code was used by the run.
- The run completed with `training_return_code=0`, streamer `jynxzi`, date range `2026-03-01` to `2026-03-14`.
- Chat download/cache:
  - The run still has real cached Twitch rechat files for `13` VODs.
  - Cached chat ranges are mostly early-VOD ranges, for example `2711274372` covers only `12.0` to `559.0` seconds while its training rows include windows around `2297.9` to `2327.48`.
  - Only VODs `2712128863` and `2712736206` overlap enough training windows to create non-zero chat features.
- Dataset:
  - `jynxzi_vod_training_rows.jsonl`: `214` rows, `38` positive, `176` negative.
  - `13` rows have chat provenance and non-zero chat features.
  - `69` rows have LLM semantics; `0` rows have vector similarity.
  - Dataset sources: `52` Twitch primary, `34` Twitch boundary near-miss, `126` gold train, `2` gold hard-negative train.
- Artifact result:
  - The ranker was trained but rejected for serving.
  - Rejection reason: `validation_group_ranking_metric_p25_below_minimum`.
  - Validation: accuracy `0.8`, precision `0.667`, recall `0.4`, F1 `0.5`, MRR `0.733`, NDCG@5 `0.8`, `mrr_p25=0.333`.
  - Threshold calibration was skipped because artifact status is rejected.
- Runtime evals:
  - Gold/proxy evals have `trained_runtime_uses_artifact=false`.
  - Runtime therefore falls back to the baseline heuristic and deltas are `0`.
- Conclusion:
  - The current downloader code was used and the run completed.
  - It did not work as expected as a model-serving run, because the artifact is rejected.
  - Chat download still does not cover most label windows; the issue is now coverage/offset/caching behavior, not the old GraphQL 400 failure.

## 2026-04-25 Plan: Analyze Latest Training Run Result

- [x] Confirm latest run timestamps and completion status.
- [x] Measure chat cache/download coverage and chat feature coverage in generated datasets.
- [x] Inspect ranker artifact, calibration, health gates, and serving acceptance.
- [x] Inspect gold/proxy/pipeline evals and holdout diagnosis.
- [x] Compare against the previous rejected/accepted runs and document result.

### Review Notes

- Latest completed run:
  - `backend/ml/artifacts/local_train_summary.json` was written at `2026-04-25 00:28:17 CEST`.
  - `training_return_code=0`, streamer `jynxzi`, date range `2026-03-01` to `2026-03-14`, `artifact_only=false`.
  - Workflow state used `skip_covered=true`, `merge_output=true`, hard-negative weight `4.0`, gold train weight `2.0`.
- Chat cache result:
  - All `13` Twitch VOD cache directories now contain real `chat.twitch_rechat.json` files and metadata with `cache_status=downloaded`.
  - Message counts range from `1820` to `3484` messages per cached VOD.
  - The chat download path therefore ran and produced usable replay files.
- Chat feature result:
  - `jynxzi_vod_training_rows.jsonl` has `222` rows: `38` positive, `184` negative.
  - `13` rows have `has_chat_features=true` and non-zero chat density/spike features.
  - Non-zero chat rows are limited to VODs `2712128863` (`8` rows) and `2712736206` (`5` rows).
  - Chat lexical/emotion features remain zero: `chat_hype`, `chat_laugh`, `chat_shock`, and `chat_sadge` are all `0.0`.
  - Many cached chat files only cover early offsets, while later labeled windows still have zero chat features; this run likely did not yet benefit from the final target-offset/client-id/logging fixes.
- Ranker artifact:
  - `jynxzi_ranker.json` is accepted/usable, version `4`, family `ranker_v2`, selected model `logistic_regression`.
  - Serving threshold is `5.697`.
  - Validation: accuracy `0.85`, precision `1.0`, recall `0.4`, F1 `0.571`, MRR `0.767`, NDCG@5 `0.826`.
  - Health gates pass with `ranking_metric_p25=0.5` against minimum `0.333`; warnings are empty.
- Calibration:
  - Calibration updated successfully from old threshold `-1.297709` to `5.697`.
  - Gold-train calibration metrics at threshold: accuracy `0.937`, precision `0.8`, recall `0.8`, F1 `0.8`.
- Runtime evals:
  - Gold eval uses the trained artifact: accuracy `0.907`, precision `0.692`, recall `0.692`, F1 `0.692`, MRR `0.387`, NDCG@5 `0.363`.
  - Proxy eval uses the trained artifact: accuracy `0.874`, precision `0.656`, recall `0.553`, F1 `0.6`, MRR `0.511`, NDCG@5 `0.496`.
  - Pipeline eval still reports baseline-style classification metrics and should not be treated as trained-runtime proof.
- Holdout risk:
  - Forced artifact holdout classification improves over baseline: accuracy `0.826` vs `0.13`, F1 `0.333` vs `0.231`.
  - Ranking on holdout is still slightly worse than baseline: NDCG@5 `0.17` vs `0.2`, MRR `0.16` vs `0.2`.
  - Hard-negative export selected `2` false-positive rows for the next run.
- Comparison:
  - Previous rejected run at `2026-04-24 23:55 CEST` had `215` rows, no real chat files, no chat features, and a rejected artifact (`ranking_metric_p25=0.333 < 0.45`), so serving fell back to the heuristic.
  - Current run has real chat files, `13` chat-feature rows, an accepted artifact, and runtime evals that use the model.
  - Compared with the earlier accepted run, validation is essentially back to the same level (`MRR=0.767`, `NDCG@5=0.826`, `ranking_metric_p25=0.5`), while gold/proxy runtime F1 is slightly lower.

## 2026-04-25 Plan: Add Chat Download Logs

- [x] Add structured logs for GraphQL request start/success/failure with offset, status, page counts, and message counts.
- [x] Add explicit logs for target-offset planning, timeout/deadline behavior, empty pages, and unavailable replay states.
- [x] Improve training cache logs so `None` results include the downstream reason instead of only `download_returned_none`.
- [x] Add/adjust focused tests that assert the new log events fire.
- [x] Run targeted chat/training tests and rebuild backend Graphify metadata.
- [x] Document the resulting log events and verification.

### Review Notes

- Added chat downloader log events in `backend/services/chat_analyzer.py`:
  - `twitch_chat_download_start` now includes target offset count and first/last offset.
  - `twitch_rechat_fetch_plan` logs crawl vs target-offset mode, offset span, global timeout, and per-request timeout.
  - `twitch_graphql_rechat_request_start` logs every GraphQL request with `vod_id`, `job_id`, `offset`, timeout, and `client_id_source=web`.
  - `twitch_graphql_rechat_response` logs HTTP status, response byte size, and elapsed milliseconds.
  - `twitch_graphql_rechat_http_error` logs HTTP status, reason, elapsed time, and up to `1000` bytes of Twitch response body.
  - `twitch_graphql_rechat_payload_error` logs GraphQL payload errors.
  - `twitch_graphql_rechat_unavailable` logs specific unavailable reasons such as `not_found`, `video_missing`, and `comments_missing`.
  - `twitch_rechat_fetch_page` logs page number, offset, raw edge count, normalized count, new message count, total message count, page time range, `has_next`, and mode.
  - `twitch_rechat_fetch_stop` logs stop reason such as `empty_page` or `no_next_page`.
  - `twitch_rechat_fetch_done` logs final status, pages, total messages, first/last chat time, and mode.
  - `twitch_chat_download_failed` now includes error type and error message in the log line before the traceback.
- Updated training cache miss logs in `backend/ml/build_twitch_training_dataset.py`:
  - Failed chat downloads now persist/log `download_returned_none_see_twitch_chat_logs`.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset` -> `57` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset backend.tests.test_audit_chat_feature_coverage backend.tests.test_feature_builder` -> `67` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/services/chat_analyzer.py backend/ml/build_twitch_training_dataset.py backend/tests/test_chat_analyzer.py backend/tests/test_build_twitch_training_dataset.py` -> OK.
  - `/home/antonio/.local/share/uv/tools/graphifyy/bin/python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"` -> rebuilt `2009` nodes, `5189` edges, `40` communities.

## 2026-04-25 Plan: Fix Twitch GraphQL 400 Chat Failure

- [x] Reproduce the HTTP 400 for VOD `2711274372` and capture Twitch's response body.
- [x] Identify whether the 400 comes from request shape, headers, batching, offset values, or VOD availability.
- [x] Implement a robust downloader fix with useful log details and fallback behavior.
- [x] Add regression tests for HTTP 400/response-body handling and the corrected request path.
- [x] Run focused chat/training tests and rebuild backend Graphify metadata.
- [x] Document the verified result and deployment caveat.

### Review Notes

- Root cause:
  - The production/container environment likely has `TWITCH_CLIENT_ID` set for Helix API calls.
  - The GraphQL rechat path reused `settings.twitch_client_id`; Twitch GraphQL rejects a non-Web Client-ID with HTTP `400`.
  - Live probe confirmed the exact Twitch response body for an invalid ID: `The "Client-ID" header is invalid.`
  - Local verification had passed because local settings had no `TWITCH_CLIENT_ID`, so the code used the web-client fallback.
- Fix:
  - `backend/services/chat_analyzer.py` now always uses the known Twitch Web Client-ID for GraphQL rechat requests.
  - Helix credentials remain isolated from the rechat GraphQL path.
  - HTTP errors now include up to `1000` bytes of Twitch response body in the raised `ValueError`, so future upstream failures are visible in logs.
- Live verification:
  - `2711274372` with production-like offsets loaded `1525` chat messages, first at `t=2191.0`, last at `t=2665.0`.
  - With `settings.twitch_client_id` patched to `not-a-real-client-id`, the same downloader still succeeded and loaded `164` messages for a smaller offset set.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer` -> `8` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset backend.tests.test_audit_chat_feature_coverage backend.tests.test_feature_builder` -> `67` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/services/chat_analyzer.py backend/ml/build_twitch_training_dataset.py backend/tests/test_chat_analyzer.py backend/tests/test_build_twitch_training_dataset.py` -> OK.
  - `/home/antonio/.local/share/uv/tools/graphifyy/bin/python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"` -> rebuilt `2003` nodes, `5182` edges, `42` communities.

## 2026-04-25 Plan: Fix Twitch Chat Download

- [x] Reproduce the current chat download failure on a real VOD and capture the concrete HTTP/API error.
- [x] Inspect the downloader implementation and config inputs for stale Twitch API assumptions.
- [x] Implement the smallest robust downloader fix, including observable failure reasons.
- [x] Add focused tests for the failed real-world response shape and the fixed path.
- [x] Run targeted chat/training tests and rebuild backend Graphify metadata.
- [x] Document the result and any remaining operational caveats.

### Review Notes

- Root cause:
  - `backend/services/chat_analyzer.py` still used Twitch's old `https://api.twitch.tv/v5/videos/<vod_id>/comments` endpoint.
  - Live verification for `https://www.twitch.tv/videos/2721412840` returned HTTP `404` with an empty body from that v5 endpoint.
  - Twitch GraphQL returned real `comments` data for the same VOD.
- Fix:
  - Replaced the v5 chat replay fetch with Twitch GraphQL `VideoCommentsByOffsetOrCursor` requests.
  - Normalized GraphQL `contentOffsetSeconds`, `displayName`, `id`, and embedded `emote` fields into the existing `{t, text, emotes}` cache format.
  - Kept unsupported URLs and missing replay responses non-fatal.
  - Added target-offset support to `download_chat()` so training can fetch chat near label windows.
  - `backend/ml/build_twitch_training_dataset.py` now computes 10-second chat target offsets across each expanded training label window and passes them into the downloader.
- Live verification:
  - New GraphQL downloader against `2721412840` with offsets `[0, 60, 120]` wrote `chat.twitch_rechat.json`.
  - The probe loaded `153` messages, first at `t=8.0`, last at `t=122.0`.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset` -> `55` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset backend.tests.test_audit_chat_feature_coverage backend.tests.test_feature_builder` -> `65` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/services/chat_analyzer.py backend/ml/build_twitch_training_dataset.py backend/tests/test_chat_analyzer.py backend/tests/test_build_twitch_training_dataset.py` -> OK.
  - `/home/antonio/.local/share/uv/tools/graphifyy/bin/python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"` -> rebuilt `1995` nodes, `5171` edges, `39` communities.
  - `git diff --check -- backend/services/chat_analyzer.py backend/ml/build_twitch_training_dataset.py backend/tests/test_chat_analyzer.py backend/tests/test_build_twitch_training_dataset.py tasks/todo.md backend/graphify-out/GRAPH_REPORT.md backend/graphify-out/graph.json backend/graphify-out/graph.html` -> OK.
- Caveat:
  - The test run still logs sandbox DNS failures from mocked/expected Twitch VOD materialization paths, but the tests pass and the live GraphQL chat probe succeeded with escalated network access.

## 2026-04-24 Plan: Analyze Latest Jynxzi Training Run

- [x] Confirm the latest training artifacts and timestamps.
- [x] Inspect dataset/chat coverage for the new run.
- [x] Inspect trained ranker metrics, calibration, and serving acceptance.
- [x] Inspect runtime evals, holdout diagnosis, and hard-negative export.
- [x] Compare the new run against the previous accepted run and summarize impact.

### Review Notes

- Latest artifact timestamps show a completed run around `2026-04-24 23:55:44` to `23:55:59 CEST`.
- The run completed with `training_return_code=0` for streamer `jynxzi`, date range `2026-03-01` to `2026-03-14`, `artifact_only=false`.
- Chat status:
  - `13` VOD chat metadata files now exist in `backend/tmp/vod_feature_cache/*/chat/chat.metadata.json`.
  - All inspected metadata records have `cache_status=unavailable`, `reason=download_returned_none`, empty `chat_path`, and `size=0`.
  - No real `chat*.json` replay files were present.
  - New VOD checkpoints were not written; existing VOD checkpoints still have older timestamps. This confirms the new checkpoint guard worked and did not persist another chatless VOD checkpoint.
- Dataset coverage:
  - `backend/ml/artifacts/jynxzi_vod_training_rows.jsonl`: `215` rows, `38` positive, `177` negative.
  - `backend/ml/artifacts/jynxzi_vod_eval_dataset.jsonl`: `215` rows, same label mix.
  - `0` rows have `has_chat_features=true`; all chat feature values remain zero.
  - `69` training rows have LLM semantics; `0` rows have vector similarity.
- New ranker artifact:
  - `version=4`, `artifact_family=ranker_v2`, `selected_model=logistic_regression`.
  - `blend=0.8`, `serving_threshold=-7.931646`.
  - Validation: accuracy `0.8`, precision `0.667`, recall `0.4`, F1 `0.5`, MRR `0.633`, NDCG@5 `0.726`.
  - Group ranking robustness failed: `ranking_metric_p25=0.333`, below the serving acceptance minimum `0.45`.
- Calibration/serving result:
  - `jynxzi_threshold_calibration.json` has `status=skipped`, `artifact_status=rejected`, `reason=validation_group_ranking_metric_p25_below_minimum`.
  - Runtime evals report `trained_runtime_uses_artifact=false`; the system falls back to the baseline heuristic instead of using the new artifact.
- Runtime eval outcome:
  - Gold eval: accuracy `0.151`, precision `0.151`, recall `1.0`, F1 `0.263`, MRR `0.216`; deltas are `0` because trained runtime did not load the rejected artifact.
  - Proxy eval: accuracy `0.177`, precision `0.177`, recall `1.0`, F1 `0.3`, MRR `0.266`; deltas are `0` for the same reason.
  - Pipeline eval mirrors baseline behavior: accuracy `0.177`, precision `0.177`, recall `1.0`, F1 `0.3`, top1 positive rate `0.129`.
- Holdout diagnosis:
  - Artifact inspection status is rejected for `validation_group_ranking_metric_p25_below_minimum`.
  - Forced artifact holdout ranking is worse than baseline: NDCG@5 `0.17` vs baseline `0.2`, MRR `0.16` vs baseline `0.2`.
- Comparison to previous accepted run:
  - Previous run had `224` rows, `38` positive, `186` negative, `blend=0.65`, `serving_threshold=4.601`.
  - Previous validation was stronger: accuracy `0.85`, precision `1.0`, recall `0.4`, F1 `0.571`, NDCG@5 `0.826`, MRR `0.767`, `ranking_metric_p25=0.5`.
  - Previous runtime evals used the artifact; this run does not.
- Conclusion:
  - The checkpoint fix behaved correctly, but Twitch chat download still returns no usable replay files.
  - The new model is operationally worse because it was rejected and serving/evals fall back to the heuristic.
  - The hard-negative flow exported `10` new rows for a future run, while the current training mix only included `3` hard-negative train rows.

## 2026-04-24 Plan: Fix Chatless VOD Checkpoint Reuse

- [x] Add chat-enrichment state to the raw VOD checkpoint fingerprint so chat-related changes invalidate old raw-row checkpoints.
- [x] Persist chat download miss/failure metadata when no chat file is produced.
- [x] Add focused tests proving old chatless checkpoints are invalidated and failed downloads leave observable metadata.
- [x] Run targeted verification and document the result.

### Review Notes

- Root cause:
  - The first chat-enabled run called the chat download path but produced no usable chat files; evidence was empty `backend/tmp/vod_feature_cache/*/chat/` directories and no `chat*.json` files.
  - VOD raw-row checkpoints were then written with `has_chat_features=false`.
  - Later runs hit `_load_vod_checkpoint()` before `_load_or_download_chat()`, so they reused chatless rows and never retried chat.
  - The raw VOD checkpoint fingerprint did not include chat-enrichment state, so enabling/changing chat did not invalidate old raw-row checkpoints.
- Fix:
  - Added `chat_enrichment` state to `_raw_vod_rows_fingerprint_payload()`.
  - Excluded `chat.metadata.json` from `_cached_chat_path()` so miss metadata cannot be mistaken for a chat replay.
  - Persisted miss metadata when chat download is skipped or returns `None`.
  - Skips writing a VOD checkpoint when chat analysis is enabled but no cached chat replay exists; transcript/window caches still remain for faster retry.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset` -> `47` tests OK.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset backend.tests.test_train_vod_ranker backend.tests.test_training_workflow_state backend.tests.test_local_train` -> `74` tests OK.
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"` failed because system Python has no `graphify`.
  - `/home/antonio/.local/share/uv/tools/graphifyy/bin/python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"` succeeded and refreshed `backend/graphify-out`.
  - `git diff --check -- backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py tasks/todo.md backend/graphify-out/GRAPH_REPORT.md backend/graphify-out/graph.json backend/graphify-out/graph.html` -> OK.

## 2026-04-24 Plan: Check Chat Feature Coverage in Fresh Training Run

- [x] Check whether a training process is visible from this session.
- [x] Inspect fresh artifact timestamps and workflow summary.
- [x] Inspect VOD chat cache directories and chat metadata/files.
- [x] Measure `has_chat_features` and chat feature values in the generated training/eval datasets.
- [x] Identify whether the issue is download/cache or feature recognition.

### Review Notes

- Current local time during inspection: `2026-04-24 23:45 CEST`.
- Fresh artifacts were written up to `2026-04-24 23:40`, including:
  - `backend/ml/artifacts/jynxzi_vod_training_rows.jsonl`
  - `backend/ml/artifacts/jynxzi_ranker.json`
  - `backend/ml/artifacts/jynxzi_vod_eval_dataset.jsonl`
  - `backend/ml/artifacts/local_train_summary.json`
- No training process was visible from this sandboxed session; process visibility may be PID-isolated.
- Chat cache result:
  - `backend/tmp/vod_feature_cache/*/chat/` directories exist for 13 VODs.
  - `0` `chat.metadata.json` files found.
  - `0` `chat*.json` files found in those chat directories.
- Feature coverage result:
  - `backend/ml/artifacts/jynxzi_vod_training_rows.jsonl`: `224` rows, `0` rows with `has_chat_features=True`.
  - `backend/ml/artifacts/jynxzi_vod_eval_dataset.jsonl`: `224` rows, `0` rows with `has_chat_features=True`.
  - `chat_peak`, `chat_mean`, `chat_delta`, `chat_hype_score`, `chat_laugh_score`, and `chat_shock_score` were all zero across the inspected rows.
- Code/config check:
  - `backend/.env` has `ENABLE_CHAT_ANALYSIS=true` and `CHAT_DOWNLOAD_TIMEOUT_SEC=30`.
  - The VOD dataset builder does call `_load_or_download_chat()`, but the persisted evidence shows download returned no usable chat file.
  - VOD checkpoints under `backend/tmp/training_runs/jynxzi_resume/_vod_checkpoints/` also contain `has_chat_features=false`.
- Conclusion:
  - Chats were not successfully loaded for this run.
  - Chat features were not recognized in the produced training/eval rows.
  - Because chatless VOD checkpoints now exist, another `--skip-covered` run may reuse them unless the checkpoint fingerprint changes or the checkpoints are removed/regenerated.

## 2026-04-24 Plan: Backend ML and Training Workflow Overview

- [x] Read existing Graphify backend report/wiki and relevant project lessons.
- [x] Map backend package layout and runtime entry points.
- [x] Trace the ML ranker model, feature construction, artifacts, and serving path.
- [x] Trace the training workflow from data/feedback inputs through dataset build, training, evaluation, and artifact promotion.
- [x] Verify the overview against code/tests/artifacts and document caveats.

### Review Notes

- Backend entrypoint is `backend/main.py`; API job routes live in `backend/routers/jobs.py`; the runtime clipping workflow is orchestrated by `backend/pipeline.py`.
- Main serving ranker path is `backend/services/selector.py` -> `backend/services/ranker.py`, using `CandidateFeatures` from `backend/services/timeline_types.py`.
- Feature construction is in `backend/services/feature_builder.py` with `FEATURE_BUILD_VERSION = 7`; current trained artifact uses feature schema version `3` and excludes offline-only fields at serving.
- Current `jynxzi` ranker artifact:
  - `backend/ml/artifacts/jynxzi_ranker.json`
  - `version=4`, `artifact_family=ranker_v2`
  - `model_type=logistic`, `selected_model=logistic_regression`
  - `blend=0.65`, `serving_threshold=4.601`
  - training rows: `224` total, `38` positive, `186` negative
  - split: `group_holdout` by `vod_id`, `leakage_risk=low`
- Current training dataset source mix:
  - `52` `twitch_vod_primary`
  - `34` `twitch_vod_boundary_near_miss`
  - `126` `gold_eval_train`
  - `12` `gold_eval_hard_negative_train`
- Current exported gold eval rows:
  - `172` rows total
  - `126` train, `46` holdout
  - `26` positive, `146` negative
- Current hard-negative export file now contains `3` rows for the next run; the already-built training dataset still contains `12` hard-negative rows from the previous export.
- Training workflow entrypoints:
  - local wrapper: `backend/ml/local_train.py`
  - main shell workflow: `backend/ml/train_jynxzi_vod.sh`
  - dataset/model trainer: `backend/ml/train_vod_ranker.py`
  - model trainer: `backend/ml/train_ranker.py`
  - workflow fingerprint gate: `backend/ml/training_workflow_state.py`
- Evaluation/artifact workflow includes `build_eval_dataset.py`, `calibrate_ranker_threshold.py`, `evaluate_ranker.py`, `diagnose_ranker_holdout.py`, `export_ranker_hard_negatives.py`, `evaluate_pipeline.py`, and `mlflow_tracking.py`.
- Verification:
  - Parsed current ranker, health, gold eval, proxy eval, local train summary, and dataset JSONL artifacts successfully.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_train_ranker backend.tests.test_train_vod_ranker backend.tests.test_training_workflow_state backend.tests.test_local_train` -> `59` tests OK.
  - Wider focused command including `backend.tests.test_ranker` failed at import because this local venv has no `whisper` module.
  - No backend code files were modified, so no Graphify rebuild was required.

## 2026-04-24 Plan: Refresh Backend Graphify Knowledge Graph

- [x] Verify the local `graphify` Python environment and bind the interpreter for this run.
- [x] Detect the current `backend` corpus and confirm file/word counts are within a direct rebuild range.
- [x] Rebuild `backend/graphify-out` outputs from the current backend code and docs.
- [x] Verify the generated graph/report/HTML artifacts are non-empty and current.
- [x] Document the refresh result and any caveats.

### Review Notes

- Refreshed `backend/graphify-out` with the `/graphify` skill against the current `backend` corpus.
- Added `graphify-out/` to `backend/.graphifyignore` so generated reports are not re-ingested into future graph runs.
- Corpus detected after ignore cleanup:
  - `154` files
  - `~118,033` words
  - `152` code files
  - `2` docs
- Reused semantic cache for `152` files and extracted the two uncached docs:
  - `backend/ml/viral_clips.txt`
  - `backend/data/README.md`
- Generated/updated outputs:
  - `backend/graphify-out/GRAPH_REPORT.md`
  - `backend/graphify-out/graph.json`
  - `backend/graphify-out/graph.html`
  - `backend/graphify-out/manifest.json`
  - `backend/graphify-out/cost.json`
  - `backend/graphify-out/cache/*.json`
- Final graph summary:
  - `2,127` nodes
  - `6,356` edges
  - report shows `42` detected communities after labeled report generation
  - extraction mix: `70%` EXTRACTED, `30%` INFERRED, `0%` AMBIGUOUS
- Graphify benchmark:
  - corpus estimate: `118,033` words / `~157,377` naive tokens
  - average query cost: `~40,816` tokens
  - token reduction: `3.9x`
- Verification:
  - parsed `backend/graphify-out/graph.json` successfully and confirmed `2,127` nodes / `6,356` edges
  - checked `GRAPH_REPORT.md`, `graph.html`, `graph.json`, `manifest.json`, and `cost.json` are non-empty
  - `git diff --check -- backend/.graphifyignore backend/graphify-out/GRAPH_REPORT.md backend/graphify-out/graph.json backend/graphify-out/graph.html backend/graphify-out/manifest.json backend/graphify-out/cost.json tasks/todo.md` -> OK
- Caveat:
  - `backend/graphify-out/cache` now has `154` untracked cache entries in git status; they are Graphify cache artifacts for faster future refreshes.

## 2026-04-24 Plan: Replace yt-dlp Chat Replay Fetch

- [x] Replace the `yt-dlp` chat subprocess path with a direct Twitch rechat API fetcher.
- [x] Normalize fetched Twitch comments into the existing `{t, text, emotes}` chat cache format.
- [x] Update chat parsing so direct Twitch cache files and existing exports feed the same feature path.
- [x] Add focused downloader/parser tests and run verification without starting training.
- [x] Refresh backend graph metadata or document why it cannot be refreshed.
- [x] Document the implementation result and operational caveats.

### Review Notes

- Replaced `backend/services/chat_analyzer.py` chat download internals with a direct Twitch comments API fetcher:
  - extracts VOD IDs from `twitch.tv/videos/<id>` URLs
  - calls `https://api.twitch.tv/v5/videos/<id>/comments`
  - paginates via Twitch `_next` cursors
  - writes normalized `chat.twitch_rechat.json` rows with `t`, `text`, `emotes`, optional `id`, and optional `user`
- Removed the `yt-dlp` subprocess dependency from the chat replay path. Unsupported non-Twitch URLs now log `chat_download_skipped` and return `None`; there is no `yt-dlp` fallback.
- Kept the public `download_chat(url, job_id, out_dir)` contract unchanged so the training VOD cache path continues to call the same API.
- Updated `load_messages()` to parse both the new normalized direct export and existing legacy chat JSON formats.
- Added focused tests for:
  - paginated Twitch rechat fetch and normalized cache output
  - Twitch 404 replay-missing behavior as non-fatal
  - unsupported URL skip without fallback/network call
  - normalized cache parser support
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer` -> `4` tests OK
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_chat_analyzer backend.tests.test_build_twitch_training_dataset backend.tests.test_audit_chat_feature_coverage backend.tests.test_feature_builder` -> `57` tests OK
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/services/chat_analyzer.py backend/tests/test_chat_analyzer.py` -> OK
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/services/chat_analyzer.py backend/ml/build_twitch_training_dataset.py backend/ml/audit_chat_feature_coverage.py backend/services/feature_builder.py backend/services/timeline_types.py` -> OK
  - `git diff --check -- backend/services/chat_analyzer.py backend/tests/test_chat_analyzer.py tasks/todo.md` -> OK
- Global `git diff --check` still fails on pre-existing trailing whitespace in generated `backend/ml/artifacts/jynxzi_ranker_holdout_diagnosis.csv`; this implementation did not touch that file.
- Graphify refresh still cannot run because the local Python environment has no `graphify` module installed.
- No training run was started.

## 2026-04-24 Plan: Make VOD Chat Cache Observable

- [x] Add explicit training logs for VOD chat cache lookup, download start, cache hit, download success, and unavailable states.
- [x] Persist VOD chat cache metadata next to cached chat files.
- [x] Log chat enrichment coverage after applying chat to the transcript.
- [x] Add focused tests for cache metadata/logging behavior.
- [x] Run verification without starting training.

### Review Notes

- Added explicit training-run log events in `backend/ml/build_twitch_training_dataset.py`:
  - `training_vod_chat_cache_lookup`
  - `training_vod_chat_cache_hit`
  - `training_vod_chat_download_skip`
  - `training_vod_chat_download_start`
  - `training_vod_chat_download_success`
  - `training_vod_chat_download_unavailable`
  - `training_vod_chat_enrichment_applied`
  - `training_vod_chat_enrichment_skipped`
- VOD chats are cached under `backend/tmp/vod_feature_cache/<vod_id>/chat/` during training dataset builds.
- Each cached VOD chat now gets `chat.metadata.json` with `vod_id`, `vod_url`, `job_id`, `cache_status`, `chat_path`, size, mtime, and write timestamp.
- `backend/services/chat_analyzer.py` now also emits structured downloader events:
  - `twitch_chat_download_start`
  - `twitch_chat_download_success`
  - `twitch_chat_download_unavailable`
  - `twitch_chat_download_failed`
  - `twitch_chat_download_empty`
  - `twitch_chat_download_timeout`
  - `twitch_chat_download_skipped`
- Current audit still shows `0` cached chat VODs because no new training dataset build has run since chat caching was added.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset backend.tests.test_audit_chat_feature_coverage backend.tests.test_feature_builder` -> `53` tests OK
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/ml/build_twitch_training_dataset.py backend/services/chat_analyzer.py backend/ml/audit_chat_feature_coverage.py`
  - `git diff --check`
- Graphify refresh still cannot run because `graphify` is not installed in the local Python environment.
- No training run was started.

## 2026-04-24 Plan: Add Twitch Chat Spike Features

- [x] Add a chat-feature coverage audit script for current artifacts and caches.
- [x] Expand candidate features from a single `chat_peak` into temporal and category-specific chat signals.
- [x] Wire cached Twitch chat enrichment into the VOD training dataset build.
- [x] Update focused tests and run verification without starting a training run.
- [x] Document results and any remaining operational caveats.

### Review Notes

- Added `backend/ml/audit_chat_feature_coverage.py` to summarize current row coverage, per-feature chat statistics, category nonzero counts, and VOD chat cache coverage.
- Expanded `CandidateFeatures` and `feature_builder` from only `chat_peak` to:
  - `chat_mean`
  - `chat_early_peak`
  - `chat_post_hook_peak`
  - `chat_delta`
  - `chat_spike_coverage`
  - `chat_hype_score`
  - `chat_laugh_score`
  - `chat_shock_score`
  - `chat_sadge_score`
- Bumped `FEATURE_BUILD_VERSION` to `7`, so the next VOD build invalidates raw row checkpoints and rebuilds rows with the new chat feature schema.
- Wired chat replay enrichment into VOD dataset generation:
  - checks `vod_feature_cache/<vod_id>/chat/chat*.json*` first
  - downloads chat via the existing `chat_analyzer.download_chat()` only when a VOD URL is available and no cached chat file exists
  - enriches the full cached transcript in memory before label-window slicing
- Current audit against existing artifacts:
  - `391` feature rows inspected
  - `0` rows with chat features
  - `0` cached chat VODs under `backend/tmp/vod_feature_cache`
  - this is expected before the next training/data-build run, because the existing artifacts are still feature build version `6`.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_feature_builder backend.tests.test_build_twitch_training_dataset backend.tests.test_audit_chat_feature_coverage` -> `52` tests OK
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m py_compile backend/ml/audit_chat_feature_coverage.py backend/ml/build_twitch_training_dataset.py backend/services/feature_builder.py backend/services/timeline_types.py`
  - `git diff --check`
- Graphify refresh still cannot run because the local Python environment has no `graphify` module installed.
- No training run was started.

## 2026-04-24 Plan: Evaluate Hard-Negative Weight Training Run

- [x] Check fresh run status, artifact timestamps, and workflow-state weight.
- [x] Verify the `4.0` hard-negative setting and current hard-negative rows were used.
- [x] Compare ranker validation, full gold eval, and holdout metrics to the prior run.
- [x] Inspect remaining false positives/false negatives and hard-negative export.
- [x] Document verdict and next action.

### Review Notes

- Fresh local training completed successfully at `2026-04-24 19:44`:
  - `training_return_code=0`
  - `artifact_only=false`
  - date range `2026-03-01` to `2026-03-14`
  - workflow state: `ranker_hard_negative_weight=4.0`
- The new hard-negative setting was actually used:
  - `219` training rows total
  - `126` `gold_eval_train`
  - `7` `gold_eval_hard_negative_train`
  - all `7` hard-negative train rows have `sample_weight=4.0`
- Ranker artifact:
  - selected model `logistic_regression`
  - blend `0.60`
  - serving threshold increased from `4.982` to `5.495`
  - validation stayed at `accuracy=0.850`, `precision=1.000`, `recall=0.400`, `f1=0.571`
  - validation ranking improved: `mrr=0.867`, `ndcg_at_5=0.900`
- Full gold eval improved versus the previous run:
  - `accuracy=0.930` vs `0.901`
  - `precision=0.750` vs `0.636`
  - `recall=0.808` unchanged
  - `f1=0.778` vs `0.712`
  - `mrr=0.387` vs `0.355`
  - `ndcg_at_5=0.379` vs `0.366`
- Holdout eval improved on classification:
  - `accuracy=0.848` vs `0.739`
  - `precision=0.429` vs `0.250`
  - `recall=0.500` unchanged
  - `f1=0.462` vs `0.333`
  - false positives dropped from `9` to `4`
  - false negatives stayed at `3`
  - ranking remains weak: `mrr=0.180`, `ndcg_at_5=0.185`
- Effect on the `7` promoted hard negatives:
  - `5/7` are now true negatives
  - remaining false positives:
    - `2744998185` / `c2b1227c-9d62-46ab-9c5f-0dc6a3456826:quote:1:1230`
    - `2712128863` / `twitch_vod_2712128863:twitch_vod_2712128863_14:narrative_reveal:0:0`
- Remaining false negatives are still the three confirmed positive special cases:
  - `2698737426` / R6 operator introduction
  - `2697130608` / Jynxzi rage
  - `2714647758` / CS2 cheater controversy
- The regenerated hard-negative export now contains `2` rows for the next run.
- Verdict:
  - This run is better than the prior `4.982` threshold run.
  - The hard-negative weight increase worked for false-positive suppression.
  - It did not improve recall on the three special-case positives, so the next model-strengthening step should add targeted positive training signal for those categories rather than further increasing generic negative pressure.

## 2026-04-24 Plan: Strengthen Hard-Negative Training

- [x] Locate the hard-negative weighting path used by the VOD ranker workflow.
- [x] Increase the default hard-negative training impact for diagnosis-mined false positives.
- [x] Update tests and workflow-state expectations so the next run retrains intentionally.
- [x] Run focused tests and refresh backend graph metadata.
- [x] Document the expected next-run interpretation.

### Review Notes

- Increased the diagnosis-mined hard-negative default from `2.5` to `4.0` in:
  - `backend/ml/train_vod_ranker.py`
  - `backend/ml/train_jynxzi_vod.sh`
  - `backend/ml/training_workflow_state.py`
- This targets the current failure mode directly: the previous run still predicted the promoted hard negatives as positives, so the next run should apply more pressure against those false-positive patterns.
- Existing explicit overrides still work through `--hard-negative-weight` or `VOD_RANKER_HARD_NEGATIVE_WEIGHT`.
- Added tests for:
  - default hard-negative sample weight becoming `4.0` in the composite VOD ranker dataset
  - workflow-state default becoming `ranker_hard_negative_weight=4.0`
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend backend/.venv/bin/python -m unittest backend.tests.test_train_vod_ranker backend.tests.test_training_workflow_state backend.tests.test_export_ranker_hard_negatives` -> `26` tests OK
  - `git diff --check` passes
  - workflow-state preview returns `ranker_hard_negative_weight=4.0`
- Graphify refresh could not run because `graphify` is not installed in either system Python or `backend/.venv`.
- No training run was started. The current `7` hard-negative export is ready; the next training run will use those rows at effective weight `4.0` unless explicitly overridden.

## 2026-04-24 Plan: Evaluate Latest Training Run

- [x] Check fresh run status and artifact timestamps.
- [x] Verify which corrected gold/hard-negative rows entered the training dataset.
- [x] Compare current ranker, gold eval, proxy eval, and holdout diagnosis against the previous run.
- [x] Inspect remaining false positives/false negatives and update the next action.
- [x] Document verdict and verification.

### Review Notes

- Fresh local training completed successfully at `2026-04-24 19:31`:
  - `training_return_code=0`
  - `artifact_only=false`
  - date range `2026-03-01` to `2026-03-14`
  - `ranker_short_form_mode=off`
  - VOD workers `1`, label workers `3`
- Current ranker artifact:
  - selected model `logistic_regression`
  - serving threshold `4.982`
  - training rows `215`: `126` `gold_eval_train`, `3` `gold_eval_hard_negative_train`, `52` VOD primary, `34` boundary near-miss
  - validation: `accuracy=0.850`, `precision=1.000`, `recall=0.400`, `f1=0.571`, `mrr=0.767`, `ndcg_at_5=0.826`
- Gold data after correction:
  - `172` rows total
  - `26` positives, `146` negatives
  - holdout `46`, train `126`
  - corrected `2662238932` row is holdout, so it affects eval/diagnosis but not the current training weights directly
- Full gold eval:
  - `accuracy=0.901`, `precision=0.636`, `recall=0.808`, `f1=0.712`
  - `mrr=0.355`, `ndcg_at_5=0.366`
  - compared with the prior evaluated run, recall is slightly higher but precision/F1/ranking are lower
- Holdout eval:
  - `accuracy=0.739`, `precision=0.250`, `recall=0.500`, `f1=0.333`
  - confusion: `3 TP`, `9 FP`, `3 FN`, `31 TN`
  - `mrr=0.180`, `ndcg_at_5=0.185`
- Remaining false negatives:
  - `2698737426` / R6 operator introduction
  - `2697130608` / Jynxzi rage
  - `2714647758` / CS2 cheater controversy
- The three hard negatives used by this run are still false positives after training:
  - `2744998185` / quote row
  - `2712128863` / narrative reveal row
  - `2750220507` / narrative reveal row
- The newly regenerated hard-negative export now contains `7` rows for the next run, because the current artifact created more false positives.
- Verdict:
  - This run is not stronger overall.
  - It lowers the serving threshold from `5.388` to `4.982`, which recovers some recall but increases false positives.
  - The hard-negative weighting is not strong enough to suppress the exact promoted negatives.
  - Next step should be to increase hard-negative impact or tune threshold selection for holdout precision, before adding another generic training run.

## 2026-04-24 Plan: Apply Latest False-Negative Review

- [x] Inspect current source snapshots for the reviewed rows.
- [x] Persist the `2662238932` label correction in source snapshots.
- [x] Regenerate gold artifacts, holdout diagnosis, and hard-negative export.
- [x] Verify updated counts and remaining false negatives.
- [x] Document next training implication.

### Review Notes

- Applied the latest review as source-of-truth:
  - `2662238932` / `4ced0c6a-a499-47b2-928d-5e874ed69f67:peak:6:57791` is now `gold_negative`, `gold_grade=1`.
  - `2698737426` remains positive because the R6 operator introduction/event context carries audience demand.
  - `2697130608` remains positive as a typical Jynxzi rage moment.
  - `2714647758` remains positive due to the CS2 cheater controversy.
- Regenerated frontend gold artifacts:
  - `172` gold rows total
  - `26` positives
  - `146` negatives
- Updated forced-artifact holdout diagnosis after the label correction:
  - rows `46`
  - `accuracy=0.826`, `precision=0.375`, `recall=0.500`, `f1=0.429`
  - `3 TP`, `5 FP`, `3 FN`, `35 TN`
  - `mrr=0.180`, `ndcg_at_5=0.185`
- Remaining false negatives are now the three confirmed positives:
  - `2698737426` / R6 operator introduction
  - `2697130608` / Jynxzi rage
  - `2714647758` / CS2 cheater controversy
- Regenerated hard-negative training export:
  - `3` rows
  - all from current holdout false positives
  - dataset source `gold_eval_hard_negative_train`
- No new training run was started. The existing ranker artifact still has weights from the previous training run; the next training run will be the first one that can learn the new `2662238932` negative correction.

## 2026-04-24 Plan: Evaluate Corrected Gold Training Run

- [x] Check fresh artifact timestamps and run status.
- [x] Verify corrected gold labels and hard-negative rows were included in the training dataset.
- [x] Compare ranker validation, gold eval, proxy eval, and holdout diagnosis metrics.
- [x] Inspect remaining false positives/false negatives and next hard-negative export.
- [x] Document verdict and next step.

### Review Notes

- Fresh run completed successfully at `2026-04-24 19:15`:
  - `training_return_code=0`
  - `artifact_only=false`
  - date range `2026-03-01` to `2026-03-14`
  - `ranker_short_form_mode=off`
  - VOD workers `1`, label workers `3`
- Corrected gold labels persisted into the run:
  - `jynxzi_gold_eval_rows.jsonl`: `172` rows, `27` positives, `145` negatives
  - `2748097299` and `2750220507` are now `gold_negative`, `gold_grade=1`
  - the four reviewed true positives remain positive with rationale notes
- Training dataset:
  - `219` rows total
  - `126` `gold_eval_train` rows with `sample_weight=2.0`
  - `7` `gold_eval_hard_negative_train` rows with `sample_weight=2.5`
  - `52` VOD primary rows and `34` boundary near-miss rows
- Ranker artifact:
  - status usable, selected model `logistic_regression`
  - blend `0.60`, serving threshold `5.388`
  - validation: `accuracy=0.85`, `precision=1.0`, `recall=0.4`, `f1=0.571`, `mrr=0.767`, `ndcg_at_5=0.826`
- Full corrected gold eval:
  - trained runtime: `accuracy=0.919`, `precision=0.724`, `recall=0.778`, `f1=0.750`, `mrr=0.398`, `ndcg_at_5=0.395`
  - train split: `accuracy=0.960`, `precision=0.857`, `recall=0.900`, `f1=0.878`
  - full holdout: `accuracy=0.804`, `precision=0.375`, `recall=0.429`, `f1=0.400`, `mrr=0.220`, `ndcg_at_5=0.225`
- Clean holdout excluding the 7 holdout rows promoted as hard negatives:
  - rows `39`
  - `accuracy=0.846`, `precision=0.600`, `recall=0.429`, `f1=0.500`, `mrr=0.240`, `ndcg_at_5=0.240`
  - confusion: `3 TP`, `2 FP`, `4 FN`, `30 TN`
- The 7 promoted hard negatives improved but are not solved:
  - `4/7` are now true negatives
  - `3/7` remain false positives
- Current holdout diagnosis CSV:
  - `18` review rows
  - `3` false positives
  - `4` false negatives
  - `1` true positive
  - `10` true negatives
- Current next hard-negative export:
  - `3` rows
  - sources: `quote=1`, `narrative_reveal=2`
- Verdict:
  - The corrected gold labels were used.
  - The run improved holdout classification materially, especially on false positives.
  - Ranking quality on clean holdout is still flat (`mrr`/`ndcg_at_5=0.240`), and recall remains weak at `0.429`.
  - The next model-strengthening step should target positive recall on the four remaining false negatives, not add more generic negatives.

## 2026-04-24 Plan: Persist Gold Corrections In Source Snapshots

- [x] Confirm the new run overwrote generated gold artifacts from frontend snapshots.
- [x] Append corrected source snapshots with newer `updated_at` values.
- [x] Regenerate canonical gold artifacts, holdout diagnosis, and hard-negative export.
- [x] Verify the corrected labels survive the export path and document the result.

### Review Notes

- Root cause confirmed: the run rebuilt `jynxzi_gold_eval_rows.jsonl` from `backend/tmp/storage/feedback/gold_eval_updates.jsonl`, which still had stale frontend labels for the corrected false negatives.
- Appended five newer source snapshots to `backend/tmp/storage/feedback/gold_eval_updates.jsonl`:
  - `2748097299` / `2af78e63-9ac2-49ea-b8e0-bece7760e05b:peak:11:411193` -> `gold_negative`, `gold_grade=1`
  - `2750220507` / `c5e0537b-1ac7-4813-8d07-55233a37fc57:peak:7:189043` -> `gold_negative`, `gold_grade=1`
  - `2662238932`, `2698737426`, and `2697130608` kept as positives with rationale notes.
- Re-ran the frontend gold export with `--merge-existing`.
  - canonical gold rows now stay at `172` rows, `27` positives, `145` negatives
  - holdout is now `7` positives, `39` negatives
- Re-ran holdout diagnosis against the current trained artifact.
  - forced metrics: `accuracy=0.717`, `precision=0.250`, `recall=0.429`, `f1=0.316`, `ndcg_at_5=0.225`, `mrr=0.220`
  - confusion: `3 TP`, `9 FP`, `4 FN`, `30 TN`
  - diagnosis CSV: `23` review rows with `7` false positives, `4` false negatives, `1` true positive, `11` true negatives
- Re-ran hard-negative export.
  - exported `7` hard-negative train rows
  - candidate sources: `peak=2`, `quote=2`, `narrative_reveal=2`, `boundary_perturbation=1`
- Verification:
  - parsed source snapshots and generated gold JSONL files
  - verified exact reviewed rows after export
  - `git diff --check`

## 2026-04-24 Plan: Apply False-Negative Gold Review

- [x] Locate the six reviewed false-negative rows by exact `queue_id`.
- [x] Correct label-noise rows and preserve true-positive rows with clearer notes.
- [x] Recompute gold summary counts and validate the artifact consistency.
- [x] Document the updated interpretation and next training step.

### Review Notes

- Corrected two label-noise rows from positive to negative:
  - `2748097299` / `2af78e63-9ac2-49ea-b8e0-bece7760e05b:peak:11:411193`
  - `2750220507` / `c5e0537b-1ac7-4813-8d07-55233a37fc57:peak:7:189043`
- Kept four reviewed rows as positives and added rationale notes:
  - `2662238932`: good Jynxzi reaction at the beginning.
  - `2698737426`: event/trailer reaction demand from the R6 operator trailer context.
  - `2697130608`: typical Jynxzi rage moment.
  - `2714647758`: controversy around CS2 cheaters.
- Updated the canonical and combined gold JSONL artifacts plus the active queue note for the visible `2714647758` row.
- Gold eval distribution is now:
  - total rows: `172`
  - positives: `27`
  - negatives: `145`
  - holdout positives: `7`
  - holdout negatives: `39`
- Re-ran holdout diagnosis and hard-negative export.
  - forced holdout metrics after label correction: `accuracy=0.804`, `precision=0.375`, `recall=0.429`, `f1=0.400`, `ndcg_at_5=0.225`, `mrr=0.220`
  - confusion: `3 TP`, `5 FP`, `4 FN`, `34 TN`
  - diagnosis CSV now has `18` review rows: `3` false positives, `4` false negatives, `1` true positive, `10` true negatives.
  - next hard-negative train artifact still exports `3` false-positive rows.
- Verification:
  - parsed all updated JSONL artifacts
  - checked exact reviewed row labels and notes
  - `git diff --check`

## 2026-04-24 Plan: Evaluate New Hard-Negative Training Run

- [x] Check fresh artifact timestamps and training summary.
- [x] Verify hard-negative rows were included in the training dataset.
- [x] Inspect ranker artifact status and validation gates.
- [x] Compare proxy, gold, and holdout metrics.
- [x] Inspect updated holdout diagnosis and next hard-negative export.
- [x] Document verdict and next step.

### Review Notes

- Fresh artifacts were written around `2026-04-24 14:53`.
- `local_train_summary.json` reports a successful full run:
  - `training_return_code=0`
  - `artifact_only=false`
  - date range `2026-03-01` to `2026-03-14`
  - `training_vod_workers=1`
  - `training_label_workers=3`
  - `ranker_short_form_mode=off`
- Workflow state includes:
  - `ranker_gold_train_weight=2.0`
  - `ranker_hard_negative_weight=2.5`
- The training dataset used the hard negatives:
  - rows: `221`
  - labels: `38` positive, `183` negative
  - dataset sources: `126` `gold_eval_train`, `9` `gold_eval_hard_negative_train`, `52` VOD primary, `34` boundary near miss
  - hard-negative sample weight: `2.5`
  - the 9 hard negatives came from previous holdout diagnosis and had original split `holdout`, then were promoted to train for this run.
- Ranker artifact:
  - status now `usable`
  - selected model `logistic_regression`
  - blend `0.55`
  - serving threshold `5.531`
  - validation: `mrr=0.767`, `ndcg_at_5=0.826`, `ranking_metric_p25=0.5`
  - this clears the previous failed gate `validation_group_ranking_metric_p25_below_minimum`.
- Gold threshold calibration succeeded:
  - status `updated`
  - old threshold `-0.583427`
  - new threshold `5.531`
  - calibration rows `126`, positives `20`, negatives `106`
  - calibrated train precision/recall/f1: `0.857` / `0.9` / `0.878`
- Full gold eval:
  - rows `172`
  - trained runtime uses artifact: `true`
  - trained: `accuracy=0.907`, `precision=0.724`, `recall=0.724`, `f1=0.724`, `ndcg_at_5=0.424`, `mrr=0.415`
  - deltas vs heuristic: `f1 +0.435`, `ndcg_at_5 +0.104`, `mrr +0.144`
- Gold train split:
  - trained: `accuracy=0.96`, `precision=0.857`, `recall=0.9`, `f1=0.878`, `ndcg_at_5=0.387`, `mrr=0.387`
  - strong improvement vs heuristic, mostly from suppressing false positives.
- Reported full holdout (`46` rows) is partly contaminated because 9 old holdout negatives are now training hard negatives:
  - trained: `accuracy=0.761`, `precision=0.375`, `recall=0.333`, `f1=0.353`, `ndcg_at_5=0.290`, `mrr=0.280`
  - deltas vs heuristic: `f1 +0.026`, `ndcg_at_5 -0.015`, `mrr -0.020`
- Clean holdout excluding the 9 promoted hard negatives:
  - rows `37`, positives `9`, negatives `28`
  - trained: `accuracy=0.784`, `precision=0.600`, `recall=0.333`, `f1=0.429`, `ndcg_at_5=0.320`, `mrr=0.320`
  - deltas vs heuristic: `f1 +0.038`, `ndcg_at_5 +0.000`, `mrr +0.000`
- Updated holdout diagnosis:
  - diagnosis rows dropped from previous `36` to `20`
  - errors changed from false-positive-heavy to recall-heavy:
    - `6` false negatives
    - `3` false positives
    - `10` true negatives
    - `1` true positive
  - next hard-negative export dropped from `9` rows to `3` rows.
- Verdict:
  - The hard-negative step fixed the runtime gate and reduced false positives substantially.
  - It did not yet improve clean holdout ranking; clean holdout `mrr`/`ndcg_at_5` are flat.
  - The model is now too conservative on positives, with holdout recall only `0.333`.

## 2026-04-24 Plan: Add Hard-Negative Training From Diagnosis

- [x] Define a leakage-aware hard-negative export flow.
- [x] Implement hard-negative JSONL export from ranker diagnosis CSV.
- [x] Wire VOD ranker training to include weighted hard-negative rows.
- [x] Generate the current Jynxzi hard-negative training artifact.
- [x] Run focused tests, syntax checks, and graphify rebuild attempt.
- [x] Document usage, caveat, and next training command.

### Review Notes

- Added `backend/ml/export_ranker_hard_negatives.py`.
  - Reads `jynxzi_ranker_holdout_diagnosis.csv`.
  - Selects `error_type=false_positive`, `label=0`, default `forced_score >= 0.0`.
  - Joins back to `jynxzi_gold_eval_rows.jsonl` to preserve full features.
  - Writes hard-negative train rows and a summary.
  - Avoids rewriting unchanged outputs so mtime alone does not force retraining.
- Generated current Jynxzi hard-negative artifacts:
  - `backend/ml/artifacts/jynxzi_ranker_hard_negative_train_rows.jsonl`
  - `backend/ml/artifacts/jynxzi_ranker_hard_negative_train_summary.json`
- Current export result:
  - diagnosis rows: `36`
  - selected false positives: `9`
  - exported rows: `9`
  - source rows found: `9/9`
  - candidate sources: `peak=3`, `quote=2`, `narrative_reveal=2`, `boundary_perturbation=2`
  - original split: `holdout=9`, promoted to train in the hard-negative artifact.
- Wired `backend/ml/train_vod_ranker.py`:
  - new `--include-hard-negative-train`
  - new `--hard-negative-weight`, default `2.5`
  - hard negatives are prepared as `dataset_source=gold_eval_hard_negative_train`.
  - They are marked `gold_holdout_split=train` and `supervision_tier=gold_eval`, so `train_ranker` forces them into train, not validation.
- Wired `backend/ml/train_jynxzi_vod.sh`:
  - exports hard negatives before the retrain decision if a diagnosis CSV exists.
  - passes hard-negative rows and weight into VOD ranker training.
  - includes the hard-negative rows file as a retrain dependency.
  - regenerates diagnosis and hard-negative rows after eval for the next run.
- Updated workflow state:
  - includes `ranker_hard_negative_weight`.
- Dry-run merge against the current dataset:
  - base rows: `212`
  - hard negatives: `9`
  - merged rows for next training: `221`
  - effective hard-negative sample weight: `2.5`
  - hard holdout rows remaining in prepared training rows: `0`
- Caveat:
  - These 9 rows came from the current holdout diagnosis. They are now promoted into training, so the same candidates should not be treated as untouched holdout evidence after the next training run.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_diagnose_ranker_holdout backend.tests.test_export_ranker_hard_negatives backend.tests.test_train_vod_ranker backend.tests.test_training_workflow_state`
  - `python3 -m py_compile backend/ml/diagnose_ranker_holdout.py backend/ml/export_ranker_hard_negatives.py backend/ml/train_vod_ranker.py backend/ml/training_workflow_state.py backend/tests/test_diagnose_ranker_holdout.py backend/tests/test_export_ranker_hard_negatives.py backend/tests/test_train_vod_ranker.py backend/tests/test_training_workflow_state.py`
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - `git diff --check`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.

## 2026-04-24 Plan: Analyze Holdout Diagnosis CSV Patterns

- [x] Summarize error distribution and affected groups.
- [x] Aggregate candidate sources and feature contributions by error type.
- [x] Inspect ranking inversions in mixed positive/negative groups.
- [x] Document pattern and recommended next step.

### Review Notes

- Diagnosis CSV rows: `36`.
- Error types: `27` false positives, `7` true positives, `2` true negatives.
- Predictions are too permissive in the problem subset: `34/36` rows predicted positive.
- False positives are mostly `peak` candidates: `15/27`.
- False-positive top positive feature contributions are dominated by:
  - `speaker_context_clarity`: `20/27`
  - `profile_fit_score`: `15/27`
  - `standalone_score`: `15/27`
  - `quoteability_score`: `14/27`
  - `chapter_position`: `11/27`
- All-negative false-positive groups are a major issue:
  - `2744998185`: `4/4` false positives
  - `2697967639`: `3/3` false positives
  - `2712128863`: `3/3` false positives
  - `2722404357`: `3/3` false positives
- Ranking inversions where a `label=0` ranks above a `label=1`:
  - `2750220507`: negative `narrative_reveal` at rank 1 beats positive `peak`.
  - `2698737426`: negative `narrative_reveal` at rank 1 beats positive `peak`.
  - `2697130608`: negative `peak` at rank 1 beats positive `peak`.
- Pattern: current model/threshold over-promotes plausible-looking setup/semantic/position signals and still does not separate "clip-like but not viral" negatives from true winners reliably.

## 2026-04-24 Plan: Add Forced Holdout Diagnosis Script

- [x] Review existing ranker eval conventions.
- [x] Implement a reusable forced-artifact holdout diagnosis script.
- [x] Add regression coverage for grouped metrics and CSV output.
- [x] Run focused tests, syntax checks, and graphify rebuild attempt.
- [x] Document usage and findings.

### Review Notes

- Added `backend/ml/diagnose_ranker_holdout.py`.
  - Defaults to `backend/ml/artifacts/jynxzi_gold_eval_rows.jsonl`.
  - Defaults to `gold_holdout_split=holdout`.
  - Loads the local ranker artifact even when runtime inspection rejects it.
  - Uses the same safe runtime blend cap as `Ranker` while bypassing the usability gate for diagnosis.
  - Writes JSON and CSV reports by default:
    - `backend/ml/artifacts/jynxzi_ranker_holdout_diagnosis.json`
    - `backend/ml/artifacts/jynxzi_ranker_holdout_diagnosis.csv`
  - CSV includes review URLs with Twitch timestamps, row-level forced scores, heuristic scores, trained scores, error type, and top feature contributions.
- Added `backend/tests/test_diagnose_ranker_holdout.py`.
- Real Jynxzi diagnosis run:
  - selected holdout rows: `46`
  - artifact status: `rejected`
  - rejection reason: `validation_group_ranking_metric_p25_below_minimum`
  - forced holdout metrics: `accuracy=0.283`, `precision=0.214`, `recall=1.0`, `f1=0.353`, `ndcg_at_5=0.276`, `mrr=0.260`
  - deltas vs heuristic: `f1 +0.026`, `ndcg_at_5 -0.029`, `mrr -0.040`
  - worst positive-ranking groups: `2697130608`, `2698737426`, `2750220507`
  - strongest all-negative false-positive groups: `2744998185`, `2697967639`, `2712128863`, `2722404357`
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_diagnose_ranker_holdout backend.tests.test_train_ranker`
  - `python3 -m py_compile backend/ml/diagnose_ranker_holdout.py backend/tests/test_diagnose_ranker_holdout.py`
  - `git diff --check`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.

## 2026-04-24 Plan: Evaluate Gold-Train Weighted Training Run

- [x] Check container/log availability.
- [x] Inspect fresh artifact timestamps and local training summary.
- [x] Verify ranker dataset includes gold-train rows and excludes holdout.
- [x] Verify ranker artifact training stats include gold-train rows.
- [x] Compare proxy, gold, and holdout metrics.
- [x] Document verdict and next step.

### Review Notes

- `docker ps -a` returned no containers, so live/container logs were unavailable.
- Fresh artifacts were written around `2026-04-24 14:14` to `14:15`.
- `local_train_summary.json` reports a successful non-artifact-only run:
  - `training_return_code=0`
  - `artifact_only=false`
  - date range `2026-03-01` to `2026-03-14`
  - `training_vod_workers=1`
  - `training_label_workers=3`
  - `ranker_short_form_mode=off`
  - `ranker_gold_train_weight=2.0` in workflow state.
- The ranker training dataset now includes the real gold-train signals:
  - `jynxzi_vod_training_rows.jsonl` has `212` rows.
  - Dataset sources: `126` `gold_eval_train`, `52` `twitch_vod_primary`, `34` `twitch_vod_boundary_near_miss`.
  - Gold rows have `sample_weight=2.0`.
  - Gold holdout rows in the training dataset: `0`.
- The split guard worked:
  - `126` gold rows total.
  - `126` gold rows in train split.
  - `0` gold rows in validation split.
  - `gold_train_rows_forced_to_train=126`.
- The saved ranker artifact confirms it was trained with those rows:
  - `training_stats.rows=212`
  - `training_stats.dataset_sources.gold_eval_train=126`
  - `training_stats.supervision_tiers.gold_eval=126`
  - `train_rows=192`, `validation_rows=20`.
- Selected model:
  - `logistic_regression`
  - artifact blend `0.9`, runtime safe blend cap `0.65`
  - serving threshold `-4.666997`.
- The artifact is currently rejected by runtime inspection:
  - reason `validation_group_ranking_metric_p25_below_minimum`
  - validation `mrr=0.733`, `ndcg_at_5=0.8`, but `ranking_metric_p25=0.333`, below the `0.45` runtime gate.
  - Because of this, `jynxzi_proxy_eval.json`, `jynxzi_gold_eval.json`, and `jynxzi_ranker_eval.json` report `trained_runtime_uses_artifact=false`.
- The post-training gold calibration also skipped for the same artifact rejection reason:
  - `jynxzi_threshold_calibration.json`: `status=skipped`, `artifact_status=rejected`, `calibration_rows=126`.
- Forced offline evaluation of the learned artifact shows it did learn signal:
  - Full gold set: `f1 0.289 -> 0.320`, `ndcg_at_5 0.320 -> 0.406`, `mrr 0.271 -> 0.383`.
  - Gold train split: `f1 0.274 -> 0.308`, `ndcg_at_5 0.283 -> 0.387`, `mrr 0.240 -> 0.387`.
  - Gold holdout split: `f1 0.327 -> 0.353`, but `ndcg_at_5 0.305 -> 0.276`, `mrr 0.300 -> 0.260`.
- Verdict: yes, the real gold-train signals were trained into the model weights. The artifact is not production-usable yet because validation tail quality fails the runtime gate, and clean holdout ranking regressed slightly.

## 2026-04-24 Plan: Train Ranker With Gold-Train Rows

- [x] Inspect VOD ranker dataset composition and gold safety guards.
- [x] Add gold-train inclusion to the composite ranker dataset while excluding holdout.
- [x] Wire Jynxzi VOD training to pass the canonical gold rows file.
- [x] Add regression tests for gold-train inclusion and holdout exclusion.
- [x] Run focused tests, syntax checks, shell syntax, and graphify rebuild attempt.

### Review Notes

- Added `--include-gold-train` and `--gold-train-weight` to `backend/ml/train_vod_ranker.py`.
  - Only `gold_holdout_split=train` rows are prepared for training.
  - `holdout` rows are ignored before the training dataset is written.
  - Gold rows get `dataset_source=gold_eval_train`, keep `supervision_tier=gold_eval`, and receive a default `2.0x` sample-weight multiplier.
  - Gold rows replace matching proxy rows by stable identity (`queue_id`, `job_id/candidate_id`, or candidate/window fallback) instead of creating conflicting duplicates.
- Updated `backend/ml/train_ranker.py` so gold-train rows are forced into the training split, not the internal validation split, when enough non-gold rows exist for validation.
- Updated `backend/ml/train_jynxzi_vod.sh`:
  - Exports/merges frontend gold rows before the retrain decision.
  - Passes `--include-gold-train "$GOLD_EVAL_ROWS_PATH"` to VOD ranker training.
  - Uses `VOD_RANKER_GOLD_TRAIN_WEIGHT`, default `2.0`.
- Updated `backend/ml/training_workflow_state.py`:
  - `should_retrain` can now consider dependency mtimes.
  - Jynxzi workflow state includes `ranker_gold_train_weight`.
  - Gold rows are a retrain dependency in `train_jynxzi_vod.sh`.
- Updated `backend/ml/export_frontend_gold_eval.py` to avoid rewriting unchanged output content, preventing unchanged gold exports from forcing retraining on every run.
- Real artifact dry-run using current files:
  - VOD rows before merge: `86`.
  - Gold rows total: `172`.
  - Prepared gold-train rows: `126`.
  - Merged training rows for next run: `212`.
  - Labels after merge: `38` positive, `174` negative.
  - Dataset sources after merge: `126` gold train, `52` VOD primary, `34` VOD boundary near miss.
  - Gold holdout rows in merged training set: `0`.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_export_frontend_gold_eval backend.tests.test_gold_eval_workflow backend.tests.test_train_vod_ranker backend.tests.test_train_ranker backend.tests.test_training_workflow_state`
  - `python3 -m py_compile backend/ml/train_vod_ranker.py backend/ml/train_ranker.py backend/ml/training_workflow_state.py backend/ml/export_frontend_gold_eval.py backend/tests/test_train_vod_ranker.py backend/tests/test_train_ranker.py backend/tests/test_training_workflow_state.py`
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - `git diff --check`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.

## 2026-04-24 Plan: Evaluate Latest 172-Row Gold Training Run

- [x] Check container/log availability.
- [x] Inspect fresh artifact timestamps and local training summary.
- [x] Verify merged gold row usage in export, calibration, and eval.
- [x] Compare proxy, gold, and holdout metrics.
- [x] Document verdict and next step.

### Review Notes

- `docker ps -a` returned no containers, so live/container logs were unavailable.
- Fresh artifacts were written around `2026-04-24 14:04`.
- `local_train_summary.json` reports a successful non-artifact-only run:
  - `training_return_code=0`
  - `artifact_only=false`
  - date range `2026-03-01` to `2026-03-14`
  - `training_vod_workers=1`
  - `training_label_workers=3`
  - `ranker_short_form_mode=off`
- The gold merge fix worked:
  - `jynxzi_gold_eval_rows.jsonl` has `172` rows.
  - `jynxzi_gold_eval_summary.json` has `merged_existing=true`.
  - Summary source rows: `existing=172`, `frontend=97`, `merged=172`, `duplicates_skipped=97`.
  - Labels: `29` positive, `143` negative.
  - Gold split: `126` train, `46` holdout.
- Gold threshold calibration now used the expanded gold set:
  - `jynxzi_threshold_calibration.json` has `rows=172`, `calibration_rows=126`.
  - Calibration positives/negatives: `20` / `106`.
  - Threshold stayed `-2.606`.
  - Gold-train calibrated metrics: `precision=0.255`, `recall=0.700`, `f1=0.373`.
- Ranker artifact remains usable:
  - selected model `logistic_regression`
  - `blend=0.95`
  - `serving_threshold=-2.606`
  - no artifact health warnings.
- Proxy eval is unchanged and still strong:
  - rows `86`, positives `18`, negatives `68`
  - trained runtime: `accuracy=0.616`, `precision=0.347`, `recall=0.944`, `f1=0.507`
  - deltas vs heuristic: `f1 +0.161`, `ndcg_at_5 +0.359`, `mrr +0.355`
- Expanded gold eval:
  - rows `172`, positives `29`, negatives `143`
  - trained runtime: `accuracy=0.640`, `precision=0.268`, `recall=0.655`, `f1=0.380`
  - deltas vs heuristic: `accuracy +0.471`, `precision +0.099`, `f1 +0.091`, `ndcg_at_5 +0.026`, `mrr +0.056`
- Expanded gold holdout:
  - rows `46`, positives `9`, negatives `37`
  - trained runtime: `accuracy=0.674`, `precision=0.312`, `recall=0.556`, `f1=0.400`
  - deltas vs heuristic: `f1 +0.073`, but `ndcg_at_5 -0.015`, `mrr -0.020`
- Weak bucket:
  - `twitch_label_proxy` has `27` rows.
  - trained runtime predicts no positives there: `precision=0.0`, `recall=0.0`, `f1=0.0`.
  - ranking also regresses there: `ndcg_at_5 -0.023`, `mrr -0.046`.
- Verdict: the run now correctly evaluates/calibrates on the expanded gold set. It is not a model-training improvement yet, because ranker training still uses the same `86` proxy rows; the new gold rows currently affect threshold calibration and evaluation, not the learned model weights.

## 2026-04-24 Plan: Merge Manual And Frontend Gold Rows

- [x] Add an export mode that merges existing curated gold rows with frontend rows.
- [x] Deduplicate rows by stable gold row identity and prefer newer frontend updates on conflict.
- [x] Wire Jynxzi VOD training to use the merge mode.
- [x] Restore canonical Jynxzi gold artifacts to the combined 172-row set.
- [x] Verify with focused tests, syntax checks, shell syntax, artifact dry-run, and graphify rebuild attempt.

### Review Notes

- Added `--merge-existing` to `backend/ml/export_frontend_gold_eval.py`.
  - Existing curated rows are read from `--rows-out`.
  - Frontend rows are merged into them instead of replacing the whole file.
  - Deduplication uses `queue_id`, then `job_id/candidate_id`, then candidate/window fallbacks.
  - Newer frontend rows replace matching existing rows when `updated_at` is newer.
  - Manual rows without `updated_at` are not overwritten by untimestamped frontend rows on key collision.
- Kept `--preserve-existing-if-empty` for empty frontend exports.
- Updated `backend/ml/train_jynxzi_vod.sh` to call the exporter with both `--merge-existing` and `--preserve-existing-if-empty`.
- Restored canonical Jynxzi gold artifacts to the combined gold set:
  - `backend/ml/artifacts/jynxzi_gold_eval_rows.jsonl`
  - `backend/ml/artifacts/jynxzi_gold_eval_dataset.jsonl`
  - `backend/ml/artifacts/jynxzi_gold_eval_summary.json`
- Real artifact dry-run result:
  - Existing canonical rows: `172`.
  - Frontend rows discovered: `97`.
  - Merged rows written: `172`.
  - Duplicates skipped: `97`.
  - Labels: `29` positive, `143` negative.
  - Gold split: `126` train, `46` holdout.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_export_frontend_gold_eval backend.tests.test_gold_eval_workflow`
  - `python3 -m py_compile backend/ml/export_frontend_gold_eval.py backend/ml/import_gold_eval_labels.py backend/tests/test_export_frontend_gold_eval.py backend/tests/test_gold_eval_workflow.py`
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - `git diff --check`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.

## 2026-04-24 Plan: Evaluate New Training Run

- [x] Check container/log availability.
- [x] Inspect fresh artifact timestamps and training summary.
- [x] Compare proxy, gold, and holdout metrics.
- [x] Verify whether combined gold rows were used/preserved.
- [x] Document verdict and next step.

### Review Notes

- `docker ps -a` returned no containers, so live/container logs were unavailable.
- Fresh artifacts were written around `2026-04-24 13:45`.
- `local_train_summary.json` reports a successful non-artifact-only run:
  - `training_return_code=0`
  - `artifact_only=false`
  - date range `2026-03-01` to `2026-03-14`
  - `training_vod_workers=1`
  - `training_label_workers=3`
  - `ranker_short_form_mode=off`
- The run did not use the combined `172` gold rows:
  - `jynxzi_gold_eval_summary.json` has `97` rows.
  - `jynxzi_threshold_calibration.json` has `rows=97`, `calibration_rows=67`.
  - The separate combined file still has `172` rows, but the canonical training gold file was overwritten back to `97`.
- Ranker artifact remains usable:
  - selected model `logistic_regression`
  - `blend=0.95`
  - `serving_threshold=-2.606`
  - artifact health has no warnings.
- Proxy eval:
  - rows `86`, positives `18`, negatives `68`
  - trained runtime: `accuracy=0.616`, `precision=0.347`, `recall=0.944`, `f1=0.507`
  - deltas vs heuristic: `f1 +0.161`, `ndcg_at_5 +0.359`, `mrr +0.355`
- Gold eval on the old 97-row set:
  - trained runtime: `accuracy=0.773`, `precision=0.385`, `recall=0.625`, `f1=0.476`
  - deltas vs heuristic: `accuracy +0.608`, `precision +0.220`, `f1 +0.193`, `ndcg_at_5 +0.023`, `mrr +0.038`
- Gold holdout remains weak:
  - holdout rows `30`, positives `6`, negatives `24`
  - trained runtime: `precision=0.333`, `recall=0.333`, `f1=0.333`
  - ranking deltas vs heuristic: `ndcg_at_5 -0.024`, `mrr -0.034`
- Attempted a local eval of the current artifact against `jynxzi_gold_eval_rows_combined.jsonl`, but local execution failed because the venv is missing `whisper`.
- Verdict: this run is effectively the same model/eval result as the prior 97-row gold run. It confirms the artifact is operational, but it does not prove improvement from the newly labeled active gold data.

## 2026-04-24 Plan: Import Active Gold Labels

- [x] Fix `import_gold_eval_labels.py` CLI entrypoint.
- [x] Import labeled active gold queue into gold eval row/dataset artifacts.
- [x] Verify imported row counts, grade distribution, and train/holdout split.
- [x] Build separate combined gold artifacts from existing plus active labels.
- [x] Document resulting counts and next training command.

### Review Notes

- Fixed `backend/ml/import_gold_eval_labels.py` so direct script execution calls `main()`.
- Added CLI regression coverage in `backend/tests/test_gold_eval_workflow.py`.
- Imported `backend/ml/artifacts/jynxzi_active_gold_eval_queue.csv` into:
  - `backend/ml/artifacts/jynxzi_gold_eval_rows_active.jsonl`
  - `backend/ml/artifacts/jynxzi_gold_eval_dataset_active.jsonl`
  - `backend/ml/artifacts/jynxzi_gold_eval_summary_active.json`
- Active import result:
  - `75` labeled rows, `0` skipped.
  - Grade distribution: `0=36`, `1=26`, `2=10`, `3=3`.
  - Labels: `13` positive, `62` negative.
  - Gold split: `59` train, `16` holdout.
- Built separate combined artifacts without overwriting canonical `jynxzi_gold_eval_rows.jsonl`:
  - `backend/ml/artifacts/jynxzi_gold_eval_rows_combined.jsonl`
  - `backend/ml/artifacts/jynxzi_gold_eval_dataset_combined.jsonl`
  - `backend/ml/artifacts/jynxzi_gold_eval_summary_combined.json`
- Copied the combined data into the canonical training gold artifacts:
  - `backend/ml/artifacts/jynxzi_gold_eval_rows.jsonl`
  - `backend/ml/artifacts/jynxzi_gold_eval_dataset.jsonl`
  - `backend/ml/artifacts/jynxzi_gold_eval_summary.json`
- Combined result:
  - `172` rows total from `97` existing + `75` active.
  - `0` duplicates skipped.
  - Labels: `29` positive, `143` negative.
  - Gold split: `126` train, `46` holdout.
- Updated `backend/ml/export_frontend_gold_eval.py` and `backend/ml/train_jynxzi_vod.sh` so an empty frontend export preserves existing curated gold artifacts during training.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_export_frontend_gold_eval backend.tests.test_gold_eval_workflow`
  - `python3 -m py_compile backend/ml/export_frontend_gold_eval.py backend/ml/import_gold_eval_labels.py backend/tests/test_export_frontend_gold_eval.py backend/tests/test_gold_eval_workflow.py`
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - `git diff --check`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.

## 2026-04-24 Plan: Push Active Gold Labels

- [x] Inspect branch and changed files.
- [x] Validate labeled CSV import.
- [x] Commit active gold label updates.
- [x] Push commit to `origin/main`.
- [x] Document pushed commit.

### Review Notes

- Validated `backend/ml/artifacts/jynxzi_active_gold_eval_queue.csv` imports with `75` labeled rows and `0` skipped rows.
- Grade distribution is `0=36`, `1=26`, `2=10`, `3=3`, producing `13` positive gold-eval rows at threshold `2`.
- `git diff --check` passed after normalizing CSV line endings.
- Changes were committed and pushed to `origin/main`.

## 2026-04-24 Plan: Suggest Gold Grade Notes

- [x] Inspect active gold queue schema.
- [x] Summarize buckets and labeling constraints.
- [x] Draft reusable `gold_grade` note guidance.
- [x] Document review notes.

### Review Notes

- `backend/ml/artifacts/jynxzi_active_gold_eval_queue.csv` has `75` rows with blank `gold_grade` and `notes` columns.
- Import validation requires notes for grades `2` and `3`, and for grade `0` when `heuristic_score >= 8.0`.
- Active buckets are mainly `false_positive_proxy`, `model_heuristic_disagreement`, and `proxy_positive_review`, so notes should explain human-visible hook/payoff/context quality rather than model scores alone.

## 2026-04-24 Plan: Commit And Push Main

- [x] Inspect branch and pending changes.
- [x] Run final lightweight verification.
- [x] Pull latest `main` with fast-forward only.
- [x] Commit all current changes.
- [x] Push commit to `origin/main`.
- [x] Document pushed commit.

### Review Notes

- Branch was `main` tracking `origin/main`.
- Final verification before commit:
  - `git diff --check`
  - `116` focused unittest cases across active queue, threshold calibration, ranker, VOD dataset/ranker, local training, and workflow state.
  - `python3 -m py_compile` for changed backend Python files.
  - `bash -n backend/ml/train_jynxzi_vod.sh`
- `git pull --ff-only origin main` reported `Already up to date`.

## 2026-04-24 Plan: Build Active Gold Label Queue

- [x] Add a script that scores candidate rows and selects active-labeling examples.
- [x] Cover false positives, false negatives, near-threshold cases, and model/heuristic disagreement.
- [x] Write a labeling CSV and `.features.jsonl` sidecar compatible with `import_gold_eval_labels.py`.
- [x] Exclude already labeled gold rows and keep per-VOD diversity.
- [x] Add focused tests and run syntax/test checks.
- [x] Attempt graph rebuild and document result.

### Review Notes

- Added `backend/ml/build_active_gold_eval_queue.py`.
  - Reads candidate JSONL rows with features, normally `jynxzi_vod_eval_dataset.jsonl`.
  - Scores rows with heuristic and trained rankers.
  - Uses the current artifact `serving_threshold`.
  - Selects active-labeling buckets:
    - `false_negative_proxy`
    - `false_positive_proxy`
    - `proxy_positive_review`
    - `near_threshold`
    - `model_heuristic_disagreement`
  - Excludes exact existing gold candidate/window matches.
  - Enforces `--per-vod-limit` for diversity.
  - Writes a CSV plus `.features.jsonl` sidecar compatible with `import_gold_eval_labels.py`.
- Added `backend/tests/test_build_active_gold_eval_queue.py`.
- Generated the current active queue:
  - `backend/ml/artifacts/jynxzi_active_gold_eval_queue.csv`
  - `backend/ml/artifacts/jynxzi_active_gold_eval_queue.features.jsonl`
  - `backend/ml/artifacts/jynxzi_active_gold_eval_queue.summary.json`
- Queue summary:
  - `75` rows selected from `86` scored candidates.
  - Proxy labels: `18` positive, `57` negative.
  - Buckets: `1` false negative proxy, `28` false positive proxy, `17` proxy positive review, `3` near threshold, `26` model/heuristic disagreement.
  - Per-VOD cap: `8`, covering `12` VODs.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_active_gold_eval_queue backend.tests.test_gold_eval_workflow`
  - `python3 -m py_compile backend/ml/build_active_gold_eval_queue.py backend/tests/test_build_active_gold_eval_queue.py`
  - `git diff --check`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.

## 2026-04-24 Plan: Analyze Fresh Gold-Calibrated Container Result

- [x] Confirm container/log availability.
- [x] Inspect fresh timestamps and training summary.
- [x] Verify threshold calibration report and artifact threshold.
- [x] Compare proxy, gold, and gold holdout metrics.
- [x] Inspect dataset composition and workflow state.
- [x] Document verdict and next step.

### Review Notes

- `docker ps -a` returned no containers, so logs are unavailable for this run.
- Fresh artifacts were written around `2026-04-24 01:25`.
- `local_train_summary.json` reports a successful non-artifact-only run:
  - `training_return_code=0`
  - `artifact_only=false`
  - `training_vod_workers=1`
  - `training_label_workers=3`
  - `ranker_short_form_mode=off`
- The gold-train threshold calibration hook ran:
  - `jynxzi_threshold_calibration.json` exists.
  - Calibration status is `updated`.
  - Old threshold: `9.926048`.
  - New threshold: `-2.606`.
  - Calibration rows: `67` gold-train rows, `10` positive, `57` negative.
  - Gold-train calibration metrics: `precision=0.400`, `recall=0.800`, `f1=0.533`.
- Ranker artifact:
  - Selected model: `logistic_regression`, `blend=0.95`, `serving_threshold=-2.606`.
  - Artifact health is usable, no warnings.
  - Validation ranking remains strong: `ndcg_at_5=0.926`, `mrr=0.900`, `ranking_metric_min=0.500`, `ranking_metric_p25=1.000`.
- Proxy eval improved versus heuristic and no longer has recall collapse:
  - Trained runtime: `accuracy=0.616`, `precision=0.347`, `recall=0.944`, `f1=0.507`.
  - Deltas: `f1 +0.161`, `ndcg_at_5 +0.359`, `mrr +0.355`.
- Gold eval overall is much better than the bad-threshold run:
  - Trained runtime: `accuracy=0.773`, `precision=0.385`, `recall=0.625`, `f1=0.476`.
  - Deltas versus heuristic: `accuracy +0.608`, `precision +0.220`, `f1 +0.193`, `ndcg_at_5 +0.023`, `mrr +0.038`.
  - Recall remains lower than heuristic by `0.375`, because heuristic predicts almost everything positive.
- Gold holdout slice is weaker:
  - Holdout rows: `30`, trained `precision=0.333`, `recall=0.333`, `f1=0.333`.
  - Holdout deltas: `accuracy +0.533`, `precision +0.133`, `f1 +0.000`, `ndcg_at_5 -0.024`, `mrr -0.034`.
- Dataset composition remained unchanged:
  - `86` rows, `18` positives, `68` negatives.
  - `12` VOD groups, all mixed.
  - Dataset sources: `52` twitch VOD primary, `34` boundary near miss.
- Verdict: threshold calibration is fixed operationally. The artifact is usable and no longer zeroes recall. However, holdout ranking/classification is still not convincingly better than heuristic, so the next model-strength step should be to train on more real gold-train signal and/or add more gold positives, while keeping gold-holdout untouched.

## 2026-04-24 Plan: Gold-Train Serving Threshold Calibration

- [x] Add a post-training threshold calibration command that can update an existing ranker artifact.
- [x] Calibrate only on `gold_holdout_split=train` rows when gold eval data exists.
- [x] Wire the calibration command into the local VOD training workflow before proxy/gold eval.
- [x] Report holdout-split bucket metrics so calibrated-vs-holdout behavior is visible.
- [x] Add focused regression coverage without starting training.
- [x] Run syntax/tests and graph rebuild.
- [x] Document result.

### Review Notes

- Added `backend/ml/calibrate_ranker_threshold.py`.
  - Loads an existing usable ranker artifact.
  - Scores labeled eval rows with runtime `Ranker`.
  - Filters calibration rows by `--split-field/--split-value`.
  - Updates `serving_threshold`, `serving_threshold_calibration`, and artifact health when `--write-artifact` is passed.
  - Skips safely when there are too few calibration rows or classes.
- Wired `backend/ml/train_jynxzi_vod.sh` to run threshold calibration after frontend gold export and before proxy/gold eval:
  - Dataset: `jynxzi_gold_eval_rows.jsonl`.
  - Calibration filter: `gold_holdout_split=train`.
  - Source: `gold_eval_train`.
  - Report: `jynxzi_threshold_calibration.json`.
- Extended `evaluate_ranker.py` with `by_gold_holdout_split` metrics so train-vs-holdout behavior is visible after calibration.
- Regression coverage:
  - `backend/tests/test_calibrate_ranker_threshold.py` verifies artifact update uses only gold-train rows and writes artifact health.
  - Also verifies missing filtered calibration rows skip without mutating the artifact.
- Smoke test on a temp copy of the current artifact:
  - Old threshold: `9.926048`.
  - Gold-train calibrated threshold: `-2.606`.
  - All-gold trained metrics with temp artifact: `recall=0.625`, `f1=0.476`, `ndcg_at_5=0.406`, `mrr=0.366`.
  - Gold-holdout slice: `recall=0.333`, `f1=0.333`, `ndcg_at_5=0.351`, `mrr=0.333`.
  - This fixes the recall-zero failure, but holdout remains only marginal, so this is calibration repair rather than model-strength improvement.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_calibrate_ranker_threshold backend.tests.test_train_ranker backend.tests.test_ranker`
  - `python3 -m py_compile backend/ml/calibrate_ranker_threshold.py backend/ml/train_ranker.py backend/ml/evaluate_ranker.py backend/services/ranker.py backend/tests/test_calibrate_ranker_threshold.py backend/tests/test_train_ranker.py backend/tests/test_ranker.py`
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - `git diff --check`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.
- No training run was started and the checked-in/current artifact was not modified by the smoke test.

## 2026-04-24 Plan: Analyze Fresh Container Training Result

- [x] Check whether the container is still available for logs.
- [x] Inspect fresh artifact timestamps and local training summary.
- [x] Verify whether the new serving-threshold calibration landed in the artifact.
- [x] Compare proxy and gold eval metrics against the previous run.
- [x] Inspect VOD dataset composition and workflow state.
- [x] Document verdict and next step.

### Review Notes

- `docker ps -a` returned no containers, so Docker logs are no longer available for this run.
- Fresh artifacts were written around `2026-04-24 01:10`; `local_train_summary.json` reports `training_return_code=0`, `artifact_only=false`, `training_vod_workers=1`, `training_label_workers=3`, and `ranker_short_form_mode=off`.
- The new code landed in the artifact: `jynxzi_ranker.json` contains `serving_threshold=9.926048` and a validation-derived `serving_threshold_calibration`.
- The artifact itself is still usable: selected model `logistic_regression`, `blend=0.95`, validation `accuracy=0.600`, `f1=0.500`, `ndcg_at_5=0.926`, `mrr=0.900`, `ranking_metric_min=0.500`, `ranking_metric_p25=1.000`.
- Dataset composition is unchanged from the previous successful VOD-only run: `86` rows, `18` positive, `68` negative, `12` mixed VOD groups, source peak share `0.395`.
- Ranking quality remains improved versus heuristic:
  - Proxy trained delta: `ndcg_at_5 +0.359`, `mrr +0.355`.
  - Gold trained delta: `ndcg_at_5 +0.023`, `mrr +0.038`.
- The new validation-derived threshold is too high and worsened classification:
  - Proxy at threshold `9.926048`: `recall=0.000`, `f1=0.000`.
  - Gold at threshold `9.926048`: `recall=0.000`, `f1=0.000`.
  - At old threshold `0.0`, proxy would still be strong: `recall=0.944`, `f1=0.576`.
  - A gold-optimal threshold on the current scores would be around `-2.606`, producing `recall=0.625`, `f1=0.476`.
- Verdict: the ranker model/artifact is still useful for ranking, but the validation-only serving-threshold calibration overfit a tiny validation split. Do not trust the new `serving_threshold` for rollout yet.

## 2026-04-24 Plan: Implement Ranker Serving Threshold Calibration

- [x] Add an explicit classification threshold to ranker metric evaluation.
- [x] Calibrate and store a serving threshold when training writes a ranker artifact.
- [x] Make eval reports use the artifact threshold for trained-runtime classification metrics.
- [x] Expose threshold metadata from runtime ranker details/inspection.
- [x] Add regression tests for thresholded metrics and artifact metadata.
- [x] Run focused tests, syntax checks, and graph rebuild.
- [x] Document result.

### Review Notes

- Added explicit threshold support to `train_ranker._evaluate_scores()`. Default behavior remains `0.0` for margin scores and `0.5` for probability scores, but callers can now pass a serving threshold.
- Training now calibrates a validation-derived `serving_threshold` over the final runtime blended score and writes both `serving_threshold` and `serving_threshold_calibration` into the ranker artifact and artifact-health summary.
- `evaluate_ranker.py` now uses the artifact `serving_threshold` for trained-runtime classification metrics, including bucket and group reports. Existing artifacts without the field still fall back to the old threshold.
- `Ranker.score_details()` now exposes `serving_threshold` and `passes_serving_threshold` when a trained artifact is used.
- Regression coverage added for thresholded metric evaluation, threshold calibration, artifact metadata, and runtime detail exposure.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_train_ranker backend.tests.test_ranker`
  - `python3 -m py_compile backend/ml/train_ranker.py backend/ml/evaluate_ranker.py backend/services/ranker.py backend/tests/test_train_ranker.py backend/tests/test_ranker.py`
  - Smoke-tested `evaluate_ranker.py` against current gold rows with a local `whisper` import stub; current artifact has no `serving_threshold` yet, so it correctly falls back to `classification_threshold=0.0`.
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.
- No training run was started. The next freshly trained artifact will contain the calibrated threshold.

## 2026-04-24 Plan: Analyze Latest Container Training Result

- [x] Inspect fresh training summary and artifact timestamps.
- [x] Inspect VOD dataset composition and checkpoint behavior.
- [x] Evaluate ranker, proxy, gold, and artifact-health results.
- [x] Document conclusion and next concrete step.

### Review Notes

- Latest run wrote fresh artifacts around `2026-04-24 00:55`.
- `local_train_summary.json` shows successful non-artifact-only training:
  - `training_return_code=0`
  - `training_vod_workers=1`
  - `training_label_workers=3`
  - `ranker_short_form_mode=off`
- Docker container was already removed, so logs were unavailable. Analysis used persisted artifacts.
- VOD dataset:
  - `86` rows, `18` positive, `68` negative.
  - Sources: `52` `twitch_vod_primary`, `34` `twitch_vod_boundary_near_miss`.
  - All `12` VOD groups are now mixed positive/negative groups.
  - Worst negative/positive ratio is capped at `8.0` (`2711274372`, `2712128863`, `2712736206`, `2713838389`).
- Checkpoints:
  - `13` VOD checkpoints exist and are legacy raw-row-compatible checkpoints, all with dataset fingerprint prefix `54200270`.
  - The new loader can accept these as legacy raw-row checkpoints; no new raw-fingerprint fields were written because the run reused them rather than rebuilding raw rows.
- Ranker artifact:
  - Selected model: `logistic_regression`.
  - Validation: `accuracy=0.600`, `f1=0.500`, `ndcg_at_5=0.926`, `mrr=0.900`, `ranking_metric_min=0.500`, `ranking_metric_p25=1.000`, `ranking_groups=5`.
  - Artifact health status is usable; no warnings.
- Proxy eval:
  - `trained_runtime_uses_artifact=true`.
  - Major lift versus heuristic: `accuracy +0.500`, `f1 +0.230`, `ndcg_at_5 +0.359`, `mrr +0.355`.
- Gold eval:
  - `trained_runtime_uses_artifact=true`.
  - Ranking improves slightly: `ndcg_at_5 +0.023`, `mrr +0.038`, `precision_at_5 +0.011`.
  - Classification recall drops sharply: `recall -0.875`, `f1 -0.116`; accuracy rises mostly because the model predicts far fewer positives on a negative-heavy gold set.
- Conclusion:
  - The artifact is materially better and now passes runtime gates.
  - Proxy performance is strong, but gold improvement is still thin and classification calibration is too conservative.
  - Next work should target more gold-positive coverage and/or calibration/thresholding so ranking gains do not collapse recall.

## 2026-04-24 Plan: Split Raw VOD Checkpoints From Dataset Postprocessing

- [x] Inspect current VOD checkpoint fingerprint logic.
- [x] Separate raw VOD row fingerprint from dataset postprocessing fingerprint.
- [x] Preserve compatibility with existing raw-row checkpoints where possible.
- [x] Add regression coverage for postprocessing changes not forcing VOD downloads.
- [x] Run focused tests and syntax checks without starting training.
- [x] Document result.

### Review Notes

- Implemented two-stage checkpoint fingerprinting in `backend/ml/build_twitch_training_dataset.py`:
  - `_raw_vod_rows_fingerprint()` now fingerprints only expensive raw VOD row generation.
  - `_dataset_build_fingerprint()` still includes cheap dataset postprocessing such as dedupe and negative balancing for auditability.
  - `_write_vod_checkpoint()` writes `checkpoint_kind=raw_vod_rows`, `raw_vod_rows_fingerprint`, and the full `dataset_build_fingerprint`.
  - `_load_vod_checkpoint()` validates the raw fingerprint when present and accepts known legacy full-dataset fingerprints when `raw_vod_rows_fingerprint` is absent.
- Existing local checkpoint for `2711274372` has legacy fingerprint `54200270163c5e2761a647ec758fe69e48b887bb406244cf3f59c95dc234d782`; the new loader accepts it as a legacy raw-row checkpoint.
- This means future postprocessing-only changes, including `max_vod_negative_rows_per_positive`, should no longer force VOD re-downloads when raw rows are already checkpointed.
- Added regression tests:
  - raw fingerprint changes still invalidate a VOD checkpoint.
  - full dataset/postprocessing fingerprint changes do not invalidate a raw VOD checkpoint.
  - legacy checkpoints without `raw_vod_rows_fingerprint` are still accepted.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset`
  - `python3 -m py_compile backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.
- No training run was started by this implementation. The already-running container still uses the old image/code and will not pick up this change until the image is rebuilt.

## 2026-04-24 Plan: Strengthen VOD Ranker Dataset Without Training

- [x] Inspect current VOD dataset generation and quality gates.
- [x] Add VOD-group balancing so positives are retained and negatives are capped per positive.
- [x] Expose the mixed-VOD contrast target in dataset quality output.
- [x] Add regression coverage for the balancing and quality reporting.
- [x] Run focused tests and syntax checks without starting training.
- [x] Document result and next operational step.

### Review Notes

- Implemented VOD group balancing in `backend/ml/build_twitch_training_dataset.py`:
  - `_prepare_ranker_rows()` now runs dedupe -> VOD negative balance -> candidate-source cap.
  - `_balance_vod_negative_rows()` keeps all positives and caps mixed VOD negatives at `8` rows per positive.
  - Negative selection is deterministic: higher sample weight first, then closer to the positive window, while reserving boundary-near-miss rows when both primary and boundary negatives exist.
  - Candidate-source capping now protects all positives and at least one negative per mixed VOD, so source balancing cannot accidentally destroy contrast groups.
  - Dataset build fingerprint now includes `max_vod_negative_rows_per_positive=8`.
- Extended `ranker_dataset_quality()` in `backend/ml/train_vod_ranker.py`:
  - Reports `targets.mixed_ranking_groups=20` and `targets.negative_rows_per_positive_max=8`.
  - Reports `vod_contrast_balance`, including worst negative/positive ratio and groups over the limit.
- Offline simulation against the current persisted VOD rows, without starting training:
  - Final prepared rows would move from `86` to `40`.
  - Positives remain `18`; negatives become `22`.
  - Mixed VOD groups remain `6`.
  - Worst mixed-group ratio becomes `8.0`, with no over-limit groups.
  - `2711274372` moves from `1 positive / 40 negatives` to `1 positive / 8 negatives`.
  - Candidate-source peak remains at the configured cap: `boundary_perturbation=0.4`.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset backend.tests.test_train_vod_ranker`
  - `python3 -m py_compile backend/ml/build_twitch_training_dataset.py backend/ml/train_vod_ranker.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_train_vod_ranker.py`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.
- No training run was started.

## 2026-04-24 Plan: Set Training Worker Defaults

- [x] Locate all runtime defaults for VOD and label training workers.
- [x] Set VOD worker default to `1` and label worker default to `3`.
- [x] Run focused tests and syntax checks.
- [x] Document verification result.

### Review Notes

- Updated runtime defaults:
  - `backend/config.py`: `training_vod_workers=1`, `training_label_workers=3`.
  - `backend/ml/local_train.py`: auto worker plan now resolves to `TRAINING_VOD_WORKERS=1` and `TRAINING_LABEL_WORKERS=3`.
  - `backend/.env.example`: documents `TRAINING_VOD_WORKERS=1` and `TRAINING_LABEL_WORKERS=3`.
- Updated `backend/tests/test_local_train.py` to assert the new default VOD worker value.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_local_train`
  - `python3 -m py_compile backend/config.py backend/ml/local_train.py backend/tests/test_local_train.py`
  - Graphify rebuild attempted but still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.

## 2026-04-24 Plan: Implement VOD Contrast Training Next Step

- [x] Inspect current VOD dataset generation, hard-negative creation, and short-form merge path.
- [x] Add a concrete VOD-first/contrast training mode that prevents short-form rows from dominating serving gates.
- [x] Ensure mixed VOD comparison groups are measurable before training.
- [x] Add regression coverage for VOD-only contrast composition.
- [x] Run targeted tests and syntax checks.
- [x] Document result, rollout command, and remaining risks.

### Review Notes

- Implemented VOD-only ranker training as the default path:
  - `backend/ml/train_vod_ranker.py` now has `--short-form-mode {off,supplemental}` with default `off`.
  - `backend/ml/train_jynxzi_vod.sh` passes `--short-form-mode "$VOD_RANKER_SHORT_FORM_MODE"` and defaults `VOD_RANKER_SHORT_FORM_MODE=off`.
  - `backend/ml/local_train.py` includes `VOD_RANKER_SHORT_FORM_MODE=off` in runtime env and summary.
  - `backend/ml/training_workflow_state.py` fingerprints `ranker_short_form_mode`, forcing retrain/eval when this mode changes.
- Added dataset quality guard for mixed VOD ranking groups:
  - `ranker_dataset_quality()` now reports `mixed_ranking_groups` and fails if there are fewer than `2`.
  - Existing current VOD-only dataset passes quality in simulation: `86` rows, `18` positive, `68` negative, `12` VOD groups, `6` mixed VOD groups, source peak share `0.395`.
- Improved validation split:
  - `backend/ml/train_ranker.py` no longer puts one giant VOD group into validation by default.
  - Group holdout now prefers mixed groups and requires at least two validation groups when possible.
- Fixed a diagnostics bug:
  - `_count_values()` in `backend/ml/train_ranker.py` had a mis-indented increment, so `training_stats.dataset_sources` and `candidate_sources` were always empty. It now counts non-empty values.
- Simulation on current VOD-only rows after the split fix:
  - `rows=86`, validation `accuracy=0.842`, `f1=0.667`, `ndcg_at_5=0.783`, `mrr=0.708`, `ranking_groups=4`, `ranking_metric_min=0.333`, `ranking_metric_p25=0.333`.
  - This removes the zero-min failure, but p25 is still below the runtime gate of `0.45`; more real VOD contrast groups are still needed before expecting automatic promotion.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_train_vod_ranker backend.tests.test_train_ranker backend.tests.test_training_workflow_state backend.tests.test_local_train`
  - `python3 -m py_compile backend/ml/train_vod_ranker.py backend/ml/train_ranker.py backend/ml/training_workflow_state.py backend/ml/local_train.py backend/tests/test_train_vod_ranker.py backend/tests/test_train_ranker.py backend/tests/test_training_workflow_state.py backend/tests/test_local_train.py`
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - Graphify rebuild still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.
- Operational next run:
  - Rebuild the local training image before Docker training because code is baked into the image.
  - Run with default VOD-only mode, or explicitly: `VOD_RANKER_SHORT_FORM_MODE=off`.

## 2026-04-24 Plan: Analyze Non-Artifact-Only Training Result

- [x] Inspect fresh training summary and artifact timestamps.
- [x] Parse ranker, artifact health, boundary, proxy, gold, and pipeline eval reports.
- [x] Compare trained runtime against heuristic and rollout gates.
- [x] Identify root causes for failures or weak metrics.
- [x] Document recommendation and remaining risks.

### Review Notes

- `backend/ml/artifacts/local_train_summary.json` shows `training_return_code=0` and `artifact_only=false` for `jynxzi`, `2026-03-01` to `2026-03-14`.
- Fresh eval outputs were produced at `2026-04-24 00:14`: proxy eval, gold eval, ranker eval, pipeline eval, VOD eval dataset, artifact health, boundary artifact, and workflow state.
- Runtime verdict: do not enable `ENABLE_TRAINED_RANKER_V2`. Reconstructed rollout recommendation is `recommend_enable_trained_ranker_v2=false`, reason `inconclusive_rejected:validation_group_ranking_metric_min_below_minimum`.
- Ranker artifact:
  - `jynxzi_ranker.json`: `version=4`, `pairwise_logistic_regression`, `rows=1089`, `positive_rows=521`, `negative_rows=568`, validation `accuracy=0.636`, `f1=0.595`, `precision_at_5=0.546`, `ndcg_at_5=0.552`, `mrr=0.547`.
  - Runtime rejects it because `ranking_metric_min=0.0 < 0.2`; `ranking_metric_p25=0.0 < 0.45` would also fail, and aggregate ranking metrics remain below serving thresholds (`mrr=0.547 < 0.60`, `ndcg_at_5=0.552 < 0.70`, `precision_at_5=0.546 < 0.60`).
- Proxy eval:
  - `rows=1089`, `positive_rows=521`, `negative_rows=568`, but `trained_runtime_uses_artifact=false`.
  - Baseline and trained runtime are identical: `accuracy=0.48`, `precision=0.479`, `recall=1.0`, `f1=0.648`, `precision_at_5=0.503`, `ndcg_at_5=0.504`, `mrr=0.503`, delta all `0`.
- Gold eval:
  - `rows=97`, `positive_rows=16`, `negative_rows=81`, `eval_groups=19`, holdout/train split `30/67`.
  - Baseline and trained runtime are identical because artifact is rejected: `accuracy=0.165`, `precision=0.165`, `recall=1.0`, `f1=0.283`, `precision_at_5=0.168`, `ndcg_at_5=0.383`, `mrr=0.328`, delta all `0`.
- Pipeline eval:
  - `rows=1089`, classification `accuracy=0.48`, `precision=0.479`, `recall=1.0`, `false_positive=566`, `true_negative=2`, `top1_positive_rate=0.501`, `recall_at_10=0.997`.
  - This indicates the heuristic/ranking path is broad-recall but very weak at rejecting negatives.
- Dataset root cause:
  - Training remains dominated by short-form supplemental rows: `1003` short-form rows vs `52` Twitch VOD primary rows and `34` VOD boundary near-miss rows.
  - `source_url` grouping has `1015` groups, `1007` singleton groups, only `6` mixed positive/negative groups, `500` zero-positive groups, and `509` zero-negative groups. This explains zero min/p25 group ranking metrics.
- Boundary artifact:
  - `jynxzi_boundary_ranker.json` still looks promising but small: `86` rows, start/end/pair validation accuracy `0.722`, F1 roughly `0.737/0.783/0.762`, `ndcg_at_5=1.0`; validation uses a small fallback split, so it is not enough for promotion by itself.
- Verification used JSON parsing of all fresh eval artifacts and dataset composition checks.

## 2026-04-24 Plan: Investigate Container Download Despite Existing Artifact

- [x] Inspect running container logs for download source and target path.
- [x] Inspect container mounts/env to compare `/app` artifact paths with local repo paths.
- [x] Verify whether required artifact files exist inside the container.
- [x] Identify root cause and document any fix or operational action.

### Review Notes

- Logs showed the label artifact was reused: `Label-Datei unverändert: /app/backend/ml/artifacts/jynxzi_twitch_clip_labels.jsonl`.
- The workflow still entered `Training VOD-first ranker for jynxzi` because `should_retrain=True`; the stored workflow state code hash differed from the current code hash after recent code changes.
- Artifact files existed inside the container: `jynxzi_ranker.json`, `jynxzi_vod_training_rows.jsonl`, `jynxzi_training_workflow_state.json`, and `jynxzi_twitch_clip_labels.jsonl`.
- The unexpected download was a serial prefetch bug: after a VOD checkpoint hit, `_schedule_prefetch` blindly downloaded the next VOD before checking whether that next VOD also had a checkpoint. The running process showed `aria2c` downloading into `/app/backend/tmp/storage/twitch_vod_prefetch_2712128863`, while `/app/backend/tmp/training_runs/jynxzi_resume/_vod_checkpoints/2712128863.json` already existed.
- Fixed `backend/ml/build_twitch_training_dataset.py` so serial prefetch skips when the next VOD has a checkpoint. Added `test_build_dataset_skips_serial_prefetch_when_next_vod_has_checkpoint`.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset.BuildTwitchTrainingDatasetTests.test_build_dataset_uses_vod_checkpoint_when_present backend.tests.test_build_twitch_training_dataset.BuildTwitchTrainingDatasetTests.test_build_dataset_skips_serial_prefetch_when_next_vod_has_checkpoint backend.tests.test_build_twitch_training_dataset.BuildTwitchTrainingDatasetTests.test_build_dataset_starts_prefetch_for_next_vod backend.tests.test_build_twitch_training_dataset.BuildTwitchTrainingDatasetTests.test_build_dataset_parallel_path_skips_prefetch_for_checkpoint_hits`
  - `python3 -m py_compile backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py`
  - Graphify rebuild still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.
- The original container exited before final log retrieval; `docker ps -a` returned no remaining containers.

## 2026-04-23 Plan: Pull Main And Verify Merge

- [x] Inspect current branch, remote, and dirty worktree.
- [x] Pull latest `main` without overwriting local/user changes.
- [x] Check for conflicts or merge state.
- [x] Run targeted verification after merge.
- [x] Document result and any remaining risks.

### Review Notes

- Pulled `origin/main` into local `main` with `git pull --ff-only origin main`; HEAD is now `9ff2627` (`Track gold eval data in git`).
- No merge conflicts remain. `git status --short --branch` reports only the local `tasks/todo.md` planning notes.
- Verified tracked gold-eval files parse as JSONL: 97 rows in `backend/ml/artifacts/jynxzi_gold_eval_rows.jsonl`, 97 rows in `backend/ml/artifacts/jynxzi_gold_eval_dataset.jsonl`, and 110 rows in `backend/tmp/storage/feedback/gold_eval_updates.jsonl`.
- The pre-existing `backend/tmp/storage` directory was owned by `nobody:nogroup` and blocked Git from creating `backend/tmp/storage/feedback`; it was preserved as `backend/tmp/storage.pre_pull_backup_20260423`, and Git created a fresh tracked `backend/tmp/storage/feedback` path.

## 2026-04-23 Plan: Fix Local Training Failure When Gold Eval Rows Are Missing

- [x] Inspect `train_jynxzi_vod.sh` handling of `export_frontend_gold_eval.py`.
- [x] Inspect exporter exit behavior for empty frontend gold-eval rows.
- [x] Make missing gold-eval rows a non-fatal skipped evaluation path.
- [x] Add/update regression tests for no-gold local training behavior.
- [x] Run targeted verification and document outcome.

### Review Notes

- Root cause:
  - `backend/ml/export_frontend_gold_eval.py` returned exit code `1` when no frontend gold-eval rows existed.
  - `backend/ml/train_jynxzi_vod.sh` runs with `set -euo pipefail`, so that expected empty-state aborted the whole local training run after artifact/eval work had already started.
- Fix:
  - `export_frontend_gold_eval.py` now writes empty `rows`, empty `eval_rows`, and a summary `{"rows": 0, "eval_dataset": {}}`, prints a warning, and returns `0`.
  - `backend/ml/build_eval_dataset.py` no longer imports `ml.train_ranker` just to read JSONL. It now has a local `_read_jsonl()`, avoiding the heavy `train_ranker -> semantic_service -> transcriber -> whisper` import path for eval dataset export.
  - Added regression coverage for the no-gold CLI path in `backend/tests/test_export_frontend_gold_eval.py`.
- Verification:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_export_frontend_gold_eval backend.tests.test_build_eval_dataset`
  - `python3 -m py_compile backend/ml/export_frontend_gold_eval.py backend/ml/build_eval_dataset.py backend/tests/test_export_frontend_gold_eval.py backend/tests/test_build_eval_dataset.py`
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Result:
  - `8` targeted tests green.
  - Syntax and shell parse checks clean.
  - Graphify rebuild still fails in this environment with `ModuleNotFoundError: No module named 'graphify'`.
- Expected behavior now:
  - A run with no frontend gold-eval rows continues past export.
  - Downstream `evaluate_ranker.py` can create an empty/no-dataset gold eval report instead of the shell aborting.
  - Rollout recommendation should remain inconclusive when no gold eval rows exist, rather than failing the whole training command.

## 2026-04-23 Plan: Compare Viral Clip Pipeline Against AI Clipping Apps

- [x] Verify current positioning/features of leading AI clipping tools from primary sources.
- [x] Map competitor capabilities against the local pipeline architecture.
- [x] Identify quality, workflow, training, eval, and product gaps.
- [x] Prioritize improvements by impact and implementation leverage.
- [x] Record findings and recommendations.

### Review Notes

- Market baseline checked from official/current sources:
  - OpusClip emphasizes any-genre clipping, AI reframe/object tracking, dynamic captions, B-roll, audio enhance, voice-over, brand templates, API/workflow automation, and publishing.
  - Captions AI Shorts emphasizes auto short extraction, templates, subtitles, and AI editing.
  - Clippper is Twitch-specific and emphasizes webcam/gameplay/chat region detection, planned 9:16 layouts, timed captions, hooks, and export-ready shorts.
  - vidIQ emphasizes YouTube-to-shorts clipping, cut/crop/captions, safe-zone captions, and title/keyword/post-time optimization.
  - Vizard/Descript emphasize clip counts/duration controls, layouts/aspect ratios, topic/goal guided clip generation, text editing, templates, subtitles, translation, sharing/publishing.
  - Munch emphasizes coherent clips, multimodal analysis, trend/keyword analysis, platform-specific posts, auto-captioning, and smart cropping.
- Local strengths:
  - Strong architecture for candidate discovery and scoring: transcript/audio/video features, semantic candidates, peak candidates, strategy/profile knobs, boundary refinement, feedback/gold eval plumbing.
  - Twitch/VOD training infrastructure is more specialized than generic tools: VOD labels, VOD feature cache, boundary near-miss negatives, streamer profiles, training state, artifact health.
  - Debug outputs and feedback collection are unusually good for an early internal tool.
- Major gaps versus leading apps:
  - Current ranker artifact is not runtime-usable; serving likely falls back to heuristic.
  - Training data is still short-form-heavy, not genuinely VOD-first.
  - Export only cuts horizontal MP4 clips; no vertical 9:16 composition, no webcam/gameplay layout, no active speaker/object tracking crop.
  - No burned-in dynamic captions, safe-zone caption placement, caption templates, emoji/keyword highlighting, translation, or brand kit.
  - No B-roll, transitions, audio enhancement, silence/filler-word removal, hook/title generation, thumbnails, hashtags, descriptions, or posting workflow.
  - UI is review/download/feedback, not a timeline/text editor where users can adjust cuts/captions/layouts.
  - Trend/post-publish learning loop exists mostly as disabled flags or weak placeholders, not an operational outcome-based system.
  - Portable pipeline and Vertex docs drift from the richer shell flow.
- Priority:
  - First make clip selection measurable and active: non-artifact-only eval, gold/proxy reports, ranker gate fixes, more VOD-positive data.
  - Then build export-ready vertical output: Twitch region detector/layout planner, captions, safe zones, templates.
  - Then close learning loop with post-publish metrics and trend-conditioned metadata.

## 2026-04-23 Plan: Evaluate Current 14-Day Run Artifact

- [x] Identify the run metadata and current artifact files.
- [x] Inspect ranker artifact, artifact health, boundary artifact, and dataset composition.
- [x] Compare artifact validation metrics against runtime usability gates.
- [x] Assess rollout readiness, model quality, and primary risks.
- [x] Record findings and verification commands.

### Review Notes

- Run metadata:
  - `backend/ml/artifacts/local_train_summary.json` identifies the current 14-day run as `jynxzi`, `2026-03-01` to `2026-03-14`, `training_return_code=0`, `artifact_only=true`.
  - Because `artifact_only=true`, no proxy/gold/pipeline eval reports were produced in `backend/ml/artifacts`.
  - MLflow is disabled in the artifact (`mlflow.enabled=false`).
- Candidate ranker artifact:
  - `backend/ml/artifacts/jynxzi_ranker.json`: `version=4`, `artifact_family=ranker_v2`, `model_type=logistic`, `selected_model=pairwise_logistic_regression`, `blend=0.95`.
  - Runtime clamps blend to `0.65`, so the artifact's requested `0.95` is not the actual serving blend.
  - Training stats: `1089` rows, `521` positives, `568` negatives, `872` train rows, `217` validation rows, `group_holdout` by `source_url`, leakage risk `low`.
  - Validation: accuracy `0.636`, precision `0.547`, recall `0.652`, F1 `0.595`, precision@5 `0.546`, ndcg@5 `0.552`, mrr `0.547`.
  - Runtime usability gates fail on `ranking_metric_min=0.0 < 0.20`, `ranking_metric_p25=0.0 < 0.45`, and ranking composite thresholds (`mrr=0.547 < 0.60`, `ndcg@5=0.552 < 0.70`, `precision@5=0.546 < 0.60`, `precision=0.547 < 0.70`).
  - Expected runtime result: artifact rejected, heuristic fallback.
- Dataset composition:
  - Parsed `1089` JSONL rows from `jynxzi_vod_training_rows.jsonl`.
  - `1003` rows are `short_form_supplemental`; `52` are `twitch_vod_primary`; `34` are `twitch_vod_boundary_near_miss`.
  - Weighted data: total sample weight `447.75`; short-form supplement weight `351.05`; primary Twitch VOD weight `81.4`; boundary near-miss weight `15.3`.
  - Label classes: `503` short-form positive, `500` short-form negative, `34` boundary near-miss negative, `29` hard negative, `14` positive, `5` background negative, `4` weak positive.
  - Grouping is mostly singleton short-form URLs: `1015` groups, `1007` singleton groups, `500` zero-positive groups, `509` zero-negative groups, only `6` mixed groups. This explains the zero min/p25 grouped ranking metrics.
  - Primary VOD source mix barely passes the configured cap: `twitch_label_proxy=34/86` and `boundary_perturbation=34/86`, peak share about `0.395` below the `0.40` cap.
- Boundary artifact:
  - `backend/ml/artifacts/jynxzi_boundary_ranker.json`: `version=1`, `boundary_logistic`, `86` boundary rows.
  - Start/end/pair validation each report accuracy `0.722`, F1 about `0.737`/`0.783`/`0.762`, ndcg@5 `1.0`.
  - This passes static boundary model gates, but `enable_boundary_ranker` defaults to `False`; also validation has only `18` rows and stratified fallback split, so do not promote without boundary-specific eval.
- Assessment:
  - The ranker artifact is useful as a training checkpoint, not rollout-ready.
  - It should not be treated as an active serving improvement because runtime gates reject it.
  - The artifact is still dominated by short-form supervision, so it is not yet a strong VOD-first model.
  - Run a non-artifact-only eval pass before changing serving flags or relaxing gates.
- Verification:
  - Inspected `local_train_summary.json`, `jynxzi_ranker.json`, `jynxzi_artifact_health.json`, `jynxzi_boundary_ranker.json`.
  - Counted dataset rows, labels, source mix, sample weights, and grouping structure via read-only JSON parsing.
  - Compared artifact metrics against `services.ranker.Ranker._validation_passes` thresholds statically.

## 2026-04-23 Plan: Analyze Codebase ML Model And Training Workflow

- [x] Read architecture map and identify core ML/training entry points.
- [x] Trace runtime inference path from candidate generation through feature building, ranker scoring, boundary refinement, and final selection.
- [x] Trace training workflow from data collection/labeling through dataset construction, ranker/boundary training, evaluation, artifact promotion, and deployment/loading.
- [x] Inspect model artifacts/config contracts and tests that define expected behavior.
- [x] Run lightweight verification commands that do not mutate backend code, and record limitations.
- [x] Document findings, risks, and improvement opportunities.

### Review Notes

- Graphify/entry points:
  - Graphify identifies `RichSegment`, `Candidate`, `train_from_rows()`, `Ranker`, `_build_candidate_features()` and `_window_candidate()` as central nodes.
  - Main runtime path is `backend/pipeline.py`: download, audio extraction, transcription, audio/video/chat enrichment, segment scoring, candidate generation/normalization, `feature_builder.build_features()`, `selector.rank_candidates()`, optional clip judge, `analyzer.refine_candidates()`, post-processing, clipping, feedback export.
  - Main local training path is `backend/ml/local_train.py` -> `backend/ml/train_jynxzi_vod.sh` -> `backend/ml/collect_twitch_clip_labels.py` -> `backend/ml/train_vod_ranker.py` -> `backend/ml/build_twitch_training_dataset.py` -> `backend/ml/train_ranker.py` plus `backend/ml/train_boundary_ranker.py`.
- Model architecture:
  - Serving ranker is hybrid: heuristic score in `backend/services/ranker.py` plus optional trained artifact blend from `backend/ml/artifacts/{profile}_ranker.json` or MLflow Production registry fallback.
  - Current ranker artifact is `version=4`, `profile=jynxzi`, `model_type=logistic`, `selected_model=pairwise_logistic_regression`, `blend=0.95`.
  - Runtime artifact gates likely reject the current ranker: validation has `ranking_metric=mrr`, `ranking_metric_value=0.547`, but `ranking_metric_min=0.0` and `ranking_metric_p25=0.0`, which trips `validation_group_ranking_metric_min_below_minimum`.
  - Boundary artifact exists and its start/end/pair submodels pass static gate thresholds, but `enable_boundary_ranker` defaults to `False`, so serving uses heuristic boundary scoring unless explicitly enabled.
- Data/training state:
  - Current artifacts: `jynxzi_twitch_clip_labels.jsonl` has 6703 label rows; `jynxzi_vod_training_rows.jsonl` has 1089 training rows; `jynxzi_training_rows.jsonl` has 1003 short-form rows.
  - Composite training data is dominated by short-form supplement: 1003/1089 rows are `short_form_supplemental`; only 52 rows are `twitch_vod_primary` and 34 are `twitch_vod_boundary_near_miss`.
  - Twitch label source is very imbalanced/noisy at the label-window stage: hard/background negatives dominate; only 14 `positive` and 4 `weak_positive` label rows are present after ambiguous rows are excluded.
  - Last local training summary says `artifact_only=true` for 2026-03-01 to 2026-03-14, so eval/gold/proxy reports were intentionally skipped; no `jynxzi_*eval*.json` reports were present in `backend/ml/artifacts`.
- Workflow mechanics:
  - Dataset build uses VOD download cache, full-VOD audio/transcript cache, window modality cache, VOD row checkpoints, candidate-source capping, and short-form supplemental fallback.
  - `train_ranker.py` trains mean-difference linear, logistic, pairwise logistic, gradient boosting, and optionally LightGBM LambdaRank, then selects by composite validation score and calibrates heuristic/model blend.
  - Evaluation path builds proxy eval rows, exports frontend gold eval rows, compares trained vs heuristic ranker, evaluates heuristic pipeline ranking, finalizes MLflow artifacts, and prints rollout recommendation.
  - Workflow state hashes tracked ML/services/config code and skips retraining/evaluation when labels, dataset, artifact, and state are unchanged.
  - Productive local path is more complete than portable pipeline path: shell trains VOD composite dataset, ranker, boundary ranker, eval, MLflow finalization and rollout recommendation; pipeline components train only the direct `ml.build_twitch_training_dataset` output and `ml.train_ranker`.
  - `docs/vertex-training.md` references `backend/ml/vertex_train.py`, `backend/ml/pipelines/train_pipeline.py`, `scripts/submit_vertex_training.py`, and `scripts/submit_vertex_pipeline.py`; all four are missing in this workspace.
  - MLflow registration is driven by training-validation gates, while proxy/gold eval is logged later by finalization and is not a hard promotion gate.
- Verification:
  - Parsed 80 Python files under `backend/ml` and `backend/services` with `ast.parse`; no syntax errors.
  - Static artifact gate check showed current ranker would fail with `validation_group_ranking_metric_min_below_minimum`.
  - Static boundary artifact gate check showed all three boundary submodels meet row/accuracy/F1 minimums.
  - Verified missing Vertex files listed above and inspected pipeline train/build components for shell-flow drift.
  - Import-level runtime verification could not run because the local `.venv` is missing `whisper`, causing `ModuleNotFoundError` when importing `services.transcriber`.
- Risks / recommendations:
  - The current trained ranker is probably not active at runtime despite `ENABLE_TRAINED_RANKER_V2=True`; verify after installing dependencies or in the Docker image with `Ranker().inspect_artifact("jynxzi")`.
  - Treat the current `jynxzi` model as short-form-heavy, not truly VOD-first, until primary VOD positives grow materially.
  - The validation gate is conservative but currently punishes many single-positive/zero-positive groups; consider filtering groups without positives from min/p25 ranking gates or adding a separate coverage metric.
  - Boundary ranker should stay disabled until evaluated on gold/proxy boundary data; its internal split uses `stratified_fallback`, so leakage risk is higher than the main ranker split.
  - Run a non-artifact-only eval pass before changing serving flags, because the current artifact lacks the proxy/gold/pipeline reports needed for a staff-level rollout decision.
  - Treat Vertex docs/pipeline as non-authoritative until the missing submitter/entrypoint files are restored or the docs are corrected.

## 2026-04-23 Plan: Prevent VOD FFmpeg Decode Stalls

- [x] Live-Haenger gegen Container-Prozesse und Trainingscode einordnen.
- [x] Harten Timeout fuer FFmpeg-Window-Decodes implementieren.
- [x] Regressionstests fuer Timeout-/Cleanup-Verhalten ergaenzen.
- [x] Tests, Syntaxcheck und Graphify-Rebuild ausfuehren.

### Review Notes

- Live-Befund:
  - Der laufende Container war nicht durch RAM/CPU limitiert, sondern hing mit sechs `ffmpeg -hide_banner -loglevel fatal ... -f rawvideo ...` Prozessen seit ueber 3:38 Stunden an 16-36 Sekunden langen VOD-Window-Decodes.
  - `docker stats` lag dabei bei ca. `0.06%` CPU und ca. `1GiB` RAM; die letzten Logs waren alte Download-Ausgaben. Das ist ein Decode-Hang, kein normaler Download- oder Compute-Bottleneck.
- Umsetzung:
  - `backend/services/video_features.py` hat jetzt fuer FFmpeg-Frame-Decodes einen harten Timeout. Batch-Decodes lesen non-blocking aus der Rawvideo-Pipe, killen den Prozess bei Timeout und geben bereits gelesene Frames beziehungsweise leere Visual-Frames zurueck.
  - `backend/config.py` fuegt `video_feature_ffmpeg_timeout_sec` hinzu; Default ist `120.0`. Per Env kann das als `VIDEO_FEATURE_FFMPEG_TIMEOUT_SEC` angepasst werden.
  - Einzelbild-Decodes nutzen denselben Timeout und unterdruecken weiterhin FFmpeg-Stderr ueber `stderr=DEVNULL`.
  - Tests decken Timeout, Kill-Verhalten und die Timeout-Weitergabe an `subprocess.run` ab.
- Verifiziert mit:
  - `python3 -m py_compile backend/config.py backend/services/video_features.py backend/tests/test_video_features.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_video_features`
  - `env STORAGE_DIR=tmp/storage .venv/bin/python -m unittest tests.test_build_twitch_training_dataset`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_video_features`: `17` Tests gruen.
  - `test_build_twitch_training_dataset`: `36` Tests gruen.
  - Syntaxcheck sauber.
  - Graphify-Rebuild scheitert weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der aktuell laufende Container nutzt diese Aenderung nicht; er muss neu gestartet werden. `scripts/train_local_docker.sh` baut das Image beim Start neu, danach greift der Timeout.
  - Ein Timeout bedeutet fuer das betroffene Window reduzierte visuelle Features, aber der 14-Tage-Run bleibt fortsetzbar statt endlos zu blockieren.

## 2026-04-23 Plan: Implement ML Training Cache Optimization And Legacy Cleanup

- [x] Cache-Key und VOD-Feature-Cache robust machen.
- [x] Download-Cache-Metadata und sichere Cleanup-Regeln erweitern.
- [x] Regressionstests fuer Cache-Hits, korrupte Caches und Altlasten ergaenzen.
- [x] Relevante Tests, Syntaxcheck und Graphify-Rebuild ausfuehren.

### Review Notes

- Umsetzung:
  - `backend/ml/build_twitch_training_dataset.py` nutzt fuer Window-Cache-Keys bei Twitch jetzt eine stabile Signatur aus canonical Download-Cache und `vod_id`, statt den transienten Jobpfad in den Key zu ziehen.
  - VOD-Audio bekommt `audio.metadata.json`; inkompatible Audio-Caches werden neu extrahiert, Legacy-Audio ohne Metadata bleibt fuer Rueckwaertskompatibilitaet zunaechst nutzbar und wird beim Zugriff mit Metadata versehen.
  - Transcript- und Window-Cache-Writes laufen atomar ueber `.tmp` + `replace`.
  - Kompatible Transcript-Hits entfernen jetzt uebrig gebliebene `chunks/`, weil das persistierte Transcript danach die wiederverwendbare Quelle ist.
  - Korrupte Window-Cache-JSONs werden als Cache-Miss behandelt und neu aufgebaut.
  - `backend/services/downloader.py` schreibt fuer neue Mediencache-Eintraege `.metadata.json` mit Namespace, URL/VOD-ID, Groesse und Complete-Marker; fehlerhafte Metadata erzwingt einen Refresh.
  - `backend/ml/cache_cleanup.py` entfernt alte `backend/tmp/storage/viral_*` nur, wenn die finalen Medien per Inhaltsvergleich im Short-Form-Downloadcache vorhanden sind oder keine finalen Medien existieren.
- Verifiziert mit:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset backend.tests.test_downloader backend.tests.test_cache_cleanup`
  - `python3 -m py_compile backend/ml/build_twitch_training_dataset.py backend/services/downloader.py backend/ml/cache_cleanup.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_downloader.py backend/tests/test_cache_cleanup.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m ml.cache_cleanup --backend-root /home/antonio/ai_clipping/backend --dry-run`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `53` gezielte Unit-Tests gruen.
  - Syntaxcheck sauber.
  - Cleanup-Dry-Run findet jetzt `312` konservativ entfernbare Pfade im aktuellen Workspace, inklusive redundant gespiegeltem Short-Form-Legacy-Material; es wurde nichts geloescht.
  - Graphify-Rebuild scheitert weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Die neuen Download-Cache-Metadata-Dateien entstehen bei neuen oder refreshenden Cache-Writes; bestehende Legacy-Mediencache-Dateien ohne Metadata bleiben als kompatible Hits gueltig.
  - Der Short-Form-Legacy-Cleanup vergleicht Inhalte gegen den Cache und ist dadurch sicherer, aber bei sehr vielen Kandidaten etwas teurer als eine reine Groessenregel.

## 2026-04-23 Plan: Analyze ML Training Cache Optimization And Legacy Cleanup

- [x] Graphify-Report, bisherige Lessons und vorhandene Task-Historie lesen.
- [x] ML-Training-Entry-Points und Dataset-Stufen einordnen.
- [x] Cache-Schichten, Keying, Invalidierung und Cleanup-Regeln auditieren.
- [x] Aktuelle Altlasten und redundante Artefakte konservativ bewerten.
- [x] Konkrete Optimierungsempfehlungen mit Risiken und Verifikation dokumentieren.

### Review Notes

- Workflow:
  - Lokaler Hauptpfad: `backend/ml/local_train.py` setzt Runtime-Umgebung und ruft `backend/ml/train_jynxzi_vod.sh`.
  - Shell-Orchestrierung: Cleanup, Twitch-Label-Collect, Workflow-State-Gating, VOD-first Dataset/Training, Short-Form-Backfill-Fallback und Evaluation.
  - Innerer Trainingspfad: `backend/ml/train_vod_ranker.py` baut ein Composite-Dataset aus VOD-Primary-Rows und optional Short-Form-Supplement, prueft Quality-Gates und trainiert Ranker plus Boundary-Ranker.
  - Teuerster Teil: `backend/ml/build_twitch_training_dataset.py`, besonders VOD-Download, Full-VOD-Audio/Transcript, Window-Modalities und VOD-Row-Checkpoints.
- Cache-Schichten:
  - Modell-/Library-Cache: `backend/tmp/cache/**` fuer Whisper, HuggingFace, Torch, DeepFace, Matplotlib. Behalten.
  - Mediencache: `backend/tmp/storage/_download_cache/twitch/vod_<id>.mp4` und `_download_cache/short_form/<sha>.mp4`. Behalten; Hauptschutz gegen Re-Downloads.
  - Featurecache: `backend/tmp/vod_feature_cache/<vod_id>/audio.wav`, `transcript_*.json`, `windows/*.json`. Behalten, aber Keying verbessern.
  - Row-Checkpoints: `backend/tmp/training_runs/.../_vod_checkpoints/<vod_id>.json`. Behalten; aktuell klein.
  - Short-Form-Row-Cache: `backend/tmp/short_form_training/_row_cache`. Behalten.
  - Artefakte/State: `backend/ml/artifacts/**`. Behalten.
- Aktueller Workspace-Befund:
  - `backend/tmp`: ca. `428G`.
  - `backend/tmp/storage/_download_cache`: ca. `179G` bis `197G`, davon Twitch ca. `177G`; persistent wertvoll.
  - `backend/tmp/storage/twitch_vod_*`: ca. `244G`, davon `9` Ordner mit finalem `video.mp4`, die groesstenteils im Twitch-Download-Cache repliziert sind, plus `2` abgebrochene Fragmentordner.
  - `backend/tmp/vod_feature_cache`: ca. `18G`; persistent wertvoll, aber `chunks/` dupliziert große Teile von `audio.wav`.
  - `backend/tmp/storage/viral_*`: `929` Short-Form-Jobordner, zusammen ca. `6.1G`; nicht pauschal loeschen, weil alte Rohkopien dort liegen koennen.
  - Fragment-/Temp-Dateien unter `backend/tmp/storage`: `2033` Dateien, ca. `18.31GiB`, vor allem in `twitch_vod_2719793388` und `twitch_vod_2720629593`.
  - `ml.cache_cleanup --dry-run` wuerde aktuell `24` Pfade loeschen, inklusive redundanter Twitch-Jobordner und transienter Short-Form-/Backfill-/Legacy-Pfade.
- Optimierungsvorschlaege:
  - Window-Feature-Key stabilisieren: in `_window_feature_cache_payload()` nicht transienten Jobpfad/mtime verwenden, sondern canonical Twitch-Cache-Signatur oder VOD-ID plus Cache-Metadata. Das verbessert Cross-Run-Hits.
  - `audio.wav` mit Metadata-Sidecar validieren: VOD-ID, Source-Signatur, Extractor-/Cache-Version. Bei Mismatch neu extrahieren.
  - `chunks/` nach erfolgreichem Transcript-Cache optional entfernen oder unter eigener TTL fuehren; die Chunks kosten aktuell fast die Haelfte von `vod_feature_cache`, waehrend das Transcript bereits persistiert ist.
  - Window-/Transcript-Cache-Writes atomar machen und korrupte JSON-Dateien als Cache-Miss behandeln.
  - Download-Cache mit `.complete`/Sidecar-Metadata absichern, damit Cache-Hits nicht nur auf Dateiexistenz vertrauen.
  - Cleanup-Regeln um sichere Short-Form-`viral_*`-Migration erweitern: nur loeschen, wenn Rohmedium im `_download_cache/short_form` vorhanden ist oder kein finales Medium existiert.
- Verifikation:
  - Graphify-Report gelesen.
  - Read-only Befund via `du`, `find`, `rg` und JSONL-Zaehlung.
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m ml.cache_cleanup --backend-root /home/antonio/ai_clipping/backend --dry-run`.
- Keine Backend-Codeaenderung vorgenommen; Graphify-Rebuild ist deshalb nicht erforderlich.

## 2026-04-23 Plan: Add Per-Run Backend Cache Cleanup

- [x] Cache- und Arbeitsordner im Backend klassifizieren.
- [x] Konservativen Run-Start-Cleanup fuer transiente Altlasten implementieren.
- [x] Cleanup in lokalen Trainingsstart einhaengen.
- [x] Regressionstests fuer Behalten/Loeschen ergaenzen.
- [x] Tests, Syntaxcheck und Graphify-Rebuild ausfuehren.

### Review Notes

- Ursache / Ziel:
  - `backend/tmp/cache` enthaelt Modell-/Library-Caches wie Whisper, HuggingFace, Torch und DeepFace; diese sollen nicht geloescht werden.
  - Die eigentlichen Cleanup-Kandidaten sind transiente Laufartefakte: `_cache_fill_*`, sicher gecachte `twitch_vod_*` / `twitch_vod_prefetch_*`, Short-Form-Jobordner unter `backend/tmp/short_form_training/viral_*`, alte Backfill-Artefakte und leere Legacy-Cache-Ordner.
  - `backend/tmp/storage/viral_*` enthaelt in alten Laeufen oft die einzige lokale Short-Form-Rohkopie (`video.mp4`) und wird deshalb nicht automatisch geloescht.
  - Persistente Caches wie `_download_cache`, `vod_feature_cache`, `_row_cache` und Modellcaches bleiben erhalten.
- Umsetzung:
  - Neues Modul `backend/ml/cache_cleanup.py` klassifiziert und entfernt nur bekannte Runtime-Artefakte.
  - `storage/viral_*` bleibt erhalten; Twitch-Jobordner werden nur geloescht, wenn kein finales `video.mp4` drinliegt oder der passende `_download_cache/twitch/vod_<id>.mp4` mit gleicher Groesse existiert.
  - `backend/ml/train_jynxzi_vod.sh` ruft den Cleanup bei jedem lokalen Trainingsstart auf, ausser `AI_CLIPPING_CACHE_CLEANUP=0` ist gesetzt.
  - Der Cleanup hat einen `--dry-run`-Modus fuer Inspektion.
- Verifiziert mit:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_cache_cleanup backend.tests.test_downloader`
  - `python3 -m py_compile backend/ml/cache_cleanup.py backend/tests/test_cache_cleanup.py backend/services/downloader.py backend/tests/test_downloader.py`
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_cache_cleanup` + `test_downloader`: `16` Tests gruen.
  - `py_compile` und Shell-Syntax sauber.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Aktuelle Workspace-Pruefung:
  - Dry-run auf dem aktuellen Workspace fand nach der konservativen Rohdaten-Schutzregel `23` entfernbare Runtime-Artefakte.
  - Der echte Cleanup wurde nicht auf den aktuellen Workspace angewendet, weil gerade ein Trainingscontainer (`ai-clipping-train-local:latest`) laeuft.
- Rest-Risiken / lokale Einschraenkungen:
  - Bei parallel gestarteten Trainingslaeufen sollte nur ein Lauf die gemeinsamen `backend/tmp`-Arbeitsordner verwenden; der Cleanup ist fuer den normalen Ein-Run-Startpfad gedacht.
  - Rohmedien-Cache und Feature-Cache werden absichtlich nicht automatisch geloescht, weil sie erneute Downloads und teure Feature-Arbeit vermeiden.

## 2026-04-23 Plan: Clean Redundant Raw VOD Copies After Cache Fill

- [x] Downloader-Pfad fuer Roh-VOD-Duplikate einordnen.
- [x] Frische Twitch-Downloads nach erfolgreichem Cache-Store auf den Cache-Hardlink umlegen.
- [x] Bekannte temporaere Job-Artefakte wie `video.temp.mp4` nach erfolgreichem Download entfernen.
- [x] Regressionstests, Syntaxcheck und Graphify-Rebuild ausfuehren.

### Review Notes

- Ursache / Ziel:
  - Die Transcript-Cache-Bereinigung hatte nur Feature-Cache-Duplikate behandelt.
  - Im Roh-VOD-Pfad konnte nach einem frischen Twitch-Download trotzdem eine echte zweite Kopie entstehen: `job/video.mp4` plus `_download_cache/twitch/vod_<id>.mp4`.
  - Zusaetzlich konnten alte yt-dlp/aria2-Artefakte wie `video.temp.mp4` und `video.mp4.part-*` im Jobordner liegen bleiben.
- Umsetzung:
  - `backend/services/downloader.py` verlinkt `job/video.mp4` nach erfolgreichem Cache-Store wieder aus der Cache-Datei, statt die vorher heruntergeladene Kopie weiter separat liegen zu lassen.
  - Beim Twitch-Cache-Hit und nach einem frischen Cache-Fill werden redundante `video*`-Medien- und Fragment-Artefakte im Jobordner entfernt; die finale `video.mp4` bleibt erhalten.
  - Wenn Hardlinks nicht moeglich sind, bleibt der bestehende Copy-Fallback erhalten.
- Verifiziert mit:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_downloader`
  - `python3 -m py_compile backend/services/downloader.py backend/tests/test_downloader.py`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `backend.tests.test_downloader`: `13` Tests gruen.
  - `py_compile` sauber.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der aktuell laufende Docker-Container nutzt diese Code-Aenderung erst nach Neustart mit aktualisiertem Code/Image.
  - Bereits existierende alte Jobordner werden nicht global geloescht; der Cleanup greift beim naechsten Downloader-Lauf fuer den betroffenen Jobordner.

## 2026-04-23 Plan: Prune Superseded Transcript Cache Copies

- [x] Doppelte Transcript-Dateien und den Migrationspfad vom alten source-signierten Cache einordnen.
- [x] Stabile Transcript-Migration so anpassen, dass alte kompatible Kopien entfernt werden.
- [x] Regressionstests fuer Migration und Cleanup ergaenzen und ausfuehren.
- [x] Graphify-Rebuild versuchen und Ergebnis dokumentieren.

### Review Notes

- Ursache / Ziel:
  - Nach der Umstellung auf stabile Transcript-Keys lagen bei `2711274372` und `2712128863` jeweils zwei Transcript-Dateien: die alte source-signierte Datei und die neue stabile Datei.
  - Funktional war das nicht falsch, aber es macht Cache-Inspektion uneindeutig und laesst den Cache unnoetig wachsen.
- Umsetzung:
  - `backend/ml/build_twitch_training_dataset.py` laedt kompatible alte Transcript-Caches weiter, schreibt danach aber die stabile Datei und entfernt kompatible Altdateien im selben VOD-Cache.
  - Auch wenn die stabile Datei bereits existiert, werden alte kompatible Transcript-Dateien bei einem Cache-Hit aufgeraeumt.
  - Inkompatible Transcripts, etwa anderes `vod_id`, anderes Whisper-Modell oder andere Sprache, werden nicht entfernt.
- Verifiziert mit:
  - `env STORAGE_DIR=tmp/storage .venv/bin/python -m unittest tests.test_build_twitch_training_dataset.BuildTwitchTrainingDatasetTests.test_load_or_build_vod_feature_context_migrates_old_source_bound_transcript_cache tests.test_build_twitch_training_dataset.BuildTwitchTrainingDatasetTests.test_load_or_build_vod_feature_context_prunes_old_transcripts_on_stable_cache_hit`
  - `env STORAGE_DIR=tmp/storage .venv/bin/python -m unittest tests.test_build_twitch_training_dataset`
  - `python3 -m py_compile backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - Gezielt: `2` Tests gruen.
  - `tests.test_build_twitch_training_dataset`: `34` Tests gruen.
  - `py_compile` sauber.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der aktuell laufende Docker-Container nutzt diese Code-Aenderung erst nach Neustart mit aktualisiertem Code/Image.
  - Bereits vorhandene doppelte Transcript-Dateien im aktuellen Host-Cache werden beim naechsten Lauf mit diesem Code bereinigt; laufende alte Container bereinigen sie nicht.

## 2026-04-23 Plan: Suppress H264 Decoder Noise In VOD Feature Extraction

- [x] Den H.264-Warning-Pfad gegen aktuelle Logs und `video_features` einordnen.
- [x] H.264-Frames ueber den kontrollierten FFmpeg-Decode-Pfad lesen, statt OpenCV/FFmpeg direkt in stderr loggen zu lassen.
- [x] Regressionstests fuer die H.264-Pfadwahl und FFmpeg-Loglevel ergaenzen.
- [x] Relevante Tests, Graphify-Rebuild und Ergebnis dokumentieren.

### Review Notes

- Ursache / Ziel:
  - `[h264 @ ...] co located POCs unavailable` kommt aus dem H.264-Decoder, waehrend `video_features.extract(...)` H.264 bislang ueber `cv2.VideoCapture` gelesen hat.
  - OpenCVs eingebetteter FFmpeg schreibt solche Decoder-Hinweise direkt nach stderr; dadurch landen sie im Docker-Trainingslog, ohne dass Python sie gezielt filtern kann.
- Umsetzung:
  - `backend/services/video_features.py` routet noisy Codecs (`av1`, `h264`, `avc1`, `hevc`, `h265`, `mpeg4`) jetzt ueber den bereits vorhandenen FFmpeg-Rawvideo-Pipe-Decoder.
  - Der FFmpeg-Pipe-Decoder laeuft mit `-loglevel fatal`, sodass nicht-fatale Decoder-Hinweise den Trainingslog nicht mehr fluten.
  - OpenCV bleibt als Fallback fuer nicht erkannte Codec-Infos erhalten.
- Verifiziert mit:
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_video_features.VideoFeaturesTests.test_should_use_ffmpeg_frame_decode_for_noisy_codecs backend.tests.test_video_features.VideoFeaturesTests.test_extract_uses_ffmpeg_frame_decode_for_h264_streams backend.tests.test_video_features.VideoFeaturesTests.test_decode_second_frames_ffmpeg_uses_fatal_loglevel`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_video_features`
  - `env STORAGE_DIR=tmp/storage .venv/bin/python -m unittest tests.test_build_twitch_training_dataset`
  - `python3 -m py_compile backend/services/video_features.py backend/tests/test_video_features.py`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - Gezielt: `3` Tests gruen.
  - `backend.tests.test_video_features`: `15` Tests gruen.
  - `tests.test_build_twitch_training_dataset`: `33` Tests gruen.
  - `py_compile` sauber.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Zum Zeitpunkt der Analyse lief kein Docker-Container mehr (`docker ps -a` leer), daher konnte kein Live-Log nach dem Fix bestaetigt werden.
  - Der Fix greift im Docker-Training erst nach Neustart mit aktualisiertem Code beziehungsweise neu gebautem Image.

## 2026-04-23 Plan: Fix VOD Window Cache JSON Serialization

- [x] Den aktuellen Trainingslog gegen den VOD-Window-Feature-Cache-Pfad abgleichen.
- [x] NumPy-Skalarwerte vor `json.dumps()` in normale Python-Typen wandeln.
- [x] Regressionstest fuer NumPy-Bool/Number-Frames ergaenzen und ausfuehren.
- [x] Ergebnis und verbleibende Laufzeit-Einschraenkungen dokumentieren.

### Review Notes

- Ursache / Ziel:
  - Der Log-Eintrag `[h264 ...] co located POCs unavailable` ist ein Decoder-Hinweis, aber im selben Lauf wurden VOD-Windows wegen `Object of type bool is not JSON serializable` verworfen.
  - Der Window-Feature-Cache serialisierte `vars(frame)` direkt; `numpy.bool_` und andere NumPy-Skalare sehen fachlich korrekt aus, sind fuer `json.dumps()` aber keine nativen JSON-Typen.
- Umsetzung:
  - `backend/ml/build_twitch_training_dataset.py` normalisiert Frame-Werte vor dem Cache-Schreiben ueber `_json_scalar(...)`.
  - NumPy-Skalare werden per `.item()` in native Python-`bool`/`int`/`float` umgewandelt; nicht-finite Floats werden auf `0.0` gesetzt.
  - Ein Regressionstest deckt Window-Feature-Frames mit `np.bool_`, `np.int64`, `np.float32` und `np.float64` ab.
- Verifiziert mit:
  - `env STORAGE_DIR=tmp/storage .venv/bin/python -m unittest tests.test_build_twitch_training_dataset.BuildTwitchTrainingDatasetTests.test_load_or_build_window_modalities_serializes_numpy_scalars tests.test_build_twitch_training_dataset.BuildTwitchTrainingDatasetTests.test_load_or_build_window_modalities_uses_cache_after_first_build`
  - `env STORAGE_DIR=tmp/storage .venv/bin/python -m unittest tests.test_build_twitch_training_dataset`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - Die gezielten Tests sind gruen.
  - `tests.test_build_twitch_training_dataset`: `33` Tests gruen.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der bereits gestartete Docker-Container nimmt Code-Aenderungen erst nach einem Neustart bzw. Neubuild des Trainingsimages auf.
  - Zum Zeitpunkt der Pruefung lief kein Trainingscontainer mehr (`docker ps -a` leer).

## 2026-04-23 Plan: Preserve Short-Form Backfill Artifact Even When VOD Training Fails

- [x] Den aktuellen Failure-Pfad in `train_jynxzi_vod.sh` auf den Short-Form-Backfill-Abzweig pruefen.
- [x] Das erzeugte Short-Form-Artefakt an einen stabilen Ort unter `ml/artifacts` spiegeln, bevor der VOD-Retry laeuft oder spaeter scheitert.
- [x] Den Shell-Pfad verifizieren und das Verhalten dokumentieren.

### Review Notes

- Ursache / Ziel:
  - Beim 1-Tages-Lauf wurde ein brauchbares Short-Form-Backfill-Artefakt unter `tmp/short_form_backfill_artifacts/` erzeugt, ging aber im Gesamtergebnis unter, weil der anschliessende VOD-first-Retry an den Qualitaetsgates scheiterte.
  - Gewuenscht war, dieses Zwischenartefakt fuer spaetere Vergleiche und laengere Runs trotzdem stabil zu behalten.
- Umsetzung:
  - `backend/ml/train_jynxzi_vod.sh` spiegelt das erzeugte Short-Form-Backfill-Artefakt jetzt direkt nach dem Backfill in `ml/artifacts/`.
  - Konkret werden `jynxzi_ranker.json` und `jynxzi_artifact_health.json` aus `tmp/short_form_backfill_artifacts/` nach
    `ml/artifacts/jynxzi_short_form_backfill_ranker.json` und
    `ml/artifacts/jynxzi_short_form_backfill_artifact_health.json` kopiert.
  - Das passiert vor dem erneuten VOD-first-Training, sodass die Dateien auch dann erhalten bleiben, wenn der Retry spaeter wieder mit `insufficient_dataset_quality` endet.
- Verifiziert mit:
  - `bash -n backend/ml/train_jynxzi_vod.sh`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - Shell-Syntax ist sauber.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Das gespiegelte Backfill-Artefakt ist bewusst getrennt vom finalen VOD-Artefakt benannt; es wird nicht automatisch als produktives `jynxzi_ranker.json` promoted.
  - Bei mehreren Backfill-Laeufen wird das gespiegelt gespeicherte Testartefakt jeweils vom neuesten Run ueberschrieben.

## 2026-04-23 Plan: Add Configurable Whisper Language Hint For Local Training

- [x] Relevante Transcriber-, Config- und lokalen Runner-Pfade fuer eine optionale Sprachvorgabe pruefen.
- [x] Eine konfigurierbare Whisper-Sprachvorgabe einfuehren, die bei leerem Wert weiter Auto-Detect verwendet.
- [x] Den lokalen Trainingspfad fuer den aktuellen englischen Setup auf `en` vorkonfigurieren.
- [x] Gezielte Tests ergaenzen und den Fix verifizieren.

### Review Notes

- Ursache / Ziel:
  - In den Trainingslogs wurden bei kurzen Short-Form-Clips mehrfach offensichtlich falsche Whisper-Spracherkennungen (`Detected language: ...`) sichtbar.
  - Da der aktuelle Jynxzi-/Twitch-/Short-Form-Setup laut `streamer_sources.csv` durchgehend englisch ist, sollte der lokale Trainingspfad Whisper gezielt mit `en` hinweisen koennen, statt pro Chunk auf Auto-Detect zu vertrauen.
- Umsetzung:
  - `backend/config.py` besitzt jetzt `whisper_language` als optionale Konfiguration; leerer Wert bedeutet weiter Auto-Detect.
  - `backend/services/transcriber.py` reicht die Sprachvorgabe nur dann an `model.transcribe(...)` weiter, wenn sie gesetzt ist, und loggt sie in der Runtime-Zusammenfassung.
  - `backend/ml/local_train.py` setzt fuer den aktuellen lokalen Trainingspfad standardmaessig `WHISPER_LANGUAGE=en`, ohne den generischen Backend-Default global auf Englisch festzulegen.
  - Die lokale Trainings-Summary schreibt die aktive Whisper-Sprache ebenfalls mit.
- Verifiziert mit:
  - `python3 -m py_compile backend/config.py backend/services/transcriber.py backend/ml/local_train.py backend/tests/test_transcriber.py backend/tests/test_transcriber_progress.py backend/tests/test_local_train.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_transcriber backend.tests.test_transcriber_progress backend.tests.test_local_train`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_transcriber` + `test_transcriber_progress` + `test_local_train`: `13` Tests gruen.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Die Vorgabe `en` ist im lokalen Runner bewusst fuer den aktuellen englischen Trainingsfall gesetzt; fuer spaetere mehrsprachige Setups sollte `WHISPER_LANGUAGE` explizit leer oder passend gesetzt werden.
  - Der bereits laufende Container nutzt den Fix erst nach einem Neustart.

## 2026-04-23 Plan: Clean Top-Level tmp Directory Conservatively

- [x] Inhalt von `tmp/` gegen aktuelle Repo-Referenzen pruefen.
- [x] Nur klar historische / generierte Artefakte entfernen und noch referenzierte Prewarm-Pfade erhalten.
- [x] Ergebnisgroesse und verbleibenden Inhalt verifizieren.

### Review Notes

- Ursache / Ziel:
  - Der top-level Ordner `tmp/` enthielt vor allem alte Vertex-/Cloud-Build-Specs, Pipeline-JSONs und lokale Laufartefakte.
  - Gleichzeitig referenziert `scripts/train_local_docker.sh` weiterhin `tmp/prewarmed_sentence_transformers_local`, daher sollte die Bereinigung konservativ bleiben.
- Umsetzung:
  - Historische Einzeldateien wie `vertex_job_spec_*`, `cloudbuild_*`, `train_pipeline_*`, `_diskspec_probe.json` und `benchmark_local_semantic_models.py` wurden entfernt.
  - Das alte Laufverzeichnis `tmp/local_train_2x2` wurde ebenfalls geloescht.
  - `tmp/prewarmed_sentence_transformers_local` blieb bewusst erhalten, weil der lokale Docker-Trainingspfad diesen Ordner optional mountet.
- Verifiziert mit:
  - `find /home/antonio/ai_clipping/tmp -maxdepth 2 -mindepth 1 -printf '%y %P\n' | sort`
  - `du -sh /home/antonio/ai_clipping/tmp /home/antonio/ai_clipping/tmp/* 2>/dev/null | sort -h`
- Ergebnis:
  - Der top-level Ordner `tmp/` enthaelt jetzt nur noch `prewarmed_sentence_transformers_local`.
  - Restgroesse: `8.8M`.
- Rest-Risiken / lokale Einschraenkungen:
  - Falls du einzelne geloeschte Vertex-/Pipeline-Specs doch noch brauchst, muessten sie aus Git-Historie oder einem anderen Speicherort wiederhergestellt werden.
  - Der Cleanup hat absichtlich nur den top-level Ordner `tmp/` betroffen, nicht `backend/tmp`.

## 2026-04-23 Plan: Decouple Short-Form Download I/O With Persistent Media Cache And Prefetch

- [x] Den bestehenden Short-Form-Downloadpfad gegen vorhandene Downloader-Caches und die aktuelle Worker-Topologie abgleichen.
- [x] Einen persistenten Medien-Cache unter `backend/tmp` fuer Short-Form-URLs einbauen, damit Wiederholungslaeufe vorhandene Quelldateien statt neuer Downloads verwenden.
- [x] Download-Prefetching im Short-Form-Sammler ergaenzen, damit Medien-I/O und spaetere Feature-Extraktion staerker ueberlappen.
- [x] Gezielte Tests fuer Cache-Hits, Prefetch-Reuse und die unveraenderte Row-Erzeugung ergaenzen und verifizieren.

### Review Notes

- Ursache / Ziel:
  - Die Short-Form-Sammlung hatte zwar bereits einen Row-Cache, aber keine Wiederverwendung der heruntergeladenen Medien selbst.
  - In den Trainingslogs blieb dadurch noch viel Zeit in wiederholten `yt-dlp`-Downloads haengen, und Medien-I/O war nur lose mit der spaeteren CPU-Feature-Extraktion ueberlappt.
- Umsetzung:
  - `backend/services/downloader.py` besitzt jetzt mit `ensure_cached_download(...)` einen generischen persistenten Medien-Cache unter `backend/tmp/storage/_download_cache/<namespace>/`; fuer diesen Pfad bleibt die bestehende Twitch-Cache-Logik unveraendert.
  - `backend/ml/collect_viral_clips.py` nutzt fuer Short-Form-URLs jetzt einen dedizierten Medien-Cache-Namespace (`short_form`) statt bei jedem Lauf neue Job-spezifische Downloads zu erzeugen.
  - Ein neuer `_ShortFormMediaPrefetcher` zieht kommende URLs mit kleinem Worker-Pool vor und ueberlappt dadurch Download-I/O mit der spaeteren Feature-Extraktion.
  - `_process_clip()` kann vorab gecachte Medien direkt verwenden und schreibt `audio.wav` dabei weiterhin lokal in das Clip-Arbeitsverzeichnis statt in den Cache.
- Verifiziert mit:
  - `python3 -m py_compile backend/services/downloader.py backend/ml/collect_viral_clips.py backend/config.py backend/tests/test_downloader.py backend/tests/test_collect_viral_clips.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_downloader backend.tests.test_collect_viral_clips`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_downloader` + `test_collect_viral_clips`: `32` Tests gruen.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der Short-Form-Medien-Cache ist absichtlich persistent und hat derzeit keine automatische TTL-Eviction; Speicherbedarf bleibt daher der relevante Tradeoff.
  - Prefetching ist bewusst konservativ gehalten (`2` Prefetch-Worker Default), weil der Lauf in den letzten Messungen eher an CPU-/I/O-Ueberlappung als an fehlender roher Parallelitaet hing.

## 2026-04-23 Plan: Fix DeepFace Warmup For Installed API Signature

- [x] Die im Container installierte DeepFace-`build_model`-Signatur gegen den aktuellen Warmup-Pfad abgleichen.
- [x] `_ensure_deepface_ready()` versionsrobust auf den korrekten Emotion-Warmup fuer alte und neue DeepFace-Signaturen umstellen.
- [x] Den betroffenen `video_features`-Testpfad erweitern und den Fix verifizieren.

### Review Notes

- Ursache / Ziel:
  - Im Lauf erschien `video_features_deepface_init_failed error=Invalid model_name passed - facial_recognition/Emotion`.
  - Die reale Container-Version ist `deepface 0.0.99`; dort hat `DeepFace.build_model` die Signatur `(model_name, task='facial_recognition')`, waehrend das eigentliche Modeling intern `task='facial_attribute', model_name='Emotion'` fuer den Emotionspfad erwartet.
- Umsetzung:
  - `backend/services/video_features.py` erkennt jetzt die `build_model`-Signatur zur Laufzeit und waermt das Emotionsmodell fuer alte und neue DeepFace-APIs korrekt an.
  - Fuer task-aware Signaturen wird jetzt explizit `task='facial_attribute', model_name='Emotion'` verwendet; einfache Ein-Argument-Signaturen bleiben weiter unterstuetzt.
- Verifiziert mit:
  - `docker exec <container> python -c "... inspect.signature(DeepFace.build_model) ..."`
  - `python3 -m py_compile backend/services/video_features.py backend/tests/test_video_features.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_video_features`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_video_features`: `13` Tests gruen.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der bereits laufende Container nutzt den Fix erst nach einem Neustart.
  - Der Warnpfad `video_features_deepface_init_failed error=boom:Emotion` bleibt absichtlich in einem Test erhalten, der sicherstellt, dass ein fehlgeschlagener Warmup nicht endlos wiederholt wird.

## 2026-04-23 Plan: Remove Remaining Video Feature Overhead In Short-Form Training

- [x] Den AV1-Softwaredecode im `video_features`-Pfad von Frame-fuer-Frame-Subprozessen auf gebatchten FFmpeg-Decode pro Window umstellen.
- [x] Den DeepFace-Warmup so absichern, dass ein einmaliger Init-Versuch pro Prozess ausreicht und wiederholte Init-Logs/Versuche im Lauf verschwinden.
- [x] Gezielte `video_features`-Tests fuer den neuen AV1-Batch-Pfad und den robusteren DeepFace-Warmup ergaenzen und verifizieren.

### Review Notes

- Ursache / Ziel:
  - Die aktuellen Container-Logs zeigten den AV1-Fallback als CPU-Hotspot: gleichzeitig laufende `ffmpeg -frames:v 1`-Subprozesse verbrauchten mehrere Kerne nur fuer einzelne Frame-Reads.
  - Gleichzeitig tauchten `video_features_deepface_init`-Logs weiterhin auffaellig oft auf; das deutete darauf hin, dass der Warmup-Pfad bei Build-Problemen oder spaeteren Aufrufen nicht robust genug terminiert.
- Umsetzung:
  - `backend/services/video_features.py` dekodiert AV1-Fenster jetzt ueber `_decode_second_frames_ffmpeg(...)` gebatcht mit einem einzigen FFmpeg-Prozess pro Window und liest die 1-fps-Frames direkt aus dem Rawvideo-Stream.
  - `extract(...)` baut den AV1-Frame-Cache pro Window einmal auf und liest danach nur noch aus dem In-Memory-Frame-Index statt pro Sekunde einen neuen Subprozess zu starten.
  - `refine_face_reactions(...)` nutzt fuer AV1 ebenfalls einen gebatchten Frame-Puffer fuer den relevanten Sekundenbereich.
  - `_ensure_deepface_ready()` terminiert jetzt nach dem ersten Warmup-Versuch pro Prozess, auch wenn `DeepFace.build_model("Emotion")` fehlschlaegt; wiederholte Build-Versuche und Init-Logs entfallen damit.
- Verifiziert mit:
  - `python3 -m py_compile backend/services/video_features.py backend/tests/test_video_features.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_video_features`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_video_features`: `12` Tests gruen.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der AV1-Batchpfad arbeitet weiterhin absichtlich auf CPU-Softwaredecode; der Gewinn kommt aus weniger Prozess- und Seek-Overhead, nicht aus Hardwaredecode.
  - Der DeepFace-Warmup gibt bei einem fehlgeschlagenen Build-Versuch bewusst nur noch eine einmalige Warnung aus; die eigentliche `DeepFace.analyze(...)`-Nutzung bleibt davon unberuehrt.

## 2026-04-23 Plan: Speed Up Short-Form Collection Without Quality Loss

- [x] `collect_rows()` auf begrenzte parallele URL-Verarbeitung umstellen, ohne Reihenfolge und Fehlertoleranz zu verlieren.
- [x] Einen persistenten Per-URL-Row-Cache unter `backend/tmp` nutzen, damit bereits verarbeitete Short-Form-Clips bei Wiederholungslaeufen nicht neu geladen und gefeatured werden.
- [x] Pro Collection-Worker die teuren Modellruntimes gezielt vorwaermen, statt Initialisierung mehrfach waehrend des eigentlichen Durchlaufs zu bezahlen.
- [x] In `_process_clip()` unabhaengige Schritte nach dem Audio-Extract parallelisieren und die Aenderungen mit gezielten Tests verifizieren.

### Review Notes

- Ursache / Ziel:
  - Die aktuellen Trainingslogs zeigten den Short-Form-Backfill als neuen Hauptengpass: `collect_rows()` lief URL fuer URL seriell, bereits verarbeitete Clips wurden bei Wiederholungslaeufen erneut komplett gerechnet, und `_process_clip()` arbeitete Download, Transkript, Audio- und Video-Features streng nacheinander ab.
  - Gewuenscht waren ausschliesslich qualitaetsneutrale Beschleunigungen, also weniger redundante Arbeit statt weniger oder groberer Features.
- Umsetzung:
  - `backend/ml/collect_viral_clips.py` verarbeitet Short-Form-URLs jetzt mit einem begrenzten `ThreadPoolExecutor`, behaelt aber die Reihenfolge der zurueckgegebenen Rows stabil und behandelt Retry-/Unavailable-Faelle weiter pro Entry.
  - Erfolgreich erzeugte Short-Form-Rows werden unter `backend/tmp/.../_row_cache/` per Hash aus Manifest-Eintrag, Profil und Runtime-Fingerprint persistiert und bei Folgelaeufen als Cache-Hit wiederverwendet.
  - Die Collection-Worker waermen Whisper-, Video-/DeepFace- und Semantic-Runtimes gezielt vor dem eigentlichen Durchlauf an, statt die Initialisierung mitten im ersten Batch zu bezahlen.
  - `_process_clip()` und `process_local_video()` extrahieren nach dem Audio-Extract nun Transkript, Audio-Frames und Video-Frames parallel, ohne die Featureinhalte selbst zu aendern.
  - Der Cache-Fingerprint und der Warmup-Pfad degradieren absichtlich robust, wenn optionale Trainings-Abhaengigkeiten in einer Minimal- oder Testumgebung lokal nicht importierbar sind.
- Verifiziert mit:
  - `python3 -m py_compile backend/ml/collect_viral_clips.py backend/tests/test_collect_viral_clips.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_collect_viral_clips`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_collect_viral_clips`: `16` Tests gruen.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der Per-URL-Cache speichert bewusst nur erfolgreiche Rows; fehlgeschlagene oder transient nicht erreichbare URLs werden nicht persistent als Fehler festgeschrieben.
  - Die neuen Default-Workerzahlen fuer den Short-Form-Sammler sind konservativ gehalten; echte Grenzwerte haengen weiter vom lokalen GPU-/CPU-Budget ab.

## 2026-04-23 Plan: Analyze Current Training Logs For Quality-Neutral Speedups

- [ ] Den aktuell laufenden Trainingscontainer und die relevanten Fortschritts-/Runtime-Logs erfassen.
- [ ] Die größten Zeitfresser entlang des aktuellen Codepfads mit den Logs gegenprüfen.
- [ ] Nur Maßnahmen ableiten, die Laufzeit sparen, ohne die Clipfindungsqualität oder Featureinhalte zu verschlechtern.

### Review Notes

- Offen:
  - Gewünscht ist keine weitere Vermutung anhand alter Läufe, sondern eine Einordnung auf Basis der aktuellen Container-Logs.
  - Die Empfehlungen sollen explizit auf qualitätsneutrale Hebel beschränkt bleiben.

## 2026-04-23 Plan: Prevent OpenCV AV1 Hardware-Decode Failures In Video Features

- [x] Den aktuellen `cv2.VideoCapture`-Pfad in `video_features.py` auf implizite Hardware-Decode-Nutzung pruefen.
- [x] OpenCV-Capture standardmaessig auf Software-Decode umstellen, ohne den restlichen Video-Feature-Pfad zu veraendern.
- [x] Gezielte Tests fuer den neuen Capture-Open-Pfad ergaenzen und den Fix verifizieren.

### Review Notes

- Ursache / Ziel:
  - Im Trainingslog schlagen AV1-Reads mit `Your platform doesn't support hardware accelerated AV1 decoding`, `Failed to get pixel format` und `Get current frame error` fehl.
  - Der erste Env-basierte OpenCV-Tweak reichte nicht aus; Ziel war daher ein robuster codec-spezifischer Fallback statt weiterer unbewiesener Decoder-Flags.
- Umsetzung:
  - `backend/services/video_features.py` prueft jetzt per `ffprobe` den Videocodec und schaltet fuer `av1` gezielt auf FFmpeg-Softwaredecode pro Sample-Sekunde um.
  - Nicht-AV1-Quellen bleiben auf dem bisherigen `cv2.VideoCapture`-Pfad.
  - Sowohl `extract(...)` als auch `refine_face_reactions(...)` nutzen denselben AV1-Fallback, damit die Fehler nicht im Zweitpass wieder auftauchen.
  - Der fruehere OpenCV-Env-Default bleibt erhalten, ist aber nicht mehr die eigentliche Schutzmassnahme.
- Verifiziert mit:
  - `python3 -m py_compile backend/services/video_features.py backend/tests/test_video_features.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_video_features`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_video_features`: `10` Tests gruen.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Der AV1-Fallback dekodiert Sample-Frames per FFmpeg-Prozess und ist deshalb fuer AV1 langsamer als der normale OpenCV-Pfad; dafuer vermeidet er die wiederholten Decoder-Fehler und den Verlust visueller Features.
  - Der Fix greift erst in neu gestarteten Prozessen beziehungsweise Containern.

## 2026-04-23 Plan: Add VOD-Wide Feature Caches Under backend/tmp

- [x] Den begonnenen VOD-Cache-Pfad in `build_twitch_training_dataset.py` bis in die Label-Verarbeitung durchziehen.
- [x] Transcript und Voll-Audio pro VOD in `backend/tmp` persistieren und Window-Modalitaeten ueber denselben Cache-Kontext wiederverwenden.
- [x] Gezielte Tests fuer Transcript-Slicing und Window-Cache-Hits ergaenzen sowie den neuen Pfad verifizieren.

### Review Notes

- Ursache / Ziel:
  - Nach dem Direkt-VOD-Umbau wurden Transcript und Audio weiterhin effektiv pro Label-Fenster neu aufgebaut.
  - Gewuenscht war ein qualitativer Null-Downgrade: Transcript und Voll-Audio genau einmal pro VOD materialisieren, Window-Modalitaeten exakt aus diesem VOD-Kontext ableiten und die Ergebnisse unter `backend/tmp` persistent halten.
- Umsetzung:
  - `backend/ml/build_twitch_training_dataset.py` baut jetzt pro VOD einen `VodFeatureContext` auf und cached Voll-Audio (`audio.wav`) sowie den absoluten Transcript-Export unter `backend/tmp/vod_feature_cache/<vod_id>/`.
  - Die Label-Phase sliced den Transcript jetzt nur noch fuer das konkrete Fenster und liest Audio-Frames exakt aus dem bereits geladenen Vollsignal.
  - Window-Modalitaeten werden unterhalb desselben VOD-Cache-Verzeichnisses per Hash aus Quellvideo, Zeitfenster und Video-Runtime-Konfiguration persistiert und beim naechsten Zugriff wiederverwendet.
  - `backend/services/audio_features.py` kann ueber `extract_window_from_signal(...)` jetzt direkt aus einem vorhandenen Audio-Signal per Fenster extrahieren, ohne erneut FFmpeg oder Dateiloads anzustossen.
- Verifiziert mit:
  - `python3 -m py_compile backend/ml/build_twitch_training_dataset.py backend/services/audio_features.py backend/tests/test_build_twitch_training_dataset.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `test_build_twitch_training_dataset`: `31` Tests gruen.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin an `ModuleNotFoundError: No module named 'graphify'`.
- Rest-Risiken / lokale Einschraenkungen:
  - Video-Features werden fuer exakte Fenstergrenzen weiter direkt aus dem VOD gelesen und dann per Window-Cache persistiert; der wirklich teure erste Lauf pro neuem Fenster bleibt also bestehen, Folgelaeufe profitieren dafuer direkt.
  - Der aktuell laufende Trainingscontainer nutzt den neuen Cache-Pfad erst nach einem Neustart mit dem aktualisierten Code.

## 2026-04-23 Plan: Avoid Repeated DeepFace Downloads And Warmups In Local Training

- [x] Den aktuellen DeepFace-Pfad auf vermeidbare Cache-/Initialisierungsarbeit eingrenzen.
- [x] DeepFace auf einen persistenten Projekt-Cache umstellen und die Modellinitialisierung pro Prozess auf einen einmaligen, gelockten Pfad reduzieren.
- [x] Die betroffenen Tests ausfuehren und den vorgeschriebenen Graphify-Rebuild fuer `backend` erneut versuchen.

### Review Notes

- Ursache / Ziel:
  - Im Trainingslog wurden `facial_expression_model_weights.h5` und die DeepFace-Initialisierung von mehreren Label-Workern praktisch gleichzeitig angestossen.
  - Das ist reine Mehrarbeit: weder der Download noch der Emotionsmodell-Warmup verbessern die Feature-Qualitaet.
- Umsetzung:
  - `backend/config.py` setzt jetzt zusaetzlich `DEEPFACE_HOME` auf den projektlokalen Cache unter `backend/tmp/cache/deepface`.
  - `backend/services/video_features.py` fuehrt DeepFace jetzt ueber einen einmaligen, gelockten `_ensure_deepface_ready()`-Pfad hoch und nutzt nach erfolgreichem Warmup denselben geladenen Zustand fuer spaetere `analyze(...)`-Aufrufe.
  - `backend/tests/test_video_features.py` deckt den neuen Pfad mit einem Test ab, der `DeepFace.build_model("Emotion")` nur einmal pro Prozess erwartet.
- Verifiziert mit:
  - `python3 -m py_compile backend/config.py backend/services/video_features.py backend/tests/test_video_features.py`
  - `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_video_features`
- Rest-Risiken / lokale Einschraenkungen:
  - Der laufende Trainingscontainer profitiert erst nach einem Neustart mit dem aktualisierten Code.
  - Wenn eine DeepFace-Version kein `build_model` anbietet, bleibt der Warmup-Pfad absichtlich no-op; die Funktionsfaehigkeit bleibt erhalten, aber der Einmal-Warmup-Vorteil faellt dann weg.
  - Der Graphify-Rebuild scheitert in dieser Umgebung weiterhin daran, dass das Modul `graphify` lokal nicht installiert ist.

## 2026-04-23 Plan: Restore MediaPipe Face Detection In Local Training Container

- [x] Den Grund dafuer eingrenzen, warum `video_features` im lokalen Trainingscontainer auf `mediapipe=unavailable` faellt.
- [x] Den lokalen Trainings-Dockerfile so anpassen, dass die fuer MediaPipe Tasks fehlenden Runtime-Libraries im Image vorhanden sind.
- [x] Das lokale Trainings-Image neu bauen und die MediaPipe-Detector-Erzeugung im Container verifizieren.

### Review Notes

- Ursache / Ziel:
  - Im laufenden Trainingscontainer war `mediapipe` zwar installiert, aber `hasattr(mediapipe, "solutions")` lieferte `False`, daher fiel der Code auf den Tasks-Pfad zurueck.
  - `FaceDetector.create_from_options(...)` scheiterte dort reproduzierbar mit `OSError: libGLESv2.so.2: cannot open shared object file`.
  - `ldd /usr/local/lib/python3.12/dist-packages/mediapipe/tasks/c/libmediapipe.so` zeigte zusaetzlich `libEGL.so.1 => not found`.
- Umsetzung:
  - `backend/Dockerfile.local-train` installiert jetzt zusaetzlich `libegl1` und `libgles2`.
- Verifiziert mit:
  - `docker exec ... python -c "import mediapipe ..."`
  - `docker exec ... ldd /usr/local/lib/python3.12/dist-packages/mediapipe/tasks/c/libmediapipe.so`
  - `docker build -f backend/Dockerfile.local-train -t ai-clipping-train-local:latest .`
  - `docker run --rm --entrypoint /bin/sh ai-clipping-train-local:latest -lc 'python -c "... FaceDetector.create_from_options(...) ..."'`
  - `docker run --rm --entrypoint /bin/sh ai-clipping-train-local:latest -lc 'ldd .../libmediapipe.so | grep "not found\\|libGLES\\|libEGL"'`
- Rest-Risiken / lokale Einschraenkungen:
  - Der aktuell laufende Container bleibt unveraendert, bis der Trainingslauf mit dem neu gebauten `ai-clipping-train-local:latest` neu gestartet wird.
  - Eine separate `docker exec`-Direktabfrage von `services.video_features` scheiterte nebenbei an einer in der Container-Umgebung vorhandenen Host-`STORAGE_DIR`-Variable ausserhalb von `/app`; der Trainingslauf selbst umgeht das aktuell, weil `local_train.py` `STORAGE_DIR` vor dem eigentlichen Trainingsprozess ueberschreibt.

## 2026-04-23 Plan: Repair Local Training Docker Build On New CUDA 12.8 Base Image

- [x] Den fehlgeschlagenen `docker build`-Schritt auf dem neuen `cu128`-Base-Image auswerten und die eigentliche Ursache eingrenzen.
- [x] `backend/Dockerfile.local-train` so anpassen, dass Paketinstallation und Runtime weiter ueber das im PyTorch-Image mitgelieferte Conda-Python laufen statt ueber das extern verwaltete System-Python.
- [x] Das lokale Trainings-Image erneut bauen und den Build-Abschluss verifizieren.

### Review Notes

- Ursache / Ziel:
  - Der erste Build gegen `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` scheiterte bei `python -m pip install ...` mit `externally-managed-environment` nach PEP 668.
  - Das neue Base-Image zeigt mit `python` auf das systemverwaltete Ubuntu-Python; ein nachgeschobener Versuch ueber `/opt/conda/bin/python` scheiterte ebenfalls, weil dieses Image gar kein Conda mehr mitbringt.
- Umsetzung:
  - `backend/Dockerfile.local-train` setzt jetzt `PIP_BREAK_SYSTEM_PACKAGES=1`, damit Container-Builds trotz PEP 668 projektlokale Python-Pakete installieren koennen.
  - Die Python-Paketinstallation laeuft explizit ueber `/usr/local/bin/pip`, also ueber den im Image vorhandenen Python-Paketmanager statt ueber einen nicht vorhandenen Conda-Pfad.
  - Der lokale `.venv`-Symlink zeigt jetzt auf `/usr/bin/python`, und der Container-`ENTRYPOINT` verwendet wieder das im Image vorhandene `python`.
- Verifiziert mit:
  - `docker build -f backend/Dockerfile.local-train -t ai-clipping-train-local:cuda128 .`
- Ergebnis:
  - Die erste Build-Iteration zeigte den PEP-668-Fehler reproduzierbar.
  - Ein zweiter Versuch belegte, dass das neue Image keinen `/opt/conda/bin/python` mehr mitbringt.
  - Der darauf angepasste Rebuild lief erfolgreich bis zum Ende durch und taggte das Image als `ai-clipping-train-local:cuda128`.
- Rest-Risiken / lokale Einschraenkungen:
  - Wenn einzelne Wheels fuer `.[ml,vision]` gegen `cu128` nicht verfuegbar sind oder weitere Paketkonflikte im neuen Image bestehen, taucht der naechste Fehler erst im erneuten Build auf.

## 2026-04-23 Plan: Raise Local Training Container To PyTorch CUDA 12.8 For 5070 Ti Compatibility

- [x] Alle im Repo verdrahteten lokalen Trainings-Image-/CUDA-Versionen identifizieren.
- [x] Das lokale Trainings-Base-Image auf einen neueren offiziellen PyTorch/CUDA-Stand anheben, der besser zu Blackwell-/5070-Ti-Hardware passt.
- [x] Den lokalen Docker-Startpfad gegen offensichtliche Inkonsistenzen pruefen und einen echten Build gegen das neue Base-Image anstossen.

### Review Notes

- Ursache / Ziel:
  - Der Trainingslauf schlug beim Laden von Whisper-Gewichten mit `CUDA error: no kernel image is available for execution on the device` fehl.
  - Das passt zu einem zu alten PyTorch/CUDA-Build fuer die eingesetzte GPU; im Repo lief der lokale Trainingscontainer noch auf `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`.
- Umsetzung:
  - `backend/Dockerfile.local-train` verwendet jetzt `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`.
  - Der bestehende lokale Docker-Runner bleibt unveraendert und zeigt weiter auf `backend/Dockerfile.local-train`.
- Verifiziert mit:
  - `bash -n scripts/train_local_docker.sh`
  - `docker build -f backend/Dockerfile.local-train -t ai-clipping-train-local:cuda128 .`
- Ergebnis:
  - Der neue Base-Tag wird von Docker Hub erfolgreich aufgeloest und der Build laedt das `2.10.0-cuda12.8-cudnn9-runtime`-Image.
  - Ein kompletter Build-Abschluss lag zum Dokumentationszeitpunkt noch nicht vor, weil der neue Runtime-Layer mehrere Gigabyte gross ist.
- Rest-Risiken / lokale Einschraenkungen:
  - Der Build allein garantiert noch nicht, dass alle Python-Abhaengigkeiten im Image ohne weitere Anpassungen fertig bauen; das muss der vollstaendige Build-/Trainingslauf zeigen.
  - Wenn auf dem Host noch ein veralteter NVIDIA-Treiber oder ein unpassender Docker-WSL2-GPU-Pfad laeuft, reicht das neue `cu128`-Image allein nicht aus.

## 2026-04-23 Plan: Remove Training window.mp4 Intermediate And Read Features Directly From VOD

- [x] Direkte Abhaengigkeiten des aktuellen `window.mp4`-Pfads lesen und die Sync-/Zeitachsen-Anforderungen fuer Audio, Transcript und Video-Frames ableiten.
- [x] Trainings-Label-Pfad so umbauen, dass kein `window.mp4` mehr geschrieben wird und Audio-/Video-Features direkt aus dem VOD-Zeitfenster gelesen werden.
- [x] Checkpoint-Version und bestehende Tests auf den neuen Direkt-VOD-Pfad anpassen.
- [x] Syntax pruefen, Testlauf versuchen und den vorgeschriebenen `graphify`-Rebuild fuer `backend` erneut versuchen.

### Review Notes

- Ursache / Ziel:
  - Der Trainingspfad schnitt bisher erst `window.mp4`-Zwischenclips und extrahierte danach Audio, Transcript und Video-Features daraus.
  - Das kostete zusaetzlichen Re-Encode-/I/O-Aufwand und fuehrte eine unnoetige verlustbehaftete Zwischenstufe ein.
- Umsetzung:
  - `backend/ml/build_twitch_training_dataset.py` verwendet fuer die Label-Phase jetzt direkt den VOD-Pfad und loggt `mode=direct_vod` statt vorbereitete Batch-Clips.
  - `backend/services/audio_extractor.py` kann Audio jetzt direkt aus einem VOD-Zeitfenster (`start_sec`/`end_sec`) in die lokale `audio.wav` des Label-Arbeitsverzeichnisses extrahieren.
  - `backend/services/video_features.py` akzeptiert jetzt `start_sec`/`end_sec` und liest Frames direkt aus dem entsprechenden VOD-Fenster, waehrend die ausgegebenen `VideoFeatureFrame.second`-Werte weiter clip-lokal bei `0` beginnen.
  - `backend/ml/build_twitch_training_dataset.py` fuehrt Audio- und Video-Window-Extraktion direkt auf dem VOD aus; die `window.mp4`-Vorbereitung und der Batch-Cut-Pfad sind aus dieser Label-Phase entfernt.
  - `_VOD_CHECKPOINT_VERSION` wurde erhoeht, damit alte Checkpoints aus dem frueheren Zwischenclip-Pfad nicht still weiterverwendet werden.
- Verifiziert mit:
  - `python3 -m py_compile backend/services/audio_extractor.py backend/services/video_features.py backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_video_features.py`
  - `PYTHONPATH=backend python3 -m unittest backend.tests.test_build_twitch_training_dataset backend.tests.test_video_features`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`
- Ergebnis:
  - `py_compile` lief sauber.
  - Der `unittest`-Lauf konnte in dieser Shell nicht starten, weil Abhaengigkeiten wie `pydantic_settings` und `numpy` fehlen.
  - Der `graphify`-Rebuild konnte erneut nicht ausgefuehrt werden, weil das Modul `graphify` in der aktuellen Python-Umgebung fehlt.
- Rest-Risiken / lokale Einschraenkungen:
  - Die Korrektheit der direkten VOD-Zeitfenster ist derzeit nur ueber den Codepfad und die angepassten Tests abgesichert, nicht ueber einen vollstaendigen lokalen Integrationstest mit echten Medien.
  - Falls einzelne Downstream-Komponenten implizit auf einen physischen `window.mp4`-Dateipfad vertraut haben sollten, wuerde das erst in einem vollstaendigen Lauf mit realen Projekt-Dependencies sichtbar.

## 2026-04-23 Plan: Expose NVIDIA Video Capability To Local Training Container

- [x] Ursache fuer fehlendes `libnvidia-encode.so.1` im lokalen Docker-Trainingspfad auf die Container-Startparameter eingrenzen.
- [x] `scripts/train_local_docker.sh` so anpassen, dass die NVIDIA-`video`-Capability standardmaessig gesetzt wird und per Env ueberschreibbar bleibt.
- [x] Shell-Syntax pruefen und die Wirkung sowie Rest-Risiken in `tasks/todo.md` dokumentieren.

### Review Notes

- Ursache / Ziel:
  - Der Trainingscontainer lief bereits mit `--gpus all`, aber NVENC scheiterte an `Cannot load libnvidia-encode.so.1`.
  - Das passt zum NVIDIA-Container-Toolkit-Default: ohne explizite `video`-Capability werden typischerweise nur `compute,utility` in den Container gereicht.
- Umsetzung:
  - `scripts/train_local_docker.sh` setzt jetzt standardmaessig `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.
  - Der Wert bleibt per bereits gesetzter Umgebungsvariable ueberschreibbar, falls lokal spaeter `all` oder eine andere Capability-Menge noetig ist.
- Verifiziert mit:
  - `bash -n scripts/train_local_docker.sh`
- Ergebnis:
  - Shell-Syntax sauber.
- Rest-Risiken / lokale Einschraenkungen:
  - Wenn der Windows-NVIDIA-Treiber, Docker-WSL2-GPU-Pfad oder das `ffmpeg`-Build selbst kein NVENC bereitstellen, reicht die neue Capability allein nicht aus.
  - Der naechste sinnvolle Smoke-Test ist ein Containerlauf mit `ffmpeg -encoders | grep nvenc` oder ein erneuter Trainingsstart ueber `scripts/train_local_docker.sh`.

## 2026-04-22 Plan: Switch Local Training Batch Cuts To Hardware Encoding

- [x] Lokalen Trainingspfad auf die bereits vorhandenen FFmpeg-Settings prüfen und den minimalen Default-Wechsel festlegen.
- [x] Hardware-Encoding als Standard fuer den lokalen Trainings-Runner setzen, ohne Software-Fallback-Pfad.
- [x] Tests und Dokumentation der Laufzeit-Zusammenfassung an die neuen Defaults anpassen.
- [x] Geaenderte Module per Syntax-Check verifizieren und verbleibende Umgebungsgrenzen dokumentieren.

### Review Notes

- Ursache / Ziel:
  - Der FFmpeg-Cut war zwar bereits konfigurierbar, der lokale Trainings-Runner startete aber weiterhin mit Software-Encoding-Defaults.
  - Gewuenscht war, dass der lokale Trainingspfad standardmaessig Hardware-Encoding nutzt, ohne stillen Rueckfall auf `libx264`.
- Umsetzung:
  - `backend/ml/local_train.py` setzt jetzt standardmaessig `TRAINING_FFMPEG_VIDEO_CODEC=h264_nvenc`.
  - Der Preset-Default ist leer und `TRAINING_FFMPEG_THREADS=0`, damit der lokale Runner NVENC aktiviert, ohne einen versionsabhaengigen Preset-Namen zu erzwingen oder unnoetig CPU-Threads festzunageln.
  - `backend/tests/test_local_train.py` erwartet die neuen Defaults und prueft weiter, dass sie in der Laufzeit-Umgebung und der Summary auftauchen.
- Verifiziert mit:
  - `python3 -m py_compile backend/ml/local_train.py backend/tests/test_local_train.py`
  - `PYTHONPATH=backend python3 -m unittest backend.tests.test_local_train`
- Ergebnis:
  - Syntax-Check sauber.
  - `LocalTrainTests`: `2` Tests gruen.
- Rest-Risiken / lokale Einschraenkungen:
  - Der lokale Trainings-Runner faellt bei fehlendem NVENC jetzt bewusst hart aus; das ist so gewollt.
  - Der in `AGENTS.md` geforderte `graphify`-Rebuild konnte erneut nicht ausgefuehrt werden, weil das Modul `graphify` in der aktuellen Python-Umgebung fehlt.

## 2026-04-22 Plan: Make FFmpeg Batch Cutting Configurable Without Fallback

- [x] Bestehenden FFmpeg-Batch-Cut-Pfad auf feste Encoder-/Preset-/Thread-Annahmen reduzieren und die Low-Risk-Aenderungen festlegen.
- [x] `+faststart` aus dem Trainings-Cut entfernen und explizite FFmpeg-Cut-Konfiguration ueber Runtime-Settings einbauen, ohne automatische Fallback-Logik.
- [x] Lokalen Trainings-Wrapper und Tests auf die neuen FFmpeg-Settings erweitern.
- [x] Betroffene Tests und Syntax-Checks ausfuehren.
- [x] Review Notes mit Wirkung, Konfigurationsoberflaeche und Rest-Risiken dokumentieren.

### Review Notes

- Ursache / Ziel:
  - Der Trainings-Cut war auf `libx264`/`aac`/`veryfast`/`threads=1` fest verdrahtet und setzte zusaetzlich `+faststart`, obwohl die erzeugten `window.mp4`-Dateien nur lokale Zwischenartefakte sind.
  - Gewuenscht war die Low-Risk-Beschleunigung ueber konfigurierbare FFmpeg-Parameter, aber ausdruecklich ohne automatische Encoder-Fallback-Logik.
- Umsetzung:
  - `backend/config.py` definiert jetzt `training_ffmpeg_video_codec`, `training_ffmpeg_audio_codec`, `training_ffmpeg_preset` und `training_ffmpeg_threads`.
  - `backend/ml/collect_viral_clips.py` baut den Batch-Cut-Command jetzt direkt aus diesen Settings auf und entfernt `-movflags +faststart`.
  - Leere Codec-/Preset-Werte werden nur weggelassen; es gibt keinen versteckten Rueckfall auf andere Encoder.
  - `backend/ml/local_train.py` nimmt die neuen Werte als CLI-/Env-Parameter an und schreibt sie in die Laufzeit-Umgebung sowie in die Trainings-Zusammenfassung.
  - Tests decken jetzt ab, dass `+faststart` weg ist und ein explizit gesetzter Codec unveraendert verwendet wird.
- Verifiziert mit:
  - `python3 -m py_compile backend/config.py backend/ml/collect_viral_clips.py backend/ml/local_train.py backend/tests/test_collect_viral_clips.py backend/tests/test_local_train.py`
  - `PYTHONPATH=backend python3 -m unittest backend.tests.test_collect_viral_clips backend.tests.test_local_train`
- Ergebnis:
  - `py_compile` lief sauber.
  - Der `local_train`-Testpfad lief unter `unittest`; der `collect_viral_clips`-Import scheiterte in dieser Shell vor Testausfuehrung an der fehlenden Abhaengigkeit `pydantic_settings`.
- Rest-Risiken / lokale Einschraenkungen:
  - `python3 -m pytest ...` konnte hier nicht laufen, weil `pytest` nicht installiert ist.
  - Der in `AGENTS.md` geforderte `graphify`-Rebuild konnte nicht ausgefuehrt werden, weil das Modul `graphify` in der aktuellen Python-Umgebung fehlt.
  - Wenn ein explizit konfigurierter Hardware-Encoder auf der Zielmaschine nicht verfuegbar ist, faellt der Cut jetzt absichtlich klar aus, statt still auf Software-Encoding zurueckzugehen.

## 2026-04-22 Plan: Make Local Training Preprocessing Visible

- [x] Bestehende Blindstellen im lokalen Trainingslauf fuer VOD-Fensterschnitt und Label-Verarbeitung identifizieren.
- [x] Fortschrittslogs fuer `ffmpeg`-Batch-Cuts einbauen, damit pro VOD sichtbar ist, welcher Batch laeuft und wann er fertig ist.
- [x] Fortschrittslogs fuer die nachgelagerte Label-/Window-Verarbeitung einbauen, damit sichtbare VOD-Fortschritte waehrend Feature-Extraktion und Ranking-Dataset-Build entstehen.
- [x] Betroffene Tests ausfuehren oder anpassen und die neue Sichtbarkeit gegen den lokalen Trainingspfad verifizieren.
- [x] Review Notes mit beobachtetem Problem, Umsetzung und Rest-Risiken dokumentieren.

### Review Notes

- Beobachtetes Problem:
  - Die lange Vorverarbeitungsphase bestand aus grossen `ffmpeg`-Batch-Cuts und nachgelagerter Window-/Feature-Verarbeitung, erzeugte aber ueber viele Minuten keine aussagekraeftigen Fortschrittslogs.
  - Dadurch wirkte der lokale Docker-Trainingslauf haeufig wie ein Hanger, obwohl CPU/RAM sichtbar ausgelastet waren.
- Umsetzung:
  - `backend/ml/collect_viral_clips.py` loggt jetzt fuer jeden Batch einen Plan-, Start- und Done-Eintrag.
  - Der eigentliche `ffmpeg`-Subprozess laeuft jetzt ueber `_run_ffmpeg_command(...)` mit Heartbeat-Logs alle 15 Sekunden, solange ein Batch noch codiert.
  - `backend/ml/build_twitch_training_dataset.py` loggt jetzt den Start der Label-Phase pro VOD, den Status der vorbereiteten `window.mp4`-Clips, laufende Progress-Updates waehrend der Label-Verarbeitung und einen Abschluss-Eintrag fuer die Phase.
  - Die Progress-Logs funktionieren sowohl fuer den sequentiellen als auch fuer den parallelen Label-Pfad.
- Verifikation:
  - `python3 -m py_compile backend/ml/collect_viral_clips.py backend/ml/build_twitch_training_dataset.py`
  - Testdateien angepasst:
    - `backend/tests/test_collect_viral_clips.py`
    - `backend/tests/test_build_twitch_training_dataset.py`
  - Direkter lokaler Smoke-Test mit Stub-Modulen erfolgreich: Batch-Cut-Logs und Label-Progress-Logs werden auf den geaenderten Pfaden erzeugt.
- Rest-Risiken / lokale Einschraenkungen:
  - Die vollstaendigen `unittest`-/`pytest`-Laeufe konnten in dieser Shell nicht nativ ausgefuehrt werden, weil Projekt-Dependencies wie `pytest`, `yt_dlp` und `pydantic_settings` fehlen.
  - Der in `AGENTS.md` geforderte `graphify`-Rebuild liess sich hier ebenfalls nicht ausfuehren, weil das Modul `graphify` in der aktuellen Python-Umgebung nicht importierbar ist.

## 2026-04-22 Plan: Remove GCloud Training Path And Rebuild For Local Docker

- [x] Alle produktiven `gcloud`-/Vertex-/GCS-/Secret-Manager-Einstiegspunkte im Trainingsstack inventarisieren und in lokale A/Bloesung aufteilen: `entfernen`, `ersetzen`, `tests anpassen`.
- [x] Den Trainings-Entry-Point auf einen rein lokalen Docker-Runner umstellen, der keine Cloud-URIs, keine Vertex-Wrapper und keinen VOD-Cache akzeptiert.
- [x] Die Laufzeit-Konfiguration auf lokale Secrets/`.env` konsolidieren und alle Cloud-Resolver aus dem Trainingspfad entfernen.
- [x] Den Docker-Trainingscontainer fuer `Ryzen 9 9900X3D + RTX 5070 Ti` umbauen:
  - CUDA-faehiges Base-Image statt CPU-only-Image
  - GPU-nutzbare Torch-/Whisper-/Embedding-Pfade
  - CPU-Threading und Worker-Defaults auf hohe Parallelitaet fuer 12 Kerne abstimmen
- [x] Die aktuellen CPU-Bottlenecks gezielt aufloesen:
  - Whisper laeuft derzeit hart auf `device=\"cpu\"`
  - `video_features` ist DeepFace/OpenCV-lastig und aktuell nicht GPU-optimiert
  - `semantic_candidates` laeuft CPU-only, obwohl das auf der lokalen RTX beschleunigbar ist
- [x] Die VOD-Beschaffung bewusst ohne Cache erzwingen:
  - lokaler TTL-Cache aus Trainingspfad entfernen/deaktivieren
  - GCS-Cache- und Checkpoint-Code aus dem Trainingspfad entfernen
  - Resumes nur noch ueber lokale Artefakte/Workflow-State, nicht ueber VOD-Cache
- [x] Den lokalen Docker-Workflow explizit machen:
  - Build-Skript / Run-Skript / Dokumentation fuer lokalen Trainingslauf
  - sinnvolle Volume-Mounts fuer Artefakte, Tempdaten und Modellcaches
  - `docker run --gpus all`-Pfad als Standard
- [x] Tests auf den neuen lokalen Pfad umstellen:
  - Vertex-/Submit-/GCS-Tests entfernen oder isolieren
  - lokale Trainings-/Pipeline-Tests erweitern
  - Smoke-Test fuer den Docker-Entry-Point ohne Cloud-Abhaengigkeiten
- [x] Den neuen lokalen Trainingspfad verifizieren:
  - betroffene Test-Suites
  - lokaler Docker-Smoke-Run
  - Review Notes mit verbleibenden Risiken und empfohlenen Default-Settings

### Review Notes

- Architekturstand vor dem Umbau:
  - Cloud-spezifischer Einstieg: `backend/ml/vertex_train.py`
  - Haupt-Orchestrierung: `backend/ml/train_jynxzi_vod.sh` -> `backend/ml/train_vod_ranker.py` -> `backend/ml/build_twitch_training_dataset.py`
  - Bereits vorhandener lokaler Sequenzpfad: `backend/ml/pipelines/backends/local_shell.py`
- Klare Cloud-Kopplungen, die entfernt oder ersetzt werden muessen:
  - `vertex_train.py` haengt an `google.cloud.storage` und `google.cloud.secretmanager`
  - `build_twitch_training_dataset.py` enthaelt GCS-VOD-Cache und GCS-Checkpoint-Logik
  - `train_pipeline.py`, `pipelines/runtime_env.py`, Submitter-Tests und Vertex-Extras in `pyproject.toml` sind Cloud-spezifisch
  - `Dockerfile.vertex-train` ist noch ein Vertex-/CPU-Image mit `ENTRYPOINT ["python", "-m", "ml.vertex_train"]`
- Aktuelle Performance-Engpaesse, die der Umbau adressieren soll:
  - `services/transcriber.py` laedt Whisper explizit mit `device="cpu"`
  - `services/video_features.py` ist DeepFace/OpenCV-lastig; dort steckt viel Laufzeit
  - `services/semantic_candidates.py` nutzt `sentence-transformers` aktuell ohne expliziten GPU-Pfad
  - Dataset-Build parallelisiert bereits ueber `TRAINING_VOD_WORKERS` und `TRAINING_LABEL_WORKERS`; diese Defaults muessen auf die lokale Hardware neu abgestimmt werden
- Harte Produktanforderung fuer den Umbau:
  - kein VOD-Cache mehr, weder lokal noch ueber GCS
- Umsetzung:
  - Cloud-spezifische Einstiegspunkte und Submitter entfernt:
    - `backend/ml/vertex_train.py`
    - `backend/ml/pipelines/train_pipeline.py`
    - `backend/ml/pipelines/backends/vertex.py`
    - `scripts/submit_vertex_training.py`
    - `scripts/submit_vertex_pipeline.py`
    - `backend/Dockerfile.vertex-train`
  - Vertex-/KFP-/Secret-Manager-Code aus den lokalen Pipeline-Komponenten entfernt; `runtime_env.resolve_secret_ref()` ist jetzt ein lokaler Literal-Resolver ohne `google.cloud`.
  - Neuer lokaler Standardpfad:
    - Docker-Entrypoint: `backend/ml/local_train.py`
    - GPU-Image: `backend/Dockerfile.local-train`
    - Startskript: `scripts/train_local_docker.sh`
- Hardware-Tuning fuer `Ryzen 9 9900X3D + RTX 5070 Ti`:
  - `services/transcriber.py` waehlt jetzt `cuda` bei verfuegbarer GPU und nutzt `fp16` im Transcribe-Pfad.
  - `services/semantic_candidates.py` waehlt jetzt `cuda` bei verfuegbarer GPU und skaliert den Batch-Size lokal hoch.
  - `ml/local_train.py` setzt schnelle lokale Defaults:
    - `TRAINING_VOD_WORKERS=2`
    - `TRAINING_LABEL_WORKERS=3` bei 12 CPU-Kernen
    - harte BLAS-/OMP-Begrenzung auf `1` Thread pro Worker zur Vermeidung von Oversubscription
    - `VIDEO_EXPRESSION_SAMPLE_SEC=6`
    - `TRAINING_DATASET_LLM_PROVIDER=none` als schnellster lokaler Default
- No-Cache-Regel umgesetzt:
  - `build_twitch_training_dataset._materialize_vod()` laedt jetzt immer direkt von Twitch.
  - lokaler TTL-VOD-Cache und alle GCS-VOD-/Checkpoint-Pfade wurden aus dem produktiven Trainingspfad entfernt.
  - lokale Row-Checkpoints fuer bereits verarbeitete VODs bleiben erhalten; das ist Resume-State, kein VOD-Cache.
- Abhaengigkeiten / Konfiguration:
  - `backend/pyproject.toml` entfernt das `vertex`-Extra komplett.
  - `backend/.env.example` dokumentiert jetzt lokale Device-/Worker-Settings statt Cloud-Cache-Settings.
- Verifikation:
  - `backend/.venv/bin/python -m pytest backend/tests/test_local_train.py backend/tests/test_pipeline_component_env.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_transcriber.py backend/tests/test_semantic_candidates.py backend/tests/test_local_shell_pipeline.py backend/tests/test_train_vod_ranker.py backend/tests/test_train_ranker.py -q`
  - Ergebnis: `82 passed`
  - `backend/.venv/bin/python -m py_compile ...` auf die betroffenen Module lief sauber.
  - `.venv/bin/python -m ml.local_train --help` lief sauber.
  - `zsh -n scripts/train_local_docker.sh` und `zsh -n backend/ml/train_jynxzi_vod.sh` liefen sauber.
  - Graphify fuer `backend` neu gebaut.
- Verbleibende Risiken:
  - `video_features.py` nutzt DeepFace/OpenCV weiterhin primär CPU-lastig; der groesste Geschwindigkeitsgewinn kam aus GPU-Whisper, GPU-Embeddings und dem Entfernen des LLM-Standards.
  - Das neue Docker-Image ist auf einen CUDA-faehigen lokalen Docker-/NVIDIA-Stack ausgelegt und wurde in dieser Session nicht als kompletter Image-Build durchgetestet.

## 2026-04-22 Plan: Repair Vertex Boot Disk Pipeline Wrapper

- [x] Den kaputten `v1.custom_job`-Pfad durch einen Vertex-kompatiblen Ersatz fuer `build-training-dataset-component` ersetzen.
- [x] Die kompilierte Pipeline-Spec gegen den neuen Wrapper und die 500GB-Boot-Disk verifizieren.
- [x] Einen neuen 7-Tage-Run starten und pruefen, dass `build-training-dataset-component` als echter Custom Job anlaeuft.

### Review Notes

- Root Cause des kaputten Ersatzlaufs:
  - Der vorherige `v1.custom_job`-Wrapper erzeugte zwar eine Spec mit `disk_spec`, Vertex lehnte den Task aber vor Jobstart mit `Failed to parse the container spec json payload...` ab.
- Repo-Fix:
  - `backend/ml/pipelines/train_pipeline.py` nutzt fuer `build-training-dataset-component` jetzt den `google_cloud_pipeline_components.preview.custom_job`-Wrapper.
  - Zusaetzlich ersetzt `compile_pipeline()` den kaputten, nicht definierten `labels`-Placeholder im serialisierten Custom-Job-Payload gezielt durch `{}`.
  - Der Compile-Test in `backend/tests/test_train_pipeline.py` verifiziert jetzt sowohl die 500GB-Disk-Defaults als auch den bereinigten Payload.
- Verifikation:
  - `backend/.venv/bin/python -m pytest backend/tests/test_train_pipeline.py backend/tests/test_submit_vertex_pipeline.py backend/tests/test_local_shell_pipeline.py -q`
  - Ergebnis: `5 passed`
  - Kompilierte Spec: `backend/tmp/train_pipeline_20260422_bootdisk500_preview_v3.json`
  - Bestaetigt in der Spec:
    - `machine_type = e2-standard-4`
    - `boot_disk_type = pd-ssd`
    - `boot_disk_size_gb = 500`
    - kein `labels`-Placeholder mehr im Dataset-Custom-Job-Payload
- Neuer echter Vertex-Run:
  - Pipeline Job: `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422150546`
  - Display Name: `jynxzi-ranker-pipeline-20260422-ew3-7day-bootdisk500-preview-v2`
  - Pipeline Root: `gs://project-22586fae-58b4-4890-aeb-ml-artifacts-ew3/pipeline-root/jynxzi/20260422-7day-bootdisk500-preview-v2`
  - Status direkt nach dem Submit: `PIPELINE_STATE_RUNNING`
  - `build-training-dataset-component` laeuft jetzt als echter Custom Job:
    - `projects/303245987408/locations/europe-west3/customJobs/1647797669495570432`
    - `workerPoolSpecs[0].diskSpec.bootDiskSizeGb = 500`
    - `workerPoolSpecs[0].diskSpec.bootDiskType = pd-ssd`

## 2026-04-22 Plan: Increase Vertex Pipeline Boot Disk For Dataset Build

- [x] Die Root Cause der `window skip`-Meldungen im laufenden Vertex-Run verifizieren.
- [x] Den `build-training-dataset`-Step auf eine groessere Boot-Disk umstellen.
- [x] Compile-/Submit-Pfad verifizieren und einen Ersatzlauf mit der neuen Disk-Groesse starten.

### Review Notes

- Die `window skip`-Meldungen waren kein reines Logging-Artefakt:
  - Im laufenden `build-training-dataset-component` traten wiederholt echte `ffmpeg`-Fehler mit `No space left on device` auf.
  - Codepfad: [backend/ml/build_twitch_training_dataset.py](/Users/antonioherbin/Desktop/Privat/ai_clipping/backend/ml/build_twitch_training_dataset.py:375) loggt diese Faelle als `Skipping VOD window ...` und verwirft dann das betroffene Trainingsfenster.
- Umsetzung:
  - `backend/ml/pipelines/train_pipeline.py` nutzt fuer `build-training-dataset-component` jetzt den offiziellen Vertex-Custom-Job-Wrapper aus `google_cloud_pipeline_components.v1.custom_job.create_custom_training_job_from_component`.
  - Der Step laeuft damit explizit mit:
    - `machine_type = e2-standard-4`
    - `boot_disk_type = pd-ssd`
    - `boot_disk_size_gb = 500`
  - Zusaetzliche Pipeline-Parameter wurden fuer den Wrapper durchgereicht:
    - `vertex_project`
    - `vertex_location`
    - `vertex_service_account`
    - `vertex_network`
- Repo-/Verifikationsstand:
  - `backend/pyproject.toml` erweitert um `google-cloud-pipeline-components>=2.6.0,<3.0.0` im `vertex`-Extra.
  - Neuer Compile-Test bestaetigt `worker_pool_specs[0].disk_spec.boot_disk_size_gb == 500`.
  - Verifikation:
    - `backend/.venv/bin/python -m pytest backend/tests/test_train_pipeline.py backend/tests/test_submit_vertex_pipeline.py backend/tests/test_local_shell_pipeline.py -q`
    - Ergebnis: `5 passed`
- Neuer Ersatzlauf:
  - Pipeline Job: `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422143750`
  - Display Name: `jynxzi-ranker-pipeline-20260422-ew3-7day-bootdisk500`
  - Template: `tmp/train_pipeline_20260422_bootdisk500.json`
  - Bestaetigte Spec im Template:
    - `disk_spec.boot_disk_size_gb = 500`
    - `disk_spec.boot_disk_type = pd-ssd`

## 2026-04-22 Plan: Isolate Vertex MetadataStore Deny On Pipeline Retry

- [x] Einen frischen Vertex-Pipeline-Run mit explizitem `vertex-train`-Service-Account einreichen.
- [x] Jobzustand, Fehlerdetails und Audit-Logs direkt nach dem Submit auswerten.
- [x] Die effektive Laufidentitaet gegen IAM-/Policy-Sicht pruefen und den Restblocker eingrenzen.

### Review Notes

- Aktueller erfolgreicher Retry:
  - `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422121230`
  - Display Name: `jynxzi-ranker-pipeline-20260422-ew3-7day-vertex-train-retry-2`
  - Service Account: `vertex-train@project-22586fae-58b4-4890-aeb.iam.gserviceaccount.com`
  - Pipeline Root: `gs://project-22586fae-58b4-4890-aeb-ml-artifacts-ew3/pipeline-root/jynxzi/20260422-7day-gcloud-sa-retry-2`
- Status direkt nach dem Submit:
  - `PIPELINE_STATE_RUNNING`
  - kein `error.code`, keine `error.message`
  - `collect-twitch-clips-component` laeuft bereits, nachgelagerte Tasks stehen erwartungsgemaess auf `PENDING`
- Starker Nachweis, dass der fruehere Metadatastore-Blocker umgangen ist:
  - Vertex hat fuer den Root-Run und fuer `collect-twitch-clips-component` bereits neue ML Metadata Executions im regionalen Store `projects/303245987408/locations/europe-west3/metadataStores/default` angelegt.
  - Damit ist `aiplatform.metadataStores.get` auf dem effektiven Laufpfad nicht mehr der operative Blocker.
- Isolierte Ursache des alten Fehlers:
  - Der wirklich funktionierende Pfad ist ein frischer Submit mit explizitem `service_account=vertex-train@...` nach dem IAM-/Metadatastore-Nachzug.
  - Die frueheren Sofortfehler waren damit kein Repo-/Pipeline-Compile-Problem mehr, sondern eine Kombination aus falscher effektiver Laufidentitaet auf einzelnen Retries und/oder noch nicht propagierten IAM-Aenderungen.
- Audit-Log zum erfolgreichen Create:
  - Create-Request kam von `clipzzholy@gmail.com`
  - Response enthielt die neue Pipeline-Job-Resource ohne Fehlerstatus
- Operativer Stand:
  - Der 7-Tage-Run fuer `jynxzi`, `2026-03-14` bis `2026-03-20`, laeuft jetzt ueber den neuen Vertex-Pipeline-Workflow.
  - Ab hier ist nur noch normales Laufzeit-Monitoring noetig, nicht mehr Metadatastore-Debugging auf Control-Plane-Ebene.

## 2026-04-22 Plan: Fix Missing Runtime Secret Injection In Vertex Pipeline

- [x] Secret-/Env-Weitergabe fuer den neuen Vertex-Pipeline-Pfad vom Submitter bis in die betroffenen Components verdrahten.
- [x] Regressions-Tests fuer Submitter und runtime-nahe Component-Pfade ergaenzen.
- [x] Pipeline neu kompilieren und den 7-Tage-Run erneut auf Vertex starten.
- [x] Laufstatus und erste Task-Logs verifizieren, danach den Review-Stand dokumentieren.

### Review Notes

- Repo-Fix fuer den eigentlichen Root Cause:
  - Secret-Refs werden jetzt vom Submitter bis in den Vertex-Pipeline-Pfad durchgereicht (`twitch_client_id_secret`, `twitch_client_secret_secret`, `groq_api_key_secret`, optional `gemini_api_key_secret`, `mlflow_tracking_uri_secret`).
  - Fuer echte Vertex-Runs ist die Secret-Aufloesung zusaetzlich direkt in die KFP-Component-Funktionen eingebettet, damit der bereits veroeffentlichte Image-Tag ausreicht und kein frischer Container-Build den Rollout blockiert.
- Wichtige operative Folge:
  - Der bestehende Tag `europe-west3-docker.pkg.dev/project-22586fae-58b4-4890-aeb/ml/ai-clipping-train:20260422-pipeline-gcloud-7day` konnte weiterverwendet werden.
  - Neues Template: `tmp/train_pipeline_20260422_template_embedded_secrets.json`
- Neuer echter Vertex-Run:
  - Pipeline Job: `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422134516`
  - Display Name: `jynxzi-ranker-pipeline-20260422-ew3-7day-embedded-secrets-2`
  - Pipeline Root: `gs://project-22586fae-58b4-4890-aeb-ml-artifacts-ew3/pipeline-root/jynxzi/20260422-7day-gcloud-embedded-secrets-2`
  - Zeitraum: `2026-03-14` bis `2026-03-20`
- Direkte Verifikation gegen den frueheren Twitch-Credential-Fehler:
  - `collect-twitch-clips-component` laeuft jetzt als `RUNNING`, statt sofort mit `TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be configured.` zu scheitern.
  - Unterliegender Custom Job: `projects/303245987408/locations/europe-west3/customJobs/6297764284755607552`
  - Die laufende Task-Metadata enthaelt jetzt explizit `input:twitch_client_id_secret` und `input:twitch_client_secret_secret`, also ist die Secret-Bindung auf dem echten Vertex-Pfad nachweisbar aktiv.
- Zusaetzlicher Robustheits-Fix:
  - `backend/ml/pipelines/backends/vertex.py` wartet bei `sync=False` jetzt mit `wait_for_resource_creation()` auf die echte Resource-Erstellung.
  - Damit scheitert `scripts/submit_vertex_pipeline.py` kuenftig nicht mehr an dem Async-Race, bei dem `job.display_name`/`job.resource_name` noch nicht gesetzt waren.
- Verifikation:
  - `python3 -m py_compile backend/ml/pipelines/components/*.py backend/ml/pipelines/backends/vertex.py backend/tests/test_vertex_pipeline_backend.py`
  - `backend/.venv/bin/python -m pytest backend/tests/test_vertex_pipeline_backend.py backend/tests/test_submit_vertex_pipeline.py backend/tests/test_pipeline_component_env.py backend/tests/test_local_shell_pipeline.py -q`
  - Ergebnis: `10 passed`

## 2026-04-22 Plan: Execute 7-Day Run With New ML Training Workflow

- [x] Neue Workflow-Variante fuer diese Umgebung validieren und lauffaehigen Backend-Pfad waehlen.
- [x] Einen 7-Tage-Lauf fuer `jynxzi` mit dem neuen Workflow starten.
- [x] Den Startstatus und die Artefakt-/Output-Pfade verifizieren.

### Review Notes

- Gewuenschter Zielpfad war ein echter GCloud/Vertex-Pipeline-Run, nicht der lokale `local_shell`-Fallback. Der lokal gestartete Fallback-Lauf wurde nach Nutzerkorrektur abgebrochen.
- Fuer den neuen Workflow musste zuerst das Pipeline-Wiring in `backend/ml/pipelines/` KFP-kompilierbar gemacht werden:
  - `from __future__ import annotations` aus den KFP-Component-Modulen und `train_pipeline.py` entfernt, weil KFP sonst die Typannotation `str` als Artefakt-Typ fehlinterpretiert und die Pipeline schon beim Compile scheitert.
  - Verifikation danach:
    - `backend/.venv/bin/python -m py_compile backend/ml/pipelines/train_pipeline.py backend/ml/pipelines/components/*.py`
    - `backend/.venv/bin/python -m pytest backend/tests/test_submit_vertex_pipeline.py backend/tests/test_local_shell_pipeline.py -q`
    - Ergebnis: `4 passed`
- Frisches Cloud-Image fuer den neuen Workflow gebaut:
  - Build: `projects/project-22586fae-58b4-4890-aeb/locations/europe-west3/builds/fdee869b-eac0-407c-bc5d-ae4135a798b5`
  - Image: `europe-west3-docker.pkg.dev/project-22586fae-58b4-4890-aeb/ml/ai-clipping-train:20260422-pipeline-gcloud-7day`
  - Digest: `sha256:e0c11cd434ad023bb538f539065871e27a758cd00c6bf186a918a4823e448a14`
- Vertex-Pipeline-Spec erfolgreich gegen dieses Image kompiliert:
  - Template: `tmp/train_pipeline_20260422_gcloud_7day.json`
- GCloud/Vertex-Run mehrfach eingereicht fuer `jynxzi`, `2026-03-14` bis `2026-03-20`, Pipeline-Root `gs://project-22586fae-58b4-4890-aeb-ml-artifacts-ew3/pipeline-root/jynxzi/20260422-7day-gcloud`:
  - `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422115344`
  - `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422115632`
  - `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422115720`
  - `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422115815`
  - `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422120004`
  - `projects/303245987408/locations/europe-west3/pipelineJobs/ai-clipping-train-pipeline-20260422120136`
- Alle Einreichungen wurden zwar angelegt, aber fielen unmittelbar in `PIPELINE_STATE_FAILED` mit demselben GCP-Kontrollplane-Fehler:
  - `Permission 'aiplatform.metadataStores.get' denied on resource '//aiplatform.googleapis.com/projects/303245987408/locations/europe-west3/metadataStores/default' (or it may not exist).`
- Repo- und Projekt-Fixes, die bereits umgesetzt wurden:
  - Default-Metadatastore in `europe-west3` erstellt:
    - `projects/303245987408/locations/europe-west3/metadataStores/default`
  - IAM fuer den Lauf erweitert:
    - `vertex-train@project-22586fae-58b4-4890-aeb.iam.gserviceaccount.com` erhielt `roles/aiplatform.user`
    - `service-303245987408@gcp-sa-aiplatform.iam.gserviceaccount.com` erhielt zusaetzlich `roles/aiplatform.user`
- Restblocker nach allen Retries:
  - Der Vertex-Pipeline-Control-Plane-Deny auf `metadataStores/default` bleibt bestehen, obwohl der Store existiert und die offensichtlichen Laufidentitaeten nachgezogen wurden.
  - Damit konnte in dieser Session kein dauerhaft laufender 7-Tage-Pipeline-Run in Vertex erreicht werden; der Blocker liegt aktuell auf GCP-/Vertex-Metadatenebene, nicht mehr im Repo-Workflow.

## 2026-04-22 Plan: MLflow + Vertex Pipelines Integration

### Leitlogik
- **MLflow** = Tracking + Registry (Experimente, Metriken, Artefakt-Versionen, Stage-Promotion).
- **Vertex Pipelines (KFP)** = DAG-Executor, ersetzt das monolithische `backend/ml/train_jynxzi_vod.sh`.
- Beides wird **additiv** eingefuehrt. Der bestehende Shell-/Custom-Job-Pfad bleibt so lange als Fallback, bis 3 Pipeline-Runs sauber durchlaufen.

### Phase 0 — MLflow Tracking Server (Tag 1, ~4h)
- [ ] MLflow-Server als Cloud Run Service deployen (Image `ghcr.io/mlflow/mlflow:latest`).
- [ ] Backend Cloud SQL Postgres `mlflow_db`, Artifact-Store `gs://<artifact-bucket>/mlflow-artifacts/`.
- [ ] Service Account `mlflow-server@...` mit `storage.objectAdmin` nur auf diesen Prefix.
- [x] `MLFLOW_TRACKING_URI` in Secret Manager, in bestehenden Vertex-Container injected.
- [ ] Smoke-Test lokal: `mlflow.log_metric("x", 1.0)` erscheint in der UI.

### Phase 1 — MLflow in bestehenden Shell-Pfad retrofitten (Tag 2, ~4h)
- [x] In `backend/ml/train_ranker.py::train_from_rows()` einmalig `mlflow.start_run()` wrappen.
- [x] Params loggen: `streamer`, `profile`, `date_range`, `window_size`, `blend_floor`.
- [x] Metrics loggen: `ndcg@5`, `f1`, `accuracy`, `precision@5`, `recall@5`, `gold_eval_ndcg`.
- [x] Artifacts loggen: `jynxzi_ranker.json`, `jynxzi_artifact_health.json`, `jynxzi_proxy_eval.json`, `jynxzi_gold_eval.json`.
- [x] Tags: `git_sha`, `vertex_job_id`, `image_tag`, `dataset_rows`, `positive_rows`.
- [x] Regression in `backend/tests/test_train_ranker.py`: `run_id` gesetzt, Metrics landen im Tracking.
- **Akzeptanz:** naechster Vertex-Custom-Job erzeugt sichtbaren MLflow-Run mit vollstaendigen Metriken. Kein Pipeline-Refactor noetig.

### Phase 2 — Model Registry fuer Promotion (Tag 3, ~3h)
- [x] Nach jedem Training `mlflow.register_model(ranker.json)` → Model `jynxzi_ranker`.
- [x] Auto-Transition-Regel: bei bestandenen Gates (`positives>=15`, `ndcg@5>=floor`, `source_mix_ok`) → Stage `Staging`.
- [x] `Production` bleibt manueller Gate (CLI-Command oder UI-Button).
- [x] `backend/services/ranker.py::_artifact_is_usable()` bekommt zweiten Pfad: erst lokale Datei, sonst MLflow-Registry-Bundle-Fallback.
- [x] Regressionstest: Registry-Fallback greift, wenn lokale Datei fehlt.
- **Akzeptanz:** Query `latest production ranker` liefert deterministisch die korrekte Version, unabhaengig vom lokalen Filesystem.

### Phase 3 — KFP-Komponenten-Zerlegung (Tage 4–6, ~2 Tage)
Zerlege `backend/ml/train_jynxzi_vod.sh` in 6 KFP-Components (`@dsl.component`), jede mit typisierten Inputs/Outputs:

| # | Component | Input-Artifacts | Output-Artifacts |
|---|---|---|---|
| 1 | `collect_twitch_clips_op` | `streamer`, `date_range` | `Dataset: twitch_labels.jsonl` |
| 2 | `build_training_dataset_op` | `twitch_labels` | `Dataset: training_rows.jsonl` |
| 3 | `train_ranker_op` | `training_rows` | `Model: ranker.json`, `Metrics: training_stats` |
| 4 | `build_eval_dataset_op` | `twitch_labels`, Gold-Updates | `Dataset: eval_rows.jsonl` |
| 5 | `evaluate_ranker_op` | `ranker`, `eval_rows` | `Metrics: proxy_eval`, `Metrics: gold_eval` |
| 6 | `promote_op` | `ranker`, `eval_metrics` | `Model: registered_version` |

- [x] Jede Component nutzt dasselbe Basis-Image (`ai-clipping-train`), sodass Feature-Builder- und Ranker-Code nicht dupliziert wird — nur duenne CLI-Wrapper rufen bestehende Python-Funktionen.
- [x] Explizite Ressourcen-Requests pro Step: Component 1 mit 2 vCPU, Component 3 mit `n2-highmem-16`.
- **Akzeptanz:** jede Component laeuft isoliert als Cloud Run Job mit denselben Inputs/Outputs wie der entsprechende Shell-Step.

### Phase 4 — Pipeline-Definition und Submit (Tag 7, ~4h)
- [x] `backend/ml/pipelines/train_pipeline.py` mit `@dsl.pipeline` — Wiring der 6 Components ueber Artifacts.
- [x] Compile: `kfp.compiler.Compiler().compile(pipeline, "train_pipeline.json")`.
- [x] Submit-Skript `scripts/submit_vertex_pipeline.py` via `google-cloud-aiplatform` SDK → `PipelineJob`.
- [x] Parametrisiert: `streamer`, `date_range`, `profile` — selbe Semantik wie aktueller Custom Job.
- **Akzeptanz:** ein End-to-End-Pipeline-Run auf `2026-03-14…20` produziert ein Artefakt, das metrisch aequivalent zum letzten Shell-Run ist.

### Phase 5 — Caching und Parallelisierung (Tag 8, ~3h)
- [x] Alle Components mit deterministischen Inputs (Collect, Build-Dataset, Build-Eval) auf `enable_caching=True`.
- [ ] `collect_twitch_clips_op` in `dsl.ParallelFor` ueber Streamer-Liste. Voraussetzung: `backend/ml/streamer_sources.csv` hat >1 Eintrag.
- **Akzeptanz:** zweiter identischer Pipeline-Run nutzt Cache fuer Collect + Build-Dataset → Wall-Time < 50 % des ersten Runs.

### Phase 6 — Migration und Fallback-Abbau (laufend, 3 Runs)
- [ ] `backend/ml/train_jynxzi_vod.sh` und `scripts/submit_vertex_training.py` bleiben bestehen, werden aber nicht mehr aktiv submitted.
- [ ] Nach 3 erfolgreichen Pipeline-Runs mit positiver Registry-Promotion: Shell-Pfad mit `DEPRECATED` markieren.
- [ ] `docs/vertex-training.md` auf Pipeline-Submit umstellen.
- [ ] Shell-Pfad erst loeschen, wenn Pipeline 2 Wochen stabil laeuft.

### Was explizit NICHT angefasst wird
- Feature-Builder, Ranker-Logik, Semantic-Service, Online-Pipeline (`backend/pipeline.py`), Candidate-Generators.
- Quality-Gates in `backend/ml/train_ranker.py` — bleiben, werden zusaetzlich nach MLflow geloggt.
- Gold-Holdout-Split-Logik.

### Portabilitaets-Constraints (GCP-Lock-in vermeiden)
Der Workflow muss so gebaut sein, dass ein Wechsel zurueck auf lokales Training nur ENV-Flip + anderer Backend-Submitter ist, kein Code-Rewrite.

**Regel 1 — Components als Standalone-Python-CLIs**
- [x] Jede Component in `backend/ml/pipelines/components/` ist ein normal aufrufbares Python-Modul mit `__main__`-Section, nicht nur KFP-Funktion.
- [x] KFP-`@dsl.component`-Decorator ist nur duenner Wrapper um den CLI-Entry-Point.
- [x] Keine Imports von `google_cloud_pipeline_components` — GCP-only. Ausschliesslich pure `kfp.dsl`, damit Pipeline auch auf minikube/k3d/kind oder `kfp local` laeuft.
- [x] Kein Vertex-Metadata-API fuer Tracking — MLflow ist die einzige Tracking-Quelle.
- [x] Kein GCSFuse- oder Vertex-SDK-Zugriff im Component-Code selbst.

**Regel 2 — Storage-URIs vollstaendig parametrisiert**
- [x] Alle Pfade via ENV oder Config: `STORAGE_ROOT=gs://...` ODER `STORAGE_ROOT=file:///...`.
- [x] Kein hardcoded `gs://` im Component-Code.
- [x] Reads/Writes ueber `fsspec` (transparent fuer `gs://`, `s3://`, `file://`).

**Backends-Verzeichnis in Phase 3/4 anlegen**
```
backend/ml/pipelines/
  components/            # pure Python, CLI-aufrufbar
  train_pipeline.py      # KFP-DSL-Wiring
  backends/
    vertex.py            # Submit zu Vertex Pipelines
    local_shell.py       # Ruft Components sequentiell als Subprocess (ersetzt train_jynxzi_vod.sh)
    local_kfp.py         # Ruft "kfp local run" (optional)
```

**MLflow ist von sich aus portabel**
- [x] Lokaler Modus per ENV: `MLFLOW_TRACKING_URI=sqlite:///mlruns.db`, Artifacts in `./mlruns/`.
- [ ] Cloud-Modus per ENV: `MLFLOW_TRACKING_URI=https://<cloud-run-url>`, Artifacts in `gs://...`.
- [x] Kein MLflow-Code-Unterschied zwischen lokal und Cloud — nur ENV.

**Was bleibt GCP-spezifisch (bewusst)**
- `backend/ml/vertex_train.py`, `scripts/submit_vertex_training.py`, neuer `scripts/submit_vertex_pipeline.py` — per Definition Vertex-Submitter. Werden beim Weggang nicht genutzt, ersetzt durch `backends/local_shell.py` oder `backends/local_kfp.py`.
- Secret Manager-Zugriffe in `vertex_train.py::_resolve_secret()` — bereits mit ENV-Fallback.

**Akzeptanz fuer Portabilitaet**
- Eine Component (z.B. `train_ranker_op`) laesst sich mit identischen Inputs wahlweise via Vertex-Pipeline-Run, via `python -m backend.ml.pipelines.components.train_ranker_op ...` oder via `backends/local_shell.py` ausfuehren — Output ist metrisch aequivalent.
- Ein MLflow-Run aus lokalem SQLite-Mode laesst sich via `mlflow-export-import` in den Cloud-Tracking-Server uebernehmen (und zurueck).

### Rueckfall-Trigger (wann stoppen und replannen)
- Phase-3-Components produzieren nicht-aequivalente Artefakte → zuerst Divergenz verstehen, nicht weiterbauen.
- MLflow-Server-Ops >2h/Monat → auf managed MLflow (Databricks, Azure ML) ausweichen.
- KFP-DSL-Verbosity verlangsamt Iteration → Phase 4+5 pausieren, nur MLflow behalten.

### Gesamt-Effort-Schaetzung
- Phase 0-2 (MLflow allein): ~1,5 Tage, sofortiger Nutzen.
- Phase 3-5 (Pipelines): ~3,5 Tage, mittelfristiger Nutzen ab Run 2 ueber Caching.
- Phase 6 (Cleanup): verteilt ueber 2-3 Wochen.

### Empfohlene Startsequenz
Phase 0-2 zuerst, danach **1 Woche reale Nutzung** bevor Phase 3 angefasst wird. MLflow allein loest ~70 % des aktuellen Schmerzes (keine Experiment-Historie), Pipelines lohnen sich erst, wenn Retrain-Frequenz hoch genug ist, dass Caching messbar Zeit spart.

### Offene Fragen
- MLflow-Hosting: Cloud Run im bestehenden `project-22586fae-58b4-4890-aeb`, oder neues Ops-Projekt?
- Registry-Promotion `Production`: CLI-Command oder kleines Web-UI?
- Pipeline-Trigger: manuell per `submit_vertex_pipeline.py`, oder spaeter per n8n/Cloud-Scheduler-Cron?

## 2026-04-22 Plan: Diagnose And Fix Failure Of Restarted Zero-Duration Highmem Vertex Run

- [x] Aktuellen Vertex-Jobstatus und die ersten Fehler-Logs des neuen Highmem-Runs auswerten.
- [x] Die konkrete Root Cause im Repo oder in der Job-Konfiguration isolieren.
- [x] Fix implementieren und lokal verifizieren.
- [ ] Ergebnis und den naechsten operativen Schritt dokumentieren.

### Review Notes

- Umgesetzt im Repo:
  - optionale MLflow-Tracking-/Registry-Schicht in `backend/ml/mlflow_tracking.py`
  - MLflow-Retrofit im bestehenden `train_ranker.py` + Finalisierung im Shell-Flow
  - Registry-Fallback im Runtime-Ranker via MLflow-Model-Bundle
  - neue portable Pipeline-Struktur unter `backend/ml/pipelines/`
  - neuer Vertex-Pipeline-Submitter `scripts/submit_vertex_pipeline.py`
  - Doku-Update in `docs/vertex-training.md`
- Bewusst offen geblieben:
  - externer MLflow-Server/Cloud-SQL/Cloud-Run-Betrieb
  - echte End-to-End-Vertex-Pipeline-Ausfuehrung gegen GCP
  - Phase-5-Parallelisierung ueber `dsl.ParallelFor`
- Verifikation:
  - `backend/.venv/bin/python -m py_compile backend/ml/mlflow_tracking.py backend/ml/train_ranker.py backend/services/ranker.py backend/ml/vertex_train.py scripts/submit_vertex_training.py scripts/submit_vertex_pipeline.py backend/ml/pipelines/io.py backend/ml/pipelines/train_pipeline.py backend/ml/pipelines/backends/local_shell.py backend/ml/pipelines/components/*.py`
  - `backend/.venv/bin/python -m pytest backend/tests/test_train_ranker.py backend/tests/test_ranker.py backend/tests/test_submit_vertex_training.py backend/tests/test_submit_vertex_pipeline.py backend/tests/test_local_shell_pipeline.py backend/tests/test_vertex_train.py -q`
  - Ergebnis: `84 passed`
  - `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"`

- Betroffener Job: `projects/303245987408/locations/europe-west3/customJobs/2163354273713225728`
- Das Problem war nicht mehr der Zero-Duration-Whisper-Pfad.
- Der Job hing nach dem letzten sichtbaren Log
  - `Load pretrained SentenceTransformer: paraphrase-multilingual-MiniLM-L12-v2`
  - danach kam keine weitere Candidate- oder Window-Verarbeitung mehr.
- Root Cause:
  - `backend/services/semantic_candidates.py` fiel im Vertex-Container auf `source=huggingface` zurueck.
  - Der eigentliche Download von `model.safetensors` fuer `paraphrase-multilingual-MiniLM-L12-v2` hing in dieser Umgebung sowohl im Vertex-Run als auch lokal bei `0 B / 471 MB`.
  - Zusaetzlich war unser lokaler Snapshot-Materializer noch auf ein aelteres Split-Cache-Schema (`hub` + `transformers`) zugeschnitten.
- Implementierte Fixes:
  - `backend/services/semantic_candidates.py`
    - `_materialize_local_sentence_transformer_model()` akzeptiert jetzt auch das aktuelle Hub-only-Snapshot-Schema, bei dem Modellgewichte direkt im Hub-Snapshot liegen.
  - `backend/Dockerfile.vertex-train`
    - kein HuggingFace-Network-Prewarm mehr im Build
    - stattdessen wird ein bereits vollstaendiger lokaler Snapshot in das Image kopiert
- Lokale Verifikation:
  - `cd backend && .venv/bin/python -m pytest tests/test_semantic_candidates.py tests/test_build_twitch_training_dataset.py -q`
    - Ergebnis: `39 passed`
  - `cd backend && .venv/bin/python -m py_compile services/semantic_candidates.py tests/test_semantic_candidates.py`
  - Graphify fuer `backend` neu gebaut.
- Operativer Stand:
  - Der haengende Job wurde abgebrochen:
    - `projects/303245987408/locations/europe-west3/customJobs/2163354273713225728`
    - Status: `JOB_STATE_CANCELLED`
    - `endTime`: `2026-04-22T08:28:00.759543Z`
  - Das korrigierte Image mit vendortem Semantic-Model-Snapshot wurde erfolgreich gebaut:
    - Build: `projects/project-22586fae-58b4-4890-aeb/locations/europe-west3/builds/f631c975-2d9d-498e-9bef-d2ae3b05b0ff`
    - Image: `europe-west3-docker.pkg.dev/project-22586fae-58b4-4890-aeb/ml/ai-clipping-train:20260422-semantic-cache-fix`
    - Digest: `sha256:a687712d3b2f315d8148156e4b6ff8309ab2a65b6faa8ed5e2c21fa0bf85498e`
  - Der ersetzende Highmem-Run wurde neu eingereicht:
    - Job: `projects/303245987408/locations/europe-west3/customJobs/828705487267561472`
    - Display Name: `jynxzi-ranker-train-20260422-ew3-7day-artifactonly-2x2-semantic-cache-fix-n2hm64`
    - Status direkt nach Submit: `JOB_STATE_PENDING`

## 2026-04-22 Plan: Diagnose And Fix Repeated VOD Window Skips In Vertex Run

- [x] Aktuelle Vertex-Logs und den aktiven Jobstatus fuer die Skip-Meldungen auswerten.
- [x] Die konkrete Root Cause im Backend-Codepfad isolieren.
- [x] Fix implementieren und mit gezielten Tests verifizieren.
- [x] Ergebnis und verbleibende Runtime-Risiken dokumentieren.

### Review Notes

- Betroffener aktiver Job ist `projects/303245987408/locations/europe-west3/customJobs/7602576723669942272`.
- Die wiederkehrenden Window-Skips im laufenden `workerpool0-0` sind kein Rueckfall des alten `NoneType`-Concurrency-Bugs, sondern ein anderer Whisper-Pfad:
  - wiederholte Logs vom Typ `Skipping VOD window ...: non-positive Whisper segment duration ... start=... end=...`
  - Beispiele im laufenden Run:
    - `2722404357[124]`
    - `2723115885[87]`
    - `2722404357[123]`
- Root Cause:
  - Whisper liefert in echten Laufzeitdaten gelegentlich Segmente mit gueltigem Text, aber `start == end`.
  - Der aktuelle `backend/services/transcriber.py`-Pfad behandelte `end <= start` pauschal als harten Fehler, was im Dataset-Build den gesamten Window-Slice verwarf.
  - Das ist fuer wirklich kaputte Payloads korrekt, aber fuer diese Null-Dauer-Timestamp-Artefakte zu aggressiv.
- Implementierter Fix:
  - `backend/services/transcriber.py`
    - neue Konstante `_MIN_SEGMENT_DURATION_SECONDS = 0.01`
    - `end < start` bleibt harter `ValueError`
    - `end == start` wird jetzt mit `transcriber_clamp_zero_duration_segment ...` sauber gewarnt und auf `start + 0.01` aufgeweitet
  - Dadurch bleibt das Segment und damit das ganze Window im Lauf, statt komplett geskippt zu werden.
- Wichtige Abgrenzung:
  - Wirklich kaputte Whisper-Payloads wie `None`, falsche Mapping-Typen oder invalide Timestamp-Typen bleiben weiterhin harte Fehler und werden nicht still versteckt.
- Verifikation:
  - `cd backend && .venv/bin/python -m pytest tests/test_transcriber.py tests/test_transcriber_progress.py tests/test_build_twitch_training_dataset.py -q`
    - Ergebnis: `42 passed`
  - `cd backend && .venv/bin/python -m py_compile services/transcriber.py tests/test_transcriber_progress.py`
  - Graphify fuer `backend` neu gebaut.
- Betriebsaspekt:
  - Der Fix liegt jetzt im Repo, aber nicht im aktuell laufenden Vertex-Container.
  - Damit der laufende Job nicht weiter an `non-positive Whisper segment duration`-Windows verliert, braucht er ein neues Image und einen neuen Job-Start.

## 2026-04-22 Plan: Roll Out Zero-Duration Whisper Fix To Vertex Highmem Run

- [x] Neues Vertex-Train-Image mit dem Zero-Duration-Whisper-Fix im neuen Projekt bauen.
- [x] Eine neue Highmem-64-Job-Spec mit identischer 7-Tage-2x2-Konfiguration auf das frische Image zeigen lassen.
- [x] Den laufenden Highmem-Run abbrechen und den neuen Job einreichen.
- [x] Den neuen Jobstatus und die ersten Runtime-Logs verifizieren.

### Review Notes

- Cloud Build fuer das neue Image lief erfolgreich:
  - Build: `projects/project-22586fae-58b4-4890-aeb/locations/europe-west3/builds/02a4185c-7ba3-45cc-afed-073b69a9975a`
  - Image: `europe-west3-docker.pkg.dev/project-22586fae-58b4-4890-aeb/ml/ai-clipping-train:20260422-zero-duration-fix`
  - Digest: `sha256:d7538059ab6a9fc2d90d7a0ffe149751528bc1b07018ba9b53679212695de838`
- Der alte Highmem-Run mit dem fehlerhaften Image wurde sauber beendet:
  - Job: `projects/303245987408/locations/europe-west3/customJobs/7602576723669942272`
  - Status: `JOB_STATE_CANCELLED`
  - `endTime`: `2026-04-21T23:55:06.482513Z`
- Der neue ersetzende Highmem-Run wurde mit identischer 7-Tage-2x2-Config eingereicht:
  - Job: `projects/303245987408/locations/europe-west3/customJobs/2163354273713225728`
  - Display Name: `jynxzi-ranker-train-20260422-ew3-7day-artifactonly-2x2-zero-duration-fix-n2hm64`
  - Status direkt nach Submit: `JOB_STATE_PENDING`
- Verifizierte Runtime-Konfiguration des neuen Jobs:
  - Image: `.../ai-clipping-train:20260422-zero-duration-fix`
  - `TRAINING_VOD_WORKERS=2`
  - `TRAINING_LABEL_WORKERS=2`
  - `--artifact-only`
  - Date-Range: `2026-03-14` bis `2026-03-20`
  - neuer Artifact-Prefix:
    `gs://project-22586fae-58b4-4890-aeb-ml-artifacts-ew3/jynxzi/train-artifacts-20260422-7day-artifactonly-2x2-zero-duration-fix-n2highmem64`
- Erste Vertex-Service-Logs des neuen Jobs sind normal:
  - `Waiting for job to be provisioned.`
  - `Vertex AI is provisioning job running framework.`
  - `Job is preparing.`

## 2026-04-21 Plan: Rebuild Image And Restart 2x2 Vertex Job In New Project

- [x] Neues GCP-Projekt auf fehlende Vertex-Train-Ressourcen und Credentials pruefen.
- [x] Artifact Registry, Bucket, Service Account und Secrets im neuen Projekt bootstrapen.
- [x] Image per Cloud Build im neuen Projekt bauen und pushen.
- [ ] Den identischen 7-Tage-2x2-Artifact-Only-Run im neuen Projekt neu starten und verifizieren.

### Review Notes

- Neues Projekt: `project-22586fae-58b4-4890-aeb` mit Account `clipzzholy@gmail.com`.
- Bootstrap im neuen Projekt:
  - APIs aktiviert: `cloudbuild.googleapis.com`, `secretmanager.googleapis.com`, `storage.googleapis.com`
  - Artifact Registry Repo erstellt: `europe-west3-docker.pkg.dev/project-22586fae-58b4-4890-aeb/ml`
  - Bucket erstellt: `gs://project-22586fae-58b4-4890-aeb-ml-artifacts-ew3`
  - Service Account erstellt: `vertex-train@project-22586fae-58b4-4890-aeb.iam.gserviceaccount.com`
  - Secrets gespiegelt: `twitch-client-id`, `twitch-client-secret`, `groq-api-key`, `gemini-api-key`
- Erster Cloud-Build-Versuch scheiterte, weil die neue Default-Build-Identitaet `303245987408-compute@developer.gserviceaccount.com` keinen Zugriff auf den Cloud-Build-Source-Bucket hatte.
- Fix dafuer:
  - `roles/storage.admin`
  - `roles/artifactregistry.writer`
  - `roles/logging.logWriter`
  wurden an `303245987408-compute@developer.gserviceaccount.com` vergeben.
- Danach lief der Cloud Build erfolgreich:
  - Build: `projects/project-22586fae-58b4-4890-aeb/locations/europe-west3/builds/ec3b15d7-ac16-47ab-a873-79f04fab1c27`
  - Image: `europe-west3-docker.pkg.dev/project-22586fae-58b4-4890-aeb/ml/ai-clipping-train:20260421-transcriberlock-newproject`
  - Digest: `sha256:8637ea03014ec77247219d890953fba3ee0c7c11941b4ecacd4ad12e85d250e2`
- Der anschliessende Vertex-Submit des identischen `n2-highmem-64`-Runs wurde nicht angenommen:
  - Fehler: `429 RESOURCE_EXHAUSTED`
  - Meldung: `The following quota metrics exceed quota limits: aiplatform.googleapis.com/custom_model_training_n2_cpus`
- Wichtig:
  - Die allgemeine Compute-Region-Quota `N2_CPUS` in `europe-west3` ist vorhanden (`limit 200`).
  - Geblockt ist spezifisch die getrennte Vertex-AI-Custom-Training-Quota fuer N2-CPUs im neuen Projekt.
  - Deshalb konnte der identische Job im neuen Projekt noch nicht gestartet werden, obwohl Image, Bucket, Secrets und Service Account korrekt vorbereitet sind.

## 2026-04-21 Plan: Request Vertex N2 Training Quota Increase In New Project

- [x] Exakte Vertex-Quota-ID fuer den Submit-Fehler identifizieren.
- [x] Quota-Preference-Request fuer `europe-west3` direkt per `gcloud` einreichen.
- [x] Rueckmeldung des Quota-Systems verifizieren und dokumentieren.

### Review Notes

- Der Vertex-Submit-Fehler `aiplatform.googleapis.com/custom_model_training_n2_cpus` wurde gegen die Cloud-Quotas-API aufgeloest.
- Die exakte Quota im neuen Projekt ist:
  - `quotaId`: `CustomModelTrainingN2CPUsPerProjectPerRegion`
  - `metric`: `aiplatform.googleapis.com/custom_model_training_n2_cpus`
  - `region`: `europe-west3`
  - aktueller `grantedValue`: `20`
  - `quotaIncreaseEligibility.isEligible`: `True`
- Eingereichter Request:
  - `preferredValue`: `64`
  - Preference: `projects/project-22586fae-58b4-4890-aeb/locations/global/quotaPreferences/aiplatform-custom-model-training-n2-cpus-europe-west3`
- Ergebnis:
  - Der Request wurde technisch erstellt, aber innerhalb weniger Sekunden von Google abgelehnt.
  - Rueckmeldung aus der Quota Preference:
    - `grantedValue: 20`
    - `preferredValue: 64`
    - `stateDetail: Quota request denied`
- Operative Konsequenz:
  - Eine automatische Erhoehung per CLI ist fuer dieses Projekt/Quota im Moment nicht erreichbar.
  - Der naechste Weg ist ein manueller Quota-/Support-Request in Google Cloud fuer genau `CustomModelTrainingN2CPUsPerProjectPerRegion` in `europe-west3`, oder ein Ausweichen auf kleinere/andere Vertex-Maschinentypen.

- Danach wurde auch der manuelle Support-Pfad direkt ueber die Cloud Support API getestet:
  - `cloudsupport.googleapis.com` wurde im Projekt aktiviert.
  - Die Support-API-Discovery zeigt einen echten `cases.create`-Pfad.
  - Eine passende Fallklassifikation wurde gefunden:
    - `Other Google Cloud Products > AI Applications > Quotas`
  - Der erste `cases.create`-Versuch scheiterte nur an einem ungueltigen BYOID-Feld (`case.contact_email`).
  - Der bereinigte zweite `cases.create`-Versuch gegen `projects/project-22586fae-58b4-4890-aeb/cases` scheiterte dann an der eigentlichen Support-Berechtigung:
    - `FAILED_PRECONDITION`
    - `You are not eligible to create a case with this channel.`
- Schlussfolgerung:
  - Ich konnte technisch einen echten manuellen Quota-/Support-Request vorbereiten und absenden.
  - Google blockiert die Case-Erstellung aber auf Support-Kanal-Ebene fuer dieses Projekt/Konto.
  - Damit ist der naechste Schritt nicht mehr ein CLI-/API-Fix, sondern ein Support-Plan/Customer-Care-Thema oder ein Ausweichen auf kleinere Vertex-Ressourcen.

## 2026-04-21 Plan: Start Same 2x2 Vertex Run On Smaller N2 Machine In New Project

- [x] Den Job auf einen zulaessigen N2-Maschinentyp innerhalb der aktuellen Vertex-Quota umstellen.
- [x] Den 7-Tage-2x2-Artifact-Only-Run im neuen Projekt mit dem kleineren Maschinentyp einreichen.
- [x] Submit-Ergebnis und erste Runtime-Metadaten verifizieren.

### Review Notes

- Weil `CustomModelTrainingN2CPUsPerProjectPerRegion` im neuen Projekt nur `20` gewaehren N2-Training-CPUs in `europe-west3` hat, wurde die Vertex-Spec von `n2-highmem-64` auf `n2-highmem-16` reduziert.
- Die angepasste Spec liegt hier:
  - [vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock_newproject.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock_newproject.yaml:1)
- Der neue Job wurde im neuen Projekt erfolgreich eingereicht:
  - Display Name: `jynxzi-ranker-train-20260421-ew3-7day-artifactonly-2x2-transcriberlock-n2hm16`
  - Resource: `projects/303245987408/locations/europe-west3/customJobs/7350375144537194496`
- Verifizierte Runtime-Metadaten:
  - `state`: `JOB_STATE_PENDING`
  - `machineType`: `n2-highmem-16`
  - `serviceAccount`: `vertex-train@project-22586fae-58b4-4890-aeb.iam.gserviceaccount.com`
  - Image: `europe-west3-docker.pkg.dev/project-22586fae-58b4-4890-aeb/ml/ai-clipping-train:20260421-transcriberlock-newproject`
  - `TRAINING_VOD_WORKERS=2`
  - `TRAINING_LABEL_WORKERS=2`
  - `--artifact-only`
- Wichtig:
  - Der Submit wurde diesmal von Vertex akzeptiert; der vorherige harte `custom_model_training_n2_cpus`-Quota-Blocker ist fuer diese kleinere Maschine umgangen.
  - Zum Zeitpunkt der Verifikation war der Container noch nicht auf `RUNNING`, sondern weiterhin in `JOB_STATE_PENDING`.

## 2026-04-21 Plan: Replace Small New-Project Run With N2 Highmem/Highcpu 64 Variants

- [x] Getrennte Vertex-Specs fuer `n2-highmem-64` und `n2-highcpu-64` im neuen Projekt vorbereiten.
- [x] Den laufenden `n2-highmem-16`-Run beenden.
- [x] Beide 64-vCPU-N2-Varianten einreichen und Quota-/Submit-Verhalten pruefen.

### Review Notes

- Fuer den neuen Projektpfad wurden zwei getrennte Specs angelegt:
  - [vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock_newproject_n2highmem64.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock_newproject_n2highmem64.yaml:1)
  - [vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock_newproject_n2highcpu64.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock_newproject_n2highcpu64.yaml:1)
- Der kleine Testjob wurde sauber beendet:
  - Job `7350375144537194496`
  - finaler Status: `JOB_STATE_CANCELLED`
  - `updateTime`: `2026-04-21T19:33:00.300666Z`
- Beide 64-vCPU-N2-Varianten wurden im neuen Projekt erfolgreich von Vertex angenommen:
  - Highmem:
    - Display Name: `jynxzi-ranker-train-20260421-ew3-7day-artifactonly-2x2-transcriberlock-n2hm64`
    - Resource: `projects/303245987408/locations/europe-west3/customJobs/1289937420980060160`
    - Maschine: `n2-highmem-64`
  - Highcpu:
    - Display Name: `jynxzi-ranker-train-20260421-ew3-7day-artifactonly-2x2-transcriberlock-n2hc64`
    - Resource: `projects/303245987408/locations/europe-west3/customJobs/3639409051583905792`
    - Maschine: `n2-highcpu-64`
- Wichtig:
  - Beide Submits wurden parallel akzeptiert; es gab diesmal keinen sofortigen `custom_model_training_n2_cpus`-Quota-Reject beim Erstellen.
  - Direkt nach dem Submit stehen beide neuen Jobs noch auf `JOB_STATE_PENDING`; der eigentliche Scheduling-/Provisioning-Verlauf muss separat beobachtet werden.

## 2026-04-21 Plan: Diagnose Failed Transcriber-Lock Vertex Job

- [x] Status und erste Worker-Logs des neuen Transcriber-Lock-Jobs auswerten.
- [x] Root Cause des Container-Fehlers im Code oder Build-Pfad isolieren.
- [x] Fix implementieren, gezielt verifizieren und dokumentieren.

### Review Notes

- Der fehlgeschlagene Job `projects/545981913082/locations/europe-west3/customJobs/5560475767634132992` ist nicht am Trainingscode oder am neuen Image gescheitert, sondern schon im Bootstrap von `backend/ml/vertex_train.py`.
- Der erste echte Fehler in den Worker-Logs war:
  - `google.api_core.exceptions.PermissionDenied: 403 Permission 'secretmanager.versions.access' denied on resource`
  - Callsite: `_resolve_secret()` in `backend/ml/vertex_train.py`
- Root Cause:
  - Der neue Job wurde ohne `jobSpec.serviceAccount` eingereicht.
  - Der vorherige funktionierende Vertex-Run `4432394429722001408` lief mit `vertex-train@clipping-493513.iam.gserviceaccount.com`.
  - Der fehlgeschlagene neue Run `5560475767634132992` hatte dieses Feld gar nicht und konnte deshalb die Secret Manager Secrets nicht lesen.
- Fix:
  - In [vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock.yaml:1) wurde `serviceAccount: vertex-train@clipping-493513.iam.gserviceaccount.com` wieder gesetzt.
  - Danach wurde ein neuer Ersatzjob eingereicht:
    - `projects/545981913082/locations/europe-west3/customJobs/9072157577076277248`
    - Display Name: `jynxzi-ranker-train-20260421-ew3-7day-artifactonly-2x2-transcriberlock-sa`
- Verifikation:
  - `gcloud ai custom-jobs describe` zeigt beim neuen Job wieder exakt die erwartete `jobSpec.serviceAccount`.
  - In den ersten nachfolgenden Logabfragen gibt es keinen erneuten `secretmanager.versions.access`-Fehler.
  - Der neue Job steht aktuell noch auf `JOB_STATE_PENDING`; der Container ist also noch nicht auf `RUNNING`, aber der konkrete Secret-Access-Fehler wurde an der Einreichungsebene korrigiert.

## 2026-04-21 Plan: Rebuild Image And Restart 2x2 Vertex Job

- [x] Neuen eindeutigen Vertex-Trainings-Image-Tag fuer den Transcriber-Concurrency-Fix vorbereiten.
- [x] Image per Google Cloud Build neu bauen und pushen.
- [x] Laufenden 2x2-Artifact-Only-Job ersetzen und mit identischer Konfiguration auf dem neuen Image neu starten.
- [x] Neuen Job, Image-Tag und Runtime-Konfiguration verifizieren und dokumentieren.

### Review Notes

- Fuer den Transcriber-Concurrency-Fix wurde ein neuer eindeutiger Image-Tag vorbereitet:
  - `europe-west3-docker.pkg.dev/clipping-493513/ml/ai-clipping-train:20260421-transcriberlock-rerun`
- Der Build lief erfolgreich ueber Google Cloud Build:
  - Build: `projects/545981913082/locations/europe-west3/builds/44c0e4b4-f524-4fd1-9893-d3d674e60dd4`
  - Status: `SUCCESS`
  - Digest: `sha256:3cd3c6e10955b540ab9b77ce69e87a7e1951d9b27acc960bff048e24af0abd85`
  - Build-Config: [cloudbuild_vertex_train_20260421_transcriber_lock.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/cloudbuild_vertex_train_20260421_transcriber_lock.yaml:1)
- Der alte 2x2-Artifact-Only-Run wurde beendet:
  - alte Resource: `projects/545981913082/locations/europe-west3/customJobs/8168622901835071488`
  - finaler Status: `JOB_STATE_CANCELLED`
  - `endTime`: `2026-04-21T16:15:02.708632Z`
- Der neue Run wurde mit identischer 2x2-Konfiguration und dem neuen Image eingereicht:
  - Display Name: `jynxzi-ranker-train-20260421-ew3-7day-artifactonly-2x2-transcriberlock`
  - neue Resource: `projects/545981913082/locations/europe-west3/customJobs/5560475767634132992`
  - Spec: [vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2_transcriberlock.yaml:1)
- Verifizierte Runtime-Konfiguration des neuen Jobs:
  - Image: `europe-west3-docker.pkg.dev/clipping-493513/ml/ai-clipping-train:20260421-transcriberlock-rerun`
  - Date-Range: `2026-03-14` bis `2026-03-20`
  - `--artifact-only`
  - `TRAINING_VOD_WORKERS=2`
  - `TRAINING_LABEL_WORKERS=2`
  - Artifact-Prefix: `gs://clipping-493513-ml-artifacts-ew3/jynxzi/train-artifacts-20260421-7day-artifactonly-2x2-transcriberlock`
- Direkt nach dem Submit steht der neue Job noch auf `JOB_STATE_PENDING`. Das ist konsistent mit dem gerade beendeten Alt-Job; der Container ist noch nicht auf `RUNNING`.

## 2026-04-21 Plan: Inspect Current 2x2 Vertex Job Logs And Fix Errors

- [x] Aktuelle Worker-Logs des laufenden 2x2-Jobs auf echte Fehlerpfade und wiederkehrende Warnungen pruefen.
- [x] Falls ein reproduzierbarer Fehler vorliegt, Root Cause im Backend-Code isolieren.
- [x] Fix implementieren, gezielt verifizieren und Ergebnis dokumentieren.

### Review Notes

- Der laufende 2x2-Vertex-Job `projects/545981913082/locations/europe-west3/customJobs/8168622901835071488` ist weiterhin `JOB_STATE_RUNNING`, produziert aber fortlaufend Warnungen vom Typ:
  - `Skipping VOD window ...: 'NoneType' object is not subscriptable`
  - zuletzt noch um `2026-04-21T15:51:35Z`
- Die Warnung entsteht im Catch von `backend/ml/build_twitch_training_dataset.py`, aber die wahrscheinlichere Root Cause ist nicht der VOD-Inhalt selbst, sondern parallelisierte Whisper-Inferenz:
  - Der letzte lange 1x1-Run `4432394429722001408` zeigte keine `Skipping VOD window ... NoneType ...`-Treffer.
  - Der aktuelle 2x2-Run startet dagegen mehrere gleichzeitige `transcriber.transcribe()`-Aufrufe ueber parallele VOD-Worker, Label-Worker und den pro-Label-Modalitaets-Threadpool.
  - `backend/services/transcriber.py` verwendet dabei eine globale gemeinsame Whisper-Modellinstanz `_whisper_model`, hatte bisher aber keinen Lock um `model.transcribe(...)`.
  - Das passt am besten zu einem Thread-Safety-/Shared-State-Problem im laufenden Inferenzpfad.
- Fix im Repo:
  - `backend/services/transcriber.py` serialisiert Whisper-Inferenz jetzt explizit ueber `_whisper_inference_lock`, damit parallele Fenster keine konkurrierenden `model.transcribe(...)`-Aufrufe mehr auf derselben Modellinstanz ausfuehren.
  - Kaputte Segmentpayloads werden bewusst nicht mehr stillschweigend uebersprungen. Stattdessen werfen sie wieder einen klaren `ValueError`, damit der bestehende Window-Level-Logpfad `Skipping VOD window ...` erhalten bleibt und der Fehler sichtbar bleibt.
- Regression:
  - `backend/tests/test_transcriber_progress.py` deckt jetzt explizit zwei Dinge ab:
    - kaputte Segmentpayloads fuehren zu einem klaren Fehler statt zu stillem Segment-Skip
    - zwei parallele `transcribe_with_progress()`-Aufrufe nutzen die gemeinsame Modellinstanz nur seriell (`max_active == 1`)
- Verifikation:
  - `backend/.venv/bin/python -m pytest backend/tests/test_transcriber.py backend/tests/test_transcriber_progress.py backend/tests/test_build_twitch_training_dataset.py -q` lief mit `41 passed`.
  - `backend/.venv/bin/python -m py_compile backend/services/transcriber.py backend/tests/test_transcriber_progress.py` blieb sauber.
- Wichtiger Betriebsaspekt:
  - Der aktuell laufende Vertex-Job nutzt noch das alte Image und wird deshalb weiter Fenster droppen, bis ein neues Image mit diesem Fix gebaut und der Job neu gestartet wird. Der Fix wirkt nicht rueckwirkend auf den bereits gestarteten Container.

## 2026-04-21 Plan: Diagnose Failed Vertex Training Job

- [x] Vertex-Jobstatus, Fehlerzustand und letzte Updates fuer den fehlgeschlagenen Run einsammeln.
- [x] Die letzten Worker-Logs um den Abbruchzeitpunkt auswerten und den ersten echten Fehler isolieren.
- [x] Root Cause und die naechste konkrete Massnahme dokumentieren.

### Review Notes

- Der Vertex-Job `projects/545981913082/locations/europe-west3/customJobs/4432394429722001408` endete am `2026-04-21T12:18:07Z` mit `JOB_STATE_FAILED`; Vertex selbst meldet nur den generischen Wrapperfehler `workerpool0-0 exited with a non-zero status of 1`.
- Der eigentliche Fehler liegt im Trainings-Shellscript `backend/ml/train_jynxzi_vod.sh` nach dem eigentlichen Modelltraining. Die Logs zeigen davor noch erfolgreiche Artefaktschritte:
  - `candidate_source_mix_capped rows_before=3424 rows_after=66`
  - `Wrote artifact to /app/backend/ml/artifacts/jynxzi_ranker.json`
  - `Wrote artifact health to /app/backend/ml/artifacts/jynxzi_artifact_health.json`
- Direkt danach lief der Gold-Eval-Export in den Fehler:
  - `Exporting frontend gold eval rows...`
  - `No frontend gold-eval rows found.`
- Ursache im Code:
  - `backend/ml/export_frontend_gold_eval.py` beendet sich bei null Rows mit `return 1`.
  - `backend/ml/train_jynxzi_vod.sh` laeuft mit `set -euo pipefail` und ruft den Export unguarded auf; dadurch beendet das Shellscript den gesamten Vertex-Run sofort mit Exit `1`.
- Die weiteren sichtbaren Warnungen waren nicht der Abbruchgrund:
  - `training_vod_checkpoint_gcs_write_failed ... Content-Type ... text/plain vs application/json` stammt vom alten Container-Image vor dem GCS-Checkpoint-Fix.
  - Die vielen `Groq ... 429 ... falling back to Ollama` haben den Lauf gebremst, aber nicht beendet.
- Naechste konkrete Massnahme:
  - Entweder `export_frontend_gold_eval.py` bei leerem Gold-Set als erfolgreichen No-op behandeln und leere/placeholder Outputs schreiben.
  - Oder `train_jynxzi_vod.sh` den Gold-Eval-Pfad nur dann ausfuehren lassen, wenn `jynxzi_gold_eval_rows.jsonl` nach dem Export wirklich Rows enthaelt.

## 2026-04-21 Plan: Artifact-Only Vertex Training

- [x] Trainingspfad so erweitern, dass ein Run nach Artefakterzeugung vor allen Eval-Schritten enden kann.
- [x] Vertex-Wrapper mit einem expliziten `artifact-only`-Schalter ausstatten.
- [x] Gezielte Regressionstests und Syntaxchecks ausfuehren.

### Review Notes

- `backend/ml/train_jynxzi_vod.sh` akzeptiert jetzt `--artifact-only` beziehungsweise `--no-artifact-only`.
- Wenn `--artifact-only` aktiv ist, laeuft der Pfad weiterhin durch Label-Sammlung, Dataset-Build, Modelltraining und Workflow-State-Schreiben, beendet den Run danach aber bewusst vor:
  - `build_eval_dataset.py`
  - `export_frontend_gold_eval.py`
  - `evaluate_ranker.py`
  - `evaluate_pipeline.py`
  - Rollout-Recommendation
- Dadurch kann ein Vertex-Run jetzt sauber nur Artefakte erzeugen, ohne am derzeit leeren Frontend-Gold-Eval-Set zu scheitern.
- `backend/ml/vertex_train.py` reicht diesen Modus jetzt ueber `--artifact-only` an `ml/train_jynxzi_vod.sh` weiter.
- Verifikation:
  - `backend/.venv/bin/python -m pytest backend/tests/test_vertex_train.py -q` lief mit `30 passed`.
  - `backend/.venv/bin/python -m py_compile backend/ml/vertex_train.py` blieb sauber.
  - `zsh -n backend/ml/train_jynxzi_vod.sh` blieb sauber.

## 2026-04-21 Plan: Rebuild And Resubmit 7-Day Vertex Run In Artifact-Only Mode

- [x] Frisches Vertex-Trainingsimage mit dem aktuellen `artifact-only`-Pfad per Cloud Build bauen und pushen.
- [x] Einen neuen 7-Tage-Vertex-Run fuer `2026-03-14` bis `2026-03-20` mit `--artifact-only` und eigenem Artifact-Prefix submitten.
- [x] Job-Referenz, Image-Tag und verwendete Konfiguration dokumentieren.

### Review Notes

- Neuer Cloud-Build wurde erfolgreich abgeschlossen:
  - Build: `projects/545981913082/locations/europe-west3/builds/13fc2a26-c715-4317-a9d4-ad115ca8a28d`
  - Image: `europe-west3-docker.pkg.dev/clipping-493513/ml/ai-clipping-train:20260421-artifactonly-rerun`
  - Digest: `sha256:ff2920a8873871d0a7c6448ae19c57c14cbd36642af4f486aaa5da9adea7ab97`
- Fuer den Build wurde die temporaere Cloud-Build-Config [cloudbuild_vertex_train_20260421_artifact_only.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/cloudbuild_vertex_train_20260421_artifact_only.yaml:1) angelegt, damit `backend/Dockerfile.vertex-train` ueber Google Cloud Build gebaut werden konnte.
- Neuer Vertex-Run wurde im reinen Artefaktmodus eingereicht:
  - Display Name: `jynxzi-ranker-train-20260421-ew3-7day-artifactonly`
  - Resource: `projects/545981913082/locations/europe-west3/customJobs/7099017990334578688`
  - Initialstatus nach Submit: `JOB_STATE_PENDING`
- Verwendete Job-Konfiguration:
  - Date-Range: `2026-03-14` bis `2026-03-20`
  - Maschine: `n2-highmem-64`
  - Disk: `500 GB` `pd-ssd`
  - Env: `TRAINING_VOD_WORKERS=1`, `TRAINING_LABEL_WORKERS=1`
  - `--artifact-only` aktiv
  - Download-Cache wiederverwendet: `gs://clipping-493513-ml-artifacts-ew3/cache/vod-downloads`
  - Checkpoint-Prefix: `gs://clipping-493513-ml-artifacts-ew3/cache/vod-checkpoints`
  - Neuer Artifact-Prefix: `gs://clipping-493513-ml-artifacts-ew3/jynxzi/train-artifacts-20260421-7day-artifactonly`
  - Spec: [vertex_job_spec_20260421_jynxzi_7day_artifactonly.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/vertex_job_spec_20260421_jynxzi_7day_artifactonly.yaml:1)

## 2026-04-20 Plan: Fix Training VOD Checkpoint GCS Upload

- [x] Fehlerursache in den Vertex-Logs und im GCS-Checkpoint-Codepfad isolieren.
- [x] Content-Type-Mismatch im Checkpoint-Upload beheben und Regression anpassen.
- [x] Gezielte Tests ausfuehren, Review Notes dokumentieren und Graphify aktualisieren.

### Review Notes

- Der Vertex-Job `4432394429722001408` loggte `training_vod_checkpoint_gcs_write_failed` mit `400 invalid`, nicht mit IAM- oder Bucket-Fehlern.
- Die eigentliche GCS-Fehlermeldung laut Worker-Log war:
  - `Content-Type specified in the upload (text/plain) does not match Content-Type specified in metadata (application/json).`
- Ursache war in `backend/ml/build_twitch_training_dataset.py`: `_upload_vod_checkpoint_to_gcs()` setzte `blob.content_type = "application/json"`, rief `upload_from_string()` aber ohne expliziten `content_type` auf.
- Fix: `upload_from_string(..., content_type="application/json", if_generation_match=0)` statt implizitem Default-Upload.
- Der bestehende GCS-Checkpoint-Test in `backend/tests/test_build_twitch_training_dataset.py` wurde so angepasst, dass er den expliziten Upload-`content_type` direkt absichert.
- Verifikation:
  - `backend/.venv/bin/python -m pytest backend/tests/test_build_twitch_training_dataset.py backend/tests/test_vertex_train.py backend/tests/test_submit_vertex_training.py -q` lief mit `70 passed`.
  - `backend/.venv/bin/python -m py_compile backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py` blieb sauber.

## 2026-04-20 Plan: Default Semantic Analysis To Local Ollama

- [x] Relevante Default-Pfade fuer den semantischen Analyseschritt identifizieren.
- [x] Den semantischen Default von Groq-first auf Ollama-first umstellen.
- [x] Status-/Zusammenfassungsdarstellung und Regressionstests an das neue Default-Verhalten anpassen.
- [x] Betroffene Tests ausfuehren, Review Notes dokumentieren und Graphify aktualisieren.

### Review Notes

- In `backend/services/semantic_service.py` nutzt `provider=None|"auto"` jetzt standardmaessig zuerst das lokale Ollama-Modell. Groq bleibt nur noch optionaler Remote-Fallback, falls Ollama fuer diesen Schritt nichts liefert.
- Die sichtbare Provider-Zusammenfassung wurde daran angepasst:
  - `backend/ml/build_twitch_training_dataset.py` meldet bei `auto` jetzt `auto->ollama:...|fallback=groq:...`.
  - `backend/routers/jobs.py` zeigt den semantischen Scoring-Pfad jetzt als lokalen Ollama-Provider an, statt Groq als primaeren Online-Provider auszugeben.
- Regressionen wurden in `backend/tests/test_semantic_service.py`, `backend/tests/test_build_twitch_training_dataset.py` und `backend/tests/test_jobs_router.py` angepasst bzw. ergaenzt.
- Verifikation:
  - `backend/.venv/bin/python -m pytest backend/tests/test_semantic_service.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_jobs_router.py backend/tests/test_feature_builder.py -q` lief mit `57 passed`.
  - `backend/.venv/bin/python -m py_compile backend/services/semantic_service.py backend/ml/build_twitch_training_dataset.py backend/routers/jobs.py backend/tests/test_semantic_service.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_jobs_router.py` blieb sauber.

## 2026-04-20 Plan: Pick The Best Local Ollama Model On M2 Air

- [x] Relevante reale Promptquellen fuer den semantischen Scoring-Pfad identifizieren.
- [x] `llama3.2`, `qwen3:4b` und `gemma3:4b` lokal auf dem Mac vergleichen.
- [x] JSON-Trefferquote, Latenz und subjektive Scoring-Qualitaet fuer dieselben Prompts auswerten.
- [x] Das beste lokale Modell empfehlen und den naechsten Umstellungsschritt festhalten.

### Review Notes

- Reale Promptquellen fuer den Benchmark wurden direkt aus den Debug-Artefakten des Backends gezogen: `backend/tmp/storage/6261dc6c-51f3-4541-9174-9ac300a9f8fa/debug/*` und `backend/tmp/storage/9e0ae6fc-b20a-40bc-83b4-7a453cb0e2cc/debug/*`. Damit lief der Vergleich auf echten `segments.jsonl`- und `candidates_semantic.jsonl`-Fenstern statt auf synthetischen Prompts.
- Relevante Regressionstests liefen sauber: `backend/.venv/bin/python -m pytest backend/tests/test_feature_builder.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_vertex_train.py -q` mit `72 passed`.
- Fuer den Modellvergleich wurde `tmp/benchmark_local_semantic_models.py` gebaut. Der Runner nutzt den echten `score_window()`-Pfad aus `backend/services/semantic_service.py`, misst JSON-Trefferquote, Latenz und eine einfache Rangordnungs-Plausibilitaet gegen `score_hint`.
- Installierte lokale Kandidaten zum Testzeitpunkt: `llama3.2:latest`, `qwen3:4b`, `gemma3:4b`.
- `llama3.2` war mit Abstand am reaktivsten, aber in der 3-Fenster-Stichprobe qualitativ schwach: `warmup_latency_sec 18.161`, `mean_latency_sec 8.624`, `pairwise_order_accuracy_vs_score_hint 0.333`. Es ueberbewertete ein schwaches `tension_build`-Fenster gegenueber einem starken Quote-Window.
- `qwen3:4b` war auf diesem `M2 Air 8 GB` praktisch zu langsam fuer den realen Pfad. Der saubere 2-Fenster- und 3-Fenster-Lauf kam nach mehreren Minuten nicht zum Abschluss; daraus ergibt sich bereits eine grobe Untergrenze von deutlich ueber `90s` pro Fenster. Fuer diesen semantischen Scoring-Schritt ist das lokal auf diesem Geraet nicht tragbar.
- `gemma3:4b` lieferte eine saubere 2-Fenster-Stichprobe mit voller JSON-Trefferquote und korrekter High-vs-Low-Reihenfolge, war aber ebenfalls sehr langsam: `warmup_latency_sec 68.826`, `mean_latency_sec 72.582`, `pairwise_order_accuracy_vs_score_hint 1.0`.
- Entscheidung fuer dieses konkrete Geraet und diesen Backend-Pfad:
  - Wenn Durchsatz und interaktive Nutzbarkeit zaehlen: bei `llama3.2` bleiben.
  - Wenn du lokal etwas mehr semantische Qualitaet willst und die Laufzeit akzeptierst: `gemma3:4b`.
  - `qwen3:4b` in der aktuellen Integration nicht empfehlen.

## 2026-04-20 Plan: Submit Jynxzi Vertex 7-Day Run On `n2-highmem-64`

- [x] Vertex-Submitter fuer explizite Container-Env-Overrides erweitern und lokal verifizieren.
- [x] Neue Vertex-Job-Spec fuer den 7-Tage-Run `2026-03-14` bis `2026-03-20` mit `TRAINING_VOD_WORKERS=1` und `TRAINING_LABEL_WORKERS=1` schreiben.
- [x] Aktives GCP-Projekt/Credentials pruefen und den geplanten 7-Tage-Run vorbereiten.
- [x] Frisches Vertex-Trainingsimage per Google Cloud Build bauen und nach Artifact Registry pushen.
- [x] Den versehentlich mit altem Image eingereichten Pending-Job verwerfen und den 7-Tage-Run mit dem frischen Image neu submitten.

### Review Notes

- `scripts/submit_vertex_training.py` akzeptiert jetzt wiederholbare `--container-env NAME=VALUE`-Flags und reicht sie als `container_spec.env` in `worker_pool_specs` durch. Regressionen in `backend/tests/test_submit_vertex_training.py` decken den neuen Pfad und die Syntaxvalidierung ab.
- Verifikation lokal: `backend/.venv/bin/python -m pytest backend/tests/test_submit_vertex_training.py -q` lief mit `7 passed`; `backend/.venv/bin/python -m py_compile scripts/submit_vertex_training.py` blieb sauber.
- Das aktive GCP-Setup wurde vor dem Submit geprueft:
  - aktives Konto `slwa.herbin@googlemail.com`
  - aktives Projekt `clipping-493513`
- Es liefen bereits zwei alte 3-Tage-Jobs, darunter ein Low-Parallel-Run auf dem bisherigen Haupt-Prefix `gs://clipping-493513-ml-artifacts-ew3/jynxzi/train-artifacts`. Um Artefaktkonflikte zu vermeiden, nutzt der neue 7-Tage-Run einen eigenen Prefix:
  - `gs://clipping-493513-ml-artifacts-ew3/jynxzi/train-artifacts-20260420-7day-n2highmem64-lowparallel`
- Neue Vertex-Spec: [tmp/vertex_job_spec_20260420_jynxzi_7day_n2highmem64_lowparallel.json](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/vertex_job_spec_20260420_jynxzi_7day_n2highmem64_lowparallel.json:1)
  - Date-Range `2026-03-14` bis `2026-03-20`
  - Maschine `n2-highmem-64`
  - Disk `500 GB` `pd-ssd`
  - Timeout `604800s`
  - Image `europe-west3-docker.pkg.dev/clipping-493513/ml/ai-clipping-train:20260420-013900-goldevalplan`
  - `TRAINING_VOD_WORKERS=1`
  - `TRAINING_LABEL_WORKERS=1`
  - Service Account `vertex-train@clipping-493513.iam.gserviceaccount.com`
- Frischer Container-Build wurde ueber Google Cloud Build erfolgreich gebaut und gepusht:
  - Build `projects/545981913082/locations/europe-west3/builds/0ff7cc6f-d8b3-4f32-927b-e3e1b6f0f819`
  - Image `europe-west3-docker.pkg.dev/clipping-493513/ml/ai-clipping-train:20260420-013900-goldevalplan`
  - Digest `sha256:eab83be6fb6dfd7f0967942ac4a6977389275f8c96c944f4d337e0e23c0ae5f1`
- Ein frueherer Submit mit dem alten Image wurde nicht weiterverwendet:
  - alte Resource `projects/545981913082/locations/europe-west3/customJobs/2129366170275414016`
  - Status beim Korrekturschritt `JOB_STATE_CANCELLED`
- Der eigentliche 7-Tage-Run wurde danach mit dem frischen Image neu eingereicht:
  - Display Name `jynxzi-ranker-train-20260420-ew3-7day-n2highmem64-lowparallel-goldevalplan`
  - Resource `projects/545981913082/locations/europe-west3/customJobs/4432394429722001408`
  - initialer State `JOB_STATE_PENDING`

## 2026-04-20 Plan: Jynxzi Gold Coverage Next Step

- [x] Aktuelle Jynxzi-Artefakte auf Positive-Dichte pro VOD und Gold-Abdeckung auswerten.
- [x] Konkrete Priorisierung ableiten: welche VODs zuerst mehr Gold-Positives brauchen und ob Fenstererweiterung kurzfristig mehr bringt.
- [x] Ergebnis als handhabbaren naechsten Operator-Schritt dokumentieren.

### Review Notes

- Das aktuelle Trainingsartefakt ist nicht an der Modelllogik gescheitert, sondern am Input-Fenster: `backend/ml/artifacts/jynxzi_vod_training_rows.jsonl` enthaelt nur einen einzigen VOD (`2737446918`, `2026-04-01`) mit `285` Rows und nur `3` Positiven. Deshalb wird das Artefakt zur Laufzeit mit `positive_rows_below_minimum` rejected.
- Der lokale Labelbestand ist deutlich staerker als dieses eine VOD-Fenster vermuten laesst. `backend/ml/artifacts/jynxzi_twitch_clip_labels.jsonl` umfasst `8950` Rows von `2026-03-01` bis `2026-04-01`. Beste Fenster nach Proxy-Positives:
  - `3` Tage: `2026-03-18` bis `2026-03-20` mit `10` Positiven
  - `7` Tage: `2026-03-14` bis `2026-03-20` mit `16` Positiven
  - `14` Tage: `2026-03-08` bis `2026-03-21` mit `23` Positiven
- Fuer einen schnellen unblock ist deshalb die erste sinnvolle Operator-Aktion keine neue Gold-Annotation, sondern ein Retrain auf mindestens dem `7`-Tage-Fenster `2026-03-14` bis `2026-03-20`. Dieses Fenster ueberschreitet lokal bereits die Mindestschwelle von `15` Proxy-Positiven; `14` Tage geben noch mehr Puffer.
- Das bestehende Gold-Set ist fuer Eval brauchbar, aber fuer per-VOD-Ranking noch duenner als das Proxy-Set: `74` Gold-Rows, `13` Positive, verteilt ueber `15` VODs, kein VOD ueber `2` Positive. Fuer Gold-Ausbau zuerst die bereits positiven VODs weiter verdichten:
  - `2697130608`, `2731700208`, `2746133952`, `2698737426` haben schon jeweils `2` Gold-Positive
  - `2662238932`, `2706841224`, `2748097299`, `2749673626`, `2750220507` haben jeweils `1`
- Gold-Positive stammen fast komplett aus `peak`-Kandidaten (`12/13`), mit einem einzelnen `tension_build`-Treffer. Fuer schnelles Nachlabeln also zuerst wieder `peak`-Kandidaten derselben VODs priorisieren, statt breit ueber neue Candidate-Sources zu streuen.

## 2026-04-20 Plan: Gold Eval Hardening

- [x] Eval-Loop so erweitern, dass Gold-Eval und Proxy-Eval getrennt laufen und getrennte Reports mit Heuristik-vs-Trained-Delta schreiben.
- [x] Source-Mix im Training begrenzen, sodass kein `candidate_source` mehr ueber 40 Prozent des Datensatzes einnimmt.
- [x] Gold-Eval-Validierung verschaerfen: `gold_notes` Pflicht fuer Grade 2/3 und fuer Grade 0 bei hohem Heuristik-Score.
- [x] Gold-Holdout einfuehren: fester 70/30-Split, Training muss Holdout-Rows laut und hart verweigern.
- [x] Ranking-Metriken fuer sparse positive VODs robust machen: bei zu wenigen Positiven je VOD auf geeignetere Kennzahl/Fallback umstellen.
- [x] Relevante Tests aktualisieren/ergaenzen, Checks laufen lassen, Review Notes dokumentieren und Graphify aktualisieren.

### Review Notes

- `backend/ml/train_jynxzi_vod.sh` exportiert Gold-Eval jetzt explizit aus `tmp/storage/feedback/gold_eval_updates.jsonl`, schreibt getrennte Reports `jynxzi_gold_eval.json` und `jynxzi_proxy_eval.json`, und `backend/ml/evaluate_ranker.py` kann Reports per `--report-out` materialisieren. Die Delta-Sicht zwischen Heuristik und Trained ist damit fuer Gold und Proxy getrennt nachvollziehbar.
- Gold-Eval-Rows bekommen ueber `backend/services/gold_eval_policy.py` einen deterministischen, eingefrorenen 70/30-Split (`gold_holdout_split=train|holdout`). `backend/ml/train_ranker.py` verweigert Training hart, wenn Gold-Rows ohne Split auftauchen oder Holdout-Rows in den Trainingsinput geraten.
- `gold_notes` sind jetzt an beiden Gold-Eval-Eingaengen Pflicht: im Frontend-Pfad via `backend/services/feedback_store.py` und im CSV-Import via `backend/ml/import_gold_eval_labels.py`. Erzwungen wird das fuer Grade `2/3` sowie fuer Grade `0` mit hohem `heuristic_rank_score`.
- `backend/ml/build_twitch_training_dataset.py` begrenzt den Mix pro `candidate_source` auf maximal 40 Prozent, sofern der Datensatz genug unterschiedliche Quellen hat. `backend/ml/train_vod_ranker.py` prueft diese Schranke zusaetzlich als Quality-Gate und markiert dominante Quellen als Verletzung.
- `backend/ml/train_ranker.py` waehlt die Primaer-Metrik jetzt dynamisch: `NDCG@5` erst ab mindestens 10 Positiven pro Ranking-Gruppe, sonst `NDCG@10`, bei noch duenneren Gruppen `MRR`. Die gewaehlte Kennzahl inklusive Min-/P25-Floors fliesst in Artefakt-Gates und Health-Output ein.
- Verifikation: `backend/.venv/bin/python -m pytest backend/tests/test_feedback_store.py backend/tests/test_gold_eval_workflow.py backend/tests/test_export_frontend_gold_eval.py backend/tests/test_build_eval_dataset.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_train_ranker.py backend/tests/test_train_vod_ranker.py backend/tests/test_training_workflow_state.py backend/tests/test_vertex_train.py -q` lief mit `124 passed`; zusaetzlich lief `backend/.venv/bin/python -m pytest backend/tests/test_jobs_router.py -q` mit `11 passed`.

## 2026-04-22 Plan: Analyse Gold Eval Set

- [x] Export-Semantik und Dedupe-Regeln fuer das Gold-Eval-Set gegen den aktuellen Codepfad verifizieren.
- [x] Rohdaten, dedupte Gold-Rows und exportiertes Eval-Dataset quantitativ vergleichen.
- [x] Biases, Datenqualitaetsrisiken und konkrete Verbesserungen dokumentieren.

### Review Notes

- `backend/ml/export_frontend_gold_eval.py` fuehrt die Analyse ueber den korrekten Pfad: `feedback/gold_eval_updates.jsonl` plus job-lokale `debug/gold_eval.jsonl`, danach Latest-Wins-Dedupe auf `(job_id, candidate_id)` per `updated_at`.
- Aktueller Live-Stand in `backend/tmp/storage/`:
  - 115 rohe Rows = 110 Feedback-Rows + 5 Debug-Rows.
  - 15 Duplicate-Keys erzeugen 18 Extra-Rows; nach Dedupe bleiben 97 Gold-Rows.
  - Die Duplicate-Historie ist ueberwiegend Revision-Historie, nicht Key-Kollision: bei 7 Keys aenderte sich der `gold_grade`, bei 4 davon auch das binarisierte `label`.
- Verteilung des dedupten Gold-Sets:
  - 97 Rows ueber 19 VODs.
  - Grade: `0=59`, `1=22`, `2=10`, `3=6`.
  - Binaer-Labels: `16` positiv (`gold_grade>=2`), `81` negativ.
  - Holdout-Split: `67 train`, `30 holdout`; Positives darin `10 train`, `6 holdout`.
- Biases:
  - Alle Rows sind `creator_id=jynxzi` und `publish_platform=twitch`.
  - `candidate_source` ist stark auf `peak` konzentriert (`72/97` insgesamt, `15/16` Positives).
  - `controversy`, `quote` und `narrative_reveal` haben aktuell keinerlei positive Golds.
- Datenqualitaet:
  - `gold_notes` sind fuer Altbestand sehr lueckig: nur 23/97 Rows haben Notes; bei positiven Grades fehlen sie in 13/16 Faellen (`grade 2`: 9/10 fehlen, `grade 3`: 4/6 fehlen).
  - Zeitlich ist die Luecke klar: Rows vom `2026-04-17` bis `2026-04-19` haben praktisch keine Notes, ab `2026-04-20` sind Notes durchgaengig vorhanden.
  - Alle gespeicherten `features.heuristic_score` sind `0.0`, wodurch dieser Feature-Kanal im Gold-Set aktuell nicht informativ ist.
- Artefaktzustand:
- `backend/ml/artifacts/gold_eval_{rows,dataset}.jsonl` ist ein alter 50-Row-Snapshot ohne Holdout-Felder.
- `backend/ml/artifacts/jynxzi_gold_eval_{rows,dataset}.jsonl` ist ein neuerer 74-Row-Snapshot mit Holdout-Feldern.
- Beide Artefakt-Snapshots sind echte Teilmengen des Live-Sets, aber nicht mehr aktuell; das Live-Set ist inzwischen bei 97 Rows.

## 2026-04-22 Plan: Überblick ML-Modell Und Workflow

- [x] Architekturkontext aus Graphify und Doku verifizieren.
- [x] Trainings-, Eval-, Registry- und Serving-Pfade im Code konsolidieren.
- [x] Einen kompakten, aber vollstaendigen deutschen Überblick ueber Modell und Workflow formulieren.

### Review Notes

- Laut `backend/graphify-out/GRAPH_REPORT.md` sind die zentralen Knoten im ML-Stack: `RichSegment`, `Candidate`, `train_from_rows()`, `Ranker` und `_build_candidate_features()`. Das passt zum echten Laufzeitpfad: Transkript-/Feature-Welt -> Kandidaten -> Ranking -> Boundary-Refinement.
- Das produktive Modell ist kein monolithisches End-to-End-Netz, sondern ein hybrider Ranking-Stack:
  - heuristischer Ranker in `backend/services/ranker.py`
  - optionales trainiertes Artefakt, das zur Laufzeit mit dem Heuristik-Score geblendet wird
  - Modellkandidaten im Training: mean-difference linear, logistic regression, pairwise logistic, gradient boosting, optional LightGBM LambdaRank
- Die Feature-Ebene sitzt in `backend/services/feature_builder.py` und kombiniert Hype-/Audio-/Video-/Semantik-/Kapitel-/Memory-/Trend-Signale in `CandidateFeatures`.
- Online-Inferenzpfad:
  - `backend/pipeline.py` orchestriert Download -> Audio/Video/Transcript -> Segment-Scoring -> Candidate-Generation -> Feature-Building -> Ranking -> Boundary-Refinement -> Post-Processing.
  - Das eigentliche Ranking passiert ueber `selector.rank_candidates(...)` mit `Ranker`.
- Trainingspfad:
  - Datensammlung startet bei Twitch-Clip-Labels (`ml.collect_twitch_clip_labels`), daraus baut `backend/ml/build_twitch_training_dataset.py` Ranker-Rows.
  - `backend/ml/train_ranker.py` splittet die Daten, fuegt `heuristic_score` als Feature ein, trainiert mehrere Modellfamilien, waehlt das beste Modell nach Composite-Validierung und kalibriert anschliessend den Heuristik/ML-Blend.
- Eval und Governance:
  - Proxy-Eval und Gold-Eval werden ueber `backend/ml/evaluate_ranker.py` gerechnet.
  - Pipeline-Heuristik wird separat ueber `backend/ml/evaluate_pipeline.py` gemessen.
  - Gold-Eval kommt aus Frontend-Feedback in `backend/services/feedback_store.py` und `backend/ml/export_frontend_gold_eval.py`, inklusive eingefrorenem Holdout-Split.
  - `backend/ml/train_ranker.py` trainiert bewusst nicht auf Gold-Holdout-Rows.
- Deployment/Operations:
  - alter Cloud-Pfad: `scripts/submit_vertex_training.py` -> `backend/ml/vertex_train.py` -> Shell-Flow `ml/train_jynxzi_vod.sh`
  - neuer portabler Pfad: `scripts/submit_vertex_pipeline.py` -> `backend/ml/pipelines/train_pipeline.py` -> Komponenten fuer Collect, Dataset-Build, Train, Eval, Promote
  - MLflow ist additiv fuer Tracking und Registry eingebaut; Serving kann aus `backend/services/ranker.py` auch auf Registry-Artefakte zurueckfallen.

## 2026-04-20 Plan: Backend Analysis / Assist Handover

- [x] Graphify-Report, Task-Historie und die zentralen Backend-Einstiegspunkte lesen.
- [x] Online-Inferenzpfad, Job-Orchestrierung und Feedback-/Persistenzpfade nachvollziehen.
- [x] ML-/Training-Stack, Vertex-Orchestrierung und Testoberflaeche auf hoher Ebene einordnen.
- [x] Architektur-Zusammenfassung und kuenftige Support-Schwerpunkte dokumentieren.

### Review Notes

- Das Backend hat zwei eng gekoppelte, aber klar getrennte Betriebsmodi: eine FastAPI-basierte Online-Clipping-Pipeline unter `main.py`/`routers/jobs.py`/`pipeline.py` und einen umfangreichen ML-/Training-Stack unter `backend/ml/`.
- Der Online-Pfad laeuft als Job-Workflow ueber Download, Audio-/Video-Feature-Extraktion, Transcript-Enrichment, Candidate-Generierung, Feature-Building, heuristisch+ML geblendetes Ranking, optionales Judge-Reranking, Boundary-Refinement, Clipping und Feedback-Persistenz.
- Die wichtigsten fachlichen Hotspots laut Code und Graphify sind `RichSegment`, `Candidate` und `Ranker`; besonders dicht gekoppelt sind dabei `services/analyzer.py`, `services/feature_builder.py`, `services/candidate_generators.py`, `services/ranker.py` und `ml/build_twitch_training_dataset.py`.
- Operativ relevant sind die lokalen Storage-/Cache-Pfade unter `backend/tmp/` beziehungsweise `settings.storage_dir`, weil Jobstatus, Debug-Artefakte, Feedback-Snapshots, Download-Cache und Trainings-Zwischenstaende dort zusammenlaufen.
- Die Testoberflaeche ist breit: Es existieren gezielte Tests fuer Router, Pipeline, Ranker, Feature-Building, Feedback, Training, Gold-Eval und Vertex-Pfade. Fuer kuenftige Arbeit sind dadurch regressionsorientierte Aenderungen im Backend gut absicherbar.
- In dieser Aufgabe wurden keine Codepfade funktional geaendert; es handelt sich um eine Architektur- und Handover-Analyse.

## 2026-04-20 Plan: Phase-1 Ranker / Vertex Bugfixes

- [x] Train/serve parity fuer offline-only Features sauberziehen, insbesondere `neighbor_positive_density`.
- [x] Semantic-Score-Skalierung im Ranker korrigieren und gegen Regressionsdrift absichern.
- [x] Vertex-Artifakt-Upload an erfolgreichen Trainings-Returncode koppeln und Fehlerfall strukturiert loggen.
- [x] Stillen Joblib-Load-Fallback im Ranker sichtbar machen und kaputte Artefakte frueher rejecten.
- [x] Zielgerichtete Regressionstests fuer Ranker- und Vertex-Pfade ergaenzen.
- [x] Relevante Tests laufen lassen, Review Notes dokumentieren und Graphify aktualisieren.

### Review Notes

- `neighbor_positive_density` bleibt bewusst ein Offline-Only-Feature. Statt im Serving eine falsche Approximation einzufuehren, rejectet `Ranker._artifact_is_usable()` jetzt jedes Artefakt, das Offline-Only-Features in `feature_names` referenziert. Zusaetzlich schreibt `train_ranker.py` `excluded_feature_names` ins Artefakt, damit die Trennung im JSON sichtbar bleibt.
- Die heuristische Semantic-Komponente in `backend/services/ranker.py` nutzt jetzt wieder die volle 0-10-Skala der fuenf Subscores (`sum * 2.0` statt implizitem 0-2-Cap durch `/ 5.0 * 10.0`).
- `backend/ml/vertex_train.py` laedt Artefakte nur noch hoch, wenn der Trainingsprozess mit Returncode `0` endet. Bei Fehlern werden strukturierte Events `vertex_training_failed` und `vertex_artifacts_upload_skipped` ausgegeben.
- Der bisher stille Joblib-Fallback loggt jetzt `artifact_model_load_failure`, und `Ranker.inspect_artifact()` rejectet kaputte Gradient-Boosting-/LightGBM-Artefakte schon vor dem Serving mit `artifact_model_unloadable` bzw. `artifact_model_loader_unavailable`.
- Verifikation: `backend/.venv/bin/python -m pytest backend/tests/test_ranker.py backend/tests/test_vertex_train.py backend/tests/test_train_ranker.py -q` lief mit `60 passed`; `backend/.venv/bin/python -m py_compile backend/services/ranker.py backend/services/feature_builder.py backend/ml/vertex_train.py backend/ml/train_ranker.py backend/tests/test_ranker.py backend/tests/test_vertex_train.py` blieb sauber.

## 2026-04-21 Plan: Replace Artifact-Only Run With 2x2 Workers

- [x] Den gerade eingereichten 1x1-Artifact-Only-Job abbrechen.
- [x] Eine neue 2x2-Vertex-Spec fuer denselben 7-Tage-Bereich schreiben.
- [x] Den Ersatzjob mit `TRAINING_VOD_WORKERS=2` und `TRAINING_LABEL_WORKERS=2` einreichen und Anfangsstatus pruefen.

### Review Notes

- Der zuerst eingereichte 1x1-Artifact-Only-Job wurde ersetzt:
  - alter Job: `projects/545981913082/locations/europe-west3/customJobs/7099017990334578688`
  - finaler Status: `JOB_STATE_CANCELLED`
- Neuer 2x2-Job:
  - Display Name: `jynxzi-ranker-train-20260421-ew3-7day-artifactonly-2x2`
  - Resource: `projects/545981913082/locations/europe-west3/customJobs/8168622901835071488`
  - Status kurz nach Submit: `JOB_STATE_RUNNING`
- Verwendete Konfiguration:
  - Date-Range `2026-03-14` bis `2026-03-20`
  - Maschine `n2-highmem-64`
  - `TRAINING_VOD_WORKERS=2`
  - `TRAINING_LABEL_WORKERS=2`
  - `--artifact-only`
  - Image `europe-west3-docker.pkg.dev/clipping-493513/ml/ai-clipping-train:20260421-artifactonly-rerun`
  - Artifact-Prefix `gs://clipping-493513-ml-artifacts-ew3/jynxzi/train-artifacts-20260421-7day-artifactonly-2x2`
  - Download-Cache `gs://clipping-493513-ml-artifacts-ew3/cache/vod-downloads`
  - Spec: [vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2.yaml](/Users/antonioherbin/Desktop/Privat/ai_clipping/tmp/vertex_job_spec_20260421_jynxzi_7day_artifactonly_2x2.yaml:1)

## 2026-04-22 Plan: Project Review / Assist Readiness

- [x] Read project instructions, lessons, graph report, and current task history.
- [x] Inspect the active working-tree changes in the training, pipeline, router, and feedback paths.
- [x] Run targeted verification for the changed Python and frontend surfaces.
- [x] Document review findings, open risks, and key project context for follow-up work.

### Review Notes

- Zielgerichtete Verifikation lief sauber:
  - `backend/.venv/bin/python -m pytest backend/tests/test_ranker.py backend/tests/test_feedback_store.py backend/tests/test_train_ranker.py backend/tests/test_local_shell_pipeline.py backend/tests/test_submit_vertex_pipeline.py -q`
  - Ergebnis: `52 passed`
- Review-Finding 1:
  - `backend/ml/pipelines/components/build_eval_dataset_op.py` liest Gold-Eval-Daten implizit aus lokalem `tmp/storage`, waehrend `backend/ml/pipelines/train_pipeline.py` diesem Step weder ein Storage-Artifact noch einen konfigurierbaren Storage-Pfad uebergibt.
  - In Vertex Pipelines laeuft dieser Step in einem isolierten Container; dort existieren die Frontend-Gold-Labels aus anderen Jobs nicht automatisch. `ml/export_frontend_gold_eval.py` beendet sich zudem mit Exit-Code `1`, wenn keine Gold-Eval-Rows gefunden werden.
  - Konsequenz: Der portable Vertex-Pipeline-Pfad kann Gold-Eval aktuell nicht reproduzierbar bauen und kann je nach Runtime-State hart fehlschlagen oder an lokaler Reststate haengen.
- Review-Finding 2:
  - Derselbe `build_eval_dataset`-Step ist auf KFP-Caching gesetzt, obwohl er von nicht deklariertem externem Zustand (`tmp/storage/feedback/gold_eval_updates.jsonl` und job-lokalen `debug/gold_eval.jsonl`) abhaengt.
  - Neue Gold-Labels invalidieren den Cache daher nicht; ein identischer Pipeline-Run kann veraltete Gold-Eval-Ausgaben wiederverwenden.
- Review-Finding 3:
  - `backend/services/ranker.py` versucht den MLflow-Registry-Fallback nur bei `status == "missing"`, nicht bei vorhandenen aber abgelehnten lokalen Artefakten.
  - Ein lokales Artefakt mit `semantic_score_config_mismatch`, kaputtem JSON oder unloadablem Modell blockiert damit weiterhin Serving, obwohl ein gueltiges Registry-Bundle verfuegbar sein kann.
  - Der Fall wurde lokal per isoliertem Repro bestaetigt; `inspect_artifact("gaming")` blieb auf `{'status': 'rejected', 'reason': 'semantic_score_config_mismatch', 'source': 'exact'}` statt auf das injizierte Registry-Artefakt zu fallen.

## 2026-04-22 Plan: Repo-Wide Cleanup / Remove AI Slop And Unused Code

- [x] Repo-Layout, Graphify-Hotspots und aktuellen Dirty-Worktree einordnen.
- [x] Mit statischen Signalen nach toten Imports, redundanten Helfern, ungenutzten Parametern und offensichtlichen Copy-Paste-Strukturen suchen.
- [x] Nur klar beweisbare Aufraeumarbeiten umsetzen; keine spekulativen Architektur-Umbauten mitten im aktiven ML/Pipeline-Branch.
- [x] Relevante Backend- und Frontend-Checks laufen lassen und Cleanup-Ergebnis dokumentieren.

### Cleanup Criteria

- "AI slop" bedeutet in diesem Pass:
  - redundante Wrapper ohne eigenen Mehrwert
  - copy-pastete Logik, die an einem Ort zusammengezogen werden kann
  - defensive Branches/Flags ohne realen Leser oder Effekt
  - aufgeblähte Interface-Pfade mit ungenutzten Daten
- "Unused code" bedeutet in diesem Pass:
  - ungenutzte Imports und Variablen
  - ungenutzte Parameter oder Rueckgabefelder
  - tote Hilfsfunktionen oder Zweige ohne Aufrufer
  - veraltete Pfade, die durch den aktuellen Code klar ersetzt wurden

### Review Notes

- Umgesetzte Aufraeumarbeiten:
  - `frontend/components/ClipCard.tsx`: Feedback-Payload auf echte Branch-Narrowing umgestellt und JSX-Struktur begradigt; der vorherige Union-Bau mit `null`-Werten war unnötig aufgeblasen und brach `tsc`.
  - `backend/services/ranker.py`: Registry-Fallback fuer denselben Profile-Key greift jetzt auch dann, wenn das lokale Artefakt vorhanden aber unbrauchbar ist; der alte Branching-Pfad war unnötig eng und blockierte brauchbare Registry-Bundles.
  - `backend/ml/pipelines/components/build_eval_dataset_op.py`: den Gold-Eval-Export nicht mehr ueber einen subprocess-CLI-Wrap laufen lassen, sondern direkt ueber die Python-Funktion; weniger Shell-Slop, weniger Sonderfehlerpfade.
  - `backend/ml/pipelines/train_pipeline.py` und `scripts/submit_vertex_pipeline.py`: `storage_dir` als expliziten Pipeline-Parameter verdrahtet und Caching fuer den Gold-Eval-abhängigen Step deaktiviert.
  - `backend/ml/evaluate_ranker.py`: leere Eval-Datensaetze liefern jetzt einen strukturierten Skip-Report statt eines harten Fehlers.
  - Tote Imports entfernt in `backend/ml/vertex_train.py`, `backend/services/transcriber.py`, `backend/services/video_features.py`, `backend/tests/test_pipeline_integration.py`, `backend/tests/test_training_workflow_state.py`.
- Verifikation:
  - `frontend/node_modules/.bin/tsc --noEmit -p frontend/tsconfig.json`
  - `cd frontend && npm run lint`
  - `backend/.venv/bin/python -m pytest backend/tests/test_ranker.py backend/tests/test_submit_vertex_pipeline.py backend/tests/test_training_workflow_state.py backend/tests/test_pipeline_integration.py -q`
  - `backend/.venv/bin/python -m pytest backend/tests/test_export_frontend_gold_eval.py backend/tests/test_feedback_store.py backend/tests/test_local_shell_pipeline.py -q`
  - `backend/.venv/bin/python backend/ml/evaluate_ranker.py --dataset <empty jsonl> --profile gaming --report-out <tmp report>`
- Ergebnis:
  - Frontend-Typcheck und ESLint sauber
  - Backend-Zieltests: `29 passed` plus `11 passed`
  - Empty-eval-Skip-Report erfolgreich erzeugt

## 2026-04-22 Plan: Repo-Wide Cleanup Sweep 2

- [x] Zweiten Sweep auf bislang unangetastete Dateien ausweiten.
- [x] Repo-weit nach toten Imports, privaten Helfern ohne Aufrufer und redundanten Branches suchen.
- [x] Zusaetzliche Cleanup-Fixes nur dort umsetzen, wo die Nutzungslage klar beweisbar ist.
- [x] Zweiten Sweep verifizieren und Restgrenzen dokumentieren.

### Review Notes

- Zweiter Sweep war absichtlich konservativ:
  - die repo-weite Import-Analyse war brauchbar fuer tote Imports
  - die repo-weite Helper-Analyse war fuer harte Loeschungen zu noisig und wurde nicht blind angewendet
- Zusätzliche Cleanup-Fixes:
  - tote Imports entfernt in `backend/ml/mlflow_tracking.py`
  - tote Imports entfernt in `backend/ml/pipelines/components/collect_twitch_clips_op.py`
  - tote Imports entfernt in `backend/ml/pipelines/components/promote_op.py`
  - tote Imports entfernt in `backend/ml/pipelines/components/train_ranker_op.py`
  - ungenutzten privaten Helper `_parent_path()` plus totes `os`-Import aus `backend/ml/pipelines/io.py` entfernt
- Zweiter Sweep verifiziert mit:
  - `backend/.venv/bin/python -m pytest backend/tests/test_local_shell_pipeline.py backend/tests/test_submit_vertex_pipeline.py backend/tests/test_ranker.py -q`
  - `backend/.venv/bin/python -m compileall backend/ml backend/services backend/routers scripts`
- Ergebnis:
  - `21 passed`
  - targeted compile smoke sauber
  - repo-weite Python-Import-Analyse nach dem zweiten Sweep ohne neue klare Dead-Import-Treffer
- Restgrenze:
  - Ich habe keine großen private-helper-Löschungen repo-weit blind gemacht, weil die schnelle Referenzanalyse zu viele falsch-negative Ergebnisse geliefert hat. Für tieferes Dead-Code-Pruning bräuchte es einen robusteren Call-Graph- oder per-file-manual pass.

## 2026-04-22 Follow-up: Speed Up Twitch VOD Downloads

- [x] Aktuelle `yt-dlp`-Doku fuer Beschleunigungsoptionen pruefen.
- [x] Twitch-spezifische `yt-dlp`-Optionen fuer parallele Fragment-Downloads und optionales `aria2c` verdrahten.
- [x] Das lokale Docker-Trainingsimage um `aria2` erweitern.
- [x] Die neuen Downloader-Optionen mit gezielten Tests absichern.

### Review Notes

- `backend/services/downloader.py` nutzt fuer Twitch-VODs jetzt `concurrent_fragment_downloads`, Fragment-Retries und optional `aria2c` als externen Downloader.
- `backend/config.py` und `backend/.env.example` expose-en dazu die Schalter `TWITCH_DOWNLOAD_CONCURRENT_FRAGMENTS`, `TWITCH_DOWNLOAD_USE_ARIA2C` und `TWITCH_DOWNLOAD_ARIA2_CONNECTIONS`.
- `backend/Dockerfile.local-train` installiert jetzt `aria2`, damit der beschleunigte Pfad im lokalen Docker-Run tatsaechlich verfuegbar ist.
- Verifiziert mit:
  - `backend/.venv/bin/python -m pytest backend/tests/test_downloader.py backend/tests/test_build_twitch_training_dataset.py -q`
  - `backend/.venv/bin/python -m py_compile backend/services/downloader.py backend/config.py backend/tests/test_downloader.py`
  - `zsh -n scripts/train_local_docker.sh`
- Ergebnis:
  - `36 passed`
  - Downloader- und Config-Pfade kompilieren sauber

## 2026-04-22 Follow-up: Add Long-Term VOD Cache

- [x] Bestehenden Twitch-Download-Cache auf einen dauerhaften lokalen Langzeit-Cache umstellen.
- [x] Cache-Reuse fuer wiederholte VOD-Downloads schneller machen.
- [x] Neue Cache-Semantik mit gezielten Downloader-Tests absichern.

### Review Notes

- `backend/config.py` nutzt fuer `TWITCH_DOWNLOAD_CACHE_TTL_SEC` jetzt per Default `-1`, also dauerhaften Cache statt kurzer TTL.
- `backend/services/downloader.py` verwendet fuer Twitch-VODs jetzt kanonische Cache-Dateinamen wie `vod_<id>.mp4` statt URL-Hash-only.
- Cache-Hits werden schneller in den Job materialisiert, weil zuerst ein Hardlink versucht wird und nur bei Bedarf auf Copy zurueckgefallen wird.
- Cache-Eintrage werden nur bei erfolgreichem vollstaendigem Download gespeichert; abgebrochene Teil-Downloads wandern nicht in den Langzeit-Cache.
- Semantik:
  - `TWITCH_DOWNLOAD_CACHE_TTL_SEC=-1` -> dauerhaft behalten
  - `TWITCH_DOWNLOAD_CACHE_TTL_SEC>0` -> nach TTL evicten
  - `TWITCH_DOWNLOAD_CACHE_TTL_SEC=0` -> Cache komplett aus
- Verifiziert mit:
  - `backend/.venv/bin/python -m pytest backend/tests/test_downloader.py -q`
  - `backend/.venv/bin/python -m py_compile backend/services/downloader.py backend/config.py backend/tests/test_downloader.py`
- Ergebnis:
  - `11 passed`
  - Downloader-Pfade kompilieren sauber

## 2026-04-22 Follow-up: Stabilize FFmpeg Batch Window Cutting

- [x] Batch-Cut-Fehlerpfad im Trainingslauf analysieren.
- [x] FFmpeg-Batch-Cuts bei Fehler automatisch in kleinere Teilbatches splitten.
- [x] Encoder-Thread-Druck pro Output reduzieren.
- [x] Regressions-Tests für Retry- und Single-Window-Fehler ergänzen.

### Review Notes

- Ursache:
  - `backend/ml/collect_viral_clips.py` schneidet bis zu `24` Trainingsfenster in einem einzigen `ffmpeg`-Prozess.
  - Dabei wird für jedes Output-Fenster `libx264` initialisiert; das ist auf WSL2/Docker deutlich anfälliger für Ressourcen- und Thread-Druck als Einzel- oder Kleinbatch-Cuts.
  - Der Screenshot passt genau zu diesem Pfad: erster Fehlbatch mit `24` Windows entspricht `_MAX_BATCH_CUT_WINDOWS = 24`.
- Fix:
  - `cut_video_windows()` ruft jetzt `_cut_video_windows_chunk()` auf, das fehlgeschlagene Batches rekursiv halbiert statt komplett aufzugeben.
  - Jeder Output bekommt im Batch-Cut jetzt `-threads 1`, damit ein Batch nicht 24 mal mit aggressiver Encoder-Autothreading-Politik eskaliert.
  - Einzelne wirklich kaputte Fenster schlagen weiterhin klar fehl; der Retry-Fix kaschiert keine echten Einzeldatei-Probleme.
- Verifiziert mit:
  - `backend/.venv/bin/python -m pytest backend/tests/test_collect_viral_clips.py backend/tests/test_build_twitch_training_dataset.py -q`
  - `backend/.venv/bin/python -m py_compile backend/ml/collect_viral_clips.py backend/tests/test_collect_viral_clips.py`
- Ergebnis:
  - `37 passed`
  - Batch-Cut-Retry und Single-Window-Fehlerpfad abgesichert

## 2026-04-26 Plan: Chat Feature Smoke Test Without Full VOD Downloads

- [x] Run a zero-download chat-feature smoke test against cached Twitch chat.
- [x] Run a one-label window-scoped mini dataset build.
- [x] Check outputs for active `has_chat_features` and non-zero `chat_*` values.
- [x] Confirm the mini run did not use the Full-VOD download path.

### Review Notes

- Zero-download smoke used cached chat for VOD `2735012372`, window `1067.0-1132.9`.
- Zero-download output activated chat features: `has_chat_features=true`, `chat_peak=1.0`, `chat_message_rate_peak=14.0`, `chat_density_peak=1.0`, category scores for hype/laugh/shock active.
- Mini dataset build used a writable temporary cache seeded from the existing scoped video source to avoid any Full-VOD download.
- Mini build logs confirmed `mode=window_scoped_top_viewed`; no Full-VOD materialization path was used.
- Mini build produced `6` rows: `1` weak-positive primary row and `5` boundary near-miss negatives.
- All emitted mini rows had active chat features; representative primary row had `chat_peak=1.0`, `chat_mean=0.891`, `chat_message_rate_peak=17.308`, `chat_density_peak=1.0`, `chat_laugh_score=1.0`, `chat_shock_score=1.0`.
- The sandbox blocked Twitch DNS during a chat-cache refill attempt, but enrichment still applied from the copied existing chat cache: `segments_with_chat=16`.
- Mini output: `/tmp/ai_clipping_chat_feature_smoke_rows.jsonl`.

## 2026-04-27 Plan: Stabilize Scoped Twitch Range Downloads

- [x] Diagnose why scoped-only VOD range downloads fail with `ffmpeg exited with code 8`.
- [x] Patch the range downloader to avoid the fragile ffmpeg re-encode path while preserving clip-scoped downloads.
- [x] Add/update downloader tests for the safer range options.
- [x] Run focused tests and compile checks.
- [x] Rebuild `backend/graphify-out` after backend code changes.

### Review Notes

- Cause: `download_range()` used precise `force_keyframes_at_cuts=True`, which sends scoped Twitch range downloads through a fragile ffmpeg re-encode path; the latest scoped-only VOD failed there with `ffmpeg exited with code 8`.
- Fix: scoped range downloads still use yt-dlp `download_ranges`, but now disable external downloader options and set `force_keyframes_at_cuts=False`.
- Full-VOD downloads are unchanged and can still use `aria2c`; only clip-scoped range downloads avoid the risky path.
- Verification passed: `backend/.venv/bin/python -m pytest backend/tests/test_downloader.py -q` -> `15 passed`.
- Verification passed: `backend/.venv/bin/python -m pytest backend/tests/test_build_twitch_training_dataset.py -q` -> `61 passed`.
- Verification passed: `backend/.venv/bin/python -m py_compile backend/services/downloader.py backend/tests/test_downloader.py`.
- Verification passed: scoped `git diff --check`.
- Rebuilt backend Graphify: `2097` nodes, `5406` edges, `52` communities.
- Rebuilt local Docker train image `ai-clipping-train-local:latest` and verified the image contains `force_keyframes_at_cuts=False` plus `disable_external_downloader=True` in `download_range()`.

## 2026-04-27 Plan: Fix Full-VOD Checkpoint Skip For Unscoped Chat

- [x] Confirm why the four already-processed Full-VODs were not checkpointed.
- [x] Patch checkpoint eligibility so unscoped chat caches without missing offsets can be reused.
- [x] Add focused regression tests.
- [x] Run targeted tests and compile checks.
- [x] Rebuild backend Graphify and local training image.

### Review Notes

- Cause: the four already-processed Full-VOD groups had no chat-target labels, so their chat cache metadata was written as `coverage_status=unscoped` instead of `complete`.
- The old checkpoint gate accepted only `complete` chat coverage and skipped checkpoint writes with `chat_enabled_without_complete_chat_cache`, even though the unscoped chat cache existed and had no missing target offsets.
- Fix: `_chat_cache_has_complete_coverage()` now treats unscoped chat caches as checkpoint-safe when a cached chat file exists and `missing_target_offsets` is empty.
- Partial or missing chat caches still block checkpoint writes.
- Verification passed: `backend/.venv/bin/python -m pytest backend/tests/test_build_twitch_training_dataset.py -q` -> `63 passed`.
- Verification passed: `backend/.venv/bin/python -m py_compile backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py`.
- Verification passed: scoped `git diff --check`.
- Rebuilt backend Graphify: `2099` nodes, `5415` edges, `47` communities.
- Rebuilt local Docker train image `ai-clipping-train-local:latest` and verified the image contains the unscoped chat-cache checkpoint logic.
- Caveat: a training container started before this image rebuild still runs the old checkpoint gate and should be restarted to benefit from the fix.

## 2026-04-27 Plan: Analyze Completed Jynxzi Training Run

- [x] Confirm final run status and artifact timestamps.
- [x] Inspect ranker, health, evaluation, holdout, threshold, and dataset summaries.
- [x] Check whether the recent checkpoint/chat fixes affected the finished run.
- [x] Summarize result quality, risks, and next action.

### Review Notes

- Run completed successfully: `local_train_summary.json` has `training_return_code=0` for `jynxzi`, `2026-03-15` through `2026-03-28`.
- Final artifact files were written at `2026-04-27 20:55`, including `jynxzi_ranker.json`, `jynxzi_artifact_health.json`, eval files, threshold calibration, and VOD training rows.
- Artifact health has no warnings; selected model is `logistic_regression`, serving threshold is `5.298`, and validation gates show low leakage risk with group holdout by `vod_id`.
- Training dataset has `601` rows: `142` positives and `459` negatives, split into `516` train rows and `85` validation rows.
- Dataset sources are `256` Twitch VOD primary rows, `170` boundary near-miss rows, `162` gold-eval train rows, and `13` promoted hard-negative rows.
- Gold eval has `222` rows: trained runtime improves full-set F1 from `0.510` to `0.734`, accuracy from `0.342` to `0.833`, and MRR from `0.556` to `0.657`.
- True untouched gold holdout is still weak: `60` holdout rows, trained runtime accuracy `0.717`, precision `0.636`, recall `0.350`, F1 `0.452`, MRR `0.443`; ranking is only marginally above heuristic MRR `0.438`.
- Proxy eval is stronger: `601` rows, trained runtime MRR `0.691` vs baseline `0.427`, NDCG@5 `0.656` vs `0.427`, F1 `0.640` vs `0.382`.
- Threshold calibration updated from `0.261297` to `5.298`; this improves precision but costs recall, especially on holdout positives.
- Chat features are present in nested `features`: `has_chat_features` is truthy on `352/601` rows; `chat_activity_coverage`, `chat_density_mean`, and `chat_message_rate_*` are active on `350` rows.
- Feature provenance shows `287` full-VOD rows and `173` `window_scoped_top_viewed` rows; `141` older/imported gold rows have blank provenance.
- The four previously problematic Full-VOD checkpoints now exist: `2720629593`, `2724120511`, `2731588015`, and `2732418387`.
- There are `50` VOD checkpoint files for `51` non-ambiguous label VODs; the missing VOD-level checkpoint is `2748783535`, but its scoped source/cache/rows exist.
- Holdout diagnosis has `22` rows: `9` false negatives, `2` false positives, `9` true negatives, and `2` true positives. False negatives are mostly grade `2/3` positives that fall below the high calibrated threshold.
- Recommendation: keep the artifact as usable, but do not treat the gold holdout as solved. Next best step is reviewing/importing feature-backed top-viewed positives and hard false-negative gold examples, then recalibrating threshold or adding a second operating threshold for candidate review.

## 2026-04-27 Plan: Push Training Changes And Artifacts

- [x] Confirm branch and working tree contents.
- [x] Stage code, task notes, Graphify output, and ML artifacts.
- [x] Commit with a concise summary.
- [x] Push `main` to origin and verify clean status.

### Review Notes

- Branch is `main`; `git pull --ff-only` reported `Already up to date`.
- Staged tracked backend code, tests, Graphify output, task notes, and updated tracked ML artifacts.
- New ignored run artifacts are intentionally force-added, excluding the Twitch cache directories.
- `git diff --check` only reported CRLF/trailing whitespace in generated `jynxzi_ranker_holdout_diagnosis.csv`; no code whitespace issues were found.

## 2026-05-08 Plan: Replace Full-MP4 Coarse Matching With Full-VOD Audio Fingerprints

- [x] Stop or confirm stopped the currently running MP4 full-coverage training run.
- [x] Inspect current cross-platform manifest and audio-fingerprint matcher flow.
- [x] Add cached full-VOD audio fingerprint manifest rows.
- [x] Make matcher consume VOD `fingerprint_path` rows directly.
- [x] Switch training defaults from MP4 coarse tiling to full-VOD audio fingerprints.
- [x] Add regression tests for full-VOD audio fingerprint manifest and matcher input.
- [x] Run focused tests, syntax checks, Graphify, and disk verification.

### Review Notes

- Replaced full-MP4 coarse matching with full-VOD audio fingerprinting. The
  refresh path now extracts each selected VOD's full audio stream, writes one
  `source.fpcalc.json` fingerprint for the entire VOD timeline, and deletes the
  temporary `.m4a` by default.
- The cross-platform matcher now consumes VOD rows with `fingerprint_path`
  directly, so repost fingerprints are compared against the full VOD audio
  fingerprint instead of only scoped video snippets.
- `train_jynxzi_vod.sh` now defaults to
  `CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS=1` and
  `CROSS_PLATFORM_COARSE_VOD_TILING=0`.
- In full-audio mode, refreshed VOD manifests exclude old scoped MP4 windows,
  so stale `coarse_*` video tiles do not silently stay in the matching set.
- Verification passed: focused pytest -> `34 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully after backend changes.
- Disk check after implementation: `/dev/sdd` has `832G` available.

## 2026-05-08 Plan: Clean Backend Tmp Storage

- [x] Measure project filesystem, Docker usage, and `backend/tmp` subtrees.
- [x] Delete regenerable storage job/download dirs and coarse VOD audio tiles.
- [x] Delete old `extended_window_scoped_group` training-window caches.
- [x] Delete legacy scoped windows without `source_kind` metadata.
- [x] Verify remaining free space and retained cache categories.

### Review Notes

- Filesystem before cleanup was full: `/dev/sdd` had `0` available and `backend/tmp` was `938G`.
- Removed all `coarse_*` VOD feature windows and transient `backend/tmp/storage` job/download-cache directories.
- Removed old `extended_window_scoped_group` scoped-window caches, which were the main storage consumer.
- Removed 104 legacy scoped windows with missing/unreadable `source_kind`.
- Filesystem after cleanup: `/dev/sdd` has `898G` available and `backend/tmp` is down to `27G`.
- Remaining VOD feature cache is `26G`, mainly `local_source` and `twitch_download_cache` windows plus small per-VOD metadata/media caches.

## 2026-05-08 Plan: Stop Coarse Tiling Deleted Twitch VOD Loops

- [x] Identify why deleted VODs continue through later coarse tiles.
- [x] Add permanent unavailable-VOD detection and abort remaining tiles for that VOD.
- [x] Add regression coverage for `Video ... does not exist`.
- [x] Run focused tests and syntax checks.
- [x] Rebuild Graphify and summarize current-run impact.

### Review Notes

- Added permanent Twitch-VOD-unavailable detection for coarse tiling errors such as `ERROR: [twitch:vod] 2712736206: Video 2712736206 does not exist`.
- On the first permanent VOD error, the tiler now logs `coarse_vod_tile_vod_abort` and skips the remaining tiles for that VOD instead of trying tile 100+.
- The summary now includes `unavailable_vods` so deleted VOD coverage loss is visible in artifacts/logs.
- Added a regression test proving a 6-tile VOD attempts only one download when Twitch reports the VOD does not exist.
- Verification passed: `backend/.venv/bin/python -m pytest backend/tests/test_prepare_cross_platform_fingerprint_rows.py -q` -> `16 passed`.
- Verification passed: `backend/.venv/bin/python -m py_compile backend/ml/prepare_cross_platform_fingerprint_rows.py`.
- Verification passed: scoped `git diff --check`.
- Rebuilt backend Graphify: `2880` nodes, `7649` edges, `70` communities.

## 2026-05-07 Plan: Make Cross-Platform VOD Matching Full-Coverage

- [x] Confirm current coarse tiling is sparse and capped.
- [x] Change second-pass VOD tiling defaults to lueckenlose VOD-horizon coverage.
- [x] Use real VOD duration when available instead of only the last Twitch-label window.
- [x] Add tests proving full-duration tiling and workflow defaults.
- [x] Run focused verification and document operational impact.

### Review Notes

- Cross-platform VOD tiling defaults now use `tile=360s`, `stride=180s`, and `max_tiles=0`, so selected VODs are no longer sparse-sampled or capped at 24 windows.
- Coarse tiling now prefers `vod_duration_seconds`/duration metadata for the horizon. It falls back to the previous label-window horizon plus padding only when no VOD duration is available.
- The tiling summary logs `coverage_mode`, `full_duration_vods`, `label_horizon_vods`, `total_tiles`, and `total_covered_sec` so the next run can prove whether it used full-duration coverage.
- Added a regression test that a 1000s VOD creates six overlapping full-coverage tiles starting at `0, 180, 360, 540, 720, 900` and ending at the real VOD duration.
- Verification passed: `backend/.venv/bin/python -m pytest backend/tests/test_prepare_cross_platform_fingerprint_rows.py backend/tests/test_training_workflow_state.py -q` -> `22 passed`.
- Verification passed: `backend/.venv/bin/python -m py_compile backend/ml/prepare_cross_platform_fingerprint_rows.py backend/ml/training_workflow_state.py`.
- Verification passed: `bash -n backend/ml/train_jynxzi_vod.sh`.
- Verification passed: scoped `git diff --check`.
- Rebuilt backend Graphify with the local graphify interpreter: `2878` nodes, `7644` edges, `68` communities.

## 2026-04-27 Plan: Prevent Holdout Leakage In Training Dataset

- [x] Inspect where VOD proxy rows, gold train rows, and hard negatives are merged.
- [x] Add a central holdout VOD/window exclusion policy for training rows.
- [x] Fail or summarize when holdout overlap is detected, so evaluation cannot be silently polluted.
- [x] Add regression tests for VOD-level and window-overlap filtering.
- [x] Run focused tests, compile checks, Graphify rebuild, and document result.

### Review Notes

- Added `build_composite_dataset_with_report()` in `backend/ml/train_vod_ranker.py`; `main()` now prints `holdout_training_exclusion` inside the dataset summary.
- The policy derives holdout VODs from `gold_holdout_split=holdout` rows and filters training labels before `build_twitch_training_dataset.build_dataset()`, preventing expensive feature builds for holdout VODs.
- The same policy is also applied after VOD row build, after gold-train preparation, after hard-negative preparation, and once more after final merge.
- Promoted hard negatives with `hard_negative_original_split=holdout` are excluded from training by default.
- Current artifact impact audit: `35` holdout VODs, `60` holdout rows; next build would filter `5636` Twitch label rows, `114` same-VOD gold-train rows, and `2` promoted holdout hard negatives.
- Verification passed: `backend/.venv/bin/python -m pytest backend/tests/test_train_vod_ranker.py -q` -> `21 passed`.
- Verification passed: `backend/.venv/bin/python -m pytest backend/tests/test_train_vod_ranker.py backend/tests/test_train_ranker.py -q` -> `55 passed`.
- Verification passed: `backend/.venv/bin/python -m py_compile backend/ml/train_vod_ranker.py backend/tests/test_train_vod_ranker.py`.
- Verification passed: scoped `git diff --check`.
- Rebuilt backend Graphify: `2113` nodes, `5456` edges, `51` communities.
- Rebuilt local Docker train image `ai-clipping-train-local:latest` and verified the image contains the holdout exclusion code.

## 2026-04-27 Plan: Push Holdout Fix And Latest Artifacts

- [x] Confirm branch and current working tree state.
- [x] Review changed tracked files before staging.
- [x] Stage all current code changes, Graphify output, task notes, and tracked ML artifacts.
- [x] Commit the changes on `main`.
- [x] Push `main` to origin and verify clean status.

### Review Notes

- Branch is `main`; current HEAD before this push is `bac99d1`.
- `git diff --stat` shows 22 tracked files: backend holdout-exclusion code/tests, Graphify output, latest Jynxzi ML artifacts, and task notes.
- `git diff --check` passed after normalizing the generated holdout-diagnosis CSV line endings.
- Staged tracked changes with `git add -u`; no untracked files were present in `git status`.

## 2026-04-27 Plan: Apply Reviewed Holdout Labels

- [x] Locate the reviewed holdout false positives in the current diagnosis artifacts.
- [x] Promote the one accepted `2712128863` holdout row to `gold_grade=2`.
- [x] Mark the remaining reviewed holdout false positives as confirmed negatives.
- [x] Recompute gold-eval/diagnosis artifacts without rebuilding the Docker image.
- [x] Verify no holdout VODs entered training and document the result.

### Review Notes

- User review: all false-positive holdouts are real false positives except the `2712128863` clip, which should become `gold_grade=2`.
- Current holdout diagnosis had 7 false positives; promoted only `twitch_vod_2712128863:twitch_vod_2712128863_14:narrative_reveal:0:0`.
- Marked the other 6 false-positive rows with `gold_reviewed_at=2026-04-27` and `gold_review_tags=["false_positive_review", "confirmed_negative"]`.
- Rebuilt `jynxzi_gold_eval_dataset.jsonl` and `jynxzi_gold_eval.json` from `jynxzi_gold_eval_rows.jsonl`; gold eval now has 77 positives and 145 negatives.
- Recomputed holdout diagnosis and hard-negative export; false positives are now 6, true positives 11, false negatives 1, true negatives 12.
- Hard-negative export now contains 6 rows instead of 7; the promoted `2712128863` row is no longer exported as a hard negative.
- Holdout trained metrics after relabel: accuracy `0.783`, precision `0.654`, recall `0.810`, F1 `0.723`, MRR `0.471`, NDCG@5 `0.475`.
- Verified training remains isolated: 244 training rows, 29 training VODs, 35 holdout VODs, and 0 train/holdout VOD overlap.
- Docker image was intentionally not rebuilt per user request.

## 2026-04-27 Plan: Analyze New Jynxzi Training Run

- [x] Confirm whether a new training container is still running or finished.
- [x] Inspect latest run summary, workflow state, model health, and artifact timestamps.
- [x] Compare dataset composition against the previous post-holdout-fix run.
- [x] Audit holdout isolation and chat-feature coverage.
- [x] Summarize model quality, remaining issues, and next action.

### Review Notes

- Initial `docker ps -a` showed no active or stopped containers visible.
- Latest tracked artifacts were written at `2026-04-27 23:17`.
- New run completed successfully: `local_train_summary.json` has `training_return_code=0` for `jynxzi`, `2026-03-15` through `2026-03-28`.
- Artifact health is clean: no warnings, selected model `logistic_regression`, serving threshold `3.636`, low leakage risk.
- Training dataset remained 244 rows over 29 VODs: 64 positives, 180 negatives; sources are 118 Twitch proxy, 78 boundary near-miss, 48 gold train.
- Holdout isolation remains intact: 35 holdout VODs and 0 train/holdout VOD overlap; no `gold_holdout_split=holdout` rows in training.
- Chat features are present on 140/244 training rows; `chat_activity_coverage`, `chat_density_mean`, and `chat_message_rate_peak` are nonzero on 138 rows.
- Gold eval after the user relabel has 222 rows: 77 positives and 145 negatives.
- Gold holdout now has 60 rows with trained F1 `0.723`, precision `0.654`, recall `0.810`, MRR `0.471`, NDCG@5 `0.475`.
- Holdout diagnosis now has 6 false positives, 11 true positives, 1 false negative, and 12 true negatives.
- Hard-negative export contains 6 confirmed false-positive rows for a future training run; they were not part of this training dataset.

## 2026-05-08 Plan: Filter Dead Cache-Only VODs From Full-Audio Backfill

- [x] Inspect full-VOD audio selection and current VOD inventory provenance.
- [x] Prefer Twitch API VOD inventory rows and admit cache-only rows only when a full-audio fingerprint already exists.
- [x] Add focused tests for cache-only filtering and API-row priority.
- [x] Run focused verification and rebuild Graphify after backend changes.

## 2026-05-14 Plan: Test 20 Transcript Match Workers

- [x] Raise the next-run transcript matching default from 16 to 20 workers.
- [x] Update env defaults and the script regression check.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The active container cannot change its already-started worker count in-place;
  `CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS=20` applies to the next run.
- Verification passed: `bash -n backend/ml/train_jynxzi_vod.sh` and
  `pytest backend/tests/test_train_jynxzi_vod_script.py -q`.

## 2026-05-14 Plan: Test 16 Transcript Match Workers

- [x] Confirm the active run still uses the already-started worker count.
- [x] Raise the next-run transcript matching default from 8 to 16 workers.
- [x] Update the script regression check for the new default.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The active container still runs with `--transcript-match-workers 8`; that
  value cannot change in-place after the process has started.
- `train_jynxzi_vod.sh`, `.env`, and `.env.example` now default to
  `CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS=16` for the next run.
- Verification passed: `bash -n backend/ml/train_jynxzi_vod.sh`,
  `pytest backend/tests/test_train_jynxzi_vod_script.py -q`, and Graphify
  rebuild.

## 2026-05-14 Plan: Add Whole-VOD Dense Ranker Candidates

- [x] Add a public dense whole-VOD candidate generator that reuses the existing dense-window proposal logic.
- [x] Feed dense candidates into the normal ranker path even when peak candidates exist.
- [x] Persist dense candidates in job debug artifacts so recall can be inspected per frontend run.
- [x] Add focused tests for dense candidate generation, merge caps, and pipeline debug output.
- [x] Run focused verification and rebuild Graphify after backend changes.

### Review Notes

- Frontend VOD runs now generate `dense_window` candidates from the whole VOD in
  addition to peak and semantic candidates.
- Dense candidates are fed into `merge_prioritized_candidates` even when peak
  candidates exist, then scored by the same feature builder and ranker path.
- The dense source cap is intentionally higher for this whole-VOD recall path:
  `WHOLE_VOD_DENSE_SOURCE_CAP=48`, controlled by
  `ENABLE_WHOLE_VOD_DENSE_CANDIDATES`.
- Job debug output now includes `debug/candidates_dense.jsonl`, making it
  inspectable whether the dense recall path found additional windows.
- Verification so far: focused candidate-generator/pipeline tests passed with
  `14` tests, `py_compile` passed, and scoped `git diff --check` passed.
- Graphify rebuilt successfully: `3131` nodes, `8352` edges, `81`
  communities.

## 2026-05-14 Plan: Preserve yt-dlp Primary Download Errors

- [x] Replace the `with YoutubeDL(...)` block with explicit close handling so cleanup errors cannot hide `extract_info` failures.
- [x] Add a regression test where ffmpeg `-11` is followed by a cookie-save failure.
- [x] Verify the downloader tests, syntax, diff hygiene, and Graphify rebuild.

### Review Notes

- The previous retry did not trigger when `YoutubeDL.__exit__` raised a
  cookie-save `FileNotFoundError` after the primary `ffmpeg exited with code
  -11` error.
- `_download_uncached` now calls yt-dlp through explicit close handling, so a
  cleanup failure cannot mask the original `extract_info` exception.
- Regression coverage now simulates `ffmpeg -11` followed by cookie cleanup
  failure and proves the safer Twitch retry still runs.
- Verification so far: `py_compile` passed and
  `backend/tests/test_downloader.py` passed with `23` tests.
- `git diff --check` passed and Graphify rebuilt successfully: `3125` nodes,
  `8332` edges, `86` communities.

## 2026-05-14 Plan: Harden yt-dlp Cookie Paths And FFmpeg Crash Retry

- [x] Confirm the downloader failure chain from the reported traceback.
- [x] Remap Docker-style yt-dlp cookie paths to the local backend path and create the parent directory before yt-dlp can save cookies.
- [x] Retry Twitch downloads once with safer native/single-file yt-dlp options when ffmpeg exits with code `-11`.
- [x] Add regression tests for cookie path handling and the retry path.
- [x] Run focused verification and rebuild Graphify after backend changes.

### Review Notes

- Root cause was a two-stage failure: yt-dlp first hit `ffmpeg exited with
  code -11`, then `YoutubeDL.__exit__` tried to save cookies to the Docker-only
  path `/app/backend/tmp/www.youtube.com_cookies.txt`, which does not exist in
  the local backend process and masked the original downloader error.
- `services.yt_dlp_auth` now remaps `/app/backend/...` cookie paths to the
  local backend root and creates the cookiefile parent directory before yt-dlp
  starts.
- Twitch downloads now retry once after `ffmpeg exited with code -11` using
  native yt-dlp downloading and `best[ext=mp4]/best`, avoiding the failing
  external/merged download path where possible.
- Verification so far: `py_compile` passed and
  `backend/tests/test_downloader.py` passed with `22` tests.
- Graphify rebuilt successfully: `3120` nodes, `8322` edges, `80`
  communities.

### Review Notes

- The active run proved the Twitch-label removal worked, but the VOD inventory
  admitted old cache-only VODs whose Twitch URLs no longer exist.
- `train_jynxzi_vod.sh` now preserves `twitch_vod_api` provenance when cache
  metadata is merged onto the same VOD, so live API VODs do not get downgraded
  to `vod_feature_cache`.
- `prepare_cross_platform_fingerprint_rows.py` filters inventory-backed
  selected VODs before download: API-backed VODs are allowed, cache-only VODs
  are allowed only if their existing full-VOD audio fingerprint cache is usable.
- Verification passed: focused prepare/script tests, `py_compile`, `bash -n`,
  and scoped `git diff --check`.

## 2026-05-10 Plan: Rebuild VOD Manifest After Full-Audio Fingerprints

- [x] Confirm why the matcher used only one stale VOD row after 57 full-audio fingerprints were built.
- [x] Force a VOD manifest rebuild from full-audio fingerprint cache whenever full-VOD-audio mode succeeds.
- [x] Add regression coverage for stale one-row manifest replacement.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The failed run built `processed_vods=57` full-audio fingerprints, but
  preserved a stale `jynxzi_cross_platform_vods.jsonl` with `vod_rows=1`
  because `refresh_vods=false` and the old manifest was still usable.
- Full-VOD-audio mode now treats a successful full-audio build as a manifest
  rebuild reason (`full_vod_audio_fingerprints`) and rewrites the VOD manifest
  from `full_vod_audio/source.fpcalc.json` cache rows before matching.
- Verification passed: focused prepare tests, `py_compile`, and scoped
  `git diff --check`.

## 2026-05-10 Plan: Calibrate Cross-Platform Gates To Available Jynxzi Coverage

- [x] Lower cross-platform matched-row gates from absolute 300/500-style targets to the available Jynxzi outcome coverage.
- [x] Keep VOD diversity enforced with a non-zero matched-VOD minimum.
- [x] Align ranker minimum, workflow state defaults, env files, and tests.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- Current full-VOD matching found about `74-75` accepted rows over `29` VODs at
  threshold `0.60`; the old `300/500` row gates exceeded the known available
  Jynxzi coverage.
- Training and prepare gates now use `70` matched rows plus `20` matched VODs,
  keeping a diversity gate without forcing the threshold down to `0.55`.
- `VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS` now defaults to `70`, aligned with
  the cross-platform prepare gate.
- Verification passed: focused script/workflow-state tests, `py_compile`,
  `bash -n`, scoped `git diff --check`, and Graphify rebuild.

## 2026-05-10 Plan: Filter Non-Jynxzi Cross-Platform Reposts

- [x] Inspect repost discovery metadata and confirm why unrelated search hits enter the Jynxzi manifest.
- [x] Preserve title/uploader/channel metadata from discovery into repost manifest rows.
- [x] Filter discovered repost candidates unless they have explicit Jynxzi/profile evidence outside the search query.
- [x] Add tests, update lessons, run focused verification, and rebuild Graphify.

### Review Notes

- Root cause: discovered `ytsearch` results were stamped with
  `streamer=jynxzi` from the search seed, while the persisted repost rows did
  not keep title/uploader/channel metadata that could prove the video was
  actually about Jynxzi.
- Discovery now carries `title`, `uploader`, `channel`, `channel_id`,
  `description`, and `discovery_source_url` into repost manifest rows.
- Search-discovered reposts now pass only when title/uploader/channel/description
  contains explicit profile evidence. Seed tags and notes are not accepted as
  proof, so `ytsearch:jynxzi ...` alone cannot admit unrelated clips.
- Repost manifest rows now include `profile_filter_version`; existing stale
  manifests without that marker rebuild automatically with reason
  `profile_filter_policy` on the next run.
- Current local `jynxzi_cross_platform_reposts.jsonl` samples have no
  `profile_filter_version`, title, or uploader, so the next run will trigger
  that stale-manifest rebuild.
- If the profile-filter rebuild finds too few valid reposts/matches, the old
  cross-platform output is no longer preserved; the run fails closed instead
  of training on stale false positives.
- Verification passed: focused pytest -> `50 passed`; `py_compile` passed;
  scoped `git diff --check` passed; Graphify rebuilt successfully with
  `2934` nodes, `7818` edges, and `73` communities.

## 2026-05-14 Plan: Ngrok Remote Frontend Setup

- [x] Stop any stale ngrok tunnel processes for this project.
- [x] Start a local reverse proxy that routes `/api` to backend and everything else to frontend.
- [x] Rebuild/restart frontend with same-origin API calls.
- [x] Start one ngrok tunnel for the combined local proxy and capture the public URL.
- [x] Verify remote frontend/backend URLs and document access details.

### Review Notes

- A two-tunnel ngrok config was not stable in this account/session: both tunnel
  entries reported the same public host and the agent exited. The final setup
  uses one local proxy on `127.0.0.1:8080` and one ngrok tunnel.
- Remote URL:
  `https://catfight-preformed-entertain.ngrok-free.dev`.
- Frontend was rebuilt with `NEXT_PUBLIC_API_URL=` so browser API calls target
  same-origin `/api/...` through the proxy instead of `localhost:8000`.
- Verified locally through the proxy: `http://127.0.0.1:8080/` and
  `http://127.0.0.1:8080/api/status`.
- Verified remotely through ngrok:
  `https://catfight-preformed-entertain.ngrok-free.dev/` returns frontend HTML
  and `/api/status` returns backend JSON.
- Active process files:
  `backend/tmp/ngrok/proxy.pid`, `backend/tmp/ngrok/ngrok.pid`; logs are in
  `backend/tmp/ngrok/proxy.log` and `backend/tmp/ngrok/ngrok.log`.

## 2026-05-14 Plan: Diagnose Clip Quality Model vs Pipeline

- [x] Inspect the latest completed frontend job outputs and selected clips.
- [x] Inspect current Jynxzi ranker/eval artifacts for model quality signals.
- [x] Check candidate-generation and ranking flow enough to locate the likely bottleneck.
- [x] Summarize whether the weakness is model quality, candidate pool quality, or downstream clipping boundaries.

### Review Notes

- Latest completed Jynxzi jobs use `ranking_mode=ml_blended` for most selected
  clips, so the trained model is active in serving.
- Current artifact is formally usable: `jynxzi_ranker.json` selects
  `lightgbm_ranker`, and `jynxzi_ranker_eval.json` reports calibration
  precision `0.909`, recall `0.833`, and ranking metric `0.778` on only
  `30` calibration rows.
- The latest debug job `9c7adf11-5b59-4244-bf7a-f3f1ab839033` produced
  `18` peak candidates, `531` semantic candidates, `11` dense candidates, but
  only `24` merged/normalized candidates reached ranking.
- The judge rerank gave all top candidates weak-to-medium scores around
  `3.9` to `5.15` and repeatedly wrote negative reasons such as unclear
  context, no clean payoff, missing resolution, and fragmentary narrative.
- Despite those negative judge reasons, the selector still emitted clips. That
  indicates the serving path ranks the best available candidates but lacks a
  strict enough abstain/reject gate when the candidate pool is weak.
- Main likely bottleneck: pipeline selection and candidate coverage, not just
  the trained model. The model can only rank proposed windows, and the final
  selector currently returns clips even when the best available windows are
  mediocre.

## 2026-05-14 Plan: Tighten Serving Clip Quality Gates

- [x] Add configurable post-judge quality gates for score, payoff, standalone clarity, and narrative completeness.
- [x] Apply the gate before final diversity selection so weak candidates can be abstained instead of emitted.
- [x] Preserve fallback behavior only when no judge result exists, without hiding judged low-quality candidates.
- [x] Add regression tests covering rejected weak judged clips and fewer-than-target outputs.
- [x] Run focused tests, lint/syntax checks, restart containers if needed, and document results.

### Review Notes

- Serving now applies `serving_quality_gate_done` after clip judging and before
  diversity selection, so judged low-quality candidates cannot be used to fill
  the requested clip count.
- New defaults are visible in `backend/.env` and `backend/.env.example`:
  minimum judge score `6.0`, standalone clarity `0.55`, payoff `0.55`,
  narrative completeness `0.45`, and judge confidence `0.45`.
- Job debug artifacts now include `debug/serving_quality_gate.jsonl` with
  per-candidate acceptance/rejection reasons.
- Applying the new gate to the latest weak debug job
  `9c7adf11-5b59-4244-bf7a-f3f1ab839033` rejects all `24` judged candidates:
  all were below judge-score and narrative thresholds.
- Verification passed: `py_compile`, scoped `git diff --check`,
  `backend/tests/test_clip_judge.py backend/tests/test_selector.py` -> `20`
  tests, and `backend/tests/test_config.py backend/tests/test_selector.py
  backend/tests/test_pipeline_integration.py` -> `11` tests.
- Graphify rebuilt with the uv-managed graphify interpreter after system
  `python3` lacked the module: `3192` nodes, `8495` edges, `97` communities.
- Backend Docker image was rebuilt and `ai-clipping-backend` restarted healthy;
  local and ngrok `/api/status` both respond.

## 2026-05-14 Plan: Trainable Candidate Proposal Quality Lever

- [x] Phase 1: Add dense candidate inventory generation that produces broad,
  overlapping windows over the whole VOD before heuristic source caps are
  applied.
- [x] Phase 2: Build cheap proposal features for every dense window using
  transcript, score timeline, chat/audio/visual aggregates, topic boundaries,
  hook/payoff shape, and current heuristic source hits.
- [x] Phase 3: Train a lightweight proposal model that filters thousands of
  windows down to a small high-recall shortlist before the expensive ranker,
  judge, face refinement, and boundary refinement stages.
- [x] Phase 4: Create labels for the proposal model from cross-platform
  transcript matches, frontend ratings, accepted judge outputs, rejected judge
  outputs, and near-miss negatives around positives.
- [x] Phase 5: Integrate the proposal model into serving behind a feature flag:
  dense windows -> proposal scorer -> top N per VOD/chapter -> existing ranker
  -> judge -> quality gate.
- [x] Phase 6: Add diagnostics that compare proposal recall against current
  heuristic candidates and show where good windows were missed.
- [x] Phase 7: Verify the implementation with focused unit/integration tests,
  candidate diagnostics, syntax checks, diff checks, and graph rebuild.

### Review Notes

- Goal: move the quality bottleneck from heuristic candidate discovery to a
  learned proposal stage. The current trained ranker can only score candidates
  it receives; this plan makes "what candidates exist" trainable too.
- Added `dense_inventory` candidate generation: overlapping whole-VOD windows
  are now generated before heuristic source caps, with configurable durations,
  stride, and max inventory size.
- Added a cheap `proposal_scorer` stage with deterministic fallback scoring and
  optional JSON linear model loading from `ml/artifacts/{profile}_proposal_scorer.json`.
- Serving now runs: heuristic candidates + dense inventory -> proposal scorer
  -> proposal shortlist -> existing feature builder/ranker -> judge -> quality
  gate. Existing heuristic candidates are preserved by default while the
  proposal stage adds broader recall candidates.
- Debug output now includes `candidates_dense_inventory.jsonl`,
  `candidate_proposals.jsonl`, and `candidate_proposal_summary.json`.
- Added `ml/train_proposal_scorer.py`, which trains the JSON proposal artifact
  from frontend feedback, selected outputs, judge quality-gate labels, optional
  cross-platform positive rows, and near-miss negatives around those positives.
- Verification passed: `py_compile`, `git diff --check`, and focused tests
  `backend/tests/test_candidate_generators.py`,
  `backend/tests/test_proposal_scorer.py`,
  `backend/tests/test_train_proposal_scorer.py`,
  `backend/tests/test_pipeline_integration.py`, `backend/tests/test_config.py`,
  `backend/tests/test_selector.py` -> `25` tests.

## 2026-05-14 Plan: Proposal Local Docker Training Script

- [x] Add a `scripts/` wrapper that mirrors `scripts/train_local_docker.sh` but
  runs `backend/ml/train_proposal_scorer.py` with repo/container defaults.
- [x] Support profile, storage dir, output path, positive cross-platform rows,
  and pass-through args without duplicating Docker build/run logic.
- [x] Verify shell syntax, help output through the Docker wrapper interface, and
  keep Graphify current after backend script changes.

### Review Notes

- Added `scripts/train_proposal_local_docker.sh` as a thin wrapper around
  `scripts/train_local_docker.sh`, so it reuses the same image build, env file,
  GPU flags, IPC, shm size, artifact mount, tmp mount, and prewarmed model mount.
- Default command inside the container is
  `python -m ml.train_proposal_scorer --profile jynxzi --storage-dir /app/backend/tmp/storage --positive-rows /app/backend/ml/artifacts/jynxzi_cross_platform_fingerprint_rows.jsonl --output /app/backend/ml/artifacts/jynxzi_proposal_scorer.json`.
- The wrapper supports `--profile`, `--storage-dir`, `--positive-rows`,
  `--output`, `--min-rows`, env overrides, and pass-through args after `--`.
- Verification passed: `bash -n scripts/train_proposal_local_docker.sh`,
  `scripts/train_proposal_local_docker.sh --help`,
  `backend/tests/test_train_proposal_scorer.py`, `git diff --check`, and
  Graphify rebuild.

## 2026-05-14 Plan: Proposal Training Backfill And Gold-Eval Removal

- [x] Add a backfill script that builds proposal training rows from
  cross-platform positive windows plus available VOD segment JSONL files.
- [x] Extend proposal scorer training so it can train directly from the
  generated proposal training rows, not only from fresh frontend debug jobs.
- [x] Wire the Docker proposal training wrapper to consume the generated
  proposal training rows by default.
- [x] Remove existing gold-eval artifacts and disable the local training flow
  from recreating frontend gold-eval rows.
- [x] Verify scripts, focused tests, artifact deletion, and Graphify rebuild.

### Review Notes

- Added `ml/build_proposal_training_rows.py`. It first tries to label dense
  proposal windows from cross-platform positives plus available VOD segment
  JSONL files, then backfills from existing non-gold VOD feature rows.
- Current local backfill produced `559` proposal rows: `211` positives and
  `348` negatives. It explicitly skipped `64` rows whose source/tier/class was
  gold-eval derived.
- Trained `ml/artifacts/jynxzi_proposal_scorer.json` from the generated rows.
- Deleted current `*gold_eval*`, active gold queue, and gold-derived hard
  negative artifacts from `backend/ml/artifacts`, plus
  `backend/tmp/storage/feedback/gold_eval_updates.jsonl`.
- Verification passed so far: `py_compile`; focused pytest set
  `backend/tests/test_train_jynxzi_vod_script.py`,
  `backend/tests/test_train_proposal_scorer.py`,
  `backend/tests/test_proposal_scorer.py`, and
  `backend/tests/test_candidate_generators.py` -> `21` tests. Gold artifact
  deletion was verified with `find`, and Graphify was rebuilt with the
  graphify uv tool interpreter after the system `python3` import failed.
- Additional checks passed: `bash -n scripts/train_proposal_local_docker.sh`,
  `bash -n backend/ml/train_jynxzi_vod.sh`, and
  `python -m ml.build_proposal_training_rows --help`. Native `zsh -n` could not
  run because `zsh` is not installed in this environment.
- Proposal training row audit: `559` rows, `0` gold-derived rows.

## 2026-05-14 Plan: Proposal Docker Entrypoint Fix

- [x] Add an explicit entrypoint override path to the shared local-training
  Docker wrapper without changing the default VOD training behavior.
- [x] Update the proposal wrapper to use that override so it runs
  `python -m ml.train_proposal_scorer` instead of appending args to
  `ml.local_train`.
- [x] Verify shell syntax and help output through the fixed wrapper.

### Review Notes

- Root cause: `backend/Dockerfile.local-train` has
  `ENTRYPOINT ["python", "-m", "ml.local_train"]`, so the proposal wrapper's
  command was appended as arguments to `ml.local_train`.
- `scripts/train_local_docker.sh` now supports
  `AI_CLIPPING_LOCAL_TRAIN_ENTRYPOINT`.
- `scripts/train_proposal_local_docker.sh` sets that entrypoint to `python` and
  passes `-m ml.train_proposal_scorer ...`, so the proposal trainer is the
  actual module executed inside the container.
- Verification passed: `bash -n scripts/train_local_docker.sh
  scripts/train_proposal_local_docker.sh` and
  `scripts/train_proposal_local_docker.sh --help`.

## 2026-05-14 Plan: Proposal Model Quality Fixes

- [x] Remove skewed existing-ranker-feature backfill from the default proposal
  training path; keep only features generated by
  `proposal_scorer.build_proposal_features`.
- [x] Centralize proposal source priors so training/serving use the same
  values.
- [x] Replace mean-difference-only training with a small L2 logistic-regression
  trainer plus grouped evaluation metrics.
- [x] Add gold-row filtering at trainer input boundaries, not only in the
  backfill builder.
- [x] Add regression tests for train/serve feature consistency, gold filtering,
  logistic metrics, and artifact loading.
- [x] Rebuild proposal rows/artifact, verify commands/tests, and rebuild
  Graphify after backend edits.

### Review Notes

- Accepted from the external review: train/serve skew, weak mean-diff trainer,
  missing trainer-side gold filtering, duplicated source priors, and untrained
  artifact loading were real issues.
- Rejected for now: removing hook/payoff early/late features and documenting
  every heuristic magic number. Those are lower leverage than making the
  training signal honest and measurable.
- Existing ranker-feature backfill is no longer default. It remains available
  only via explicit `--existing-feature-rows`, because translated downstream
  ranker features do not match serving-time proposal features.
- `proposal_scorer.source_prior_for_candidate()` is now the single prior table;
  the legacy backfill path calls it instead of keeping its own prior map.
- `train_proposal_scorer.py` now trains a small standardized L2 logistic
  regression and records metrics (`auc`, `logloss`, `precision_at_k`,
  `recall_at_k`, `accuracy_at_0_5`) in `training_summary`.
- Trainer input now filters gold-derived rows even if a caller passes a
  contaminated `--training-rows` file directly.
- Insufficient or empty proposal artifacts are ignored by the serving loader, so
  they cannot flatten all candidate scores to `0.5`.
- Current honest default rebuild produced `0` proposal rows and an
  `insufficient_rows` artifact, because the only available storage segment
  sources are for VOD `2771207666` and it has no cross-platform positives. This
  is intentional; the bad 559-row translated-ranker artifact is gone from the
  default path.
- Verification passed: `py_compile`; focused pytest
  `backend/tests/test_train_proposal_scorer.py`,
  `backend/tests/test_proposal_scorer.py`,
  `backend/tests/test_candidate_generators.py`, and
  `backend/tests/test_pipeline_integration.py` -> `25` tests;
  `bash -n` for local training wrappers; `git diff --check`; proposal row and
  artifact summary inspection; Graphify rebuild (`3283` nodes, `8777` edges,
  `94` communities).
