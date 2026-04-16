# Task Plan

## 2026-04-17 Speedup: Quality-Safe Training Pipeline

- [x] Inspect the training hot path to separate quality-safe speedups from quality-risky shortcuts.
- [x] Parallelize per-window modality extraction in the VOD training dataset build without changing feature logic.
- [x] Add targeted regression coverage for the new extraction helper.
- [x] Run targeted verification, rebuild graphify metadata, and document the result.

## 2026-04-16 Review: ML + Google-Cloud-Training

- [x] Read Graphify overview for repository shape and core ML/training nodes.
- [x] Inspect the main ML training entrypoints and dataset-build pipeline.
- [x] Inspect the Vertex/Google-Cloud wrapper, submit helper, and deployment docs.
- [x] Cross-check critical workflow assumptions against tests and artifact paths.
- [x] Deliver a findings-first review with severity, file/line references, and residual risks.

### Review Notes

- `train_jynxzi_vod.sh` skips retraining based only on label-vs-artifact mtimes, which can cause new Vertex container versions to reuse old downloaded artifacts without rebuilding datasets or models.
- The short-form backfill path trains into the same `jynxzi_ranker.json` artifact namespace as the primary VOD workflow, so a failed retry can leave the shared GCS prefix with the wrong model family.
- `auto_train_short_form.py` uses a VOD-first branch that bypasses `train_vod_ranker._train_models_for_dataset()`, which means it skips dataset quality gates and boundary-ranker training despite building a VOD dataset.
- Verification during review: `backend/.venv/bin/python -m pytest backend/tests/test_auto_train_short_form.py backend/tests/test_train_vod_ranker.py -q` passed (`22 passed`).
- There is currently no dedicated test coverage for `backend/ml/vertex_train.py` or `scripts/submit_vertex_training.py`.

## 2026-04-16 Fixes: ML + Google-Cloud-Training

- [x] Inspect the exact code paths behind the three review findings.
- [x] Add a workflow-state guard so stale downloaded artifacts do not suppress retraining/evaluation after code or config changes.
- [x] Isolate short-form backfill artifacts from the primary `jynxzi` VOD artifact namespace.
- [x] Route VOD-first training in `auto_train_short_form.py` through `train_vod_ranker._train_models_for_dataset()`.
- [x] Add regression coverage for the new workflow-state logic and the unified VOD training path.
- [x] Run targeted verification, rebuild graphify metadata, and document results.

### Fix Notes

- `backend/ml/training_workflow_state.py` now fingerprints the effective Jynxzi VOD workflow from code, runtime knobs, and date/flag inputs, and the shell wrapper uses that state instead of raw mtimes alone when deciding whether to retrain or reevaluate.
- `backend/ml/train_jynxzi_vod.sh` now writes `jynxzi_training_workflow_state.json`, so a Vertex job that downloads older artifacts still retrains when the code path or runtime configuration changed.
- The short-form backfill now writes its dataset to the expected `jynxzi_training_rows.jsonl` path but trains any temporary short-form artifact into `backend/tmp/short_form_backfill_artifacts/`, avoiding collisions with `jynxzi_ranker.json`.
- `backend/ml/auto_train_short_form.py` now routes VOD-first training through `train_vod_ranker._train_models_for_dataset()`, restoring dataset quality gates and boundary-ranker training in that path.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_auto_train_short_form.py backend/tests/test_train_vod_ranker.py backend/tests/test_training_workflow_state.py -q` passed (`26 passed`).
- Verification: `zsh -n backend/ml/train_jynxzi_vod.sh` passed.
- Verification: `backend/.venv/bin/python -m py_compile backend/ml/auto_train_short_form.py backend/ml/training_workflow_state.py backend/ml/train_vod_ranker.py` passed.
- Verification: graphify metadata was rebuilt after the code changes.

- [x] Rebuild and push the updated Vertex training image with the logging changes.
- [x] Cancel the currently running old Vertex training job.
- [x] Submit the same training run again in `europe-west3` and verify the new job was accepted.

- [x] Identify the main sources of Vertex training log noise and missing runtime attribution.
- [x] Reduce non-actionable warning spam while keeping download progress untouched.
- [x] Add explicit runtime log lines for Whisper, sentence-transformers, semantic-provider fallback, and vision backends.
- [x] Run targeted verification, rebuild graphify metadata, and document the review.

- [x] Confirm why the Vertex training container falls back to Haar instead of MediaPipe/DeepFace.
- [x] Patch the Vertex training image so it installs the vision dependency set used by `video_features`.
- [x] Run targeted verification, rebuild graphify metadata, and document the review.

- [x] Stop skipping short-form clips on transient network failures during training downloads.
- [x] Classify temporary downloader network errors separately from true clip unavailability.
- [x] Pause and retry the same clip download with backoff instead of marking it `skipped_unavailable`.
- [x] Add regression tests for downloader network classification and collection pause/retry behavior.
- [x] Run targeted verification, rebuild graphify metadata, and document the review.

- [x] Simplify clip feedback input to a single rating (`good` / `medium` / `bad`) plus optional comment.
- [x] Replace frontend feedback controls and payload types to match the simplified schema.
- [x] Replace backend feedback models/storage fields and map the new rating into training labels.
- [x] Update regression tests for feedback persistence and training export.
- [x] Run targeted verification, rebuild graphify metadata, and document the review.

- [x] Add a VOD-memory layer that tracks topic, speaker, setup/payoff, callback, and open-loop spans across a stream.
- [x] Retrieve the most relevant prior VOD spans per clip candidate and expose them as structured context instead of only local neighbors.
- [x] Extend feature building with memory-aware context scores such as setup depth, callback strength, payoff-match, and memory dependency.
- [x] Feed structured VOD memory into boundary refinement so clip starts keep the minimum required setup while staying concise.
- [x] Export memory-aware training features and verify the ranker/training paths accept the expanded schema.
- [x] Add regression tests for callback-heavy and long-setup clips across memory, feature, analyzer, and export paths.
- [x] Tighten Twitch clip-region merging so dense clip chains do not over-merge into giant windows.
- [x] Preserve strong positive Twitch clusters by shrinking oversized regions around the strongest anchor before marking them ambiguous.
- [x] Add regression tests for tighter merge gaps and strong-cluster shrink behavior.
- [x] Run targeted Twitch-labeling tests, rebuild graphify metadata, and document the review.
- [x] Review the current job lifecycle and upload session flow.
- [x] Add regression tests for long-running job retention and upload session binding.
- [x] Fix job eviction so active jobs are not dropped after two hours and terminal jobs retain for two hours after completion.
- [x] Fix uploaded-file submissions to bind and renew client sessions like URL submissions.
- [x] Run backend tests, frontend lint, and rebuild graphify metadata.
- [x] Extend job and feedback schemas for prompt-/platform-aware clip strategy and richer publish outcomes.
- [x] Add goal-aware candidate generation, dynamic trend memory, and platform-aware ranking signals.
- [x] Expand refined moments into platform/style variants with projected hook and retention metadata.
- [x] Harden router and integration tests for the new strategy-aware upload/pipeline flow.
- [x] Re-run backend tests, frontend lint, and rebuild graphify metadata after the roadmap slice.
- [x] Build an eval dataset artifact from exported training rows with bucketed metadata.
- [x] Add a pipeline evaluation CLI for baseline recall/top-k analysis.
- [x] Extend ranker training/evaluation to use richer outcome labels, sample weights, and feature reporting.
- [x] Gate trained-ranker v2 in runtime with explicit debug artifacts for heuristic/model/blended scores.
- [x] Add regression coverage for eval dataset building, evaluation CLI helpers, and runtime ranker fallback.
- [x] Run backend verification, rebuild graphify metadata, and document the review.
- [x] Align `backend/ml/train_jynxzi_vod.sh` with the new eval/ranker architecture and rollout guidance.
- [x] Add automatic short-form backfill and retry logic for one-class Jynxzi VOD training runs.

- [x] Make ranker eval report whether the trained artifact was actually usable or rejected.
- [x] Gate Jynxzi rollout recommendations on usable artifacts instead of heuristic fallback reports.
- [x] Enforce minimum VOD-training dataset quality before ranker training starts.
- [x] Dedupe overlapping positive proxy windows down to one canonical ranker example per Twitch clip cluster.
- [x] Add regression tests for artifact rejection reporting, dataset quality gates, and positive dedupe behavior.
- [x] Run targeted backend verification, shell syntax checks, rebuild graphify metadata, and document the review.
- [x] Add a Vertex AI training entrypoint that wraps the existing Jynxzi VOD training flow and syncs artifacts to GCS.
- [x] Make the Twitch VOD dataset build choose its LLM provider from environment instead of a hardcoded local Ollama default.
- [x] Add a Vertex custom-training Dockerfile, a submit helper, and deployment docs for the cloud workflow.
- [x] Run syntax/help checks for the new Vertex scripts, rebuild graphify metadata, and document the review.
- [x] Add a `.dockerignore` so the Vertex container build excludes local virtualenvs, artifacts, caches, and frontend build outputs.
- [x] Build the Vertex training image locally and fix the packaging path so torch resolves to CPU-only wheels instead of downloading CUDA runtimes.

# Review

- The VOD training dataset builder now starts Whisper transcription, per-second audio features, and per-second video features concurrently for each window, instead of waiting for them strictly one after another.
- The feature set and clip-finding logic are unchanged; only the scheduling of independent extraction steps was tightened, so this is a runtime optimization rather than a ranking-logic change.
- The dataset build now also supports bounded VOD-level parallelism, with `training_vod_workers` defaulting to `2` and deterministic result assembly preserved in input VOD order.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_build_twitch_training_dataset.py -q` passed (`22 passed`).
- Verification: `backend/.venv/bin/python -m py_compile backend/ml/build_twitch_training_dataset.py backend/config.py` passed.
- Verification: `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; ok=_rebuild_code(Path('.')); raise SystemExit(0 if ok else 1)"` passed and refreshed `graphify-out/`.

- Active jobs are no longer evicted solely because their original creation time is older than two hours.
- Terminal jobs now keep their full two-hour retention window starting from cancellation/failure/completion.
- Uploaded-file jobs now attach the same client session id used by URL jobs, so session expiry and polling renewal behave consistently.
- Jobs can now carry platform, goal, topic, tone, speaker, time-range, style, and variant-count strategy hints from both URL and upload flows.
- Candidate generation, feature building, ranking, and refinement now consume a shared normalized strategy, including dynamic feedback-backed trend memory and projected hook/retention scores.
- Final moments now expand into platform/style variants instead of a single packaging choice, and feedback/training export preserve the new publish outcome fields for future learning loops.
- Exported training rows now carry dataset-source, content-bucket, candidate-source, and final-variant metadata, which feeds a new eval-dataset builder and stronger reporting around the ranker.
- New ML tooling exists in `backend/ml/build_eval_dataset.py` and `backend/ml/evaluate_pipeline.py`, while `backend/ml/evaluate_ranker.py` now compares heuristic baseline vs. trained runtime behavior with metric deltas across buckets.
- The runtime ranker is now explicitly gated by `ENABLE_TRAINED_RANKER_V2`, and ranking debug artifacts now expose heuristic score, trained score, blend, and judge effects for each candidate.
- `backend/ml/train_jynxzi_vod.sh` now handles positional dates explicitly, runs the new eval-dataset/ranker/pipeline reports, and prints a concrete recommendation for whether `ENABLE_TRAINED_RANKER_V2` should be enabled after training.
- The Jynxzi training wrapper now detects one-class VOD datasets, generates a targeted `jynxzi` short-form backfill via `auto_train_short_form.py`, and retries VOD-first training once with the supplemental rows instead of failing immediately.
- Twitch clip clusters now merge on actual clip proximity instead of padded windows, which prevents dense clip chains from ballooning into one giant region.
- Oversized strong-positive clusters now shrink to the highest-view anchor clip instead of being downgraded wholesale to `ambiguous`, while large weak clusters still stay ambiguous.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_twitch_clip_labels.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_train_vod_ranker.py -q` passed (`42 passed`).
- A new `backend/services/vod_memory.py` layer now derives topic, speaker, setup, payoff, callback, and open-loop spans from transcripts and retrieves the most relevant prior spans per candidate.
- `feature_builder` now emits memory-aware scores including setup depth, callback strength, arc/payoff match, topic continuity, speaker clarity, memory dependency, story stage, and required setup lead; `feature_build_version` is now `6`.
- `analyzer` now attaches structured context metadata (`context_summary`, `active_topic`, `relevant_setup`, `why_this_is_a_payoff`, `minimum_required_setup_start`) and uses it to avoid trimming away required setup.
- Training/export provenance now marks whether a row includes VOD-memory-derived features, and the ranker/train-vod test paths accept the expanded feature schema.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_vod_memory.py backend/tests/test_feature_builder.py backend/tests/test_analyzer.py backend/tests/test_training_export.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_pipeline_integration.py -q` passed (`43 passed`).
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_ranker.py backend/tests/test_train_ranker.py backend/tests/test_train_vod_ranker.py -q` passed (`28 passed`).
- Verification: `zsh -n backend/ml/train_jynxzi_vod.sh` passed, `backend/.venv/bin/python -m pytest backend/tests/test_auto_train_short_form.py backend/tests/test_train_vod_ranker.py backend/tests/test_build_eval_dataset.py backend/tests/test_evaluate_pipeline.py backend/tests/test_ranker.py backend/tests/test_train_ranker.py -q` passed (`47 passed`), and graphify metadata was rebuilt.
- Clip feedback is now a minimal `good` / `medium` / `bad` rating plus optional comment in both frontend and backend, instead of the previous metrics-heavy form.
- Training export now derives outcome supervision directly from the simplified feedback rating, while `medium` falls back to weak pipeline supervision instead of pretending to be a strong outcome signal.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_feedback_store.py backend/tests/test_training_export.py backend/tests/test_trend_prior.py -q` passed (`10 passed`).
- Verification: `npm run lint` in `frontend/` passed.
- Short-form training downloads no longer treat transient network failures as `skipped_unavailable`; downloader network errors are now classified separately and pause the run with retry/backoff on the same clip.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_downloader.py backend/tests/test_collect_viral_clips.py backend/tests/test_auto_train_short_form.py -q` passed (`27 passed`).
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_train_vod_ranker.py -q` passed (`5 passed`).
- Ranker eval now reports explicit artifact status (`usable` / `rejected` / `missing`) and includes the rejection reason instead of silently presenting heuristic fallback as trained-runtime output.
- `backend/ml/train_jynxzi_vod.sh` now marks rollout advice as inconclusive whenever the eval report says the artifact was rejected or missing, instead of deriving an enable/disable recommendation from fallback scores.
- VOD-first ranker training now refuses to start when the primary dataset lacks enough independent signal: too few positive rows, too few unique VODs, too few positive clip clusters, or fewer than two ranking groups.
- Twitch VOD dataset building now collapses overlapping positive rows from the same `vod_id + clip_ids` cluster down to one canonical ranker example, preferring the primary proxy-refined row over supplemental overlaps.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_build_twitch_training_dataset.py backend/tests/test_train_vod_ranker.py backend/tests/test_train_ranker.py backend/tests/test_ranker.py -q` passed (`49 passed`).
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_build_eval_dataset.py -q` passed (`2 passed`).
- Verification: `zsh -n backend/ml/train_jynxzi_vod.sh` passed.
- Verification: `backend/.venv/bin/python backend/ml/evaluate_ranker.py --dataset backend/ml/artifacts/jynxzi_vod_eval_dataset.jsonl --profile jynxzi --artifact-dir backend/ml/artifacts` now reports `artifact.status = rejected` with `reason = positive_rows_below_minimum`.
- Verification: graphify metadata was rebuilt after the code changes.
- `backend/ml/vertex_train.py` now restores prior training artifacts from a GCS prefix, resolves Secret Manager credentials, runs the existing `train_jynxzi_vod.sh` flow inside the container, emits a compact JSON summary, and syncs updated artifacts back to GCS.
- `backend/ml/build_twitch_training_dataset.py` now reads `TRAINING_DATASET_LLM_PROVIDER` from the environment, so Vertex jobs can use `auto`, `groq`, or a remote Ollama endpoint instead of assuming local Ollama.
- A deployable Vertex custom-training scaffold now exists in `backend/Dockerfile.vertex-train`, `scripts/submit_vertex_training.py`, and `docs/vertex-training.md`.
- Verification: `backend/.venv/bin/python -m py_compile backend/ml/vertex_train.py scripts/submit_vertex_training.py` passed.
- Verification: `backend/.venv/bin/python backend/ml/vertex_train.py --help` passed.
- Verification: `backend/.venv/bin/python scripts/submit_vertex_training.py --help` passed.
- `.dockerignore` now keeps the Vertex build context small by excluding local virtualenvs, artifacts, caches, `frontend` build output, and transcripts.
- `backend/Dockerfile.vertex-train` now preinstalls `torch` from the CPU wheel index before installing `.[ml,vertex]`, which prevents the Linux container build from pulling multi-hundred-megabyte CUDA packages for a CPU training image.
- Verification: `docker build -f backend/Dockerfile.vertex-train -t ai-clipping-vertex-train:local .` passed after the CPU-only torch change.
- Verification: `docker images ai-clipping-vertex-train:local --format '{{.Repository}}:{{.Tag}} {{.Size}} {{.ID}}'` reports `ai-clipping-vertex-train:local 3.36GB b63163707885`.
- The Vertex training image now installs `.[ml,vision,vertex]` instead of omitting the vision stack, so `video_features` no longer falls back to Haar simply because MediaPipe, MTCNN, and DeepFace are missing from the container.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_video_features.py -q` passed (`3 passed`).
- Verification: `docker buildx build --platform linux/amd64 -f backend/Dockerfile.vertex-train -t ai-clipping-vertex-train:visionfix --load .` passed, and `docker run --rm --entrypoint python ai-clipping-vertex-train:visionfix -c "import mediapipe; from mtcnn import MTCNN; from deepface import DeepFace; ..."` confirmed `mediapipe_ok 0.10.33`, `mtcnn_ok True`, and `deepface_ok True`.
- Vertex training logs now suppress the repeated `Ignoring update for missing job ...` warnings by demoting those best-effort progress misses to debug level; this removes the noisiest spam without affecting actual download progress output.
- The dataset-build path now logs a one-line runtime summary with configured Whisper model, semantic-provider resolution/fallback chain, sentence-transformer family, face-detector backend, DeepFace status, and cache mode before VOD processing starts.
- Whisper and sentence-transformers now emit compact `*_init` / `*_runtime` lines instead of vague “loaded” messages, and video features now log one runtime backend summary rather than scattered fallback warnings.
- Verification: `backend/.venv/bin/python -m pytest backend/tests/test_job_manager.py backend/tests/test_build_twitch_training_dataset.py backend/tests/test_video_features.py -q` passed (`29 passed`).
- Verification: `backend/.venv/bin/python -m py_compile backend/job_manager.py backend/ml/build_twitch_training_dataset.py backend/services/transcriber.py backend/services/video_features.py backend/services/semantic_candidates.py` passed.
- The rebuilt Vertex training image was repushed to Artifact Registry and `latest` now points at digest `sha256:cbe4bd75acb2d2e7b359c2b79680684dead2cf72a1547b0b690b240b4b16468d`.
- The old running `europe-west4` job `1494314779790016512` was cancelled successfully.
- A fresh `europe-west3` custom job named `jynxzi-ranker-train-20260416-ew3-logfix` was submitted as `projects/545981913082/locations/europe-west3/customJobs/4705341594247626752` and entered `JOB_STATE_PENDING`.

## 2026-04-16 Fixes: Graphify Codex Skill Install

- [x] Reproduce `graphify install --platform codex` and inspect where it writes the Codex skill.
- [x] Confirm whether Codex discovery is blocked by stale install paths or missing skill metadata.
- [x] Patch the local Graphify package so the Codex install target uses `~/.codex/skills/graphify/SKILL.md`.
- [x] Mirror the already-installed Graphify skill into `~/.codex/skills/graphify/` for immediate pickup on this machine.
- [x] Verify the installed Graphify config and document the outcome.

### Review Notes

- `graphifyy` `0.4.1` shipped the Codex `install --platform` target as `~/.agents/skills/graphify/SKILL.md`, while this Codex client indexes home-local skills from `~/.codex/skills/`.
- The packaged `skill-codex.md` already contains the correct `trigger: /graphify`; the discovery failure was the stale destination path, not missing skill metadata.
- The local package file `/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/graphify/__main__.py` now points the Codex install target at `.codex/skills/graphify/SKILL.md`.
- Verification: `python3 -c "from graphify.__main__ import _PLATFORM_CONFIG; print(_PLATFORM_CONFIG['codex']['skill_dst'])"` now prints `.codex/skills/graphify/SKILL.md`.
- Verification: `~/.codex/skills/graphify/SKILL.md` exists with `trigger: /graphify` and matching `.graphify_version`.

## 2026-04-17 Diagnosis: Graphify Invocation In Codex

- [x] Re-check the installed Graphify skill under `~/.codex/skills/graphify/`.
- [x] Confirm whether the current Codex session sees the `graphify` skill at all.
- [x] Distinguish skill-install success from UI slash-command behavior.
- [x] Document the current limitation and the operator workaround.

### Review Notes

- `~/.codex/skills/graphify/SKILL.md` still exists and still declares `trigger: /graphify`; the local install path fix remains intact.
- This session exposes `graphify` in the available skill list, so the skill is being discovered by Codex even though the user cannot manually type `/graphify` in the input box.
- The remaining gap is therefore not skill installation but client interaction behavior: this Codex client can have the skill loaded without offering `/graphify` as a user-entered slash command.
- Operator workaround: request Graphify in plain language (for example, `nutze graphify auf .` or `run graphify on this repo`) and let the agent invoke the installed skill from the session context.

## 2026-04-17 Graphify Run: Repo Root With Requested Excludes

- [x] Check how Graphify detects and ignores files for this repo.
- [x] Measure the corpus for `.` with the requested exclusions `backend/tmp/` and `.venv/`.
- [x] Narrow the scope to a graphable subfolder if the root corpus remains too large.
- [x] Run Graphify for the approved scope and report the outputs.

### Review Notes

- Graphify already excludes hidden directories by default, so a root-level `.venv/` does not materially affect detection here.
- Excluding `backend/tmp/` removes the transient media/cache bulk, but the repo root still remains far above Graphify's practical full-run threshold.
- With the requested excludes applied, the corpus is still `17,632` files and about `21.9M` words.
- At the time of that measurement, a local Google Cloud SDK tree dominated the root corpus with `17,494` files; it has since been moved out of the repo to `~/google-cloud-sdk`.
- The user narrowed the run to `backend/`, and `backend/.graphifyignore` now excludes `tmp/` so the backend graph stays focused on application code instead of transient downloaded media.
- The resulting backend corpus is `114` files / `~79.9k` words (`111` code files, `3` documents, `0` videos after the exclude).
- Verification: `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; ok=_rebuild_code(Path('backend')); raise SystemExit(0 if ok else 1)"` succeeded and wrote `backend/graphify-out/graph.json` plus `backend/graphify-out/GRAPH_REPORT.md`.
- Verification: the rebuilt backend report summarizes `1243` nodes, `2128` edges, and `87` detected communities.
