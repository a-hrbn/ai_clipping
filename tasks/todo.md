# Task Plan

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

# Review

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
- Verification: `zsh -n backend/ml/train_jynxzi_vod.sh` passed, `backend/.venv/bin/python -m pytest backend/tests/test_auto_train_short_form.py backend/tests/test_train_vod_ranker.py backend/tests/test_build_eval_dataset.py backend/tests/test_evaluate_pipeline.py backend/tests/test_ranker.py backend/tests/test_train_ranker.py -q` passed (`47 passed`), and graphify metadata was rebuilt.
