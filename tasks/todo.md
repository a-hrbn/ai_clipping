# Task Plan

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
