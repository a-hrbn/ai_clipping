# Graph Report - .  (2026-04-20)

## Corpus Check
- 137 files · ~474,056 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1572 nodes · 2743 edges · 86 communities detected
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 199 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 81 edges
2. `Candidate` - 49 edges
3. `BuildTwitchTrainingDatasetTests` - 36 edges
4. `TwitchClipLabelsTests` - 26 edges
5. `train_from_rows()` - 25 edges
6. `TrainRankerTests` - 23 edges
7. `Ranker` - 21 edges
8. `RankedCandidate` - 19 edges
9. `AutoTrainShortFormTests` - 18 edges
10. `_build_candidate_features()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Load negative rows from a JSONL file. Defaults to synthetic data if available.` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Post-processing of LLM-returned clip moments.  Two passes run after the LLM retu` --uses--> `RichSegment`  [INFERRED]
  backend/services/post_processor.py → backend/services/transcriber.py
- `Align each clip's start/end to real segment boundaries.      Start snapping:` --uses--> `RichSegment`  [INFERRED]
  backend/services/post_processor.py → backend/services/transcriber.py
- `Greedily keep only clips that are sufficiently spread apart in time.      Proces` --uses--> `RichSegment`  [INFERRED]
  backend/services/post_processor.py → backend/services/transcriber.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (88): Build a 2-3 sentence narrative summary of what happened before the clip window., Refine clip boundaries. Uses Gemini if configured, else falls back to Ollama., Ask Gemini Flash to refine clip boundaries via its OpenAI-compatible endpoint., Ask Ollama to refine clip boundaries., Regex fallback — used only when sentence-transformers is unavailable., Finds moments of rising tension before an explosion.      Two complementary sign, Finds twists, reveals, and information drops that reframe what came before., Finds genuine disagreement, hot-takes, and confrontation — goes well beyond (+80 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (27): ensureClientSessionId(), formatApiErrorDetail(), getJobStatus(), submitClipFeedback(), submitJob(), CandidateGeneratorTests, ChapterSegmenterTests, ClipSimilarityTests (+19 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (38): BaseModel, append_feedback_events(), append_feedback_snapshot(), append_gold_eval_snapshot(), _as_float(), build_pipeline_feedback(), CandidateFeedback, _creator_id_from_url() (+30 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (63): _absolute_candidate(), _apply_vod_context_features(), _boundary_perturbation_candidates(), build_dataset(), _build_dataset_parallel(), _build_rows_for_label(), _build_rows_for_vod(), _build_rows_for_vod_labels() (+55 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (37): BaseSettings, _resolve_local_project_path(), Settings, _validate_storage_dir(), _validate_training_vod_cache_dir(), _build_context(), main(), _objective() (+29 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (51): _blended_gap_duration(), build_label_rows(), _clamp_negative_duration(), _classify_clip_views(), _clip_merge_end(), _clip_window_bounds(), collect_clip_labels(), _collection_chunks() (+43 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (52): _artifact_health_summary(), _artifact_hpo_summary(), _best_hpo_params(), _bucketed_validation(), _calibrate_blend(), _candidate_is_promoted(), _compute_scales(), _count_values() (+44 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (9): _args(), BuildSummaryTests, ConfigureRuntimeEnvTests, DownloadUploadArtifactsTests, MainTests, ParseGcsUriTests, PositiveClusterKeyTests, ResolveSecretTests (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (5): Ranker, score_candidates(), _FakeLightGBMModel, RankerTests, test_ranker_scores_with_lightgbm_ranker_artifact()

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (36): _allowed_boundaries(), analyze(), _append_unique_reason(), _apply_variant_spec(), _boundary_at_or_after(), _boundary_at_or_before(), _boundary_selection_reason(), _build_context_summary() (+28 more)

### Community 10 - "Community 10"
Cohesion: 0.06
Nodes (1): BuildTwitchTrainingDatasetTests

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (33): _candidate_baselines(), _coerce_publish_timestamp(), _coerce_upload_date(), _dedupe_candidates(), _discover_seed(), _discover_seeds(), _extract_entry_url(), _extract_single_entry_metadata() (+25 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (30): _adaptive_window(), _dense_coarse_score(), _dense_window_proposals(), _filter_candidates(), _generate_audio_burst_candidates(), generate_candidates(), _generate_chat_sync_candidates(), _generate_clean_reaction_candidates() (+22 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (2): _FakeLGBMRanker, TrainRankerTests

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (1): TwitchClipLabelsTests

### Community 15 - "Community 15"
Cohesion: 0.15
Nodes (25): _build_training_row(), _canonical_entry_url(), collect_rows(), cut_video_window(), cut_video_windows(), _cut_video_windows_batch(), _dedupe_entries(), _feature_provenance() (+17 more)

### Community 16 - "Community 16"
Cohesion: 0.15
Nodes (22): _cache_dir(), _cache_path_for_url(), _copy_into_job(), download(), DownloadCancelledError, DownloadPausedError, DownloadUnavailableError, _evict_expired_cache() (+14 more)

### Community 17 - "Community 17"
Cohesion: 0.2
Nodes (23): _audio_pop_score(), _build_candidate_features(), build_features(), _candidate_window(), _clamp01(), _context_dependency_score(), _empty_chapter_context(), _empty_memory_context() (+15 more)

### Community 18 - "Community 18"
Cohesion: 0.28
Nodes (21): _arc_payoff_match(), build_memory_spans(), _callback_spans(), candidate_memory_context(), _clamp01(), _context_summary(), _jaccard(), _memory_relevance() (+13 more)

### Community 19 - "Community 19"
Cohesion: 0.14
Nodes (15): _audience_signal(), build_score_timeline(), extract_candidates(), _hook_signal(), _payoff_signal(), Pre-scoring and candidate window extraction.  Runs before the LLM analysis step, Measure the emotional arc quality of a candidate clip window.      Supports mult, Compute a composite hype_score for each segment and store it in-place.      Scor (+7 more)

### Community 20 - "Community 20"
Cohesion: 0.18
Nodes (16): _check_gemini(), _check_groq(), _coerce_csv_list(), _coerce_optional_number(), _coerce_optional_text(), get_status(), _package_available(), _pipeline_component_statuses() (+8 more)

### Community 21 - "Community 21"
Cohesion: 0.16
Nodes (19): _calc_tempo(), _chunk_frame_count(), _deduplicate(), _diarize(), _dominant_speaker(), _filter_transcribable_chunks(), _fmt(), format_jsonl() (+11 more)

### Community 22 - "Community 22"
Cohesion: 0.22
Nodes (19): _blend_score_maps(), _blend_value(), _boundary_artifact_is_usable(), _boundary_feature_row(), _boundary_model_is_usable(), _boundary_model_scores(), _heuristic_end_scores(), _heuristic_start_scores() (+11 more)

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (1): AutoTrainShortFormTests

### Community 24 - "Community 24"
Cohesion: 0.26
Nodes (18): _binary_label(), _build_end_examples(), _build_pair_examples(), _build_start_examples(), _clamp01(), _count_values(), _duration(), _duration_prior() (+10 more)

### Community 25 - "Community 25"
Cohesion: 0.25
Nodes (18): _aggregate_score(), _as_float(), _clamp01(), _dynamic_scores(), _empty_result(), _extract_feedback_texts(), _feedback_row_labels(), _feedback_row_score() (+10 more)

### Community 26 - "Community 26"
Cohesion: 0.16
Nodes (17): AudioFeatureFrame, _best_k(), diarize_speakers(), enrich_segments(), extract(), LaughterDetector, Per-second audio feature extraction — runs in parallel with Whisper transcriptio, Load audio and return one AudioFeatureFrame per second.      Expects 16 kHz mono (+9 more)

### Community 27 - "Community 27"
Cohesion: 0.23
Nodes (18): build_variant_specs(), candidate_in_time_range(), _clean_text(), _coerce_float(), _coerce_list(), _coerce_variant_count(), infer_goal_family(), _normalize_platform() (+10 more)

### Community 28 - "Community 28"
Cohesion: 0.2
Nodes (17): _apply_min_gap(), _ends_sentence(), enforce_diversity(), _extend_min_duration(), _find_clean_start(), _find_sentence_end(), _is_mid_thought(), Post-processing of LLM-returned clip moments.  Two passes run after the LLM retu (+9 more)

### Community 29 - "Community 29"
Cohesion: 0.25
Nodes (17): _axes_to_score(), _blend_scores(), _build_context_summary(), _call_gemini(), _call_gemini_multimodal(), _call_groq(), _call_llm_judge(), _call_ollama() (+9 more)

### Community 30 - "Community 30"
Cohesion: 0.16
Nodes (2): TrainVodRankerTests, TrainVodRankerWorkflowStateTests

### Community 31 - "Community 31"
Cohesion: 0.26
Nodes (14): _build_summary(), _configure_runtime_env(), _download_artifacts(), GcsLocation, main(), _parse_args(), _parse_gcs_uri(), _positive_cluster_key() (+6 more)

### Community 32 - "Community 32"
Cohesion: 0.3
Nodes (14): _as_float(), _boundary_training_modules(), build_composite_dataset(), dataset_summary(), _filter_short_form_rows(), _load_workflow_state(), main(), _positive_cluster_key() (+6 more)

### Community 33 - "Community 33"
Cohesion: 0.14
Nodes (2): DownloaderTests, _FakeYoutubeDL

### Community 34 - "Community 34"
Cohesion: 0.26
Nodes (13): find_semantic_candidates(), _get_model(), _get_pattern_embeddings(), _has_sentence_transformer_metadata(), _has_transformer_weights(), _link_snapshot_contents(), _local_sentence_transformer_model_dir(), _materialize_local_sentence_transformer_model() (+5 more)

### Community 35 - "Community 35"
Cohesion: 0.28
Nodes (12): _format_mmss(), main(), _platform_from_url(), _preview_url(), _queue_id(), Generate a labeling queue for a gold evaluation set.  Reads a manifest of (job_i, _read_jsonl(), _read_manifest() (+4 more)

### Community 36 - "Community 36"
Cohesion: 0.35
Nodes (11): build_eval_dataset(), _content_bucket(), _count_values(), _dataset_role(), eval_dataset_summary(), _eval_group(), _has_outcome_feedback(), main() (+3 more)

### Community 37 - "Community 37"
Cohesion: 0.18
Nodes (9): get_content_type_for_streamer(), get_effective_heuristic_weights(), _init(), _load_custom_profiles(), Clip profile configuration.  Built-in profiles are defined in CLIP_PROFILES.  Cu, Merge profiles from a JSON file into CLIP_PROFILES (in-place)., Load custom profiles at import time (deferred to avoid circular imports)., Return the content-type profile for a known streamer, or None if unknown. (+1 more)

### Community 38 - "Community 38"
Cohesion: 0.22
Nodes (3): _FakeStudy, _FakeTrial, HpoRankerTests

### Community 39 - "Community 39"
Cohesion: 0.36
Nodes (10): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), _quality_score() (+2 more)

### Community 40 - "Community 40"
Cohesion: 0.36
Nodes (10): _container_args(), _default_cache_base_uri(), _job_display_name(), _machine_spec(), main(), _parse_args(), _parse_gcs_uri(), _resolved_training_vod_checkpoint_gcs_uri() (+2 more)

### Community 41 - "Community 41"
Cohesion: 0.38
Nodes (9): build_manifest(), _job_summary(), main(), _read_job_map(), _read_jsonl(), _recover_vod_url(), _url_from_path(), _url_from_payload() (+1 more)

### Community 42 - "Community 42"
Cohesion: 0.33
Nodes (7): _boundary_score(), _label_for_chapter(), _lexical_distance(), _make_chapter(), _merge_label(), _merge_short_chapters(), segment_chapters()

### Community 43 - "Community 43"
Cohesion: 0.38
Nodes (9): _dedupe_candidates(), _extend_to_sentence_end(), _nearest_boundary_at_or_after(), _nearest_boundary_at_or_before(), _nearest_end_at_or_after(), _nearest_end_at_or_before(), _normalize_candidate(), normalize_candidates() (+1 more)

### Community 44 - "Community 44"
Cohesion: 0.36
Nodes (1): SubmitVertexTrainingTests

### Community 45 - "Community 45"
Cohesion: 0.39
Nodes (8): _clamp(), main(), _make_negative(), _make_positive(), Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro, Strong viral clip: early hook, high hype, clear arc, audience confirmed., Weak clip: late or absent hook, low energy, no arc, no audience signal., _uniform()

### Community 46 - "Community 46"
Cohesion: 0.61
Nodes (8): _bucket_metrics(), evaluate_pipeline_rows(), _group_recall_at_k(), _group_rows(), main(), _mean_reciprocal_rank(), _top1_positive_rate(), _top_groups()

### Community 47 - "Community 47"
Cohesion: 0.42
Nodes (8): _as_float(), build_transcript_fields(), _clip_window(), enrich_reference_row(), main(), _materialize_reference_clip(), _read_jsonl(), _transcribe_reference_clip()

### Community 48 - "Community 48"
Cohesion: 0.42
Nodes (8): build_boundary_features(), _build_feature(), _duration_prior(), _hook_signal(), _lexical_break(), _mid_thought_risk(), _reaction_signal(), _segments_between()

### Community 49 - "Community 49"
Cohesion: 0.44
Nodes (8): _content_bucket(), _dataset_source(), export_training_rows(), _feature_provenance(), _has_real_outcome_feedback(), _read_jsonl(), _resolved_feedback_label(), _should_export_feedback()

### Community 50 - "Community 50"
Cohesion: 0.46
Nodes (7): import_gold_eval_labels(), main(), _matching_sidecar(), _parse_gold_grade(), _read_csv(), _read_jsonl(), _write_jsonl()

### Community 51 - "Community 51"
Cohesion: 0.39
Nodes (5): _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), _sample_id()

### Community 52 - "Community 52"
Cohesion: 0.29
Nodes (1): TrainingExportTests

### Community 53 - "Community 53"
Cohesion: 0.62
Nodes (6): _bucket_key(), _bucket_metrics(), _group_metrics(), main(), _metric_delta(), _read_jsonl()

### Community 54 - "Community 54"
Cohesion: 0.52
Nodes (6): _empty_similarity_features(), _load_artifact(), _neighbor_scores(), _read_jsonl(), score_candidate_window(), _window_texts()

### Community 55 - "Community 55"
Cohesion: 0.33
Nodes (1): TrainingWorkflowStateTests

### Community 56 - "Community 56"
Cohesion: 0.33
Nodes (1): BuildGoldEvalManifestTests

### Community 57 - "Community 57"
Cohesion: 0.33
Nodes (1): ExportFrontendGoldEvalTests

### Community 58 - "Community 58"
Cohesion: 0.6
Nodes (5): export_frontend_gold_eval(), main(), _read_jsonl(), _summary(), _write_jsonl()

### Community 59 - "Community 59"
Cohesion: 0.67
Nodes (5): _boundary_metrics(), _bucket_by_kind(), _has_boundary_reference(), main(), _read_jsonl()

### Community 60 - "Community 60"
Cohesion: 0.4
Nodes (1): GoldEvalWorkflowTests

### Community 61 - "Community 61"
Cohesion: 0.4
Nodes (1): ClipStrategyTests

### Community 62 - "Community 62"
Cohesion: 0.7
Nodes (4): main(), _queue_reason(), _read_jsonl(), _should_queue_row()

### Community 63 - "Community 63"
Cohesion: 0.7
Nodes (4): rank_candidates(), select_diverse_ranked_candidates(), select_top_candidates(), _too_close_to_any()

### Community 64 - "Community 64"
Cohesion: 0.6
Nodes (3): detect_peaks(), _fallback_peaks(), _walk_base()

### Community 65 - "Community 65"
Cohesion: 0.7
Nodes (4): build_boundary_points(), _delta(), _ends_sentence(), _merge_boundary()

### Community 66 - "Community 66"
Cohesion: 0.5
Nodes (1): ProfileConfigTests

### Community 67 - "Community 67"
Cohesion: 0.5
Nodes (1): EvaluateBoundaryRefinementTests

### Community 68 - "Community 68"
Cohesion: 0.5
Nodes (1): AnnotationFlowTests

### Community 69 - "Community 69"
Cohesion: 0.5
Nodes (1): ChatAnalyzerDownloadTests

### Community 70 - "Community 70"
Cohesion: 0.5
Nodes (1): BuildEvalDatasetTests

### Community 71 - "Community 71"
Cohesion: 0.67
Nodes (1): EvaluatePipelineTests

### Community 72 - "Community 72"
Cohesion: 0.67
Nodes (1): AuditRankerArtifactsTests

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (2): _group_metrics(), main()

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (2): Invoke-PythonStep(), Write-RunLog()

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (2): audit_artifacts(), main()

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (2): build_boundary_pairs(), refine_candidate_boundaries()

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): ML utilities for training, annotation, and dataset preparation.

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (0): 

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (0): 

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (0): 

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (0): 

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (0): 

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **23 isolated node(s):** `Generate a labeling queue for a gold evaluation set.  Reads a manifest of (job_i`, `Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro`, `Strong viral clip: early hook, high hype, clear arc, audience confirmed.`, `Weak clip: late or absent hook, low energy, no arc, no audience signal.`, `ML utilities for training, annotation, and dataset preparation.` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 79`** (2 nodes): `__init__.py`, `ML utilities for training, annotation, and dataset preparation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (2 nodes): `diagnose_dataset.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (2 nodes): `audio_extractor.py`, `extract()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `next-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `tailwind.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `layout.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RichSegment` connect `Community 0` to `Community 1`, `Community 34`, `Community 4`, `Community 10`, `Community 19`, `Community 21`, `Community 28`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `Candidate` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 8`, `Community 10`, `Community 15`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `BuildTwitchTrainingDatasetTests` connect `Community 10` to `Community 0`, `Community 1`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 78 inferred relationships involving `RichSegment` (e.g. with `CandidateGeneratorTests` and `_FakeModel`) actually correct?**
  _`RichSegment` has 78 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `Candidate` (e.g. with `AnalyzerValidationTests` and `AnalyzerHelperTests`) actually correct?**
  _`Candidate` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `BuildTwitchTrainingDatasetTests` (e.g. with `Candidate` and `RichSegment`) actually correct?**
  _`BuildTwitchTrainingDatasetTests` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Generate a labeling queue for a gold evaluation set.  Reads a manifest of (job_i`, `Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro`, `Strong viral clip: early hook, high hype, clear arc, audience confirmed.` to the rest of the system?**
  _23 weakly-connected nodes found - possible documentation gaps or missing edges._