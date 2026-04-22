# Graph Report - backend  (2026-04-22)

## Corpus Check
- 140 files · ~100,603 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1599 nodes · 2757 edges · 105 communities detected
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 170 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 67 edges
2. `Candidate` - 45 edges
3. `BuildTwitchTrainingDatasetTests` - 30 edges
4. `TrainRankerTests` - 28 edges
5. `train_from_rows()` - 27 edges
6. `TwitchClipLabelsTests` - 26 edges
7. `Ranker` - 22 edges
8. `AutoTrainShortFormTests` - 18 edges
9. `_build_candidate_features()` - 18 edges
10. `_window_candidate()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Load negative rows from a JSONL file. Defaults to synthetic data if available.` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Semantic candidate generation using sentence embeddings.  Uses sentence-transfor` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_candidates.py → backend/services/transcriber.py
- `Return a dict mapping category → list of segment indices that exceed the     sim` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_candidates.py → backend/services/transcriber.py
- `Remove jobs older than _JOB_TTL_SECONDS from the in-memory store.` --uses--> `JobStatus`  [INFERRED]
  backend/job_manager.py → backend/models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (91): build_frames(), ChatFrame, ChatSpike, _classify(), detect_spikes(), download_chat(), enrich_segments(), _is_missing_twitch_chat_replay() (+83 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (43): BaseModel, append_feedback_events(), append_feedback_snapshot(), append_gold_eval_snapshot(), _as_float(), build_pipeline_feedback(), CandidateFeedback, _creator_id_from_url() (+35 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (22): BaseSettings, _resolve_local_project_path(), Settings, _validate_storage_dir(), _build_context(), main(), _objective(), run_hpo() (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (18): AnalyzerApplyVariantSpecTests, AnalyzerBoundaryReasonTests, AnalyzerDedupMomentsTests, AnalyzerHelperTests, AnalyzerPairSelectionTests, AnalyzerValidationTests, _row(), TrainBoundaryRankerTests (+10 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (58): _artifact_health_summary(), _artifact_hpo_summary(), _best_hpo_params(), _bucketed_validation(), _calibrate_blend(), _candidate_is_promoted(), _candidate_source_mix(), _compute_scales() (+50 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (51): _blended_gap_duration(), build_label_rows(), _clamp_negative_duration(), _classify_clip_views(), _clip_merge_end(), _clip_window_bounds(), collect_clip_labels(), _collection_chunks() (+43 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (47): _absolute_candidate(), _apply_vod_context_features(), _boundary_perturbation_candidates(), build_dataset(), _build_dataset_parallel(), _build_rows_for_label(), _build_rows_for_vod(), _build_rows_for_vod_labels() (+39 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (3): _FakeLGBMRanker, _FakeMlflowSession, TrainRankerTests

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (41): _allowed_boundaries(), analyze(), _append_unique_reason(), _apply_variant_spec(), _boundary_at_or_after(), _boundary_at_or_before(), _boundary_selection_reason(), _build_context_summary() (+33 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (5): Ranker, score_candidates(), _FakeLightGBMModel, RankerTests, test_ranker_scores_with_lightgbm_ranker_artifact()

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (35): _adaptive_window(), _dense_coarse_score(), _dense_window_proposals(), _filter_candidates(), _generate_audio_burst_candidates(), generate_candidates(), _generate_chat_sync_candidates(), _generate_clean_reaction_candidates() (+27 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (33): _candidate_baselines(), _coerce_publish_timestamp(), _coerce_upload_date(), _dedupe_candidates(), _discover_seed(), _discover_seeds(), _extract_entry_url(), _extract_single_entry_metadata() (+25 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (27): _aria2c_available(), _base_ydl_opts(), _cache_dir(), _cache_path_for_url(), download(), DownloadCancelledError, DownloadPausedError, DownloadUnavailableError (+19 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (23): _check_gemini(), _check_groq(), _coerce_csv_list(), _coerce_optional_number(), _coerce_optional_text(), get_status(), _package_available(), _pipeline_component_statuses() (+15 more)

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (1): TwitchClipLabelsTests

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (26): _build_training_row(), _canonical_entry_url(), collect_rows(), cut_video_window(), cut_video_windows(), _cut_video_windows_batch(), _cut_video_windows_chunk(), _dedupe_entries() (+18 more)

### Community 16 - "Community 16"
Cohesion: 0.22
Nodes (19): _clean_mapping(), _coerce_scalar(), experiment_name(), finalize_training_run(), _git_sha(), _import_mlflow(), is_configured(), load_registered_ranker_artifact() (+11 more)

### Community 17 - "Community 17"
Cohesion: 0.15
Nodes (24): _calc_tempo(), _chunk_frame_count(), _deduplicate(), _diarize(), _dominant_speaker(), _filter_transcribable_chunks(), _fmt(), format_jsonl() (+16 more)

### Community 18 - "Community 18"
Cohesion: 0.2
Nodes (23): _audio_pop_score(), _build_candidate_features(), build_features(), _candidate_window(), _clamp01(), _context_dependency_score(), _empty_chapter_context(), _empty_memory_context() (+15 more)

### Community 19 - "Community 19"
Cohesion: 0.28
Nodes (21): _arc_payoff_match(), build_memory_spans(), _callback_spans(), candidate_memory_context(), _clamp01(), _context_summary(), _jaccard(), _memory_relevance() (+13 more)

### Community 20 - "Community 20"
Cohesion: 0.22
Nodes (19): _blend_score_maps(), _blend_value(), _boundary_artifact_is_usable(), _boundary_feature_row(), _boundary_model_is_usable(), _boundary_model_scores(), _heuristic_end_scores(), _heuristic_start_scores() (+11 more)

### Community 21 - "Community 21"
Cohesion: 0.11
Nodes (1): AutoTrainShortFormTests

### Community 22 - "Community 22"
Cohesion: 0.26
Nodes (18): _binary_label(), _build_end_examples(), _build_pair_examples(), _build_start_examples(), _clamp01(), _count_values(), _duration(), _duration_prior() (+10 more)

### Community 23 - "Community 23"
Cohesion: 0.25
Nodes (18): _aggregate_score(), _as_float(), _clamp01(), _dynamic_scores(), _empty_result(), _extract_feedback_texts(), _feedback_row_labels(), _feedback_row_score() (+10 more)

### Community 24 - "Community 24"
Cohesion: 0.16
Nodes (17): AudioFeatureFrame, _best_k(), diarize_speakers(), enrich_segments(), extract(), LaughterDetector, Per-second audio feature extraction — runs in parallel with Whisper transcriptio, Load audio and return one AudioFeatureFrame per second.      Expects 16 kHz mono (+9 more)

### Community 25 - "Community 25"
Cohesion: 0.23
Nodes (18): build_variant_specs(), candidate_in_time_range(), _clean_text(), _coerce_float(), _coerce_list(), _coerce_variant_count(), infer_goal_family(), _normalize_platform() (+10 more)

### Community 26 - "Community 26"
Cohesion: 0.15
Nodes (2): TrainVodRankerTests, TrainVodRankerWorkflowStateTests

### Community 27 - "Community 27"
Cohesion: 0.11
Nodes (2): DownloaderTests, _FakeYoutubeDL

### Community 28 - "Community 28"
Cohesion: 0.25
Nodes (17): _axes_to_score(), _blend_scores(), _build_context_summary(), _call_gemini(), _call_gemini_multimodal(), _call_groq(), _call_llm_judge(), _call_ollama() (+9 more)

### Community 29 - "Community 29"
Cohesion: 0.26
Nodes (15): copy_directory(), copy_file(), ensure_parent(), is_uri(), materialize_directory(), materialize_file(), normalize_uri(), path_exists() (+7 more)

### Community 30 - "Community 30"
Cohesion: 0.28
Nodes (15): _as_float(), _boundary_training_modules(), build_composite_dataset(), _candidate_source_mix(), dataset_summary(), _filter_short_form_rows(), _load_workflow_state(), main() (+7 more)

### Community 31 - "Community 31"
Cohesion: 0.26
Nodes (15): find_semantic_candidates(), _get_model(), _get_pattern_embeddings(), _has_sentence_transformer_metadata(), _has_transformer_weights(), _link_snapshot_contents(), _local_sentence_transformer_model_dir(), _materialize_local_sentence_transformer_model() (+7 more)

### Community 32 - "Community 32"
Cohesion: 0.3
Nodes (13): _annotated_eval_row(), build_eval_dataset(), _content_bucket(), _count_values(), _dataset_role(), eval_dataset_summary(), _eval_group(), _has_outcome_feedback() (+5 more)

### Community 33 - "Community 33"
Cohesion: 0.21
Nodes (13): _audience_signal(), build_score_timeline(), extract_candidates(), _hook_signal(), _payoff_signal(), Pre-scoring and candidate window extraction.  Runs before the LLM analysis step, Measure the emotional arc quality of a candidate clip window.      Supports mult, Compute a composite hype_score for each segment and store it in-place.      Scor (+5 more)

### Community 34 - "Community 34"
Cohesion: 0.28
Nodes (12): _format_mmss(), main(), _platform_from_url(), _preview_url(), _queue_id(), Generate a labeling queue for a gold evaluation set.  Reads a manifest of (job_i, _read_jsonl(), _read_manifest() (+4 more)

### Community 35 - "Community 35"
Cohesion: 0.17
Nodes (1): CollectViralClipsTests

### Community 36 - "Community 36"
Cohesion: 0.18
Nodes (9): get_content_type_for_streamer(), get_effective_heuristic_weights(), _init(), _load_custom_profiles(), Clip profile configuration.  Built-in profiles are defined in CLIP_PROFILES.  Cu, Merge profiles from a JSON file into CLIP_PROFILES (in-place)., Load custom profiles at import time (deferred to avoid circular imports)., Return the content-type profile for a known streamer, or None if unknown. (+1 more)

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (3): _FakeStudy, _FakeTrial, HpoRankerTests

### Community 38 - "Community 38"
Cohesion: 0.18
Nodes (2): _FakeSentenceTransformer, SemanticCandidatesTests

### Community 39 - "Community 39"
Cohesion: 0.36
Nodes (10): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), _quality_score() (+2 more)

### Community 40 - "Community 40"
Cohesion: 0.2
Nodes (1): CandidateGeneratorTests

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
Cohesion: 0.22
Nodes (4): BuildDatasetComponentEnvTests, CollectComponentEnvTests, PromoteComponentEnvTests, TrainRankerComponentEnvTests

### Community 45 - "Community 45"
Cohesion: 0.42
Nodes (8): _as_optional_float(), import_gold_eval_labels(), main(), _matching_sidecar(), _parse_gold_grade(), _read_csv(), _read_jsonl(), _write_jsonl()

### Community 46 - "Community 46"
Cohesion: 0.39
Nodes (8): _clamp(), main(), _make_negative(), _make_positive(), Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro, Strong viral clip: early hook, high hype, clear arc, audience confirmed., Weak clip: late or absent hook, low energy, no arc, no audience signal., _uniform()

### Community 47 - "Community 47"
Cohesion: 0.61
Nodes (8): _bucket_metrics(), evaluate_pipeline_rows(), _group_recall_at_k(), _group_rows(), main(), _mean_reciprocal_rank(), _top1_positive_rate(), _top_groups()

### Community 48 - "Community 48"
Cohesion: 0.42
Nodes (8): _as_float(), build_transcript_fields(), _clip_window(), enrich_reference_row(), main(), _materialize_reference_clip(), _read_jsonl(), _transcribe_reference_clip()

### Community 49 - "Community 49"
Cohesion: 0.42
Nodes (8): build_boundary_features(), _build_feature(), _duration_prior(), _hook_signal(), _lexical_break(), _mid_thought_risk(), _reaction_signal(), _segments_between()

### Community 50 - "Community 50"
Cohesion: 0.44
Nodes (8): _content_bucket(), _dataset_source(), export_training_rows(), _feature_provenance(), _has_real_outcome_feedback(), _read_jsonl(), _resolved_feedback_label(), _should_export_feedback()

### Community 51 - "Community 51"
Cohesion: 0.25
Nodes (1): TranscriberProgressTests

### Community 52 - "Community 52"
Cohesion: 0.25
Nodes (1): ClipJudgeTests

### Community 53 - "Community 53"
Cohesion: 0.54
Nodes (7): _bucket_key(), _bucket_metrics(), _empty_report(), _group_metrics(), main(), _metric_delta(), _read_jsonl()

### Community 54 - "Community 54"
Cohesion: 0.39
Nodes (5): _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), _sample_id()

### Community 55 - "Community 55"
Cohesion: 0.29
Nodes (1): TrainingWorkflowStateTests

### Community 56 - "Community 56"
Cohesion: 0.29
Nodes (1): GoldEvalWorkflowTests

### Community 57 - "Community 57"
Cohesion: 0.29
Nodes (1): TrainingExportTests

### Community 58 - "Community 58"
Cohesion: 0.29
Nodes (1): ExportFrontendGoldEvalTests

### Community 59 - "Community 59"
Cohesion: 0.52
Nodes (6): _empty_similarity_features(), _load_artifact(), _neighbor_scores(), _read_jsonl(), score_candidate_window(), _window_texts()

### Community 60 - "Community 60"
Cohesion: 0.4
Nodes (2): ClipSimilarityTests, _FakeModel

### Community 61 - "Community 61"
Cohesion: 0.33
Nodes (1): BuildGoldEvalManifestTests

### Community 62 - "Community 62"
Cohesion: 0.6
Nodes (5): export_frontend_gold_eval(), main(), _read_jsonl(), _summary(), _write_jsonl()

### Community 63 - "Community 63"
Cohesion: 0.67
Nodes (5): _boundary_metrics(), _bucket_by_kind(), _has_boundary_reference(), main(), _read_jsonl()

### Community 64 - "Community 64"
Cohesion: 0.53
Nodes (4): annotate_gold_row(), annotate_gold_rows(), _gold_holdout_bucket(), _gold_holdout_key()

### Community 65 - "Community 65"
Cohesion: 0.4
Nodes (1): TranscriberChunkingTests

### Community 66 - "Community 66"
Cohesion: 0.4
Nodes (1): PeakDetectorTests

### Community 67 - "Community 67"
Cohesion: 0.4
Nodes (1): ClipStrategyTests

### Community 68 - "Community 68"
Cohesion: 0.7
Nodes (4): main(), _queue_reason(), _read_jsonl(), _should_queue_row()

### Community 69 - "Community 69"
Cohesion: 0.7
Nodes (4): rank_candidates(), select_diverse_ranked_candidates(), select_top_candidates(), _too_close_to_any()

### Community 70 - "Community 70"
Cohesion: 0.6
Nodes (3): detect_peaks(), _fallback_peaks(), _walk_base()

### Community 71 - "Community 71"
Cohesion: 0.7
Nodes (4): build_boundary_points(), _delta(), _ends_sentence(), _merge_boundary()

### Community 72 - "Community 72"
Cohesion: 0.5
Nodes (1): JobCancellationTests

### Community 73 - "Community 73"
Cohesion: 0.5
Nodes (1): ProfileConfigTests

### Community 74 - "Community 74"
Cohesion: 0.5
Nodes (1): EvaluateBoundaryRefinementTests

### Community 75 - "Community 75"
Cohesion: 0.5
Nodes (1): KeyframeSamplerTests

### Community 76 - "Community 76"
Cohesion: 0.5
Nodes (1): ChapterSegmenterTests

### Community 77 - "Community 77"
Cohesion: 0.5
Nodes (1): LocalTrainTests

### Community 78 - "Community 78"
Cohesion: 0.5
Nodes (1): AnnotationFlowTests

### Community 79 - "Community 79"
Cohesion: 0.5
Nodes (1): EnrichVectorReferenceTranscriptTests

### Community 80 - "Community 80"
Cohesion: 0.5
Nodes (1): ChatAnalyzerDownloadTests

### Community 81 - "Community 81"
Cohesion: 0.5
Nodes (1): BuildEvalDatasetTests

### Community 82 - "Community 82"
Cohesion: 0.83
Nodes (3): apply_env(), build_env(), resolve_secret_ref()

### Community 83 - "Community 83"
Cohesion: 0.83
Nodes (3): main(), _parse_args(), run_local_pipeline()

### Community 84 - "Community 84"
Cohesion: 0.83
Nodes (3): main(), _parse_args(), run()

### Community 85 - "Community 85"
Cohesion: 0.83
Nodes (3): main(), _parse_args(), run()

### Community 86 - "Community 86"
Cohesion: 0.83
Nodes (3): main(), _parse_args(), run()

### Community 87 - "Community 87"
Cohesion: 0.83
Nodes (3): main(), _parse_args(), run()

### Community 88 - "Community 88"
Cohesion: 0.83
Nodes (3): main(), _parse_args(), run()

### Community 89 - "Community 89"
Cohesion: 0.83
Nodes (3): main(), _parse_args(), run()

### Community 90 - "Community 90"
Cohesion: 0.67
Nodes (1): EvaluatePipelineTests

### Community 91 - "Community 91"
Cohesion: 0.67
Nodes (1): CandidateNormalizerTests

### Community 92 - "Community 92"
Cohesion: 0.67
Nodes (1): AuditRankerArtifactsTests

### Community 93 - "Community 93"
Cohesion: 0.67
Nodes (1): LocalShellPipelineTests

### Community 94 - "Community 94"
Cohesion: 0.67
Nodes (1): ScoreTimelineTests

### Community 95 - "Community 95"
Cohesion: 0.67
Nodes (1): PostProcessorTests

### Community 96 - "Community 96"
Cohesion: 1.0
Nodes (2): _group_metrics(), main()

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 98 - "Community 98"
Cohesion: 1.0
Nodes (2): Invoke-PythonStep(), Write-RunLog()

### Community 99 - "Community 99"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 100 - "Community 100"
Cohesion: 1.0
Nodes (2): audit_artifacts(), main()

### Community 101 - "Community 101"
Cohesion: 1.0
Nodes (2): build_boundary_pairs(), refine_candidate_boundaries()

### Community 102 - "Community 102"
Cohesion: 1.0
Nodes (1): CLI-first pipeline components.

### Community 103 - "Community 103"
Cohesion: 1.0
Nodes (0): 

### Community 104 - "Community 104"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **27 isolated node(s):** `Generate a labeling queue for a gold evaluation set.  Reads a manifest of (job_i`, `Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro`, `Strong viral clip: early hook, high hype, clear arc, audience confirmed.`, `Weak clip: late or absent hook, low energy, no arc, no audience signal.`, `CLI-first pipeline components.` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 102`** (2 nodes): `__init__.py`, `CLI-first pipeline components.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 103`** (2 nodes): `diagnose_dataset.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 104`** (2 nodes): `audio_extractor.py`, `extract()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RichSegment` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 8`, `Community 10`, `Community 17`, `Community 31`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `Candidate` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 8`, `Community 9`, `Community 10`, `Community 15`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Why does `_PlatformUnavailableError` connect `Community 11` to `Community 12`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `RichSegment` (e.g. with `AnalyzerValidationTests` and `AnalyzerHelperTests`) actually correct?**
  _`RichSegment` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 43 inferred relationships involving `Candidate` (e.g. with `AnalyzerValidationTests` and `AnalyzerHelperTests`) actually correct?**
  _`Candidate` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `BuildTwitchTrainingDatasetTests` (e.g. with `Candidate` and `RichSegment`) actually correct?**
  _`BuildTwitchTrainingDatasetTests` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Generate a labeling queue for a gold evaluation set.  Reads a manifest of (job_i`, `Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro`, `Strong viral clip: early hook, high hype, clear arc, audience confirmed.` to the rest of the system?**
  _27 weakly-connected nodes found - possible documentation gaps or missing edges._