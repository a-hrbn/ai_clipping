# Graph Report - backend  (2026-05-19)

## Corpus Check
- 210 files · ~209,907 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3419 nodes · 9188 edges · 103 communities detected
- Extraction: 61% EXTRACTED · 39% INFERRED · 0% AMBIGUOUS · INFERRED: 3543 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 102|Community 102]]
- [[_COMMUNITY_Community 103|Community 103]]
- [[_COMMUNITY_Community 104|Community 104]]
- [[_COMMUNITY_Community 105|Community 105]]

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 278 edges
2. `write_text()` - 268 edges
3. `Candidate` - 189 edges
4. `read_text()` - 176 edges
5. `BuildTwitchTrainingDatasetTests` - 88 edges
6. `Ranker` - 86 edges
7. `CandidateFeatures` - 70 edges
8. `RankedCandidate` - 67 edges
9. `TrainRankerTests` - 56 edges
10. `run_pipeline()` - 50 edges

## Surprising Connections (you probably didn't know these)
- `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Load negative rows from a JSONL file. Defaults to synthetic data if available.` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `read_jsonl()` --calls--> `read_text()`  [INFERRED]
  backend/ml/row_identity.py → backend/ml/pipelines/io.py
- `_prepare_chapter_contexts()` --calls--> `chapter_for_time()`  [INFERRED]
  backend/services/feature_builder.py → backend/services/chapter_segmenter.py
- `Semantic candidate generation using sentence embeddings.  Uses sentence-transfor` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_candidates.py → backend/services/transcriber.py

## Hyperedges (group relationships)
- **Viral Clip Training Input Formats** — viral_clips_training_data, viral_clips_youtube_shorts_url, viral_clips_tiktok_url, viral_clips_csv_tsv_metadata [EXTRACTED 1.00]
- **Trend Prior Scoring Outputs** — readme_trend_prior_service, readme_trend_hook_score, readme_trend_topic_score, readme_trend_prior_score [EXTRACTED 1.00]
- **Trend Prior Update Inputs** — readme_update_process, readme_qualitative_review, readme_creator_domain_needs, readme_offline_eval_results [EXTRACTED 1.00]
- **Shorts Pattern Seed Domains** — readme_seed_phrases, readme_short_categories, readme_hook_patterns_manifest [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.01
Nodes (245): _allowed_boundaries(), analyze(), _append_unique_reason(), _apply_variant_spec(), _boundary_at_or_after(), _boundary_at_or_before(), _boundary_selection_reason(), _build_context_summary() (+237 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (156): audit_artifacts(), main(), build_manifest(), _job_summary(), main(), _read_job_map(), _read_jsonl(), _recover_vod_url() (+148 more)

### Community 2 - "Community 2"
Cohesion: 0.02
Nodes (201): _eval_group(), _aggregate_reference_metrics(), _calibrate_recall_precision_threshold(), calibrate_threshold(), _calibration_rows(), _filter_gate_from_calibration_preserving_bucket_routing(), main(), _read_jsonl() (+193 more)

### Community 3 - "Community 3"
Cohesion: 0.02
Nodes (175): _absolute_candidate(), _apply_vod_context_features(), _audio_cache_is_usable(), _audio_cache_metadata_path(), _audio_cache_payload(), _background_features_are_quiet(), _background_negative_training_row(), _background_quiet_assessment() (+167 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (96): BaseModel, append_feedback_events(), append_feedback_snapshot(), append_gold_eval_snapshot(), _apply_judge_fields(), _as_float(), build_pipeline_feedback(), CandidateFeedback (+88 more)

### Community 5 - "Community 5"
Cohesion: 0.03
Nodes (82): _score_with_artifact(), bucket_from_feature(), bucket_from_row(), bucket_from_source(), feature_value(), diagnose_holdout(), _diagnostic_csv_rows(), _empty_report() (+74 more)

### Community 6 - "Community 6"
Cohesion: 0.03
Nodes (102): _append_judge_snapshot(), DownloadUnavailableError, main(), _read_jsonl(), _await_clip_judge_or_fallback(), _await_with_optional_timeout(), _cleanup_audio_artifacts(), _cleanup_later() (+94 more)

### Community 7 - "Community 7"
Cohesion: 0.03
Nodes (80): _candidate_baselines(), _canonical_tiktok_video_url(), _coerce_publish_timestamp(), _coerce_upload_date(), _cooldown_after_short_form_yt_dlp_error(), _dedupe_candidates(), _discover_seed(), _discover_seeds() (+72 more)

### Community 8 - "Community 8"
Cohesion: 0.04
Nodes (69): _blended_gap_duration(), build_label_rows(), _clamp_negative_duration(), _classify_clip_views(), _clip_merge_end(), _clip_window_bounds(), collect_clip_labels(), collect_clip_labels_with_summary() (+61 more)

### Community 9 - "Community 9"
Cohesion: 0.04
Nodes (87): main(), _parse_args(), run(), _as_float(), _best_positive_match(), build_rows_from_existing_feature_rows(), _dedupe_paths(), _discover_segment_sources() (+79 more)

### Community 10 - "Community 10"
Cohesion: 0.04
Nodes (72): clip_ids(), is_positive_label(), _optional_float(), primary_identity_key(), read_jsonl(), _replace_text(), view_count(), write_json() (+64 more)

### Community 11 - "Community 11"
Cohesion: 0.05
Nodes (86): _adaptive_window(), _dense_coarse_score(), _dense_inventory_candidate(), _dense_inventory_ranges(), _dense_window_proposals(), _filter_candidates(), _generate_audio_burst_candidates(), _generate_chat_sync_candidates() (+78 more)

### Community 12 - "Community 12"
Cohesion: 0.06
Nodes (67): _active_bucket(), _active_reason(), build_queue(), _candidate_key(), _existing_gold_keys(), _feature_row(), main(), _priority_score() (+59 more)

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (61): score_window(), _tokens(), best_fingerprint_alignment(), _best_match_diagnostic_row(), Fingerprint, _fingerprint_from_row(), fingerprint_media_to_file(), _fingerprint_similarity() (+53 more)

### Community 14 - "Community 14"
Cohesion: 0.06
Nodes (49): _build_training_row(), _cacheable_entry(), _canonical_entry_url(), _collect_clip_modalities(), _collect_row_for_entry(), collect_rows(), cut_video_windows(), _cut_video_windows_batch() (+41 more)

### Community 15 - "Community 15"
Cohesion: 0.06
Nodes (60): _anthropic_prompt_blocks(), _anthropic_text_content(), _axes_to_score(), _base64_for_frame(), _batch_judge_prompt(), _blend_scores(), _build_context_summary(), _call_anthropic() (+52 more)

### Community 16 - "Community 16"
Cohesion: 0.06
Nodes (35): _ensure_short_form_media_cached(), _apply_short_form_sleep_opts(), _aria2c_available(), _base_ydl_opts(), _cache_dir(), _cache_entry_usable(), _cache_lock(), _cache_metadata_path() (+27 more)

### Community 17 - "Community 17"
Cohesion: 0.05
Nodes (46): _activity_category_scores(), _activity_score(), _annotate_chat_ema(), _attach_chat_activity(), build_frames(), _chat_dedupe_key(), ChatFrame, ChatSpike (+38 more)

### Community 18 - "Community 18"
Cohesion: 0.07
Nodes (42): _annotated_eval_row(), build_eval_dataset(), _content_bucket(), _count_values(), _dataset_role(), eval_dataset_summary(), _has_outcome_feedback(), _is_gold_eval_row() (+34 more)

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (38): build_boundary_points(), _delta(), _ends_sentence(), _merge_boundary(), build_boundary_features(), _build_feature(), _duration_prior(), _hook_signal() (+30 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (35): _base_candidate_id(), build_candidate_funnel_summary(), build_candidate_recall_report(), build_candidate_trace(), _candidate_ids(), _candidate_sample(), _compact_trace_sample(), _dedupe_strings() (+27 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (26): _as_float(), _cached_windows_by_vod(), _date_from_row(), _dedupe_vod_urls(), _discover_vod_urls(), _discover_vod_urls_from_label_manifest(), _distinct_vod_ids(), _enable_headless_video_feature_mode() (+18 more)

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (32): _auto_negative_row(), _auto_negative_sort_key(), build_auto_outcome_negative_rows(), _candidate_score(), _clamp_ratio(), _clip_ids(), _ensure_holdout_split_floor(), _holdout_rebalance_key() (+24 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (33): test_collect_training_rows_from_files_accepts_generated_rows(), test_collect_training_rows_from_files_filters_gold_eval_rows(), test_collect_training_rows_uses_feedback_and_quality_gate_labels(), test_collect_training_rows_uses_only_judged_quality_gate_labels(), test_existing_feature_row_backfill_excludes_gold_eval_rows(), _as_float(), _auc(), _classification_metrics() (+25 more)

### Community 24 - "Community 24"
Cohesion: 0.11
Nodes (28): AudioFeatureFrame, _best_k(), diarize_speakers(), enrich_segments(), extract(), _extract_frames(), extract_window_from_signal(), LaughterDetector (+20 more)

### Community 25 - "Community 25"
Cohesion: 0.19
Nodes (24): VodMemoryTests, MemorySpan, _arc_payoff_match(), build_memory_spans(), _callback_spans(), candidate_memory_context(), _clamp01(), _context_summary() (+16 more)

### Community 26 - "Community 26"
Cohesion: 0.07
Nodes (29): Candidate Clip, Clip-Worthy Topics, Cheap Deterministic Prior Rationale, High-Performing Short-Form Hooks, hook_patterns.json Manifest, Judge, Avoid Scraping Proprietary or Copyrighted Corpora Rationale, Ranker (+21 more)

### Community 27 - "Community 27"
Cohesion: 0.11
Nodes (14): _empty_similarity_features(), _load_artifact(), _neighbor_scores(), _read_jsonl(), score_candidate_window(), _window_texts(), _build_context(), main() (+6 more)

### Community 28 - "Community 28"
Cohesion: 0.13
Nodes (18): build_peak_candidates(), detect_peaks(), _fallback_peaks(), _walk_base(), _audience_signal(), build_score_timeline(), extract_candidates(), _hook_signal() (+10 more)

### Community 29 - "Community 29"
Cohesion: 0.22
Nodes (19): _clean_mapping(), _coerce_scalar(), experiment_name(), finalize_training_run(), _git_sha(), _import_mlflow(), is_configured(), load_registered_ranker_artifact() (+11 more)

### Community 30 - "Community 30"
Cohesion: 0.17
Nodes (18): _as_float(), _as_int(), _diagnosis_key(), export_hard_negatives(), _first_row_key(), _gold_row_index(), _hard_negative_row(), main() (+10 more)

### Community 31 - "Community 31"
Cohesion: 0.19
Nodes (17): analyze_confounds(), analyze_feature_correlation_audit(), _default_excluded_training_features(), _feature_value(), _is_penalty_score_pair(), _label_summary(), main(), _pearson() (+9 more)

### Community 32 - "Community 32"
Cohesion: 0.16
Nodes (14): _cached_media_by_size(), cleanup_backend_runtime_caches(), CleanupResult, _display_path(), _is_safe_to_remove_legacy_short_form_job_dir(), _is_safe_to_remove_twitch_job_dir(), main(), _media_exists_in_cache() (+6 more)

### Community 33 - "Community 33"
Cohesion: 0.15
Nodes (13): _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), read_annotations(), _sample_id(), main(), _queue_reason() (+5 more)

### Community 34 - "Community 34"
Cohesion: 0.25
Nodes (18): _aggregate_score(), _as_float(), _clamp01(), _dynamic_scores(), _empty_result(), _extract_feedback_texts(), _feedback_row_labels(), _feedback_row_score() (+10 more)

### Community 35 - "Community 35"
Cohesion: 0.24
Nodes (13): build_summary(), _cache_summary(), _feature_values(), _features(), _float(), _has_chat_signal(), main(), _parse_args() (+5 more)

### Community 36 - "Community 36"
Cohesion: 0.18
Nodes (9): BaseSettings, _configure_local_cache_env(), _configure_local_tool_path(), _is_writable_cache_dir(), _prepare_local_cache_path(), _resolve_local_project_path(), Settings, _validate_storage_dir() (+1 more)

### Community 37 - "Community 37"
Cohesion: 0.31
Nodes (5): CalibrateRankerThresholdTests, _reference_set(), _row(), _write_artifact(), _write_rows()

### Community 38 - "Community 38"
Cohesion: 0.26
Nodes (10): _as_float(), _candidate_summary(), _delta_summary(), main(), _mean(), _numeric_delta(), _read_json(), _read_jsonl() (+2 more)

### Community 39 - "Community 39"
Cohesion: 0.31
Nodes (12): _ends_sentence(), _extend_to_sentence_end(), _is_mid_thought(), _nearest_boundary_at_or_after(), _nearest_boundary_at_or_before(), _nearest_end_at_or_after(), _nearest_end_at_or_before(), _normalize_candidate() (+4 more)

### Community 40 - "Community 40"
Cohesion: 0.27
Nodes (9): _first_words(), main(), main_with_args_for_test(), _read_jsonl(), _sample_rows(), _twitch_link(), _write_csv(), InspectBackgroundNegativesTests (+1 more)

### Community 41 - "Community 41"
Cohesion: 0.36
Nodes (10): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), _quality_score() (+2 more)

### Community 42 - "Community 42"
Cohesion: 0.29
Nodes (8): _build_summary(), _configure_runtime_env(), main(), _parse_args(), _recommended_worker_plan(), _train_command(), WorkerPlan, LocalTrainTests

### Community 43 - "Community 43"
Cohesion: 0.33
Nodes (6): _boundary_metrics(), _bucket_by_kind(), _has_boundary_reference(), main(), _read_jsonl(), EvaluateBoundaryRefinementTests

### Community 44 - "Community 44"
Cohesion: 0.39
Nodes (8): _clamp(), main(), _make_negative(), _make_positive(), Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro, Strong viral clip: early hook, high hype, clear arc, audience confirmed., Weak clip: late or absent hook, low energy, no arc, no audience signal., _uniform()

### Community 45 - "Community 45"
Cohesion: 0.44
Nodes (8): _cache_key(), empty_scores(), _frame_times(), _load_model(), model_availability(), _score_candidate(), score_candidates(), _score_images_with_prompts()

### Community 46 - "Community 46"
Cohesion: 0.22
Nodes (9): Broad Pattern Groups, Creator or Domain Needs, Prefer Broad Patterns Rationale, Offline Eval Results, Highly Specific One-Off Phrases, Qualitative Review, Trend Prior Update Process, Trend Prior Validation Tests (+1 more)

### Community 47 - "Community 47"
Cohesion: 0.52
Nodes (6): build_manifest(), _diagnosis_error_counts(), main(), _read_csv(), _read_jsonl(), _write_csv()

### Community 48 - "Community 48"
Cohesion: 0.4
Nodes (1): Pipeline backend adapters.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (2): Invoke-PythonStep(), Write-RunLog()

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Words per second using the full segment duration as the denominator.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Words per second using the full segment duration as the denominator.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Words per second using the full segment duration as the denominator.

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Words per second using the full segment duration as the denominator.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Words per second using the full segment duration as the denominator.

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): Discover and cross-deduplicate manifest rows from a list of seeds.

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Train a separate artifact for each unique streamer in the seeds list.

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Discover and cross-deduplicate manifest rows from a list of seeds.

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Train a separate artifact for each unique streamer in the seeds list.

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Discover and cross-deduplicate manifest rows from a list of seeds.

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Train a separate artifact for each unique streamer in the seeds list.

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): Discover and cross-deduplicate manifest rows from a list of seeds.

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): Train a separate artifact for each unique streamer in the seeds list.

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): Discover and cross-deduplicate manifest rows from a list of seeds.

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Train a separate artifact for each unique streamer in the seeds list.

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): Discover and cross-deduplicate manifest rows from a list of seeds.

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): Train a separate artifact for each unique streamer in the seeds list.

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): Assign STREAMER / GUEST_N labels to each RichSegment using voice clustering.

### Community 86 - "Community 86"
Cohesion: 1.0
Nodes (1): Return k with the highest silhouette score over the given range.

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (1): Count consecutive silent whole-second frames ending immediately before seg_start

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (1): Words per second using the full segment duration as the denominator.

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 92 - "Community 92"
Cohesion: 1.0
Nodes (1): Words per second using the full segment duration as the denominator.

### Community 93 - "Community 93"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 94 - "Community 94"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 95 - "Community 95"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 96 - "Community 96"
Cohesion: 1.0
Nodes (1): Words per second using the full segment duration as the denominator.

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 98 - "Community 98"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 99 - "Community 99"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 100 - "Community 100"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 101 - "Community 101"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 102 - "Community 102"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

### Community 103 - "Community 103"
Cohesion: 1.0
Nodes (1): Return the speaker with the most overlap with [seg_start, seg_end].

### Community 104 - "Community 104"
Cohesion: 1.0
Nodes (1): Format segments as a transcript string.      Args:         segments: List of Ric

### Community 105 - "Community 105"
Cohesion: 1.0
Nodes (1): Return one JSON object per line, preserving all numerical features.

## Knowledge Gaps
- **154 isolated node(s):** `Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro`, `Strong viral clip: early hook, high hype, clear arc, audience confirmed.`, `Weak clip: late or absent hook, low energy, no arc, no audience signal.`, `Discover and cross-deduplicate manifest rows from a list of seeds.`, `Train a separate artifact for each unique streamer in the seeds list.` (+149 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 48`** (5 nodes): `Pipeline backend adapters.`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (3 nodes): `train_jynxzi_vod.ps1`, `Invoke-PythonStep()`, `Write-RunLog()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Words per second using the full segment duration as the denominator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Words per second using the full segment duration as the denominator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Words per second using the full segment duration as the denominator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Words per second using the full segment duration as the denominator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Words per second using the full segment duration as the denominator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `Discover and cross-deduplicate manifest rows from a list of seeds.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Train a separate artifact for each unique streamer in the seeds list.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Discover and cross-deduplicate manifest rows from a list of seeds.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Train a separate artifact for each unique streamer in the seeds list.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Discover and cross-deduplicate manifest rows from a list of seeds.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Train a separate artifact for each unique streamer in the seeds list.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `Discover and cross-deduplicate manifest rows from a list of seeds.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `Train a separate artifact for each unique streamer in the seeds list.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `Discover and cross-deduplicate manifest rows from a list of seeds.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Train a separate artifact for each unique streamer in the seeds list.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `Discover and cross-deduplicate manifest rows from a list of seeds.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `Train a separate artifact for each unique streamer in the seeds list.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `Assign STREAMER / GUEST_N labels to each RichSegment using voice clustering.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 86`** (1 nodes): `Return k with the highest silhouette score over the given range.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (1 nodes): `Count consecutive silent whole-second frames ending immediately before seg_start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (1 nodes): `Words per second using the full segment duration as the denominator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 92`** (1 nodes): `Words per second using the full segment duration as the denominator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 94`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 95`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 96`** (1 nodes): `Words per second using the full segment duration as the denominator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 97`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 98`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 99`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 100`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 101`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 102`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 103`** (1 nodes): `Return the speaker with the most overlap with [seg_start, seg_end].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 104`** (1 nodes): `Format segments as a transcript string.      Args:         segments: List of Ric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 105`** (1 nodes): `Return one JSON object per line, preserving all numerical features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RichSegment` connect `Community 0` to `Community 1`, `Community 3`, `Community 6`, `Community 9`, `Community 11`, `Community 14`, `Community 15`, `Community 17`, `Community 24`, `Community 25`, `Community 27`, `Community 28`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `write_text()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 12`, `Community 13`, `Community 14`, `Community 16`, `Community 17`, `Community 18`, `Community 19`, `Community 21`, `Community 22`, `Community 23`, `Community 24`, `Community 27`, `Community 30`, `Community 31`, `Community 32`, `Community 33`, `Community 35`, `Community 36`, `Community 37`, `Community 38`, `Community 40`, `Community 41`, `Community 42`, `Community 44`, `Community 45`, `Community 47`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `Candidate` connect `Community 0` to `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 39`, `Community 9`, `Community 11`, `Community 14`, `Community 20`, `Community 24`, `Community 25`, `Community 28`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 274 inferred relationships involving `RichSegment` (e.g. with `VodFeatureContext` and `WindowScopedSourcePlan`) actually correct?**
  _`RichSegment` has 274 INFERRED edges - model-reasoned connections that need verification._
- **Are the 263 inferred relationships involving `write_text()` (e.g. with `_write_json()` and `run_pipeline()`) actually correct?**
  _`write_text()` has 263 INFERRED edges - model-reasoned connections that need verification._
- **Are the 187 inferred relationships involving `Candidate` (e.g. with `_ShortFormMediaPrefetcher` and `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m`) actually correct?**
  _`Candidate` has 187 INFERRED edges - model-reasoned connections that need verification._
- **Are the 172 inferred relationships involving `read_text()` (e.g. with `_load_jobs()` and `_read_jsonl()`) actually correct?**
  _`read_text()` has 172 INFERRED edges - model-reasoned connections that need verification._