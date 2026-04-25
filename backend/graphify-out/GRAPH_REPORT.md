# Graph Report - backend  (2026-04-25)

## Corpus Check
- 152 files · ~121,281 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2042 nodes · 5284 edges · 38 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 2011 edges (avg confidence: 0.76)
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

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 178 edges
2. `write_text()` - 139 edges
3. `Candidate` - 106 edges
4. `read_text()` - 103 edges
5. `BuildTwitchTrainingDatasetTests` - 58 edges
6. `Ranker` - 50 edges
7. `train_from_rows()` - 40 edges
8. `CandidateFeatures` - 37 edges
9. `TrainRankerTests` - 35 edges
10. `RankedCandidate` - 35 edges

## Surprising Connections (you probably didn't know these)
- `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Load negative rows from a JSONL file. Defaults to synthetic data if available.` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Semantic candidate generation using sentence embeddings.  Uses sentence-transfor` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_candidates.py → backend/services/transcriber.py
- `Return a dict mapping category → list of segment indices that exceed the     sim` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_candidates.py → backend/services/transcriber.py
- `RichSegment` --uses--> `Post-processing of LLM-returned clip moments.  Two passes run after the LLM retu`  [INFERRED]
  backend/services/transcriber.py → backend/services/post_processor.py

## Hyperedges (group relationships)
- **Viral Clip Training Input Formats** — viral_clips_training_data, viral_clips_youtube_shorts_url, viral_clips_tiktok_url, viral_clips_csv_tsv_metadata [EXTRACTED 1.00]
- **Trend Prior Scoring Outputs** — readme_trend_prior_service, readme_trend_hook_score, readme_trend_topic_score, readme_trend_prior_score [EXTRACTED 1.00]
- **Trend Prior Update Inputs** — readme_update_process, readme_qualitative_review, readme_creator_domain_needs, readme_offline_eval_results [EXTRACTED 1.00]
- **Shorts Pattern Seed Domains** — readme_seed_phrases, readme_short_categories, readme_hook_patterns_manifest [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (134): extract(), build_frames(), ChatFrame, ChatSpike, _classify(), _comment_emotes(), _comment_text(), detect_spikes() (+126 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (136): _eval_group(), _row_identity_keys(), _split_leak_summary(), main(), diagnose_holdout(), _diagnostic_csv_rows(), _empty_report(), _error_type() (+128 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (112): _absolute_candidate(), _apply_vod_context_features(), _audio_cache_is_usable(), _audio_cache_metadata_path(), _audio_cache_payload(), _balance_vod_negative_rows(), _boundary_perturbation_candidates(), build_dataset() (+104 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (93): BaseModel, append_feedback_events(), append_feedback_snapshot(), append_gold_eval_snapshot(), _as_float(), build_pipeline_feedback(), CandidateFeedback, _creator_id_from_url() (+85 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (63): build_manifest(), _job_summary(), main(), _read_job_map(), _read_jsonl(), _recover_vod_url(), _url_from_path(), _url_from_payload() (+55 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (73): _allowed_boundaries(), analyze(), _append_unique_reason(), _apply_variant_spec(), _boundary_at_or_after(), _boundary_at_or_before(), _boundary_selection_reason(), _build_context_summary() (+65 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (87): _adaptive_window(), _dense_coarse_score(), _dense_window_proposals(), _filter_candidates(), _generate_audio_burst_candidates(), generate_candidates(), _generate_chat_sync_candidates(), _generate_clean_reaction_candidates() (+79 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (53): _blended_gap_duration(), build_label_rows(), _clamp_negative_duration(), _classify_clip_views(), _clip_merge_end(), _clip_window_bounds(), collect_clip_labels(), _collection_chunks() (+45 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (46): _build_training_row(), _cacheable_entry(), _canonical_entry_url(), _collect_clip_modalities(), _collect_row_for_entry(), collect_rows(), cut_video_windows(), _cut_video_windows_batch() (+38 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (38): find_semantic_candidates(), _get_model(), _get_pattern_embeddings(), _has_sentence_transformer_metadata(), _has_transformer_weights(), _link_snapshot_contents(), _local_sentence_transformer_model_dir(), _materialize_local_sentence_transformer_model() (+30 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (30): _boundary_metrics(), _bucket_by_kind(), _has_boundary_reference(), main(), _read_jsonl(), EvaluateBoundaryRefinementTests, TrainVodRankerTests, TrainVodRankerWorkflowStateTests (+22 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (32): _ensure_short_form_media_cached(), _aria2c_available(), _base_ydl_opts(), _cache_dir(), _cache_entry_usable(), _cache_lock(), _cache_metadata_path(), _cache_metadata_payload() (+24 more)

### Community 12 - "Community 12"
Cohesion: 0.08
Nodes (34): _candidate_baselines(), _coerce_publish_timestamp(), _coerce_upload_date(), _dedupe_candidates(), _discover_seed(), _discover_seeds(), _extract_entry_url(), _extract_single_entry_metadata() (+26 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (31): cut_video_window(), _as_float(), build_transcript_fields(), _clip_window(), enrich_reference_row(), main(), _materialize_reference_clip(), _read_jsonl() (+23 more)

### Community 14 - "Community 14"
Cohesion: 0.1
Nodes (36): main(), _parse_args(), run(), main(), _parse_args(), run(), main(), _parse_args() (+28 more)

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (37): _annotated_eval_row(), build_eval_dataset(), _content_bucket(), _count_values(), _dataset_role(), eval_dataset_summary(), _has_outcome_feedback(), _is_gold_eval_row() (+29 more)

### Community 16 - "Community 16"
Cohesion: 0.09
Nodes (36): build_boundary_points(), _delta(), _ends_sentence(), _merge_boundary(), build_boundary_features(), _build_feature(), _duration_prior(), _hook_signal() (+28 more)

### Community 17 - "Community 17"
Cohesion: 0.07
Nodes (25): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), _quality_score() (+17 more)

### Community 18 - "Community 18"
Cohesion: 0.1
Nodes (34): _active_bucket(), _active_reason(), build_queue(), _candidate_key(), _existing_gold_keys(), _feature_row(), main(), _priority_score() (+26 more)

### Community 19 - "Community 19"
Cohesion: 0.07
Nodes (29): Candidate Clip, Clip-Worthy Topics, Cheap Deterministic Prior Rationale, High-Performing Short-Form Hooks, hook_patterns.json Manifest, Judge, Avoid Scraping Proprietary or Copyrighted Corpora Rationale, Ranker (+21 more)

### Community 20 - "Community 20"
Cohesion: 0.22
Nodes (19): _clean_mapping(), _coerce_scalar(), experiment_name(), finalize_training_run(), _git_sha(), _import_mlflow(), is_configured(), load_registered_ranker_artifact() (+11 more)

### Community 21 - "Community 21"
Cohesion: 0.25
Nodes (22): MemorySpan, _arc_payoff_match(), build_memory_spans(), _callback_spans(), candidate_memory_context(), _clamp01(), _context_summary(), _jaccard() (+14 more)

### Community 22 - "Community 22"
Cohesion: 0.15
Nodes (18): _semantic_scores_for_candidates(), _call_groq(), _call_llm(), _call_ollama_semantic(), context_dependency_score(), contradiction_score(), payoff_clarity_score(), quoteability_score() (+10 more)

### Community 23 - "Community 23"
Cohesion: 0.16
Nodes (14): _cached_media_by_size(), cleanup_backend_runtime_caches(), CleanupResult, _display_path(), _is_safe_to_remove_legacy_short_form_job_dir(), _is_safe_to_remove_twitch_job_dir(), main(), _media_exists_in_cache() (+6 more)

### Community 24 - "Community 24"
Cohesion: 0.15
Nodes (13): _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), read_annotations(), _sample_id(), main(), _queue_reason() (+5 more)

### Community 25 - "Community 25"
Cohesion: 0.15
Nodes (19): AudioFeatureFrame, _best_k(), diarize_speakers(), enrich_segments(), extract(), _extract_frames(), extract_window_from_signal(), LaughterDetector (+11 more)

### Community 26 - "Community 26"
Cohesion: 0.25
Nodes (18): _aggregate_score(), _as_float(), _clamp01(), _dynamic_scores(), _empty_result(), _extract_feedback_texts(), _feedback_row_labels(), _feedback_row_score() (+10 more)

### Community 27 - "Community 27"
Cohesion: 0.22
Nodes (15): _as_float(), _as_int(), _diagnosis_key(), export_hard_negatives(), _gold_row_index(), _hard_negative_row(), main(), _read_csv() (+7 more)

### Community 28 - "Community 28"
Cohesion: 0.24
Nodes (13): build_summary(), _cache_summary(), _feature_values(), _features(), _float(), _has_chat_signal(), main(), _parse_args() (+5 more)

### Community 29 - "Community 29"
Cohesion: 0.19
Nodes (8): BaseSettings, _configure_local_cache_env(), _configure_local_tool_path(), _prepare_local_cache_path(), _resolve_local_project_path(), Settings, _validate_storage_dir(), ConfigPathTests

### Community 30 - "Community 30"
Cohesion: 0.26
Nodes (8): _build_summary(), _configure_runtime_env(), main(), _parse_args(), _recommended_worker_plan(), _train_command(), WorkerPlan, LocalTrainTests

### Community 31 - "Community 31"
Cohesion: 0.38
Nodes (9): _dedupe_candidates(), _extend_to_sentence_end(), _nearest_boundary_at_or_after(), _nearest_boundary_at_or_before(), _nearest_end_at_or_after(), _nearest_end_at_or_before(), _normalize_candidate(), normalize_candidates() (+1 more)

### Community 32 - "Community 32"
Cohesion: 0.39
Nodes (8): _clamp(), main(), _make_negative(), _make_positive(), Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro, Strong viral clip: early hook, high hype, clear arc, audience confirmed., Weak clip: late or absent hook, low energy, no arc, no audience signal., _uniform()

### Community 33 - "Community 33"
Cohesion: 0.25
Nodes (7): get_content_type_for_streamer(), _init(), _load_custom_profiles(), Clip profile configuration.  Built-in profiles are defined in CLIP_PROFILES.  Cu, Merge profiles from a JSON file into CLIP_PROFILES (in-place)., Load custom profiles at import time (deferred to avoid circular imports)., Return the content-type profile for a known streamer, or None if unknown.

### Community 34 - "Community 34"
Cohesion: 0.22
Nodes (9): Broad Pattern Groups, Creator or Domain Needs, Prefer Broad Patterns Rationale, Offline Eval Results, Highly Specific One-Off Phrases, Qualitative Review, Trend Prior Update Process, Trend Prior Validation Tests (+1 more)

### Community 35 - "Community 35"
Cohesion: 0.4
Nodes (3): audit_artifacts(), main(), AuditRankerArtifactsTests

### Community 36 - "Community 36"
Cohesion: 0.4
Nodes (1): Pipeline backend adapters.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (2): Invoke-PythonStep(), Write-RunLog()

## Knowledge Gaps
- **71 isolated node(s):** `Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro`, `Strong viral clip: early hook, high hype, clear arc, audience confirmed.`, `Weak clip: late or absent hook, low energy, no arc, no audience signal.`, `Discover and cross-deduplicate manifest rows from a list of seeds.`, `Train a separate artifact for each unique streamer in the seeds list.` (+66 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 36`** (5 nodes): `Pipeline backend adapters.`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (3 nodes): `train_jynxzi_vod.ps1`, `Invoke-PythonStep()`, `Write-RunLog()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `write_text()` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 23`, `Community 24`, `Community 27`, `Community 28`, `Community 30`, `Community 32`, `Community 35`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `RichSegment` connect `Community 0` to `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 9`, `Community 13`, `Community 17`, `Community 21`, `Community 22`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `read_text()` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 20`, `Community 24`, `Community 26`, `Community 27`, `Community 28`, `Community 30`, `Community 33`, `Community 35`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Are the 175 inferred relationships involving `RichSegment` (e.g. with `VodFeatureContext` and `ClipJudgeTests`) actually correct?**
  _`RichSegment` has 175 INFERRED edges - model-reasoned connections that need verification._
- **Are the 134 inferred relationships involving `write_text()` (e.g. with `run_pipeline()` and `_write_jsonl()`) actually correct?**
  _`write_text()` has 134 INFERRED edges - model-reasoned connections that need verification._
- **Are the 104 inferred relationships involving `Candidate` (e.g. with `_ShortFormMediaPrefetcher` and `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m`) actually correct?**
  _`Candidate` has 104 INFERRED edges - model-reasoned connections that need verification._
- **Are the 99 inferred relationships involving `read_text()` (e.g. with `_load_jobs()` and `_read_jsonl()`) actually correct?**
  _`read_text()` has 99 INFERRED edges - model-reasoned connections that need verification._