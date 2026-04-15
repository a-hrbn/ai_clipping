# Graph Report - .  (2026-04-15)

## Corpus Check
- 118 files · ~85,192 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1153 nodes · 2001 edges · 68 communities detected
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 160 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 69 edges
2. `Candidate` - 41 edges
3. `TwitchClipLabelsTests` - 21 edges
4. `BuildTwitchTrainingDatasetTests` - 20 edges
5. `Ranker` - 19 edges
6. `train_from_rows()` - 18 edges
7. `AutoTrainShortFormTests` - 17 edges
8. `_window_candidate()` - 17 edges
9. `normalize_strategy()` - 17 edges
10. `_build_candidate_features()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Load negative rows from a JSONL file. Defaults to synthetic data if available.` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Semantic candidate generation using sentence embeddings.  Uses sentence-transfor` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_candidates.py → backend/services/transcriber.py
- `Return a dict mapping category → list of segment indices that exceed the     sim` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_candidates.py → backend/services/transcriber.py
- `Score the five viral dimensions with an optional provider preference.` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_service.py → backend/services/transcriber.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (16): ensureClientSessionId(), getJobStatus(), submitJob(), BuildTwitchTrainingDatasetTests, CandidateGeneratorTests, ChapterSegmenterTests, EnrichVectorReferenceTranscriptTests, FeatureBuilderTests (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (68): Audio Extractor (FFmpeg WAV Export), Speaker Diarization (MFCC KMeans), Audio Segment Enrichment, Audio Feature Extraction (Librosa), AudioFeatureFrame Data Model, LaughterDetector (Acoustic Rule-Based Classifier), Candidate Deduplication (_dedupe_candidates), normalize_candidates Function (+60 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (36): BaseModel, append_feedback_events(), append_feedback_snapshot(), build_pipeline_feedback(), CandidateFeedback, _matching_moment(), _read_feedback_rows(), update_job_feedback() (+28 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (45): _blended_gap_duration(), build_label_rows(), _clamp_negative_duration(), _classify_clip_views(), _clip_merge_end(), _clip_window_bounds(), collect_clip_labels(), _collection_chunks() (+37 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (40): _bucketed_validation(), _compute_scales(), _count_values(), _evaluate(), _evaluate_scores(), _feature_importance_map(), _feature_matrix(), _fit_lightgbm_ranker() (+32 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (40): _allowed_boundaries(), analyze(), _append_unique_reason(), _apply_variant_spec(), _boundary_at_or_after(), _boundary_at_or_before(), _boundary_selection_reason(), _build_context_summary() (+32 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (25): BaseSettings, _resolve_local_project_path(), Settings, _validate_storage_dir(), _validate_training_vod_cache_dir(), _cache_dir(), _cache_path_for_url(), _copy_into_job() (+17 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (35): _adaptive_window(), _dense_coarse_score(), _dense_window_proposals(), _filter_candidates(), _generate_audio_burst_candidates(), generate_candidates(), _generate_chat_sync_candidates(), _generate_clean_reaction_candidates() (+27 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (23): _audience_signal(), build_score_timeline(), extract_candidates(), _hook_signal(), _payoff_signal(), Pre-scoring and candidate window extraction.  Runs before the LLM analysis step, Measure the emotional arc quality of a candidate clip window.      Supports mult, Compute a composite hype_score for each segment and store it in-place.      Scor (+15 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (33): _candidate_baselines(), _coerce_publish_timestamp(), _coerce_upload_date(), _dedupe_candidates(), _discover_seed(), _discover_seeds(), _extract_entry_url(), _extract_single_entry_metadata() (+25 more)

### Community 10 - "Community 10"
Cohesion: 0.13
Nodes (32): _absolute_candidate(), _apply_vod_context_features(), _boundary_perturbation_candidates(), build_dataset(), _build_rows_for_label(), _build_vod_context_index(), _candidate_alignment_score(), _checkpoint_dir() (+24 more)

### Community 11 - "Community 11"
Cohesion: 0.1
Nodes (5): Ranker, score_candidates(), _FakeLightGBMModel, RankerTests, test_ranker_scores_with_lightgbm_ranker_artifact()

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (1): TwitchClipLabelsTests

### Community 13 - "Community 13"
Cohesion: 0.23
Nodes (20): _audio_pop_score(), _build_candidate_features(), build_features(), _clamp01(), _context_dependency_score(), _empty_chapter_context(), _event_surprise_score(), _hook_strength() (+12 more)

### Community 14 - "Community 14"
Cohesion: 0.19
Nodes (18): _build_training_row(), collect_rows(), _feature_provenance(), _infer_creator_id(), _infer_platform(), _load_negatives(), main(), _ordinal_label() (+10 more)

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (19): _blend_score_maps(), _blend_value(), _boundary_artifact_is_usable(), _boundary_feature_row(), _boundary_model_is_usable(), _boundary_model_scores(), _heuristic_end_scores(), _heuristic_start_scores() (+11 more)

### Community 16 - "Community 16"
Cohesion: 0.18
Nodes (15): _check_gemini(), _check_groq(), _coerce_csv_list(), _coerce_optional_number(), _coerce_optional_text(), get_status(), _package_available(), _pipeline_component_statuses() (+7 more)

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (2): _FakeLGBMRanker, TrainRankerTests

### Community 18 - "Community 18"
Cohesion: 0.26
Nodes (18): _binary_label(), _build_end_examples(), _build_pair_examples(), _build_start_examples(), _clamp01(), _count_values(), _duration(), _duration_prior() (+10 more)

### Community 19 - "Community 19"
Cohesion: 0.23
Nodes (18): build_variant_specs(), candidate_in_time_range(), _clean_text(), _coerce_float(), _coerce_list(), _coerce_variant_count(), infer_goal_family(), _normalize_platform() (+10 more)

### Community 20 - "Community 20"
Cohesion: 0.11
Nodes (1): AutoTrainShortFormTests

### Community 21 - "Community 21"
Cohesion: 0.27
Nodes (17): _aggregate_score(), _as_float(), _clamp01(), _dynamic_scores(), _empty_result(), _extract_feedback_texts(), _feedback_row_score(), _groups_for() (+9 more)

### Community 22 - "Community 22"
Cohesion: 0.25
Nodes (17): _axes_to_score(), _blend_scores(), _build_context_summary(), _call_gemini(), _call_gemini_multimodal(), _call_groq(), _call_llm_judge(), _call_ollama() (+9 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (16): TranscriberProgressTests, _calc_tempo(), _deduplicate(), _diarize(), _dominant_speaker(), _fmt(), format_jsonl(), format_transcript() (+8 more)

### Community 24 - "Community 24"
Cohesion: 0.21
Nodes (16): _call_groq(), _call_llm(), _call_ollama_semantic(), context_dependency_score(), contradiction_score(), payoff_clarity_score(), quoteability_score(), Score the five viral dimensions with an optional provider preference. (+8 more)

### Community 25 - "Community 25"
Cohesion: 0.17
Nodes (6): _extract_frame(), _frame_times_for_candidate(), _safe_id(), sample_candidate_keyframes(), SampledKeyframe, ClipJudgeTests

### Community 26 - "Community 26"
Cohesion: 0.26
Nodes (13): find_semantic_candidates(), _get_model(), _get_pattern_embeddings(), _has_sentence_transformer_metadata(), _has_transformer_weights(), _link_snapshot_contents(), _local_sentence_transformer_model_dir(), _materialize_local_sentence_transformer_model() (+5 more)

### Community 27 - "Community 27"
Cohesion: 0.15
Nodes (1): JobsRouterTests

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (2): DownloaderTests, _FakeYoutubeDL

### Community 29 - "Community 29"
Cohesion: 0.35
Nodes (11): build_eval_dataset(), _content_bucket(), _count_values(), _dataset_role(), eval_dataset_summary(), _eval_group(), _has_outcome_feedback(), main() (+3 more)

### Community 30 - "Community 30"
Cohesion: 0.39
Nodes (11): _as_float(), _boundary_training_modules(), build_composite_dataset(), dataset_summary(), _filter_short_form_rows(), main(), _prepare_short_form_rows(), _read_jsonl() (+3 more)

### Community 31 - "Community 31"
Cohesion: 0.18
Nodes (9): get_content_type_for_streamer(), get_effective_heuristic_weights(), _init(), _load_custom_profiles(), Clip profile configuration.  Built-in profiles are defined in CLIP_PROFILES.  Cu, Merge profiles from a JSON file into CLIP_PROFILES (in-place)., Load custom profiles at import time (deferred to avoid circular imports)., Return the content-type profile for a known streamer, or None if unknown. (+1 more)

### Community 32 - "Community 32"
Cohesion: 0.36
Nodes (10): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), _quality_score() (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.2
Nodes (2): _FakeSentenceTransformer, SemanticCandidatesTests

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (7): _boundary_score(), _label_for_chapter(), _lexical_distance(), _make_chapter(), _merge_label(), _merge_short_chapters(), segment_chapters()

### Community 35 - "Community 35"
Cohesion: 0.4
Nodes (9): _as_float(), _content_bucket(), _dataset_source(), export_training_rows(), _feature_provenance(), _has_real_outcome_feedback(), _read_jsonl(), _resolved_feedback_label() (+1 more)

### Community 36 - "Community 36"
Cohesion: 0.39
Nodes (8): _clamp(), main(), _make_negative(), _make_positive(), Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro, Strong viral clip: early hook, high hype, clear arc, audience confirmed., Weak clip: late or absent hook, low energy, no arc, no audience signal., _uniform()

### Community 37 - "Community 37"
Cohesion: 0.61
Nodes (8): _bucket_metrics(), evaluate_pipeline_rows(), _group_recall_at_k(), _group_rows(), main(), _mean_reciprocal_rank(), _top1_positive_rate(), _top_groups()

### Community 38 - "Community 38"
Cohesion: 0.42
Nodes (8): _as_float(), build_transcript_fields(), _clip_window(), enrich_reference_row(), main(), _materialize_reference_clip(), _read_jsonl(), _transcribe_reference_clip()

### Community 39 - "Community 39"
Cohesion: 0.42
Nodes (8): build_boundary_features(), _build_feature(), _duration_prior(), _hook_signal(), _lexical_break(), _mid_thought_risk(), _reaction_signal(), _segments_between()

### Community 40 - "Community 40"
Cohesion: 0.39
Nodes (5): _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), _sample_id()

### Community 41 - "Community 41"
Cohesion: 0.29
Nodes (1): TrainVodRankerTests

### Community 42 - "Community 42"
Cohesion: 0.62
Nodes (6): _bucket_key(), _bucket_metrics(), _group_metrics(), main(), _metric_delta(), _read_jsonl()

### Community 43 - "Community 43"
Cohesion: 0.52
Nodes (6): _empty_similarity_features(), _load_artifact(), _neighbor_scores(), _read_jsonl(), score_candidate_window(), _window_texts()

### Community 44 - "Community 44"
Cohesion: 0.33
Nodes (1): TrainingExportTests

### Community 45 - "Community 45"
Cohesion: 0.67
Nodes (5): _boundary_metrics(), _bucket_by_kind(), _has_boundary_reference(), main(), _read_jsonl()

### Community 46 - "Community 46"
Cohesion: 0.4
Nodes (1): CollectViralClipsTests

### Community 47 - "Community 47"
Cohesion: 0.4
Nodes (1): ClipStrategyTests

### Community 48 - "Community 48"
Cohesion: 0.7
Nodes (4): main(), _queue_reason(), _read_jsonl(), _should_queue_row()

### Community 49 - "Community 49"
Cohesion: 0.7
Nodes (4): rank_candidates(), select_diverse_ranked_candidates(), select_top_candidates(), _too_close_to_any()

### Community 50 - "Community 50"
Cohesion: 0.6
Nodes (3): detect_peaks(), _fallback_peaks(), _walk_base()

### Community 51 - "Community 51"
Cohesion: 0.7
Nodes (4): build_boundary_points(), _delta(), _ends_sentence(), _merge_boundary()

### Community 52 - "Community 52"
Cohesion: 0.5
Nodes (1): ProfileConfigTests

### Community 53 - "Community 53"
Cohesion: 0.5
Nodes (1): EvaluateBoundaryRefinementTests

### Community 54 - "Community 54"
Cohesion: 0.5
Nodes (1): AnnotationFlowTests

### Community 55 - "Community 55"
Cohesion: 0.5
Nodes (1): ChatAnalyzerDownloadTests

### Community 56 - "Community 56"
Cohesion: 0.5
Nodes (1): BuildEvalDatasetTests

### Community 57 - "Community 57"
Cohesion: 0.67
Nodes (1): EvaluatePipelineTests

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (2): _group_metrics(), main()

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (2): Invoke-PythonStep(), Write-RunLog()

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (2): build_boundary_pairs(), refine_candidate_boundaries()

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (2): Root Layout Component, Tailwind CSS Custom Theme Config

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): ML utilities for training, annotation, and dataset preparation.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (0): 

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (0): 

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Next.js TypeScript Type References

## Knowledge Gaps
- **24 isolated node(s):** `Next.js TypeScript Type References`, `Tailwind CSS Custom Theme Config`, `Root Layout Component`, `SemanticServiceTests`, `TranscriberProgressTests` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 63`** (2 nodes): `Root Layout Component`, `Tailwind CSS Custom Theme Config`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (2 nodes): `__init__.py`, `ML utilities for training, annotation, and dataset preparation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (2 nodes): `diagnose_dataset.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Next.js TypeScript Type References`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RichSegment` connect `Community 1` to `Community 0`, `Community 2`, `Community 5`, `Community 7`, `Community 8`, `Community 23`, `Community 24`, `Community 25`, `Community 26`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `Candidate` connect `Community 1` to `Community 0`, `Community 2`, `Community 5`, `Community 7`, `Community 8`, `Community 11`, `Community 14`, `Community 25`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `_PlatformUnavailableError` connect `Community 9` to `Community 6`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 66 inferred relationships involving `RichSegment` (e.g. with `CandidateGeneratorTests` and `_FakeModel`) actually correct?**
  _`RichSegment` has 66 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `Candidate` (e.g. with `AnalyzerValidationTests` and `_FFmpegStub`) actually correct?**
  _`Candidate` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `BuildTwitchTrainingDatasetTests` (e.g. with `Candidate` and `RichSegment`) actually correct?**
  _`BuildTwitchTrainingDatasetTests` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Ranker` (e.g. with `_FakeLightGBMModel` and `RankerTests`) actually correct?**
  _`Ranker` has 3 INFERRED edges - model-reasoned connections that need verification._