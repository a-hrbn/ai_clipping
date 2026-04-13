# Graph Report - .  (2026-04-14)

## Corpus Check
- 98 files · ~93,615 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 816 nodes · 1370 edges · 54 communities detected
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 150 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 67 edges
2. `Candidate` - 39 edges
3. `Ranker` - 18 edges
4. `train_from_rows()` - 16 edges
5. `AutoTrainShortFormTests` - 14 edges
6. `RankedCandidate` - 13 edges
7. `_window_candidate()` - 13 edges
8. `_FFmpegStub` - 12 edges
9. `ClipJudgeTests` - 12 edges
10. `CandidateGeneratorTests` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Load negative rows from a JSONL file. Defaults to synthetic data if available.` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend/services/timeline_types.py
- `Score the five viral dimensions with an optional provider preference.` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_service.py → backend/services/transcriber.py
- `Ask Groq (Llama 3.3 70B) to score the five viral dimensions.` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_service.py → backend/services/transcriber.py
- `Ask Ollama to score the five viral dimensions. Returns None on any failure.` --uses--> `RichSegment`  [INFERRED]
  backend/services/semantic_service.py → backend/services/transcriber.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (69): Audio Extractor (FFmpeg WAV Export), Speaker Diarization (MFCC KMeans), Audio Segment Enrichment, Audio Feature Extraction (Librosa), AudioFeatureFrame Data Model, LaughterDetector (Acoustic Rule-Based Classifier), Candidate Deduplication (_dedupe_candidates), normalize_candidates Function (+61 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (11): CandidateGeneratorTests, ChapterSegmenterTests, EnrichVectorReferenceTranscriptTests, FeatureBuilderTests, KeyframeSamplerTests, PeakDetectorTests, ScoreTimelineTests, applyFile() (+3 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (19): BaseModel, append_feedback_events(), append_feedback_snapshot(), build_pipeline_feedback(), CandidateFeedback, _matching_moment(), _read_feedback_rows(), update_job_feedback() (+11 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (19): BaseSettings, Settings, _call_groq(), _call_llm(), _call_ollama_semantic(), context_dependency_score(), contradiction_score(), payoff_clarity_score() (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (32): _bucketed_validation(), _compute_scales(), _count_values(), _evaluate(), _evaluate_scores(), _feature_matrix(), _fit_lightgbm_ranker(), _fit_pairwise_logistic() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (30): _adaptive_window(), _dense_coarse_score(), _dense_window_proposals(), generate_candidates(), _generate_chat_sync_candidates(), _generate_clean_reaction_candidates(), _generate_controversy_candidates(), _generate_dense_window_candidates() (+22 more)

### Community 6 - "Community 6"
Cohesion: 0.14
Nodes (31): _candidate_baselines(), _coerce_publish_timestamp(), _coerce_upload_date(), _dedupe_candidates(), _discover_seed(), _discover_seeds(), _extract_entry_url(), _extract_single_entry_metadata() (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (27): _allowed_boundaries(), analyze(), _boundary_at_or_after(), _boundary_at_or_before(), _build_context_summary(), _call_gemini(), _call_ollama(), _call_ollama_refine() (+19 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (22): _cache_dir(), _cache_path_for_url(), _copy_into_job(), download(), DownloadUnavailableError, _evict_expired_cache(), _is_twitch_url(), _resolve_downloaded_video() (+14 more)

### Community 9 - "Community 9"
Cohesion: 0.1
Nodes (6): Ranker, score_candidates(), _FakeLightGBMModel, RankerTests, test_ranker_scores_with_lightgbm_ranker_artifact(), CandidateFeatures

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (22): _blended_gap_duration(), build_label_rows(), _clamp_negative_duration(), _classify_clip_views(), collect_clip_labels(), _cycled_duration(), _dedupe_negative_rows(), _duration_for_positive_window() (+14 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (21): _apply_vod_context_features(), build_dataset(), _build_rows_for_label(), _build_vod_context_index(), _candidate_alignment_score(), _clone_candidate(), dataset_summary(), _evict_expired_training_cache() (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.19
Nodes (18): _build_training_row(), collect_rows(), _feature_provenance(), _infer_creator_id(), _infer_platform(), _load_negatives(), main(), _ordinal_label() (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.25
Nodes (17): _axes_to_score(), _blend_scores(), _build_context_summary(), _call_gemini(), _call_gemini_multimodal(), _call_groq(), _call_llm_judge(), _call_ollama() (+9 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (16): TranscriberProgressTests, _calc_tempo(), _deduplicate(), _diarize(), _dominant_speaker(), _fmt(), format_jsonl(), format_transcript() (+8 more)

### Community 15 - "Community 15"
Cohesion: 0.12
Nodes (2): _FakeLGBMRanker, TrainRankerTests

### Community 16 - "Community 16"
Cohesion: 0.2
Nodes (14): _audience_signal(), build_score_timeline(), extract_candidates(), _hook_signal(), _payoff_signal(), Pre-scoring and candidate window extraction.  Runs before the LLM analysis step, Measure the emotional arc quality of a candidate clip window.      Supports mult, Compute a composite hype_score for each segment and store it in-place.      Scor (+6 more)

### Community 17 - "Community 17"
Cohesion: 0.13
Nodes (1): AutoTrainShortFormTests

### Community 18 - "Community 18"
Cohesion: 0.17
Nodes (6): _extract_frame(), _frame_times_for_candidate(), _safe_id(), sample_candidate_keyframes(), SampledKeyframe, ClipJudgeTests

### Community 19 - "Community 19"
Cohesion: 0.33
Nodes (13): _build_candidate_features(), build_features(), _clamp01(), _context_dependency_score(), _empty_chapter_context(), _hook_strength(), _jaccard_similarity(), _payoff_score() (+5 more)

### Community 20 - "Community 20"
Cohesion: 0.23
Nodes (6): get_status(), _package_available(), _pipeline_component_statuses(), _probe_face_detection_status(), _provider_payload(), _status_payload()

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (9): get_content_type_for_streamer(), get_effective_heuristic_weights(), _init(), _load_custom_profiles(), Clip profile configuration.  Built-in profiles are defined in CLIP_PROFILES.  Cu, Merge profiles from a JSON file into CLIP_PROFILES (in-place)., Load custom profiles at import time (deferred to avoid circular imports)., Return the content-type profile for a known streamer, or None if unknown. (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.18
Nodes (2): DownloaderTests, _FakeYoutubeDL

### Community 23 - "Community 23"
Cohesion: 0.36
Nodes (10): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), _quality_score() (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.2
Nodes (1): TwitchClipLabelsTests

### Community 25 - "Community 25"
Cohesion: 0.44
Nodes (9): _as_float(), build_composite_dataset(), dataset_summary(), _filter_short_form_rows(), main(), _prepare_short_form_rows(), _read_jsonl(), _train_per_streamer() (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.33
Nodes (7): _boundary_score(), _label_for_chapter(), _lexical_distance(), _make_chapter(), _merge_label(), _merge_short_chapters(), segment_chapters()

### Community 27 - "Community 27"
Cohesion: 0.42
Nodes (9): _aggregate_score(), _clamp01(), _empty_result(), _groups_for(), _load_manifest(), _match_groups(), _normalize_text(), _opening_text() (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.39
Nodes (8): _clamp(), main(), _make_negative(), _make_positive(), Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro, Strong viral clip: early hook, high hype, clear arc, audience confirmed., Weak clip: late or absent hook, low energy, no arc, no audience signal., _uniform()

### Community 29 - "Community 29"
Cohesion: 0.42
Nodes (8): _as_float(), build_transcript_fields(), _clip_window(), enrich_reference_row(), main(), _materialize_reference_clip(), _read_jsonl(), _transcribe_reference_clip()

### Community 30 - "Community 30"
Cohesion: 0.25
Nodes (1): JobsRouterTests

### Community 31 - "Community 31"
Cohesion: 0.39
Nodes (5): _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), _sample_id()

### Community 32 - "Community 32"
Cohesion: 0.46
Nodes (7): _as_float(), export_training_rows(), _feature_provenance(), _has_real_outcome_feedback(), _read_jsonl(), _resolved_feedback_label(), _should_export_feedback()

### Community 33 - "Community 33"
Cohesion: 0.52
Nodes (6): _empty_similarity_features(), _load_artifact(), _neighbor_scores(), _read_jsonl(), score_candidate_window(), _window_texts()

### Community 34 - "Community 34"
Cohesion: 0.4
Nodes (2): ClipSimilarityTests, _FakeModel

### Community 35 - "Community 35"
Cohesion: 0.33
Nodes (1): TrainVodRankerTests

### Community 36 - "Community 36"
Cohesion: 0.4
Nodes (1): CollectViralClipsTests

### Community 37 - "Community 37"
Cohesion: 0.4
Nodes (1): TrainingExportTests

### Community 38 - "Community 38"
Cohesion: 0.7
Nodes (4): _bucket_metrics(), _group_metrics(), main(), _read_jsonl()

### Community 39 - "Community 39"
Cohesion: 0.7
Nodes (4): main(), _queue_reason(), _read_jsonl(), _should_queue_row()

### Community 40 - "Community 40"
Cohesion: 0.7
Nodes (4): rank_candidates(), select_diverse_ranked_candidates(), select_top_candidates(), _too_close_to_any()

### Community 41 - "Community 41"
Cohesion: 0.6
Nodes (3): detect_peaks(), _fallback_peaks(), _walk_base()

### Community 42 - "Community 42"
Cohesion: 0.5
Nodes (1): ProfileConfigTests

### Community 43 - "Community 43"
Cohesion: 0.5
Nodes (1): AnnotationFlowTests

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (2): _group_metrics(), main()

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (2): Invoke-PythonStep(), Write-RunLog()

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (2): Root Layout Component, Tailwind CSS Custom Theme Config

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): ML utilities for training, annotation, and dataset preparation.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Next.js TypeScript Type References

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): TranscriberChunkingTests

## Knowledge Gaps
- **25 isolated node(s):** `Next.js TypeScript Type References`, `Tailwind CSS Custom Theme Config`, `Root Layout Component`, `TranscriberChunkingTests`, `SemanticServiceTests` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 48`** (2 nodes): `Root Layout Component`, `Tailwind CSS Custom Theme Config`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (2 nodes): `__init__.py`, `ML utilities for training, annotation, and dataset preparation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (2 nodes): `diagnose_dataset.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Next.js TypeScript Type References`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `TranscriberChunkingTests`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RichSegment` connect `Community 0` to `Community 1`, `Community 34`, `Community 2`, `Community 3`, `Community 5`, `Community 7`, `Community 14`, `Community 16`, `Community 18`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Why does `Candidate` connect `Community 0` to `Community 1`, `Community 2`, `Community 5`, `Community 7`, `Community 9`, `Community 12`, `Community 18`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `RichSegment` (e.g. with `CandidateGeneratorTests` and `_FakeModel`) actually correct?**
  _`RichSegment` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `Candidate` (e.g. with `AnalyzerValidationTests` and `_FFmpegStub`) actually correct?**
  _`Candidate` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Ranker` (e.g. with `_FakeLightGBMModel` and `RankerTests`) actually correct?**
  _`Ranker` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Next.js TypeScript Type References`, `Tailwind CSS Custom Theme Config`, `Root Layout Component` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.04 - nodes in this community are weakly interconnected._