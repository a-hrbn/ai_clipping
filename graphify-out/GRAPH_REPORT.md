# Graph Report - .  (2026-04-12)

## Corpus Check
- 86 files · ~60,690 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 673 nodes · 1114 edges · 47 communities detected
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 130 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 64 edges
2. `Candidate` - 37 edges
3. `Ranker` - 18 edges
4. `train_from_rows()` - 16 edges
5. `AutoTrainShortFormTests` - 14 edges
6. `RankerTests` - 11 edges
7. `TrainRankerTests` - 11 edges
8. `_build_training_row()` - 11 edges
9. `_FFmpegStub` - 10 edges
10. `RankedCandidate` - 10 edges

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
Cohesion: 0.04
Nodes (64): Audio Extractor (FFmpeg WAV Export), Speaker Diarization (MFCC KMeans), Audio Segment Enrichment, Audio Feature Extraction (Librosa), AudioFeatureFrame Data Model, LaughterDetector (Acoustic Rule-Based Classifier), Candidate Deduplication (_dedupe_candidates), normalize_candidates Function (+56 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (11): getDotState(), ProviderPill(), CandidateGeneratorTests, ClipSimilarityTests, _FakeModel, EnrichVectorReferenceTranscriptTests, ScoreTimelineTests, applyFile() (+3 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (21): BaseSettings, Settings, _cache_dir(), _cache_path_for_url(), _copy_into_job(), download(), DownloadUnavailableError, _evict_expired_cache() (+13 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (19): BaseModel, append_feedback_events(), append_feedback_snapshot(), build_pipeline_feedback(), CandidateFeedback, _matching_moment(), _read_feedback_rows(), update_job_feedback() (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (32): _bucketed_validation(), _compute_scales(), _count_values(), _evaluate(), _evaluate_scores(), _feature_matrix(), _fit_lightgbm_ranker(), _fit_pairwise_logistic() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.1
Nodes (6): Ranker, score_candidates(), _FakeLightGBMModel, RankerTests, CandidateFeatures, DetectedPeak

### Community 6 - "Community 6"
Cohesion: 0.14
Nodes (31): _candidate_baselines(), _coerce_publish_timestamp(), _coerce_upload_date(), _dedupe_candidates(), _discover_seed(), _discover_seeds(), _extract_entry_url(), _extract_single_entry_metadata() (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (23): _allowed_boundaries(), analyze(), _boundary_at_or_after(), _boundary_at_or_before(), _build_context_summary(), _call_gemini(), _call_ollama(), _call_ollama_refine() (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.16
Nodes (23): _dense_coarse_score(), _dense_window_proposals(), generate_candidates(), _generate_chat_sync_candidates(), _generate_controversy_candidates(), _generate_dense_window_candidates(), _generate_dialogue_candidates(), generate_fallback_candidates() (+15 more)

### Community 9 - "Community 9"
Cohesion: 0.2
Nodes (17): build_label_rows(), _classify_clip_views(), collect_clip_labels(), _dedupe_negative_rows(), _get_clips(), _get_user(), _get_vods(), main() (+9 more)

### Community 10 - "Community 10"
Cohesion: 0.14
Nodes (15): _audience_signal(), build_score_timeline(), extract_candidates(), _hook_signal(), _payoff_signal(), Pre-scoring and candidate window extraction.  Runs before the LLM analysis step, Measure the emotional arc quality of a candidate clip window.      Supports mult, Compute a composite hype_score for each segment and store it in-place.      Scor (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.19
Nodes (18): _build_training_row(), collect_rows(), _feature_provenance(), _infer_creator_id(), _infer_platform(), _load_negatives(), main(), _ordinal_label() (+10 more)

### Community 12 - "Community 12"
Cohesion: 0.18
Nodes (16): TranscriberProgressTests, _calc_tempo(), _deduplicate(), _diarize(), _dominant_speaker(), _fmt(), format_jsonl(), format_transcript() (+8 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (2): _FakeLGBMRanker, TrainRankerTests

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (1): AutoTrainShortFormTests

### Community 15 - "Community 15"
Cohesion: 0.23
Nodes (14): _call_groq(), _call_llm(), _call_ollama_semantic(), contradiction_score(), quoteability_score(), Score the five viral dimensions with an optional provider preference., Ask Groq (Llama 3.3 70B) to score the five viral dimensions., Ask Ollama to score the five viral dimensions. Returns None on any failure. (+6 more)

### Community 16 - "Community 16"
Cohesion: 0.31
Nodes (12): _apply_vod_context_features(), build_dataset(), _build_vod_context_index(), dataset_summary(), _entry_from_label(), _evict_expired_training_cache(), main(), _materialize_vod() (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (9): get_content_type_for_streamer(), get_effective_heuristic_weights(), _init(), _load_custom_profiles(), Clip profile configuration.  Built-in profiles are defined in CLIP_PROFILES.  Cu, Merge profiles from a JSON file into CLIP_PROFILES (in-place)., Load custom profiles at import time (deferred to avoid circular imports)., Return the content-type profile for a known streamer, or None if unknown. (+1 more)

### Community 18 - "Community 18"
Cohesion: 0.36
Nodes (10): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), _quality_score() (+2 more)

### Community 19 - "Community 19"
Cohesion: 0.44
Nodes (9): _as_float(), build_composite_dataset(), dataset_summary(), _filter_short_form_rows(), main(), _prepare_short_form_rows(), _read_jsonl(), _train_per_streamer() (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.28
Nodes (2): DownloaderTests, _FakeYoutubeDL

### Community 21 - "Community 21"
Cohesion: 0.39
Nodes (8): _clamp(), main(), _make_negative(), _make_positive(), Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro, Strong viral clip: early hook, high hype, clear arc, audience confirmed., Weak clip: late or absent hook, low energy, no arc, no audience signal., _uniform()

### Community 22 - "Community 22"
Cohesion: 0.42
Nodes (8): _as_float(), build_transcript_fields(), _clip_window(), enrich_reference_row(), main(), _materialize_reference_clip(), _read_jsonl(), _transcribe_reference_clip()

### Community 23 - "Community 23"
Cohesion: 0.25
Nodes (1): TwitchClipLabelsTests

### Community 24 - "Community 24"
Cohesion: 0.39
Nodes (5): _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), _sample_id()

### Community 25 - "Community 25"
Cohesion: 0.46
Nodes (7): _as_float(), export_training_rows(), _feature_provenance(), _has_real_outcome_feedback(), _read_jsonl(), _resolved_feedback_label(), _should_export_feedback()

### Community 26 - "Community 26"
Cohesion: 0.52
Nodes (6): _empty_similarity_features(), _load_artifact(), _neighbor_scores(), _read_jsonl(), score_candidate_window(), _window_texts()

### Community 27 - "Community 27"
Cohesion: 0.6
Nodes (5): _build_candidate_features(), build_features(), _hook_strength(), _payoff_score(), _profile_fit_score()

### Community 28 - "Community 28"
Cohesion: 0.47
Nodes (5): find_semantic_candidates(), _get_model(), _get_pattern_embeddings(), Semantic candidate generation using sentence embeddings.  Uses sentence-transfor, Return a dict mapping category → list of segment indices that exceed the     sim

### Community 29 - "Community 29"
Cohesion: 0.4
Nodes (1): TrainVodRankerTests

### Community 30 - "Community 30"
Cohesion: 0.4
Nodes (1): CollectViralClipsTests

### Community 31 - "Community 31"
Cohesion: 0.4
Nodes (1): TrainingExportTests

### Community 32 - "Community 32"
Cohesion: 0.7
Nodes (4): _bucket_metrics(), _group_metrics(), main(), _read_jsonl()

### Community 33 - "Community 33"
Cohesion: 0.7
Nodes (4): main(), _queue_reason(), _read_jsonl(), _should_queue_row()

### Community 34 - "Community 34"
Cohesion: 0.6
Nodes (3): detect_peaks(), _fallback_peaks(), _walk_base()

### Community 35 - "Community 35"
Cohesion: 0.5
Nodes (1): AnnotationFlowTests

### Community 36 - "Community 36"
Cohesion: 0.83
Nodes (3): _select_diverse_ranked_candidates(), select_top_candidates(), _too_close_to_any()

### Community 37 - "Community 37"
Cohesion: 0.67
Nodes (1): ProfileConfigTests

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (2): _group_metrics(), main()

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (2): Root Layout Component, Tailwind CSS Custom Theme Config

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (0): 

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Next.js TypeScript Type References

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): TranscriberChunkingTests

## Knowledge Gaps
- **24 isolated node(s):** `Next.js TypeScript Type References`, `Tailwind CSS Custom Theme Config`, `Root Layout Component`, `TranscriberChunkingTests`, `SemanticServiceTests` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 41`** (2 nodes): `Root Layout Component`, `Tailwind CSS Custom Theme Config`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (2 nodes): `diagnose_dataset.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Next.js TypeScript Type References`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `TranscriberChunkingTests`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RichSegment` connect `Community 0` to `Community 1`, `Community 3`, `Community 7`, `Community 8`, `Community 10`, `Community 12`, `Community 15`, `Community 28`?**
  _High betweenness centrality (0.196) - this node is a cross-community bridge._
- **Why does `Candidate` connect `Community 0` to `Community 3`, `Community 5`, `Community 7`, `Community 8`, `Community 11`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Are the 61 inferred relationships involving `RichSegment` (e.g. with `CandidateGeneratorTests` and `_FakeModel`) actually correct?**
  _`RichSegment` has 61 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `Candidate` (e.g. with `AnalyzerValidationTests` and `_FFmpegStub`) actually correct?**
  _`Candidate` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Ranker` (e.g. with `_FakeLightGBMModel` and `RankerTests`) actually correct?**
  _`Ranker` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Next.js TypeScript Type References`, `Tailwind CSS Custom Theme Config`, `Root Layout Component` to the rest of the system?**
  _24 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.04 - nodes in this community are weakly interconnected._