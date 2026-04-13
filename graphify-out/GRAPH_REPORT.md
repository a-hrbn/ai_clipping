# Graph Report - .  (2026-04-13)

## Corpus Check
- 97 files · ~58,599 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 839 nodes · 1333 edges · 78 communities detected
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 163 edges (avg confidence: 0.51)
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
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
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

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 77 edges
2. `Candidate` - 43 edges
3. `Ranker` - 18 edges
4. `train_from_rows()` - 16 edges
5. `AutoTrainShortFormTests` - 14 edges
6. `RankedCandidate` - 13 edges
7. `_FFmpegStub` - 12 edges
8. `_build_training_row()` - 11 edges
9. `ClipJudgeTests` - 11 edges
10. `PipelineIntegrationTests` - 11 edges

## Surprising Connections (you probably didn't know these)
- `enrich_segments()` --semantically_similar_to--> `enrich_segments()`  [INFERRED] [semantically similar]
  backend\services\audio_features.py → backend\services\video_features.py
- `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m` --uses--> `Candidate`  [INFERRED]
  backend/ml/collect_viral_clips.py → backend\services\timeline_types.py
- `Load negative rows from a JSONL file. Defaults to synthetic data if available.` --uses--> `Candidate`  [INFERRED]
  backend\ml\collect_viral_clips.py → backend\services\timeline_types.py
- `Post-processing of LLM-returned clip moments.  Two passes run after the LLM retu` --uses--> `RichSegment`  [INFERRED]
  backend/services/post_processor.py → backend\services\transcriber.py
- `Align each clip's start/end to real segment boundaries.      Start snapping:` --uses--> `RichSegment`  [INFERRED]
  backend/services/post_processor.py → backend\services\transcriber.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (78): Regex fallback — used only when sentence-transformers is unavailable., Finds moments of rising tension before an explosion.      Two complementary sign, Regex fallback — used only when sentence-transformers is unavailable., Finds moments of rising tension before an explosion.      Two complementary sign, Finds twists, reveals, and information drops that reframe what came before., Finds genuine disagreement, hot-takes, and confrontation — goes well beyond, Finds twists, reveals, and information drops that reframe what came before., Finds genuine disagreement, hot-takes, and confrontation — goes well beyond (+70 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (30): _allowed_boundaries(), analyze(), _boundary_at_or_after(), _boundary_at_or_before(), _build_context_summary(), _call_gemini(), _call_ollama(), _call_ollama_refine() (+22 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (7): Ranker, score_candidates(), FeedbackStoreTests, _FakeLightGBMModel, RankerTests, test_ranker_scores_with_lightgbm_ranker_artifact(), CandidateFeatures

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (32): _bucketed_validation(), _compute_scales(), _count_values(), _evaluate(), _evaluate_scores(), _feature_matrix(), _fit_lightgbm_ranker(), _fit_pairwise_logistic() (+24 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (32): _candidate_baselines(), _coerce_publish_timestamp(), _coerce_upload_date(), _dedupe_candidates(), _discover_seed(), _discover_seeds(), _extract_entry_url(), _extract_single_entry_metadata() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (22): BaseModel, append_feedback_events(), append_feedback_snapshot(), build_pipeline_feedback(), CandidateFeedback, _matching_moment(), _read_feedback_rows(), update_job_feedback() (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.19
Nodes (21): _dense_coarse_score(), _dense_window_proposals(), generate_candidates(), _generate_chat_sync_candidates(), _generate_controversy_candidates(), _generate_dense_window_candidates(), _generate_dialogue_candidates(), generate_fallback_candidates() (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.2
Nodes (17): build_label_rows(), _classify_clip_views(), collect_clip_labels(), _dedupe_negative_rows(), _get_clips(), _get_user(), _get_vods(), main() (+9 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (18): extract(), AudioFeatureFrame, _best_k(), diarize_speakers(), enrich_segments(), extract(), LaughterDetector, Per-second audio feature extraction — runs in parallel with Whisper transcriptio (+10 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (15): _audience_signal(), build_score_timeline(), extract_candidates(), _hook_signal(), _payoff_signal(), Pre-scoring and candidate window extraction.  Runs before the LLM analysis step, Measure the emotional arc quality of a candidate clip window.      Supports mult, Compute a composite hype_score for each segment and store it in-place.      Scor (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.19
Nodes (18): _build_training_row(), collect_rows(), _feature_provenance(), _infer_creator_id(), _infer_platform(), _load_negatives(), main(), _ordinal_label() (+10 more)

### Community 11 - "Community 11"
Cohesion: 0.25
Nodes (17): _axes_to_score(), _blend_scores(), _build_context_summary(), _call_gemini(), _call_gemini_multimodal(), _call_groq(), _call_llm_judge(), _call_ollama() (+9 more)

### Community 12 - "Community 12"
Cohesion: 0.21
Nodes (17): _apply_min_gap(), _ends_sentence(), enforce_diversity(), _extend_min_duration(), _find_clean_start(), _find_sentence_end(), _is_mid_thought(), Post-processing of LLM-returned clip moments.  Two passes run after the LLM retu (+9 more)

### Community 13 - "Community 13"
Cohesion: 0.19
Nodes (15): _calc_tempo(), _deduplicate(), _diarize(), _dominant_speaker(), _fmt(), format_jsonl(), format_transcript(), _get_model() (+7 more)

### Community 14 - "Community 14"
Cohesion: 0.12
Nodes (2): _FakeLGBMRanker, TrainRankerTests

### Community 15 - "Community 15"
Cohesion: 0.23
Nodes (14): _call_groq(), _call_llm(), _call_ollama_semantic(), contradiction_score(), quoteability_score(), Score the five viral dimensions with an optional provider preference., Ask Groq (Llama 3.3 70B) to score the five viral dimensions., Ask Ollama to score the five viral dimensions. Returns None on any failure. (+6 more)

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (1): AutoTrainShortFormTests

### Community 17 - "Community 17"
Cohesion: 0.31
Nodes (12): _apply_vod_context_features(), build_dataset(), _build_vod_context_index(), dataset_summary(), _entry_from_label(), _evict_expired_training_cache(), main(), _materialize_vod() (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.18
Nodes (9): get_content_type_for_streamer(), get_effective_heuristic_weights(), _init(), _load_custom_profiles(), Clip profile configuration.  Built-in profiles are defined in CLIP_PROFILES.  Cu, Merge profiles from a JSON file into CLIP_PROFILES (in-place)., Load custom profiles at import time (deferred to avoid circular imports)., Return the content-type profile for a known streamer, or None if unknown. (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.36
Nodes (10): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), _quality_score() (+2 more)

### Community 20 - "Community 20"
Cohesion: 0.38
Nodes (10): _build_candidate_features(), build_features(), _clamp01(), _empty_chapter_context(), _hook_strength(), _jaccard_similarity(), _payoff_score(), _prepare_chapter_contexts() (+2 more)

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (2): DownloaderTests, _FakeYoutubeDL

### Community 22 - "Community 22"
Cohesion: 0.44
Nodes (9): _as_float(), build_composite_dataset(), dataset_summary(), _filter_short_form_rows(), main(), _prepare_short_form_rows(), _read_jsonl(), _train_per_streamer() (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.38
Nodes (9): _dedupe_candidates(), _extend_to_sentence_end(), _nearest_boundary_at_or_after(), _nearest_boundary_at_or_before(), _nearest_end_at_or_after(), _nearest_end_at_or_before(), _normalize_candidate(), normalize_candidates() (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.33
Nodes (7): _boundary_score(), _label_for_chapter(), _lexical_distance(), _make_chapter(), _merge_label(), _merge_short_chapters(), segment_chapters()

### Community 25 - "Community 25"
Cohesion: 0.42
Nodes (9): _aggregate_score(), _clamp01(), _empty_result(), _groups_for(), _load_manifest(), _match_groups(), _normalize_text(), _opening_text() (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.42
Nodes (8): _as_float(), build_transcript_fields(), _clip_window(), enrich_reference_row(), main(), _materialize_reference_clip(), _read_jsonl(), _transcribe_reference_clip()

### Community 27 - "Community 27"
Cohesion: 0.39
Nodes (8): _clamp(), main(), _make_negative(), _make_positive(), Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro, Strong viral clip: early hook, high hype, clear arc, audience confirmed., Weak clip: late or absent hook, low energy, no arc, no audience signal., _uniform()

### Community 28 - "Community 28"
Cohesion: 0.25
Nodes (2): get_status(), _provider_payload()

### Community 29 - "Community 29"
Cohesion: 0.44
Nodes (8): _cache_dir(), _cache_path_for_url(), _copy_into_job(), download(), DownloadUnavailableError, _evict_expired_cache(), _is_twitch_url(), _resolve_downloaded_video()

### Community 30 - "Community 30"
Cohesion: 0.22
Nodes (1): BuildTwitchTrainingDatasetTests

### Community 31 - "Community 31"
Cohesion: 0.39
Nodes (5): _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), _sample_id()

### Community 32 - "Community 32"
Cohesion: 0.46
Nodes (7): _as_float(), export_training_rows(), _feature_provenance(), _has_real_outcome_feedback(), _read_jsonl(), _resolved_feedback_label(), _should_export_feedback()

### Community 33 - "Community 33"
Cohesion: 0.25
Nodes (1): JobsRouterTests

### Community 34 - "Community 34"
Cohesion: 0.25
Nodes (1): TwitchClipLabelsTests

### Community 35 - "Community 35"
Cohesion: 0.52
Nodes (6): _empty_similarity_features(), _load_artifact(), _neighbor_scores(), _read_jsonl(), score_candidate_window(), _window_texts()

### Community 36 - "Community 36"
Cohesion: 0.29
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 0.43
Nodes (4): applyFile(), handleDrop(), handleFileChange(), validateFile()

### Community 38 - "Community 38"
Cohesion: 0.29
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 0.6
Nodes (5): _cleanup_later(), run_pipeline(), _score_records(), _segment_records(), _write_jsonl()

### Community 40 - "Community 40"
Cohesion: 0.4
Nodes (2): ClipSimilarityTests, _FakeModel

### Community 41 - "Community 41"
Cohesion: 0.33
Nodes (1): JobManagerTests

### Community 42 - "Community 42"
Cohesion: 0.33
Nodes (1): TrainVodRankerTests

### Community 43 - "Community 43"
Cohesion: 0.33
Nodes (0): 

### Community 44 - "Community 44"
Cohesion: 0.7
Nodes (4): _bucket_metrics(), _group_metrics(), main(), _read_jsonl()

### Community 45 - "Community 45"
Cohesion: 0.7
Nodes (4): main(), _queue_reason(), _read_jsonl(), _should_queue_row()

### Community 46 - "Community 46"
Cohesion: 0.6
Nodes (3): detect_peaks(), _fallback_peaks(), _walk_base()

### Community 47 - "Community 47"
Cohesion: 0.7
Nodes (4): rank_candidates(), select_diverse_ranked_candidates(), select_top_candidates(), _too_close_to_any()

### Community 48 - "Community 48"
Cohesion: 0.4
Nodes (1): CollectViralClipsTests

### Community 49 - "Community 49"
Cohesion: 0.4
Nodes (1): TrainingExportTests

### Community 50 - "Community 50"
Cohesion: 0.5
Nodes (1): AnnotationFlowTests

### Community 51 - "Community 51"
Cohesion: 0.5
Nodes (1): ProfileConfigTests

### Community 52 - "Community 52"
Cohesion: 0.5
Nodes (1): TrendPriorTests

### Community 53 - "Community 53"
Cohesion: 0.67
Nodes (2): BaseSettings, Settings

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (2): _group_metrics(), main()

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (2): main(), _read_jsonl()

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (2): Invoke-PythonStep(), Write-RunLog()

### Community 58 - "Community 58"
Cohesion: 0.67
Nodes (1): TranscriberChunkingTests

### Community 59 - "Community 59"
Cohesion: 0.67
Nodes (1): TranscriberProgressTests

### Community 60 - "Community 60"
Cohesion: 0.67
Nodes (0): 

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (0): 

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (0): 

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (0): 

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (0): 

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (0): 

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (0): 

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (0): 

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (0): 

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (0): 

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (0): 

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (0): 

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (0): 

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (0): 

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Return the content-type profile for a known streamer, or None if unknown.

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Return heuristic weights merged with any streamer-specific overrides.      Falls

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Return the content-type profile for a known streamer, or None if unknown.

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Return heuristic weights merged with any streamer-specific overrides.      Falls

## Knowledge Gaps
- **27 isolated node(s):** `Discover and cross-deduplicate manifest rows from a list of seeds.`, `Train a separate artifact for each unique streamer in the seeds list.`, `Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro`, `Strong viral clip: early hook, high hype, clear arc, audience confirmed.`, `Weak clip: late or absent hook, low energy, no arc, no audience signal.` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 61`** (2 nodes): `diagnose_dataset.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (2 nodes): `verify_twitch_api_live.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (2 nodes): `clipper.py`, `cut_clips()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (2 nodes): `getDotState()`, `AIStatus.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (2 nodes): `ClipGrid()`, `ClipGrid.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `next-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `tailwind.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `layout.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `types.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Return the content-type profile for a known streamer, or None if unknown.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Return heuristic weights merged with any streamer-specific overrides.      Falls`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Return the content-type profile for a known streamer, or None if unknown.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Return heuristic weights merged with any streamer-specific overrides.      Falls`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RichSegment` connect `Community 0` to `Community 1`, `Community 40`, `Community 9`, `Community 12`, `Community 13`, `Community 15`, `Community 52`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `Candidate` connect `Community 0` to `Community 1`, `Community 10`, `Community 2`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Are the 74 inferred relationships involving `RichSegment` (e.g. with `Build a 2-3 sentence narrative summary of what happened before the clip window.` and `Refine clip boundaries. Uses Gemini if configured, else falls back to Ollama.`) actually correct?**
  _`RichSegment` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `Candidate` (e.g. with `Collect real viral clip training data from YouTube Shorts / TikTok URLs.  Each m` and `Load negative rows from a JSONL file. Defaults to synthetic data if available.`) actually correct?**
  _`Candidate` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Ranker` (e.g. with `CandidateFeatures` and `_FakeLightGBMModel`) actually correct?**
  _`Ranker` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Discover and cross-deduplicate manifest rows from a list of seeds.`, `Train a separate artifact for each unique streamer in the seeds list.`, `Generate synthetic pre-training data for the ML ranker.  Produces 80 training ro` to the rest of the system?**
  _27 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.02 - nodes in this community are weakly interconnected._