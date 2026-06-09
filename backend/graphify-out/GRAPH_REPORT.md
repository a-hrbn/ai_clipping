# Graph Report - backend  (2026-06-09)

## Corpus Check
- 127 files · ~86,143 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1733 nodes · 3787 edges · 62 communities (47 shown, 15 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 133 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8e497123`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 112|Community 112]]
- [[_COMMUNITY_Community 113|Community 113]]
- [[_COMMUNITY_Community 114|Community 114]]
- [[_COMMUNITY_Community 115|Community 115]]
- [[_COMMUNITY_Community 119|Community 119]]
- [[_COMMUNITY_Community 123|Community 123]]
- [[_COMMUNITY_Community 134|Community 134]]
- [[_COMMUNITY_Community 136|Community 136]]
- [[_COMMUNITY_Community 137|Community 137]]
- [[_COMMUNITY_Community 141|Community 141]]
- [[_COMMUNITY_Community 142|Community 142]]
- [[_COMMUNITY_Community 143|Community 143]]
- [[_COMMUNITY_Community 145|Community 145]]
- [[_COMMUNITY_Community 157|Community 157]]
- [[_COMMUNITY_Community 160|Community 160]]
- [[_COMMUNITY_Community 163|Community 163]]
- [[_COMMUNITY_Community 166|Community 166]]
- [[_COMMUNITY_Community 169|Community 169]]
- [[_COMMUNITY_Community 176|Community 176]]
- [[_COMMUNITY_Community 178|Community 178]]
- [[_COMMUNITY_Community 179|Community 179]]
- [[_COMMUNITY_Community 191|Community 191]]
- [[_COMMUNITY_Community 192|Community 192]]
- [[_COMMUNITY_Community 193|Community 193]]
- [[_COMMUNITY_Community 194|Community 194]]
- [[_COMMUNITY_Community 197|Community 197]]
- [[_COMMUNITY_Community 206|Community 206]]
- [[_COMMUNITY_Community 207|Community 207]]
- [[_COMMUNITY_Community 208|Community 208]]
- [[_COMMUNITY_Community 211|Community 211]]
- [[_COMMUNITY_Community 212|Community 212]]
- [[_COMMUNITY_Community 213|Community 213]]

## God Nodes (most connected - your core abstractions)
1. `RichSegment` - 58 edges
2. `JobsRouterTests` - 32 edges
3. `TwitchClipLabelsTests` - 31 edges
4. `match_reposts_to_vods_with_diagnostics()` - 28 edges
5. `DownloaderTests` - 28 edges
6. `Path` - 24 edges
7. `Any` - 23 edges
8. `Any` - 22 edges
9. `Path` - 22 edges
10. `TranscriptMatch` - 21 edges

## Surprising Connections (you probably didn't know these)
- `Remove jobs older than _JOB_TTL_SECONDS from the in-memory store.` --uses--> `JobStatus`  [INFERRED]
  job_manager.py → models.py
- `Flush current job state to disk without holding the state lock during I/O.` --uses--> `JobStatus`  [INFERRED]
  job_manager.py → models.py
- `Any` --uses--> `RichSegment`  [INFERRED]
  ml/match_cross_platform_transcripts.py → services/transcriber.py
- `BaseException` --uses--> `RichSegment`  [INFERRED]
  ml/match_cross_platform_transcripts.py → services/transcriber.py
- `create_job()` --calls--> `JobStatus`  [INFERRED]
  job_manager.py → models.py

## Import Cycles
- 1-file cycle: `ml/collect_twitch_clip_labels.py -> ml/collect_twitch_clip_labels.py`
- 1-file cycle: `ml/discover_iboclipss_clips.py -> ml/discover_iboclipss_clips.py`

## Hyperedges (group relationships)
- **Viral Clip Training Input Formats** — viral_clips_training_data, viral_clips_youtube_shorts_url, viral_clips_tiktok_url, viral_clips_csv_tsv_metadata [EXTRACTED 1.00]
- **Trend Prior Scoring Outputs** — readme_trend_prior_service, readme_trend_hook_score, readme_trend_topic_score, readme_trend_prior_score [EXTRACTED 1.00]
- **Trend Prior Update Inputs** — readme_update_process, readme_qualitative_review, readme_creator_domain_needs, readme_offline_eval_results [EXTRACTED 1.00]
- **Shorts Pattern Seed Domains** — readme_seed_phrases, readme_short_categories, readme_hook_patterns_manifest [EXTRACTED 1.00]

## Communities (62 total, 15 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (75): _dedup_moments(), main(), _overlap_ratio(), Path, Convert cross-platform match rows into a fixed eval manifest.  Reads iboclipss_m, Deduplicate near-overlapping windows per VOD. Highest views wins a slot., _read_jsonl(), main() (+67 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (81): Event, configure_job_logging(), Configure root logging for a detached pipeline subprocess.      The orchestrator, aggregate_overall(), coverage_ratio(), evaluate_all(), evaluate_vod(), is_hit() (+73 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (26): _build_gql_payload(), _build_per_second_counts(), _build_per_second_emote_classes(), _classify_emote(), _compute_spike_scores(), download_chat(), enrich_segments(), _extract_vod_id() (+18 more)

### Community 3 - "Community 3"
Cohesion: 0.47
Nodes (6): get_status(), _package_available(), _pipeline_component_statuses(), _probe_face_detection_status(), _provider_payload(), _status_payload()

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (50): anthropic_text_content(), balanced_json_objects_from_list(), bounded_ratio(), _build_chat_marker(), build_moment_finder_prompt(), call_anthropic_moment_finder(), call_gemini_moment_finder(), call_moment_finder_provider() (+42 more)

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (35): main(), _parse_args(), run_local_pipeline(), main(), _parse_args(), run(), main(), _parse_args() (+27 more)

### Community 25 - "Community 25"
Cohesion: 0.06
Nodes (55): build_outcome_holdout_dataset(), _fallback_candidate_id(), _feature_replay_index(), _is_cross_platform_positive_split(), main(), _normalize_split(), _prepare_rows(), Counter (+47 more)

### Community 32 - "Community 32"
Cohesion: 0.07
Nodes (29): Trend Prior Data, Candidate Clip, Clip-Worthy Topics, Cheap Deterministic Prior Rationale, High-Performing Short-Form Hooks, hook_patterns.json Manifest, Judge, Avoid Scraping Proprietary or Copyrighted Corpora Rationale (+21 more)

### Community 35 - "Community 35"
Cohesion: 0.15
Nodes (32): _audio_cache_metadata(), _audio_cache_metadata_path(), _audio_cache_path(), _cleanup_audio_artifacts(), _cleanup_later(), _cleanup_serving_source_media(), _debug_job(), _extract_audio_cached() (+24 more)

### Community 36 - "Community 36"
Cohesion: 0.09
Nodes (65): Fingerprint, _acceptance_decision(), _accepted_matches_by_vod(), _append_unique_match(), _audio_fallback_confidence_threshold(), _audio_fingerprint_fallback(), _best_match_diagnostic_row(), _best_transcript_match() (+57 more)

### Community 38 - "Community 38"
Cohesion: 0.09
Nodes (26): main(), Path, _read_jsonl(), find_semantic_candidates(), _get_model(), _get_pattern_embeddings(), _has_sentence_transformer_metadata(), _has_transformer_weights() (+18 more)

### Community 44 - "Community 44"
Cohesion: 0.11
Nodes (26): BaseSettings, _configure_local_cache_env(), _configure_local_tool_path(), _is_writable_cache_dir(), _prepare_local_cache_path(), Path, _resolve_local_project_path(), Settings (+18 more)

### Community 46 - "Community 46"
Cohesion: 0.06
Nodes (77): Segment, compute_segment_energy(), _mean_rms(), ndarray, Path, RichSegment, Compute per-segment RMS energy from a WAV file.  Uses only the Python standard l, Compute RMS energy for each integer second of audio. (+69 more)

### Community 47 - "Community 47"
Cohesion: 0.22
Nodes (9): Broad Pattern Groups, Creator or Domain Needs, Prefer Broad Patterns Rationale, Offline Eval Results, Highly Specific One-Off Phrases, Qualitative Review, Trend Prior Update Process, Trend Prior Validation Tests (+1 more)

### Community 49 - "Community 49"
Cohesion: 0.08
Nodes (36): best_fingerprint_alignment(), _best_match_diagnostic_row(), Fingerprint, _fingerprint_from_row(), fingerprint_media_to_file(), _fingerprint_similarity(), _log_event(), _log_repost_progress() (+28 more)

### Community 77 - "Community 77"
Cohesion: 0.18
Nodes (20): build_variant_specs(), candidate_in_time_range(), _clean_text(), _coerce_float(), _coerce_list(), _coerce_variant_count(), infer_goal_family(), _normalize_platform() (+12 more)

### Community 78 - "Community 78"
Cohesion: 0.18
Nodes (33): _aggregate_jobs(), _as_float(), _debug_dirs(), _default_output(), _default_positive_rows(), _default_storage_dir(), evaluate_holdout(), _evaluate_job() (+25 more)

### Community 80 - "Community 80"
Cohesion: 0.11
Nodes (40): bind_job_session(), _as_float(), _as_optional_float(), cancel_job_status(), _candidate_funnel_payload(), _check_gemini(), _check_groq(), _cleanup_serving_job_source_media() (+32 more)

### Community 83 - "Community 83"
Cohesion: 0.09
Nodes (25): build_manifest(), _job_summary(), main(), Path, _read_job_map(), _read_jsonl(), _recover_vod_url(), _url_from_path() (+17 more)

### Community 84 - "Community 84"
Cohesion: 0.12
Nodes (33): _auto_negative_row(), build_auto_outcome_negative_rows(), _candidate_score(), _clamp_ratio(), _clip_ids(), _ensure_holdout_split_floor(), _holdout_rebalance_key(), _identity_key_set() (+25 more)

### Community 86 - "Community 86"
Cohesion: 0.23
Nodes (13): BaseModel, JobRequest, ClipFeedbackPayload, ClipFeedbackResponse, JobRequest, JobStatus, ReviewCandidate, ReviewCandidateResponse (+5 more)

### Community 87 - "Community 87"
Cohesion: 0.17
Nodes (25): cancel_job(), create_job(), delete_job(), _evict_expired(), get_job(), is_job_cancelled(), _load_jobs(), _persist_jobs() (+17 more)

### Community 88 - "Community 88"
Cohesion: 0.17
Nodes (32): _as_float(), _debug_dirs(), export_serving_hard_negatives(), _fair_cap_rows_by_vod(), _feature_similarity(), _hard_negative_rows_for_job(), _holdout_vod_ids(), main() (+24 more)

### Community 89 - "Community 89"
Cohesion: 0.07
Nodes (69): _blended_gap_duration(), build_label_rows(), _clamp_negative_duration(), _classify_clip_views(), _clip_merge_end(), _clip_window_bounds(), collect_clip_labels(), collect_clip_labels_with_summary() (+61 more)

### Community 113 - "Community 113"
Cohesion: 0.19
Nodes (22): _as_float(), _candidate_summary(), _delta_summary(), _job_vod_id(), main(), _matched_positive_count(), _mean(), _numeric_delta() (+14 more)

### Community 114 - "Community 114"
Cohesion: 0.05
Nodes (6): _CaptureOptsYoutubeDL, DownloaderTests, _FailOnceFfmpegCrashThenCookieCloseErrorYoutubeDL, _FailOnceFfmpegCrashYoutubeDL, _FakeYoutubeDL, _FakeYoutubeDLWithTemp

### Community 115 - "Community 115"
Cohesion: 0.06
Nodes (54): CandidateFeatures, ClipFeedbackResponse, LlmMoment, append_feedback_events(), append_feedback_snapshot(), append_gold_eval_snapshot(), _apply_judge_fields(), _as_float() (+46 more)

### Community 119 - "Community 119"
Cohesion: 0.18
Nodes (16): _cached_media_by_size(), cleanup_backend_runtime_caches(), CleanupResult, _display_path(), _is_safe_to_remove_legacy_short_form_job_dir(), _is_safe_to_remove_twitch_job_dir(), main(), _media_exists_in_cache() (+8 more)

### Community 123 - "Community 123"
Cohesion: 0.13
Nodes (17): append_annotations(), _apply_manual_label(), _manual_label_class(), _manual_ordinal_label(), merge_annotations(), Path, read_annotations(), _sample_id() (+9 more)

### Community 134 - "Community 134"
Cohesion: 0.08
Nodes (48): Downloader, Lock, ensure_vod_sources(), Any, Path, resolve_downloader(), source_video_path(), twitch_vod_url() (+40 more)

### Community 136 - "Community 136"
Cohesion: 0.22
Nodes (18): _apply_min_gap(), _ends_sentence(), enforce_diversity(), _extend_min_duration(), _find_clean_start(), _find_sentence_end(), _is_mid_thought(), RichSegment (+10 more)

### Community 137 - "Community 137"
Cohesion: 0.18
Nodes (5): SimpleNamespace, CrossPlatformTranscriptMatchTests, segment(), transcript_match(), TranscriptMatch

### Community 141 - "Community 141"
Cohesion: 0.13
Nodes (18): ClipResult, Error, ClipResult, Moment, _as_float(), cut_clips(), cut_review_clips(), _decode_ffmpeg_error() (+10 more)

### Community 143 - "Community 143"
Cohesion: 0.12
Nodes (16): hook_patterns, topic_patterns, global, hook_patterns, topic_patterns, hook_patterns, topic_patterns, hook_patterns (+8 more)

### Community 145 - "Community 145"
Cohesion: 0.06
Nodes (49): main(), _parse_args(), run(), _annotated_eval_row(), build_eval_dataset(), _content_bucket(), _count_values(), _dataset_role() (+41 more)

### Community 157 - "Community 157"
Cohesion: 0.25
Nodes (11): _first_words(), main(), main_with_args_for_test(), Path, _read_jsonl(), _sample_rows(), _twitch_link(), _write_csv() (+3 more)

### Community 160 - "Community 160"
Cohesion: 0.32
Nodes (10): _artifact_snapshot(), build_audit(), _count_jsonl_rows(), main(), Any, Path, _read_json(), _sha256() (+2 more)

### Community 163 - "Community 163"
Cohesion: 0.35
Nodes (11): _as_int(), build_reference_manifest(), _default_dataset_path(), _full_text(), _hook_text(), main(), _payoff_text(), Path (+3 more)

### Community 169 - "Community 169"
Cohesion: 0.29
Nodes (7): _boundary_metrics(), _bucket_by_kind(), _has_boundary_reference(), main(), Path, _read_jsonl(), EvaluateBoundaryRefinementTests

### Community 178 - "Community 178"
Cohesion: 0.26
Nodes (12): build_manifest(), _diagnosis_error_counts(), main(), Counter, Path, _read_csv(), _read_jsonl(), _write_csv() (+4 more)

### Community 179 - "Community 179"
Cohesion: 0.39
Nodes (5): BackgroundWindow, _overlap_ratio(), sample_background_windows(), _stable_seed(), SampleBackgroundNegativesTests

### Community 192 - "Community 192"
Cohesion: 0.60
Nodes (5): main(), _numeric_feature_names(), Path, _read_jsonl(), _row_timestamp()

### Community 193 - "Community 193"
Cohesion: 0.67
Nodes (5): import_rows(), main(), Path, _read_jsonl(), _safe_float()

### Community 194 - "Community 194"
Cohesion: 0.67
Nodes (5): import_retention_rows(), main(), Path, _read_jsonl(), _safe_float()

### Community 197 - "Community 197"
Cohesion: 0.40
Nodes (4): reason, review_required, rows, status

## Knowledge Gaps
- **50 isolated node(s):** `version`, `updated_at`, `source_note`, `hook_patterns`, `topic_patterns` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RichSegment` connect `Community 46` to `Community 2`, `Community 36`, `Community 38`, `Community 136`, `Community 141`, `Community 115`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `RankedCandidate` connect `Community 115` to `Community 86`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `JobsRouterTests` connect `Community 112` to `Community 86`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `RichSegment` (e.g. with `CandidateFeatures` and `Fingerprint`) actually correct?**
  _`RichSegment` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `JobsRouterTests` (e.g. with `ClipFeedbackPayload` and `ClipFeedbackResponse`) actually correct?**
  _`JobsRouterTests` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `version`, `updated_at`, `source_note` to the rest of the system?**
  _117 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05826330532212885 - nodes in this community are weakly interconnected._