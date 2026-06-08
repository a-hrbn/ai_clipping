export interface UploadProgress {
  pct: number;
  phase: "preparing" | "uploading" | "processing";
  loadedBytes: number;
  totalBytes: number;
  speedBps: number | null;   // bytes per second, null if not yet calculable
  etaSec: number | null;     // seconds remaining, null if not yet calculable
}

export interface ClipResult {
  candidate_id: string;
  filename: string;
  download_url: string;
  start_time: number;
  end_time: number;
  viral_score: number;
  reason: string;
  selection_reasons: string[];
  ranking_mode: "heuristic" | "ml_blended";
  platform?: "general" | "tiktok" | "youtube_shorts" | "instagram_reels" | "linkedin";
  variant_key?: string;
  variant_label?: string | null;
  projected_hook_score?: number | null;
  projected_retention_score?: number | null;
}

export type ClipFeedbackRating = "excellent" | "good" | "maybe" | "medium" | "bad" | "boundary_bad";
export type ClipFeedbackTag =
  | "good_clip"
  | "false_positive"
  | "missed_viral"
  | "wrong_boundary"
  | "wrong_duration"
  | "strong_hook"
  | "funny_reaction"
  | "shock_or_rage"
  | "fanmail_reveal"
  | "quoteable"
  | "good_context"
  | "weak_context"
  | "slow_start"
  | "no_payoff"
  | "too_long"
  | "bad_boundary";
export type ClipFeedbackMode = "training_feedback" | "gold_eval";

export interface ClipFeedbackPayload {
  candidate_id: string;
  mode?: ClipFeedbackMode;
  rating?: ClipFeedbackRating;
  gold_grade?: 0 | 1 | 2 | 3;
  feedback_tags?: ClipFeedbackTag[];
  notes?: string;
}

export interface ClipRequestOptions {
  platform?: "general" | "tiktok" | "youtube_shorts" | "instagram_reels" | "linkedin";
  goal?: string;
  tone?: string;
  topic?: string;
  must_include?: string[];
  exclude?: string[];
  speaker?: string;
  time_range?: { start_sec?: number; end_sec?: number };
  clip_style?: "balanced" | "aggressive_hook" | "context_balanced" | "reaction_first" | "quote_first";
  variant_count?: number;
}

export interface ReviewCandidate {
  candidate_id: string;
  preview_url?: string | null;
  source?: string | null;
  clip_type?: string | null;
  start_time: number;
  end_time: number;
  duration: number;
  rank_score?: number | null;
  pre_judge_rank?: number | null;
  judge_score?: number | null;
  selected: boolean;
  rejected_reason?: string | null;
  rejection_source?: string | null;
  judge_reason?: string | null;
  feedback_rating?: ClipFeedbackRating | null;
  feedback_tags?: ClipFeedbackTag[];
  feedback_notes?: string | null;
}

export interface ReviewCandidateResponse {
  job_id: string;
  candidates: ReviewCandidate[];
}

export type PipelineDebugStatus = "queued" | "running" | "done" | "failed";
export type PipelineDebugStage = "queued" | "transcribe" | "moment_finder" | "cut_clips" | "done" | "failed";
export type PipelineDebugViralLabel = "yes" | "no" | "unsure" | "";
export type PipelineDebugBoundaryLabel = "good" | "too_wide" | "too_short" | "wrong" | "unsure" | "";

export interface PipelineDebugStartRequest {
  profile?: string;
  vod_ids?: string[];
  max_moments?: number;
  provider?: "gemini" | "anthropic";
  model?: string;
  name?: string;
}

export interface PipelineDebugRow {
  review_id: string;
  vod_id: string;
  job_id?: string;
  review_rank: number;
  source_rank?: number | null;
  start_time: number;
  end_time: number;
  duration: number;
  virality?: number | null;
  llm_virality?: number | null;
  hook?: number | null;
  flow?: number | null;
  value?: number | null;
  trend?: number | null;
  rationale?: string;
  transcript_excerpt?: string;
  clip_status?: string;
  clip_url?: string;
  human_viral_label?: PipelineDebugViralLabel;
  human_boundary_label?: PipelineDebugBoundaryLabel;
  human_notes?: string;
  label_updated_at?: string;
  // GT annotation (set when eval manifest is available)
  gt_hit?: boolean;
  gt_desc?: string;
  gt_coverage?: number | null;
  gt_views?: number | null;
  gt_rank?: number | null;
}

export interface PipelineDebugGtHit {
  gt_index: number;
  gt_start: number;
  gt_end: number;
  desc: string;
  views: number;
  source_url: string;
  hit: boolean;
  hit_rank: number | null;
  coverage: number;
  pred_start_time: number | null;
  pred_end_time: number | null;
}

export interface PipelineDebugRecall {
  "5": number | null;
  "10": number | null;
  "20": number | null;
}

export interface PipelineDebugVodEval {
  vod_id: string;
  has_gt: boolean;
  gt_count: number;
  pred_count: number;
  recall: PipelineDebugRecall;
  avg_rank: number | null;
  miss_rate: number | null;
  hit_count: number;
  miss_count: number;
  gt_hits: PipelineDebugGtHit[];
}

export interface PipelineDebugOverallEval {
  vods_with_gt: number;
  vods_total: number;
  gt_count: number;
  pred_count: number;
  recall: PipelineDebugRecall;
  avg_rank: number | null;
  miss_rate: number | null;
  hit_count: number;
  miss_count: number;
}

export interface PipelineDebugEval {
  per_vod: PipelineDebugVodEval[];
  overall: PipelineDebugOverallEval;
  ks: number[];
  coverage_threshold: number;
  computed_at: string;
}

export interface PipelineDebugAnchorsResponse {
  vod_ids: string[];
  vods: { vod_id: string; url: string }[];
  count: number;
}

export interface PipelineDebugWorkflow {
  workflow_id: string;
  name?: string;
  status: PipelineDebugStatus;
  stage: PipelineDebugStage;
  progress_pct: number;
  message: string;
  error?: string | null;
  created_at?: string;
  updated_at?: string;
  workflow_dir?: string;
  request?: PipelineDebugStartRequest;
  summary?: Record<string, unknown>;
  label_summary?: Record<string, unknown>;
  row_count: number;
  llm_moment_count?: number;
  eval?: PipelineDebugEval;
  rows: PipelineDebugRow[];
}

export interface PipelineDebugListResponse {
  workflows: PipelineDebugWorkflow[];
}

export interface PipelineDebugLabelRequest {
  review_id: string;
  human_viral_label: PipelineDebugViralLabel;
  human_boundary_label: PipelineDebugBoundaryLabel;
  human_notes?: string;
}

export type PipelineStageStatus = "ok" | "warning" | "pending" | "disabled" | "error";

export interface PipelineMetric {
  label: string;
  value: string | number | boolean | null;
}

export interface PipelineDiagnosticStage {
  id: string;
  label: string;
  status: PipelineStageStatus;
  summary: string;
  metrics: PipelineMetric[];
  details: string[];
}

export interface CandidateFunnelLoss {
  transition: string;
  from_stage: string;
  to_stage: string;
  input_count: number;
  output_count: number;
  lost_count: number;
  retention_ratio?: number | null;
}

export interface CandidateFunnelSample {
  candidate_id: string;
  source?: string | null;
  clip_type?: string | null;
  start?: number | null;
  end?: number | null;
  status?: string | null;
  dropped_stage?: string | null;
  drop_reason?: string | null;
  pre_judge_rank?: number | null;
  pre_judge_rank_score?: number | null;
  judge_score?: number | null;
  final_rank?: number | null;
  sample_group?: string;
}

export interface NumericDistribution {
  count?: number;
  min?: number | null;
  mean?: number | null;
  max?: number | null;
}

export interface CandidateFunnelQualityBreakdown {
  judged_kept?: number;
  unjudged_kept?: number;
  judged_rejected?: number;
  unjudged_rejected?: number;
}

export interface CandidateFunnelClipTypeLift {
  rows?: number;
  final_selected?: number;
  high_judge?: number;
}

export interface CandidateFunnelRankerLift {
  high_judge_rows?: number;
  high_judge_pre_judge_top20_ratio?: number | null;
  high_judge_pre_judge_top40_ratio?: number | null;
  high_judge_pre_judge_rank?: NumericDistribution;
  final_selected_pre_judge_rank?: NumericDistribution;
  final_selected_judge_score?: NumericDistribution;
}

export interface CandidateFunnelDiagnostics {
  available: boolean;
  rows: number;
  final_selected_rows?: number;
  stage_counts?: Record<string, number>;
  losses?: CandidateFunnelLoss[];
  largest_losses?: CandidateFunnelLoss[];
  drop_stage_counts?: Record<string, number>;
  rejection_reason_counts?: Record<string, number>;
  source_counts?: Record<string, number>;
  final_source_counts?: Record<string, number>;
  clip_type_counts?: Record<string, number>;
  final_clip_type_counts?: Record<string, number>;
  quality_breakdown?: CandidateFunnelQualityBreakdown;
  ranker_lift?: CandidateFunnelRankerLift;
  clip_type_lift?: Record<string, CandidateFunnelClipTypeLift>;
  samples?: CandidateFunnelSample[];
}

export interface PipelineDiagnostics {
  available: boolean;
  job_status?: string | null;
  last_modified?: string | null;
  summary: string;
  stages: PipelineDiagnosticStage[];
  candidate_funnel?: CandidateFunnelDiagnostics;
}

export interface StatusInfo {
  label?: string;
  name: string;
  model: string;
  active: boolean;
  error: string | null;
  mode: "active" | "fallback" | "error" | "inactive";
  fallback_name?: string | null;
  fallback_model?: string | null;
}

export interface AIProviderStatus {
  providers: Record<string, StatusInfo>;
  pipeline?: Record<string, StatusInfo>;
}

export interface JobStatus {
  job_id: string;
  status:
    | "queued"
    | "downloading"
    | "downloading_chat"
    | "extracting_audio"
    | "transcribing"
    | "extracting_features"
    | "diarizing"
    | "analyzing"
    | "judging"
    | "clipping"
    | "done"
    | "cancelled"
    | "failed";
  progress_pct: number;
  profile: string;
  clips: ClipResult[];
  error: string | null;
  download_speed: number | null;
  download_eta: number | null;
  download_cached: boolean;
  pipeline_diagnostics?: PipelineDiagnostics | null;
}
