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
  proposal_score?: number | null;
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
  proposal_score?: number | null;
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

export interface CandidateFunnelProposalLift {
  scored_rows?: number;
  positive_proxy_rows?: number;
  precision_at_positive_k?: number | null;
  recall_at_positive_k?: number | null;
  high_judge_proposal_kept_ratio?: number | null;
  final_selected_proposal_score?: NumericDistribution;
  high_judge_proposal_score?: NumericDistribution;
  source_breakdown?: Record<string, { rows?: number; final_selected?: number; high_judge?: number }>;
}

export interface CandidateFunnelClipTypeLift {
  rows?: number;
  final_selected?: number;
  high_judge?: number;
  proposal_kept?: number;
  mean_proposal_score?: number | null;
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
  proposal_lift?: CandidateFunnelProposalLift;
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
