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

export type ClipFeedbackTag =
  | "strong_hook"
  | "too_much_context"
  | "missing_payoff"
  | "too_long";

export interface ClipFeedbackPayload {
  candidate_id: string;
  tags: ClipFeedbackTag[];
  notes?: string;
  publish_platform?: "general" | "tiktok" | "youtube_shorts" | "instagram_reels" | "linkedin";
  view_count?: number;
  like_count?: number;
  comment_count?: number;
  share_count?: number;
  save_count?: number;
  impressions?: number;
  avg_watch_time?: number;
  retention_3s?: number;
  retention_50pct?: number;
  completion_rate?: number;
  views_1h?: number;
  views_24h?: number;
  baseline_view_rate?: number;
  observed_view_rate?: number;
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
    | "extracting_audio"
    | "transcribing"
    | "extracting_features"
    | "diarizing"
    | "analyzing"
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
}
