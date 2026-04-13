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
  ranking_mode: "heuristic" | "ml_blended";
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
  view_count?: number;
  like_count?: number;
  comment_count?: number;
  share_count?: number;
  save_count?: number;
  views_1h?: number;
  views_24h?: number;
  baseline_view_rate?: number;
  observed_view_rate?: number;
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
    | "failed";
  progress_pct: number;
  profile: string;
  clips: ClipResult[];
  error: string | null;
  download_speed: number | null;
  download_eta: number | null;
  download_cached: boolean;
}
