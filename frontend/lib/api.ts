import {
  ActiveClipProfile,
  AIProviderStatus,
  ClipFeedbackPayload,
  ClipRequestOptions,
  JobStatus,
  PipelineDebugAnchorsResponse,
  PipelineDebugLabelRequest,
  PipelineDebugListResponse,
  PipelineDebugRunPrompt,
  PipelineDebugStartRequest,
  PipelineDebugWorkflow,
  ReviewCandidateResponse,
  UploadProgress,
} from "./types";

const API_BASE = normalizeApiBase(process.env.NEXT_PUBLIC_API_URL);

function normalizeApiBase(value: string | undefined): string {
  const trimmed = (value ?? "").trim();
  if (!trimmed || trimmed === "/") {
    return "";
  }
  return trimmed.replace(/\/+$/, "");
}

function formatApiErrorDetail(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((item) => {
        if (typeof item === "string") {
          return item.trim();
        }
        if (item && typeof item === "object") {
          const record = item as { msg?: unknown; loc?: unknown };
          const msg = typeof record.msg === "string" ? record.msg.trim() : "";
          const loc = Array.isArray(record.loc)
            ? record.loc.map((segment) => String(segment)).join(".")
            : "";
          if (msg && loc) return `${loc}: ${msg}`;
          return msg;
        }
        return "";
      })
      .filter(Boolean);
    return parts.length ? parts.join(" | ") : null;
  }
  if (value && typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return null;
}

export async function submitJob(
  url: string,
  profile?: string | null,
  options?: ClipRequestOptions,
): Promise<JobStatus> {
  const sessionId = ensureClientSessionId();
  const res = await fetch(`${API_BASE}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url,
      session_id: sessionId,
      ...(profile ? { profile } : {}),
      ...(options ?? {}),
    }),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const json = await res.json();
      msg = json.detail ?? json.message ?? msg;
    } catch {
      const text = await res.text().catch(() => "");
      if (text) msg = text;
    }
    throw new Error(msg);
  }
  return res.json();
}

export async function submitUpload(
  file: File,
  profile?: string | null,
  onProgress?: (info: UploadProgress) => void,
  options?: ClipRequestOptions,
): Promise<JobStatus> {
  return new Promise((resolve, reject) => {
    const sessionId = ensureClientSessionId();
    const form = new FormData();
    form.append("file", file);
    if (profile) form.append("profile", profile);
    form.append("session_id", sessionId);
    if (options?.platform) form.append("platform", options.platform);
    if (options?.goal) form.append("goal", options.goal);
    if (options?.tone) form.append("tone", options.tone);
    if (options?.topic) form.append("topic", options.topic);
    if (options?.must_include?.length) form.append("must_include", options.must_include.join(","));
    if (options?.exclude?.length) form.append("exclude", options.exclude.join(","));
    if (options?.speaker) form.append("speaker", options.speaker);
    if (options?.time_range?.start_sec !== undefined) form.append("time_range_start", String(options.time_range.start_sec));
    if (options?.time_range?.end_sec !== undefined) form.append("time_range_end", String(options.time_range.end_sec));
    if (options?.clip_style) form.append("clip_style", options.clip_style);
    if (options?.variant_count !== undefined) form.append("variant_count", String(options.variant_count));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/jobs/upload`);

    let startTime: number | null = null;
    let lastLoaded = 0;
    let lastTime: number | null = null;
    // Smoothed speed using exponential moving average
    let smoothedSpeed: number | null = null;
    const ALPHA = 0.3;

    onProgress?.({
      pct: 0,
      phase: "preparing",
      loadedBytes: 0,
      totalBytes: file.size,
      speedBps: null,
      etaSec: null,
    });

    xhr.upload.addEventListener("progress", (e) => {
      if (!e.lengthComputable || !onProgress) return;

      const now = Date.now();
      if (startTime === null) startTime = now;

      // Calculate instantaneous speed since last event
      let speedBps: number | null = null;
      let etaSec: number | null = null;
      if (lastTime !== null && now > lastTime) {
        const instantSpeed = (e.loaded - lastLoaded) / ((now - lastTime) / 1000);
        smoothedSpeed =
          smoothedSpeed === null
            ? instantSpeed
            : ALPHA * instantSpeed + (1 - ALPHA) * smoothedSpeed;
        speedBps = smoothedSpeed;
        const remaining = e.total - e.loaded;
        etaSec = smoothedSpeed > 0 ? Math.round(remaining / smoothedSpeed) : null;
      }

      lastLoaded = e.loaded;
      lastTime = now;

      onProgress({
        pct: Math.round((e.loaded / e.total) * 100),
        phase: "uploading",
        loadedBytes: e.loaded,
        totalBytes: e.total,
        speedBps,
        etaSec,
      });
    });

    // All bytes sent — server is now processing the file
    xhr.upload.addEventListener("load", () => {
      onProgress?.({
        pct: 100,
        phase: "processing",
        loadedBytes: file.size,
        totalBytes: file.size,
        speedBps: null,
        etaSec: null,
      });
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Ungültige Serverantwort"));
        }
      } else {
        let msg = `HTTP ${xhr.status}`;
        try {
          const json = JSON.parse(xhr.responseText);
          msg = json.detail ?? json.message ?? msg;
        } catch {
          if (xhr.responseText) msg = xhr.responseText;
        }
        reject(new Error(msg));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("Netzwerkfehler beim Upload – ist der Server erreichbar?")));
    xhr.send(form);
  });
}

export function ensureClientSessionId(): string {
  if (typeof window === "undefined") {
    return "server";
  }
  const key = "ai-clipping-session-id";
  let sessionId = window.sessionStorage.getItem(key);
  if (!sessionId) {
    sessionId = window.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    window.sessionStorage.setItem(key, sessionId);
  }
  return sessionId;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const sessionId = ensureClientSessionId();
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}?session_id=${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/cancel`, {
    method: "POST",
    keepalive: true,
  });
  if (!res.ok && res.status !== 404) {
    throw new Error(`HTTP ${res.status}`);
  }
}

export async function deleteJobRun(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.detail ?? `HTTP ${res.status}`);
  }
}

export async function deletePipelineDebugWorkflow(workflowId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/pipeline-debug/workflows/${encodeURIComponent(workflowId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.detail ?? `HTTP ${res.status}`);
  }
}

export function cancelJobBeacon(jobId: string): void {
  const url = `${API_BASE}/api/jobs/${jobId}/cancel`;
  if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
    const queued = navigator.sendBeacon(url);
    if (queued) {
      return;
    }
  }
  void fetch(url, { method: "POST", keepalive: true }).catch(() => {});
}

export function getDownloadUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export async function getAIStatus(): Promise<AIProviderStatus> {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function submitClipFeedback(jobId: string, payload: ClipFeedbackPayload): Promise<void> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const json = await res.json();
      msg = formatApiErrorDetail(json.detail) ?? formatApiErrorDetail(json.message) ?? msg;
    } catch {
      const text = await res.text().catch(() => "");
      if (text) msg = text;
    }
    throw new Error(msg);
  }
}

export async function getReviewCandidates(jobId: string, limit = 80): Promise<ReviewCandidateResponse> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/review-candidates?limit=${encodeURIComponent(String(limit))}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getPipelineDebugAnchors(): Promise<PipelineDebugAnchorsResponse> {
  const res = await fetch(`${API_BASE}/api/pipeline-debug/anchors`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function startPipelineDebugWorkflow(
  payload: PipelineDebugStartRequest,
): Promise<PipelineDebugWorkflow> {
  const res = await fetch(`${API_BASE}/api/pipeline-debug/workflows`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const json = await res.json();
      msg = formatApiErrorDetail(json.detail) ?? formatApiErrorDetail(json.message) ?? msg;
    } catch {
      const text = await res.text().catch(() => "");
      if (text) msg = text;
    }
    throw new Error(msg);
  }
  return res.json();
}

export async function listPipelineDebugWorkflows(limit = 10): Promise<PipelineDebugListResponse> {
  const res = await fetch(`${API_BASE}/api/pipeline-debug/workflows?limit=${encodeURIComponent(String(limit))}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getPipelineDebugWorkflow(workflowId: string): Promise<PipelineDebugWorkflow> {
  const res = await fetch(`${API_BASE}/api/pipeline-debug/workflows/${encodeURIComponent(workflowId)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getActiveClipProfile(profile = "jynxzi"): Promise<ActiveClipProfile> {
  const res = await fetch(`${API_BASE}/api/pipeline-debug/active-profile?profile=${encodeURIComponent(profile)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getPipelineDebugRunPrompt(workflowId: string, vodId: string): Promise<PipelineDebugRunPrompt> {
  const res = await fetch(
    `${API_BASE}/api/pipeline-debug/workflows/${encodeURIComponent(workflowId)}/prompt?vod_id=${encodeURIComponent(vodId)}`,
  );
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function savePipelineDebugLabel(
  workflowId: string,
  payload: PipelineDebugLabelRequest,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/pipeline-debug/workflows/${encodeURIComponent(workflowId)}/labels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const json = await res.json();
      msg = formatApiErrorDetail(json.detail) ?? formatApiErrorDetail(json.message) ?? msg;
    } catch {
      const text = await res.text().catch(() => "");
      if (text) msg = text;
    }
    throw new Error(msg);
  }
}
