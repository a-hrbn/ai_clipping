import { AIProviderStatus, ClipFeedbackPayload, JobStatus, UploadProgress } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function submitJob(url: string, profile?: string | null): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, ...(profile ? { profile } : {}) }),
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
): Promise<JobStatus> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    if (profile) form.append("profile", profile);
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

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
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
      msg = json.detail ?? json.message ?? msg;
    } catch {
      const text = await res.text().catch(() => "");
      if (text) msg = text;
    }
    throw new Error(msg);
  }
}
