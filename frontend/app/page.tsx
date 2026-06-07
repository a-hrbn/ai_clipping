"use client";

import { useState, useCallback, useEffect } from "react";
import { cancelJobBeacon, submitJob, submitUpload } from "@/lib/api";
import { ClipResult, PipelineDiagnostics, UploadProgress } from "@/lib/types";
import UrlInput from "@/components/UrlInput";
import JobStatus from "@/components/JobStatus";
import ClipGrid from "@/components/ClipGrid";
import AIStatus from "@/components/AIStatus";
import PipelineDiagnosticsPanel from "@/components/PipelineDiagnosticsPanel";
import CandidateReviewPanel from "@/components/CandidateReviewPanel";
import LlmMomentReviewWorkbench from "@/components/LlmMomentReviewWorkbench";

type AppState = "idle" | "submitting" | "uploading" | "polling" | "done" | "error";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatSpeed(bps: number): string {
  if (bps < 1024) return `${Math.round(bps)} B/s`;
  if (bps < 1024 * 1024) return `${(bps / 1024).toFixed(0)} KB/s`;
  return `${(bps / (1024 * 1024)).toFixed(1)} MB/s`;
}

function formatEta(sec: number): string {
  if (sec < 60) return `${sec} Sek.`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")} Min.`;
}

const PHASE_LABEL: Record<UploadProgress["phase"], string> = {
  preparing: "Verbindung aufbauen",
  uploading: "Datei übertragen",
  processing: "Server verarbeitet",
};

const PHASE_DETAIL: Record<UploadProgress["phase"], string> = {
  preparing: "Verbindung zum Server wird hergestellt…",
  uploading: "Datei wird hochgeladen",
  processing: "Upload abgeschlossen — Server verarbeitet die Datei…",
};

function UploadProgressPanel({ info }: { info: UploadProgress | null }) {
  const pct = info?.pct ?? 0;
  const phase = info?.phase ?? "preparing";

  return (
    <div className="w-full max-w-3xl mt-6 animate-fade-up delay-300">
      {/* Phase header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {/* Animated pulse dot */}
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75" />
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-accent" />
          </span>
          <span className="text-xs font-medium uppercase tracking-widest text-dim">
            {PHASE_LABEL[phase]}
          </span>
        </div>
        <span className="font-mono text-xs text-accent">{pct}%</span>
      </div>

      {/* Progress bar */}
      <div className="h-px w-full bg-border overflow-hidden relative mb-3">
        <div
          className="absolute inset-y-0 left-0 bg-accent progress-shimmer transition-[width] duration-300"
          style={{ width: `${pct}%`, height: "1px" }}
        />
      </div>

      {/* Detail row */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted">{PHASE_DETAIL[phase]}</span>

        {/* Stats: bytes and speed/eta — only during active upload */}
        {info && phase === "uploading" && (
          <div className="flex items-center gap-4 shrink-0 ml-4">
            {/* Bytes transferred */}
            <span className="font-mono text-xs text-dim">
              {formatBytes(info.loadedBytes)}
              {info.totalBytes > 0 && (
                <span className="text-muted"> / {formatBytes(info.totalBytes)}</span>
              )}
            </span>
            {/* Speed */}
            {info.speedBps !== null && (
              <span className="font-mono text-xs text-accent">
                {formatSpeed(info.speedBps)}
              </span>
            )}
            {/* ETA */}
            {info.etaSec !== null && info.etaSec > 0 && (
              <span className="font-mono text-xs text-muted">
                noch ~{formatEta(info.etaSec)}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [clips, setClips] = useState<ClipResult[]>([]);
  const [error, setError] = useState<string>("");
  const [uploadInfo, setUploadInfo] = useState<UploadProgress | null>(null);
  const [pipelineDiagnostics, setPipelineDiagnostics] = useState<PipelineDiagnostics | null>(null);

  useEffect(() => {
    if (!jobId || state !== "polling") {
      return;
    }

    const cancelActiveJob = () => {
      cancelJobBeacon(jobId);
    };

    window.addEventListener("pagehide", cancelActiveJob);
    window.addEventListener("beforeunload", cancelActiveJob);
    return () => {
      window.removeEventListener("pagehide", cancelActiveJob);
      window.removeEventListener("beforeunload", cancelActiveJob);
    };
  }, [jobId, state]);

  async function handleSubmitUrl(url: string, profile: string | null) {
    setState("submitting");
    setError("");
    setPipelineDiagnostics(null);
    try {
      const job = await submitJob(url, profile);
      setJobId(job.job_id);
      setState("polling");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState("error");
    }
  }

  async function handleSubmitFile(file: File, profile: string | null) {
    setState("uploading");
    setUploadInfo(null);
    setError("");
    setPipelineDiagnostics(null);
    try {
      const job = await submitUpload(file, profile, setUploadInfo);
      setJobId(job.job_id);
      setState("polling");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState("error");
    }
  }

  const handleDone = useCallback((results: ClipResult[]) => {
    setClips(results);
    setState("done");
  }, []);

  const handleError = useCallback((msg: string) => {
    setError(msg);
    setState("error");
  }, []);

  function reset() {
    setState("idle");
    setJobId(null);
    setClips([]);
    setError("");
    setUploadInfo(null);
    setPipelineDiagnostics(null);
  }

  return (
    <main className="min-h-screen flex flex-col items-center px-4 pb-24 relative">
      {/* Background glow */}
      <div
        aria-hidden
        className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-[900px] h-[500px]"
        style={{
          background:
            "radial-gradient(ellipse at 50% 0%, rgba(188,255,87,0.055) 0%, transparent 65%)",
        }}
      />

      {/* Header */}
      <header className="w-full max-w-3xl pt-16 pb-12 animate-fade-up">
        <div className="flex items-start gap-4">
          {/* Logo mark */}
          <div className="mt-1.5 flex h-8 w-8 shrink-0 items-center justify-center rounded border border-accent/30 bg-accent-dim">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 3L8 1L13 3V8C13 11.5 8 15 8 15C8 15 3 11.5 3 8V3Z" stroke="#BCFF57" strokeWidth="1.2" fill="none" />
              <path d="M6 8L7.5 9.5L10 6.5" stroke="#BCFF57" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <h1
              className="text-5xl font-extrabold leading-none tracking-tight text-tx"
              style={{ letterSpacing: "-0.03em" }}
            >
              AI Clipping
            </h1>
            <p className="mt-2 text-sm font-medium uppercase tracking-widest text-muted">
              Virale Momente — automatisch erkannt
            </p>
          </div>
        </div>
      </header>

      {/* AI provider status — below the title */}
      <div className="w-full max-w-3xl pb-5 animate-fade-up delay-100">
        <AIStatus />
      </div>

      <div className="w-full pb-8 animate-fade-up delay-100">
        <LlmMomentReviewWorkbench />
      </div>

      {/* Divider */}
      <div className="w-full max-w-3xl border-t border-border animate-fade-up delay-100" />

      {/* Input section */}
      {(state === "idle" || state === "submitting" || state === "uploading" || state === "error") && (
        <section className="relative z-10 w-full max-w-3xl mt-10 animate-fade-up delay-200">
          <UrlInput
            onSubmitUrl={handleSubmitUrl}
            onSubmitFile={handleSubmitFile}
            submitting={state === "submitting"}
            uploading={state === "uploading"}
          />
        </section>
      )}

      {/* Upload progress */}
      {state === "uploading" && <UploadProgressPanel info={uploadInfo} />}

      {/* Error */}
      {state === "error" && (
        <div className="w-full max-w-3xl mt-6 animate-fade-up">
          <div className="border border-danger/30 bg-danger/5 px-5 py-4 flex items-start gap-3">
            <div className="mt-0.5 shrink-0 text-danger">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.2" />
                <path d="M7 4.5V7.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                <circle cx="7" cy="9.5" r="0.6" fill="currentColor" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-danger font-medium">Fehler</p>
              <p className="mt-0.5 text-sm text-dim">{error}</p>
            </div>
            <button
              onClick={reset}
              className="shrink-0 text-xs font-medium text-muted hover:text-tx transition-colors uppercase tracking-wider"
            >
              Wiederholen
            </button>
          </div>
        </div>
      )}

      {/* Job polling */}
      {state === "polling" && jobId && (
        <section className="w-full max-w-3xl mt-10 animate-fade-up">
          <JobStatus
            jobId={jobId}
            onDone={handleDone}
            onError={handleError}
            onDiagnostics={setPipelineDiagnostics}
          />
        </section>
      )}

      {/* Results */}
      {state === "done" && (
        <section className="w-full max-w-3xl mt-10 animate-fade-up">
          {pipelineDiagnostics?.available && (
            <div className="mb-8">
              <PipelineDiagnosticsPanel diagnostics={pipelineDiagnostics} />
            </div>
          )}
          {jobId && (
            <div className="mb-8">
              <CandidateReviewPanel jobId={jobId} />
            </div>
          )}
          <ClipGrid clips={clips} jobId={jobId ?? ""} />
          <div className="mt-12 flex justify-center">
            <button
              onClick={reset}
              className="text-xs font-medium uppercase tracking-widest text-muted hover:text-tx border-b border-transparent hover:border-muted transition-all pb-0.5"
            >
              Neues Video analysieren
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
