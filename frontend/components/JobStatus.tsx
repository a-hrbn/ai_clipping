"use client";

import { useEffect, useState } from "react";
import { getJobStatus } from "@/lib/api";
import { JobStatus as JobStatusType, ClipResult, PipelineDiagnostics } from "@/lib/types";
import PipelineDiagnosticsPanel from "@/components/PipelineDiagnosticsPanel";

const STAGES = [
  "queued",
  "downloading",
  "extracting_audio",
  "transcribing",
  "extracting_features",
  "diarizing",
  "analyzing",
  "judging",
  "clipping",
  "done",
] as const;

const STAGE_META: Record<string, { label: string; detail: string }> = {
  queued:          { label: "Eingereiht",          detail: "Job wartet auf Verarbeitung" },
  downloading:     { label: "VOD laden",            detail: "Twitch VOD wird heruntergeladen" },
  extracting_audio:{ label: "Audio extrahieren",    detail: "Tonspur wird isoliert" },
  transcribing:    { label: "Transkribieren",        detail: "Whisper analysiert den Ton — das dauert einen Moment" },
  extracting_features:{ label: "Features extrahieren", detail: "Audio-, Video- und Chat-Signale werden ausgewertet" },
  diarizing:       { label: "Sprecher trennen",      detail: "Sprecher werden per Audioanalyse unterschieden" },
  analyzing:       { label: "KI analysiert",        detail: "Llama sucht nach viralen Momenten" },
  judging:          { label: "Clip Judge",           detail: "Ollama bewertet die Top-Kandidaten" },
  clipping:        { label: "Clips schneiden",      detail: "FFmpeg schneidet die Szenen" },
  done:            { label: "Fertig",               detail: "Alle Clips wurden erstellt" },
  cancelled:       { label: "Abgebrochen",          detail: "Verarbeitung wurde beendet" },
  failed:          { label: "Fehler",               detail: "Verarbeitung fehlgeschlagen" },
};

function formatSpeed(bps: number): string {
  if (bps >= 1_000_000) return `${(bps / 1_000_000).toFixed(1)} MB/s`;
  if (bps >= 1_000) return `${(bps / 1_000).toFixed(0)} KB/s`;
  return `${bps.toFixed(0)} B/s`;
}

function formatEta(s: number): string {
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

interface Props {
  jobId: string;
  onDone: (clips: ClipResult[]) => void;
  onError: (msg: string) => void;
  onDiagnostics?: (diagnostics: PipelineDiagnostics | null) => void;
}

export default function JobStatus({ jobId, onDone, onError, onDiagnostics }: Props) {
  const [status, setStatus] = useState("queued");
  const [pct, setPct] = useState(0);
  const [speed, setSpeed] = useState<number | null>(null);
  const [eta, setEta] = useState<number | null>(null);
  const [downloadCached, setDownloadCached] = useState(false);
  const [pipelineDiagnostics, setPipelineDiagnostics] = useState<PipelineDiagnostics | null>(null);

  useEffect(() => {
    let stopped = false;

    const poll = async () => {
      try {
        const job: JobStatusType = await getJobStatus(jobId);
        if (stopped) return;
        setStatus(job.status);
        setPct(job.progress_pct);
        setSpeed(job.download_speed ?? null);
        setEta(job.download_eta ?? null);
        setDownloadCached(job.download_cached ?? false);
        const diagnostics = job.pipeline_diagnostics ?? null;
        setPipelineDiagnostics(diagnostics);
        onDiagnostics?.(diagnostics);

        if (job.status === "done") {
          stopped = true;
          onDone(job.clips);
        } else if (job.status === "cancelled") {
          stopped = true;
          onError(job.error ?? "Verarbeitung abgebrochen");
        } else if (job.status === "failed") {
          stopped = true;
          onError(job.error ?? "Unbekannter Fehler");
        }
      } catch {
        if (stopped) return;
        stopped = true;
        onError("Verbindung zum Server verloren");
      }
    };

    poll();
    const interval = setInterval(() => {
      if (!stopped) {
        poll();
      }
    }, 2000);

    return () => {
      stopped = true;
      clearInterval(interval);
    };
  }, [jobId, onDone, onError, onDiagnostics]);

  const meta = STAGE_META[status] ?? { label: status, detail: "" };
  const stageIndex = STAGES.indexOf(status as typeof STAGES[number]);

  return (
    <div className="space-y-8">
      {/* Main progress */}
      <div className="space-y-4">
        {/* Status header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="animate-pulse-dot inline-block w-2 h-2 rounded-full bg-accent" />
            <span className="text-sm font-semibold text-tx">{meta.label}</span>
            <span className="text-xs text-muted">{meta.detail}</span>
          </div>
          <span className="font-mono text-sm font-semibold text-accent">{pct}%</span>
        </div>

        {/* Progress bar */}
        <div className="h-px w-full bg-border relative overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-accent progress-shimmer transition-all duration-700 ease-out"
            style={{ width: `${pct}%`, height: "1px" }}
          />
        </div>

        {/* Download stats */}
        {status === "downloading" && (speed != null || eta != null) && (
          <div className="flex gap-6 font-mono text-xs text-muted">
            {speed != null && (
              <span>
                <span className="text-dim mr-1">↑</span>{formatSpeed(speed)}
              </span>
            )}
            {eta != null && (
              <span>
                <span className="text-dim mr-1">ETA</span>{formatEta(eta)}
              </span>
            )}
          </div>
        )}

        {status === "downloading" && downloadCached && (
          <div className="text-xs text-muted">
            Twitch-VOD aus 30-Minuten-Cache geladen.
          </div>
        )}
      </div>

      {/* Pipeline stages */}
      <div className="border border-border divide-y divide-border">
        {STAGES.filter(s => s !== "done").map((stage, i) => {
          const stageMeta = STAGE_META[stage];
          const done = stageIndex > i;
          const active = status === stage;
          return (
            <div
              key={stage}
              className={[
                "flex items-center gap-4 px-4 py-3 transition-colors",
                active ? "bg-accent-dim" : "",
              ].join(" ")}
            >
              {/* State indicator */}
              <div className="shrink-0 w-5 h-5 flex items-center justify-center">
                {done ? (
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <circle cx="7" cy="7" r="6" fill="var(--accent)" fillOpacity="0.15" stroke="var(--accent)" strokeWidth="1" />
                    <path d="M4.5 7L6.2 8.7L9.5 5.5" stroke="var(--accent)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ) : active ? (
                  <span className="w-2 h-2 rounded-full bg-accent animate-pulse-dot" />
                ) : (
                  <span className="w-2 h-2 rounded-full bg-border-bright" />
                )}
              </div>

              {/* Label */}
              <span
                className={[
                  "text-xs font-medium uppercase tracking-widest",
                  done ? "text-muted line-through" : active ? "text-tx" : "text-muted",
                ].join(" ")}
              >
                {stageMeta.label}
              </span>

              {active && (
                <span className="ml-auto">
                  <span className="inline-block w-3 h-3 border border-accent border-t-transparent rounded-full animate-spin-slow" />
                </span>
              )}
            </div>
          );
        })}
      </div>

      <PipelineDiagnosticsPanel diagnostics={pipelineDiagnostics} />
    </div>
  );
}
