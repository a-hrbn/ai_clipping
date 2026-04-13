"use client";

import { useEffect, useState } from "react";
import { getAIStatus } from "@/lib/api";
import { AIProviderStatus, StatusInfo } from "@/lib/types";

type DotState = "active" | "fallback" | "error" | "inactive";

function getDotState(info: StatusInfo): DotState {
  if (info.mode === "fallback") return "fallback";
  if (info.mode === "active") return "active";
  if (info.mode === "error") return "error";
  if (info.error) return "error";
  return "inactive";
}

const DOT_COLOR: Record<DotState, string> = {
  active:   "var(--accent)",
  fallback: "var(--warn)",
  error:    "var(--warn)",
  inactive: "var(--text-muted)",
};

const BADGE_BG: Record<DotState, string> = {
  active:   "rgba(188,255,87,0.08)",
  fallback: "rgba(255,178,64,0.08)",
  error:    "rgba(255,178,64,0.08)",
  inactive: "rgba(255,255,255,0.04)",
};

const BADGE_COLOR: Record<DotState, string> = {
  active:   "var(--accent)",
  fallback: "var(--warn)",
  error:    "var(--warn)",
  inactive: "var(--text-muted)",
};

const BADGE_BORDER: Record<DotState, string> = {
  active:   "rgba(188,255,87,0.2)",
  fallback: "rgba(255,178,64,0.25)",
  error:    "rgba(255,178,64,0.25)",
  inactive: "var(--border)",
};

function ProviderPill({ info }: { info: StatusInfo }) {
  const state = getDotState(info);
  const fallbackTarget =
    info.mode === "fallback" && info.fallback_name && info.fallback_model
      ? `${info.fallback_name} · ${info.fallback_model}`
      : null;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-1.5">
        {/* Status dot */}
        <span className="relative flex h-1.5 w-1.5 shrink-0">
          {state === "active" && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-60" />
          )}
          <span
            className="relative inline-flex rounded-full h-1.5 w-1.5"
            style={{ background: DOT_COLOR[state] }}
          />
        </span>

        {/* Label */}
        <span className="text-xs font-medium uppercase tracking-widest text-dim">
          {info.label}
        </span>

        {/* Model badge */}
        <span
          className="font-mono text-xs px-1.5 py-0.5 rounded"
          style={{
            background: BADGE_BG[state],
            color: BADGE_COLOR[state],
            border: `1px solid ${BADGE_BORDER[state]}`,
          }}
        >
          {info.name} · {info.model}
        </span>
      </div>

      {state === "fallback" && (
        <p className="ml-3 text-xs font-mono" style={{ color: "var(--warn)" }}>
          {fallbackTarget
            ? `Fallback aktiv: ${fallbackTarget}${info.error ? ` — ${info.error}` : ""}`
            : info.error
              ? `Fallback aktiv: ${info.error}`
              : "Lokaler Fallback aktiv"}
        </p>
      )}

      {/* Inline error hint */}
      {(state === "error" || state === "inactive") && info.error && (
        <p className="ml-3 text-xs font-mono" style={{ color: "var(--warn)" }}>
          {info.error}
        </p>
      )}
    </div>
  );
}

export default function AIStatus() {
  const [status, setStatus] = useState<AIProviderStatus | null>(null);

  useEffect(() => {
    getAIStatus()
      .then(setStatus)
      .catch(() => {/* server not yet running — silent */});
  }, []);

  if (!status) return null;

  const providerLabels: Record<string, string> = {
    semantic: "Scoring",
    refinement: "Refinement",
  };
  const pipeline = status.pipeline ?? {};
  const normalizeInfo = (key: string, info: StatusInfo): StatusInfo => ({
    ...info,
    label: info.label ?? providerLabels[key] ?? key.replaceAll("_", " "),
  });

  const providerOrder = ["semantic", "refinement"];
  const pipelineOrder = [
    "transcription",
    "semantic_embeddings",
    "face_detection",
    "expression_analysis",
    "diarization",
  ];
  const orderedProviders = [
    ...providerOrder
      .filter((key) => status.providers[key])
      .map((key) => normalizeInfo(key, status.providers[key])),
    ...Object.entries(status.providers)
      .filter(([key]) => !providerOrder.includes(key))
      .map(([key, info]) => normalizeInfo(key, info)),
  ];
  const orderedPipeline = [
    ...pipelineOrder.filter((key) => pipeline[key]).map((key) => normalizeInfo(key, pipeline[key])),
    ...Object.entries(pipeline)
      .filter(([key]) => !pipelineOrder.includes(key))
      .map(([key, info]) => normalizeInfo(key, info)),
  ];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-start gap-x-5 gap-y-2">
        {orderedProviders.map((info) => (
          <ProviderPill key={`${info.label}:${info.model}`} info={info} />
        ))}
      </div>
      {orderedPipeline.length > 0 && (
        <div className="pt-3 border-t border-border/70">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.22em] text-muted">
            Pipeline
          </p>
          <div className="flex flex-wrap items-start gap-x-5 gap-y-2">
            {orderedPipeline.map((info) => (
              <ProviderPill key={`${info.label}:${info.model}`} info={info} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
