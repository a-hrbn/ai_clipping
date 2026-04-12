"use client";

import { useEffect, useState } from "react";
import { getAIStatus } from "@/lib/api";
import { AIProviderStatus, ProviderInfo } from "@/lib/types";

type DotState = "active" | "error" | "inactive";

function getDotState(info: ProviderInfo): DotState {
  if (info.active) return "active";
  if (info.error) return "error";
  return "inactive";
}

const DOT_COLOR: Record<DotState, string> = {
  active:   "var(--accent)",
  error:    "var(--warn)",
  inactive: "var(--text-muted)",
};

const BADGE_BG: Record<DotState, string> = {
  active:   "rgba(188,255,87,0.08)",
  error:    "rgba(255,178,64,0.08)",
  inactive: "rgba(255,255,255,0.04)",
};

const BADGE_COLOR: Record<DotState, string> = {
  active:   "var(--accent)",
  error:    "var(--warn)",
  inactive: "var(--text-muted)",
};

const BADGE_BORDER: Record<DotState, string> = {
  active:   "rgba(188,255,87,0.2)",
  error:    "rgba(255,178,64,0.25)",
  inactive: "var(--border)",
};

function ProviderPill({ label, info }: { label: string; info: ProviderInfo }) {
  const state = getDotState(info);

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
          {label}
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

      {/* Inline error hint */}
      {state === "error" && info.error && (
        <p className="ml-3 text-xs font-mono" style={{ color: "var(--warn)" }}>
          ⚠ {info.error} — Fallback: Ollama
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

  return (
    <div className="flex flex-wrap items-start gap-x-5 gap-y-2">
      <ProviderPill label="Scoring" info={status.providers.semantic} />
      <span className="text-border select-none hidden sm:block mt-1">·</span>
      <ProviderPill label="Refinement" info={status.providers.refinement} />
    </div>
  );
}
