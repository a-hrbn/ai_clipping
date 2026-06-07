"use client";

import { useEffect, useMemo, useState } from "react";

import {
  attachLatestLlmMomentReviewRecallFloor,
  getDownloadUrl,
  getLlmMomentReviewWorkflow,
  listLlmMomentReviewWorkflows,
  saveLlmMomentReviewLabel,
  startLlmMomentReviewWorkflow,
} from "@/lib/api";
import {
  LlmMomentReviewBoundaryLabel,
  LlmMomentReviewRow,
  LlmMomentReviewViralLabel,
  LlmMomentReviewWorkflow,
} from "@/lib/types";

type LabelState = {
  viral: LlmMomentReviewViralLabel;
  boundary: LlmMomentReviewBoundaryLabel;
  notes: string;
  saving: boolean;
  saved: boolean;
  error: string | null;
};

const VIRAL_OPTIONS: Array<{ value: LlmMomentReviewViralLabel; label: string }> = [
  { value: "yes", label: "Viral" },
  { value: "unsure", label: "Unsicher" },
  { value: "no", label: "Nein" },
];

const BOUNDARY_OPTIONS: Array<{ value: LlmMomentReviewBoundaryLabel; label: string }> = [
  { value: "good", label: "Gut" },
  { value: "too_wide", label: "Zu breit" },
  { value: "too_short", label: "Zu kurz" },
  { value: "wrong", label: "Falsch" },
];

function splitVodIds(value: string): string[] {
  return value
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function fmtTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

function fmtScore(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "-";
}

function fmtMaybeTime(value: number | string | null | undefined): string {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? fmtTime(numeric) : "-";
}

function recordNumber(record: Record<string, unknown> | undefined, key: string): number | null {
  const value = record?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function fmtPct(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "-";
}

function gateReasons(value: unknown): string {
  return Array.isArray(value) ? value.map((item) => String(item)).join(", ") : "";
}

function defaultLabelState(row?: LlmMomentReviewRow): LabelState {
  return {
    viral: row?.human_viral_label ?? "",
    boundary: row?.human_boundary_label ?? "",
    notes: row?.human_notes ?? "",
    saving: false,
    saved: false,
    error: null,
  };
}

function completion(rows: LlmMomentReviewRow[]): string {
  if (!rows.length) return "0/0";
  const done = rows.filter((row) => row.human_viral_label || row.human_boundary_label).length;
  return `${done}/${rows.length}`;
}

export default function LlmMomentReviewWorkbench() {
  const [vodIds, setVodIds] = useState(
    "2785392475,2783829169,2783108932,2782348214,2781590654,2780984594,2779787387,2779061928,2778362653,2777621798",
  );
  const [vodLimit, setVodLimit] = useState(10);
  const [topNPerVod, setTopNPerVod] = useState(12);
  const [maxMoments, setMaxMoments] = useState(180);
  const [maxTokens, setMaxTokens] = useState(16000);
  const [workflow, setWorkflow] = useState<LlmMomentReviewWorkflow | null>(null);
  const [workflows, setWorkflows] = useState<LlmMomentReviewWorkflow[]>([]);
  const [labels, setLabels] = useState<Record<string, LabelState>>({});
  const [starting, setStarting] = useState(false);
  const [attachingRecall, setAttachingRecall] = useState(false);
  const [loadingList, setLoadingList] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isActive = workflow?.status === "queued" || workflow?.status === "running";
  const rows = useMemo(() => workflow?.rows ?? [], [workflow?.rows]);
  const groupedRows = useMemo(() => {
    const groups: Record<string, LlmMomentReviewRow[]> = {};
    for (const row of rows) {
      const key = row.vod_id || "unknown";
      groups[key] = [...(groups[key] ?? []), row];
    }
    return groups;
  }, [rows]);

  useEffect(() => {
    void refreshWorkflows();
  }, []);

  useEffect(() => {
    if (!workflow?.workflow_id || !isActive) return;
    const timer = window.setInterval(() => {
      void refreshWorkflow(workflow.workflow_id);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [workflow?.workflow_id, isActive]);

  useEffect(() => {
    setLabels((current) => {
      const next = { ...current };
      for (const row of rows) {
        if (!next[row.review_id]) {
          next[row.review_id] = defaultLabelState(row);
        }
      }
      return next;
    });
  }, [rows]);

  async function refreshWorkflows() {
    setLoadingList(true);
    try {
      const response = await listLlmMomentReviewWorkflows(8);
      setWorkflows(response.workflows);
    } catch {
      // The workbench remains usable even before the backend route exists.
    } finally {
      setLoadingList(false);
    }
  }

  async function refreshWorkflow(workflowId: string) {
    try {
      const next = await getLlmMomentReviewWorkflow(workflowId);
      setWorkflow(next);
      setWorkflows((current) => {
        const rest = current.filter((item) => item.workflow_id !== next.workflow_id);
        return [next, ...rest].slice(0, 8);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Workflow konnte nicht geladen werden");
    }
  }

  async function startWorkflow() {
    setStarting(true);
    setError(null);
    try {
      const next = await startLlmMomentReviewWorkflow({
        profile: "jynxzi",
        vod_ids: splitVodIds(vodIds),
        vod_limit: Math.max(1, vodLimit),
        top_n: maxMoments,
        max_moments: maxMoments,
        max_tokens: maxTokens,
        provider: "gemini",
        top_n_per_vod: topNPerVod,
        max_duration_sec: 120,
        exclude_known_reposts: false,
        phase2_min_non_repost_precision: 0.5,
        cut_clips: true,
        allow_holdout: true,
      });
      setWorkflow(next);
      setWorkflows((current) => [next, ...current.filter((item) => item.workflow_id !== next.workflow_id)].slice(0, 8));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Workflow konnte nicht gestartet werden");
    } finally {
      setStarting(false);
    }
  }

  function patchLabel(reviewId: string, patch: Partial<LabelState>) {
    setLabels((current) => ({
      ...current,
      [reviewId]: {
        ...defaultLabelState(),
        ...(current[reviewId] ?? {}),
        ...patch,
        saved: patch.saved ?? false,
      },
    }));
  }

  async function saveLabel(row: LlmMomentReviewRow) {
    if (!workflow?.workflow_id) return;
    const state = labels[row.review_id] ?? defaultLabelState(row);
    patchLabel(row.review_id, { saving: true, saved: false, error: null });
    try {
      await saveLlmMomentReviewLabel(workflow.workflow_id, {
        review_id: row.review_id,
        human_viral_label: state.viral,
        human_boundary_label: state.boundary,
        human_notes: state.notes,
      });
      patchLabel(row.review_id, { saving: false, saved: true });
      void refreshWorkflow(workflow.workflow_id);
    } catch (err) {
      patchLabel(row.review_id, {
        saving: false,
        error: err instanceof Error ? err.message : "Label konnte nicht gespeichert werden",
      });
    }
  }

  async function attachRecallFloor() {
    if (!workflow?.workflow_id) return;
    setAttachingRecall(true);
    setError(null);
    try {
      const next = await attachLatestLlmMomentReviewRecallFloor(workflow.workflow_id);
      setWorkflow(next);
      setWorkflows((current) => {
        const rest = current.filter((item) => item.workflow_id !== next.workflow_id);
        return [next, ...rest].slice(0, 8);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recall-Floor konnte nicht geladen werden");
    } finally {
      setAttachingRecall(false);
    }
  }

  return (
    <section className="w-full max-w-5xl border border-border bg-surface p-4 md:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-muted">LLM Moment Review</p>
          <h2 className="mt-1 text-xl font-bold text-tx">Moment-Finder und Precision Review</h2>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={refreshWorkflows}
            disabled={loadingList}
            className="border border-border px-3 py-2 text-[10px] uppercase tracking-widest text-muted hover:border-border-bright hover:text-tx disabled:opacity-60"
          >
            {loadingList ? "Lädt" : "Aktualisieren"}
          </button>
          <button
            type="button"
            onClick={startWorkflow}
            disabled={starting || isActive}
            className="border border-accent/40 bg-accent-dim px-3 py-2 text-[10px] uppercase tracking-widest text-accent hover:border-accent disabled:opacity-50"
          >
            {starting ? "Startet" : isActive ? "Läuft" : "Run starten"}
          </button>
          {workflow && (
            <button
              type="button"
              onClick={attachRecallFloor}
              disabled={attachingRecall || isActive}
              className="border border-border px-3 py-2 text-[10px] uppercase tracking-widest text-muted hover:border-border-bright hover:text-tx disabled:opacity-60"
            >
              {attachingRecall ? "Lädt" : "Recall Floor"}
            </button>
          )}
        </div>
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-[2fr_1fr_1fr_1fr]">
        <label className="block">
          <span className="text-[10px] uppercase tracking-widest text-muted">VOD IDs</span>
          <input
            value={vodIds}
            onChange={(event) => setVodIds(event.target.value)}
            className="mt-1 w-full border border-border bg-bg px-3 py-2 font-mono text-xs text-tx focus:border-border-bright focus:outline-none"
          />
        </label>
        <NumberField label="VOD Limit" value={vodLimit} min={1} max={10} onChange={setVodLimit} />
        <NumberField label="Top/VOD" value={topNPerVod} min={1} max={80} onChange={setTopNPerVod} />
        <NumberField label="Momente" value={maxMoments} min={1} max={500} onChange={setMaxMoments} />
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-4">
        <NumberField label="Max Tokens" value={maxTokens} min={512} max={64000} onChange={setMaxTokens} />
        <ReadOnlyPill label="Provider" value="gemini" />
        <ReadOnlyPill label="Queue" value="non-reposts" />
        <ReadOnlyPill label="Holdout" value="allowed" />
      </div>

      {error && <p className="mt-4 border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger">{error}</p>}

      {workflow && (
        <div className="mt-5 border border-border bg-bg/40 p-3">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="font-mono text-xs text-muted">{workflow.workflow_id}</p>
              <p className="mt-1 text-sm font-medium text-tx">{workflow.message}</p>
              {workflow.error && <p className="mt-1 text-xs text-danger">{workflow.error}</p>}
            </div>
            <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-widest text-muted">
              <span className="border border-border px-2 py-1">{workflow.status}</span>
              <span className="border border-border px-2 py-1">{workflow.stage}</span>
              <span className="border border-border px-2 py-1">Labels {completion(rows)}</span>
            </div>
          </div>
          <div className="mt-3 h-px w-full bg-border">
            <div className="h-px bg-accent transition-[width]" style={{ width: `${workflow.progress_pct}%` }} />
          </div>
          <Phase2GatePanel workflow={workflow} />
        </div>
      )}

      {!workflow && workflows.length > 0 && (
        <div className="mt-5 flex flex-wrap gap-2">
          {workflows.map((item) => (
            <button
              key={item.workflow_id}
              type="button"
              onClick={() => refreshWorkflow(item.workflow_id)}
              className="border border-border px-3 py-2 text-left text-[11px] text-muted hover:border-border-bright hover:text-tx"
            >
              <span className="block font-mono">{item.workflow_id}</span>
              <span className="block">{item.status} · {item.row_count} Clips</span>
            </button>
          ))}
        </div>
      )}

      {workflow?.status === "done" && rows.length > 0 && (
        <div className="mt-6 flex flex-col gap-6">
          {Object.entries(groupedRows).map(([vodId, vodRows]) => (
            <section key={vodId}>
              <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
                <h3 className="font-mono text-sm font-semibold text-tx">VOD {vodId}</h3>
                <span className="text-[10px] uppercase tracking-widest text-muted">{vodRows.length} Clips</span>
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                {vodRows.map((row) => (
                  <ReviewCard
                    key={row.review_id}
                    row={row}
                    state={labels[row.review_id] ?? defaultLabelState(row)}
                    onPatch={(patch) => patchLabel(row.review_id, patch)}
                    onSave={() => saveLabel(row)}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </section>
  );
}

function Phase2GatePanel({ workflow }: { workflow: LlmMomentReviewWorkflow }) {
  const labelSummary = workflow.label_summary;
  const gate = workflow.phase2_gate;
  const recallFloor = gate?.recall_floor as Record<string, unknown> | undefined;
  const nonRepostRows = recordNumber(labelSummary, "non_repost_rows");
  const labeledRows = recordNumber(labelSummary, "labeled_non_repost_rows");
  const yesCount = recordNumber(labelSummary, "non_repost_yes");
  const noCount = recordNumber(labelSummary, "non_repost_no");
  const unsureCount = recordNumber(labelSummary, "non_repost_unsure");
  const precision = labelSummary?.non_repost_yes_no_precision;
  const strictPrecision = labelSummary?.non_repost_strict_precision;
  const minPrecision = gate?.min_non_repost_precision;
  const recallRatio = recallFloor?.ratio;
  const blockingReasons = gateReasons(gate?.blocking_reasons);
  return (
    <div className="mt-4 grid gap-3 border-t border-border pt-3 md:grid-cols-3">
      <MetricBlock
        label="Non-Repost Labels"
        value={`${labeledRows ?? 0}/${nonRepostRows ?? 0}`}
        detail={`Ja ${yesCount ?? 0} · Nein ${noCount ?? 0} · Unsicher ${unsureCount ?? 0}`}
      />
      <MetricBlock
        label="Human Precision"
        value={fmtPct(precision)}
        detail={`Strict ${fmtPct(strictPrecision)} · Gate ${fmtPct(minPrecision)}`}
      />
      <MetricBlock
        label="Recall Floor"
        value={fmtPct(recallRatio)}
        detail={recallFloor?.available === true ? (recallFloor?.passes ? "über Baseline" : "unter Baseline") : "Holdout fehlt"}
      />
      {blockingReasons && (
        <p className="md:col-span-3 border border-border bg-surface px-3 py-2 text-[11px] text-muted">
          {blockingReasons}
        </p>
      )}
    </div>
  );
}

function MetricBlock({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="border border-border bg-surface px-3 py-2">
      <p className="text-[10px] uppercase tracking-widest text-muted">{label}</p>
      <p className="mt-1 font-mono text-sm font-semibold text-tx">{value}</p>
      <p className="mt-1 text-[11px] text-dim">{detail}</p>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="text-[10px] uppercase tracking-widest text-muted">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-1 w-full border border-border bg-bg px-3 py-2 font-mono text-xs text-tx focus:border-border-bright focus:outline-none"
      />
    </label>
  );
}

function ReadOnlyPill({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-[10px] uppercase tracking-widest text-muted">{label}</span>
      <div className="mt-1 border border-border bg-bg px-3 py-2 font-mono text-xs text-dim">{value}</div>
    </div>
  );
}

function ReviewCard({
  row,
  state,
  onPatch,
  onSave,
}: {
  row: LlmMomentReviewRow;
  state: LabelState;
  onPatch: (patch: Partial<LabelState>) => void;
  onSave: () => void;
}) {
  const clipUrl = row.clip_url ? getDownloadUrl(row.clip_url) : null;
  return (
    <article className="border border-border bg-bg/50 p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] uppercase tracking-widest text-muted">#{row.review_rank} · Rank {row.source_rank ?? "-"}</p>
          <p className="mt-1 font-mono text-xs text-dim">
            {fmtTime(row.start)} → {fmtTime(row.end)} · {Math.round(row.duration)}s
          </p>
        </div>
        {row.known_repost_match && (
          <span className="shrink-0 border border-accent/40 bg-accent-dim px-2 py-1 text-[10px] uppercase tracking-widest text-accent">
            Repost
          </span>
        )}
        {row.timestamp_repaired && (
          <span className="shrink-0 border border-warn/40 bg-warn/10 px-2 py-1 text-[10px] uppercase tracking-widest text-warn">
            Timestamp fixed
          </span>
        )}
      </div>

      <div className="mt-3">
        {clipUrl && row.clip_status === "ready" ? (
          <video src={clipUrl} controls preload="metadata" playsInline className="block w-full bg-black border border-border" />
        ) : (
          <div className="flex aspect-video items-center justify-center border border-border bg-surface text-xs text-muted">
            {row.clip_error || "Clip nicht verfügbar"}
          </div>
        )}
      </div>

      <div className="mt-3 grid grid-cols-5 gap-2 text-[11px] text-dim">
        <span>V {fmtScore(row.virality)}</span>
        <span>H {fmtScore(row.hook)}</span>
        <span>F {fmtScore(row.flow)}</span>
        <span>W {fmtScore(row.value)}</span>
        <span>T {fmtScore(row.trend)}</span>
      </div>

      {row.transcript_excerpt && (
        <div className="mt-3">
          <p className="text-[10px] uppercase tracking-widest text-muted">Transcript</p>
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap border border-border bg-surface p-2 text-[11px] leading-relaxed text-muted">
            {row.transcript_excerpt}
          </pre>
        </div>
      )}
      {row.rationale && (
        <div className="mt-3 border border-border bg-surface/60 p-2">
          <p className="text-[10px] uppercase tracking-widest text-muted">LLM Rationale</p>
          <p className="mt-1 text-xs leading-relaxed text-dim">{row.rationale}</p>
          {row.timestamp_repaired && (
            <p className="mt-2 text-[11px] text-warn">
              Timestamp fixed from {fmtMaybeTime(row.original_start)} → {fmtMaybeTime(row.original_end)}
            </p>
          )}
        </div>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        {VIRAL_OPTIONS.map((option) => (
          <ToggleButton
            key={option.value}
            active={state.viral === option.value}
            label={option.label}
            onClick={() => onPatch({ viral: option.value })}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        {BOUNDARY_OPTIONS.map((option) => (
          <ToggleButton
            key={option.value}
            active={state.boundary === option.value}
            label={option.label}
            onClick={() => onPatch({ boundary: option.value })}
          />
        ))}
      </div>

      <textarea
        value={state.notes}
        onChange={(event) => onPatch({ notes: event.target.value })}
        placeholder="Notiz"
        className="mt-3 min-h-[60px] w-full border border-border bg-surface px-2 py-2 text-xs text-tx placeholder:text-muted focus:border-border-bright focus:outline-none"
      />

      <div className="mt-3 flex items-center gap-3">
        <button
          type="button"
          onClick={onSave}
          disabled={state.saving || (!state.viral && !state.boundary && !state.notes.trim())}
          className="border border-border-bright px-3 py-2 text-[10px] uppercase tracking-widest text-tx hover:border-accent/40 hover:text-accent disabled:cursor-not-allowed disabled:opacity-60"
        >
          {state.saving ? "Speichert" : state.saved ? "Gespeichert" : "Label speichern"}
        </button>
        {state.error && <span className="text-[11px] text-danger">{state.error}</span>}
      </div>
    </article>
  );
}

function ToggleButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "border px-2 py-1 text-[10px] uppercase tracking-widest transition-colors",
        active ? "border-accent/40 bg-accent-dim text-accent" : "border-border text-muted hover:border-border-bright hover:text-tx",
      ].join(" ")}
    >
      {label}
    </button>
  );
}
