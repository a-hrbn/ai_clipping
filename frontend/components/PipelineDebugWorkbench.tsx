"use client";

import { useEffect, useMemo, useState } from "react";

import {
  deletePipelineDebugWorkflow,
  getDownloadUrl,
  getPipelineDebugAnchors,
  getPipelineDebugWorkflow,
  listPipelineDebugWorkflows,
  savePipelineDebugLabel,
  startPipelineDebugWorkflow,
} from "@/lib/api";
import {
  PipelineDebugBoundaryLabel,
  PipelineDebugEval,
  PipelineDebugGtHit,
  PipelineDebugRow,
  PipelineDebugViralLabel,
  PipelineDebugVodEval,
  PipelineDebugWorkflow,
} from "@/lib/types";

type LabelState = {
  viral: PipelineDebugViralLabel;
  boundary: PipelineDebugBoundaryLabel;
  notes: string;
  saving: boolean;
  saved: boolean;
  error: string | null;
};

const VIRAL_OPTIONS: Array<{ value: PipelineDebugViralLabel; label: string }> = [
  { value: "yes", label: "Viral" },
  { value: "unsure", label: "Unsicher" },
  { value: "no", label: "Nein" },
];

const BOUNDARY_OPTIONS: Array<{ value: PipelineDebugBoundaryLabel; label: string }> = [
  { value: "good", label: "Gut" },
  { value: "too_wide", label: "Zu breit" },
  { value: "too_short", label: "Zu kurz" },
  { value: "wrong", label: "Falsch" },
];

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

function fmtPct(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function fmtViews(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${Math.round(value / 1_000)}k`;
  return String(Math.round(value));
}

function defaultLabelState(row?: PipelineDebugRow): LabelState {
  return {
    viral: row?.human_viral_label ?? "",
    boundary: row?.human_boundary_label ?? "",
    notes: row?.human_notes ?? "",
    saving: false,
    saved: false,
    error: null,
  };
}

function completion(rows: PipelineDebugRow[]): string {
  if (!rows.length) return "0/0";
  const done = rows.filter((row) => row.human_viral_label || row.human_boundary_label).length;
  return `${done}/${rows.length}`;
}

function fmtWorkflowId(id: string): string {
  const m = id.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/);
  if (!m) return id;
  return `${m[3]}.${m[2]} · ${m[4]}:${m[5]}`;
}

export default function PipelineDebugWorkbench() {
  const [anchors, setAnchors] = useState<string[]>([]);
  const [selectedAnchors, setSelectedAnchors] = useState<Set<string>>(new Set());
  const [maxMoments, setMaxMoments] = useState(180);
  const [runName, setRunName] = useState("");

  // List of shallow workflow summaries (no rows) shown in the run list
  const [workflows, setWorkflows] = useState<PipelineDebugWorkflow[]>([]);
  // Currently expanded run (full, with rows)
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [expandedWorkflow, setExpandedWorkflow] = useState<PipelineDebugWorkflow | null>(null);
  const [loadingExpanded, setLoadingExpanded] = useState(false);

  const [labels, setLabels] = useState<Record<string, LabelState>>({});
  const [starting, setStarting] = useState(false);
  const [loadingList, setLoadingList] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isActive = expandedWorkflow?.status === "queued" || expandedWorkflow?.status === "running";
  const rows = useMemo(() => expandedWorkflow?.rows ?? [], [expandedWorkflow?.rows]);
  const groupedRows = useMemo(() => {
    const groups: Record<string, PipelineDebugRow[]> = {};
    for (const row of rows) {
      const key = row.vod_id || "unknown";
      groups[key] = [...(groups[key] ?? []), row];
    }
    return groups;
  }, [rows]);

  useEffect(() => {
    void refreshWorkflows();
    void getPipelineDebugAnchors()
      .then((r) => {
        setAnchors(r.vod_ids);
        setSelectedAnchors(new Set(r.vod_ids));
      })
      .catch(() => {});
  }, []);

  // Poll the expanded workflow while it's active
  useEffect(() => {
    if (!expandedId || !isActive) return;
    const timer = window.setInterval(() => {
      void fetchExpanded(expandedId);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [expandedId, isActive]);

  // Sync labels when rows change
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

  async function handleDelete(workflowId: string) {
    setDeletingId(workflowId);
    try {
      await deletePipelineDebugWorkflow(workflowId);
      if (expandedId === workflowId) {
        setExpandedId(null);
        setExpandedWorkflow(null);
      }
      setWorkflows((prev) => prev.filter((w) => w.workflow_id !== workflowId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Löschen fehlgeschlagen");
    } finally {
      setDeletingId(null);
    }
  }

  async function refreshWorkflows() {
    setLoadingList(true);
    try {
      const response = await listPipelineDebugWorkflows(20);
      setWorkflows(response.workflows);
    } catch {
      // workbench remains usable
    } finally {
      setLoadingList(false);
    }
  }

  async function fetchExpanded(workflowId: string) {
    try {
      const full = await getPipelineDebugWorkflow(workflowId);
      setExpandedWorkflow(full);
      // Keep shallow list entry in sync (status/progress, not rows)
      setWorkflows((current) =>
        current.map((w) =>
          w.workflow_id === workflowId
            ? { ...w, status: full.status, stage: full.stage, progress_pct: full.progress_pct, message: full.message, row_count: full.row_count, llm_moment_count: full.llm_moment_count }
            : w
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Workflow konnte nicht geladen werden");
    }
  }

  async function toggleExpand(workflowId: string) {
    if (expandedId === workflowId) {
      setExpandedId(null);
      setExpandedWorkflow(null);
      return;
    }
    setExpandedId(workflowId);
    setExpandedWorkflow(null);
    setLoadingExpanded(true);
    try {
      const full = await getPipelineDebugWorkflow(workflowId);
      setExpandedWorkflow(full);
      setWorkflows((current) =>
        current.map((w) =>
          w.workflow_id === workflowId
            ? { ...w, status: full.status, stage: full.stage, progress_pct: full.progress_pct, message: full.message, row_count: full.row_count, llm_moment_count: full.llm_moment_count }
            : w
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Workflow konnte nicht geladen werden");
      setExpandedId(null);
    } finally {
      setLoadingExpanded(false);
    }
  }

  async function startWorkflow() {
    setStarting(true);
    setError(null);
    try {
      const next = await startPipelineDebugWorkflow({
        profile: "jynxzi",
        vod_ids: Array.from(selectedAnchors),
        max_moments: maxMoments,
        provider: "gemini",
        name: runName.trim(),
      });
      setWorkflows((current) => [next, ...current.filter((w) => w.workflow_id !== next.workflow_id)].slice(0, 20));
      setExpandedId(next.workflow_id);
      setExpandedWorkflow(next);
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

  async function saveLabel(row: PipelineDebugRow) {
    if (!expandedWorkflow?.workflow_id) return;
    const state = labels[row.review_id] ?? defaultLabelState(row);
    patchLabel(row.review_id, { saving: true, saved: false, error: null });
    try {
      await savePipelineDebugLabel(expandedWorkflow.workflow_id, {
        review_id: row.review_id,
        human_viral_label: state.viral,
        human_boundary_label: state.boundary,
        human_notes: state.notes,
      });
      patchLabel(row.review_id, { saving: false, saved: true });
      void fetchExpanded(expandedWorkflow.workflow_id);
    } catch (err) {
      patchLabel(row.review_id, {
        saving: false,
        error: err instanceof Error ? err.message : "Label konnte nicht gespeichert werden",
      });
    }
  }

  return (
    <section className="w-full max-w-5xl border border-border bg-surface p-4 md:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-muted">Pipeline Debugging</p>
          <h2 className="mt-1 text-xl font-bold text-tx">Pipeline Debug Workbench</h2>
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
        </div>
      </div>

      {/* Anchor VODs — selectable subset */}
      <div className="mt-4">
        <div className="flex items-center gap-3">
          <p className="text-[10px] uppercase tracking-widest text-muted">Anchor VODs</p>
          {anchors.length > 0 && selectedAnchors.size < anchors.length && (
            <button
              type="button"
              onClick={() => setSelectedAnchors(new Set(anchors))}
              className="text-[10px] text-accent hover:underline"
            >
              alle auswählen
            </button>
          )}
        </div>
        <div className="mt-1 flex flex-wrap gap-1">
          {anchors.length > 0 ? anchors.map((id) => {
            const checked = selectedAnchors.has(id);
            return (
              <button
                key={id}
                type="button"
                onClick={() => {
                  setSelectedAnchors((prev) => {
                    const next = new Set(prev);
                    if (next.has(id)) { if (next.size > 1) next.delete(id); }
                    else next.add(id);
                    return next;
                  });
                }}
                className={[
                  "border px-2 py-1 font-mono text-[11px] transition-colors",
                  checked
                    ? "border-accent/40 bg-accent-dim text-accent"
                    : "border-border bg-bg text-muted opacity-50",
                ].join(" ")}
              >
                {id}
              </button>
            );
          }) : (
            <span className="font-mono text-[11px] text-muted">wird geladen…</span>
          )}
        </div>
        {anchors.length > 0 && (
          <p className="mt-1 text-[10px] text-muted">
            {selectedAnchors.size} von {anchors.length} ausgewählt
          </p>
        )}
      </div>

      {/* Start config */}
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <NumberField label="Max Momente" value={maxMoments} min={1} max={500} onChange={setMaxMoments} />
        <label className="block flex-1 min-w-[160px]">
          <span className="text-[10px] uppercase tracking-widest text-muted">Run Name (optional)</span>
          <input
            type="text"
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            placeholder="z.B. baseline-v1"
            className="mt-1 w-full border border-border bg-bg px-3 py-2 text-xs text-tx placeholder:text-muted/60 focus:border-border-bright focus:outline-none"
          />
        </label>
      </div>

      {error && <p className="mt-4 border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger">{error}</p>}

      {/* Run list */}
      {workflows.length > 0 && (
        <div className="mt-5 flex flex-col divide-y divide-border border border-border">
          {workflows.map((item) => {
            const isExpanded = expandedId === item.workflow_id;
            const displayName = item.name || fmtWorkflowId(item.workflow_id);
            const statusColor =
              item.status === "done" ? "text-accent" :
              item.status === "failed" ? "text-danger" :
              item.status === "running" ? "text-yellow-400" : "text-muted";
            return (
              <div key={item.workflow_id}>
                {/* Row header */}
                <div className="flex w-full items-center">
                  <button
                    type="button"
                    onClick={() => toggleExpand(item.workflow_id)}
                    className="flex min-w-0 flex-1 items-center justify-between gap-3 px-3 py-2.5 text-left hover:bg-border/10"
                  >
                    <div className="flex min-w-0 flex-1 items-center gap-3">
                      <span className="truncate text-sm font-medium text-tx">{displayName}</span>
                      {item.name && (
                        <span className="shrink-0 font-mono text-[10px] text-muted/60">{fmtWorkflowId(item.workflow_id)}</span>
                      )}
                    </div>
                    <div className="flex shrink-0 items-center gap-2 text-[10px] uppercase tracking-widest">
                      <span className={statusColor}>{item.status}</span>
                      {item.llm_moment_count != null && (
                        <span className="text-muted">{item.llm_moment_count} LLM</span>
                      )}
                      <span className="text-muted">{item.row_count} Clips</span>
                      <span className="ml-1 text-muted">{isExpanded ? "▲" : "▼"}</span>
                    </div>
                  </button>
                  {["done", "failed", "cancelled"].includes(item.status) && (
                    <button
                      type="button"
                      title="Run löschen"
                      disabled={deletingId === item.workflow_id}
                      onClick={(e) => { e.stopPropagation(); void handleDelete(item.workflow_id); }}
                      className="shrink-0 px-3 py-2.5 text-muted hover:text-danger disabled:opacity-40"
                    >
                      {deletingId === item.workflow_id ? "…" : "✕"}
                    </button>
                  )}
                </div>

                {/* Expanded content */}
                {isExpanded && (
                  <div className="border-t border-border bg-bg/30 px-3 pb-4 pt-3">
                    {loadingExpanded && !expandedWorkflow && (
                      <p className="py-8 text-center text-xs text-muted">Wird geladen…</p>
                    )}

                    {expandedWorkflow && (
                      <>
                        {/* Status bar */}
                        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                          <div>
                            <p className="text-sm font-medium text-tx">{expandedWorkflow.message}</p>
                            {expandedWorkflow.error && <p className="mt-1 text-xs text-danger">{expandedWorkflow.error}</p>}
                          </div>
                          <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-widest text-muted">
                            <span className="border border-border px-2 py-1">{expandedWorkflow.stage}</span>
                            <span className="border border-border px-2 py-1">Labels {completion(rows)}</span>
                          </div>
                        </div>
                        <div className="mt-2 h-px w-full bg-border">
                          <div className="h-px bg-accent transition-[width]" style={{ width: `${expandedWorkflow.progress_pct}%` }} />
                        </div>

                        {/* Eval panel */}
                        {expandedWorkflow.eval && Object.keys(expandedWorkflow.eval).length > 0 && (
                          <EvalPanel eval={expandedWorkflow.eval} />
                        )}

                        {/* Clips */}
                        {expandedWorkflow.status === "done" && rows.length > 0 && (
                          <div className="mt-6 flex flex-col gap-6">
                            {Object.entries(groupedRows).map(([vodId, vodRows]) => (
                              <section key={vodId}>
                                <div className="mb-2 flex items-center justify-between border-b border-border pb-2">
                                  <h3 className="font-mono text-sm font-semibold text-tx">VOD {vodId}</h3>
                                  <span className="text-[10px] uppercase tracking-widest text-muted">{vodRows.length} Clips</span>
                                </div>
                                <div className="mt-3 grid gap-4 lg:grid-cols-2">
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
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-VOD GT match overview (shown above clip grid)
// ─────────────────────────────────────────────────────────────────────────────

function VodGtOverview({ vod }: { vod: PipelineDebugVodEval }) {
  if (!vod.has_gt || !vod.gt_hits.length) return null;
  return (
    <div className="mb-3 border border-border/60 bg-surface/40 px-3 py-2">
      <p className="mb-1.5 text-[9px] uppercase tracking-widest text-muted">
        GT Matches · {vod.hit_count}/{vod.gt_count} getroffen
      </p>
      <div className="flex flex-col gap-1">
        {vod.gt_hits.map((h) => (
          <div key={h.gt_index} className="flex items-baseline gap-2 text-[11px]">
            <span className={h.hit ? "shrink-0 text-accent" : "shrink-0 text-danger/70"}>
              {h.hit ? "✓" : "✗"}
            </span>
            <span className="shrink-0 font-mono text-dim">
              {fmtTime(h.gt_start)}–{fmtTime(h.gt_end)}
            </span>
            {h.hit && h.hit_rank != null && (
              <span className="shrink-0 text-muted">→ Clip #{h.hit_rank}</span>
            )}
            <span className="shrink-0 text-muted">{fmtViews(h.views)}v</span>
            <span className="min-w-0 truncate text-muted/80" title={h.desc}>{h.desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Eval metrics panel
// ─────────────────────────────────────────────────────────────────────────────

function EvalPanel({ eval: ev }: { eval: PipelineDebugEval }) {
  const o = ev.overall;
  return (
    <div className="mt-5 border border-border bg-bg/40 p-3">
      <p className="text-[10px] uppercase tracking-widest text-muted">
        Baseline — LLM vs Ground Truth · coverage ≥ {Math.round(ev.coverage_threshold * 100)}%
      </p>

      {/* Overall */}
      <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-6">
        <MetricCell label="recall@5" value={fmtPct(o.recall?.["5"])} />
        <MetricCell label="recall@10" value={fmtPct(o.recall?.["10"])} />
        <MetricCell label="recall@20" value={fmtPct(o.recall?.["20"])} />
        <MetricCell label="avg rank" value={o.avg_rank != null ? o.avg_rank.toFixed(1) : "-"} />
        <MetricCell label="miss rate" value={fmtPct(o.miss_rate)} />
        <MetricCell label="hits" value={o.gt_count > 0 ? `${o.hit_count}/${o.gt_count}` : "-"} />
      </div>

      {/* Per-VOD breakdown */}
      {ev.per_vod.length > 0 && (
        <div className="mt-4 flex flex-col gap-3">
          {ev.per_vod.map((v) => (
            <VodEvalSection key={v.vod_id} vod={v} />
          ))}
        </div>
      )}
    </div>
  );
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-border bg-surface p-2 text-center">
      <p className="text-[9px] uppercase tracking-widest text-muted">{label}</p>
      <p className="mt-1 font-mono text-sm font-semibold text-tx">{value}</p>
    </div>
  );
}

function VodEvalSection({ vod }: { vod: PipelineDebugVodEval }) {
  const [open, setOpen] = useState(false);

  if (!vod.has_gt) {
    return (
      <div className="border border-border/50 bg-surface/40 px-3 py-2 text-[11px] text-muted">
        <span className="font-mono">{vod.vod_id}</span>
        {" — "}
        <span className="italic">Keine GT (Matching läuft noch)</span>
        {" · "}
        {vod.pred_count} Preds
      </div>
    );
  }

  return (
    <div className="border border-border bg-surface/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start justify-between px-3 py-2 text-left hover:bg-border/20"
      >
        <div className="flex flex-wrap gap-3 text-[11px]">
          <span className="font-mono text-tx">{vod.vod_id}</span>
          <span className="text-muted">r@5 <b className="text-tx">{fmtPct(vod.recall?.["5"])}</b></span>
          <span className="text-muted">r@10 <b className="text-tx">{fmtPct(vod.recall?.["10"])}</b></span>
          <span className="text-muted">r@20 <b className="text-tx">{fmtPct(vod.recall?.["20"])}</b></span>
          <span className="text-muted">avg_rank <b className="text-tx">{vod.avg_rank != null ? vod.avg_rank.toFixed(1) : "-"}</b></span>
          <span className="text-muted">hits <b className="text-tx">{vod.hit_count}/{vod.gt_count}</b></span>
        </div>
        <span className="ml-2 shrink-0 text-[10px] text-muted">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="border-t border-border px-3 py-2">
          <GtHitsTable hits={vod.gt_hits} />
        </div>
      )}
    </div>
  );
}

function GtHitsTable({ hits }: { hits: PipelineDebugGtHit[] }) {
  if (!hits.length) return <p className="text-[11px] text-muted">Keine GT-Momente.</p>;
  return (
    <table className="w-full text-[11px]">
      <thead>
        <tr className="text-left text-[9px] uppercase tracking-widest text-muted">
          <th className="pb-1 pr-2">Hit</th>
          <th className="pb-1 pr-2">Rang</th>
          <th className="pb-1 pr-2">Cov</th>
          <th className="pb-1 pr-2">Views</th>
          <th className="pb-1 pr-2">Fenster</th>
          <th className="pb-1">Beschreibung</th>
        </tr>
      </thead>
      <tbody>
        {hits.map((h) => (
          <tr
            key={h.gt_index}
            className={h.hit ? "text-tx" : "text-muted opacity-70"}
          >
            <td className="py-0.5 pr-2">
              <span className={h.hit ? "text-accent" : "text-danger"}>
                {h.hit ? "✓" : "✗"}
              </span>
            </td>
            <td className="py-0.5 pr-2 font-mono">{h.hit_rank ?? "-"}</td>
            <td className="py-0.5 pr-2">{fmtPct(h.coverage)}</td>
            <td className="py-0.5 pr-2">{fmtViews(h.views)}</td>
            <td className="py-0.5 pr-2 font-mono whitespace-nowrap">
              {fmtTime(h.gt_start)}–{fmtTime(h.gt_end)}
            </td>
            <td className="py-0.5 max-w-xs truncate" title={h.desc}>{h.desc}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

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
    <label className="block w-32">
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

function ReviewCard({
  row,
  state,
  onPatch,
  onSave,
}: {
  row: PipelineDebugRow;
  state: LabelState;
  onPatch: (patch: Partial<LabelState>) => void;
  onSave: () => void;
}) {
  const clipUrl = row.clip_url ? getDownloadUrl(row.clip_url) : null;
  return (
    <article className={[
      "border bg-bg/50 p-3",
      row.gt_hit ? "border-accent/50" : "border-border",
    ].join(" ")}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] uppercase tracking-widest text-muted">#{row.review_rank} · Rank {row.source_rank ?? "-"}</p>
          <p className="mt-1 font-mono text-xs text-dim">
            {fmtTime(row.start_time)} → {fmtTime(row.end_time)} · {Math.round(row.duration)}s
          </p>
        </div>
        {row.gt_hit && (
          <div className="shrink-0 border border-accent/40 bg-accent-dim px-2 py-0.5 text-[9px] uppercase tracking-widest text-accent">
            GT ✓ &nbsp;rank {row.gt_rank} · {fmtPct(row.gt_coverage)} · {fmtViews(row.gt_views)}v
          </div>
        )}
      </div>

      <div className="mt-3">
        {clipUrl && row.clip_status === "ready" ? (
          <video src={clipUrl} controls preload="metadata" playsInline className="block w-full bg-black border border-border" />
        ) : (
          <div className="flex aspect-video items-center justify-center border border-border bg-surface text-xs text-muted">
            {row.clip_status === "error" ? "Clip-Fehler" : "Clip nicht verfügbar"}
          </div>
        )}
      </div>

      <div className="mt-3 grid grid-cols-5 gap-2 text-[11px] text-dim">
        <span>V {fmtScore(row.virality ?? row.llm_virality)}</span>
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
        </div>
      )}

      {row.gt_hit && row.gt_desc && (
        <div className="mt-2 border border-accent/20 bg-accent-dim/30 px-2 py-1">
          <p className="text-[9px] uppercase tracking-widest text-accent/70">GT Viral Moment</p>
          <p className="mt-0.5 text-[11px] leading-relaxed text-dim">{row.gt_desc}</p>
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
