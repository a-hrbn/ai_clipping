"use client";

import {
  CandidateFunnelDiagnostics,
  CandidateFunnelSample,
  PipelineDiagnostics,
  PipelineDiagnosticStage,
  PipelineStageStatus,
} from "@/lib/types";

const STATUS_LABEL: Record<PipelineStageStatus, string> = {
  ok: "OK",
  warning: "Warnung",
  pending: "Wartet",
  disabled: "Aus",
  error: "Fehler",
};

const STATUS_CLASS: Record<PipelineStageStatus, string> = {
  ok: "border-ok/40 bg-ok/10 text-ok",
  warning: "border-warn/40 bg-warn/10 text-warn",
  pending: "border-border-bright bg-surface-2 text-muted",
  disabled: "border-border bg-surface text-muted",
  error: "border-danger/40 bg-danger/10 text-danger",
};

function formatMetricValue(value: string | number | boolean | null): string {
  if (value === null) return "-";
  if (typeof value === "boolean") return value ? "ja" : "nein";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return value;
}

function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(3);
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function formatMean(value: { mean?: number | null } | null | undefined): string {
  return formatScore(value?.mean);
}

function topEntries(values: Record<string, number> | undefined, limit = 5): [string, number][] {
  return Object.entries(values ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);
}

function sampleLabel(sample: CandidateFunnelSample): string {
  if (sample.sample_group === "final") return "Final";
  if (sample.sample_group === "quality_rejected") return "Quality Reject";
  if (sample.sample_group === "highest_ranked_rejected") return "Ranked Reject";
  return sample.status ?? "Trace";
}

function StageRow({ stage }: { stage: PipelineDiagnosticStage }) {
  const status = stage.status in STATUS_CLASS ? stage.status : "pending";

  return (
    <div className="grid gap-3 px-4 py-4 md:grid-cols-[minmax(140px,0.9fr)_minmax(0,1.8fr)]">
      <div className="flex min-w-0 items-start justify-between gap-3 md:block">
        <div className="min-w-0">
          <div className="text-xs font-semibold uppercase tracking-widest text-tx">
            {stage.label}
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span className={["inline-flex rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider", STATUS_CLASS[status]].join(" ")}>
              {STATUS_LABEL[status]}
            </span>
          </div>
        </div>
      </div>

      <div className="min-w-0 space-y-3">
        <p className="text-sm text-dim">{stage.summary}</p>
        {stage.metrics.length > 0 && (
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
            {stage.metrics.map((metric) => (
              <div key={`${stage.id}-${metric.label}`} className="min-w-0">
                <div className="truncate text-[10px] font-medium uppercase tracking-widest text-muted">
                  {metric.label}
                </div>
                <div className="mt-0.5 truncate font-mono text-xs text-tx">
                  {formatMetricValue(metric.value)}
                </div>
              </div>
            ))}
          </div>
        )}
        {stage.details.length > 0 && (
          <div className="space-y-1">
            {stage.details.map((detail, index) => (
              <p key={`${stage.id}-detail-${index}`} className="text-xs text-warn">
                {detail}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CandidateFunnelPanel({ funnel }: { funnel: CandidateFunnelDiagnostics | null | undefined }) {
  if (!funnel?.available) {
    return null;
  }

  const largestLosses = funnel.largest_losses ?? [];
  const samples = funnel.samples ?? [];
  const quality = funnel.quality_breakdown ?? {};
  const proposal = funnel.proposal_lift ?? {};
  const ranker = funnel.ranker_lift ?? {};

  return (
    <div className="border-t border-border bg-surface/40 px-4 py-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(320px,1.3fr)]">
        <div className="space-y-4">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-widest text-tx">
              Candidate Funnel
            </h3>
            <p className="mt-1 text-xs text-muted">
              {funnel.rows} Kandidaten getrackt, {funnel.final_selected_rows ?? 0} final ausgewählt.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {Object.entries(funnel.stage_counts ?? {}).map(([stage, count]) => (
              <div key={stage} className="border border-border px-3 py-2">
                <div className="truncate text-[10px] uppercase tracking-widest text-muted">{stage}</div>
                <div className="mt-1 font-mono text-sm text-tx">{count}</div>
              </div>
            ))}
          </div>

          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
            <FunnelMetricCard label="Judge behalten" value={quality.judged_kept ?? 0} detail={`${quality.judged_rejected ?? 0} judged rejected`} />
            <FunnelMetricCard label="Ungejudged behalten" value={quality.unjudged_kept ?? 0} detail={`${quality.unjudged_rejected ?? 0} ungejudged rejected`} />
            <FunnelMetricCard label="Proposal P@K" value={formatPercent(proposal.precision_at_positive_k)} detail={`Final mean ${formatMean(proposal.final_selected_proposal_score)}`} />
            <FunnelMetricCard label="Ranker Top20" value={formatPercent(ranker.high_judge_pre_judge_top20_ratio)} detail={`${ranker.high_judge_rows ?? 0} High-Judge Kandidaten`} />
          </div>

          <div className="grid gap-2 sm:grid-cols-3">
            <FunnelMetricCard label="Proposal Recall@K" value={formatPercent(proposal.recall_at_positive_k)} detail={`${proposal.positive_proxy_rows ?? 0} positive Proxy Rows`} />
            <FunnelMetricCard label="High-Judge Proposal" value={formatPercent(proposal.high_judge_proposal_kept_ratio)} detail={`Mean ${formatMean(proposal.high_judge_proposal_score)}`} />
            <FunnelMetricCard label="Final Rank Mean" value={formatMean(ranker.final_selected_pre_judge_rank)} detail={`Judge mean ${formatMean(ranker.final_selected_judge_score)}`} />
          </div>

          {largestLosses.length > 0 && (
            <div className="space-y-2">
              <div className="text-[10px] font-semibold uppercase tracking-widest text-muted">
                Größte Verluste
              </div>
              {largestLosses.slice(0, 5).map((loss) => (
                <div key={loss.transition} className="flex items-center justify-between gap-3 border border-border px-3 py-2 text-xs">
                  <span className="min-w-0 truncate text-dim">{loss.transition}</span>
                  <span className="shrink-0 font-mono text-tx">
                    {loss.lost_count} verloren ({loss.input_count} → {loss.output_count})
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="grid gap-3 sm:grid-cols-2">
            <ReasonList title="Drop-Stufen" values={funnel.drop_stage_counts} />
            <ReasonList title="Reject-Gründe" values={funnel.rejection_reason_counts} />
          </div>
        </div>

        <div className="min-w-0">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted">
            Kandidaten-Stichprobe
          </div>
          <div className="overflow-x-auto border border-border">
            <table className="min-w-full text-left text-xs">
              <thead className="border-b border-border text-[10px] uppercase tracking-widest text-muted">
                <tr>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Zeit</th>
                  <th className="px-3 py-2 font-medium">Quelle</th>
                  <th className="px-3 py-2 font-medium">Rank</th>
                  <th className="px-3 py-2 font-medium">Scores</th>
                  <th className="px-3 py-2 font-medium">Grund</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {samples.slice(0, 24).map((sample) => (
                  <tr key={`${sample.sample_group ?? "sample"}-${sample.candidate_id}`} className="align-top">
                    <td className="px-3 py-2 text-tx">{sampleLabel(sample)}</td>
                    <td className="px-3 py-2 font-mono text-muted">
                      {formatSeconds(sample.start)}-{formatSeconds(sample.end)}
                    </td>
                    <td className="max-w-[140px] truncate px-3 py-2 text-dim">{sample.source ?? "-"}</td>
                    <td className="px-3 py-2 font-mono text-muted">
                      {sample.final_rank ? `F${sample.final_rank}` : sample.pre_judge_rank ? `R${sample.pre_judge_rank}` : "-"}
                    </td>
                    <td className="px-3 py-2 font-mono text-muted">
                      P {formatScore(sample.proposal_score)} / R {formatScore(sample.pre_judge_rank_score)} / J {formatScore(sample.judge_score)}
                    </td>
                    <td className="max-w-[220px] px-3 py-2 text-warn">
                      {sample.drop_reason ?? sample.dropped_stage ?? "-"}
                    </td>
                  </tr>
                ))}
                {samples.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-3 py-4 text-muted">
                      Noch keine Kandidaten-Trace-Stichprobe vorhanden.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function FunnelMetricCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string | number;
  detail?: string;
}) {
  return (
    <div className="border border-border px-3 py-2">
      <div className="truncate text-[10px] uppercase tracking-widest text-muted">{label}</div>
      <div className="mt-1 font-mono text-sm text-tx">{value}</div>
      {detail && <div className="mt-1 truncate text-[10px] text-muted">{detail}</div>}
    </div>
  );
}

function ReasonList({ title, values }: { title: string; values: Record<string, number> | undefined }) {
  const entries = topEntries(values);
  return (
    <div className="border border-border px-3 py-3">
      <div className="text-[10px] font-semibold uppercase tracking-widest text-muted">{title}</div>
      <div className="mt-2 space-y-1">
        {entries.length > 0 ? (
          entries.map(([label, count]) => (
            <div key={label} className="flex items-center justify-between gap-3 text-xs">
              <span className="min-w-0 truncate text-dim">{label}</span>
              <span className="shrink-0 font-mono text-tx">{count}</span>
            </div>
          ))
        ) : (
          <div className="text-xs text-muted">Keine Daten</div>
        )}
      </div>
    </div>
  );
}

export default function PipelineDiagnosticsPanel({
  diagnostics,
}: {
  diagnostics: PipelineDiagnostics | null | undefined;
}) {
  if (!diagnostics) {
    return null;
  }

  return (
    <section className="w-full border border-border">
      <div className="flex items-start justify-between gap-4 border-b border-border px-4 py-3">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-widest text-tx">
            Pipeline Überblick
          </h2>
          <p className="mt-1 text-xs text-muted">{diagnostics.summary}</p>
        </div>
        {diagnostics.last_modified && (
          <span className="shrink-0 font-mono text-[10px] text-muted">
            {new Date(diagnostics.last_modified).toLocaleTimeString("de-DE", {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })}
          </span>
        )}
      </div>
      {diagnostics.available && diagnostics.stages.length > 0 ? (
        <>
          <div className="divide-y divide-border">
            {diagnostics.stages.map((stage) => (
              <StageRow key={stage.id} stage={stage} />
            ))}
          </div>
          <CandidateFunnelPanel funnel={diagnostics.candidate_funnel} />
        </>
      ) : (
        <div className="px-4 py-4 text-sm text-muted">
          Detaildaten erscheinen, sobald Kandidaten, Ranker oder Judge-Artefakte geschrieben wurden.
        </div>
      )}
    </section>
  );
}
