"use client";

import { useEffect, useState } from "react";

import { getDownloadUrl, getReviewCandidates, submitClipFeedback } from "@/lib/api";
import { ClipFeedbackRating, ReviewCandidate } from "@/lib/types";

const FEEDBACK_OPTIONS: Array<{ rating: ClipFeedbackRating; label: string }> = [
  { rating: "excellent", label: "Exzellent" },
  { rating: "good", label: "Gut" },
  { rating: "maybe", label: "Vielleicht" },
  { rating: "bad", label: "Schlecht" },
  { rating: "boundary_bad", label: "Boundary falsch" },
];

function fmtTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

function fmtMetric(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function defaultCandidateState(): CandidateState {
  return {
    rating: null,
    notes: "",
    saving: false,
    saved: false,
    error: null,
  };
}

interface CandidateState {
  rating: ClipFeedbackRating | null;
  notes: string;
  saving: boolean;
  saved: boolean;
  error: string | null;
}

export default function CandidateReviewPanel({ jobId }: { jobId: string }) {
  const [candidates, setCandidates] = useState<ReviewCandidate[]>([]);
  const [states, setStates] = useState<Record<string, CandidateState>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getReviewCandidates(jobId, 80)
      .then((response) => {
        if (cancelled) return;
        setCandidates(response.candidates);
        setStates((current) => {
          const next = { ...current };
          for (const candidate of response.candidates) {
            if (next[candidate.candidate_id]) continue;
            next[candidate.candidate_id] = {
              rating: candidate.feedback_rating ?? null,
              notes: candidate.feedback_notes ?? "",
              saving: false,
              saved: false,
              error: null,
            };
          }
          return next;
        });
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Kandidaten konnten nicht geladen werden");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  function updateCandidate(candidateId: string, patch: Partial<CandidateState>) {
    setStates((current) => ({
      ...current,
      [candidateId]: {
        ...defaultCandidateState(),
        ...(current[candidateId] ?? {}),
        ...patch,
      },
    }));
  }

  async function saveCandidate(candidate: ReviewCandidate) {
    const state = states[candidate.candidate_id];
    if (!state?.rating) return;
    updateCandidate(candidate.candidate_id, { saving: true, saved: false, error: null });
    try {
      await submitClipFeedback(jobId, {
        candidate_id: candidate.candidate_id,
        mode: "training_feedback",
        rating: state.rating,
        notes: state.notes.trim() || undefined,
      });
      updateCandidate(candidate.candidate_id, { saving: false, saved: true });
    } catch (err) {
      updateCandidate(candidate.candidate_id, {
        saving: false,
        error: err instanceof Error ? err.message : "Feedback konnte nicht gespeichert werden",
      });
    }
  }

  if (!jobId) return null;

  return (
    <section className="border border-border bg-surface p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-muted">Candidate Labeling</p>
          <h2 className="mt-1 text-lg font-bold text-tx">Review-Kandidaten bewerten</h2>
          <p className="mt-1 text-xs text-dim">
            Bewerte nicht nur finale Clips, sondern auch starke Near-Misses aus Proposal, Ranker und Judge.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="shrink-0 px-3 py-2 text-[10px] uppercase tracking-widest border border-border text-muted hover:text-tx hover:border-border-bright"
        >
          {expanded ? "Einklappen" : "Öffnen"}
        </button>
      </div>

      {expanded && (
        <div className="mt-4 flex flex-col gap-4">
          {loading && <p className="text-sm text-muted">Lade Review-Kandidaten…</p>}
          {error && <p className="text-sm text-danger">{error}</p>}
          {!loading && !error && candidates.length === 0 && (
            <p className="text-sm text-muted">Keine Review-Kandidaten gefunden.</p>
          )}

          {candidates.map((candidate, index) => {
            const state = states[candidate.candidate_id] ?? defaultCandidateState();
            const previewUrl = candidate.preview_url ? getDownloadUrl(candidate.preview_url) : null;
            return (
              <article key={candidate.candidate_id} className="border border-border bg-bg/40 p-3">
                <div className="flex flex-col gap-3 md:flex-row">
                  <div className="md:w-[260px] shrink-0">
                    {previewUrl ? (
                      <video
                        src={previewUrl}
                        controls
                        preload="metadata"
                        playsInline
                        className="block w-full bg-black border border-border"
                      />
                    ) : (
                      <div className="flex aspect-video items-center justify-center border border-border bg-surface text-xs text-muted">
                        Keine Preview verfügbar
                      </div>
                    )}
                    <div className="mt-2 font-mono text-[11px] text-muted">
                      #{String(index + 1).padStart(2, "0")} · {fmtTime(candidate.start_time)} → {fmtTime(candidate.end_time)} · {Math.round(candidate.duration)}s
                    </div>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-widest text-muted">
                      {candidate.selected && <span className="border border-accent/40 bg-accent-dim px-2 py-1 text-accent">Final</span>}
                      {candidate.source && <span className="border border-border px-2 py-1">{candidate.source}</span>}
                      {candidate.clip_type && <span className="border border-border px-2 py-1">{candidate.clip_type}</span>}
                      {candidate.rejection_source && <span className="border border-border px-2 py-1">{candidate.rejection_source}</span>}
                    </div>

                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-dim sm:grid-cols-4">
                      <span>Rank: {fmtMetric(candidate.rank_score)}</span>
                      <span>Proposal: {fmtMetric(candidate.proposal_score, 3)}</span>
                      <span>Pre-Judge: {candidate.pre_judge_rank ?? "-"}</span>
                      <span>Judge: {fmtMetric(candidate.judge_score)}</span>
                    </div>

                    {candidate.judge_reason && (
                      <p className="mt-3 text-xs leading-relaxed text-dim">{candidate.judge_reason}</p>
                    )}

                    <div className="mt-4 flex flex-wrap gap-2">
                      {FEEDBACK_OPTIONS.map((option) => {
                        const active = state.rating === option.rating;
                        return (
                          <button
                            key={option.rating}
                            type="button"
                            onClick={() => updateCandidate(candidate.candidate_id, { rating: option.rating, saved: false })}
                            className={[
                              "px-2 py-1 text-[10px] uppercase tracking-widest border transition-colors",
                              active ? "border-accent/40 bg-accent-dim text-accent" : "border-border text-muted hover:text-tx",
                            ].join(" ")}
                          >
                            {option.label}
                          </button>
                        );
                      })}
                    </div>

                    <textarea
                      value={state.notes}
                      onChange={(event) => updateCandidate(candidate.candidate_id, { notes: event.target.value, saved: false })}
                      placeholder="Optional: warum gut/schlecht, was fehlt, Boundary-Hinweis"
                      className="mt-3 min-h-[58px] w-full bg-surface border border-border px-2 py-2 text-xs text-tx placeholder:text-muted focus:outline-none focus:border-border-bright"
                    />

                    <div className="mt-3 flex items-center gap-3">
                      <button
                        type="button"
                        onClick={() => saveCandidate(candidate)}
                        disabled={state.saving || !state.rating}
                        className="px-3 py-2 text-[10px] uppercase tracking-widest border border-border-bright text-tx hover:border-accent/40 hover:text-accent disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
                      >
                        {state.saving ? "Speichert…" : state.saved ? "Gespeichert" : "Label speichern"}
                      </button>
                      {state.error && <span className="text-[11px] text-danger">{state.error}</span>}
                    </div>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
