"use client";

import { useState } from "react";

import { ClipFeedbackRating, ClipResult } from "@/lib/types";
import { getDownloadUrl, submitClipFeedback } from "@/lib/api";

function fmtTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

function scoreColor(score: number): string {
  if (score >= 8) return "var(--accent)";
  if (score >= 6) return "var(--warn)";
  return "var(--danger)";
}

function scoreLabel(score: number): string {
  if (score >= 8) return "Viral";
  if (score >= 6) return "Stark";
  return "Mittel";
}

interface Props {
  clip: ClipResult;
  rank: number;
  jobId: string;
}

const FEEDBACK_OPTIONS: Array<{ rating: ClipFeedbackRating; label: string }> = [
  { rating: "good", label: "Gut" },
  { rating: "medium", label: "Mittel" },
  { rating: "bad", label: "Schlecht" },
];

export default function ClipCard({ clip, rank, jobId }: Props) {
  const duration = Math.round(clip.end_time - clip.start_time);
  const color = scoreColor(clip.viral_score);
  const [rating, setRating] = useState<ClipFeedbackRating | null>(null);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSaveFeedback() {
    if (!jobId || !rating) return;
    setSaving(true);
    setSaved(false);
    setError(null);
    try {
      await submitClipFeedback(jobId, {
        candidate_id: clip.candidate_id,
        rating,
        notes: notes.trim() || undefined,
      });
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Feedback konnte nicht gespeichert werden");
    } finally {
      setSaving(false);
    }
  }

  return (
    <article className="group flex flex-col border border-border hover:border-border-bright bg-surface transition-all duration-200">
      {/* Score bar — colored top edge */}
      <div className="h-px w-full" style={{ background: color }} />

      <div className="p-5 flex flex-col gap-4 flex-1">
        {/* Header: rank + score */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1.5">
            <span className="font-mono text-xs text-muted leading-none">
              #{String(rank).padStart(2, "0")}
            </span>
            {clip.ranking_mode === "ml_blended" ? (
              <span className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider px-1.5 py-0.5 border border-[#4ade80]/40 text-[#4ade80] bg-[#4ade80]/5 leading-none">
                <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] animate-pulse inline-block" />
                KI-Modell
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider px-1.5 py-0.5 border border-border text-muted bg-transparent leading-none">
                <span className="w-1.5 h-1.5 rounded-full bg-muted inline-block" />
                Heuristik
              </span>
            )}
          </div>
          <div className="text-right">
            <div
              className="font-mono text-3xl font-extrabold leading-none"
              style={{ color }}
            >
              {clip.viral_score.toFixed(1)}
            </div>
            <div className="text-xs font-medium uppercase tracking-widest mt-1" style={{ color, opacity: 0.7 }}>
              {scoreLabel(clip.viral_score)}
            </div>
          </div>
        </div>

        {/* Reason */}
        <p className="text-sm text-dim leading-relaxed flex-1">{clip.reason}</p>

        {clip.selection_reasons.length > 0 && (
          <div className="border border-border bg-bg/50 p-3 flex flex-col gap-2">
            <p className="text-[10px] font-medium uppercase tracking-widest text-muted">
              Warum ausgewählt
            </p>
            <ul className="space-y-1.5">
              {clip.selection_reasons.map((reason, index) => (
                <li key={`${clip.candidate_id}-reason-${index}`} className="text-xs leading-relaxed text-tx flex gap-2">
                  <span className="mt-[6px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Timecode */}
        <div className="flex items-center gap-3 font-mono text-xs text-muted">
          <span>{fmtTime(clip.start_time)} → {fmtTime(clip.end_time)}</span>
          <span className="text-border-bright">·</span>
          <span>{duration}s</span>
        </div>

        {/* Download */}
        <a
          href={getDownloadUrl(clip.download_url)}
          download={clip.filename}
          className="flex items-center justify-center gap-2 w-full py-2.5 text-xs font-bold uppercase tracking-widest transition-all bg-accent text-bg hover:brightness-110 active:brightness-95"
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M6 1v7M3 5.5L6 8l3-2.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M1 10h10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
          </svg>
          Clip laden
        </a>

          <div className="border border-border bg-bg/50 p-3 flex flex-col gap-3">
            <div className="flex flex-wrap gap-2">
              {FEEDBACK_OPTIONS.map((option) => {
              const active = rating === option.rating;
              return (
                <button
                  key={option.rating}
                  type="button"
                  onClick={() => setRating(option.rating)}
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
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              placeholder="Optionaler Kommentar zur Bewertung"
              className="min-h-[64px] bg-surface border border-border px-2 py-2 text-xs text-tx placeholder:text-muted focus:outline-none focus:border-border-bright"
            />

          <button
            type="button"
            onClick={handleSaveFeedback}
            disabled={saving || !jobId || !rating}
            className="w-full py-2 text-[10px] uppercase tracking-widest border border-border-bright text-tx hover:border-accent/40 hover:text-accent disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
          >
            {saving ? "Speichert…" : saved ? "Feedback gespeichert" : "Feedback speichern"}
          </button>

          {error && <p className="text-[11px] text-danger">{error}</p>}
        </div>
      </div>
    </article>
  );
}
