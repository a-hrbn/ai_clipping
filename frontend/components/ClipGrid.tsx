import { ClipResult } from "@/lib/types";
import ClipCard from "./ClipCard";

interface Props {
  jobId: string;
  clips: ClipResult[];
}

export default function ClipGrid({ jobId, clips }: Props) {
  const avgScore = clips.length
    ? (clips.reduce((s, c) => s + c.viral_score, 0) / clips.length).toFixed(1)
    : "—";
  const usesMlRanking = clips.some((clip) => clip.ranking_mode === "ml_blended");
  const rankingLabel = usesMlRanking ? "Trainierte ML" : "Heuristiken";
  const rankingDetail = usesMlRanking
    ? "Ranking mit trainiertem Modell und heuristischem Blend"
    : "Ranking nur mit heuristischen Signalen";

  return (
    <div className="space-y-6">
      {/* Section header */}
      <div className="flex items-end justify-between border-b border-border pb-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-muted mb-1">
            Ergebnisse
          </p>
          <h2 className="text-2xl font-extrabold text-tx" style={{ letterSpacing: "-0.02em" }}>
            {clips.length} virale{clips.length === 1 ? "r Moment" : " Momente"}
          </h2>
        </div>
        <div className="text-right">
          <p className="text-xs font-medium uppercase tracking-widest text-muted mb-1">Ø Score</p>
          <p className="font-mono text-lg font-bold text-accent">{avgScore}</p>
        </div>
      </div>

      <div className="flex items-center justify-between border border-border bg-surface px-4 py-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-muted">Ranking</p>
          <p className="mt-1 text-sm text-dim">{rankingDetail}</p>
        </div>
        <span
          className={[
            "px-2.5 py-1 text-xs font-semibold uppercase tracking-widest border",
            usesMlRanking
              ? "border-accent/40 bg-accent-dim text-accent"
              : "border-border-bright text-muted",
          ].join(" ")}
        >
          {rankingLabel}
        </span>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-px bg-border">
        {clips.map((clip, i) => (
          <div key={clip.filename} className="bg-bg animate-fade-up" style={{ animationDelay: `${i * 0.06}s` }}>
            <ClipCard clip={clip} rank={i + 1} jobId={jobId} />
          </div>
        ))}
      </div>
    </div>
  );
}
