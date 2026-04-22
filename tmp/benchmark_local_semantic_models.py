from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from services import semantic_service
from services.transcriber import RichSegment, format_transcript


@dataclass(frozen=True)
class CandidateSpec:
    debug_dir: Path
    candidate_id: str


SAMPLE_CANDIDATES = [
    CandidateSpec(
        BACKEND_ROOT / "tmp" / "storage" / "6261dc6c-51f3-4541-9174-9ac300a9f8fa" / "debug",
        "quote:0:92",
    ),
    CandidateSpec(
        BACKEND_ROOT / "tmp" / "storage" / "6261dc6c-51f3-4541-9174-9ac300a9f8fa" / "debug",
        "tension_build:4:2406",
    ),
    CandidateSpec(
        BACKEND_ROOT / "tmp" / "storage" / "6261dc6c-51f3-4541-9174-9ac300a9f8fa" / "debug",
        "narrative_reveal:1:10228",
    ),
    CandidateSpec(
        BACKEND_ROOT / "tmp" / "storage" / "6261dc6c-51f3-4541-9174-9ac300a9f8fa" / "debug",
        "quote:9:12394",
    ),
    CandidateSpec(
        BACKEND_ROOT / "tmp" / "storage" / "9e0ae6fc-b20a-40bc-83b4-7a453cb0e2cc" / "debug",
        "narrative_reveal:0:546",
    ),
    CandidateSpec(
        BACKEND_ROOT / "tmp" / "storage" / "9e0ae6fc-b20a-40bc-83b4-7a453cb0e2cc" / "debug",
        "quote:0:8648",
    ),
    CandidateSpec(
        BACKEND_ROOT / "tmp" / "storage" / "9e0ae6fc-b20a-40bc-83b4-7a453cb0e2cc" / "debug",
        "quote:3:13568",
    ),
    CandidateSpec(
        BACKEND_ROOT / "tmp" / "storage" / "9e0ae6fc-b20a-40bc-83b4-7a453cb0e2cc" / "debug",
        "tension_build:5:2436",
    ),
]


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_segments(debug_dir: Path) -> list[RichSegment]:
    rows = _load_jsonl(debug_dir / "segments.jsonl")
    return [
        RichSegment(
            start=float(row["start"]),
            end=float(row["end"]),
            text=str(row["text"]),
            speaker=str(row.get("speaker") or "UNKNOWN"),
            emotions=list(row.get("tags") or []),
            words_per_second=float(row.get("words_per_second") or 0.0),
            laughter_prob=float(row.get("laughter_prob") or 0.0),
            max_rms_db=float(row.get("max_rms_db") or 0.0),
            mean_rms_db=float(row.get("mean_rms_db") or 0.0),
            max_pitch_hz=float(row.get("max_pitch_hz") or 0.0),
            max_onset_rate=float(row.get("max_onset_rate") or 0.0),
            hype_score=float(row.get("hype_score") or 0.0),
            arc_score=float(row.get("arc_score") or 0.0),
            max_motion_score=float(row.get("max_motion_score") or 0.0),
            mean_face_size=float(row.get("mean_face_size") or 0.0),
            max_face_count=int(row.get("max_face_count") or 0),
            max_face_reaction=float(row.get("max_face_reaction") or 0.0),
            has_scene_change=bool(row.get("has_scene_change")),
            visual_activity=float(row.get("visual_activity") or 0.0),
            chat_spike_score=float(row.get("chat_spike_score") or 0.0),
        )
        for row in rows
    ]


def _load_candidates(debug_dir: Path) -> dict[str, dict[str, object]]:
    rows = _load_jsonl(debug_dir / "candidates_semantic.jsonl")
    return {str(row["id"]): row for row in rows}


def _window_for_candidate(segments: list[RichSegment], candidate: dict[str, object]) -> list[RichSegment]:
    start = float(candidate["start"])
    end = float(candidate["end"])
    return [
        segment
        for segment in segments
        if segment.end > start and segment.start < end
    ]


def _composite_score(scores: dict[str, float]) -> float:
    positive = [
        scores["quoteability"],
        scores["surprise"],
        scores["contradiction"],
        scores["standalone"],
        scores["hook_strength"],
        scores["payoff_clarity"],
    ]
    return round(sum(positive) / len(positive) - 0.5 * scores["context_dependency"], 3)


def _pairwise_order_accuracy(rows: list[dict[str, object]]) -> float:
    total = 0
    correct = 0
    for idx, left in enumerate(rows):
        for right in rows[idx + 1:]:
            left_hint = float(left["score_hint"])
            right_hint = float(right["score_hint"])
            if math.isclose(left_hint, right_hint, rel_tol=0.0, abs_tol=1e-9):
                continue
            total += 1
            left_score = float(left["composite_score"])
            right_score = float(right["composite_score"])
            if (left_hint > right_hint and left_score > right_score) or (
                left_hint < right_hint and left_score < right_score
            ):
                correct += 1
    if total == 0:
        return 0.0
    return round(correct / total, 3)


def _transcript_preview(window: list[RichSegment], *, limit: int = 220) -> str:
    transcript = format_transcript(window, use_seconds=True).replace("\n", " | ")
    return transcript[:limit] + ("..." if len(transcript) > limit else "")


def benchmark_model(model: str, samples: list[dict[str, object]]) -> dict[str, object]:
    settings.ollama_model = model
    warmup_window = samples[0]["window"]
    warmup_hook_delay = samples[0]["hook_delay_sec"]
    warmup_start = time.perf_counter()
    semantic_service.score_window(warmup_window, warmup_hook_delay, use_llm=True, llm_provider="ollama")
    warmup_latency_sec = time.perf_counter() - warmup_start

    rows: list[dict[str, object]] = []
    for sample in samples:
        started_at = time.perf_counter()
        scores = semantic_service.score_window(
            sample["window"],
            sample["hook_delay_sec"],
            use_llm=True,
            llm_provider="ollama",
        )
        latency_sec = time.perf_counter() - started_at
        row = {
            "candidate_id": sample["candidate_id"],
            "source": sample["source"],
            "score_hint": sample["score_hint"],
            "hook_delay_sec": sample["hook_delay_sec"],
            "llm_used": bool(scores["llm_used"]),
            "latency_sec": round(latency_sec, 3),
            "composite_score": _composite_score(scores),
            "scores": scores,
            "transcript_preview": sample["transcript_preview"],
        }
        rows.append(row)

    latencies = [float(row["latency_sec"]) for row in rows]
    llm_success_rate = round(sum(1 for row in rows if row["llm_used"]) / len(rows), 3)
    return {
        "model": model,
        "warmup_latency_sec": round(warmup_latency_sec, 3),
        "mean_latency_sec": round(statistics.mean(latencies), 3),
        "median_latency_sec": round(statistics.median(latencies), 3),
        "p95_latency_sec": round(sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)], 3),
        "llm_success_rate": llm_success_rate,
        "pairwise_order_accuracy_vs_score_hint": _pairwise_order_accuracy(rows),
        "rows": rows,
    }


def build_samples() -> list[dict[str, object]]:
    by_dir: dict[Path, tuple[list[RichSegment], dict[str, dict[str, object]]]] = {}
    samples: list[dict[str, object]] = []
    for spec in SAMPLE_CANDIDATES:
        if spec.debug_dir not in by_dir:
            by_dir[spec.debug_dir] = (
                _load_segments(spec.debug_dir),
                _load_candidates(spec.debug_dir),
            )
        segments, candidates = by_dir[spec.debug_dir]
        candidate = candidates[spec.candidate_id]
        window = _window_for_candidate(segments, candidate)
        samples.append(
            {
                "candidate_id": spec.candidate_id,
                "source": str(candidate["source"]),
                "score_hint": round(float(candidate["score_hint"]), 3),
                "hook_delay_sec": round(max(0.0, float(candidate["anchor_time"]) - float(candidate["start"])), 3),
                "window": window,
                "transcript_preview": _transcript_preview(window),
            }
        )
    return samples


def main() -> int:
    argv = sys.argv[1:]
    summary_only = False
    sample_limit: int | None = None
    if "--summary" in argv:
        summary_only = True
        argv = [arg for arg in argv if arg != "--summary"]
    if "--limit" in argv:
        limit_index = argv.index("--limit")
        sample_limit = int(argv[limit_index + 1])
        del argv[limit_index:limit_index + 2]
    requested_models = argv or ["llama3.2", "qwen3:4b", "gemma3:4b"]
    samples = build_samples()
    if sample_limit is not None:
        samples = samples[:sample_limit]
    results = {
        "sample_count": len(samples),
        "samples": [
            {
                "candidate_id": sample["candidate_id"],
                "source": sample["source"],
                "score_hint": sample["score_hint"],
                "hook_delay_sec": sample["hook_delay_sec"],
                "transcript_preview": sample["transcript_preview"],
            }
            for sample in samples
        ],
        "models": [benchmark_model(model, samples) for model in requested_models],
    }
    if summary_only:
        compact = {
            "sample_count": results["sample_count"],
            "models": [
                {
                    "model": model_result["model"],
                    "warmup_latency_sec": model_result["warmup_latency_sec"],
                    "mean_latency_sec": model_result["mean_latency_sec"],
                    "median_latency_sec": model_result["median_latency_sec"],
                    "p95_latency_sec": model_result["p95_latency_sec"],
                    "llm_success_rate": model_result["llm_success_rate"],
                    "pairwise_order_accuracy_vs_score_hint": model_result["pairwise_order_accuracy_vs_score_hint"],
                    "top_rows_by_composite": sorted(
                        [
                            {
                                "candidate_id": row["candidate_id"],
                                "score_hint": row["score_hint"],
                                "composite_score": row["composite_score"],
                                "hook_strength": row["scores"]["hook_strength"],
                                "context_dependency": row["scores"]["context_dependency"],
                            }
                            for row in model_result["rows"]
                        ],
                        key=lambda row: row["composite_score"],
                        reverse=True,
                    )[:3],
                    "bottom_rows_by_composite": sorted(
                        [
                            {
                                "candidate_id": row["candidate_id"],
                                "score_hint": row["score_hint"],
                                "composite_score": row["composite_score"],
                                "hook_strength": row["scores"]["hook_strength"],
                                "context_dependency": row["scores"]["context_dependency"],
                            }
                            for row in model_result["rows"]
                        ],
                        key=lambda row: row["composite_score"],
                    )[:3],
                }
                for model_result in results["models"]
            ],
        }
        print(json.dumps(compact, indent=2))
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
