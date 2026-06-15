"""Archetype-Judge Re-Test — Arm A (the neutral control).

PRD #95 / issue #96. The first arm of a two-arm offline re-test that isolates a
single variable — archetypes yes/no — on the existing replay infrastructure. Arm A
is the *neutral control*: a gemini-2.5-pro judge (temp 0) reads a raw transcript
excerpt (±12 s, no ``chat=``/``spk=`` markers) per candidate and scores it on a
neutral repost basis (0-1, score-only, NO archetypes, NO anti-loudness clause). The
judge plugs into the replay harness as the ``rank_score_fn`` over the deduped
top-200 of each VOD and is scored on recall@20 via ``directional_eval``.

Highest-possible reuse, new code only where the variable lives:
  * The deduped top-``depth`` scoping + surfaced cut comes from
    ``judge_spike.judge_shortlist_rerank`` unchanged.
  * The compact ``{"scores": [{"id", "repost_virality"}]}`` parse + fail-closed,
    truncation-salvaging apply comes from ``repost_virality_judge`` unchanged.
  * Swapped for Arm A: the provider seam (Anthropic Sonnet → **Gemini**), the
    prompt builder (neutral base, no anti-loudness), and the candidate format (the
    ``build_excerpt`` raw transcript instead of the generator's reason).

Two-Adapter-Disziplin: the paid run binds the provider to gemini-2.5-pro and
persists every raw response BEFORE parsing; a ``--replay`` binds it to those
persisted responses, so the verdict reproduces bit-for-bit network-free.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Callable

from ml import directional_eval, judge_spike, replay_harness
from ml import llm_moment_finder as d0
from ml import repost_virality_judge
from ml.repost_virality_judge import CallProviderFn, RerankFn, Row
from ml.strong_model_window_probe import build_excerpt

logger = logging.getLogger(__name__)

# The symmetric transcript pad around each candidate window. ±12 s by design (PRD
# #95 AC): enough context to read the beat, deliberately narrow so the judge scores
# the moment, not a whole chunk.
EXCERPT_PAD_SEC = 12.0

# The Sonnet-class provider both arms run on, over the existing Gemini seam.
GEMINI_PRO_MODEL = "gemini-2.5-pro"

# The deduped top-`depth` the judge scores (the seam the buried-real gold sit in) and
# the served final-set size — both reused from the existing harness, no new numbers.
DEFAULT_SHORTLIST_DEPTH = judge_spike.DEFAULT_SHORTLIST_DEPTH
DEFAULT_TOP_K = replay_harness.DEFAULT_TOP_K

# The population the re-test scores against: gold ∪ silver, equally weighted (PRD #95
# — silver is a strong transcript match missing only audio-position confirmation).
NEUTRAL_GOLD_TIERS = ("gold", "silver")

# vod_id, segments -> the per-VOD judge (paid Gemini+persist, or replay-from-disk).
# Unlike judge_spike's MakeJudgeFn, Arm A threads the VOD's segments in so the judge
# can build each candidate's transcript excerpt.
MakeJudgeFn = Callable[[str, list], RerankFn]
# vod_id -> (raw_rows, segments) or None when the VOD has no replayable artifacts.
LoadInputsFn = Callable[[str], "tuple[list[dict], list] | None"]


def neutral_repost_judge(
    *,
    call_provider: CallProviderFn,
    persist_raw: Callable[[str], None],
    segments: list,
    pad_sec: float = EXCERPT_PAD_SEC,
) -> RerankFn:
    """The real Arm-A adapter: judge a deduped candidate set on the neutral base.

    For each candidate it builds the ±``pad_sec`` raw transcript excerpt from
    ``segments`` (``build_excerpt`` — plain text, never the chat=/spk= annotated
    finder prompt), renders the neutral score-only prompt, calls the injected
    provider, persists the raw response BEFORE parsing, then decodes the compact
    scores and overwrites each scored row's rank-score. Parse + apply are the reused
    ``repost_virality_judge`` seam: fail-closed, truncation-salvaging, item-level no
    fabrication. The excerpt is a prompt input only — the returned rows stay clean
    serving currency.
    """

    def _rerank(rows: list[Row]) -> list[Row]:
        enriched = _attach_excerpts(rows, segments, pad_sec)
        prompt = build_neutral_judge_prompt(enriched)
        raw = call_provider(prompt)
        persist_raw(raw)
        scores = repost_virality_judge.parse_judge_scores(raw)
        reranked = repost_virality_judge.apply_scores(rows, scores)
        logger.info(
            json.dumps(
                {
                    "event": "neutral_repost_judge_scored",
                    "candidates": len(rows),
                    "scored": len(reranked),
                    "dropped": len(rows) - len(reranked),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return reranked

    return _rerank


def gemini_pro_provider(
    *,
    model: str = GEMINI_PRO_MODEL,
    temperature: float = 0.0,
    max_tokens: int = repost_virality_judge.DEFAULT_JUDGE_MAX_TOKENS,
    timeout_sec: float = d0.DEFAULT_TIMEOUT_SEC,
    retries: int = d0.DEFAULT_API_RETRIES,
    retry_backoff_sec: float = d0.DEFAULT_RETRY_BACKOFF_SEC,
) -> CallProviderFn:
    """Bind the provider seam to gemini-2.5-pro (temp 0) over the existing Gemini leaf.

    Mirrors ``repost_virality_judge.anthropic_sonnet_provider`` but routes through
    ``call_gemini_moment_finder`` — the Sonnet-class model the re-test runs on. temp 0
    keeps the paid run deterministic (replayable). Returns just the response text; the
    judge's own parser owns the structured decode.
    """

    def _call(prompt: str) -> str:
        text, _usage = d0.call_gemini_moment_finder(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        return text

    return _call


def fake_gemini_provider(raw: str) -> CallProviderFn:
    """A deterministic, network-free provider: returns ``raw`` for any prompt.

    The provider-seam fake the unit tests and a 1-VOD smoke can wire instead of paid
    Gemini — the judge's own parser decodes it on the same path the paid run uses, so
    the wiring is exercised without a paid call.
    """

    def _call(_prompt: str) -> str:
        return raw

    return _call


def _attach_excerpts(rows: list[Row], segments: list, pad_sec: float) -> list[Row]:
    """Each row's ±``pad_sec`` raw transcript excerpt, written onto a copy's ``text``.

    The single source of the candidate's transcript: ``build_excerpt`` over the VOD
    segments, the same pure padding the window-probe uses. Copies the row so the
    caller's serving-currency rows are never mutated.
    """
    enriched: list[Row] = []
    for row in rows:
        excerpt = build_excerpt(
            float(row.get("start_time") or 0.0),
            float(row.get("end_time") or 0.0),
            segments,
            pad_sec=pad_sec,
        )
        candidate = dict(row)
        candidate["text"] = excerpt["text"]
        enriched.append(candidate)
    return enriched


def build_neutral_judge_prompt(rows: list[Row]) -> str:
    """Render the neutral-base judge prompt: one line per candidate, scores only.

    The neutral repost rubric — 0-1, score-only, no archetypes, and crucially NO
    anti-loudness clause (the #50 confound the re-test removes). Each candidate is
    addressed by the same 1-based id the reused parser maps back and carries its raw
    transcript excerpt (the ``text`` field), so the judge scores the actual content
    rather than the generator's rationale.
    """
    candidate_lines = "\n".join(
        _format_candidate(index, row) for index, row in enumerate(rows, start=1)
    )
    return f"""You are an expert judge of REPOST virality for short-form Twitch clips.

You are scoring an already-selected, deduplicated candidate set — do NOT propose new \
moments. Score each candidate on how likely the moment is to go viral as a REPOST: the \
quotable, narrative beats (a sharp explanation, a memorable anecdote, a clean hot take) \
that get reposted and re-shared days later.

Use the full 0.0-1.0 scale and score each candidate independently on its own merit.

Return raw JSON only, no markdown, scores only (no explanations or rationale), this exact shape:
{{"scores": [{{"id": <candidate id>, "repost_virality": <0.0-1.0>}}]}}

Candidates:
{candidate_lines}
"""


def _format_candidate(index: int, row: Row) -> str:
    """``[id] | t=<start>s-<end>s | transcript: <excerpt>`` — the raw transcript
    excerpt only, no generator reason (Arm A reads the content, not the rationale)."""
    start = d0.float_or_none(row.get("start_time"))
    end = d0.float_or_none(row.get("end_time"))
    window = f"t={start}s-{end}s" if start is not None and end is not None else "t=?"
    text = str(row.get("text") or "").strip()
    parts = [f"[{index}]", window]
    if text:
        parts.append(f"transcript: {text}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — judge each VOD's deduped top-`depth`, score recall@20
# ─────────────────────────────────────────────────────────────────────────────

def run_arm(
    anchor_ids: list[str],
    manifest: dict[str, list[dict]],
    *,
    make_judge: MakeJudgeFn,
    load_inputs: LoadInputsFn,
    depth: int = DEFAULT_SHORTLIST_DEPTH,
    top_k: int = DEFAULT_TOP_K,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = NEUTRAL_GOLD_TIERS,
) -> dict:
    """Run one arm end-to-end: judge each VOD's deduped top-``depth``, score recall@20.

    For every anchor VOD it loads the raw rows + segments, builds the per-VOD judge
    (``make_judge(vod_id, segments)``), scopes it to the deduped top-``depth`` via
    ``judge_spike.judge_shortlist_rerank`` (the same surfaced cut as #46), and runs
    the real serving post-processing under it to get the served top-``top_k``. The
    aggregated final + raw sets feed ``directional_eval.run`` over gold ∪ silver, so
    the result is a directional_eval result the B-vs-A gate (#98) can diff against the
    other arm. VODs with no replayable artifacts are a surfaced coverage gap — never
    silent misses; the per-VOD shortlist cut is recorded.
    """
    raw_by_vod: dict[str, list[dict]] = {}
    final_by_vod: dict[str, list[dict]] = {}
    shortlist_cuts: dict[str, dict] = {}
    gaps: list[str] = []

    for vod_id in anchor_ids:
        loaded = load_inputs(vod_id)
        if loaded is None:
            gaps.append(vod_id)
            continue
        raw_rows, segments = loaded
        judge = make_judge(vod_id, segments)
        rank_score_fn = judge_spike.judge_shortlist_rerank(
            judge, depth=depth,
            on_cut=lambda cut, v=vod_id: shortlist_cuts.__setitem__(v, cut),
        )
        raw_by_vod[vod_id] = raw_rows
        final_by_vod[vod_id] = replay_harness.replay_postprocess(
            raw_rows, segments, rank_score_fn=rank_score_fn, top_k=top_k)

    result = directional_eval.run(
        manifest, final_by_vod, raw_by_vod,
        anchor_vod_ids=list(anchor_ids), ks=ks,
        headline_min_views=headline_min_views, gold_tiers=tuple(gold_tiers),
    )
    result["coverage_gap_vods"] = gaps
    result["shortlist_cuts"] = shortlist_cuts
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Paid / replay wiring — the two make_judge factories + raw-response persistence
# ─────────────────────────────────────────────────────────────────────────────

def _raw_path(artifacts_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return pathlib.Path(artifacts_dir) / f"{vod_id}.raw.txt"


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def paid_make_judge(
    artifacts_dir: pathlib.Path,
    *,
    provider: CallProviderFn | None = None,
    pad_sec: float = EXCERPT_PAD_SEC,
) -> MakeJudgeFn:
    """Per-VOD judge bound to the paid provider, persisting each raw response.

    Defaults the provider to gemini-2.5-pro (temp 0); tests inject a network-free
    fake. The raw response is persisted (``{vod_id}.raw.txt``) BEFORE parsing — the
    re-evaluable artifact a ``--replay`` re-reads. ``run_arm`` calls the judge once
    per VOD, so this is a single paid call per VOD.
    """
    call_provider = provider or gemini_pro_provider()

    def _make(vod_id: str, segments: list) -> RerankFn:
        raw_path = _raw_path(artifacts_dir, vod_id)
        return neutral_repost_judge(
            call_provider=call_provider,
            persist_raw=lambda text: _write_text(raw_path, text),
            segments=segments,
            pad_sec=pad_sec,
        )

    return _make


def replay_make_judge(
    artifacts_dir: pathlib.Path, *, pad_sec: float = EXCERPT_PAD_SEC
) -> MakeJudgeFn:
    """Per-VOD judge bound to the persisted raw response — re-scores without paying.

    The provider seam is the persisted-raw fake, so the adapter's own parser decodes
    it on the *same* path the paid run used — bit-for-bit, no network.
    """

    def _make(vod_id: str, segments: list) -> RerankFn:
        raw = _raw_path(artifacts_dir, vod_id).read_text(encoding="utf-8")
        return neutral_repost_judge(
            call_provider=fake_gemini_provider(raw),
            persist_raw=lambda _t: None,
            segments=segments,
            pad_sec=pad_sec,
        )

    return _make


def replay_load_inputs(
    storage: pathlib.Path, artifacts_dir: pathlib.Path
) -> LoadInputsFn:
    """Replay loader: a VOD with no persisted raw was not part of the paid run, so it
    is a surfaced coverage gap (``None``) rather than a re-priced call."""

    def _load(vod_id: str):
        if not _raw_path(artifacts_dir, vod_id).exists():
            return None
        return replay_harness.load_replay_inputs(vod_id, pathlib.Path(storage))

    return _load
