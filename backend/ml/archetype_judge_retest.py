"""Archetype-Judge Re-Test — both arms + the pre-fixed verdict.

PRD #95 (issues #96, #98). A two-arm offline re-test that isolates a single variable —
archetypes yes/no — on the existing replay infrastructure. Both arms are a gemini-2.5-pro
judge (temp 0) that reads a raw transcript excerpt (±12 s, no ``chat=``/``spk=`` markers)
per candidate and scores 0-1 (score-only):
  * **Arm A** (the neutral control, #96): the neutral repost base — NO archetypes, NO
    anti-loudness clause (the #50 confound this re-test removes).
  * **Arm B** (the archetype arm, #98): the SAME neutral base + the 8 Jynxzi archetypes
    injected (name + description + ``transcript_signals`` via ``render_profile_rules``,
    archetypes only) + a score-higher-on-match instruction. The archetype block is the
    ONLY difference between the arms.

Both arms judge the **full deduped pool** of each VOD — no top-N cut (#98) — so every
``clean_but_buried`` gold is in scope (a 200-cut hid 16/50 of them on gold∪silver), and
both run at the SAME depth so the A/B comparison is symmetric. Each arm is scored on
recall@20 via ``directional_eval`` over gold∪silver (equally weighted).

The **verdict** (#98) reads the combined (gold∪silver) floor recall@20 from both arms and
gates on a pre-fixed cut: PASS iff Arm B beats Arm A by ≥ +0.03, else NO-GO. Read-only
breakdowns (gold-only, silver-only, combined-headline recall@20) and the per-gold moves
(golds B served in its top-20 that A did not) are carried beside the gate, never gated on.

Highest-possible reuse, new code only where the variable lives:
  * The deduped top-``depth`` scoping (and ``depth=None`` full-pool) + surfaced cut comes
    from ``judge_spike.judge_shortlist_rerank``.
  * The compact ``{"scores": [{"id", "repost_virality"}]}`` parse + fail-closed,
    truncation-salvaging apply comes from ``repost_virality_judge`` unchanged.
  * The two arms share ONE adapter; ``build_prompt`` is the only thing they vary.

Two-Adapter-Disziplin: a paid run binds the provider to gemini-2.5-pro and persists every
raw response (per arm, own dir) BEFORE parsing; a ``--replay`` binds it to those persisted
responses, so the verdict reproduces bit-for-bit network-free.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Callable

from ml import directional_eval, judge_spike, replay_harness
from ml import llm_moment_finder as d0
from ml import repost_virality_judge
from ml import transcribe_only_rerun
from ml.repost_virality_judge import CallProviderFn, RerankFn, Row
from ml.strong_model_window_probe import build_excerpt
from models import ClipResult, JobStatus
from producers.clipping import moments_from_post_processed

logger = logging.getLogger(__name__)

# The symmetric transcript pad around each candidate window. ±12 s by design (PRD
# #95 AC): enough context to read the beat, deliberately narrow so the judge scores
# the moment, not a whole chunk.
EXCERPT_PAD_SEC = 12.0

# The Sonnet-class provider both arms run on, over the existing Gemini seam.
GEMINI_PRO_MODEL = "gemini-2.5-pro"

# The output-token cap for the judge. gemini-2.5-pro is a REASONING model: it spends a
# (hidden) thinking budget out of ``max_tokens`` BEFORE emitting visible content, so the
# 8k judge default let thinking eat the whole budget and return an EMPTY response
# (finish_reason=length, completion content 0) on a full-pool judge — a fake 0.000
# verdict. The full deduped pool runs to ~1300 candidates, whose compact score JSON alone
# is ~26k tokens; the model's max output (65535) leaves room for both the thinking budget
# and that JSON. The largest pool still tail-truncates — the truncation-salvage parser
# recovers the complete items (the buried golds sit near the top, far from the dropped tail).
GEMINI_JUDGE_MAX_TOKENS = 65535

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


class EmptyJudgeResponseError(RuntimeError):
    """The judge scored ZERO of N real candidates — an empty / over-truncated response (e.g.
    gemini-2.5-pro's thinking budget ate the whole ``max_tokens`` cap). Raised so a failed
    paid call aborts the run LOUDLY instead of silently scoring recall to 0 — a fake NO-GO
    that reads like a valid verdict. With ``GEMINI_JUDGE_MAX_TOKENS`` this should never fire;
    it is the integrity net the 8k-cap empty-response failure (#101 smoke) revealed."""


def neutral_repost_judge(
    *,
    call_provider: CallProviderFn,
    persist_raw: Callable[[str], None],
    segments: list,
    pad_sec: float = EXCERPT_PAD_SEC,
    build_prompt: "Callable[[list[Row]], str] | None" = None,
) -> RerankFn:
    """The real judge adapter: judge a deduped candidate set on the chosen rubric.

    For each candidate it builds the ±``pad_sec`` raw transcript excerpt from
    ``segments`` (``build_excerpt`` — plain text, never the chat=/spk= annotated
    finder prompt), renders the prompt via ``build_prompt``, calls the injected
    provider, persists the raw response BEFORE parsing, then decodes the compact
    scores and overwrites each scored row's rank-score. Parse + apply are the reused
    ``repost_virality_judge`` seam: fail-closed, truncation-salvaging, item-level no
    fabrication. The excerpt is a prompt input only — the returned rows stay clean
    serving currency.

    ``build_prompt`` is the ONE thing the two arms vary: it defaults to Arm A's neutral
    ``build_neutral_judge_prompt``; Arm B passes ``archetype_prompt_builder(archetypes)``
    so the same adapter renders the archetype variant.
    """
    render_prompt = build_prompt or build_neutral_judge_prompt

    def _rerank(rows: list[Row]) -> list[Row]:
        enriched = _attach_excerpts(rows, segments, pad_sec)
        prompt = render_prompt(enriched)
        raw = call_provider(prompt)
        persist_raw(raw)
        scores = repost_virality_judge.parse_judge_scores(raw)
        reranked = repost_virality_judge.apply_scores(rows, scores)
        if rows and not reranked:
            raise EmptyJudgeResponseError(
                f"judge scored 0 of {len(rows)} candidates — empty/over-truncated "
                "response (raise max_tokens / check the provider), not a real 0.0 verdict"
            )
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
    max_tokens: int = GEMINI_JUDGE_MAX_TOKENS,
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
    """Arm A's prompt: the neutral-base judge prompt, one line per candidate, scores only.

    The neutral repost rubric — 0-1, score-only, no archetypes, and crucially NO
    anti-loudness clause (the #50 confound the re-test removes). Each candidate is
    addressed by the same 1-based id the reused parser maps back and carries its raw
    transcript excerpt (the ``text`` field), so the judge scores the actual content
    rather than the generator's rationale.
    """
    return _render_judge_prompt(rows)


def build_archetype_judge_prompt(rows: list[Row], *, archetypes: list) -> str:
    """Arm B's prompt: the SAME neutral base + the archetype section (#98).

    The single isolated variable. The Jynxzi ``archetypes`` (name + description +
    ``transcript_signals``) are rendered through the SAME ``render_profile_rules`` the
    serving finder uses — **archetypes only**, so the profile's ``selection_rules`` and
    ``avoid`` never leak in and B − A differs by exactly the archetypes — followed by
    the one instruction that rewards a transcript matching an archetype. The block
    appears ONLY here; ``build_neutral_judge_prompt`` (Arm A) never carries it.
    """
    return _render_judge_prompt(rows, archetype_section=_archetype_section(archetypes))


def _archetype_section(archetypes: list) -> str:
    """The Arm-B-only archetype block: ``render_profile_rules`` over the archetypes
    ALONE (no selection_rules/avoid), plus the score-higher instruction. Empty when
    there are no archetypes, so the section never injects a bare instruction."""
    block = d0.render_profile_rules({"archetypes": list(archetypes or [])})
    if not block.strip():
        return ""
    return (
        "\n\nScore HIGHER when the transcript matches one of these Jynxzi clip "
        "archetypes (a transcript that fits an archetype is strong evidence of repost "
        "virality):\n" + block
    )


def archetype_prompt_builder(archetypes: list) -> "Callable[[list[Row]], str]":
    """Arm B's ``build_prompt`` seam: a 1-arg prompt builder bound to ``archetypes``.

    Threads the archetypes into ``build_archetype_judge_prompt`` so a make_judge factory
    (paid or replay) renders Arm B's prompt with the same 1-arg signature Arm A's neutral
    builder has — the single difference between the two arms' judges.
    """

    def _build(rows: list[Row]) -> str:
        return build_archetype_judge_prompt(rows, archetypes=archetypes)

    return _build


def load_jynxzi_archetypes() -> list:
    """The 8 Jynxzi archetypes from the committed ``jynxzi_profile.json`` (#98 AC).

    Each carries name + description + ``transcript_signals`` — exactly the fields
    ``render_profile_rules`` injects into Arm B's prompt. This is the canonical wiring an
    external paid driver hands to ``archetype_prompt_builder``.
    """
    profile = d0.load_clip_profile("jynxzi") or {}
    return list(profile.get("archetypes") or [])


def _render_judge_prompt(rows: list[Row], *, archetype_section: str = "") -> str:
    """The shared judge prompt: neutral repost base + an optional archetype section.

    Both arms render through here; ``archetype_section`` is the *only* thing that
    differs between Arm A ("") and Arm B (the rendered archetypes).
    """
    candidate_lines = "\n".join(
        _format_candidate(index, row) for index, row in enumerate(rows, start=1)
    )
    return f"""You are an expert judge of REPOST virality for short-form Twitch clips.

You are scoring an already-selected, deduplicated candidate set — do NOT propose new \
moments. Score each candidate on how likely the moment is to go viral as a REPOST: the \
quotable, narrative beats (a sharp explanation, a memorable anecdote, a clean hot take) \
that get reposted and re-shared days later.{archetype_section}

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
    depth: int | None = DEFAULT_SHORTLIST_DEPTH,
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
    # Expose the served + raw predictions so the verdict (#98) can re-score per tier
    # (gold-only/silver-only breakdowns) without re-judging.
    result["final_by_vod"] = final_by_vod
    result["raw_by_vod"] = raw_by_vod
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
    build_prompt: "Callable[[list[Row]], str] | None" = None,
) -> MakeJudgeFn:
    """Per-VOD judge bound to the paid provider, persisting each raw response.

    Defaults the provider to gemini-2.5-pro (temp 0); tests inject a network-free
    fake. The raw response is persisted (``{vod_id}.raw.txt``) BEFORE parsing — the
    re-evaluable artifact a ``--replay`` re-reads. ``run_arm`` calls the judge once
    per VOD, so this is a single paid call per VOD. ``build_prompt`` selects the rubric:
    omitted = Arm A (neutral); ``archetype_prompt_builder(archetypes)`` = Arm B.
    """
    call_provider = provider or gemini_pro_provider()

    def _make(vod_id: str, segments: list) -> RerankFn:
        raw_path = _raw_path(artifacts_dir, vod_id)
        return neutral_repost_judge(
            call_provider=call_provider,
            persist_raw=lambda text: _write_text(raw_path, text),
            segments=segments,
            pad_sec=pad_sec,
            build_prompt=build_prompt,
        )

    return _make


def replay_make_judge(
    artifacts_dir: pathlib.Path,
    *,
    pad_sec: float = EXCERPT_PAD_SEC,
    build_prompt: "Callable[[list[Row]], str] | None" = None,
) -> MakeJudgeFn:
    """Per-VOD judge bound to the persisted raw response — re-scores without paying.

    The provider seam is the persisted-raw fake, so the adapter's own parser decodes
    it on the *same* path the paid run used — bit-for-bit, no network. ``build_prompt``
    mirrors the paid arm's rubric (the persisted-raw fake ignores the prompt, but keeping
    it identical means the replay code path matches the paid one exactly).
    """

    def _make(vod_id: str, segments: list) -> RerankFn:
        raw = _raw_path(artifacts_dir, vod_id).read_text(encoding="utf-8")
        return neutral_repost_judge(
            call_provider=fake_gemini_provider(raw),
            persist_raw=lambda _t: None,
            segments=segments,
            pad_sec=pad_sec,
            build_prompt=build_prompt,
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


def fresh_segment_load_inputs(
    storage: pathlib.Path, predictions_dir: pathlib.Path
) -> LoadInputsFn:
    """Pair the candidate POOL with the FRESH re-transcribed segments (#100, Slice 5).

    The candidate pool (raw rows) is the deduped generation the #46/#50 pool already
    holds — loaded from the VOD's original replayable job dir via
    ``replay_harness.load_replay_inputs``. The segment timeline, however, is REPLACED by
    the fresh transcribe-only ``segments.jsonl`` (``transcribe_only_<vod>/debug/``, the
    consistent ``large-v3-turbo`` basis #100 minted), so the judge builds every excerpt
    from the re-transcribed transcript — never the stale one the pool dir was generated
    against. A VOD missing EITHER the pool OR the fresh transcript is a surfaced coverage
    gap (``None``), never judged on a half-stale input.
    """

    def _load(vod_id: str):
        loaded = replay_harness.load_replay_inputs(vod_id, pathlib.Path(storage))
        if loaded is None:
            return None
        raw_rows, _stale_segments = loaded
        fresh_path = transcribe_only_rerun.fresh_segments_path(
            pathlib.Path(predictions_dir), vod_id)
        if not fresh_path.exists():
            return None
        return raw_rows, replay_harness.load_segments(fresh_path)

    return _load


# ─────────────────────────────────────────────────────────────────────────────
# Issue #101 — the consistency pre-condition: no paid call until the #100 report is
# clean for the VODs being judged (the documented no-video exclusions never block).
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptConsistencyError(RuntimeError):
    """The paid run's pre-condition failed: a judged VOD's fresh transcript is flagged
    (or missing) in the #100 consistency report. Raised BEFORE any paid Gemini call so a
    drifted transcript never reaches the judge (PRD #95 / #101 — the paid run starts only
    after a green consistency report)."""


def assert_transcripts_consistent(
    report: "transcribe_only_rerun.ConsistencyReport", judged_vod_ids: list[str]
) -> None:
    """Hard gate: refuse the paid run unless EVERY judged VOD passed the #100 report.

    ``report`` is the ``transcribe_only_rerun.ConsistencyReport`` over the fresh
    transcripts. The documented exclusions (anchor VODs with no local video) make
    ``report.clean`` False but are NOT in ``judged_vod_ids`` — they never block. A judged
    VOD that is FLAGGED, or absent from the report entirely (no fresh transcript minted),
    is fail-closed: it raises ``TranscriptConsistencyError`` naming the offenders, so the
    paid run aborts before a single drifted transcript is judged.
    """
    by_id = {r.vod_id: r for r in report.vod_reports}
    offenders: list[str] = []
    for vod_id in judged_vod_ids:
        vod_report = by_id.get(vod_id)
        if vod_report is None:
            offenders.append(f"{vod_id} (no fresh transcript)")
        elif not vod_report.passed:
            offenders.append(f"{vod_id} ({','.join(vod_report.flags)})")
    if offenders:
        raise TranscriptConsistencyError(
            "consistency report not clean for judged VODs — aborting paid run: "
            + "; ".join(offenders)
        )


# ─────────────────────────────────────────────────────────────────────────────
# The verdict — the pre-fixed combined-tier B-vs-A gate (#98)
# ─────────────────────────────────────────────────────────────────────────────

# The pre-fixed cut, IN CODE before the run (PRD #95): Arm B (archetype) must beat
# Arm A (neutral) on combined (gold∪silver) floor recall@20 by at least this much, or
# the archetype variable is NO-GO. Read explicitly, never rationalised after the fact.
COMBINED_GATE_CUT = 0.03

# The gate column — the product headline K (= clip_top_k = 20), the served top-K.
GATE_K = directional_eval.PRODUCT_HEADLINE_K


def _signed(delta: float | None) -> str:
    return "—" if delta is None else f"{delta:+.6g}"


def _recall_at_k(arm_result: dict, tier: str, k: int = GATE_K) -> float | None:
    """A tier's product recall@k from a directional_eval.run result, or None."""
    value = arm_result.get(tier, {}).get("recall", {}).get(str(k))
    return float(value) if value is not None else None


def _cell(current: float | None, baseline: float | None) -> dict:
    """A {current, baseline, delta} cell; delta is None unless both are numbers."""
    delta = round(current - baseline, 6) if (current is not None and baseline is not None) else None
    return {"current": current, "baseline": baseline, "delta": delta}


def evaluate_combined_gate(current: dict, baseline: dict, *, cut: float = COMBINED_GATE_CUT) -> dict:
    """The pre-fixed verdict: PASS iff Arm B beats Arm A on combined floor recall@20 by ≥ cut.

    Both args are ``directional_eval.run`` results over gold∪silver (``current`` = Arm B,
    ``baseline`` = Arm A). The gate reads the **combined** floor recall@20 — the floor
    tier already spans gold∪silver, equally weighted, because both arms run with
    ``gold_tiers=(gold, silver)``. PASS iff the rise (B − A) meets the pre-fixed ``cut``
    (+0.03); a smaller rise — or a missing recall on either side — fails closed (an
    unmeasurable move is never read as an improvement). Returns the verdict, the delta
    cell, and a one-line reason so the read is explicit, not rationalised.
    """
    cell = _cell(_recall_at_k(current, "floor"), _recall_at_k(baseline, "floor"))
    delta = cell["delta"]
    passed = delta is not None and delta >= cut
    reason = (
        f"combined floor r@{GATE_K} B {_fmt(cell['current'])} − A {_fmt(cell['baseline'])} "
        f"= {_signed(delta)} (cut {cut:+g}) → {'PASS' if passed else 'NO-GO'}"
    )
    return {
        "passed": passed,
        "k": GATE_K,
        "cut": cut,
        "combined_floor_recall_at_20": cell,
        "reason": reason,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _served_within_k(hit: dict | None, k: int) -> bool:
    """True iff the gold was served within the top-k — a hit at rank ≤ k."""
    return bool(
        hit and hit.get("hit") and hit.get("hit_rank") is not None and hit["hit_rank"] <= k
    )


def per_gold_moves(current: dict, baseline: dict, *, k: int = GATE_K) -> list[dict]:
    """The golds Arm B served in its top-k that Arm A's did not — the per-gold moves.

    Joins the two arms' ``directional_eval`` ``per_vod`` gt-hit tables by
    ``(vod_id, gt_index)`` — both arms score the same manifest, so the index is a
    stable key — and keeps each gold B serves within its top-k (``hit_rank ≤ k``) that
    A does not. Each move carries both ranks (A's is ``None`` when A never served it),
    so a reader sees exactly which clean_but_buried golds the archetype rubric lifted.
    Ordered by views desc, so the biggest moves read first.
    """
    a_by_vod = {
        v["vod_id"]: {h["gt_index"]: h for h in v.get("gt_hits", [])}
        for v in baseline.get("per_vod", [])
    }
    moves: list[dict] = []
    for vb in current.get("per_vod", []):
        a_hits = a_by_vod.get(vb["vod_id"], {})
        for hb in vb.get("gt_hits", []):
            ha = a_hits.get(hb["gt_index"])
            if _served_within_k(hb, k) and not _served_within_k(ha, k):
                moves.append({
                    "vod_id": vb["vod_id"],
                    "gt_index": hb["gt_index"],
                    "desc": hb.get("desc", ""),
                    "views": float(hb.get("views") or 0.0),
                    "a_rank": ha.get("hit_rank") if ha else None,
                    "b_rank": hb.get("hit_rank"),
                })
    moves.sort(key=lambda m: m["views"], reverse=True)
    return moves


def build_breakdowns(
    manifest: dict[str, list[dict]],
    anchor_ids: list[str],
    current: dict,
    baseline: dict,
    *,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
) -> dict:
    """The read-only breakdown cells the verdict carries (never gates on) — #98 AC.

    gold-only and silver-only recall@20 re-score each arm's served + raw predictions
    through ``directional_eval`` restricted to that single tier, so the combined gate
    number is decomposed without re-judging. combined-headline recall@20 is read
    straight from each arm's headline tier (gold∪silver, views ≥ ``headline_min_views``).
    Each is a {current=B, baseline=A, delta} cell — colour beside the gated combined
    floor, surfaced so a reader sees *which* tier moved.
    """

    def _floor20(arm: dict, tier: str) -> float | None:
        result = directional_eval.run(
            manifest, arm.get("final_by_vod", {}), arm.get("raw_by_vod", {}),
            anchor_vod_ids=list(anchor_ids), ks=ks,
            headline_min_views=headline_min_views, gold_tiers=(tier,),
        )
        return _recall_at_k(result, "floor")

    return {
        "gold_only_recall_at_20": _cell(_floor20(current, "gold"), _floor20(baseline, "gold")),
        "silver_only_recall_at_20": _cell(_floor20(current, "silver"), _floor20(baseline, "silver")),
        "combined_headline_recall_at_20": _cell(
            _recall_at_k(current, "headline"), _recall_at_k(baseline, "headline")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — run both arms at full depth, read the pre-fixed verdict
# ─────────────────────────────────────────────────────────────────────────────

# Judge the WHOLE deduped pool, no top-N cut (#98): both arms run at full depth so the
# A/B comparison is symmetric and every clean_but_buried gold is in the judge's scope.
FULL_POOL_DEPTH: int | None = None


def run_verdict(
    anchor_ids: list[str],
    manifest: dict[str, list[dict]],
    *,
    make_judge_a: MakeJudgeFn,
    make_judge_b: MakeJudgeFn,
    load_inputs_a: LoadInputsFn,
    load_inputs_b: LoadInputsFn | None = None,
    depth: int | None = FULL_POOL_DEPTH,
    top_k: int = DEFAULT_TOP_K,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = NEUTRAL_GOLD_TIERS,
    cut: float = COMBINED_GATE_CUT,
) -> dict:
    """Run both arms at full depth, then read the pre-fixed combined-tier verdict (#98).

    Arm A (neutral) is the baseline, Arm B (archetype) the current — both judged at the
    SAME unrestricted ``depth`` (default the full pool) so the A/B comparison is symmetric
    and every clean_but_buried gold is in scope. Returns the gate (B − A combined floor
    recall@20 against the pre-fixed +0.03 ``cut``), the read-only breakdowns, the per-gold
    moves, and both full arm results. ``load_inputs_b`` defaults to ``load_inputs_a`` (a
    paid run shares one storage); a ``--replay`` passes each arm its own persisted-raw
    loader, so a VOD only one arm judged is a surfaced gap in that arm, never a silent miss.
    """
    load_inputs_b = load_inputs_b or load_inputs_a
    arm_kwargs = dict(depth=depth, top_k=top_k, ks=ks,
                      headline_min_views=headline_min_views, gold_tiers=gold_tiers)
    arm_a = run_arm(anchor_ids, manifest, make_judge=make_judge_a,
                    load_inputs=load_inputs_a, **arm_kwargs)
    arm_b = run_arm(anchor_ids, manifest, make_judge=make_judge_b,
                    load_inputs=load_inputs_b, **arm_kwargs)
    return {
        "anchor_vods": list(anchor_ids),
        "depth": depth,
        "gate": evaluate_combined_gate(arm_b, arm_a, cut=cut),
        "breakdowns": build_breakdowns(manifest, anchor_ids, arm_b, arm_a,
                                       ks=ks, headline_min_views=headline_min_views),
        "per_gold_moves": per_gold_moves(arm_b, arm_a),
        "arm_a": arm_a,
        "arm_b": arm_b,
    }


# The pre-fixed follow-on decision, IN CODE before the run (PRD #95 verdict table):
# a PASS (combined floor r@20 B−A ≥ +0.03) means the archetype-judge lever lives →
# follow-on serving/blend-α PRD; a NO-GO closes the ranking-track cell directly.
DECISION_ARCHETYPE_JUDGE_LIVES = "ARCHETYPE_JUDGE_LIVES"
DECISION_RANKING_TRACK_CLOSED = "RANKING_TRACK_CLOSED"


def verdict_decision(verdict: dict) -> str:
    """The pre-fixed follow-on decision a verdict reads to — ABLESEN, nicht rationalisiert.

    PASS (the gate cleared the +0.03 combined-floor cut) → ``ARCHETYPE_JUDGE_LIVES``;
    otherwise ``RANKING_TRACK_CLOSED``. The mapping is fixed in code before the paid run.
    """
    return (DECISION_ARCHETYPE_JUDGE_LIVES if verdict["gate"]["passed"]
            else DECISION_RANKING_TRACK_CLOSED)


def format_verdict_table(verdict: dict) -> str:
    """The pre-fixed verdict table the runbook reads off (the verdict, not a rationalisation).

    One block: the gated headline (PASS/NO-GO + the follow-on decision label), the gate's
    own reason line (B − A combined floor r@20 against the +0.03 cut), the read-only
    breakdown cells (gold-only / silver-only / combined-headline), and the per-gold moves
    (the golds B served in its top-20 that A did not). Everything is read straight from the
    verdict — the reader sees the numbers and the fixed decision, not an interpretation.
    """
    gate = verdict["gate"]
    lines = [
        f"VERDICT: {'PASS' if gate['passed'] else 'NO-GO'} → {verdict_decision(verdict)}",
        f"  gate: {gate['reason']}",
        "  read-only breakdowns (carried, not gated):",
    ]
    breakdowns = verdict.get("breakdowns", {})
    for key in ("gold_only_recall_at_20", "silver_only_recall_at_20",
                "combined_headline_recall_at_20"):
        cell = breakdowns.get(key)
        if cell:
            lines.append(
                f"    {key}: B {_fmt(cell['current'])} − A {_fmt(cell['baseline'])} "
                f"= {_signed(cell['delta'])}")
    moves = verdict.get("per_gold_moves", [])
    lines.append(
        f"  per-gold moves (golds B served in top-20 that A did not): {len(moves)}")
    for move in moves:
        lines.append(
            f"    {move['vod_id']} '{move['desc']}' views={move['views']:.0f} "
            f"A=#{move['a_rank']} → B=#{move['b_rank']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Issue #101 — the paid / replay drivers: wire both arms onto the FRESH transcripts,
# gate the paid run on the #100 report, persist per arm, replay free.
#
# arm_a / arm_b raw-response dirs live under one ``artifacts_root`` so the two arms
# never share a persisted file; both read the same candidate pool + fresh transcripts
# (only the prompt differs), so B − A isolates exactly the archetypes.
# ─────────────────────────────────────────────────────────────────────────────

def _arm_dirs(artifacts_root: pathlib.Path) -> "tuple[pathlib.Path, pathlib.Path]":
    root = pathlib.Path(artifacts_root)
    return root / "arm_a", root / "arm_b"


def _excluding(load_inputs: LoadInputsFn, exclude_vods: "list[str] | None") -> LoadInputsFn:
    """Wrap a loader so excluded VODs return ``None`` — a SURFACED coverage gap, never
    judged and never paid for. Used to drop VODs whose deduped pool is too large for a
    single reliable gemini-2.5-pro judge call (#101 — over-generation / empty response at
    ~1000 candidates), so they are documented out, not silently folded as misses."""
    excluded = set(exclude_vods or [])
    if not excluded:
        return load_inputs

    def _load(vod_id: str):
        return None if vod_id in excluded else load_inputs(vod_id)

    return _load


def _judged_in_report(
    anchor_ids: list[str], report: "transcribe_only_rerun.ConsistencyReport"
) -> list[str]:
    """The anchors the paid run will actually judge — those covered by the consistency
    report (a fresh transcript was minted). The no-video exclusions, absent from the
    report, fall out here and surface later as coverage gaps, never as gate failures."""
    covered = {r.vod_id for r in report.vod_reports}
    return [v for v in anchor_ids if v in covered]


def run_paid_verdict(
    anchor_ids: list[str],
    manifest: dict[str, list[dict]],
    *,
    storage: pathlib.Path,
    predictions_dir: pathlib.Path,
    artifacts_root: pathlib.Path,
    report: "transcribe_only_rerun.ConsistencyReport",
    provider: CallProviderFn | None = None,
    archetypes: "list | None" = None,
    exclude_vods: "list[str] | None" = None,
    **verdict_kwargs,
) -> dict:
    """The paid two-arm run: gate on the #100 report, then judge both arms and persist.

    The HARD gate fires first — ``assert_transcripts_consistent`` over the judged anchors
    — so a single drifted transcript aborts BEFORE any paid Gemini call (no money spent,
    nothing persisted). Then both arms run ``run_verdict`` over the shared fresh-transcript
    inputs (``fresh_segment_load_inputs`` — #100's basis, never the stale pool transcript),
    with the SAME ``provider`` but different prompts: Arm A neutral, Arm B the archetype
    rubric (``archetypes`` defaults to the 8 committed Jynxzi archetypes). Each arm persists
    its raw response to its OWN dir, so a later ``run_replay_verdict`` reproduces the verdict
    network-free.
    """
    assert_transcripts_consistent(report, _judged_in_report(anchor_ids, report))
    archetypes = load_jynxzi_archetypes() if archetypes is None else archetypes
    arm_a_dir, arm_b_dir = _arm_dirs(artifacts_root)
    load_inputs = _excluding(fresh_segment_load_inputs(storage, predictions_dir), exclude_vods)
    return run_verdict(
        anchor_ids, manifest,
        make_judge_a=paid_make_judge(arm_a_dir, provider=provider),
        make_judge_b=paid_make_judge(
            arm_b_dir, provider=provider,
            build_prompt=archetype_prompt_builder(archetypes)),
        load_inputs_a=load_inputs,
        **verdict_kwargs,
    )


def _gated_replay_inputs(artifacts_dir: pathlib.Path, base_load: LoadInputsFn) -> LoadInputsFn:
    """Per-arm replay loader: a VOD with no persisted raw for THIS arm is a surfaced gap,
    so a VOD only the other arm judged never folds in as a silent miss."""

    def _load(vod_id: str):
        if not _raw_path(artifacts_dir, vod_id).exists():
            return None
        return base_load(vod_id)

    return _load


def run_replay_verdict(
    anchor_ids: list[str],
    manifest: dict[str, list[dict]],
    *,
    storage: pathlib.Path,
    predictions_dir: pathlib.Path,
    artifacts_root: pathlib.Path,
    archetypes: "list | None" = None,
    exclude_vods: "list[str] | None" = None,
    **verdict_kwargs,
) -> dict:
    """Reproduce the verdict free, network-free, from the persisted A and B raw responses.

    Each arm re-scores its OWN persisted raw (``replay_make_judge``) over the SAME fresh
    transcripts the paid run used; the per-arm gate (``_gated_replay_inputs``) surfaces a
    VOD an arm never judged as a coverage gap rather than re-pricing it. With identical
    inputs and identical persisted responses, the result is bit-for-bit the paid verdict —
    the read-only ``--replay`` the runbook reads the verdict table from.
    """
    archetypes = load_jynxzi_archetypes() if archetypes is None else archetypes
    arm_a_dir, arm_b_dir = _arm_dirs(artifacts_root)
    base_load = _excluding(fresh_segment_load_inputs(storage, predictions_dir), exclude_vods)
    return run_verdict(
        anchor_ids, manifest,
        make_judge_a=replay_make_judge(arm_a_dir),
        make_judge_b=replay_make_judge(
            arm_b_dir, build_prompt=archetype_prompt_builder(archetypes)),
        load_inputs_a=_gated_replay_inputs(arm_a_dir, base_load),
        load_inputs_b=_gated_replay_inputs(arm_b_dir, base_load),
        **verdict_kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Render-Step (#99) — each arm's gated top-20 as raw 16:9 cuts → frontend jobs
#
# The recall measured exactly ``replay_postprocess``'s served top-20; this renders
# that SAME set as raw cuts and registers it as a frontend job, so the existing
# job/clip viewer shows what the recall scored — no Pipeline-Re-Run, no reframe,
# no video copy. The ffmpeg cutter is an injected seam (the module stays import-light
# — ffmpeg is a Docker-only leaf), with a real default bound lazily inside Docker.
# ─────────────────────────────────────────────────────────────────────────────

# Raw 16:9 cuts deliberately carry no platform packaging (PRD #95: judge the moment
# *selection*, not the Aufmachung — no Packaging-Confound).
RENDER_PLATFORM = "general"

# The frontend job-id namespace for an arm render: ``archetype_retest_<vod>_<A|B>``.
# Keeps A and B in collision-free job dirs / ``/static`` URLs, and matches the static
# route's job-id charset (alnum + ``-_``), so the existing viewer serves them unchanged.
ARM_JOB_PREFIX = "archetype_retest"

# ``vod_id -> the cached video.mp4 path, or None`` when the VOD has no local video — a
# surfaced render gap, not a silent drop. The real adapter resolves the download cache
# (``cached_twitch_video``); tests inject a vod→path map.
ResolveVideoFn = Callable[[str], "pathlib.Path | None"]

# ``video_path, start, end, output_path -> None`` — cut one raw 16:9 segment. The
# real adapter is ffmpeg (``services.clipper.cut_raw_segment``); tests inject a
# recording fake, so the renderer is exercised ffmpeg-free.
CutSegmentFn = Callable[[pathlib.Path, float, float, pathlib.Path], None]

# ``JobStatus -> None`` — the frontend-registry sink. The real adapter upserts the
# job into ``jobs.json`` (``upsert_jobs_registry``); tests collect into a list, so
# registration is verifiable without touching the on-disk registry.
RegisterJobFn = Callable[[JobStatus], None]


def _clip_result(moment: dict, *, filename: str, job_id: str, start: float, end: float) -> ClipResult:
    """A ClipResult for one cut moment — the same fields the serving cutter emits,
    so the existing viewer renders an arm clip exactly like a pipeline clip; only the
    ``download_url`` points at this arm's own ``/static/<job_id>/clips/`` namespace."""
    return ClipResult(
        candidate_id=str(moment.get("candidate_id") or filename),
        filename=filename,
        download_url=f"/static/{job_id}/clips/{filename}",
        start_time=start,
        end_time=end,
        viral_score=float(moment.get("viral_score") or 1.0),
        reason=str(moment.get("reason") or ""),
        selection_reasons=list(moment.get("selection_reasons") or []),
        ranking_mode=moment.get("ranking_mode", "llm_moment_finder"),
        platform=moment.get("platform", RENDER_PLATFORM),
        variant_key=str(moment.get("variant_key") or "balanced"),
        variant_label=moment.get("variant_label"),
        projected_hook_score=moment.get("projected_hook_score"),
        projected_retention_score=moment.get("projected_retention_score"),
        composite_score=moment.get("composite_score"),
        value_score=moment.get("value_score"),
        trend_score=moment.get("trend_score"),
    )


def render_arm_clips(
    video_path: pathlib.Path,
    top20: list[dict],
    job_id: str,
    clips_dir: pathlib.Path,
    *,
    cut_segment: CutSegmentFn,
) -> list[ClipResult]:
    """Cut a gated top-20 as raw 16:9 clips into ``clips_dir/clip_NN.mp4`` → ClipResults.

    ``top20`` is exactly ``replay_postprocess``'s output (the served set the recall
    measured). ``moments_from_post_processed`` gives it the SAME final-moment shape the
    serving pipeline writes (candidate_id ``llm_moment_NN``, viral/composite scores,
    reason). Each moment NN is cut at its EXACT window by the injected ``cut_segment``
    leaf-op — reading the cached video IN PLACE, no reframe/packaging, no copy — and gets
    a ClipResult whose ``download_url`` is ``/static/<job_id>/clips/clip_NN.mp4``, the
    exact shape the existing static route + job/clip viewer already serve.
    """
    moments = moments_from_post_processed(list(top20), RENDER_PLATFORM)
    clips: list[ClipResult] = []
    for rank, moment in enumerate(moments, start=1):
        filename = f"clip_{rank:02d}.mp4"
        start = float(moment["start_time"])
        end = float(moment["end_time"])
        cut_segment(video_path, start, end, clips_dir / filename)
        clips.append(_clip_result(moment, filename=filename, job_id=job_id, start=start, end=end))
    return clips


def register_arm_job(
    job_id: str,
    top20: list[dict],
    clips: list[ClipResult],
    *,
    store,
    register_job: RegisterJobFn,
    profile: str,
) -> JobStatus:
    """Register one (VOD, arm) as a frontend job: write ``final_moments`` + the JobStatus.

    Writes ``final_moments`` = ``{"moments": <the gated top-20, serving-shaped>, "count"}``
    via the store — the SAME artifact + shape the diagnostics endpoint already reads — and
    registers a done JobStatus carrying the ClipResults through the injected ``register_job``
    sink (the real one upserts ``jobs.json``). No cut happens here (the clips are passed in),
    so registration is verifiable without rendering (#99 AC). ``profile`` labels the arm so
    A and B are distinguishable in the viewer.
    """
    moments = moments_from_post_processed(list(top20), RENDER_PLATFORM)
    store.write_final_moments(job_id, {"moments": moments, "count": len(moments)})
    job = JobStatus(
        job_id=job_id, status="done", progress_pct=100, profile=profile, clips=list(clips)
    )
    register_job(job)
    return job


def arm_job_id(vod_id: str, arm: str) -> str:
    """The collision-free frontend job id for a (VOD, arm): ``archetype_retest_<vod>_<A|B>``."""
    return f"{ARM_JOB_PREFIX}_{vod_id}_{arm}"


def render_and_register_arm(
    vod_id: str,
    arm: str,
    top20: list[dict],
    *,
    storage_dir: pathlib.Path,
    store,
    resolve_video: ResolveVideoFn,
    cut_segment: CutSegmentFn,
    register_job: RegisterJobFn,
) -> dict:
    """Render one (VOD, arm)'s gated top-20 as raw cuts and register the frontend job.

    Resolves the VOD's cached ``video.mp4``; a VOD with NO local video is a SURFACED render
    gap (``{"rendered": False, "reason": "no_local_video", …}``) — never silently dropped
    (#99 AC). Otherwise it cuts the top-20 into ``storage_dir/<arm_job_id>/clips/`` and
    registers the done job (final_moments + ClipResults). ``arm`` (``A``/``B``) gives A and B
    distinct ``arm_job_id``s, clip dirs, and ``/static`` URLs, so the two never collide.
    Returns a per-(VOD, arm) outcome row the orchestrator surfaces (rendered or gap).
    """
    job_id = arm_job_id(vod_id, arm)
    video = resolve_video(vod_id)
    if video is None:
        return {"vod_id": vod_id, "arm": arm, "job_id": job_id,
                "rendered": False, "reason": "no_local_video", "clip_count": 0}
    clips_dir = pathlib.Path(storage_dir) / job_id / "clips"
    clips = render_arm_clips(video, top20, job_id, clips_dir, cut_segment=cut_segment)
    register_arm_job(job_id, top20, clips, store=store,
                     register_job=register_job, profile=f"{ARM_JOB_PREFIX}_{arm}")
    return {"vod_id": vod_id, "arm": arm, "job_id": job_id,
            "rendered": True, "reason": None, "clip_count": len(clips)}


def render_both_arms(
    final_by_vod_a: dict[str, list[dict]],
    final_by_vod_b: dict[str, list[dict]],
    *,
    storage_dir: pathlib.Path,
    store,
    resolve_video: ResolveVideoFn,
    cut_segment: CutSegmentFn,
    register_job: RegisterJobFn,
) -> dict:
    """Render BOTH arms' gated top-20s as raw cuts → one frontend job per (VOD, arm).

    ``final_by_vod_a`` / ``final_by_vod_b`` are the served top-20 maps from the two arms
    (the verdict's ``arm_a["final_by_vod"]`` / ``arm_b["final_by_vod"]``) — exactly what the
    recall measured. For each VOD an arm served, it renders + registers that arm (``_A`` /
    ``_B``), so the user sees A and B side by side in the existing viewer. VODs with no local
    video are SURFACED render gaps, never silent drops. Returns the per-(VOD, arm) outcomes
    plus the registered job ids and the surfaced render-gap job ids.
    """
    outcomes: list[dict] = []
    for arm, final_by_vod in (("A", final_by_vod_a), ("B", final_by_vod_b)):
        for vod_id, top20 in final_by_vod.items():
            outcomes.append(render_and_register_arm(
                vod_id, arm, top20,
                storage_dir=storage_dir, store=store, resolve_video=resolve_video,
                cut_segment=cut_segment, register_job=register_job))
    return {
        "outcomes": outcomes,
        "registered_job_ids": [o["job_id"] for o in outcomes if o["rendered"]],
        "render_gap_jobs": [o["job_id"] for o in outcomes if not o["rendered"]],
    }


def render_verdict(
    verdict: dict,
    *,
    storage_dir: pathlib.Path,
    store,
    resolve_video: ResolveVideoFn,
    cut_segment: CutSegmentFn,
    register_job: RegisterJobFn,
) -> dict:
    """Render BOTH arms straight from a verdict (paid or replay) → one job per (VOD, arm).

    Reads each arm's ``final_by_vod`` — the served top-20 the recall measured — so what the
    user reviews in the existing job/clip viewer is byte-identical to what the verdict
    scored (PRD #95 User Story 19). A thin read over ``run_verdict``'s output; the rendering,
    collision-free arm dirs, and surfaced no-video gaps are all ``render_both_arms``.
    """
    return render_both_arms(
        verdict["arm_a"]["final_by_vod"], verdict["arm_b"]["final_by_vod"],
        storage_dir=storage_dir, store=store, resolve_video=resolve_video,
        cut_segment=cut_segment, register_job=register_job,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Real adapters (#99) — the ffmpeg cutter wrapper, the jobs.json registry sink, and
# the cached-video resolver the offline driver binds (the two-adapter "real" side).
# ─────────────────────────────────────────────────────────────────────────────

def upsert_jobs_registry(registry_path: pathlib.Path, job: JobStatus) -> None:
    """Merge ``job`` into the ``jobs.json`` registry by job_id (the real register sink).

    Reads the existing registry list (a missing or unparseable file starts empty — the
    same tolerance ``job_manager._load_jobs`` applies), replaces any entry with the same
    job_id, appends otherwise, and atomically rewrites (tempfile + ``os.replace``, mirroring
    ``job_manager._persist_jobs``). Merging — not overwriting — means the offline render
    leaves a running server's live jobs intact; on its next mtime reload the server picks
    the arm jobs up and the existing job/clip viewer serves them.
    """
    registry_path = pathlib.Path(registry_path)
    existing: list[dict] = []
    if registry_path.exists():
        try:
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = [row for row in loaded if isinstance(row, dict)]
        except (OSError, json.JSONDecodeError):
            existing = []
    row = job.model_dump(mode="json")
    merged = [r for r in existing if r.get("job_id") != job.job_id]
    merged.append(row)

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_path.parent / f"{registry_path.name}.{os.getpid()}.tmp"
    tmp_path.write_text(json.dumps(merged, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp_path, registry_path)


def cached_twitch_video(vod_id: str, storage_dir: pathlib.Path) -> "pathlib.Path | None":
    """The cached ``video.mp4`` for a Twitch VOD, or ``None`` when there is no local copy.

    Resolves the download cache the serving pipeline writes
    (``<storage_dir>/_download_cache/twitch/vod_<vod_id>.mp4`` — the layout
    ``downloader._cache_path_for_url`` owns). ``None`` (no local video) is a SURFACED render
    gap (the 6 VODs without a download are reported, not re-fetched), never a silent drop.
    """
    path = pathlib.Path(storage_dir) / "_download_cache" / "twitch" / f"vod_{vod_id}.mp4"
    return path if path.exists() else None


def staged_video_resolver(video_source_root: pathlib.Path) -> ResolveVideoFn:
    """Resolve ``vod_<id>.mp4`` from a source root instead of the download cache.

    The anchor videos live in their ``pipeline_debug_*`` dirs, surfaced via the #100
    staging root (``transcribe_only_sources/vod_<id>.mp4`` — relative symlinks). The
    render reads through those in place (``Path.exists`` follows the symlink, ffmpeg cuts
    the real file). A VOD with no staged source surfaces as ``None`` — a render gap, not a
    silent drop."""
    root = pathlib.Path(video_source_root)

    def _resolve(vod_id: str) -> "pathlib.Path | None":
        path = root / f"vod_{vod_id}.mp4"
        return path if path.exists() else None

    return _resolve


# ─────────────────────────────────────────────────────────────────────────────
# Issue #101 — the copy-paste CLI: `run` (gate + paid + persist), `replay` (free
# verdict), `render` (both arms → frontend jobs). The runbook drives these; the heavy
# leaves (Gemini, ffmpeg, the jobs.json sink) are injected so unit tests stay net-free.
# ─────────────────────────────────────────────────────────────────────────────

# The artifacts root holding the two arms' persisted raw responses, under the storage
# dir so `run` writes them and a later `replay`/`render` re-reads the same files.
DEFAULT_ARTIFACTS_SUBDIR = "archetype_judge_retest"


def build_fresh_consistency_report(
    storage: pathlib.Path, anchor_ids: list[str]
) -> "transcribe_only_rerun.ConsistencyReport":
    """Re-derive the #100 consistency report from the fresh transcripts on disk.

    A VOD is *completed* (re-transcribed) when its fresh ``segments.jsonl`` exists; the
    rest are *excluded* (no local video — the documented 6), carried through so they are
    surfaced, never silently judged. The report over the completed VODs is the same
    fail-closed verdict #100 reads — the pre-condition the paid run gates on.
    """
    storage = pathlib.Path(storage)
    completed = [
        v for v in anchor_ids
        if transcribe_only_rerun.fresh_segments_path(storage, v).exists()
    ]
    excluded = [v for v in anchor_ids if v not in completed]
    per_vod = transcribe_only_rerun.load_report_inputs(storage, completed)
    return transcribe_only_rerun.build_consistency_report(per_vod, excluded=excluded)


def _default_render_adapters(storage: pathlib.Path):
    """The real render side, imported lazily (ffmpeg + the prod store are Docker-only)."""
    from services.clipper import cut_raw_segment
    from services.job_artifact_store import RealFsJobArtifactStore

    store = RealFsJobArtifactStore(storage)

    def register(job: JobStatus) -> None:
        upsert_jobs_registry(store.jobs_registry_path(), job)

    return store, cut_raw_segment, register


def main(
    argv: "list[str] | None" = None,
    *,
    provider: "CallProviderFn | None" = None,
    cut_segment: "CutSegmentFn | None" = None,
    register_job: "RegisterJobFn | None" = None,
    store=None,
) -> int:
    """The archetype-judge re-test driver — ``run`` / ``replay`` / ``render`` (PRD #95 #101).

    ``run`` gates on the fresh consistency report, then judges both arms over the
    re-transcribed segments and persists each raw (``--only`` restricts to one VOD — the
    1-VOD smoke). ``replay`` reproduces the verdict free from the persisted raws. ``render``
    re-reads them and renders both arms' gated top-20s into frontend jobs. The verdict table
    is read off (PASS → ARCHETYPE_JUDGE_LIVES, NO-GO → RANKING_TRACK_CLOSED). Returns 0 on a
    read verdict / completed render; non-zero only when the consistency gate aborts the run.
    """
    import argparse

    from ml import eval_clip_finder

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("run", "replay", "render"))
    parser.add_argument("--storage", required=True,
                        help="The storage dir: candidate pools + fresh #100 transcripts + arm dirs")
    parser.add_argument("--artifacts-root", default=None,
                        help="Where the two arms' raw responses persist "
                             "(default <storage>/" + DEFAULT_ARTIFACTS_SUBDIR + ")")
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--only", default=None,
                        help="Restrict to one VOD id — the 1-VOD smoke (both arms)")
    parser.add_argument("--exclude-vods", default=None,
                        help="Comma-separated VOD ids to surface as coverage gaps (never "
                             "judged) — e.g. pools too large for one reliable judge call")
    parser.add_argument("--video-source-root", default=None,
                        help="render: resolve vod_<id>.mp4 from here (e.g. the #100 staging "
                             "root) instead of the download cache")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)

    storage = pathlib.Path(args.storage)
    artifacts_root = (pathlib.Path(args.artifacts_root) if args.artifacts_root
                      else storage / DEFAULT_ARTIFACTS_SUBDIR)
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))
    if args.only:
        anchors = [v for v in anchors if v == args.only]
    exclude_vods = [v.strip() for v in (args.exclude_vods or "").split(",") if v.strip()]

    common = dict(storage=storage, predictions_dir=storage,
                  artifacts_root=artifacts_root, top_k=args.top_k,
                  exclude_vods=exclude_vods)

    if args.mode == "run":
        report = build_fresh_consistency_report(storage, anchors)
        try:
            verdict = run_paid_verdict(
                anchors, manifest, report=report, provider=provider, **common)
        except TranscriptConsistencyError as exc:
            print(f"ABORTED — paid run refused: {exc}")
            return 1
        print(format_verdict_table(verdict))
        return 0

    if args.mode == "replay":
        verdict = run_replay_verdict(anchors, manifest, **common)
        print(format_verdict_table(verdict))
        return 0

    # render: reconstruct both arms' served top-20s free (replay), then cut + register.
    verdict = run_replay_verdict(anchors, manifest, **common)
    if store is None or cut_segment is None or register_job is None:
        d_store, d_cut, d_register = _default_render_adapters(storage)
        store = store or d_store
        cut_segment = cut_segment or d_cut
        register_job = register_job or d_register
    resolve_video = (staged_video_resolver(pathlib.Path(args.video_source_root))
                     if args.video_source_root
                     else (lambda vod_id: cached_twitch_video(vod_id, storage)))
    summary = render_verdict(
        verdict, storage_dir=storage, store=store,
        resolve_video=resolve_video,
        cut_segment=cut_segment, register_job=register_job)
    print(f"rendered {len(summary['registered_job_ids'])} arm jobs; "
          f"{len(summary['render_gap_jobs'])} render gaps (no local video)")
    for job_id in summary["registered_job_ids"]:
        print(f"  registered: {job_id}")
    for job_id in summary["render_gap_jobs"]:
        print(f"  render gap (no local video): {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
