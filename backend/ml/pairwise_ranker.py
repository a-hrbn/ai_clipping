"""Within-VOD pairwise numpy ranker + from-scratch PU dataset — PRD #51 Slice A.

The cheap null-test PRD #46 never ran: does a **nonlinear interaction** among the
existing serving features lift the buried-real gold that the linear composite sweep
could not see? This module builds, from scratch, the dataset and the model to answer
it, and plugs the model into the #52 grouped-by-VOD OOF runner so the answer is read
out-of-fold against the pre-fixed A-Gate.

**Dataset (from scratch, not the old corpus).** The training unit is a candidate in
its VOD's own deduped pool. A candidate is a **positive** iff it is a #49 *real*
candidate (clip-sized, gold-proportioned — ``gap_decomposition.is_real_candidate``)
that *covers* a verified gold (``eval_clip_finder.is_hit``); an accidental/oversized
cover is **not** a positive. The rest of the pool is **unlabeled** (PU — no synthetic
hard-negatives). Headline positives (gold views >= 100k) are up-weighted. VODs with no
real positive are dropped from training but kept in the eval denominator. The circular
``jynxzi_training_rows.jsonl`` is deliberately **not** read — the dataset is built
purely from the replayed pools.

**Model.** A dependency-free numpy pairwise logistic with explicit interaction terms
(``value*duration_bucket``, ``value*trend``), L1-regularized; pairs are formed **only
within a VOD** (a positive ranked above its pool siblings). The artifact is a diffable
JSON weight vector + feature list + ``feature_build_version`` — no pickle/joblib. The
same ``features -> score`` function is plugged in as the offline ``rank_score_fn``.

Two adapters (Two-Adapter discipline): a **real** one that trains/loads the JSON and a
deterministic, network-free **fake** one. No paid LLM call anywhere.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import sys
from typing import Sequence

import numpy as np

from ml import directional_eval, eval_clip_finder, gap_decomposition, replay_harness

# Deliberately NOT a training source. ``jynxzi_training_rows.jsonl`` mixed the
# serving finder's own outputs back into the labels — a circular corpus that teaches
# the ranker to agree with itself. The from-scratch dataset here is built only from
# the replayed pools + the GT-flywheel gold; a ``*.DEPRECATED.md`` marker sits beside
# the file in ``ml/artifacts``. Surfaced as a constant so the exclusion is explicit.
DEPRECATED_CIRCULAR_TRAINING_ROWS = "jynxzi_training_rows.jsonl"

# The two PU labels. Negatives are never minted synthetically: every non-positive
# pool candidate is *unlabeled*, the defining move of PU learning.
POSITIVE = "positive"
UNLABELED = "unlabeled"

# Sample weights. Unlabeled candidates and sub-headline positives carry the base
# weight; a positive that covers a headline gold (views >= the headline floor) is
# up-weighted so the ranker pays more for burying a proven viral repost.
BASE_WEIGHT = 1.0
DEFAULT_HEADLINE_WEIGHT = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Features — row-only, so the same φ runs at train and at the rank_score_fn seam
# ─────────────────────────────────────────────────────────────────────────────

# The model artifact pins this so a stale weight vector (built against a different
# feature layout) is caught instead of silently mis-scored.
FEATURE_BUILD_VERSION = 1

# Duration normaliser (≈1.0 for a 60s clip) and the bucket boundaries the
# value×duration_bucket interaction reads. A short/medium/long ordinal is exactly the
# nonlinearity a linear sweep over {hook,flow,value,trend} cannot represent: it lets
# `value` count differently depending on clip length.
_DURATION_NORM_S = 60.0
_DURATION_BUCKETS_S = (20.0, 45.0)

# The ordered feature names. φ is row-derivable only (hook/flow/value/trend + the
# window): the offline rank_score_fn seam never sees the chat/voice segments, so
# keeping φ row-only is what guarantees train/serve parity.
FEATURES: tuple[str, ...] = (
    "hook", "flow", "value", "trend", "dur_n",
    "value_x_durbucket", "value_x_trend",
)


def _duration(row: dict) -> float:
    return float(row.get("end_time") or 0.0) - float(row.get("start_time") or 0.0)


def duration_bucket(duration_s: float) -> int:
    """Short/medium/long ordinal (0/1/2) the value×duration interaction multiplies.

    Boundaries at ``_DURATION_BUCKETS_S``: <20s → 0, 20–45s → 1, ≥45s → 2.
    """
    short, long = _DURATION_BUCKETS_S
    if duration_s < short:
        return 0
    return 1 if duration_s < long else 2


def feature_row(row: dict) -> dict[str, float]:
    """Build the row-only feature dict (the four LLM axes + duration + interactions).

    ``value_x_durbucket`` = value × the duration-bucket ordinal and ``value_x_trend``
    = value × trend are the explicit nonlinear interaction terms; the rest are the
    serving axes and a normalised duration. Missing fields read as 0.0.
    """
    hook = float(row.get("hook") or 0.0)
    flow = float(row.get("flow") or 0.0)
    value = float(row.get("value") or 0.0)
    trend = float(row.get("trend") or 0.0)
    dur = _duration(row)
    return {
        "hook": hook,
        "flow": flow,
        "value": value,
        "trend": trend,
        "dur_n": dur / _DURATION_NORM_S,
        "value_x_durbucket": value * duration_bucket(dur),
        "value_x_trend": value * trend,
    }


def feature_vector(row: dict) -> list[float]:
    """The feature dict flattened to a list ordered by ``FEATURES`` (the model's φ)."""
    feats = feature_row(row)
    return [feats[name] for name in FEATURES]


def score_features(weights: list[float], row: dict) -> float:
    """The model score for one row: w · φ(row). Higher = better candidate."""
    return float(np.dot(np.asarray(weights, dtype=float), feature_vector(row)))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — the from-scratch PU corpus, built within each VOD's own pool
# ─────────────────────────────────────────────────────────────────────────────

def label_pool(
    gold_moments: list[dict],
    pool: list[dict],
    *,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
) -> list[dict]:
    """Label each deduped-pool candidate as a real-only positive or unlabeled (PU).

    A candidate is a **positive** iff it is a #49 *real* candidate (clip-sized,
    gold-proportioned) that covers at least one gold moment; an oversized/random
    window that merely overlaps a gold (accidental) and a candidate that covers no
    gold are both **unlabeled**. Each label carries ``gold_views`` — the max views
    over the gold a positive really-covers (``None`` for unlabeled) — so the headline
    up-weighting can read it without re-deriving the match. Order mirrors ``pool``.
    """
    labeled: list[dict] = []
    for cand in pool:
        covered_views = [
            float(gold.get("views") or 0.0)
            for gold in gold_moments
            if eval_clip_finder.is_hit(gold, cand)
            and gap_decomposition.is_real_candidate(gold, cand, thresholds)
        ]
        if covered_views:
            labeled.append({"label": POSITIVE, "row": cand, "gold_views": max(covered_views)})
        else:
            labeled.append({"label": UNLABELED, "row": cand, "gold_views": None})
    return labeled


def build_pu_dataset(
    pool_by_vod: dict[str, list[dict]],
    gold_by_vod: dict[str, list[dict]],
    *,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    headline_weight: float = DEFAULT_HEADLINE_WEIGHT,
    base_weight: float = BASE_WEIGHT,
) -> dict:
    """Label every VOD's pool into the from-scratch PU dataset, with sample weights.

    For each VOD the deduped pool is labelled (``label_pool``): real-only covering
    candidates are positives, the rest is unlabeled. A positive that covers a
    **headline** gold (``gold_views >= headline_min_views``) gets ``headline_weight``;
    every other candidate gets ``base_weight``. A VOD with no positive is **train-
    dropped** (it cannot form a within-VOD pair) but its labelled pool is **kept** in
    ``vods`` so the eval denominator still sees it. Returns ``{"vods", "train_vods",
    "train_dropped_vods", "headline_min_views"}``.
    """
    vods: dict[str, list[dict]] = {}
    train_vods: list[str] = []
    train_dropped_vods: list[str] = []
    for vod_id, pool in pool_by_vod.items():
        labelled = label_pool(gold_by_vod.get(vod_id, []), pool, thresholds=thresholds)
        for cand in labelled:
            headline = (
                cand["label"] == POSITIVE
                and cand["gold_views"] is not None
                and cand["gold_views"] >= headline_min_views
            )
            cand["weight"] = headline_weight if headline else base_weight
        vods[vod_id] = labelled
        if any(cand["label"] == POSITIVE for cand in labelled):
            train_vods.append(vod_id)
        else:
            train_dropped_vods.append(vod_id)
    return {
        "vods": vods,
        "train_vods": train_vods,
        "train_dropped_vods": train_dropped_vods,
        "headline_min_views": headline_min_views,
    }


def within_vod_pairs(dataset: dict) -> list[dict]:
    """Form the pairwise comparisons: a positive ranked above each pool sibling.

    Within **one** VOD, every positive is paired with every non-positive (unlabeled)
    sibling — the PU comparison "the real gold-covering candidate should outrank its
    deduped pool siblings". Pairs never cross a VOD boundary (so the model never
    learns a cross-VOD scale it could not see at serving), and a VOD with no positive
    contributes none (train-drop). Each pair carries the **positive's** weight, so a
    buried headline repost costs more. Returns ``[{vod_id, pos, neg, weight}]``.
    """
    pairs: list[dict] = []
    for vod_id, labelled in dataset["vods"].items():
        positives = [c for c in labelled if c["label"] == POSITIVE]
        siblings = [c for c in labelled if c["label"] != POSITIVE]
        for pos in positives:
            for neg in siblings:
                pairs.append({"vod_id": vod_id, "pos": pos, "neg": neg, "weight": pos["weight"]})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Model — dependency-free numpy pairwise logistic, L1-regularised
# ─────────────────────────────────────────────────────────────────────────────

# Defaults for the proximal-gradient fit. Zeros init + full-batch descent makes the
# whole train deterministic — no seed, reproducible run-to-run for the OOF baseline.
DEFAULT_L1 = 0.01
DEFAULT_LR = 0.5
DEFAULT_EPOCHS = 600


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_pairwise_logistic(
    pairs: list[dict],
    *,
    l1: float = DEFAULT_L1,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
) -> list[float]:
    """Fit a within-VOD pairwise logistic ranker by L1-proximal gradient descent.

    Each pair contributes the feature difference ``Δ = φ(pos) − φ(neg)``; the model
    maximises ``σ(w·Δ)`` (the positive should outrank its sibling), weighted by the
    pair weight. Optimised by full-batch gradient descent on the smooth logistic loss
    followed by a soft-threshold proximal step that applies the L1 penalty — giving
    *exact* zeros, the sparsity L1 is chosen for. Init is the zero vector, so the fit
    is fully deterministic (no seed). Returns the weights aligned to ``FEATURES``.
    """
    n_features = len(FEATURES)
    if not pairs:
        return [0.0] * n_features

    diffs = np.array(
        [np.subtract(feature_vector(p["pos"]["row"]), feature_vector(p["neg"]["row"]))
         for p in pairs],
        dtype=float,
    )
    sample_w = np.array([float(p.get("weight", 1.0)) for p in pairs], dtype=float)
    total_w = sample_w.sum() or 1.0

    weights = np.zeros(n_features, dtype=float)
    for _ in range(epochs):
        # dL/dw = Σ w_i (σ(w·Δ_i) − 1) Δ_i   (mean over total sample weight)
        grad = (sample_w * (_sigmoid(diffs @ weights) - 1.0)) @ diffs / total_w
        weights = weights - lr * grad
        # Proximal L1 soft-threshold — shrinks toward 0, clamps small weights to 0.
        shrink = lr * l1
        weights = np.sign(weights) * np.maximum(np.abs(weights) - shrink, 0.0)
    return [float(w) for w in weights]


@dataclasses.dataclass(frozen=True)
class PairwiseModel:
    """The serialisable ranker: a weight vector over ``features`` at a build version.

    Deliberately a plain JSON document (``to_dict``/``from_dict``) — a diffable weight
    vector + feature list + ``feature_build_version``, never a pickle/joblib blob — so
    a model change shows up as a readable diff and a stale artifact is caught by the
    version, not silently mis-scored. ``score`` recomputes φ in the artifact's own
    feature order, so it stays correct even if the module's ``FEATURES`` later grows.
    """

    weights: list[float]
    features: tuple[str, ...] = FEATURES
    feature_build_version: int = FEATURE_BUILD_VERSION

    def score(self, row: dict) -> float:
        feats = feature_row(row)
        return float(np.dot(
            np.asarray(self.weights, dtype=float),
            [feats[name] for name in self.features],
        ))

    def to_dict(self) -> dict:
        return {
            "feature_build_version": self.feature_build_version,
            "features": list(self.features),
            "weights": [float(w) for w in self.weights],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PairwiseModel":
        return cls(
            weights=[float(w) for w in data["weights"]],
            features=tuple(data.get("features", FEATURES)),
            feature_build_version=int(data.get("feature_build_version", FEATURE_BUILD_VERSION)),
        )


def rank_score_fn(model: PairwiseModel):
    """Adapt a model into the offline ``rank_score_fn`` seam (rows → rescored rows).

    Returns a function that overwrites each deduped row's ``composite_score`` with
    ``model.score(row)`` (on a copy — the input rows are not mutated), so the
    downstream serving diversity + ranking honour the learned order. This is the one
    ``features → score`` function the issue requires: the same φ the dataset trained
    on, plugged straight into the replay harness / #52 OOF runner.
    """
    def fn(rows: list[dict]) -> list[dict]:
        return [dict(row, composite_score=model.score(row)) for row in rows]

    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Adapters — Two-Adapter discipline: a deterministic Fake, a real trainer (below)
# ─────────────────────────────────────────────────────────────────────────────

# Canned weights for the Fake adapter: it leans on `value` and the value×duration
# interaction so its scoring visibly differs from identity, without any fit. Aligned
# to FEATURES = (hook, flow, value, trend, dur_n, value_x_durbucket, value_x_trend).
FAKE_WEIGHTS: list[float] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0]
FAKE_MODEL = PairwiseModel(weights=FAKE_WEIGHTS)


def fake_train_fn(_train_vods) -> "object":
    """A deterministic, network-free #52 ``TrainFn`` — the canned Fake adapter.

    Ignores the training VOD ids and returns a ``rank_score_fn`` backed by the fixed
    ``FAKE_MODEL``. No disk, no network, no fit — so it is reproducible run-to-run and
    safe to drive the OOF plumbing in a bare test env (mirrors the Fake-adapter
    discipline of the producer tests and the #52 identity/reorder stubs).
    """
    return rank_score_fn(FAKE_MODEL)


def train_model_from_pools(
    pool_by_vod: dict[str, list[dict]],
    gold_by_vod: dict[str, list[dict]],
    *,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    headline_weight: float = DEFAULT_HEADLINE_WEIGHT,
    l1: float = DEFAULT_L1,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
) -> PairwiseModel:
    """Build the PU dataset from the deduped pools and fit the within-VOD ranker.

    The from-scratch path end to end: ``build_pu_dataset`` → ``within_vod_pairs`` →
    ``train_pairwise_logistic``, returning the serialisable ``PairwiseModel``. The
    pools are the *only* training source — the circular ``jynxzi_training_rows.jsonl``
    is never read.
    """
    dataset = build_pu_dataset(
        pool_by_vod, gold_by_vod,
        thresholds=thresholds, headline_min_views=headline_min_views,
        headline_weight=headline_weight)
    pairs = within_vod_pairs(dataset)
    weights = train_pairwise_logistic(pairs, l1=l1, lr=lr, epochs=epochs)
    return PairwiseModel(weights=weights)


def make_oof_train_fn(
    storage: pathlib.Path,
    manifest: dict[str, list[dict]],
    *,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    headline_weight: float = DEFAULT_HEADLINE_WEIGHT,
    l1: float = DEFAULT_L1,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
):
    """The Real adapter as a #52 ``TrainFn``: replay the fold's pools, train, score.

    Returns ``train_fn(train_vods) -> rank_score_fn``. For each training VOD it replays
    the **identity** (composite) deduped pool — the candidate currency the dataset is
    built on — joins it with the VOD's gold from ``manifest``, fits the within-VOD
    pairwise model over only those fold VODs, and adapts it into the offline
    ``rank_score_fn`` seam. The OOF runner calls this per fold on the VODs that fold did
    not hold out, so the model never sees the VOD it later ranks (leakage-free). No
    paid call: it re-reads the same replay artifacts the rest of the eval uses.
    """
    gold = directional_eval.gold_manifest(manifest, gold_tiers)

    def train_fn(train_vods: Sequence[str]):
        pool_by_vod: dict[str, list[dict]] = {}
        gold_by_vod: dict[str, list[dict]] = {}
        for vod_id in train_vods:
            loaded = replay_harness.load_replay_inputs(vod_id, storage)
            if loaded is None:
                continue  # a coverage-gap VOD contributes no training rows
            raw_rows, segments = loaded
            pool_by_vod[vod_id] = replay_harness.replay_ranked_pool(raw_rows, segments)
            gold_by_vod[vod_id] = gold.get(vod_id, [])
        model = train_model_from_pools(
            pool_by_vod, gold_by_vod,
            thresholds=thresholds, headline_min_views=headline_min_views,
            headline_weight=headline_weight, l1=l1, lr=lr, epochs=epochs)
        return rank_score_fn(model)

    return train_fn


# ─────────────────────────────────────────────────────────────────────────────
# The A-Gate — read the #52 two-metric delta against the pre-fixed ship rule
# ─────────────────────────────────────────────────────────────────────────────

SHIP = "ship"
PROBE = "→ Probe"

# The pre-fixed cuts, set BEFORE the run so the read is mechanical, never
# rationalised after the fact. ship iff the engineering clean_but_buried recall@20
# rises by a **majority** of the re-rankable set (> 0.5) AND the floor recall@20
# guardrail does not fall AND avg_rank does not explode.
DEFAULT_MAJORITY_CBB_DELTA = 0.5
DEFAULT_MAX_AVG_RANK_INCREASE = 5.0


def evaluate_a_gate(
    two_metric_delta: dict,
    *,
    majority_cbb_delta: float = DEFAULT_MAJORITY_CBB_DELTA,
    max_avg_rank_increase: float = DEFAULT_MAX_AVG_RANK_INCREASE,
    floor_tolerance: float = 1e-9,
) -> dict:
    """Read the A-Gate verdict (ship vs → Probe) off the #52 two-metric delta.

    Three pre-fixed conditions, all required for **ship**:

      * **cbb_majority** — the engineering ``clean_but_buried`` recall@20 (floor tier,
        the full re-rankable set) rises by more than ``majority_cbb_delta`` (a majority
        of the re-rankable gold lifted). This is the *claim*, so it fails closed: a
        missing delta is not a pass.
      * **floor_not_falling** — the ``guardrail_floor_recall`` (full-denominator floor
        recall@20) does not fall. A guardrail: it only vetoes on evidence of a drop, so
        a missing delta does not veto.
      * **avg_rank_not_exploding** — the floor ``avg_rank`` does not increase by more
        than ``max_avg_rank_increase`` (the ranker did not push served hits deeper).
        Also a fail-open guardrail (you cannot explode from an undefined baseline).

    Returns the verdict, the three condition booleans, the cuts, and a one-line reason
    so the read stays explicit and auditable.
    """
    cbb = two_metric_delta["engineering_clean_but_buried_recall"]["floor"].get("delta")
    floor = two_metric_delta["guardrail_floor_recall"].get("delta")
    avg = two_metric_delta["avg_rank"]["floor"].get("delta")

    cbb_majority = cbb is not None and cbb > majority_cbb_delta
    floor_not_falling = floor is None or floor >= -floor_tolerance
    avg_rank_not_exploding = avg is None or avg <= max_avg_rank_increase

    ship = cbb_majority and floor_not_falling and avg_rank_not_exploding
    verdict = SHIP if ship else PROBE

    cbb_str = f"{cbb:+.3f}" if cbb is not None else "—"
    floor_str = f"{floor:+.3f}" if floor is not None else "—"
    avg_str = f"{avg:+.3f}" if avg is not None else "—"
    reason = (
        f"clean_but_buried recall@20 Δ {cbb_str} (> {majority_cbb_delta} majority? "
        f"{cbb_majority}) · floor recall@20 Δ {floor_str} (not falling? {floor_not_falling}) · "
        f"avg_rank Δ {avg_str} (≤ {max_avg_rank_increase} not exploding? {avg_rank_not_exploding}) "
        f"→ {verdict}"
    )
    return {
        "verdict": verdict,
        "ship": ship,
        "conditions": {
            "cbb_majority": cbb_majority,
            "floor_not_falling": floor_not_falling,
            "avg_rank_not_exploding": avg_rank_not_exploding,
        },
        "thresholds": {
            "majority_cbb_delta": majority_cbb_delta,
            "max_avg_rank_increase": max_avg_rank_increase,
        },
        "observed": {"cbb_floor_delta": cbb, "guardrail_floor_delta": floor, "avg_rank_floor_delta": avg},
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — Slice A end to end: real ranker → #52 OOF → A-Gate verdict
# ─────────────────────────────────────────────────────────────────────────────

def run_slice_a(
    anchor_ids: Sequence[str],
    storage: pathlib.Path,
    manifest: dict[str, list[dict]],
    frozen_baseline: dict,
    *,
    k: int = 5,
    top_k: int = replay_harness.DEFAULT_TOP_K,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
    headline_weight: float = DEFAULT_HEADLINE_WEIGHT,
    l1: float = DEFAULT_L1,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    shortlist_depth: int = gap_decomposition.DEFAULT_SHORTLIST_DEPTH,
    majority_cbb_delta: float = DEFAULT_MAJORITY_CBB_DELTA,
    max_avg_rank_increase: float = DEFAULT_MAX_AVG_RANK_INCREASE,
    on_fold=None,
) -> dict:
    """Fit the real within-VOD ranker, drive the #52 OOF runner, read the A-Gate.

    Builds the Real adapter (``make_oof_train_fn``) and hands it to the #52 grouped-by-
    VOD OOF runner as the ``train_fn`` — so the model ranks every eval VOD out-of-fold
    and the report carries the two-metric delta against the frozen composite baseline.
    The mechanically-read A-Gate verdict is attached under ``a_gate``. No paid call.
    """
    from ml import oof_eval_runner as oof  # lazy: keeps the module import light

    train_fn = make_oof_train_fn(
        storage, manifest, thresholds=thresholds, headline_min_views=headline_min_views,
        headline_weight=headline_weight, l1=l1, lr=lr, epochs=epochs)
    report = oof.build_report(
        anchor_ids, storage, manifest, frozen_baseline, train_fn=train_fn,
        k=k, top_k=top_k, ks=ks, headline_min_views=headline_min_views,
        thresholds=thresholds, shortlist_depth=shortlist_depth, on_fold=on_fold)
    report["a_gate"] = evaluate_a_gate(
        report["two_metric_delta"],
        majority_cbb_delta=majority_cbb_delta, max_avg_rank_increase=max_avg_rank_increase)
    report["model"] = {
        "feature_build_version": FEATURE_BUILD_VERSION,
        "features": list(FEATURES),
        "l1": l1, "lr": lr, "epochs": epochs, "headline_weight": headline_weight,
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Reporting + CLI
# ─────────────────────────────────────────────────────────────────────────────

def format_report(report: dict) -> str:
    """The #52 two-metric delta, then the A-Gate verdict read explicitly."""
    from ml import oof_eval_runner as oof  # lazy: shares the #52 formatter

    gate = report["a_gate"]
    mark = "✅" if gate["ship"] else "⛔"
    cond = gate["conditions"]
    return "\n".join([
        oof.format_report(report),
        "",
        "=" * 78,
        "A-GATE (Slice A null-test — within-VOD pairwise interaction ranker)",
        "=" * 78,
        f"  cbb recall@20 rises by a majority (> {gate['thresholds']['majority_cbb_delta']}): "
        f"{cond['cbb_majority']}",
        f"  floor recall@20 does not fall:                {cond['floor_not_falling']}",
        f"  avg_rank does not explode "
        f"(≤ {gate['thresholds']['max_avg_rank_increase']}):       {cond['avg_rank_not_exploding']}",
        "",
        f"  {mark}  {gate['verdict'].upper()}   —   {gate['reason']}",
        "",
        f"  model: pairwise-L1 over {report['model']['features']}",
        f"         feature_build_version={report['model']['feature_build_version']}  "
        f"l1={report['model']['l1']}  epochs={report['model']['epochs']}",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--baseline", default=str(gap_decomposition.DEFAULT_BASELINE),
                        help="Frozen composite baseline JSON the Ship/Guardrail gates delta against")
    parser.add_argument("--k", type=int, default=5, help="Number of grouped-by-VOD folds")
    parser.add_argument("--top-k", type=int, default=replay_harness.DEFAULT_TOP_K)
    parser.add_argument("--ks", default="5,10,20,50", help="Recall@K comma-separated")
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS)
    parser.add_argument("--l1", type=float, default=DEFAULT_L1)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--headline-weight", type=float, default=DEFAULT_HEADLINE_WEIGHT)
    parser.add_argument("--shortlist-depth", type=int, default=gap_decomposition.DEFAULT_SHORTLIST_DEPTH)
    parser.add_argument("--majority-cbb-delta", type=float, default=DEFAULT_MAJORITY_CBB_DELTA)
    parser.add_argument("--max-avg-rank-increase", type=float, default=DEFAULT_MAX_AVG_RANK_INCREASE)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))
    baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))

    report = run_slice_a(
        anchors, pathlib.Path(args.predictions_dir), manifest, baseline,
        k=args.k, top_k=args.top_k, ks=ks, headline_min_views=args.headline_min_views,
        l1=args.l1, epochs=args.epochs, headline_weight=args.headline_weight,
        shortlist_depth=args.shortlist_depth,
        majority_cbb_delta=args.majority_cbb_delta,
        max_avg_rank_increase=args.max_avg_rank_increase)
    if report["coverage_gap_vods"]:
        print(
            f"pairwise_ranker: {len(report['coverage_gap_vods'])}/{len(anchors)} anchor VODs have "
            f"no replayable artifacts — coverage gap: {', '.join(report['coverage_gap_vods'])}",
            file=sys.stderr)
    report["computed_at"] = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.output_json:
        print(payload)
    else:
        print(format_report(report))
        print(f"\ncomputed_at: {report['computed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
