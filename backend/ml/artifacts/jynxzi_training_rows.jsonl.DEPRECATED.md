# DEPRECATED — `jynxzi_training_rows.jsonl` is a circular corpus, not a training source

This file (and its sibling `jynxzi_vod_training_rows.jsonl`) folded the **serving
finder's own outputs** back into the training labels. A model trained on it learns to
agree with the existing composite ranker rather than with the ground truth — a circular
signal that PRD #46's linear sweep could never escape.

**Do not use it as a training source.** PRD #51 Slice A (`ml/pairwise_ranker.py`) builds
its PU dataset **from scratch** instead:

- training unit = a candidate in its VOD's own deduped replay pool,
- **positive** = a #49 *real* (clip-sized, gold-proportioned) candidate that covers a
  GT-flywheel gold moment,
- the rest of the pool = **unlabeled** (PU; no synthetic hard-negatives).

The exclusion is pinned in code as
`ml.pairwise_ranker.DEPRECATED_CIRCULAR_TRAINING_ROWS` and proven by the
`FromScratchTests` (the dataset/model build performs no file I/O at all).

These rows may still be read by descriptive **audit** scripts
(`audit_chat_feature_coverage.py`, `inspect_background_negatives.py`,
`expand_eval_manifest.py`) — that is fine; those do not train a ranker.
