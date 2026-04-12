## Trend Prior Data

`hook_patterns.json` contains a lightweight, hand-curated runtime prior for short-form clip selection.

Purpose:
- reward openings that resemble common high-performing short-form hooks
- reward topics that often correlate with clip-worthy moments
- stay cheap and deterministic; this is not a replacement for the ranker or judge

How it is used:
- `backend/services/trend_prior.py` loads the manifest
- hook phrases are matched against the opening of a candidate clip
- topic phrases are matched against the full candidate transcript window
- the service emits `trend_hook_score`, `trend_topic_score`, and `trend_prior_score`

Update process:
1. Add or remove phrases based on qualitative review, creator/domain needs, or offline eval results.
2. Keep weights in the `0.4` to `0.75` range unless there is strong evidence for a larger shift.
3. Prefer broad pattern groups over highly specific one-off phrases.
4. Validate changes with `tests.test_trend_prior`, `tests.test_feature_builder`, and `tests.test_ranker`.

Data source note:
- The seed phrases are hand-curated from common patterns seen in gaming, reaction, just-chatting, and podcast shorts.
- This file intentionally avoids scraping proprietary trend feeds or external copyrighted corpora.
