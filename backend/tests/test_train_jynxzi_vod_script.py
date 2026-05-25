from __future__ import annotations

import unittest
from pathlib import Path


class TrainJynxziVodScriptTests(unittest.TestCase):
    def test_full_vod_audio_is_not_limited_to_refresh_pass(self) -> None:
        script = Path(__file__).resolve().parents[1] / "ml" / "train_jynxzi_vod.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn(
            '"$([[ "$CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS" == "1" ]] && echo --full-vod-audio-fingerprints || echo --no-full-vod-audio-fingerprints)"',
            text,
        )
        self.assertNotIn(
            '"$([[ "$refresh_vods" == "1" && "$CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS" == "1" ]] && echo --full-vod-audio-fingerprints || echo --no-full-vod-audio-fingerprints)"',
            text,
        )

    def test_cross_platform_prepare_fails_on_blocked_status(self) -> None:
        script = Path(__file__).resolve().parents[1] / "ml" / "train_jynxzi_vod.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn("--fail-on-blocked", text)

    def test_cross_platform_gates_match_available_jynxzi_coverage(self) -> None:
        script = Path(__file__).resolve().parents[1] / "ml" / "train_jynxzi_vod.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn('CROSS_PLATFORM_MIN_MATCHED_ROWS="${CROSS_PLATFORM_MIN_MATCHED_ROWS:-70}"', text)
        self.assertIn('CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS="${CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS:-70}"', text)
        self.assertIn('CROSS_PLATFORM_MIN_MATCHED_VODS="${CROSS_PLATFORM_MIN_MATCHED_VODS:-20}"', text)
        self.assertIn('CROSS_PLATFORM_MIN_REPOSTS="${CROSS_PLATFORM_MIN_REPOSTS:-0}"', text)
        self.assertIn('VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS="${VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS:-70}"', text)

    def test_twitch_clip_labels_are_removed_from_training_flow(self) -> None:
        script = Path(__file__).resolve().parents[1] / "ml" / "train_jynxzi_vod.sh"
        text = script.read_text(encoding="utf-8")

        self.assertNotIn("TWITCH_CLIP_LABELS_ENABLED", text)
        self.assertNotIn("collect_twitch_clip_labels.py", text)
        self.assertIn("--full-vod-audio-vod-inventory", text)
        self.assertIn("--full-vod-audio-workers", text)
        self.assertIn('CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS="${CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS:-2}"', text)
        self.assertIn('CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS="${CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS:-20}"', text)
        self.assertIn('--transcript-match-workers "$CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS"', text)
        self.assertIn("Twitch clip-label collection removed", text)
        self.assertIn('"twitch_vod_api" in sources', text)
        self.assertIn('merged["inventory_sources"] = sorted(sources)', text)
        self.assertIn('for key in ("created_at", "title", "vod_duration_seconds"):', text)

    def test_frontend_gold_rows_are_disabled_by_default(self) -> None:
        script = Path(__file__).resolve().parents[1] / "ml" / "train_jynxzi_vod.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn('VOD_RANKER_USE_USER_LABELS="${VOD_RANKER_USE_USER_LABELS:-0}"', text)
        self.assertIn('VOD_RANKER_USE_GOLD_TRAIN="${VOD_RANKER_USE_GOLD_TRAIN:-0}"', text)
        self.assertIn("--include-gold-train \"$GOLD_EVAL_ROWS_PATH\"", text)
        self.assertIn("Preparing gold eval artifacts for training merge", text)
        self.assertNotIn("--merge-existing", text)
        self.assertNotIn("--preserve-existing-if-empty", text)
        self.assertIn("reset_user_label_artifacts", text)
        self.assertIn("User-labeled gold eval export disabled", text)
        self.assertIn("removed gold-eval artifacts", text)
        self.assertIn("Skipping gold eval; user-labeled rows are disabled.", text)
        self.assertIn("Skipping gold-derived hard-negative diagnosis; user-labeled rows are disabled.", text)
        self.assertIn("Skipping rollout review queue; user-labeled rows are disabled.", text)
        self.assertNotIn('--candidate-rows "$ARTIFACT_DIR/jynxzi_active_gold_eval_queue.features.jsonl"', text)


if __name__ == "__main__":
    unittest.main()
