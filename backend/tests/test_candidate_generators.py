import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import candidate_generators
from services.timeline_types import Chapter
from services.transcriber import RichSegment


class CandidateGeneratorTests(unittest.TestCase):
    def test_generate_candidates_defaults_to_semantic_candidates(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", speaker="STREAMER", hype_score=1.0),
            RichSegment(
                start=2.0,
                end=5.0,
                text="what is happening?!",
                speaker="STREAMER",
                emotions=["exclamation"],
                hype_score=5.5,
                arc_score=3.0,
                chat_spike_score=0.7,
                max_face_reaction=0.5,
            ),
            RichSegment(start=5.0, end=7.0, text="haha no way", speaker="GUEST_1", emotions=["laughter"], hype_score=4.5),
            RichSegment(start=7.0, end=10.0, text="listen to this take.", speaker="STREAMER", arc_score=3.2),
        ]

        candidates = candidate_generators.generate_candidates(segments)
        sources = {candidate.source for candidate in candidates}

        self.assertIn("quote", sources)
        self.assertTrue(sources.issubset({"quote", "chat_velocity", "tension_build", "narrative_reveal", "controversy"}))

    def test_generate_semantic_candidates_only_uses_semantic_sources(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", speaker="STREAMER", hype_score=1.0),
            RichSegment(
                start=2.0,
                end=5.0,
                text="listen to this insane take!",
                speaker="STREAMER",
                emotions=["exclamation"],
                hype_score=5.0,
                arc_score=3.2,
            ),
            RichSegment(start=5.0, end=8.0, text="nobody expected that reveal", speaker="GUEST_1", hype_score=2.0),
        ]

        candidates = candidate_generators.generate_semantic_candidates(segments)
        sources = {candidate.source for candidate in candidates}

        self.assertTrue(candidates)
        self.assertTrue(sources.issubset({"quote", "chat_velocity", "tension_build", "narrative_reveal", "controversy"}))

    def test_generate_semantic_candidates_includes_chat_velocity_peaks(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", speaker="STREAMER"),
            RichSegment(
                start=3.0,
                end=6.0,
                text="chat suddenly goes crazy",
                speaker="STREAMER",
                chat_spike_score=0.32,
                chat_rate_derivative=1.4,
                chat_ema_crossing_score=0.62,
            ),
        ]

        candidates = candidate_generators.generate_semantic_candidates(segments)
        chat_velocity = [candidate for candidate in candidates if candidate.source == "chat_velocity"]

        self.assertEqual(len(chat_velocity), 1)
        self.assertIn("chat_velocity:0:300", chat_velocity[0].id)

    def test_merge_prioritized_candidates_keeps_peak_when_semantic_candidate_overlaps(self) -> None:
        peak = candidate_generators.Candidate(
            id="peak:0:1000",
            start=8.0,
            end=24.0,
            source="peak",
            anchor_time=12.0,
            score_hint=8.0,
        )
        overlapping_semantic = candidate_generators.Candidate(
            id="quote:0:1100",
            start=9.0,
            end=22.0,
            source="quote",
            anchor_time=12.5,
            score_hint=6.0,
        )
        distant_semantic = candidate_generators.Candidate(
            id="controversy:0:5000",
            start=50.0,
            end=68.0,
            source="controversy",
            anchor_time=56.0,
            score_hint=5.0,
        )

        merged = candidate_generators.merge_prioritized_candidates([peak], [overlapping_semantic, distant_semantic])

        self.assertEqual([candidate.id for candidate in merged], ["peak:0:1000", "controversy:0:5000"])

    def test_merge_prioritized_candidates_can_preserve_stronger_context_variant(self) -> None:
        short_peak = candidate_generators.Candidate(
            id="peak:0:1200",
            start=10.0,
            end=20.0,
            source="peak",
            anchor_time=13.0,
            score_hint=7.5,
        )
        setup_variant = candidate_generators.Candidate(
            id="narrative_reveal:0:800",
            start=2.0,
            end=34.0,
            source="narrative_reveal",
            anchor_time=17.0,
            score_hint=4.0,
            tags=["question"],
        )

        default_merged = candidate_generators.merge_prioritized_candidates([short_peak], [setup_variant])
        context_merged = candidate_generators.merge_prioritized_candidates(
            [short_peak],
            [setup_variant],
            preserve_context_variants=True,
            max_context_variants_per_region=3,
        )

        self.assertEqual([candidate.id for candidate in default_merged], ["peak:0:1200"])
        self.assertEqual({candidate.id for candidate in context_merged}, {"peak:0:1200", "narrative_reveal:0:800"})

    def test_merge_prioritized_candidates_keeps_source_diverse_supplemental_variants(self) -> None:
        quote = candidate_generators.Candidate(
            id="quote:0:1100",
            start=9.0,
            end=22.0,
            source="quote",
            anchor_time=12.5,
            score_hint=6.0,
        )
        reveal = candidate_generators.Candidate(
            id="narrative_reveal:0:950",
            start=7.0,
            end=25.0,
            source="narrative_reveal",
            anchor_time=13.0,
            score_hint=5.5,
        )
        duplicate_quote = candidate_generators.Candidate(
            id="quote:1:1150",
            start=9.5,
            end=22.5,
            source="quote",
            anchor_time=12.6,
            score_hint=5.0,
        )

        merged = candidate_generators.merge_prioritized_candidates(
            [],
            [quote, reveal, duplicate_quote],
        )

        self.assertEqual([candidate.id for candidate in merged], ["narrative_reveal:0:950", "quote:0:1100"])

    def test_dense_window_candidates_add_recall_for_gradual_build(self) -> None:
        segments = [
            RichSegment(start=0.0, end=4.0, text="setup", hype_score=1.2, arc_score=0.6),
            RichSegment(start=4.0, end=8.0, text="getting tense", hype_score=2.3, arc_score=1.5),
            RichSegment(start=8.0, end=12.0, text="chat is waking up", hype_score=3.2, arc_score=2.2, chat_spike_score=0.35),
            RichSegment(start=12.0, end=16.0, text="no way this works", hype_score=4.8, arc_score=3.4, max_face_reaction=0.6),
            RichSegment(start=16.0, end=20.0, text="that's insane", hype_score=5.5, arc_score=4.0, chat_spike_score=0.55, emotions=["exclamation"]),
        ]

        candidates = candidate_generators._generate_dense_window_candidates(segments)

        self.assertTrue(candidates)
        self.assertTrue(any(candidate.source == "dense_window" for candidate in candidates))

    def test_generate_dense_window_candidates_is_public_whole_vod_recall_path(self) -> None:
        segments = [
            RichSegment(start=0.0, end=4.0, text="setup", hype_score=1.2, arc_score=0.6),
            RichSegment(start=4.0, end=8.0, text="getting tense", hype_score=2.3, arc_score=1.5),
            RichSegment(start=8.0, end=12.0, text="chat is waking up", hype_score=3.2, arc_score=2.2, chat_spike_score=0.35),
            RichSegment(start=12.0, end=16.0, text="no way this works", hype_score=4.8, arc_score=3.4, max_face_reaction=0.6),
            RichSegment(start=16.0, end=20.0, text="that's insane", hype_score=5.5, arc_score=4.0, chat_spike_score=0.55, emotions=["exclamation"]),
        ]

        candidates = candidate_generators.generate_dense_window_candidates(segments)

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.source == "dense_window" for candidate in candidates))

    def test_dense_candidate_inventory_scans_broad_overlapping_windows_before_caps(self) -> None:
        segments = [
            RichSegment(
                start=float(index * 5),
                end=float(index * 5 + 4),
                text=f"segment {index}",
                hype_score=1.0 + (index % 5),
                arc_score=0.5 + (index % 3),
            )
            for index in range(18)
        ]

        inventory = candidate_generators.generate_dense_candidate_inventory(
            segments,
            durations=(12.0, 20.0),
            stride_sec=5.0,
            max_candidates=0,
        )

        self.assertGreater(len(inventory), len(candidate_generators._generate_dense_window_candidates(segments)))
        self.assertTrue(all(candidate.source == "dense_inventory" for candidate in inventory))
        self.assertTrue(any(18.0 <= candidate.end - candidate.start <= 20.1 for candidate in inventory))

    def test_merge_prioritized_candidates_can_raise_dense_source_cap_for_ranker_recall(self) -> None:
        dense = [
            candidate_generators.Candidate(
                id=f"dense_window:16:{index}:{int(start * 100)}",
                start=start,
                end=start + 12.0,
                source="dense_window",
                anchor_time=start + 6.0,
                score_hint=3.0,
            )
            for index, start in enumerate((0.0, 20.0, 40.0, 60.0, 80.0))
        ]

        default_merged = candidate_generators.merge_prioritized_candidates([], dense)
        recall_merged = candidate_generators.merge_prioritized_candidates(
            [],
            dense,
            source_cap_overrides={"dense_window": 5},
        )

        self.assertEqual(len(default_merged), 3)
        self.assertEqual(len(recall_merged), 5)

    def test_merge_prioritized_candidates_caps_supplemental_by_quality_not_chronology(self) -> None:
        dense = [
            candidate_generators.Candidate(
                id=f"dense_window:low:{index}",
                start=float(index * 20),
                end=float(index * 20 + 20),
                source="dense_window",
                anchor_time=float(index * 20 + 10),
                score_hint=1.0,
            )
            for index in range(3)
        ]
        late_strong = candidate_generators.Candidate(
            id="dense_window:late_strong",
            start=120.0,
            end=148.0,
            source="dense_window",
            anchor_time=132.0,
            score_hint=7.5,
            tags=["exclamation"],
        )

        merged = candidate_generators.merge_prioritized_candidates(
            [],
            [*dense, late_strong],
            source_cap_overrides={"dense_window": 3},
        )

        merged_ids = {candidate.id for candidate in merged}
        self.assertIn("dense_window:late_strong", merged_ids)
        self.assertEqual(len([candidate for candidate in merged if candidate.source == "dense_window"]), 3)

    def test_merge_prioritized_candidates_audits_duplicate_and_source_cap_drops(self) -> None:
        kept = candidate_generators.Candidate(
            id="quote:kept",
            start=10.0,
            end=35.0,
            source="quote",
            anchor_time=20.0,
            score_hint=7.0,
        )
        duplicate = candidate_generators.Candidate(
            id="quote:duplicate",
            start=11.0,
            end=34.0,
            source="quote",
            anchor_time=20.5,
            score_hint=4.0,
        )
        dense_kept = candidate_generators.Candidate(
            id="dense_window:kept",
            start=80.0,
            end=105.0,
            source="dense_window",
            anchor_time=92.0,
            score_hint=5.0,
        )
        capped = candidate_generators.Candidate(
            id="dense_window:capped",
            start=130.0,
            end=155.0,
            source="dense_window",
            anchor_time=142.0,
            score_hint=2.0,
        )
        audit_rows: list[dict] = []

        merged = candidate_generators.merge_prioritized_candidates(
            [],
            [kept, duplicate, dense_kept, capped],
            source_cap_overrides={"quote": 4, "dense_window": 1},
            audit_rows=audit_rows,
            audit_stage="test_merge",
        )

        self.assertEqual([candidate.id for candidate in merged], ["quote:kept", "dense_window:kept"])
        duplicate_row = next(row for row in audit_rows if row["candidate_id"] == "quote:duplicate")
        capped_row = next(row for row in audit_rows if row["candidate_id"] == "dense_window:capped")
        self.assertEqual(duplicate_row["decision"], "dropped")
        self.assertEqual(duplicate_row["reason"], "duplicate")
        self.assertEqual(duplicate_row["duplicate_of"], "quote:kept")
        self.assertEqual(capped_row["decision"], "dropped")
        self.assertEqual(capped_row["reason"], "source_cap")

    def test_select_temporal_recall_candidates_keeps_time_coverage_before_global_score(self) -> None:
        early_strong = candidate_generators.Candidate(
            id="quote:early_strong",
            start=5.0,
            end=25.0,
            source="quote",
            anchor_time=12.0,
            score_hint=8.0,
        )
        early_weak = candidate_generators.Candidate(
            id="quote:early_weak",
            start=30.0,
            end=50.0,
            source="quote",
            anchor_time=40.0,
            score_hint=1.0,
        )
        late_weaker = candidate_generators.Candidate(
            id="quote:late_weaker",
            start=360.0,
            end=385.0,
            source="quote",
            anchor_time=372.0,
            score_hint=2.0,
        )

        selected = candidate_generators.select_temporal_recall_candidates(
            [early_weak, early_strong, late_weaker],
            bucket_sec=60.0,
            per_source_bucket=1,
            max_candidates=10,
        )

        selected_ids = {candidate.id for candidate in selected}
        self.assertIn("quote:early_strong", selected_ids)
        self.assertIn("quote:late_weaker", selected_ids)
        self.assertNotIn("quote:early_weak", selected_ids)

    def test_merge_prioritized_candidates_preserves_explicit_ids_through_duplicate_and_cap_checks(self) -> None:
        primary = [
            candidate_generators.Candidate(
                id="peak:primary",
                start=10.0,
                end=30.0,
                source="peak",
                anchor_time=20.0,
                score_hint=8.0,
            )
        ]
        preserved = candidate_generators.Candidate(
            id="dense_inventory:preserved-selected",
            start=8.0,
            end=32.0,
            source="dense_inventory",
            anchor_time=20.0,
            score_hint=1.0,
        )
        cap_filler = candidate_generators.Candidate(
            id="dense_inventory:filler",
            start=80.0,
            end=100.0,
            source="dense_inventory",
            anchor_time=90.0,
            score_hint=1.0,
        )

        merged = candidate_generators.merge_prioritized_candidates(
            primary,
            [cap_filler, preserved],
            supplemental_source_cap=1,
            preserve_supplemental_ids={preserved.id},
        )

        self.assertEqual(
            {candidate.id for candidate in merged},
            {"peak:primary", "dense_inventory:filler", "dense_inventory:preserved-selected"},
        )

    def test_generate_fallback_candidates_preserves_generic_recall_sources(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", speaker="STREAMER", hype_score=1.0),
            RichSegment(
                start=2.0,
                end=5.0,
                text="what is happening?!",
                speaker="STREAMER",
                emotions=["exclamation"],
                hype_score=5.5,
                arc_score=3.0,
                chat_spike_score=0.7,
                max_face_reaction=0.5,
            ),
            RichSegment(start=5.0, end=7.0, text="haha no way", speaker="GUEST_1", emotions=["laughter"], hype_score=4.5),
            RichSegment(start=7.0, end=10.0, text="listen to this take.", speaker="STREAMER", arc_score=3.2),
        ]

        candidates = candidate_generators.generate_fallback_candidates(segments)
        sources = {candidate.source for candidate in candidates}

        self.assertIn("hype_peak", sources)
        self.assertIn("chat_sync", sources)
        self.assertIn("chat_velocity", {candidate.source for candidate in candidate_generators.generate_semantic_candidates(segments)})
        self.assertIn("reaction", sources)
        self.assertIn("clean_reaction", sources)

    def test_generate_fallback_candidates_adds_new_shortform_sources(self) -> None:
        segments = [
            RichSegment(start=0.0, end=3.0, text="setup", speaker="STREAMER", hype_score=1.0),
            RichSegment(start=3.0, end=6.0, text="bro listen i'm telling you this is insane", speaker="STREAMER", hype_score=2.1, arc_score=2.4),
            RichSegment(start=6.0, end=9.0, text="wait for it", speaker="STREAMER", hype_score=2.8, arc_score=3.0),
            RichSegment(start=9.0, end=12.0, text="no way!", speaker="STREAMER", emotions=["exclamation", "laughter"], hype_score=5.8, arc_score=4.1, chat_spike_score=0.6, max_face_reaction=0.7),
        ]

        candidates = candidate_generators.generate_fallback_candidates(segments)
        sources = {candidate.source for candidate in candidates}

        self.assertIn("hot_take", sources)
        self.assertIn("story_payoff", sources)

    def test_window_candidate_adapts_window_for_fast_hook(self) -> None:
        urgent = RichSegment(
            start=10.0,
            end=12.0,
            text="what?!",
            emotions=["exclamation"],
            hype_score=5.5,
            arc_score=1.0,
            chat_spike_score=0.8,
            max_face_reaction=0.7,
        )
        slower = RichSegment(
            start=10.0,
            end=12.0,
            text="let me explain what happened here",
            hype_score=2.5,
            arc_score=4.0,
            chat_spike_score=0.0,
            max_face_reaction=0.0,
        )

        urgent_candidate = candidate_generators._window_candidate(urgent, 0, "reaction", lead=8.0, tail=12.0)
        slow_candidate = candidate_generators._window_candidate(slower, 0, "reaction", lead=8.0, tail=12.0)

        self.assertGreater(urgent_candidate.start, slow_candidate.start)
        self.assertLess(urgent_candidate.end, slow_candidate.end)

    def test_generate_semantic_candidates_runs_per_chapter_when_provided(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="first chapter"),
            RichSegment(start=2.0, end=4.0, text="still first chapter"),
            RichSegment(start=10.0, end=12.0, text="second chapter"),
            RichSegment(start=12.0, end=14.0, text="still second chapter"),
        ]
        chapters = [
            Chapter(id="chapter_00", start=0.0, end=4.0, index=0),
            Chapter(id="chapter_01", start=10.0, end=14.0, index=1),
        ]

        with patch("services.candidate_generators._generate_semantic_candidates_for_segments") as helper_mock:
            helper_mock.side_effect = [
                [candidate_generators.Candidate(id="quote:0:100", start=0.0, end=4.0, source="quote", anchor_time=1.0)],
                [candidate_generators.Candidate(id="quote:1:1100", start=10.0, end=14.0, source="quote", anchor_time=11.0)],
            ]
            candidates = candidate_generators.generate_semantic_candidates(segments, chapters=chapters)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(helper_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
