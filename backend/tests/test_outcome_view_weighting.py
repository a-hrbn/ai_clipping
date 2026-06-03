from ml import outcome_view_weighting


def test_outcome_view_weighting_tiers_views_and_preserves_unknown_weight() -> None:
    assert outcome_view_weighting.outcome_view_tier(0) == "unknown"
    assert outcome_view_weighting.outcome_view_sample_weight(0) == 1.0
    assert outcome_view_weighting.outcome_view_tier(30_000) == "strong"
    assert outcome_view_weighting.outcome_view_sample_weight(30_000) == 1.75
    assert outcome_view_weighting.outcome_view_tier(700_000) == "breakout_viral"
    assert outcome_view_weighting.outcome_view_rank_target(700_000) == 5


def test_annotate_cross_platform_view_weight_marks_viral_positive() -> None:
    row = outcome_view_weighting.annotate_cross_platform_view_weight(
        {
            "views": 250_000,
            "rank_target": 3,
            "ordinal_label": 3,
            "sample_weight": 0.2,
        }
    )

    assert row["outcome_view_tier"] == "viral"
    assert row["outcome_view_sample_weight"] == 2.0
    assert row["sample_weight"] == 2.0
    assert row["rank_target"] == 4
    assert row["ordinal_label"] == 4
    assert row["label_quality_class"] == "cross_platform_viral_positive"
