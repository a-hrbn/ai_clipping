from services import context_patterns, feature_builder
from services.timeline_types import Candidate
from services.transcriber import RichSegment


def _fake_context_embeddings(texts):
    vectors = []
    for text in texts:
        lowered = text.lower()
        if any(token in lowered for token in ("fanmail", "package", "box", "gift", "letter")):
            vectors.append([0.0, 1.0, 0.0])
        elif any(token in lowered for token in ("world cup", "football final", "soccer championship", "suite", "vip event", "luxury lounge")):
            vectors.append([1.0, 0.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return vectors


def test_context_pattern_store_clusters_repeated_short_context(tmp_path) -> None:
    rows = [
        {
            "repost_title": "Jynxzi opens a wild fanmail box",
            "transcript_repost_excerpt": "this box has another box inside just for you",
            "source_url": "https://www.youtube.com/watch?v=a",
            "repost_views": 100_000,
        },
        {
            "repost_title": "Jynxzi gets a crazy fanmail package",
            "transcript_repost_excerpt": "there is a fanmail letter and a joke in the box",
            "source_url": "https://www.youtube.com/watch?v=b",
            "repost_views": 50_000,
        },
        {
            "repost_title": "Jynxzi booked a World Cup suite",
            "transcript_repost_excerpt": "world cup quarter final in miami suite tickets",
            "source_url": "https://www.youtube.com/watch?v=c",
            "repost_views": 200_000,
        },
        {
            "repost_title": "Jynxzi explains his World Cup tickets",
            "transcript_repost_excerpt": "booked a suite for the world cup in miami",
            "source_url": "https://www.youtube.com/watch?v=d",
            "repost_views": 70_000,
        },
    ]

    payload = context_patterns.write_pattern_store(
        rows,
        profile="jynxzi",
        output=tmp_path / "jynxzi_context_patterns.json",
        min_support=2,
        embedding_fn=_fake_context_embeddings,
    )

    assert len(payload["patterns"]) >= 2
    fanmail_match = context_patterns.score_text(
        "Jynxzi opens fanmail and finds a letter inside the box",
        profile="jynxzi",
        artifact_dir=tmp_path,
        embedding_fn=_fake_context_embeddings,
    )
    world_cup_match = context_patterns.score_text(
        "he booked a suite for the world cup in Miami",
        profile="jynxzi",
        artifact_dir=tmp_path,
        embedding_fn=_fake_context_embeddings,
    )
    assert fanmail_match.confidence > 0.4
    assert world_cup_match.confidence > 0.4
    assert fanmail_match.pattern_id != world_cup_match.pattern_id


def test_context_patterns_match_semantic_rewording_with_low_keyword_overlap(tmp_path) -> None:
    payload = context_patterns.write_pattern_store(
        [
            {
                "repost_title": "Jynxzi booked a World Cup suite",
                "transcript_repost_excerpt": "world cup quarter final in miami suite tickets",
                "source_url": "https://www.youtube.com/watch?v=a",
            },
            {
                "repost_title": "Jynxzi reserved a VIP event lounge",
                "transcript_repost_excerpt": "football final luxury lounge reservation for the event",
                "source_url": "https://www.youtube.com/watch?v=b",
            },
        ],
        profile="jynxzi",
        output=tmp_path / "jynxzi_context_patterns.json",
        min_support=2,
        embedding_fn=_fake_context_embeddings,
    )

    assert payload["semantic_matching_enabled"] is True
    assert payload["patterns"][0]["embedding_centroid"]

    match = context_patterns.score_text(
        "he rented a private lounge for the soccer championship",
        profile="jynxzi",
        artifact_dir=tmp_path,
        embedding_fn=_fake_context_embeddings,
    )

    assert match.confidence > 0.6
    assert match.score > 0.7


def test_candidate_features_use_dynamic_context_pattern_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(context_patterns, "DEFAULT_ARTIFACT_DIR", tmp_path)
    context_patterns.write_pattern_store(
        [
            {
                "repost_title": "Jynxzi opens a fanmail box",
                "transcript_repost_excerpt": "fanmail letter inside the box",
                "repost_views": 120_000,
            },
            {
                "repost_title": "Jynxzi reacts to fanmail package",
                "transcript_repost_excerpt": "package and fanmail gift in the box",
                "repost_views": 80_000,
            },
        ],
        profile="jynxzi",
        output=tmp_path / "jynxzi_context_patterns.json",
        min_support=2,
        embedding_fn=_fake_context_embeddings,
    )
    candidate = Candidate(
        id="dense_inventory:20:0:0:0",
        start=0.0,
        end=20.0,
        source="dense_inventory",
        anchor_time=10.0,
    )
    segments = [
        RichSegment(start=0.0, end=10.0, text="look at this fanmail letter inside the box"),
        RichSegment(start=10.0, end=20.0, text="this package is actually crazy"),
    ]

    features = feature_builder.build_features(
        [candidate],
        segments,
        "jynxzi",
        use_llm_semantics=False,
    )

    assert features[0].context_pattern_confidence > 0.4
    assert features[0].context_pattern_id
