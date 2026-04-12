import unittest

from ml import annotation_store, export_annotation_queue


class AnnotationFlowTests(unittest.TestCase):
    def test_export_annotation_queue_includes_multiple_review_buckets(self) -> None:
        rows = [
            {"vod_id": "1", "vod_url": "https://example.com/1", "window_start": 10.0, "window_end": 20.0, "label_class": "ambiguous"},
            {"vod_id": "1", "vod_url": "https://example.com/1", "window_start": 30.0, "window_end": 40.0, "label_class": "weak_positive"},
            {"vod_id": "1", "vod_url": "https://example.com/1", "window_start": 50.0, "window_end": 60.0, "label_class": "hard_negative", "negative_source": "boundary_mined"},
            {"vod_id": "1", "vod_url": "https://example.com/1", "window_start": 70.0, "window_end": 80.0, "label_class": "positive"},
        ]

        queued = [row for row in rows if export_annotation_queue._should_queue_row(row)]
        self.assertEqual(len(queued), 3)
        self.assertEqual({export_annotation_queue._queue_reason(row) for row in queued}, {
            "ambiguous_window",
            "weak_positive_review",
            "boundary_hard_negative_review",
        })

    def test_merge_annotations_promotes_manual_labels(self) -> None:
        dataset_rows = [
            {
                "vod_id": "1",
                "window_start": 10.0,
                "window_end": 20.0,
                "label": 0,
                "label_class": "hard_negative",
                "supervision_tier": "twitch_clip_proxy",
            }
        ]
        annotations = [
            {
                "sample_id": "1:10.0:20.0",
                "is_good_clip": True,
                "quality_score": 5,
                "notes": "clear winner",
            }
        ]

        merged = annotation_store.merge_annotations(dataset_rows, annotations)

        self.assertEqual(merged[0]["label"], 1)
        self.assertEqual(merged[0]["label_source"], "manual_annotation")
        self.assertEqual(merged[0]["supervision_tier"], "manual_annotation")
        self.assertEqual(merged[0]["label_class"], "viral_positive")
        self.assertEqual(merged[0]["ordinal_label"], 4)


if __name__ == "__main__":
    unittest.main()
