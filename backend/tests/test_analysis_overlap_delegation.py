import unittest
from unittest.mock import patch

from ml import (
    evaluate_candidate_funnel,
    evaluate_serving_recall_holdout,
    export_serving_hard_negatives,
    sample_background_negatives,
)

SENTINEL = 0.4242


class OverlapDelegationTests(unittest.TestCase):
    """Die Analyse-Skripte delegieren die *reine* Geometrie an
    ``row_identity.window_overlap_ratio``; ihre dünnen Coercion-Wrapper
    (Float-Coercion, None-Handling, Dict-Key-Extraktion) bleiben lokal."""

    def test_funnel_delegates_pure_geometry(self) -> None:
        with patch.object(evaluate_candidate_funnel, "window_overlap_ratio", return_value=SENTINEL) as m:
            result = evaluate_candidate_funnel._window_overlap_ratio(0.0, 10.0, 0.0, 5.0)
        self.assertEqual(result, SENTINEL)
        m.assert_called_once_with(0.0, 10.0, 0.0, 5.0)

    def test_funnel_keeps_coercion_wrapper(self) -> None:
        with patch.object(evaluate_candidate_funnel, "window_overlap_ratio", return_value=SENTINEL) as m:
            result = evaluate_candidate_funnel._window_overlap_ratio(0.0, 10.0, "nan-ish", None)
        self.assertEqual(result, 0.0)
        m.assert_not_called()

    def test_holdout_delegates_pure_geometry(self) -> None:
        positive = {"start": 0.0, "end": 10.0}
        row = {"start": 0.0, "end": 5.0}
        with patch.object(evaluate_serving_recall_holdout, "window_overlap_ratio", return_value=SENTINEL) as m:
            result = evaluate_serving_recall_holdout._window_overlap_ratio(positive, row)
        self.assertEqual(result, SENTINEL)
        m.assert_called_once_with(0.0, 10.0, 0.0, 5.0)

    def test_holdout_keeps_coercion_wrapper(self) -> None:
        with patch.object(evaluate_serving_recall_holdout, "window_overlap_ratio", return_value=SENTINEL) as m:
            result = evaluate_serving_recall_holdout._window_overlap_ratio({"start": 0.0}, {"start": 0.0, "end": 5.0})
        self.assertEqual(result, 0.0)
        m.assert_not_called()

    def test_export_delegates_pure_geometry(self) -> None:
        with patch.object(export_serving_hard_negatives, "window_overlap_ratio", return_value=SENTINEL) as m:
            result = export_serving_hard_negatives._window_overlap_ratio(0.0, 5.0, 0.0, 10.0)
        self.assertEqual(result, SENTINEL)
        m.assert_called_once_with(0.0, 5.0, 0.0, 10.0)

    def test_background_negatives_delegates_pure_geometry(self) -> None:
        with patch.object(sample_background_negatives, "window_overlap_ratio", return_value=SENTINEL) as m:
            result = sample_background_negatives._overlap_ratio((0.0, 5.0), (0.0, 10.0))
        self.assertEqual(result, SENTINEL)
        m.assert_called_once_with(0.0, 5.0, 0.0, 10.0)


if __name__ == "__main__":
    unittest.main()
