import unittest

from ml import evaluate_boundary_refinement


class EvaluateBoundaryRefinementTests(unittest.TestCase):
    def test_boundary_metrics_aggregate_error_distribution(self) -> None:
        rows = [
            {
                "boundary_abs_start_error_sec": 1.0,
                "boundary_abs_end_error_sec": 0.5,
                "boundary_abs_total_error_sec": 1.5,
            },
            {
                "boundary_abs_start_error_sec": 2.0,
                "boundary_abs_end_error_sec": 1.0,
                "boundary_abs_total_error_sec": 3.0,
            },
        ]

        metrics = evaluate_boundary_refinement._boundary_metrics(rows)

        self.assertEqual(metrics["mean_abs_start_error_sec"], 1.5)
        self.assertEqual(metrics["mean_abs_end_error_sec"], 0.75)
        self.assertEqual(metrics["mean_abs_total_error_sec"], 2.25)
        self.assertEqual(metrics["within_2s_total_rate"], 0.5)
        self.assertEqual(metrics["severe_error_rate"], 0.5)

    def test_bucket_by_kind_groups_rows(self) -> None:
        rows = [
            {"boundary_reference_start": 10.0, "boundary_reference_end": 20.0, "boundary_abs_total_error_sec": 1.5, "boundary_perturbation_kind": "start_late_small"},
            {"boundary_reference_start": 10.0, "boundary_reference_end": 20.0, "boundary_abs_total_error_sec": 2.5, "boundary_perturbation_kind": "start_late_small"},
            {"boundary_reference_start": 10.0, "boundary_reference_end": 20.0, "boundary_abs_total_error_sec": 3.0, "boundary_perturbation_kind": "end_early_large"},
        ]

        grouped = evaluate_boundary_refinement._bucket_by_kind(rows, top_kinds=10)

        self.assertEqual(grouped[0]["kind"], "start_late_small")
        self.assertEqual(grouped[0]["rows"], 2)
        self.assertEqual(grouped[1]["kind"], "end_early_large")


if __name__ == "__main__":
    unittest.main()
