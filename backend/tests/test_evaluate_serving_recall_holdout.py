import json
import tempfile
import unittest
from pathlib import Path

from ml import evaluate_serving_recall_holdout


class EvaluateServingRecallHoldoutTests(unittest.TestCase):
    def test_evaluate_holdout_aggregates_oracle_free_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            self._write_job(
                storage,
                "job-a",
                vod_id="123",
                recall={
                    "positive_windows": 2,
                    "recall_by_stage": {
                        "dense_inventory": {"matched": 2, "total": 2, "ratio": 1.0},
                        "ranked_pre_judge": {"matched": 1, "total": 2, "ratio": 0.5},
                        "final_moments": {"matched": 0, "total": 2, "ratio": 0.0},
                    },
                    "details": [
                        {
                            "positive_id": "short-a",
                            "start": 10.0,
                            "end": 30.0,
                            "last_matched_stage": "ranked_pre_judge",
                        }
                    ],
                },
            )
            self._write_job(
                storage,
                "job-b",
                vod_id="456",
                recall={
                    "positive_windows": 1,
                    "recall_by_stage": {
                        "dense_inventory": {"matched": 1, "total": 1, "ratio": 1.0},
                        "ranked_pre_judge": {"matched": 1, "total": 1, "ratio": 1.0},
                        "final_moments": {"matched": 1, "total": 1, "ratio": 1.0},
                    },
                    "details": [],
                },
            )

            report = evaluate_serving_recall_holdout.evaluate_holdout(
                storage_dir=storage,
                positive_rows_path=None,
            )

        self.assertEqual(report["jobs_evaluated"], 2)
        self.assertEqual(report["aggregate"]["positive_windows"], 3)
        self.assertEqual(report["aggregate"]["recall_by_stage"]["dense_inventory"]["ratio"], 1.0)
        self.assertEqual(report["aggregate"]["recall_by_stage"]["ranked_pre_judge"]["matched"], 2)
        self.assertEqual(report["aggregate"]["recall_by_stage"]["final_moments"]["ratio"], 0.333333)

    def test_evaluate_holdout_skips_jobs_with_recall_target_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            self._write_job(
                storage,
                "job-injected",
                vod_id="123",
                recall={
                    "positive_windows": 1,
                    "recall_by_stage": {"final_moments": {"matched": 1, "total": 1, "ratio": 1.0}},
                    "details": [],
                },
                trace_rows=[
                    {
                        "candidate_id": "oracle",
                        "rationale": ["cross_platform_recall_target"],
                    }
                ],
            )

            report = evaluate_serving_recall_holdout.evaluate_holdout(
                storage_dir=storage,
                positive_rows_path=None,
            )

        self.assertEqual(report["jobs_evaluated"], 0)
        self.assertEqual(report["jobs_skipped"], {"oracle_injection_detected": 1})

    def test_evaluate_holdout_skips_unknown_oracle_state_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            self._write_job(
                storage,
                "job-old",
                vod_id="123",
                recall={
                    "positive_windows": 1,
                    "recall_by_stage": {"ranked_pre_judge": {"matched": 1, "total": 1, "ratio": 1.0}},
                    "details": [],
                },
                diagnostics={},
            )

            strict = evaluate_serving_recall_holdout.evaluate_holdout(
                storage_dir=storage,
                positive_rows_path=None,
            )
            relaxed = evaluate_serving_recall_holdout.evaluate_holdout(
                storage_dir=storage,
                positive_rows_path=None,
                require_known_oracle_free=False,
            )

        self.assertEqual(strict["jobs_evaluated"], 0)
        self.assertEqual(strict["jobs_skipped"], {"unknown_oracle_state": 1})
        self.assertEqual(relaxed["jobs_evaluated"], 1)

    def test_evaluate_holdout_uses_latest_valid_job_per_vod_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            self._write_job(
                storage,
                "job-old",
                vod_id="123",
                recall={
                    "positive_windows": 1,
                    "recall_by_stage": {"final_moments": {"matched": 0, "total": 1, "ratio": 0.0}},
                    "details": [],
                },
            )
            self._write_job(
                storage,
                "job-new",
                vod_id="123",
                recall={
                    "positive_windows": 1,
                    "recall_by_stage": {"final_moments": {"matched": 1, "total": 1, "ratio": 1.0}},
                    "details": [],
                },
            )

            report = evaluate_serving_recall_holdout.evaluate_holdout(
                storage_dir=storage,
                positive_rows_path=None,
            )

        self.assertEqual(report["jobs_evaluated"], 1)
        self.assertEqual(report["jobs_skipped"], {"duplicate_vod_run": 1})
        self.assertEqual(report["jobs"][0]["job_id"], "job-new")
        self.assertEqual(report["aggregate"]["recall_by_stage"]["final_moments"]["ratio"], 1.0)

    def test_evaluate_holdout_reports_product_precision_and_recall_at_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            self._write_job(
                storage,
                "job-a",
                vod_id="123",
                recall={
                    "positive_windows": 2,
                    "recall_by_stage": {"final_ranked": {"matched": 1, "total": 2, "ratio": 0.5}},
                    "details": [
                        {"positive_id": "short-a", "start": 10.0, "end": 20.0},
                        {"positive_id": "short-b", "start": 100.0, "end": 110.0},
                    ],
                },
                trace_rows=[
                    {
                        "candidate_id": "good",
                        "start": 10.0,
                        "end": 20.0,
                        "final_rank": 1,
                        "final_selected": True,
                        "stages": {
                            "final_ranked": {"status": "selected", "rank": 1},
                            "final_moments": {"status": "selected", "index": 1},
                        },
                    },
                    {
                        "candidate_id": "miss",
                        "start": 30.0,
                        "end": 40.0,
                        "final_rank": 2,
                        "stages": {
                            "final_ranked": {"status": "selected", "rank": 2},
                        },
                    },
                ],
            )

            report = evaluate_serving_recall_holdout.evaluate_holdout(
                storage_dir=storage,
                positive_rows_path=None,
                product_k_values=[1, 2],
            )

        job_metrics = report["jobs"][0]["product_selection"]["final_ranked"]
        self.assertEqual(job_metrics["k1"]["selected_candidate_rows"], 1)
        self.assertEqual(job_metrics["k1"]["selected_positive_candidates"], 1)
        self.assertEqual(job_metrics["k1"]["precision"], 1.0)
        self.assertEqual(job_metrics["k1"]["per_vod_recall"], 0.5)
        self.assertEqual(job_metrics["k2"]["selected_candidate_rows"], 2)
        self.assertEqual(job_metrics["k2"]["precision"], 0.5)
        aggregate = report["aggregate"]["product_selection"]["final_ranked@k2"]
        self.assertEqual(aggregate["precision"], 0.5)
        self.assertEqual(aggregate["precision_at_capacity"], 0.5)
        self.assertEqual(aggregate["per_vod_recall"], 0.5)

    def test_evaluate_holdout_reports_median_metrics_for_repeat_vod_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            recall = {
                "positive_windows": 1,
                "recall_by_stage": {"final_ranked": {"matched": 1, "total": 1, "ratio": 1.0}},
                "details": [{"positive_id": "short-a", "start": 10.0, "end": 20.0}],
            }
            for index, selected in enumerate([True, False, True], start=1):
                trace_rows = []
                if selected:
                    trace_rows.append(
                        {
                            "candidate_id": f"candidate-{index}",
                            "start": 10.0,
                            "end": 20.0,
                            "final_rank": 1,
                            "stages": {"final_ranked": {"status": "selected", "rank": 1}},
                        }
                    )
                self._write_job(
                    storage,
                    f"job-{index}",
                    vod_id="123",
                    recall=recall,
                    trace_rows=trace_rows or [{"candidate_id": f"candidate-{index}"}],
                )

            report = evaluate_serving_recall_holdout.evaluate_holdout(
                storage_dir=storage,
                positive_rows_path=None,
                latest_per_vod=False,
                product_k_values=[1],
            )

        robust = report["aggregate"]["robust_product_selection"]
        self.assertEqual(robust["vods_with_3plus_runs"], 1)
        self.assertEqual(robust["by_vod"]["123"]["runs"], 3)
        self.assertEqual(robust["by_metric"]["final_ranked@k1"]["median_precision"], 1.0)
        self.assertEqual(robust["by_metric"]["final_ranked@k1"]["median_per_vod_recall"], 1.0)

    def _write_job(
        self,
        storage: Path,
        job_id: str,
        *,
        vod_id: str,
        recall: dict,
        diagnostics: dict | None = None,
        trace_rows: list[dict] | None = None,
    ) -> None:
        job_dir = storage / job_id
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True)
        (job_dir / "request.json").write_text(
            json.dumps({"url": f"https://www.twitch.tv/videos/{vod_id}"}),
            encoding="utf-8",
        )
        (debug_dir / "candidate_recall_report.json").write_text(
            json.dumps({"positive_window_recall": recall}),
            encoding="utf-8",
        )
        funnel = {}
        if diagnostics is None:
            diagnostics = {
                "enabled": False,
                "active": True,
                "diagnostic_only": True,
                "status": "loaded",
                "matched_rows": recall.get("positive_windows", 0),
            }
        if diagnostics:
            funnel["cross_platform_positive_diagnostics"] = diagnostics
        (debug_dir / "candidate_funnel_summary.json").write_text(
            json.dumps(funnel),
            encoding="utf-8",
        )
        rows = trace_rows if trace_rows is not None else [{"candidate_id": "candidate"}]
        (debug_dir / "candidate_trace.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
