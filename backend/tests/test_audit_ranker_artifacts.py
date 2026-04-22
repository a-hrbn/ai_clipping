import json
import tempfile
import unittest
from pathlib import Path

from ml import audit_ranker_artifacts


class AuditRankerArtifactsTests(unittest.TestCase):
    def test_audit_artifacts_reports_deploy_and_scale_gaps(self) -> None:
        artifact = {
            "profile": "jynxzi",
            "selected_model": "logistic_regression",
            "training_stats": {
                "rows": 120,
                "positive_rows": 20,
                "negative_rows": 100,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jynxzi_ranker.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")

            payload = audit_ranker_artifacts.audit_artifacts(Path(tmp))

        self.assertEqual(len(payload["profiles"]), 1)
        audited = payload["profiles"][0]
        self.assertTrue(audited["deploy_gate"]["ready"])
        self.assertFalse(audited["scale_target"]["ready"])
        self.assertEqual(audited["scale_gap"]["rows"], 4880)


if __name__ == "__main__":
    unittest.main()
