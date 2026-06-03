import json
import tempfile
import unittest
from pathlib import Path

from ml import audit_clipfinding_artifacts


class AuditClipfindingArtifactsTests(unittest.TestCase):
    def test_audit_reports_artifact_hashes_and_row_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "jynxzi_proposal_scorer.json").write_text(
                json.dumps(
                    {
                        "training_summary": {
                            "status": "trained",
                            "rows": 10,
                            "positive_rows": 4,
                            "negative_rows": 6,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / "jynxzi_cross_platform_fingerprint_rows.jsonl").write_text(
                "{}\n{}\n",
                encoding="utf-8",
            )

            payload = audit_clipfinding_artifacts.build_audit(profile="jynxzi", artifact_dir=artifact_dir)

        self.assertEqual(payload["summary"]["existing_artifacts"], 2)
        self.assertEqual(payload["summary"]["cross_platform_rows"], 2)
        self.assertEqual(payload["summary"]["proposal_status"], "trained")
        self.assertRegex(payload["artifacts"]["proposal_scorer"]["sha256"], r"^[a-f0-9]{64}$")


if __name__ == "__main__":
    unittest.main()
