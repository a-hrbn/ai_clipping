import json
import pathlib
import tempfile
import unittest

from ml.audit_gt_match_recall import build_audit


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class AuditGtMatchRecallTests(unittest.TestCase):
    def test_audio_fallback_rows_do_not_count_as_matched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            _write_jsonl(run_dir / "iboclipss_discovered.jsonl", [
                {"source_url": "https://www.youtube.com/shorts/transcripted1", "title": "Jynxzi explains", "view_count": 100},
                {"source_url": "https://www.youtube.com/shorts/audioonly002", "title": "Jynxzi rages", "view_count": 200},
                {"source_url": "https://www.youtube.com/shorts/unmatched0003", "title": "Jynxzi", "view_count": 50},
            ])
            _write_jsonl(run_dir / "matches.jsonl", [
                {"source_url": "https://www.youtube.com/shorts/transcripted1", "transcript_match_kind": "single_window"},
                {"source_url": "https://www.youtube.com/shorts/audioonly002", "transcript_match_kind": "audio_fingerprint_fallback"},
            ])

            audit = build_audit(run_dir)

        self.assertEqual(audit["matched_clips"], 1)
        self.assertEqual(audit["clip_match_recall"], round(1 / 3, 3))
        self.assertEqual(audit["audio_fallback_only_clips"], 1)
        self.assertEqual(audit["match_rows_by_kind"], {"single_window": 1, "audio_fingerprint_fallback": 1})
        unmatched_urls = {row["url"] for row in audit["top_unmatched_by_views"]}
        self.assertIn("https://www.youtube.com/shorts/audioonly002", unmatched_urls)


if __name__ == "__main__":
    unittest.main()
