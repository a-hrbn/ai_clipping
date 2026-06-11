import re
import unittest
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1] / "ml"

# Gold-eval & manifest builder family (PRD #30 / Tracer #35). Each must pull its
# JSONL/JSON I/O from the row_identity seam instead of rolling a local copy.
GROUP = [
    "build_gold_eval_manifest",
    "build_gold_eval_queue",
    "build_gold_holdout_annotation_manifest",
    "export_frontend_gold_eval",
    "import_gold_eval_labels",
    "build_eval_dataset",
]

LOCAL_IO_DEF = re.compile(r"^\s*def _(?:read_jsonl|write_jsonl|write_json)\b", re.MULTILINE)


class GoldEvalIoConsolidationTests(unittest.TestCase):
    """The builders no longer roll their own JSONL/JSON plumbing: the
    strict/lenient decision lives at the call-site (``row_identity.read_jsonl``)
    and writes go through ``row_identity.write_jsonl`` / ``write_json``. This
    guards AC #1/#5 of the tracer — ``grep`` finds no local I/O definition."""

    def test_no_module_defines_local_jsonl_or_json_io(self) -> None:
        offenders: dict[str, list[str]] = {}
        for name in GROUP:
            source = (ML_DIR / f"{name}.py").read_text(encoding="utf-8")
            hits = LOCAL_IO_DEF.findall(source)
            if hits:
                offenders[name] = hits
        self.assertEqual(
            offenders,
            {},
            f"local JSONL/JSON I/O still defined in: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
