import json
import pathlib
import tempfile
import unittest

from ml import row_identity


class WindowOverlapRatioTests(unittest.TestCase):
    def test_identical_windows_return_one(self) -> None:
        self.assertEqual(row_identity.window_overlap_ratio(10.0, 20.0, 10.0, 20.0), 1.0)

    def test_disjoint_windows_return_zero(self) -> None:
        self.assertEqual(row_identity.window_overlap_ratio(0.0, 5.0, 10.0, 15.0), 0.0)

    def test_touching_windows_return_zero(self) -> None:
        # End of A coincides with start of B → no positive overlap.
        self.assertEqual(row_identity.window_overlap_ratio(0.0, 10.0, 10.0, 20.0), 0.0)

    def test_partial_overlap_uses_shorter_duration(self) -> None:
        # A = [0, 10] (dur 10), B = [8, 12] (dur 4). Overlap = [8, 10] = 2.
        # Ratio = overlap / shorter duration = 2 / 4 = 0.5.
        self.assertEqual(row_identity.window_overlap_ratio(0.0, 10.0, 8.0, 12.0), 0.5)

    def test_short_clip_fully_inside_long_window_returns_one(self) -> None:
        # A short window entirely inside a long one covers 100% of the shorter.
        self.assertEqual(row_identity.window_overlap_ratio(0.0, 100.0, 40.0, 50.0), 1.0)

    def test_zero_length_window_returns_zero(self) -> None:
        self.assertEqual(row_identity.window_overlap_ratio(5.0, 5.0, 0.0, 10.0), 0.0)

    def test_inverted_window_returns_zero(self) -> None:
        self.assertEqual(row_identity.window_overlap_ratio(10.0, 5.0, 0.0, 20.0), 0.0)

    def test_is_commutative(self) -> None:
        forward = row_identity.window_overlap_ratio(0.0, 10.0, 8.0, 12.0)
        backward = row_identity.window_overlap_ratio(8.0, 12.0, 0.0, 10.0)
        self.assertEqual(forward, backward)


class JsonIoRoundTripTests(unittest.TestCase):
    def test_read_json_round_trips_write_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "payload.json"
            payload = {"vod_id": "123", "views": 42, "nested": {"k": [1, 2, 3]}}
            row_identity.write_json(path, payload)
            self.assertEqual(row_identity.read_json(path), payload)

    def test_read_json_missing_file_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "absent.json"
            self.assertEqual(row_identity.read_json(path), {})

    def test_read_json_non_dict_payload_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "list.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            self.assertEqual(row_identity.read_json(path), {})

    def test_read_json_corrupt_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "broken.json"
            path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                row_identity.read_json(path)

    def test_read_jsonl_round_trips_write_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "rows.jsonl"
            rows = [{"a": 1}, {"b": 2}, {"c": [3, 4]}]
            row_identity.write_jsonl(path, rows)
            self.assertEqual(row_identity.read_jsonl(path), rows)

    def test_read_jsonl_missing_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "absent.jsonl"
            self.assertEqual(row_identity.read_jsonl(path), [])

    def test_read_jsonl_corrupt_line_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "broken.jsonl"
            path.write_text('{"a": 1}\n{not valid json\n', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                row_identity.read_jsonl(path)

    def test_read_jsonl_strict_false_skips_corrupt_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "mixed.jsonl"
            path.write_text('{"a": 1}\n{not valid json\n{"b": 2}\n', encoding="utf-8")
            self.assertEqual(
                row_identity.read_jsonl(path, strict=False),
                [{"a": 1}, {"b": 2}],
            )

    def test_read_jsonl_strict_false_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "blanks.jsonl"
            path.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
            self.assertEqual(
                row_identity.read_jsonl(path, strict=False),
                [{"a": 1}, {"b": 2}],
            )

    def test_read_jsonl_strict_false_missing_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "absent.jsonl"
            self.assertEqual(row_identity.read_jsonl(path, strict=False), [])


if __name__ == "__main__":
    unittest.main()
