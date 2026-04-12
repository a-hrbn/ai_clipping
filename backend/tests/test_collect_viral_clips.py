import tempfile
import unittest
from pathlib import Path

from ml import collect_viral_clips


class CollectViralClipsTests(unittest.TestCase):
    def test_read_entries_supports_manifest_metadata_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clips.csv"
            path.write_text(
                "\n".join([
                    "url,streamer,platform,language,tags,label,notes",
                    'https://www.youtube.com/shorts/abc,shroud,youtube,en,"fps,clutch",1,viral',
                    'https://www.tiktok.com/@tarik/video/123,tarik,tiktok,en,"reaction,chatting",0,flat',
                ]),
                encoding="utf-8",
            )

            entries = collect_viral_clips._read_entries(path)

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["streamer"], "shroud")
            self.assertEqual(entries[1]["label"], "0")
            self.assertEqual(entries[1]["notes"], "flat")

    def test_parse_label_accepts_positive_and_negative_aliases(self) -> None:
        self.assertEqual(collect_viral_clips._parse_label("1"), 1)
        self.assertEqual(collect_viral_clips._parse_label("negative"), 0)
        self.assertEqual(collect_viral_clips._parse_label("weak"), 0)
        self.assertEqual(collect_viral_clips._parse_label("viral"), 1)

    def test_infer_platform_accepts_manifest_entry_dict(self) -> None:
        self.assertEqual(
            collect_viral_clips._infer_platform(
                {"url": "https://www.tiktok.com/@tarik/video/123", "platform": "tiktok"}
            ),
            "tiktok",
        )
        self.assertEqual(
            collect_viral_clips._infer_platform(
                {"url": "https://www.youtube.com/shorts/abc", "platform": ""}
            ),
            "youtube",
        )


if __name__ == "__main__":
    unittest.main()
