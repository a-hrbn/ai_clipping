import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import collect_viral_clips
from services import downloader


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

    def test_collect_rows_pauses_and_retries_same_clip_after_transient_network_failure(self) -> None:
        entry = {
            "url": "https://www.tiktok.com/@tarik/video/123",
            "streamer": "tarik",
            "platform": "tiktok",
            "language": "en",
            "tags": "",
            "label": "1",
            "notes": "",
        }
        recovered_row = {
            "label": 1,
            "features": {
                "hook_strength": 0.5,
                "max_hype": 7.0,
                "arc_score": 1.2,
            },
        }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips._process_clip", side_effect=[
                 downloader.DownloadPausedError("Temporary failure in name resolution"),
                 recovered_row,
             ]) as process_mock, \
             patch("ml.collect_viral_clips.time.sleep") as sleep_mock:
            rows = collect_viral_clips.collect_rows([entry], "gaming", Path(tmp))

        self.assertEqual(rows, [recovered_row])
        self.assertEqual(process_mock.call_count, 2)
        sleep_mock.assert_called_once_with(30)


if __name__ == "__main__":
    unittest.main()
