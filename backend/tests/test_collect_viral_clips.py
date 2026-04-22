import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from ml import collect_viral_clips
from services import downloader


class CollectViralClipsTests(unittest.TestCase):
    def test_cut_video_windows_uses_single_ffmpeg_process_within_batch_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips.subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")) as run_mock:
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / "window_0001" / "window.mp4", 10.0, 20.0),
                (Path(tmp) / "window_0002" / "window.mp4", 30.0, 45.0),
            ]

            result = collect_viral_clips.cut_video_windows(video_path, outputs)

        self.assertEqual(result, [item[0] for item in outputs])
        command = run_mock.call_args.args[0]
        self.assertEqual(command[:4], ["ffmpeg", "-y", "-nostdin", "-loglevel"])
        self.assertEqual(command.count("-i"), 2)
        self.assertIn(str(outputs[0][0]), command)
        self.assertIn(str(outputs[1][0]), command)
        self.assertEqual(run_mock.call_count, 1)

    def test_cut_video_windows_splits_large_batches_across_multiple_ffmpeg_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ml.collect_viral_clips.subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")) as run_mock:
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / f"window_{index:04d}" / "window.mp4", float(index * 10), float(index * 10 + 8))
                for index in range(1, collect_viral_clips._MAX_BATCH_CUT_WINDOWS + 3)
            ]

            result = collect_viral_clips.cut_video_windows(video_path, outputs)

        self.assertEqual(result, [item[0] for item in outputs])
        self.assertEqual(run_mock.call_count, 2)
        first_command = run_mock.call_args_list[0].args[0]
        second_command = run_mock.call_args_list[1].args[0]
        self.assertEqual(first_command.count("-i"), collect_viral_clips._MAX_BATCH_CUT_WINDOWS)
        self.assertEqual(second_command.count("-i"), 2)
        self.assertIn(str(outputs[0][0]), first_command)
        self.assertIn(str(outputs[-1][0]), second_command)

    def test_cut_video_windows_retries_failed_batch_by_splitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / f"window_{index:04d}" / "window.mp4", float(index * 10), float(index * 10 + 8))
                for index in range(1, 5)
            ]

            def _fake_run(command, **_kwargs):
                input_count = command.count("-i")
                if input_count == 4:
                    return SimpleNamespace(returncode=1, stderr="pthread_create failed")
                return SimpleNamespace(returncode=0, stderr="")

            with patch("ml.collect_viral_clips.subprocess.run", side_effect=_fake_run) as run_mock:
                result = collect_viral_clips.cut_video_windows(video_path, outputs)

        self.assertEqual(result, [item[0] for item in outputs])
        self.assertEqual(run_mock.call_count, 3)
        self.assertEqual(run_mock.call_args_list[0].args[0].count("-i"), 4)
        self.assertEqual(run_mock.call_args_list[1].args[0].count("-i"), 2)
        self.assertEqual(run_mock.call_args_list[2].args[0].count("-i"), 2)

    def test_cut_video_windows_raises_when_single_window_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch(
                 "ml.collect_viral_clips.subprocess.run",
                 return_value=SimpleNamespace(returncode=1, stderr="decoder blew up"),
             ):
            video_path = Path(tmp) / "vod.mp4"
            outputs = [
                (Path(tmp) / "window_0001" / "window.mp4", 10.0, 20.0),
            ]

            with self.assertRaisesRegex(RuntimeError, "decoder blew up"):
                collect_viral_clips.cut_video_windows(video_path, outputs)

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

    def test_read_entries_dedupes_cross_source_urls_by_canonical_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clips.csv"
            path.write_text(
                "\n".join([
                    "url,streamer,platform,language,tags,label,notes",
                    "https://www.youtube.com/shorts/abc?feature=share,jynxzi,youtube,en,gaming,1,first",
                    "https://www.youtube.com/shorts/abc,jynxzi,youtube,en,gaming,0,duplicate",
                    "https://www.tiktok.com/@jynxzi/video/123/,jynxzi,tiktok,en,gaming,1,unique",
                ]),
                encoding="utf-8",
            )

            entries = collect_viral_clips._read_entries(path)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["notes"], "first")

    def test_canonical_entry_url_drops_query_and_trailing_slash(self) -> None:
        self.assertEqual(
            collect_viral_clips._canonical_entry_url("https://www.youtube.com/shorts/abc/?feature=share"),
            "https://www.youtube.com/shorts/abc",
        )

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
