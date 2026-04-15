import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import chat_analyzer


class ChatAnalyzerDownloadTests(unittest.TestCase):
    def test_download_chat_accepts_twitch_json_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "chat"

            def fake_run(args, **kwargs):
                self.assertEqual(args[:3], [sys.executable, "-m", "yt_dlp"])
                self.assertIn("--sub-langs", args)
                self.assertIn("rechat,live_chat", args)
                self.assertIn("--sub-format", args)
                self.assertIn("json3/json", args)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "chat.rechat.json").write_text("[]", encoding="utf-8")
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            with patch.object(chat_analyzer.subprocess, "run", side_effect=fake_run), \
                 patch.object(chat_analyzer.logger, "info") as info_log:
                path = chat_analyzer.download_chat(
                    "https://www.twitch.tv/videos/123456",
                    "job-1",
                    out_dir,
                )

        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.name, "chat.rechat.json")
        info_messages = [call.args[0] for call in info_log.call_args_list]
        self.assertIn("Starting Twitch chat download for job %s: %s", info_messages)
        self.assertIn("Twitch chat download finished for job %s: %s", info_messages)

    def test_download_chat_treats_missing_twitch_rechat_as_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "chat"

            def fake_run(*_args, **_kwargs):
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 1,
                        "stdout": "",
                        "stderr": "ERROR: Unable to download video subtitles for 'rechat': HTTP Error 404: Not Found",
                    },
                )()

            with patch.object(chat_analyzer.subprocess, "run", side_effect=fake_run), \
                 patch.object(chat_analyzer.logger, "info") as info_log, \
                 patch.object(chat_analyzer.logger, "warning") as warning_log:
                path = chat_analyzer.download_chat(
                    "https://www.twitch.tv/videos/123456",
                    "job-2",
                    out_dir,
                )

        self.assertIsNone(path)
        warning_log.assert_not_called()
        info_messages = [call.args[0] for call in info_log.call_args_list]
        self.assertIn(
            "Twitch chat replay unavailable for job %s (exit=%s); continuing without chat enrichment",
            info_messages,
        )


if __name__ == "__main__":
    unittest.main()
