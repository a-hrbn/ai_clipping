import unittest
from unittest.mock import patch

from services import yt_dlp_auth


class YtDlpAuthTests(unittest.TestCase):
    def test_applies_cookiefile_when_configured(self) -> None:
        opts: dict[str, object] = {}

        with patch.object(yt_dlp_auth.settings, "yt_dlp_cookiefile", "/tmp/youtube.cookies.txt"), \
             patch.object(yt_dlp_auth.settings, "yt_dlp_cookies_from_browser", "chrome:Default"):
            yt_dlp_auth.apply_yt_dlp_auth_options(opts)

        self.assertEqual(opts["cookiefile"], "/tmp/youtube.cookies.txt")
        self.assertNotIn("cookiesfrombrowser", opts)

    def test_applies_browser_cookies_when_cookiefile_is_not_configured(self) -> None:
        opts: dict[str, object] = {}

        with patch.object(yt_dlp_auth.settings, "yt_dlp_cookiefile", ""), \
             patch.object(yt_dlp_auth.settings, "yt_dlp_cookies_from_browser", "firefox:default-release"):
            yt_dlp_auth.apply_yt_dlp_auth_options(opts)

        self.assertEqual(opts["cookiesfrombrowser"], ("firefox", "default-release"))

    def test_leaves_options_unchanged_without_auth_config(self) -> None:
        opts: dict[str, object] = {"quiet": True}

        with patch.object(yt_dlp_auth.settings, "yt_dlp_cookiefile", ""), \
             patch.object(yt_dlp_auth.settings, "yt_dlp_cookies_from_browser", ""):
            yt_dlp_auth.apply_yt_dlp_auth_options(opts)

        self.assertEqual(opts, {"quiet": True})
