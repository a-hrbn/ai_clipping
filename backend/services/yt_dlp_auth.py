from __future__ import annotations

from pathlib import Path

from config import settings


def apply_yt_dlp_auth_options(opts: dict[str, object]) -> dict[str, object]:
    cookiefile = str(settings.yt_dlp_cookiefile or "").strip()
    browser = str(settings.yt_dlp_cookies_from_browser or "").strip()
    if cookiefile:
        opts["cookiefile"] = str(Path(cookiefile).expanduser())
    if browser and not cookiefile:
        opts["cookiesfrombrowser"] = _cookies_from_browser_tuple(browser)
    return opts


def _cookies_from_browser_tuple(value: str) -> tuple[str, ...]:
    parts = [part.strip() for part in value.split(":") if part.strip()]
    if not parts:
        return ("",)
    return tuple(parts)
