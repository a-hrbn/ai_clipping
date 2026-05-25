from __future__ import annotations

from pathlib import Path

from config import BACKEND_ROOT, settings


_DOCKER_BACKEND_ROOT = Path("/app/backend")


def apply_yt_dlp_auth_options(opts: dict[str, object]) -> dict[str, object]:
    cookiefile = str(settings.yt_dlp_cookiefile or "").strip()
    browser = str(settings.yt_dlp_cookies_from_browser or "").strip()
    if cookiefile:
        cookiefile = _prepare_cookiefile_path(cookiefile)
        if cookiefile:
            opts["cookiefile"] = cookiefile
    if browser and not cookiefile:
        opts["cookiesfrombrowser"] = _cookies_from_browser_tuple(browser)
    return opts


def _prepare_cookiefile_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            path = BACKEND_ROOT / path.relative_to(_DOCKER_BACKEND_ROOT)
        except ValueError:
            pass
    else:
        path = BACKEND_ROOT / path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return ""
    return str(path)


def _cookies_from_browser_tuple(value: str) -> tuple[str, ...]:
    parts = [part.strip() for part in value.split(":") if part.strip()]
    if not parts:
        return ("",)
    return tuple(parts)
