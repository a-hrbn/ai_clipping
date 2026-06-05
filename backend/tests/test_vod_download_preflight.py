from __future__ import annotations

from pathlib import Path

from ml import vod_download_preflight as preflight


def test_ensure_vod_sources_uses_existing_cache_file(tmp_path: Path) -> None:
    source_root = tmp_path / "twitch"
    source_root.mkdir()
    source = source_root / "vod_123.mp4"
    source.write_bytes(b"video")

    summary = preflight.ensure_vod_sources(
        ["123"],
        video_source_root=source_root,
        download_missing=False,
        require_all=True,
    )

    assert summary["all_available"] is True
    assert summary["downloaded_count"] == 0
    assert summary["by_vod"]["123"]["source_video_path"] == str(source)


def test_ensure_vod_sources_downloads_missing_vod(tmp_path: Path) -> None:
    downloaded = tmp_path / "twitch" / "vod_456.mp4"
    downloaded.parent.mkdir()

    def fake_downloader(url: str, *, cache_namespace: str, cache_ttl_sec: int):
        assert url == "https://www.twitch.tv/videos/456"
        assert cache_namespace == "twitch"
        assert cache_ttl_sec == -1
        downloaded.write_bytes(b"video")
        return downloaded

    summary = preflight.ensure_vod_sources(
        ["456"],
        video_source_root=tmp_path / "twitch",
        download_missing=True,
        require_all=True,
        downloader=fake_downloader,
    )

    assert summary["all_available"] is True
    assert summary["downloaded_count"] == 1
    assert summary["by_vod"]["456"]["status"] == "downloaded"


def test_ensure_vod_sources_raises_when_required_download_fails(tmp_path: Path) -> None:
    def fake_downloader(*args, **kwargs):
        raise RuntimeError("unavailable")

    try:
        preflight.ensure_vod_sources(
            ["789"],
            video_source_root=tmp_path / "twitch",
            download_missing=True,
            require_all=True,
            downloader=fake_downloader,
        )
    except preflight.VodDownloadPreflightError as exc:
        assert exc.summary["failed_vod_ids"] == ["789"]
        assert exc.summary["by_vod"]["789"]["status"] == "download_failed"
    else:
        raise AssertionError("expected VodDownloadPreflightError")
