from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_FILE_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".ts", ".m4v"}


@dataclass
class CleanupResult:
    removed_paths: list[str]
    failed_paths: list[str]

    @property
    def removed_count(self) -> int:
        return len(self.removed_paths)

    @property
    def failed_count(self) -> int:
        return len(self.failed_paths)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["removed_count"] = self.removed_count
        data["failed_count"] = self.failed_count
        return data


def cleanup_backend_runtime_caches(
    backend_root: Path = BACKEND_ROOT,
    *,
    dry_run: bool = False,
) -> CleanupResult:
    backend_root = backend_root.resolve()
    result = CleanupResult(removed_paths=[], failed_paths=[])

    tmp_dir = backend_root / "tmp"
    storage_dir = tmp_dir / "storage"
    short_form_dir = tmp_dir / "short_form_training"

    for path in _storage_runtime_artifacts(storage_dir):
        _remove_path(path, backend_root, result, dry_run=dry_run)

    for path in _short_form_runtime_artifacts(short_form_dir):
        _remove_path(path, backend_root, result, dry_run=dry_run)

    _remove_path(tmp_dir / "short_form_backfill_artifacts", backend_root, result, dry_run=dry_run)
    _remove_empty_legacy_cache_dir(tmp_dir / "cache_deepface", backend_root, result, dry_run=dry_run)

    return result


def _storage_runtime_artifacts(storage_dir: Path) -> list[Path]:
    if not storage_dir.exists():
        return []

    artifacts: list[Path] = []
    for path in storage_dir.iterdir():
        name = path.name
        if path.is_dir() and name.startswith("_cache_fill_"):
            artifacts.append(path)
            continue
        if path.is_dir() and _is_safe_to_remove_twitch_job_dir(path, storage_dir):
            artifacts.append(path)
            continue
        if path.is_dir() and _is_safe_to_remove_legacy_short_form_job_dir(path, storage_dir):
            artifacts.append(path)
            continue
        if path.is_file() and (
            name.endswith(".tmp")
            or ".part" in name
            or name.endswith(".aria2")
            or name.endswith(".ytdl")
        ):
            artifacts.append(path)
    return artifacts


def _is_safe_to_remove_twitch_job_dir(path: Path, storage_dir: Path) -> bool:
    match = re.fullmatch(r"twitch_vod(?:_prefetch)?_(\d+)", path.name)
    if not match:
        return False

    video_path = path / "video.mp4"
    if not video_path.exists():
        return True

    cache_path = storage_dir / "_download_cache" / "twitch" / f"vod_{match.group(1)}.mp4"
    if not cache_path.exists():
        return False

    try:
        return cache_path.stat().st_size == video_path.stat().st_size
    except OSError:
        return False


def _is_safe_to_remove_legacy_short_form_job_dir(path: Path, storage_dir: Path) -> bool:
    if not re.fullmatch(r"viral_[0-9a-fA-F]+", path.name):
        return False

    final_media = [
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and candidate.suffix.lower() in _VIDEO_FILE_SUFFIXES
        and ".part" not in candidate.name
        and not candidate.name.endswith((".aria2", ".ytdl", ".tmp"))
        and not candidate.name.endswith(".temp.mp4")
    ]
    if not final_media:
        return True

    short_form_cache = storage_dir / "_download_cache" / "short_form"
    if not short_form_cache.exists():
        return False
    cached_media = _cached_media_by_size(short_form_cache)
    if not cached_media:
        return False
    try:
        return all(_media_exists_in_cache(media, cached_media) for media in final_media)
    except OSError:
        return False


def _cached_media_by_size(cache_dir: Path) -> dict[int, list[Path]]:
    media_by_size: dict[int, list[Path]] = {}
    for path in cache_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _VIDEO_FILE_SUFFIXES:
            continue
        try:
            media_by_size.setdefault(int(path.stat().st_size), []).append(path)
        except OSError:
            continue
    return media_by_size


def _media_exists_in_cache(media_path: Path, cached_media: dict[int, list[Path]]) -> bool:
    size = int(media_path.stat().st_size)
    for cached_path in cached_media.get(size, []):
        try:
            if filecmp.cmp(media_path, cached_path, shallow=False):
                return True
        except OSError:
            continue
    return False


def _short_form_runtime_artifacts(short_form_dir: Path) -> list[Path]:
    if not short_form_dir.exists():
        return []
    return [
        path
        for path in short_form_dir.iterdir()
        if path.is_dir() and path.name.startswith("viral_")
    ]


def _remove_empty_legacy_cache_dir(
    path: Path,
    backend_root: Path,
    result: CleanupResult,
    *,
    dry_run: bool,
) -> None:
    if not path.exists():
        return
    try:
        next(path.iterdir())
    except StopIteration:
        _remove_path(path, backend_root, result, dry_run=dry_run)
    except OSError:
        result.failed_paths.append(_display_path(path, backend_root))


def _remove_path(path: Path, backend_root: Path, result: CleanupResult, *, dry_run: bool) -> None:
    if not path.exists():
        return
    display_path = _display_path(path, backend_root)
    if dry_run:
        result.removed_paths.append(display_path)
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        result.failed_paths.append(display_path)
        return
    result.removed_paths.append(display_path)


def _display_path(path: Path, backend_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(backend_root))
    except ValueError:
        return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean per-run backend cache artifacts without removing persistent caches.")
    parser.add_argument("--backend-root", default=str(BACKEND_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = cleanup_backend_runtime_caches(Path(args.backend_root), dry_run=bool(args.dry_run))
    print(json.dumps({"event": "backend_cache_cleanup", **result.to_dict()}, ensure_ascii=True))
    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
