from __future__ import annotations

import json
import pathlib
import posixpath
import shutil
import tempfile
from contextlib import contextmanager
from urllib.parse import urlparse

import fsspec


def is_uri(value: str) -> bool:
    parsed = urlparse(str(value))
    return bool(parsed.scheme and parsed.scheme != "file")


def normalize_uri(value: str) -> str:
    text = str(value).strip()
    if text.startswith("file://"):
        return urlparse(text).path
    return text


def _url_to_fs(path_or_uri: str):
    return fsspec.core.url_to_fs(normalize_uri(path_or_uri))


def ensure_parent(path_or_uri: str) -> None:
    fs, path = _url_to_fs(path_or_uri)
    parent = posixpath.dirname(path) if is_uri(path_or_uri) else str(pathlib.Path(path).parent)
    if parent:
        fs.makedirs(parent, exist_ok=True)


def write_text(path_or_uri: str, payload: str) -> None:
    ensure_parent(path_or_uri)
    fs, path = _url_to_fs(path_or_uri)
    with fs.open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)


def read_text(path_or_uri: str) -> str:
    fs, path = _url_to_fs(path_or_uri)
    with fs.open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def write_json(path_or_uri: str, payload: dict | list) -> None:
    write_text(path_or_uri, json.dumps(payload, ensure_ascii=True, indent=2))


def read_json(path_or_uri: str) -> dict | list:
    return json.loads(read_text(path_or_uri))


def write_jsonl(path_or_uri: str, rows: list[dict]) -> None:
    write_text(path_or_uri, "\n".join(json.dumps(row, ensure_ascii=True) for row in rows))


def read_jsonl(path_or_uri: str) -> list[dict]:
    try:
        payload = read_text(path_or_uri)
    except FileNotFoundError:
        return []
    rows: list[dict] = []
    for line in payload.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def path_exists(path_or_uri: str) -> bool:
    fs, path = _url_to_fs(path_or_uri)
    return fs.exists(path)


def copy_file(local_path: pathlib.Path, destination_uri: str) -> None:
    ensure_parent(destination_uri)
    fs, path = _url_to_fs(destination_uri)
    fs.put_file(str(local_path), path)


def materialize_file(source_uri: str, local_path: pathlib.Path) -> pathlib.Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fs, path = _url_to_fs(source_uri)
    fs.get_file(path, str(local_path))
    return local_path


def copy_directory(local_dir: pathlib.Path, destination_uri: str) -> None:
    destination_uri = normalize_uri(destination_uri)
    if not is_uri(destination_uri):
        target = pathlib.Path(destination_uri)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(local_dir, target)
        return

    fs, base_path = _url_to_fs(destination_uri)
    fs.makedirs(base_path, exist_ok=True)
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        remote_path = posixpath.join(base_path, relative)
        remote_parent = posixpath.dirname(remote_path)
        if remote_parent:
            fs.makedirs(remote_parent, exist_ok=True)
        fs.put_file(str(path), remote_path)


def materialize_directory(source_uri: str, local_dir: pathlib.Path) -> pathlib.Path:
    source_uri = normalize_uri(source_uri)
    if not is_uri(source_uri):
        source = pathlib.Path(source_uri)
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.copytree(source, local_dir)
        return local_dir

    fs, base_path = _url_to_fs(source_uri)
    local_dir.mkdir(parents=True, exist_ok=True)
    for remote_path in fs.find(base_path):
        relative = remote_path[len(base_path):].lstrip("/")
        destination = local_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(remote_path, str(destination))
    return local_dir


@contextmanager
def temporary_work_dir(prefix: str):
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield pathlib.Path(tmp)
