from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping


def resolve_secret_ref(secret_ref: str) -> str:
    payload = str(secret_ref).strip()
    if not payload:
        return ""
    return payload


def build_env(
    *,
    secret_bindings: Mapping[str, str] | None = None,
    value_bindings: Mapping[str, str] | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env) if base_env is not None else os.environ.copy()
    for name, value in (value_bindings or {}).items():
        resolved = str(value).strip()
        if resolved:
            env[str(name)] = resolved
    for name, secret_ref in (secret_bindings or {}).items():
        resolved = resolve_secret_ref(secret_ref)
        if resolved:
            env[str(name)] = resolved
    return env


@contextlib.contextmanager
def apply_env(
    *,
    secret_bindings: Mapping[str, str] | None = None,
    value_bindings: Mapping[str, str] | None = None,
) -> Iterator[dict[str, str]]:
    updated = build_env(secret_bindings=secret_bindings, value_bindings=value_bindings, base_env=os.environ)
    previous: dict[str, str | None] = {}
    for name, value in updated.items():
        previous.setdefault(name, os.environ.get(name))
        os.environ[name] = value
    try:
        yield updated
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
