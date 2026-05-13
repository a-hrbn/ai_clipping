from __future__ import annotations

import hashlib
import json
import pathlib

from config import settings

_MODEL_VERSION = "local_heuristic_v1"


def score_hook(first_seconds_text: str, *, model_version: str = _MODEL_VERSION) -> dict[str, object]:
    text = " ".join(str(first_seconds_text or "").split())
    key = _cache_key(text, model_version)
    path = pathlib.Path(settings.storage_dir) / "hook_rewrite" / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)

    lower = text.lower()
    curiosity_hits = sum(token in lower for token in ("wait", "what", "why", "how", "no way", "?"))
    surprise_hits = sum(token in lower for token in ("bro", "insane", "crazy", "look", "watch"))
    score = min(10.0, 2.0 + curiosity_hits * 2.0 + surprise_hits * 1.25 + min(len(text), 80) / 40.0)
    gap = min(10.0, curiosity_hits * 3.0 + ("?" in text) * 2.0)
    payload = {
        "score": round(score, 3),
        "gap": round(gap, 3),
        "reason": "local_fail_open_heuristic",
        "provider": "local",
        "model_version": model_version,
        "llm_used": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return payload


def _cache_key(text: str, model_version: str) -> str:
    payload = f"{model_version}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
