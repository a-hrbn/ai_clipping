"""Session-Lease-State für Jobs (Kandidat B aus dem Architecture-Review, Slice B1).

Ein Browser-Tab hält „seinen" Job am Leben, indem er pro Status-Poll die Lease
`touch`t; läuft die Lease ab (Tab geschlossen), darf der verwaiste Job gecancelt
werden. Dieses Modul **besitzt** den Lease-State und die reine Lease-Logik
(`bind` / `touch` / `clear` / `is_expired`). Es kennt den Job-Store nicht — der
`job_manager` orchestriert (reload/evict/Job-Lookup) und ruft diese Funktionen.

Locking: Dieser Modul-State hat **keinen eigenen Lock**. Alle Mutatoren werden
ausschließlich vom `job_manager` aufgerufen, während dieser seinen `_lock` hält —
die Lease teilt damit dieselbe Lock-Disziplin wie der Job-State (keine neuen
Races). Niemand ruft diese Funktionen ohne den `job_manager._lock`.

Zeitquelle: `now()` liest über die überschreibbare `_time_source`, damit Expiry
ohne `sleep` testbar ist. Per Default ist das `time.monotonic` — identisch zur
früheren Inline-Logik im `job_manager`.
"""
import time
from typing import Callable, Dict

# Lease-State — Eigentum dieses Moduls. Der job_manager re-exportiert diese Dicts
# per Referenz (Test-/Aufrufer-Kompatibilität); daher NIE neu binden, nur mutieren.
_job_session_id: Dict[str, str] = {}
_job_session_expires_at: Dict[str, float] = {}

# Überschreibbare Zeitquelle — injizierbarer Seam für Expiry-Tests ohne sleep.
# Default time.monotonic: verhaltens-identisch zur früheren Inline-Uhr.
_time_source: Callable[[], float] = time.monotonic

# Server-seitige Lease-Lebensdauer. Browser-Tabs drosseln Timer/Polling im
# Hintergrund stark; lang genug, dass solche Tabs nicht als getrennt gelten,
# kurz genug, dass verwaiste Jobs stoppen.
_SESSION_TTL_SECONDS = 300.0


def now() -> float:
    """Aktuelle Lease-Zeit über die injizierbare Zeitquelle."""
    return _time_source()


def bind(job_id: str, session_id: str) -> None:
    """Binde einen Job an eine Session und starte die Lease-Frist neu.

    Vorbedingung (vom job_manager geprüft): der Job existiert und session_id ist
    nicht leer.
    """
    _job_session_id[job_id] = session_id
    _job_session_expires_at[job_id] = now() + _SESSION_TTL_SECONDS


def touch(job_id: str, session_id: str | None) -> bool:
    """Erneuere die Lease, wenn die Session zum gebundenen Job passt.

    Gibt True zurück, wenn erneuert wurde, sonst False (No-Op bei fehlender oder
    nicht passender Session).
    """
    if session_id and _job_session_id.get(job_id) == session_id:
        _job_session_expires_at[job_id] = now() + _SESSION_TTL_SECONDS
        return True
    return False


def is_expired(job_id: str) -> bool:
    """True, wenn der Job eine gebundene Lease hat, die abgelaufen ist."""
    session_id = _job_session_id.get(job_id)
    expires_at = _job_session_expires_at.get(job_id)
    return bool(session_id) and expires_at is not None and now() > expires_at


def clear(job_id: str) -> None:
    """Räume den Lease-State eines Jobs (Eviction/Delete/Cancel-Hook)."""
    _job_session_id.pop(job_id, None)
    _job_session_expires_at.pop(job_id, None)
