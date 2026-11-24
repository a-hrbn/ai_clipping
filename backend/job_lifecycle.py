"""Job-Status-Vokabular — Source-of-Truth für die terminalen Stati.

Dieses Modul ist der einzige Owner der Antwort auf „welche Job-Stati sind
terminal?". Das `JobState`-`Literal` selbst bleibt der pydantic-Validierungstyp
in `models` (die bestehende Validierungs-seam); dieses Modul ergänzt es um das
Lifecycle-Prädikat, ersetzt es nicht.

Genau eine Regel wird erzwungen: terminal ist sticky. `job_manager.update_job`
ignoriert jedes Update an einem Job, der bereits terminal ist — sonst überschrieb
der nächste Fortschritts-Tick des Workers ein `cancelled` mit einem laufenden
Status und der Job lief weiter (#115). Nur `create_job` setzt einen Job für
einen neuen Lauf zurück.

Bewusst KEINE volle Transition-Tabelle / State-Machine: zwischen den laufenden
Stati ist jede Reihenfolge erlaubt.
"""

# Benannte terminal-Status-Konstanten. Ein Serving-Job erreicht genau einen
# dieser Stati höchstens einmal; danach läuft er nicht mehr.
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

# Single source of truth: Eviction, Cancel und Delete konsultieren alle dieses
# Set. Frozenset, weil das Vokabular unveränderlich ist — Aufrufer mutieren nie.
TERMINAL: frozenset[str] = frozenset({DONE, FAILED, CANCELLED})


def is_terminal(status: str | None) -> bool:
    """True iff `status` ein terminaler Job-Status ist (done/failed/cancelled)."""
    return status in TERMINAL
