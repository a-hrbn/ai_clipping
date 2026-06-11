"""Job-Status-Vokabular — Source-of-Truth für die terminalen Stati.

Dieses Modul ist der einzige Owner der Antwort auf „welche Job-Stati sind
terminal?". Das `JobState`-`Literal` selbst bleibt der pydantic-Validierungstyp
in `models` (die bestehende Validierungs-seam); dieses Modul ergänzt es um das
Lifecycle-Prädikat, ersetzt es nicht.

Bewusst KEINE Transition-Tabelle und KEINE Enforcement: der Conductor ist der
einzige Schreiber von `status` (Invariante durch Konstruktion gehalten),
`update_job` läuft überwiegend mit Nicht-Status-Kwargs, und `jobs.json` wird
Cross-Process neu geladen — ein Guard würde legitime Catch-ups abweisen. Es
existiert kein beobachteter Illegal-Transition-Bug.

Eskalationspfad (dokumentiert, nicht implementiert): Sollten illegale Übergänge
je real auftreten, ist der nächste Schritt *advisory logging* — Übergänge
protokollieren, nie raisen — und nicht eine erzwingende State-Machine.
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
