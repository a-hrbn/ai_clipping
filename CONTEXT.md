# ai_clipping

Full-Stack-App, die lange Twitch-VODs in gerankte Kurzclips verwandelt. Der Code
besteht aus zwei Hälften:

**1. Serving-Pipeline (online).** Erzeugt für einen einzelnen Job Clips aus einem VOD.
Fluss: `main.py` (FastAPI) → `routers/jobs.py` (`/api/jobs`-Endpoints) →
`job_manager.py` (Job-State + `jobs.json`, Cancellation, Session-Leases) →
`pipeline_worker.py` (isolierter Subprozess pro Job) → `run_pipeline` in
`pipeline.py` (der **Conductor**) → die **Producers** in `producers/` → gerankte Clips.

**2. Offline-ML / GT-Flywheel (offline).** Erzeugt die Ground-Truth-Labels, gegen die
der Clip-Finder bewertet und verbessert wird. Liegt komplett in `backend/ml/`.
Entry-Points: `ml/gt_pipeline.py` (orchestriert den Flywheel), `ml/gt_capture.py`
(täglicher Capture-Job), `ml/gt_match_verify.py` (wöchentliches gold-minting),
`ml/eval_clip_finder.py` (Bewertung gegen das Eval-Manifest). Shell-Wrapper in `scripts/`.

Das Frontend (`frontend/`, Next.js) reicht Jobs ein und zeigt Clips zur Review.

## Sprache

Arbeits- und Antwortsprache: **Deutsch**. Domänen-/Code-Begriffe (`Producer`,
`RichSegment`, `GT-Flywheel` …) bleiben englisch.

## Glossar

### Clip pipeline

**Conductor**: Der schlanke Orchestrator eines einzelnen Job-Laufs (`run_pipeline`).
Besitzt Job-Status-Übergänge, Fortschritts-Prozente, Cancellation-Checks, den
Chat-Prefetch und das Ressourcen-Teardown. Enthält selbst keine Clip-Finding-Logik
— er sequenziert Producers. _Avoid_: orchestrator, manager, runner.

**Terminal status**: Ein Job-Status, der einen abgeschlossenen Lauf markiert (`done`,
`failed`, `cancelled`); danach läuft der Job nicht mehr. Das Vokabular lebt an genau
einer Stelle — `job_lifecycle` (`TERMINAL` + `is_terminal()`); Eviction, Cancel und
Delete konsultieren dasselbe Prädikat. Bewusst ohne Transition-Enforcement: der
Conductor ist der einzige Schreiber von `status`, und `jobs.json` wird Cross-Process
neu geladen, sodass ein Guard legitime Catch-ups abweisen würde. _Avoid_: final state,
end status.

**Producer**: Ein deep module, das seine Inputs hinter einer kleinen Schnittstelle in
**einen** Pipeline-Output verwandelt (transcript, voice-energy annotation, chat
enrichment, moments, clips). Pur: liest nie globale Settings, fasst keinen Job-State
an; seine Leaf-Operationen (ffmpeg, Whisper, das LLM) sind injiziert, damit er isoliert
testbar ist. _Avoid_: stage, step, handler.

**Resource**: Ein Producer, der lazy und memoized ist statt fire-once — liefert seinen
Wert höchstens einmal, auch wenn mehrere Producers ihn ziehen. `AudioResource` (das
16-kHz-wav) ist der kanonische Fall: Transkription und voice-energy ziehen beide daran,
Audio wird höchstens einmal extrahiert, der Conductor räumt es ab. _Avoid_: provider,
cache (für diese Rolle).

**Cache source**: Die stabile Identität des Job-Mediums, genutzt als Cache-Key — z. B.
`twitch_vod:123`, `url:…`, `path:…`. `None` heißt: Caching ist für den Job aus.
_Avoid_: cache key, source id.

**Moment**: Ein gescorter, zeitlich begrenzter VOD-Abschnitt, der zum Clippen
vorgeschlagen wird (start/end + viral/composite scores + reason). Vom Moment-Finder
erzeugt; die Top-K Moments werden zu Clips. _Avoid_: highlight, segment.

**RichSegment**: Die angereicherte Transkript-Einheit (text + start/end + Per-Signal-
Scores wie voice-energy burst und chat spike). Die geteilte Timeline-Währung, die die
Clip-Finding-Producers annotieren. _Avoid_: segment (unqualifiziert), chunk.

### Offline-ML / Eval

**GT-Flywheel** (Ground-Truth-Flywheel): Der Offline-Kreislauf, der echte Eval-/
Trainings-Labels prägt. @Iboclipss-**Reposts** werden gegen gecapturte Jynxzi-VODs
gematcht (Transcript primär, Audio-Fingerprint als Fallback), um „Gold"-Momente zu
gewinnen. Capture läuft täglich (`ml/gt_capture.py`), Match+Verify wöchentlich
(`ml/gt_match_verify.py`); Orchestrierung in `ml/gt_pipeline.py`. _Avoid_: training
loop, data pipeline.

**Eval-Manifest** (Gold-Eval-Manifest): Der vom GT-Flywheel geprägte Ground-Truth-
Datensatz bekannt-guter Momente (jsonl). Maßstab, gegen den der Clip-Finder bewertet
wird — ein Treffer deckt ≥ 50 % des GT-Fensters ab (`ml/eval_clip_finder.py`).
_Avoid_: ground truth file, labels.
