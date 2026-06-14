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

## Produkt-These & Nordstern

**Produkt:** ein **Autopilot**-Viral-Clip-Finder, spezialisiert auf *einen* Streamer
(Test-Streamer: **Jynxzi**). Differentiator gegenüber generischen Findern (OpusClips)
ist **Streamer-Spezialisierung** — aber *nicht* als besseres Content-Verständnis: vier
Diagnose-Tracks haben belegt, dass virale Diskriminierung aus dem VOD-Content tot ist
(Ranking, Generation, generische Kontext-Virality, und zuletzt clipper-spezifische
Kontext-Ähnlichkeit als Text-Signal → Blind-Sheet 0/10 → **WALL**, #69). _Offen (#90):_
diese Text-Signal-Evidenz lief auf einem qualitativ schwachen, sprecher-losen Transkript;
ein Re-Test auf High-Fidelity-Transkript prüft, ob die WALL ein Transkript-Artefakt ist —
bis das Verdikt vorliegt, ist sie Arbeits-Annahme, nicht geschlossener Beweis. Die
Spezialisierung überlebt nur als **First-party-Outcome-Flywheel**: Jynxzi-spezifische
Retention-Daten aus *eigenen* geposteten Clips, die kein generischer Finder sammelt.

**Mechanismus:** nach WALL ist der Autopilot kein Pre-Publication-Ranker, sondern ein
**Harvesting-Loop** — der Generator liefert ein *diverses* Kandidaten-Set (gute
Abdeckung statt Ranking), das gepostet, am First-party-Outcome gemessen und dessen
Gewinner geerntet/skaliert werden. Der „Ranker" ist der Markt.

**Nordstern:** Retention/Performance der *selbst geposteten* Clips (nicht mehr recall@20
gegen @Iboclipss). Der gesamte Offline-Apparat (recall@20, Gold-Manifest, GT-Flywheel)
ist auf **Bootstrap-Saat + Generator-Regressions-Wächter** demoted, nicht gelöscht.

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

**Job-Artifact-Store**: Der einzige Owner des On-Disk-Job-Layouts — der Artefakt-
**Namen** (`segments.jsonl`, `final_moments.json`, die `llm_moment_finder_*`-Dumps,
`request.json`, `progress.json` …) UND der json/jsonl/text-I/O. Aufrufer übergeben nur
die `job_id` und eine Artefakt-Rolle, nie selbst einen Pfad; umbenennen passiert an
genau einer Stelle, sodass Schreiber und Leser nicht auseinanderlaufen. Zwei Adapter:
`RealFsJobArtifactStore` (prod) und `InMemoryJobArtifactStore` (Test-Fake). Kennt nur
Pfade + I/O — die `jobs.json`-Registry-Mechanik bleibt im `job_manager`. _Avoid_:
artifact manager, file store.

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

### Outcome-Signal

**Third-party outcome**: Publikums-Performance an *fremden* geposteten Clips — heute
@Iboclipss-/andere Jynxzi-Kanal-View-Counts, die der GT-Flywheel auf die Quell-VODs
matcht. Nur sichtbar für Momente, die ein Fremd-Clipper *gewählt* hat (kein Signal über
dessen übersprungene Kandidaten), und auf ~View-Count reduziert. _Avoid_: engagement,
virality (unqualifiziert).

**First-party outcome**: Publikums-Performance an *eigenen* geposteten Clips — die
Retention-Kurve (`retention_3s`/`retention_50pct`/`completion`) plus Views-vs-Baseline,
wie sie `import_yt_retention` zu Outcome-Rows verlabelt. Reicher als third-party
(Abbruchkurve statt nur View-Count) und misst *selbst gewählte Kandidaten inkl.
Verlierer*. _Avoid_: engagement, performance (unqualifiziert).

**Harvesting-Loop** (Test-and-Learn): der Produktkern nach WALL. Statt Virality *vor dem
Posten* vorherzusagen (Content-Ranking, widerlegt), wird ein diverses Kandidaten-Set
gepostet, am First-party-Outcome gemessen und die Gewinner geerntet/skaliert — der
„Ranker" ist der Markt. Der Generator liefert *Abdeckung* (`raw@any`), nicht Reihenfolge.
_Avoid_: ranker, predictor (für diese Rolle).

**Facette** (Stratum): eine grobe, *berechenbare* Kategorie eines Moments (Goal-Family
via `infer_goal_family` aus dem Transkript des Shorts, Längen-Bucket), über die der
Harvesting-Loop stratifiziert sampelt (`derive_facets`). Chat-Emotion ist bewusst
**ausgeschlossen** — ein VOD-only-Signal, das im geposteten Short nicht verfügbar ist.
WALL-konform: keine feinkörnige Virality-Vorhersage, nur *Abdeckung* — die am Outcome
gemessene **Gewinner-Facette IST** das streamer-spezifische, lernbare Moat-Wissen, das
kein generischer Finder hat. _Avoid_: cluster, bucket (unqualifiziert), label.

**Packaging**: Die *Aufmachung* eines geposteten Shorts als Outcome-Hebel — alles
außer der Moment-Wahl: Reframe-Layout, Caption-Stil, Hook-Titel. Die Gegen-Größe, gegen
die der **Moment** im Harvesting-Loop disentangled wird ("ist Moment-Auswahl überhaupt
mein Hebel, oder dominiert Packaging?"). _Avoid_: presentation, style (unqualifiziert).

**Packaging-Vorlage** (constant packaging template): Das über einen Forward-Loop-Batch
*konstant gehaltene* Packaging, damit Retention-Streuung dem **Moment** zurechenbar ist
statt der Aufmachung — das Herz des Moment-vs-Packaging-Disentanglements. Ihre kanonische
Form ist aus den bestehenden 52 @donni_clips-Shorts abgeleitet (ein vertikaler
„Reaction-Stack": Facecam über Gameplay). Bewusst *kein* generisches Face-Tracking — das
führte pro-Clip-Variation ein und wäre selbst ein Packaging-Confound. Welcher Crop/Font/
Titel das konkret ist, ist Implementierung und lebt nicht hier. _Avoid_: template, format,
layout (unqualifiziert).
