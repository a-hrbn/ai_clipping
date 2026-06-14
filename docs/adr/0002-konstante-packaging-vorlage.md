# 2. Konstante Packaging-Vorlage als erster Forward-Loop-Build (fixed-policy/content-derived Rendering)

Status: Accepted — 2026-06-14

## Kontext

ADR 0001 hat auf den First-party-Outcome-Flywheel (Harvesting-Loop) gepivotet. Dessen
**erste Build-Komponente** ist die *konstante Packaging-Vorlage*: der Renderer, der einen
**Moment** (Quell-VOD + Zeitspanne) in einen fertigen, postbaren 9:16-Short verwandelt.

Der Nordstern für diesen Build ist **Streamer-Spezialisierung, nicht ein generischer
Clipper**. Das schärft, *was* der Renderer ist und *was nicht*:

- **Packaging ist Commodity.** 9:16-Reframe, Captions, Auto-Titel, konstantes Format sind
  exakt OpusClips' Stärke. Der Renderer ist notwendige Infrastruktur — er differenziert
  nicht.
- **Der Moat ist das Lernen.** Die einzig nicht-widerlegte Spezialisierung (ADR 0001) ist
  der First-party-Outcome-Flywheel: Jynxzi-spezifische Retention, an **Facetten** gemessen.
  Damit der Flywheel überhaupt lernen kann, muss die Retention-Streuung dem **Moment**
  zurechenbar bleiben — also darf das Packaging kein *unabhängiger* Hebel sein.

Empirische Grundlage (Sample-Short + zwei Quell-VODs inspiziert, 2026-06-14): die 52
@donni_clips-Shorts nutzen ein **konstantes „Reaction-Stack"-Format** (Facecam oben,
Gameplay unten, word-level Karaoke-Captions an der Naht, Hook-Titel im Intro). In den
Quell-VODs sitzt Jynxzis Facecam an **fester Position** (unten-links), konstant über
völlig verschiedene Inhalte (React-Stream wie Rocket League).

## Entscheidung

Die konstante Packaging-Vorlage wird als **committetes Offline-Modul** gebaut: ein
`ml/`-Orchestrator (CLI wie `historical_short_mine`/`pull_yt_retention_curve`), der
injizierte Leaf-Ops in `services/` komponiert — `render` (ffmpeg crop+stack+caption),
`transcribe`/words (bestehender Seam), `make_title` (LLM). Fakebar; Tests fassen nie
ffmpeg/LLM/Netz an.

Tragende Mechanik-Entscheidungen (gegrillt 2026-06-14):

- **Reframe = fester Facecam-Crop (Single-Streamer), bewusst *kein* generisches
  Face-Tracking.** Das **korrigiert ADR 0001s wörtliches „9:16 Face-Track"**. Gründe: (a)
  generisches Face-Tracking ist genau OpusClips' Commodity-Verhalten und zieht einen
  schweren CV-Dep ein; (b) Jynxzis festes Layout macht einen hartcodierten Crop machbar —
  ein Stück Spezialisierung, das ein generisches Tool nicht hartcodieren kann; (c) ein
  fester Crop *hält das Packaging konstant*, ein Face-Track bewegte den Ausschnitt pro
  Clip.
- **Leitprinzip: feste Policy, content-abgeleiteter Output.** Jedes Packaging-Element ist
  eine *deterministische Funktion des Moments*: Crop (feste Geometrie), Captions (fester
  Stil, Wörter aus dem Transkript), Titel (fester Prompt, Text aus dem Moment). Damit ist
  keines ein *unabhängiger* Hebel → Retention bleibt dem Moment (und seiner Facette)
  zurechenbar — die Größe, aus der der Flywheel lernt.
- **Pro-Clip *handgeschriebene* Titel werden verworfen.** Sie injizieren menschliches
  Können unabhängig vom Moment (Confound genau bei `retention_3s`) und skalieren nicht für
  einen Autopilot (~180 Shorts/Monat). Auto-Titel aus fester Policy erfüllt *beides*:
  Autopilot-Pflicht und Lern-Disziplin.
- **Der Renderer emittiert Moment-/Facetten-/Quell-Metadaten** (stabiler lokaler Key aus
  `vod_id`+Zeitspanne), damit der First-party-Outcome-Flywheel den Outcome später *zurück
  auf den Moment und seine Facette* attribuieren kann. Diese Naht koppelt den
  Commodity-Renderer an den Moat — sie ist der Zweck, nicht ein Nice-to-have.
- **Crop-Geometrie als *relative* Coords in Config** (auflösungsunabhängig: Quelle 720p,
  Serving evtl. 1080p; anpassbar, falls das Layout übers Fenster driftet).
- **Slice-1-Grenze:** der Renderer burnt einen *übergebenen* `title: str` (konstanter
  Stil); die Auto-Titel-Policy (fester Prompt → Titel) ist Slice 2 — hält Slice 1 auf das
  novel ffmpeg-Rendering fokussiert.

## Konsequenzen

**Positiv:**

- Der Renderer ist billig auf bestehenden Seams gebaut (transcribe, ffmpeg-Leaf,
  Job-Layout) und bleibt offline, ohne den Serving-Pfad zu berühren.
- Die Spezialisierung lebt dort, wo sie hingehört — in der (späteren) Messung/Ernte über
  Facetten —, und der Renderer hält sie via fixed-policy-Rendering + Metadaten-Naht offen.
- „Auto-alles" macht den Autopilot tatsächlich autonom.

**Negativ / Kosten:**

- Ein fester Crop bricht, falls Jynxzi sein Layout ändert (gemildert durch Config /
  relative Coords; eine billige Validierung wäre nachrüstbar).
- Auto-Titel sind vermutlich schwächer als handgeschriebene — bewusst akzeptiert:
  Konsistenz/Attribution > Per-Clip-Politur.
- Man verzichtet auf die Per-Clip-Optimierung, die ein reiner „mach jeden Short maximal
  gut"-Produktreflex wollte; genau diese Disziplin hält den Loop aber lernbar.

**Verworfene Alternativen:**

- *Generisches Face-Tracking* — Commodity (OpusClips), CV-Dep, führt Per-Clip-Variation
  ein.
- *Pro-Clip handgeschriebene Titel* — Confound bei `retention_3s`, skaliert nicht.
- *Renderer als Serving-Producer* — verheiratete den Serving-Pfad mit der
  Offline-Content-Op; die Schleife ist keine Per-Job-Serving-Sorge.
- *Die Messung/Attribution ganz fallen lassen* („nur das Produkt bauen") — würde das
  Einzige streichen, was spezialisiert, und das Produkt zu einem OpusClips-Klon machen.
  Gegen den Nordstern verworfen.

## Abgrenzung

Dieses ADR gibt den **Moment-Moat nicht auf**. Die schwer-umkehrbare „Moment-vs-Packaging"-
Entscheidung bleibt ADR 0001s *sauberem* Cold-Start-Gate vorbehalten (der vorab fixierte
~180-Post-Vergleich best-vs-worst Goal-Family) — nicht den konfundierten Vorab-Blicken
(#75 Step 0, #80 Step 0.5). Lineage: **WALL (#69) → ADR 0001 → #75/#80 (Pre-Looks) →
dieses ADR (erster Forward-Loop-Build)**.

ADR 0001 wird bewusst *nicht* umgeschrieben (ADRs sind append-only); die „Face-Track"-
Verfeinerung ist hier dokumentiert und supersedet die Formulierung dort.
