# Runbook — Archetype-Judge Paid-Lauf + Verdikt + Render + Frontend-Review (Issue #101, PRD #95)

**Status:** HITL — der zahlende Zwei-Arm-Lauf (gemini-2.5-pro) + der ffmpeg-Render
werden vom Menschen gefahren. Alles außer dem Lauf selbst ist vorab verifiziert
(s. u.). Dieser Runbook ist die copy-paste-Anleitung; das Verdikt liest die
in-Code fixierte Tabelle ab — **abgelesen, nicht hinterher rationalisiert**.

Treiber: `ml/archetype_judge_retest.py` (CLI `run` / `replay` / `render`, gebaut &
test-first abgesichert unter #96/#98/#99/#101 — netz- und ffmpeg-freie Fakes).

## Ziel (der zahlende Schritt)

Genau **eine Variable** isolieren — Archetypes ja/nein — auf der bestehenden
Replay-Infrastruktur, und die resultierende Top-20 beider Arme als geschnittene
Clips im Frontend reviewen.

- **Arm A (Kontrolle):** neutraler Repost-Rubric (0–1, score-only, *keine*
  Archetypes, *keine* Anti-Loudness-Klausel).
- **Arm B (Treatment):** gleiche neutrale Basis **+ die 8 Jynxzi-Archetypes**
  (Name + Description + `transcript_signals`) + „score higher on transcript match".

Beide Arme: gemini-2.5-pro (temp 0), lesen pro Kandidat den **Roh-Text-Excerpt
±12 s** der **re-transkribierten Segments aus #100** (Slice 5), scoren den
**vollen deduped Pool** je VOD, recall@20 gegen **gold ∪ silber** (gleich
gewichtet). Der Archetype-Effekt = recall@20(B) − recall@20(A).

## Vorab-fixiertes Verdikt (im Code: `ml/archetype_judge_retest.py`)

| combined FLOOR recall@20 (B − A) | Verdikt | Folge-Entscheidung (in-Code-Label) |
|---|---|---|
| **≥ +0,03** (≈ 5 gerettete Golds) | **PASS** | `ARCHETYPE_JUDGE_LIVES` → Folge-PRD Serving/Blend-α (#51) |
| **< +0,03** | **NO-GO** | `RANKING_TRACK_CLOSED` → Ranking-Track-Zelle *direkt* geschlossen |

`COMBINED_GATE_CUT = 0.03`. Fail-closed: ein nicht messbarer Move (fehlender
Recall auf einer Seite) liest **nie** als Verbesserung. Read-only mitgeführt
(nicht gegatet): `gold-only`, `silver-only`, `combined-headline` recall@20 +
die **per-gold moves** (welche konkreten Golds B neu in die Top-20 hob).

## ⚠️ Vorbedingung — der Paid-Lauf startet erst nach grünem Konsistenz-Report

Der `run`-Modus baut den #100-Konsistenz-Report **fail-closed neu** aus den
frischen Transkripten auf Disk und **bricht vor jedem bezahlten Call ab**
(`TranscriptConsistencyError`, Exit 1), wenn eine **gejudgte** VOD geflaggt ist
(`schema_mismatch` / `late_start` / `truncated` / `over_run` / `density_outlier`
/ `unreplayable`). Die **6 dokumentierten Exclusions** (kein lokales Video,
`transcribe_only_*` fehlt) blocken **nicht** — sie sind keine gejudgten VODs und
erscheinen als surfaced coverage gap, nie still als Miss.

**#100-Befund (ausgeführt 2026-06-16/17):** 20/20 PASS, 0 Drops, Report clean
modulo der 6 Exclusions. Damit ist die Vorbedingung erfüllt; das harte Gate im
`run` re-prüft sie bei jedem Start trotzdem.

## Verifizierter Stand VOR dem Lauf (2026-06-17, ohne Compute)

- Treiber + Verdikt + Render + CLI gebaut & test-first abgesichert: **39 Tests
  grün** (`tests/test_archetype_judge_retest.py`), alle über netz-/ffmpeg-freie
  Fakes (Fake-Gemini-Provider, Recording-Cutter, In-Memory-Store + List-Sink).
  Geprüft u. a.: (a) `fresh_segment_load_inputs` paart den Kandidaten-Pool mit den
  **frischen #100-Segments**, nie den stale Pool-Transkripten; (b) das harte
  Konsistenz-Gate bricht ohne bezahlten Call ab; (c) Paid→Replay reproduziert das
  Verdikt **bit-genau**; (d) der Archetype-Block erscheint **nur** in Arm Bs Prompt;
  (e) Render registriert je (VOD, Arm) einen Frontend-Job, A/B kollidieren nicht,
  VODs ohne Video surfacen als render gap.
- **Transkript-Basis = #100.** Die frischen `tmp/storage/transcribe_only_<vod>/
  debug/segments.jsonl` (20 VODs, `large-v3-turbo`, plain) sind die einheitliche
  Basis, die der Judge liest.
- `.env`: `GEMINI_API_KEY` vorhanden (der `call_gemini_moment_finder`-Seam braucht
  ihn; gemini-2.5-pro über die OpenAI-kompatible Gemini-API).

## Voraussetzungen

- Docker Desktop läuft. **Kein GPU nötig** — der Judge ist ein reiner API-Lauf,
  der Render reiner ffmpeg-Cut (CPU). Der teure Teil ist die Gemini-API.
- `GEMINI_API_KEY` in `backend/.env`.
- Die 20 Quell-`video.mp4` liegen lokal in `tmp/storage` (aus #100 / dem
  GT-Capture) für den Render. Die 6 ohne Video surfacen als render gap (kein
  Re-Download — Out of Scope).
- ~Platz für 800 Roh-Cuts (20 VODs × 2 Arme × 20 Cuts) unter
  `tmp/storage/archetype_retest_<vod>_<A|B>/clips/`.

**Kosten:** 2 gemini-2.5-pro-Pässe je VOD über 20 VODs (~18,6k Input-Tokens/Pass),
**replaybar** (Verdikt + Render laufen danach netzfrei aus den persistierten
Rohantworten).

---

## Schritt 0 — Container hochfahren

```bash
./scripts/start_containerized.sh
```

(Kein `AI_CLIPPING_BACKEND_RUNTIME=cuda` nötig — reiner CPU/API-Lauf.)

## Schritt 1 — Konsistenz-Report bestätigen (kein Compute, kein Netz)

Den #100-Report report-only neu erzeugen (alle 20 frisch → resume-skip, nur
Reload + Report):

```bash
docker exec ai-clipping-backend python -m ml.transcribe_only_rerun \
  --predictions-dir tmp/storage \
  --video-source-root tmp/storage/transcribe_only_sources
```

Erwartung (s. #100): jede der 20 Zeilen `vod=<id> PASS`, **0** `FLAG`, `excluded`
== die bekannten 6 IDs, `clean: false` (erwartet — allein durch die 6 Exclusions).
**Ein FLAG auf einer der 20** ⇒ der `run` in Schritt 3 bricht hart ab; die VOD
einzeln einordnen (für #101 rein/raus dokumentieren), nicht still folden.

## Schritt 2 — 1-VOD-Smoke (A+B, echtes Gemini, de-risk vor dem Volllauf)

Kleinster VOD = `2751373742`. Nur diesen durch **beide Arme** (echter
gemini-2.5-pro) laufen lassen — beweist Prompt-Bau, Parse und Persistenz beider
Arme, bevor 20× investiert wird:

```bash
docker exec ai-clipping-backend python -m ml.archetype_judge_retest run \
  --storage tmp/storage --only 2751373742 --top-k 20
```

**Smoke-Checkpoints:**

1. **Gate clear:** keine `ABORTED — paid run refused`-Zeile (sonst flaggt die VOD
   den Konsistenz-Report → Schritt 1 erneut prüfen).
2. **Beide Arme persistiert** (die replaybaren Artefakte):
   ```bash
   docker exec ai-clipping-backend sh -c \
     "ls -la tmp/storage/archetype_judge_retest/arm_a/2751373742.raw.txt tmp/storage/archetype_judge_retest/arm_b/2751373742.raw.txt"
   ```
3. **`VERDICT:`-Tabelle gedruckt** (1 VOD ist statistisch bedeutungslos — der Smoke
   prüft das Wiring, nicht das Verdikt).

## Schritt 3 — Voller Paid-Lauf über die 20 (der zahlende Schritt)

```bash
docker exec ai-clipping-backend python -m ml.archetype_judge_retest run \
  --storage tmp/storage --top-k 20 \
  | tee backend/tmp/archetype_judge_retest_run.log
```

- **Hartes Gate:** flaggt eine gejudgte VOD im Konsistenz-Report, bricht der Lauf
  **vor jedem bezahlten Call** ab (Exit 1, `ABORTED — paid run refused: …`). Kein
  Geld verbrannt, nichts persistiert. Den Befund dokumentieren (AC #101).
- **Resume/Re-run:** jeder Arm persistiert pro VOD eine `<vod>.raw.txt`. Bricht der
  Lauf ab, re-rechnet ein erneuter `run` die schon persistierten VODs (kein
  skip-if-exists im Judge); willst du nur Drops nachziehen, nutze `--only <vod>`.
- Der `--only`-Smoke-VOD (Schritt 2) ist bereits persistiert; der Volllauf re-judged
  ihn mit (1 zusätzlicher Paid-Pass je Arm — vernachlässigbar).

## Schritt 4 — Verdikt ablesen (gratis `replay`, netzfrei)

```bash
docker exec ai-clipping-backend python -m ml.archetype_judge_retest replay \
  --storage tmp/storage --top-k 20 \
  | tee backend/tmp/archetype_judge_retest_verdict.log
```

Liest die **bit-genau aus den persistierten Rohantworten** reproduzierte
Verdikt-Tabelle:

- `VERDICT: PASS|NO-GO → ARCHETYPE_JUDGE_LIVES|RANKING_TRACK_CLOSED`
- `gate: combined floor r@20 B … − A … = ±… (cut +0.03) → …`
- read-only breakdowns: `gold_only` / `silver_only` / `combined_headline` recall@20
- `per-gold moves`: die konkreten Golds, die B in seine Top-20 hob, A nicht
  (mit A-Rang → B-Rang, nach Views sortiert).

**Abgelesen, nicht rationalisiert.** Das `cut +0.03` steht im Code vor dem Lauf.

## Schritt 5 — Beide Arme rendern (Roh-16:9-Cuts → Frontend-Jobs)

```bash
docker exec ai-clipping-backend python -m ml.archetype_judge_retest render \
  --storage tmp/storage --top-k 20
```

- Re-liest die persistierten Rohantworten (gratis, netzfrei), schneidet die **exakt
  gegatete Top-20** beider Arme als Roh-16:9-Cuts (kein Reframe/Packaging) und
  registriert je (VOD, Arm) einen Frontend-Job `archetype_retest_<vod>_<A|B>`.
- Druckt `rendered N arm jobs; M render gaps (no local video)` + die Job-IDs. Die
  **6 VODs ohne lokales Video** erscheinen als `render gap (no local video)` —
  ausgewiesen, nie still gedroppt.
- `jobs.json` wird **gemerged** (nicht überschrieben) — laufende Server-Jobs bleiben
  intakt; der Server zieht die Arm-Jobs beim nächsten mtime-Reload.

## Schritt 6 — Im Frontend reviewen (A vs. B nebeneinander)

Im bestehenden Job/Clip-Viewer (keine neue UI): je VOD die zwei Jobs
`archetype_retest_<vod>_A` und `_B` öffnen und die Top-20-Clips vergleichen. Was
du siehst, ist **identisch** mit dem, was der recall gemessen hat (dieselbe
`replay_postprocess`-Top-20). Beurteile die Moment-*Auswahl* (nicht die
Aufmachung — bewusst Roh-Cut).

## Schritt 7 — WALL-Folge-Entscheidung dokumentieren (#101 → PRD #95)

Aus `backend/tmp/archetype_judge_retest_verdict.log` + dem Frontend-Review:

- Das **abgelesene** Verdikt (`PASS`/`NO-GO`) + die Zahl: combined floor r@20
  (B−A), die Breakdowns, die per-gold moves.
- **Ehrliche Einordnung (PRD #95):** Prior niedrig (Blind-Sheet 0/10). Ein NO-GO
  bei sauberem Test schließt die `clean_but_buried`-Ranking-Zelle **direkt** —
  kein Rückschlag, sondern eine der vier WALL-Säulen bestätigt. Ein PASS kippt
  diese Säule → Folge-PRD Serving/Blend-α (#51).
- **Decke des Hebels:** max. ~31 % der recall-Lücke (`clean_but_buried`); die
  ~69 % Generation/Accidental sind durch keinen Re-Ranker erreichbar.
- Den qualitativen Frontend-Eindruck (A vs. B) notieren.

Kommentar an **PRD #95** mit der Folge-Entscheidung; **#101 schließen**. #101 ist
das letzte offene Kind von #95 — danach **#95 cascade-schließen** und den Branch
`prd/95-archetype-judge-retest` fast-forward in `main` integrieren (s.
`scripts/ralph_tdd.sh` / TDD-Skill Schritt 6).

---

## Ergebnis des Laufs (auszufüllen nach Ausführung)

> _Verdikt: <PASS|NO-GO> → <LABEL>. combined floor r@20 (B−A) = <…>. Breakdowns:
> gold-only <…>, silver-only <…>, headline <…>. per-gold moves: <N> (<Liste>).
> Render: <N> Arm-Jobs, <M> render gaps. Frontend-Eindruck: <…>. Folge: <…>._
