# Runbook — Transcribe-only Re-Transkription der 20 Anchor-VODs (Issue #100, PRD #95)

**Status:** HITL — der schwere GPU-Lauf wird vom Menschen gefahren. Alles außer
dem Lauf selbst ist vorab verifiziert (s. u.). Dieser Runbook ist die
copy-paste-Anleitung; danach liest der Konsistenz-Report (`transcribe_only_rerun`,
fail-closed) den Befund ab. Er liefert die konsistente Transkript-Basis, die der
Archetype-Judge-Lauf (#101) liest.

## Ziel (der GPU-Schritt)

Die **20 Anchor-VODs mit lokalem Video** mit *einem* einheitlichen Modell
(`large-v3-turbo`, **plain** — kein Glossar-`initial_prompt`, keine Diarization)
transcribe-only neu transkribieren → frische, schema-einheitliche
`segments.jsonl` je VOD in eigenen `transcribe_only_<vod>/`-Dirs. Der
Kandidaten-Pool (`llm_moment_finder_moments_raw.jsonl`) wird **nie** zum Schreiben
geöffnet — das ist transcribe-only, **kein** Pipeline-Re-Run. Resumebar (eine VOD
mit bereits frischer `segments.jsonl` wird übersprungen). Danach den
Konsistenz-Report fail-closed ablesen.

Treiber: `ml/transcribe_only_rerun.py` (gebaut & getestet unter #97).

## Verifizierter Stand VOR dem Lauf (2026-06-15, ohne Compute)

- Treiber #97 gebaut & test-first abgesichert (netzfreier Fake-Transcriber).
- **Source-Staging gebaut & verifiziert.** Die 20 Ziel-VODs liegen auf Disk als
  `video.mp4` in ihren `pipeline_debug_*/` (bzw. einem `gt_vod_*/`) Job-Dirs —
  **nicht** im Default-`--video-source-root`. Ein dedizierter Staging-Root
  `tmp/storage/transcribe_only_sources/` mit **relativen** Symlinks
  (`vod_<id>.mp4 → ../<job-dir>/video.mp4`) macht sie für den Treiber auflösbar.
  Relativ ⇒ mount-agnostisch: derselbe Link funktioniert auf Host **und** im
  Container (beide unter dem `tmp/storage`-Mount).
- **Split verifiziert = 20 / 6:** `split_by_local_video` über den Staging-Root
  liefert exakt **20** `with_video` und **6** surfaced `without_video` —
  letztere ohne jegliches Video auf Disk:
  `2764976309 2766519381 2768960946 2770356205 2771207666 2773650615`.
- **End-to-end-Wiring ohne GPU bestätigt** (Fake-Transcribe-Leaf über den realen
  Staging-Root, throwaway predictions-dir): `completed=20`, `excluded=6`,
  `drops=0`, **alle 20 per-VOD-Verdikte = PASS**, Resume re-skippt alle 20.
- `.env`: `WHISPER_MODEL=large-v3-turbo`, `WHISPER_DEVICE=auto`,
  `WHISPER_LANGUAGE=en` vorhanden.

### ⚠️ Fail-closed lesen — `clean: false` ist hier der ERWARTETE Erfolgsfall

Der Report setzt `clean = (keine Exclusions) AND (alle VODs PASS)`. Da es **6
dokumentierte Exclusions** (keine lokalen Videos) gibt, ist `clean` **immer
`false`** und der CLI-Exit-Code **non-zero** — *auch bei einem perfekten Lauf*.
Behandle den non-zero Exit **nicht** als Fehler. Erfolgskriterium ist stattdessen:

1. `completed == 20` und `drops == []`,
2. **alle 20** `vod_reports[].passed == true` (0 reale Flags),
3. `excluded` == genau die bekannten **6** IDs oben.

Ein FLAG auf einer der **20** (`schema_mismatch` / `late_start` / `truncated` /
`over_run` / `density_outlier` / `unreplayable`) ist ein **echtes** Drift-Signal →
einzeln surfacen und dokumentieren (PRD #95 / AC #100), nicht still als Miss folden.

`over_run` (span_end > VOD-Länge × 1,05) und das chunk-granularitäts-bewusste
`late_start` (Default 4 × chunk_seconds = 720 s, statt flat 120 s) kamen während
dieses Laufs dazu — s. „Ergebnis des Laufs".

## Voraussetzungen

- Docker Desktop läuft + GPU-Zugang. **GTX 1660 (6 GB)** ist für `large-v3-turbo`
  **eng** — rechne mit **overnight** über 20 Voll-VODs (~180 GB Quell-Video,
  längster ~15 GB). (#100 akzeptiert GPU/overnight bewusst.)
- Die 20 Quell-Videos liegen lokal in `tmp/storage` (Staging-Symlinks zeigen
  darauf — Schritt 1 verifiziert das, **kein** Re-Download nötig).
- ~10 GB frei für die extrahierten 16-kHz-`.wav` (landen neben den Symlinks in
  `transcribe_only_sources/`; nach dem Lauf wegräumbar). Die frischen
  `segments.jsonl` selbst sind winzig (Text).

---

## Schritt 0 — CUDA-Container hochfahren

```bash
AI_CLIPPING_BACKEND_RUNTIME=cuda ./scripts/start_containerized.sh
```

GPU-Sichtbarkeit prüfen (muss `True` + den Device-Namen zeigen):

```bash
docker exec ai-clipping-backend python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Schritt 1 — Source-Staging bestätigen (idempotent, **kein** Compute)

Die Symlinks sind bereits gebaut (s. „Verifizierter Stand"). Dieser Schritt baut
sie reproduzierbar neu (z. B. auf einer frischen Checkout) und bestätigt den
20/6-Split. **Re-runnable, kein Download, kein GPU.**

```bash
docker exec ai-clipping-backend python - <<'PY'
import os, pathlib
from ml import eval_clip_finder, transcribe_only_rerun as T

STORAGE = pathlib.Path("tmp/storage").resolve()
STAGING = STORAGE / "transcribe_only_sources"
anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(eval_clip_finder.DEFAULT_ANCHOR_CSV))

def best_source(vod_id):
    names = {"video.mp4", f"vod_{vod_id}.mp4", f"{vod_id}.mp4"}
    cands = []
    for p in STORAGE.rglob("*.mp4"):
        if p.name not in names or vod_id not in str(p):
            continue
        try: sz = p.stat().st_size
        except OSError: continue
        if sz < 1_000_000_000:          # skip rendered clips
            continue
        cands.append((sz, str(p), p))
    cands.sort(key=lambda t: (t[0], t[1]))   # deterministic: largest, then newest path
    return cands[-1][2] if cands else None

if STAGING.exists():
    for f in STAGING.iterdir():
        if f.is_symlink() or f.is_file(): f.unlink()
STAGING.mkdir(parents=True, exist_ok=True)

staged, missing = [], []
for vid in anchors:
    src = best_source(vid)
    if src is None:
        missing.append(vid); continue
    os.symlink(os.path.relpath(src, STAGING), STAGING / f"vod_{vid}.mp4")  # RELATIVE
    staged.append(vid)

wv, wo = T.split_by_local_video(anchors, STAGING)
print(f"staged={len(staged)}  resolvable with_video={len(wv)}  without_video={len(wo)}")
print("excluded (no local video):", sorted(wo))
assert len(wv) == 20 and len(wo) == 6, "expected the 20/6 split"
print("OK: 20 targets resolvable, 6 surfaced exclusions")
PY
```

Erwartung: `with_video=20`, `without_video=6` (exakt die 6 IDs oben).

## Schritt 2 — 1-VOD Smoke-Test (echtes Modell, de-risk vor overnight)

Kleinster gestageter VOD = `2751373742` (2,1 GB). Nur diesen durch den **echten**
transcribe-only-Leaf laufen lassen (`_default_transcribe_vod`: Audio extrahieren +
plain Whisper). Schreibt in den **realen** predictions-dir `tmp/storage`, sodass
der Volllauf (Schritt 3) diese VOD danach resume-skippt — kein doppelter Compute.

```bash
docker exec ai-clipping-backend python - <<'PY'
import pathlib
from ml.transcribe_only_rerun import retranscribe_vod, _default_transcribe_vod, load_report_input
vod = "2751373742"
src = pathlib.Path("tmp/storage/transcribe_only_sources") / f"vod_{vod}.mp4"
out = retranscribe_vod(vod, src, pathlib.Path("tmp/storage"),
                       transcribe_vod=_default_transcribe_vod)
recs, dur = load_report_input(pathlib.Path("tmp/storage"), vod)
print("fresh segments:", out)
print("exists:", out.exists(), "bytes:", out.stat().st_size, "records:", len(recs), "vod_len_sec:", dur)
PY
```

**Smoke-Checkpoints** (das beweist der Smoke, bevor du 20× investierst):

1. **Lädt es NICHT neu herunter** (Logs beobachten): es extrahiert Audio aus dem
   lokalen Symlink, kein Twitch-Download. Tut es das doch, stimmt das Staging
   nicht (Schritt 1 erneut prüfen).
2. **Frische `segments.jsonl`** in `transcribe_only_2751373742/debug/`, non-empty,
   `records > 0`, plain (keine `spk=`/Diarization-Marker):
   ```bash
   docker exec ai-clipping-backend sh -c \
     "ls -la tmp/storage/transcribe_only_2751373742/debug/segments.jsonl && head -2 tmp/storage/transcribe_only_2751373742/debug/segments.jsonl"
   ```
3. **Pool unangetastet** (transcribe-only, kein Pipeline-Re-Run): der Treiber legt
   **nur** `transcribe_only_<vod>/` an — die `pipeline_debug_*_2751373742`-Dirs und
   ihr `llm_moment_finder_moments_raw.jsonl` werden nie geöffnet. Verifizieren,
   dass kein `transcribe_only_*` unter den Pool-Dirs auftaucht und nur das eigene
   Dir neu ist:
   ```bash
   docker exec ai-clipping-backend sh -c \
     "ls -d tmp/storage/transcribe_only_2751373742 && find tmp/storage/pipeline_debug_*_2751373742 -name 'llm_moment_finder_moments_raw.jsonl' -printf '%T+ %p\n'"
   ```

## Schritt 3 — Voller transcribe-only-Lauf über die 20 (der overnight-Schritt)

```bash
docker exec ai-clipping-backend python -m ml.transcribe_only_rerun \
  --predictions-dir tmp/storage \
  --video-source-root tmp/storage/transcribe_only_sources \
  --json | tee backend/tmp/transcribe_only_rerun.log
```

- **Resumebar:** bereits frische VODs (inkl. der Smoke-VOD aus Schritt 2) werden
  übersprungen. Bricht der Lauf ab, einfach denselben Befehl erneut absetzen — er
  rechnet nur die fehlenden VODs nach.
- **Compute-Zeit + Drops/Excludes protokollieren** (#100-AC). `--json` gibt
  `completed`, `drops`, `excluded` und die `vod_reports` aus; Drops/Exclusions
  gehen zusätzlich einzeln auf stderr.
- ⚠️ **Exit-Code ist non-zero** (fail-closed wegen der 6 Exclusions) — **kein
  Fehler.** Erfolg nach der Tabelle oben lesen.

## Schritt 4 — Konsistenz-Report-Verdikt ablesen

Den Report netzfrei/ohne Recompute neu erzeugen (alle 20 sind frisch → resume-skip,
nur Reload + Report):

```bash
docker exec ai-clipping-backend python -m ml.transcribe_only_rerun \
  --predictions-dir tmp/storage \
  --video-source-root tmp/storage/transcribe_only_sources
```

Ablesen (Erfolgskriterium, s. Fail-closed-Box oben):

- `... completed, 6 excluded (no video), 0 dropped`
- **jede** der 20 Zeilen `vod=<id> PASS` — **0** `FLAG ...`
- `excluded (no local video):` == die bekannten 6 IDs
- `clean: false` ⇐ **erwartet**, allein durch die 6 Exclusions getrieben

Bei einem **FLAG** auf einer der 20: die Flag-Art (`late_start`/`truncated`/
`density_outlier`/`schema_mismatch`/`unreplayable`) je VOD notieren und einzeln
einordnen — geflaggte VODs werden für #101 explizit aus- oder eingeschlossen
(dokumentiert), nie still gefoldet.

## Schritt 5 — Befund dokumentieren (#100 → #101)

Aus `backend/tmp/transcribe_only_rerun.log`:

- `completed` (Soll 20), `drops` (Soll leer), `excluded` (die 6).
- Pro geflaggter VOD (falls vorhanden) die Flag-Art + Entscheidung (für #101 rein
  oder raus). Sind alle 20 PASS: „Konsistenz-Report clean modulo der 6
  dokumentierten Exclusions".
- **Compute-Zeit** des Volllaufs (Start/Ende aus den Logs).
- Kommentar an **#100** mit dem Befund; **#100 schließen**. Die frischen
  `tmp/storage/transcribe_only_<vod>/debug/segments.jsonl` sind die einheitliche
  Transkript-Basis für den Archetype-Judge-Lauf **#101**.

### Aufräumen (optional, nach erfolgreichem Lauf)

Die extrahierten `.wav` (nicht die `segments.jsonl`!) wegräumen:

```bash
docker exec ai-clipping-backend sh -c \
  "rm -f tmp/storage/transcribe_only_sources/transcribe_only_*.wav"
```

---

## Ergebnis des Laufs (ausgeführt 2026-06-16/17)

**Verdikt: 20/20 PASS, 0 Drops, Report clean modulo der 6 dokumentierten
Exclusions.** Compute ~**11,5 h** über die 20 (GTX 1660, `large-v3-turbo`,
int8_float16). Die frischen `segments.jsonl` sind die Transkript-Basis für #101.

Der Lauf legte zwei Bugs offen (beide gefixt, TDD):

1. **Chunk-Bleed (Korrektheit) — `fix(transcriber)` `7899899`.** Alle 20 `.wav`
   teilen sich einen Staging-Dir, also einen `chunks/`-Dir. `transcribe_with_progress`
   globt *alle* `chunk_*.wav`; ffmpegs Segment-Muxer (`-y`) überschreibt aber nur,
   was er schreibt — eine **kürzere** VOD nach einer **längeren** erbte deren
   Rest-Chunks und transkribierte fremde Stream-Audio auf ihre Timeline (end_cov bis
   299 %). Fix: stale `chunk_*.wav` vor dem Re-Segmentieren löschen. Erst-Lauf war
   davon zu 14/20 betroffen → komplett neu transkribiert.
2. **Report-Lücken (fail-closed) — `feat(ml)` `50c10fa`.** (a) Kein oberes
   Coverage-Limit → der Over-Run oben lief *durch*. Neuer `over_run`-Flag
   (span_end > Länge × 1,05). (b) `late_start` war flat 120 s < 180 s Chunk → jede
   VOD mit kurzem Intro flaggte (11/20, alle benigne „starting soon"-Musik, per
   Inhalt verifiziert). Jetzt chunk-granularitäts-bewusst (Default 4 × chunk = 720 s).

**Infra-Hinweise fürs nächste Mal:** der GPU-Lauf brach einmal mit `CUDA failed
with error unknown error` ab (frischer Prozess = frischer CUDA-Context behebt es,
Driver ist resumebar) und Docker Desktop verlor einmal die WSL-Integration
(`Start-Process 'Docker Desktop.exe'`, Container kam per restart-policy zurück).
Beide Male ging **kein** Fortschritt verloren (Resume-Skip pro fertiger VOD).
