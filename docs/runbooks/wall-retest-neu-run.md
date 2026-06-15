# Runbook — WALL-Re-Test neu Best-Shot Run (Slice #93, PRD #90)

**Status:** HITL — der schwere Compute-Lauf wird vom Menschen gefahren. Alles
außer dem Lauf selbst ist vorab verifiziert (s. u.). Dieser Runbook ist die
copy-paste-Anleitung; danach liest die Verdikt-Zelle in `wall_retest_compare`
das **vorab-fixierte Verdikt** ab.

## Ziel (der zahlende Schritt)

Das 13-VOD-Banter/Tic-Subset durch die **echte** Pipeline neu laufen lassen mit
den zwei billigen Hebeln an — Jynxzi-**Glossar** in Whisper (`initial_prompt`) +
pyannote-**Diarization** mit Sprecher ins Finder-Transkript gerendert — Modell
bleibt `large-v3-turbo` (kein Flip). Dann `base` (Status quo, unangetastet) vs.
`neu` recall@20 diffen und das vorab-fixierte Verdikt ablesen.

## Vorab-fixiertes Verdikt (im Code: `ml/wall_retest_compare.py`)

| recall@20-Bewegung (neu − base, headline-Tier) | Verdikt | Folge |
|---|---|---|
| **≥ 0,25** | `WALL_FALLS` | Finder-Pfad revivebar → Lever-Decomposition (Glossar vs. Diarization) |
| **≤ 0,15** | `ESCALATE_PASS2` | WALL felsenfest → First-party-Loop (ADR 0001) |
| dazwischen | `INCONCLUSIVE` | Demucs / Blind-Sheet-Fallback erwägen |

Wird nach dem Lauf **abgelesen, nicht hinterher rationalisiert** (PRD #90 / #93).

## Verifizierter Stand VOR dem Lauf (2026-06-15, ohne Compute)

- Harness #91 + Driver #92 gebaut & getestet — **21 Tests grün**
  (`tests/test_wall_retest_compare.py`, `test_wall_retest_rerun.py`,
  `test_wall_retest_e2e.py`).
- **Subset deterministisch = 13 VODs** (glossary-`desc` ∩ replaybare base-Dirs):
  `2745548882 2747242788 2751373742 2752903388 2757793134 2758541573 2759304833
  2768134699 2769594798 2772745645 2780984594 2783108932 2787844740`
- **Alle 13 base-VODs haben lokales `video.mp4`** (~115 GB in `tmp/storage`) —
  re-download evtl. vermeidbar (s. Smoke-Test-Checkpoint).
- Compare-Harness läuft **base-vs-base auf realen Dirs** → lift `+0.000` überall,
  Verdikt-Logik gesund (lift 0 → `ESCALATE_PASS2` = WALL hält).
- **Live-Baseline auf dem Subset** (base, kein Glossar/keine Diarization):
  recall@20 floor **0,108** / headline **0,059**; raw@any **0,595**;
  clean_but_buried floor **0,378**.
- `.env`: `WHISPER_MODEL=large-v3-turbo`, `HF_TOKEN`, `PIPELINE_ENABLE_DIARIZATION`
  alle vorhanden.

## Voraussetzungen

- Docker Desktop läuft + GPU-Zugang. **GTX 1660 (6 GB)** ist für
  `large-v3-turbo` + pyannote **eng** — rechne mit **overnight/mehrtägig** über
  13 Voll-VODs. (PRD #90 akzeptiert das bewusst für den best-shot.)
- `HF_TOKEN` in `backend/.env` (pyannote ist ein gated Model — Lizenz auf HF
  einmal akzeptieren).
- ~120 GB frei (`tmp/storage_neu` wächst; `tmp/storage` = base bleibt unberührt).

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

## Schritt 1 — Subset bestätigen (dry-run, **kein** Compute)

```bash
docker exec ai-clipping-backend python -m ml.wall_retest_rerun \
  --base-dir tmp/storage --predictions-dir tmp/storage_neu --dry-run --json
```

Erwartung: `subset_count: 13`, exakt die 13 IDs oben.

## Schritt 2 — 1-VOD Smoke-Test (echte Modelle, de-risk vor overnight)

Kleinster gecachter VOD = `2751373742` (2,0 GB). Nur diesen durch den echten
`run_vod`-Leaf laufen lassen:

```bash
docker exec ai-clipping-backend python - <<'PY'
import pathlib
from ml.wall_retest_rerun import pipeline_run_vod
job = pipeline_run_vod("2751373742", pathlib.Path("tmp/storage_neu"), workflow_id="smoke")
print("job_dir:", job)   # None = Lauf fehlgeschlagen / keine replaybaren Artefakte
PY
```

**Smoke-Checkpoints** (das beweist der Smoke-Test, bevor du 13× investierst):

1. **Lädt es NICHT stundenlang den Voll-VOD neu** (Logs beobachten). Tut es das
   doch, ist das die Kosten ×13, die du einplanen musst.
2. **Diarization → Sprecher gerendert.** Der Finder-Prompt trägt `| spk=…`:
   ```bash
   docker exec ai-clipping-backend sh -c \
     "grep -o '| spk=[^ ]*' tmp/storage_neu/pipeline_debug_smoke_2751373742/debug/llm_moment_finder_prompt.txt | sort | uniq -c | head"
   ```
   ≥1 Treffer mit einem nicht-`UNKNOWN`-Sprecher ⇒ Diarization-Rendering hat real
   gefeuert (nicht nur im Fake-Test).
3. **Replaybar** (`segments.jsonl` + `llm_moment_finder_moments_raw.jsonl` da):
   ```bash
   docker exec ai-clipping-backend sh -c \
     "ls -la tmp/storage_neu/pipeline_debug_smoke_2751373742/debug/{segments.jsonl,llm_moment_finder_moments_raw.jsonl}"
   ```

Smoke-Dir danach wegräumen, damit es den echten Lauf nicht stört:
```bash
docker exec ai-clipping-backend rm -rf tmp/storage_neu/pipeline_debug_smoke_2751373742
```

## Schritt 3 — Voller neu-Lauf über die 13 (der overnight-Schritt)

```bash
docker exec ai-clipping-backend python -m ml.wall_retest_rerun \
  --base-dir tmp/storage \
  --predictions-dir tmp/storage_neu \
  --workflow-id wallretest \
  --json | tee backend/tmp/wall_retest_rerun.log
```

- **Compute-Zeit + Drops + Subset-Größe protokollieren** (#93-AC). `--json` gibt
  `completed` + `drops` aus; Drops gehen einzeln auf stderr.
- **Caveat Resume:** der Driver re-runt *alle* Subset-VODs (kein
  skip-if-exists). Bricht der Lauf ab, sind bereits fertige `neu`-Dirs in
  `tmp/storage_neu` da; ein erneuter Aufruf re-rechnet sie aber. Willst du nur
  die Drops nachziehen, ruf `pipeline_run_vod` (s. Schritt 2) je Drop einzeln.

## Schritt 4 — Verdikt ablesen (base vs. neu)

Subset-CSV für den Anchor schreiben (die 13 IDs, deterministisch):

```bash
docker exec ai-clipping-backend python - <<'PY'
import pathlib, csv
from ml import eval_clip_finder
from ml.wall_retest_rerun import select_subset
m = eval_clip_finder.load_manifest(eval_clip_finder.DEFAULT_MANIFEST)
subset = select_subset(m, pathlib.Path("tmp/storage"))
out = pathlib.Path("tmp/wall_retest_subset.csv")
with out.open("w", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["vod_id","vod_date","note"])
    for v in subset: w.writerow([v, "", "wall-retest-subset"])
print("wrote", out, len(subset), "VODs")
PY
```

Compare laufen lassen (CPU-only Replay, schnell):

```bash
docker exec ai-clipping-backend python -m ml.wall_retest_compare \
  --base-dir tmp/storage \
  --predictions-dir tmp/storage_neu \
  --anchor-csv tmp/wall_retest_subset.csv \
  --output tmp/wall_retest_verdict.json
```

Liest: `VERDICT: …`, recall@20 base/neu/lift (floor+headline), raw@any,
clean_but_buried. Anchor bleibt die **vollen 13** — neu-Drops erscheinen als
`replay_gaps`-Warnung (nicht als Misses gefoldet).

## Schritt 5 — Befund dokumentieren (#90 + #93)

Aus `tmp/wall_retest_verdict.json`:

- recall@20-Diff (headline) + raw@any + clean_but_buried, base vs. neu.
- Das **abgelesene** Verdikt (`WALL_FALLS` / `ESCALATE_PASS2` / `INCONCLUSIVE`)
  + Subset-Größe, Drops, Compute-Zeit.
- **Ehrliche Einordnung:** Prior niedrig (starkes Modell hat schon versagt) —
  ein Fail bei best-shot ist die **starke WALL-Bestätigung**, kein Rückschlag.
- **Bei `WALL_FALLS`:** Lever-Decomposition als Folgefrage notieren (Glossar vs.
  Diarization — je einen isoliert nachfahren).
- Kommentar an **#90** mit der Folge-Entscheidung; **#93** schließen.
