# Jynxzi Viral Clip Finder — Roadmap

Ziel: LLM-basierter Clip-Finder, der aus einem Jynxzi-VOD die besten ~10 viralen
Momente liefert. Fokus: **Clipfindung**, keine Nachbearbeitung.

Aktueller Stand: Pipeline läuft (Download → Whisper → LLM → Cuts).
Kritische Lücken: kein Chat-Signal, kein Eval-Rahmen, kein Review-Werkzeug.

---

## Phase 1 — Ground Truth aufbauen
> **Ziel:** Wissen, welche Momente aus echten Jynxzi-VODs viral gegangen sind.
> Ohne das ist jede Verbesserung blind.

- [ ] **1.1 — 5 Anchor-VODs auswählen**
  - 5 Jynxzi Twitch VODs aus den letzten 3 Monaten identifizieren
  - Für jeden VOD: 3-5 bekannte virale Clips finden (YouTube Shorts, TikTok)
  - Format: `{vod_id, clip_url, views, platform}` in `ml/artifacts/jynxzi_gt_anchors.csv`

- [ ] **1.2 — Timestamps aus Clip-Material herleiten**
  - Option A (schnell): manuell — Clip-URL anschauen, Transcript-Suche nach
    dem gesprochenen Text → `{start_sec, end_sec}` notieren
  - Option B (skalierbar): `match_cross_platform_transcripts.py` auf die VODs laufen
    lassen, wenn Reposts bekannt sind
  - Ergebnis: `{vod_id, start_sec, end_sec, description, views}` pro Viral-Moment

- [ ] **1.3 — Eval-Manifest bauen**
  - Datei: `ml/artifacts/jynxzi_eval_manifest.jsonl`
  - Schema: `{"vod_id": "...", "gt_moments": [{"start": 1184.0, "end": 1208.0,
    "desc": "clutch 1v3 mit Ace", "views": 120000}]}`
  - Dieses Manifest ist **unveränderlich nach Fertigstellung** — nie für Training nutzen

---

## Phase 2 — Baseline messen
> **Ziel:** Aktuelle Recall@K-Zahlen kennen, bevor etwas geändert wird.
> Script: `ml/eval_clip_finder.py` (neu zu bauen in Schritt 2.2).

- [ ] **2.1 — Pipeline auf 5 Anchor-VODs ausführen**
  - `run_pipeline()` mit aktuellem LLM Moment Finder (Gemini, kein Chat)
  - Debug-Artifacts pro VOD in `tmp/storage/<job_id>/debug/` sichern
  - `llm_moment_finder_moments.jsonl` für jedes VOD exportieren

- [ ] **2.2 — `ml/eval_clip_finder.py` schreiben**
  - Lädt `jynxzi_eval_manifest.jsonl` + `llm_moment_finder_moments.jsonl` pro VOD
  - Overlap-Definition: Kandidat gilt als Treffer wenn ≥ 50% des GT-Fensters überdeckt
  - Metriken:
    - `recall@5`, `recall@10`, `recall@20`
    - `avg_rank` — Durchschnittsrang eines gefundenen GT-Moments
    - `miss_rate` — Anteil GT-Momente, die kein Kandidat im Top-20 abdeckt

- [ ] **2.3 — Top-10-Clips pro VOD manuell sichten**
  - FFmpeg-Cuts der Top-10 Kandidaten jedes VODs nach `tmp/review/<vod_id>/`
  - Manuell bewerten: gut / okay / schlecht
  - Ergebnis in `ml/artifacts/jynxzi_baseline_precision.jsonl`

---

## Phase 3 — Chat-Signal einbauen
> **Ziel:** Twitch-Chat-Spikes als stärkstes Publikums-Signal in den LLM-Prompt.
> `services/chat_analyzer.py` existiert bereits vollständig — muss nur verdrahtet werden.

- [ ] **3.1 — Chat-Download in `pipeline.py` einbauen**
  - Nach Video-Download:
    ```python
    chat_path = chat_analyzer.download_chat(url, job_id, debug_dir)
    ```
  - Nur für Twitch-URLs (VOD-ID aus URL extrahieren)
  - Fail-safe: `chat_path = None` → Pipeline läuft unverändert weiter
  - Chat-JSON in `debug/<job_id>/chat.json` cachen

- [ ] **3.2 — Segments mit Chat-Daten anreichern**
  - Nach Transkription:
    ```python
    if chat_path:
        segments = chat_analyzer.enrich_segments(segments, chat_path)
    ```
  - Fügt `chat_spike_score`, `ema_fast`, `ema_slow`, `categories` zu jedem Segment hinzu

- [ ] **3.3 — Chat-Annotationen in LLM-Prompt injizieren**
  - In `llm_moment_finder.format_transcript_for_prompt()`:
    - Neuer optionaler Parameter: `segment_chat: dict[float, ChatSpike] | None`
    - Segment-Zeile mit Spike:
      ```
      [t=1184s - t=1196s | clock=19:44-19:56 | chat=🔥HIGH(+287msg/s, hype)] "bro what was that??"
      ```
    - Segment-Zeile ohne Spike: unverändert (kein Rauschen für normale Segmente)
  - Prompt-Präambel ergänzen: kurze Erklärung der `chat=` Annotation

- [ ] **3.4 — Eval nach Chat-Integration wiederholen**
  - Gleiche 5 VODs, gleicher eval_clip_finder.py-Lauf
  - Delta-Report: recall@K und avg_rank vor vs. nach Chat
  - Spike-Schwellenwert anpassen (0.5 / 0.7 / 0.9) wenn nötig

---

## Phase 4 — Review-Werkzeug
> **Ziel:** Schnelles manuelles Bewerten ohne Entwicklungsaufwand.

- [ ] **4.1 — CLI Review-Script `ml/review_candidates.py`**
  - Aufruf: `python ml/review_candidates.py --vod_id 2345678901`
  - Zeigt pro Kandidat: Rang, Zeit, Dauer, virality, Rationale, Chat-Score (wenn vorhanden)
  - Option `--cut`: Clip schneiden und mit `mpv`/`vlc` abspielen
  - Tastatur: `g` = gut, `b` = schlecht, `s` = skip, `q` = quit
  - Schreibt nach `ml/artifacts/jynxzi_clip_reviews.jsonl`

- [ ] **4.2 — Review-Labels in Eval einbinden**
  - `eval_clip_finder.py` liest `jynxzi_clip_reviews.jsonl`
  - Ergänzt `precision@10` (Human-Labels) neben GT-Recall

---

## Phase 5 — Prompt-Iteration
> **Ziel:** recall@5 ≥ 60% und precision@10 ≥ 50% erreichen.
> Jede Iteration gegen das fixe Eval-Manifest aus Phase 1 messen.

- [ ] **5.1 — Miss-Analyse aus Baseline**
  - Für jeden verpassten GT-Moment: Ursache klassifizieren
    - `A` — LLM hat Moment nicht vorgeschlagen (Recall-Lücke im Finder)
    - `B` — LLM hat vorgeschlagen, aber virality zu niedrig → unterhalb Rank-Cutoff
    - `C` — Dedup hat ihn rausgefiltert (Überschneidung mit höherem Kandidaten)
  - Protokoll: `ml/artifacts/jynxzi_miss_analysis.md`

- [ ] **5.2 — Jynxzi-spezifische Prompt-Schärfung**
  - Aktueller Prompt ist zu generisch ("Jynxzi-style clips").
  - Konkrete Muster ergänzen:
    - **R6 Siege:** Clutch-Rounds (1v3, 1v4, Ace), Operator-Picks, Bomb-Plant-Momente
    - **Rage:** Controller-Sprache ("I'm done", "this game is cooked"), Tilt-Spiralen,
      Argument mit Random-Teammates
    - **Chat-Interaktion:** Auf Spender reagieren, Trolls countern, parasoziale Momente
    - **Duo-Segmente:** Teamspieler-Fail-Reaktion, Callout-Fails, Duo-Synergy-Payoffs
    - **IRL:** Fanmail-Unboxings, Familien-Telefonat, Surprise-Reveals

- [ ] **5.3 — Längen-Kalibrierung**
  - Clip-Längen-Distribution der bekannten Jynxzi-Virals analysieren
  - Aktuell: `min=8s, max=90s` — vermutlich zu breit
  - Hypothese: typische Jynxzi-Virals sind 25-60s (Setup + Reaction + Payoff)
  - `min_duration_sec` / `max_duration_sec` in `pipeline.py` anpassen

- [ ] **5.4 — Model-Vergleich**
  - Gleiche 5 VODs, gleiche Metriken:
    | Model | Provider | Kosten/VOD (est.) |
    |-------|----------|-------------------|
    | gemini-2.5-flash-lite | Gemini | ~$0.05 |
    | gemini-2.5-flash | Gemini | ~$0.20 |
    | claude-sonnet-4-6 | Anthropic | ~$0.40 |
  - Entscheidung: recall@10 × (1 / cost) — Effizienz vor roher Qualität

- [ ] **5.5 — `max_moments` tunen**
  - Aktuell 180 — sehr viele Kandidaten, viel Noise
  - Testen: 60 / 100 / 180
  - Ziel: bester recall@10 bei precision@10 ≥ 0.5

---

## Erfolgskriterien

| Metrik | Baseline (Phase 2) | Ziel (Phase 5) |
|--------|-------------------|----------------|
| recall@5 | TBD | ≥ 60% |
| recall@10 | TBD | ≥ 75% |
| precision@10 | TBD | ≥ 50% |
| avg_rank (GT-Moments) | TBD | ≤ 8 |

---

## Offene Entscheidungen — vor Phase 1 klären

1. **Welche 5 Anchor-VODs?** — IDs oder Datumsbereich (z.B. April-Mai 2026)
2. **Ground Truth manuell oder automatisch?** — Option A (schnell) oder Option B (skalierbar)
3. **Review-Werkzeug Scope** — CLI reicht, oder soll das Review-Workbench im Frontend zurückkommen?
4. **Gemini API-Key konfiguriert?** — Nötig für Phase 2 (Baseline)

---

## Review-Ergebnisse
*(wird nach Abschluss jeder Phase befüllt)*
