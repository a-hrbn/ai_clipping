# Task Plan

## 2026-06-05 Plan (KONKRET): LLM-Moment-Finder produktiv — Lokalisierung → Integration → Entscheidungs-Gate

**Leit-Entscheidung (begründet):** LLM-Pfad ist die einzige ungetestete Stellschraube, die zum Produktziel ("virale Jynxzi-Momente finden") passt; Ranker-Reparatur ist empirisch erschöpft. ABER: kein bewiesener Gewinner — die **tragende, noch unmessene Metrik ist Top-N-Precision auf `known_repost_match=false`**. Erst additiv bauen → in Phase 2 messen → DANN Ranker löschen. Exact-Pick-@top-12 (~8 %, für beide Pfade) ist NICHT mehr die Zielmetrik.

### Schritt 0 — Lokalisierungs-Bug fixen (BLOCKER, zuerst) — konfundiert jede spätere Messung
- [x] **0.1** In `ml/run_llm_moment_finder_d0.py` nach `parse_moments`/vor Eval die Momente durch eine Längen-Disziplin schicken: harte Kappung auf `max_duration_sec` (90s) + Drop/Split von Blobs >120s statt sie als „Treffer" zählen zu lassen. Aktuell wird nur nach `virality` sortiert (Z. 891–897) — keine Längen-Enforcement.
- [x] **0.2** Eval härten: Ein Positiv-Treffer durch ein >max_duration-Fenster zählt NICHT (sonst „trifft" ein 67-min-Blob jedes Positiv). `evaluate_moments`/`evaluate_operating_points` so anpassen, dass überlange Momente vor dem Matching verworfen werden — Recall wird dann lokalisierungs-ehrlich.
- [x] **0.3** Re-run gegen die 3 bestehenden D0.6b-Moment-Artefakte via `--reuse-responses` (KEIN neuer API-Call) → prüfen wie viel echter Recall nach ehrlicher Lokalisierung übrig bleibt. Das ist die reale Generation-Baseline.
- [x] **0.4** Tests in `test_run_llm_moment_finder_d0.py` für Längen-Enforcement + ehrliches Matching.

#### Schritt-0 Review Notes
- Längen-Policy implementiert: `<=90s` bleibt, `90-120s` wird auf 90s ab `start` gekappt, `>120s` wird als Blob verworfen.
- Eval ist jetzt lokalisierungs-ehrlich: `evaluate_moments` und `evaluate_operating_points` ignorieren überlange Momente vor dem Matching und reporten `ignored_long_moment_count`.
- Re-run ohne API-Call: `--reuse-responses --output-prefix backend/ml/artifacts/jynxzi_llm_moment_finder_d0_step0_localized`.
- Ergebnis: D0.6b lokalisiert `8/36 = 22.2222%` Recall@Top-180, Gate 15% bestanden. Per VOD: `2757793134=1/12`, `2763470626=5/12`, `2787027855=2/12`.
- Blob-/Cap-Befund: `2757793134` 2 Drops/3 Caps, `2763470626` 0 Drops/1 Cap, `2787027855` 74 Drops/35 Caps. Die alte 67%-Generierungszahl war damit durch überlange Fenster deutlich konfundiert.

### Phase 1 — `services/moment_finder.py` additiv + Pipeline-Rewire (Flag default AUS, kein Delete)
- [x] **1.1** `services/moment_finder.py`: Kern aus `ml/run_llm_moment_finder_d0.py` **teilen, nicht duplizieren** (`build_moment_finder_prompt`, `call_moment_finder_provider`, `parse_moments`, `LlmMoment`, `format_transcript_for_prompt`, `transcript_noise_reason`, `normalize_transcript_text`). Eingang: Live-`list[RichSegment]` aus der Pipeline (Adapter `RichSegment → TranscriptSegment`). Ausgang: `list[LlmMoment]`, bereits durch `post_processor.snap_boundaries` + `enforce_diversity` lokalisiert.
- [x] **1.2** Schema-Brücke `post_processor`: D0-Momente (`start/end/virality`) auf das `post_processor`-Dict-Schema (`start_time/end_time/viral_score/rank`) mappen, damit `snap_boundaries` (90s-Cap, Satzgrenzen) + `enforce_diversity` (min_gap) wiederverwendet werden. Reuse statt Neubau.
- [x] **1.3** Adapter `LlmMoment → RankedCandidate` (`virality`→`rank_score`; `hook/flow/value/trend`+`rationale` in Reasons). So konsumieren Judge/Selektion/Quality-Gate/Boundary/Cutter den Output unverändert.
- [x] **1.4** `pipeline.py` rewire hinter Flag `enable_llm_moment_finder` (default AUS): Block `pipeline.py:1723–1799` (`normalize_candidates`→`feature_builder.build_features`→`selector.rank_candidates` inkl. zweitem Rank) durch `moment_finder.find_moments(transcript)`→Adapter→`ranked_candidates` ersetzen. Ab `1800` (`ranked_candidates_pre_judge`) + Judge auf **Multimodal-Confirm-only** + Select/Cut-Tail UNVERÄNDERT.
- [x] **1.5** Config-Flags in `config.py`: `enable_llm_moment_finder` (False), `moment_finder_provider` (gemini), `moment_finder_model`, `moment_finder_max_moments`, `moment_finder_multimodal_confirm_top_n`. **Fail-open** bei Provider-Fehler (Serving-Lesson): Job sauber degradieren, Backend nicht crashen.
- [x] **1.6** `test_moment_finder.py`: gemockter Provider, RichSegment-Adapter, post_processor-Brücke, RankedCandidate-Adapter, Noise-Filter, Fail-open.

#### Phase-1 Review Notes
- Neuer additiver Service `backend/services/moment_finder.py` nutzt die D0-Bausteine direkt: Prompt-Build, Provider-Call, JSON-Parsing, Noise-Filter und Längen-Policy werden nicht dupliziert.
- Pipeline-Branch ist hinter `enable_llm_moment_finder=False` default-aus. Wenn der LLM-Pfad erfolgreich ist, ersetzt er nur den Ranker-Block und liefert `RankedCandidate`s an Judge/Quality-Gate/Analyzer/Cutter weiter.
- Fail-open ist implementiert: `MomentFinderError` schreibt `debug/llm_moment_finder_stage.json` mit `fallback=ranker_pipeline` und nutzt danach den bisherigen Ranker-Pfad.
- Debug-Artefakte bei Flag AN: `llm_moment_finder_stage.json`, `llm_moment_finder_prompt.txt`, `llm_moment_finder_response.txt`, `llm_moment_finder_usage.json`, `llm_moment_finder_moments.jsonl`.
- Verification mit gemocktem Provider/Pipeline: `test_moment_finder.py`, D0-Tests, Precision-Review-Tests, Workflow-Router-Tests und Pipeline-Integration (`71 passed`).
- Echter Docker/GPU-Serving-Smoke mit Flag AN: `llm_phase1_verify_docker_20260605T214307Z`, VOD `2785392475`, Judge AUS, `status=done`, `clips=10`, elapsed `181.43s`.
- Smoke-Artefakt bestaetigt LLM-Pfad statt Ranker: `debug/llm_moment_finder_stage.json` mit `status=ok`, `provider=gemini`, `model=gemini-2.5-flash-lite`, `raw_moment_count=65`, `post_processed_moment_count=11`, `ranked_candidate_count=11`, `dropped_long_moment_count=0`.
- Beim Smoke gefunden und gefixt: Pipeline setzte kurz `status=finding_moments`, aber `JobState` kennt diesen Wert nicht. Jetzt nutzt die Moment-Finder-Phase den bestehenden Status `analyzing` bei `progress_pct=73`; Regressionstest deckt das ab.
- Qualitaetsbefund aus diesem einen Holdout-VOD bleibt rot gegen alte Repost-Metrik: `final_moments@k10=0/18`, `pre_judge_top80=0/18`. Das ist technische Integration OK, aber KEIN Phase-2-Gate-Gruen.

### Phase 2 — ENTSCHEIDUNGS-GATE (misst, ob der Pfad trägt)
- [x] **2.1** `run_serving_recall_holdout.py` mit `enable_llm_moment_finder=True` → `evaluate_serving_recall_holdout.py`: **Recall-Floor** des LLM-Pfads gegen die 111 Repost-Holdout-Fenster. Erwartung ≫ `final_ranked` 2,7 %.
- [ ] **2.2** **DIE tragende Messung: Top-N-Precision auf `known_repost_match=false`.** `build_llm_moment_precision_review` + Workbench nutzen: menschliches viral-j/n-Rating der Top-N-Momente, die NICHT Clipper-Picks sind. Das beantwortet: sind die LLM-Top-N viral-würdig, wenn sie nicht der exakte Repost sind?
- [ ] **2.3 GATE:** (a) Recall-Floor klar > 2,7 % UND (b) Top-N-Precision auf Nicht-Reposts akzeptabel → Phase 3. Sonst STOPPEN, Ranker NICHT löschen, zurückmelden (dann ist es „anderer Ranker, gleiches Problem").

#### Phase-2 Umsetzung jetzt
- [x] Holdout-Runner bekommt expliziten `--enable-llm-moment-finder` Modus inkl. Provider/Model/Token/Top-N Env und Summary-Metadaten.
- [x] Precision-Review kann eine Non-Repost-Queue bauen: `known_repost_match=true` wird aus der Human-Precision-Queue ausgeschlossen und aus tieferen LLM-Raengen aufgefuellt.
- [x] Workflow-API aggregiert Human-Labels zu `phase2_gate`: Non-Repost Label-Coverage, yes/no Precision, strict Precision und Gate-Readiness.
- [x] Frontend-Workbench zeigt Phase-2-Gate-Status und Non-Repost-Fokus direkt im Review.
- [x] Focused Tests, `import pipeline`, `git diff --check`, Graphify-Rebuild.

#### Phase-2 Implementierungs-Review Notes
- `ml.run_serving_recall_holdout` hat jetzt `--enable-llm-moment-finder` plus `--moment-finder-provider`, `--moment-finder-model`, `--moment-finder-max-moments`, `--moment-finder-max-tokens` und `--moment-finder-confirm-top-n`. Worker-Env setzt `ENABLE_LLM_MOMENT_FINDER=false` ohne Flag und alle Moment-Finder-Env-Werte mit Flag.
- `scripts/run_serving_recall_holdout.sh --wrapper-help` dokumentiert den Phase-2-Recall-Floor-Lauf.
- `ml.build_llm_moment_precision_review` hat `--exclude-known-reposts`. In diesem Modus werden bekannte Repost-Matches aus der Human-Precision-Queue ausgeschlossen und aus tieferen LLM-Raengen aufgefuellt; Summary reportet `non_repost_rows_in_queue`, `excluded_known_repost_count`, `scanned_moment_count` und `max_source_rank_in_queue`.
- Workflow-API startet Precision-Reviews standardmaessig mit `exclude_known_reposts=true`, aggregiert Labels zu `label_summary`, baut `phase2_gate` und kann per `POST /api/llm-moment-review/workflows/{workflow_id}/recall-floor/latest` den neuesten LLM-Holdout-Summary als Recall-Floor anhaengen.
- Frontend-Workbench sendet `exclude_known_reposts=true`, zeigt Non-Repost-Label-Coverage, Human Precision, Strict Precision, Recall-Floor und Blocking Reasons.
- Frontend-Workbench ist auf den Phase-2-Holdout-Review voreingestellt: 10 Holdout-VODs, `VOD Limit=10`, `exclude_known_reposts=true` und `allow_holdout=true`, damit diagnostische Holdout-Reviews nicht am D0-Holdout-Schutz scheitern.
- Gate-Checkboxes 2.1-2.3 blieben nach der Implementierung zunaechst offen: Der echte LLM-Holdout-Lauf und die menschlichen Labels waren zu diesem Zeitpunkt noch nicht ausgefuehrt.
- Verification: `23 passed` fuer Runner/Review/Router/Moment-Finder/Pipeline-Smoke, groessere relevante Backend-Suite `80 passed`, `npm --prefix frontend run lint`, `npm --prefix frontend run build`, `import pipeline`, `git diff --check`, Graphify-Rebuild.

#### Phase-2 Recall-Floor Review Notes
- Erster echter LLM-Holdout `phase2_llm_recall_20260605T220245Z` fand einen Analyzer-Fallback-Bug: `_fallback_moment` crashte bei LLM-Candidates ausserhalb der Boundary-Snap-Toleranz (`KeyError: timestamp outside allowed tolerance`) auf VOD `2778362653`.
- Bugfix: Analyzer-Fallback snappt weiter strikt, clamp't aber bei Fallback-Momenten auf erlaubte Boundary-Spanne statt den Serving-Job zu crashen. Regressionstest in `backend/tests/test_analyzer.py`.
- Sauberer Re-Run: `phase2_llm_recall_fixed_20260605T222634Z`, `enable_llm_moment_finder=true`, `provider=gemini`, `model=gemini-2.5-flash-lite`, `max_moments=180`, `max_tokens=16000`, `jobs_by_status={"done": 10}`.
- Recall-Floor Ergebnis: `final_moments@k12 = 7/110 = 6.3636%`, `final_ranked@k12 = 7/110 = 6.3636%`, `pre_judge_top80 = 10/110 = 9.0909%`, `ranked_pre_judge = 10/110 = 9.0909%`.
- Interpretation: 2.1 ist technisch abgeschlossen und liegt ueber dem alten finalen Ranker-Floor von 2.7%. Phase 2 insgesamt ist noch NICHT abgeschlossen, weil 2.2/2.3 die menschliche Top-N-Precision auf `known_repost_match=false` brauchen.
- Verification nach Bugfix: relevante Backend-Suite `110 passed`.

#### Phase-2 Precision-Review Fix Notes
- Beim ersten Human-Review fiel ein Rationale/Clip-Mismatch auf: `2777621798` Rank 1 beschrieb Pokemon-Fanmail, der geschnittene Clip startete aber erst im Aftermath (`5114-5204`).
- Root Cause: Ein passenderes, aber 210s breites LLM-Fenster (`5006-5216`) wurde durch die alte Blob-Policy komplett verworfen (`>120s`). Uebrig blieb ein kuerzeres, spaeteres Fenster mit stale Rationale. Zusaetzlich enthielt das VOD massive Duplicate-Momente.
- Fix: Moderate lange Momente werden jetzt bis 300s als verwertbar behandelt und auf 90s gekappt; echte Marathon-Blobs bleiben Drop. Moment-Finder und Precision-Review deduplizieren identische/ueberlappende bzw. lange nahe Fenster vor Top-N-Auswahl.
- Frontend-Review zeigt den Transcript-Excerpt jetzt vor der LLM-Rationale und markiert die Beschreibung explizit als `LLM Rationale`, damit sie nicht wie Ground Truth wirkt.
- Bestehender Workflow `20260605T230019201558Z` wurde ohne neuen LLM-Call aus gespeicherten Responses regeneriert und die Review-Clips neu geschnitten. Beispielclip ist jetzt `2777621798_rank01_005006_005096.mp4` mit passendem Pokemon/20,000-Transcript.
- Verification: relevante Backend-Suite `113 passed`, Frontend Lint/Build gruen, `import pipeline`, `git diff --check`, Graphify-Rebuild.
- Zweiter Human-Review-Fund: Airsoft-Shotgun-Rationale war semantisch richtig, aber an falsche LLM-Timestamps (`1913-2003`, ca. `31:53-33:23`) gebunden; passende Transcript-Evidence lag bei ca. `19:46-21:24`.
- Fix: Precision-Review zeigt die LLM-Rationale nur noch als Beschreibung und nutzt sie nicht zur Timestamp-Reparatur. Zeitfenster werden rein numerisch repariert, wenn sie wie kompakte Clock-Werte (`MMSS`/`HHMMSS`) aussehen.
- Bestehender Workflow `20260605T230019201558Z` wurde erneut ohne neuen LLM-Call regeneriert. Airsoft ist nach rein numerischer Clock-Reparatur jetzt `2777621798_rank02_001153_001203.mp4` (`19:13-20:03`) aus dem alten LLM-Fenster `1913-2003`.
- Rebuild-Summary nach Nutzerkorrektur: `queue_rows=120`, `non_repost_rows_in_queue=120`, `known_repost_matches_in_queue=0`, `timestamp_repaired_rows_in_queue=57`, `clip_ready=120`, `all_clips_ready=true`.
- Stale-Clip-Cleanup ergaenzt: Pro Review-Run-Ordner werden nicht mehr referenzierte `.mp4` entfernt. Bestehender Run-Ordner wurde beim letzten Rebuild auf `120` MP4s bereinigt.
- Root-Cause-Deep-Dive danach: Das Transcript war korrekt; der Prompt zeigte aber nur display clocks wie `[19:44 - 19:46]`, waehrend das JSON-Schema Sekunden verlangte. Gemini kodierte mehrfach `MM:SS` ohne Doppelpunkt (`19:38` -> `1938`) statt Sekunden (`1178`).
- Kernfix: Transcript-Prompt-Zeilen enthalten jetzt absolute Sekunden als autoritative Werte (`[t=1184s - t=1186s | clock=19:44 - 19:46]`), die Prompt-Regeln verbieten explizit `19:44 -> 1944`, und D0/Serving/Precision-Review reparieren clock-encoded Outputs rein numerisch. Alte Airsoft-Response repariert im Verify-Run von `1913-2027` auf `1153-1227`.
- Nutzerkorrektur: Rationale-/Transcript-Evidence-basierte Timestamp-Reparatur wurde entfernt. Der Timestamp-Repair benutzt keine Rationale-Terme, keine Stopwords und keine Support-Scores mehr.

### Phase 3 — Erst nach grünem Gate: Ranker offiziell raus
- [ ] **3.1** Flag default AN; alten Ranker-Zweig (`pipeline.py:1723–1799`) entfernen.
- [ ] **3.2** Löschen gemäß RAUS/BLEIBT-Liste in `~/.claude/plans/cozy-sniffing-mango.md` (services/ranker, feature_builder, candidate_generators/normalizer, boundary_*, scorer; ml/ train_*/evaluate_*/hpo/calibrate/audit; ~25 Tests). BLEIBT: Repost-Match-Erzeuger, Holdout-Builder, serving-recall-Eval, `build_llm_moment_precision_review` + Workbench, `post_processor`, `clip_judge` (Multimodal), `selector` (Selektion/Quality-Gate), `timeline_types`.
- [ ] **3.3** Referenzen säubern (routers/jobs, analyzer, high_recall_protection, profile_config, feedback_store, training_workflow_state); Frontend-Funnel-Diagnostik auf Moment-Finder umstellen/entfernen.

### Verifikation (jede Phase)
- [ ] `pytest` grün; `import pipeline`; nach Phase 1 ein echter Serving-Lauf Flag AN (Clips geschnitten, debug zeigt Moment-Finder-Stage); Recall-Floor + Top-N-Precision dokumentiert; nach Phase 3 `grep` ohne Ranker-Reste außerhalb der Validierungs-Eval; Graphify-Rebuild.

### Risiken / Ehrlichkeit
- Phase-2-Gate kann ROT werden (LLM-Top-N nur ~8 % Exact-Pick — Nicht-Repost-Precision unbekannt). Dann Ranker NICHT löschen. Das Gate ist genau dafür da.
- Lokalisierung (Schritt 0) muss VOR Phase 2 sitzen, sonst ist „Recall" bedeutungslos.
- `feature_builder` breit referenziert → Konsumenten vor Delete prüfen (Phase 3).

---



## 2026-06-05 Plan: Frontend-Workflow fuer LLM-Moment-Finder + Precision Review

**Ziel:** Der bisher manuelle Ablauf `ml.run_llm_moment_finder_d0` ->
`ml.build_llm_moment_precision_review --cut-clips` soll aus dem Frontend
bedienbar sein. Ein Klick startet den kompletten Workflow, die UI pollt den
Status und zeigt die geschnittenen Rohclips direkt zum menschlichen Review an.

- [x] Backend-Workflow-API bauen: Start, Status/List, Label speichern und
  sichere Clip-Auslieferung fuer Workflow-Artefakte.
- [x] Workflow-Artefakte pro Run in eigenem Ordner ablegen: Moment-Finder,
  Review-CSV/JSONL/Summary, Clips und Status/Labels.
- [x] Frontend-API-Typen und Client-Funktionen ergaenzen.
- [x] Frontend-Review-Workbench bauen: Konfiguration, Startbutton, Polling,
  Clip-Review-Karten und Label-Speicherung.
- [x] In die bestehende Startseite integrieren, ohne die normale Clip-Pipeline
  zu blockieren.
- [x] Verifizieren: Backend-Tests, Frontend-Typecheck/Build, Graphify-Rebuild
  und Plan-Review.

### Review Notes

- Neuer Backend-Router `routers/llm_moment_review.py` stellt den Workflow unter
  `/api/llm-moment-review/workflows` bereit. Ein Start erzeugt einen eigenen
  Ordner unter `backend/ml/artifacts/llm_moment_review_workflows/<workflow_id>/`.
- Der Workflow ruft im Hintergrund den bestehenden D0-Moment-Finder auf, legt
  dessen Artefakte unter `moment_finder/` ab und baut danach automatisch die
  Precision-Review mit Rohclips unter `review/`.
- Die UI kann Workflows starten, laufende Runs pollen, fertige Review-Rows mit
  Videos anzeigen und menschliche Labels in `review/review_labels.json` sowie
  `review/review_label_events.jsonl` speichern.
- Die normale URL-/Upload-Clip-Pipeline bleibt unveraendert; die neue Workbench
  ist eine zusaetzliche Review-Flaeche auf der Startseite.
- Verification bisher: `backend/tests/test_llm_moment_review_router.py`,
  `backend/tests/test_build_llm_moment_precision_review.py`,
  `backend/tests/test_run_llm_moment_finder_d0.py` (`39 passed`),
  Frontend-Lint fuer die beruehrten Dateien, `npx tsc --noEmit`,
  `npm run build`, `git diff --check` und Graphify-Rebuild via installiertem
  Graphify-Interpreter (`4073 nodes`, `11183 edges`, `109 communities`).

## 2026-06-05 Plan: Umstieg auf OpusClip-Pattern (LLM-Moment-Finder statt Ranker-Gate)

**Strategischer Kontext:** Phase A hat bewiesen, dass der Engpass die **Ranking-Diskrimination** ist
(88/111 im Pool, nur 9 in Top-80, final_ranked 2,7 %). Phase B versucht den lightgbm_ranker zu *reparieren*.
Dieser Plan *umgeht* ihn: Der LLM liest das **ganze** Transkript und IST der Moment-Finder (OpusClip-Muster:
ClipAnything + Virality-Score Hook/Flow/Value/Trend). Damit gibt es das Failure-Mode „Ranker begräbt Clip vor
dem Judge" strukturell nicht mehr.

**Kostenrahmen (verifiziert, Median-VOD 6,7h ≈ 90k Tok):** Voll-Transkript-Pass ~$0,15 (Haiku) – $0,46 (Sonnet)
pro VOD, ~$4–12/Monat bei 25 VODs — **billiger als der jetzige 80-Fenster-Judge** (~200k vs ~90k Input-Tok) bei
~11× Recall. Kostentreiber bleibt allein der multimodale Pass → klein halten auf Top-N.

**Leitmetrik (unverändert):** `run_serving_recall_holdout.py` → `evaluate_serving_recall_holdout.py`,
10 VODs / 111 Positiv-Fenster. Ziel: `final_ranked` 2,7 % → deutlich höher. Neue LLM-Stage in den Funnel instrumentieren.

**Beziehung zu B/C:** Wenn D0 trägt, wird Phase B (Ranker-Reparatur) für den kritischen Pfad **obsolet** —
der lightgbm_ranker sinkt zum nachgelagerten Sortierer/Tie-Breaker. B daher PAUSIEREN bis D0 entschieden ist.
C (mehr/saubere Labels) bleibt orthogonal nützlich (Few-Shot-Kalibrierung + Holdout).

### D0 — Cheap Spike / Go-No-Go (1 VOD, KEINE Integration) ⟵ ZUERST
- [x] D0.1: 1 Train-VOD (NICHT Holdout) mit bekannten Positives nehmen, volles Transkript an Sonnet, Moment-Finder-Prompt,
  JSON-Momente (start/end + hook/flow/value/trend + rationale) emittieren. Standalone-Skript, ~$0,50.
- [x] D0.2: Overlap der LLM-Top-Momente mit den bekannten Clipper-Positives messen (window_overlap ≥ 0.45).
- [x] **D0-Gate:** Findet der LLM die Positives im Volltext (Recall@Top-N deutlich > 8 %)? JA → D1. NEIN → Pattern trägt für
  Jynxzi nicht; zurück zu Phase B. (Entscheidung für $0,50 statt nach 2 Wochen Integration.)

#### D0 Implementation Notes

- Implementiert als Standalone-Skript `backend/ml/run_llm_moment_finder_d0.py`.
  Keine Pipeline-Integration, keine Serving-Flags, kein Ranker-Pfad-Umbau.
- Das Skript waehlt automatisch einen nicht-Holdout-Train-VOD mit bekannten
  Cross-Platform-Positives und vorhandenem Full-VOD-Transcript-Cache; explizite
  `--vod-id`-Auswahl blockt Holdout-VODs, ausser `--allow-holdout` wird bewusst
  gesetzt.
- Output-Schema: JSON-Momente mit `start`, `end`, `hook`, `flow`, `value`,
  `trend`, `virality`, `rationale`; Auswertung misst Recall@Top-N gegen
  bekannte Positives mit `window_overlap >= 0.45`.
- Outputs: `jynxzi_llm_moment_finder_d0.prompt.txt`,
  `jynxzi_llm_moment_finder_d0.moments.json` und
  `jynxzi_llm_moment_finder_d0.eval.json`.
- Dry-run-Smoke waehlte VOD `2757793134` mit 12 bekannten Positives und
  8457 Transcript-Segmenten.
- Echter Sonnet-D0-Run: `llm_moment_finder_d0_done`, VOD `2757793134`,
  30 Momente, 12 Positives, 1 Treffer bei `window_overlap >= 0.45`,
  `recall_at_top_n=0.083333`. Das ist nur minimal ueber 8 % und damit kein
  klarer Go fuer D1; D0 traegt in dieser Form nicht ausreichend.
- Die Gate-Logik wurde nach dem Run von `> 0.08` auf einen expliziten
  `--gate-min-recall`-Default von `0.15` gehaertet, damit 1/12-Grenzfaelle
  nicht als Go zaehlen.
- Verification: `python3 -m py_compile
  backend/ml/run_llm_moment_finder_d0.py
  backend/tests/test_run_llm_moment_finder_d0.py`,
  `backend/.venv/bin/python -m pytest
  backend/tests/test_run_llm_moment_finder_d0.py -q` (`3 passed`) und
  `backend/.venv/bin/python -m ml.run_llm_moment_finder_d0 --dry-run
  --top-n 5 --max-moments 5`.
- Vorgeschriebener Graphify-Befehl mit `python3` scheiterte wegen fehlendem
  Modul `graphify`; Rebuild via installiertem Graphify-Interpreter erfolgreich
  (`3916 nodes`, `10627 edges`, `108 communities`).

### D0.5 — Re-run-Haertung nach Lokalisierungsdiagnose

- [x] Coverage-Cap als Confounder entfernen: `max_moments`/`top_n` fuer D0-
  Re-runs auf OpusClip-aehnliche Dichte anheben.
- [x] Transcript-Noise vor dem Prompt filtern, insbesondere Whisper-Filler wie
  repetitive `you`/kurze Einwort-Stille-Segmente.
- [x] Prompt auf exakte Segment-Grenzen und keine runden Minutenfenster
  schaerfen.
- [x] Eval zusaetzlich operativ messen: Positiv-Center liegt im LLM-Moment
  zaehlt als Treffer, nicht nur `window_overlap >= 0.45`.
- [x] Batch-Support fuer 2-3 Train-VODs ergaenzen, damit D0.5 nicht erneut
  an `n=12` haengt.
- [x] Verifizieren; der echte D0.5-Re-run bleibt der naechste API-kostenpflichtige
  Schritt.

#### D0.5 Implementation Notes

- Defaults fuer den D0.5-Re-run: `max_moments=180`, `top_n=180`,
  `max_tokens=16000`, `match_mode=center_or_overlap`.
- `--vod-limit 3` laesst das Skript die drei besten nicht-Holdout-Train-VODs
  mit Positives und Transcript-Cache auswaehlen; pro VOD werden eigene Prompt-,
  Moment- und Eval-Artefakte geschrieben, plus eine aggregierte Eval-Datei am
  angegebenen `--output-prefix`.
- Transcript-Prompt filtert kurze Filler-Segmente wie einzelnes oder repetitives
  `you`, `uh`, `yeah`, `[music]` und protokolliert Dropped-Segment-Counts im
  Eval-JSON.
- Prompt fordert exakte Transcript-Bracket-Grenzen, kurze lokalisierte Fenster
  und verbietet gerundete Minuten-/Region-Grenzen ausser wenn sie wirklich im
  Transcript stehen.
- Eval misst neben dem bisherigen Overlap auch, ob das LLM-Moment den
  Positiv-Center enthaelt; `match_mode=overlap` bleibt als Diagnosemodus
  verfuegbar.
- Dry-run-Smoke mit `--vod-limit 3` waehlte VODs `2757793134`, `2763470626`
  und `2787027855`, zusammen 36 bekannte Positives.

### D0.5a — Timeout-Robustheit fuer echten Re-run

- [x] Default-Read-Timeout fuer grosse D0.5-Prompts von 120s auf einen
  realistischen Wert anheben.
- [x] Anthropic-Call mit Retry/Backoff gegen Read-Timeouts und transiente
  URL-Fehler absichern.
- [x] Batch-Lauf darf bei einem VOD-Fehler nicht alle vorherigen Artefakte
  verlieren; Fehler sollen als Eval-JSON sichtbar sein und das Gate blockieren.
- [x] Focused Tests und Dry-run erneut ausfuehren.

#### D0.5a Review Notes

- Root Cause des Fehlers: Der echte D0.5-Lauf sendet pro VOD sehr grosse
  Prompts und erlaubt bis zu 16k Output-Tokens; der alte 120s Read-Timeout war
  dafuer zu knapp.
- Neue Defaults: `timeout_sec=900`, `api_retries=2`,
  `retry_backoff_sec=20`. Retry-Events werden als JSON nach stderr geloggt.
- Wenn ein VOD trotz Retries fehlschlaegt, schreibt das Skript ein
  `llm_moment_finder_d0_failed` Eval-JSON fuer diesen VOD. Der Batch laeuft
  weiter, markiert `failed_vod_count`/`failed_vod_ids` und blockiert das Gate,
  statt einen Stacktrace ohne verwertbares Artefakt zu hinterlassen.
- Verification: `py_compile`, `backend/tests/test_run_llm_moment_finder_d0.py`
  (`9 passed`) und Batch-Dry-run mit `--vod-limit 3`.

### D0.6 — Breiten Moment-Finder auf Gemini Flash Lite umstellen

- [x] D0-Skript providerfaehig machen: `gemini` und `anthropic`, aber
  `gemini` als Default.
- [x] Default-Modell fuer den breiten Spike auf `gemini-2.5-flash-lite`
  setzen und Config-Default an `.env`/`.env.example` angleichen.
- [x] Gemini-Call ueber die vorhandene OpenAI-kompatible
  `generativelanguage.googleapis.com`-Schnittstelle implementieren.
- [x] Tests fuer Provider-Routing, Gemini-Parsing und Dry-run/Mock-Pfade
  ergaenzen.
- [x] Verifizieren: focused Tests, Compile, Dry-run und Graphify-Rebuild.

#### D0.6 Review Notes

- `backend/ml/run_llm_moment_finder_d0.py` nutzt jetzt standardmaessig
  `--provider gemini` und `model=gemini-2.5-flash-lite`; Anthropic bleibt nur
  noch als expliziter Diagnosepfad via `--provider anthropic`.
- `backend/config.py` ist an `.env`/`.env.example` angeglichen:
  `gemini_model = "gemini-2.5-flash-lite"`.
- Gemini wird ueber die bereits im Backend genutzte OpenAI-kompatible
  Schnittstelle `https://generativelanguage.googleapis.com/v1beta/openai/`
  aufgerufen.
- Verification bisher: `py_compile`, `backend/tests/test_run_llm_moment_finder_d0.py`
  (`11 passed`) und Batch-Dry-run mit `--vod-limit 3`, der
  `provider=gemini`, `model=gemini-2.5-flash-lite` ausweist. Zusaetzlicher
  Anthropic-Dry-run bestaetigt, dass `--provider anthropic` weiterhin
  `claude-sonnet-4-6` nutzt.

### D0.6a — Gemini JSON-Robustheit und Raw-Response-Artefakte

- [x] Gemini-Call auf strukturierte JSON-Ausgabe konfigurieren.
- [x] Provider-Raw-Response pro VOD immer als Artefakt persistieren, auch wenn
  Parsing danach fehlschlaegt.
- [x] Parser um konservative JSON-Reparatur erweitern, damit typische Gemini-
  Ausgaben mit fehlenden Kommas zwischen Objekten gerettet werden koennen.
- [x] Tests fuer Raw-Response-Persistenz und JSON-Recovery ergaenzen.
- [x] Verifizieren: focused Tests, Compile, Dry-run und Graphify-Rebuild.

#### D0.6a Review Notes

- Gemini-Calls fordern jetzt `response_format={"type": "json_object"}` ueber
  die OpenAI-kompatible Gemini-Schnittstelle.
- Pro VOD wird die rohe Provider-Antwort als
  `*.response.txt` geschrieben, bevor `parse_moments()` laeuft. Dadurch bleibt
  auch bei `JSONDecodeError` das eigentliche Modell-Output-Artefakt erhalten.
- Batch-Evals enthalten jetzt pro VOD `response_path`, damit fehlgeschlagene
  Mehr-VOD-Runs direkt aus dem Aggregat debuggt werden koennen.
- Der Parser repariert konservativ typische fehlende Kommas zwischen JSON-
  Objekten/Feldern, ohne Scores oder Momentdaten zu erfinden.
- Verification: `py_compile`, `backend/tests/test_run_llm_moment_finder_d0.py`
  (`13 passed`) und Batch-Dry-run mit `--vod-limit 3`, der
  `provider=gemini`, `model=gemini-2.5-flash-lite` und die neuen
  `response_path`-Felder ausweist.
- Vorgeschriebener Graphify-Rebuild via `python3` scheiterte wegen fehlendem
  Modul `graphify`; Rebuild via installiertem Graphify-Interpreter erfolgreich
  (`3947 nodes`, `10727 edges`, `110 communities`).

### D0.6b — Gemini Partial-JSON aus Raw-Responses retten

- [x] Fehlermuster der drei echten Gemini-Rohantworten analysieren.
- [x] Parser um balancierte Moment-Objekt-Extraktion erweitern, damit ein
  Fremdzeichen oder ein trunkiertes letztes Objekt nicht den ganzen VOD
  verwirft.
- [x] Tests fuer Fremdzeichen zwischen Objekten und abgeschnittenes letztes
  Objekt ergaenzen.
- [x] Verifizieren: echte `.response.txt`-Artefakte parsebar, focused Tests,
  Compile, Dry-run und Graphify-Rebuild.

#### D0.6b Review Notes

- Der echte Gemini-Run scheiterte nicht an Rate-Limits, sondern an invalidem
  JSON: ein Fremdzeichen vor einem Moment-Objekt und zwei am Ende trunkierte
  Antworten.
- `parse_json_payload()` gibt bei Decode-Fehlern nicht mehr sofort auf,
  sondern versucht danach eine Partial-Rettung der `moments`-Liste.
- Die Partial-Rettung extrahiert nur balancierte `{...}`-Objekte aus der
  `moments`-Liste; unvollstaendige Fragmente werden verworfen.
- Neuer Offline-Modus: `--reuse-responses` liest vorhandene per-VOD
  `*.response.txt`-Artefakte und wertet sie ohne neuen Provider-Call aus.
- Nachtraegliche Auswertung deines Runs via `--reuse-responses`:
  `llm_moment_finder_d0_batch_done`, 0 Fehl-VODs, 406 gerettete Momente,
  24/36 Positives getroffen, `recall_at_top_n=0.666667`; D0.6-Gate damit
  klar bestanden.
- Verification: `py_compile`, `backend/tests/test_run_llm_moment_finder_d0.py`
  (`16 passed`), Batch-Dry-run mit `--vod-limit 3`, `diff --check` und
  Graphify-Rebuild via installiertem Graphify-Interpreter (`3953 nodes`,
  `10742 edges`, `112 communities`).

### D0.7 — Strong-Gemini Re-Scoring fuer Top-Ranking

- [x] Bestehende D0.6b-Momente ohne neuen Generierungsrun als Kandidaten-Set
  laden.
- [x] Leave-one-VOD-out Few-Shots aus echten Train-Positives bauen, damit der
  Ziel-VOD nicht als Oracle-Beispiel in seinen eigenen Prompt gelangt.
- [x] Re-Scoring-Prompt fuer Gemini Pro mit Moment-Kontext, Ursprungsscore und
  Clipper-Pick-Kalibrierung implementieren.
- [x] Re-scored Artefakte schreiben: Prompt, Response, rescored moments,
  per-VOD Eval und Aggregateval mit `recall@12/20/80/180`.
- [x] Offline-Modi fuer Tests und bezahlte Runs: Mock-Response und
  `--reuse-rescore-responses`.
- [x] Verifizieren: Unit-Tests, Compile, Dry-run und Graphify-Rebuild.

#### D0.7 Review Notes

- Implementiert als `--rescore-existing` im D0-Skript. Der Pfad laedt
  vorhandene `*.moments.json`-Artefakte und generiert keine neuen Momente.
- Starkes Modell fuer den Re-Score ist `gemini-2.5-pro`; Flash Lite bleibt der
  guenstige breite Moment-Generator.
- Few-Shots werden Leave-one-VOD-out gebaut: Der Ziel-VOD wird aus seinen
  eigenen Kalibrierungsbeispielen ausgeschlossen; Holdout-VODs bleiben ohne
  explizites `--allow-holdout` ausgeschlossen.
- Outputs pro VOD: `*.rescore_prompt.txt`, `*.rescore_response.txt`,
  `*.rescored_moments.json`, `*.rescore_eval.json`. Aggregat:
  `jynxzi_llm_moment_finder_d0.rescore_eval.json`.
- Eval misst `recall@12`, `recall@20`, `recall@80` und `recall@180`; Gate liegt
  bewusst auf `recall@12 >= 0.25`, nicht auf Generierungs-Recall.
- Dry-run auf den echten D0.6b-Moment-Artefakten funktionierte fuer alle drei
  VODs mit `model=gemini-2.5-pro` und je 12 Few-Shots. Es wurde noch kein
  kostenpflichtiger Gemini-Pro-Run gestartet.
- Verification: `py_compile`, `backend/tests/test_run_llm_moment_finder_d0.py`
  (`19 passed`), Re-Score-Dry-run, CLI-Help-Check, `diff --check` und
  Graphify-Rebuild via installiertem Graphify-Interpreter (`3969 nodes`,
  `10829 edges`, `107 communities`).

### D0.7a — Gemini-Pro Re-Score Partial-JSON retten

- [x] Fehlermuster der echten `*.rescore_response.txt` analysieren.
- [x] Re-Score-Parser um Partial-Rettung vollstaendiger `scores`-Objekte
  erweitern.
- [x] Tests fuer Fremdzeichen/trunkierte Re-Score-Antworten ergaenzen.
- [x] Bestehende Gemini-Pro-Raw-Responses via `--reuse-rescore-responses`
  ohne neuen API-Call neu auswerten.
- [x] Verifizieren: Compile, Tests, Reuse-Eval, Dry-run, Graphify-Rebuild.

#### D0.7a Review Notes

- Der echte Gemini-Pro-Re-Score schrieb alle drei `*.rescore_response.txt`,
  scheiterte aber initial an invalidem/trunkiertem JSON.
- `parse_rescore_response()` rettet jetzt vollstaendige Objekte aus der
  `scores`-Liste. Defekte/trunkierte Score-Fragmente werden verworfen; fehlende
  Kandidaten fallen beim Re-Ranking auf ihren Flash-Lite-Virality-Score zurueck.
- Der Re-Score-Prompt fordert fuer kuenftige Runs kompakte `id`/`score`-
  Objekte ohne Rationale, damit lange Erklaertexte die JSON-Antwort nicht wieder
  abschneiden.
- Gerettete Score-Counts: VOD `2757793134` = 73 Scores, `2763470626` = 95
  Scores, `2787027855` = 61 Scores.
- Reuse-Auswertung ohne neuen API-Call: `llm_moment_finder_d0_rescore_batch_done`,
  0 Fehl-VODs. Operating Points: `recall@12=0.138889`, `recall@20=0.166667`,
  `recall@80=0.305556`, `recall@180=0.666667`.
- D0.7-Gate bleibt fachlich rot: `recall@12` liegt unter der 0.25-Schwelle.
  Der Parser-/Run-Fix ist abgeschlossen, aber Gemini-Pro-Re-Scoring loest die
  Top-12-Selektion noch nicht ausreichend.
- Verification: `py_compile`, `backend/tests/test_run_llm_moment_finder_d0.py`
  (`21 passed`), Re-Score-Dry-run, Reuse-Eval, `diff --check` und
  Graphify-Rebuild via installiertem Graphify-Interpreter (`3971 nodes`,
  `10834 edges`, `106 communities`).

### D0.7b — Vollstaendiges Re-Scoring ohne Fallback-Muell

- [x] Base-Momente mit `duration > 120s` vor dem Re-Score ausfiltern.
- [x] Re-Score-Prompts in kleinere Batches splitten, damit Gemini Pro alle
  Kandidaten scoren kann.
- [x] Fehlende Scores als harten VOD-Fehler behandeln, nicht als
  Virality-Fallback.
- [x] Artefakte fuer Batch-Prompts/-Responses sichtbar schreiben.
- [x] Tests fuer Duration-Filter, Batching und Missing-Score-Failure ergaenzen.
- [x] Verifizieren: Compile, Tests, Dry-run, Graphify-Rebuild.

#### D0.7b Review Notes

- Neuer Default fuer Re-Score: `--rescore-max-duration-sec 120` filtert
  ueberlange Base-Momente vor dem Gemini-Pro-Rerank.
- Neuer Default `--rescore-batch-size 40` splittet den Rerank in mehrere
  kompakte Gemini-Pro-Calls. Bei mehreren Batches werden Artefakte als
  `*.rescore_prompt_001.txt` / `*.rescore_response_001.txt` usw. geschrieben.
- Fehlende Scores sind jetzt ein harter Fehler: Der VOD wird nicht evaluiert,
  wenn ein erwartetes `m<rank>` fehlt. Es gibt keinen stillen Virality-Fallback
  mehr fuer das D0.7-Gate.
- Dry-run auf den echten D0.6b-Momenten: VOD `2757793134` 133 -> 131
  Kandidaten, `2763470626` 141 -> 141, `2787027855` 132 -> 58. Damit wurden
  die 67-Minuten-/Riesenfenster vor dem Re-Score entfernt.
- Der naechste echte Test muss neu laufen; alte Single-Response-Artefakte sind
  fuer D0.7b absichtlich nicht mehr gueltig, weil sie nicht alle Scores
  vollstaendig abdecken.
- Verification: `py_compile`, `backend/tests/test_run_llm_moment_finder_d0.py`
  (`25 passed`), Re-Score-Dry-run mit echten Artefakten, CLI-Help-Check,
  `diff --check` und Graphify-Rebuild via installiertem Graphify-Interpreter
  (`3980 nodes`, `10862 edges`, `109 communities`).

### E2 — Human-Precision-Spot-Check fuer LLM-Momente

- [x] Eigenes Review-Queue-Artefakt fuer LLM-Moment-Top-N bauen, getrennt von
  Ranker/Gold-Training-Queues.
- [x] Pro VOD Top-12 aus bestehenden LLM-Momenten ziehen, `duration > 120s`
  ausfiltern und Transcript-Excerpts anhaengen.
- [x] Bekannte Cross-Platform-Repost-Matches nur als Recall-Floor-Hinweis
  markieren, nicht als Zielmetrik oder Label.
- [x] CSV/JSONL/HTML/Summary schreiben mit Feldern fuer menschliche Labels:
  `human_viral_label`, `human_boundary_label`, `human_notes`.
- [x] Tests fuer Top-N-Auswahl, Duration-Filter, Excerpts und Summary
  ergaenzen.
- [x] Aktuelle 3-VOD-E2-Queue erzeugen und verifizieren.
- [x] Raw-Review-Clips optional aus lokalen Full-VOD-MP4s schneiden und in
  CSV/JSONL/HTML verlinken.
- [x] Fehlende VOD-Quellen im Summary hart sichtbar machen, damit kein Review
  faelschlich als vollstaendig gilt.
- [x] Clip-Cutter testen, aktuelle Queue mit Clips regenerieren und Graphify
  aktualisieren.

#### E2 Review Notes

- Implementiert als `backend/ml/build_llm_moment_precision_review.py`.
  Das Skript schreibt review-only Artefakte und beruehrt keine Trainingslabels.
- Outputs: `jynxzi_llm_moment_precision_review.csv`, `.jsonl`, `.html` und
  `.summary.json`.
- Human-Label-Felder: `human_viral_label`, `human_boundary_label`,
  `human_notes`. Guardrails: `review_required=true`,
  `auto_publish_allowed=false`.
- Aktuelle Queue fuer VODs `2757793134`, `2763470626`, `2787027855` erzeugt:
  36 Rows, Top-12 pro VOD, `duration > 120s` gefiltert.
- Duration-Filter in der Queue: `2757793134` 133 -> 131, `2763470626`
  141 -> 141, `2787027855` 132 -> 58.
- Bekannte Repost-Matches in der Top-12-Queue: 1/36. Das ist nur ein
  Recall-Floor-Hinweis; die 35 nicht-matchenden Momente sind jetzt
  menschlich zu bewertende Produktkandidaten, keine automatischen Fehler.
- Verification: `py_compile`,
  `backend/tests/test_build_llm_moment_precision_review.py` plus
  `backend/tests/test_run_llm_moment_finder_d0.py` (`27 passed`),
  Artefakt-Zaehlung CSV/JSONL/HTML/Summary und `diff --check`.
- Graphify-Rebuild via installiertem Graphify-Interpreter erfolgreich:
  `3997 nodes`, `10926 edges`, `110 communities`.
- Raw-Clip-Erweiterung: `--cut-clips` schneidet rohe MP4-Review-Clips aus
  lokalen Full-VOD-Dateien unter
  `backend/tmp/storage/_download_cache/twitch/vod_<vod_id>.mp4` und verlinkt
  sie in CSV/JSONL/HTML. `--require-clips` kann den Lauf hart fehlschlagen
  lassen, wenn nicht alle Clips erzeugt wurden.
- Aktuelle Queue mit Clips regeneriert: 36 Rows, 12 fertige MP4s fuer VOD
  `2763470626`, 24 Rows mit `clip_status=missing_source` fuer VODs
  `2757793134` und `2787027855`, weil deren Full-VOD-MP4s lokal nicht im
  Download-Cache liegen. Keine ffmpeg-Fehler.
- Raw-Clip-Output:
  `backend/ml/artifacts/jynxzi_llm_moment_precision_review_clips/`.
- Verification nach Raw-Clip-Erweiterung: `py_compile`,
  `backend/tests/test_build_llm_moment_precision_review.py` plus
  `backend/tests/test_run_llm_moment_finder_d0.py` (`29 passed`),
  Artefakt-Statuscheck, HTML-Video-Link-Check und `diff --check`.

### E2a — Full-VOD-Download-Preflight fuer LLM und Raw-Clips

- [x] Gemeinsamen Helper bauen, der Twitch-VOD-IDs in den bestehenden
  Downloader-Cache nachlaedt und fehlende/unverfuegbare Quellen als harte
  Fehler sammelt.
- [x] LLM-D0-Runner so absichern, dass echte neue Provider-Calls erst nach
  erfolgreichem Full-VOD-Preflight starten.
- [x] Review-Builder um `--download-missing-vods` erweitern, damit fehlende
  Quellen vor dem Clip-Schnitt nachgeladen werden.
- [x] Tests fuer Cache-Hit, Download-Failure und "kein Provider-Call bei
  fehlendem VOD" ergaenzen.
- [x] Verifizieren: focused Tests, Compile, Dry-run/Help, Graphify-Rebuild und
  Plan-Review.

#### E2a Review Notes

- Neuer Helper `backend/ml/vod_download_preflight.py` prueft lokale Full-VODs
  unter `backend/tmp/storage/_download_cache/twitch/vod_<vod_id>.mp4` und
  nutzt bei fehlenden Quellen den bestehenden
  `services.downloader.ensure_cached_download(...)` gegen
  `https://www.twitch.tv/videos/<vod_id>`.
- `run_llm_moment_finder_d0.py` macht diesen Preflight standardmaessig vor
  echten neuen Provider-Calls. Wenn ein VOD nicht heruntergeladen werden kann,
  schreibt das Skript `llm_moment_finder_d0_preflight_failed` bzw.
  `llm_moment_finder_d0_rescore_preflight_failed` und ruft Gemini/Anthropic
  nicht auf. `--dry-run`, `--reuse-responses`, `--reuse-rescore-responses` und
  Mock-Responses bleiben offline; `--skip-vod-download-preflight` ist nur der
  explizite Escape-Hatch.
- `build_llm_moment_precision_review.py` hat jetzt `--download-missing-vods`;
  ausserdem impliziert `--cut-clips` den Download-Preflight. Wenn Clips
  angefordert werden, muessen alle VODs vor dem Clip-Schnitt verfuegbar sein;
  sonst beendet das Skript mit JSON-Fehler statt einer halbvollstaendigen
  Review-Queue.
- Verification bisher: `py_compile`, focused Tests
  `backend/tests/test_vod_download_preflight.py`,
  `backend/tests/test_build_llm_moment_precision_review.py`,
  `backend/tests/test_run_llm_moment_finder_d0.py` (`35 passed`),
  CLI-Help-Smokes, D0-Dry-run-Smoke mit temporaerem Output und `diff --check`.
- Vorgeschriebener Graphify-Rebuild via `python3` scheiterte wegen fehlendem
  Modul `graphify`; Fallback-Rebuild via installiertem Graphify-Interpreter
  erfolgreich (`4026 nodes`, `11007 edges`, `106 communities`).

### E2b — Run-spezifische Raw-Clip-Ordner

- [x] `--clip-output-dir` als stabilen Root behandeln und pro `--cut-clips`
  Lauf einen eigenen Child-Ordner erzeugen.
- [x] Optionales `--clip-run-id` fuer reproduzierbare Review-Run-Pfade
  ergaenzen.
- [x] CSV/JSONL/HTML-Links auf den run-spezifischen Ordner schreiben.
- [x] Tests fuer deterministische Run-ID und relative Clip-Pfade ergaenzen.

#### E2b Review Notes

- Raw Clips werden nicht mehr flach in
  `backend/ml/artifacts/jynxzi_llm_moment_precision_review_clips/`
  geschrieben. Default ist jetzt ein timestamp-basierter Unterordner wie
  `..._clips/20260605T...Z/`.
- Wer reproduzierbare Pfade will, kann `--clip-run-id <id>` setzen; der Wert
  wird dateisystem-sicher normalisiert.
- CSV/JSONL/HTML enthalten weiterhin `clip_path` und `clip_rel_path`, aber
  `clip_rel_path` zeigt jetzt auf den Run-Unterordner.
- Verification: `py_compile`,
  `backend/tests/test_vod_download_preflight.py`,
  `backend/tests/test_build_llm_moment_precision_review.py`,
  `backend/tests/test_run_llm_moment_finder_d0.py` (`36 passed`),
  CLI-Help-Smoke, `diff --check` und Graphify-Rebuild via installiertem
  Graphify-Interpreter (`4028 nodes`, `11012 edges`, `108 communities`).

### E2c — Run-spezifische LLM-Moment-Finder-Artefaktordner

- [x] LLM-Moment-Finder-CLI so umstellen, dass jede normale Ausfuehrung einen
  eigenen Run-Ordner unter `backend/ml/artifacts` anlegt.
- [x] Alle generierten Prompt-/Response-/Moments-/Eval-Dateien und ein Manifest
  in diesen Run-Ordner schreiben.
- [x] Precision-Review-Builder um `--moment-run-dir` erweitern, damit ein
  konkreter LLM-Run-Ordner als Input gewaehlt werden kann.
- [x] Tests fuer neuen Run-Ordner, Manifest und Review aus Run-Ordner
  ergaenzen.
- [x] Verifizieren: focused Tests, Compile, CLI-Smokes, `diff --check`,
  Graphify-Rebuild und Plan-Review.

#### E2c Review Notes

- Normale `ml.run_llm_moment_finder_d0`-CLI-Generierung ohne explizites
  `--output-prefix` schreibt jetzt in
  `backend/ml/artifacts/jynxzi_llm_moment_finder_runs/<run-id>/`.
- Der Run-Ordner enthaelt Prompt-, Response-, Moments- und Eval-Artefakte mit
  dem bestehenden Prefix-Namen plus `manifest.json`. Das Manifest enthaelt
  `run_id`, `run_dir`, `output_prefix`, `moment_prefix`, `vod_ids`, Provider,
  Modell und den finalen Report.
- `--run-id` setzt einen reproduzierbaren Run-Ordnernamen; `--run-dir` erlaubt
  einen expliziten Ordner. Explizites `--output-prefix` bleibt als Legacy-
  Override erhalten.
- `ml.build_llm_moment_precision_review` kann jetzt mit `--moment-run-dir`
  einen konkreten LLM-Run-Ordner als Moment-Input verwenden. Wenn ein
  `manifest.json` vorhanden ist, wird dessen `moment_prefix` genutzt; sonst
  faellt der Builder auf den Standardprefix im Ordner zurueck. Ohne explizite
  `--vod-ids` liest der Builder die VOD-Auswahl ebenfalls aus dem Manifest.
- Verification bisher: `py_compile`, focused Tests
  `backend/tests/test_run_llm_moment_finder_d0.py`,
  `backend/tests/test_build_llm_moment_precision_review.py`,
  `backend/tests/test_vod_download_preflight.py` (`38 passed`),
  CLI-Help-Smokes und `diff --check`.
- Vorgeschriebener Graphify-Rebuild via `python3` scheiterte wegen fehlendem
  Modul `graphify`; Fallback-Rebuild via installiertem Graphify-Interpreter
  erfolgreich (`4037 nodes`, `11045 edges`, `107 communities`).

### D1 — Transcript-Chunker + Moment-Schema
- [ ] D1.1: Chunker: Median-VOD (~90k) in EINEM Call; Marathons (28–46h, bis ~600k Tok) in überlappende Fenster (~60k, Overlap ~3 min).
- [ ] D1.2: Strukturiertes Output-Schema (Pydantic): `start,end,hook,flow,value,trend,virality,rationale`. Boundary-Dedup an Chunk-Grenzen.
- [ ] D1.3: Unit-Tests auf Transkript-Fixtures (kein API-Call im Test).

### D2 — `services/moment_finder.py` (neuer Service)
- [ ] D2.1: Volltext → Momente via Anthropic. Konfigurierbares Modell (Haiku→Sonnet-Eskalation), **Prompt-Caching** des Transkripts
  (für Multi-Achsen-Pass), Jynxzi-Curation-Prompt.
- [ ] D2.2: Few-Shot aus echten Clipper-Reposts — **nur aus TRAIN-VODs**, NIE Holdout (Oracle-Leakage-Lesson beachten).
- [ ] D2.3: Fail-open: API-Ausfall → degradieren auf bestehenden Generator/Ranker-Pfad, Degradation loggen (Serving-Lesson).
- [ ] D2.4: Tests (gemockter Provider).

### D3 — Pipeline-Integration hinter Flag `enable_llm_moment_finder`
- [ ] D3.1: Flag AN: Momente werden Kandidaten-Quelle, gehen direkt in Selektion; lightgbm_ranker nur noch Tie-Breaker/Sortierer (oder bypass).
- [ ] D3.2: Alten Pfad intakt lassen (A/B). Neue Stage in Recall-Funnel-Instrumentierung (`candidate_recall_report.py` / eval) aufnehmen.
- [ ] D3.3: Selector/Quality-Gate auf LLM-Momente abstimmen (virality statt ranker_score als primäres Selektionssignal).

### D4 — Eval-Gate gegen die 111-Positiv-Baseline
- [ ] D4.1: Holdout-Lauf mit Flag AN (Cache-Bust beachten). Batch-API −50 % für Offline-Eval nutzen.
- [ ] **D4-Gate:** `final_ranked`-Recall vs aktuell 2,7 %. Schlägt LLM-Pattern den Ranker-Pfad? JA → D5. NEIN → Prompt/Schema iterieren oder verwerfen.

### D5 — Multimodal-Confirm + Härtung (nur falls D4 besteht)
- [ ] D5.1: Bestehenden multimodalen Judge auf Top-N LLM-Momente behalten (Präzisions-Bestätigung, kein Recall-Gate).
- [ ] D5.2: Prompt-Caching für 4-Achsen-Scoring (Input −~65 %), Latenz-Budget, Marathon-VOD-Chunking validieren.
- [ ] D5.3: Kosten/Latenz pro VOD messen, Modell-Tier final wählen (Haiku-wide vs Sonnet-wide).

### Nicht-Ziele / Caveats
- Keine Cuts/Untertitel/Reframing — nur roher Clip.
- Few-Shot/Kalibrierung NIE aus Holdout-VODs (Oracle-Leakage).
- D0 ist Pflicht-Gate VOR jeder Integration — der $0,50-Test entscheidet die ganze Strategie.
- Whisper/Feature-Extraction bleibt lokal/GPU (kein API-Kostenfaktor, aber die echte Wall-Clock/OOM-Wand — unverändert).

## 2026-06-05 Plan: Training-Transcriber Speed-Hebel

- [x] Pruefen, ob `faster-whisper` im Trainingsimage/Backend verfuegbar ist.
- [x] Training-Docker-Wrapper so anpassen, dass `backend/.env` die GPU-Speed-
  Defaults nicht wieder auf `auto`/leer zuruecksetzt.
- [x] Full-VOD-Audio-Worker fuer Trainingslaeufe konservativ von 2 auf 3
  anheben, aber per Host-Env ueberschreibbar lassen.
- [x] Verifizieren: Shell-Syntax und Wrapper-Dry-Run/Help.

### Review Notes

- Laufend: Der aktuelle Container wird dadurch nicht beeinflusst; die Hebel
  greifen erst beim naechsten Start von `scripts/train_local_docker.sh` bzw.
  `scripts/train_clipfinding_models_docker.sh`.
- `scripts/train_local_docker.sh` setzt trainingsspezifisch nach dem Env-File:
  `WHISPER_BACKEND=faster-whisper`, `WHISPER_DEVICE=cuda`,
  `WHISPER_COMPUTE_TYPE=float16` und
  `CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS=3`.
- Bei expliziter Backend-CPU-Runtime erzwingt der Wrapper kein `cuda`; dann
  bleibt das Device `auto` und der Transcriber faellt intern auf CPU zurueck.
- Alle Werte bleiben ueberschreibbar ueber
  `AI_CLIPPING_LOCAL_TRAIN_WHISPER_BACKEND`,
  `AI_CLIPPING_LOCAL_TRAIN_WHISPER_DEVICE`,
  `AI_CLIPPING_LOCAL_TRAIN_WHISPER_COMPUTE_TYPE` und
  `AI_CLIPPING_LOCAL_TRAIN_FULL_VOD_AUDIO_WORKERS`.
- Verification: `bash -n scripts/train_local_docker.sh
  scripts/train_clipfinding_models_docker.sh`,
  `scripts/train_local_docker.sh --wrapper-help`,
  `scripts/train_clipfinding_models_docker.sh --date-from 2026-03-01
  --date-to 2026-06-04 --dry-run` und scoped `git diff --check`.

## 2026-06-05 Plan: ML-Artefakte gezielt aufraeumen

- [x] Alte Proposal-Artefakte entfernen, weil die alte Proposal-Version nicht
  mehr verfuegbar sein soll.
- [x] Backup-/Snapshot-/Pre-Fix-Artefakte entfernen, die nicht mehr als
  aktueller Trainings- oder Holdout-Input dienen.
- [x] Leere bzw. no-op Review-/Diagnose-Artefakte entfernen, die beim naechsten
  Lauf bei Bedarf neu erzeugt werden.
- [x] Verifizieren: keine Proposal-Artefakte, keine `.bak`-/`pre_*`-/partial-
  Reste und keine leeren Artefaktdateien in `backend/ml/artifacts`.

### Review Notes

- Abgeschlossen: Trainingsdaten, Modellfiles, Phase-B-Diagnose,
  Hard-Negative-Training und letzter Serving-Holdout bleiben erhalten.
  Die aktuellen Cross-Platform-Fingerprint-Outputs wurden danach gezielt als
  B3-Cache-Bust entfernt.
- Entfernt: alte `jynxzi_proposal_*`-Artefakte, `.bak`-Fingerprints,
  `pre_*`-/partial-Snapshots, alte Holdout-Eval-Varianten (`newmodel`,
  `gatefix`, `allfixes`, `lgbm`), der alte Clipfinding-Audit-Snapshot und
  leere/no-op Review-/Diagnose-Artefakte.
- Umfang: 32 Dateien, ca. 47 MB.
- Verification: `find` zeigt keine Proposal-/Backup-/Snapshot-Muster mehr,
  `find -empty` zeigt keine leeren Artefaktdateien, `rg` findet keine
  Proposal-Keys mehr in `backend/ml/artifacts`. `jynxzi_artifact_health.json`
  verweist nicht auf geloeschte Artefaktpfade.

## 2026-06-05 Plan: Phase B Ranking-Diskrimination

- [x] 5-Min-Diagnose ohne Retrain: Score-/Rang-Verteilung der 88
  In-Pool-Positives gegen `serving_threshold=4.707` aus dem aktuellen
  Serving-Holdout ziehen.
- [x] B1 umsetzen: Same-VOD-Look-alike-Hard-Negs at scale exportieren
  (Ziel: mindestens 20 pro Positiv-VOD, Holdout-Leakage bleibt verboten).
- [x] B2 umsetzen: Chat-Velocity/Chat-Spike im Ranker-Training gezielt
  aufwerten, ohne Serving-Feature-Paritaet zu brechen.
- [x] Verification vor Retrain: focused Tests, Syntax/Compile, Artefakt-
  Smoke-Diagnose und Graphify-Rebuild.
- [x] Cache-Bust fuer B3: die drei `jynxzi_cross_platform_fingerprint_*`-
  Output-Dateien entfernen, damit der Retrain frisch rematcht.
- [ ] Danach ein einziger B3-Retrain; B-Gate:
  `pre_judge_top80 >= 30%`.

### Review Notes

- Diagnose-Artefakte:
  `backend/ml/artifacts/jynxzi_phase_b_positive_rank_score_diagnosis.json`
  und `.csv`. Ergebnis: 88/111 Positives erreichen `ranked_pre_judge`;
  65/88 liegen ueber `serving_threshold=4.707`, aber Median-Rang ist 636
  und 48/88 liegen hinter Rang 500. Threshold-only ist damit kein sinnvoller
  Fix; B1+B2 sind erforderlich.
- B1: `ml/export_serving_hard_negatives.py` exportiert jetzt
  `selection_strategy=same_vod_positive_lookalike`. Safe-Negatives werden
  pro VOD nach Feature-Aehnlichkeit zu bekannten Positivfenstern derselben VOD
  sortiert, nicht mehr nur nach Pre-Judge-Rang. Holdout-VODs bleiben per
  Policy ausgeschlossen, ausser bei explizitem Diagnose-Flag.
- B1-Scale: Default `pre_judge_top_k` fuer Serving-Hard-Negatives ist 500;
  globaler `max_rows`-Default ist 0/unbegrenzt, damit der Per-VOD-Zielwert
  von 20 nicht durch einen 300er-Globalcap unterlaufen wird. Wenn ein
  Globalcap gesetzt wird, verteilt der Export fair ueber VODs.
- B2: `ml/train_ranker.py` behandelt Chat-Velocity-Features als positive
  Trainingssignale: Prioritaet in der Feature-Auswahl, positive
  Sign-Constraints und LightGBM-`feature_contri` fuer
  `chat_velocity_score`, `chat_rate_derivative_score`,
  `chat_ema_crossing_score`, `chat_spike_coverage`,
  `chat_activity_coverage`, `chat_delta`.
- Verification: focused Tests fuer Hard-Negative-Export, Ranker-Training und
  Workflow-State (`68 passed`), `py_compile`, Shell-Syntax,
  scoped `git diff --check`, Export-Smoke (`40` Rows fuer 2 VODs, 20/VOD,
  keine under-target VODs). Vorgeschriebener Graphify-Befehl mit `python3`
  scheiterte wegen fehlendem Modul; Fallback-Rebuild via installiertem
  Graphify-Interpreter erfolgreich (`3880 nodes`, `10528 edges`,
  `108 communities`).
- B3-Cache-Bust: `jynxzi_cross_platform_fingerprint_rows.jsonl`,
  `jynxzi_cross_platform_fingerprint_summary.json` und
  `jynxzi_cross_platform_fingerprint_diagnostics.jsonl` sind geloescht;
  `find backend/ml/artifacts -name 'jynxzi_cross_platform_fingerprint_*'`
  liefert keine Treffer.

## 2026-06-05 Plan: Proposal-Cleanup Rest-Audit

- [x] Aktiven Code, Scripts, Frontend-Typen und Env-Dateien auf alte
  Proposal-/Trainable-Flags, Debug-Schemas und tote Helfer durchsuchen.
- [x] Nur erreichbare Altpfade entfernen; generierte historische Artefakte
  nicht als Produktcode behandeln.
- [x] Focused Verification ausfuehren: `rg`-Audit, `py_compile`, relevante
  Pytest-Suites, Shell-Syntax und Graphify-Rebuild.
- [x] Review Notes mit entfernten Resten und verbleibenden Nicht-Code-Artefakten
  ergaenzen.

### Review Notes

- Aktiver Code-Audit ist sauber: keine `proposal`-/`PROPOSAL`-/
  `TRAINABLE_CANDIDATE`-/`proposal_score`-/`proposal_lift`-Treffer mehr in
  `backend`, `scripts` oder `frontend`, wenn generierte Artefakte und Graphify
  ausgeklammert sind.
- Entfernt: tote Settings `chat_join_timeout_sec`,
  `enable_post_publish_labels`, `enable_word_level_boundaries`; tote Env-Zeile
  `TRAINING_VOD_CACHE_TTL_SEC`; letzter aktiver Proposal-Wortlaut im
  Holdout-Runner-Test.
- Historische `proposal_*`-Keys bleiben in alten Debug-/Holdout-Artefakten und
  alten Plan-Notizen stehen. Sie sind keine erreichbare Runtime-Oberflaeche.
- Verification: `py_compile` fuer betroffene Dateien, Runner-Test
  `backend/tests/test_run_serving_recall_holdout.py` (`6 passed`),
  Shell-Syntax fuer Holdout-/Training-Wrapper, Rest-`rg`-Audit.
- Vorgeschriebener Graphify-Befehl mit `python3` scheiterte wegen fehlendem
  Modul `graphify`; Rebuild via installiertem Graphify-Interpreter erfolgreich
  (`3871 nodes`, `10500 edges`, `109 communities`).

## 2026-06-05 Plan: Alte Proposal-Version komplett entfernen

- [x] Alle erreichbaren Opt-in-Schalter, Env-Variablen und Config-Felder fuer
  die alte trainable Proposal-Schicht entfernen.
- [x] Alte Proposal-Module, Trainings-/Materialize-Skripte und direkte Tests
  loeschen.
- [x] Serving-Pipeline auf direkten Dense-Inventory-/Heuristik-Merge in den
  Ranker-Pool umstellen, ohne leere Proposal-Platzhalter weiterzugeben.
- [x] Diagnose-, Holdout- und Frontend-Schemas von `proposal_input`,
  `proposal_kept`, Proposal-Scores und Proposal-Lift befreien.
- [ ] Verifizieren: Rest-Audit ohne Proposal-Treffer, focused Tests,
  Syntaxchecks, Shell-Syntax und Graphify-Rebuild.

### Review Notes

- Laufend: Die alte Proposal-Schicht soll nicht mehr per Flag reaktivierbar
  sein. Der Phase-A-Holdout nutzt nur noch Dense Inventory + direkten Ranker-
  Merge; historische Phase-A-Notizen weiter unten sind durch diesen Removal-
  Plan ueberholt.

## 2026-06-05 Plan: Phase-A-Holdout-Start hart absichern

- [x] Runner-Worker-Environment so setzen, dass ein altes Container-Environment
  `ENABLE_TRAINABLE_CANDIDATE_PROPOSAL=true` nicht mehr in den Holdout durchsickert.
- [x] Wrapper-Startpfad ebenfalls mit Phase-A-Env versehen, damit auch der
  Parent-Prozess im Container den richtigen Modus sieht.
- [x] Tests fuer Env-Override und Summary/Progress-Metadaten ergaenzen.
- [x] Verifizieren: focused runner tests, Wrapper-Syntax/Help, Graphify-Rebuild.

### Review Notes

- `backend/ml/run_serving_recall_holdout.py` erzwingt fuer jeden Pipeline-Worker
  jetzt den Phase-A-Vertrag: `ENABLE_DENSE_CANDIDATE_INVENTORY=true`,
  `DENSE_CANDIDATE_INVENTORY_SOURCE_CAP=700`,
  `ENABLE_TRAINABLE_CANDIDATE_PROPOSAL=false`. Das ueberschreibt explizit auch
  ein altes Container-Environment mit Proposal-Opt-in.
- `scripts/run_serving_recall_holdout.sh` exportiert dieselben Phase-A-Flags im
  `--local`-Pfad und gibt sie bei `docker exec` mit `-e` an den Parent-Prozess
  weiter. Die Wrapper-Hilfe dokumentiert den erzwungenen Phase-A-Modus.
- Runner-Progress und Run-Summary enthalten `phase_a_pipeline_env`, damit ein
  laufender/abgeschlossener Holdout sofort auditierbar ist.
- Verification: `backend/tests/test_run_serving_recall_holdout.py` (`6 passed`),
  `py_compile`, `bash -n scripts/run_serving_recall_holdout.sh`, Wrapper-Help,
  lokaler Wrapper-Dry-Run mit absichtlich falscher Aussen-Env
  (`true 700 false` im Summary), scoped `git diff --check`.
- Vorgeschriebener Graphify-Befehl mit `python3` scheiterte wegen fehlendem
  Modul `graphify`; Graphify-Rebuild via installiertem Graphify-Interpreter
  erfolgreich (`3992 nodes`, `10920 edges`, `108 communities`).

## 2026-06-05 Plan: Clip-Finding auf OpusClips-Niveau — Funnel-Sanierung (aus Deep-Dive-Review)

**Leitmetrik (Gate jeder Phase):** `ml/run_serving_recall_holdout.py` → `ml/evaluate_serving_recall_holdout.py`,
oracle-free, 10 disjunkte neueste VODs / 111 Positiv-Fenster. Getrackt: `merged`, `ranked_pre_judge`,
`pre_judge_top80`, `final_ranked`. Cache-Bust vor Retrain: die 3 `jynxzi_cross_platform_fingerprint_*`-Output-Dateien
wegmoven (Match-Signatur ist input-, nicht code-basiert).

**Baseline (Lauf 2026-06-05):** semantic 79 % · dense 72 % · merged 50 % · **pre_judge_top80 8 % (9/111)** · final_ranked 2,7 %.

**Kern-Diagnose:** Generierung findet ~80 %. Bindender Engpass = **Ranking-Diskrimination**: von 56 Positives
im Ranker-Pool landen nur **9 in den Top-80**. Merge-Leak (79 %→50 %) und finale Selektion (`final_top_k=12`,
pipeline.py:2214) sind nachgelagerte Verstärker, nicht die Wurzel — selbst k=80 ergäbe nur 9. Architektur hat
zwei redundante gelernte Scoring-Schichten; die obere (Proposal) hat **Netto-Recall-Beitrag 0** (Beleg:
`merged`=56, `proposal_input`=65, `normalized`=56 → Proposal-Scorer verwirft genau die 9 Inventory-Positives).

### Phase A — Simplifikation (Surface −1 Schicht; darf NICHT regressieren)

- [x] Phase-A-Umsetzung 2026-06-05: Dense-Inventory aus dem Proposal-Block herausziehen, direkt in den
  Ranker-Pool mergen, Proposal-Scorer standardmaessig deaktivieren, Judge im Holdout-Pfad off lassen und
  fokussiert gegen Tests + Holdout-Eval pruefen.
- [x] A1: **Proposal-Scorer-Schicht entfernen.** `dense_inventory`-Kandidaten direkt in den Heuristik-Merge geben
  statt durch `proposal_scorer.select_candidates`. pipeline.py:1747-1896-Block ausbauen; config.py:261
  `enable_trainable_candidate_proposal=False` als Default (Pfad tot behalten). Erwartung: `normalized`/`ranked_pre_judge` ≥ 56,
  idealerweise 65 (Inventory-Positives nicht mehr verworfen). Vorher isolierten Inventory-Beitrag bestätigen.
- [x] A2: **4 Merge-Stufen auf 1 reduzieren** (heuristic + recall_carry + inventory in einen Pool), Source-Caps lockern.
  services/candidate_generators.py `merge_prioritized_candidates`, candidate_normalizer.py. Gate: `merged`-Recall ↑
  Richtung 79 %; Pool-Größe protokollieren (kein unkontrollierter Blowup).
- [x] A3: **Clip-Judge aus dem kritischen Pfad.** Optionaler Präzisions-Rerank auf guter Shortlist, nie als Gate
  (Memory: Judge = Co-Signal, Gate No-op, judge_scope 9 %). config.py `enable_clip_judge` für Holdout aus;
  Gate-Bypass in selector.py. Gate: `final_ranked` darf nicht sinken.
- [ ] A-Gate: Holdout-Lauf. Akzeptanz: `pre_judge_top80` + `final_ranked` ≥ Baseline (keine Regression), Schichten −1.

### Phase B — Echter Hebel: Ranking-Diskrimination (Ziel: pre_judge_top80 8 %→≥30 %)

Vorheriger Hard-Neg-Versuch (40 Negs / 2 VODs) war zu dünn → flach. Jetzt systematisch:

- [ ] B1: **Same-VOD Hard-Negative-Mining at scale.** Pro Positiv-VOD die look-alike, NICHT geclippten Fenster als
  Negatives ziehen (Chat-Spike-ähnlich / Embedding-nah am Positiv, aber kein Repost-Match). Ziel ≥ 20 Hard-Negs
  pro Positiv-VOD statt 40 gesamt. ml/export_serving_hard_negatives.py: pro-VOD-look-alike statt globaler Pool.
  Kein Leakage; temporaler Split (build_recent_vod_holdout) bleibt.
- [ ] B2: **Chat-Velocity stärker gewichten** in Generation + Ranking. Importances: `chat_*` niedrig,
  `duration`/`chapter_coherence` dominant → Modell lernt Form statt Hype. services/chat_analyzer.py +
  feature_builder.py: Chat-Spike-zentrierte Seeds; Monotonie-Constraint/Gewicht für Chat-Features im lightgbm_ranker.
- [ ] B3: Retrain ranker (train_jynxzi_vod.sh) nach Cache-Bust.
- [ ] B-Gate: Holdout. Akzeptanz: `pre_judge_top80` ≥ 30 %. Flach → Diagnose: Daten-Decke (→ Phase C) vs Feature-Lücke.

### Phase C — Daten-Decke heben (parallel zu B startbar)

- [ ] C1: Repost-Match-Rate erhöhen (aktuell ~50 %) → mehr saubere Positives.
- [ ] C2: Zweiten/dritten Jynxzi-Clipper-Kanal als Label-Quelle erschließen (Decke = eine Person × Geschmack).
- [ ] C-Gate: Anzahl saubere Positives ↑, Holdout-Recall stabil/↑.

### Nicht-Ziele / Caveats
- Keine Cuts/Untertitel/Reframing — nur Roh-Clip-Findung.
- Precision (~0,01–0,03) ist Lower Bound (partielle Ground Truth, ein Clipper), kein Müll-Beweis. Recall = Leitmetrik.
- Reihenfolge bewusst: A senkt Komplexität ohne Recall-Risiko, B ist der eigentliche Fix, C hebt die Decke. Nur EINE
  Phase gleichzeitig gegen den Holdout messen, sonst sind die Deltas nicht zuordenbar.

### Review Notes
- **Phase A Code umgesetzt (2026-06-05):** `dense_inventory` wird jetzt unabhaengig
  vom trainable Proposal-Scorer erzeugt. Im Default (`ENABLE_TRAINABLE_CANDIDATE_PROPOSAL=false`)
  wird der Ranker-Pool direkt aus Peak/Fallback plus einem deduplizierten Supplemental-Pool
  aus Recall-Carry, Semantic, Dense und Dense-Inventory gebaut. Der alte Proposal-Scorer-Pfad
  bleibt als expliziter Opt-in erhalten.
- Neue Defaults: `ENABLE_DENSE_CANDIDATE_INVENTORY=true`,
  `DENSE_CANDIDATE_INVENTORY_SOURCE_CAP=700`,
  `ENABLE_TRAINABLE_CANDIDATE_PROPOSAL=false` in `backend/config.py`,
  `backend/.env` und `backend/.env.example`.
- A2 ist als Merge-Simplifikation umgesetzt, ohne den 2026-06-04-Recall-Carry
  zu duplizieren: Carry-Kandidaten werden vor dem direkten Merge selektiert,
  dedupliziert und als Preserve-IDs in denselben Merge gegeben. Proposal-
  Inventory- und Proposal-Shortlist-Merges laufen nur noch bei Proposal-Opt-in.
- A3 war im Holdout-Runner bereits Default (`ENABLE_CLIP_JUDGE=false`) und wurde
  durch die Runner/Eval-Tests abgesichert; der normale Produkt-Default
  `enable_clip_judge=True` bleibt unveraendert.
- Verification: `py_compile`, Phase-A/Unique/Judge-Pipeline-Tests (`5 passed`),
  Candidate/Runner/Evaluator-Tests (`30 passed`), scoped `git diff --check`,
  vorgeschriebener System-Graphify-Befehl scheiterte wegen fehlendem Modul,
  Graphify-Rebuild via installiertem Graphify-Interpreter erfolgreich
  (`3991 nodes`, `10917 edges`, `111 communities`).
- Nicht ausgefuehrt: A-Gate-Holdout-Lauf gegen die 10-VOD/111-Positive-Baseline.
  Naechster Schritt: `scripts/run_serving_recall_holdout.sh` bzw.
  `ml.run_serving_recall_holdout` mit neuem Run-ID laufen lassen und
  `pre_judge_top80`/`final_ranked` gegen die Baseline vergleichen.

## 2026-06-04 Plan: Serving Holdout Download Cache dauerhaft halten

- [x] Serving-Holdout-Runner so konfigurieren, dass Twitch-VOD-MP4s nicht nach 24h evicted werden.
- [x] Sicherstellen, dass der normale Frontend-/Serving-Cache nicht global auf unendlich gestellt wird.
- [x] Tests fuer Pipeline-Env ergaenzen.
- [x] Verifizieren: focused tests, `git diff --check`, Graphify-Rebuild.

### Review Notes

- `backend/ml/run_serving_recall_holdout.py` setzt fuer Pipeline-Subprozesse
  jetzt `TWITCH_DOWNLOAD_CACHE_TTL_SEC=-1`. Im Downloader bedeutet `-1`:
  Twitch-Cache bleibt aktiv, aber `_evict_expired_cache` loescht nichts wegen
  TTL. `0` waere falsch, weil damit der Twitch-Cache deaktiviert wuerde.
- Scope ist bewusst nur der Serving-Holdout-Runner. Die globale
  `backend/config.py`-Default-TTL bleibt `86400`, damit normale Frontend-/
  Serving-Jobs weiterhin den 24h-Raw-VOD-Cache verwenden.
- Verification bisher: `py_compile`, focused
  `backend/tests/test_run_serving_recall_holdout.py` (`6 passed`),
  Host-Env-Smoke und Container-Env-Smoke zeigen
  `TWITCH_DOWNLOAD_CACHE_TTL_SEC=-1`, scoped `git diff --check`.
- Graphify-Rebuild via
  `/home/antonio/.local/share/uv/tools/graphifyy/bin/python` erfolgreich:
  `3988 nodes`, `10909 edges`, `108 communities`; finaler scoped
  `git diff --check` inklusive Graphify-Artefakten sauber.

## 2026-06-04 Plan: Recall-Lecks an Merge + Finaler Selektion (die zwei echten Bruchstellen)

**Kontext / Evidenz:** Phase 0 (10 VODs / 111 Positiv-Fenster, oracle-free) + Phase 1 (Daten
abgeschlossen) zeigen empirisch: Kandidaten-Generierung FINDET ~80 % der echten Clips
(semantic 88, dense 80 absolut), aber zwei Stellen **außerhalb des Rankers** vernichten den Recall.
Der Phase-1-Retrain (Hard-Negs) hat — wie erwartet — den Produkt-Recall **nicht** bewegt
(flat im Rauschen, KEINE Regression: Funnel bis quality_kept absolut flach, proposal_kept 34→40).
Ranker-Tuning ist also nicht der Hebel. Die zwei echten Bruchstellen (absolute Positiv-Fenster):

- **Merge/Dedup + Source-Caps:** semantic 88 → merged 42 (−46). Positives werden gefunden, aber
  vor dem Ranking weggedupt/gekappt.
- **Finale Selektion:** quality_kept 41 → final_ranked 2 → final_moments 0. Strukturell durch
  `final_top_k=12` (pipeline.py:2214) + Diversity gekappt.

**Disziplin (nicht verhandelbar):** Gegen Phase-0-Holdout mit **absoluten Zahlen** messen, nicht
Ratios (Nenner schwankt: 103→111 durch Re-Match). Final-Stage-Counts sind einstellig + LLM-Judge
nicht-deterministisch → Median über ≥3 Läufe ODER judge-off-Variante reporten, sonst misst man Rauschen.

### Stufe A — Metrik zuerst geradeziehen (BLOCKIEREND)
Sonst optimiert man gegen eine strukturell gekappte Zahl.
- [x] Klären: `final_top_k=12` kappt `final_ranked`-Recall hart bei ~12/111 ≈ 11 % — selbst ein
      perfekter Ranker kann nicht mehr. Entscheiden, was „beste Clips finden" misst:
      **precision@k** (von 12 Ausgaben wie viele clip-würdig) + **per-VOD recall@k** statt Recall
      über ALLE Clipper-Fenster.
- [x] Holdout-Eval um precision@k / recall@k erweitern; einstellige Final-Counts mit Judge-Varianz
      über Mehrfachlauf (Median) absichern.
- [x] Günstigen Judge-Mehrfach-Sampling-Pfad operationalisieren: Holdout-Pipeline soll den
      bestehenden Median-Consensus im Judge mit N=3 Samples pro Kandidat nutzen, ohne das ganze
      VOD dreimal durch Download/ASR/Feature-Building laufen zu lassen.
- [x] Judge-off als deterministischen Standard-Messpfad fuer Holdout-Eval setzen; Judge-on nur
      explizit als Referenzlauf aktivieren.
- [x] Akzeptanz: Eval gibt eine produkt-sinnvolle, rausch-robuste Kennzahl aus.

### Stufe B — Merge/Dedup-Leck (88 → 42) diagnostizieren & fixen  [HÖCHSTER HEBEL]

#### Arbeitsplan Stufe B Umsetzung (2026-06-04)
- [x] Merge-Diagnose aus Debug-Artefakten ableiten: semantic/dense/dense_inventory gegen merged vergleichen und Drop-Gründe approximieren.
- [x] `merge_prioritized_candidates` so erweitern, dass Duplicate-Gruppen nicht mehr blind "first wins" behalten, sondern der label-freie beste Repräsentant überlebt.
- [x] Source-Caps weiterhin als Compute-Schutz behalten, aber dedupte Ersatzfenster nicht schlechter ausrichten lassen.
- [x] Regressionstests für besseren Duplicate-Repräsentanten und Source-Cap-Verhalten ergänzen.
- [x] Focused Tests, `git diff --check`, Graphify-Rebuild und Review Notes ergänzen.

Knüpft an `services/candidate_generators.py` (`merge_prioritized_candidates`,
`_is_duplicate_candidate`, `_candidate_windows_overlap_like_duplicates` mit overlap/shorter≥0.7;
`source_caps`/`supplemental_source_cap`).
- [x] Diagnose: pro verlorenem Positiv-Fenster (semantic→merged) bestimmen, welche Regel es entfernt
      — Dedup (overlap≥0.7) vs. Source-Cap. Label-frei über die Holdout-Debug-Dirs.
- [x] Prüfen, ob bei überlappenden Kandidaten der BEHALTENE Repräsentant schlechter zur
      clip-würdigen Boundary alignt als der entfernte (Dedup behält das falsche Fenster).
- [x] Fix label-frei: Dedup so, dass der beste Repräsentant (Boundary-Qualität/score_hint) überlebt;
      kein Positiv-tragendes Fenster soll ohne Repräsentant in den Ranker gelangen.
- [x] Akzeptanz: merged-Recall (absolut) steigt Richtung semantic-Niveau (Ziel ≥60/111), ohne die
      Kandidatenzahl in den teuren Stufen zu sprengen.

### Stufe C — Finale Selektion (quality_kept 41 → final_ranked 2)
`services/selector.py` (`select_post_judge_final_candidates`, `select_diverse_ranked_candidates`:
`source_cap=min(4,...)`, temporaler min-gap; `_select_with_gap`).
- [ ] Diversity-Logik prüfen: tauscht der min-gap/source-cap echte Positives gegen schwächere
      „diverse" Kandidaten? Beispiele aus Holdout-Trace ziehen.
- [ ] `judge_scope` (proposal_kept 40 → judge_scope 17): Recall-Reserve-Slots so tunen, dass tiefere
      Positives gejudged werden (ohne Kosten zu sprengen).
- [ ] Akzeptanz: bei festem Produkt-k steigt precision@k / per-VOD recall@k messbar (Median, robust).

### Stufe D — Boundary-Refine / Post-Processing (final_ranked 2 → final_moments 0)
`analyzer.refine_candidates` + `post_processor.snap_boundaries`/`enforce_diversity`.
- [ ] Prüfen, ob Boundary-Snap die letzten Positives aus dem Positiv-Fenster schiebt oder
      enforce_diversity sie dropt. Falls ja: Snap an Positiv-Overlap koppeln statt nur an Satzgrenzen.
- [ ] Akzeptanz: final_moments verliert keine final_ranked-Positives mehr (Δ=0 im Holdout).

### Querschnitt / Done-Kriterien
- [ ] Jede Stufe gegen Phase-0-Holdout in **absoluten Zahlen** messen (Median ≥3 Läufe bei Final-Stage).
- [ ] Tests grün (`PYTHONPATH=backend pytest backend/tests -q`), `git diff --check`.
- [ ] Nach Backend-Änderungen Graphify-Rebuild.
- [ ] Nach jeder Stufe Review-Notes hier ergänzen (Baseline absolut → Ergebnis absolut).

### Review Notes
- **Stufe A umgesetzt (2026-06-04):** `backend/ml/evaluate_serving_recall_holdout.py`
  berichtet jetzt zusaetzlich zu den alten globalen Stage-Recall-Zahlen eine
  produktnahe `product_selection`-Metrik fuer `final_ranked` und
  `final_moments` bei `k=3,5,10,12`. Pro `k` werden
  `selected_positive_candidates`, `selected_candidate_rows`,
  `matched_positive_windows`, `positive_windows`, `precision`,
  `precision_at_capacity` und `per_vod_recall` ausgegeben. Damit wird
  `final_top_k=12` nicht mehr faelschlich als unbounded global recall bewertet.
- Fuer Mehrfachlaeufe gibt `aggregate.robust_product_selection` Medianwerte pro
  VOD und Metrik aus; `vods_with_3plus_runs` zeigt, ob die im Plan geforderte
  Judge-Varianz-Absicherung tatsaechlich belastbar ist. Der normale
  `latest_per_vod=True`-Pfad bleibt erhalten; Median wird relevant, wenn mit
  `--include-repeat-vod-runs` evaluiert wird.
- Smoke gegen echte Debug-Artefakte:
  `PYTHONPATH=backend backend/.venv/bin/python -m ml.evaluate_serving_recall_holdout
  --output /tmp/jynxzi_stage_a_eval_smoke.json --max-jobs 2
  --allow-unknown-oracle-state` erzeugte die neuen Aggregate-Felder
  `product_selection` und `robust_product_selection` erfolgreich.
- Verification: `py_compile`, `backend/tests/test_evaluate_serving_recall_holdout.py`
  (`6 passed`), `backend/tests/test_run_serving_recall_holdout.py` +
  `backend/tests/test_evaluate_candidate_funnel.py` (`9 passed`), scoped
  `git diff --check`, and Graphify rebuild (`3970 nodes`, `10850 edges`,
  `105 communities`).
- **Stufe A Nachtrag Judge-Varianz operationalisiert (2026-06-04):**
  `scripts/run_serving_recall_holdout.sh`/`ml.run_serving_recall_holdout`
  fuehrt weiterhin nur einen Pipeline-Lauf pro VOD aus, setzt aber fuer den
  Pipeline-Subprozess `CLIP_JUDGE_SAMPLE_COUNT=3` und
  `CLIP_JUDGE_UNCERTAIN_SAMPLE_COUNT=3`. Damit sampelt der bestehende
  `services.clip_judge`-Median-Consensus jeden gejudgten Kandidaten dreimal und
  stabilisiert genau die nicht-deterministische Judge-Stufe, ohne Download,
  ASR, Feature-Building oder Candidate-Generation dreimal zu wiederholen.
- Der Judge-Cache-Key wurde auf Version 3 erweitert und enthaelt jetzt
  `sample_count`, `uncertain_sample_count` und `uncertain_margin`; alte
  1-Sample-Cache-Hits koennen dadurch keine 3-Sample-Holdout-Messung
  verfaelschen. `clip_judge_stage.json` und `clip_judge_stage_start` loggen
  Sampling/Cache-Settings sichtbar mit.
- Verification Nachtrag: `py_compile`, `backend/tests/test_clip_judge.py` +
  `backend/tests/test_run_serving_recall_holdout.py` (`32 passed`),
  `backend/tests/test_pipeline_integration.py -k 'clip_judge or judge'`
  (`3 passed, 22 deselected`), Env-Smoke (`CLIP_JUDGE_SAMPLE_COUNT=3`,
  `CLIP_JUDGE_UNCERTAIN_SAMPLE_COUNT=3`), scoped `git diff --check`, and
  Graphify rebuild (`3972 nodes`, `10863 edges`, `109 communities`).
- **Stufe A Korrektur: Judge-off ist jetzt Standard (2026-06-04):**
  Der guenstigste und sauberste A2-Pfad ist nicht, den Judge zu glaetten,
  sondern ihn fuer Merge-/Final-Selection-Eval auszuschalten. Der
  Holdout-Runner setzt fuer normale Runs jetzt `ENABLE_CLIP_JUDGE=false`.
  Damit misst `scripts/run_serving_recall_holdout.sh` deterministisch den
  Ranker-/Diversity-/Boundary-Pfad ohne LLM-Rauschen und ohne Gemini/Ollama-
  Kosten. Ein Judge-on-Referenzlauf bleibt explizit moeglich mit
  `--enable-judge`; nur dann werden `CLIP_JUDGE_SAMPLE_COUNT=3` und
  `CLIP_JUDGE_UNCERTAIN_SAMPLE_COUNT=3` gesetzt.
- Hard-Negative-Collection nutzt weiterhin den bisherigen Produktpfad mit
  Judge-on (`enable_clip_judge=True`, `judge_sample_count=1`), damit diese
  Workflow-Aenderung nicht versehentlich das Training/Mining-Verhalten
  umstellt.
- Verification Judge-off-Korrektur: `py_compile`,
  `backend/tests/test_run_serving_recall_holdout.py` +
  `backend/tests/test_run_serving_hard_negative_collection.py` (`9 passed`),
  `backend/tests/test_clip_judge.py` +
  `backend/tests/test_pipeline_integration.py -k 'clip_judge or judge'`
  (`30 passed, 22 deselected`), Env-Smoke (`default false None None`,
  `judge_on true 3 3`), scoped `git diff --check`, and Graphify rebuild
  (`3974 nodes`, `10868 edges`, `105 communities`).
- **Stufe B umgesetzt (2026-06-04):** Diagnose gegen Phase-0-Holdout
  `20260604T162338Z` bestaetigte die Plan-Baseline: `semantic=88/111`,
  `dense=80/111`, `merged=42/111`. Global hoehere Source-Caps und reine
  Score-Top-N-Auswahl waren nicht ausreichend (`max_total=600` nur
  `58/111`). Der wirksame label-freie Fix ist deshalb temporale
  Recall-Abdeckung vor dem Merge: pro Source und 60s-Bucket wird der beste
  Semantic/Dense-Kandidat als bounded Carry-Forward ausgewaehlt, maximal 700
  Kandidaten pro VOD.
- `services.candidate_generators.merge_prioritized_candidates` sortiert Merge-
  Kandidaten jetzt nach einem label-freien Merge-Selection-Score statt nach
  chronologischer Eingabereihenfolge und schreibt optional Audit-Zeilen mit
  `duplicate`/`source_cap`/`outside_time_range`. Preserved Kandidaten umgehen
  weiterhin Cap/Dedup und verbrauchen den normalen Source-Cap-Zaehler nicht.
- `pipeline.py` schreibt neue Debug-Artefakte:
  `debug/candidate_merge_audit.jsonl` und `debug/candidate_merge_summary.json`.
  Die Summary enthaelt `merge_recall_carry_forward` mit Poolgroesse,
  ausgewaehlten Carry-Kandidaten, Added-Count und Settings.
- Replay-Akzeptanz gegen bestehende Holdout-Debug-Artefakte:
  `old_merged=42/111`, `new_merged_replay=60/111`,
  `old_rows=3145`, `new_rows=7577`, `carry_selected=6811`.
  Damit erreicht Stufe B das Ziel `>=60/111`; die Kandidatenzahl steigt auf
  ca. 758 pro VOD und bleibt deutlich unter dem Dense-Inventory/Proposal-Input-
  Pool.
- Verification Stufe B: `py_compile`, focused Pipeline-Smoke
  (`3 passed, 40 deselected`), Candidate/Runner/Evaluator tests
  (`30 passed`), scoped `git diff --check`, required system Graphify command
  failed because system Python lacks `graphify`, then installed Graphify
  rebuilt successfully via
  `/home/antonio/.local/share/uv/tools/graphifyy/bin/python` (`3988 nodes`,
  `10909 edges`, `108 communities`).

---

## 2026-06-04 Plan: Serving Holdout Default Flags

- [x] Set serving holdout fail-fast on by default.
- [x] Set serving holdout progress interval to 3 seconds by default.
- [x] Keep an explicit opt-out for non-fail-fast runs.
- [x] Verify parser defaults and focused tests.

### Review Notes

- `backend/ml/run_serving_recall_holdout.py` now defaults to
  `fail_fast=True`, `progress_interval_sec=3.0`, and still defaults to
  `workers=2`.
- `--fail-fast` now uses `argparse.BooleanOptionalAction`, so
  `--no-fail-fast` is the explicit opt-out.
- The run summary and initial progress event include `fail_fast` and
  `progress_interval_sec`, making defaults visible in artifacts/logs.
- `scripts/run_serving_recall_holdout.sh --wrapper-help` now documents the
  defaults instead of recommending redundant flags.
- Verification passed: `py_compile`, focused
  `backend/tests/test_run_serving_recall_holdout.py` (`5 passed`),
  `bash -n scripts/run_serving_recall_holdout.sh`, wrapper help,
  scoped `git diff --check`, and Graphify rebuild via the installed graphify
  tool.

## 2026-06-04 Plan: Long-VOD Video Feature Cache

- [x] Confirm why the current holdout has no `video_frames.json`.
- [x] Make cached video frames reusable even when full visual extraction would
  be skipped for long VODs.
- [x] Add explicit diagnostics for video feature cache hit/miss/skip.
- [x] Add focused regression coverage and verify.
- [x] Rebuild Graphify after backend changes.

### Review Notes

- `video_frames.json` caching already existed, but long VODs over
  `pipeline_full_visual_feature_max_duration_sec` were not even attempting a
  cache read because the full visual extraction branch was skipped first.
- `backend/pipeline.py` now separates cache reuse from new extraction:
  `_extract_video_features_cached(..., allow_extract=False)` loads an existing
  `video_frames.json` but returns `[]` without recomputing if no cache exists.
  This keeps the 2h long-VOD extraction guard while allowing previously cached
  visual features to be reused.
- The pipeline now calls the video feature cache path whenever a stable feature
  cache source exists, even when long-VOD full extraction is disabled.
- Added regression tests for long-VOD cache reuse and no-cache skip behavior.
- Verification passed:
  `python3 -m py_compile backend/pipeline.py backend/tests/test_pipeline_integration.py`,
  `PYTHONPATH=backend backend/.venv/bin/python -m pytest -q
  backend/tests/test_pipeline_integration.py -k 'video_feature_cache or frontend_feature_cache'`,
  scoped `git diff --check`, and Graphify rebuild via the installed
  graphify tool. The required system `python3 ... graphify.watch...` still lacks
  the `graphify` module.

## 2026-06-04 Plan: Serving Recall Holdout CUDA Auto Runtime

- [x] Fix `scripts/run_serving_recall_holdout.sh` so `AI_CLIPPING_BACKEND_RUNTIME=auto`
  adds `docker-compose.cuda.yml` whenever Docker GPU access is available.
- [x] Verify shell syntax and wrapper help still work.
- [x] Confirm the CUDA branch is exercised without starting a long holdout run.
- [x] Document the result.

### Review Notes

- `scripts/run_serving_recall_holdout.sh` now appends `docker-compose.cuda.yml`
  in the `auto` runtime branch when Docker GPU access is available. Previously
  only an explicit `AI_CLIPPING_BACKEND_RUNTIME=cuda` appended the CUDA Compose
  override.
- `scripts/lib_backend_container.sh` now detects GPU availability through
  Docker's `--gpus all` path instead of requiring host `nvidia-smi` to succeed.
  On this WSL host, host `nvidia-smi` returns `GPU access blocked by the
  operating system`, while Docker GPU access succeeds.
- Verification passed: `bash -n scripts/lib_backend_container.sh`,
  `bash -n scripts/run_serving_recall_holdout.sh`,
  `scripts/run_serving_recall_holdout.sh --wrapper-help`, escalated wrapper
  trace showing `BACKEND_RUNTIME=cuda` plus `docker-compose.cuda.yml`, and
  scoped `git diff --check`.

## 2026-06-04 Plan: Restore MediaPipe Solutions Face Detector

- [x] Pin MediaPipe to a version range that still provides `mp.solutions`.
- [x] Verify that backend serving and ranker training share the same face
      extraction path and are both affected by the dependency.
- [x] Run focused syntax/tests and document expected runtime impact.

### Review Notes

- A direct `pyproject.toml` pin to `mediapipe==0.10.21` restores
  `mp.solutions`, but breaks dependency resolution because that MediaPipe
  version requires `protobuf<5` while the current TensorFlow stack resolves to
  protobuf 6.
- Kept the normal project dependency broad and added a controlled Docker image
  override in both `backend/Dockerfile.local-train` and `backend/Dockerfile.cuda`:
  install the full ML/vision stack first, then replace MediaPipe with
  `mediapipe==0.10.21` via `--no-deps`.
- Verified `ai-clipping-train-local:latest` and `ai-clipping-backend-cuda:latest`
  both report `face_detector=mediapipe` and
  `mediapipe_status=solutions_face_detection`; MTCNN remains available only as
  fallback.
- The same issue affected the main pipeline: `backend/pipeline.py` and
  `backend/ml/build_twitch_training_dataset.py` both call
  `services.video_features.extract(...)`, so both serving and training used the
  same MTCNN fallback before this image fix.
- Focused verification: `backend/.venv/bin/python -m pytest -q
  backend/tests/test_video_features.py` passed with 19 tests.
- Expected runtime impact: the gain applies to the video feature stage only.
  The code budget lists MediaPipe face detection at about 55s per 1h video on
  CPU; MTCNN fallback is typically slower, so this should save roughly the
  face-detector delta during visual extraction. End-to-end gain is smaller when
  Whisper, download, transcript matching, or LLM judging dominate.

## 2026-06-04 Plan: Ranker Train New Match Delta Logs

- [x] Add explicit JSON logs around Cross-Platform matching in the ranker train flow.
- [x] Report previous, current, added, removed, and retained match counts per prepare phase.
- [x] Include enough context to tell whether new Shorts created new usable matches.
- [x] Verify shell syntax and the delta helper with a small offline fixture.

### Review Notes

- Added `ml.cross_platform_match_delta`, a small CLI that compares the
  cross-platform match rows before and after each prepare/rematch phase.
- `train_jynxzi_vod.sh` now snapshots
  `jynxzi_cross_platform_fingerprint_rows.jsonl` before
  `prepare_cross_platform_fingerprint_rows`, then logs a JSON event:
  `ranker_train_cross_platform_match_delta`.
- The event includes `previous_rows`, `current_rows`, `added_rows`,
  `removed_rows`, `retained_rows`, unique-match counts, affected VOD counts,
  summary status/counts, and up to five `added_match_samples`.
- Verification passed: `bash -n backend/ml/train_jynxzi_vod.sh`,
  `py_compile backend/ml/cross_platform_match_delta.py`, a fixture run of the
  delta CLI, and Graphify rebuild.

## 2026-06-04 Plan: Restore Legacy Local Training Image Default

- [x] Set `scripts/train_local_docker.sh` default image mode back to `legacy`.
- [x] Update wrapper help so the documented default matches behavior.
- [x] Verify syntax/help and document the result.

### Review Notes

- `scripts/train_local_docker.sh` now defaults to
  `AI_CLIPPING_LOCAL_TRAIN_IMAGE_MODE=legacy` again. This restores the previous
  training behavior using `backend/Dockerfile.local-train`, which provides
  `zsh` and `/app/backend/.venv/bin/python`.
- The backend-image mode is still available explicitly with
  `AI_CLIPPING_LOCAL_TRAIN_IMAGE_MODE=backend`, but it is no longer the default.
- Verification passed: `bash -n scripts/train_local_docker.sh` and
  `scripts/train_local_docker.sh --wrapper-help`.

## 2026-06-04 Plan: Fix Ranker Training Shell In Backend Image

- [x] Reproduce/confirm missing `/bin/zsh` in the backend CUDA image.
- [x] Switch local ranker training launcher to a shell available in the backend image.
- [x] Verify syntax and a non-destructive Docker help/path check.
- [x] Document the fix and exact retry command.

### Review Notes

- The CUDA backend image has `/bin/bash` and `/bin/sh`, but no `/bin/zsh`.
  `ml.local_train` was hard-coded to launch `ml/train_jynxzi_vod.sh` through
  `/bin/zsh`, causing `FileNotFoundError`.
- `backend/ml/local_train.py` now launches the ranker shell flow via
  `/bin/bash`, and `backend/ml/train_jynxzi_vod.sh` now has a matching
  `#!/bin/bash` shebang.
- Verification passed: `py_compile backend/ml/local_train.py`,
  `bash -n backend/ml/train_jynxzi_vod.sh`,
  `scripts/train_local_docker.sh --wrapper-help`, Graphify rebuild, scoped
  `git diff --check`, and a Docker-wrapper smoke test against the rebuilt
  `ai-clipping-backend-cuda:latest` image confirming `_train_command()[0]` is
  `/bin/bash`.
- Retry command:
  `./scripts/train_local_docker.sh --date-from 2026-03-01 --date-to 2026-05-31`

## 2026-06-04 Plan: Combined Proposal + Ranker Training Script

- [x] Add a Docker wrapper that trains Proposal first and Ranker second.
- [x] Require and forward `--date-from` / `--date-to` to both underlying scripts.
- [x] Support explicit extra args for Proposal and Ranker without mixing them.
- [x] Verify shell syntax/help/dry-run output and document the result.

### Review Notes

- Added `scripts/train_clipfinding_models_docker.sh`.
- The script enforces `--date-from` and `--date-to`, then forwards both flags
  to `scripts/train_proposal_local_docker.sh` and `scripts/train_local_docker.sh`.
  Proposal currently accepts those flags for compatibility; Ranker requires
  them.
- Training order is fixed: Proposal first, Ranker second. If Proposal fails,
  the Ranker step does not start.
- Extra args are intentionally explicit: repeat `--proposal-arg ARG` or
  `--ranker-arg ARG` so options cannot accidentally go to the wrong training
  wrapper.
- Verification passed: `bash -n scripts/train_clipfinding_models_docker.sh`,
  `scripts/train_clipfinding_models_docker.sh --help`, and a dry-run confirming
  both generated subcommands contain `--date-from 2026-03-01 --date-to
  2026-05-23`.

## 2026-06-04 Plan: Hard-Negative Collection Runner fuer Mai-VODs

Ziel: Jynxzi-VODs im Zeitraum 2026-04-27 bis 2026-05-26 automatisch
oracle-frei durch die Serving-Pipeline schicken und danach Hard-Negative-Rows
aus hochgerankten, nicht positiven Kandidaten exportieren.

- [x] Bestehenden Serving-Runner wiederverwenden/parametrisieren, damit
      Progress, Env und Media-Cleanup identisch bleiben.
- [x] Date-Range-VOD-Auswahl aus `jynxzi_twitch_vod_inventory.jsonl` bauen,
      Training-/Review-Holdout-VODs standardmaessig ausschliessen.
- [x] Neues Container-Wrapper-Skript fuer den Hard-Negative-Collection-Run
      ergaenzen.
- [x] Nach den Pipeline-Laeufen automatisch `export_serving_hard_negatives.py`
      ausfuehren und Summary schreiben.
- [x] Tests/Syntax/Dry-Run/Graphify verifizieren und Review dokumentieren.

**Review 2026-06-04:** `ml.run_serving_hard_negative_collection` selektiert
standardmaessig Jynxzi-VODs im Bereich 2026-04-27 bis 2026-05-26, zieht bei
gesetzten Twitch-Credentials frische VOD-Metadaten aus der Twitch-API nach und
faellt sonst auf `jynxzi_twitch_vod_inventory.jsonl` zurueck. VODs aus
`jynxzi_training_holdout_policy_rows.jsonl` sind standardmaessig ausgeschlossen,
damit die Collection keine Holdout-Leakage erzeugt. Pipeline-Ausfuehrung,
Live-Progress, oracle-free Env und Source-Media-Cleanup werden ueber den
parametrisierten `run_pipeline_job` des Holdout-Runners wiederverwendet.
`export_serving_hard_negatives.py` kann jetzt optional auf konkrete VOD-IDs
filtern; der Collection-Runner nutzt diesen Filter, damit nur Mai-Collection-
Jobs in den Hard-Negative-Export eingehen. `scripts/run_serving_hard_negative_collection.sh`
startet wie die anderen Runner standardmaessig im Backend-Container mit
Auto-Rebuild/Auto-Start und CUDA-Auto-Runtime. Verifiziert:
`bash -n scripts/run_serving_hard_negative_collection.sh`,
`py_compile` fuer Runner/Exporter/Holdout-Runner,
`PYTHONPATH=backend backend/.venv/bin/python -m pytest -q
backend/tests/test_run_serving_hard_negative_collection.py
backend/tests/test_export_serving_hard_negatives.py` (`7 passed`) und lokaler
Dry-Run ohne Twitch-Refresh. Der Collection-Zeitraum wurde anschliessend auf
2026-04-27..2026-05-26 umgestellt, weil das aktuelle Inventar damit exakt 10
verfuegbare Nicht-Holdout-VODs liefert. Geplante VODs:
`2758541573`, `2759304833`, `2763470626`, `2764197142`, `2766519381`,
`2768960946`, `2771207666`, `2771989634`, `2774602304`, `2776013057`.
20 VODs im Zeitraum werden wegen Holdout-Policy korrekt ausgeschlossen. Fuer
den echten Lauf sollte Twitch-Refresh aktiv bleiben. Graphify wurde
aktualisiert (`3936 nodes`, `10748 edges`, `106 communities`).

## 2026-06-03 Plan: Whisper Transcription Speedups ohne Qualitaetsverlust

Ziel: Die langen Frontend-/Holdout-VOD-Transcriptions schneller und
robuster machen, ohne das Whisper-Modell bewusst zu verkleinern oder
Transcript-Qualitaet zu opfern.

- [x] Konfigurierbare Transcriber-Parameter ergaenzen: Backend,
      Chunk-Laenge, Word-Timestamps und isolierte Worker-Batch-Groesse.
- [x] `word_timestamps` standardmaessig deaktivieren, weil downstream nur
      Segmenttext/-zeiten und `words_per_second` genutzt werden.
- [x] Isolierte Chunk-Worker streaming machen, damit jeder fertige Chunk sofort
      Cache + Progress schreibt.
- [x] Isolierte Worker in kleinere Batches splitten, statt 172 Chunks in einem
      einzigen Batch zu verarbeiten.
- [x] Optionales `faster-whisper`/CTranslate2 Backend mit OpenAI-Whisper
      Fallback integrieren.
- [x] Tests/Syntax/Graphify pruefen und Review-Notizen dokumentieren.

**Review 2026-06-03:** `services/transcriber.py` nutzt jetzt dynamische
Transcription-Settings: `WHISPER_BACKEND=auto`, `WHISPER_COMPUTE_TYPE`,
`WHISPER_WORD_TIMESTAMPS=false`, `TRANSCRIPTION_CHUNK_SECONDS=180` und
`TRANSCRIPTION_WORKER_BATCH_SIZE=12`. Der OpenAI-Whisper-Pfad setzt
`word_timestamps` nicht mehr hart auf `True`; `words_per_second` wird weiterhin
aus Whisper-Segmentdauer und Text/Words berechnet. Isolierte Worker werden in
kleinere Batches gesplittet und `_transcribe_chunks_isolated` yieldet Resultate
streaming, sodass der Parent jeden fertigen Chunk sofort in den durable
Transcript-Cache schreibt und Progress melden kann. `faster-whisper` ist als
ML-Dependency ergaenzt und wird im `auto` Backend bevorzugt, falls installiert;
sonst faellt der Code auf den bestehenden OpenAI-Whisper-Pfad zurueck.
Die aktive `.env` und `.env.example` enthalten die neuen Runtime-Schalter.
Verifiziert: `PYTHONPATH=backend backend/.venv/bin/python -m pytest -q
backend/tests/test_transcriber.py backend/tests/test_transcriber_progress.py`
(`22 passed`), Pipeline-Cache-Fokus-Tests (`4 passed`), Component-Env-Tests
(`4 passed`) und `py_compile`.

## 2026-06-03 Plan: Trainingsscripts auf Backend-Image umstellen

Ziel: `scripts/train_local_docker.sh` und dadurch auch
`scripts/train_proposal_local_docker.sh` nutzen standardmaessig das aktuelle
Backend-Image statt eines separat driftenden Training-Images.

- [x] Backend-Images um fehlende lokale Trainings-Runtime-Dependency ergaenzen.
- [x] `train_local_docker.sh` mit Backend-Image-Default und Legacy-Fallback
      umstellen.
- [x] Auto-Rebuild-Helper fuer Training-Wrapper wiederverwenden.
- [x] Shell-Syntax und ungefaehrliche Help/Mode-Pfade verifizieren.

**Review 2026-06-03:** `train_local_docker.sh` nutzt jetzt
`AI_CLIPPING_LOCAL_TRAIN_IMAGE_MODE=backend` als Default und laeuft damit auf
dem aktuellen Backend-Image (`ai-clipping-backend` oder
`ai-clipping-backend-cuda`). `AI_CLIPPING_LOCAL_TRAIN_IMAGE_MODE=legacy` baut
weiterhin `backend/Dockerfile.local-train`. Der Wrapper nutzt
`scripts/lib_backend_container.sh`, `AI_CLIPPING_REBUILD=auto` und Runtime-
Auswahl ueber `AI_CLIPPING_LOCAL_TRAIN_BACKEND_RUNTIME`/`AI_CLIPPING_BACKEND_RUNTIME`.
Ohne Entry-Point-Override startet Backend-Mode `python -m ml.local_train`; mit
`AI_CLIPPING_LOCAL_TRAIN_ENTRYPOINT=python` funktionieren Spezialcommands wie
`train_proposal_local_docker.sh`. `backend/Dockerfile` enthaelt jetzt auch
`libchromaprint-tools`, damit die alte Training-Dependency fuer Audio-
Fingerprinting im CPU-Backend-Image nicht fehlt. Verifiziert:
`bash -n scripts/train_local_docker.sh`,
`bash -n scripts/train_proposal_local_docker.sh`,
`scripts/train_local_docker.sh --wrapper-help`,
`scripts/train_proposal_local_docker.sh --help`.

## 2026-06-03 Plan: Serving-Holdout im Backend-Container ausfuehren

Ziel: `scripts/run_serving_recall_holdout.sh` soll standardmaessig die gleiche
Runtime wie das containerisierte Backend nutzen, damit Eval-Laeufe nicht von der
lokalen `.venv`/CUDA/ffmpeg-Umgebung abweichen.

- [x] Script-Default von lokalem Python auf `docker exec ai-clipping-backend`
      umstellen.
- [x] Expliziten lokalen Debug-Modus behalten.
- [x] Containername per Option/Env ueberschreibbar machen.
- [x] Shell-Syntax, lokalen Dry-Run und Container-Fehlerpfad verifizieren.

**Review 2026-06-03:** `scripts/run_serving_recall_holdout.sh` nutzt jetzt
standardmaessig `docker exec -w /app -e PYTHONPATH=/app/backend
ai-clipping-backend python -m ml.run_serving_recall_holdout`. Lokaler Host-Run
ist nur noch explizit per `--local` moeglich; der Containername ist per
`--container NAME` oder `AI_CLIPPING_BACKEND_CONTAINER` ueberschreibbar.
Verifikation: `bash -n scripts/run_serving_recall_holdout.sh`,
`scripts/run_serving_recall_holdout.sh --wrapper-help`,
`scripts/run_serving_recall_holdout.sh --local --dry-run --limit 1`.
Ohne laufenden Backend-Container bricht der Default-Aufruf bewusst mit
`Backend container not found: ai-clipping-backend` ab, statt auf die lokale
`.venv` zurueckzufallen. Ein echter Container-Dry-Run ist der naechste Check,
sobald `ai-clipping-backend` laeuft.

**Update 2026-06-03:** Auto-Rebuild/Auto-Start fuer Backend-Container ist jetzt
in `scripts/lib_backend_container.sh` zentralisiert. `start_containerized.sh`
nutzt `AI_CLIPPING_REBUILD=auto` als Default und baut das Backend-Image nur,
wenn Build-Inputs neuer als das Image sind oder das Image fehlt. Im Dev-Modus
zaehlen nur Docker-/Dependency-/Compose-Inputs, weil Code per Bind-Mount
eingespielt wird. `run_serving_recall_holdout.sh` nutzt dieselbe Pruefung,
startet `backend` bei Bedarf und wartet auf den Backend-Container, bevor es
`docker exec` ausfuehrt. Verifiziert: `bash -n` fuer Helper/Runner/Startscript,
Wrapper-Help, lokaler Dry-Run und Custom-Container-Fehlerpfad.

## 2026-06-03 Plan: Recent-Holdout einfrieren

Ziel: Die aktuell definierten Serving-/Review-Holdout-VODs bleiben stabil und
werden bei normalen Trainingslaeufen nicht mehr automatisch durch die neuesten
10 VODs ersetzt.

- [x] `build_recent_vod_holdout.py` so aendern, dass bestehende Policy-Rows
      standardmaessig preserved werden.
- [x] Expliziten Refresh-Schalter ergaenzen, damit ein bewusstes Neuwaehlen der
      neuesten VODs weiterhin moeglich bleibt.
- [x] Ranker-Train-Script so verdrahten, dass normale Runs preserve nutzen und
      Refresh nur per Env aktiviert wird.
- [x] Tests fuer Preserve- und Refresh-Verhalten ergaenzen.
- [x] Verifizieren, dass die aktuelle Holdout-Liste unveraendert bleibt.

**Review 2026-06-03:** Normale Aufrufe von `ml.build_recent_vod_holdout`
preserven jetzt eine vorhandene Policy-Datei. Refresh ist nur noch explizit per
`--refresh` bzw. im Ranker-Train per `RECENT_VOD_HOLDOUT_REFRESH=1` moeglich.
Gezielte Tests: `PYTHONPATH=backend pytest -q backend/tests/test_recent_vod_holdout_policy.py`
und `bash -n backend/ml/train_jynxzi_vod.sh`. Der echte Artefakt-Check meldete
`recent_vod_holdout_policy_preserved` und behielt die 10 aktuellen VODs
unveraendert.

## 2026-06-01 Plan: OpusClips-Niveau Clipfindung (Deep-Dive-Umsetzung)

**Kontext / Befund aus dem Deep Dive:**
Die Architektur ist gut. Der Engpass ist NICHT die Pipeline, sondern (a) **Messbarkeit**
(gesamter 6-Stufen-Funnel wird gegen 3 Positiv-Fenster in 1 VOD getunt → `final_ranked 0/3`),
(b) **Datenmenge/Outcome-Signal**, (c) **Judge-Rubrik** passt nur halb zu Gaming-Reaction-Content.
Architektur-Hygiene (Gate entschärfen, Patch-Wildwuchs konsolidieren) bringt **Tuning-Stabilität,
nicht Recall** — sie ist Vorbereitung für die Tuning-Phase, kein Ersatz dafür.
Reihenfolge der Phasen = Priorität. Phase 0 ist blockierend für alles andere.

**Leitprinzip:** Keine weitere Gate/Judge/Ranker-Schwelle anfassen, bis Phase 0 steht.
Jeder Tweak gegen n=3 verschiebt nur das Versagen (Beleg: Bottleneck-Log).

### Phase 0 — Messbarkeit herstellen (BLOCKIEREND, zuerst)
Ziel: belastbarer Holdout, gegen den überhaupt getunt werden kann.

- [x] Holdout-VOD-Set definieren: 10 neueste VODs (temporaler Split). Disjunktheit empirisch
      verifiziert — 0 Leakage in `jynxzi_vod_training_rows.jsonl` + `jynxzi_proposal_training_rows.jsonl`.
- [x] Aktuellste 10 VODs automatisch als `recent_vod_holdout_policy` exportieren und in
      Ranker- sowie Proposal-Training hart ausschließen.
- [x] Skript ergänzen, das alle Holdout-VODs oracle-frei durch die Serving-Pipeline laufen lässt
      und anschließend den Stage-Recall aggregiert auswertet.
- [x] Live-Progress im Holdout-Runner ergänzen: pro VOD aktuelle Pipeline-Stufe,
      Prozent, Dauer und finales Eval sichtbar machen.
- [x] Reposts pro Holdout-VOD gemappt: 10/10 VODs haben Positives (5–29 je VOD, 103 gesamt)
      in `jynxzi_cross_platform_fingerprint_rows.jsonl`.
- [x] Holdout oracle-frei durch die Pipeline gelaufen + ausgewertet (run 20260603T190707Z,
      ~6,6h, 10/10 done) → `jynxzi_serving_recall_holdout_eval.json`, jobs_evaluated 10.
- [x] Akzeptanz erfüllt: 103 Positiv-Fenster / 10 VODs (Ziel ≥40–60/≥10), n=3-Rauschen beendet.
      ⬜ Latenter Schönheitsfehler offen: Recency-Auswahl filtert nicht gegen Positives-Existenz
      (aktuell unkritisch, 10/10 Treffer).

### Phase 1 — Daten: Match-Rate & saubere Positives konsolidieren
Ziel: den bereits identifizierten Haupt-Hebel (139→521 Positives) verstetigen.

- [x] Prüfen, ob der letzte Retrain die 521 sauberen Fenster wirklich konsumiert hat
      (Train-Timestamp vs. Cleanup-Lauf 2026-05-29 16:49); falls nein, Retrain auslösen.
- [x] Rest-Schönheitsfehler abarbeiten: 83 Reposts noch Multi-VOD (Cross-VOD-Filter greift nur
      bei Chunk-Matches, nicht zwischen chunk vs. single/audio) — Filter auf alle Match-Kinds
      ausweiten (knüpft an `_filter_multi_window_matches_to_top_vod`).
- [x] `export_serving_hard_negatives.py` über das erweiterte VOD-Set neu erzeugen
      (aktuell nur 40 Hard-Negs aus 2 VODs → mit mehr VODs skalieren).
- [~] Akzeptanz: Match-Rate ≥90% gehalten, token_recall-Median ≥0.78, Hard-Negs aus ≥5 VODs.
      Match-/Token-Ziele erfüllt; trainierbare Hard-Negs aus ≥5 VODs blockiert, weil die aktuell
      verfügbaren Serving-Jobs ausschließlich Holdout-VODs sind und aus Leakage-Gründen nicht in
      `*_train_rows` geschrieben werden.

### Phase 2 — Outcome/Views als Ranking-Target (statt nur sample_weight)
Ziel: von „was der Clipper genommen hätte" → „was performt" (OpusClips-Kante).

- [ ] `import_yt_retention.py` / View-Daten-Abdeckung prüfen (Memo: 152/153 Positives haben Views).
- [x] In `train_vod_ranker.py` View/Retention-Tier als **Ziel-Signal** (graded relevance für den
      LightGBM-Ranker), nicht nur als `sample_weight` aus `outcome_view_weighting.py`, evaluieren.
- [ ] `build_outcome_holdout_dataset.py` als Outcome-Eval gegen das Phase-0-Holdout fahren.
- [ ] Akzeptanz: Ranking-Metrik (NDCG@5 / MRR) gegen Outcome-Labels verbessert sich ggü.
      Clipper-Binary-Labels, ohne Recall auf Phase-0-Holdout zu verschlechtern.

### Phase 3 — Judge-Rubrik few-shot gegen echte Jynxzi-Clips kalibrieren
Ziel: Judge bewertet Gaming-Reaction-Content korrekt (Memo: aktuell 4.5–5.8 / clarity 0.4–0.5).

- [ ] 5–10 reale Clipper-Picks (verschiedene Typen: Clutch, Rage, Funny, Chat-Igniter) als
      Few-Shot-Anker in den Judge-Prompt (`services/clip_judge.py`, `_AXIS_NAMES`/Prompt-Block).
- [ ] Prüfen, ob Achsen-Gewichte (clarity 0.18 / payoff 0.28 / narrative 0.12) zu Reaction-Content
      passen; ggf. narrative_completeness abwerten, reaction/payoff aufwerten.
- [ ] Determinismus: Multi-Sampling (n>1, Median) statt einzelnem temp=0-Call gegen Run-zu-Run-
      Varianz testen (Memo: derselbe Moment narr 0.6 vs 0.4).
- [ ] Akzeptanz: gejudgte echte Positives erreichen Median judge_score ≥6.0 und überstehen das
      Gate reproduzierbar über 3 Wiederholungsläufe.

### Phase 4 — Chat-Velocity als Primärsignal
Ziel: das stärkste billige Streamer-Clip-Signal aus dem vergrabenen Ranker-Feature herausheben.

- [x] Chat-Spike-Rate (`chat_analyzer.py`) als eigenen Candidate-Generator/Prior verdrahten,
      analog zu `peak_detector` (Chat-EMA-Peaks → Kandidatenfenster).
- [ ] Gewichtung im Ranker/Proposal-Prior erhöhen; gegen Phase-0-Holdout messen.
- [ ] Akzeptanz: Stage-Recall „peak/semantic→merged" steigt für chat-getriebene Positives.

### Architektur-Hygiene (parallel möglich, KEIN Recall-Hebel — Stabilität/Wartbarkeit)
- [x] Serving-Quality-Gate von harten Schwellen auf Soft-Blend umstellen
      (`serving_quality_min_*` in `config.py` / `selector.py`) — harte Cuts auf nicht-
      deterministischen Judge-Subscores sind per Konstruktion instabil.
- [ ] Proposal-Scorer als reinen High-Recall-Pre-Filter retunen (Durchsatz-Job, nicht Präzision;
      Präzision macht der Ranker). `proposal_min_score`/`proposal_shortlist_top_k` entsprechend.
- [ ] Die 5 Preserve/Reserve/Bypass-Pfade konsolidieren (`_promote_cross_platform_recall_candidates`,
      `_cross_platform_recall_preserve_ids`, `_add_context_variant_candidates`,
      `_build_judge_recall_reserved_order`, `proposal_preserve_heuristic_candidates`) → ein
      einziger „known-good protection"-Pfad.
- [ ] Akzeptanz: gleiche oder bessere Phase-0-Holdout-Recall bei deutlich weniger Code-Pfaden;
      bestehende Tests grün.

### Querschnitt / Done-Kriterien (gilt für jede Phase)
- [ ] Jede Änderung gegen das **Phase-0-Holdout** messen, nicht gegen n=3.
- [ ] Tests grün (`PYTHONPATH=backend pytest backend/tests -q`), `git diff --check`.
- [ ] Nach Backend-Änderungen Graphify-Rebuild laufen lassen.
- [ ] Nach jeder Phase: Review-Notes hier ergänzen (Baseline → Ergebnis, Zahlen).

### Review Notes
- 2026-06-03 **PHASE 0 ABGESCHLOSSEN** (run 20260603T190707Z, ~6,6h, 10/10 VODs done,
  oracle-free, 0 Leakage verifiziert). Holdout: 10 disjunkte neueste VODs / 103 Positiv-Fenster
  → Messbarkeit hergestellt, n=3-Rauschen beendet. **Offizielle Baseline (Aggregat-Recall/Stage):**
  semantic 81% · dense 74% · **merged 41%** · proposal_kept 33% · judge_scope 17% · clip_judge 41% ·
  quality_kept 41% · **final_ranked 3,9% (4/103)** · final_moments 2,9%. Zwei Bruchstellen:
  (1) Merge/Normalize 81%→41%, (2) finale Selektion 41%→3,9%. Kandidaten-Gen findet ~80% — der
  Funnel blutet sie aus. `final_ranked 3,9%` = Start-Baseline für Phase 1–4, kein Erfolg.
  Artefakt: `jynxzi_serving_recall_holdout_eval.json`.
- 2026-06-04 **PHASE 1 TEILWEISE ABGESCHLOSSEN:** Cross-Platform-Match-Artefakt hält die
  Datenziele: `516` Rows aus `368` Reposts, `349` gematchte Reposts (~94,8%), Token-Recall
  Median `0.837719`, Mean `0.817630`; Cleanup hat bereits `100` Cross-VOD-Treffer gefiltert
  und `549` Near-Duplicate-Merges gemacht. Ranker-Artifact ist nach dem Match-Artefakt gebaut
  und hat `201` Cross-Platform-Positive aus `cross_platform_outcome_train` +
  `cross_platform_transcript_match_train` konsumiert. Zusaetzlich wurde der dominante-VOD-Cleanup
  gehärtet: starke `single_window`/`audio_fingerprint_fallback`-Matches schlagen jetzt mehrere
  schwache falsche Chunk-Treffer. Serving-Hard-Negatives wurden neu exportiert, aber sicher auf
  `0` trainierbare Rows gesetzt: alle `25` aktuell vorhandenen Serving-Job-Debugs liegen auf
  Holdout-VODs (`holdout_vod_id`) und duerfen nicht als Training-Negatives leaken. Naechster
  Daten-Schritt: organische Serving-Runs auf Nicht-Holdout-VODs sammeln, dann Hard-Negs aus
  ≥5 trainierbaren VODs exportieren.
- 2026-06-01: View-Tier ist jetzt ein echtes Ranking-Ziel: `rank_target` wird aus
  Cross-Platform/View-Signalen in Trainingszeilen übernommen, im Ranker vor
  `ordinal_label` verwendet und in Eval-/Ranker-Berichten als
  `by_outcome_view_tier` / `by_rank_target` ausgewiesen.
- 2026-06-01: `chat_velocity` ist als eigener Candidate-Generator im
  semantischen Candidate-Pool verdrahtet, bekommt Reaction-Bucket/Prior und
  kann dadurch vor Proposal/Ranker als primäres Streamer-Signal auftauchen.
- 2026-06-01: Oracle-Cross-Platform-Feature-Boosts sind im normalen Serving
  deaktiviert (`serving_cross_platform_feature_boosts_enabled=False`). Bekannte
  Positives bleiben damit Trainings-/Eval-Signal, aber pushen Produkt-Runs
  nicht mehr automatisch hoch.
- 2026-06-01: Ein zentraler label-freier
  `high_recall_protection.oracle_free_recall_score` speist Judge-Reserve und
  Quality-Gate-Audit. Cross-Platform-Targets sind darin bewusst ausgeschlossen.
  Das ist die erste Konsolidierungsstufe; die alten Preserve-/Reserve-Pfade
  sind noch nicht vollständig entfernt.
- 2026-06-01: Judge-Prompt hat Jynxzi-spezifische Kalibrierungsanker
  (Clutch/Rage/Fanmail-Unboxing/Chat-Igniter vs. schwache generische Takes),
  ohne konkrete bekannte Positives als Runtime-Leak einzubauen. Der explizite
  5–10-Beispiel-Few-Shot-Block bleibt offen, bis ein sauberes Holdout-/Anchor-
  Set festgelegt ist.
- 2026-06-01: Verifikation bisher: fokussierte Tests `93 passed`, Pipeline-
  Oracle-/Reserve-Tests `2 passed`, View-Target-Row-Regressionscheck
  `1 passed`, `py_compile` und scoped `git diff --check` grün.
  Graphify-Rebuild: `3827` Nodes, `10434` Edges, `106` Communities.
  Phase-0-Holdout über 10–20 VODs ist weiterhin offen und bleibt die
  blockierende Mess-Aufgabe vor weiteren Schwellentweaks.
- 2026-06-01: Die aktuellsten 10 VODs werden jetzt aus
  `jynxzi_twitch_vod_inventory.jsonl` als
  `jynxzi_recent_vod_holdout_policy.jsonl` exportiert und mit dem bestehenden
  Outcome-Holdout zu `jynxzi_training_holdout_policy_rows.jsonl` gemerged.
  Der aktuelle Snapshot enthält die VODs `2781590654`, `2780984594`,
  `2779787387`, `2779061928`, `2778362653`, `2777621798`, `2776799704`,
  `2776013057`, `2775214183`, `2774602304`; 66 Cross-Platform-Rows auf diesen
  VODs werden dadurch vom Ranker-/Proposal-Training ausgeschlossen.
- 2026-06-01: Ranker-Training nutzt die gemergte Holdout-Policy statt nur des
  Outcome-Holdouts. Proposal-Training filtert dieselben Holdout-VODs vor
  Materialisierung, Row-Building und Debug-Row-Collection. Verification:
  `py_compile`, `bash -n scripts/train_proposal_local_docker.sh`, fokussierte
  Tests `12 passed`, scoped `git diff --check`, Graphify-Rebuild
  (`3853` Nodes, `10511` Edges, `104` Communities). Die Holdout-Policy-Builder
  schreiben Dateien nur bei Inhaltsänderung, damit unveränderte Runs nicht
  allein durch mtime neu trainieren. `zsh -n` konnte lokal nicht laufen, weil
  `zsh` in der Umgebung nicht installiert ist.
- 2026-06-01: `ml.run_serving_recall_holdout` und
  `scripts/run_serving_recall_holdout.sh` orchestrieren jetzt den Phase-0-Test:
  Holdout-VODs aus der Policy lesen, pro VOD `pipeline_worker.py` oracle-frei
  starten (`SERVING_CROSS_PLATFORM_RECALL_ENABLED=false`,
  `SERVING_CROSS_PLATFORM_FEATURE_BOOSTS_ENABLED=false`) und danach
  `evaluate_serving_recall_holdout.evaluate_holdout` für aggregierten
  Stage-Recall schreiben. Verification: `py_compile`, `bash -n`, neue Tests
  `3 passed`, Dry-Run über 2 Holdout-VODs, Graphify-Rebuild (`3869` Nodes,
  `10556` Edges, `103` Communities).
- 2026-06-01: Holdout-Runner gibt jetzt Live-Progress als strukturierte
  stdout-Events aus: `holdout_run_started`, `holdout_vod_started`,
  `holdout_vod_progress` mit `stage_label`/`progress_pct`/elapsed time,
  `holdout_vod_done`, `holdout_eval_started`, `holdout_eval_done` und
  `holdout_run_done`. Verification: `py_compile`, `bash -n`, neue Tests
  `4 passed`, Dry-Run zeigt die Progress-Events, Graphify-Rebuild (`3873`
  Nodes, `10567` Edges, `102` Communities).


## 2026-05-31 Plan: Cross-VOD Match Cleanup Across All Match Kinds

- [x] Inspect the existing transcript-match cleanup path and identify why weak second-VOD labels survive.
- [x] Extend per-repost dominant-VOD filtering across all accepted match kinds, not only chunk matches.
- [x] Preserve strong secondary-VOD matches so true multi-stream recaps are not removed.
- [x] Add regression tests for single-window plus chunk cross-VOD false positives.
- [x] Run focused tests, syntax checks, diff checks, rebuild Graphify, and document the result.

### Review Notes

- `_filter_multi_window_matches_to_top_vod` now computes the dominant VOD from
  every accepted match kind for the repost, including `single_window`,
  `multi_window_chunk`, merged windows, and audio fallback rows.
- Weak non-dominant VOD matches are removed when their token recall is below
  `0.55`. Strong secondary matches stay intact, preserving genuine multi-VOD
  recap evidence.
- Added regression coverage for the exact failure mode: a strong single-window
  match on the true VOD plus a weak chunk match on a second VOD now emits only
  the true VOD row.
- Verification passed: focused transcript matcher tests, Python compile checks,
  scoped `git diff --check`, and Graphify rebuild (`3808` nodes, `10375`
  edges, `105` communities).


## 2026-05-30 Plan: Fix Local Training Cache Permissions

- [x] Repair existing root-owned training cache, VOD feature cache, and artifact files.
- [x] Run local train containers as the host user by default so new cache files are writable.
- [x] Verify no root-owned files remain in the affected cache roots.
- [x] Smoke-test `train_vod_ranker.py` far enough to prove the previous `PermissionError` is gone.
- [x] Rebuild Graphify after script/code changes and document the result.


### Review Notes

- Existing root-owned files under `backend/tmp/training_runs/jynxzi_resume`,
  `backend/tmp/vod_feature_cache`, and `backend/ml/artifacts` were reassigned
  to the host user via the local training Docker image. A follow-up ownership
  scan returned no remaining non-user-owned files in those roots.
- `scripts/train_local_docker.sh` now runs the training container as the host
  UID/GID by default using `--user`, with `AI_CLIPPING_LOCAL_TRAIN_USER` left
  as an override. `HOME` and `MPLCONFIGDIR` point at writable paths under
  `/app/backend/tmp`.
- `bash -n scripts/train_local_docker.sh` passed and `git diff --check` passed
  for the touched files.
- A 140s isolated `train_vod_ranker.py` smoke run reached later VOD phases
  without `PermissionError` or `Permission denied`. It timed out intentionally
  while continuing work, so the previous checkpoint write failure is gone.
- Graphify rebuilt successfully: `3803` nodes, `10361` edges, `108`
  communities.


## 2026-05-29 Plan: Faster Ranker Label Phases

- [x] Raise the default local ranker training VOD phase parallelism.
- [x] Make extended window-scoped groups use `TRAINING_LABEL_WORKERS` for labels inside a group.
- [x] Preserve deterministic row ordering and existing unavailable-VOD skip behavior.
- [x] Update focused tests for runtime defaults and scoped-group label parallelism.
- [x] Run verification, rebuild Graphify, and document the result.

### Review Notes

- `backend/ml/local_train.py` now defaults to `TRAINING_VOD_WORKERS=2` on
  machines with at least `12` CPUs, while keeping `1` VOD worker on smaller
  machines. Explicit `--training-vod-workers` still overrides the default.
- Extended window-scoped VOD label phases now log and use the resolved
  `TRAINING_LABEL_WORKERS` value. Labels inside the same scoped group are
  processed in parallel when a group has multiple labels, then emitted in the
  original label order for deterministic training rows.
- Existing unavailable-VOD behavior stays scoped to source/context loading:
  unavailable VOD errors still skip the remaining groups for that VOD.
- Verification passed for changed training code: `py_compile`, focused
  `test_local_train.py` and `test_build_twitch_training_dataset.py`
  (`85 passed`), scoped `git diff --check`, and Graphify rebuild
  (`3803` nodes, `10361` edges, `108` communities).
- A broader `test_train_vod_ranker.py` + `test_pipeline_integration.py` run
  exposed an unrelated existing failure in pipeline tests:
  `services.clipper.cut_review_clips` missing. The ranker-training focused
  tests passed.


## 2026-05-29 Plan: Multi-Window Positive Cleanup

- [x] Add per-repost cleanup for accepted transcript matches before rows are emitted.
- [x] Restrict multi-window chunk matches from one repost to the top-confidence VOD.
- [x] Merge accepted windows less than `30s` apart inside the same repost/VOD.
- [x] Add focused tests for cross-VOD chunk filtering and near-duplicate merging.
- [x] Run focused verification, rebuild Graphify, and document the outcome.

### Review Notes

- Transcript cross-platform matching now cleans accepted rows per repost before
  writing training positives.
- Multi-window chunk matches from one repost are restricted to the dominant VOD
  by aggregate chunk confidence, preventing one Short from creating positives
  across unrelated VODs.
- Accepted windows less than `30s` apart on the same VOD are merged into one
  `multi_window_merged` row, preserving the strongest confidence fields while
  removing duplicate label weight.
- Diagnostics now report `accepted_rows_before_cleanup`,
  `multi_window_cross_vod_filtered`, and
  `multi_window_near_duplicate_merges`.
- Verification passed: `py_compile`, `56` focused transcript/cross-platform
  pytest cases, scoped `git diff --check`, and Graphify rebuild via the
  uv-managed Graphify interpreter (`3802` nodes, `10358` edges,
  `107` communities).
- `scripts/train_local_docker.sh` rebuilds the local-train Docker image at
  startup, so the cleanup is picked up by the next ranker training run. Any
  already-running training container still uses the rows/code it started with.

## 2026-05-28 Plan: Outcome Views And Serving Hard Negatives

- [x] Inspect current Cross-Platform positive rows, sample weights, and hard-negative export paths.
- [x] Centralize repost view-count tiers and sample weights for Cross-Platform positives.
- [x] Export hard negatives from high-ranked Serving candidates that miss known positive windows.
- [x] Wire the new hard-negative artifact into the local ranker training flow.
- [x] Add focused tests, run artifact smoke checks, rebuild Graphify, and document results.

### Review Notes

- Added view-count tiering in `backend/ml/outcome_view_weighting.py`.
  Cross-Platform positives now carry `outcome_view_tier`,
  `outcome_view_sample_weight`, `outcome_view_weight_version`, and stronger
  rank targets for high-view reposts.
- Transcript and audio Cross-Platform matchers now annotate matched rows with
  view-tier weights at creation time, and `train_vod_ranker.py` re-annotates
  older Cross-Platform rows during training preparation.
- Added `backend/ml/export_serving_hard_negatives.py`. It mines recent serving
  debug jobs for high-ranked organic candidates that do not overlap known
  Cross-Platform positives, skips selected or strongly judged candidates, and
  writes `jynxzi_serving_hard_negative_train_rows.jsonl`.
- Wired serving hard negatives into `train_jynxzi_vod.sh`,
  `train_vod_ranker.py`, and the workflow-state/dependency checks. This path is
  independent of manual user-label/gold hard negatives.
- Smoke export on current storage produced `40` serving hard-negative rows from
  `2` VODs, with source mix `dense_inventory=22`, `peak=12`,
  `narrative_reveal=3`, `quote=1`, `tension_build=2`.
- Verification passed: Python compile checks, `52` focused pytest cases,
  scoped `git diff --check`, and Graphify rebuild
  (`3743` nodes, `10143` edges, `109` communities).

## 2026-05-28 Plan: Serving-Recall Holdout Measurement

- [x] Persist the new OpusClips-gap plan and define the first measurable step.
- [x] Add an oracle-free serving-recall holdout evaluator for existing debug jobs.
- [x] Aggregate known cross-platform positives by stage without serving-time injection.
- [x] Run focused tests and a local evaluation over current storage jobs.
- [x] Rebuild Graphify after backend code edits and document results.

### Review Notes

- Added `backend/ml/evaluate_serving_recall_holdout.py`, a strict holdout
  evaluator for existing serving debug jobs.
- The evaluator only includes jobs whose debug summary says positives were
  loaded for diagnostics while serving recall was disabled, and whose
  `candidate_trace.jsonl` has no `cross_platform_recall_target` or
  `proposal_recall_target` markers.
- It dedupes by default to the newest valid run per VOD, so repeated test runs
  of the same VOD do not inflate the aggregate.
- Current command:
  `backend/.venv/bin/python backend/ml/evaluate_serving_recall_holdout.py --profile jynxzi --storage-dir backend/tmp/storage --positive-rows backend/ml/artifacts/jynxzi_cross_platform_fingerprint_rows.jsonl --output backend/ml/artifacts/jynxzi_serving_recall_holdout_eval.json`
- Current local result used one valid oracle-free VOD run:
  job `61acfc25-bc9c-496e-97cd-019507f715a4`, VOD `2768134699`,
  `3` positive windows.
- Serving-recall result: semantic `3/3`, dense `3/3`, proposal input `3/3`,
  proposal kept `1/3`, ranked pre-Judge `2/3`, pre-Judge top20 `0/3`,
  pre-Judge top40 `0/3`, pre-Judge top80 `1/3`, judge scope `1/3`,
  final moments `0/3`.
- Verification: `14` focused backend tests passed, `git diff --check` passed,
  and Graphify rebuilt to `3711` nodes, `10058` edges, `105` communities.

## 2026-05-28 Plan: Frontend Candidate Labeling Loop

- [ ] Inspect existing feedback models, storage, API routes, and frontend clip preview flow.
- [ ] Expose reviewable pipeline candidates with preview URLs and enough context for labeling.
- [ ] Add frontend labeling controls for excellent/good/maybe/bad/boundary_bad plus reason tags.
- [ ] Persist labels through the existing feedback/training export path.
- [ ] Add focused backend/frontend tests or type checks, rebuild Graphify, and verify the app.

## 2026-05-28 Plan: Analyze Latest Job After Variant Preservation

- [x] Identify newest completed debug-backed job and VOD.
- [x] Inspect funnel, Proposal, Ranker, Judge, Quality Gate, and final clips.
- [x] Compare known cross-platform positives against each pipeline stage.
- [x] Summarize whether the new context-variant preservation improved recall.
- [x] Document remaining bottleneck and next action.

### Review Notes

- Latest completed debug-backed job is
  `30916b50-8466-48c2-a9b9-cb0d0af03d7f` for
  `https://www.twitch.tv/videos/2768134699`.
- Funnel: `5000` dense inventory, `533` Proposal inputs, `240`
  Proposal-kept, `275` ranked pre-Judge, `80` judged, `13` judged
  quality-kept, `12` final moments.
- The new context-variant preservation is active and added `96` candidates.
  Known-positive recall improved at Proposal input from the previous `2/3`
  to `3/3`.
- Known-positive recall still fails after that: Proposal-kept `1/3`,
  ranked-pre-Judge `2/3`, Judge scope `1/3`, final `0/3`.
- The World-Cup positive now has a full-context `narrative_reveal`
  Proposal-input candidate, but it is still rejected by Proposal
  (`score=0.383283`). The short dense variant is ranked only `272`.
- The Wholesome Fanmail positive now has a `quote` Proposal-input candidate,
  but it is rejected by Proposal (`score=0.404498`).
- The Darth/Funko positive reaches Proposal-kept and Judge scope. The judged
  variant scores only `5.8` and fails `judge_score_below_min` plus
  `standalone_clarity_below_min`; the better full-overlap variant remains
  unjudged.
- Main bottleneck has moved from missing Proposal input to Proposal scoring
  and Judge-scope variant selection. Semantic/quote context candidates need a
  dedicated reserve path or ranking boost before the Judge.

## 2026-05-28 Plan: Check Clip Context Understanding Status

- [x] Read project lessons and Graphify report for current architecture context.
- [x] Inspect pipeline stages for ASR, chat, visual features, VLM/Judge, and context memory.
- [x] Verify latest debug artifacts for actual Judge behavior.
- [x] Summarize implemented versus missing clip-understanding capability.

### Review Notes

- The serving pipeline already combines transcript, audio, visual, chat,
  semantic/context, Proposal/Ranker, and LLM Judge signals.
- The latest debug-backed job `89bc39b1-b472-46c2-bf45-0e0b01c6528d` judged
  `80/80` candidates with Gemini multimodal and wrote context summaries plus
  Judge reasons into `final_moments.json`.
- The current implementation is still primarily a ranking/judging system. It
  does not yet emit a dedicated structured clip-understanding artifact with
  timestamped event timeline, detected visible actions, transcript-grounded
  cause/effect, uncertainty, and evidence spans for every final clip.

## 2026-05-28 Plan: Preserve Best Context Variants Through Judge

- [x] Add context-variant preservation during candidate merge/normalization.
- [x] Add Proposal input context-variant enrichment so semantic setup windows are not replaced by short dense windows.
- [x] Reweight Judge reserve toward context completeness, setup, payoff, and dynamic pattern fit.
- [x] Add focused tests for overlap-region variant preservation and judge reserve scoring.
- [x] Run focused backend tests, syntax checks, Graphify rebuild, and restart backend.

### Review Notes

- `merge_prioritized_candidates` can now preserve stronger setup/context
  variants that overlap an existing dense or peak candidate, instead of always
  dropping them as duplicates.
- Proposal input construction now adds label-free context variants from
  semantic, dense, and dense-inventory candidates before Proposal scoring.
  Offline replay on job `89bc39b1-b472-46c2-bf45-0e0b01c6528d` showed the
  previously lost World-Cup narrative setup and Wholesome quote variants being
  added to Proposal input.
- Judge recall reserve now gives more weight to context completeness, setup,
  payoff, and dynamic context-pattern fit, so context-complete variants are
  more likely to reach the Judge shortlist.
- Verification: `py_compile` passed locally and inside the rebuilt backend
  container; focused backend tests passed (`80 passed`); `git diff --check`
  passed; Graphify rebuilt (`3666` nodes, `9936` edges, `103` communities);
  backend container rebuilt/restarted and is healthy.

## 2026-05-28 Plan: Analyze Latest Job After Context-Primary Changes

- [x] Identify newest completed debug-backed job.
- [x] Inspect funnel, judge stage, quality gate, and final moments.
- [x] Compare known cross-platform positives against each pipeline stage.
- [x] Check whether the new context-primary features affected Proposal/Ranker behavior.
- [x] Summarize the remaining bottleneck and next concrete step.

### Review Notes

- Latest completed debug-backed job is
  `89bc39b1-b472-46c2-bf45-0e0b01c6528d` for
  `https://www.twitch.tv/videos/2768134699`.
- Funnel: `5000` dense inventory, `414` Proposal inputs, `240`
  Proposal-kept, `284` ranked pre-Judge, `80` judged, `10` judged
  quality-kept, `8` final moments. Gemini judged all `80` candidates in
  `189.886s`.
- Context-primary serving logic is active: final selected clips have high
  context values (`context_completeness` median about `0.774`) and final
  rationales include `context_primary`, `context_complete`,
  `context_strong`, and context-pattern labels.
- This VOD has `3` known cross-platform positives. Recall: semantic `3/3`,
  dense `3/3`, merged `2/3`, proposal input `2/3`, proposal kept `1/3`,
  judge scope `1/3`, judged quality-kept `0/3`, final moments `0/3`.
- World Cup positive reached Proposal input with perfect overlap, but only as
  a short `dense_window`; Proposal rejected it (`score=0.300935`) and Ranker
  buried it at pre-Judge rank `282`. Its semantic full-context candidate
  existed but was replaced before merge/proposal.
- Darth/Funko positive improved materially: it reached Proposal-kept and
  Judge scope, with context pattern `darth_his_shoes`, but the judged variant
  scored only `5.8` and failed `judge_score_below_min` plus
  `standalone_clarity_below_min`. A better longer overlapping variant was
  unjudged but quality-accepted.
- Wholesome fanmail positive existed in semantic/dense candidates but had
  `0` merged/normalized/proposal matches, so it still dies in merge/proposal
  input construction.
- Main remaining bottleneck is not raw candidate generation. It is preserving
  the best context-rich semantic/dense variants into merge/proposal and
  sending the best setup-containing variant to the Judge.

## 2026-05-28 Plan: Make Context Primary In Proposal/Ranker

- [x] Add explicit context-understanding features shared by Proposal and Ranker.
- [x] Reweight Proposal heuristic/serving floor so context completeness drives candidate survival.
- [x] Reweight Ranker heuristic blend so context quality can promote candidates before Judge.
- [x] Add tests proving context-rich low-hype candidates outrank context-poor high-hype candidates.
- [x] Run focused tests, syntax checks, rebuild Graphify after backend edits, and restart backend.

### Review Notes

- Added shared context scoring in `backend/services/context_quality.py` and
  exposed `setup_present_score`, `payoff_present_score`,
  `context_strength_score`, and `context_completeness_score` on both Proposal
  and Ranker feature paths.
- Proposal scoring now treats context as the primary survival signal:
  context-primary/completeness/strength/setup/payoff dominate the heuristic,
  and contextual dense candidates get a stronger serving floor before the
  trained proposal artifact is retrained.
- Ranker serving now has a context-primary rerank stage before the existing
  boundary/hook reranker. Context-rich candidates receive explicit
  `context_primary_boost` / `context_primary_signal`; high-hype candidates with
  missing context receive `context_missing_penalty`.
- Proposal training feature defaults now include the new context fields, so the
  next proposal retrain can learn them directly instead of relying only on the
  serving heuristic.
- Focused verification passed: changed-file `py_compile`, `88` backend tests
  across proposal/ranker/training/feature/pipeline/recall modules, and
  `git diff --check`.
- Offline replay on the latest debug job shows the previously buried
  Darth/Funko context-positive candidate now gets strong context signals
  (`context_completeness=0.848833`, `context_strength=0.745556`) and a ranker
  score of `17.278` with `context_primary` rationales.
- Graphify was rebuilt with the local Graphify interpreter:
  `3648` nodes, `9875` edges, `102` communities.
- Backend image was rebuilt and restarted; `docker compose ps backend` reports
  `healthy`, `/api/status` responds, and the running container exposes the new
  context feature fields.

## 2026-05-28 Plan: Analyze Latest Serving Job After Activation

- [x] Identify the newest completed debug-backed serving job.
- [x] Inspect funnel counts, Proposal, Ranker, Judge, quality gate, and final moments.
- [x] Check embedded positive-window recall now that the new diagnostics are active.
- [x] Summarize the dominant bottleneck and next concrete step.

### Review Notes

- Latest completed debug-backed job is
  `546163cf-d290-44dc-9a34-a169913f420b` for
  `https://www.twitch.tv/videos/2768134699`.
- The latest implementation is active in this run: `positive_window_recall` is
  embedded in `candidate_recall_report.json`, and Judge reserve used `16` slots
  from pool `240`.
- Funnel: `5000` dense inventory, `414` Proposal inputs, `240` Proposal-kept,
  `281` ranked pre-Judge, `80` judged, `8` judged quality-kept, `7` final
  clips. Gemini judged all `80` candidates in `115.667s`.
- Post-Judge fallback is now correctly disabled: `selected_unjudged_candidates=0`
  and `fallback_unjudged_candidates=0`. The system emitted `7` clips instead of
  filling weak unjudged slots.
- VOD has `3` known cross-platform positives. Recall: semantic `3/3`, dense
  `3/3`, proposal input `2/3`, proposal kept `1/3`, real Judge scope `1/3`,
  final moments `0/3`.
- Per-positive bottlenecks: World-Cup/suite reached Judge with perfect overlap
  but failed the quality gate on `standalone_clarity_below_min`; Darth
  Maul/Funko survived Proposal but was buried at pre-Judge rank `190` and stayed
  outside Judge scope; wholesome fanmail appeared in semantic/dense candidates
  but disappeared before merge/Proposal.
- Ranker remains the dominant bottleneck: high-Judge candidates had mean
  pre-Judge rank `105.6`, and none were in top 40 before Judge. Final selected
  clips had weak Proposal scores too (`mean=0.218885`), so Proposal is also not
  confidently identifying winners.

## 2026-05-28 Plan: Analyze Latest Serving Job

- [x] Identify newest debug-backed serving job.
- [x] Inspect funnel, Proposal, Judge, quality gate, and final selection.
- [x] Compare known cross-platform positives for the VOD against pipeline stages.
- [x] Summarize current bottleneck and next step.

### Review Notes

- Latest debug-backed job is `ec5492c7-732d-4f57-bfc9-d60b82606cef` for
  `https://www.twitch.tv/videos/2768134699`, profile `jynxzi`.
- This job was generated before the newest diagnostic/default changes were
  active in the running backend: `candidate_recall_report.json` has no embedded
  `positive_window_recall`, and Judge reserve still shows `10` slots from pool
  `160` instead of the new defaults `16/240`.
- Funnel: `5000` dense inventory, `416` Proposal inputs, `240` Proposal-kept,
  `284` ranked pre-Judge, `80` judged, `7` judged quality-kept, `6` final
  moments. Gemini judged `80/80` candidates in about `102.8s`.
- Final selection correctly avoided unjudged fallback clips:
  `selected_unjudged_candidates=0`, `fallback_unjudged_candidates=0`,
  `missing_final_slots=5`.
- VOD has `3` known cross-platform positives. Offline recall: Proposal input
  `2/3`, Proposal-kept `1/3`, ranked pre-Judge `2/3`, Judge scope `1/3`,
  final `0/3`.
- Per-positive bottlenecks: World-Cup/suite had a perfect semantic candidate
  but the merged/normalized surviving quote only overlapped `0.347`, was
  Proposal-rejected and Judge-rejected. Darth Maul/Funko survived Proposal with
  perfect overlap but was buried at pre-Judge rank `191`. Wholesome fanmail had
  a semantic overlap but disappeared before merge/Proposal.
- Main current issue: Proposal/merge still lose context variants, and the
  Ranker still buries a known good unboxing positive far outside Judge scope.

## 2026-05-28 Plan: Improve Candidate Recall Before Judge

- [x] Add per-positive recall diagnostics to each serving job.
- [x] Preserve more candidate-source variants through merge/normalization so short known-good windows are not replaced by poor neighbors.
- [x] Improve Proposal/Judge scope recall without serving-time cross-platform boosts.
- [x] Add tests for positive-window diagnostics and source-diverse merge behavior.
- [x] Run focused backend verification.
- [x] Rebuild Graphify after backend edits.
- [x] Document result and remaining bottlenecks.

### Review Notes

- Serving now loads cross-platform positives in diagnostic-only mode when normal
  serving recall boosts are disabled. These rows feed only
  `candidate_recall_report.json` / `candidate_funnel_summary.json`; they do
  not promote, boost, or preserve candidates during inference.
- `positive_window_recall` now reports per known positive window which stages
  matched it, the best candidate per stage, the last matched stage, and the
  first missed stage. This should make future "where did the good clip die?"
  checks much faster.
- Candidate merge keeps primary heuristic candidates strict, but no longer lets
  one supplemental source such as `quote` wipe out a different source variant
  such as `narrative_reveal`. Normalization also preserves source-diverse
  recall variants for semantic/dense sources while keeping generic
  peak/reaction dedupe strict.
- Judge recall reserve defaults now inspect a deeper pool and reserve more
  diverse slots (`16` from top `240`) without using cross-platform positives as
  serving-time labels.
- Focused verification passed:
  `backend/.venv/bin/python -m pytest backend/tests/test_candidate_generators.py backend/tests/test_candidate_normalizer.py backend/tests/test_candidate_recall_report.py backend/tests/test_pipeline_integration.py -q`
  (`43 passed`), changed-file `py_compile`, and targeted `git diff --check`.
- Graphify was rebuilt via the local Graphify interpreter after system
  `python3` did not have the module installed: `3638` nodes, `9838` edges,
  `103` communities.
- Remaining bottleneck to measure with the next job: whether the known positives
  now survive merge/normalization/proposal better, and whether the larger
  generic Judge reserve moves more low-rank but strong candidates into the
  actual Judge scope without label leakage.

## 2026-05-28 Plan: Remove Final Unjudged Fallback Clips

- [x] Change post-Judge final selection to use only judged candidates.
- [x] Update diagnostics/tests so unjudged fallback counts stay at zero.
- [x] Run focused backend verification.
- [x] Rebuild Graphify after backend edits.
- [x] Document result and caveats.

### Review Notes

- `selector.select_post_judge_final_candidates()` now only selects candidates
  with `judge_result`; it returns fewer than `top_k` instead of filling final
  slots with unjudged candidates.
- Pipeline behavior is split intentionally: if the Judge actually evaluated
  candidates, final selection is judge-only; if the Judge is disabled or
  produced zero judged results, the existing pre-Judge diverse selection path
  remains available for no-Judge/fallback operation.
- Diagnostics now report `unjudged_fallback_enabled=false`,
  `fallback_unjudged_candidates=0`, and `missing_final_slots` when judged clips
  are fewer than the requested final count.
- Focused verification passed:
  `backend/.venv/bin/python -m pytest backend/tests/test_selector.py backend/tests/test_candidate_recall_report.py backend/tests/test_pipeline_integration.py`
  (`32 passed`), `py_compile`, and targeted `git diff --check`.
- Graphify was rebuilt after backend edits:
  `3615` nodes, `9767` edges, `100` communities.

## 2026-05-27 Plan: Analyze Latest Job

- [x] Identify the newest job and its request/status.
- [x] Inspect available debug artifacts and stage summaries.
- [x] Check logs for failures or fallback behavior.
- [x] Summarize bottleneck and next action.

### Review Notes

- Latest job is `341a9e9b-a1f3-49ce-84b7-767697404016` for
  `https://www.twitch.tv/videos/2768134699`, profile `jynxzi`.
- The job completed successfully after the SIGSEGV fix: status `done`,
  progress `100`, `11` clip files generated and served through `/static/...`.
- Funnel: `5000` dense inventory, `416` proposal input, `240` proposal kept,
  `284` ranked pre-Judge, `80` judged, `7` judged quality-kept, `11` final
  moments.
- Gemini Judge ran successfully: `80` attempted/judged candidates in about
  `162s`, no stage timeout and no fallback provider.
- Final quality is still mixed: only `6/11` final clips had a real Judge score;
  `5/11` were unjudged fallback candidates from ranks `71-77`.
- Cross-platform recall for VOD `2768134699`: `3` known positive windows.
  Recall reached `2/3` at proposal input/ranked pre-Judge, `1/3` at Judge
  scope, and `0/3` final.
  The known fanmail joke was kept but buried at pre-Judge rank `191`; the
  unboxing/box positive died before Proposal input; the World Cup suite positive
  reached Judge but was rejected with `judge_score=4.48`.
- Current bottleneck is not a crash anymore. It is final selection still filling
  with unjudged candidates plus weak promotion/evaluation of known viral windows.

## 2026-05-27 Plan: Fix Pipeline Signal 11 Crash

- [x] Identify the exact stage and native library involved in the latest
  signal-11 serving crashes.
- [x] Patch the pipeline to avoid the crashing native path instead of only
  masking the subprocess exit.
- [x] Verify the fix with focused tests and a direct backend/container check.
- [x] Rebuild Graphify after backend code edits.
- [x] Document the result and any remaining runtime caveats.

### Review Notes

- Latest failed jobs for `https://www.twitch.tv/videos/2768134699` exited from
  signal 11 around `30%`, before normal debug artifacts were written.
- `PYTHONFAULTHANDLER=1` reproduced the native crash in the isolated worker:
  `transformers -> torch._dynamo -> triton/knobs.py`, triggered by lazy
  SentenceTransformer loading from the semantic candidate path.
- Transcript cache for the same VOD loaded successfully (`7507` segments), so
  the crash was not Whisper/transcript generation.
- `pipeline_worker.py` now preloads `services.semantic_candidates` before
  importing `pipeline`, so Torch/Triton initializes before the TensorFlow/
  DeepFace stack. The late preload inside `pipeline.py` was removed because it
  still crashed after the CV stack had initialized.
- `_pipeline_subprocess_env()` now sets `PYTHONFAULTHANDLER=1`,
  `TORCHDYNAMO_DISABLE=1`, and `TORCH_COMPILE_DISABLE=1` for better native
  diagnostics and to keep dynamic compile paths disabled in serving workers.
- Verification passed: focused backend tests (`56` tests total across worker,
  pipeline integration, jobs router, semantic candidates, context patterns),
  Python compile checks, targeted `git diff --check`, Graphify rebuild, Docker
  backend image rebuild/recreate, `/api/status`, and a direct container replay
  of the old crash point (`worker_preload True`, `segments 7507`,
  `semantic_candidates 14`) without SIGSEGV.

## 2026-05-27 Plan: Disable Serving Cross-Platform Rescue

- [x] Disable Cross-Platform-positive rescue/boosting in the live frontend
  serving path by default.
- [x] Update the real backend environment so the running container does not
  keep the old behavior through `.env`.
- [x] Keep Cross-Platform rows available for diagnostics/training/evaluation,
  but not as a serving-time gate bypass.
- [x] Add/adjust tests to lock the default behavior.
- [x] Run focused verification and rebuild Graphify after backend code edits.

### Review Notes

- Serving Cross-Platform recall is now disabled by default in config and in the
  live backend `.env` via `SERVING_CROSS_PLATFORM_RECALL_ENABLED=false`.
- `_load_cross_platform_positive_windows` now defaults to disabled when the
  setting is absent, so known repost positives cannot silently act as a
  serving-time oracle.
- Cross-Platform rows remain available for training, evaluation, diagnostics,
  and offline recall reports.
- Runtime verification in the rebuilt Docker backend returned
  `status=disabled`, `enabled=false`, and `rows=0` for VOD `2768134699`.
- Verification passed: focused Cross-Platform recall tests (`4 passed`),
  `backend/tests/test_pipeline_integration.py` (`19 passed`),
  `backend/tests/test_clip_judge.py backend/tests/test_selector.py`
  (`29 passed`), Python compile, and `git diff --check`.
- The combined pytest process for those three files still has pre-existing
  module-stub isolation issues when `test_pipeline_integration.py` runs before
  clip-judge tests, so the suites were verified separately.
- Graphify was rebuilt with the project-local Graphify interpreter after the
  system `python3` command failed because `graphify` is not installed there.

## 2026-05-27 Plan: Analyze Latest Serving Job

- [x] Identify the latest completed frontend/serving job and requested VOD.
- [x] Inspect funnel summaries from candidate generation through final moments.
- [x] Check Proposal, Ranker, Judge, quality gate, and final selection diagnostics.
- [x] Run Cross-Platform-positive recall analysis when the VOD has known positives.
- [x] Summarize the dominant bottleneck and next concrete step.

### Review Notes

- Latest completed job inspected: `ac9c313e-199b-410b-a6e0-5232fc5ff04a`
  for Twitch VOD `2768134699`.
- The run used the trained Proposal model and Gemini multimodal Judge. The
  Judge processed all `80` scoped candidates in about `151.5s`.
- Known Cross-Platform positives for this VOD: `3`. They all reached
  Proposal, Ranker, and Judge scope, but only one survived in the actual final
  output boundaries.
- Dominant remaining issue: Ranker still buries known positives around ranks
  `110-186`, and boundary refinement can cut a rescued positive before its
  actual payoff.

## 2026-05-27 Plan: Use Unmatched Shorts For Context Discovery Only

- [x] Add a separate Context Pattern source path for eligible repost/unmatched shorts without feeding those rows into Proposal labels.
- [x] Merge titles/metadata/transcript diagnostics by short URL so Pattern Discovery sees richer short context.
- [x] Expose summary counts proving which rows went into context discovery versus training labels.
- [x] Add regression tests that unmatched repost rows create context patterns but do not become training rows.
- [x] Run focused tests, syntax/diff checks, and Graphify rebuild.

### Review Notes

- `train_proposal_pipeline.py` now builds the context-pattern artifact from a
  separate discovery-only row set. By default it merges
  `jynxzi_cross_platform_fingerprint_rows.jsonl`,
  `jynxzi_cross_platform_reposts.jsonl`, and
  `jynxzi_cross_platform_fingerprint_diagnostics.jsonl`.
- Rows are deduped by short URL and merged so manifest titles/metadata and
  diagnostics transcript excerpts enrich the same short context.
- These rows only flow into `context_patterns.write_pattern_store`; Proposal
  labels still come from materialized/debug rows and
  `jynxzi_proposal_training_rows.jsonl`.
- Current Jynxzi artifacts would increase context discovery coverage from 157
  matched positive rows to 353 deduped short rows.
- Verification passed: focused Proposal/context tests (`5 passed`),
  broader focused backend suite (`38 passed`), Python compile, and targeted
  `git diff --check`; Graphify rebuilt `3603` nodes / `9747` edges / `103`
  communities.

## 2026-05-27 Plan: Complete SentenceTransformer Prewarm Snapshot

- [ ] Add a repeatable prewarm command for the local SentenceTransformer
  snapshot used by Proposal training and Serving context-pattern embeddings.
- [ ] Download/save the full model weights into
  `tmp/prewarmed_sentence_transformers_local`.
- [ ] Verify required files include model weights, tokenizer, SentenceTransformer
  metadata, and pooling config.
- [ ] Verify `semantic_candidates._get_model()` loads from the completed local
  snapshot.
- [ ] Re-run Proposal training or provide the exact next command.

## 2026-05-27 Plan: Semantic Context Pattern Embeddings

- [x] Extend the dynamic context pattern store with semantic embedding
  centroids while keeping keyword labels only for debug/explanation.
- [x] Use semantic similarity as the primary context-pattern score during
  serving feature construction, with lexical overlap as a secondary signal.
- [x] Keep the implementation deterministic/testable with injectable fake
  embeddings and graceful lexical behavior when the local embedding model is
  unavailable.
- [x] Update tests to prove semantically similar wording can match even with
  low direct keyword overlap.
- [x] Run focused tests, syntax/diff checks, and Graphify rebuild.

### Review Notes

- `services/context_patterns.py` now writes version-2 context-pattern stores
  with optional `embedding_centroid`, `embedding_model`, and
  `semantic_matching_enabled` fields. Keyword labels remain debug/explanation
  only.
- Pattern clustering uses semantic cosine similarity when the local
  SentenceTransformer is available; otherwise it falls back to the previous
  lexical clustering.
- Candidate scoring uses semantic similarity as the primary score and blends in
  lexical overlap as a secondary signal. Existing generic
  `context_pattern_*` feature names remain unchanged, so Proposal/Ranker schema
  stays stable.
- Proposal training exposes
  `--context-pattern-semantic-cluster-similarity`, and the Docker wrapper wires
  it through.
- `docker-compose.yml` now mounts the prewarmed SentenceTransformer snapshot
  into the backend container so Serving can use the same semantic context
  matching path as Proposal training.
- Verification passed: focused context/proposal/pipeline tests (`74 passed`),
  Python compile for touched files, wrapper syntax/help checks, Docker Compose
  config validation for the embedding-model mount, `git diff --check`, and
  Graphify rebuild (`3596` nodes, `9727` edges, `102` communities).

## 2026-05-27 Plan: Remove Hardcoded Context Classes

- [x] Remove hardcoded context taxonomy buckets for the manually added
  unboxing/fanmail/story/event contexts.
- [x] Remove their feature columns from Proposal, CandidateFeatures, Ranker,
  and training feature lists.
- [x] Keep dynamic `context_patterns` features as the sole context-expansion
  path.
- [x] Update tests and task notes so they no longer assert hardcoded context
  classes.
- [x] Run focused tests, syntax/diff checks, and Graphify rebuild.

### Review Notes

- Removed the manually added context buckets and all direct feature fields for
  those buckets from `clip_taxonomy`, Proposal features, VOD candidate
  features, Ranker scoring, training feature lists, serving recall boosts, and
  Proposal training rows.
- Kept the dynamic context-pattern feature path:
  `context_pattern_score`, `context_pattern_support`,
  `context_pattern_views_prior`, `context_pattern_confidence`, and
  `context_pattern_similarity_margin`.
- Removed tests that expected the manual context classes and updated the
  pipeline integration assertion to check dynamic pattern propagation instead.
- Verification passed: no remaining backend code/test references to the removed
  context class names, Python compile for touched files, focused pytest suite
  (`73 passed`), `git diff --check`, and Graphify rebuild (`3587` nodes,
  `9696` edges, `101` communities).

## 2026-05-27 Plan: Dynamic Context Pattern Store

- [x] Add a reusable context-pattern service that clusters Cross-Platform short titles/transcripts into learned pattern artifacts.
- [x] Generate/update the profile pattern artifact during Proposal training.
- [x] Add generic dynamic pattern features to Proposal and Ranker feature rows.
- [x] Use the pattern artifact in serving feature construction without hardcoded class names.
- [x] Add tests for pattern building, scoring, and Proposal integration.
- [x] Run focused tests, syntax/diff checks, and Graphify rebuild.

### Review Notes

- Added `services/context_patterns.py`, which clusters repeated
  Cross-Platform title/transcript contexts into a profile artifact such as
  `jynxzi_context_patterns.json`.
- The learned patterns are dynamic: the model sees generic numeric features
  (`context_pattern_score`, `context_pattern_support`,
  `context_pattern_views_prior`, `context_pattern_confidence`,
  `context_pattern_similarity_margin`) plus debug IDs/labels, not one hardcoded
  feature per pattern.
- `train_proposal_pipeline.py` now writes the pattern artifact before
  materialization/build/train so Proposal training rows and serving-style
  feature construction can use the same artifact.
- Proposal features, VOD candidate features, Ranker heuristic scoring,
  Proposal training feature names, and VOD-ranker feature-signal preflight now
  include the dynamic context-pattern features.
- Serving recall-target boosts can also carry a matched pattern into known
  positive candidates via `matched_cross_platform_pattern`.
- Smoke check against current Jynxzi cross-platform rows found usable dynamic
  patterns (`source_rows=163`, `usable_rows=161`, `patterns=6` with
  `min_support=2`), so the extractor has enough signal to produce an artifact
  on the next Proposal training run.
- Verification passed: Python compile for touched files, dynamic pattern tests
  (`2 passed`), focused context/proposal/pipeline tests (`4 passed`), broader
  Proposal/Pipeline/Training suite (`74 passed`), touched-file
  `git diff --check`, and local Graphify rebuild (`3585` nodes, `9699` edges,
  `103` communities).

## 2026-05-27 Plan: Add Context Features For Missed Jynxzi Positives

- [x] Add Jynxzi-specific context taxonomy for unboxing/fanmail and event/status-story clips.
- [x] Pipe those context scores through Proposal features, VOD candidate features, and Ranker heuristic scoring.
- [x] Attach cross-platform title/transcript context to recall targets so known positives can carry their clip type into serving.
- [x] Extend Proposal training feature lists/backfill rows so retraining can learn the new context dimensions.
- [x] Add regression tests for taxonomy detection and cross-platform context boosts.
- [x] Run focused backend tests, syntax/diff checks, and Graphify rebuild.

### Review Notes

- Root cause: the three inspected positives were semantically meaningful
  Jynxzi moments (`unboxing/fanmail` and `World Cup suite/status story`), but
  the feature taxonomy only exposed generic buckets such as rage, clutch,
  funny reaction, chat shock, and streamer persona. That made these clips look
  like weak generic quote/chat windows to Proposal/Ranker.
- Added taxonomy buckets and scores for `unboxing_reveal`, `fanmail_item`,
  `story_flex`, and `event_status_story`.
- Proposal features, VOD candidate features, Ranker heuristic signal, Proposal
  training feature names, and legacy/backfill Proposal rows now all carry those
  context scores.
- Cross-platform transcript/audio match rows now preserve `repost_title` when
  available. Serving recall targets also classify title/transcript excerpt
  context and transfer the context scores into matching candidate features.
- Smoke check: an unboxing/fanmail text classifies as `fanmail_item` with
  `unboxing_reveal=0.58`; the World Cup suite text classifies as `story_flex`
  with `event_status_story=0.56`.
- Verification passed: syntax compile for touched Python files, targeted
  Proposal/Pipeline context tests (`2 passed`), focused backend suite
  (`40 passed`), Proposal/Ranker training tests (`46 passed`), touched-file
  `git diff --check`, and local Graphify rebuild (`3558` nodes, `9628`
  edges, `104` communities).

## 2026-05-27 Plan: Fix Candidate Recall Before Judge

- [x] Trace why the `5698.42s-5723.12s` cross-platform positive reached Proposal input but disappeared before Judge.
- [x] Inspect existing recall-target hooks in `backend/pipeline.py`, `proposal_scorer`, `ranker`, and candidate trace reporting.
- [x] Implement a minimal serving/eval recall fix that preserves normal production behavior but keeps known positives alive when recall positives are available.
- [x] Add focused regression coverage for Proposal/Ranker/Judge-scope recall promotion.
- [x] Run backend tests, syntax/diff checks, and Graphify rebuild after backend changes.
- [x] Document results and next validation run.

### Review Notes

- Root cause from last run: the known positive `5698.42s-5723.12s` had a
  Proposal-kept dense candidate but only reached Pre-Judge rank `82`, outside
  the `80`-candidate Judge scope. Other positive windows had overlapping dense
  or semantic candidates that were deduped away before Proposal/Ranker.
- Enabled serving cross-platform recall by default so VODs that already have
  matching cross-platform outcome rows can use those rows as supervised recall
  targets. VODs with no matching rows still run normally.
- Added recall-preserve IDs: per known positive, only a few strongest
  overlapping dense candidates bypass merge dedupe and Proposal top-k pruning.
  This avoids flooding the serving funnel while preventing known positives from
  disappearing before Ranker/Judge.
- Added merge support for `preserve_supplemental_ids` and applied it both when
  constructing Proposal input and when merging Proposal-kept candidates back
  with heuristic candidates.
- Replay check on job `038009c9-1169-4ace-8c55-62e9c314aae0`: the three known
  positives now have Proposal/normalized hits after the fix (`1`, `4`, and `3`
  overlapping candidates respectively) instead of being lost before Judge.
- Verification passed: targeted recall tests (`4 passed`), focused backend
  tests (`39 passed`), Python compile, `git diff --check`, and local Graphify
  rebuild (`3543` nodes, `9589` edges, `103` communities). The required system
  Graphify command still fails because system Python has no `graphify` module.

## 2026-05-27 Plan: Analyze Last Completed Serving Run

- [x] Identify the latest run, VOD, final status, and stop reason.
- [x] Inspect produced artifacts and stage progress.
- [x] Check logs for download/chat/transcription errors.
- [x] Determine what can be reused by the next run.
- [x] Summarize bottleneck and next step.

### Review Notes

- Latest completed serving job is `038009c9-1169-4ace-8c55-62e9c314aae0`
  for `https://www.twitch.tv/videos/2768134699`; status `done`, `12`
  final clips, `download_cached=true`.
- The previous job for the same VOD,
  `a6a6c1c9-4295-4d2e-9c43-10a775e4936e`, was cancelled manually and is
  not the last completed run.
- Funnel summary: `5000` dense-inventory candidates, `410` Proposal inputs,
  `240` Proposal-kept, `270` pre-judge ranked, `80` judged, `14` judged
  quality-kept, and `12` final moments.
- Gemini judge completed successfully: `80` judged candidates in about
  `146.6s`, no timeout. Final clips are all judged candidates.
- Candidate recall against cross-platform positives for this VOD found
  `3` known positives: `2/3` reached Proposal input, `1/3` reached
  Proposal-kept, `0/3` reached Judge scope, and `0/3` reached final moments.
- Manual positive-window check: positives around `1283.6s-1304.2s` and
  `5531.64s-5549.62s` were not covered by dense inventory under the current
  overlap rule. The positive around `5698.42s-5723.12s` reached dense
  inventory and Proposal input, but did not survive into ranked/Judge stages.
- Current bottleneck is not the Judge; it is candidate recall plus the
  Proposal/Ranker handoff before Judge. Next fix should trace the accepted
  Proposal candidate for `5698.42s-5723.12s` and broaden/repair recall for
  dense windows around known positives.

## 2026-05-27 Plan: Diagnose Current VOD Download Failure

- [x] Identify the failed/current job and exact VOD URL.
- [x] Inspect job state, partial files, and per-job artifacts.
- [x] Inspect backend logs for downloader/aria2/ffmpeg/cache errors.
- [x] Determine root cause and next corrective step.

### Review Notes

- Current job `a6a6c1c9-4295-4d2e-9c43-10a775e4936e` is for `https://www.twitch.tv/videos/2768134699`.
- The VOD download did complete: `video.mp4` exists in the job directory at about `10.85 GB`.
- The job moved to `transcribing`; `ffmpeg` is currently extracting `audio.wav` from the completed video.
- The apparent download failure was an API/status validation bug: the pipeline sets status `downloading_chat`, but `JobState` did not allow that value. FastAPI response validation then caused `/api/jobs/{job_id}` to fail while chat replay was being downloaded.
- Added `downloading_chat` to the backend `JobState`, frontend `JobStatus` type, and frontend stage metadata.
- Verification passed: backend model compile, `backend/tests/test_job_manager.py backend/tests/test_jobs_router.py` (`33 passed`), frontend lint, live job endpoint now returns `200 OK`, and local Graphify rebuild refreshed `3533` nodes, `9557` edges, `102` communities.

## 2026-05-27 Plan: Analyze Latest Serving Run After Feature Cache Fix

- [x] Identify the newest completed serving job and request VOD.
- [x] Inspect cache hit/write behavior after the feature-cache serialization fix.
- [x] Run Candidate-Recall against cross-platform positives for the VOD.
- [x] Inspect Proposal, Ranker, Judge, Quality Gate, and final selected clips.
- [x] Summarize the current bottleneck and next concrete step.

### Review Notes

- Latest completed serving job: `67585434-a826-44bc-880b-25f50efd0ee7`
  for `https://www.twitch.tv/videos/2777621798`; status `done`, `10`
  final clips.
- The feature-cache serialization fix is active in the running container.
  The latest job wrote reusable audio feature frames to
  `backend/tmp/storage/_feature_cache/9190b03c9f50c66c3fddc5aa4ec6a10c17fa3a7c0f6b30af03c2cce955d687f2/audio_frames.json`
  with `13430` frames. The earlier `pipeline_feature_cache_write_failed`
  log belonged to the previous job, not this latest run.
- Funnel summary: `6537` total trace rows, `414` proposal inputs, `240`
  proposal-kept, `250` pre-judge ranked, `80` judged, `16` judged
  quality-kept, `12` final-ranked, and `10` final moments.
- Gemini judge completed without timeout: `80/80` judged in `88.454s`.
- Candidate recall against cross-platform positives for this VOD:
  `3/3` reached `proposal_input`, `2/3` reached `proposal_kept`,
  `2/3` reached `judge_scope`/`clip_judge`, and only `1/3` reached
  `final_moments`.
- Concrete miss: the known positive window `3005.82s-3018.54s`
  (`https://www.youtube.com/watch?v=6cHbKPpSMNk`, similarity `0.565155`)
  had overlapping dense inventory candidates, but Proposal rejected the
  overlapping windows with scores around `0.124` and `0.134`.
- Current next step: rebuild/retrain the Proposal model with the latest
  cross-platform proposal false-negative rows, then rerun the same VOD and
  require this `3005.82s-3018.54s` positive to reach Proposal-kept,
  Judge scope, and final selection.

## 2026-05-27 Plan: Feature Cache JSON Serialization Fix

- [x] Reproduce the last-run cache-write failure from logs and inspect the cache write path.
- [x] Normalize feature-cache payloads to plain JSON-safe Python primitives before writing.
- [x] Add a regression test for NumPy-like scalar values in cached feature frames.
- [x] Run focused backend tests and syntax/diff checks.
- [x] Rebuild Graphify after backend code changes.
- [x] Document results.

### Review Notes

- Fixed the last-run cache write failure:
  `pipeline_feature_cache_write_failed ... TypeError: Object of type bool is not JSON serializable`.
- Added `_json_safe_cache_value()` in `backend/pipeline.py` so feature-cache metadata and frame payloads are recursively converted to plain JSON-safe values before writing.
- The normalizer handles dataclasses, dict/list/tuple/set values, `pathlib.Path`, and NumPy-like scalar/array values via `.tolist()` / `.item()`.
- Added a regression test that writes and reloads feature frames containing NumPy-like bool/float/int scalars.
- Verification passed:
  `backend/.venv/bin/python -m pytest backend/tests/test_pipeline_integration.py -k 'feature_cache'` -> `4 passed`;
  `backend/.venv/bin/python -m pytest backend/tests/test_pipeline_integration.py backend/tests/test_transcriber_progress.py` -> `29 passed`;
  `python3 -m py_compile backend/pipeline.py backend/tests/test_pipeline_integration.py`;
  scoped `git diff --check`.
- Required system Graphify command still fails because system `python3` has no `graphify` module; local Graphify interpreter succeeded and rebuilt `3533` nodes, `9557` edges, and `102` communities.

## 2026-05-27 Plan: 24h Frontend VOD Download Cache

- [x] Enable stable Twitch VOD download cache for frontend serving jobs.
- [x] Keep job-local source media cleanup unchanged after pipeline completion.
- [x] Set the default Twitch download cache TTL to 24h.
- [x] Add regression coverage for frontend cache hit and TTL eviction.
- [x] Verify with targeted downloader/pipeline tests, syntax checks, diff checks, and Graphify rebuild.

### Review Notes

- Changed `TWITCH_DOWNLOAD_CACHE_TTL_SEC` default from unlimited (`-1`) to
  `86400`, including `backend/.env` and `backend/.env.example`.
- Frontend serving jobs now use the existing Twitch VOD download cache even
  when `FRONTEND_JOB_CACHE_ENABLED=false`. The downloaded source media is still
  linked/copied into the per-job directory and cleaned up by the pipeline after
  completion.
- Cache metadata now stores `cached_at`; TTL eviction uses that timestamp when
  present and falls back to file mtime for legacy cache entries.
- Cache hits no longer refresh the Twitch cache TTL by touching the cached VOD,
  so a cached VOD expires after 24h instead of becoming indefinitely alive
  through repeated access.
- Added downloader tests covering frontend cache use, cache population, cache
  metadata, and stale-cache eviction.
- Verification passed: `test_downloader.py + test_cache_cleanup.py` `30
  passed`, `test_job_cancellation.py` `2 passed`, focused Pipeline cleanup/cache
  tests `7 passed`, Python compile checks, targeted `git diff --check`, and
  Graphify rebuild. Graphify rebuilt `3527` nodes, `9535` edges, and `101`
  communities.

## 2026-05-27 Plan: Safe Runtime Cache Cleanup

- [x] Inventory current `backend/tmp` usage and identify disposable runtime job data.
- [x] Delete old frontend serving job directories from `backend/tmp/storage`.
- [x] Clear the frontend job index so old jobs no longer appear in the UI/API.
- [x] Remove clearly derived temporary materialization cache that can be regenerated.
- [x] Preserve durable caches: download cache, transcript cache, feature cache, VOD feature cache, feedback, model/runtime caches.
- [x] Verify reclaimed space and remaining cache layout.

### Review Notes

- Removed stale UUID job directories from `backend/tmp/storage`, including
  cancelled/done job workspaces and their source media, chunks, debug files,
  clips, and keyframes.
- Removed the derived `_proposal_training_materialized` directory; it can be
  regenerated by Proposal training when needed.
- Cleared `backend/tmp/storage/jobs.json` to `[]` and restarted only the
  backend container so the UI/API no longer sees stale in-memory jobs.
- Preserved durable caches and source-of-truth data:
  `_download_cache`, `_transcript_cache`, `feedback`,
  `vision_language_features`, `backend/tmp/cache`, and
  `backend/tmp/vod_feature_cache`.
- `backend/tmp/storage` dropped from about `47G` to `4.1G`. Remaining storage
  is primarily `_download_cache` (`4.1G`), `_transcript_cache` (`612K`),
  `feedback` (`6.6M`), and `vision_language_features` (`16K`).
- `backend/tmp` is still about `162G` because the durable
  `backend/tmp/vod_feature_cache` is about `157G`; this was intentionally not
  deleted because it prevents expensive VOD feature rebuilds.

## 2026-05-27 Plan: Reuse Frontend VOD Feature Frames

- [x] Add a stable frontend Twitch VOD cache for reusable audio feature frames.
- [x] Add a stable frontend Twitch VOD cache for reusable video feature frames.
- [x] Keep candidate/ranker feature rows uncached so model, strategy, and candidate changes apply immediately.
- [x] Add logs and focused tests for feature cache hit/miss behavior.
- [x] Verify with targeted backend tests, syntax checks, diff checks, and Graphify rebuild.

### Review Notes

- Added `FRONTEND_FEATURE_CACHE_ENABLED=true` as the default companion to
  transcript caching.
- Frontend Twitch VOD jobs now use a stable `twitch_vod:<vod_id>` source to
  read/write reusable modality frames under `storage_dir/_feature_cache/`.
- Cached audio features can skip both audio extraction and audio feature
  recomputation when diarization is not needed. If local diarization is enabled,
  the pipeline still extracts audio before diarizing.
- Cached video features skip full-VOD visual extraction when the cache metadata
  matches the current feature-cache version, source, rounded media duration,
  visual runtime summary, and visual duration setting.
- Candidate/ranker feature rows are intentionally not cached; they still rebuild
  from the current candidate set, strategy, model artifacts, and
  `FEATURE_BUILD_VERSION`.
- Verification passed: feature-cache focused tests `5 passed`,
  `test_transcriber_progress.py + test_pipeline_integration.py` `28 passed`,
  Python compile checks, targeted `git diff --check`, and Graphify rebuild.
  Graphify rebuilt `3527` nodes, `9531` edges, and `100` communities.

## 2026-05-27 Plan: Reuse Frontend VOD Transcriptions

- [x] Enable stable transcript chunk caching for frontend Twitch VOD jobs without keeping source VOD media.
- [x] Keep audio feature extraction available after transcript cache hits so cached ASR does not degrade downstream quality.
- [x] Add focused tests for cache-source behavior and cached-transcript audio extraction.
- [x] Verify with targeted backend tests and syntax checks.
- [x] Rebuild Graphify after backend changes.

### Review Notes

- Added `FRONTEND_TRANSCRIPT_CACHE_ENABLED=true` as a separate default from
  `FRONTEND_JOB_CACHE_ENABLED=false`. Frontend VOD jobs can now reuse Whisper
  transcript chunks by stable Twitch VOD id while still deleting job-local
  source media after completion.
- Cache source for Twitch URLs is `twitch_vod:<vod_id>`, so rerunning the same
  VOD uses the same transcript chunk directory even with a new job id.
- On transcript cache hit, the pipeline still extracts audio and computes audio
  features. This avoids the earlier quality regression where cached transcript
  runs would silently lose audio feature signals.
- Verification passed: focused pipeline cache tests `3 passed`, extended
  pipeline/transcriber tests `25 passed`, Python compile checks, and targeted
  `git diff --check`.
- Graphify rebuilt successfully: `3511` nodes, `9463` edges, `107`
  communities.

## 2026-05-27 Plan: Proposal False-Negative Repair Loop

- [x] Locate the current Proposal training row builder, training pipeline, and candidate-recall evaluator.
- [x] Export cross-platform-positive candidates that reach `proposal_input` but miss `proposal_kept` as hard Proposal positives.
- [x] Integrate those hard positives into the Proposal training pipeline without enabling serving-time cross-platform boosts.
- [x] Add focused tests for the missed-positive export and training integration.
- [x] Verify with tests and a targeted recall/export run against the latest job.
- [x] Rebuild Graphify after backend code changes.

### Review Notes

- Started after the latest serving job showed the missing known-positive window
  `3005.82-3018.54s` reached Proposal input but was rejected at Proposal rank
  `420/429`.
- `train_proposal_scorer.collect_training_rows()` now treats cross-platform
  overlaps rejected by Proposal as `cross_platform_proposal_false_negative`
  positives with sample weight `5.0`.
- Cross-platform positives now override automatic Judge/Quality labels during
  Proposal training, but manual frontend feedback remains authoritative.
- `train_proposal_pipeline.py` writes those hard examples to
  `<profile>_proposal_false_negative_training_rows.jsonl` and records the count
  in the training pipeline summary.
- Verified against current `backend/tmp/storage`: `331` Proposal debug training
  rows collected, including `6` `cross_platform_proposal_false_negative` rows.
  The latest missed candidate `db433142... / 2777621798 /
  dense_inventory:20:12:0:299460` is now exported with sample weight `5.0`.
- Focused tests passed for Proposal training, Proposal pipeline integration,
  Proposal scorer, candidate funnel evaluation, candidate recall report, Python
  compile checks, and targeted `git diff --check`.
- Graphify rebuilt successfully: `3508` nodes, `9454` edges, `100`
  communities.

## 2026-05-27 Plan: Container Live Reload For Frontend/Backend

- [x] Add a dev compose override that bind-mounts backend/frontend source code and runs reload/watch servers.
- [x] Make `scripts/start_containerized.sh` use dev live-reload mode by default with a production opt-out.
- [x] Avoid forced image rebuilds on normal starts; keep an explicit rebuild switch for dependency/image changes.
- [x] Verify compose config, shell syntax, and relevant frontend/backend smoke checks.
- [x] Rebuild Graphify after backend-related code/config changes.

### Review Notes

- Added `docker-compose.dev.yml`. Backend source is bind-mounted to `/app/backend` and runs
  `uvicorn main:app --reload`; frontend source is bind-mounted to `/app` and runs `next dev`.
- Added Docker/WSL-safe polling defaults: `WATCHFILES_FORCE_POLLING=true`, `WATCHPACK_POLLING=true`, and
  `CHOKIDAR_USEPOLLING=true`.
- `scripts/start_containerized.sh` now enables dev live reload by default via `AI_CLIPPING_DEV=1`.
  Use `AI_CLIPPING_DEV=0` for image-only containers.
- Normal startup no longer forces `--build`. Use `AI_CLIPPING_REBUILD=1 ./scripts/start_containerized.sh` after
  dependency or Dockerfile changes.
- Verified `bash -n scripts/start_containerized.sh`, CPU/dev and CUDA/dev `docker compose config`, backend `/api/status`,
  frontend `/api/status`, and container logs.
- Restarted the containers once to apply the new dev mode. Backend logs show `Started reloader process ... using
  WatchFiles`; frontend logs show `next dev -H 0.0.0.0 -p 3000`.
- Required system `python3 ... graphify.watch...` failed because this shell has no `graphify` module; rebuilding with the
  local Graphify interpreter succeeded with `3505` nodes, `9445` edges, and `103` communities.

## 2026-05-27 Plan: Make Judge The Primary Final Selector

- [x] Add a post-Judge final selection policy that prioritizes judged Quality-Gate candidates by Judge score and confidence.
- [x] Keep only transparent post-Judge removals: temporal overlap/diversity, hard Quality-Gate rejection, and boundary refinement.
- [x] Prevent unjudged candidates and source caps from displacing judged Quality-Gate candidates when enough judged candidates exist.
- [x] Surface the post-Judge selection policy in debug/funnel artifacts.
- [x] Add focused selector tests and run backend verification.
- [x] Rebuild Graphify after backend code changes.

### Review Notes

- Started after deciding the Clip Judge should be the final quality selector and later stages should only do Quality Gate,
  dedupe/diversity, and boundary refinement.
- Added `select_post_judge_final_candidates()`: after the Quality Gate, judged candidates are ordered by Judge score and
  confidence. Source caps no longer run after the Judge, and unjudged candidates are only fallback fill.
- The post-Judge selection summary is logged and written to `debug/post_judge_final_selection.json`.
- Removed the hidden hard `5` candidate limit before boundary refinement; the complete Post-Judge selection pool now reaches
  boundary refinement.
- Offline replay against job `cd2e678d-b067-472a-84b1-ee4d4856defd` selected `12` judged candidates and `0` unjudged
  fallbacks from `194` Quality-Gate accepted rows. The previously displaced known-positive windows around `425s` and
  `3268s` are now in the boundary-refinement pool.
- Verification passed: targeted backend suite `27 passed`, Python compile checks, and targeted `git diff --check`.
- Required system `python3 ... graphify.watch...` failed because this shell has no `graphify` module; rebuilding with the
  local Graphify interpreter succeeded with `3505` nodes, `9445` edges, and `103` communities.

## 2026-05-26 Plan: Cross-Platform Recall Calibration

- [x] Set serving Judge shortlist to 80 in the active container env defaults.
- [x] Load cross-platform positive windows for the submitted Twitch VOD during serving.
- [x] Preserve dense candidates that overlap known cross-platform positives before dense/proposal caps.
- [x] Boost and annotate cross-platform recall targets through Ranker, Judge reserve, and Quality Gate.
- [x] Add focused tests and replay the latest job diagnostics against the new logic.
- [x] Run focused backend tests, syntax/diff checks, and Graphify rebuild.

### Review Notes

- Started after latest job showed recall-reserve pulled `12008/12052s` into Judge, but the LLM/quality gate rejected the
  known repost-positive at `3.74`, while `4691s` stayed outside Judge scope and `11695s` had no candidate overlap.
- Implemented Serving-side cross-platform recall calibration: VOD-specific positive windows are loaded from
  `jynxzi_cross_platform_fingerprint_rows.jsonl`, dense inventory is generated uncapped before Recall promotion, matching
  recall targets are boosted through Ranker/Judge/Quality Gate, and active env defaults now use `SHORTLIST_JUDGE_TOP_K=80`.
- Offline replay against job `474a0e5b-8471-4579-8f21-423736b72e96` found `4/4` cross-platform positives in promoted dense
  candidates. The previously missing `11695.36-11716.58s` positive now has a promoted matching window
  `11659.04-11719.04s`.
- Replaying the old normalized candidate set with the new feature boost moved the retained recall targets into Judge range
  at ranks `12`, `14`, `15`, and `22`.
- Verification passed: focused backend suite `54 passed`, changed Python files compile, `git diff --check` is clean for the
  touched files, and Graphify rebuilt `3495` nodes / `9409` edges / `104` communities.
- Follow-up correction: normal frontend Serving must not use known cross-platform positives as an inference boost, because
  that leaks labels into the product run. `SERVING_CROSS_PLATFORM_RECALL_ENABLED` is now disabled by default; the code path
  remains available only for explicit debug/eval/training calibration.

## 2026-05-26 Plan: Analyze Latest Serving Job After Recall Reserve

- [x] Identify the newest completed debug-backed serving job.
- [x] Inspect Judge shortlist size, recall-reserve diagnostics, and final clips.
- [x] Run Candidate-Recall against cross-platform positives for the job VOD.
- [x] Summarize the current bottleneck and next concrete improvement.

### Review Notes

- Started after request to analyze the last job following the recall-reserved Judge shortlist change.
- Latest completed job is `474a0e5b-8471-4579-8f21-423736b72e96` for
  `https://www.twitch.tv/videos/2774440901`.
- Recall-reserve is active, but the running container has `SHORTLIST_JUDGE_TOP_K=50`, so it judged `40` normal Ranker
  candidates plus `10` recall-reserved candidates, not `80`.
- Cross-platform-positive recall improved at Judge scope from `0/4` to `2/4`; one known positive around `12008/12052s`
  was explicitly pulled in by recall reserve. It was then rejected by the LLM/quality gate with judge score `3.74`.
- The `4691s` positive still died outside Judge scope at pre-Judge rank `150`; the `11695s` positive still has no matching
  candidate overlap.
- Current bottleneck is no longer only Judge scope. It is now: better candidate boundary/coverage for `11695s`, better
  reserve/ranker promotion for `4691s`, and Judge calibration for transcript-matched repost positives that the LLM
  currently underrates.

## 2026-05-26 Plan: Recall-Reserved Judge Shortlist

- [x] Add serving config for Judge recall reserve slots and candidate pool size.
- [x] Build a Judge input ordering that keeps most Ranker top candidates but reserves slots for high-recall candidates from deeper ranks.
- [x] Surface recall-reserve diagnostics in `clip_judge_stage.json` and candidate rationales.
- [x] Add focused tests for reserved Judge shortlist behavior.
- [x] Run focused backend tests, syntax/diff checks, and Graphify rebuild.

### Review Notes

- Started after latest job showed Proposal recall improved but cross-platform positives were ranked at `52`, `78`, and `138`,
  outside the actual Judge Top-50.
- Added a deterministic Judge input reorder: default `80` judged candidates now means `70` normal Ranker slots plus `10`
  recall-reserved slots selected from the top `160` pre-Judge candidates by Proposal/heuristic/feature signal.
- Verified with focused tests: `19 passed` across `test_pipeline_integration.py`, `test_candidate_recall_report.py`, and
  `test_evaluate_candidate_funnel.py`.
- Offline replay on latest job `1815a324-0c4a-4e62-b4ee-a1869a278d69` selected `10` recall-reserved candidates; the known
  positive around `12052s` moved into the Judge top-80 via reserve, and the known positive at pre-Judge rank `52` would now
  be naturally included by the default top-80.
- Graphify backend graph rebuilt after backend edits.

## 2026-05-26 Plan: Analyze Latest Serving Job

- [x] Identify the newest completed debug-backed serving job.
- [x] Run Candidate-Recall against cross-platform positives.
- [x] Inspect Proposal, Ranker, Judge, quality gate, and final clip diagnostics.
- [x] Verify whether the rebuilt backend image includes the latest Serving fixes.
- [x] Summarize the dominant bottleneck and next action.

### Review Notes

- Started after request to analyze the latest job.
- Latest completed job is `1815a324-0c4a-4e62-b4ee-a1869a278d69` for
  `https://www.twitch.tv/videos/2774440901`.
- The rebuilt backend image is active: `/app/backend/services/proposal_scorer.py`
  contains `_serving_score`, so the Proposal guardrail was exercised.
- Proposal improved materially versus the previous stale-container run:
  final Proposal mean rose from about `0.003115` to `0.180727`, and
  cross-platform recall at Proposal-kept rose from `2/4` to `3/4`.
- Cross-platform-positive recall still fails downstream: `3/4` reach
  ranked pre-Judge, only `0/4` reach real Judge scope/top50, and `0/4`
  become final moments.
- Per-positive bottlenecks: the `4691s` positive is kept by Proposal but buried
  at pre-Judge rank `138`; the `12008s/12052s` positives are kept by Proposal
  but rank `52`/`78`, just outside or below Judge scope; the `11695s` positive
  has no candidate overlap because dense inventory windows in that area end
  before the target interval.
- Dominant next bottleneck is Ranker/Judge-scope promotion of cross-platform
  positives, plus dense coverage around the `11695s` interval.

## 2026-05-26 Plan: Analyze Latest Serving Job After Dense Recall Fix

- [x] Identify newest completed serving job and request VOD.
- [x] Re-run Candidate-Recall against cross-platform positives.
- [x] Inspect Proposal, Ranker, Judge, quality gate, and final clips.
- [x] Verify whether the running backend container included the latest Serving fixes.

### Review Notes

- Latest completed job is `fe90204f-0831-490c-a26a-403728673f4c` for
  `https://www.twitch.tv/videos/2774440901`.
- The job completed successfully with 5 final clips and 50 judged candidates.
  The Clip Judge finished without timeout in about `1488s`.
- Cross-platform-positive recall stayed weak: 4 positives total, 3/4 reached
  Proposal input, 2/4 reached Proposal-kept / pre-Judge / Judge scope, 0/4
  survived quality gate or final moments.
- The local Serving fixes did not actually run in this job. The Docker backend
  container does not contain `_serving_score` in `/app/backend/services/proposal_scorer.py`,
  and only `tmp` plus `ml/artifacts` are bind-mounted. A backend restart alone
  is insufficient; the backend image must be rebuilt/recreated.
- Proposal scores remained near-zero for final clips (`mean=0.003115`) because
  the old container code still used raw model scores without the new guardrail.

## 2026-05-26 Plan: Preserve Dense Recall Candidates In Serving Funnel

- [x] Stop dense-inventory boundary variants from being collapsed by the generic candidate dedupe.
- [x] Add a Proposal serving guardrail so a poorly calibrated model cannot bury strong dense/heuristic candidates at near-zero scores.
- [x] Add focused regression tests for dense variant preservation and Proposal model guardrail behavior.
- [x] Run focused backend tests, syntax/diff checks, and Graphify rebuild after backend edits.

### Review Notes

- Started after latest job analysis showed known cross-platform positives were present in dense inventory but lost by
  Proposal input, Proposal scoring, or pre-Judge promotion.
- `candidate_normalizer` now uses stricter duplicate criteria for dense-vs-dense candidates. Nearby dense variants are
  only collapsed when their anchor, start, end, and duration are effectively the same, so alternate boundaries can survive
  into Proposal input.
- `proposal_scorer` now applies a serving-time floor from the heuristic score when a trained model assigns a very low score
  to strong dense/heuristic candidates. The model still ranks normally above that floor.
- Added regression coverage for dense boundary preservation and for the low-model-score Proposal guardrail.
- Verification passed: `./.venv/bin/python -m pytest -q tests/test_candidate_normalizer.py tests/test_proposal_scorer.py
  tests/test_evaluate_candidate_funnel.py tests/test_train_proposal_scorer.py`, Python compile checks, targeted
  `git diff --check`, and Graphify rebuild.

## 2026-05-26 Plan: Cross-Platform Recall Targets For Proposal And Ranker

- [x] Fix Candidate-Recall evaluation so excluded `judge_scope` / `quality_gate` rows do not count as actual judged/kept matches.
- [x] Promote dense-inventory windows overlapping cross-platform positives into Proposal input during proposal-training materialization.
- [x] Ensure Proposal training labels overlapping dense windows as high-weight recall positives instead of letting low model scores reject them.
- [x] Add Ranker training rows/features for cross-platform-positive dense windows so known viral windows become Top-K targets before the Judge.
- [x] Add or update tests covering Proposal recall promotion, ranker training inclusion, and accurate judged-scope recall.
- [x] Run focused backend tests, `git diff --check`, and Graphify rebuild.

### Review Notes

- Implemented effective-stage recall in `backend/ml/evaluate_candidate_funnel.py`.
  `judge_scope` now only counts `included`, `clip_judge` only counts actually
  judged rows, and `quality_gate` only counts kept rows with a Judge result.
- Proposal-training dense inventory is now generated uncapped first, then
  positive-overlap windows are promoted into the capped candidate set. Those
  dense Cross-Platform overlaps become `cross_platform_dense_recall_positive`
  rows with higher sample weight and `proposal_recall_target=true`.
- Proposal materialization now writes `positive_recall_targets.jsonl` and
  annotates matching `candidate_proposals.jsonl` rows with recall-target
  metadata, so failed/preserved materialized jobs show whether known positives
  survived Proposal input.
- Ranker Cross-Platform labels now use `cross_platform_outcome_positive`
  instead of `weak_positive`, with stronger sample weight and ordinal label.
  This makes known viral repost matches a hard Top-K learning target.
- Verification passed: focused backend tests via
  `./.venv/bin/python -m pytest -q tests/test_evaluate_candidate_funnel.py
  tests/test_train_proposal_scorer.py
  tests/test_train_vod_ranker.py::TrainVodRankerTests::test_cross_platform_fingerprint_rows_are_sent_through_vod_feature_builder`,
  Python compile checks, targeted `git diff --check` for touched files, and
  Graphify rebuild. Full `git diff --check` is still blocked by pre-existing
  CRLF/trailing-whitespace artifact CSV changes in `backend/ml/artifacts/`.

## 2026-05-26 Plan: Inspect New Latest Job Candidate Recall

- [x] Identify the newest debug-backed serving job and its VOD.
- [x] Run Candidate-Recall against cross-platform positives for that job.
- [x] Inspect funnel, Judge, quality gate, and final clip diagnostics.
- [x] Summarize whether the new job changes the next improvement step.

### Review Notes

- Started after request to rerun the Candidate-Recall inspection on the new job.
- Newest debug-backed job is `c1fb71f3-af31-46e9-ac6b-6c6d618b0e43`
  for `https://www.twitch.tv/videos/2774440901`, profile `jynxzi`.
- This VOD has 4 cross-platform positive windows, so recall is now
  measurable. Aggregated recall: 3/4 matched Proposal input, 2/4 matched
  ranked pre-Judge, 0/4 reached actual Clip Judge, 0/4 final.
- Per-window diagnosis: the `4691.26-4699.42` positive reached Proposal input
  but was Proposal-rejected with score `0.054082`; the `11695.36-11716.58`
  positive was present in dense inventory but did not reach Proposal input;
  the `12008.38-12038.16` and `12052.16-12061.86` positives shared a kept
  dense-inventory candidate with Proposal score `0.394469`, but Ranker buried
  it at pre-Judge rank `167`, so it was outside the judged set.
- Normal funnel: 5,000 dense inventory windows, 317 Proposal inputs, 240
  Proposal-kept, 237 ranked pre-Judge, 50 judged, 10 judged quality-kept, 12
  final-ranked, 5 final moments.
- Judge took about `1952.16s` for 50 candidates. Final selected clips averaged
  Judge score `7.152`, pre-Judge rank `26.0`, and Proposal score `0.118012`.
- Current actionable bottleneck is not dense candidate generation; it is
  Proposal/Ranker promotion of known outcome-positive dense-inventory windows
  into the actual judged shortlist.

## 2026-05-25 Plan: Inspect Latest Job Candidate Recall

- [x] Identify the latest debug-backed serving job and its VOD.
- [x] Run Candidate-Recall against cross-platform positives for that job.
- [x] Inspect funnel, Judge, quality gate, and final clip diagnostics.
- [x] Summarize what the result means for the next improvement step.

### Review Notes

- Started after request to inspect Candidate-Recall on the last job.
- Latest debug-backed job is `14109705-7138-4721-a6bb-b5fb8ad233bf`,
  request URL `https://www.twitch.tv/videos/2774602304`, profile `jynxzi`.
- Cross-platform outcome file currently has no positive windows for VOD
  `2774602304`, so `positive_window_recall.positive_windows=0` and recall is
  not computable for this job.
- Normal funnel diagnostics are still useful: 5,000 dense inventory windows,
  319 Proposal inputs, 240 Proposal-kept, 238 ranked pre-Judge, 50 judged, 9
  judged quality-kept, 12 final-ranked, 5 final moments.
- Judge took about 2,018.6 seconds for 50 candidates. Three judged candidates
  reached judge score >= 7.0; all three became final clips. High-Judge
  pre-Judge top20 ratio was 0.667 and top40 ratio was 1.0.
- Final selected clips still have low Proposal scores on average
  (`0.205732`), with three of five below `0.15`; Proposal confidence remains
  the weak link even though Ranker placement improved versus prior jobs.
- No backend code was changed; this task log was updated with the inspection.

## 2026-05-25 Plan: Proposal Training CLI Compatibility

- [x] Make Proposal training tolerate legacy/local-train `--date-from` and `--date-to` arguments.
- [x] Add regression coverage for direct `train_proposal_pipeline.py` invocation with ignored date arguments.
- [x] Update lessons for the command-interface mismatch.
- [x] Verify with focused tests and `git diff --check`.

### Review Notes

- `scripts/train_proposal_local_docker.sh` now consumes `--date-from` and
  `--date-to` as compatibility no-ops instead of forwarding them into the
  Proposal trainer.
- `backend/ml/train_proposal_pipeline.py` also accepts those flags directly as
  hidden no-ops, so direct module calls no longer fail on reused local-train
  date arguments.
- Added regression coverage to
  `backend/tests/test_proposal_training_pipeline.py`.
- Verification passed: focused Proposal pipeline test, wrapper syntax check,
  Python compile check, direct CLI smoke test with date arguments,
  `git diff --check`, and Graphify rebuild through the local Graphify
  interpreter.

## 2026-05-25 Plan: Implement Clipfinding Quality Levers

- [x] Add versioned artifact/eval audit output so every run can show which Ranker/Proposal artifacts and training/eval summaries were actually used.
- [x] Add Proposal-recall diagnostics against positive/outcome windows across generated, proposal input, proposal kept, pre-Judge top-k, Judge scope, and final stages.
- [x] Add explicit Jynxzi clip-type taxonomy signals to Proposal/Ranker diagnostics and training rows.
- [x] Promote existing multimodal proxy signals before the Judge through Proposal features and funnel slices.
- [x] Extend feedback/review exports so future training can distinguish good clips, false positives, missed virals, bad boundaries, and bad durations.
- [x] Verify with focused tests, full backend tests where practical, lint/build if frontend changes, `git diff --check`, and Graphify rebuild after backend edits.

### Review Notes

- Added a deterministic Jynxzi clip taxonomy (`rage`, `clutch`,
  `funny_reaction`, `trash_talk_payoff`, `chat_shock`, `ranked_tension`,
  `failed_setup`, `streamer_persona_beat`) and surfaced it in Proposal
  features, Ranker features, trace rows, funnel summaries, and frontend
  diagnostics.
- Added `multimodal_proxy_score` from existing pre-Judge signals
  (chat/activity, face/motion, audio pop, game events, laughter/exclamation,
  hype) so Proposal and Ranker can score non-transcript evidence before the
  LLM Judge.
- Extended Proposal training feature names and legacy row conversion so new
  taxonomy/multimodal fields are learnable on the next Proposal retrain.
- Extended frontend feedback with tags for `good_clip`, `false_positive`,
  `missed_viral`, `wrong_boundary`, and `wrong_duration`; exports now preserve
  those tags and weight false positives vs boundary/duration mistakes
  differently.
- Added `backend/ml/audit_clipfinding_artifacts.py` and wired it into
  `train_proposal_pipeline.py`; Proposal training now writes a current
  clipfinding artifact audit next to the output artifact.
- Extended `backend/ml/evaluate_candidate_funnel.py` with
  `--positive-windows` so cross-platform/outcome positives can be measured
  against Proposal input, Proposal kept, Ranker top-k, Judge scope, quality
  gate, and final stages.
- Verification: focused Proposal/Funnel/Feedback/Training/Audit tests
  passed, related Feature/Ranker tests passed, full backend test suite passed,
  frontend lint/build passed, and `git diff --check` passed.
- Graphify rebuild passed via the local Graphify interpreter after system
  `python3` did not have the `graphify` module installed.

## 2026-05-25 Plan: Backend Clipfinding Gap Analysis

- [x] Map the current serving clipfinding pipeline from job API through candidate generation, feature building, ranking, Judge reranking, and final selection.
- [x] Review the Jynxzi-specific training and label workflow: cross-platform labels, frontend feedback, Judge snapshots, Proposal/Ranker artifacts, and holdout/eval tooling.
- [x] Inspect current artifact/debug evidence and recent review notes for measurable quality bottlenecks.
- [x] Compare the current system against the target of OpusClip-level clip discovery without post-processing.
- [x] Produce prioritized concrete next steps with verification criteria.

### Review Notes

- Started after request to analyze the whole backend for market-leader-level
  viral clip finding for one streamer, before post-processing.
- Serving path is a broad audited funnel: ASR/audio/video/chat enrichment,
  peak/semantic/dense/dense-inventory recall, Proposal shortlist, Ranker,
  Clip Judge, quality gate, diverse top selection, and boundary refinement.
- Current biggest serving bottleneck is not raw candidate breadth; latest
  debug job still generated 5,000 dense-inventory windows and judged 80
  candidates, but final winners had weak upstream positions: mean pre-Judge
  rank 53 and mean Proposal score about 0.136.
- Current Jynxzi supervision is still small for market-leader-level quality:
  144 cross-platform positive matches across 52 VODs, current Ranker training
  around 327 policy rows with 138 positives / 189 negatives, and outcome
  holdout only 24 positives / 17 negatives against a 300-positive target.
- Proposal model is under-proven: it has thousands of rows, but heldout AUC is
  only about 0.62 / precision@K about 0.17, and the pipeline summary artifact
  appears overwritten by a pytest-temp run, so it should not be treated as a
  production training summary.
- Product eval and latest debug evidence point to the same gap: Judge rescues
  candidates late, while Proposal/Pre-Judge Ranker do not rank future winners
  confidently enough before the expensive LLM stage.
- The OpusClip-level target implies four missing pillars before post-processing:
  larger outcome-labeled gold/silver data, explicit Proposal/Recall gates,
  stronger multimodal discovery beyond transcript matching, and streamer-
  specific Jynxzi taxonomy/eval slices.
- No backend code was changed in this analysis; only this task log was updated.

## 2026-05-19 Plan: Speed Up Ollama Clip Judge

- [x] Change serving Judge defaults to use a smaller Ollama model.
- [x] Increase Ollama Judge batch size and reduce judged shortlist size.
- [x] Ensure the selected Ollama model is available in the running Ollama container.
- [x] Restart backend runtime and verify `/api/status` reports the new Judge model.
- [x] Document verification results.

### Review Notes

- Changed serving config from `qwen2.5:14b`, Judge batch `4`, Judge Top-K
  `80` to `qwen2.5:7b`, Judge batch `8`, Judge Top-K `50`.
- Updated `backend/.env.example` with the same defaults so future env files do
  not drift back to the slower setup.
- Pulled `qwen2.5:7b` into `ai-clipping-ollama`; `ollama list` now shows both
  `qwen2.5:7b` and the older `qwen2.5:14b`.
- Recreated the backend container and verified it is healthy.
- Runtime verification inside the container reports:
  `ollama_model=qwen2.5:7b`, `clip_judge_batch_size=8`,
  `shortlist_judge_top_k=50`, `judge_provider=ollama`.
- `/api/status` reports Ollama active on `qwen2.5:7b`.
- Ollama smoke test passed. First request cold-loaded the model in about
  `29s`; immediate warm request completed in about `307ms`.

## 2026-05-19 Plan: Fix Context Leakage And Judge Feedback Calibration

- [x] Reproduce the stale/global callback leakage in VOD memory context selection.
- [x] Fix context retrieval so setup/callback evidence must be locally relevant to the candidate window.
- [x] Increase Judge-accepted/rejected feedback strength for Proposal and Ranker training without letting unjudged rows become positives.
- [x] Add/update focused tests for context relevance and feedback weighting.
- [x] Run focused backend tests, syntax checks, Graphify rebuild, and document results.

### Review Notes

- Triggered by latest job analysis: final clips reused the same unrelated
  callback summary and `minimum_required_setup_start=358.48` across far-apart
  windows, while low Proposal scores showed Judge feedback needs stronger
  training influence.
- Fixed `vod_memory` so setup/open-loop/callback spans must be locally relevant
  to the candidate and callback spans cannot bridge beyond the configured
  lookback window through weak one-token overlap.
- Verified the fix against the last job: final clip setup starts are now local
  (`516.82`, `2081.54`, `1480.38`, `8367.92`, `1397.48`) instead of all
  leaking the same stale `358.48` callback.
- Increased Judge feedback influence: serving Judge quality exports now receive
  score/confidence-aware sample weights, Proposal training uses stronger
  Judge-accepted/rejected weights, and Ranker `judge_feedback` / judge quality
  class multipliers were raised.
- Verification passed: focused VOD memory/training export/Proposal/Ranker tests
  (`70 passed`), analyzer/candidate recall/pipeline integration tests
  (`42 passed`), `py_compile`, live context check on the latest job, Graphify
  rebuild, backend CUDA image rebuild/restart with healthy container and
  `/api/status` response, and `git diff --check`.

## 2026-05-19 Plan: Analyze Latest Job

- [x] Identify the latest job and its status.
- [x] Inspect job debug artifacts, candidate funnel, selected clips, and judge/ranker/proposal diagnostics.
- [x] Check backend logs for errors or fallback behavior around that job.
- [x] Summarize quality bottlenecks and concrete next actions.

### Review Notes

- Started after request to analyze the latest job after backend restart and
  artifact retraining.
- Latest completed artifact job is `1e29506f-bed5-4234-8b2b-dae688e07ca0`
  for Twitch VOD `2774602304`; it finished with 5 clips and the frontend
  successfully requested all five clip files.
- Candidate funnel: 5,000 dense inventory windows, 249 merged candidates, 241
  normalized, 339 Proposal inputs, 240 Proposal-kept, 241 pre-Judge ranked, 80
  judged, 170 quality-kept pool entries, 12 final-ranked, 5 final clips.
- Proposal artifact was loaded, but final clip Proposal ranks were only 17, 27,
  86, 102, and 170. Two final clips still had very low Proposal scores
  (`0.076561` and `0.052378`), so Proposal is preserving coverage more than
  confidently selecting winners.
- Clip Judge worked through Ollama without timeout or fallback and judged 80
  candidates, but took 2,470.581 seconds. It accepted only 9 judged candidates;
  71 judged candidates failed mostly on low judge score, narrative completeness,
  payoff strength, and standalone clarity.
- Final clips were mostly rescued by Judge: one dense inventory candidate moved
  from pre-Judge score 2.399 to judged score 8.122, and one peak moved from
  2.720 to 7.162. This confirms the Ranker/Proposal still miss some candidates
  before Judge.
- Chat replay was effectively unusable for most of the VOD: only 66 chat rows
  from seconds 17-22 were fetched, with log reason
  `twitch_rechat_fetch_stopped ... no_offset_progress offset=23`. As a result
  chat features are zero for the selected clips.
- Transcript/context diagnostics show some bad global context leakage: all
  final moments had the same unrelated callback summary and
  `minimum_required_setup_start=358.48`, which is not credible for clips hours
  later. This can mislead Judge/final explanations and should be fixed before
  trusting context summaries.

## 2026-05-18 Plan: Retrain Proposal And Ranker Artifacts

- [x] Run Proposal training through `scripts/train_proposal_local_docker.sh`.
- [x] Inspect the Proposal training summary and artifact metrics.
- [x] Determine the safest Ranker retrain path from existing staged/training artifacts.
- [x] Run or prepare the Ranker retrain with the current corrected training signals.
- [x] Verify generated artifacts and document the result.

### Review Notes

- Triggered after quality-fix implementation because training-signal changes do
  not affect serving until artifacts are retrained.
- Proposal retrain completed successfully through Docker. New
  `jynxzi_proposal_scorer.json` is trained on 9,937 rows with 755 positives.
  Heldout metrics: AUC 0.61846, logloss 0.321553, precision@K/recall@K
  0.169863. Label sources now include `judge_quality_accepted`,
  `judge_quality_rejected`, `pipeline_selected`, and `pipeline_low_rank`.
- Ranker retrain should use the artifact-only local training path with the last
  successful date range (`2026-01-01` to `2026-05-12`) so the workflow exports
  current feedback/training rows instead of directly training on a stale dataset
  file.
- Ranker artifact-only retrain completed successfully with return code 0.
  `jynxzi_ranker.json`, `jynxzi_boundary_ranker.json`, and
  `jynxzi_vod_training_rows.jsonl` were rewritten at the end of the run.
- The refreshed transcript matching stage used 64 VOD rows and 356 repost rows,
  loaded 64 VOD transcripts, reused 328 repost transcript caches, fetched 28
  missing repost transcripts, and produced 144 transcript matches across 52
  VODs at the current 0.5 threshold.
- Current training rows after the retrain: 450 VOD-ranker rows, 143
  cross-platform rows on disk, and 9,852 proposal training rows.
- Ranker eval artifact exists and reports 30 eval rows. Runtime model precision
  improved over the heuristic at the selected threshold (0.909 vs 0.800), while
  ranking metrics stayed flat on this small eval set.
- Verification passed after retrain: artifact timestamps confirmed, summaries
  inspected, row counts checked, and `git diff --check` passed after normalizing
  retrain-generated CSV line endings.

## 2026-05-18 Plan: Fix Clip Quality Weaknesses

- [x] Fix funnel semantics before changing scoring: split `quality_kept` into judged quality pass, unjudged retained pool, and final-ranked pool so every report shows whether candidates were actually judged.
- [x] Promote Clip-Judge and frontend feedback into durable training rows for both Proposal and Ranker, with explicit labels for selected, rejected, skipped-by-user, and low-judge-score candidates.
- [ ] Retrain the Proposal model on real serving windows plus cross-platform positives, not only transcript-match overlap, and require heldout metrics plus live-job lift checks before trusting the artifact.
- [x] Add Proposal calibration diagnostics: final selected candidate proposal-score distribution, missed-high-judge candidates, precision@top-k, recall@top-k, and source-specific lift for `dense_inventory`, `peak`, `quote`, and `dense_window`.
- [x] Improve the Pre-Judge Ranker with Judge/feedback outcomes so high-judge candidates rank in the top 20 before the LLM stage instead of being rescued from ranks 24-32.
- [x] Tighten candidate generation: replace pure brute-force dense windows with boundary-aware dense windows, transcript sentence boundaries, local context expansion, and overlap pruning before Proposal scoring.
- [x] Add candidate boundary repair before Judge: expand weak windows to include setup/payoff context, snap starts/ends to transcript boundaries, and penalize windows with incomplete standalone text.
- [x] Change the Clip-Judge rubric from "complete moment" to "short-form viral potential": require hook, tension/reaction/payoff, standalone clarity, and Jynxzi-specific replay value; report separate sub-scores instead of one opaque score.
- [x] Use Judge only as a teacher/reranker, not as final truth: feed high/low judge outcomes into training, but keep final validation tied to frontend feedback and cross-platform positives.
- [x] Add a before/after evaluation command that runs the same VOD through old vs new artifacts and reports final clip overlap, judge-score lift, proposal-score lift, and manual review samples.

### Success Criteria

- Final selected clips should usually have non-trivial Proposal scores; obvious winners should not depend on broad shortlist survival.
- At least 80% of high-Judge candidates should appear in the pre-Judge top 20 on heldout jobs.
- The frontend funnel should clearly show whether a bad final result came from generation, Proposal, Ranker, Judge, Quality Gate, or final selection.
- The Proposal artifact should show meaningful heldout lift, not just high accuracy from many easy negatives.
- Manual sample review should show fewer generic/low-payoff dense windows and more standalone moments with clear setup and payoff.

### Review Notes

- Current weakness is not one broken stage. The broad candidate search reaches enough windows, but Proposal and Ranker do not yet rank the truly promising windows strongly enough.
- The last analyzed job proves Judge lift exists: several final candidates moved up from weak pre-Judge ranks, but their Proposal scores were still very low.
- The most important first implementation target is diagnostics/label flow, because changing thresholds before the system learns from real outcomes would only hide the root cause.
- Implemented the first two quality levers: funnel reports now split `judged_quality_kept` and `unjudged_quality_kept`, and transition retention uses intersection counts so non-nested stages cannot report impossible retention above 100%.
- Clip-Judge quality gate rows now become durable training feedback only when a real judge result exists. Accepted judged rows export as weak positives and rejected judged rows export as judge-feedback negatives for both Proposal and Ranker training paths.
- Unjudged quality passes are no longer treated as positive Judge labels for Proposal training.
- Verification passed: candidate recall tests, feedback-store/training-export/proposal-training tests, `py_compile`, `git diff --check`, and Graphify rebuild through the local Graphify interpreter.
- Implemented the remaining serving-quality levers: Proposal input candidates are boundary-normalized before Proposal scoring, dense windows keep more setup context, and starts that begin mid-thought are repaired to a cleaner transcript boundary before Ranker/Judge.
- Added Proposal/Ranker lift diagnostics to the candidate funnel: positive-proxy precision/recall@K, final/high-Judge Proposal score distributions, high-Judge Proposal keep ratio, source-specific quality breakdown, and high-Judge pre-Judge top-20/top-40 ratios.
- `/api/jobs/{job_id}` and the frontend Pipeline overview now expose Judge quality split, Proposal lift, and Ranker lift so weak jobs can be diagnosed without reading raw JSONL.
- Added `backend/ml/evaluate_candidate_funnel.py` for current-vs-baseline debug-dir comparisons of final selections, Judge lift, Proposal lift, Ranker lift, and missed high-Judge candidates.
- The Clip Judge prompt now scores viral short-form potential more strictly: immediate scroll-stop, Jynxzi-style reaction/quote/payoff, pacing/replay value, and standalone clarity are weighted above generic completeness.
- Ranker training now gives explicit Judge-feedback rows a small supervision boost so future artifacts can learn from judged accepts/rejects instead of treating them like generic weak labels.
- Still not done automatically: the Proposal/Ranker artifacts must be retrained before these training-signal changes affect production scores.

## 2026-05-18 Plan: Analyze Latest Job Results

- [x] Check whether current job artifacts still exist after storage cleanup.
- [x] Inspect backend logs/artifacts for the most recent completed frontend job.
- [x] Summarize result quality, candidate funnel, and any missing evidence.

### Review Notes

- Latest artifact-backed job is `d6bfbbe2-605f-40bd-8c48-7c77f72d74de` for Twitch VOD `2774602304`; it completed with 5 clips.
- Source cleanup worked for this job: `video.mp4`, `audio.wav`, and `chunks/` are absent while clips/debug artifacts remain.
- Candidate funnel shows 5,000 dense inventory windows plus heuristic sources, 357 proposal inputs, 240 proposal-kept candidates, 268 pre-judge ranked candidates, 80 judged candidates, 12 final-ranked candidates, and 5 final moments.
- Clip Judge ran through Ollama successfully but took about 2,184 seconds for 80 candidates.
- Funnel diagnostics still need a semantic cleanup: `quality_kept=195` includes 188 unjudged candidates, so it is misleading as a post-Judge quality metric.


## 2026-05-18 Plan: Delete Serving VOD After Job

- [x] Inspect where frontend URL/upload jobs store the source VOD, derived audio, clips, and debug artifacts.
- [x] Delete the source VOD after pipeline completion/failure without removing output clips, debug diagnostics, feedback material, or training caches.
- [x] Add focused tests and run syntax/Graphify verification for the backend change.

### Review Notes

- Serving/frontend jobs now delete their job-local `video.mp4` after the pipeline
  finishes, fails, or is cancelled. `audio.wav` and `chunks/` are also removed
  as transient source artifacts.
- Output clips and debug artifacts are preserved because only the job-local
  source media paths are removed, not the whole job directory.
- Training/backfill jobs with durable cache semantics are excluded:
  `twitch_vod*`, `proposal_training_*`, and `judge_backfill_*` keep their
  source media.
- The router parent process also runs the same cleanup after the isolated
  pipeline subprocess exits, so SIGKILL/OOM worker failures still remove
  `video.mp4`.
- Verification passed: `py_compile`, focused pipeline cleanup tests, focused
  router subprocess cleanup tests, `git diff --check`, and Graphify rebuild via
  the local Graphify interpreter after system `python3` lacked `graphify`.

## 2026-05-18 Plan: Start ngrok By Default

- [x] Inspect the container startup script and current remote-mode default.
- [x] Make ngrok the default remote mode while keeping an explicit local-only opt-out.
- [x] Verify script syntax and document the operational behavior.

### Review Notes

- `scripts/start_containerized.sh` now defaults `AI_CLIPPING_REMOTE` to `ngrok`, so running `./scripts/start_containerized.sh` starts the ngrok tunnel automatically after the containers come up.
- Local-only startup is still explicit and available with `AI_CLIPPING_REMOTE=none ./scripts/start_containerized.sh`.
- Verification passed: `bash -n scripts/start_containerized.sh`.

## 2026-05-18 Plan: Candidate Filtering Observability

- [x] Map the exact serving funnel from candidate generation through Proposal, Ranker, Judge, Quality Gate, final selection, and frontend display.
- [x] Extend the existing candidate recall report into a per-candidate trace with stable candidate ids, source, time window, scores, pass/fail status, and rejection reason at every stage.
- [x] Add aggregate funnel metrics for each transition: input count, kept count, loss count, loss percentage, top rejection reasons, score distributions, and source mix changes.
- [x] Add “lost good candidates” analysis by comparing final/high-ranker clips against earlier stages to identify whether quality loss starts in Proposal, Ranker, Judge, Quality Gate, or final diversity trimming.
- [ ] Surface the funnel in the frontend as an inspectable table/timeline, not only summary cards: stage columns, candidate rows, filters for rejected/kept/source/reason, and direct links to preview the candidate window.
- [x] Add debug artifacts and tests so every pipeline run can prove which stage removed each candidate and why.
- [ ] Verify with one known weak frontend job and one known good job, then document the bottleneck and next quality lever.

### Review Notes

- The backend now writes `debug/candidate_trace.jsonl` with one row per
  candidate across generation, Proposal, Ranker, Judge scope, Clip Judge,
  Quality Gate, final ranked, and final moment stages.
- The backend also writes `debug/candidate_funnel_summary.json` with stage
  counts, losses, drop-stage counts, rejection-reason counts, source counts,
  score distributions, and trace samples for final and rejected candidates.
- `/api/jobs/{job_id}` now exposes the consolidated funnel as
  `pipeline_diagnostics.candidate_funnel`, so the frontend no longer needs to
  infer the funnel from separate debug artifacts.
- The frontend Pipeline overview now includes a Candidate Funnel section with
  stage counts, largest losses, rejection/drop reasons, and a candidate sample
  table showing time window, source, rank, Proposal/Ranker/Judge scores, and
  drop reason. Dedicated preview links/interactive filters are still the next
  UI refinement.
- Verification passed: backend syntax check, focused candidate trace tests,
  focused jobs-router diagnostics test, frontend lint, `git diff --check`, and
  Graphify rebuild via CLI fallback because the direct Python module import is
  unavailable in this shell.
- Remaining validation is empirical: run one known weak frontend job and one
  good job, then use the new funnel to identify the actual quality bottleneck.

## 2026-05-15 Plan: Align Frontend Whisper Runtime With Local Training

- [x] Compare `scripts/train_local_docker.sh` and `ml.local_train` runtime defaults against the containerized frontend/backend path.
- [x] Apply the matching Whisper/CUDA/thread/shared-memory defaults to serving without increasing concurrent Whisper model count.
- [x] Add/adjust tests for the pipeline subprocess environment.
- [x] Verify backend tests, Graphify rebuild, container runtime settings, and document results.

### Review Notes

- `scripts/train_local_docker.sh` optimizes Whisper mainly through CUDA
  exposure, host IPC, 16 GB shared memory, fixed English transcription, CUDA
  allocator settings, and native thread caps.
- The containerized frontend/backend path now uses the same serving defaults:
  `WHISPER_LANGUAGE=en`, CUDA allocator defaults when `WHISPER_DEVICE=cuda`,
  `CUDA_VISIBLE_DEVICES=0`, tokenizers disabled, and BLAS/OpenMP thread pools
  capped at one thread.
- `docker-compose.yml` now gives the backend `ipc: host` and `shm_size: 16gb`;
  `docker-compose.cuda.yml` carries the CUDA-specific runtime defaults.
- This does not increase Whisper model concurrency. It keeps the persistent
  isolated chunk worker from the previous optimization and removes avoidable
  per-chunk language detection/runtime overhead.
- Verification passed: focused backend tests, `py_compile`, `git diff --check`,
  compose config inspection, Graphify rebuild, backend image rebuild, API
  healthcheck, and live container runtime checks. Runtime reports
  `whisper_device=cuda`, `whisper_language=en`, CUDA available, 16 GB shared
  memory, and the expected thread/CUDA environment variables.

## 2026-05-15 Plan: Optimize Frontend Whisper Transcription

- [x] Confirm current transcription bottleneck and relevant runtime defaults.
- [x] Replace per-chunk isolated Whisper subprocesses with a persistent isolated chunk worker.
- [x] Preserve chunk cache/progress semantics and parent-process isolation.
- [x] Add focused transcriber tests for persistent isolated processing.
- [x] Verify backend tests, Graphify rebuild, container runtime config, and document results.

### Review Notes

- The slow path was `PIPELINE_ISOLATE_TRANSCRIPTION_CHUNKS=true` spawning a
  separate Python process per audio chunk, which reloaded Whisper for every
  chunk.
- `transcribe_with_progress` now keeps isolation but sends all uncached chunks
  to one persistent isolated worker via `--transcribe-chunks`, so Whisper loads
  once per VOD transcription batch instead of once per chunk.
- Cached chunk transcripts still short-circuit before model loading, progress
  updates are still emitted per chunk, and non-isolated transcription keeps the
  existing in-process path.
- Verification passed: transcriber `py_compile`, focused transcriber tests,
  pipeline/config tests, `git diff --check`, Graphify rebuild, and
  containerized runtime checks.
- Backend was rebuilt and restarted. Runtime reports
  `pipeline_isolate_transcription_chunks=True`, `whisper_device=cuda`,
  `frontend_job_cache_enabled=False`, and `shortlist_judge_top_k=80`.

## 2026-05-14 Plan: Speed Up Clip Judge

- [x] Inspect current Clip Judge provider flow and config/test coverage.
- [x] Add Ollama batch judging so multiple candidates can be scored per LLM call.
- [x] Keep non-Ollama providers on the existing single-candidate path.
- [x] Add tests for batch judging and fallback behavior.
- [x] Verify backend tests, frontend build hygiene if types change, Graphify rebuild, and container runtime.

### Review Notes

- Added `CLIP_JUDGE_BATCH_SIZE=4` as the local/default Ollama Judge batch
  size.
- The Ollama text Judge now scores consecutive recall-passing candidates in
  one strict JSON batch request. With `SHORTLIST_JUDGE_TOP_K=80`, this reduces
  the worst-case Judge calls from about 80 to about 20 before recall filtering.
- Non-Ollama providers and multimodal/keyframe Judge paths stay on the
  existing single-candidate path.
- If a batch call fails or omits a candidate result, that candidate falls back
  to the existing single-candidate `judge_candidate` path instead of being
  silently dropped.
- Verification passed: backend `py_compile`, focused Clip-Judge/config tests,
  jobs-router/pipeline/recall-report tests, frontend lint/build,
  `git diff --check`, Graphify rebuild, and containerized runtime checks.
- Containers were rebuilt and restarted. Backend is healthy, frontend/API
  respond, and runtime settings report `judge_provider=ollama`,
  `clip_judge_batch_size=4`, and `clip_judge_timeout_sec=0.0`.

## 2026-05-14 Plan: Add Candidate Recall Debug Report

- [x] Inspect current debug artifact schema for candidate, proposal, judge, and quality-gate stages.
- [x] Add a per-job recall report that tracks candidate counts, source mix, and losses across pipeline stages.
- [x] Surface the recall report in backend pipeline diagnostics so the frontend can show where candidates drop.
- [x] Verify backend tests, frontend build hygiene, Graphify rebuild, and container runtime if needed.

### Review Notes

- Added `services/candidate_recall_report.py` to build a per-job funnel report
  from generated candidates, proposal audit rows, ranker output, judge scope,
  quality-gate audit rows, final ranked candidates, and final moments.
- Pipeline now writes `debug/candidate_recall_report.json` after final moments
  are selected. The report includes stage counts, source counts, transition
  losses, largest losses, top Proposal rejects, candidates outside Judge scope,
  Quality-Gate rejects, and final candidate ids.
- `/api/jobs/<job_id>` now exposes a `Candidate Recall` diagnostics stage so
  the frontend can show whether candidates dropped at Proposal, Ranker/Judge,
  Quality Gate, or final selection.
- Frontend `JobStatus` type now includes the `judging` status used by the
  Clip-Judge progress phase.
- Verification passed: backend `py_compile`, focused pytest suite
  (`test_candidate_recall_report`, `test_jobs_router`,
  `test_pipeline_integration`, `test_config`), frontend lint/build,
  `git diff --check`, Graphify rebuild, and containerized runtime import/API
  checks.
- Containers were rebuilt and restarted; backend is healthy and frontend/API
  respond.

## 2026-05-14 Plan: Remove Clip Judge Stage Timeout And Add Live Progress

- [x] Inspect current Clip Judge timeout path and frontend diagnostics flow.
- [x] Disable the whole-stage Clip Judge timeout so the Judge can finish instead of falling back.
- [x] Persist per-candidate Clip Judge progress and surface it in job status diagnostics.
- [x] Add frontend status label for the judging phase.
- [x] Verify focused backend/frontend tests and update running containers.

### Review Notes

- Removed the whole-stage Clip Judge timeout from the serving pipeline. The
  Judge no longer falls back to pre-judge ranking just because the full stage
  takes longer than 90 seconds.
- `CLIP_JUDGE_TIMEOUT_SEC=0` is now the local default, so provider request
  timeout is disabled unless explicitly configured otherwise.
- Added live Clip Judge progress: processed, attempted, judged,
  recall-filtered, elapsed seconds, and current candidate window are persisted
  to `debug/clip_judge_stage.json` while judging runs.
- Job status now has a dedicated `judging` phase and advances progress during
  candidate judging; the frontend stage list shows `Clip Judge`.
- The frontend diagnostics panel now summarizes Clip Judge progress while the
  job is still running instead of only showing the final judge result.
- Verification passed: backend `py_compile`, focused pytest suite
  (`test_clip_judge`, `test_jobs_router`, `test_pipeline_integration`,
  `test_config`), frontend lint/build, `git diff --check`, Graphify rebuild,
  and containerized runtime checks.
- Containers were rebuilt and restarted. Runtime config now reports
  `clip_judge_timeout_sec=0.0` and `judge_provider=ollama`; frontend and backend
  endpoints respond.

## 2026-05-14 Plan: Keep Backend Diagnostics Visible Before Debug Artifacts

- [x] Confirm why the frontend hides backend diagnostics after cache cleanup.
- [x] Show the backend diagnostics summary even when detailed debug stages are not available yet.
- [x] Verify frontend lint/build and document the behavior.

### Review Notes

- Root cause: the backend did return `pipeline_diagnostics` during early phases,
  but with `available=false` before `debug/` artifacts existed. The frontend
  returned `null` for that state, so the whole backend diagnostics panel
  disappeared while the job was still downloading/transcribing.
- `PipelineDiagnosticsPanel` now renders the backend summary even before
  detailed stages exist, with a short note that detailed candidate/ranker/judge
  data appears after those artifacts are written.
- The current job had no `debug/` directory yet because it was still in
  `transcribing`; that is expected after disabling transcript cache reuse.
- During the frontend image rebuild, Compose also recreated the backend
  container and killed the active worker. I marked job
  `8840d552-24ab-4362-b46a-77591479f00e` as failed so the UI does not wait
  forever.
- Verification passed: `npm --prefix frontend run lint`,
  `npm --prefix frontend run build`, live `/api/jobs/...` inspection, and
  frontend/backend containers are healthy.

## 2026-05-14 Plan: Disable Frontend Job Caches And Purge Old Job Artifacts

- [x] Map frontend-serving cache write paths before deleting anything.
- [x] Add a serving setting that disables frontend download/transcript cache reads and writes.
- [x] Keep training/cross-platform caches untouched unless explicitly requested.
- [x] Delete existing frontend job cache/artifact directories and job metadata.
- [x] Verify tests, runtime config, and storage cleanup.

### Review Notes

- Added `FRONTEND_JOB_CACHE_ENABLED=false` as the serving default in backend
  settings, local `.env`, and `.env.example`.
- Normal frontend/serving job IDs no longer use the Twitch VOD download cache
  even when `TWITCH_DOWNLOAD_CACHE_TTL_SEC=-1`; training-style job IDs such as
  `twitch_vod*` can still use the durable training cache path.
- Frontend pipeline transcription no longer reads or writes
  `backend/tmp/storage/_transcript_cache` when serving cache is disabled; it
  transcribes the current job input directly.
- Deleted old frontend job artifacts and caches from `backend/tmp/storage`,
  including UUID job directories, `jobs.json`, `_transcript_cache`,
  `_download_cache/twitch`, `vision_language_features`, and old materialized
  proposal job debug material. About `90.1 GiB` was removed.
- Preserved non-frontend training/repost caches under `_download_cache`
  (`short_form`, `short_form_metadata`, `short_form_transcripts`) and
  `backend/tmp/vod_feature_cache`.
- Verification passed: `py_compile`, `backend/tests/test_downloader.py`,
  `backend/tests/test_pipeline_integration.py`, `backend/tests/test_config.py`,
  scoped `git diff --check`, runtime settings inspection, storage inventory,
  and Graphify rebuild (`3312` nodes, `8869` edges, `94` communities).

## 2026-05-14 Plan: Fix Clip Judge Timeout Observability And Load

- [x] Confirm whether the frontend warning is a real Judge miss or a diagnostics false positive.
- [x] Persist Clip-Judge stage outcome/degradation details into job debug artifacts.
- [x] Reduce default local Ollama Judge load so serving judges the shortlist instead of every candidate.
- [x] Verify tests, runtime config, and live diagnostics.

### Review Notes

- The warning was real: the latest test job `bec13437-9dac-48b9-94d0-6457c6053d09`
  logged `clip_judge_stage_degraded` because the Clip Judge hit the 90s timeout
  and fell back to 248 pre-judge ranked candidates.
- Root cause was local serving load: `JUDGE_ALL_CANDIDATES=true` plus
  `SHORTLIST_JUDGE_TOP_K=80` meant Ollama could be asked to judge far more than
  the useful shortlist within the timeout.
- Serving defaults now judge only the shortlist:
  `JUDGE_ALL_CANDIDATES=false` and `SHORTLIST_JUDGE_TOP_K=24` in config,
  local `.env`, and `.env.example`.
- Pipeline runs now persist `debug/clip_judge_stage.json`, so the frontend can
  distinguish timeout, disabled judge, empty judge result, and provider failure
  instead of only showing a generic "no judge result" warning.
- Verification passed: `py_compile`, focused jobs-router tests, config and
  pipeline integration tests, and scoped `git diff --check`.
- Graphify rebuilt successfully after backend edits: `3307` nodes, `8858`
  edges, `96` communities.
- Running backend container was recreated with the new env and patched with the
  updated backend files; runtime settings now report `judge_provider=ollama`,
  `judge_all_candidates=False`, and `shortlist_judge_top_k=24`.

## 2026-05-14 Plan: Add Frontend Pipeline Observability

- [x] Inspect current backend debug artifacts and job-status payload shape.
- [x] Add a backend per-job pipeline diagnostics summary from existing debug files.
- [x] Add frontend types and a compact diagnostics panel for candidates, proposal, ranker, judge, quality gate, and final clips.
- [x] Verify backend tests, frontend lint/build hygiene, Graphify rebuild after backend edits, and runtime API shape.
- [x] Document review notes with what the UI can and cannot prove.

### Review Notes

- Added `pipeline_diagnostics` to the job-status payload. It is derived from
  the existing per-job debug artifacts under `backend/tmp/storage/<job_id>/debug`
  and does not add a second tracking system.
- The diagnostics summarize candidate generation, proposal scoring, ranker,
  clip judge, serving quality gate, and final selection with status, counts,
  and warning details.
- The frontend now shows a `Pipeline Überblick` panel while a job is polling
  and preserves the latest panel above the results after completion.
- Older completed jobs can show `disabled` or `warning` for stages whose audit
  files did not exist when those jobs were produced. That is intentional:
  the panel reports what can be proven from artifacts instead of inventing a
  green status.
- Verification passed: backend `py_compile`, focused
  `backend/tests/test_jobs_router.py`, `npm --prefix frontend run lint`,
  `npm --prefix frontend run build`, `git diff --check`, Graphify rebuild, and
  live `/api/jobs/<job_id>` runtime check from the containerized backend.

## 2026-05-14 Plan: Force Clip Judge To Ollama

- [x] Add an explicit Judge provider setting and thread it through the pipeline.
- [x] Set local and example Docker env defaults to use Ollama for Clip Judge.
- [x] Verify tests, rebuild Graphify after backend edits, rebuild backend container, and confirm runtime settings.

### Review Notes

- Added `JUDGE_PROVIDER` to backend settings and set `JUDGE_PROVIDER=ollama`
  in both `.env.example` and the local Docker `.env`.
- The pipeline now passes the configured provider into `clip_judge.rerank_candidates`.
  With `JUDGE_PROVIDER=ollama`, the Anthropic two-tier path is disabled even if
  an Anthropic key is still present.
- When Ollama is forced, the pipeline skips unnecessary multimodal keyframe
  sampling for cloud multimodal providers and uses the text-only Ollama judge.
- `scripts/start_containerized.sh` now auto-enables `docker-compose.ollama.yml`
  when `backend/.env` contains `JUDGE_PROVIDER=ollama`.
- `docker-compose.ollama.yml` now points the backend at `http://ollama:11434`
  inside the Compose network.
- Verification passed: `py_compile`, focused Ollama provider test, full
  `backend/tests/test_clip_judge.py`, `backend/tests/test_config.py` plus
  pipeline integration tests, `bash -n`, Compose config rendering,
  `git diff --check`, and Graphify rebuild.
- Runtime verification passed: backend is healthy, Ollama container is running,
  backend reports `judge_provider=ollama`, `ollama_host=http://ollama:11434`,
  and the backend can reach Ollama with model `qwen2.5:14b` available.

## 2026-05-14 Plan: Make Judge Timeout Non-Fatal In Serving

- [x] Inspect the current Judge timeout failure and fail-fast configuration.
- [x] Change pipeline timeout handling so pre-judge ranked candidates continue to serving.
- [x] Set serving Docker defaults so Judge provider failures are not fail-fast by default.
- [x] Add focused regression coverage for timeout degradation.
- [x] Run tests, rebuild Graphify, and document the result.

### Review Notes

- Root cause: serving jobs already had ranked candidates, but the optional
  Clip Judge stage was wrapped in a hard timeout path that raised when
  `CLIP_JUDGE_FAIL_FAST=true`.
- Judge timeouts now degrade to the pre-judge ranked candidates and log
  `clip_judge_stage_degraded` with the fallback candidate count. Non-timeout
  `ClipJudgeError` still respects the fail-fast setting.
- Serving defaults are now non-fatal: `CLIP_JUDGE_FAIL_FAST=false` and
  `JUDGE_PROVIDER_FALLBACK_ENABLED=true` in both `.env.example` and the local
  `.env` used by Docker.
- Verification passed: `py_compile` for `backend/pipeline.py` and
  `backend/config.py`, focused timeout regression test, full
  `backend/tests/test_pipeline_integration.py`, `backend/tests/test_config.py`
  plus pipeline integration tests, `git diff --check`, and Graphify rebuild.
- Rebuilt/restarted the backend Docker container with CUDA compose overrides.
  Runtime verification passed: `/api/status` responds and the container reports
  `clip_judge_fail_fast=False`, `judge_provider_fallback_enabled=True`, and
  `clip_judge_timeout_sec=90.0`.
- Follow-up verification caught and fixed a keyword typo in the new fallback
  helper call sites (`timeout` -> `timeout_sec`). The backend image was rebuilt
  again and `/api/status` still responds.

## 2026-05-14 Plan: Fix Ngrok Failed Fetch

- [x] Reproduce local and ngrok API reachability from the current running containers.
- [x] Inspect frontend fetch configuration for direct localhost/backend URLs that break remotely.
- [x] Make browser API/media requests use same-origin by default.
- [x] Add focused verification for remote API/media fetch behavior.
- [x] Document the root cause and verification result.

### Review Notes

- Root cause: the current remote/ngrok container was built with browser API
  calls targeting `http://localhost:8000`. On another PC, that means the remote
  client tries to fetch its own local machine, which surfaces as `failed to
  fetch`.
- There was also one ngrok log entry with `connect: connection refused` while
  the frontend port was down during a restart, but the persistent remote-fetch
  problem was the browser-side API base.
- `frontend/lib/api.ts` now normalizes an unset or empty `NEXT_PUBLIC_API_URL`
  to same-origin requests, so API calls use `/api/...` and clip URLs use
  `/static/...` through the Next.js rewrite.
- Docker defaults now build the frontend with `NEXT_PUBLIC_API_URL=` rather
  than `http://localhost:8000`.
- Remote was rebuilt/restarted with `AI_CLIPPING_REMOTE=ngrok
  ./scripts/start_containerized.sh`.
- Verification passed: local `/api/status`, remote
  `https://catfight-preformed-entertain.ngrok-free.dev/api/status`, remote
  frontend HTML, remote JS chunk, no `localhost:8000` in the frontend
  container bundle, no `localhost:8000` in the fetched remote page chunk,
  `node` frontend config tests, `npm --prefix frontend run lint`, and
  `git diff --check`.

## 2026-05-14 Plan: Wire Proposal Training Materialization Into Script

- [x] Add a materializer that creates proposal debug folders from cached VOD transcripts with cross-platform positives.
- [x] Add a proposal-training pipeline entrypoint that runs materialization, row building, then model training.
- [x] Wire `scripts/train_proposal_local_docker.sh` to the new pipeline entrypoint.
- [x] Add focused tests for materialization/orchestration behavior.
- [x] Verify syntax, tests, script help, diff hygiene, and rebuild Graphify after backend edits.

### Review Notes

- `scripts/train_proposal_local_docker.sh` now invokes
  `ml.train_proposal_pipeline` instead of calling the scorer trainer directly.
- The pipeline first materializes cached positive VOD transcripts into isolated
  proposal debug jobs under `<storage-dir>/_proposal_training_materialized`,
  then builds serving-shape proposal rows, then trains the logistic proposal
  scorer.
- The materializer uses compatible VOD transcript caches first, then safely
  falls back to the latest transcript cache for the same VOD ID when only the
  Whisper runtime metadata differs.
- A local non-destructive verification run against two real Jynxzi positive
  VODs materialized 2 VODs and built 34 proposal rows with 1 positive and 33
  negatives using reduced candidate limits.
- Verification passed: `py_compile`, `bash -n`, script help rendering, focused
  proposal tests (`11 passed`), `git diff --check`, and Graphify rebuild
  (`3294` nodes, `8818` edges, `95` communities).

## 2026-05-14 Plan: Fix Ngrok Clip Playback

- [x] Confirm why remote job status works but returned clip media does not render.
- [x] Route generated clip media through the same ngrok/Next.js origin.
- [x] Add focused frontend config verification for `/static` rewrites.
- [x] Rebuild/restart frontend in remote mode and verify existing clip media over ngrok.
- [x] Rebuild Graphify if backend files change and document results.

### Review Notes

- Root cause: remote mode only rewrote `/api/:path*` from the frontend/ngrok
  origin to the backend. Job polling succeeded, but clip files use
  `/static/{job_id}/clips/{filename}`, so Next.js returned `404` for generated
  media over ngrok.
- `frontend/next.config.mjs` now rewrites both `/api/:path*` and
  `/static/:path*` to the internal backend service.
- Added `frontend/tests/next-config-rewrites.mjs` to lock the API and static
  media rewrites.
- Rebuilt/restarted with `AI_CLIPPING_REMOTE=ngrok ./scripts/start_containerized.sh`.
- Verified the previously failing clip path:
  `https://catfight-preformed-entertain.ngrok-free.dev/static/7141a1f5-2231-4c16-aff0-efca5908c448/clips/clip_01.mp4`
  now returns `200` for full GET and `206` with `content-type: video/mp4` for
  range requests.
- Verification passed: `node --check frontend/next.config.mjs`,
  `node frontend/tests/next-config-rewrites.mjs`,
  `npm --prefix frontend run lint`, frontend rebuild, local `/static` route,
  remote `/static` route, and remote clip byte download.
- No backend code changed, so Graphify rebuild was not required.

## 2026-05-14 Plan: Add Optional Ngrok Remote Startup

- [x] Replace the temporary reverse-proxy setup with a versioned Next.js API rewrite.
- [x] Add an explicit remote mode to `scripts/start_containerized.sh`.
- [x] Preserve the default local-only startup behavior.
- [x] Verify local mode, remote mode, ngrok URL extraction, and frontend API base.
- [x] Document how to stop the remote tunnel.

### Review Notes

- `AI_CLIPPING_REMOTE=ngrok ./scripts/start_containerized.sh` now starts the
  normal containers, rebuilds the frontend with same-origin API calls, and then
  starts one ngrok tunnel against frontend port `3000`.
- Next.js rewrites `/api/:path*` to the backend service
  `http://backend:8000/api/:path*`, so no extra Python reverse proxy is needed.
- Default local startup remains unchanged: without `AI_CLIPPING_REMOTE=ngrok`,
  the frontend still builds with `http://localhost:8000` as the browser API
  base.
- Runtime verification passed with CUDA auto-selection: backend/frontend
  containers are running, local `http://127.0.0.1:3000/api/status` returns
  backend JSON through the frontend rewrite, and remote
  `https://catfight-preformed-entertain.ngrok-free.dev/api/status` returns the
  same backend status through ngrok.
- Frontend bundle check found no `http://localhost:8000` reference after the
  remote-mode build.
- Verification passed: `bash -n scripts/start_containerized.sh`,
  `node --check frontend/next.config.mjs`, Compose config rendering,
  `npm --prefix frontend run lint`, scoped `git diff --check`, and a real
  `AI_CLIPPING_REMOTE=ngrok ./scripts/start_containerized.sh` run.
- Active tunnel PID/log are `backend/tmp/ngrok/ngrok.pid` and
  `backend/tmp/ngrok/ngrok.log`. Stop command:
  `kill "$(cat backend/tmp/ngrok/ngrok.pid)"`.

## 2026-05-14 Plan: Test VOD 2771207666 Through Frontend

- [x] Confirm frontend/backend containers and the frontend submission path.
- [x] Submit `https://www.twitch.tv/videos/2771207666` through the frontend-equivalent submit path.
- [x] Watch job status, backend logs, and cgroup OOM state until the next bottleneck.
- [x] Re-plan and remove the full-audio extraction requirement when transcript chunks are already cached.
- [x] Re-run the frontend-equivalent VOD test and record whether the fix holds.

### Review Notes

- Frontend and backend containers were running; frontend returned HTTP `200`,
  backend `/api/status` returned active pipeline providers.
- Job submitted with the same frontend JSON contract plus `profile=jynxzi`:
  `2ee4f409-2bbe-4331-a5ff-60a10b70de76`.
- First frontend-equivalent run exposed a second root cause before transcription: the pipeline still extracts a full `audio.wav` from the 6.4G VOD even when all transcript chunks are already cached. This drove cgroup memory/current above 8 GB and incremented `oom_kill` from 0 to 1 while `ffmpeg` was still extracting audio.
- The pipeline now probes VOD duration with `ffprobe`, verifies that all
  expected transcript chunk-cache rows exist, and skips `audio.wav` extraction,
  Whisper, and audio-feature extraction on a complete transcript cache hit.
- A second frontend-equivalent retry confirmed the transcript/audio fix, then
  exposed the next memory path: full-VOD visual extraction decoded one frame per
  second from a 4h38m VOD and grew worker RSS beyond 8 GB. The pipeline now
  skips full-VOD visual feature extraction above
  `PIPELINE_FULL_VISUAL_FEATURE_MAX_DURATION_SEC=7200`; candidate-window face
  refinement and keyframe sampling remain enabled.
- A third frontend-equivalent retry confirmed those two skips, then exposed the
  next memory path: shortlist face-reaction refinement decoded one huge
  min-to-max ffmpeg span for sparse candidates. Refinement now decodes only
  contiguous candidate-second ranges instead of hours of non-candidate video.
- Verification so far: `py_compile`, focused tests (`67 passed`), scoped
  `git diff --check`, and Graphify rebuild (`3181` nodes, `8464` edges,
  `96` communities).
- Final frontend-equivalent test job
  `232c93ec-e58c-4231-b30f-24046d4e8a18` completed successfully with
  `status=done`, `progress_pct=100`, `download_cached=true`, `clips=5`, and
  `oom_kill=0`. Peak observed memory stayed around `3.3 GB`, then dropped to
  about `0.5 GB` after completion. The job output contains five clip files
  under `backend/tmp/storage/232c93ec-e58c-4231-b30f-24046d4e8a18/clips/`.
## 2026-05-14 Plan: Fix Remaining Frontend VOD Transcriber SIGKILL

- [x] Re-check latest failed job state instead of assuming the previous OOM pattern.
- [x] Determine whether SIGKILL hits the parent pipeline worker or the isolated Whisper chunk worker.
- [x] Identify the actual memory trigger for VOD `2771207666`.
- [x] Implement the smallest durable fix and update regression coverage.
- [x] Verify in tests/container and document the result.

### Review Notes

- Latest failed job is `9fe48cd9-e8fc-4d51-a849-1a52cd92c5eb`, now failing
  at `progress_pct=30` with `download_cached=True`. That points to the start
  of transcription or the first isolated chunk, not to later feature extraction.
- The latest job had all 28 Whisper chunk cache files written under
  `_transcript_cache`, so ASR itself completed. The subsequent log shows
  `transcriber_diarization_unavailable ... AudioDecoder`, followed by cgroup
  `oom_kill=1`, which means post-ASR Pyannote/TorchCodec diarization was the
  remaining parent-worker memory trigger.
- Frontend pipeline jobs now pass `enable_diarization=False` by default and skip
  the MFCC fallback unless `PIPELINE_ENABLE_DIARIZATION=true` is explicitly set.
- Verification passed: `py_compile`, focused backend tests (`46 passed`),
  scoped `git diff --check`, and Graphify rebuild (`3163` nodes, `8421`
  edges, `95` communities).
- Runtime verification passed after rebuilding the backend CUDA image:
  container is `healthy`, `OOMKilled=false`, cgroup `oom_kill=0`, CUDA is
  visible, and `pipeline_enable_diarization=False` inside the container.
- Direct container proof for VOD `2771207666`: loading the existing transcript
  chunk cache with `enable_diarization=False` returned `3540` segments, while a
  patched `_diarize` function that would raise was never called.

## 2026-05-14 Plan: Clean Backend Tmp Runtime Altlasten

- [x] Inspect `backend/tmp` size distribution and current Docker/backend process state.
- [x] Identify persistent caches that must be preserved.
- [x] Stop stale cancelled/failed runtime workers that still hold temp files.
- [x] Delete only stale per-job runtime directories and orphan temp fragments.
- [x] Verify freed disk space and document what was preserved.

### Review Notes

- Persistent caches to preserve: `backend/tmp/vod_feature_cache`,
  `backend/tmp/storage/_download_cache`, `backend/tmp/storage/_transcript_cache`,
  `backend/tmp/cache`, and short-form metadata/cache directories.
- Current largest safe cleanup candidates are failed/cancelled per-job runtime
  directories under `backend/tmp/storage/<uuid>`.
- `ffeabed2-c22a-4cc9-9cb5-053da047ff89` is marked `cancelled` in
  `jobs.json`, but still has a live `pipeline_worker` and `aria2c` process,
  so it must be stopped before deleting its partial fragments.
- Stopped the stale cancelled worker and removed stale per-job directories for
  failed/cancelled frontend jobs. Removed all leftover `.part`/fragment temp
  files under `backend/tmp`.
- `backend/tmp` dropped from about `165G` to `133G`. Remaining usage is mostly
  intentional cache: `122G` VOD feature cache and `11G` storage cache.
- `backend/ml/cache_cleanup.py --dry-run` reports no additional conservative
  cleanup candidates after the manual stale UUID cleanup.
- Restarted the backend after cleanup to clear defunct child processes from the
  cancelled download. Container health is `healthy`, status is `running`, and
  Docker reports `OOMKilled=false`.

## 2026-05-14 Plan: Isolate Whisper Chunk Transcription From OOM

- [x] Confirm repeated SIGKILL still occurs in the Whisper transcribing phase.
- [x] Add process isolation for each non-cached Whisper chunk in frontend jobs.
- [x] Keep the durable chunk cache and make cached chunks skip Whisper entirely.
- [x] Add focused tests for isolated chunk execution and pipeline wiring.
- [x] Run verification, rebuild Graphify, and restart the backend container.

### Review Notes

- The current failed/active frontend VOD jobs still reach `progress_pct=55`
  with `SIGKILL`, and Docker reports `OOMKilled=true`.
- Logs show the worker repeatedly dies during Whisper chunk processing around
  the high-teen chunks, before feature extraction starts. The previous chunk
  cache preserves completed work, but it does not prevent the long-lived
  Whisper process from accumulating memory within one run.
- Frontend pipeline runs now pass `isolate_chunks=True` by default via
  `PIPELINE_ISOLATE_TRANSCRIPTION_CHUNKS=true`. Cache hits still bypass
  Whisper, while cache misses run one chunk in one child process and then
  return native/PyTorch memory to the OS when that child exits.
- Verification passed: `py_compile`; focused transcriber/pipeline/jobs/config
  tests -> `38 passed`; scoped `git diff --check`; Graphify rebuild (`3162`
  nodes, `8419` edges, `94` communities); CUDA backend image rebuild/restart;
  backend healthcheck healthy with `OOMKilled=false`; runtime probe reports
  `pipeline_isolate_transcription_chunks=True`, `WHISPER_DEVICE=cuda`, and
  CUDA available.

## 2026-05-14 Plan: Make Frontend VOD Transcription Resumable

- [x] Confirm the repeat SIGKILL phase from current job/log state.
- [x] Inspect frontend transcription chunking and existing cache boundaries.
- [x] Add durable per-chunk transcript checkpointing for URL/VOD jobs.
- [x] Release transient Whisper/GPU memory after each chunk.
- [x] Run focused transcriber/pipeline tests, diff hygiene, and Graphify.

### Review Notes

- New failed job `9dec6885-139b-4e51-bfd0-e61d9fa90136` reached
  `progress_pct=55` and was killed while Whisper was still processing VOD
  chunks. The log never reached `extracting_features`, so the previous
  feature-overlap fix did not cover this failure.
- The VOD media itself was reused through hardlinks/cache; the repeated cost is
  the non-resumable Whisper chunk loop.
- Frontend transcription now writes each completed 10-minute Whisper chunk to
  `backend/tmp/storage/_transcript_cache/<source-hash>/chunks/`. The cache key
  is stable for Twitch VOD URLs via `twitch_vod:<vod_id>`, so a fresh frontend
  job for the same VOD can resume completed chunks from an earlier killed job.
- Fully cached chunks do not load Whisper at all; cache misses still use the
  existing singleton model and call CUDA cache cleanup after each completed
  chunk.
- Verification passed: `py_compile`; focused transcriber/pipeline/jobs tests
  -> `33 passed`; scoped `git diff --check`; Graphify rebuild (`3153` nodes,
  `8401` edges, `84` communities); CUDA backend image rebuild/restart;
  backend healthcheck healthy with `OOMKilled=false`; runtime probe reports
  `WHISPER_DEVICE=cuda`, CUDA available, and feature overlap disabled.

## 2026-05-14 Plan: Prevent Frontend VOD Worker SIGKILL/OOM

- [x] Confirm the failed job phase and runtime kill signal from current logs.
- [x] Reduce peak memory by disabling overlapped feature extraction by default.
- [x] Cap frontend Twitch VOD downloads to a serving-safe default height.
- [x] Improve signal-9 job error text and add focused regression coverage.
- [x] Run focused tests, syntax checks, diff hygiene, and Graphify after backend changes.

### Review Notes

- The failed frontend VOD job `c4599af9-4daf-42f0-827f-1000e5d1c677`
  stopped at `progress_pct=55`, right after Whisper progress and before feature
  extraction completed. Docker also reported `OOMKilled=true` for the backend
  container state, so the failure mode was SIGKILL/peak-memory, not a Python
  exception.
- Pipeline workers now default to sequential feature extraction:
  `PIPELINE_OVERLAP_FEATURE_EXTRACTION=false`. Whisper no longer overlaps with
  audio/video feature extraction unless explicitly re-enabled.
- Frontend Twitch VOD downloads now default to `TWITCH_DOWNLOAD_MAX_HEIGHT=720`.
  Training VOD jobs still use `TRAINING_VOD_MAX_HEIGHT`.
- Signal-9 failures now surface as
  `Pipeline subprocess exited from signal 9 (SIGKILL; likely OOM or external kill)`.
- Verification passed: focused pytest suite (`48 passed`), `py_compile`, scoped
  `git diff --check`, Graphify rebuild (`3137` nodes, `8366` edges,
  `87` communities), and CUDA container restart/healthcheck.

## 2026-05-14 Plan: Add CUDA Backend Container With CPU Fallback

- [x] Mirror the training workflow's CUDA runtime for the app backend.
- [x] Keep the current CPU backend image as the fallback path.
- [x] Add a Compose override that requests Docker GPU access and CUDA devices.
- [x] Make the start script auto-select CUDA when available, otherwise CPU.
- [x] Verify Compose configs, build paths, runtime selection, and Graphify.

### Review Notes

- Added a CUDA backend image using the same CUDA family as the local training
  workflow: `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`.
- Added `docker-compose.cuda.yml` as an override for the backend only. It uses
  the CUDA Dockerfile, requests `gpus: all`, and sets
  `WHISPER_DEVICE=cuda`, `SEMANTIC_DEVICE=cuda`,
  `NVIDIA_VISIBLE_DEVICES=all`, and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.
- `scripts/start_containerized.sh` now supports
  `AI_CLIPPING_BACKEND_RUNTIME=auto|cuda|gpu|cpu`. `auto` probes host NVIDIA
  visibility and Docker `--gpus all`; if either is unavailable it starts the
  CPU backend.
- Verified forced CPU startup and automatic CUDA startup on the current host.
  Auto selected CUDA on the RTX 5070 Ti host.
- Live CUDA backend verification passed:
  `torch=2.10.0+cu128`, `torch.version.cuda=12.8`,
  `torch.cuda.is_available()=True`, `WHISPER_DEVICE=cuda`, and
  `SEMANTIC_DEVICE=cuda`.
- Compose config validation, CUDA backend build, CPU fallback build, frontend
  HTTP `200`, backend `/api/status`, backend healthcheck, scoped
  `git diff --check`, and Graphify rebuild passed.

## 2026-05-14 Plan: Containerize Frontend And Backend Runtime

- [x] Confirm frontend/backend runtime commands and env boundaries.
- [x] Add production Dockerfiles for FastAPI backend and Next.js frontend.
- [x] Add Docker Compose wiring with persistent backend runtime/cache volumes.
- [x] Add a start script that builds and starts the stack correctly.
- [x] Verify Compose config/build path and rebuild Graphify after backend-related files change.

### Review Notes

- Added container builds for the FastAPI backend and Next.js frontend, plus
  scoped `.dockerignore` files so local venvs, caches, generated artifacts, and
  secrets are not copied into images.
- `docker-compose.yml` now wires `backend` on port `8000` and `frontend` on
  port `3000`, mounts persistent backend runtime/cache/artifact directories,
  and exposes `host.docker.internal` for host Ollama access.
- `scripts/start_containerized.sh` creates required runtime directories,
  bootstraps missing env files from examples, then runs
  `docker compose -f docker-compose.yml up --build -d`.
- The backend image preinstalls CPU PyTorch wheels before app extras to avoid
  pulling CUDA/NVIDIA packages into the default container image.
- Verification passed: Compose config validation, frontend image build, backend
  image build, start script launch, backend healthcheck, frontend HTTP `200`,
  backend `/api/status`, CPU Torch probe, scoped `git diff --check`, and
  Graphify rebuild.

## 2026-05-14 Plan: Fix Frontend Worker Cache/Headless Runtime Warnings

- [x] Confirm which warning is an actual broken runtime path.
- [x] Make local cache env selection require write access, not just directory existence.
- [x] Set stable headless CV defaults for frontend pipeline workers and face probes.
- [x] Repair current root-owned runtime cache directory.
- [x] Run focused tests, diff hygiene, and Graphify after backend changes.

### Review Notes

- The real broken path was `backend/tmp/matplotlib`: it existed, but was owned
  by `root:root`, so Matplotlib fell back to a temporary `/tmp/matplotlib-*`
  cache on every worker start.
- Local cache setup now probes write access before accepting a cache directory
  and falls back to a writable runtime cache if a directory only exists but is
  not writable.
- Frontend pipeline workers and face-detection probes now set software/headless
  defaults for OpenCV/Mesa: `LIBGL_ALWAYS_SOFTWARE=1`,
  `MESA_LOADER_DRIVER_OVERRIDE=swrast`, and
  `OPENCV_FFMPEG_CAPTURE_OPTIONS=hwaccel;none`.
- The current `backend/tmp/matplotlib` directory ownership was repaired to
  `antonio:antonio`, and direct Matplotlib cache resolution now stays on that
  path.
- Verification passed: `backend/tests/test_config.py` and
  `backend/tests/test_jobs_router.py` -> `22` tests, `py_compile`, scoped
  `git diff --check`, direct Matplotlib cache probe, and Graphify rebuild.
- Graphify now reports `3135` nodes, `8362` edges, and `81` communities.

## 2026-05-14 Plan: Enable Twitch VOD Download Cache And Remove Stale Job Media

- [x] Confirm current cache setting and identify large stale frontend VOD job directories.
- [x] Enable Twitch VOD download cache in config, `.env`, and `.env.example`.
- [x] Delete stale failed/cancelled job media directories and obsolete partial VOD-range downloads.
- [x] Run focused verification, diff hygiene, and Graphify after backend config changes.

### Review Notes

- Twitch VOD download caching is now enabled by default with
  `TWITCH_DOWNLOAD_CACHE_TTL_SEC=-1` in config, `.env`, and `.env.example`.
- The existing downloaded Twitch VOD `2770356205` was migrated into the
  canonical cache at `backend/tmp/storage/_download_cache/twitch/`.
- Removed the stale failed/cancelled frontend job media copies and two obsolete
  coarse VOD `.part` directories. `backend/tmp/storage` dropped from `69G` to
  `26G`.
- Cache usability was verified for `https://www.twitch.tv/videos/2770356205`;
  focused downloader tests passed with `23` tests.
- `py_compile`, scoped `git diff --check`, and Graphify rebuild passed.
  Graphify now reports `3132` nodes, `8354` edges, and `84` communities.

## 2026-05-14 Plan: Re-enable Face Detection For Frontend Pipeline Jobs

- [x] Change the pipeline subprocess default so native face detection is enabled again.
- [x] Preserve an explicit environment override for disabling face detection if it crashes.
- [x] Update regression coverage for the new worker default.
- [x] Run focused verification and rebuild Graphify after backend changes.

### Review Notes

- Frontend pipeline workers now default to
  `AI_CLIPPING_DISABLE_FACE_DETECTION=0`, so MediaPipe/MTCNN/DeepFace can run
  again inside the isolated worker process.
- Explicit host override still works: launching the backend with
  `AI_CLIPPING_DISABLE_FACE_DETECTION=1` keeps native face stacks disabled.
- Verification passed: `backend/tests/test_jobs_router.py` -> `18` tests,
  `py_compile`, scoped `git diff --check`, and Graphify rebuild.
- Graphify rebuilt successfully: `3132` nodes, `8354` edges, `84`
  communities.

## 2026-05-14 Plan: Make Pipeline Worker Strategy JSON-Safe

- [x] Trace why URL submit task failed before starting the worker.
- [x] Normalize strategy payloads before serializing for the subprocess.
- [x] Add regression coverage for set-valued strategy fields.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- `clip_strategy.normalize_strategy()` returns set-valued fields such as
  `keywords` and `exclude_keywords`. The worker-launch command now converts the
  full strategy payload to JSON-safe primitives before `json.dumps`.
- Sets are serialized as sorted lists, so command construction is deterministic
  and the worker receives the same keyword content.
- Verification passed: `py_compile`; focused jobs/job-manager tests ->
  `24 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3113` nodes, `8301` edges, and
  `79` communities.

## 2026-05-14 Plan: Isolate Pipeline URL Submits From Native CV Crashes

- [x] Trace the Twitch URL submit crash path.
- [x] Move pipeline execution out of the FastAPI process.
- [x] Keep job progress/cancellation visible across the subprocess boundary.
- [x] Add regression coverage for subprocess launch/failure behavior.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The crash path was `routers/jobs.py` importing `pipeline` during URL submit;
  `pipeline.py` imports `services.video_features`, which initializes native
  MediaPipe/MTCNN/DeepFace stacks at import time. In a headless EGL/WSL
  context, those native imports can segfault before Python can catch anything.
- URL and upload jobs now launch `pipeline_worker.py` in a separate Python
  subprocess. If the worker exits from a signal, FastAPI marks the job failed
  instead of dying with the worker.
- Worker jobs set a writable `MPLCONFIGDIR` and default
  `AI_CLIPPING_DISABLE_FACE_DETECTION=1`, while preserving motion/scene visual
  features. The env can still override this if native face detection is stable.
- `job_manager` now reloads `jobs.json` when another process updates it, so
  progress, terminal state, and cancellation remain visible across the worker
  boundary.
- Verification passed: `py_compile`; focused jobs/job-manager tests ->
  `23 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3111` nodes, `8297` edges, and
  `86` communities.

## 2026-05-14 Plan: Feed Frontend Ratings Into Training

- [x] Trace how frontend gold-eval ratings are stored and exported.
- [x] Enable frontend-rated clips as default VOD training signal.
- [x] Prevent stale generated gold artifacts from being preserved as training input.
- [x] Update regression coverage for the new default.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- Frontend gold-eval ratings already flow into
  `tmp/storage/feedback/gold_eval_updates.jsonl` and export as
  `gold_eval_frontend` rows with deterministic `gold_holdout_split`.
- The normal Jynxzi VOD training run now enables
  `VOD_RANKER_USE_USER_LABELS=1` and `VOD_RANKER_USE_GOLD_TRAIN=1` by default,
  so frontend-rated train-split rows are passed to `--include-gold-train`.
- The script no longer calls `export_frontend_gold_eval.py` with
  `--merge-existing` or `--preserve-existing-if-empty`; generated gold
  artifacts are rebuilt from the feedback store each run instead of preserving
  stale rows.
- A dry run against the current local feedback store exported `97` Jynxzi
  frontend gold rows: `67` train and `30` holdout.
- Verification passed: `bash -n`; `py_compile`; focused frontend/gold/training
  tests -> `57 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3094` nodes, `8249` edges, and
  `86` communities.

## 2026-05-14 Plan: Balance Auto-Negative Outcome Holdout

- [x] Trace why the outcome holdout has positives but no auto-negative rows.
- [x] Add deterministic holdout-floor balancing for exported auto-outcome negatives.
- [x] Ensure split metadata and dataset_source stay consistent after balancing.
- [x] Add regression coverage and verify current artifacts can produce mixed holdout rows.
- [x] Rebuild Graphify after backend changes.

### Review Notes

- The last run's auto-outcome negative export produced `49` rows but all landed
  in the train split, leaving the outcome holdout with `24` positives and `0`
  negatives. That made threshold calibration skip with
  `calibration_class_rows_below_minimum`.
- Auto-outcome negative export now applies a deterministic holdout floor after
  `max_rows` capping. If the stable split under-fills holdout, selected train
  negatives are promoted until the requested holdout ratio is represented.
- Split promotion updates `gold_holdout_split` and `dataset_source` together,
  so downstream builders see `auto_outcome_negative_holdout` rows correctly.
- A dry run with the current Jynxzi artifacts now exports `58` auto negatives:
  `46` train and `12` holdout. Building the outcome holdout from that output
  produces `36` rows: `24` positives and `12` negatives.
- Threshold calibration on the dry-run holdout now returns `status=computed`,
  `reason=ok`, with both classes present.
- Verification passed: `py_compile`; focused auto-negative + outcome-holdout
  tests -> `9 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3094` nodes, `8249` edges, and
  `86` communities.

## 2026-05-14 Plan: Inventory-Calibrate Rollout Readiness Gates

- [x] Inspect the current rollout-readiness targets in the diagnostic report and ranker artifact.
- [x] Replace impossible fixed Jynxzi rollout gates with inventory-calibrated targets when available VOD groups are capped.
- [x] Keep raw historical targets visible in reports for context.
- [x] Add focused regression coverage and verify against the current Jynxzi artifact.
- [x] Rebuild Graphify after backend changes.

### Review Notes

- Jynxzi rollout readiness now uses inventory-calibrated targets when the
  available VOD groups cannot plausibly satisfy the old absolute rollout
  gates. The raw historical targets remain visible under `raw_targets`.
- With the current maxed-out artifact, the calibrated targets are
  `training_rows=304`, `positive_rows=152`, and `validation_groups=15` for
  `38` available policy-surviving groups.
- Re-running the coverage diagnosis against the current Jynxzi training rows
  now returns `passed=true`: actuals are `367` training rows, `159` positive
  rows, and `15` validation groups.
- Verification passed: `py_compile`; focused diagnose + train-ranker tests ->
  `58 passed`; scoped `git diff --check`.
- Graphify rebuilt successfully with `3089` nodes, `8234` edges, and
  `85` communities.

## 2026-05-14 Plan: Parallelize Transcript Repost Matching

- [x] Add a configurable process-worker count for transcript repost matching.
- [x] Parallelize only the independent per-repost best-match step while keeping
  transcript cache loading and scoring semantics unchanged.
- [x] Preserve deterministic output/diagnostic ordering.
- [x] Add focused regression coverage and run verification.
- [x] Rebuild Graphify after backend changes.

### Review Notes

- Transcript matching now accepts `--transcript-match-workers` in
  `prepare_cross_platform_fingerprint_rows.py` and `--workers` in the direct
  transcript matcher CLI.
- `train_jynxzi_vod.sh` defaults `CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS=4`,
  exports it, logs it at startup, and passes it into the prepare stage.
- The implementation keeps media resolution and Short transcript cache loading
  in the parent process. Only the independent per-repost VOD best-match scan is
  distributed through `ProcessPoolExecutor`.
- Worker results are consumed in input order, so matched rows and diagnostic
  rows remain deterministic relative to the serial path.
- Verification passed: `py_compile`; `bash -n`; focused transcript/script
  tests -> `9 passed`; focused prepare tests -> `47 passed`; scoped
  `git diff --check`.
- Graphify rebuilt successfully with `3085` nodes, `8223` edges, and
  `84` communities.

## 2026-05-13 Plan: Reuse Existing Repost Media During Freshness Rebuild

- [x] Document why the IboClips freshness rebuild is rematerializing many Shorts.
- [x] Reuse existing repost `video_path` rows by canonical source/video identity
  before attempting `yt-dlp` media materialization.
- [x] Add regression coverage for `/shorts/<id>` vs `watch?v=<id>` reuse.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The live run is downloading many Shorts because the curated-channel freshness
  rebuild rewrites the repost manifest from fresh discovery rows. Those rows do
  not naturally carry the previous `video_path`, and the short-form media cache
  is keyed by exact URL, so `https://youtube.com/shorts/<id>` and
  `https://youtube.com/watch?v=<id>` can miss the same already downloaded MP4.
- Freshness rebuilds now pass previous manifest rows into repost
  materialization and reuse existing media paths by canonical repost identity.
  For YouTube rows, that identity is the normalized video id, so Shorts URLs
  and watch URLs for the same clip share the cached media.
- Reused rows skip `_ensure_short_form_media_cached`, so the next run should
  download only genuinely missing selected Shorts rather than re-downloading
  already materialized equivalents.
- Verification passed: `py_compile`; focused
  `test_prepare_cross_platform_fingerprint_rows.py` -> `47 passed`; scoped
  `git diff --check`. Graphify rebuilt successfully with the local Graphify
  interpreter: `3080` nodes, `8211` edges, `82` communities.

## 2026-05-13 Plan: Refresh IboClips Shorts Before Each Run

- [x] Trace the current repost manifest reuse logic and the short-form metadata cache.
- [x] Add a pre-run freshness check so curated YouTube channel sources rebuild
  the repost manifest when new Shorts appear.
- [x] Preserve existing metadata/media cache reuse while fetching only missing
  metadata/downloads needed for the current VOD date window.
- [x] Add focused regression coverage and run verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- Curated YouTube Shorts channel sources such as
  `https://www.youtube.com/@Iboclipss/shorts` now trigger a repost-manifest
  freshness rebuild before matching, even when the previous manifest already
  meets `min_reposts`.
- The rebuild goes through the existing discovery path, so flat channel rows
  are checked again, missing per-Short metadata is cached, date-window filtering
  is reapplied, and only selected Shorts whose media is missing are downloaded.
- Existing short media and metadata remain cache hits; this does not delete or
  redownload the 355 already materialized Shorts.
- If no effective inputs change, the transcript match signature remains
  unchanged and the existing match result can still be reused after the
  manifest refresh.
- Verification passed: `py_compile`; focused
  `test_prepare_cross_platform_fingerprint_rows.py` -> `46 passed`; scoped
  `git diff --check`. Required system `python3 ... graphify.watch...` still
  fails because this interpreter lacks `graphify`; rebuilding with the local
  Graphify interpreter succeeded with `3076` nodes, `8196` edges, and `88`
  communities.

## 2026-05-13 Plan: Lower Transcript Match Threshold To 0.50

- [x] Change the local Jynxzi training default from `0.60` to `0.50`.
- [x] Keep workflow-state reporting and env templates aligned with the new default.
- [x] Run focused verification and rebuild Graphify after backend changes.

### Review Notes

- `CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE` now defaults to `0.50` in the
  Jynxzi VOD training script and in workflow-state reporting.
- `backend/.env` and `backend/.env.example` are aligned to `0.50`, so the
  Docker local-train wrapper will not keep injecting the old `0.60` value.
- The next run's transcript match input signature will change because the
  threshold is part of the signature, so transcript matching should recompute
  once and then be reusable again on unchanged later runs.
- Verification passed: `py_compile`, `bash -n`, focused unittest suite
  `test_training_workflow_state` + `test_train_jynxzi_vod_script` ->
  `12 passed`, and scoped `git diff --check`.
- Graphify rebuilt successfully with `3073` nodes, `8183` edges, and
  `86` communities.

## 2026-05-13 Plan: Reuse Cross-Platform Transcript Matching

- [x] Inspect the current prepare-stage transcript matching inputs, outputs, and summaries.
- [x] Add a deterministic input signature so transcript matching can be reused when inputs/config have not changed.
- [x] Persist and validate the signature in the prepare summary before invoking matching again.
- [x] Add regression coverage and run focused verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- Transcript-mode prepare now computes `match_input_signature` from VOD rows,
  repost rows, media path state, verification mode, threshold, cache versions,
  profile-filter version, and VOD cache root.
- If the current signature matches the previous prepare summary and both output
  and diagnostics still exist, the run reuses the prior match result. It writes
  `reused_match_result=true` and `reuse_reason=match_input_signature_unchanged`
  without calling `build_full_vod_transcript_caches()` or
  `match_reposts_to_vods_with_diagnostics()`.
- The reused result preserves the previous status, including
  `insufficient_*`, so `--fail-on-insufficient` still behaves honestly while
  avoiding duplicate Whisper/transcript matching work.
- The old pre-manifest output reuse is now limited to audio-fingerprint mode;
  transcript mode checks manifests/signature before deciding reuse.
- Verification passed: `py_compile`; focused
  `test_prepare_cross_platform_fingerprint_rows.py` -> `45 passed`; scoped
  `git diff --check`. Required system `python3 ... graphify.watch...` still
  fails because that interpreter lacks `graphify`; rebuilding with the local
  Graphify interpreter succeeded with `3073` nodes, `8183` edges, and `86`
  communities.


## 2026-05-13 Plan: Remove User-Labeled Rows From Jynxzi Training

- [x] Disable frontend/manual/gold label export in the normal VOD training flow.
- [x] Empty generated gold-eval and hard-negative artifacts before training/eval so stale self-labeled rows cannot be reused.
- [x] Stop using active gold-eval queue rows as auto-negative candidates.
- [x] Add script regression coverage and run focused verification.

### Review Notes

- User-labeled rows are now disabled by default with
  `VOD_RANKER_USE_USER_LABELS=0`. Gold train and hard-negative train merges
  require explicitly opting back in with `VOD_RANKER_USE_USER_LABELS=1`.
- The training script resets generated model-facing user-label artifacts before
  training: gold eval rows/datasets/reports, gold-derived hard negatives,
  ranker holdout diagnosis outputs, and rollout review queue artifacts.
- The flow now skips frontend/manual gold export, gold eval, gold-derived
  hard-negative export/diagnosis, and rollout review queue generation while
  user labels are disabled.
- `jynxzi_active_gold_eval_queue.features.jsonl` is no longer passed into
  auto-outcome negative export, so manually reviewed queue rows cannot affect
  negative mining.
- Existing generated model-facing user-label artifacts were reset in-place:
  gold eval JSONL files, gold-derived hard-negative rows, rollout review queue
  features, and active gold-eval queue features now have `0` rows; their
  summaries are marked `status=skipped` with
  `reason=user_labeled_rows_disabled`.
- A follow-up content scan over `backend/ml/artifacts` found additional old
  gold/manual snapshot artifacts (`*_active`, `*_combined`, backup active
  queues, rollout-reviewed gold rows, and related reports). Those were reset
  too, and `rg` no longer finds `gold_eval_manual`, `gold_eval_frontend`,
  `gold_eval_rollout_review`, `gold_positive`, or `gold_negative` in the
  artifact directory.
- Verification passed: `py_compile`; `bash -n`; focused unittest suite ->
  `12 passed`; scoped `git diff --check`. Required system
  `python3 ... graphify.watch...` still fails because that interpreter lacks
  `graphify`; rebuilding with the local Graphify interpreter succeeded with
  `3068` nodes, `8163` edges, and `87` communities.

## 2026-05-12 Plan: Let VOD Audio Downloader Drain Full Queue

- [x] Confirm current lookahead only queues one next VOD and can idle while Whisper transcribes.
- [x] Queue all missing VOD audio downloads on the single download worker before transcription consumes them.
- [x] Add focused tests and run verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- The previous one-item lookahead submitted only the next VOD audio download,
  so the single download worker could go idle while Whisper was still
  transcribing the current VOD.
- Missing VOD audio downloads are now all submitted to a single-worker executor
  before transcription starts consuming them. The worker still downloads
  serially, but it drains the full missing-audio queue independently of Whisper.
- Pipeline metadata is now `pipeline_mode=download_queue_ahead`; transcribe
  start logs expose `queued_audio_downloads_remaining`.
- Regression coverage proves the third VOD audio download occurs before the
  first VOD transcribe call finishes.
- Verification passed: `py_compile`; focused prepare tests -> `44 passed`;
  combined transcript/prepare tests -> `47 passed`; scoped `git diff --check`.
- Required system `python3 ... graphify.watch...` still fails because that
  interpreter lacks `graphify`; rebuilding with the local Graphify interpreter
  succeeded with `3067` nodes, `8161` edges, and `87` communities.

## 2026-05-12 Plan: Pipeline VOD Audio Download With Whisper Transcription

- [x] Confirm current VOD transcript cache builder downloads and transcribes each missing VOD serially.
- [x] Add one-item audio prefetch so the next VOD downloads while Whisper transcribes the current VOD.
- [x] Add focused tests and run verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- The VOD transcript cache builder now separates missing VODs into cached,
  skipped, and missing groups before the build.
- Missing VODs run through a one-item lookahead pipeline:
  `download_workers=1`, `transcribe_workers=1`,
  `pipeline_mode=prefetch_next_audio`. While Whisper transcribes the current
  VOD, the single download worker can prepare the next VOD's `source.m4a`.
- Whisper remains single-threaded and the next audio prefetch is visible in
  `full_vod_transcript_cache_vod_transcribe_start` via
  `next_audio_prefetch_active`.
- Verification passed: `py_compile`; focused prepare tests -> `44 passed`;
  combined transcript/prepare tests -> `47 passed`; scoped `git diff --check`.
- Required system `python3 ... graphify.watch...` still fails because that
  interpreter lacks `graphify`; rebuilding with the local Graphify interpreter
  succeeded with `3067` nodes, `8161` edges, and `87` communities.

## 2026-05-12 Plan: Prebuild VOD Transcript Caches Before Transcript Matching

- [x] Confirm why the current transcript run loaded only part of the VOD transcript pool.
- [x] Add a prepare-stage VOD transcript cache builder for the exact VOD manifest being matched.
- [x] Add regression coverage and run focused verification.
- [x] Rebuild Graphify and document the result.

### Review Notes

- The latest transcript run loaded only `20` VOD transcripts and skipped `37`,
  so transcript matching compared Shorts against less than half of the VOD
  manifest.
- `prepare_cross_platform_fingerprint_rows.py` now prebuilds compatible VOD
  transcript caches for every unique VOD row before transcript matching starts.
  Existing compatible caches are counted as cache hits; missing caches download
  full VOD audio, transcribe with diarization disabled, write the normal
  transcript cache, and then proceed to matching.
- The next run should emit `full_vod_transcript_cache_build_start`,
  per-VOD cached/build events, and `full_vod_transcript_cache_build_done`
  before `cross_platform_transcript_matching_call`.
- Verification passed: `py_compile`; focused prepare tests -> `43 passed`;
  combined transcript/prepare tests -> `46 passed`; scoped `git diff --check`.
- Required system `python3 ... graphify.watch...` still fails because that
  interpreter lacks `graphify`; rebuilding with the local Graphify interpreter
  succeeded with `3066` nodes, `8158` edges, and `88` communities.

## 2026-05-12 Plan: Stop Preserving Audio Rows In Transcript Mode

- [x] Confirm why the latest transcript run still outputs old audio-fingerprint rows.
- [x] Require preserved existing output to be reusable for the active verification mode.
- [x] Add regression coverage and run focused verification.

### Review Notes

- The latest run matched `31` transcript rows, but `output_rows=73` because
  `preserved_existing_output=true` kept the prior
  `cross_platform_audio_fingerprint` artifact after the transcript result missed
  the `70` row gate.
- Transcript mode now preserves an existing output only when every preserved row
  is reusable for the active transcript verification mode. Old audio rows are no
  longer carried forward into transcript-only training.
- If transcript matching is insufficient, the output file is still replaced
  with the current transcript matches so downstream training cannot silently use
  stale audio-fingerprint positives.
- Verification passed: `py_compile`; focused
  `test_prepare_cross_platform_fingerprint_rows.py` -> `41 passed`; scoped
  `git diff --check`.

## 2026-05-12 Plan: Skip Diarization In Cross-Platform Transcript Matching

- [x] Add an explicit transcriber option to skip speaker diarization for text-only matching.
- [x] Wire cross-platform Short transcription to use text-only mode.
- [x] Run focused verification and document results.

### Review Notes

- The pasted run log showed Whisper transcription completing, then optional
  Pyannote diarization failing because `torchcodec`/`AudioDecoder` is broken in
  the container.
- Cross-platform transcript matching does not need speaker labels, so Short
  transcription now calls `transcriber.transcribe(..., enable_diarization=False)`.
- Normal transcriber callers keep the existing default
  `enable_diarization=True`, so VOD feature paths are unchanged.
- Verification passed: `py_compile`; focused tests
  `test_match_cross_platform_transcripts.py` + `test_transcriber.py` ->
  `10 passed`; scoped `git diff --check`.

## 2026-05-12 Plan: Make Transcript Matching Primary Cross-Platform Verification

- [x] Trace current audio-fingerprint prepare path and available transcript caches.
- [x] Add transcript-based repost-to-VOD matcher and wire the prepare flow to use it as the primary label source.
- [x] Update downstream source labels/tests so outcome rows no longer present as audio-fingerprint positives.
- [x] Run focused verification, update Graphify, and document results.

### Review Notes

- Cross-platform verification now defaults to `transcript` via
  `CROSS_PLATFORM_VERIFICATION_MODE=transcript` in `train_jynxzi_vod.sh` and
  `--verification-mode transcript` in the prepare CLI.
- Added `match_cross_platform_transcripts.py`, which loads existing VOD
  transcript caches, transcribes/caches downloaded Shorts under
  `_download_cache/short_form_transcripts`, and writes positives as
  `cross_platform_transcript_match`.
- Audio fingerprints are still available only as explicit
  `--verification-mode audio_fingerprint` legacy mode; transcript mode does not
  fall back to audio matching.
- Outcome and VOD-ranker dataset prep now preserve transcript label provenance
  instead of rewriting cross-platform positives to
  `cross_platform_audio_fingerprint`.
- Verification passed: `py_compile` for changed ML modules; new transcript
  matcher tests -> `2 passed`; focused combined tests -> `84 passed`;
  `bash -n backend/ml/train_jynxzi_vod.sh`; scoped `git diff --check`.
- Required system `python3 ... graphify.watch...` still fails because this
  shell's Python has no `graphify` module; rebuilding with the local Graphify
  interpreter succeeded with `3056` nodes, `8129` edges, and `78` communities.

## 2026-05-11 Plan: Switch Cross-Platform Source To Iboclipss

- [x] Replace the curated YouTube Shorts source with `youtube.com/@Iboclipss`.
- [x] Remove stale Clips.Delivery-derived short media, metadata, and cross-platform output artifacts so the next run starts clean.
- [x] Verify source/caches/artifact state and document the cleanup.

### Review Notes

- `backend/ml/cross_platform_sources.csv` now points Jynxzi cross-platform
  discovery at `https://www.youtube.com/@Iboclipss/shorts` with
  `curated_repost_source;iboclipss;youtube_shorts`.
- Removed stale short-form media and metadata caches:
  `backend/tmp/storage/_download_cache/short_form` and
  `backend/tmp/storage/_download_cache/short_form_metadata`.
- Removed stale cross-platform derived artifacts:
  `jynxzi_cross_platform_reposts.jsonl`,
  `jynxzi_cross_platform_fingerprint_rows.jsonl`,
  `jynxzi_cross_platform_fingerprint_summary.json`, and
  `jynxzi_cross_platform_fingerprint_diagnostics.jsonl`.
- Kept VOD full-audio fingerprint caches intact because they are independent of
  the YouTube repost channel and avoid redownloading/re-fingerprinting VODs.
- Verification: source manifest parses as one Iboclipss row; old short/media
  cache file count is `0`; stale cross-platform repost/output artifacts are
  absent.

## 2026-05-10 Plan: Restore Cached Per-Short Metadata Dates

- [x] Re-enable cached per-Short metadata enrichment for date-scoped repost discovery.
- [x] Keep the existing serial yt-dlp pacing/cooldown behavior for every metadata/download request outcome.
- [x] Add regression coverage for metadata cache reuse and exact upload-date filtering, then verify and rebuild Graphify.

### Review Notes

- Clips.Delivery channel rows still start from one flat playlist discovery, but
  rows with no recovered channel-page publish date now fall back to an exact
  per-Short yt-dlp metadata request.
- Successful per-Short metadata payloads are cached under
  `_download_cache/short_form_metadata`, keyed by URL, so later runs reuse the
  exact upload date/timestamp instead of requesting the same Short again.
- Cache misses sleep for the configured 5-10s short-form interval after a
  successful metadata request; metadata errors keep the existing error cooldown.
  Cache hits do not sleep.
- Verification passed: targeted metadata-date tests -> `2 passed`; focused
  auto-train/prepare/downloader tests -> `87 passed`; `py_compile` passed;
  scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3019` nodes, `8014` edges, and
  `78` communities.

## 2026-05-10 Plan: Recover Channel-Scoped Short Dates Without Per-Short Metadata

- [x] Enrich YouTube channel/Shorts flat discovery rows from the channel page's publish-time data.
- [x] Convert relative publish text to approximate UTC upload dates and keep missing-date rows fail-closed.
- [x] Add regression coverage, run verification, update lessons, and rebuild Graphify.

### Review Notes

- The failed run discovered `1738` Clips.Delivery candidates but skipped all of
  them as `missing_publish_date`, because flat YouTube channel extraction
  returned titles/URLs without dates and per-Short metadata enrichment was
  intentionally disabled.
- YouTube channel `/shorts` discovery now performs one channel-page fetch and
  parses per-video `publishedTimeText` blocks keyed by `videoId`.
- Relative publish text such as `2 weeks ago` / `3 months ago` is converted into
  approximate UTC `timestamp` and `upload_date` fields before date-window
  filtering.
- Rows still fail closed if no publish text can be recovered; the pipeline does
  not assume undated Shorts belong to the VOD window.
- Verification passed: targeted channel-date tests -> `2 passed`; focused
  auto-train/prepare/downloader tests -> `86 passed`; `py_compile` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3009` nodes, `7986` edges, and
  `75` communities.

## 2026-05-10 Plan: Add Post-Attempt Delay For Repost Short Downloads

- [x] Move repost Short download pacing to a `finally` path so every attempted yt-dlp cache fill sleeps after the attempt.
- [x] Keep cache hits fast; apply the delay only to actual missing-media download attempts.
- [x] Add regression coverage for successful and failed attempts, run verification, update lessons, and rebuild Graphify.

### Review Notes

- The serial repost Short download loop now emits
  `cross_platform_repost_download_request_cooldown` and sleeps after every
  attempted missing-media Short cache fill.
- Success and failure outcomes both pause before the next Short is attempted.
  Existing usable media rows remain cache hits and do not add artificial delay.
- Short-form pacing defaults and env values are now `5..10` seconds:
  `YTDLP_SHORT_FORM_SLEEP_REQUESTS_SEC=5`,
  `YTDLP_SHORT_FORM_MIN_SLEEP_INTERVAL_SEC=5`,
  `YTDLP_SHORT_FORM_MAX_SLEEP_INTERVAL_SEC=10`.
- Verification passed: targeted success/failure delay tests -> `2 passed`;
  focused prepare/downloader/auto-train tests -> `84 passed`; `py_compile`
  passed; scoped `git diff --check` passed; settings resolution prints
  `5.0 5.0 10.0`.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3001` nodes, `7965` edges, and
  `77` communities.

## 2026-05-10 Plan: Avoid Per-Short Metadata Enrichment For Date Filtering

- [x] Require repost candidates to carry publish/upload date from discovery or manifest rows before date-window filtering.
- [x] Stop per-Short yt-dlp metadata enrichment during repost manifest filtering so age/rate-limited Shorts do not create extra requests.
- [x] Add regression coverage proving missing-date rows are skipped without metadata lookup, then run focused verification and rebuild Graphify.

### Review Notes

- Repost date-window filtering now uses only `publish_timestamp` /
  `upload_date` already present on discovered or manifest rows.
- Rows without a publish/upload date still fail closed as
  `skipped_missing_publish_date`; they are not recovered with a per-Short
  yt-dlp metadata request.
- This keeps the Clips.Delivery source date-scoped to the actual usable VOD
  window without adding extra YouTube requests during manifest filtering.
- Verification passed: targeted regression -> `1 passed`; focused
  auto-train/prepare tests -> `64 passed`; `py_compile` passed; scoped
  `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3000` nodes, `7964` edges, and
  `76` communities.

## 2026-05-10 Plan: Replace One-Hour Short Error Cooldown

- [x] Remove the explicit 3600s short-form error cooldown behavior.
- [x] Reuse the existing short-form 5-10s style sleep range for all yt-dlp short error cooldowns.
- [x] Update tests/env/docs, run verification, and rebuild Graphify.

### Review Notes

- Removed `YTDLP_SHORT_FORM_RATE_LIMIT_COOLDOWN_SEC` from `backend/.env`,
  `.env.example`, and `Settings`; stale env values are ignored because the
  setting no longer exists.
- Short-form yt-dlp error cooldowns now sample the same configured short-form
  range used by normal yt-dlp pacing:
  `YTDLP_SHORT_FORM_MIN_SLEEP_INTERVAL_SEC=6` to
  `YTDLP_SHORT_FORM_MAX_SLEEP_INTERVAL_SEC=10`.
- The final repost download path and the discovery/metadata paths all use this
  short jittered cooldown.
- Verification passed: targeted cooldown tests -> `3 passed`; focused
  auto-train/prepare/downloader tests -> `84 passed`; Python `py_compile`
  passed; scoped `git diff --check` passed; env resolution prints
  `6.0 10.0 False`.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `3001` nodes, `7971` edges, and
  `81` communities.

## 2026-05-10 Plan: Apply Short Cooldown To All yt-dlp Short Errors

- [x] Trace why current YouTube errors bypass the existing materialization cooldown.
- [x] Add cooldown handling to discovery/metadata yt-dlp exceptions as well as final download exceptions.
- [x] Add focused tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Root cause: some YouTube errors happen before final media cache
  materialization, during source discovery or per-entry metadata extraction.
  Those exceptions were swallowed or converted to fallback rows before reaching
  the cooldown logic in `prepare_cross_platform_fingerprint_rows.py`.
- Added `short_form_yt_dlp_error_cooldown` handling to yt-dlp source discovery,
  date-search extraction, and metadata extraction in `auto_train_short_form.py`.
- The same `YTDLP_SHORT_FORM_RATE_LIMIT_COOLDOWN_SEC` setting is used. Any
  exception in those yt-dlp paths sleeps before the workflow continues; YouTube
  rate-limit errors are marked with `rate_limited=True` in the warning log.
- Verification passed: targeted cooldown tests -> `3 passed`; focused
  auto-train/prepare/downloader tests -> `84 passed`; Python `py_compile`
  passed; scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2997` nodes, `7967` edges, and
  `75` communities.

## 2026-05-10 Plan: Cool Down And Continue After Any Short Download Error

- [x] Change repost Short download errors from retry-same-URL to cooldown-and-continue.
- [x] Apply the cooldown to any download exception, while still flagging YouTube rate-limit errors in logs.
- [x] Update tests/lessons, run verification, and rebuild Graphify.

### Review Notes

- Repost Short download failures now emit
  `cross_platform_repost_download_failed_cooldown`, sleep for
  `YTDLP_SHORT_FORM_RATE_LIMIT_COOLDOWN_SEC`, and then continue with the next
  Short.
- The same URL is no longer retried after the cooldown. Any exception triggers
  the cooldown; YouTube rate-limit errors are marked with `rate_limited=true`.
- The manifest summary now includes `repost_download_error_cooldowns` plus the
  existing rate-limit-specific cooldown count.
- Verification passed: focused pytest -> `59 passed`; Python `py_compile`
  passed; targeted cooldown-and-continue test -> `1 passed`; scoped
  `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2989` nodes, `7953` edges, and
  `73` communities.

## 2026-05-10 Plan: Cool Down After YouTube Rate-Limit Errors

- [x] Detect yt-dlp YouTube "try again later" rate-limit errors during serial repost Short downloads.
- [x] Sleep for a configured cooldown and retry the same Short once before counting it as failed.
- [x] Add env wiring/tests/lesson, run verification, and rebuild Graphify.

### Review Notes

- Added `YTDLP_SHORT_FORM_RATE_LIMIT_COOLDOWN_SEC=3600` to the backend env
  files and `yt_dlp_short_form_rate_limit_cooldown_sec` to settings.
- During serial repost Short cache fills, YouTube rate-limit errors such as
  `This content isn't available, try again later` now emit
  `cross_platform_repost_download_rate_limited`, sleep for the configured
  cooldown, and retry the same URL once.
- The manifest summary now includes rate-limit cooldown/retry counters.
- Verification passed: focused pytest -> `59 passed`; targeted rate-limit test
  -> `1 passed`; Python `py_compile` passed; scoped `git diff --check` passed;
  env resolution prints `6.0 6.0 10.0 3600.0`.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2989` nodes, `7953` edges, and
  `73` communities.

## 2026-05-10 Plan: Add YouTube yt-dlp Sleep For Shorts

- [x] Add configurable yt-dlp request/data-extraction sleep for short-form media downloads.
- [x] Apply the sleep only to `short_form` cache fills so Twitch/full-VOD audio paths stay unchanged.
- [x] Add tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Added Short-form-specific yt-dlp sleep settings:
  `YTDLP_SHORT_FORM_SLEEP_REQUESTS_SEC=6`,
  `YTDLP_SHORT_FORM_MIN_SLEEP_INTERVAL_SEC=6`, and
  `YTDLP_SHORT_FORM_MAX_SLEEP_INTERVAL_SEC=10`.
- The settings map to yt-dlp native options `sleep_interval_requests`,
  `sleep_interval`, and `max_sleep_interval`.
- These options are applied only when `ensure_cached_download(...,
  cache_namespace="short_form")` fills a missing media cache entry. Existing
  Short cache hits, Twitch VOD downloads, generic downloads, and full-VOD audio
  metadata extraction are not slowed by this change.
- Verification passed: focused pytest -> `61 passed`; Python `py_compile`
  passed; scoped `git diff --check` passed; env resolution prints `6.0`,
  `6.0`, `10.0`.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2986` nodes, `7943` edges, and
  `73` communities.

## 2026-05-10 Plan: Revert Repost Shorts To Serial Download

- [x] Remove the cross-platform repost download worker setting and shell/env wiring.
- [x] Change repost media materialization back to one-by-one cache downloads.
- [x] Replace the parallel regression with a serial-order regression, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Removed `CROSS_PLATFORM_REPOST_DOWNLOAD_WORKERS` from `backend/.env`,
  `backend/.env.example`, `backend/config.py`, and `backend/ml/train_jynxzi_vod.sh`.
- Repost Short media is now materialized by a simple for-loop in
  `build_repost_manifest_from_sources`; no `ThreadPoolExecutor` is used for
  this path.
- The manifest summary now reports `repost_download_mode: serial` and
  `repost_download_submitted_rows`.
- Verification passed: focused pytest -> `59 passed`; Python `py_compile`
  passed; `bash -n` passed; scoped `git diff --check` passed; no repost-worker
  config or parallel-test leftovers remain in touched paths.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2983` nodes, `7937` edges, and
  `75` communities.

## 2026-05-10 Plan: Batch Download Required YouTube Shorts

- [x] Keep yt-dlp request-delay settings out of the workflow.
- [x] Download all selected date/profile-valid repost Shorts into the short-form cache as one parallel batch.
- [x] Add regression tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Added `CROSS_PLATFORM_REPOST_DOWNLOAD_WORKERS` / `cross_platform_repost_download_workers`
  with default `8`.
- `build_repost_manifest_from_sources` now filters and sorts repost candidates
  first, then materializes every selected uncached Short in a parallel batch
  before writing the repost manifest.
- No yt-dlp sleep/request-delay options were added.
- Verification passed: focused pytest -> `59 passed`; Python `py_compile`
  passed; `bash -n` passed; scoped `git diff --check` passed; no sleep-related
  yt-dlp options present in the touched paths.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2983` nodes, `7937` edges, and
  `75` communities.

## 2026-05-10 Plan: Remove yt-dlp Cookie Diagnostics

- [x] Remove cookie-file logging from shared yt-dlp auth helpers.
- [x] Remove cookie-auth status/success/failure logging from discovery, metadata, download, and full-VOD metadata call sites.
- [x] Verify cookie support still applies options without emitting cookie-file logs.

### Review Notes

- Removed the cookie-file diagnostic helpers/events from
  `backend/services/yt_dlp_auth.py`; it now only applies `cookiefile` or
  `cookiesfrombrowser` options.
- Removed auth status/success/failure logging from Shorts discovery, Shorts
  metadata, generic media download, Twitch download, and full-VOD audio
  metadata extraction.
- Verification passed: no backend matches for cookie diagnostic fields/events;
  focused pytest -> `81 passed`; `py_compile` passed; scoped
  `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2981` nodes, `7929` edges, and
  `73` communities.

## 2026-05-10 Plan: Add yt-dlp Cookie Diagnostics

- [x] Superseded by the removal request above.

## 2026-05-10 Plan: Add yt-dlp YouTube Cookie Support

- [x] Add shared yt-dlp auth option handling for cookie files / browser cookies.
- [x] Apply it to Shorts discovery, Shorts metadata enrichment, Shorts downloads, and full-VOD audio metadata extraction.
- [x] Add env placeholders, tests, lessons, verification, and Graphify rebuild.

### Review Notes

- Added shared yt-dlp auth option handling in `services/yt_dlp_auth.py`.
- New settings/env keys:
  `YTDLP_COOKIEFILE` and `YTDLP_COOKIES_FROM_BROWSER`.
- Cookiefile auth is applied to generic downloader downloads, Shorts discovery,
  Shorts metadata enrichment, and full-VOD audio metadata extraction.
- Cookiefile takes precedence over browser cookies because it works better in
  Docker/non-interactive training runs.
- Verification passed: focused pytest -> `81 passed`; `py_compile` passed;
  scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2981` nodes, `7929` edges, and
  `73` communities.

## 2026-05-10 Plan: Use Clips.Delivery As Curated Repost Source

- [x] Replace broad YouTube/TikTok keyword discovery with only `youtube.com/@Clips.Delivery/shorts`.
- [x] Mark this channel as a trusted curated repost source while preserving the real VOD date-window filter.
- [x] Bump policy version, add regression tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- `backend/ml/cross_platform_sources.csv` now contains exactly one configured
  cross-platform source for Jynxzi:
  `https://www.youtube.com/@Clips.Delivery/shorts`.
- The source is tagged `curated_repost_source`, so its clips bypass only the
  profile-text filter. This means titles/uploader names do not need to contain
  `jynxzi` or `junko`.
- The real VOD publish-date window still applies. Curated clips outside that
  VOD-derived date range are skipped, and clips without publish metadata still
  fail closed unless metadata can be enriched.
- `REPOST_PROFILE_FILTER_VERSION` is now `8`, so the next run rebuilds repost
  manifests and does not reuse older cross-platform rows from the broad
  keyword/TikTok policy.
- Verification passed: focused pytest -> `67 passed`; `py_compile` passed;
  scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command still fails because
  this shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2970` nodes, `7912` edges, and
  `72` communities.

## 2026-05-10 Plan: Expand Cross-Platform Label Discovery Coverage

- [x] Add broader Jynxzi/Junko YouTube and TikTok source queries for the same VOD date-window policy.
- [x] Add a TikTok Bing HTML fallback so zero RSS results do not mean zero TikTok candidates.
- [x] Bump repost policy version, add tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Cross-platform source discovery now has `27` source rows instead of the
  narrow keyword set: broader YouTube searches for `jynxzi shorts`,
  `jynxzi viral`, `jynxzi twitch`, `jynxzi league of legends`, and
  `junko clips`, plus matching TikTok keyword searches.
- `tiktoksearch...` no longer depends only on Bing RSS. If RSS returns no
  TikTok videos, the discovery code also parses Bing HTML result links and
  canonicalizes direct TikTok and Bing redirect URLs.
- `REPOST_PROFILE_FILTER_VERSION` is now `7`, so stale repost manifests from
  the smaller source/filter policy rebuild on the next run.
- Existing cross-platform match rows now also carry `profile_filter_version`;
  stale rows below version `7` are not considered reusable. That prevents the
  next run from short-circuiting before the broader discovery policy runs.
- Verification passed: focused pytest -> `66 passed`; `py_compile` passed;
  scoped `git diff --check` passed.
- The required `python3 -c ... graphify.watch...` command failed because this
  shell's `python3` has no `graphify` module; rebuilding with the local
  Graphify interpreter succeeded with `2968` nodes, `7905` edges, and
  `72` communities.

## 2026-05-10 Plan: Force Repost Rediscovery After Date-Search Fix

- [x] Confirm the run reused the 7-row repost manifest instead of rediscovering sources.
- [x] Bump the repost manifest policy version so the next run rebuilds with the `ytsearchdate` handler.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The posted run did not exercise the `ytsearchdate` handler because
  `repost_manifest_built=false` and the existing 7-row manifest had
  `profile_filter_version=5`.
- `REPOST_PROFILE_FILTER_VERSION` is now `6`, so those version-5 rows are stale
  and the next run should show `repost_manifest_profile_filter_rebuild_required=true`
  followed by a rebuilt manifest.
- Verification passed: focused pytest -> `56 passed`; stale-manifest targeted
  test -> `1 passed`; `py_compile` passed; scoped `git diff --check` passed;
  Graphify rebuilt successfully with `2957` nodes, `7880` edges, and
  `69` communities.

## 2026-05-10 Plan: Fix Unsupported `ytsearchdate` Source

- [x] Add an internal `ytsearchdateN:<query>` handler instead of passing that scheme to yt-dlp.
- [x] Keep the same fresh-source CSV policy, but make it compatible with the installed yt-dlp version.
- [x] Add focused tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- The active run showed `Unsupported url scheme: "ytsearchdate500"` because the
  installed yt-dlp does not support that pseudo-scheme.
- `auto_train_short_form.py` now intercepts `ytsearchdateN:<query>` before
  yt-dlp sees it and converts it to a real YouTube results URL with upload-date
  sort parameter `sp=CAI%253D`.
- If the date-sorted search page extraction returns no rows, the code falls
  back to normal `ytsearchN:<query>` instead of failing the source.
- Verification passed: focused pytest -> `56 passed`; targeted
  `ytsearchdate` test -> `1 passed`; `py_compile` passed; scoped
  `git diff --check` passed; Graphify rebuilt successfully with `2957` nodes,
  `7880` edges, and `69` communities.

## 2026-05-10 Plan: Pick Most-Viewed Reposts Inside Actual VOD Window

- [x] Move cross-platform repost selection to a global post-filter ranking.
- [x] Rank date-valid/profile-valid reposts by views and viral score before applying `max_reposts`.
- [x] Add focused tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Cross-platform repost manifest building now dedupes all discovered/manual
  candidates first, then applies explicit profile evidence and the actual VOD
  date-window filter.
- Only after those filters does it sort candidates globally by `views`,
  `viral_score`, and publish time, then apply `max_reposts`.
- This means an older/out-of-window high-view video can no longer consume a
  slot, and the selected labels are the most-viewed valid candidates from the
  concrete usable VOD window.
- Added manifest summary counters: `eligible_profile_date_rows` and
  `selected_repost_candidates`.
- Bumped `REPOST_PROFILE_FILTER_VERSION` to `5`, so the next run rebuilds the
  repost manifest with this selection policy instead of reusing the previous
  one.
- Verification passed: focused pytest -> `55 passed`; targeted most-viewed
  regression -> `1 passed`; `py_compile` passed; scoped `git diff --check`
  passed; Graphify rebuilt successfully with `2953` nodes, `7869` edges, and
  `70` communities.

## 2026-05-10 Plan: Increase Date-Scoped Repost Label Discovery

- [x] Add fresh/date-sorted YouTube sources for the same four Jynxzi keywords.
- [x] Make TikTok keyword search use multiple search variants and keep RSS fallback rows when metadata extraction fails.
- [x] Add focused tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- Added `ytsearchdate500` sources for the existing four requested Jynxzi
  keywords, so discovery can pull newer/date-sorted YouTube results for the
  actual VOD window instead of only best-match search results.
- TikTok keyword search now queries multiple Bing RSS variants per keyword and
  dedupes URLs across them.
- TikTok RSS rows are kept as fallback candidates when detailed metadata
  extraction fails; their title, description, uploader, timestamp, and
  upload-date are still available for downstream profile/date filtering.
- Bumped `REPOST_PROFILE_FILTER_VERSION` to `4`, forcing the next run to rebuild
  the repost manifest with the expanded source policy instead of reusing the
  small version-3 manifest.
- Verification passed: focused pytest -> `54 passed`; TikTok-specific tests ->
  `2 passed`; `py_compile` passed; scoped `git diff --check` passed; Graphify
  rebuilt successfully with `2951` nodes, `7861` edges, and `72` communities.

## 2026-05-10 Plan: Remove Raw Repost Minimum After Date Scoping

- [x] Confirm the run fails before matching because raw repost count is below the old global minimum.
- [x] Set the default raw repost minimum to zero and rely on matched-row/VOD gates.
- [x] Update tests, lessons, run verification, and rebuild Graphify.

### Review Notes

- The new source policy worked but yielded only `10` date-valid reposts.
  `min_reposts=400` blocked the run before audio matching, so no diagnostics
  about actual matches could be produced.
- `CROSS_PLATFORM_MIN_REPOSTS` now defaults to `0` in the shell workflow,
  Docker env file, env example, and workflow-state defaults.
- The remaining meaningful gates are still `CROSS_PLATFORM_MIN_MATCHED_ROWS`
  and `CROSS_PLATFORM_MIN_MATCHED_VODS`; with only `10` reposts the next run
  may still fail there, but it will fail after real matching instead of before
  it.
- Verification passed: focused pytest -> `45 passed`; `py_compile` passed;
  `bash -n` passed; scoped `git diff --check` passed; Graphify rebuilt
  successfully with `2944` nodes, `7850` edges, and `72` communities.

## 2026-05-10 Plan: Add TikTok Keyword Repost Sources

- [x] Replace broad/overlapping Jynxzi repost keywords with the requested four-keyword set.
- [x] Add TikTok keyword-source support and source rows alongside YouTube.
- [x] Add focused tests, run verification, and rebuild Graphify.

### Review Notes

- `cross_platform_sources.csv` now has exactly the requested keywords for both
  platforms: `jynxzi`, `jynxzi funny`, `jynxzi rage`, and `jynxzi clips`.
- Added `tiktoksearchN:<query>` source support. It discovers TikTok video URLs
  through Bing RSS, then still runs normal metadata extraction/download logic.
- Search-source strings (`ytsearch...` and `tiktoksearch...`) are not trusted
  as profile evidence. Rows still need explicit Jynxzi/Junko evidence in
  title/uploader/channel/description.
- Bumped `REPOST_PROFILE_FILTER_VERSION` to `2`, so the next run rebuilds the
  repost manifest with the new source policy instead of reusing the old one.
- Verification passed: focused pytest -> `51 passed`; `py_compile` passed;
  scoped `git diff --check` passed; Graphify rebuilt successfully with
  `2938` nodes, `7827` edges, and `69` communities.

## 2026-05-10 Plan: Scope Reposts To Actual VOD Date Window

- [x] Identify the actual VOD date source and why current inventory rows lost `created_at`.
- [x] Preserve non-empty VOD API dates when merging cache-backed inventory rows.
- [x] Filter discovered reposts to the date window of VODs present in the usable VOD manifest.
- [x] Add tests, update lessons, run verification, and rebuild Graphify.

### Review Notes

- The VOD inventory merge could overwrite non-empty API fields like
  `created_at` with empty cache metadata. The merge now preserves non-empty
  `created_at`, `title`, and `vod_duration_seconds`.
- Repost manifest rebuilds now compute a date window from the actual VOD rows
  used for matching, joined back to `jynxzi_twitch_vod_inventory.jsonl`.
- Reposts are kept only when their `publish_timestamp` or `upload_date` falls
  in that actual VOD window, with a small default margin: `1` day before the
  earliest usable VOD and `30` days after the latest usable VOD.
- If flat discovery omits publish/upload date, the builder tries one metadata
  lookup for that URL before deciding whether to skip it as date-unknown.
- Bumped `REPOST_PROFILE_FILTER_VERSION` to `3`, so any manifest from the
  pre-date-window policy is rebuilt on the next run.
- Verification passed: focused pytest -> `57 passed`; `py_compile` passed;
  `bash -n` passed; scoped `git diff --check` passed; Graphify rebuilt
  successfully with `2944` nodes, `7850` edges, and `72` communities.

## 2026-05-10 Plan: Calibrate Repost Minimum After Profile Filter

- [x] Confirm the new run failed because filtered repost rows dropped below the old pre-match minimum.
- [x] Lower the repost-source minimum to the filtered Jynxzi source pool while keeping match-row and VOD-diversity gates.
- [x] Update env/workflow-state defaults, tests, lessons, and run verification plus Graphify.

### Review Notes

- The profile filter worked: the run rebuilt with
  `repost_manifest_build_reason=profile_filter_policy`, removed
  `152` non-profile search hits, and did not preserve the old false-positive
  output.
- The failure was the remaining pre-match source guard:
  `repost_rows=493 < min_reposts=600`, so matching never started.
- Lowered `CROSS_PLATFORM_MIN_REPOSTS` to `400` in the shell default,
  Docker env file, env example, and workflow-state default. The hard quality
  gates remain `70` matched rows and `20` matched VODs for the strict pass.
- Verification passed: focused pytest -> `43 passed`; `py_compile` passed;
  `bash -n` passed; scoped `git diff --check` passed; Graphify rebuilt
  successfully with `2934` nodes, `7818` edges, and `73` communities.

## 2026-05-08 Plan: Parallelize Full-VOD Audio Fingerprint Builds

- [x] Add a bounded `--full-vod-audio-workers` option to cross-platform fingerprint prep.
- [x] Run Full-VOD audio cache/download/fingerprint work with two parallel workers by default in Jynxzi training.
- [x] Keep per-VOD logging and summary counters accurate under parallel execution.
- [x] Update env/workflow/tests, run focused verification, and rebuild Graphify.

### Review Notes

- `prepare_cross_platform_fingerprint_rows.py` now accepts
  `--full-vod-audio-workers` and runs per-VOD full-audio cache/download/
  fingerprint work through a bounded `ThreadPoolExecutor` when workers > 1.
- Jynxzi training defaults `CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS=2`, passes it
  into prepare, exports it into workflow state, and records it in `.env` and
  `.env.example`.
- Summary/log payloads include `workers`, while per-VOD success/cache/failure
  counters are aggregated after worker completion.
- Verification passed: focused prepare/script/workflow-state tests,
  `py_compile`, `bash -n`, and scoped `git diff --check`.

## 2026-05-08 Plan: Remove Twitch Clip Labels From Outcome Training Flow

- [x] Map the remaining Twitch-label dependencies in the Jynxzi training script and cross-platform fingerprint prep.
- [x] Replace Twitch-label VOD seeding with a VOD inventory path built from available non-label artifacts/cache.
- [x] Remove Twitch clip-label collection/pruning from the Jynxzi training workflow.
- [x] Update tests, env defaults, lessons, and run focused verification plus Graphify.

### Review Notes

- `train_jynxzi_vod.sh` no longer collects or prunes Twitch clip labels, and
  there is no `TWITCH_CLIP_LABELS_ENABLED` opt-in switch left in the training
  workflow or env files.
- The workflow now builds `jynxzi_twitch_vod_inventory.jsonl` from the Twitch
  VOD API plus existing `vod_feature_cache` metadata. Full-VOD audio matching
  receives that inventory through `--full-vod-audio-vod-inventory`.
- `prepare_cross_platform_fingerprint_rows.py` prefers the VOD inventory over
  Twitch label rows when selecting full-VOD audio fingerprints. Label rows are
  only a legacy fallback if no inventory exists.
- `train_vod_ranker.py` no longer exits just because Twitch label rows are
  empty; with outcome-only training it can continue from cross-platform rows.
- Local cache inspection found `67` VOD inventory candidates, so the next run
  should not be limited to the single stale Twitch-label VOD.
- Verification passed: focused prepare/script/workflow-state tests, full
  `test_train_vod_ranker.py`, `py_compile`, `bash -n`, and scoped
  `git diff --check`.

## 2026-05-08 Plan: Fix Stale Twitch Label Progress Blocking Wider Windows

- [x] Confirm the wider date window is skipped because progress says covered while labels contain only one VOD.
- [x] Add a guard so stale label progress does not block collection for wider training windows.
- [x] Keep the fix scoped to Twitch label collection; do not change cross-platform matching behavior.
- [x] Run focused tests and syntax checks.

### Review Notes

- Current progress claimed `2025-10-01..2025-12-14` and
  `2026-01-01..2026-05-05` were covered, but the active label file contained
  only `483` rows from one VOD on `2026-05-01`.
- `collect_twitch_clip_labels.py` now validates progress ranges against actual
  existing label dates before honoring `--skip-covered`.
- With current artifacts, the validated covered range becomes only
  `2026-05-01..2026-05-01`, so the next wider run will recollect the missing
  days instead of skipping the whole window.
- Verification passed: `test_twitch_clip_labels.py` -> `30 passed`;
  `py_compile` passed; scoped `git diff --check` passed.

## 2026-05-08 Plan: Fix Env Override For Cross-Platform Matching Defaults

- [x] Confirm why the restarted run still used `min_confidence=0.5` and `reuse_existing_output=true`.
- [x] Update the real Docker env file and example so they no longer override the code defaults with stale values.
- [x] Verify the effective env-file values and run focused checks.
- [x] Document the restart expectation.

### Review Notes

- `scripts/train_local_docker.sh` passes `backend/.env` through Docker
  `--env-file`, and that file still had
  `CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE=0.50` plus
  `CROSS_PLATFORM_REUSE_EXISTING_ROWS=1`.
- Updated both `backend/.env` and `backend/.env.example` to `0.60` and `0`.
- Verified with `rg` that the active env file, example, and shell defaults now
  agree.
- Verification passed: shell syntax check and scoped `git diff --check`.

## 2026-05-08 Plan: Stop Reusing Stale Cross-Platform Rows By Default

- [x] Confirm the failed run reused the old cross-platform artifact instead of rematching at the new threshold.
- [x] Change the Jynxzi training default so cross-platform rows are rematched unless reuse is explicitly enabled.
- [x] Keep the state snapshot and tests aligned with the new default.
- [x] Run focused verification and rebuild Graphify.

### Review Notes

- The run showed `prepare_status=reused_existing_output` and
  `prepare_preserved_existing_output=true`, so it did not rematch with the new
  `0.60` threshold.
- `CROSS_PLATFORM_REUSE_EXISTING_ROWS` now defaults to `0` in the Jynxzi
  training script. Existing cross-platform rows can still be reused explicitly
  by setting `CROSS_PLATFORM_REUSE_EXISTING_ROWS=1`.
- The workflow-state snapshot default and test expectation now match the shell
  default.
- Verification passed: focused workflow-state pytest, shell syntax check,
  `py_compile`, and scoped `git diff --check`.

## 2026-05-08 Plan: Prevent One-VOD Cross-Platform Overmatching

- [ ] Diagnose the one-VOD full-audio match explosion from the run summary.
- [ ] Raise the cross-platform audio confidence default for full-VOD matching.
- [ ] Add a seed matched-VOD diversity guard.
- [ ] Update workflow state/tests and run focused verification.

## 2026-05-08 Plan: Fix Full-VOD Audio Temp Extension

- [x] Diagnose ffmpeg returncode 234 from the new structured logs.
- [x] Change full-VOD audio temp output path so ffmpeg can infer the muxer.
- [x] Add regression coverage for the temp-path extension.
- [x] Run focused tests, syntax checks, and Graphify.

### Review Notes

- The ffmpeg failure was caused by the temporary output path
  `source.m4a.part`. FFmpeg could not infer the output container from that
  final `.part` suffix.
- Full-VOD audio extraction now writes to `source.part.m4a` and then atomically
  replaces `source.m4a`, preserving the real media extension as the final
  suffix.
- Added regression coverage for the temp path.
- Verification passed: focused pytest -> `42 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2913` nodes, `7758` edges,
  `69` communities.

## 2026-05-08 Plan: Diagnose Full-VOD Audio FFmpeg Failure

- [x] Inspect the full-VOD audio download implementation and current error
  logging.
- [x] Preserve useful ffmpeg stderr and source-format diagnostics in the
  failure log.
- [x] Fix obvious ffmpeg invocation issues if found.
- [x] Add focused regression coverage and run verification.

### Review Notes

- The run now fails in the intended place: Full-VOD audio extraction for
  `2761754957`, no MP4 fallback, and immediate blocked training exit.
- The previous log only showed a truncated `CalledProcessError` command because
  the ffmpeg command included long HTTP headers. It hid the real stderr.
- Added structured `FullVodAudioDownloadError` wrapping ffmpeg failures with
  `ffmpeg_returncode`, `ffmpeg_stderr`, `ffmpeg_stdout`, and selected format
  metadata.
- Added `full_vod_audio_download_input_selected` logging so the next run shows
  selected format id, protocol, codecs, bitrate, height, and media URL prefix.
- Verification passed: focused pytest -> `41 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2911` nodes, `7754` edges,
  `70` communities.

## 2026-05-08 Plan: Stop After Blocked Audio-Only Cross-Platform Prep

- [x] Diagnose the completed 1-day run's blocked full-audio failure.
- [x] Make blocked cross-platform prep abort the shell workflow immediately.
- [x] Stop rebuilding repost manifests merely because they are below
  `max_reposts`.
- [x] Add regression coverage and run focused verification.

### Review Notes

- The run correctly produced `blocked_full_vod_audio_fingerprints`, but
  `train_jynxzi_vod.sh` only passed `--fail-on-insufficient`, so the workflow
  continued and built negative-only outcome datasets before failing later.
- The training script now passes `--fail-on-blocked`, so blocked audio-only
  cross-platform prep exits immediately and does not continue into dataset
  construction.
- Repost manifests no longer rebuild just because cached rows are below
  `max_reposts`. `max_reposts` is now only a cap for explicit rebuilds; rebuild
  still happens when rows are missing, below `min_reposts`, or refresh is
  requested.
- Verification passed: focused pytest -> `39 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2904` nodes, `7736` edges,
  `68` communities.

## 2026-05-08 Plan: Remove Old MP4 Cross-Platform Caches

- [x] Inspect old MP4 cache categories and current container state.
- [x] Delete obsolete `coarse_*` scoped-window MP4 cache directories.
- [x] Delete leftover full-VOD audio temp `.m4a` files if present.
- [x] Verify freed space and remaining cache state.

### Review Notes

- Removed 203 obsolete `scoped_windows/coarse_*` directories from
  `backend/tmp/vod_feature_cache`.
- These files were root-owned from Docker, so normal `rm` failed with
  permission errors. Deletion succeeded via a one-shot Docker container mounted
  on the repo.
- Removed stale cross-platform generated artifacts that were derived from the
  old MP4-window matches:
  `jynxzi_cross_platform_vods.jsonl`,
  `jynxzi_cross_platform_fingerprint_rows.jsonl`,
  `jynxzi_cross_platform_fingerprint_summary.json`, and
  `jynxzi_cross_platform_fingerprint_diagnostics.jsonl`.
- Verified `coarse_*` directories remaining: `0`.
- Verified leftover `full_vod_audio/source.m4a` files: `0`.
- `backend/tmp/vod_feature_cache` is now `31G`; filesystem free space is
  `885G`.

## 2026-05-08 Plan: Enforce Full-VOD Audio In Seed Cross-Platform Pass

- [x] Confirm why the current seed pass still used scoped MP4 windows.
- [x] Change the training orchestrator so full-VOD audio mode is active for
  all cross-platform prepare calls, not only the second pass.
- [x] Add regression coverage for the shell wiring.
- [x] Run focused verification and document the result.

### Review Notes

- Root cause: `train_jynxzi_vod.sh` only passed
  `--full-vod-audio-fingerprints` when `refresh_vods=1`, so the seed pass
  explicitly passed `--no-full-vod-audio-fingerprints` and rebuilt from old
  scoped MP4 windows.
- The shell wrapper now passes `--full-vod-audio-fingerprints` whenever
  `CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS=1`, including the seed pass.
- Added a regression test so the shell script cannot reintroduce the
  `refresh_vods && full_audio` coupling.
- Verification passed: focused pytest -> `37 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2902` nodes, `7731` edges,
  `67` communities.

## 2026-05-08 Plan: Fail Closed On Failed Full-VOD Audio Fingerprints

- [x] Diagnose the 1-day smoke-test failure path from the provided logs.
- [x] Stop empty full-audio VOD refreshes from overwriting usable manifests or
  cross-platform outcome rows.
- [x] Add regression tests for failed full-audio backfill and empty-input
  preservation.
- [x] Run focused tests, syntax checks, Graphify, and document the result.

### Review Notes

- The smoke test selected one VOD for full-audio fingerprinting, failed that
  one VOD, wrote a refreshed VOD manifest with `0` rows, then overwrote
  `jynxzi_cross_platform_fingerprint_rows.jsonl` with `0` rows. That removed
  all positives before training.
- Full-audio fingerprint builds now report
  `failed_no_full_vod_audio_fingerprints` when selected VODs exist but no
  cached/built full-audio fingerprints remain.
- `prepare_cross_platform_fingerprint_rows` now blocks with
  `blocked_full_vod_audio_fingerprints` before matching when that happens.
  Existing VOD manifests and existing cross-platform outcome rows are preserved.
- Empty-input and below-minimum match results now preserve the last valid
  output artifact when one exists, instead of writing an empty outcome file.
- Verification passed: focused pytest -> `36 passed`; `py_compile` passed;
  `bash -n backend/ml/train_jynxzi_vod.sh` passed; scoped `git diff --check`
  passed.
- Graphify rebuilt successfully: `2899` nodes, `7728` edges,
  `68` communities.

## 2026-05-07 Plan: Coarse Audio Tiling fuer Cross-Platform-Matching

Ziel: Cross-platform Reposts sollen nicht nur dann matchen koennen, wenn sie
zufaellig in einem Twitch-Top-Clip scoped window liegen. Der Rematch soll
optional grobe Audio-Tiles ueber ausgewaehlte VODs bauen und in das
VOD-Manifest aufnehmen.

- [x] Coarse-Tiling Parameter und Cache-Layout im Fingerprint-Prepare
  einfuehren.
- [x] Aus selected targeted-backfill VODs und Label-Manifest grobe VOD-Tiles
  erzeugen.
- [x] Jynxzi-Trainingsscript so verdrahten, dass der zweite Rematch diese
  Tiles nutzt.
- [x] Tests/Syntax/Graphify ausfuehren.
- [x] Ergebnis dokumentieren.

### Review

- `prepare_cross_platform_fingerprint_rows.py` hat jetzt einen optionalen
  `--coarse-vod-tiling` Pfad. Er liest die selected targeted-backfill VODs,
  nimmt das Twitch-Label-Manifest nur zur VOD-/Horizon-Bestimmung und erzeugt
  grobe `coarse_vod_audio_tile` scoped windows im bestehenden
  `vod_feature_cache`.
- Defaults fuer den Trainingsrun: `30s` Tile-Laenge, `600s` Stride,
  maximal `24` Tiles pro VOD. Damit deckt ein selected VOD grob bis zu vier
  Stunden ab, ohne den kompletten VOD voll zu analysieren.
- Der zweite Cross-platform Rematch aktiviert das Tiling nur bei
  `refresh_vods=1`. Seed-Reuse bleibt billig; der teure breitere Suchraum
  entsteht erst im strikten Rematch nach targeted backfill.
- Die groben Tiles landen im normalen VOD-Manifest, weil der Manifest-Builder
  alle `scoped_windows/*/source.metadata.json` einliest. Der bestehende
  Audio-Fingerprint-Matcher kann sie dadurch unveraendert vergleichen.
- Verifikation: `py_compile` erfolgreich; `bash -n train_jynxzi_vod.sh`
  erfolgreich; fokussierte Tests
  `test_prepare_cross_platform_fingerprint_rows.py`,
  `test_training_workflow_state.py`, `test_targeted_backfill.py`,
  `test_train_vod_ranker.py` -> `60 passed`; scoped `git diff --check`
  sauber.
- Graphify-Rebuild erfolgreich:
  `2876` Nodes, `7638` Edges, `68` Communities.

## 2026-04-27 Plan: ML Model Verbesserungsplan (basierend auf Eval-Analyse)

### Kontext
Modell: Logistic Regression, 71 Features, 244 Trainingsrows (64 positiv).
Kernprobleme: kleines Goldlabeled-Dataset, Train-Holdout-Gap, twitch_proxy schadet, slow_open_penalty-Confound.

---

### Priorität 1 — Mehr Gold-Labels (kritischer Engpass)

**Ziel:** Von 48 auf ≥150 gold_eval-Trainingsrows in 4 Wochen.

- [ ] **P1.1 — Batch-Annotierung starten (laufend)**
  - 5–8 neue VODs pro Woche in `jynxzi_gold_eval_manifest.csv` aufnehmen
  - `build_gold_eval_queue.py --manifest ... --storage-dir ... --out jynxzi_gold_eval_queue` ausführen
  - Queue in Google Sheets öffnen, `gold_grade` 0–3 blind annotieren (ohne Modell-Scores)
  - Importieren via `import_gold_eval_labels.py`

- [x] **P1.2 — Holdout-VODs für Annotierung priorisieren**
  - Die 35 Holdout-VODs aus `jynxzi_training_workflow_state.json` sind ungenutzte Labelpotenziale
  - 10 davon in den nächsten Annotierungsbatch aufnehmen (ohne sie aus dem Holdout zu entfernen bis genug Train-Daten)

- [x] **P1.3 — Erfolgsmetrik festlegen**
  - Retrain wenn gold_eval-Rows ≥ 80 (Zwischenmeilenstein) und ≥ 150 (Hauptmeilenstein)
  - Erwarteter Effekt: MRR-Holdout sollte von 0.471 auf ≥ 0.55 steigen

---

### Priorität 2 — twitch_label_proxy-Gewichtung reduzieren

**Ziel:** Source-Bucket `twitch_label_proxy` schadet dem Modell (MRR-Delta −0.046). Gewicht senken oder ausschließen.

- [x] **P2.1 — Diagnose vertiefen**
  - In `train_ranker.py:_sample_weight()` ist `twitch_clip_proxy`-Tier aktuell auf 1.15 gesetzt (leicht übergewichtet)
  - Mit einem temporären Run testen: Gewicht auf 0.5 senken und MRR auf Holdout vergleichen
  - Kommando: `ranker_gold_train_weight=2.0, ranker_hard_negative_weight=4.0` beibehalten, nur `twitch_clip_proxy`-Multiplikator anpassen

- [x] **P2.2 — Code-Änderung in `train_ranker.py`**
  - `"twitch_clip_proxy": 1.15` → `"twitch_clip_proxy": 0.5` in `_sample_weight()` (Zeile ~1721)
  - Retrain durchführen, gold_holdout MRR und nDCG@5 vergleichen
  - Wenn MRR auf Holdout steigt: Wert dauerhaft übernehmen

- [ ] **P2.3 — Alternativ: Proxy-Tier aus Training ausschließen**
  - Falls Gewichtsreduktion nicht ausreicht: `--no-proxy-tier`-Flag zu `train_vod_ranker.py` hinzufügen
  - Nur sinnvoll wenn gold_eval-Rows ≥ 80 (sonst zu wenig Daten)

---

### Priorität 3 — slow_open_penalty Confound aufklären

**Ziel:** Feature hat Gewicht +1.14 — das bedeutet "mehr slow_open = besser", was semantisch falsch ist.

- [x] **P3.1 — Korrelationsanalyse**
  - Feature-Korrelation zwischen `slow_open_penalty` und `chapter_local_density`, `duration`, `payoff_score` berechnen
  - In Python: `pd.DataFrame([r["features"] for r in training_rows]).corr()[["slow_open_penalty"]]`
  - Vermutung: `slow_open_penalty` korreliert positiv mit `chapter_local_density` (dichte Chapters = mehr slow-open-erkannte Clips)

- [x] **P3.2 — Entscheidung treffen**
  - Option A: Feature zu `excluded_feature_names` verschieben (konservativ, einfach)
  - Option B: Feature-Semantik im Feature Builder invertieren (wenn Bedeutung klar)
  - Option C: Feature umbenennen und Polarität dokumentieren
  - Empfehlung: erst Korrelation messen, dann entscheiden

- [x] **P3.3 — Retrain mit angepasstem Feature-Set**
  - Wenn Feature ausgeschlossen: erwarten dass `profile_fit_score` und `setup_depth_score` mehr Gewicht aufnehmen

---

### Priorität 4 — Hard Negatives systematisch ausbauen

**Ziel:** Aktuell nur 6 Hard-Negatives aus Holdout. Ziel: 20–30 pro Trainingslauf.

- [x] **P4.1 — Rollout Review Queue nutzen**
  - `jynxzi_rollout_review_queue.csv` (63 Rows, 26 VODs) manuell reviewen
  - False Positives identifizieren (Modell sagt gut, aber tatsächlich schlecht)
  - Diese als Hard Negatives in den nächsten Trainingslauf einbringen

- [x] **P4.2 — Pipeline für strukturiertes FP-Feedback**
  - Nach jedem Trainingslauf: `diagnose_ranker_holdout.py` ausführen
  - Alle False Positives (max 100) via `export_ranker_hard_negatives.py` exportieren
  - Direkt in `jynxzi_ranker_hard_negative_train_rows.jsonl` mergen

- [ ] **P4.3 — Hard-Negative-Gewicht evaluieren**
  - Aktuell: `ranker_hard_negative_weight=4.0` (label_class=hard_negative hat base_weight 1.8, dann x4 = 7.2 effektiv)
  - Testen ob 2.0 oder 3.0 besser ist wenn mehr Hard Negatives vorhanden

---

### Priorität 5 — Gradient Boosting als Alternative evaluieren (nach P1)

**Voraussetzung:** ≥ 150 gold_eval-Trainingsrows.

- [x] **P5.1 — Model-Selection-Metrik verstehen**
  - In `train_ranker.py`: `composite_validation_score` identifizieren und verstehen warum LR über GB gewählt wurde
  - GB hat auf Validation MRR 0.885 vs LR 0.846 — warum wurde LR bevorzugt?

- [ ] **P5.2 — Expliziter GB-Vergleich**
  - Wenn genug Daten: beide Modelle auf separaten Seeds evaluieren und Holdout-MRR vergleichen
  - GB hat potenzielle Overfitting-Risiken bei kleinem Datensatz — daher erst bei >150 positiv

---

### Erfolgsmetriken (Zielzustand nach 4 Wochen)

| Metrik | Aktuell | Ziel |
|---|---|---|
| Gold-Holdout MRR | 0.471 | ≥ 0.55 |
| Gold-Holdout nDCG@5 | 0.475 | ≥ 0.55 |
| MRR_p25 (volle Eval) | 0.0 | > 0.0 |
| gold_eval Training-Rows | 48 | ≥ 150 |
| Hard Negatives | 6 | ≥ 25 |

---

### Umsetzung 2026-04-27

- P1.1 bleibt als laufender manueller Annotierungsprozess offen; die technischen Vorarbeiten sind erledigt, aber blindes Human-Rating und Re-Import kann ich nicht automatisch korrekt ausführen.
- P1.2 ist umgesetzt über `backend/ml/artifacts/jynxzi_gold_holdout_annotation_manifest.csv` mit 10 priorisierten Holdout-VODs.
- P1.3 ist als Zielmetrik im Plan fixiert und in den erzeugten Eval-Artefakten prüfbar: Gold-Holdout MRR, nDCG@5, `mrr_p25`, Gold-Train-Rows und Hard-Negative-Anzahl.
- P2.1/P2.2 sind umgesetzt: `twitch_clip_proxy`-Supervision wird in `train_ranker._sample_weight()` von 1.15 auf 0.5 reduziert.
- P2.3 bleibt bewusst offen: Proxy-Tier komplett auszuschließen ist erst sinnvoll, wenn mindestens 80 Gold-Rows im Training stehen; aktuell sind es 48.
- P3.1 ist umgesetzt über `backend/ml/analyze_ranker_feature_confounds.py` und `backend/ml/artifacts/jynxzi_ranker_feature_confounds.json`.
- P3.2/P3.3 sind umgesetzt: `slow_open_penalty` wird konservativ aus dem supervised Ranker-Feature-Set ausgeschlossen; das aktuelle Ranker-Artefakt wurde neu trainiert.
- P4.1 ist operationalisiert über `backend/ml/artifacts/jynxzi_rollout_review_queue.csv` mit 63 manuellen Review-Kandidaten, `review_required=true`, `auto_publish_allowed=false`.
- P4.2 ist umgesetzt über `backend/ml/refresh_ranker_hard_negative_artifacts.py`, das Holdout-Diagnose und Hard-Negative-Export in einem Schritt erneuert.
- P4.3 bleibt offen, weil aktuell weiterhin nur 6 exportierte Hard Negatives vorliegen und alle aus Holdout stammen; die Training-Pipeline lässt Holdout-origin Rows wegen Leakage-Schutz nicht ins Training.
- P5.1 ist umgesetzt über `artifact.model_selection`: GB/LightGBM werden unter 400 Trainingsrows nicht promotet, auch wenn LightGBM auf Validation besser ist.
- P5.2 bleibt offen, weil die Voraussetzung `>=150` Gold-Train-Rows nicht erfüllt ist.

### Aktueller Ergebnisstand

- Re-train ohne neue VOD-Downloads abgeschlossen: `jynxzi_ranker.json` ist Logistic Regression, `feature_schema_version=4`, `serving_threshold=5.52`.
- `slow_open_penalty` ist in `excluded_feature_names`; Confound-Report bestätigt Positiv-Mittelwert `0.719625` vs Negativ-Mittelwert `0.654`.
- Gold full eval: accuracy `0.779`, precision `0.649`, recall `0.792`, F1 `0.713`, MRR `0.645`, nDCG@5 `0.647`.
- Gold holdout: accuracy `0.783`, precision `0.667`, recall `0.762`, F1 `0.711`, MRR `0.471`, nDCG@5 `0.475`.
- Proxy eval: accuracy `0.877`, precision `0.774`, recall `0.75`, F1 `0.762`, MRR `0.81`, nDCG@5 `0.785`.
- Hard-negative export bleibt bei 6 Rows, alle `hard_negative_original_split=holdout`; diese werden vom Training weiterhin ausgeschlossen, um Evaluation nicht zu verfälschen.



## 2026-04-27 Plan: Conservative Rollout Review Queue

- [x] Read the existing queue/scoring flow and Graphify backend overview.
- [x] Add a manual-review rollout queue that never auto-publishes candidates.
- [x] Exclude existing gold rows and gold-holdout VODs from rollout review candidates.
- [x] Add focused tests for holdout exclusion, per-VOD top-k selection, and guardrail metadata.
- [x] Generate the current Jynxzi rollout review artifacts.
- [x] Run verification and rebuild backend Graphify after backend code edits.
- [x] Document the result and remaining rollout constraints.

### Review Notes

- Added `backend/ml/build_rollout_review_queue.py` as a separate conservative rollout queue, so active gold-eval sampling keeps its existing semantics.
- The queue scores candidates with the current trained ranker, excludes exact existing gold candidates/windows, and excludes any VOD that appears in the gold holdout split.
- Guardrails are explicit in CSV, sidecar, and summary: `review_required=true` and `auto_publish_allowed=false`.
- Generated current Jynxzi rollout review artifacts with `top_k_per_vod=10`, `max_total=200`, and `min_gap_sec=20`.
- Current queue output has `63` rows across `26` VODs; no queue row overlaps a gold-holdout VOD, and the maximum per-VOD count is `10`.
- The current scoring dataset already excludes holdout VODs, so `excluded_gold_holdout_vod_rows=0`; the artifact still records all `35` holdout VOD IDs and enforces the exclusion if they appear later.
- Verification passed: syntax compile for the new script/test.
- Verification passed: `backend/tests/test_build_rollout_review_queue.py` and `backend/tests/test_build_active_gold_eval_queue.py`.
- Verification passed: `git diff --check`.
- Rebuilt backend Graphify with the uv tool interpreter: `2142` nodes, `5558` edges, `46` communities.
- The generated rollout review artifacts are currently ignored by `.gitignore` like other new artifact names; they exist locally under `backend/ml/artifacts/`.

## 2026-04-26 Plan: Clip-First Top-Viewed Scoped Training

- [x] Trace the current top-viewed scoped path and full-VOD materialization boundary.
- [x] Add a clip/scoped media materialization path that avoids full-VOD download for all-top-viewed VOD groups.
- [x] Preserve VOD-chat alignment by reusing `vod_id` and rebased chat windows.
- [x] Add focused tests proving the top-viewed path does not call full-VOD download.
- [x] Run syntax checks, targeted tests, and scoped diff checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Document behavior, caveats, and current-run impact.

### Review Notes

- Added `downloader.download_range(...)`, which uses yt-dlp range options and bypasses the persistent Twitch full-VOD cache.
- Changed all-top-viewed `window_scoped_top_viewed` VOD groups to skip `_materialize_vod` and build each scoped source directly from the needed VOD time range.
- Serial prefetch and parallel download scheduling now skip Full-VOD futures for all-top-viewed scoped groups.
- The scoped source metadata version is now `2`, and the raw VOD row fingerprint includes that version so future checkpoints reflect the new source behavior.
- Chat alignment is preserved: scoped sources still use the canonical `vod_id`, VOD chat cache, target offsets, and rebased `chat.local.json`.
- Current label impact: `24` scoped-only VODs containing the `47` top-viewed labels will use the range/scoped path; the remaining `27` VODs still require Full-VOD processing because they contain normal training labels.
- Verification passed: `python3 -m py_compile` for touched modules/tests.
- Verification passed: full `backend.tests.test_downloader`.
- Verification passed: full `backend.tests.test_build_twitch_training_dataset`.
- Verification passed: scoped `git diff --check`.
- Rebuilt backend Graphify with the uv tool interpreter: `2097` nodes, `5406` edges, `52` communities.
- Caveat: already-running training containers use the code baked/mounted into that container at start; this change applies to the next run or a restarted container using the updated workspace/image.

## 2026-04-26 Plan: Inspect Current Training Run Downloads

- [x] Inspect running container, command, and recent training logs.
- [x] Separate cache hits from real external downloads for VOD, scoped windows, chat, and model assets.
- [x] Cross-check artifact/cache state for what changed in this run.
- [x] Summarize what is being re-downloaded and why.

### Review Notes

- Current container `boring_keller` started `2026-04-25T22:18:24Z` and is running `train_vod_ranker.py` for `2026-03-15` through `2026-03-28` with `--skip-covered --merge-output`.
- The run uses `FEATURE_BUILD_VERSION=8`, `_CHAT_ENRICHMENT_FINGERPRINT_VERSION=4`, and `TWITCH_DOWNLOAD_CACHE_TTL_SEC=0`.
- Checkpoint status inside the running container: `51` VODs in the label file, `1` currently valid VOD checkpoint, `45` invalid old checkpoints, and `5` missing checkpoints.
- Because the raw VOD fingerprint changed, old VOD row checkpoints are being rebuilt; this requires full VOD video materialization again for each invalid/missing VOD.
- Confirmed active real Twitch media downloads: `2711274372` was downloaded and checkpointed; `2712128863` was prefetched and being processed; `2712736206` is actively downloading via `aria2c`.
- Existing chat cache files are present for all `51` VODs, but only `46` currently have complete chat coverage; `2720629593`, `2724120511`, `2731588015`, `2732418387`, and `2748783535` may trigger targeted chat refill/download for missing offsets.
- Existing feature cache is mixed: `28/51` VODs have full-VOD audio, `27/51` have full-VOD transcripts, and all have chat cache; missing audio/transcript may still be recomputed from the re-materialized video.
- Current storage state is healthy: mount has about `847G` free; `backend/tmp/storage` is about `50G`, mostly active/stale VOD materialization directories.
- There is a stale-looking `twitch_vod_prefetch_2720629593` from before this container start; active current directories are `twitch_vod_prefetch_2712128863` and `twitch_vod_prefetch_2712736206`.

## 2026-04-25 Plan: Improve Scoped Chat Features

- [x] Trace how rebased scoped chat becomes segment and candidate chat features.
- [x] Add a density/volume fallback so chat presence is represented even without detected spikes.
- [x] Add focused tests for non-spike scoped chat coverage.
- [x] Run targeted tests and syntax checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Document the behavioral impact and remaining caveats.

### Review Notes

- Scoped chat now contributes segment-level `chat_message_count`, `chat_message_rate`, `chat_density_score`, and `chat_emote_score` before spike tagging, so normal sustained chat is no longer invisible.
- Candidate features now use the maximum of spike and density as chat activity, while keeping `chat_spike_coverage` separate from the new `chat_activity_coverage`.
- Added candidate-level `chat_message_rate_peak`, `chat_message_rate_mean`, `chat_density_peak`, and `chat_density_mean`.
- Bumped `FEATURE_BUILD_VERSION` to `8` and `_CHAT_ENRICHMENT_FINGERPRINT_VERSION` to `4`, so VOD training rows are rebuilt with the new chat feature schema.
- Verified a real scoped cache sample: `52` segments, `46` with density, `0` with spike; candidate features reported `has_chat_features=True`, `chat_peak=1.0`, `chat_activity_coverage=0.694`, and `chat_spike_coverage=0.0`.
- Verification passed: `python3 -m py_compile` for touched backend modules and tests.
- Verification passed: `backend.tests.test_chat_analyzer` and `backend.tests.test_feature_builder` with offline HF settings.
- Verification passed: selected scoped-chat/checkpoint tests from `backend.tests.test_build_twitch_training_dataset`.
- Rebuilt backend Graphify with the uv tool interpreter: `2088` nodes, `5378` edges, `53` communities.
- Caveat: existing static gold eval row artifacts remain on their old feature payload until they are refreshed or re-imported from a new feature-backed run.

## 2026-04-25 Plan: Import Feature-Backed Top-Viewed Positives Into Gold Eval

- [x] Identify the 47 newly materialized `window_scoped_top_viewed` positive rows.
- [x] Convert them into gold-positive rows using the existing gold-eval schema and dedupe keys.
- [x] Merge them into canonical gold eval artifacts without duplicating the earlier 3 imported rows.
- [x] Rebuild the gold eval dataset and summaries.
- [x] Verify row counts, split distribution, duplicate keys, and feature completeness.

### Review Notes

- Imported `47/47` newly feature-backed `window_scoped_top_viewed` positive-like rows as gold grade `3`; no new rows were skipped as duplicates.
- Canonical `jynxzi_gold_eval_rows.jsonl` now has `222` rows: `76` gold positives and `146` gold negatives.
- Top-viewed Twitch gold rows now total `50`: the earlier `3` plus the new `47`.
- The new batch spans `24` VODs and keeps the source label mix: `29` positive, `17` weak_positive, and `1` viral_positive.
- Deterministic gold split added `34` train rows and `13` holdout rows; total gold split is now `162` train and `60` holdout.
- Rebuilt `jynxzi_gold_eval_dataset.jsonl` and `jynxzi_gold_eval_summary.json`; summary now reports `222` eval rows and `54` eval groups.
- Verification passed: no duplicate queue IDs, no duplicate clip IDs, no missing features, and no missing `features.candidate_id`.
- The full gold set still has one pre-existing duplicate `candidate_id` from frontend rows with different queue IDs and grades; the imported batch has zero duplicate queue IDs or candidate IDs.
- Verification passed: `backend.tests.test_gold_eval_workflow` and `backend.tests.test_build_eval_dataset`.
- `git diff --check` still reports trailing whitespace in the pre-existing `jynxzi_ranker_holdout_diagnosis.csv`; scoped diff-check on the files touched by this import passed.

## 2026-04-25 Plan: Analyze Latest Training Run Failure

- [x] Locate latest run logs, workflow state, and artifact timestamps.
- [x] Quantify progress before failure: VODs processed, checkpoints, rows, and caches.
- [x] Identify fatal error and separate it from non-fatal warnings.
- [x] Check whether artifacts were finalized or only intermediate state changed.
- [x] Summarize next action and rerun impact.

### Review Notes

- Latest persisted run artifacts were written at `2026-04-25 20:24-20:25`; `local_train_summary.json` reports `training_return_code=0`, so the earlier no-space failure was not the final state.
- The workflow produced final artifacts for `jynxzi` over `2026-03-15` through `2026-03-28`; `jynxzi_ranker.json`, eval JSONs, dataset JSONL, and workflow state were updated.
- The final VOD eval dataset has `556` rows: `108` positives and `448` negatives across `64` eval groups.
- The new top-viewed scoped path contributed `139` final rows from `24` VODs: `47` positive-like rows and `92` boundary near-miss negatives.
- Checkpoints exist for `46` VODs, including the post-cleanup VODs `2752903388`, `2753667907`, `2754503460`, and `2756055866`; total checkpoint rows are `11375`.
- Chat downloads completed for the latest VODs with `coverage_status=complete`, but final scoped top-viewed rows only have `2/139` rows with non-zero chat spike features; full-VOD rows have `174/287`.
- The trained model was rejected by the quality gate with `validation_accuracy_below_minimum`; validation accuracy was `0.591`, precision `0.519`, recall `0.737`, F1 `0.609`, and precision@5 `0.485`.
- Threshold calibration was skipped because the artifact was rejected; serving threshold in artifact health is `-1.449484`.
- Next recommendation: treat the run as successful for feature materialization, but not successful for model promotion; inspect whether scoped chat should use density/volume features, not only spike features, before expecting chat to help those top-viewed examples.

## 2026-04-25 Plan: Prevent Twitch VOD Download Cache Disk Exhaustion

- [x] Confirm the failed run root cause and current disk usage.
- [x] Remove only regenerable full-VOD download cache data and failed prefetch fragments.
- [x] Change local/default Twitch full-VOD download cache behavior to avoid unbounded growth.
- [x] Run focused downloader/config checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Document the outcome and rerun caveat.

### Review Notes

- Root cause: the failed VOD-first run hit `[Errno 28] No space left on device`; the host mount was `100%` full, with `backend/tmp/storage/_download_cache` at about `934G` and the failed `twitch_vod_prefetch_2752903388` at about `20G`.
- Removed only regenerable full-VOD download cache data and the failed prefetch fragments via a short-lived root container because the files were Docker/root-owned.
- Disk recovered from `0B` available to about `902G` available; `backend/tmp` is now about `24G`, mostly `vod_feature_cache`.
- Changed `TWITCH_DOWNLOAD_CACHE_TTL_SEC` from `-1` to `0` in `backend/.env`, `backend/.env.example`, and the config default so Twitch full-VOD downloads are not cached indefinitely.
- Verification passed: `python3 -m py_compile backend/config.py backend/services/downloader.py`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_downloader backend.tests.test_config`.
- Rebuilt backend Graphify with the uv tool interpreter: `2069` nodes, `5350` edges, `46` communities.
- Caveat: the failed training run stopped; it needs to be restarted after the cleanup/config change.

## 2026-04-25 Plan: Make Gated Pyannote Diarization Non-Fatal

- [x] Inspect the new Hugging Face gated-model failure path.
- [x] Treat pyannote load/inference failures as optional diarization misses instead of training failures.
- [x] Add a regression test for gated/forbidden pyannote access.
- [x] Run targeted transcriber tests and syntax checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Document the behavior and caveat.

### Review Notes

- Root cause: Hugging Face accepted the `token=` kwarg but returned `403 Forbidden` for the gated `pyannote/speaker-diarization-3.1` model, so optional diarization aborted the whole VOD training run.
- Updated `_diarize(...)` to fail open when pyannote pipeline load or inference fails: it logs `transcriber_diarization_unavailable` with a classified reason and returns the Whisper segments unchanged.
- Added `hf_gated_model_access`, `hf_auth`, `pipeline_download`, and generic `pipeline_error` reason classification for compact logs.
- Added a regression test that simulates the gated `403` access failure and verifies training receives the original `UNKNOWN` speaker segments.
- Verification passed: `python3 -m py_compile backend/services/transcriber.py backend/tests/test_transcriber.py`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_transcriber backend.tests.test_transcriber_progress`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset`.
- Rebuilt backend Graphify with the uv tool interpreter: `2069` nodes, `5350` edges, `46` communities.
- Caveat: diarization-derived speaker labels are skipped until the HF token has accepted/access to the gated pyannote model, but transcript and all non-speaker features continue.

## 2026-04-25 Plan: Fix Pyannote Diarization API Compatibility

- [x] Inspect the current diarization call and installed dependency contract.
- [x] Make `Pipeline.from_pretrained` compatible with pyannote 3.x and 4.x token kwargs.
- [x] Add a focused regression test for the new kwarg fallback.
- [x] Run targeted transcriber tests and quick syntax checks.
- [x] Rebuild backend Graphify after backend code edits.
- [x] Record the failure cause and result.

### Review Notes

- Root cause: `backend/pyproject.toml` allows `pyannote.audio>=4.0.0,<5.0.0`, and the installed `Pipeline.from_pretrained` signature uses `token=...`; the old code still passed `use_auth_token=...`.
- Updated `services/transcriber.py` to load pyannote with `token=...` first, with a fallback to `use_auth_token=...` for older pyannote releases.
- Added regression tests for both the current token kwarg path and the legacy fallback path.
- Verification passed: `python3 -m py_compile backend/services/transcriber.py backend/tests/test_transcriber.py`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_transcriber backend.tests.test_transcriber_progress`.
- Verified local installed signature: `(checkpoint, revision=None, hparams_file=None, token=None, cache_dir=None)`.
- Rebuilt backend Graphify with the uv tool interpreter: `2063` nodes, `5343` edges, `45` communities.
- Added the lesson to verify optional ML dependency signatures before long training runs.

## 2026-04-25 Plan: Window-Scoped Top-Viewed Twitch Feature Build

- [x] Inspect current full-VOD materialization, audio, transcript, chat, and window feature cache flow.
- [x] Add a scoped source mode for manually confirmed top-viewed Twitch labels.
- [x] Make scoped mode cut only the expanded label window before audio/transcript/window features.
- [x] Preserve normal full-VOD training behavior for regular labels and mixed VODs.
- [x] Add focused tests for scoped-materialization decisions and offset mapping.
- [x] Run targeted tests and rebuild backend Graphify after backend code edits.
- [x] Document the result and any remaining caveats.

### Review Notes

- Added a `window_scoped_top_viewed` feature context for positive labels from `manual_gold_review_source=top_viewed_twitch_30d`.
- For eligible VOD label groups, the workflow now cuts the expanded label window into `scoped_windows/<digest>/source.mp4` and builds audio, transcript, chat enrichment, and window modalities against that short source.
- Chat still uses the root VOD chat cache, but scoped windows write a local rebased chat file so chat offsets align with the clipped source.
- Regular labels and mixed VOD groups still use the existing full-VOD feature context to avoid changing hard-negative and normal training behavior.
- Feature provenance now records `vod_feature_source_kind`, making scoped rows auditable in downstream artifacts.
- Verification passed: `python3 -m py_compile backend/ml/build_twitch_training_dataset.py backend/tests/test_build_twitch_training_dataset.py`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_build_twitch_training_dataset`.
- Verification passed: `PYTHONPATH=/home/antonio/ai_clipping/backend /home/antonio/ai_clipping/backend/.venv/bin/python -m unittest backend.tests.test_train_vod_ranker`.
- Verification passed: `git diff --check`.
- Rebuilt backend Graphify with the uv tool interpreter: `2057` nodes, `5337` edges, `40` communities.
- Caveat: this removes the expensive full-VOD transcription/window-feature bottleneck for the confirmed top-viewed positives, but it still needs the source VOD available so the short window can be cut.

## 2026-04-25 Plan: Wire Top-Viewed Backlog Into Next Training Run

- [x] Confirm the next training run reads the canonical Jynxzi Twitch label file.
- [x] Merge the 47 feature-backlog positive labels into the canonical label file without duplicates.
- [x] Verify the canonical label file now contains those VOD/window labels and will trigger retraining.
- [x] Document remaining caveat: they become gold only after feature-backed import.

### Review Notes

- `backend/ml/train_jynxzi_vod.sh` reads `backend/ml/artifacts/jynxzi_twitch_clip_labels.jsonl`; `MERGE_OUTPUT=1` is the default, so the collect step preserves existing labels on the normal path.
- Merged all `47` feature-backlog top-viewed positives from `backend/ml/artifacts/jynxzi_top_viewed_30d_gold_positive_needs_features.jsonl` into `backend/ml/artifacts/jynxzi_twitch_clip_labels.jsonl`.
- Canonical label file now has `13604` rows and `86` positive-like labels, including `47` rows marked with `manual_gold_review_source=top_viewed_twitch_30d`.
- Added backlog labels span `24` VODs: `29` positive, `17` weak_positive, and `1` viral_positive.
- Verification found `0` duplicate label keys after merge.
- `training_workflow_state.should_retrain(...)` returned `True`, so the next normal run should not skip because of unchanged inputs.
- Caveat: these `47` will be loaded as Twitch/VOD proxy labels during feature build; they become canonical gold rows only after the run produces matching feature rows and we import them into `jynxzi_gold_eval_rows.jsonl`.

## 2026-04-25 Plan: Add Top-Viewed Jynxzi Twitch Clips As Gold Positives

- [x] Collect or load Jynxzi Twitch clips for the last 30 days into a separate artifact path.
- [x] Select top-viewed positive clip windows and dedupe against existing gold rows.
- [x] Build a labeled gold-positive CSV plus features sidecar for import.
- [x] Import into gold-eval artifacts with a preserved train/holdout split.
- [x] Verify row counts, grade distribution, and no duplicate gold keys.
- [x] Document result and next training step.

### Review Notes

- Collected `12766` Jynxzi Twitch clip label rows for `2026-03-26` through `2026-04-25` into `backend/ml/artifacts/jynxzi_top_viewed_30d_twitch_clip_labels.jsonl`.
- Selected `51` user-confirmed positive-like top-viewed clip windows: `33` positive, `17` weak_positive, and `1` viral_positive.
- Only `4/51` positive windows currently overlap usable feature rows in the existing VOD training/eval artifacts; `1` of those was already present in gold eval.
- Imported the `3` new feature-backed top-viewed positives as gold grade `3`; canonical `jynxzi_gold_eval_rows.jsonl` now has `175` rows, `29` gold positives, and `146` gold negatives.
- Gold split after merge is `128` train and `47` holdout; the new positives add `2` train rows and `1` holdout row.
- Verification found `0` duplicate gold holdout keys, `0` duplicate queue IDs, and `0` gold rows missing features.
- Wrote the remaining `47` confirmed positive clip labels that need feature materialization to `backend/ml/artifacts/jynxzi_top_viewed_30d_gold_positive_needs_features.jsonl`.
- Targeted verification passed: `python -m unittest backend.tests.test_gold_eval_workflow backend.tests.test_build_eval_dataset`.

### Next Step

- Run a focused VOD feature build/training run for `backend/ml/artifacts/jynxzi_top_viewed_30d_gold_positive_needs_features.jsonl`, then import the newly feature-backed windows into gold eval as the second batch.

---

## 2026-06-08 Phase 3 — Chat-Signal von Grund auf neu + Dead-Code-Bereinigung

### Teil D — Dead-Code löschen
- [ ] 19 tote Service-Module + Tests löschen
- [ ] Tote Config-Flags entfernen
- [ ] services/__init__.py prüfen

### Teil A — Neues Chat-Signal-Modul
- [ ] backend/services/chat_signal.py (GraphQL download + spike + emotion)
- [ ] backend/tests/test_chat_signal.py

### Teil B — RichSegment schlankmachen
- [ ] 12 chat_*-Felder → 3 (chat_spike_score, chat_message_count, chat_category)
- [ ] Tote audio/visual-Felder entfernen
- [ ] to_dict/from_dict/format_transcript aktualisieren

### Teil C — Verdrahtung + LLM-Prompt
- [ ] pipeline.py: chat_signal import + wiring
- [ ] llm_moment_finder.py: TranscriptSegment chat fields + format_transcript_for_prompt + preamble
- [ ] moment_finder.py: rich_segments_to_transcript_segments chat fields durchreichen

### Verifikation
- [ ] Import-Check: python -c "import pipeline, main"
- [ ] Tests: python -m pytest backend/tests/ -q
- [ ] Graphify update
