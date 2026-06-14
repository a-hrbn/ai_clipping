# 1. First-party-Outcome-Flywheel statt Content-Ranking

Status: Accepted — 2026-06-14

## Kontext

Das Produkt ist ein auf *einen* Streamer (Jynxzi) spezialisierter Autopilot-Viral-Clip-
Finder. Der angenommene Moat war: den Content des Streamers besser verstehen als
generische Finder (OpusClips) und so die viralen Momente genauer erkennen.

Über ~74 Issues hat eine „diagnose-before-build"-Disziplin jeden content-basierten
Diskriminierungs-Hebel einzeln getestet und **widerlegt**:

- Ranking des bestehenden Kandidaten-Pools (#46/#51) — Serving-Features saturiert, kein
  diskriminierendes Signal.
- Generation-Swaps auf stärkere Modelle (#56/#60/#64/#68) — erschöpft.
- Generische Kontext-Virality-Judge (#54/#55) — trennt viral/nicht-viral nicht.
- Clipper-spezifische Kontext-Ähnlichkeit als Text-Signal (#69) — **Blind-Sheet: ein
  Jynxzi-kundiger Proband 0/10 (Zufall ≈ 1,2/10) → WALL.**

Die Eval-Zahlen zeigen das Muster: der Generator *produziert* den viralen Moment ~63 %
der Zeit (`raw@any`), liefert ihn aber bei recall@20 ≈ 12 % aus. Die Lücke ist fast
reines Sortieren — und Sortieren aus dem Content ist tot. Der Blind-Sheet misst eine
*obere Schranke* (was ein kundiger Mensch aus dem Material kann ≥ was ein Text/Embedding-
Modell lernen kann); 0/10 heißt: das Signal ist nicht im (Text-)Content. Caveat: n=1
Proband, 1 Lauf — eine Zweitmessung bliebe sinnvoll, ändert aber die Richtung nicht.

## Entscheidung

Pivot von **content-basiertem Pre-Publication-Ranking** auf einen **First-party-Outcome-
Flywheel (Harvesting-Loop)**:

- **North Star** wechselt von „reproduce @Iboclipss (recall@20)" auf **Retention/
  Performance der selbst geposteten Clips**.
- Der Autopilot **sagt Virality nicht mehr vorher**; er **postet ein diverses
  Kandidaten-Set, misst First-party-Outcome und erntet die Gewinner** — der „Ranker" ist
  der Markt.
- Der gesamte Offline-Apparat (recall@20, Gold-Manifest, GT-Flywheel, Ranker-Tracks)
  wird **demoted, nicht gelöscht**: Generator bleibt zentral (liefert Abdeckung),
  GT-Flywheel/Iboclipss → Bootstrap-Saat + Markt-Prior, Eval-Harness →
  Generator-Regressions-Wächter (`raw@any`).

Realisierte Form (Mechanik-Entscheidungen, gegrillt 2026-06-14):

- **Plattform:** YouTube Shorts zuerst — einzige Plattform mit programmatischem
  Retention-*Export* (YT Analytics API) UND Upload (YT Data API); der Loop stirbt ohne
  sauberen Retention-Export im Volumen. TikTok später für Reichweite.
- **„Win" im Cold-Start:** Seed-Batch-Retention (`retention_3s`/`completion`),
  **views-entkoppelt** (Views erst nach Warmup als Bestätigung) — das views-Gate in
  `import_yt_retention` verhungert sonst den Cold-Start.
- **Erste Build-Komponente:** eine *konstante* Packaging-Vorlage (9:16 Face-Track +
  fester Caption-Stil), gehalten über den Batch → Retention-Streuung ist dem *Moment*
  zurechenbar. Der Loop ist als **Moment-vs-Packaging-Disentanglement** designt (falls
  Packaging der dominante Hebel ist → These pivotet auf Auto-Packaging).
- **Diversity:** stratifiziertes Explore-Sampling über berechenbare Facetten
  (Goal-Family × Länge; Chat-Emotion mitgeloggt), permanenter Explore-Floor;
  Content-Priors nur milder Tilt (WALL).
- **Cold-Start-Gate (vorab fixiert):** ~180 Posts, ≥ ~15/Goal-Family; beste vs.
  schlechteste Goal-Family im Median-Seed-Retention > Packaging-A/B-Swing + Rauschen →
  Moment ist ein Hebel; sonst → Packaging/Exogenes dominiert.
- **Step 0 (cheap-before-expensive):** den bestehenden Kanal (@donni_clips, 52 Shorts,
  28.02.–10.04.2026) standalone minen — 52 transkribieren + Facette + YT-Retention
  ziehen — als konfundierter Vorab-Blick, bevor die Vorwärts-Schleife gebaut wird.
  VOD-Attribution (via Cross-Platform-Matcher) später nur fürs Generator-Feedback.

## Konsequenzen

**Positiv:**

- Der Moat überlebt in seiner einzig nicht-widerlegten Form: Jynxzi-spezifische
  First-party-Outcome-Daten, die kein generischer Finder sammelt.
- Der Loop beantwortet empirisch „ist Moment-Auswahl überhaupt mein Hebel?" — validiert
  oder kippt damit die ganze Produkt-These.
- Das gebaute Kapital (Generator + Pipeline) trägt den Post-Betrieb direkt.

**Negativ / Kosten:**

- **Cold-Start:** keine First-party-Daten ohne Posten; erfordert einen laufenden
  Content-Betrieb (~5–8 Shorts/Tag, ~1 Monat bis zum ersten belastbaren Gate).
- **Verlust der Offline-Messbarkeit:** kein freies recall@20 gegen ein eingefrorenes
  Manifest mehr; der North Star ist jetzt extern, langsam und verrauscht.
- **Packaging-Risiko:** falls Packaging/Exogenes den Outcome dominiert, ist der
  Moment-Moat schwach und das Produkt rückt näher an OpusClips' Stärke.
- Der Großteil von ~260 Offline-ML-Dateien wird zu Saat/Wächter — bewusst gehalten,
  aber nicht mehr Produkt-Rückgrat.

**Verworfene Alternativen:**

- *Content→Outcome-Predictor auf First-party-Labels* — stößt an dieselbe WALL (kein
  Content-Signal → Retention-Labels zaubern es nicht herbei). First-party-Outcome wird
  daher als Harvesting-/Bandit-Signal genutzt, nicht als Predictor-Trainingsziel.
- *Video-Achse (multimodal)* — vom Text-Blind-Sheet nicht widerlegt, nur als
  Hebel-Klasse zurückgestellt; könnte über einen Video-Blind-Sheet später revisited
  werden.
- *(b) reines Assistenz-Produkt* — sidesteppt Diskriminierung, verzichtet aber auf den
  Autopilot-Anspruch; First-party-Outcome macht (a) erreichbar, daher nicht nötig.
