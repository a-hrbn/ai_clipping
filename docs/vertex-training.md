# Vertex AI Training

Dieses Repo hat jetzt zwei additive Cloud-Trainingspfade:

- `CustomJob`-Lift-and-Shift: ruft weiter `backend/ml/train_jynxzi_vod.sh` im Container auf.
- `Vertex Pipelines`: nutzt portable Komponenten unter `backend/ml/pipelines/` und den Submitter `scripts/submit_vertex_pipeline.py`.

Beide Pfade koexistieren. Der bestehende Shell-/Custom-Job-Flow bleibt der Fallback, bis die Pipeline-Runs stabil sind.

## Scope

- Der erste Cloud-Entrypoint trainiert nur den bestehenden `jynxzi`-Flow.
- Vorhandene Artefakte aus `gs://...` werden vor dem Lauf nach `backend/ml/artifacts/` synchronisiert.
- Neue Artefakte werden nach dem Lauf wieder in denselben GCS-Prefix hochgeladen.
- `TRAINING_DATASET_LLM_PROVIDER` ist jetzt umschaltbar. In Vertex sollte standardmaessig `auto` oder `groq` verwendet werden, nicht lokales Ollama ohne expliziten Remote-Host.
- Optionales MLflow-Tracking ist additiv verdrahtet. Ohne `MLFLOW_TRACKING_URI` bleibt der Trainingspfad unveraendert.

## Voraussetzungen

- Vertex AI API, Artifact Registry, Cloud Storage und Secret Manager sind im GCP-Projekt aktiviert.
- Das Service Account des Jobs hat mindestens Zugriff auf:
  - Vertex AI Custom Jobs
  - Artifact Registry Pull
  - Cloud Storage Read/Write fuer den Artefakt-Bucket
  - Secret Manager Accessor fuer Twitch- und optional Groq-/Gemini-Secrets
- Ein GCS-Prefix fuer Trainingsartefakte existiert, z. B. `gs://my-bucket/jynxzi/train-artifacts`.

## Container bauen

Aus dem Repo-Root:

```bash
docker build -f backend/Dockerfile.vertex-train -t europe-west3-docker.pkg.dev/PROJECT_ID/ml/ai-clipping-train:latest .
docker push europe-west3-docker.pkg.dev/PROJECT_ID/ml/ai-clipping-train:latest
```

Der Container installiert `backend` mit den Extras `ml`, `vision` und `vertex` und startet standardmaessig `python -m ml.vertex_train`.

Die neuen Pipeline-Komponenten benoetigen dieselben Extras plus `kfp`, `mlflow`, `fsspec` und fuer `gs://`-URIs `gcsfs`. Diese sind jetzt ueber die `ml`-/`vertex`-Extras verdrahtet.

## Job starten

Beispiel:

```bash
backend/.venv/bin/python scripts/submit_vertex_training.py \
  --project PROJECT_ID \
  --region europe-west3 \
  --staging-bucket gs://PROJECT_ID-vertex-staging \
  --container-uri europe-west3-docker.pkg.dev/PROJECT_ID/ml/ai-clipping-train:latest \
  --artifacts-gcs-uri gs://PROJECT_ID-ml/jynxzi/train-artifacts \
  --service-account vertex-train@PROJECT_ID.iam.gserviceaccount.com \
  --date-from 2026-03-21 \
  --date-to 2026-04-10 \
  --twitch-client-id-secret projects/PROJECT_NUMBER/secrets/twitch-client-id/versions/latest \
  --twitch-client-secret-secret projects/PROJECT_NUMBER/secrets/twitch-client-secret/versions/latest \
  --groq-api-key-secret projects/PROJECT_NUMBER/secrets/groq-api-key/versions/latest \
  --gemini-api-key-secret projects/PROJECT_NUMBER/secrets/gemini-api-key/versions/latest \
  --mlflow-tracking-uri-secret projects/PROJECT_NUMBER/secrets/mlflow-tracking-uri/versions/latest \
  --dataset-llm-provider auto
```

### MLflow im bestehenden Custom-Job-Pfad

Wenn `MLFLOW_TRACKING_URI` gesetzt ist, loggt `backend/ml/train_ranker.py`:

- Params: `streamer`, `profile`, `date_range`, `window_size`, `blend_floor`
- Metrics: `ndcg_at_5`, `f1`, `accuracy`, `precision_at_5`, `recall_at_5`
- Artifacts: `*_ranker.json`, `*_artifact_health.json`, spaeter im Shell-Flow zusaetzlich `*_proxy_eval.json`, `*_gold_eval.json`, `*_pipeline_eval.json`
- Tags: `git_sha`, `vertex_job_id`, `image_tag`, `dataset_rows`, `positive_rows`

Der Ranker kann bei fehlendem lokalem Artefakt zusaetzlich ein in MLflow Registry registriertes Production-Bundle als Fallback laden.

### Lokaler MLflow-Modus

Beispiel fuer lokalen Tracking-Modus ohne Cloud:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
export MLFLOW_EXPERIMENT_NAME=ai-clipping
```

Der Codepfad unterscheidet nicht zwischen lokalem und Cloud-Tracking; nur die ENV-Konfiguration aendert sich.

### Cloud-MLflow-Modus

Beispiel fuer denselben Codepfad gegen einen externen Tracking-Server:

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.run.app
export MLFLOW_EXPERIMENT_NAME=ai-clipping
```

## Vertex Pipelines

Die portable Pipeline lebt unter:

```text
backend/ml/pipelines/
  components/
  backends/
  train_pipeline.py
```

Wesentliche Eigenschaften:

- Komponenten sind CLI-first und koennen direkt per `python -m ...` gestartet werden.
- Das KFP-Wiring ist ein duenner Wrapper um diese CLI-Module.
- Der lokale sequenzielle Backend-Pfad ist `python -m ml.pipelines.backends.local_shell`.
- Der Vertex-Submitter ist `scripts/submit_vertex_pipeline.py`.

### Pipeline lokal sequenziell ausfuehren

```bash
cd backend
.venv/bin/python -m ml.pipelines.backends.local_shell \
  --streamer jynxzi \
  --profile jynxzi \
  --date-from 2026-03-14 \
  --date-to 2026-03-20 \
  --pipeline-root tmp/pipeline-local
```

### Pipeline kompilieren und submitten

```bash
backend/.venv/bin/python scripts/submit_vertex_pipeline.py \
  --project PROJECT_ID \
  --region europe-west3 \
  --staging-bucket gs://PROJECT_ID-vertex-staging \
  --pipeline-root gs://PROJECT_ID-ml/pipeline-root \
  --service-account vertex-train@PROJECT_ID.iam.gserviceaccount.com \
  --base-image europe-west3-docker.pkg.dev/PROJECT_ID/ml/ai-clipping-train:latest \
  --streamer jynxzi \
  --profile jynxzi \
  --date-from 2026-03-14 \
  --date-to 2026-03-20 \
  --dataset-llm-provider groq \
  --twitch-client-id-secret projects/PROJECT_NUMBER/secrets/twitch-client-id/versions/latest \
  --twitch-client-secret-secret projects/PROJECT_NUMBER/secrets/twitch-client-secret/versions/latest \
  --groq-api-key-secret projects/PROJECT_NUMBER/secrets/groq-api-key/versions/latest \
  --mlflow-tracking-uri-secret projects/PROJECT_NUMBER/secrets/mlflow-tracking-uri/versions/latest
```

Der Submitter kompiliert zuerst `backend/ml/pipelines/train_pipeline.py` nach JSON und startet danach einen `google.cloud.aiplatform.PipelineJob`.

## Wichtige Laufzeitparameter

- `--artifacts-gcs-uri`: GCS-Prefix fuer Labels, Progress-Datei, Datensaetze, Modellartefakte und `vertex_job_summary.json`.
- `--dataset-llm-provider`: `auto`, `groq` oder `ollama`.
- `--training-vod-gcs-cache-uri`: optionaler `gs://`-Prefix fuer einen reinen Cloud-VOD-Cache. Wenn der Flag weggelassen wird, leitet das Submit-Script automatisch `gs://<artifact-bucket>/cache/vod-downloads` aus `--artifacts-gcs-uri` ab.
- `--training-vod-checkpoint-gcs-uri`: optionaler `gs://`-Prefix fuer persistente VOD-Checkpoint-Dateien mit bereits aufbereiteten Trainingsrows. Wenn der Flag weggelassen wird, leitet das Submit-Script automatisch `gs://<artifact-bucket>/cache/vod-checkpoints` aus `--artifacts-gcs-uri` ab.
- `--ollama-host`: nur setzen, wenn in Vertex wirklich ein externer Ollama-Endpunkt existiert.
- `--storage-dir`: lokaler Pfad innerhalb des Containers. Default ist `tmp/storage`, damit die bestehende Backend-Konfiguration gueltig bleibt.
- `--train-tmpdir`: lokaler Arbeitsbereich fuer zwischengespeicherte Trainingsdownloads.

### Zukuenftige Qualitaetsphasen bleiben standardmaessig aus

Die naechsten ML-Ausbaustufen sind bewusst hinter Feature-Flags geparkt und bleiben bis zu gruenen Shadow-/Trainingsmetriken deaktiviert:

- `ENABLE_HOOK_SUBCLASSIFIER=false`
- `ENABLE_MULTIMODAL_EMBEDDINGS=false`
- `ENABLE_POST_PUBLISH_LABELS=false`

Das haelt den Vertex-Trainingspfad rueckwaertskompatibel, waehrend die zugehoerigen Daten- und Modellpfade schrittweise aufgebaut werden.

### Kostenrahmen fuer spaetere Phasen

Vor Phase B und E sollten die externen Budgets explizit geklaert sein:

- Twitch/Helix-Quota fuer breitere Label-Sammler
- YouTube-Data- und TikTok-API-Quota fuer den spaeteren Post-Publish-Loop
- Vertex-CPU-Stunden fuer HPO und Embedding-Experimente

### GCS-only VOD-Cache

Wenn mehrere Vertex-Runs dieselben Twitch-VODs wiederverwenden sollen, ohne einen persistenten Cache auf dem lokalen Rechner zu brauchen, setze einen separaten GCS-Prefix, z. B.:

```bash
backend/.venv/bin/python scripts/submit_vertex_training.py \
  ... \
  --training-vod-gcs-cache-uri gs://PROJECT_ID-ml-cache/vod-downloads
```

Verhalten:

- Der Cache ist nur in Google Cloud persistent.
- Jeder Vertex-Job laedt ein Cache-Hit-Objekt trotzdem kurz in seinen lokalen Arbeitsbereich, weil `ffmpeg` und die Feature-Extraktion mit lokalen Dateien arbeiten.
- Bei Cache-Miss wird das VOD wie bisher von Twitch geladen und anschliessend mit einem conditional upload in den GCS-Cache geschrieben.
- Der Cache sollte einen eigenen Bucket oder Prefix mit Lifecycle-Regel bekommen, z. B. automatische Loeschung nach 30 Tagen.

### GCS-only VOD-Checkpoint-Cache

Wenn nicht nur die MP4-Downloads, sondern auch bereits gebaute VOD-Trainingsrows ueber Vertex-Runs hinweg wiederverwendet werden sollen, setze zusaetzlich einen zweiten GCS-Prefix:

```bash
backend/.venv/bin/python scripts/submit_vertex_training.py \
  ... \
  --training-vod-gcs-cache-uri gs://PROJECT_ID-ml-cache/vod-downloads \
  --training-vod-checkpoint-gcs-uri gs://PROJECT_ID-ml-cache/vod-checkpoints
```

Verhalten:

- Der Checkpoint-Key haengt von `vod_id`, `dataset_build_fingerprint` und `label_signature` ab.
- Aendert sich die Feature-Logik, der relevante Dataset-Fingerprint oder das zugrunde liegende Label-Set fuer ein VOD, wird automatisch ein anderer Checkpoint-Key verwendet.
- Bei passendem Checkpoint kann der Vertex-Run das bereits aufbereitete VOD-Row-Set direkt laden, statt das VOD erneut komplett zu schneiden, zu transkribieren und zu featurisieren.
- MP4-Cache und Checkpoint-Cache sollten getrennte Prefixe behalten, damit Lifecycle und Debugging sauber bleiben.

## Verhalten des Vertex-Entrypoints

`backend/ml/vertex_train.py` macht in dieser Reihenfolge:

1. Secrets aus Secret Manager lesen und in die Runtime-Umgebung legen.
2. Vorhandene Artefakte aus GCS nach `backend/ml/artifacts/` synchronisieren.
3. `backend/ml/train_jynxzi_vod.sh DATE_FROM DATE_TO` im Container ausfuehren.
4. Eine kompakte `vertex_job_summary.json` erzeugen.
5. Alle Artefakte wieder in den GCS-Prefix hochladen.

## Bekannte Grenzen

- Der Entrypoint ist absichtlich noch auf `jynxzi` fest verdrahtet.
- Das Training bleibt single-worker. Es ist ein Lift-and-Shift des bestehenden Flows, keine verteilte Pipeline.
- Wenn weder `GROQ_API_KEY` noch ein expliziter `--ollama-host` gesetzt ist, blockiert `TRAINING_DATASET_LLM_PROVIDER=auto` bewusst frueh, statt spaeter an einem impliziten lokalen Ollama-Fallback zu scheitern.
