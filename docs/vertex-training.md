# Vertex AI Training

Dieses Scaffold packt den bestehenden `jynxzi`-Trainingsworkflow in einen Vertex-AI-Custom-Container. Es ersetzt den aktuellen lokalen Wrapper nicht, sondern ruft intern weiter `backend/ml/train_jynxzi_vod.sh` auf.

## Scope

- Der erste Cloud-Entrypoint trainiert nur den bestehenden `jynxzi`-Flow.
- Vorhandene Artefakte aus `gs://...` werden vor dem Lauf nach `backend/ml/artifacts/` synchronisiert.
- Neue Artefakte werden nach dem Lauf wieder in denselben GCS-Prefix hochgeladen.
- `TRAINING_DATASET_LLM_PROVIDER` ist jetzt umschaltbar. In Vertex sollte standardmaessig `auto` oder `groq` verwendet werden, nicht lokales Ollama ohne expliziten Remote-Host.

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

Der Container installiert `backend` mit den Extras `ml` und `vertex` und startet standardmaessig `python -m ml.vertex_train`.

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
  --dataset-llm-provider auto
```

## Wichtige Laufzeitparameter

- `--artifacts-gcs-uri`: GCS-Prefix fuer Labels, Progress-Datei, Datensaetze, Modellartefakte und `vertex_job_summary.json`.
- `--dataset-llm-provider`: `auto`, `groq` oder `ollama`.
- `--ollama-host`: nur setzen, wenn in Vertex wirklich ein externer Ollama-Endpunkt existiert.
- `--storage-dir`: lokaler Pfad innerhalb des Containers. Default ist `tmp/storage`, damit die bestehende Backend-Konfiguration gueltig bleibt.
- `--train-tmpdir`: lokaler Arbeitsbereich fuer zwischengespeicherte Trainingsdownloads.

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
