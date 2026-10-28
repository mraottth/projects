export PYTHONPATH := src
PY := uv run python -m

.PHONY: serve dev frontend test ingest-resumes docker apis-setup bucket-setup deploy iap-setup \
        anthropic-secret scheduler-secret scheduler-setup

serve:        ## API (serves frontend/dist if built) on http://localhost:8000
	uv run uvicorn jobhunt.api.main:app --reload --port 8000

dev:          ## Vite dev server (proxies /api to :8000); run `make serve` alongside
	cd frontend && npm run dev

frontend:     ## build frontend/dist
	cd frontend && npm install && npm run build

test:
	uv run pytest -q

ingest-resumes:  ## re-runnable recursive bulk import: make ingest-resumes DIR=~/Documents/Resumes
	$(PY) jobhunt.cli.ingest_resumes $(DIR)

docker:
	docker build -t jobhunt .

# ---- Cloud Run + Cloud Scheduler (M6). GCP project id lives in the git-ignored .deploy.local.mk.
-include .deploy.local.mk
GCP_PROJECT ?= $(error set GCP_PROJECT in .deploy.local.mk or on the command line)
GCP_REGION  ?= us-central1
IMAGE        = $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/jobhunt/app
BUCKET      ?= $(GCP_PROJECT)-jobhunt-data

# One-time setup, in order: apis-setup -> bucket-setup -> anthropic-secret -> scheduler-secret
# -> deploy -> iap-setup -> scheduler-setup. `deploy` alone is safe to re-run for every later update.

apis-setup:   ## once: enable every GCP API this app needs
	gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
	  secretmanager.googleapis.com iap.googleapis.com cloudscheduler.googleapis.com storage.googleapis.com \
	  --project $(GCP_PROJECT)
	gcloud artifacts repositories describe jobhunt --project $(GCP_PROJECT) --location $(GCP_REGION) >/dev/null 2>&1 \
	  || gcloud artifacts repositories create jobhunt --project $(GCP_PROJECT) --location $(GCP_REGION) \
	       --repository-format docker

bucket-setup:  ## once: create the GCS bucket backing persistent SQLite + resume storage (mounted
               ## read/write into the container via a Cloud Run gen2 volume mount, see `deploy`)
	gcloud storage buckets describe gs://$(BUCKET) >/dev/null 2>&1 \
	  || gcloud storage buckets create gs://$(BUCKET) --project $(GCP_PROJECT) --location $(GCP_REGION) \
	       --uniform-bucket-level-access
	gcloud storage buckets add-iam-policy-binding gs://$(BUCKET) \
	  --member="serviceAccount:$$(gcloud projects describe $(GCP_PROJECT) --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
	  --role roles/storage.objectAdmin

# The frontend is built inside the Dockerfile's node stage -- no separate local build step needed
# before `gcloud builds submit`. --iap locks the service down directly (GA Cloud Run + IAP
# integration, no load balancer needed) -- run `make iap-setup` once afterward to actually grant
# your own account access, or nobody (including you) can reach it.
deploy:       ## build the image in Cloud Build and deploy to Cloud Run, locked behind IAP
	gcloud builds submit --project $(GCP_PROJECT) --region $(GCP_REGION) --tag $(IMAGE) .
	gcloud run deploy jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) --image $(IMAGE) \
	  --memory 1Gi --cpu 1 --max-instances 1 --no-allow-unauthenticated --iap \
	  --execution-environment gen2 \
	  --add-volume name=data,type=cloud-storage,bucket=$(BUCKET) \
	  --add-volume-mount volume=data,mount-path=/data \
	  --set-env-vars JOBHUNT_DATA_DIR=/data \
	  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest,SCHEDULER_SECRET=scheduler-secret:latest

IAP_USER ?= $(error set IAP_USER=you@example.com on the command line)

iap-setup:    ## once (after the first deploy): restrict the service to IAP_USER's Google account only
	gcloud run services add-iam-policy-binding jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) \
	  --member="serviceAccount:service-$$(gcloud projects describe $(GCP_PROJECT) --format='value(projectNumber)')@gcp-sa-iap.iam.gserviceaccount.com" \
	  --role roles/run.invoker
	gcloud iap web add-iam-policy-binding --project $(GCP_PROJECT) --region $(GCP_REGION) \
	  --resource-type cloud-run --service jobhunt \
	  --member="user:$(IAP_USER)" --role roles/iap.httpsResourceAccessor

anthropic-secret:  ## once: store $ANTHROPIC_API_KEY in Secret Manager (re-run to rotate)
	@test -n "$$ANTHROPIC_API_KEY" || (echo "set ANTHROPIC_API_KEY first" && exit 1)
	gcloud secrets describe anthropic-key --project $(GCP_PROJECT) >/dev/null 2>&1 \
	  || gcloud secrets create anthropic-key --project $(GCP_PROJECT) --replication-policy automatic
	printf %s "$$ANTHROPIC_API_KEY" | gcloud secrets versions add anthropic-key --project $(GCP_PROJECT) --data-file=-

scheduler-secret:  ## once: store $SCHEDULER_SECRET in Secret Manager (re-run to rotate)
	@test -n "$$SCHEDULER_SECRET" || (echo "set SCHEDULER_SECRET first" && exit 1)
	gcloud secrets describe scheduler-secret --project $(GCP_PROJECT) >/dev/null 2>&1 \
	  || gcloud secrets create scheduler-secret --project $(GCP_PROJECT) --replication-policy automatic
	printf %s "$$SCHEDULER_SECRET" | gcloud secrets versions add scheduler-secret --project $(GCP_PROJECT) --data-file=-

scheduler-setup:  ## create/update the daily Cloud Scheduler job (M6)
	gcloud scheduler jobs describe run-daily --project $(GCP_PROJECT) --location $(GCP_REGION) >/dev/null 2>&1 \
	  && gcloud scheduler jobs update http run-daily --project $(GCP_PROJECT) --location $(GCP_REGION) \
	       --uri "$$(gcloud run services describe jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) --format 'value(status.url)')/internal/run-daily" \
	       --http-method POST --headers "X-Scheduler-Secret=$$SCHEDULER_SECRET" --schedule "0 6 * * *" \
	  || gcloud scheduler jobs create http run-daily --project $(GCP_PROJECT) --location $(GCP_REGION) \
	       --uri "$$(gcloud run services describe jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) --format 'value(status.url)')/internal/run-daily" \
	       --http-method POST --headers "X-Scheduler-Secret=$$SCHEDULER_SECRET" --schedule "0 6 * * *"
