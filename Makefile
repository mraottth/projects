export PYTHONPATH := src
PY := uv run python -m

.PHONY: serve dev frontend test ingest-resumes docker deploy anthropic-secret scheduler-secret scheduler-setup

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

# The frontend is built inside the Dockerfile's node stage -- no separate local build step needed
# before `gcloud builds submit`.
deploy:       ## build the image in Cloud Build and deploy to Cloud Run (private; IAP setup is a one-time manual step -- see README)
	gcloud builds submit --project $(GCP_PROJECT) --region $(GCP_REGION) --tag $(IMAGE) .
	gcloud run deploy jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) --image $(IMAGE) \
	  --memory 1Gi --cpu 1 --max-instances 1 --no-allow-unauthenticated \
	  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest,SCHEDULER_SECRET=scheduler-secret:latest

anthropic-secret:  ## once: store $ANTHROPIC_API_KEY in Secret Manager (re-run to rotate)
	@test -n "$$ANTHROPIC_API_KEY" || (echo "set ANTHROPIC_API_KEY first" && exit 1)
	gcloud services enable secretmanager.googleapis.com --project $(GCP_PROJECT)
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
