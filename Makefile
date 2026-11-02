export PYTHONPATH := src
PY := uv run python -m

.PHONY: serve serve-sandbox reset-sandbox dev frontend test ingest-resumes docker apis-setup bucket-setup deploy iap-setup \
        anthropic-secret scheduler-secret email-secret scheduler-setup bucket-protect iap-open iap-close

# Locally there's no IAP: JOBHUNT_DEV_EMAIL stands in as the signed-in person (api/auth.py). It
# defaults to the admin's placeholder address, which owns everything in a local database.
DEV_EMAIL ?= admin@localhost

serve:        ## API (serves frontend/dist if built) on http://localhost:8000
	JOBHUNT_DEV_EMAIL=$(DEV_EMAIL) uv run uvicorn jobhunt.api.main:app --reload --port 8000

# A throwaway data dir (its own SQLite DB + uploaded files) -- starts empty, so the first-run
# onboarding wizard shows up, and nothing touches the real data/jobhunt.db or your resumes. Lives
# under data/, so it's git-ignored like everything else there. Claude calls are still real (and
# still logged/costed, in the sandbox DB's own usage log).
SANDBOX_DIR ?= data/sandbox

serve-sandbox:  ## like `serve`, but on :8001 against SANDBOX_DIR (run `make frontend` first)
	JOBHUNT_DEV_EMAIL=$(DEV_EMAIL) JOBHUNT_DATA_DIR=$(SANDBOX_DIR) uv run uvicorn jobhunt.api.main:app --reload --port 8001

reset-sandbox:  ## wipe SANDBOX_DIR to see onboarding from scratch again
	rm -rf $(SANDBOX_DIR)

dev:          ## Vite dev server (proxies /api to :8000); run `make serve` alongside
	cd frontend && npm run dev

frontend:     ## Timeline data, then build frontend/dist
	python3 scripts/build_changelog.py
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

# The daily email (core/email_digest.py) is on only when EMAIL_ADDRESS is set (in .deploy.local.mk
# or on the command line) -- after `make email-secret` has stored the Gmail app password, since
# deploy then mounts that secret. APP_URL (for "Open in the app" links) is the service's own URL.
EMAIL_ADDRESS ?=
comma := ,
SERVICE_URL = $$(gcloud run services describe jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) --format 'value(status.url)')
EMAIL_ENV = $(if $(EMAIL_ADDRESS),$(comma)EMAIL_ADDRESS=$(EMAIL_ADDRESS))
EMAIL_SECRET = $(if $(EMAIL_ADDRESS),$(comma)GMAIL_APP_PASSWORD=gmail-app-password:latest)

# The live database sits on the server's own disk and Litestream streams it to the bucket's
# litestream/ folder (scripts/start.sh, DECISIONS.md D-020). The first start seeds it from the old
# /data/jobhunt.db, which is then left alone as a fallback. --no-cpu-throttling keeps the CPU on
# while the server is up, so streaming never stalls between requests; with --min-instances 0 the
# server still shuts down (and Litestream flushes) after ~15 idle minutes.
LITESTREAM_ENV = JOBHUNT_DB_PATH=/tmp/db/jobhunt.db,JOBHUNT_SEED_DB=/data/jobhunt.db,LITESTREAM_REPLICA_URL=gcs://$(BUCKET)/litestream/jobhunt

# Multi-user (D-021). ADMIN_EMAIL is the Google account you sign in with: it owns everything from
# before multi-user and is the admin who approves everyone else. IAP_AUDIENCE is what the app checks
# IAP's signed sign-in token against (api/auth.py); if it's ever wrong, every page says so, naming
# the audience the token carried, so it can be fixed with `make deploy IAP_AUDIENCE=...`.
ADMIN_EMAIL ?= $(EMAIL_ADDRESS)
PROJECT_NUMBER = $$(gcloud projects describe $(GCP_PROJECT) --format='value(projectNumber)')
IAP_AUDIENCE ?= /projects/$(PROJECT_NUMBER)/locations/$(GCP_REGION)/services/jobhunt
AUTH_ENV = JOBHUNT_ADMIN_EMAIL=$(ADMIN_EMAIL),IAP_AUDIENCE=$(IAP_AUDIENCE),APP_URL=$(SERVICE_URL)

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

# The app snapshots its database into backups/ in this same bucket (core/backup.py); this guards
# the bucket itself. Soft delete keeps every deleted *or overwritten* object restorable for 7 days
# (Google's default for new buckets; this makes sure it's on). Object versioning isn't added on
# top: the database is rewritten on every save, so it would double-store each version for little
# extra safety. Settings' "Download a backup" is the copy kept off Google Cloud.
bucket-protect:  ## once: make sure the data bucket keeps deleted/overwritten files for 7 days, and show the result
	gcloud storage buckets update gs://$(BUCKET) --project $(GCP_PROJECT) --soft-delete-duration=7d
	gcloud storage buckets describe gs://$(BUCKET) --project $(GCP_PROJECT) --format="yaml(soft_delete_policy)"
	gcloud storage ls gs://$(BUCKET)/backups/ 2>/dev/null || echo "no backups/ yet (the first is made at the next daily run)"

# The frontend is built inside the Dockerfile's node stage -- no separate local build step needed
# before `gcloud builds submit`. --iap locks the service down directly (GA Cloud Run + IAP
# integration, no load balancer needed) -- run `make iap-setup` once afterward to actually grant
# your own account access, or nobody (including you) can reach it. --timeout 3600 (Cloud Run's max,
# default is 300s) because a full daily run -- polling plus up to daily_llm_cap serial scoring calls
# -- and a web-search discovery call can both outlast 5 minutes.
deploy:       ## Timeline data, then build the image in Cloud Build and deploy to Cloud Run, locked behind IAP
	@test -n "$(ADMIN_EMAIL)" || (echo "set ADMIN_EMAIL=you@gmail.com (the Google account you sign in with)" && exit 1)
	python3 scripts/build_changelog.py
	gcloud builds submit --project $(GCP_PROJECT) --region $(GCP_REGION) --tag $(IMAGE) .
	gcloud run deploy jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) --image $(IMAGE) \
	  --memory 2Gi --cpu 1 --max-instances 1 --min-instances 0 --no-cpu-throttling \
	  --no-allow-unauthenticated --iap --timeout 3600 \
	  --execution-environment gen2 \
	  --add-volume name=data,type=cloud-storage,bucket=$(BUCKET) \
	  --add-volume-mount volume=data,mount-path=/data \
	  --set-env-vars JOBHUNT_DATA_DIR=/data,$(LITESTREAM_ENV),$(AUTH_ENV)$(EMAIL_ENV) \
	  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest,SCHEDULER_SECRET=scheduler-secret:latest$(EMAIL_SECRET)

IAP_USER ?= $(error set IAP_USER=you@example.com on the command line)

iap-setup:    ## once (after the first deploy): restrict the service to IAP_USER's Google account only
	gcloud run services add-iam-policy-binding jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) \
	  --member="serviceAccount:service-$$(gcloud projects describe $(GCP_PROJECT) --format='value(projectNumber)')@gcp-sa-iap.iam.gserviceaccount.com" \
	  --role roles/run.invoker
	gcloud iap web add-iam-policy-binding --project $(GCP_PROJECT) --region $(GCP_REGION) \
	  --resource-type cloud-run --service jobhunt \
	  --member="user:$(IAP_USER)" --role roles/iap.httpsResourceAccessor

# Lets any Google account reach the app's sign-in. The app itself still admits only people the
# admin approves on /admin (D-021); everyone else sees a request-access page and nothing more. Run
# it once you're ready to invite people; `make iap-close` undoes it.
iap-open:     ## when ready to invite people: any Google account can sign in and request access
	gcloud iap web add-iam-policy-binding --project $(GCP_PROJECT) --region $(GCP_REGION) \
	  --resource-type cloud-run --service jobhunt \
	  --member=allAuthenticatedUsers --role roles/iap.httpsResourceAccessor

iap-close:    ## undo iap-open: back to only the accounts granted with iap-setup
	gcloud iap web remove-iam-policy-binding --project $(GCP_PROJECT) --region $(GCP_REGION) \
	  --resource-type cloud-run --service jobhunt \
	  --member=allAuthenticatedUsers --role roles/iap.httpsResourceAccessor

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

email-secret:  ## once: store $GMAIL_APP_PASSWORD (a Google app password) for the daily email
	@test -n "$$GMAIL_APP_PASSWORD" || (echo "set GMAIL_APP_PASSWORD first" && exit 1)
	gcloud secrets describe gmail-app-password --project $(GCP_PROJECT) >/dev/null 2>&1 \
	  || gcloud secrets create gmail-app-password --project $(GCP_PROJECT) --replication-policy automatic
	printf %s "$$GMAIL_APP_PASSWORD" | gcloud secrets versions add gmail-app-password --project $(GCP_PROJECT) --data-file=-
	gcloud secrets add-iam-policy-binding gmail-app-password --project $(GCP_PROJECT) \
	  --member="serviceAccount:$$(gcloud projects describe $(GCP_PROJECT) --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
	  --role roles/secretmanager.secretAccessor

SCHEDULER_SA = $(shell gcloud projects describe $(GCP_PROJECT) --format='value(projectNumber)')-compute@developer.gserviceaccount.com
IAP_CLIENT_ID ?= $(error set IAP_CLIENT_ID on the command line -- the OAuth client id from `gcloud iap settings get`)

# A plain custom header is NOT enough once --iap protects the whole service: IAP itself rejects
# the request (401 "Invalid IAP credentials: empty token") before it ever reaches the app's own
# X-Scheduler-Secret check (confirmed live -- this is exactly what happened on the first attempt).
# Cloud Scheduler instead needs a real Google-signed OIDC identity, scoped to the IAP OAuth
# client as its audience, from a service account IAP has been told to trust. The custom header is
# kept too (defense in depth, and internal.py still checks it) -- just no longer sufficient alone.
# --attempt-deadline 1800s (Cloud Scheduler's maximum): a full run takes 5-25 minutes, and the
# default 180s made every run show DEADLINE_EXCEEDED in the scheduler's log even when it succeeded,
# hiding the real 500s (2026-11-02).
scheduler-setup:  ## create/update the daily Cloud Scheduler job (M6) -- also grants the default
                  ## compute service account IAP + Cloud Run invoker access, and grants Cloud
                  ## Scheduler's own service agent permission to mint tokens as it
	gcloud run services add-iam-policy-binding jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) \
	  --member="serviceAccount:$(SCHEDULER_SA)" --role roles/run.invoker
	gcloud iap web add-iam-policy-binding --project $(GCP_PROJECT) --region $(GCP_REGION) \
	  --resource-type cloud-run --service jobhunt \
	  --member="serviceAccount:$(SCHEDULER_SA)" --role roles/iap.httpsResourceAccessor
	gcloud iam service-accounts add-iam-policy-binding $(SCHEDULER_SA) --project $(GCP_PROJECT) \
	  --member="serviceAccount:service-$$(gcloud projects describe $(GCP_PROJECT) --format='value(projectNumber)')@gcp-sa-cloudscheduler.iam.gserviceaccount.com" \
	  --role roles/iam.serviceAccountTokenCreator
	gcloud scheduler jobs describe run-daily --project $(GCP_PROJECT) --location $(GCP_REGION) >/dev/null 2>&1 \
	  && gcloud scheduler jobs update http run-daily --project $(GCP_PROJECT) --location $(GCP_REGION) \
	       --uri "$$(gcloud run services describe jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) --format 'value(status.url)')/internal/run-daily" \
	       --http-method POST --update-headers "X-Scheduler-Secret=$$(gcloud secrets versions access latest --secret=scheduler-secret --project $(GCP_PROJECT))" \
	       --schedule "0 14 * * *" --time-zone "America/New_York" --attempt-deadline 1800s --oidc-service-account-email $(SCHEDULER_SA) --oidc-token-audience $(IAP_CLIENT_ID) \
	  || gcloud scheduler jobs create http run-daily --project $(GCP_PROJECT) --location $(GCP_REGION) \
	       --uri "$$(gcloud run services describe jobhunt --project $(GCP_PROJECT) --region $(GCP_REGION) --format 'value(status.url)')/internal/run-daily" \
	       --http-method POST --headers "X-Scheduler-Secret=$$(gcloud secrets versions access latest --secret=scheduler-secret --project $(GCP_PROJECT))" \
	       --schedule "0 14 * * *" --time-zone "America/New_York" --attempt-deadline 1800s --oidc-service-account-email $(SCHEDULER_SA) --oidc-token-audience $(IAP_CLIENT_ID)
