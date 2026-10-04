# PYTHONPATH is set explicitly: on macOS uv's editable-install .pth file can get the
# "hidden" flag, which makes Python skip it.
export PYTHONPATH := src
PY := uv run python -m
P := goodrec.pipeline

.PHONY: data artifacts genres-draft genres search-db package eval test serve frontend dev docker deploy

data:            ## download + parse raw UCSD data into data/interim
	$(PY) $(P).s00_download
	$(PY) $(P).s01_books
	$(PY) $(P).s02_meta
	$(PY) $(P).s03_ratings
	$(PY) $(P).s04_catalog
	$(PY) $(P).s05_matrix

genres-draft:    ## one-time: ask Claude to draft config/shelf_genres.yaml (needs ANTHROPIC_API_KEY), then review it
	$(PY) $(P).s06_genres --draft

genres:          ## apply config/shelf_genres.yaml -> per-work genres
	$(PY) $(P).s06_genres

artifacts: genres  ## models + search DB + packaging
	$(PY) $(P).s07_item_knn
	$(PY) $(P).s08_als
	$(PY) $(P).s09_search_db
	$(PY) $(P).s11_rating_mf   # before s10, which hashes it into the manifest
	$(PY) $(P).s10_package

eval:            ## temporal hide-and-predict evaluation vs baselines -> eval/reports/ (see goodrec.eval.run)
	$(PY) goodrec.eval.run

test:
	uv run pytest -q

serve:           ## API (serves frontend/dist if built) on http://localhost:8000
	uv run uvicorn goodrec.api.main:app --reload --port 8000

frontend:        ## code-link anchors for the About page, changelog data, then build the React app
	python3 scripts/code_links.py
	python3 scripts/build_changelog.py
	python3 scripts/build_evaluations.py
	cd frontend && npm install && npm run build

dev:             ## frontend dev server with API proxy (run `make serve` in another shell)
	cd frontend && npm run dev

docker:
	docker build -t goodrec .

# ---- Cloud Run (one warm instance to avoid cold starts). Needs `gcloud auth login`; builds remotely with Cloud Build, no local Docker.
GCP_PROJECT ?= v2-book-recommender
GCP_REGION  ?= us-central1
IMAGE       := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/goodrec/app

deploy:          ## build the image in Cloud Build and deploy to Cloud Run (uploads artifacts/)
	python3 scripts/code_links.py
	python3 scripts/build_changelog.py
	python3 scripts/build_evaluations.py
	gcloud builds submit --project $(GCP_PROJECT) --region $(GCP_REGION) --tag $(IMAGE) .
	gcloud run deploy goodrec --project $(GCP_PROJECT) --region $(GCP_REGION) --image $(IMAGE) \
	  --memory 1Gi --cpu 1 --cpu-boost --min-instances 1 --max-instances 3 --concurrency 20 \
	  --allow-unauthenticated $(CHAT_SECRET_FLAG)

# Assistant tab: the Anthropic key lives in Secret Manager; deploy mounts it only if the secret exists.
CHAT_SECRET_FLAG = $(shell gcloud secrets describe anthropic-key --project $(GCP_PROJECT) >/dev/null 2>&1 \
                     && echo --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest)

chat-secret:     ## store $$ANTHROPIC_API_KEY in Secret Manager for the Assistant tab (run once; re-run to rotate)
	@test -n "$$ANTHROPIC_API_KEY" || (echo "set ANTHROPIC_API_KEY first" && exit 1)
	gcloud services enable secretmanager.googleapis.com --project $(GCP_PROJECT)
	gcloud secrets describe anthropic-key --project $(GCP_PROJECT) >/dev/null 2>&1 \
	  || gcloud secrets create anthropic-key --project $(GCP_PROJECT) --replication-policy automatic
	printf %s "$$ANTHROPIC_API_KEY" | gcloud secrets versions add anthropic-key --project $(GCP_PROJECT) --data-file=-
	gcloud secrets add-iam-policy-binding anthropic-key --project $(GCP_PROJECT) \
	  --member serviceAccount:$$(gcloud projects describe $(GCP_PROJECT) --format 'value(projectNumber)')-compute@developer.gserviceaccount.com \
	  --role roles/secretmanager.secretAccessor
