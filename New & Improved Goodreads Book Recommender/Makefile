# PYTHONPATH is set explicitly: on macOS uv's editable-install .pth file can get the
# "hidden" flag, which makes Python skip it.
export PYTHONPATH := src
PY := uv run python -m
P := goodrec.pipeline

.PHONY: data artifacts genres-draft genres search-db package eval test serve frontend dev docker

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
	$(PY) $(P).s10_package

eval:            ## offline evaluation on held-out users -> eval/reports/
	$(PY) goodrec.eval.run

test:
	uv run pytest -q

serve:           ## API (serves frontend/dist if built) on http://localhost:8000
	uv run uvicorn goodrec.api.main:app --reload --port 8000

frontend:
	cd frontend && npm install && npm run build

dev:             ## frontend dev server with API proxy (run `make serve` in another shell)
	cd frontend && npm run dev

docker:
	docker build -t goodrec .
