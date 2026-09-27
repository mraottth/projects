# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Goodreads book recommender (web app) built on the UCSD Book Graph data (Wan & McAuley; ratings through 2017). Users either upload a Goodreads library export CSV or search and star-rate books. They get "For you" recommendations (item-item neighbors blended with ALS fold-in), "readers like you" lists (popular / top rated among nearest users), genre filters and a book detail view with similar books. This folder lives inside the larger `projects` portfolio repo; `../Goodreads Book Recommender` is the original 2023 version and is unrelated code.

## Commands

All Python runs through uv (Python 3.12). The Makefile exports `PYTHONPATH=src`. On macOS, uv's editable-install `.pth` file sometimes gets the "hidden" flag and Python skips it, so outside `make` prefix commands with `PYTHONPATH=src`.

```bash
make data            # s00-s05: download raw UCSD files (~7.5 GB, resumable) and build data/interim/*
make genres          # s06: apply config/shelf_genres.yaml -> per-work genres (make genres-draft = regenerate map via Claude API)
make artifacts       # s06-s10: item-kNN, ALS, SQLite catalog, packaging + sanity asserts -> artifacts/
make eval            # offline eval on held-out users -> eval/reports/<stamp>.md (add --grid via: PYTHONPATH=src uv run python -m goodrec.eval.run --grid)
make test            # pytest (API tests skip unless artifacts/ is built)
make serve           # uvicorn on :8000; serves frontend/dist if built
make frontend        # npm install + build frontend/dist
make dev             # Vite dev server (proxies /api to :8000); run `make serve` alongside

PYTHONPATH=src uv run pytest tests/test_api.py::test_filters -q         # single test
PYTHONPATH=src uv run python -m goodrec.pipeline.s07_item_knn --force   # re-run one stage (stages skip if outputs exist)
PYTHONPATH=src uv run python -m goodrec.eval.tune_als --users 800       # ALS hyperparameter sweep
```

## Architecture

**Offline pipeline → artifacts → stateless API.** Heavy work is precomputed; a recommend request costs ~10–70 ms.

- `src/goodrec/pipeline/s00…s10` run in order. Each stage skips if its outputs exist (`--force` to redo). Thresholds and hyperparameters live in `config/pipeline.yaml`.
  - **Editions are collapsed to works** (`work_id`) in s03. The catalog (s04) is works with at least `catalog.min_raters` distinct raters, sorted by popularity. `work_idx` (dense 0..N-1) is the row/column index for every matrix and artifact, so s04 through s10 must be rebuilt together when the catalog changes.
  - s05 holds out `eval.n_test_users` users that **no model sees**; the eval harness folds them in, so eval measures real fold-in quality. Production artifacts are trained on the same train split.
  - s06 genres come from Goodreads reader shelves mapped through `config/shelf_genres.yaml` (hand-reviewed; parents list is fixed and mirrored in `s06_genres.PARENTS` and `frontend/src/genres.ts`).
  - The raw data URL moved to `mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/` (the old `datarepo.eng.ucsd.edu` URL 404s).
- `src/goodrec/core/` is shared by the API **and** eval, so eval always exercises serving code:
  - `artifacts.py` loads everything.
  - `scoring.py` has fold-in, item-item scoring, blend `a(n)=n/(n+k_a)`, filters, series rules and explanations.
  - `similar_readers.py` does nearest users in ALS space, then aggregates their actual shelves from `readers_csr.npz`.
  - Serving needs numpy/scipy only. `implicit`, polars and anthropic are pipeline-only dependency groups.
- `src/goodrec/api/`: FastAPI.
  - **Public ids are Goodreads `work_id`s**, mapped to `work_idx` in `catalog.py`, so browser-stored ratings survive rebuilds.
  - Per-user model scores are LRU-cached by ratings hash (`main.py`), so filter/tab changes only re-filter.
  - CSV import matching (`matching.py`): Book Id → edition table, then ISBN, then normalized title + author last name (`core/textnorm.py`).
- `frontend/`: React + Vite + TS, no UI library.
  - The user's shelf lives in localStorage (`store.tsx`). View, tab and filters live in the URL query string (`useUrlState.ts`).
  - Genre colors are 8 validated family hues (`genres.ts`); text labels always carry identity.

## Gotchas

- Changing `blend`/`als` params in `config/pipeline.yaml` affects serving immediately (`Params.from_config`). ALS factor params require re-running s08 onward.
- Popular Goodreads titles embed series as `"Title (Series, #N)"`. `textnorm.parse_series` drives the "hide later volumes / show next in series" rule.
- Cover URLs are 2017 Goodreads links (still live as of 2026-09). The frontend falls back to Open Library by ISBN, then a typographic placeholder.
- The data license is non-commercial; keep the citation in the footer.
