# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Goodreads book recommender (web app) built on the UCSD Book Graph data (Wan & McAuley; ratings through 2017). Users either upload a Goodreads library export CSV or search and star-rate books. They get "For you" recommendations (item-item neighbors blended with ALS fold-in), "readers like you" lists (popular / top rated among nearest users), genre filters and a book detail view with similar books. The app is branded "Shelf Life". This folder lives inside the larger `projects` portfolio repo, whose top-level folders are number-prefixed to set their order on GitHub; `../03 - Goodreads Recommender (2023 original)` is the original 2023 version and is unrelated code. The About page's GitHub links (`REPO_URL`, `CODE_BASE` in `AboutPage.tsx`) encode this folder's name, so update them if it's renamed.

## Commands

All Python runs through uv (Python 3.12). The Makefile exports `PYTHONPATH=src`. On macOS, uv's editable-install `.pth` file sometimes gets the "hidden" flag and Python skips it, so outside `make` prefix commands with `PYTHONPATH=src`.

```bash
make data            # s00-s05: download raw UCSD files (~7.5 GB, resumable) and build data/interim/*
make genres          # s06: apply config/shelf_genres.yaml -> per-work genres (make genres-draft = regenerate map via Claude API)
make artifacts       # s06-s10: item-kNN, ALS, SQLite catalog, packaging + sanity asserts -> artifacts/
make eval            # temporal hide-and-predict eval on the test users vs all baselines -> eval/reports/<stamp>_<model>.md
make test            # pytest (API tests skip unless artifacts/ is built)
make serve           # uvicorn on :8000; serves frontend/dist if built
make frontend        # npm install + build frontend/dist
make dev             # Vite dev server (proxies /api to :8000); run `make serve` alongside
make chat-secret     # once: store $ANTHROPIC_API_KEY in Secret Manager (make deploy mounts it if present)
make deploy          # Cloud Build + Cloud Run (project v2-book-recommender, us-central1, min 1 warm instance); uploads artifacts/ via .gcloudignore

PYTHONPATH=src uv run pytest tests/test_api.py::test_filters -q         # single test
PYTHONPATH=src uv run python -m goodrec.pipeline.s07_item_knn --force   # re-run one stage (stages skip if outputs exist)
PYTHONPATH=src uv run python -m goodrec.eval.run --set validation --users 300 --models shelf_life,popular   # quick check
PYTHONPATH=src uv run python -m goodrec.eval.run --set validation --grid  # blend grid (validation only; tuning never uses test)
PYTHONPATH=src uv run python -m goodrec.eval.run --ablations --promote    # full test run; record as champion if it wins
PYTHONPATH=src uv run python -m goodrec.eval.tune_als --users 800       # ALS hyperparameter sweep (validation users)
```

## Working conventions

This project's history is public: every prompt, decision and commit shows on the site's Changelog page (`?view=changelog`).

- **Small commits, one concern each**, on a feature branch. Ask before pushing, merging or deploying.
- **Commit messages:** `<category>: <imperative summary>` (≤ 72 characters), then a body saying what changed and why. Categories: `ui`, `model`, `eval`, `data`, `assistant`, `infra`, `docs`. The prefix is how the Changelog page categorizes commits; anything else shows as "uncategorized". Messages are public, so keep them neutral.
- **DECISIONS.md:** add an entry (next `D-NNN`, same fields as the others) whenever a choice or tradeoff is made, by the user or proposed by Claude and accepted. Link the prompt ids and commits, and quote the measured numbers. If the reasoning wasn't stated, say so rather than inventing it.
- **prompts/:** a `Stop` hook (`.claude/settings.json`, here and at the repo root) runs `scripts/export_prompts.py` after every turn, keeping `prompts/<date>_<session>.md/.json` current. Commit those files with the related work, and add each new prompt id to `prompts/categories.json` (categories above plus `discussion`), or, if it doesn't affect the product, to `prompts/omit.json` with a reason. That covers commit/push/merge/deploy requests, running the app locally, environment or API-key setup, troubleshooting the dev setup, and housekeeping. Questions about how the product behaves and discussions behind decisions stay in. `python3 scripts/build_changelog.py` warns about unlabeled ones. Sessions run in Claude Code on the web aren't captured automatically (their commits carry a `Claude-Session:` trailer and show a badge). To import one, run `claude --teleport <session id>` from the repo root with a clean working tree, send one message so the transcript is saved, exit, then run `python3 scripts/export_prompts.py ~/.claude/projects/<project>/<new session>.jsonl`. List housekeeping messages like that one in `prompts/omit.json`. Commits link to the latest prompt within 12 hours before them.
- **Milestones:** when a change is significant, add a milestone to `prompts/milestones.json` (or extend the latest one with its prompts, commits and decisions). These feed the visual timeline at the top of the Changelog page, and the build warns about unknown references.
- **Changelog data:** `scripts/build_changelog.py` writes `frontend/src/changelog.json` (git-ignored) from git, `prompts/` and DECISIONS.md. `make frontend` and `make deploy` run it.

## Architecture

**Offline pipeline → artifacts → stateless API.** Heavy work is precomputed; a recommend request costs ~10–70 ms.

- `src/goodrec/pipeline/s00…s10` run in order. Each stage skips if its outputs exist (`--force` to redo). Thresholds and hyperparameters live in `config/pipeline.yaml`.
  - **Editions are collapsed to works** (`work_id`) in s03. The catalog (s04) is works with at least `catalog.min_raters` distinct raters, sorted by popularity. `work_idx` (dense 0..N-1) is the row/column index for every matrix and artifact, so s04 through s10 must be rebuilt together when the catalog changes.
  - s05 holds out `eval.n_test_users` users that **no model sees**; the eval harness folds them in, so eval measures real fold-in quality. Production artifacts are trained on the same train split. Item statistics (s10: raters, averages and `bayes`, the population stats and calibration prior) are computed without them too; `tests/test_eval.py` checks this.
- `src/goodrec/eval/`: the offline evaluation (DECISIONS D-037 to D-043).
  - `split.py`: per-user temporal split. Each held-out user's most recent 30% of ratings (by `date_added`, moved to a date boundary) are hidden. Eligible users are fixed into validation (30%, tuning) and test (70%, reported) sets. Saved to `data/interim/eval_split.npz`; its hash goes in every report, and it rebuilds when the split settings in `eval:` change.
  - `models.py`: the `Recommender` interface. Shelf Life is the served For you `ranking()` (YA included), plus ablations and the random / popular / genre + popularity baselines. `legacy2023.py` re-implements the 2023 project's three methods (similar readers, SVD, gradient-descent MF).
  - `run.py` runs everything in a forked process pool (`eval.workers`) and writes the report (`report.py`) and per-user results (`eval/runs/`, git-ignored except the champion's). `eval/champion.json` (via `--promote`, only with the user's go-ahead) is the "previous best" baseline for later runs. `metrics.py` has P/R/NDCG@k and the per-user paired comparisons. Baseline results are cached in `eval/runs/cache_*` keyed by their code, the split, the `eval:` settings and the artifacts manifest (`--fresh` recomputes); a cold run takes about 4 hours on an 8 GB laptop, mostly the 2023 gradient-descent baseline, and a cached one about 6 minutes.
  - s06 genres come from Goodreads reader shelves mapped through `config/shelf_genres.yaml` (hand-reviewed; parents list is fixed and mirrored in `s06_genres.PARENTS` and `frontend/src/genres.ts`).
  - The raw data URL moved to `mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/` (the old `datarepo.eng.ucsd.edu` URL 404s).
- `src/goodrec/core/` is shared by the API **and** eval, so eval always exercises serving code:
  - `artifacts.py` loads everything.
  - `scoring.py` has fold-in, item-item scoring, blend `a(n)=n/(n+k_a)`, filters, series rules and explanations.
  - Best match adds a fame-gated prediction boost, `a(n)·delta_pred·fame·z(pred)` (`score_all(preds=...)`, `fame_weight()`, `add_pred_candidates()`), applied in both `ranking()` and `blend(user=...)` so eval measures it. It's gated by Goodreads ratings count because raw predictions for obscure books mostly echo their high average.
  - For you excludes books whose calibrated prediction is below the user's average minus `blend.pred_floor_offset` (`prediction_floor()`, applied inside `ranking()` before the pool is built; needs `prior`, the dataset rating distribution, passed by the API). Popular / Top rated / To-read are not floored. `make eval` reports the floor's NDCG cost.
  - Young adult books (`is_ya`, built in s10: parent genre YA or YA vote share ≥ `catalog.ya_min_share`) are hidden unless `include_ya` is set or the Young Adult genre is selected. The YA toggle changes the ranking universe, so `ranking()` caches per `include_ya` and ranks are recomputed (not just filtered) when it flips. Explore defaults to showing YA.
  - Displayed predictions come from `shown_predictions()` / `apply_calibration(raw, cal, evidence)`: quantile calibration applied in proportion to each book's evidence weight (`predict_ratings(return_evidence=True)`, `CALIB_EVIDENCE_K`). This is **not monotone** in the raw prediction, so every "sort by predicted" must sort by displayed values (`ranking()["shown"]`, `main._shown`). The prediction boost still uses raw predictions. Item means (`bayes` in item_meta) are the dataset average shrunk toward the Goodreads-wide average (s10).
  - `predict_ratings` (predicted stars shown on every card) is a separate baseline + item-item residual model, not the blend score. The rank and the predicted rating can disagree by design; `make eval` reports its RMSE. `sort="predicted"` re-orders the whole blend candidate pool by prediction (ties by blend score) before paging. `readers_avg`/`readers_n` on each book come from `similar_readers()` (`item_avg`/`item_n`) and only exist once a user has at least 5 ratings.
  - `similar_readers.py` does nearest users in ALS space, then aggregates their actual shelves from `readers_csr.npz`.
  - Serving needs numpy/scipy (plus `anthropic` for the Assistant). `implicit` and polars are pipeline-only dependency groups.
- `src/goodrec/api/`: FastAPI.
  - **Public ids are Goodreads `work_id`s**, mapped to `work_idx` in `catalog.py`, so browser-stored ratings survive rebuilds.
  - Per-user model scores are LRU-cached by ratings hash (`main.py`), so filter/tab changes only re-filter.
  - CSV import matching (`matching.py`): Book Id → edition table, then ISBN, then normalized title + author last name (`core/textnorm.py`). Title matches **must** also match an author (main or additional). A title-only fallback matched post-2017 books to unrelated same-titled books, e.g. *The Anarchy* (Dalrymple) → *Anarchy* (Jaymin Eve).
  - `/api/chat` (`chat.py`, Assistant tab): Claude in a server-side tool loop, two tiers (`chat.models` in `pipeline.yaml`: Haiku for simple messages, Sonnet for complex ones; starter buttons carry a fixed tier, typed messages are labeled by a quick Haiku call in `choose_tier()`, and a conversation that reaches Sonnet stays there) streamed as SSE (`status`/`text`/`books`/`done`/`error`). Tools call the route functions directly (`recommend_route`, `insights`, `book_personal`, `Catalog.search`), so answers match the UI; plus Anthropic's server-side web search. The system prompt holds a library digest (ratings by star, to-read, out-of-catalog reads) and is prompt-cached. Catalog books come back as `[[book:<work_id>]]` markers that `ChatPage.tsx` renders as cover cards. Stateless: the browser sends shelf + transcript each turn (transcript in localStorage `goodrec.chat.v1`). Caps are in-memory per instance (`Limits`). `chat.py` reaches `main` through a lazy proxy (`api`) because `main` mounts the router. Tests use a fake Anthropic client (`tests/test_chat.py`); no test calls the real API.
  - The homepage offers two buttons, "See a demo" and "Use my own ratings" (the latter expands the upload / rate cards). "See a demo" calls `GET /api/demo`, which runs `demo/goodreads_demo.csv` (the site author's real Goodreads export) through the normal import matching; the store marks the shelf `demo: true` and Recommendations shows a demo banner. The Dockerfile copies `demo/`.
  - Import keeps review text (`My Review`, HTML stripped, 1,500 chars) and unmatched rows with rating/shelf; the store keeps them as `reviews` / `outside` for the Assistant only.
  - `/api/browse` (Explore page) reuses `filter_mask` with an empty user; the Explore filter set is separate from Recommendations' in `useUrlState.ts` and defaults to showing everything.
- `frontend/`: React + Vite + TS, no UI library.
  - Pages: Recommendations (list or map), Rate books, Import, Explore (`/api/browse`), Your books (`/api/books/batch` fills in full details for shelf ids), About (`pages/AboutPage.tsx`, holds `REPO_URL`), Changelog (`pages/ChangelogPage.tsx`, linked from About and the footer; a horizontally scrolling milestone timeline, `components/MilestoneTimeline.tsx`, sits above the full log). `components/Markdown.tsx` is the shared safe Markdown renderer (Assistant replies and the Changelog).
  - The user's shelf lives in localStorage (`store.tsx`). View, tab and filters live in the URL query string (`useUrlState.ts`).
  - Going to Recommendations from an import, "See a demo" or the Rate books shelf plays `components/CoverFlow.tsx`: 30 covers (the user's top-rated first, topped up from the homepage cover wall via `fillCovers()`), then their top 20 recommendations as a second wave once a prefetched `/api/recommend` answers (skipped if it's late). 4 s; the demo runs 5 s after a 0.75 s `lead` where only its caption ("…a real Goodreads user's ratings library") shows. Each cover eases to a pause halfway through its flight at its own horizontal spot (`--mid`, 18–74vw) rather than all at the centre. Recommendations is shown underneath at the flow's halfway point. Timings are `lead` + fractions of `duration`; the ending starts `FADE_EARLY_MS` (750 ms) before the last 15%: a white veil fades in over the covers, then the white overlay fades away to reveal Recommendations (`--fade-at` / `--reveal-at` / `--step` in CSS), ending at ~3.4 s (demo ~5.2 s). `App` owns the flow state and fetches the wall list once. Skipped for `prefers-reduced-motion`.
  - Genre colors are 8 validated family hues (`genres.ts`); text labels always carry identity.

## Gotchas

- About-page code links come from `scripts/code_links.py` (run by `make frontend`), which writes `frontend/src/codeLinks.json` with the current line of each linked function. It fails if a linked symbol is renamed or moved; update its `LINKS` table when that happens. Links target `main` on GitHub (`CODE_BASE` in `AboutPage.tsx`).

- Changing `blend`/`als` params in `config/pipeline.yaml` affects serving immediately (`Params.from_config`). ALS factor params require re-running s08 onward.
- Popular Goodreads titles embed series as `"Title (Series, #N)"`. `textnorm.parse_series` drives the "hide later volumes / show next in series" rule.
- The homepage cover wall is hand-picked in `config/home_wall.yaml` (literary / popular, alternated), resolved by title + author at API startup (`_home_wall()`). A test fails if any entry stops resolving. The Rate books grid still uses the computed `starter_shelf.json`.
- Cover URLs are 2017 Goodreads links (still live as of 2026-09). The frontend falls back to Open Library by ISBN, then a typographic placeholder.
- The data license is non-commercial; keep the citation in the footer.
