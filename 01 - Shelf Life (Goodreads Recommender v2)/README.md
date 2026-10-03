### Shelf Life: Goodreads Book Recommender (v2)

A web app that recommends books from your Goodreads ratings, in two ways:

1. **Upload your Goodreads library export** (CSV). The app uses your whole reading history.
2. **Search and rate a few books**. There's no account or export, and recommendations update as you rate.

No Goodreads history? **See a demo** on the homepage loads a real 770-book Goodreads library.

Results come in three views: **For you** (with "Because you liked X" explanations), **Popular with readers like you** and **Top rated by readers like you**. Every book shows your predicted rating next to its average among readers like you and its Goodreads average, and any list can be sorted by best match or predicted rating. A **Map** view plots your top 50 by Goodreads popularity (x, log scale) against average rating (y), colored by predicted rating; hover a dot for the cover and details, click for the full card. An **Explore** page browses the whole catalog with the same filters (genre, author, year, rating, number of ratings), sorted by most rated, highest rated (weighted by rating count), newest, oldest or title. There's also a genre profile and filters for genre, author, year, average rating and number of ratings. Clicking a book shows its description and similar books.

An **Assistant** tab is a reading assistant (Claude: Haiku 4.5 for simple messages, Sonnet 5 for complex ones): it sees your ratings, reviews, to-read list and out-of-catalog reads, calls the recommender as tools (For you, readers like you, catalog search, book details), can search the web for books published after 2017, and talks about books book-club style. Starter buttons cover the common requests.

**Description:**
Built on the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads) Goodreads data collected by Mengting Wan and Julian McAuley: 15.7M ratings from 465k users, collapsed from 2.36M editions to a catalog of about 105k works with at least 20 raters. It replaces the 2023 version, which refit a KNN model over the full ratings matrix on every request and took about a minute. Now all the heavy work happens offline, and a request takes milliseconds.

**How it works:**
* **Similar books (item-item):** precomputed top-50 neighbors per book, using adjusted cosine similarity with shrinkage on co-ratings. This works well with only a handful of ratings and drives the explanations.
* **Taste model (ALS):** implicit-feedback matrix factorization. A new user's vector is solved at request time from their ratings (fold-in).
* **Blend:** `a(n)·z(ALS) + (1−a(n))·z(item-item) + β·z(popularity)`, where `a(n) = n/(n+k_a)`. It leans on item-item for new users and on ALS as ratings accumulate.
* **Predicted rating:** the stars you'd likely give each book: the book's average (dataset readers' mean, shrunk toward its Goodreads-wide average), adjusted for how harshly or generously you rate, plus a correction from similar books you've rated. The raw estimate misses held-out ratings by 0.85 stars (RMSE) with a full history, vs 0.95 for the book's average. The displayed value is stretched toward your own rating distribution in proportion to how much of it rests on your ratings, so well-supported top picks still reach 4.5+ while unconnected books aren't pushed to extremes (displayed RMSE 0.87).
* **Prediction boost:** Best match adds `a(n)·0.75·fame·z(predicted rating)`, where fame runs from 0 at 10k Goodreads ratings to 1 at ~316k, so well-known books you'd likely love move up while obscure high-average books don't (about a 4% full-history NDCG cost).
* **Prediction floor:** once you've rated 5+ books, For you skips books predicted below your average minus 0.25★ (about a 4% NDCG cost, traded for recommendations that match their own predicted ratings).
* **Readers like you:** the nearest users in ALS embedding space, and what they actually read and rated.
* **Genres:** mined from Goodreads reader shelves (e.g. `cozy-mystery`, `space-opera`) and mapped to 205 descriptive tags and 34 parent genres. This replaces the old LDA topics.

**Offline evaluation** (9,906 held-out users no model saw; NDCG@20 after hiding 30% of each user's ratings; report in `eval/reports/`):

| method | 1 rating | 3 | 10 | 25 | all |
|---|---|---|---|---|---|
| **blend (tuned)** | 0.042 | 0.064 | 0.102 | 0.123 | 0.139 |
| item-item only | 0.038 | 0.061 | 0.095 | 0.112 | 0.134 |
| ALS only | 0.042 | 0.059 | 0.084 | 0.097 | 0.110 |
| most popular | 0.032 | 0.032 | 0.033 | 0.034 | 0.036 |

**Quickstart:**
```bash
make data        # download + parse the UCSD data (~7.5 GB; resumable)
make artifacts   # genres, models, search DB (~15 min on a laptop)
make frontend    # build the React app
make serve       # http://localhost:8000
```

**How it was built:** Shelf Life was built in conversation with Claude Code, and the whole process is public. The site's **Changelog** page (linked from About) shows every prompt, Claude's replies, the commits they produced and the decisions along the way. The same records are in [`prompts/`](prompts/) and [`DECISIONS.md`](DECISIONS.md).

**Assistant tab setup:** export `ANTHROPIC_API_KEY` before `make serve` (without it the tab says it isn't configured). For Cloud Run, run `make chat-secret` once (stores the key in Secret Manager); `make deploy` then mounts it. Usage caps live in `config/pipeline.yaml` under `chat:` (messages per visitor per day, global per day, turns per conversation); also set a monthly spend limit in the Anthropic console as a hard backstop.

**Filetree:**
```
├── config/            pipeline.yaml (thresholds, hyperparameters), shelf_genres.yaml (genre map)
├── src/goodrec/
│   ├── pipeline/      s00_download … s10_package (offline)
│   ├── core/          scoring, similar readers, artifacts, text normalization (shared by API + eval)
│   ├── api/           FastAPI app, CSV import matching, catalog search, chat (Assistant tab)
│   └── eval/          offline evaluation + ALS sweep
├── frontend/          React + Vite + TypeScript
├── tests/
├── Dockerfile
└── Makefile
```

Data: M. Wan & J. McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018; M. Wan, R. Misra, N. Nakashole & J. McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", ACL 2019. Non-commercial use.
