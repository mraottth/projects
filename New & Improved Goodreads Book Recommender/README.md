### Goodreads Book Recommender (v2)

A web app that recommends books from your Goodreads ratings, in two ways:

1. **Upload your Goodreads library export** (CSV). The app uses your whole reading history.
2. **Search and rate a few books**. There's no account or export, and recommendations update as you rate.

Results come in three views: **For you** (with "Because you liked X" explanations), **Popular with readers like you** and **Top rated by readers like you**. There's also a genre profile and filters for genre, author, year, average rating and number of ratings. Clicking a book shows its description and similar books.

**Description:**
Built on the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads) Goodreads data collected by Mengting Wan and Julian McAuley: 15.7M ratings from 465k users, collapsed from 2.36M editions to a catalog of about 105k works with at least 20 raters. It replaces the 2023 version, which refit a KNN model over the full ratings matrix on every request and took about a minute. Now all the heavy work happens offline, and a request takes milliseconds.

**How it works:**
* **Similar books (item-item):** precomputed top-50 neighbors per book, using adjusted cosine similarity with shrinkage on co-ratings. This works well with only a handful of ratings and drives the explanations.
* **Taste model (ALS):** implicit-feedback matrix factorization. A new user's vector is solved at request time from their ratings (fold-in).
* **Blend:** `a(n)·z(ALS) + (1−a(n))·z(item-item) + β·z(popularity)`, where `a(n) = n/(n+k_a)`. It leans on item-item for new users and on ALS as ratings accumulate.
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

**Filetree:**
```
├── config/            pipeline.yaml (thresholds, hyperparameters), shelf_genres.yaml (genre map)
├── src/goodrec/
│   ├── pipeline/      s00_download … s10_package (offline)
│   ├── core/          scoring, similar readers, artifacts, text normalization (shared by API + eval)
│   ├── api/           FastAPI app, CSV import matching, catalog search
│   └── eval/          offline evaluation + ALS sweep
├── frontend/          React + Vite + TypeScript
├── tests/
├── Dockerfile
└── Makefile
```

Data: M. Wan & J. McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018; M. Wan, R. Misra, N. Nakashole & J. McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", ACL 2019. Non-commercial use.
