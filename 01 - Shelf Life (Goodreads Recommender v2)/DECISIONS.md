# Decisions

The choices that shaped Shelf Life, oldest first. Each entry links the prompts where it was discussed (ids from
[`prompts/`](prompts/)) and the commits that carried it out. Numbers come from the offline evaluations or
measurements quoted in those conversations. Where the reasoning wasn't recorded, the entry says so.

Categories: `ui` · `model` · `eval` · `data` · `assistant` · `infra` · `docs`

## D-001 · Keep the UCSD Goodreads data, with the ratings file swappable
- **Date:** 2026-09-26
- **Category:** data
- **Prompts:** fe7c091a-004, fe7c091a-006
- **Commits:** 7ec278f

**Decision.** Rebuild on the same UCSD Book Graph data as the 2023 app: the reviews file (15.7M ratings from 465k readers). The pipeline keeps the ratings source swappable.

**Context.** There is no newer Goodreads dataset. The same repository also has an interactions file with about 104M ratings from 876k readers.

**Alternatives considered.** Switching to the interactions file now: a much stronger signal, but an ~11 GB download and a far slower pipeline on an 8 GB laptop.

**Why.** It keeps the project buildable on a laptop, and the stronger file can be swapped in later without code changes.

## D-002 · A hybrid recommender: similar books plus a taste model
- **Date:** 2026-09-26
- **Category:** model
- **Prompts:** fe7c091a-004, fe7c091a-005
- **Commits:** 7ec278f

**Decision.** Blend item-item similarity ("readers who liked X also liked Y") with an implicit-feedback ALS taste model, folding each new user in at request time.

**Context.** The 2023 app refit a nearest-neighbor model over 245k users on every request and took about a minute.

**Alternatives considered.** Item embeddings only; item-item only; fast user-user neighbors.

**Why.** Item-item works from a handful of ratings and gives "because you liked X" explanations. ALS captures broad taste once a reader has rated more. Both are precomputed offline, so a request takes milliseconds.

## D-003 · Python API with a React front end, heavy work offline
- **Date:** 2026-09-26
- **Category:** infra
- **Prompts:** fe7c091a-005, fe7c091a-006
- **Commits:** 7ec278f

**Decision.** An offline pipeline writes model artifacts, a FastAPI server loads them, and a React + Vite + TypeScript front end calls it.

**Alternatives considered.** A fully static, in-browser app; HTMX/Alpine instead of React.

**Why.** The models are too large to ship to the browser, and keeping training out of the request path is what makes responses fast.

## D-004 · Genres from readers' shelves, not LDA topics
- **Date:** 2026-09-26
- **Category:** data
- **Prompts:** fe7c091a-005, fe7c091a-006
- **Commits:** 7ec278f

**Decision.** Derive genres from the shelves Goodreads readers apply (e.g. `cozy-mystery`, `space-opera`), mapped to 205 descriptive tags and 34 parent genres. The mapping was drafted by an LLM and then reviewed.

**Alternatives considered.** Keep the 2023 LDA topics from book descriptions; hand-curate the whole mapping.

**Why.** Reader shelves are more accurate than inferred topics and give the genres readable names.

## D-005 · Delete the old notebooks from the copy
- **Date:** 2026-09-26
- **Category:** docs
- **Prompts:** fe7c091a-006

**Decision.** The rebuild started as a copy of the 2023 project; its notebooks and Flask app were deleted from the copy rather than moved to a `legacy/` folder.

**Why.** The original project still exists in its own folder, so the copy didn't need to carry it.

## D-006 · No cap on books per author
- **Date:** 2026-09-26
- **Category:** model
- **Prompts:** fe7c091a-007

**Decision.** Recommendations may include many books by the same author.

**Alternatives considered.** At most two books per author in the top 20, as first proposed.

**Why.** The user's call: if a reader would love several books by one author, the list should show them.

## D-007 · Filters for genre, author, year, rating and popularity
- **Date:** 2026-09-26
- **Category:** ui
- **Prompts:** fe7c091a-007
- **Commits:** 7ec278f, dde6c54

**Decision.** Beyond the proposed genre and year filters, add author, average rating, number of ratings and search-within-results. These later also became click-to-filter on cards.

## D-008 · One catalog entry per work, not per edition
- **Date:** 2026-09-26
- **Category:** data
- **Prompts:** fe7c091a-007
- **Commits:** 7ec278f

**Decision.** Collapse all editions of a book into one work, and keep works with at least 20 raters (about 105,000 books).

**Why.** Ratings for the same book are spread across editions. Merging them strengthens every model and stops the same book from appearing several times.

## D-009 · Measure on readers the models never saw
- **Date:** 2026-09-26
- **Category:** eval
- **Prompts:** fe7c091a-007
- **Commits:** 7ec278f, 7d7493b

**Decision.** Hold out 9,906 test readers that no model trains on. Fold them in exactly as the app does, hide 30% of their ratings and score with NDCG@20. The blend weights (k_a = 20, the popularity weight easing toward −0.3 for long histories) and the ALS settings (64 factors, regularization 1.0, alpha 30) were tuned this way.

**Why.** It measures what users actually experience, a cold fold-in, rather than in-sample fit.

## D-010 · A separate model for the predicted star rating
- **Date:** 2026-09-27
- **Category:** model
- **Prompts:** fe7c091a-019, fe7c091a-020
- **Commits:** 7d7493b

**Decision.** Show a predicted rating on every card from its own model: the book's average, adjusted for how harshly the reader rates, plus a correction from similar books they've rated. Also add "readers like you" averages and sorting by predicted rating.

**Why.** The ranking score says what a reader is likely to pick up, not how much they'd like it. The separate model beats the book's average on held-out ratings (RMSE 0.86 vs 0.97).

## D-011 · Title matches must also match an author
- **Date:** 2026-09-27
- **Category:** data
- **Prompts:** fe7c091a-022, fe7c091a-025
- **Commits:** 7d7493b, dde6c54

**Decision.** When importing a Goodreads export, a match by title alone isn't accepted; an author must match too. Re-imports replace earlier imports, and any book can be removed from the shelf.

**Context.** Post-2017 books were matching unrelated older books with the same title (*The Anarchy* by Dalrymple → *Anarchy* by Jaymin Eve; *Promised Land* → a Robert B. Parker novel).

## D-012 · Single-column ranked list with a prominent rank and score
- **Date:** 2026-09-27
- **Category:** ui
- **Prompts:** fe7c091a-019, fe7c091a-021, fe7c091a-023, fe7c091a-026
- **Commits:** 7d7493b, dde6c54

**Decision.** Recommendations are a single ranked column with a large rank number and a predicted-rating box. The tabs are ordered Recommendations, Explore, Your books, Rate books, Import, About.

**Context.** The first grid layout made the ranking hard to read.

## D-013 · An About page that explains the models, with math and code links
- **Date:** 2026-09-27
- **Category:** docs
- **Prompts:** fe7c091a-024, fe7c091a-026
- **Commits:** dde6c54

**Decision.** The About page explains each model in plain language, with LaTeX formulas and links to the exact functions on GitHub in collapsible technical sections. "Your books" adds reading insights (percentile, harshness, genre mix).

## D-014 · Host on Cloud Run, scaling to zero, with a $5 budget alert
- **Date:** 2026-09-27
- **Category:** infra
- **Prompts:** fe7c091a-035, fe7c091a-036, fe7c091a-037, fe7c091a-038, fe7c091a-039
- **Commits:** ebef015

**Decision.** Deploy to Google Cloud Run with no always-on instance and at most 3 instances, and set a $5/month budget alert.

**Alternatives considered.** Hugging Face Spaces (free, but a sleeping Space can take a minute or more to wake); Cloud Run with a warm instance (about $10–15/month).

**Why.** Close to free for portfolio traffic, with cold starts of seconds rather than minutes. *Superseded by D-015.*

## D-015 · Keep one warm Cloud Run instance
- **Date:** 2026-09-28
- **Category:** infra
- **Prompts:** ec278335-002, ec278335-003
- **Commits:** d7f2ad9

**Decision.** Set Cloud Run's minimum instances to 1, so visitors never wait for a cold start. `make deploy` keeps the setting.

**Context.** Claude's estimate (not measured): after the site sits idle, a first visit would wait about 5–15 seconds with no warm instance (starting the container, loading Python and the ~300 MB of models) versus under 1 second with one.

**Alternatives considered.** Keep scaling to zero (D-014): free when idle, but with that cold-start wait.

**Why.** A fast first impression matters for a portfolio site, worth roughly $10–15/month.

## D-016 · Ranks don't change when filters are applied
- **Date:** 2026-09-27
- **Category:** ui
- **Prompts:** fe7c091a-041
- **Commits:** 32555e9

**Decision.** Each sort has one full ranking. Filters hide books but keep their original numbers, so #12 stays #12 under any filter.

**Why.** Renumbering made every filtered list start at #1, which hid how a book ranks overall.

## D-017 · Calibrate predicted ratings to the reader's own scale
- **Date:** 2026-09-27
- **Category:** model
- **Prompts:** fe7c091a-042
- **Commits:** 32555e9

**Decision.** Show predictions with two decimals, and map them onto the reader's own rating distribution so a tough grader's top picks can still reach 4.5+.

**Context.** Raw predictions cluster near each reader's average. *Refined by D-027.*

## D-018 · A floor on predicted ratings in For you
- **Date:** 2026-09-27
- **Category:** model
- **Prompts:** fe7c091a-043, fe7c091a-044
- **Commits:** 32555e9

**Decision.** For you skips books predicted below the reader's average minus 0.25 stars.

**Context.** A book with a 2.42 prediction appeared in the user's top recommendations.

**Alternatives considered.** A floor exactly at the average (−10% full-history NDCG); a weight on the predicted rating in the score (−5% to −14%).

**Why.** −0.25 keeps most of the benefit for about a 4% NDCG cost.

## D-019 · No extra ubiquity correction for "readers like you" lists
- **Date:** 2026-09-27
- **Category:** model
- **Prompts:** fe7c091a-043, fe7c091a-044

**Decision.** Don't add a further correction to the Popular and Top rated lists to discount books that everyone reads. (The Popular score already divides a book's reach among similar readers by the square root of its reach among all readers.)

**Alternatives considered.** A "ubiquity" correction that would push widely read books further down.

**Why.** The user's call: "That feels like putting our finger on the scales too much. If a book is extremely popular with readers like me that's a useful signal."

## D-020 · Hide young adult books by default
- **Date:** 2026-09-27
- **Category:** model
- **Prompts:** fe7c091a-045
- **Commits:** 32555e9

**Decision.** Recommendations leave out young adult books unless the reader ticks "Include young adult books", which re-ranks the whole list rather than just filtering it. A book counts as YA if its main genre is Young Adult or at least 22% of readers' genre votes say so (about 11,700 books).

**Why 22%.** 25% let YA novels like *Steelheart* and *The Rithmatist* through. 22% catches them while leaving adult novels like *Red Rising*, *The Giver* and *The Book Thief* in.

## D-021 · A simpler homepage with a hand-picked cover wall
- **Date:** 2026-09-27
- **Category:** ui
- **Prompts:** fe7c091a-046, fe7c091a-047, fe7c091a-048
- **Commits:** 32555e9

**Decision.** Remove the explanatory copy, and fill the cover wall with a hand-picked list (`config/home_wall.yaml`): half literary fiction, half popular or famous titles, with no YA.

## D-022 · An Assistant tab built on the recommender
- **Date:** 2026-09-27
- **Category:** assistant
- **Prompts:** fe7c091a-050, fe7c091a-051
- **Commits:** 8ab67ef

**Decision.** Add a Claude-powered reading assistant that calls the recommender as tools (For you, readers like you, catalog search, book details, the reader's shelves and reviews). It can search the web for books published after 2017, and it can discuss books like a book club. It runs on the site owner's API key with per-visitor and daily caps.

**Alternatives considered.** Visitors bring their own key; a passphrase; local-only.

**Why.** Tools keep its answers consistent with the app's rankings, while Claude adds judgment and books outside the 2017 data.

## D-023 · Haiku for simple messages, Sonnet for complex ones
- **Date:** 2026-09-27
- **Category:** assistant
- **Prompts:** fe7c091a-058, fe7c091a-059, fe7c091a-060, fe7c091a-061
- **Commits:** 8ab67ef

**Decision.** A quick Haiku call labels each typed message simple or complex. Starter buttons carry a fixed tier, and a conversation that reaches Sonnet stays there.

**Context.** In a side-by-side test, Haiku answered catalog questions well at a fraction of the cost, but it handled the "books since 2017" question poorly (no web search, recommended books the user had read).

**Why.** Most of the savings with little quality loss. Staying on Sonnet avoids re-paying for the cached library summary.

## D-024 · Give the Assistant room to think
- **Date:** 2026-09-27
- **Category:** assistant
- **Prompts:** fe7c091a-057
- **Commits:** 8ab67ef

**Decision.** Raise the reply budget to 16,000 tokens and set Sonnet's effort to medium. Show an error instead of an empty bubble if a reply has no text.

**Context.** Replies came back blank because thinking used the entire 1,500-token budget.

## D-025 · A fame-gated boost for high predicted ratings
- **Date:** 2026-09-27
- **Category:** model
- **Prompts:** fe7c091a-063, fe7c091a-066
- **Commits:** 8511b03

**Decision.** Best match adds `a(n) × 0.75 × fame × z(predicted rating)`, where fame goes from 0 at 10k Goodreads ratings to 1 at about 316k.

**Context.** *The Green Mile* was the user's #1 by predicted rating but #384 by Best match, because none of their 300 most similar readers had read it.

**Alternatives considered.** No gate (−14% NDCG); boost strength 1.0 (−7.9%); a gate on similar-reader overlap, which would have given *The Green Mile* no boost at all.

**Why.** Well-known books the reader would likely love move up (*The Green Mile* #384 → #96), while obscure books with high averages don't. The cost was about −5.6% full-history NDCG, later −4.1% after D-026.

## D-026 · Shrink book averages toward the Goodreads-wide average
- **Date:** 2026-09-27
- **Category:** model
- **Prompts:** fe7c091a-069, fe7c091a-070
- **Commits:** 9aba0e6

**Decision.** A book's baseline is the average among dataset readers, shrunk toward its Goodreads-wide average by 50 pseudo-ratings.

**Alternatives considered.** Use the Goodreads average outright. It was worse on held-out readers (RMSE 0.876 vs 0.864).

**Why.** It improved prediction RMSE at every history length (full history 0.864 → 0.852).

## D-027 · Stretch predictions only as far as personal evidence supports
- **Date:** 2026-09-27
- **Category:** model
- **Prompts:** fe7c091a-069, fe7c091a-070
- **Commits:** 9aba0e6

**Decision.** Calibration moves each prediction toward the reader's scale in proportion to how strongly the book is linked to books they rated (weight den / (den + 0.1)).

**Context.** Full calibration multiplied small, noisy differences between book averages about 4×. *The Magic Mountain* and *Buddenbrooks* showed 2.64 vs 3.89, though they have the same Goodreads average.

**Alternatives considered.** Cap the overall stretch, which would have compressed every prediction again; K = 0.5 (most accurate, but top picks never reached 4.5).

**Why.** Displayed RMSE fell from 0.998 to 0.879, while the top picks still reach 4.5+.

## D-028 · Number-prefixed folders to order the portfolio repo
- **Date:** 2026-09-27
- **Category:** docs
- **Prompts:** fe7c091a-072, fe7c091a-073, fe7c091a-075, fe7c091a-077, fe7c091a-078
- **Commits:** 3148292

**Decision.** Prefix the repo's top-level folders 01–10 in the owner's chosen order (Shelf Life first), mirror that order in the root README, and give the two Goodreads projects clearer names. Commit messages stay neutral ("Reorganize repo") because they're public.

**Why.** GitHub always lists folders alphabetically.

## D-029 · A public demo library from a real Goodreads export
- **Date:** 2026-09-28
- **Category:** ui
- **Prompts:** fe7c091a-080
- **Commits:** a0ea42a

**Decision.** "See a demo" loads the site owner's real Goodreads export (772 books) through the normal import, so visitors without a Goodreads history can see the app work.

**Why.** Most portfolio visitors won't have their own export. A real library shows real results. Publishing it was the owner's choice.

## D-030 · A two-choice homepage and a cover-flow transition
- **Date:** 2026-09-28
- **Category:** ui
- **Prompts:** fe7c091a-081, fe7c091a-082, fe7c091a-083, fe7c091a-084, fe7c091a-085, fe7c091a-086, fe7c091a-087, fe7c091a-088, ec278335-010, ec278335-013, ec278335-014
- **Commits:** a0ea42a, 9e6ef1f, 8c6eb2a, 7c50abd

**Decision.** The homepage offers "See a demo" or "Use my own ratings", which expands the upload and rate options. Going to Recommendations plays a short transition: the reader's 30 top-rated covers (topped up from the cover wall), then their top 20 recommendations, flow across the screen and fade to white. The demo adds a caption lead-in.

**Context.** Timings were tuned by eye over several prompts. Later, to stop covers bunching up mid-screen, Claude tried a constant-speed version with spaced lanes (on-screen covers ~24 → ~17). The user preferred the original feel, so it was reverted, and each cover now pauses at its own random point (18–74% across) instead of the center (covers in the middle fifth: 26% → 15%). The demo lead-in was shortened to 0.75 s.

## D-031 · Collapse the top nav into a menu on phones
- **Date:** 2026-09-28
- **Category:** ui
- **Prompts:** ec278335-004, ec278335-005
- **Commits:** 4c01f31

**Decision.** At 640 px wide or less, the seven top-bar links collapse into a drop-down menu. The bar shows the current page, and a dot on the menu button stands in for the Your books count. The recommendation tabs keep sideways scrolling, but snap to tabs and scroll the selected one into view.

**Context.** The two toolbars were squished on phones.

**Alternatives considered.** A bottom tab bar, as in native apps.

**Why.** It matches what the user suggested, and leaves the desktop layout unchanged.

## D-032 · Short tooltips on the nav links
- **Date:** 2026-09-28
- **Category:** ui
- **Prompts:** ec278335-010, ec278335-011
- **Commits:** 9e6ef1f, 0aa244a

**Decision.** Each top-nav link gets a few words on what the page is for (e.g. "See your personal picks"). It appears on hover or focus on desktop and as a subtitle in the phone menu. The user chose the wording for three of them.

## D-033 · A plainer About page
- **Date:** 2026-09-28
- **Category:** docs
- **Prompts:** ec278335-010, ec278335-016
- **Commits:** 9e6ef1f, 28950bf

**Decision.** The About page's visible text explains what each model is, why it's used, how it works and what the results mean. Memory use, data types and run times were removed from the technical sections; the math and code links stay.

**Why.** The user found those engineering details beside the point for readers.

## D-034 · Keep the name Shelf Life; add a logo and favicon
- **Date:** 2026-09-28
- **Category:** ui
- **Prompts:** ec278335-015, ec278335-016, ec278335-017
- **Commits:** 28950bf, 75171ed

**Decision.** Keep "Shelf Life", and use the user's logo (book-and-worm icon plus wordmark) in the top bar and as the favicon.

**Alternatives considered.** Next Chapter, Kindred Reads, Dog-Ear.

**Why.** Claude still recommended Shelf Life (memorable, hints at a reading history over time, already throughout the site and repo), and the user kept it.

## D-035 · Leave the readers-like-you rankings as they are, for now
- **Date:** 2026-09-28
- **Category:** model
- **Prompts:** ec278335-019, ec278335-020, ec278335-021

**Decision.** Keep the Popular and Top rated tabs unchanged: no prediction floor, and no minimum share of similar readers.

**Context.** Popular's Best match ranks by reach among similar readers ÷ √(reach among all readers). That favors books over-represented among similar readers over books most of them read. *Superintelligence* led the user's list with 5% reach and a 2.56 predicted rating.

**Alternatives considered.** Apply the For you prediction floor to these tabs (about a 10-line server change); require at least ~10% of similar readers to have read a book.

**Why.** The user's call ("let's keep it as is for now"), consistent with D-019. It may be revisited.

## D-036 · Log every prompt, decision and commit in the open
- **Date:** 2026-10-03
- **Category:** docs
- **Prompts:** fe7c091a-090, fe7c091a-091, fe7c091a-092

**Decision.** Export each conversation with Claude Code to `prompts/` (automatically, via a hook), keep this file, and show both with the commits on a public changelog page. Sessions run in Claude Code on the web are imported with `claude --teleport` and the same exporter. Commits from here on are small, with messages of the form `category: summary`.

**Why.** Shelf Life was built through conversation, so the conversation is part of how it was made.

## D-037 · Evaluate with a per-user temporal split
- **Date:** 2026-10-03
- **Category:** eval
- **Prompts:** fe7c091a-110, fe7c091a-111, fe7c091a-112, fe7c091a-115
- **Commits:** da21a59, d57a932, 9aae987, 457b7c4

**Decision.** Offline evaluation hides each held-out user's most recent 30% of ratings and asks the model to predict them from everything before. Ratings are ordered by the date the user read the book (the data's `read_at`, given for 82.5% of ratings), or the date it was shelved when no plausible read date is given; the data has no date-rated field. The split point is relative to each user's own history, not a calendar date. Books with the same date stay on the same side (the split moves to the nearest date boundary), because their order within a day is unknown. A user is evaluated with at least 10 ratings, at least 7 visible and at least 3 hidden books rated 4+: 8,422 of the 10,000 held-out users.

**Context.** The previous evaluation hid a random 30% of ratings, so models were partly asked to predict the past.

**Ordering.** The first version ordered by shelving date (8,278 users); the user asked to order by date read or rated when the data has it, since back-filled shelves say little about reading order.

**Alternatives considered.** A single global cutoff (2017-01-01, with models retrained on earlier data) was proposed first. It would have evaluated 3,535 users and left out the 2,600 held-out users whose last rating was before 2016. The user preferred a per-user split: "the cutoff should be relative to the user's history, not an arbitrary calendar date. We want enough history to make a meaningful profile and enough future observations to evaluate it."

## D-038 · NDCG@10 as the primary metric; hits are hidden books rated 4+
- **Date:** 2026-10-03
- **Category:** eval
- **Prompts:** fe7c091a-110, fe7c091a-111
- **Commits:** 0e0f582, d57a932

**Decision.** NDCG@10 is the primary metric, with Recall@10 and Precision@10 as secondary metrics and all three also at 20. Relevance is binary: a hidden book the user rated 4 or 5 stars. Results are broken down by cutting each user's visible history to their 1, 3, 5, 10 and 25 most recent ratings, plus the full history. Reports compare the model with each baseline per user: wins, ties and losses, the mean difference with a bootstrap 95% interval, relative lift and the median gain among wins.

**Alternatives considered.** Grouping users by their real history size plus a separate cold-start section (Claude's suggestion); the user chose truncation only, as in the earlier evaluation. Counting only 5-star books, or any book read, as hits; the user chose 4+.

## D-039 · Evaluate the production models and state their exposure
- **Date:** 2026-10-03
- **Category:** eval
- **Prompts:** fe7c091a-111, fe7c091a-113
- **Commits:** d57a932

**Decision.** The harness evaluates the production models. Evaluation users are excluded from all training (s05), and the For you ranking is measured as served: prediction floor and boost on, young adult books included, no other content filters. Every report lists the exposure that remains: other training readers' ratings dated after a user's split date, Goodreads-wide rating counts and averages (a 2017 snapshot), and catalog membership counted over all users.

**Why.** With a different split date per user, the models can't be retrained to stop at each user's date. Retraining was only possible with the single global cutoff the user decided against (D-037).

## D-040 · Compute item means and the calibration prior from training users only
- **Date:** 2026-10-03
- **Category:** model
- **Prompts:** fe7c091a-113
- **Commits:** fa55ca5

**Decision.** s10 computes each book's rater count and average (the inputs to its item mean) from the training matrix, and the population statistics, including the calibration prior, exclude the held-out users.

**Context.** Checking for leakage, at the user's request, found that item means used the catalog's counts and averages over every user, including the evaluation users. Their hidden ratings fed their own predicted ratings, the prediction floor and calibration. 90% of books' item means changed by up to 0.13 stars after the fix.

**Numbers (validation users, full history).** Predicted-rating RMSE 0.8732 with the leak, 0.8777 without; item-mean RMSE 0.9689 → 0.9740. NDCG@10 0.0557 → 0.0558 (ranking was barely affected). Reports: `eval/reports/2026-10-03_1201_validation.md`, `eval/reports/2026-10-03_1203_validation.md`.

## D-041 · Tune on validation users, report on test users
- **Date:** 2026-10-03
- **Category:** eval
- **Prompts:** fe7c091a-113
- **Commits:** da21a59, d57a932, 12f8f9d

**Decision.** The evaluation users are split once, with a fixed seed, into validation (30%, 2,527 users) and test (70%, 5,895 users). Grid searches and the ALS sweep run only on validation; reported results and decisions about the best model use the test set.

**Why.** Part of the leakage precautions the user asked for: tuning on the same users the results are reported on would overstate them.

## D-042 · Re-implement the 2023 recommender's three methods as baselines
- **Date:** 2026-10-03
- **Category:** eval
- **Prompts:** fe7c091a-110, fe7c091a-112
- **Commits:** 05a0fbb

**Decision.** The 2023 project's methods run as three baselines on this project's training data, with the original parameters and filters: similar readers (the 150 nearest readers' most-rated books), SVD of the 3,000 nearest readers (what the 2023 web app served) and gradient-descent matrix factorization of the 1,000 nearest. The gradient descent reproduces the original update exactly, including a quirk: NumPy keeps only the last write for repeated indices, so each step updates each reader and book from a single rating. Gradient descent runs on a fixed subsample of 1,000 users because it is slow: the first full run took 3 h 51 min on an 8 GB laptop, almost all of it this baseline. Baseline results are cached and reused until their code, the split, the settings or the artifacts change.

**Result (first, shelving-date split; test set, full history, NDCG@10).** Similar readers 0.0338, SVD 0.0279, gradient descent 0.0012 (random scores 0.0002; with one rating per reader and book per step it barely learns). Shelf Life scores 0.0565: 67% above the best 2023 method, winning for 24.1% of users, tying for 62.7% and losing for 13.3% (`eval/reports/2026-10-03_1556_shelf_life.md`).

**Why.** The original code reads its own data files, which use edition ids and a different set of users and may include the evaluation users, so it can't be run as is. Reproducing the original update, quirk included, compares against what the 2023 project actually did. The user pointed out that the 2023 project used matrix factorization as well as SVD.

## D-043 · Keep a record of the best model so far
- **Date:** 2026-10-03
- **Category:** eval
- **Prompts:** fe7c091a-110, fe7c091a-114
- **Commits:** d57a932

**Decision.** `eval/champion.json` records the best model so far (parameters, commit, split hash, report). Later runs include it as the "previous best" baseline, reusing its per-user results when the split is unchanged. A model becomes champion only with `--promote`, when it beats the current one on test-set NDCG@10, and only when the user confirms.

**First champion.** The current configuration, recorded on the date-read split (`54b863801d57`) at user's request: test-set NDCG@10 0.0668 with full histories. It beats the best 2023 method (similar readers, 0.0403) by +0.0264 (95% CI +0.0237 to +0.0294, a 66% lift), winning for 27.0% of users, tying for 59.5% and losing for 13.5%; popular books score 0.0229 (`eval/reports/2026-10-03_1817_shelf_life.md`).

## D-044 · Drop the 2023 gradient-descent baseline
- **Date:** 2026-10-03
- **Category:** eval
- **Prompts:** fe7c091a-114

**Decision.** Remove the 2023 notebook's gradient-descent matrix factorization from the evaluation baselines. The 2023 similar-readers and SVD methods stay.

**Why.** The 2023 web app never served it; it existed only in the notebook. And because of how NumPy handles repeated indices in its update (D-042), each step learned from a single rating per reader and book, so it scored close to random (NDCG@10 0.0012 against random's 0.0002). It also accounted for most of the first full run's 3 h 51 min.

## D-045 · Cap the taste model's share of the blend at 0.4
- **Date:** 2026-10-03
- **Category:** model
- **Prompts:** fe7c091a-114
- **Commits:** c7e2c2a, 70a21cd, 68a1886, 9ba1059

**Decision.** The blend gives the taste model (ALS) a weight of a(n) = n/(n+20), now capped at 0.4 (`blend.a_max`), and the popularity penalty for heavy readers goes from -0.3 to -0.5. Readers with up to about 13 ratings are almost unaffected; above that, the similar-books model keeps at least 60% of the say.

**Context.** The temporal evaluation showed the similar-books model alone beating the blend with long histories (test NDCG@10 0.0759 against 0.0668). The old weights were tuned with the random-holdout evaluation, which favoured the taste model.

**Numbers.** Chosen on the validation users only: a 30-point sweep (k_a, a_max, beta_pop_many) on 1,000 users, then the four best on all 2,527 (full-history NDCG@10 0.0682 → 0.0777, +0.0096, 95% CI +0.0074 to +0.0115, with no history size significantly worse). Re-checking the prediction boost and floor at this setting changed NDCG@10 by at most 0.0003, so they stay. Confirmed on the 5,895 test users against the champion: full history 0.0668 → 0.0770 (+0.0102, CI +0.0088 to +0.0118; wins for 16.9% of users, losses for 9.9%), 25 ratings +0.0011 (CI +0.0004 to +0.0019), 5 ratings -0.0003 (CI -0.0005 to -0.0001, from the steeper popularity penalty), other sizes unchanged (`eval/reports/2026-10-03_2023_shelf_life.md`).

## D-046 · Recent ratings should count more; this needs rating dates in the app
- **Date:** 2026-10-03
- **Category:** model
- **Prompts:** fe7c091a-114, fe7c091a-115
- **Commits:** 0740cac, 9ba1059

**Finding.** Weighting each rating by how recent it is (the k-th most recent counts 0.5^(k/25)) raises full-history NDCG@10 on the validation users from 0.0777 to 0.0872 (+0.0095, 95% CI +0.0075 to +0.0116), on top of the blend cap (D-045). Half-lives of 10, 50 and 100 gained less (+0.0076, +0.0074, +0.0052). Shorter histories are essentially unchanged, since a handful of ratings are all recent. The order comes from when each book was read (date shelved when no read date is given).

**Decision.** Not shipped yet. The app keeps no dates for ratings: Goodreads imports drop "Date Read" and "Date Added", and ratings made on the site aren't timestamped. Shipping it means keeping those dates from import to scoring, which is planned as a separate step for the user to approve. The scoring code supports it already (`UserInput.recency`, `recency_weights()`), with no effect while no dates are passed.

## D-047 · Weight recent reading more in the app
- **Date:** 2026-10-03
- **Category:** model
- **Prompts:** fe7c091a-116
- **Commits:** 9ac0ca0, 1145474, ff5efd2, 5d7f534

**Decision.** Recommendations weight each rating by how recently the book was read: 0.5^(k/25), where k is the number of the reader's ratings dated later. Dates come from a Goodreads import ("Date Read", else "Date Added"); ratings made on the site are dated the day they're made. Ratings with the same date share a weight, so a shelf rated in one sitting is unaffected; undated ratings count as oldest, and a shelf with no dates (saved before this change) is scored exactly as before. The dates live in the browser with the rest of the shelf and are sent with each request, like the ratings.

**Why.** D-046's offline result. Re-checked with the app's tie rule on the validation users (full-history NDCG@10 0.0777 → 0.0870 at a half-life of 25; 15 and 40 slightly lower), then confirmed on the test users against the champion: 0.0770 → 0.0874 (+0.0104, 95% CI +0.0090 to +0.0118; wins for 17.8% of users, losses for 9.4%), with shorter, undated histories unchanged (`eval/reports/2026-10-03_2124_shelf_life.md`). On the demo library, 7 of the top 10 recommendations stay the same.

## D-048 · Chart the model's versions on today's test split
- **Date:** 2026-10-03
- **Category:** eval
- **Prompts:** fe7c091a-118, fe7c091a-119

**Decision.** An Evaluation page, linked from About, charts the model's five versions on the same temporal test split, with every evaluation report below it (the champion's first). Versions that predate the temporal evaluation (launch, and the prediction floor) were re-scored with their own settings, at the user's request, rather than shown only from the point the evaluation existed.

**Caveat.** Only the blend settings changed between versions (item-kNN and ALS settings are unchanged since launch), so each version can be re-scored by its settings, but every version uses today's data: the Goodreads-shrunk item means and the leak fix (D-026, D-040) changed the data, not settings. The page says re-scored versions differ slightly from what they scored at the time.

**Numbers (test, full history, NDCG@10).** v1 launch 0.0645, v2 floor 0.0651, v3 fame-gated boost 0.0668, v4 blend cap 0.0770, v5 recency 0.0874 (+35% since launch). At 5 ratings, launch scores higher (0.057 against 0.052): the floor costs accuracy for short histories, as D-018 accepted.

## D-049 · Vary authors in the Best match list (display only)
- **Date:** 2026-10-03
- **Category:** model
- **Prompts:** fe7c091a-132, fe7c091a-133, fe7c091a-134, fe7c091a-135
- **Commits:** 00dd8a1, 42342b6, 0d3b97f

**Decision.** The Recommendations page's Best match list (which the Assistant follows) re-orders the model's top 100 so repeat authors are nudged down: each next book scores its blend score minus 0.5 for every book by the same author already above it. The model's own ranking is unchanged, so the evaluation, the champion and the Evaluation chart keep scoring the model; reports add a "Best match as displayed" row and an author-variety table so the cost stays visible. Predicted ratings and the predicted-rating sort are untouched.

**Context.** This partly reverses D-006 (no cap on books per author). The user's list had five Brandon Sanderson books in the top 10 after reading and loving three recently: "part of me wants to leave the recommendations as they are predicted, but I think variety is important."

**Alternatives considered.** A hard cap per author (rejected in D-006). The same penalty inside the model, making a lower-scoring v6 the champion (Claude's first proposal); the user preferred to leave the model untouched and change only the displayed order. Re-ordering in the browser, which only sees one page at a time and would shift ranks under filters.

**Numbers (validation, full-history NDCG@10 cost; distinct authors in the top 10; most from one author).** Penalty 0.1: −0.3%, 8.6, 2.1. 0.25: −0.8%, 8.8, 1.9. **0.5: −1.9%, 9.1, 1.7.** 1.0: −3.1%, 9.4, 1.5 (no penalty: 8.4, 2.3). On the demo library, five Sanderson books in the top 10 become four, spread out (#1, 3, 7, 10) instead of clustered; 0.75 would leave three. The user chose 0.5. On the test readers: the model scores 0.0874 (unchanged) and the displayed list 0.0860 (−1.6%); distinct authors in the top 10 rise from 8.4 to 9.1 and the most from one author falls from 2.3 to 1.7 (`eval/reports/2026-10-04_0004_shelf_life.md`).

## D-050 · Evaluate rating prediction as a second track
- **Date:** 2026-10-04
- **Category:** eval
- **Prompts:** fe7c091a-139, fe7c091a-140, fe7c091a-141
- **Commits:** e08af3c, a316c08, 8fd70c5, 3b069fc, 00d9b98, 24f06ad

**Decision.** The evaluation has two clearly separated tracks on the same test readers, truncated histories and hidden books. Track 1, recommendation quality, is unchanged (NDCG, Recall and Precision at 10 and 20). Track 2, rating prediction, scores the predicted rating a book card shows against every hidden rating (1–5★).
- **Metrics.** MAE is primary. Secondary: RMSE, Pearson correlation pooled over all ratings, and Spearman within each reader's hidden books (readers with at least 3 varied ratings, 92% of them). Also: share within ±1★ and ±0.5★, mean signed error, a half-star calibration table, error by actual star and prediction spread. Metrics are averaged per reader, then over readers, as in Track 1.
- **Rating scales.** The model already adapts to them (the reader's offset b_u and calibration to their own histogram). The evaluation checks this with:
  - baselines that know the reader's scale (book average + offset, the reader's average);
  - per-reader Spearman, which ignores the scale;
  - MAE divided by σ_u, the spread of the reader's visible ratings shrunk toward the population's;
  - results by rating style (narrow, typical, wide), with cut points from the validation readers' terciles.
- **Leakage.** Predictions and σ_u see only visible ratings and training-only item statistics; a test checks that changing hidden ratings changes no prediction.
- **Baselines.** Book average (the requested one), book average + reader offset, the reader's average, the Goodreads average (reference; 2017 snapshot) and both 2023 methods' ratings. The 2023 methods fall back to the book average when they have no estimate (similar readers: 50% of books; SVD: 10%). The better one is labelled "2023 best". Gradient descent stays excluded (D-044).
- **Champion.** NDCG@10 still decides, with a rating guardrail: no promotion if per-reader MAE is significantly worse than the champion's (bootstrap CI of the difference entirely above 0). This was the user's choice.
- **Evaluation page.** A Ranking | Rating switch changes the intro, the chart, the panel and every report card (the user's request).

**Context.** The user wanted to know whether the model predicts ratings well, not just which books someone will like, and to account for how differently people use the scale without test-set leakage. Before this, reports carried one RMSE number for the raw, uncalibrated prediction, not the rating users see.

**Alternatives considered.** Normalizing ratings per reader with z-scores from hidden ratings (leaks; σ_u uses visible ratings only). Picking the champion on MAE, or a combined score (the user chose NDCG@10 with the guardrail). Style groups from test-set terciles (would let test ratings define the groups).

**Numbers (test, 5,895 readers, full history, v5).**
- Displayed rating: MAE 0.675, RMSE 0.879, Pearson 0.504, per-reader Spearman 0.314, 77.6% within one star.
- Uncalibrated: MAE 0.672.
- Book average + reader offset: 0.678 (the model wins for 49% of readers and loses for 45%; mean difference −0.0025, CI −0.0036 to −0.0012).
- Book average: 0.763. The reader's average: 0.727. Goodreads average: 0.764. 2023 similar readers: 0.792. 2023 SVD: 0.797.
- With one visible rating: displayed 0.740, book average + offset 0.742.
- Every method over-predicts the books readers disliked (+2.3★ bias at 1★) and under-predicts 5★ books (−0.65).

So most of the gain over book averages comes from the reader's offset; the similar-books residual adds little on MAE. Calibration costs 0.003 MAE but keeps predictions more spread out (59% of the actual spread vs 53%).

## D-051 · Reconstruct earlier rating models for the version history
- **Date:** 2026-10-04
- **Category:** eval
- **Prompts:** fe7c091a-140
- **Commits:** a316c08, c85dd3f

**Decision.** The five versions differ mostly in ranking settings, so with today's code their predicted ratings would be identical. To chart a real rating history, the evaluation can reconstruct earlier rating models (`RatingSettings`, `run.py --rating`, evaluation only; `core/` is unchanged).
- **Item means.** "global" is the training average shrunk toward the global mean, as before D-026.
- **Calibration.** "full" applies the whole stretch to every book (the first calibration, before the evidence gate of D-027); "none" is the raw prediction.
- **Per-version settings** (from the git history; recorded in `eval/versions.json`):
  - v1: global, none;
  - v2: global, full (calibration arrived in 32555e9);
  - v3–v5: today's (9aba0e6).

Each version was re-run on the test set against the previous one, so the chart shows per-reader intervals between neighbours. Reconstructed versions run on today's data and artifacts; the page says so. This was the user's choice ("Reconstruct them").

**Numbers (full-history MAE).** v1 0.677, v2 0.755, v3–v5 0.675 (v4 and v5 change only the ranking: identical predictions). Full calibration made predictions worse (v2 vs v1: +0.078, CI 0.074 to 0.083), and the evidence gate recovered it (v3 vs v2: −0.080), matching the earlier finding behind D-027. Track 1 numbers of all five re-runs match the earlier reports exactly.

## D-052 · Predict ratings with a factorization model under the item-kNN residual
- **Date:** 2026-10-04
- **Category:** model
- **Prompts:** fe7c091a-143, fe7c091a-144, fe7c091a-145, fe7c091a-146
- **Commits:** a325aa9, c05c7a4, 854dfe3, e49f528, ab2c868, aa67b98, cf0a36e, 5c9e05e

**Decision.** Predicted ratings start from a biased matrix-factorization model, r̂ = μ + b_u + b_i + p_u·q_i with 16 factors. Today's item-item residual correction and evidence-weighted calibration go on top (the "hybrid"). This is v6, and the predictor is used everywhere: cards, the predicted-rating sort, and the For you floor and boost (the user's choice).
- **Fitting a new reader.** Book parameters are trained on the training readers by SGD (numba, `pipeline/train_rating_mf.py`, s11). A reader is fitted at request time by a ridge solve over their own ratings: λ_b = 2 on their offset, λ_p = 20 on their taste vector. Their own books get leave-one-out predictions, so a rating never predicts itself. Serving stays NumPy.
- **Size and speed.** The model file is 7 MB, and a whole-catalog prediction costs a few ms more than before.

**Context.** The rating track (D-050) showed the item-kNN predictor barely beating "book average + reader offset". The user asked to try SVD with biases and latent factors, SVD++, and anything else worth trying, each over a sensible range of hyperparameters, and to promote a clear winner.

**How it was chosen** (validation readers only; score = per-reader MAE averaged over history sizes 1–25 and all).
- **36 distinct configurations, 22 biased MF and 14 SVD++** (`eval/reports/rating_sweep_2026-10-04_1207.md`). Biased MF covered:
  - k ∈ {4, 8, 16, 32, 64, 128}, factor regularization 0.012–0.1, bias regularization 0–0.1, learning rate 0.003–0.01;
  - best epoch checked every 5;
  - fold-in penalties λ_b ∈ {0.5, 2, 5, 15} × λ_p ∈ {1, 5, 20, 50, 150, 500}.
- **Results:**
  - Biased MF reached 0.6808 (today's displayed rating: 0.6916).
  - Every factor count from 4 to 128 lands within 0.002, so most of the gain comes from the learned book and reader biases, and less bias regularization helped.
  - SVD++ (implicit factors from every book rated or read) peaked at 0.6831. It overfits within 5–15 epochs and is weakest for one-rating readers.
  - Under the item-item residual, MF reached 0.6788 calibrated (0.6779 uncalibrated); calibration costs about 0.001 but keeps predictions spread out (D-027), so it stays.
  - On validation the MF hybrid's NDCG@10 was 0.0864 vs 0.0870. Re-tuning the boost and floor (delta_pred × pred_floor_offset) moved it only within 0.0864–0.0868, so the ranking settings are unchanged.

**Test set** (5,895 readers, one run per finalist against v5, full history):

| Finalist | MAE | MAE vs v5 (95% CI) | NDCG@10 | NDCG@10 vs v5 (95% CI) |
|---|---|---|---|---|
| **MF hybrid** | 0.6643 | −0.0110 [−0.0124, −0.0095] | 0.0875 | +0.0001 [−0.0004, +0.0007] |
| Biased MF alone | 0.6688 | −0.0065 | 0.0869 | −0.0004 (not significant) |
| SVD++ hybrid | 0.6695 | −0.0058 | 0.0869 | −0.0004 (not significant) |

All three qualify under D-053. The MF hybrid gains the most.
- **The promotion run** (`eval/reports/2026-10-04_1505_shelf_life.md`, the production model rebuilt by s11, identical bit for bit) reproduced it exactly.
- **v6 against v5:**
  - MAE by history size: 0.728 vs 0.740 at 1 rating, 0.701 vs 0.716 at 3, 0.672 vs 0.685 at 10, 0.664 vs 0.675 at all.
  - Full history: RMSE 0.874 vs 0.879, per-reader Spearman 0.335 vs 0.314, 78.3% within one star vs 77.6%.
  - v6 is more accurate for 59% of readers.
  - Against book average + offset: −0.0134 [−0.0152, −0.0116].
  - Better for narrow, typical and wide raters alike (0.503 / 0.663 / 0.818 vs 0.521 / 0.672 / 0.826).

**Alternatives considered.** SVD++ (above). Biased MF alone (less accurate, and a slightly lower NDCG@10). A learned linear blend of the predictors was not tried: the hybrid already uses both signals and needs no extra fitting. Re-tuning the boost and floor for the new predictor (no measurable gain).

## D-053 · Promote a rating-model improvement that leaves the ranking at least as good
- **Date:** 2026-10-04
- **Category:** eval
- **Prompts:** fe7c091a-144
- **Commits:** 854dfe3

**Decision.** `promote_if_better` accepts either of two paths, both on full history against the champion:
- **Ranking path:** NDCG@10 is higher and per-reader MAE isn't significantly worse (the D-050 rule).
- **Rating path:** per-reader MAE is significantly better (95% bootstrap CI of the difference entirely below 0) and NDCG@10 isn't significantly worse (its CI not entirely below 0).

**Context.** D-050 made NDCG@10 decide promotions, with MAE as a guardrail. A change aimed at rating prediction could never qualify on its own, even with a clear MAE gain and an unchanged ranking. The user chose "MAE better, NDCG not worse" for a clear winner.

**Alternatives considered.** Requiring NDCG@10 to be at least the champion's (stricter; a pure rating change would pass or fail on ranking noise). Promoting on MAE alone (offered only if the new model were limited to the displayed rating; the user chose to use it everywhere).

## D-054 · Keep confidence intervals off the Evaluation chart
- **Date:** 2026-10-04
- **Category:** eval
- **Prompts:** fe7c091a-149, fe7c091a-150, fe7c091a-151, fe7c091a-152, fe7c091a-153, fe7c091a-154, fe7c091a-155

**Decision.** The Evaluation chart stays as it was: points with value labels, no confidence-interval whiskers or bands. Confidence intervals stay where they already are: the version panel's per-reader paired CI against the previous version, and the reports' head-to-head tables. The ranking intro also drops its sentence about the first two versions being re-scored. The page and the version settings already say which versions were re-scored.

**Context.** The user asked whether saved runs could put CIs on the charts without re-running evaluations. They could: a bootstrap over the per-reader results already in `eval/runs/` (1,000 resamples of the 5,895 test readers) gave intervals for every version and baseline, e.g. v6 NDCG@10 0.0875 [0.0841, 0.0913] and MAE 0.664 [0.657, 0.672]. Claude tried three designs on a branch, which was deleted unmerged:
- **Whiskers with the existing large points.** The user found the intervals too small to make out.
- **Small dots, with the trophy in the value label.**
- **Small dots without value labels.**

The user's verdict on the last two: "It's visually less clear and we have CIs presented elsewhere."

**Alternatives considered.** The three chart designs above. Unpaired per-version intervals would also have needed a caveat: v6's and v5's MAE intervals overlap ([0.657, 0.672] vs [0.668, 0.682]) although the per-reader paired test is clearly significant (−0.0110 [−0.0124, −0.0095]).

## D-055 · Learn from all ratings: switch to the interactions data (v7)
- **Date:** 2026-10-06
- **Category:** data
- **Prompts:** fe7c091a-160, fe7c091a-161, fe7c091a-162, fe7c091a-163, fe7c091a-164, fe7c091a-165, fe7c091a-166, fe7c091a-167
- **Commits:** 9436388, 09d8fa5, 134eaac, d4d8c41, 434fc73, 39d7772, bec5f67

**Decision.** Build every model from `goodreads_interactions_dedup.json.gz` (every rating readers gave, plus their read and to-read shelves) instead of `goodreads_reviews_dedup.json.gz` (only ratings that came with a written review).

| | Reviews data (before) | Interactions data (now) |
|---|---|---|
| Ratings | 15.1M | 104.0M |
| Readers | 465k | 876k |
| Readers' median history | 37 ratings | 199 ratings |

What changes with the switch:
- **To-read shelves.** 116.5M rows (51% of the file) are kept out of the reads. Before, a rating of 0 meant "read, unrated"; in this file most rating-0 rows are to-read shelves.
- **Catalog.** Every book in the earlier catalog stays, plus books with 75+ raters: 144,112 books (105,230 before), so no saved book disappears.
- **"Readers like you" pool.** A seeded random 150k of the ~676k eligible readers.
- **Re-tuned ranking settings** (validation readers only):
  - taste-model cap `a_max` 0.4 → 0.6;
  - `k_a` 20 → 10;
  - popularity penalty for heavy readers −0.5 → −0.8;
  - prediction boost 0.75 → 0.5;
  - recency half-life stays at 25.

**Context.** The user asked for a plan to swap the base data, covering:
- what changes;
- user-experience risks;
- hosting cost;
- build and workflow time;
- how to evaluate the swap fairly.

They chose the catalog and pool sizes at a decision point after profiling the file (`eval_interactions/profile.md`). They also asked for a guarantee that nothing on the live site changes during the experiment, which was met:
- the experiment ran in separate folders (`GOODREC_DATA`, `GOODREC_EVAL`, `GOODREC_CONFIG`, `make exp-*`);
- `make deploy` refuses experiment settings;
- all 471 production files were checksummed before and after every step.

**How it was evaluated.** The swap changes the training data, readers' histories and the hidden books all at once, so the old numbers can't be compared with new ones. Instead:
- **Bridge comparison.** The same 10,000 held-out readers (matched by Goodreads user id; neither build saw them) are scored on the new test split by:
  - v6 on its own reviews build (`ForeignShelfLife`: books translated by work id; books outside its catalog get the new data's book average, which can only flatter it);
  - v7 on the new build.
- **The earlier versions keep their numbers.** v1–v6 stay as the reviews-data era on the Evaluation page.

**Results** (6,921 test readers, full history unless noted; paired 95% CIs):
- **v6's settings on the new data:** better ratings everywhere, but full-history NDCG@10 was significantly worse (0.1930 vs 0.1955, −0.0025 [−0.0047, −0.0003]) and so was n=25 (−0.0037). Not promotable (D-053).
- **After re-tuning, on validation:** full-history NDCG@10 0.1852 → 0.1900 (+2.6%); n=25 0.1292 → 0.1319. Also tried, but worse: the 0.8 cap, half-lives of 12/50/100, a −1.1 penalty and a boost of 1.0.
- **v7 against v6 on the test:**
  - NDCG@10 0.1973 vs 0.1955 (+0.0018 [−0.0005, +0.0042]); significantly better at n=5 (+0.0033), not significantly different elsewhere.
  - MAE 0.659 vs 0.672 (−0.0128 [−0.0142, −0.0112]), significantly better at every history length.
  - Precision@10 0.1725 vs 0.1681; Recall@20 0.070 vs 0.066.
  - The top 20s cover 11.7% of the catalog (8.5% before) and are slightly less popular.
  - Promoted under D-053.

**Costs (measured).**
- **Serving:**
  - artifacts 426 MB (319 MB before);
  - peak memory 528 MB (474 MB);
  - startup 2.8 s (2.2 s);
  - uncached recommendations 65 ms vs 51 ms (demo) and 99 ms vs 86 ms (1,000 ratings) locally.

  It still fits the 1 GiB Cloud Run instance, so no hosting change.
- **Workflow:**
  - download 5 min, profile 7 min;
  - `make artifacts` 58 min (item-kNN 30 min);
  - a fresh test evaluation 75 min (42 min with cached baselines);
  - a full validation grid point ~6.5 min;
  - the 2023 SVD baseline is the slowest part.

  This Mac's 8 GB swaps during evaluations with 6 workers.
- **Other changes:**
  - only 41% of ratings have a read date (83% before), so recency more often uses the shelving date;
  - 83.7% of books have a Goodreads cover (90.7%); the rest fall back to Open Library or a placeholder.

**Alternatives considered.**
- **Catalog size:**
  - the same 20-rater rule (313k books, about 3× the serving cost);
  - a higher threshold alone (drops up to a quarter of today's books, and their saved ratings);
  - today's catalog only.
- **Reader pool:** everyone eligible, or the 150k most active (both past 1 GiB).
- **Evaluation:** re-scoring v1–v6 on the new data (4–8 h; not done) instead of the bridge.

## D-056 · The 2023 Book Recommender as the first point of each dataset's line
- **Date:** 2026-10-06
- **Category:** ui
- **Prompts:** fe7c091a-167, fe7c091a-168

**Decision.** On the Evaluation chart the 2023 Book Recommender is no longer a dashed baseline across the chart. It's the first point of the line, v0 ("2023 baseline"), styled differently from versions (a hollow amber diamond).

The chart now has one segment per dataset ("era"). Each era has its own test split, so scores are only compared within an era:
- **Trained on 15M ratings** ("only ratings that came with a written review"): v0 → v1 … v6.
- **Switched to use all 104M ratings** ("new data and test: scores restart"): v0 → v6 on the new test (the bridge, a hollow point) → v7.

Each era has a coloured header with a rule across its part of the plot.

The popularity and book-average baselines stay as dashed lines ("Baseline: Popular books"), drawn across their own era only. The panel's "change from" compares each point with the previous point on its line, with the paired per-reader CI from the report's head-to-head:
- v1 is compared with 2023;
- v7 is compared with the bridge.

**Context.** The user asked for the 2023 baseline to be "just another point in the line (styled differently to make it clear it's pre-v1)", in the same request as preparing the data switch. The switch needed the eras anyway.

## D-057 · One "From similar readers" list, sortable absolute or relative
- **Date:** 2026-10-07
- **Category:** ui
- **Prompts:** 81e316a2-001, 81e316a2-002, 81e316a2-003
- **Commits:** 4042e08, 98b1911, f8ac63b

**Decision.** The "Popular with readers like you" and "Top rated by readers like you" tabs are merged into one "From similar readers" tab. It has three sorts (Popularity, Similar readers' rating, Predicted rating), shown absolute by default, and a "Compared to all readers" checkbox at the right end of the sort row (after the List / Map view switch) that switches to the relative view. For the rating sorts, "all readers" means the Goodreads average. What each sort ranks by:
- **Popularity:** the similarity-weighted share of the 300 nearest readers who read the book; relative is the lift over the share of all readers who read it, with 5 pseudo-readers added to observed and expected counts.
- **Similar readers' rating:** the neighbors' shrunk average (at least max(5, M/100) of them rated it); relative is how far that average sits above the Goodreads average, shrunk toward 0.
- **Predicted rating:** the reader's calibrated predicted rating over books any neighbor read; relative subtracts the Goodreads average.

Absolute popularity is now plain reach. The old Popular score divided reach by the square root of the global read rate, a midpoint between the two ends that the switch now offers. Each card's note shows the number being sorted on. Old `tab=popular` / `tab=top-rated` links open the new tab with the matching sort. The Assistant's readers_like_you tool takes the same options.

**Context.** The user asked to condense the two tabs into a single page with Popularity, Rating (from similar users) and Predicted rating sorts, plus "an option to switch between relative and absolute values for all 3 of those", where relative means "books that similar readers rated higher than the Goodreads average or books that are more popular with similar readers than among all readers or books where the predicted rating is much higher than the Goodreads average."

**Alternatives considered.** An Absolute / Relative segmented switch (the first version; the user found it "a little confusing" and asked for absolute by default with a button for the relative view, with clearer wording); a per-sort button ("Compare to all readers" / "Compare to Goodreads average"; the user then chose the single checkbox wording). Keeping the square-root-damped score as "absolute" popularity (proposed by Claude, not chosen: the switch already covers both ends); unsmoothed lift, which puts books two or three neighbors read at the top.

**Why.** The pseudo-reader count (5) and the rating shrinkage (5, as before) were picked by Claude without a measurement; there's no offline evaluation of these lists. For you is unchanged.
