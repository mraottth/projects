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
