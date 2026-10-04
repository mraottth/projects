"""Markdown report for an evaluation run (goodrec.eval.run): shared header, Track 1 (recommendation quality),
Track 2 (rating prediction), then notes and descriptions. scripts/build_evaluations.py splits reports at the
"## Track" headings, so keep them."""

from __future__ import annotations

EXPOSURE = [
    "Each user has their own split date, so the production models can't be retrained to stop at it: they "
    "have seen other (training) readers' ratings dated after a user's split. This mildly helps similarity "
    "and popularity signals; popularity baselines benefit most.",
    "Goodreads-wide metadata (each book's ratings count and average rating) is a 2017 snapshot over all "
    "Goodreads users. The fame-gated boost, the item-mean prior and the 2023 baselines' filters use it.",
    "Catalog membership (at least 20 raters in the dataset) is counted over all users.",
]
LABEL = {"precision": "Precision", "recall": "Recall", "ndcg": "NDCG"}
COLS = [("ndcg", 10), ("precision", 10), ("recall", 10), ("ndcg", 20), ("precision", 20), ("recall", 20)]   # NDCG first, everywhere


def _n(n: int) -> str:
    return "all" if n < 0 else f"n={n}"


def _f(x, d=4) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def _lift(x) -> str:
    return "–" if x is None else f"{100 * x:+.0f}%"


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _bucket_title(n: int) -> str:
    return _n(n) + (" (full visible history)" if n < 0 else " (the most recent visible rating)" if n == 1
                    else f" (the {n} most recent visible ratings)")


def render(r: dict) -> str:
    """`r` is the run summary written to the JSON report (see run.summarize)."""
    m = r["model"]
    buckets = sorted(r["buckets"], key=lambda n: (n >= 0, -n))     # full history first, then 25, 10, 5, 3, 1
    lines = [
        f"# Eval {r['stamp']}: {m['name']}", "",
        "## Model", "",
        f"- **Model:** {m['name']} (`{m['key']}`). {m['description']}",
        f"- **Git commit:** `{r['commit']}`{' (uncommitted changes)' if r.get('dirty') else ''}",
        f"- **Parameters:** `{r['params_text']}`",
        f"- **Rating model:** `{r.get('rating_text', 'item_means=goodreads, calibration=evidence')}`"
        + (" (as the book cards show it)" if r.get("rating_text", "").endswith("calibration=evidence")
           and "goodreads" in r.get("rating_text", "") else " (reconstructed earlier version)"),
        f"- **Artifacts:** `{r['artifacts']}` (built {r.get('artifacts_built', '?')})",
        f"- **Split:** per-user temporal; each user's most recent {r['split']['holdout_frac']:.0%} of ratings hidden, "
        f"ordered by date read (date shelved when no read date is given), at the nearest date boundary; "
        f"relevant = hidden and rated ≥ {r['split']['relevant_min_rating']}★. "
        f"Hash `{r['split_hash']}`. Both tracks use the same users, visible histories and hidden books.",
        f"- **Users:** {r['n_users']:,} from the **{r['set']}** set"
        + (f" (subsample of {r['users_arg']:,})" if r.get("users_arg") else "") + ".",
        f"- **Runtime:** {r['runtime_s'] / 60:.1f} min.", "",
        "## Accepted exposure", "", *[f"- {e}" for e in EXPOSURE], "",
        "## Track 1 · Recommendation quality", "",
        "Does the model put the books a user went on to like (rated "
        f"≥ {r['split']['relevant_min_rating']}★) near the top of their recommendations? Higher is better.", "",
        "### Metrics", "",
        "Rows are sorted as model, previous best, baselines, ablations. Users per row in brackets when a "
        "baseline ran on a fixed subsample.", "",
    ]
    rows = r["rows"]
    for n in buckets:
        lines += [f"#### {_bucket_title(n)}", "",
                  "| | " + " | ".join(f"{LABEL[c[0]]}@{c[1]}" for c in COLS) + " |",
                  "|---|" + "---|" * len(COLS)]
        for row in rows:
            cell = row["by_n"][str(n)]
            label = f"**{row['name']}**" if row["kind"] == "model" else row["name"]
            if row.get("n_users") and row["n_users"] != r["n_users"]:
                label += f" [{row['n_users']:,}]"
            lines.append(f"| {label} | " + " | ".join(_f(cell[f'{a}@{k}']) for a, k in COLS) + " |")
        lines.append("")

    identical = False
    lines += ["### Head-to-head (NDCG@10, per user)", "",
              f"How **{m['name']}** compares with each baseline on the same users. Ties are common: many users "
              "score 0 under both. The CI is a bootstrap 95% interval on the mean per-user difference; lift is "
              "that difference relative to the baseline's mean.", ""]
    for n in buckets:
        lines += [f"#### {_n(n)}", "",
                  "| Comparison | users | wins | ties | losses | mean diff [95% CI] | lift | median gain when it wins |",
                  "|---|---|---|---|---|---|---|---|"]
        for h in r["head_to_head"]:
            c = h["by_n"][str(n)]
            if not c.get("n"):
                continue
            if c["tie"] >= 1 - 1e-9:      # the two models ranked every user's books identically
                identical = True
                lines.append(f"| vs. {h['name']} | {c['n']:,} | – | 100% | – | identical rankings† | – | – |")
                continue
            lines.append(f"| vs. {h['name']} | {c['n']:,} | {_pct(c['win'])} | {_pct(c['tie'])} | {_pct(c['loss'])} | "
                         f"{c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                         f"{_lift(c['lift'])} | {c['median_gain_when_win']:.4f} |")
        lines.append("")

    if identical:
        lines += ["† Identical rankings: both models ranked every user's books the same at that history size, so they "
                  "can't differ. For example, recency weighting needs rating dates, and the truncated histories carry "
                  "none, so it only changes full-history results.", ""]
    lines += ["### Diagnostics (not used to rank models)", "",
              "Catalog coverage: share of the catalog appearing in any user's top 20. Popularity: mean "
              "log(1 + training readers) of top-10 books (lower = less popularity bias).", "",
              "| | " + " | ".join(f"coverage {_n(n)}" for n in buckets) + " | " + " | ".join(f"popularity {_n(n)}" for n in buckets) + " |",
              "|---|" + "---|" * (2 * len(buckets))]
    for row in rows:
        lines.append(f"| {row['name']} | " + " | ".join(_f(row['by_n'][str(n)]['coverage']) for n in buckets) + " | "
                     + " | ".join(_f(row['by_n'][str(n)]['popularity'], 2) for n in buckets) + " |")
    if any("authors10" in row["by_n"][str(buckets[0])] for row in rows):
        lines += ["", "Author variety of each user's top 10: distinct authors, and the most books by a single author "
                  "(averages over users; the model doesn't aim for variety, the displayed Best match list does).", "",
                  "| | " + " | ".join(f"authors {_n(n)}" for n in buckets) + " | "
                  + " | ".join(f"one author {_n(n)}" for n in buckets) + " |",
                  "|---|" + "---|" * (2 * len(buckets))]
        for row in rows:
            lines.append(f"| {row['name']} | " + " | ".join(_f(row['by_n'][str(n)].get('authors10'), 2) for n in buckets) + " | "
                         + " | ".join(_f(row['by_n'][str(n)].get('top_author10'), 2) for n in buckets) + " |")
    if r.get("rating_rows"):
        lines += ["", *_rating_track(r, buckets)]
    if r.get("notes"):
        lines += ["", "## Notes", "", *[f"- {x}" for x in r["notes"]]]
    described = {}
    for row in rows + r.get("rating_rows", []):
        described.setdefault(row["name"], row["description"])
    lines += ["", "## Descriptions", "", *[f"- **{k}:** {v}" for k, v in described.items()]]
    return "\n".join(lines) + "\n"


RATING_HEAD = ["MAE", "RMSE", "MAE/σ", "Pearson", "Spearman", "±1★", "±0.5★", "Bias", "Spread"]


def _stars(x, d=2) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def _rating_track(r: dict, buckets: list[int]) -> list[str]:
    m = r["model"]
    rows = r["rating_rows"]
    st = r["rating_styles"]
    full = next(row for row in rows if row["key"] == "shelf_life")
    lines = [
        "## Track 2 · Rating prediction", "",
        "How close is the predicted rating to the one the user gave each hidden book (all of them, 1–5★)? Every "
        "prediction uses only the user's visible ratings and training-only item statistics. MAE is the primary "
        "metric; lower is better.", "",
        "### Metrics", "",
        "MAE and RMSE in stars, averaged per user and then over users (RMSE: the root of the mean per-user squared "
        "error). MAE/σ: each user's MAE divided by the spread of their visible ratings (σ_u, shrunk toward the "
        f"population's {st['pop_sd']:.2f}), so a one-star miss counts more for a reader who rates everything 4–5★. "
        "Pearson: correlation over all held-out ratings. Spearman: rank correlation within each user's hidden books, "
        "averaged over users with at least 3 hidden ratings that aren't all the same "
        f"({_pct(full['by_n']['-1']['spearman_users'])} of users; a method that predicts the same rating for every "
        "book scores 0). ±1★ / ±0.5★: share of predictions within one / half a star. Bias: mean prediction − "
        "actual. Spread: the SD of predictions as a share of the SD of actual ratings (below 100% = squeezed "
        "toward the middle). A method with no estimate for a book uses the book average; the share is in brackets.", "",
    ]
    for n in buckets:
        lines += [f"#### {_bucket_title(n)}", "", "| | " + " | ".join(RATING_HEAD) + " |", "|---|" + "---|" * len(RATING_HEAD)]
        for row in rows:
            c = row["by_n"][str(n)]
            label = f"**{row['name']}**" if row["kind"] == "model" else row["name"]
            if c.get("fallback"):
                label += f" (book average for {_pct(c['fallback'])})"
            spread = c["sd_pred"] / c["sd_actual"] if c.get("sd_actual") else None
            lines.append(f"| {label} | {_f(c['mae'])} | {_f(c['rmse'])} | {_f(c['nmae'], 3)} | {_f(c['pearson'], 3)} | "
                         f"{_f(c['spearman'], 3)} | {_pct(c['within1'])} | {_pct(c['within05'])} | {c['bias']:+.3f} | "
                         f"{_pct(spread)} |")
        lines.append("")

    lines += ["### Head-to-head (MAE, per user)", "",
              f"How **{m['name']}** compares with each baseline on per-user MAE. A win is a lower MAE. MAE diff is the "
              "model's MAE minus the baseline's (negative = more accurate), with a bootstrap 95% interval; relative is "
              "that difference over the baseline's MAE.", ""]
    for n in buckets:
        lines += [f"#### {_n(n)}", "",
                  "| Comparison | users | wins | ties | losses | MAE diff [95% CI] | relative | median gain when it wins |",
                  "|---|---|---|---|---|---|---|---|"]
        for h in r["rating_head_to_head"]:
            c = h["by_n"][str(n)]
            if not c.get("n"):
                continue
            lines.append(f"| vs. {h['name']} | {c['n']:,} | {_pct(c['win'])} | {_pct(c['tie'])} | {_pct(c['loss'])} | "
                         f"{c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                         f"{_lift(c['lift'])} | {c['median_gain_when_win']:.4f} |")
        lines.append("")

    g = st["groups"]
    lines += ["### By rating style (full history)", "",
              "Users grouped by the spread of their visible ratings (σ_u): narrow (below "
              f"{st['cuts'][0]:.2f}, {g[0]['users']:,} users), typical ({g[1]['users']:,}) and wide (above "
              f"{st['cuts'][1]:.2f}, {g[2]['users']:,}). The cut points are the validation users' terciles, so "
              "test ratings don't define the groups.", "",
              "| | " + " | ".join(f"{h} {s['style']}" for h in ("MAE", "MAE/σ", "Spearman", "Bias") for s in g) + " |",
              "|---|" + "---|" * (4 * len(g))]
    for row in rows:
        label = f"**{row['name']}**" if row["kind"] == "model" else row["name"]
        cells = [_f(s["mae"]) for s in row["by_style"]] + [_f(s["nmae"], 3) for s in row["by_style"]] \
            + [_f(s["spearman"], 3) for s in row["by_style"]] \
            + ["–" if s["bias"] is None else f"{s['bias']:+.3f}" for s in row["by_style"]]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    stars = range(1, 6)
    lines += ["", "### By actual rating (full history)", "",
              "Mean prediction − actual and MAE for the hidden books the user rated 1★ … 5★ (all ratings pooled). "
              "Most methods over-predict the books readers disliked and under-predict their favourites: predictions "
              "regress toward the middle.", "",
              "| | " + " | ".join(f"bias {s}★" for s in stars) + " | " + " | ".join(f"MAE {s}★" for s in stars) + " |",
              "|---|" + "---|" * 10]
    for row in rows:
        b = row["by_star"]
        label = f"**{row['name']}**" if row["kind"] == "model" else row["name"]
        lines.append(f"| {label} | " + " | ".join("–" if x is None else f"{x:+.2f}" for x in b["bias"]) + " | "
                     + " | ".join(_stars(x) for x in b["mae"]) + " |")
    cal = full["calibration"]
    lines += ["", f"### Calibration (full history, {m['name']})", "",
              "Predictions in half-star bins: how many fell in each, their mean, and the mean rating users actually "
              "gave. Well calibrated means the two means match.", "",
              "| Predicted | ratings | mean prediction | mean actual |", "|---|---|---|---|"]
    edges = cal["edges"]
    for i in range(len(edges) - 1):
        if cal["n"][i]:
            lines.append(f"| {edges[i]:.1f}–{edges[i + 1]:.1f}★ | {cal['n'][i]:,} | {_stars(cal['mean_pred'][i])} | "
                         f"{_stars(cal['mean_actual'][i])} |")
    return lines
