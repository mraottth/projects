"""Markdown report for an evaluation run (goodrec.eval.run)."""

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
COLS = [("precision", 10), ("recall", 10), ("ndcg", 10), ("precision", 20), ("recall", 20), ("ndcg", 20)]


def _n(n: int) -> str:
    return "all" if n < 0 else f"n={n}"


def _f(x, d=4) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def _lift(x) -> str:
    return "–" if x is None else f"{100 * x:+.0f}%"


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def render(r: dict) -> str:
    """`r` is the run summary written to the JSON report (see run.summarize)."""
    m = r["model"]
    buckets = r["buckets"]
    lines = [
        f"# Eval {r['stamp']}: {m['name']}", "",
        "## Model", "",
        f"- **Model:** {m['name']} (`{m['key']}`). {m['description']}",
        f"- **Git commit:** `{r['commit']}`{' (uncommitted changes)' if r.get('dirty') else ''}",
        f"- **Parameters:** `{r['params_text']}`",
        f"- **Artifacts:** `{r['artifacts']}` (built {r.get('artifacts_built', '?')})",
        f"- **Split:** per-user temporal; each user's most recent {r['split']['holdout_frac']:.0%} of ratings hidden, "
        f"ordered by date read (date shelved when no read date is given), at the nearest date boundary; "
        f"relevant = hidden and rated ≥ {r['split']['relevant_min_rating']}★. "
        f"Hash `{r['split_hash']}`.",
        f"- **Users:** {r['n_users']:,} from the **{r['set']}** set"
        + (f" (subsample of {r['users_arg']:,})" if r.get("users_arg") else "") + ".",
        f"- **Runtime:** {r['runtime_s'] / 60:.1f} min.", "",
        "## Accepted exposure", "", *[f"- {e}" for e in EXPOSURE], "",
        "## Metrics", "",
        "Rows are sorted as model, previous best, baselines, ablations. Users per row in brackets when a "
        "baseline ran on a fixed subsample.", "",
    ]
    rows = r["rows"]
    for n in buckets:
        lines += [f"### {_n(n)}" + (" (full visible history)" if n < 0 else " (the most recent visible rating)" if n == 1
                                         else f" (the {n} most recent visible ratings)"), "",
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
    lines += ["## Head-to-head (NDCG@10, per user)", "",
              f"How **{m['name']}** compares with each baseline on the same users. Ties are common: many users "
              "score 0 under both. The CI is a bootstrap 95% interval on the mean per-user difference; lift is "
              "that difference relative to the baseline's mean.", ""]
    for n in buckets:
        lines += [f"### {_n(n)}", "",
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
    lines += ["## Diagnostics (not used to rank models)", "",
              "Catalog coverage: share of the catalog appearing in any user's top 20. Popularity: mean "
              "log(1 + training readers) of top-10 books (lower = less popularity bias).", "",
              "| | " + " | ".join(f"coverage {_n(n)}" for n in buckets) + " | " + " | ".join(f"popularity {_n(n)}" for n in buckets) + " |",
              "|---|" + "---|" * (2 * len(buckets))]
    for row in rows:
        lines.append(f"| {row['name']} | " + " | ".join(_f(row['by_n'][str(n)]['coverage']) for n in buckets) + " | "
                     + " | ".join(_f(row['by_n'][str(n)]['popularity'], 2) for n in buckets) + " |")
    if r.get("rating_rmse"):
        lines += ["", "Predicted-rating error on all hidden ratings (RMSE, stars; lower is better):", "",
                  "| | " + " | ".join(_n(n) for n in buckets) + " |", "|---|" + "---|" * len(buckets)]
        for name, by_n in r["rating_rmse"].items():
            lines.append(f"| {name} | " + " | ".join(_f(by_n[str(n)]) for n in buckets) + " |")
    if r.get("notes"):
        lines += ["", "## Notes", "", *[f"- {x}" for x in r["notes"]]]
    lines += ["", "## Descriptions", "", *[f"- **{row['name']}:** {row['description']}" for row in rows]]
    return "\n".join(lines) + "\n"
