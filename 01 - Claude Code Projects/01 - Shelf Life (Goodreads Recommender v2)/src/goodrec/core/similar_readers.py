"""'Readers like you': nearest training users in ALS user-embedding space, then their actual shelves.

Per-book scores behind the "From similar readers" list (each absolute, then relative):
popularity: similarity-weighted share of neighbors who read the book (reach); relative = lift, how many
            times more of them read it than readers overall would predict, with POP_PRIOR pseudo-readers
            added to both sides so a book two neighbors read doesn't top the list.
rating:     Bayesian neighbor average rating, requiring >= max(5, M/100) neighbor raters (the old app's
            max(N/300, 5) idea); relative = how far that average sits above the Goodreads average,
            shrunk toward it (0) with the same prior weight.
The predicted-rating sort lives in the API (it needs the user's calibrated predictions).
"""

import numpy as np

from goodrec.config import load_config
from goodrec.core.artifacts import Artifacts

READ_UNRATED = 6
RATING_PRIOR = 5     # neighbor-weight pseudo-ratings at the prior (item mean, or Goodreads avg for relative)
POP_PRIOR = 5        # neighbor-weight pseudo-readers added to observed and expected readers (relative popularity)


def neighbors(art: Artifacts, u: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    un = u / max(np.linalg.norm(u), 1e-8)
    sims = art.user_factors @ un
    top = np.argpartition(-sims, m)[:m]
    top = top[np.argsort(-sims[top])]
    return top, np.maximum(sims[top], 0)


def similar_readers(art: Artifacts, u: np.ndarray | None) -> dict | None:
    cfg = load_config()["similar_readers"]
    if u is None or art.user_factors is None or art.readers is None:
        return None
    m = min(cfg["n_neighbors"], len(art.user_factors) - 1)
    rows, w = neighbors(art, u, m)
    sub = art.readers[rows]                          # m x N int8
    read = sub.copy()
    read.data = np.ones_like(read.data, dtype=np.float32)
    rated = sub.copy().astype(np.float32)
    rated.data[rated.data == READ_UNRATED] = 0
    rated.eliminate_zeros()
    rated_bin = rated.copy()
    rated_bin.data[:] = 1

    wsum = max(float(w.sum()), 1e-8)
    w_read = np.asarray(read.T @ w).ravel()
    reach = w_read / wsum                                                # weighted share who read it
    pct_read = np.asarray(read.sum(axis=0)).ravel() / m                  # unweighted, for display
    rate = art.meta.reader_rate.astype(np.float64)                       # share of all readers who read it
    lift = (w_read + POP_PRIOR) / (rate * wsum + POP_PRIOR)              # observed / expected readers, smoothed
    n_rated = np.asarray(rated_bin.sum(axis=0)).ravel()
    w_rated = np.asarray(rated_bin.T @ w).ravel()
    w_sum_r = np.asarray(rated.T @ w).ravel()
    nbr_avg = (w_sum_r + RATING_PRIOR * art.meta.bayes) / (w_rated + RATING_PRIOR)   # weighted, shrunk to item mean
    gr = art.meta.avg_rating.astype(np.float64)
    nbr_vs_gr = (w_sum_r - w_rated * gr) / (w_rated + RATING_PRIOR)      # shrunk toward "same as Goodreads"
    raw_avg = np.divide(np.asarray(rated.sum(axis=0)).ravel(), n_rated,
                        out=np.zeros_like(n_rated, dtype=np.float64), where=n_rated > 0)
    min_raters = max(5, m // 100)

    return {
        "n_neighbors": int(m),
        # Per-book arrays (all N books): scores for ranking the "From similar readers" list, and the average
        # rating among these neighbors shown on every card.
        "score": {("popularity", False): reach.astype(np.float32), ("popularity", True): lift.astype(np.float32),
                  ("rating", False): nbr_avg.astype(np.float32), ("rating", True): nbr_vs_gr.astype(np.float32)},
        "pct_read_all": pct_read.astype(np.float32),
        "min_raters": int(min_raters),
        "item_avg": raw_avg.astype(np.float32),
        "item_n": n_rated.astype(np.int32),
        "genre_share": _genre_share(art, read, w),
    }


def _genre_share(art: Artifacts, read, w: np.ndarray) -> np.ndarray:
    """Share of neighbors' (weighted) reads in each parent genre."""
    g = art.meta.parent_genre
    per_item = np.asarray(read.T @ w).ravel()
    ok = g >= 0
    share = np.bincount(g[ok], weights=per_item[ok], minlength=len(art.meta.genre_names))
    return share / max(share.sum(), 1e-8)


def user_genre_share(art: Artifacts, ratings: dict[int, int]) -> np.ndarray:
    """Share of the user's liked (>= 4 stars, or all if none) books in each parent genre."""
    liked = [i for i, r in ratings.items() if r >= 4] or list(ratings)
    g = art.meta.parent_genre[np.asarray(liked, dtype=np.int64)] if liked else np.zeros(0, np.int16)
    g = g[g >= 0]
    share = np.bincount(g, minlength=len(art.meta.genre_names)).astype(np.float64)
    return share / max(share.sum(), 1e-8)
