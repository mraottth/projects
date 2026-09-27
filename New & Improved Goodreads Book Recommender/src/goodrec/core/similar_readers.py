"""'Readers like you': nearest training users in ALS user-embedding space, then their actual shelves.

popular:   similarity-weighted share of neighbors who read the book, damped by global popularity
           (reach / reader_rate^damping) so universally-read books don't dominate.
top_rated: Bayesian neighbor average rating, requiring >= max(5, M/100) neighbor raters
           (the old app's max(N/300, 5) idea).
"""

import numpy as np

from goodrec.config import load_config
from goodrec.core.artifacts import Artifacts

READ_UNRATED = 6


def neighbors(art: Artifacts, u: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    un = u / max(np.linalg.norm(u), 1e-8)
    sims = art.user_factors @ un
    top = np.argpartition(-sims, m)[:m]
    top = top[np.argsort(-sims[top])]
    return top, np.maximum(sims[top], 0)


def similar_readers(art: Artifacts, u: np.ndarray | None, mask: np.ndarray, limit: int = 40) -> dict | None:
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
    reach = np.asarray(read.T @ w).ravel() / wsum                       # weighted share who read it
    pct_read = np.asarray(read.sum(axis=0)).ravel() / m                  # unweighted, for display
    n_rated = np.asarray(rated_bin.sum(axis=0)).ravel()
    w_rated = np.asarray(rated_bin.T @ w).ravel()
    w_sum_r = np.asarray(rated.T @ w).ravel()
    mu = art.meta.bayes
    nbr_avg = (w_sum_r + 5 * mu) / (w_rated + 5)                         # weighted, shrunk to item mean
    raw_avg = np.divide(np.asarray(rated.sum(axis=0)).ravel(), n_rated,
                        out=np.zeros_like(n_rated, dtype=np.float64), where=n_rated > 0)

    pop_score = reach / np.power(np.maximum(art.meta.reader_rate, 1e-6), cfg["pop_damping"])
    pop_ok = mask & (pct_read > 0)
    pop_idx = np.flatnonzero(pop_ok)
    pop_idx = pop_idx[np.argsort(-pop_score[pop_idx])][:limit]

    min_raters = max(5, m // 100)
    tr_ok = mask & (n_rated >= min_raters)
    tr_idx = np.flatnonzero(tr_ok)
    tr_idx = tr_idx[np.argsort(-nbr_avg[tr_idx])][:limit]

    return {
        "n_neighbors": int(m),
        "popular": [{"idx": int(i), "pct_read": round(float(pct_read[i]) * 100, 1)} for i in pop_idx],
        "top_rated": [{"idx": int(i), "neighbor_avg": round(float(raw_avg[i]), 2), "n_raters": int(n_rated[i])}
                      for i in tr_idx],
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
