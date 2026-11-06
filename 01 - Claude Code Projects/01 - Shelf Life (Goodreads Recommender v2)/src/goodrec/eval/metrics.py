"""Ranking metrics with binary relevance, rating-prediction metrics, and paired per-user comparisons."""

from __future__ import annotations

import numpy as np

METRICS = ("precision", "recall", "ndcg")


def at_k(top: np.ndarray, relevant: set[int], k: int) -> tuple[float, float, float]:
    """(Precision@k, Recall@k, NDCG@k) for a ranked list. Precision = hits / k, Recall = hits / #relevant,
    NDCG's ideal ranking puts min(k, #relevant) hits first. A list shorter than k counts missing slots as misses."""
    if not relevant:
        return 0.0, 0.0, 0.0
    hits = np.array([i in relevant for i in top[:k]], dtype=float)
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    idcg = (1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum()
    return hits.sum() / k, hits.sum() / len(relevant), dcg / idcg


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0, lower_is_better: bool = False) -> dict:
    """Compare per-user scores of model `a` against baseline `b` (same users, NaNs dropped pairwise). The mean
    difference and its CI are always a - b; with lower_is_better (errors) a win is a lower score."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    if not n:
        return {"n": 0}
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    sign = -1.0 if lower_is_better else 1.0
    wins = sign * d > 1e-12
    losses = sign * d < -1e-12
    return {
        "n": n,
        "win": float(wins.mean()), "tie": float((~wins & ~losses).mean()), "loss": float(losses.mean()),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "lift": float(d.mean() / b.mean()) if b.mean() > 0 else None,
        "median_gain_when_win": float(np.median(sign * d[wins])) if wins.any() else 0.0,
    }


# ---- rating prediction --------------------------------------------------------------------------------------

# Per user (one value per user and history size; averaged over users, so each user counts equally):
RATING_COLS = ("mae", "mse", "within1", "within05", "bias", "nmae", "spearman")
# Pooled over every held-out rating (sums, so worker results add up): Pearson and spread, the calibration
# table (predictions in 0.5-star bins) and error by actual star.
CAL_EDGES = np.arange(1.0, 5.01, 0.5)                  # 8 bins: [1, 1.5), ..., [4.5, 5]
N_BINS = len(CAL_EDGES) - 1
POOL_LEN = 7 + 3 * N_BINS + 3 * 5                     # n, Σp, Σa, Σp², Σa², Σpa, fallbacks | bins | stars


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    xs = x[order]
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or xs[i] != xs[start]:
            ranks[order[start:i]] = (start + i - 1) / 2
            start = i
    return ranks


def spearman(pred: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation of one user's predictions with their actual ratings: does the model order this user's
    books right, whatever their scale? NaN when it can't be measured (fewer than 3 ratings, or all the same);
    0 when the predictions are all equal (no ordering information)."""
    if len(actual) < 3 or np.ptp(actual) == 0:
        return float("nan")
    if np.ptp(pred) < 1e-9:
        return 0.0
    rp, ra = _ranks(np.asarray(pred, float)), _ranks(np.asarray(actual, float))
    rp -= rp.mean()
    ra -= ra.mean()
    return float((rp @ ra) / np.sqrt((rp @ rp) * (ra @ ra)))


def rating_user(pred: np.ndarray, actual: np.ndarray, sigma: float) -> np.ndarray:
    """RATING_COLS for one user: MAE, MSE, share within 1 and 0.5 stars, mean signed error (prediction -
    actual), MAE / σ_u (σ_u from their visible ratings) and per-user Spearman."""
    e = np.asarray(pred, float) - np.asarray(actual, float)
    a = np.abs(e)
    return np.array([a.mean(), (e ** 2).mean(), (a <= 1 + 1e-9).mean(), (a <= 0.5 + 1e-9).mean(), e.mean(),
                     a.mean() / sigma, spearman(pred, actual)])


def rating_pool(pred: np.ndarray, actual: np.ndarray, fallbacks: int = 0) -> np.ndarray:
    p, a = np.asarray(pred, float), np.asarray(actual, float)
    out = np.zeros(POOL_LEN)
    out[:7] = [len(p), p.sum(), a.sum(), (p ** 2).sum(), (a ** 2).sum(), (p * a).sum(), fallbacks]
    b = np.clip(np.searchsorted(CAL_EDGES, p, side="right") - 1, 0, N_BINS - 1)
    for i, w in enumerate((np.ones_like(p), p, a)):
        out[7 + i * N_BINS: 7 + (i + 1) * N_BINS] = np.bincount(b, weights=w, minlength=N_BINS)
    s = np.clip(np.rint(a).astype(int), 1, 5) - 1
    base = 7 + 3 * N_BINS
    for i, w in enumerate((np.ones_like(p), p - a, np.abs(p - a))):
        out[base + i * 5: base + (i + 1) * 5] = np.bincount(s, weights=w, minlength=5)
    return out


def pooled_summary(v: np.ndarray) -> dict:
    """Pearson, spreads, fallback share, calibration table and by-star error from summed rating_pool vectors."""
    n, sp, sa, spp, saa, spa, fb = v[:7]
    if n == 0:
        return {"n": 0}
    var_p, var_a = spp / n - (sp / n) ** 2, saa / n - (sa / n) ** 2
    cov = spa / n - (sp / n) * (sa / n)
    bins = v[7: 7 + 3 * N_BINS].reshape(3, N_BINS)
    stars = v[7 + 3 * N_BINS:].reshape(3, 5)
    div = lambda x, c: [float(xx / cc) if cc else None for xx, cc in zip(x, c)]  # noqa: E731
    return {
        "n": int(n),
        "pearson": float(cov / np.sqrt(var_p * var_a)) if var_p > 1e-12 and var_a > 1e-12 else 0.0,
        "sd_pred": float(np.sqrt(max(var_p, 0))), "sd_actual": float(np.sqrt(max(var_a, 0))),
        "fallback": float(fb / n),
        "calibration": {"edges": CAL_EDGES.tolist(), "n": bins[0].astype(int).tolist(),
                        "mean_pred": div(bins[1], bins[0]), "mean_actual": div(bins[2], bins[0])},
        "by_star": {"n": stars[0].astype(int).tolist(), "bias": div(stars[1], stars[0]), "mae": div(stars[2], stars[0])},
    }
