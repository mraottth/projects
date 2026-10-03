"""Ranking metrics with binary relevance, and paired per-user comparisons between two models."""

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


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Compare per-user scores of model `a` against baseline `b` (same users, NaNs dropped pairwise)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    if not n:
        return {"n": 0}
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    wins = d > 1e-12
    losses = d < -1e-12
    return {
        "n": n,
        "win": float(wins.mean()), "tie": float((~wins & ~losses).mean()), "loss": float(losses.mean()),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "lift": float(d.mean() / b.mean()) if b.mean() > 0 else None,
        "median_gain_when_win": float(np.median(d[wins])) if wins.any() else 0.0,
    }
