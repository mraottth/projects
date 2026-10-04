"""Train matrix-factorization rating predictors on the training readers (R_train, plus Read_train for SVD++).

Stochastic gradient descent in numba (training only; serving folds readers in with NumPy, core.rating_mf).
Biased MF visits ratings in a fresh random order each epoch. SVD++ visits readers in random order and,
for each, updates the implicit factors y_j of every book they rated or read once, after their ratings
(the usual batched form of the per-rating update). An optional `checkpoint(model, epoch)` callback runs
every `check_every` epochs; it returns a validation score (lower is better), and the best snapshot is kept.
Used by goodrec.eval.tune_rating (the sweep) and goodrec.pipeline.s11_rating_mf (production).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit
from scipy import sparse

from goodrec.config import INTERIM_DIR
from goodrec.core.rating_mf import MFModel


@dataclass(frozen=True)
class TrainConfig:
    kind: str = "mf"             # "mf" | "svdpp"
    k: int = 64
    lr: float = 0.005
    reg: float = 0.05            # factors
    reg_b: float | None = None   # biases (None = reg)
    reg_y: float | None = None   # SVD++ implicit factors (None = reg)
    epochs: int = 60
    init_std: float = 0.1
    seed: int = 0

    def label(self) -> str:
        s = f"{self.kind} k={self.k} lr={self.lr} reg={self.reg}"
        if self.reg_b is not None:
            s += f" reg_b={self.reg_b}"
        if self.reg_y is not None:
            s += f" reg_y={self.reg_y}"
        return s


@njit(cache=True, fastmath=True)
def _epoch_mf(order, u, i, r, mu, bu, bi, P, Q, lr, reg, reg_b):
    k = P.shape[1]
    sse = 0.0
    for t in order:
        uu, ii = u[t], i[t]
        pred = mu + bu[uu] + bi[ii]
        for f in range(k):
            pred += P[uu, f] * Q[ii, f]
        e = r[t] - pred
        sse += e * e
        bu[uu] += lr * (e - reg_b * bu[uu])
        bi[ii] += lr * (e - reg_b * bi[ii])
        for f in range(k):
            pf, qf = P[uu, f], Q[ii, f]
            P[uu, f] += lr * (e * qf - reg * pf)
            Q[ii, f] += lr * (e * pf - reg * qf)
    return sse


@njit(cache=True, fastmath=True)
def _epoch_svdpp(users, indptr, idx, val, nptr, nidx, mu, bu, bi, P, Q, Y, lr, reg, reg_b, reg_y, seed):
    np.random.seed(seed)
    k = P.shape[1]
    sse = 0.0
    z = np.zeros(k)
    gz = np.zeros(k)
    for uu in users:
        s, e_ = indptr[uu], indptr[uu + 1]
        if s == e_:
            continue
        ns, ne = nptr[uu], nptr[uu + 1]
        norm = 1.0 / np.sqrt(max(ne - ns, 1))
        z[:] = 0.0
        for t in range(ns, ne):
            for f in range(k):
                z[f] += Y[nidx[t], f]
        for f in range(k):
            z[f] *= norm
        gz[:] = 0.0
        for t in s + np.random.permutation(e_ - s):
            ii = idx[t]
            pred = mu + bu[uu] + bi[ii]
            for f in range(k):
                pred += Q[ii, f] * (P[uu, f] + z[f])
            e = val[t] - pred
            sse += e * e
            bu[uu] += lr * (e - reg_b * bu[uu])
            bi[ii] += lr * (e - reg_b * bi[ii])
            for f in range(k):
                pf, qf = P[uu, f], Q[ii, f]
                P[uu, f] += lr * (e * qf - reg * pf)
                Q[ii, f] += lr * (e * (pf + z[f]) - reg * qf)
                gz[f] += e * qf
        for t in range(ns, ne):
            j = nidx[t]
            for f in range(k):
                Y[j, f] += lr * (gz[f] * norm - reg_y * Y[j, f])
    return sse


def load_training(R: sparse.csr_matrix | None = None, Read: sparse.csr_matrix | None = None):
    R = sparse.csr_matrix(R if R is not None else sparse.load_npz(INTERIM_DIR / "R_train.npz"))
    R.sort_indices()
    if Read is None:
        p = INTERIM_DIR / "Read_train.npz"
        Read = sparse.load_npz(p) if p.exists() else sparse.csr_matrix(R.shape, dtype=np.int8)
    N = ((R > 0).astype(np.int8) + (sparse.csr_matrix(Read) > 0).astype(np.int8)).tocsr()
    N.data[:] = 1
    N.sort_indices()
    return R, N


def train(cfg: TrainConfig, R: sparse.csr_matrix | None = None, Read: sparse.csr_matrix | None = None,
          checkpoint: Callable[[MFModel, int], float] | None = None, check_every: int = 5,
          verbose: bool = True) -> tuple[MFModel, list[tuple[int, float, float]]]:
    """Train; returns (best model, curve of (epoch, train RMSE, validation score or nan))."""
    R, N = load_training(R, Read)
    n_users, n_items = R.shape
    rng = np.random.default_rng(cfg.seed)
    vals = R.data.astype(np.float64)
    mu = float(vals.mean())
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    P = rng.normal(0, cfg.init_std, (n_users, cfg.k))
    Q = rng.normal(0, cfg.init_std, (n_items, cfg.k))
    Y = rng.normal(0, cfg.init_std, (n_items, cfg.k)) if cfg.kind == "svdpp" else None
    reg_b = cfg.reg if cfg.reg_b is None else cfg.reg_b
    reg_y = cfg.reg if cfg.reg_y is None else cfg.reg_y
    u = np.repeat(np.arange(n_users, dtype=np.int32), np.diff(R.indptr))
    i = R.indices.astype(np.int32)

    def snapshot() -> MFModel:
        return MFModel(kind=cfg.kind, mu=mu, bi=bi.astype(np.float32), Q=Q.astype(np.float32),
                       Y=Y.astype(np.float32) if Y is not None else None, name=cfg.label())

    curve, best, best_score = [], None, np.inf
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        if cfg.kind == "mf":
            sse = _epoch_mf(rng.permutation(len(vals)), u, i, vals, mu, bu, bi, P, Q, cfg.lr, cfg.reg, reg_b)
        else:
            sse = _epoch_svdpp(rng.permutation(n_users), R.indptr, R.indices, vals, N.indptr, N.indices,
                               mu, bu, bi, P, Q, Y, cfg.lr, cfg.reg, reg_b, reg_y, int(rng.integers(1 << 30)))
        score = np.nan
        if checkpoint is not None and (ep % check_every == 0 or ep == cfg.epochs):
            m = snapshot()
            score = float(checkpoint(m, ep))
            if score < best_score:
                best, best_score = m, score
            elif ep >= 2 * check_every and len(curve) and score > best_score + 0.002:
                curve.append((ep, float(np.sqrt(sse / len(vals))), score))
                if verbose:
                    print(f"    {cfg.label()} epoch {ep}: train RMSE {curve[-1][1]:.4f}, val {score:.4f} "
                          f"(stopping; best {best_score:.4f})", flush=True)
                break
        curve.append((ep, float(np.sqrt(sse / len(vals))), score))
        if verbose and (not np.isnan(score) or ep == 1):
            print(f"    {cfg.label()} epoch {ep}: train RMSE {curve[-1][1]:.4f}"
                  + (f", val {score:.4f}" if not np.isnan(score) else "") + f" ({time.time() - t0:.0f}s)", flush=True)
    return (best if best is not None else snapshot()), curve
