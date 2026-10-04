"""Matrix-factorization rating predictors (biased MF and SVD++) for readers the model never trained on.

Item parameters are learned offline from the training readers (goodrec.pipeline.train_rating_mf). A reader
is *folded in* at request time from their own ratings, with the item parameters fixed: a small ridge
regression for their offset b_u and taste vector p_u (DECISIONS D-052).

    biased MF:  r̂ = μ + b_u + b_i + q_i · p_u
    SVD++:      r̂ = μ + b_u + b_i + q_i · (p_u + |N(u)|^-½ Σ_{j∈N(u)} y_j),  N(u) = books rated or read

Fold-in solves  min Σ_i (r_i − μ − b_i − q_i·z − b_u − q_i·p_u)² + λ_b b_u² + λ_p |p_u|²  (z = the SVD++
implicit term, 0 for MF). A rated book's own prediction is leave-one-out (the ridge hat-matrix identity), so,
as with the item-kNN predictor, a rating never predicts itself; calibration relies on that.
NumPy only: serving doesn't need the training dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MFModel:
    kind: str                    # "mf" | "svdpp"
    mu: float
    bi: np.ndarray               # (N,) float32 item biases
    Q: np.ndarray                # (N, k) float32 item factors
    Y: np.ndarray | None = None  # (N, k) float32 implicit factors (SVD++)
    lam_b: float = 5.0           # fold-in ridge penalty on the reader's offset
    lam_p: float = 20.0          # ... and on their taste vector
    name: str = ""

    @property
    def k(self) -> int:
        return self.Q.shape[1]

    def implicit(self, ratings: dict, read: set | None = None) -> np.ndarray:
        """SVD++'s |N(u)|^-½ Σ y_j over books rated or read (zeros for biased MF)."""
        if self.Y is None:
            return np.zeros(self.k, np.float64)
        n = np.fromiter(set(ratings) | set(read or ()), dtype=np.int64)
        if not len(n):
            return np.zeros(self.k, np.float64)
        return self.Y[n].sum(axis=0, dtype=np.float64) / np.sqrt(len(n))

    def fold_in(self, ratings: dict, read: set | None = None):
        """(b_u, p_u, z, ri, rv, X, A_inv) for a reader; ratings: work_idx -> stars."""
        z = self.implicit(ratings, read)
        if not ratings:
            return 0.0, np.zeros(self.k), z, None, None, None, None
        ri = np.fromiter(ratings, dtype=np.int64)
        rv = np.fromiter(ratings.values(), dtype=np.float64)
        Qr = self.Q[ri].astype(np.float64)
        y = rv - self.mu - self.bi[ri] - Qr @ z
        X = np.hstack([np.ones((len(ri), 1)), Qr])
        A = X.T @ X
        A[np.diag_indices_from(A)] += np.r_[self.lam_b, np.full(self.k, self.lam_p)]
        A_inv = np.linalg.inv(A)
        w = A_inv @ (X.T @ y)
        return float(w[0]), w[1:], z, ri, rv, X, A_inv

    def predict(self, ratings: dict, items, read: set | None = None, clip: bool = True, loo: bool = True,
                fit=None) -> np.ndarray:
        """Predicted ratings for `items`. The reader's own rated books get leave-one-out predictions unless
        loo=False (the in-sample fit, used as the hybrid's baseline). `fit` reuses a fold_in() result."""
        items = np.asarray(items, dtype=np.int64)
        b_u, p_u, z, ri, rv, X, A_inv = fit if fit is not None else self.fold_in(ratings, read)
        pred = self.mu + self.bi[items] + b_u + self.Q[items].astype(np.float64) @ (p_u + z)
        if loo and ri is not None:                            # leave-one-out for the reader's own books
            order = np.argsort(ri)                            # position of each item among the rated books
            at = np.minimum(np.searchsorted(ri[order], items), len(ri) - 1)
            m = ri[order][at] == items
            own = np.full(len(items), -1)
            own[m] = order[at[m]]
            if m.any():
                Xo = X[own[m]]
                h = np.einsum("ij,jk,ik->i", Xo, A_inv, Xo)
                fit = pred[m]
                pred[m] = rv[own[m]] - (rv[own[m]] - fit) / np.maximum(1 - h, 1e-6)
        return np.clip(pred, 1, 5) if clip else pred

    # ---- storage -----------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        extra = {"Y": self.Y} if self.Y is not None else {}
        np.savez(path, kind=self.kind, mu=self.mu, bi=self.bi.astype(np.float32), Q=self.Q.astype(np.float32),
                 lam_b=self.lam_b, lam_p=self.lam_p, name=self.name, **extra)

    @classmethod
    def load(cls, path: Path) -> "MFModel":
        z = np.load(path, allow_pickle=False)
        return cls(kind=str(z["kind"]), mu=float(z["mu"]), bi=z["bi"], Q=z["Q"], Y=z["Y"] if "Y" in z.files else None,
                   lam_b=float(z["lam_b"]), lam_p=float(z["lam_p"]), name=str(z["name"]))
