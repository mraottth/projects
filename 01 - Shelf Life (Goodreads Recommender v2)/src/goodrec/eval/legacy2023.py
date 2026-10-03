"""The 2023 Goodreads recommender (../03 - Goodreads Recommender (2023 original)), re-implemented on this
project's training data as evaluation baselines.

All three methods start from the same step as the original: cosine nearest neighbours among readers, on
L2-normalised rating rows (`NearestNeighbors(metric="cosine")` on `Normalizer()` output), with the
target reader's row appended to the neighbourhood matrix. Parameters and filters are the original's
(02_book_recommender.ipynb, Web App/recommender_classes.py):

- Similar readers (notebook Part 1): the 150 nearest readers; books ranked by how many of them rated
  the book, then by average rating; keep Goodreads ratings_count > 100 and average > 3.75.
- SVD (notebook 2.1-2.2 and the 2023 web app): the 3,000 nearest readers; `svds(k=42)` of their rating
  matrix; rank by the target's reconstructed row; keep average > 3.5.
- Gradient-descent matrix factorization (notebook 2.3): the 1,000 nearest readers; k=45, lr 0.02 divided
  by 1.5 at step 1 and every 140 steps (floor 2e-5), beta=1, 700 steps, uniform random init. The
  original updates P and Q with fancy-indexed assignment (`P[i,k] = P[i,k] + ...`) over all observed
  ratings at once; NumPy keeps only the last write for a repeated index, so each step updates each
  reader from their last rating (in column order) and each book from its last reader (in row order).
  This replicates that exactly, computing only the entries that take effect.

All three drop the reader's own books and later volumes in a series. Differences from the original:
work-level ids (editions collapsed) instead of edition ids; this project's catalog and training users;
the children's-book filter uses this catalog's children's and comics flags instead of the 2023 LDA
genres; column order within a neighbourhood is by work id (it decides which rating is "last" in GD).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from goodrec.config import INTERIM_DIR
from goodrec.core.artifacts import Artifacts
from goodrec.core.scoring import UserInput
from goodrec.eval.models import Recommender


class Neighbourhood:
    """Row-normalised training ratings and a cosine nearest-reader search."""

    def __init__(self, R: sparse.csr_matrix | None = None):
        R = R if R is not None else sparse.load_npz(INTERIM_DIR / "R_train.npz")
        R = sparse.csr_matrix(R, dtype=np.float32)
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel())
        self.Rn = sparse.diags(1 / np.maximum(norms, 1e-12)).astype(np.float32) @ R
        self.Rn = self.Rn.tocsr()
        self.Rn.sort_indices()
        self.Rc = self.Rn.tocsc()

    def target_row(self, user: UserInput, n_items: int) -> sparse.csr_matrix:
        items = np.fromiter(user.ratings, dtype=np.int64)
        vals = np.fromiter(user.ratings.values(), dtype=np.float32)
        vals /= max(float(np.linalg.norm(vals)), 1e-12)
        return sparse.csr_matrix((vals, (np.zeros(len(items), int), items)), shape=(1, n_items))

    def nearest(self, user: UserInput, n: int) -> np.ndarray:
        """The n readers with the highest cosine similarity to `user`, most similar first."""
        t = self.target_row(user, self.Rn.shape[1])
        sims = np.asarray((self.Rc[:, t.indices] @ t.data)).ravel()
        n = min(n, len(sims))
        top = np.argpartition(-sims, n - 1)[:n]
        return top[np.argsort(-sims[top], kind="stable")]

    def matrix(self, user: UserInput, nbrs: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Neighbour rows plus the target as the last row, restricted to books any of them rated.
        Returns (matrix, column work_idx)."""
        t = self.target_row(user, self.Rn.shape[1])
        sub = sparse.vstack([self.Rn[nbrs], t]).tocsr()
        cols = np.unique(sub.indices)
        M = sub[:, cols].tocsr()
        M.sort_indices()
        return M, cols


def _keep(art: Artifacts) -> np.ndarray:
    """Books the 2023 recommender could show at all: not a later series volume, not children's or comics."""
    m = art.meta
    later = (m.series_id >= 0) & (m.series_pos > 1)
    return ~later & ~m.is_children & ~m.is_comic


def _rank(scores: np.ndarray, cols: np.ndarray, ok: np.ndarray, user: UserInput, k: int,
          tiebreak: np.ndarray | None = None) -> np.ndarray:
    good = ok[cols]
    if user.seen:
        good &= ~np.isin(cols, np.fromiter(user.seen, dtype=np.int64))
    c, s = cols[good], scores[good]
    order = np.lexsort((-tiebreak[c], -s)) if tiebreak is not None else np.argsort(-s, kind="stable")
    return c[order][:k]


@dataclass
class SimilarReaders2023(Recommender):
    art: Artifacts = None
    nb: Neighbourhood = field(default=None, repr=False)
    n_neighbours: int = 150

    def recommend(self, user, k, rng):
        m = self.art.meta
        nbrs = self.nb.nearest(user, self.n_neighbours)
        sub = self.nb.Rn[nbrs]
        counts = np.bincount(sub.indices, minlength=m.n).astype(np.float64)
        cols = np.flatnonzero(counts)
        ok = _keep(self.art) & (m.ratings_count > 100) & (m.avg_rating > 3.75)
        return _rank(counts[cols], cols, ok, user, k, tiebreak=m.avg_rating)


@dataclass
class SVD2023(Recommender):
    art: Artifacts = None
    nb: Neighbourhood = field(default=None, repr=False)
    n_neighbours: int = 3000
    factors: int = 42

    def recommend(self, user, k, rng):
        M, cols = self.nb.matrix(user, self.nb.nearest(user, self.n_neighbours))
        kk = min(self.factors, min(M.shape) - 1)
        U, s, Vt = svds(M.astype(np.float64), k=kk, random_state=0)
        pred = (U[-1] * s) @ Vt
        ok = _keep(self.art) & (self.art.meta.avg_rating > 3.5)
        return _rank(pred, cols, ok, user, k)


def gd_factorize(M: sparse.csr_matrix, k: int = 45, steps: int = 700, lr: float = 0.02, beta: float = 1.0,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The 2023 notebook's matrix_factorization(), with identical arithmetic but computed only for the
    observed entries whose updates survive NumPy's last-write-wins assignment (see module docstring)."""
    rng = rng or np.random.default_rng(0)
    M = M.tocsr()
    M.sort_indices()
    P = rng.random((M.shape[0], k))
    Q = rng.random((k, M.shape[1]))
    nz_rows = np.flatnonzero(np.diff(M.indptr))
    last_r = M.indptr[nz_rows + 1] - 1                      # each reader's last observed entry (row-major)
    iP, jP, vP = nz_rows, M.indices[last_r], M.data[last_r].astype(np.float64)
    C = M.tocsc()
    C.sort_indices()
    nz_cols = np.flatnonzero(np.diff(C.indptr))
    last_c = C.indptr[nz_cols + 1] - 1                      # each book's last observed entry (highest row)
    iQ, jQ, vQ = C.indices[last_c], nz_cols, C.data[last_c].astype(np.float64)
    for step in range(steps):
        rP = vP - np.einsum("ek,ke->e", P[iP], Q[:, jP])    # residuals use P, Q from the start of the step
        rQ = vQ - np.einsum("ek,ke->e", P[iQ], Q[:, jQ])
        P[iP] = P[iP] + lr * (2 * rP[:, None] * Q[:, jP].T - beta * P[iP])
        Q[:, jQ] = Q[:, jQ] + lr * (2 * rQ[:, None] * P[iQ] - beta * Q[:, jQ].T).T   # uses the updated P
        if (((step + 1) / (steps / 5)) % 1 == 0) or step == 0:
            if lr > 0.00002:
                lr = lr / 1.5
    return P, Q


@dataclass
class GDMF2023(Recommender):
    art: Artifacts = None
    nb: Neighbourhood = field(default=None, repr=False)
    n_neighbours: int = 1000

    def recommend(self, user, k, rng):
        M, cols = self.nb.matrix(user, self.nb.nearest(user, self.n_neighbours))
        P, Q = gd_factorize(M, rng=rng)
        pred = P[-1] @ Q
        ok = _keep(self.art) & (self.art.meta.avg_rating > 3.5)
        return _rank(pred, cols, ok, user, k)


def legacy_baselines(art: Artifacts, subsample: dict) -> list[Recommender]:
    nb = Neighbourhood()
    note = ("Re-implemented on this project's training data (work ids, this catalog); see "
            "goodrec/eval/legacy2023.py for the differences from the original.")
    return [
        SimilarReaders2023(key="similar_readers_2023", name="2023: similar readers", art=art, nb=nb,
                           subsample=subsample.get("similar_readers_2023"),
                           description="The 150 most similar readers (cosine); their most-rated books with >100 "
                                       "ratings and average > 3.75. " + note),
        SVD2023(key="svd_2023", name="2023: SVD", art=art, nb=nb, subsample=subsample.get("svd_2023"),
                description="SVD (k=42) of the 3,000 most similar readers' ratings, ranked by predicted "
                            "rating; average > 3.5. What the 2023 web app served. " + note),
        GDMF2023(key="gd_mf_2023", name="2023: gradient-descent MF", art=art, nb=nb,
                 subsample=subsample.get("gd_mf_2023"),
                 description="Gradient-descent matrix factorization (k=45, 700 steps) of the 1,000 most similar "
                             "readers' ratings, replicating the original's update exactly; average > 3.5. " + note),
    ]
