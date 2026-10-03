"""The 2023 Goodreads recommender (../03 - Goodreads Recommender (2023 original)), re-implemented on this
project's training data as evaluation baselines.

Both methods start from the same step as the original: cosine nearest neighbours among readers, on
L2-normalised rating rows (`NearestNeighbors(metric="cosine")` on `Normalizer()` output), with the
target reader's row appended to the neighbourhood matrix. Parameters and filters are the original's
(02_book_recommender.ipynb, Web App/recommender_classes.py):

- Similar readers (notebook Part 1): the 150 nearest readers; books ranked by how many of them rated
  the book, then by average rating; keep Goodreads ratings_count > 100 and average > 3.75.
- SVD (notebook 2.1-2.2 and the 2023 web app): the 3,000 nearest readers; `svds(k=42)` of their rating
  matrix; rank by the target's reconstructed row; keep average > 3.5.

The notebook's third method, gradient-descent matrix factorization (2.3), isn't included: the 2023 web
app never served it, and its update kept only NumPy's last write for repeated indices, so each step
learned from one rating per reader and book (it scored close to random; DECISIONS D-044).

Both drop the reader's own books and later volumes in a series. Differences from the original:
work-level ids (editions collapsed) instead of edition ids; this project's catalog and training users;
the children's-book filter uses this catalog's children's and comics flags instead of the 2023 LDA
genres.
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
    ]
