"""similar_readers() scores on a tiny hand-built neighborhood (no artifacts needed)."""

from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from goodrec.core.similar_readers import similar_readers


def _art():
    # 8 users close to the query vector, 2 far away; 4 books.
    uf = np.array([[1.0, 0.0]] * 8 + [[0.0, 1.0]] * 2, dtype=np.float32)
    r = np.zeros((10, 4), dtype=np.int8)
    r[:8, 0] = 6                  # book 0: every neighbor read it, but so does everyone
    r[:4, 1] = 6                  # book 1: half the neighbors, almost nobody else
    r[:8, 2] = 5                  # book 2: neighbors give 5 stars to a book Goodreads rates 4.6
    r[:8, 3] = 4                  # book 3: neighbors give 4 stars to a book Goodreads rates 3.0
    meta = NS(bayes=np.array([4.0, 4.0, 4.6, 3.1], np.float32), avg_rating=np.array([4.0, 4.0, 4.6, 3.0], np.float32),
              reader_rate=np.array([0.95, 0.001, 0.3, 0.3], np.float32),
              parent_genre=np.zeros(4, np.int16), genre_names=["Fiction"])
    return NS(user_factors=uf, readers=sparse.csr_matrix(r), meta=meta)


def test_absolute_and_relative_scores():
    sr = similar_readers(_art(), np.array([1.0, 0.0]))
    s = sr["score"]
    assert s["popularity", False][0] > s["popularity", False][1]      # most read by neighbors
    assert s["popularity", True][1] > s["popularity", True][0]        # most distinctive to them
    assert s["popularity", True][0] < 1.1                             # read by neighbors about as often as by everyone
    assert s["rating", False][2] > s["rating", False][3]              # highest neighbor average
    assert s["rating", True][3] > s["rating", True][2] > 0            # furthest above Goodreads
    assert s["rating", True][0] == 0                                  # no neighbor ratings: same as Goodreads
    assert sr["item_n"][2] == 8 and sr["item_avg"][3] == 4.0
    assert np.isclose(sr["pct_read_all"][1], 4 / 9)                   # unweighted share of the 9 neighbors
