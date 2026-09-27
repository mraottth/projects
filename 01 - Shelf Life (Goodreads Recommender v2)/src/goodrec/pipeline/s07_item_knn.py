"""s07: precomputed top-K similar books (item-item) from co-ratings.

Adjusted cosine on user-mean-centered ratings, shrunk toward 0 by co-rater count:
    sim(i,j) = cos(r'_i, r'_j) * n_ij / (n_ij + shrinkage),  require n_ij >= min_corated
Computed in blocks of items so the dense (block x N) product stays bounded in memory.

Outputs (artifacts): item_nbrs_idx.npy (N x K int32, -1 = none), item_nbrs_sim.npy (N x K float16)
"""

import argparse
import time

import numpy as np
from scipy import sparse
from tqdm import tqdm

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, load_config
from goodrec.pipeline.io import skip_if_done


def center_rows(R: sparse.csr_matrix) -> sparse.csr_matrix:
    """Subtract each user's mean rating from their nonzero entries."""
    R = R.astype(np.float32)
    counts = np.diff(R.indptr)
    means = np.asarray(R.sum(axis=1)).ravel() / np.maximum(counts, 1)
    X = R.copy()
    X.data -= np.repeat(means, counts).astype(np.float32)
    # A rating exactly at the mean carries no signal; keep a tiny value so co-counts still see it.
    X.data[X.data == 0] = 1e-6
    return X


def item_knn(R: sparse.csr_matrix, k: int, shrinkage: float, min_corated: int, block: int):
    n_items = R.shape[1]
    X = center_rows(R)
    B = R.copy()
    B.data = np.ones_like(B.data, dtype=np.float32)
    B = B.astype(np.float32)
    norms = np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel()).astype(np.float32)
    norms[norms == 0] = 1.0
    Xt, Bt = X.T.tocsr(), B.T.tocsr()
    X, B = X.tocsc(), B.tocsc()

    nbr_idx = np.full((n_items, k), -1, dtype=np.int32)
    nbr_sim = np.zeros((n_items, k), dtype=np.float16)
    for start in tqdm(range(0, n_items, block), desc="item-knn", mininterval=10):
        stop = min(start + block, n_items)
        dots = (Xt[start:stop] @ X).toarray()
        co = (Bt[start:stop] @ B).toarray()
        sim = dots / (norms[start:stop, None] * norms[None, :])
        sim *= co / (co + shrinkage)
        sim[co < min_corated] = -np.inf
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # no self-links
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        top_sim = np.take_along_axis(sim, top, axis=1)
        order = np.argsort(-top_sim, axis=1)
        top, top_sim = np.take_along_axis(top, order, axis=1), np.take_along_axis(top_sim, order, axis=1)
        valid = np.isfinite(top_sim) & (top_sim > 0)
        nbr_idx[start:stop] = np.where(valid, top, -1)
        nbr_sim[start:stop] = np.where(valid, top_sim, 0).astype(np.float16)
    return nbr_idx, nbr_sim


def main(force: bool = False, out_dir=ARTIFACTS_DIR, **overrides) -> None:
    out_i, out_s = out_dir / "item_nbrs_idx.npy", out_dir / "item_nbrs_sim.npy"
    if skip_if_done(out_i, out_s, force=force):
        return
    cfg = {**load_config()["item_knn"], **overrides}
    R = sparse.load_npz(INTERIM_DIR / "R_train.npz")
    t = time.time()
    idx, sim = item_knn(R, cfg["k"], cfg["shrinkage"], cfg["min_corated"], cfg["block"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_i, idx)
    np.save(out_s, sim)
    filled = (idx >= 0).sum(axis=1)
    print(f"  item-knn: {R.shape[1]:,} items, K={cfg['k']} in {time.time() - t:.0f}s; "
          f"median neighbors={np.median(filled):.0f}, items with 0 neighbors={(filled == 0).sum():,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
