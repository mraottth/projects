"""s08: implicit-feedback ALS book embeddings (serving uses numpy fold-in, not `implicit`).

Confidence c = 1 + alpha * g(r), with g from config als.confidence (5 > 4 > 3 > read).
1-2 star ratings are not positives and are left out of the matrix.

Outputs (artifacts):
  item_factors.npy   N x F float32
  yty.npy            F x F float32 (Y^T Y, precomputed for fold-in)
  user_factors.npy   L2-normalized float16 vectors for train users with enough ratings (similar readers)
  user_rows.npy      row in R_train for each user_factors row
"""

import argparse
import os
import time

import numpy as np
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, load_config
from goodrec.pipeline.io import skip_if_done

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit recommends this; it parallelizes itself


def confidence_matrix(R: sparse.csr_matrix, Read: sparse.csr_matrix, alpha: float, g: dict) -> sparse.csr_matrix:
    """Map ratings (and read-unrated flags) to ALS confidence values."""
    lut = np.zeros(6, dtype=np.float32)
    for r, w in g.items():
        if int(r) > 0:
            lut[int(r)] = 1 + alpha * w
    C = R.astype(np.float32)
    C.data = lut[R.data.astype(np.int64)]
    if Read is not None and Read.nnz:
        Cr = Read.astype(np.float32) * (1 + alpha * g.get(0, 0.0))
        C = C + Cr
    C.eliminate_zeros()
    return C.tocsr()


def train_als(C: sparse.csr_matrix, factors: int, regularization: float, iterations: int, seed: int = 42):
    from implicit.cpu.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                    iterations=iterations, random_state=seed, calculate_training_loss=False)
    model.fit(C, show_progress=True)
    return np.asarray(model.user_factors, dtype=np.float32), np.asarray(model.item_factors, dtype=np.float32)


def main(force: bool = False, out_dir=ARTIFACTS_DIR, **overrides) -> None:
    outs = [out_dir / f for f in ("item_factors.npy", "yty.npy", "user_factors.npy", "user_rows.npy")]
    if skip_if_done(*outs, force=force):
        return
    cfg = {**load_config()["als"], **overrides}
    R = sparse.load_npz(INTERIM_DIR / "R_train.npz")
    Read = sparse.load_npz(INTERIM_DIR / "Read_train.npz")
    C = confidence_matrix(R, Read, cfg["alpha"], {int(k): v for k, v in cfg["confidence"].items()})

    t = time.time()
    U, Y = train_als(C, cfg["factors"], cfg["regularization"], cfg["iterations"])
    print(f"  ALS {C.shape} nnz={C.nnz:,} F={cfg['factors']} in {time.time() - t:.0f}s")

    rows = np.flatnonzero(np.diff(R.indptr) >= cfg["min_user_ratings_for_neighbors"]).astype(np.int32)
    cap = cfg.get("max_neighbor_users")      # readers-like-you pool: a seeded random sample bounds serving memory
    if cap and len(rows) > cap:
        rows = np.sort(np.random.default_rng(0).choice(rows, cap, replace=False)).astype(np.int32)
    Un = U[rows]
    Un /= np.maximum(np.linalg.norm(Un, axis=1, keepdims=True), 1e-8)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(outs[0], Y)
    np.save(outs[1], (Y.T @ Y).astype(np.float32))
    np.save(outs[2], Un.astype(np.float16))
    np.save(outs[3], rows)
    print(f"  user_factors for {len(rows):,} users with >= {cfg['min_user_ratings_for_neighbors']} ratings"
          + (f" (random sample, cap {cap:,})" if cap else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
