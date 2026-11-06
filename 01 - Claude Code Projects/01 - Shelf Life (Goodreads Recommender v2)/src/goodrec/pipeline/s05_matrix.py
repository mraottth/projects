"""s05: users x catalog rating matrices, with a fixed held-out test-user split.

Test users are excluded from every model (item-kNN, ALS, similar readers) so
`make eval` measures true fold-in performance. At ~5% of users the production
cost of excluding them is negligible. With eval.keep_test_users_from (another build's interim folder, relative
to the project root), the same readers are held out, matched by Goodreads user id, so models built on both
datasets can be compared on readers neither saw.

Outputs (data/interim):
  R_train.npz / R_test.npz          CSR int8, ratings 1-5 (users x work_idx)
  Read_train.npz / Read_test.npz    CSR int8, 1 where read but unrated
  train_users.npy / test_users.npy  original user_idx for each matrix row
"""

import argparse

import numpy as np
import polars as pl
from scipy import sparse

from goodrec.config import INTERIM_DIR, ROOT, load_config
from goodrec.pipeline.io import skip_if_done


def _csr(rows, cols, vals, shape):
    m = sparse.csr_matrix((vals.astype(np.int8), (rows, cols)), shape=shape)
    m.sort_indices()
    return m


def main(force: bool = False) -> None:
    outs = [INTERIM_DIR / f for f in ("R_train.npz", "R_test.npz", "Read_train.npz", "Read_test.npz",
                                      "train_users.npy", "test_users.npy")]
    if skip_if_done(*outs, force=force):
        return
    cfg = load_config()
    cat = pl.read_parquet(INTERIM_DIR / "catalog.parquet", columns=["work_idx", "work_id"])
    r = (pl.scan_parquet(INTERIM_DIR / "ratings.parquet")
           .join(cat.lazy(), on="work_id")
           .select("user_idx", "work_idx", "rating")
           .collect())

    counts = r.filter(pl.col("rating") > 0).group_by("user_idx").len()
    keep = counts.filter(pl.col("len") >= cfg["catalog"]["min_user_ratings"])
    if cfg["eval"].get("keep_test_users_from"):
        src = ROOT / cfg["eval"]["keep_test_users_from"]
        old = pl.read_parquet(src / "users.parquet").filter(
            pl.col("user_idx").is_in(np.load(src / "test_users.npy").tolist()))["user_id"]
        mine = pl.read_parquet(INTERIM_DIR / "users.parquet")
        test_users = np.sort(mine.filter(pl.col("user_id").is_in(old.implode()))["user_idx"].to_numpy())
        test_users = np.intersect1d(test_users, keep["user_idx"].to_numpy())
        print(f"  held-out readers: {len(test_users):,} of {old.len():,} from {cfg['eval']['keep_test_users_from']}")
    else:
        eligible = keep.filter(pl.col("len") >= cfg["eval"]["min_test_user_ratings"])["user_idx"].to_numpy()
        rng = np.random.default_rng(cfg["eval"]["seed"])
        test_users = np.sort(rng.choice(eligible, size=min(cfg["eval"]["n_test_users"], len(eligible)),
                                        replace=False))
    train_users = np.setdiff1d(keep["user_idx"].to_numpy(), test_users)

    n_items = cat.height
    for name, users in (("train", train_users), ("test", test_users)):
        sub = r.join(pl.DataFrame({"user_idx": users, "row": np.arange(len(users), dtype=np.int32)}),
                     on="user_idx")
        rated, read = sub.filter(pl.col("rating") > 0), sub.filter(pl.col("rating") == 0)
        shape = (len(users), n_items)
        R = _csr(rated["row"].to_numpy(), rated["work_idx"].to_numpy(), rated["rating"].to_numpy(), shape)
        Rd = _csr(read["row"].to_numpy(), read["work_idx"].to_numpy(), np.ones(read.height), shape)
        sparse.save_npz(INTERIM_DIR / f"R_{name}.npz", R)
        sparse.save_npz(INTERIM_DIR / f"Read_{name}.npz", Rd)
        np.save(INTERIM_DIR / f"{name}_users.npy", users.astype(np.int32))
        print(f"  {name}: users={shape[0]:,} items={n_items:,} ratings={R.nnz:,} read-unrated={Rd.nnz:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
