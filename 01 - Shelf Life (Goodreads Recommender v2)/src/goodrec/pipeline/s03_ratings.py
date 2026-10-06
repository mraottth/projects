"""s03: stream the ratings source and collapse editions to works.

Input is config data.files.ratings: goodreads_reviews_dedup.json.gz by default, or
goodreads_interactions_dedup.json.gz (the experiment in config/experiments/interactions.yaml). Interactions also
hold to-read shelves: rows with is_read false and no rating. Those are not reads, so they go to to_read.parquet
instead of ratings.parquet (a rating of 0 there means read but unrated, as in the reviews file, which has no
is_read field). Editions of one work are collapsed per user in user-id ranges, so memory stays bounded at
~100M rows.

Outputs (data/interim):
  ratings.parquet  user_idx int32, work_id int64, rating int8 (0 = read, unrated), date int32 (yyyymmdd, shelved),
                   read_date int32 (yyyymmdd from read_at, the date the user finished the book; 0 if not given)
  users.parquet    user_idx, user_id (hex)
  to_read.parquet  user_idx, work_id, date (interactions only: shelved to read, not read)
"""

import argparse

import numpy as np
import polars as pl
from tqdm import tqdm

from goodrec.config import INTERIM_DIR, RAW_DIR, load_config
from goodrec.pipeline.io import ColumnWriter, iter_jsonl_gz, skip_if_done

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _date(s: str) -> int:
    """'Fri Aug 25 13:55:02 -0700 2017' -> 20170825 (0 if unparseable)."""
    try:
        p = s.split()
        return int(p[5]) * 10000 + _MONTHS[p[1]] * 100 + int(p[2])
    except (AttributeError, IndexError, KeyError, ValueError):
        return 0


def main(force: bool = False) -> None:
    out_r, out_u = INTERIM_DIR / "ratings.parquet", INTERIM_DIR / "users.parquet"
    if skip_if_done(out_r, out_u, force=force):
        return
    src = RAW_DIR / load_config()["data"]["files"]["ratings"]

    ed = pl.read_parquet(INTERIM_DIR / "editions.parquet", columns=["book_id", "work_id"])
    book_to_work = dict(zip(ed["book_id"].to_list(), ed["work_id"].to_list()))
    del ed

    users: dict[str, int] = {}
    raw = INTERIM_DIR / "ratings_raw.parquet"
    raw_tr = INTERIM_DIR / "to_read_raw.parquet"
    missing = to_read = 0
    schema = {"user_idx": pl.Int32, "work_id": pl.Int64, "rating": pl.Int8, "date": pl.Int32, "read_date": pl.Int32}
    tr_schema = {"user_idx": pl.Int32, "work_id": pl.Int64, "date": pl.Int32}
    with ColumnWriter(raw, schema, 2_000_000) as w, ColumnWriter(raw_tr, tr_schema, 2_000_000) as wt:
        for r in tqdm(iter_jsonl_gz(src), desc="ratings", mininterval=10):
            work = book_to_work.get(int(r["book_id"]))
            if work is None:
                missing += 1
                continue
            uid = users.setdefault(r["user_id"], len(users))
            rating = int(r.get("rating") or 0)
            if rating == 0 and r.get("is_read") is False:          # interactions: a to-read shelf, not a read
                wt.add(user_idx=uid, work_id=work, date=_date(r.get("date_added")))
                to_read += 1
                continue
            w.add(user_idx=uid, work_id=work, rating=rating, date=_date(r.get("date_added")),
                  read_date=_date(r.get("read_at")))

    # Several editions of one work shelved by the same user -> keep the max rating, latest dates. Done in
    # contiguous user-id ranges so the group-by fits in memory; concatenating the ranges keeps the sort order.
    n_parts = max(1, -(-w.n // 20_000_000))
    bounds = np.linspace(0, len(users), n_parts + 1).astype(np.int64)
    parts = []
    for k in range(n_parts):
        part = INTERIM_DIR / f"ratings_part{k:02d}.parquet"
        (pl.scan_parquet(raw)
           .filter((pl.col("user_idx") >= bounds[k]) & (pl.col("user_idx") < bounds[k + 1]))
           .group_by("user_idx", "work_id")
           .agg(pl.col("rating").max(), pl.col("date").max(), pl.col("read_date").max())
           .sort("user_idx", "work_id")
           .collect()
           .write_parquet(part, compression="zstd"))
        parts.append(part)
    pl.concat([pl.scan_parquet(p) for p in parts]).sink_parquet(out_r, compression="zstd")
    for p in parts:
        p.unlink()
    raw.unlink()
    if wt.n:
        (pl.scan_parquet(raw_tr).group_by("user_idx", "work_id").agg(pl.col("date").max())
           .sort("user_idx", "work_id").sink_parquet(INTERIM_DIR / "to_read.parquet", compression="zstd"))
    raw_tr.unlink(missing_ok=True)
    pl.DataFrame({"user_idx": np.arange(len(users), dtype=np.int32), "user_id": list(users)}).write_parquet(out_u)

    stats = pl.scan_parquet(out_r).select(
        pl.len().alias("rows"), (pl.col("rating") > 0).sum().alias("rated")).collect()
    print(f"  raw rows={w.n:,} to-read={to_read:,} unmapped books={missing:,} users={len(users):,} "
          f"{stats.row(0, named=True)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
