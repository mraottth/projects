"""s03: stream the ratings source and collapse editions to works.

Input is config data.files.ratings: goodreads_reviews_dedup.json.gz by default.
goodreads_interactions_dedup.json.gz uses the same fields (user_id, book_id,
rating, date_added), so it can be swapped in without code changes.

Outputs (data/interim):
  ratings.parquet  user_idx int32, work_id int64, rating int8 (0 = read, unrated), date int32 (yyyymmdd)
  users.parquet    user_idx, user_id (hex)
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
    missing = 0
    schema = {"user_idx": pl.Int32, "work_id": pl.Int64, "rating": pl.Int8, "date": pl.Int32}
    with ColumnWriter(raw, schema, 2_000_000) as w:
        for r in tqdm(iter_jsonl_gz(src), desc="ratings", mininterval=10):
            work = book_to_work.get(int(r["book_id"]))
            if work is None:
                missing += 1
                continue
            uid = users.setdefault(r["user_id"], len(users))
            w.add(user_idx=uid, work_id=work, rating=int(r.get("rating") or 0), date=_date(r.get("date_added")))

    # Several editions of one work rated by the same user -> keep the max rating, latest date.
    (pl.scan_parquet(raw)
       .group_by("user_idx", "work_id")
       .agg(pl.col("rating").max(), pl.col("date").max())
       .sort("user_idx", "work_id")
       .sink_parquet(out_r, compression="zstd"))
    raw.unlink()
    pl.DataFrame({"user_idx": np.arange(len(users), dtype=np.int32), "user_id": list(users)}).write_parquet(out_u)

    stats = pl.scan_parquet(out_r).select(
        pl.len().alias("rows"), (pl.col("rating") > 0).sum().alias("rated")).collect()
    print(f"  raw rows={w.n:,} unmapped books={missing:,} users={len(users):,} {stats.row(0, named=True)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
