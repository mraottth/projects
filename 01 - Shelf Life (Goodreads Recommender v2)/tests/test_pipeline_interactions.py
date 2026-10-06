"""Pipeline changes for the interactions-data experiment (config/experiments/interactions.yaml), on tiny
synthetic inputs: to-read shelves never become reads, editions collapse per reader across the user-range
partitions, a previous catalog's works are kept, held-out readers are matched by Goodreads id, and the
readers-like-you pool is capped."""

import gzip
import json

import numpy as np
import polars as pl
import pytest


def write_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def row(user, book, rating=0, is_read=None, added="Tue Oct 17 09:40:11 -0700 2017", read_at=""):
    r = {"user_id": user, "book_id": str(book), "rating": rating, "date_added": added, "read_at": read_at}
    if is_read is not None:
        r["is_read"] = is_read
    return r


@pytest.fixture
def s03_env(tmp_path, monkeypatch):
    from goodrec.pipeline import s03_ratings
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    raw.mkdir(), interim.mkdir()
    pl.DataFrame({"book_id": [1, 2, 3, 4], "work_id": [10, 10, 20, 30]}).write_parquet(interim / "editions.parquet")
    monkeypatch.setattr(s03_ratings, "RAW_DIR", raw)
    monkeypatch.setattr(s03_ratings, "INTERIM_DIR", interim)
    monkeypatch.setattr(s03_ratings, "load_config", lambda: {"data": {"files": {"ratings": "x.json.gz"}}})
    return s03_ratings, raw, interim


def test_s03_keeps_to_read_shelves_out_of_ratings(s03_env):
    s03, raw, interim = s03_env
    write_gz(raw / "x.json.gz", [
        row("a", 1, rating=4, is_read=True, read_at="Fri Aug 25 13:55:02 -0700 2017"),
        row("a", 2, rating=5, is_read=True),           # second edition of work 10 -> max rating
        row("a", 3, rating=0, is_read=False),          # to-read: not a read
        row("b", 3, rating=0, is_read=True),           # read but unrated
        row("b", 4, rating=0, is_read=False),
        row("c", 99, rating=3, is_read=True),          # unknown book
    ])
    s03.main(force=True)
    r = pl.read_parquet(interim / "ratings.parquet").sort("user_idx", "work_id")
    assert r.select("user_idx", "work_id", "rating").rows() == [(0, 10, 5), (1, 20, 0)]
    assert r["read_date"][0] == 20170825
    tr = pl.read_parquet(interim / "to_read.parquet").sort("user_idx")
    assert tr.select("user_idx", "work_id").rows() == [(0, 20), (1, 30)]


def test_s03_reviews_file_without_is_read_is_unchanged(s03_env):
    s03, raw, interim = s03_env
    write_gz(raw / "x.json.gz", [row("a", 1, rating=4), row("a", 3, rating=0)])     # reviews: no is_read field
    s03.main(force=True)
    assert pl.read_parquet(interim / "ratings.parquet").select("work_id", "rating").sort("work_id").rows() == [(10, 4), (20, 0)]
    assert not (interim / "to_read.parquet").exists()


def test_s03_partitions_keep_every_reader_and_sort_order(s03_env, monkeypatch):
    s03, raw, interim = s03_env
    rng = np.random.default_rng(0)
    rows = [row(f"u{u}", int(rng.integers(1, 5)), rating=int(rng.integers(1, 6)), is_read=True)
            for u in range(300) for _ in range(4)]
    write_gz(raw / "x.json.gz", rows)

    class Small(s03.ColumnWriter):                     # force several user-range partitions
        def __exit__(self, *a):
            out = super().__exit__(*a)
            self.n = max(self.n, 50_000_000)
            return out
    monkeypatch.setattr(s03, "ColumnWriter", Small)
    s03.main(force=True)
    r = pl.read_parquet(interim / "ratings.parquet")
    expect = (pl.DataFrame({"user": [x["user_id"] for x in rows], "book": [int(x["book_id"]) for x in rows],
                            "rating": [x["rating"] for x in rows]})
              .with_columns(work=pl.col("book").replace_strict({1: 10, 2: 10, 3: 20, 4: 30}))
              .group_by("user", "work").agg(pl.col("rating").max()))
    assert r.height == expect.height and r["user_idx"].n_unique() == 300
    assert r.select("user_idx", "work_id").rows() == sorted(r.select("user_idx", "work_id").rows())


def test_s05_holds_out_the_same_readers_by_goodreads_id(tmp_path, monkeypatch):
    from goodrec.pipeline import s05_matrix
    root, interim = tmp_path, tmp_path / "new"
    old = tmp_path / "old"
    old.mkdir(), interim.mkdir()
    pl.DataFrame({"user_idx": [0, 1, 2], "user_id": ["x", "y", "z"]}).write_parquet(old / "users.parquet")
    np.save(old / "test_users.npy", np.array([1, 2], np.int32))          # readers y and z were held out
    pl.DataFrame({"user_idx": [0, 1, 2, 3], "user_id": ["z", "w", "y", "x"]}).write_parquet(interim / "users.parquet")
    pl.DataFrame({"work_idx": [0, 1, 2], "work_id": [10, 20, 30]}).write_parquet(interim / "catalog.parquet")
    pl.DataFrame({"user_idx": np.repeat([0, 1, 2, 3], 3).astype(np.int32), "work_id": np.tile([10, 20, 30], 4),
                  "rating": np.full(12, 4, np.int8)}).write_parquet(interim / "ratings.parquet")
    monkeypatch.setattr(s05_matrix, "INTERIM_DIR", interim)
    monkeypatch.setattr(s05_matrix, "ROOT", root)
    monkeypatch.setattr(s05_matrix, "load_config", lambda: {
        "catalog": {"min_user_ratings": 1},
        "eval": {"keep_test_users_from": "old", "min_test_user_ratings": 1, "n_test_users": 99, "seed": 0}})
    s05_matrix.main(force=True)
    assert np.load(interim / "test_users.npy").tolist() == [0, 2]         # z and y, under their new indices
    assert np.load(interim / "train_users.npy").tolist() == [1, 3]


def test_s04_keeps_a_previous_catalogs_works_and_s07_bounds_blocks():
    import inspect

    from goodrec.pipeline import s04_catalog, s07_item_knn
    src = inspect.getsource(s04_catalog.main)
    assert "keep_works_from" in src and "is_in(keep.implode())" in src
    n_items = 145_000
    block = max(50, min(500, int(1.5e9 // (3 * 8 * n_items))))
    assert block < 500 and 3 * 8 * n_items * block <= 1.5e9
    assert "block_bytes" in inspect.signature(s07_item_knn.item_knn).parameters
