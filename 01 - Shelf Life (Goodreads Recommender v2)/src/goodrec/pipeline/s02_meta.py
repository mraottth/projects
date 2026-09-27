"""s02: works, authors, and UCSD coarse genre votes.

Outputs (data/interim): works.parquet, authors.parquet, ucsd_genres.parquet (per book_id)
"""

import argparse

import polars as pl
from tqdm import tqdm

from goodrec.config import INTERIM_DIR, RAW_DIR, load_config
from goodrec.pipeline.io import ColumnWriter, iter_jsonl_gz, skip_if_done
from goodrec.pipeline.s01_books import _int

# UCSD's 10 fuzzy genres (from shelf keywords) -> short column names
UCSD_GENRES = {
    "history, historical fiction, biography": "g_history",
    "fiction": "g_fiction",
    "fantasy, paranormal": "g_fantasy",
    "mystery, thriller, crime": "g_mystery",
    "poetry": "g_poetry",
    "romance": "g_romance",
    "non-fiction": "g_nonfiction",
    "children": "g_children",
    "young-adult": "g_ya",
    "comics, graphic": "g_comic",
}


def _parse_rating_dist(s: str) -> dict:
    """'5:1|4:1|3:1|2:0|1:0|total:3' -> {'r5': 1, ..., 'r1': 0}"""
    out = {f"r{i}": 0 for i in range(1, 6)}
    for part in (s or "").split("|"):
        k, _, v = part.partition(":")
        if k in "12345" and k:
            out[f"r{k}"] = _int(v, 0)
    return out


def main(force: bool = False) -> None:
    files = load_config()["data"]["files"]
    out_w, out_a, out_g = (INTERIM_DIR / f for f in ("works.parquet", "authors.parquet", "ucsd_genres.parquet"))
    if skip_if_done(out_w, out_a, out_g, force=force):
        return

    w_schema = {"work_id": pl.Int64, "best_book_id": pl.Int64, "original_title": pl.String,
                "original_pub_year": pl.Int32, "work_ratings_count": pl.Int64, "work_ratings_sum": pl.Int64,
                **{f"r{i}": pl.Int64 for i in range(1, 6)}}
    with ColumnWriter(out_w, w_schema) as w:
        for r in tqdm(iter_jsonl_gz(RAW_DIR / files["works"]), desc="works", mininterval=5):
            w.add(work_id=int(r["work_id"]), best_book_id=_int(r.get("best_book_id")),
                  original_title=r.get("original_title") or "",
                  original_pub_year=_int(r.get("original_publication_year")),
                  work_ratings_count=_int(r.get("ratings_count"), 0),
                  work_ratings_sum=_int(r.get("ratings_sum"), 0), **_parse_rating_dist(r.get("rating_dist")))

    a_schema = {"author_id": pl.Int64, "name": pl.String, "author_ratings_count": pl.Int64}
    with ColumnWriter(out_a, a_schema) as a:
        for r in tqdm(iter_jsonl_gz(RAW_DIR / files["authors"]), desc="authors", mininterval=5):
            a.add(author_id=int(r["author_id"]), name=r.get("name") or "",
                  author_ratings_count=_int(r.get("ratings_count"), 0))

    g_schema = {"book_id": pl.Int64, **{c: pl.Int32 for c in UCSD_GENRES.values()}}
    with ColumnWriter(out_g, g_schema) as g:
        for r in tqdm(iter_jsonl_gz(RAW_DIR / files["genres"]), desc="genres", mininterval=5):
            votes = r.get("genres") or {}
            if votes:
                g.add(book_id=int(r["book_id"]), **{c: int(votes.get(k, 0)) for k, c in UCSD_GENRES.items()})
    print(f"  works={w.n:,} authors={a.n:,} genre rows={g.n:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
