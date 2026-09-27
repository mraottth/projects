"""s01: one streaming pass over goodreads_books.json.gz.

Outputs (data/interim):
  editions.parquet     one row per edition (book_id -> work_id + display metadata)
  descriptions.parquet book_id, description (truncated)
  shelves.parquet      book_id, shelf, count  (count >= 2; filtered to the catalog later)
"""

import argparse

import polars as pl
from tqdm import tqdm

from goodrec.config import INTERIM_DIR, RAW_DIR, load_config
from goodrec.pipeline.io import ColumnWriter, iter_jsonl_gz, skip_if_done

DESC_MAX = 1200
MIN_SHELF_COUNT = 2


def _int(s, default=None):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def _float(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def main(force: bool = False) -> None:
    out_ed = INTERIM_DIR / "editions.parquet"
    out_desc = INTERIM_DIR / "descriptions.parquet"
    out_sh = INTERIM_DIR / "shelves.parquet"
    if skip_if_done(out_ed, out_desc, out_sh, force=force):
        return
    src = RAW_DIR / load_config()["data"]["files"]["books"]

    ed_schema = {
        "book_id": pl.Int64, "work_id": pl.Int64, "title": pl.String, "title_without_series": pl.String,
        "isbn": pl.String, "isbn13": pl.String, "language_code": pl.String, "pub_year": pl.Int32,
        "num_pages": pl.Int32, "ratings_count": pl.Int64, "avg_rating": pl.Float32,
        "image_url": pl.String, "url": pl.String, "author_id": pl.Int64, "format": pl.String,
    }
    with ColumnWriter(out_ed, ed_schema) as ed, \
         ColumnWriter(out_desc, {"book_id": pl.Int64, "description": pl.String}) as desc, \
         ColumnWriter(out_sh, {"book_id": pl.Int64, "shelf": pl.String, "count": pl.Int32}, 2_000_000) as sh:
        for b in tqdm(iter_jsonl_gz(src), total=2_360_655, desc="books", mininterval=5):
            book_id = int(b["book_id"])
            work_id = _int(b.get("work_id")) or -book_id  # editions without a work stand alone
            authors = b.get("authors") or []
            primary = next((a for a in authors if not a.get("role")), authors[0] if authors else None)
            ed.add(
                book_id=book_id, work_id=work_id, title=b.get("title") or "",
                title_without_series=b.get("title_without_series") or "",
                isbn=b.get("isbn") or "", isbn13=b.get("isbn13") or "",
                language_code=b.get("language_code") or "", pub_year=_int(b.get("publication_year")),
                num_pages=_int(b.get("num_pages")), ratings_count=_int(b.get("ratings_count"), 0),
                avg_rating=_float(b.get("average_rating")), image_url=b.get("image_url") or "",
                url=b.get("url") or b.get("link") or "",
                author_id=_int(primary["author_id"]) if primary else None, format=b.get("format") or "",
            )
            if b.get("description"):
                desc.add(book_id=book_id, description=b["description"][:DESC_MAX])
            for s in b.get("popular_shelves") or []:
                c = _int(s.get("count"), 0)
                if c >= MIN_SHELF_COUNT:
                    sh.add(book_id=book_id, shelf=s["name"], count=c)
    print(f"  editions={ed.n:,} descriptions={desc.n:,} shelf rows={sh.n:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
