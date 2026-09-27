"""s09: SQLite catalog for search, book detail, and CSV import matching.

Tables:
  works        one row per catalog work (display metadata, genre tags, description snippet)
  works_fts    FTS5 over title / author / series (prefix search for typeahead)
  authors      author_id, name, total ratings across catalog works; authors_fts for the filter typeahead
  editions     book_id -> work_idx for every edition of a catalog work (Goodreads export "Book Id")
  isbns        isbn10/isbn13 -> work_idx
  titlekeys    normalized title + author last name -> work_idx
Genres (from s06) are optional so the DB can be built before the genre map is reviewed.
"""

import argparse
import sqlite3

import orjson
import polars as pl

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR
from goodrec.core.textnorm import author_key, clean_isbn, isbn10_to_13, titlekey
from goodrec.pipeline.io import skip_if_done

SNIPPET = 1200  # = s01 DESC_MAX: keep everything we ingested

SCHEMA = """
CREATE TABLE works (
  work_idx INTEGER PRIMARY KEY, work_id INTEGER, book_id INTEGER, title TEXT, base_title TEXT,
  author TEXT, author_id INTEGER, year INTEGER, avg_rating REAL, ratings_count INTEGER, n_raters INTEGER,
  cover_url TEXT, isbn TEXT, url TEXT, num_pages INTEGER, series_name TEXT, series_pos REAL,
  is_boxset INTEGER, is_children INTEGER, is_comic INTEGER, parent_genre TEXT, tags TEXT, description TEXT
);
CREATE VIRTUAL TABLE works_fts USING fts5(
  title, author, series_name, content='works', content_rowid='work_idx',
  tokenize='unicode61 remove_diacritics 2', prefix='1 2 3'
);
CREATE TABLE authors (author_id INTEGER PRIMARY KEY, name TEXT, total_ratings INTEGER, n_works INTEGER);
CREATE VIRTUAL TABLE authors_fts USING fts5(
  name, content='authors', content_rowid='author_id', tokenize='unicode61 remove_diacritics 2', prefix='1 2 3'
);
CREATE TABLE editions (book_id INTEGER PRIMARY KEY, work_idx INTEGER) WITHOUT ROWID;
CREATE TABLE isbns (isbn TEXT PRIMARY KEY, work_idx INTEGER) WITHOUT ROWID;
CREATE TABLE titlekeys (titlekey TEXT, author_key TEXT, work_idx INTEGER, ratings_count INTEGER);
CREATE INDEX titlekeys_idx ON titlekeys (titlekey, author_key);
"""


def main(force: bool = False) -> None:
    out = ARTIFACTS_DIR / "catalog.db"
    if skip_if_done(out, force=force):
        return
    cat = pl.read_parquet(INTERIM_DIR / "catalog.parquet")
    genres_path = INTERIM_DIR / "work_genres.parquet"
    if genres_path.exists():
        g = pl.read_parquet(genres_path).select("work_id", "parent_genre", "tags")
        cat = cat.join(g, on="work_id", how="left")
    else:
        print("  note: work_genres.parquet not found; building without genres (re-run after s06)")
        cat = cat.with_columns(parent_genre=pl.lit(None, pl.String), tags=pl.lit(None, pl.List(pl.String)))

    desc = (pl.scan_parquet(INTERIM_DIR / "descriptions.parquet")
              .join(cat.lazy().select("book_id"), on="book_id", how="semi")
              .with_columns(pl.col("description").str.slice(0, SNIPPET))
              .collect())
    cat = cat.join(desc, on="book_id", how="left")

    ed = (pl.scan_parquet(INTERIM_DIR / "editions.parquet")
            .select("book_id", "work_id", "isbn", "isbn13", "title")
            .join(cat.lazy().select("work_id", "work_idx", "ratings_count", "author"), on="work_id")
            .collect())

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)

    cols = ["work_idx", "work_id", "book_id", "title", "base_title", "author", "author_id", "year",
            "avg_rating", "ratings_count", "n_raters", "cover_url", "isbn_any", "url", "num_pages",
            "series_name", "series_pos", "is_boxset", "is_children", "is_comic", "parent_genre", "tags",
            "description"]
    rows = []
    for r in cat.select(cols).iter_rows():
        r = list(r)
        r[-2] = orjson.dumps(r[-2]).decode() if r[-2] is not None else "[]"
        rows.append(r)
    con.executemany(f"INSERT INTO works VALUES ({','.join('?' * len(cols))})", rows)
    con.execute("INSERT INTO works_fts(works_fts) VALUES ('rebuild')")

    authors = (cat.filter(pl.col("author_id").is_not_null())
                  .group_by("author_id").agg(pl.col("author").first(), pl.col("ratings_count").sum(), pl.len()))
    con.executemany("INSERT INTO authors VALUES (?,?,?,?)", authors.iter_rows())
    con.execute("INSERT INTO authors_fts(authors_fts) VALUES ('rebuild')")

    con.executemany("INSERT INTO editions VALUES (?,?)", ed.select("book_id", "work_idx").iter_rows())

    isbn_map: dict[str, tuple[int, int]] = {}
    for isbn, isbn13, widx, cnt in ed.select("isbn", "isbn13", "work_idx", "ratings_count").iter_rows():
        i10, i13 = clean_isbn(isbn), clean_isbn(isbn13)
        for key in {i10, i13, isbn10_to_13(i10)} - {""}:
            if key not in isbn_map or cnt > isbn_map[key][1]:
                isbn_map[key] = (widx, cnt)
    con.executemany("INSERT INTO isbns VALUES (?,?)", ((k, v[0]) for k, v in isbn_map.items()))

    tk = {(titlekey(t), author_key(a), w, c) for t, a, w, c in
          ed.select("title", "author", "work_idx", "ratings_count").iter_rows()}
    tk |= {(titlekey(t), author_key(a), w, c) for t, a, w, c in
           cat.select("base_title", "author", "work_idx", "ratings_count").iter_rows()}
    con.executemany("INSERT INTO titlekeys VALUES (?,?,?,?)", (t for t in tk if t[0]))

    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp.rename(out)
    print(f"  catalog.db: works={cat.height:,} editions={ed.height:,} isbns={len(isbn_map):,} "
          f"titlekeys={len(tk):,} authors={authors.height:,} size={out.stat().st_size / 1e6:.0f}MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
