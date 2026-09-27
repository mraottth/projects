"""s04: choose the catalog of works and their display metadata.

A work is in the catalog if it has >= catalog.min_raters distinct 1-5 star raters
in our ratings data and at least one edition in an allowed language.

Output: data/interim/catalog.parquet with dense work_idx (0 = most rated).
"""

import argparse

import polars as pl

from goodrec.config import INTERIM_DIR, load_config
from goodrec.core.textnorm import ascii_fold, is_boxset, parse_series
from goodrec.pipeline.io import skip_if_done
from goodrec.pipeline.s02_meta import UCSD_GENRES

GENRE_COLS = list(UCSD_GENRES.values())


def _latin_share(title: str) -> float:
    letters = [ch for ch in title if ch.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for ch in ascii_fold("".join(letters)) if ch.isalpha()) / len(letters)


def main(force: bool = False) -> None:
    out = INTERIM_DIR / "catalog.parquet"
    if skip_if_done(out, force=force):
        return
    cfg = load_config()["catalog"]

    raters = (pl.scan_parquet(INTERIM_DIR / "ratings.parquet")
                .filter(pl.col("rating") > 0)
                .group_by("work_id")
                .agg(pl.len().alias("n_raters"), pl.col("rating").mean().alias("data_avg"))
                .filter(pl.col("n_raters") >= cfg["min_raters"])
                .collect())

    ed = (pl.scan_parquet(INTERIM_DIR / "editions.parquet")
            .join(raters.lazy().select("work_id"), on="work_id", how="semi")
            .with_columns(
                lang_ok=pl.col("language_code").is_in(cfg["languages"]),
                has_cover=(pl.col("image_url") != "") & ~pl.col("image_url").str.contains("nophoto"))
            .collect())
    works = pl.read_parquet(INTERIM_DIR / "works.parquet")

    # Display edition: best_book_id if it's in an allowed language, else the most-rated allowed edition.
    ed = ed.join(works.select("work_id", "best_book_id"), on="work_id", how="left")
    ranked = ed.filter(pl.col("lang_ok")).with_columns(
        is_best=pl.col("book_id") == pl.col("best_book_id")
    ).sort(["work_id", "is_best", "ratings_count"], descending=[False, True, True])
    display = ranked.group_by("work_id", maintain_order=True).first()
    # Cover: display edition's image if real, else the most-rated allowed edition with a real image.
    covers = (ranked.filter(pl.col("has_cover")).sort(["work_id", "is_best", "ratings_count"],
                                                      descending=[False, True, True])
                    .group_by("work_id", maintain_order=True).first()
                    .select("work_id", cover_url="image_url"))
    # Any ISBN on an allowed edition (for the Open Library cover fallback).
    isbns = (ranked.with_columns(isbn_any=pl.when(pl.col("isbn13") != "").then("isbn13")
                                 .when(pl.col("isbn") != "").then("isbn"))
                   .drop_nulls("isbn_any").group_by("work_id", maintain_order=True).first()
                   .select("work_id", "isbn_any"))

    genres = (pl.read_parquet(INTERIM_DIR / "ucsd_genres.parquet")
                .join(ed.select("book_id", "work_id"), on="book_id")
                .group_by("work_id").agg(pl.col(GENRE_COLS).sum()))
    total = pl.sum_horizontal(GENRE_COLS)
    genres = genres.with_columns(
        is_children=(pl.col("g_children") / total >= 0.25) & (pl.col("g_children") >= pl.col("g_ya")),
        is_comic=pl.col("g_comic") / total >= 0.25,
    )
    authors = pl.read_parquet(INTERIM_DIR / "authors.parquet", columns=["author_id", "name"])

    cat = (display.select("work_id", "book_id", "title", "author_id", "url", "pub_year", "num_pages")
           .join(raters, on="work_id")
           .join(works.select("work_id", "original_pub_year", "work_ratings_count", "work_ratings_sum",
                              "r1", "r2", "r3", "r4", "r5"), on="work_id", how="left")
           .join(covers, on="work_id", how="left")
           .join(isbns, on="work_id", how="left")
           .join(genres.select("work_id", "is_children", "is_comic", *GENRE_COLS), on="work_id", how="left")
           .join(authors.rename({"name": "author"}), on="author_id", how="left"))

    # Many non-English works have no language_code on any edition; drop titles not in Latin script.
    cat = cat.filter(pl.Series([_latin_share(t) >= 0.5 for t in cat["title"].to_list()]))

    series = [parse_series(t) for t in cat["title"].to_list()]
    cat = cat.with_columns(
        base_title=pl.Series([s[0] for s in series]),
        series_name=pl.Series([s[1] for s in series], dtype=pl.String),
        series_pos=pl.Series([s[2] for s in series], dtype=pl.Float32),
        is_boxset=pl.Series([is_boxset(t) for t in cat["title"].to_list()]),
        year=pl.coalesce("original_pub_year", "pub_year"),
        # Goodreads-wide totals (all editions) for display and filters; our data counts drive the model.
        ratings_count=pl.max_horizontal("work_ratings_count", "n_raters"),
        avg_rating=pl.when(pl.col("work_ratings_count") > 0)
                     .then(pl.col("work_ratings_sum") / pl.col("work_ratings_count"))
                     .otherwise(pl.col("data_avg")).round(2).cast(pl.Float32),
        is_children=pl.col("is_children").fill_null(False),
        is_comic=pl.col("is_comic").fill_null(False),
        author=pl.col("author").fill_null(""),
    ).sort(["n_raters", "work_id"], descending=[True, False]).with_row_index("work_idx")  # deterministic order

    cat = cat.select(
        pl.col("work_idx").cast(pl.Int32), "work_id", "book_id", "title", "base_title", "author", "author_id",
        "year", "avg_rating", "ratings_count", "n_raters", "data_avg", "r1", "r2", "r3", "r4", "r5",
        "cover_url", "isbn_any", "url", "num_pages", "series_name", "series_pos", "is_boxset",
        "is_children", "is_comic", *GENRE_COLS,
    )
    cat.write_parquet(out)
    print(f"  catalog works={cat.height:,} (raters>={cfg['min_raters']}); "
          f"covers={cat['cover_url'].is_not_null().mean():.1%} boxsets={cat['is_boxset'].sum():,} "
          f"children={cat['is_children'].sum():,} comics={cat['is_comic'].sum():,} "
          f"series={cat['series_name'].is_not_null().sum():,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
