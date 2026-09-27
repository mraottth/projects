"""s06: descriptive genres from Goodreads reader shelves (replaces the old LDA topics).

Two modes:
  --draft  one-time: rank candidate shelves across the catalog and ask Claude to map each to
           {keep, display, parent} using a fixed parent-genre list -> config/shelf_genres.yaml.
           Review/edit that file by hand before applying.
  (default) apply config/shelf_genres.yaml -> data/interim/work_genres.parquet:
           work_id, parent_genre, tags (top-3 display names), genre weights

Per-work tag weight = shelf count share x idf across the catalog, so ubiquitous shelves
("fiction") don't crowd out specific ones ("cozy mystery"). The parent genre is the parent
with the largest undamped share, falling back to UCSD's coarse genre votes.
"""

import argparse
import json
import math
import re

import numpy as np
import polars as pl
import yaml

from goodrec.config import CONFIG_DIR, INTERIM_DIR
from goodrec.pipeline.io import skip_if_done

GENRE_MAP = CONFIG_DIR / "shelf_genres.yaml"

PARENTS = [
    "Literary Fiction", "Contemporary Fiction", "Historical Fiction", "Classics", "Mystery & Crime",
    "Thriller & Suspense", "Horror", "Science Fiction", "Fantasy", "Paranormal & Urban Fantasy", "Romance",
    "Erotica", "Young Adult", "Children's & Middle Grade", "Graphic Novels & Comics", "Poetry",
    "Plays & Drama", "Humor", "Women's Fiction", "Christian & Inspirational", "Adventure & Action",
    "Short Stories & Anthologies", "Biography & Memoir", "History", "Science & Nature",
    "Self-Help & Psychology", "Business & Economics", "Philosophy & Religion", "Politics & Society",
    "True Crime", "Travel", "Food, Health & Lifestyle", "Arts & Culture", "Sports",
]

# Obviously non-genre shelves, dropped before drafting to save tokens (the LLM catches the rest).
_STOP = re.compile(
    r"^(to-?read|currently-?reading|read|re-?read|owned?|i-own|books-i-own|my-books|my-library|library|"
    r"favou?rites?|all-time-favou?rites?|faves?|default|wish-?list|tbr|to-?buy|kindle|e-?books?|nook|"
    r"audio(books?)?|audible|paperback|hardcover|dnf|did-not-finish|abandoned|unfinished|on-hold|maybe|"
    r"borrowed|library-books?|book-?club|school|arc|arcs|netgalley|series|giveaways?|first-reads|"
    r"\d-stars?|five-stars?|\w+-stars?|books|novels?|to-read-.*|read-in-.*|.*-read|read-.*|\d{4}.*|"
    r".*-\d{4}|.*owned.*|.*kindle.*|.*shelf.*|.*tbr.*|.*library.*|p|w|r|x)$"
)

UCSD_FALLBACK = {
    "g_fantasy": "Fantasy", "g_mystery": "Mystery & Crime", "g_romance": "Romance", "g_poetry": "Poetry",
    "g_children": "Children's & Middle Grade", "g_ya": "Young Adult", "g_comic": "Graphic Novels & Comics",
    "g_history": "Historical Fiction", "g_nonfiction": "History", "g_fiction": "Contemporary Fiction",
}


def work_shelves() -> pl.DataFrame:
    """work_id, shelf (normalized), count, summed over the editions of each catalog work."""
    cat = pl.read_parquet(INTERIM_DIR / "catalog.parquet", columns=["work_id"])
    ed = pl.scan_parquet(INTERIM_DIR / "editions.parquet").select("book_id", "work_id") \
           .join(cat.lazy(), on="work_id", how="semi")
    return (pl.scan_parquet(INTERIM_DIR / "shelves.parquet")
              .join(ed, on="book_id")
              .with_columns(shelf=pl.col("shelf").str.to_lowercase().str.strip_chars()
                                     .str.replace_all(r"[\s_]+", "-").str.replace_all(r"-+", "-")
                                     .str.strip_chars("-"))
              .group_by("work_id", "shelf").agg(pl.col("count").sum())
              .collect())


# ---------------------------------------------------------------- draft (one-time, uses Claude)

def candidates(n_shelves: int = 500) -> list[tuple[str, int, list[str]]]:
    """Top shelves by catalog coverage (after the stoplist) with 3 example titles each."""
    ws = work_shelves()
    titles = pl.read_parquet(INTERIM_DIR / "catalog.parquet", columns=["work_id", "base_title", "n_raters"])
    # Coverage: number of works where the shelf is among that work's top-30 shelves.
    ranked = ws.with_columns(rk=pl.col("count").rank("ordinal", descending=True).over("work_id")) \
               .filter(pl.col("rk") <= 30)
    cand = (ranked.group_by("shelf").agg(pl.len().alias("works"))
                  .filter(~pl.col("shelf").str.contains(_STOP.pattern))
                  .sort("works", descending=True).head(n_shelves))
    examples = (ranked.join(cand.select("shelf"), on="shelf").join(titles, on="work_id")
                      .sort("n_raters", descending=True).group_by("shelf", maintain_order=True)
                      .agg(pl.col("base_title").head(3)))
    ex = dict(zip(examples["shelf"].to_list(), examples["base_title"].to_list()))
    return [(s, n, ex.get(s, [])) for s, n in cand.iter_rows()]


def draft(n_shelves: int = 500, batch: int = 100) -> None:
    import anthropic
    from pydantic import BaseModel

    rows = candidates(n_shelves)
    print(f"  drafting {len(rows)} shelves in batches of {batch}")

    class Mapping(BaseModel):
        shelf: str
        keep: bool
        display: str
        parent: str | None

    schema = {
        "type": "object",
        "properties": {"mappings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "shelf": {"type": "string"},
                "keep": {"type": "boolean"},
                "display": {"type": "string"},
                "parent": {"anyOf": [{"type": "string", "enum": PARENTS}, {"type": "null"}]},
            },
            "required": ["shelf", "keep", "display", "parent"],
            "additionalProperties": False,
        }}},
        "required": ["mappings"],
        "additionalProperties": False,
    }
    system = (
        "You label Goodreads user shelf names for a book recommender's genre system. For each shelf decide:\n"
        "- keep: true only if the shelf describes a genre, subgenre, audience, form, setting, or theme that "
        "helps a reader browse (e.g. 'cozy-mystery', 'space-opera', 'ww2', 'memoir', 'lgbt'). false for "
        "reading status, ownership, formats, ratings, dates, personal lists, and labels too broad to "
        "discriminate ('fiction', 'nonfiction', 'adult', 'literature', 'novels', 'books', 'general').\n"
        "- display: a short human-readable Title Case name ('Cozy Mystery', 'Space Opera', 'World War II', "
        "'LGBTQ+'). Give synonyms the identical display name so they merge ('sci-fi', 'scifi', "
        "'science-fiction' -> 'Science Fiction').\n"
        "- parent: the single best parent genre from the allowed list, or null if keep is false or it is a "
        "cross-cutting theme with no natural parent.\n"
        "Example titles shelved under each shelf are given to disambiguate. Return one mapping per input "
        "shelf, with the shelf string copied exactly."
    )
    client = anthropic.Anthropic()
    out: dict[str, dict] = {}
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        listing = "\n".join(f"- {s} (on {n} books; e.g. {'; '.join(t)})" for s, n, t in chunk)
        resp = client.beta.messages.create(
            model="claude-opus-5",
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": "Allowed parent genres:\n" + "\n".join(PARENTS)
                       + f"\n\nShelves:\n{listing}"}],
        )
        if resp.stop_reason == "refusal":
            print(f"  batch {start // batch}: refused ({resp.stop_details}); those shelves are left out")
            continue
        text = next(b.text for b in resp.content if b.type == "text")
        for m in json.loads(text)["mappings"]:
            m = Mapping(**m)
            out[m.shelf] = {"keep": m.keep, "display": m.display, "parent": m.parent}
        print(f"  batch {start // batch + 1}/{math.ceil(len(rows) / batch)}: {len(out)} shelves mapped")

    missing = [s for s, _, _ in rows if s not in out]
    header = ("# Drafted by `make genres-draft` (Claude), then reviewed by hand.\n"
              "# keep=false shelves are ignored. Synonyms share a display name. Edit freely, then `make genres`.\n")
    doc = {"parents": PARENTS,
           "shelves": {s: out[s] for s, _, _ in rows if s in out}}
    GENRE_MAP.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))
    kept = sum(v["keep"] for v in out.values())
    print(f"  wrote {GENRE_MAP.name}: {len(out)} shelves ({kept} kept); unmapped: {missing[:10]}")


# ---------------------------------------------------------------- apply

def apply(force: bool = False) -> None:
    out = INTERIM_DIR / "work_genres.parquet"
    if skip_if_done(out, force=force):
        return
    if not GENRE_MAP.exists():
        raise SystemExit(f"{GENRE_MAP} not found; run `make genres-draft` first (or write it by hand)")
    doc = yaml.safe_load(GENRE_MAP.read_text())
    kept = {s: v for s, v in doc["shelves"].items() if v and v.get("keep")}
    mapping = pl.DataFrame({"shelf": list(kept), "display": [v["display"] for v in kept.values()],
                            "parent": [v.get("parent") for v in kept.values()]})

    ws = work_shelves().join(mapping, on="shelf")
    # Merge synonyms (same display name) within each work.
    ws = ws.group_by("work_id", "display").agg(pl.col("count").sum(), pl.col("parent").drop_nulls().first())
    n_works = pl.read_parquet(INTERIM_DIR / "catalog.parquet", columns=["work_id"]).height
    df = ws.group_by("display").agg(pl.len().alias("df"))
    ws = (ws.join(df, on="display")
            .with_columns(share=pl.col("count") / pl.col("count").sum().over("work_id"))
            .with_columns(weight=pl.col("share") * (pl.lit(n_works) / pl.col("df")).log()))

    tags = (ws.sort("weight", descending=True).group_by("work_id", maintain_order=True)
              .agg(pl.col("display").head(3).alias("tags")))
    parents = (ws.drop_nulls("parent").group_by("work_id", "parent").agg(pl.col("share").sum())
                 .sort("share", descending=True).group_by("work_id", maintain_order=True)
                 .agg(pl.col("parent").first().alias("parent_genre")))

    cat = pl.read_parquet(INTERIM_DIR / "catalog.parquet", columns=["work_id", *UCSD_FALLBACK])
    g = cat.select(list(UCSD_FALLBACK)).fill_null(0).to_numpy()
    fallback = np.array(list(UCSD_FALLBACK.values()))[g.argmax(axis=1)]
    fallback = np.where(g.sum(axis=1) > 0, fallback, None)
    res = (cat.select("work_id").with_columns(fallback=pl.Series(fallback, dtype=pl.String))
              .join(parents, on="work_id", how="left").join(tags, on="work_id", how="left")
              .with_columns(parent_genre=pl.coalesce("parent_genre", "fallback"),
                            tags=pl.col("tags").fill_null(pl.lit([], pl.List(pl.String))))
              .drop("fallback"))
    res.write_parquet(out)
    has_tags = (res["tags"].list.len() > 0).mean()
    print(f"  work_genres: {res.height:,} works; with tags {has_tags:.1%}; "
          f"with parent {res['parent_genre'].is_not_null().mean():.1%}")
    print(res.group_by("parent_genre").len().sort("len", descending=True).head(40))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draft", action="store_true", help="draft config/shelf_genres.yaml with Claude")
    p.add_argument("--candidates", action="store_true",
                   help="write ranked candidate shelves to data/interim/shelf_candidates.tsv (for manual drafting)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.candidates:
        rows = candidates()
        path = INTERIM_DIR / "shelf_candidates.tsv"
        path.write_text("".join(f"{s}\t{n}\t{' | '.join(t)}\n" for s, n, t in rows))
        print(f"  wrote {len(rows)} candidates to {path}")
    elif a.draft:
        draft()
    else:
        apply(force=a.force)
