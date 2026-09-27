"""Match a Goodreads library export CSV to catalog works.

Order (extends the old app's Book Id -> title matching):
  1. Book Id -> edition -> work (covers every edition of every catalog work)
  2. ISBN13, then ISBN (export wraps them as ="..."), including ISBN10 -> 13 conversion
  3. normalized title + an author last name that must match (ties -> most-rated work)
Shelves: My Rating > 0 -> rating; Exclusive Shelf 'read' with rating 0 -> read-unrated;
'to-read' -> to_read (re-ranked separately; never used as a signal).
"""

import csv
import io

from goodrec.api.catalog import Catalog
from goodrec.core.textnorm import author_key, clean_isbn, isbn10_to_13, titlekey

REQUIRED = {"Book Id", "Title", "My Rating"}


def match_row(cat: Catalog, row: dict) -> tuple[int | None, str | None]:
    book_id = (row.get("Book Id") or "").strip()
    if book_id.isdigit():
        idx = cat.by_edition(int(book_id))
        if idx is not None:
            return idx, "book_id"
    for key in ("ISBN13", "ISBN"):
        isbn = clean_isbn(row.get(key))
        for cand in (isbn, isbn10_to_13(isbn)):
            if cand:
                idx = cat.by_isbn(cand)
                if idx is not None:
                    return idx, "isbn"
    tkey = titlekey(row.get("Title") or "")
    if tkey:
        # Any credited author may be the one the catalog lists (co-authors, editors, translators).
        names = [row.get("Author") or "", row.get("Author l-f") or "", *(row.get("Additional Authors") or "").split(",")]
        idx = cat.by_titlekey(tkey, {author_key(n.strip()) for n in names})
        if idx is not None:
            return idx, "title"
    return None, None


def parse_export(cat: Catalog, raw: bytes) -> dict:
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not REQUIRED <= set(reader.fieldnames):
        raise ValueError("This doesn't look like a Goodreads library export (missing Book Id / Title / My Rating).")

    rated: dict[int, dict] = {}
    read_unrated: set[int] = set()
    to_read: set[int] = set()
    unmatched: list[dict] = []
    counts = {"rows": 0, "book_id": 0, "isbn": 0, "title": 0}
    for row in reader:
        counts["rows"] += 1
        try:
            rating = int(float(row.get("My Rating") or 0))
        except ValueError:
            rating = 0
        shelf = (row.get("Exclusive Shelf") or "").strip()
        idx, how = match_row(cat, row)
        if idx is None:
            if rating > 0 or shelf == "read":
                unmatched.append({"title": row.get("Title", ""), "author": row.get("Author", ""),
                                  "year": row.get("Original Publication Year") or row.get("Year Published")})
            continue
        counts[how] += 1
        if 1 <= rating <= 5:
            prev = rated.get(idx)
            if prev is None or rating > prev["rating"]:  # duplicate editions -> keep the max
                rated[idx] = {"idx": idx, "rating": rating, "match": how}
        elif shelf == "read" or shelf == "currently-reading":
            read_unrated.add(idx)
        elif shelf == "to-read":
            to_read.add(idx)
    read_unrated -= set(rated)
    to_read -= set(rated) | read_unrated
    matched = counts["book_id"] + counts["isbn"] + counts["title"]
    return {
        "rated": list(rated.values()), "read_unrated": sorted(read_unrated), "to_read": sorted(to_read),
        "unmatched": unmatched,
        "stats": {**counts, "matched": matched, "unmatched_rated_or_read": len(unmatched),
                  "match_rate": round(matched / counts["rows"], 4) if counts["rows"] else 0.0},
    }
