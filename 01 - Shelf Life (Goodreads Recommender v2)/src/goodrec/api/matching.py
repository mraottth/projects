"""Match a Goodreads library export CSV to catalog works.

Order (extends the old app's Book Id -> title matching):
  1. Book Id -> edition -> work (covers every edition of every catalog work)
  2. ISBN13, then ISBN (export wraps them as ="..."), including ISBN10 -> 13 conversion
  3. normalized title + an author last name that must match (ties -> most-rated work)
Shelves: My Rating > 0 -> rating; Exclusive Shelf 'read' with rating 0 -> read-unrated;
'to-read' -> to_read (re-ranked separately; never used as a signal).
"""

import csv
import datetime as dt
import html
import io
import re

from goodrec.api.catalog import Catalog
from goodrec.core.textnorm import author_key, clean_isbn, isbn10_to_13, titlekey

REQUIRED = {"Book Id", "Title", "My Rating"}
MAX_REVIEW = 1500   # characters kept per review (the chat assistant reads them)


def clean_review(text: str | None) -> str:
    """Goodreads exports reviews as HTML fragments: tags -> plain text, trimmed."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>|</p>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) <= MAX_REVIEW else text[:MAX_REVIEW].rsplit(" ", 1)[0] + "…"


def export_date(text: str | None) -> str | None:
    """A Goodreads export date ("2023/05/14") as ISO "2023-05-14"; None if blank, malformed or in the future."""
    try:
        d = dt.datetime.strptime((text or "").strip(), "%Y/%m/%d").date()
    except ValueError:
        return None
    return d.isoformat() if d <= dt.date.today() else None


def read_date(row: dict) -> str | None:
    """When the book was read ("Date Read"), else when it was shelved ("Date Added"): orders ratings for recency."""
    return export_date(row.get("Date Read")) or export_date(row.get("Date Added"))


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
    unmatched: list[dict] = []           # rated/read books not in the catalog (mostly published after 2017)
    unmatched_to_read: list[dict] = []
    reviews: dict[int, str] = {}
    counts = {"rows": 0, "book_id": 0, "isbn": 0, "title": 0}
    for row in reader:
        counts["rows"] += 1
        try:
            rating = int(float(row.get("My Rating") or 0))
        except ValueError:
            rating = 0
        shelf = (row.get("Exclusive Shelf") or "").strip()
        review = clean_review(row.get("My Review"))
        idx, how = match_row(cat, row)
        if idx is None:
            entry = {"title": row.get("Title", ""), "author": row.get("Author", ""),
                     "year": row.get("Original Publication Year") or row.get("Year Published"),
                     "rating": rating if 1 <= rating <= 5 else None, "shelf": shelf} | ({"review": review} if review else {})
            if rating > 0 or shelf == "read":
                unmatched.append(entry)
            elif shelf == "to-read":
                unmatched_to_read.append(entry)
            continue
        counts[how] += 1
        if review:
            reviews[idx] = review
        if 1 <= rating <= 5:
            prev = rated.get(idx)
            date = read_date(row)
            if prev is None or rating > prev["rating"]:  # duplicate editions -> keep the max rating, latest date
                rated[idx] = {"idx": idx, "rating": rating, "match": how,
                              "date": max(filter(None, (date, prev and prev["date"])), default=None)}
            elif date and (prev["date"] is None or date > prev["date"]):
                prev["date"] = date
        elif shelf == "read" or shelf == "currently-reading":
            read_unrated.add(idx)
        elif shelf == "to-read":
            to_read.add(idx)
    read_unrated -= set(rated)
    to_read -= set(rated) | read_unrated
    matched = counts["book_id"] + counts["isbn"] + counts["title"]
    return {
        "rated": list(rated.values()), "read_unrated": sorted(read_unrated), "to_read": sorted(to_read),
        "unmatched": unmatched, "unmatched_to_read": unmatched_to_read, "reviews": reviews,
        "stats": {**counts, "matched": matched, "unmatched_rated_or_read": len(unmatched),
                  "match_rate": round(matched / counts["rows"], 4) if counts["rows"] else 0.0},
    }
