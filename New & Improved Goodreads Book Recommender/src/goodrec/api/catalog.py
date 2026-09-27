"""Read-only access to artifacts/catalog.db: book payloads, typeahead search, id mapping.

Public ids are Goodreads work_ids (stable across artifact rebuilds); work_idx is internal.
"""

import math
import re
import sqlite3
import threading

import orjson

from goodrec.core.textnorm import ascii_fold, titlekey

BOOK_COLS = ("work_idx", "work_id", "title", "base_title", "author", "author_id", "year", "avg_rating",
             "ratings_count", "cover_url", "isbn", "url", "series_name", "series_pos", "parent_genre", "tags", "description")
_TOKEN = re.compile(r"[a-z0-9]+")


def _clean_description(text: str | None) -> str | None:
    """Description with whitespace collapsed; marks text that was truncated at ingest (1200 chars)."""
    if not text:
        return None
    truncated = len(text) >= 1200
    text = " ".join(text.split())
    return text[:text.rfind(" ")].rstrip(",.;:") + "…" if truncated and " " in text else text


def _cover_large(url: str | None) -> str | None:
    # Goodreads 2017 URLs end in "<id>m/<book>.jpg" (small); "l" is the large rendition.
    return re.sub(r"(\d+)m/(\d+\.jpg)$", r"\1l/\2", url) if url else None


class Catalog:
    def __init__(self, db_path):
        self._local = threading.local()
        self.db_path = str(db_path)
        con = self._con()
        rows = con.execute("SELECT work_id, work_idx FROM works").fetchall()
        self.idx_of = {wid: idx for wid, idx in rows}
        self.id_of = {idx: wid for wid, idx in rows}

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
            self._local.con = con
        return con

    # -------------------------------------------------------------- payloads

    def books(self, idxs) -> list[dict]:
        idxs = [int(i) for i in idxs]
        if not idxs:
            return []
        q = f"SELECT {','.join(BOOK_COLS)} FROM works WHERE work_idx IN ({','.join('?' * len(idxs))})"
        by_idx = {r[0]: r for r in self._con().execute(q, idxs)}
        return [self._payload(by_idx[i]) for i in idxs if i in by_idx]

    def _payload(self, r) -> dict:
        d = dict(zip(BOOK_COLS, r))
        return {
            "id": d["work_id"], "title": d["base_title"] or d["title"], "full_title": d["title"],
            "author": d["author"], "author_id": d["author_id"], "year": d["year"] or None,
            "avg_rating": d["avg_rating"], "ratings_count": d["ratings_count"],
            "cover_url": _cover_large(d["cover_url"]), "cover_url_small": d["cover_url"], "isbn": d["isbn"],
            "url": d["url"], "series": d["series_name"],
            "series_pos": d["series_pos"] if d["series_pos"] is None else float(d["series_pos"]),
            "genre": d["parent_genre"], "tags": orjson.loads(d["tags"] or "[]"),
            "description": _clean_description(d["description"]),
        }

    def detail(self, idx: int) -> dict | None:
        rows = self.books([idx])
        if not rows:
            return None
        desc = self._con().execute("SELECT description FROM works WHERE work_idx=?", (idx,)).fetchone()[0]
        return {**rows[0], "description": desc}

    # -------------------------------------------------------------- search

    @staticmethod
    def _fts_query(q: str) -> str | None:
        tokens = _TOKEN.findall(ascii_fold(q).lower())
        return " ".join(f'"{t}"*' for t in tokens) if tokens else None

    def search(self, q: str, limit: int = 10) -> list[dict]:
        """Prefix search over title/author/series; relevance first, then popularity."""
        fts = self._fts_query(q)
        if not fts:
            return []
        rows = self._con().execute(
            """SELECT w.work_idx, w.base_title, w.author, w.ratings_count, bm25(works_fts, 8.0, 3.0, 2.0)
               FROM works_fts JOIN works w ON w.work_idx = works_fts.rowid
               WHERE works_fts MATCH ? ORDER BY bm25(works_fts, 8.0, 3.0, 2.0) LIMIT 200""", (fts,)).fetchall()
        # Compare with titlekey() normalization so "hunger games" fully matches "The Hunger Games".
        ql = titlekey(q)
        qtok = _TOKEN.findall(ql)

        def score(r):
            _, title, author, cnt, bm = r
            t = titlekey(title or "")
            ttok = _TOKEN.findall(t)
            s = -bm + 2.5 * math.log10(max(cnt or 1, 1))
            if t == ql:
                s += 12
            elif t.startswith(ql):
                s += 8
            if all(any(w.startswith(tk) for w in ttok) for tk in qtok):
                s += 3  # every query token matches the title (not just the author)
            return s

        best = sorted(rows, key=score, reverse=True)[:limit]
        return self.books([r[0] for r in best])

    def search_authors(self, q: str, limit: int = 10) -> list[dict]:
        fts = self._fts_query(q)
        if not fts:
            return []
        rows = self._con().execute(
            """SELECT a.author_id, a.name, a.total_ratings, a.n_works FROM authors_fts
               JOIN authors a ON a.author_id = authors_fts.rowid WHERE authors_fts MATCH ?
               ORDER BY a.total_ratings DESC LIMIT ?""", (fts, limit)).fetchall()
        return [{"id": r[0], "name": r[1], "ratings_count": r[2], "n_books": r[3]} for r in rows]

    def author_names(self, ids) -> dict[int, str]:
        ids = [int(i) for i in ids]
        if not ids:
            return {}
        q = f"SELECT author_id, name FROM authors WHERE author_id IN ({','.join('?' * len(ids))})"
        return dict(self._con().execute(q, ids).fetchall())

    # -------------------------------------------------------------- import matching lookups

    def by_edition(self, book_id: int) -> int | None:
        r = self._con().execute("SELECT work_idx FROM editions WHERE book_id=?", (book_id,)).fetchone()
        return r[0] if r else None

    def by_isbn(self, isbn: str) -> int | None:
        r = self._con().execute("SELECT work_idx FROM isbns WHERE isbn=?", (isbn,)).fetchone()
        return r[0] if r else None

    def by_titlekey(self, tkey: str, akeys: set[str]) -> int | None:
        """Title match that also requires an author match. Titles alone are too ambiguous: a post-2017
        book missing from the catalog would otherwise latch onto an older book with the same title."""
        akeys = {a for a in akeys if a}
        if not tkey or not akeys:
            return None
        q = (f"SELECT work_idx FROM titlekeys WHERE titlekey=? AND author_key IN ({','.join('?' * len(akeys))}) "
             "ORDER BY ratings_count DESC LIMIT 1")
        r = self._con().execute(q, (tkey, *akeys)).fetchone()
        return r[0] if r else None
