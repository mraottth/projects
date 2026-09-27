"""FastAPI app: search, CSV import, recommendations, book detail. Stateless; ratings live in the browser.

Serves the built frontend (frontend/dist) at / when present.
"""

import hashlib
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

import numpy as np
import orjson
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from goodrec.api.catalog import Catalog
from goodrec.api.matching import parse_export
from goodrec.api.schemas import FilterSpec, RecommendRequest
from goodrec.config import ARTIFACTS_DIR, ROOT
from goodrec.core.artifacts import load_artifacts
from goodrec.core.scoring import Filters, Params, RawScores, UserInput, next_in_series, raw_scores, recommend, \
    similar_books, filter_mask, zscore
from goodrec.core.similar_readers import similar_readers, user_genre_share

MAX_UPLOAD = 10 * 1024 * 1024
MIN_RATINGS_FOR_READERS = 5
state: dict = {}


class RawCache:
    """Tiny LRU of per-user model scores so filter/tab changes don't re-score."""

    def __init__(self, size: int = 256):
        self.size, self.d = size, OrderedDict()

    def get(self, key):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        return None

    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        while len(self.d) > self.size:
            self.d.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.time()
    art = load_artifacts(ARTIFACTS_DIR)
    state.update(art=art, cat=Catalog(art.db_path), params=Params.from_config(), cache=RawCache(),
                 genre_idx={g: i for i, g in enumerate(art.meta.genre_names)},
                 starter=orjson.loads((ARTIFACTS_DIR / "starter_shelf.json").read_bytes()))
    print(f"artifacts loaded in {time.time() - t:.1f}s: {art.meta.n:,} books")
    yield
    state.clear()


app = FastAPI(title="Goodreads Recommender", lifespan=lifespan)


# ------------------------------------------------------------------ helpers

def _to_idx(ids) -> list[int]:
    idx_of = state["cat"].idx_of
    return [idx_of[i] for i in ids if i in idx_of]


def _user(req: RecommendRequest) -> UserInput:
    idx_of = state["cat"].idx_of
    ratings = {idx_of[r.id]: r.rating for r in req.ratings if r.id in idx_of}
    return UserInput(ratings=ratings, read=set(_to_idx(req.read)), dismissed=set(_to_idx(req.dismissed)))


def _filters(f: FilterSpec) -> Filters:
    gi = state["genre_idx"]
    return Filters(
        genres=[gi[g] for g in f.genres if g in gi], authors_include=f.authors_include,
        authors_exclude=f.authors_exclude, year_min=f.year_min, year_max=f.year_max,
        min_avg_rating=f.min_avg_rating, min_ratings_count=f.min_ratings_count,
        max_ratings_count=f.max_ratings_count, text=f.text, include_children=f.include_children,
        include_comics=f.include_comics, include_series_continuations=f.include_series_continuations,
    )


def _raw(user: UserInput) -> RawScores:
    key = hashlib.sha1(orjson.dumps([sorted(user.ratings.items()), sorted(user.read)])).hexdigest()
    raw = state["cache"].get(key)
    if raw is None:
        raw = raw_scores(state["art"], user, state["params"])
        state["cache"].put(key, raw)
    return raw


def _decorate(idxs, extra: list[dict] | None = None) -> list[dict]:
    books = state["cat"].books(idxs)
    if extra:
        for b, e in zip(books, extra):
            b.update(e)
    return books


# ------------------------------------------------------------------ routes

@app.get("/api/health")
def health():
    art = state["art"]
    return {"ok": True, "books": art.meta.n, "neighbor_users": 0 if art.user_factors is None else len(art.user_factors)}


@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(10, ge=1, le=50)):
    return state["cat"].search(q, limit)


@app.get("/api/authors")
def authors(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(10, ge=1, le=50)):
    return state["cat"].search_authors(q, limit)


@app.get("/api/authors/names")
def author_names(ids: list[int] = Query(..., max_length=100)):
    return {str(k): v for k, v in state["cat"].author_names(ids).items()}


@app.get("/api/genres")
def genres():
    art = state["art"]
    counts = np.bincount(art.meta.parent_genre[art.meta.parent_genre >= 0], minlength=len(art.meta.genre_names))
    return [{"name": g, "books": int(c)} for g, c in zip(art.meta.genre_names, counts)]


@app.get("/api/starter")
def starter():
    return _decorate(state["starter"])


@app.get("/api/books/{work_id}")
def book(work_id: int):
    cat = state["cat"]
    idx = cat.idx_of.get(work_id)
    if idx is None:
        raise HTTPException(404, "book not in catalog")
    detail = cat.detail(idx)
    detail["similar"] = _decorate(similar_books(state["art"], idx, 20))
    return detail


@app.post("/api/import")
async def import_csv(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (10 MB max)")
    try:
        res = parse_export(state["cat"], raw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    id_of = state["cat"].id_of
    rated_books = {b["id"]: b for b in _decorate([r["idx"] for r in res["rated"]])}
    return {
        "rated": [{**rated_books[id_of[r["idx"]]], "rating": r["rating"], "match": r["match"]}
                  for r in res["rated"] if id_of[r["idx"]] in rated_books],
        "read_unrated": [id_of[i] for i in res["read_unrated"]],
        "to_read": [id_of[i] for i in res["to_read"]],
        "unmatched": res["unmatched"][:500],
        "stats": res["stats"],
    }


@app.post("/api/recommend")
def recommend_route(req: RecommendRequest):
    t = time.time()
    art, p = state["art"], state["params"]
    user, f = _user(req), _filters(req.filters)
    raw = _raw(user)
    res = recommend(art, user, f, p, limit=req.limit, offset=req.offset, raw=raw)

    id_of = state["cat"].id_of
    because_idx = sorted({i for b in res["because"] for i in b})
    because_titles = {b["id"]: b["title"] for b in state["cat"].books(because_idx)}
    for_you = _decorate(res["items"], [
        {"score": round(float(s), 3), "source": {1: "similar-books", 2: "taste-model", 3: "both"}.get(int(src), "popular"),
         "because": [{"id": id_of[i], "title": because_titles.get(id_of[i], "")} for i in b],
         "next_in_series": nx}
        for s, src, b, nx in zip(res["scores"], res["source"], res["because"], res["next_in_series"])])

    # To-read picks: the user's own to-read shelf, ordered by the same blended model scores.
    to_read = [i for i in _to_idx(req.to_read) if i not in user.seen]
    to_read_picks = []
    if to_read and raw.n:
        tr = np.asarray(to_read)
        a = res["alpha"]
        s = a * zscore(raw.s_als[tr]) + (1 - a) * zscore(raw.s_ii[tr]) if raw.u is not None else zscore(raw.s_ii[tr])
        to_read_picks = _decorate(tr[np.argsort(-s)][:req.limit])

    readers = None
    if len(user.ratings) >= MIN_RATINGS_FOR_READERS and raw.u is not None:
        mask = filter_mask(art, user, f, allow=next_in_series(art, user))
        sr = similar_readers(art, raw.u, mask, limit=req.limit)
        if sr:
            names = art.meta.genre_names
            you = user_genre_share(art, user.ratings)
            order = np.argsort(-(you + sr["genre_share"]))[:10]
            readers = {
                "n_neighbors": sr["n_neighbors"],
                "popular": _decorate([x["idx"] for x in sr["popular"]], [{"pct_read": x["pct_read"]} for x in sr["popular"]]),
                "top_rated": _decorate([x["idx"] for x in sr["top_rated"]],
                                       [{"neighbor_avg": x["neighbor_avg"], "neighbor_raters": x["n_raters"]}
                                        for x in sr["top_rated"]]),
                "genres": [{"genre": names[g], "you": round(float(you[g]), 3),
                            "similar_readers": round(float(sr["genre_share"][g]), 3)} for g in order],
            }
    return {
        "for_you": for_you, "to_read_picks": to_read_picks, "similar_readers": readers,
        "meta": {"n_ratings": len(user.ratings), "alpha": round(res["alpha"], 3),
                 "total_candidates": res["total"], "min_ratings_for_readers": MIN_RATINGS_FOR_READERS,
                 "ms": round((time.time() - t) * 1000, 1)},
    }


# ------------------------------------------------------------------ frontend

DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = DIST / path
        if path and target.is_file() and DIST in target.resolve().parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
