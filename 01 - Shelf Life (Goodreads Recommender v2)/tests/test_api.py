"""End-to-end API tests against the built artifacts (skipped if `make artifacts` hasn't run)."""

from pathlib import Path

import pytest

from goodrec.config import ARTIFACTS_DIR

pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "manifest.json").exists(), reason="artifacts not built")
FIXTURE = Path(__file__).parent / "fixtures" / "goodreads_library_export.csv"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from goodrec.api.main import app

    with TestClient(app) as c:
        yield c


def _first(client, q):
    return client.get("/api/search", params={"q": q}).json()[0]


def _genres(books):
    return [b["genre"] for b in books]


def test_search_prefers_title_matches(client):
    assert _first(client, "sapiens")["title"].startswith("Sapiens")
    assert _first(client, "harry potter sorc")["title"].startswith("Harry Potter and the Sorcerer")
    assert _first(client, "gone girl")["author"] == "Gillian Flynn"
    # Box sets rank below the books in them.
    for q, expect in [("mistborn", "The Final Empire"), ("harry potter", "Harry Potter and the Sorcerer"),
                      ("hunger games", "The Hunger Games")]:
        assert _first(client, q)["title"].startswith(expect), q


def test_import_fixture(client):
    with open(FIXTURE, "rb") as f:
        res = client.post("/api/import", files={"file": ("export.csv", f, "text/csv")}).json()
    stats = res["stats"]
    assert stats["rows"] > 400
    matched = len(res["rated"]) + len(res["read_unrated"])
    # The dataset ends in 2017, so measure the match rate on books published before 2018.
    pre2018_unmatched = [u for u in res["unmatched"] if not (u["year"] or "").isdigit() or int(u["year"]) < 2018]
    assert matched / (matched + len(pre2018_unmatched)) >= 0.85
    assert all(1 <= r["rating"] <= 5 for r in res["rated"])


def test_import_title_matches_require_author():
    import csv

    from goodrec.api.main import state
    from goodrec.api.matching import match_row
    from goodrec.core.textnorm import author_key

    cat = state["cat"]
    rows = list(csv.DictReader(open(FIXTURE, encoding="utf-8-sig")))
    anarchy = next(r for r in rows if r["Title"].startswith("The Anarchy"))
    assert match_row(cat, anarchy) == (None, None)  # 2019 book: not "Anarchy" by Jaymin Eve
    for r in rows:
        idx, how = match_row(cat, r)
        if how == "title":
            keys = {author_key(r["Author"])} | {author_key(a.strip()) for a in r["Additional Authors"].split(",")}
            assert author_key(cat.books([idx])[0]["author"]) in keys, r["Title"]


def test_import_rejects_non_export(client):
    r = client.post("/api/import", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400


@pytest.mark.parametrize("queries,genre", [
    (["mistborn final empire", "name of the wind"], "Fantasy"),
    (["gone girl", "girl on the train", "dark places"],  # the UI's "suspense" genre family
     {"Mystery & Crime", "Thriller & Suspense", "Horror", "True Crime"}),
    (["sapiens", "guns germs steel", "thinking fast slow"], None),
])
def test_personas(client, queries, genre):
    ratings = [{"id": _first(client, q)["id"], "rating": 5} for q in queries]
    res = client.post("/api/recommend", json={"ratings": ratings, "limit": 20}).json()
    top = res["for_you"]
    assert len(top) == 20
    rated = {r["id"] for r in ratings}
    assert not rated & {b["id"] for b in top}
    genres = _genres(top)
    if genre is None:  # nonfiction persona: mostly nonfiction parents
        fiction = {"Fantasy", "Romance", "Young Adult", "Mystery & Crime", "Thriller & Suspense",
                   "Science Fiction", "Paranormal & Urban Fantasy", "Contemporary Fiction"}
        assert sum(g not in fiction for g in genres) >= 12
    else:
        want = {genre} if isinstance(genre, str) else genre
        assert sum(g in want for g in genres) >= 10
    assert all(b["because"] for b in top)


def test_filters(client):
    ratings = [{"id": _first(client, q)["id"], "rating": r} for q, r in
               [("hunger games", 5), ("divergent", 4), ("the maze runner", 4), ("the giver", 3), ("matched", 2)]]
    base = {"ratings": ratings, "limit": 30}
    filters = {"year_min": 2010, "year_max": 2015, "min_avg_rating": 4.0, "min_ratings_count": 50000,
               "genres": ["Science Fiction", "Young Adult"]}
    res = client.post("/api/recommend", json={**base, "filters": filters}).json()
    assert len(res["for_you"]) == 30  # heavily filtered request still fills the page
    for b in res["for_you"]:
        assert 2010 <= b["year"] <= 2015 and b["avg_rating"] >= 4.0 and b["ratings_count"] >= 50000
        assert b["genre"] in filters["genres"]
    # author include / exclude and text search
    collins = _first(client, "suzanne collins")["author_id"]
    only = client.post("/api/recommend", json={**base, "filters": {"authors_include": [collins]}}).json()
    assert only["for_you"] and all(b["author_id"] == collins for b in only["for_you"])
    excl = client.post("/api/recommend", json={**base, "filters": {"authors_exclude": [collins]}}).json()
    assert all(b["author_id"] != collins for b in excl["for_you"])
    txt = client.post("/api/recommend", json={**base, "filters": {"text": "dystopian"}}).json()
    assert txt["for_you"] and all("Dystopian" in b["tags"] or "dystopian" in b["full_title"].lower()
                                  for b in txt["for_you"])
    # similar readers appear with >= 5 ratings and respect filters
    sr = res["similar_readers"]
    assert sr and sr["popular"] and sr["top_rated"] and sr["genres"]
    assert all(b["genre"] in filters["genres"] for b in sr["popular"] + sr["top_rated"])


def test_series_continuations(client):
    hg = _first(client, "mistborn final empire")
    res = client.post("/api/recommend", json={"ratings": [{"id": hg["id"], "rating": 5}], "limit": 40}).json()
    nxt = [b for b in res["for_you"] if b["next_in_series"]]
    assert nxt and nxt[0]["series"] == hg["series"] and nxt[0]["series_pos"] == 2.0
    later = [b for b in res["for_you"] if b["series_pos"] and b["series_pos"] > 1 and not b["next_in_series"]]
    assert not later


def test_book_detail_and_starter(client):
    gg = _first(client, "gone girl")
    d = client.get(f"/api/books/{gg['id']}").json()
    assert d["description"] and len(d["similar"]) == 20
    starter = client.get("/api/starter").json()
    assert len(starter) >= 40 and len({b["genre"] for b in starter}) >= 8
    from goodrec.api.main import state
    wall = client.get("/api/home_wall").json()
    assert len(wall) == 48 and len({b["id"] for b in wall}) == 48      # every config entry resolved, no repeats
    assert not any(state["art"].meta.is_ya[state["cat"].idx_of[b["id"]]] for b in wall)


def test_predicted_ratings(client):
    ratings = [{"id": _first(client, q)["id"], "rating": r} for q, r in
               [("mistborn final empire", 5), ("name of the wind", 5), ("twilight", 1), ("hunger games", 4),
                ("way of kings", 5)]]
    res = client.post("/api/recommend", json={"ratings": ratings, "limit": 20}).json()
    lists = [res["for_you"], res["similar_readers"]["popular"], res["similar_readers"]["top_rated"]]
    for books in lists:
        assert books and all(1.0 <= b["predicted_rating"] <= 5.0 for b in books)
    # A Sanderson fan should get a high predicted rating for Words of Radiance, a low one for New Moon.
    from goodrec.api.main import state
    from goodrec.core.scoring import UserInput, predict_ratings
    idx = state["cat"].idx_of
    user = UserInput(ratings={idx[r["id"]]: r["rating"] for r in ratings})
    wor, nm = _first(client, "words of radiance")["id"], _first(client, "new moon")["id"]
    p_wor, p_nm = predict_ratings(state["art"], user, [idx[wor], idx[nm]])
    assert p_wor > 4.3 and p_nm < 3.5 and p_wor - p_nm > 1.0


def _fantasy_reader(client):
    return [{"id": _first(client, q)["id"], "rating": r} for q, r in
            [("mistborn final empire", 5), ("name of the wind", 5), ("twilight", 1), ("hunger games", 4),
             ("way of kings", 5), ("the hobbit", 4)]]


def test_sort_by_predicted(client):
    ratings = _fantasy_reader(client)
    match = client.post("/api/recommend", json={"ratings": ratings, "limit": 40}).json()
    pred = client.post("/api/recommend", json={"ratings": ratings, "limit": 40, "sort": "predicted"}).json()
    for books in (pred["for_you"], pred["similar_readers"]["popular"], pred["similar_readers"]["top_rated"]):
        p = [b["predicted_rating"] for b in books]
        assert p == sorted(p, reverse=True)
    # Sorting by predicted draws on the whole candidate pool, so its top is at least as high as match order's.
    assert pred["for_you"][0]["predicted_rating"] >= max(b["predicted_rating"] for b in match["for_you"][:40])
    assert pred["meta"]["sort"] == "predicted"
    assert client.post("/api/recommend", json={"ratings": ratings, "sort": "bogus"}).status_code == 422


def test_readers_avg_on_every_book(client):
    res = client.post("/api/recommend", json={"ratings": _fantasy_reader(client), "limit": 20}).json()
    books = res["for_you"] + res["similar_readers"]["popular"] + res["similar_readers"]["top_rated"]
    assert all("readers_n" in b for b in books)
    for b in books:
        if b["readers_n"]:
            assert 1.0 <= b["readers_avg"] <= 5.0
        else:
            assert b["readers_avg"] is None
    assert all(b["readers_n"] >= 5 for b in res["similar_readers"]["top_rated"])  # top-rated requires raters
    # Fewer than 5 ratings: no neighbor set, so no readers average.
    few = client.post("/api/recommend", json={"ratings": _fantasy_reader(client)[:2]}).json()
    assert few["similar_readers"] is None and "readers_avg" not in few["for_you"][0]


def test_browse(client):
    sand = _first(client, "brandon sanderson")["author_id"]
    res = client.post("/api/browse", json={"filters": {"authors_include": [sand], "include_series_continuations": True},
                                          "sort": "newest", "limit": 50}).json()
    books = res["books"]
    assert res["total"] >= 15 and all(b["author_id"] == sand for b in books)
    years = [b["year"] for b in books if b["year"]]
    assert years == sorted(years, reverse=True)
    assert any(b["series_pos"] and b["series_pos"] > 1 for b in books)   # later volumes shown when asked
    f = {"genres": ["Science Fiction"], "year_min": 1960, "year_max": 1990, "min_avg_rating": 4.0,
         "min_ratings_count": 10000}
    top = client.post("/api/browse", json={"filters": f, "sort": "rating", "limit": 30}).json()["books"]
    assert top and all(b["genre"] == "Science Fiction" and 1960 <= b["year"] <= 1990 and b["avg_rating"] >= 4.0
                       and b["ratings_count"] >= 10000 for b in top)
    assert "predicted_rating" not in top[0]
    with_preds = client.post("/api/browse", json={"filters": f, "ratings": _fantasy_reader(client)}).json()["books"]
    assert all(1 <= b["predicted_rating"] <= 5 for b in with_preds)


def test_books_batch(client):
    ids = [_first(client, q)["id"] for q in ("dune", "gone girl", "sapiens")]
    res = client.post("/api/books/batch", json={"ids": ids + [999999999]}).json()
    assert [b["id"] for b in res] == ids and all(b["avg_rating"] and b["url"] for b in res)


def test_insights(client):
    tough = [{"id": r["id"], "rating": 2} for r in _fantasy_reader(client)]
    kind = [{"id": r["id"], "rating": 5} for r in _fantasy_reader(client)]
    t = client.post("/api/insights", json={"ratings": tough}).json()
    k = client.post("/api/insights", json={"ratings": kind}).json()
    assert t["harshness"]["bias"] < 0 < k["harshness"]["bias"]
    assert t["harshness"]["harsher_than"] > 0.9 and k["harshness"]["harsher_than"] < 0.2
    assert t["books"]["n"] == 6 and 0 < t["books"]["percentile"] < 1
    many = [{"id": b["id"], "rating": 4} for b in client.get("/api/starter").json()]
    assert client.post("/api/insights", json={"ratings": many}).json()["books"]["percentile"] > t["books"]["percentile"]
    g = {x["genre"]: x for x in t["genres"]}
    assert g["Fantasy"]["books"] >= 3 and g["Fantasy"]["your_avg"] == 2.0 and g["Fantasy"]["readers_genre_avg"] > 3.5
    few = client.post("/api/insights", json={"ratings": tough[:2]}).json()
    assert few["harshness"] is None


def test_tag_filter_and_accent_folded_search(client):
    base = {"ratings": _fantasy_reader(client), "limit": 30}
    res = client.post("/api/recommend", json={**base, "filters": {"tags": ["Epic Fantasy"]}}).json()["for_you"]
    assert res and all("Epic Fantasy" in b["tags"] for b in res)
    both = client.post("/api/recommend", json={**base, "filters": {"tags": ["Epic Fantasy", "Dragons"]}}).json()["for_you"]
    assert all({"Epic Fantasy", "Dragons"} <= set(b["tags"]) for b in both)   # tags narrow (all-of)
    assert client.post("/api/recommend", json={**base, "filters": {"tags": ["No Such Tag"]}}).json()["for_you"] == []
    br = client.post("/api/browse", json={"filters": {"text": "bronte jane eyre"}}).json()["books"]
    assert br and br[0]["author"].startswith("Charlotte Bront")                  # "bronte" finds "Brontë"


def test_tags_endpoint_and_genre_mix(client):
    tags = client.get("/api/tags").json()
    assert len(tags) > 100 and tags[0]["books"] >= tags[-1]["books"]
    res = client.post("/api/insights", json={"ratings": _fantasy_reader(client)}).json()
    mix = res["genre_mix"]
    assert mix and any(r["genre"] == "Fantasy" and r["you"] > 0.3 for r in mix)
    assert client.post("/api/insights", json={"ratings": _fantasy_reader(client)[:2]}).json()["genre_mix"] == []


def test_book_personal(client):
    wor = _first(client, "words of radiance")["id"]
    res = client.post(f"/api/books/{wor}/personal", json={"ratings": _fantasy_reader(client)}).json()
    assert 4.0 < res["predicted_rating"] <= 5.0 and res["readers_n"] > 0 and res["readers_avg"] >= 4.0
    few = client.post(f"/api/books/{wor}/personal", json={"ratings": _fantasy_reader(client)[:2]}).json()
    assert few["predicted_rating"] is not None and few["readers_n"] is None       # readers need >= 5 ratings
    assert client.post(f"/api/books/{wor}/personal", json={}).json()["predicted_rating"] is None
    assert client.post("/api/books/999999999/personal", json={}).status_code == 404


def test_ranks_are_stable_under_filters(client):
    base = {"ratings": _fantasy_reader(client), "limit": 200}
    for sort in ("match", "predicted"):
        full = client.post("/api/recommend", json={**base, "sort": sort}).json()
        rank_of = {b["id"]: b["rank"] for b in full["for_you"]}
        assert [b["rank"] for b in full["for_you"]] == list(range(1, 201))          # unfiltered = 1..n
        filt = client.post("/api/recommend", json={**base, "sort": sort, "filters": {"genres": ["Science Fiction"]}}).json()
        ranks = [b["rank"] for b in filt["for_you"]]
        assert ranks == sorted(ranks) and ranks[0] >= 1                                # original numbers, in order
        assert any(r > i + 1 for i, r in enumerate(ranks))                            # gaps where other books were
        for b in filt["for_you"]:
            if b["id"] in rank_of:
                assert b["rank"] == rank_of[b["id"]], (sort, b["title"])
        # readers-like-you tabs keep their numbers too
        for tab in ("popular", "top_rated"):
            full_t = {b["id"]: b["rank"] for b in full["similar_readers"][tab]}
            for b in filt["similar_readers"][tab]:
                assert b["genre"] == "Science Fiction"
                if b["id"] in full_t:
                    assert b["rank"] == full_t[b["id"]]
    match = client.post("/api/recommend", json={**base, "limit": 50}).json()["for_you"]
    pred = client.post("/api/recommend", json={**base, "limit": 50, "sort": "predicted"}).json()["for_you"]
    assert [b["id"] for b in match] != [b["id"] for b in pred]                        # separate rankings


def test_vectorized_predictions_match_reference():
    """predict_ratings (sparse, vectorized) == the original per-pair loop, for the item-kNN predictor and for the
    hybrid (the same residual on a factorization baseline, D-052) when production has a factorization model."""
    import dataclasses

    import numpy as np

    from goodrec.api.main import state
    from goodrec.core.scoring import UserInput, predict_ratings

    knn = dataclasses.replace(state["art"], rating_mode="knn", rating_mf=None)
    m = knn.meta
    mf = state["art"].rating_mf

    def reference(user, items, shrink=0.5, user_shrink=5.0, hybrid=False):
        art = knn
        ri = np.fromiter(user.ratings, dtype=np.int64)
        rv = np.fromiter(user.ratings.values(), dtype=np.float64)
        b_u = (rv - m.bayes[ri]).sum() / (len(ri) + user_shrink)
        if hybrid:
            base_r = mf.predict(user.ratings, ri, clip=False, loo=False)
            base = dict(zip(items.tolist(), mf.predict(user.ratings, items, clip=False).tolist()))
        else:
            base_r = m.bayes[ri] + b_u
            base = {int(j): m.bayes[j] + b_u for j in items}
        resid = dict(zip(ri.tolist(), (rv - base_r).tolist()))
        rated = set(ri.tolist())
        out = []
        for j in items:
            pairs = {int(i): float(s) for i, s in zip(art.nbr_idx[j], art.nbr_sim[j]) if i in rated}
            for i in ri:
                hit = np.flatnonzero(art.nbr_idx[i] == j)
                if len(hit):
                    pairs[int(i)] = max(pairs.get(int(i), 0.0), float(art.nbr_sim[i][hit[0]]))
            num = sum(s * resid[i] for i, s in pairs.items())
            den = sum(abs(s) for s in pairs.values())
            out.append(np.clip(base[int(j)] + num / (den + shrink), 1, 5))
        return np.array(out)

    rng = np.random.default_rng(1)
    for n in (1, 5, 40):
        user = UserInput(ratings={int(i): int(rng.integers(1, 6)) for i in rng.choice(3000, n, replace=False)})
        items = rng.choice(m.n, 150, replace=False)
        assert np.allclose(predict_ratings(knn, user, items), reference(user, items), atol=1e-5)
        if mf is not None:
            hy = dataclasses.replace(knn, rating_mode="hybrid", rating_mf=mf)
            assert np.allclose(predict_ratings(hy, user, items), reference(user, items, hybrid=True), atol=1e-5)


def test_prediction_calibration():
    import numpy as np

    from goodrec.api.main import state
    from goodrec.core.scoring import UserInput, apply_calibration, calibration, predict_ratings

    art, prior = state["art"], state["population"]["rating_dist"]
    rng = np.random.default_rng(3)
    # A tough grader: mostly 2-3 stars, but ~15% 5-stars.
    books = rng.choice(20000, 80, replace=False)
    stars = rng.choice([2, 3, 3, 3, 4, 5], size=80, p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.15])
    user = UserInput(ratings={int(b): int(s) for b, s in zip(books, stars)})
    cal = calibration(art, user, prior)
    raw = predict_ratings(art, user, np.arange(art.meta.n))
    shown = apply_calibration(raw, cal)
    order = np.argsort(raw)
    assert np.all(np.diff(shown[order]) >= -1e-6)                  # monotone: never changes an ordering
    assert shown.max() < 5.0 and shown.min() >= 1.0                 # asymptotic tails: nothing pinned at 5
    assert (shown >= 4.5).mean() > 5 * max((raw >= 4.5).mean(), 1e-4)   # a tough grader's 5s show up
    assert (shown >= 4.95).mean() < 0.05                            # ...without piling up at the ceiling
    few = UserInput(ratings=dict(list(user.ratings.items())[:3]))
    assert calibration(art, few, prior) is None                     # too few ratings: raw predictions


def test_predictions_two_decimals(client):
    res = client.post("/api/recommend", json={"ratings": _fantasy_reader(client), "limit": 20}).json()
    vals = [b["predicted_rating"] for b in res["for_you"]]
    assert all(round(v, 2) == v for v in vals) and any(round(v, 1) != v for v in vals)


def test_prediction_floor(client):
    from goodrec.api.main import state
    from goodrec.core.scoring import Filters, Params, UserInput, recommend

    ratings = _fantasy_reader(client)                    # 6 ratings, average 4.0
    avg = sum(r["rating"] for r in ratings) / len(ratings)
    offset = state["params"].pred_floor_offset
    for sort in ("match", "predicted"):
        res = client.post("/api/recommend", json={"ratings": ratings, "limit": 100, "sort": sort}).json()
        assert all(b["predicted_rating"] >= avg - offset - 0.005 for b in res["for_you"]), sort
        # filters can't bring floored books back in as unranked extras
        filt = client.post("/api/recommend", json={"ratings": ratings, "limit": 100, "sort": sort,
                                                   "filters": {"include_series_continuations": True}}).json()
        assert all(b["predicted_rating"] >= avg - offset - 0.005 for b in filt["for_you"])
        # readers-like-you tabs are not floored (visible in match order; the predicted sort puts the highest first)
        if sort == "match":
            assert any(b["predicted_rating"] < avg - offset for b in res["similar_readers"]["popular"])

    # Fewer than 5 ratings: no calibration, so no floor -> identical to the unfloored ranking.
    art, prior = state["art"], state["population"]["rating_dist"]
    idx = state["cat"].idx_of
    few = UserInput(ratings={idx[r["id"]]: r["rating"] for r in ratings[:3]})
    with_floor = recommend(art, few, Filters(), Params.from_config(), limit=40, prior=prior)["items"]
    without = recommend(art, few, Filters(), Params.from_config(pred_floor_offset=None), limit=40)["items"]
    assert list(with_floor) == list(without)


def test_young_adult_hidden_by_default(client):
    from goodrec.api.main import state

    m, idx = state["art"].meta, state["cat"].idx_of
    is_ya = lambda b: bool(m.is_ya[idx[b["id"]]])  # noqa: E731
    assert is_ya(_first(client, "hunger games")) and is_ya(_first(client, "six of crows"))
    assert not is_ya(_first(client, "ender's game")) and not is_ya(_first(client, "the martian"))

    base = {"ratings": _fantasy_reader(client), "limit": 100}
    for sort in ("match", "predicted"):
        off = client.post("/api/recommend", json={**base, "sort": sort}).json()
        on = client.post("/api/recommend", json={**base, "sort": sort, "filters": {"include_ya": True}}).json()
        for books in (off["for_you"], off["similar_readers"]["popular"], off["similar_readers"]["top_rated"]):
            assert not any(is_ya(b) for b in books)
        assert any(is_ya(b) for b in on["for_you"])
        # Including YA re-ranks: consecutive numbers 1..n over the larger universe, not gaps.
        assert [b["rank"] for b in on["for_you"]] == list(range(1, 101))
        assert any(is_ya(b) for b in on["similar_readers"]["popular"])
    # Choosing the Young Adult genre implies including YA books.
    ya = client.post("/api/recommend", json={**base, "filters": {"genres": ["Young Adult"]}}).json()
    assert ya["for_you"] and all(b["genre"] == "Young Adult" for b in ya["for_you"])
    for flag in (False, True):
        browse = client.post("/api/browse", json={"filters": {"include_ya": flag}, "limit": 200}).json()["books"]
        assert any(is_ya(b) for b in browse) == flag


def test_prediction_boost_favors_famous_books(client):
    """The Best-match boost lifts well-known books with high predictions, not obscure ones."""
    from goodrec.api.main import state
    import numpy as np

    from goodrec.core.scoring import Params, UserInput, fame_weight, ranking, raw_scores

    art, idx, prior = state["art"], state["cat"].idx_of, state["population"]["rating_dist"]
    user = UserInput(ratings={idx[r["id"]]: r["rating"] for r in _fantasy_reader(client)})
    fame = fame_weight(art, Params.from_config())
    orders = {}
    for d in (0.0, 1.0):
        p = Params.from_config(delta_pred=d)
        rk = ranking(art, raw_scores(art, user, p), user, p, "match", prior)
        orders[d] = rk["order"][:40]
        preds = rk["preds"]
    assert np.nanmean(preds[orders[1.0]]) > np.nanmean(preds[orders[0.0]])      # higher predicted ratings on top
    new = np.setdiff1d(orders[1.0], orders[0.0])
    assert len(new) and fame[new].mean() > 0.5                                   # and the newcomers are well known


def test_calibration_scales_with_evidence(client):
    """Books with no link to the user's ratings keep their raw prediction; linked books get the stretch."""
    import numpy as np

    from goodrec.api.main import state
    from goodrec.core.scoring import UserInput, apply_calibration, calibration, predict_ratings

    art, prior, idx = state["art"], state["population"]["rating_dist"], state["cat"].idx_of
    user = UserInput(ratings={idx[r["id"]]: r["rating"] for r in _fantasy_reader(client)})
    cal = calibration(art, user, prior)
    items = np.arange(art.meta.n)
    raw, ev = predict_ratings(art, user, items, return_evidence=True)
    shown = apply_calibration(raw, cal, ev)
    full = apply_calibration(raw, cal)
    none, strong = ev == 0, ev > 0.5
    assert none.any() and strong.any()
    assert np.allclose(shown[none], raw[none])                             # no evidence: unstretched
    assert np.allclose(shown[strong], raw[strong] + ev[strong] * (full[strong] - raw[strong]), atol=1e-5)
    assert ((ev >= 0) & (ev < 1)).all()


def test_demo_library(client):
    """The homepage's "See a demo" loads the bundled Goodreads export through the normal import matching."""
    res = client.get("/api/demo").json()
    assert len(res["rated"]) > 150 and len(res["to_read"]) > 200   # the rest are mostly post-2017 (outside the catalog)
    assert res["stats"]["rows"] > 700 and res["stats"]["match_rate"] > 0.5
    assert {"reviews", "unmatched", "unmatched_to_read"} <= res.keys()


def test_export_dates_prefer_date_read():
    from goodrec.api.matching import export_date, read_date
    assert read_date({"Date Read": "2021/03/09", "Date Added": "2020/01/01"}) == "2021-03-09"
    assert read_date({"Date Read": "", "Date Added": "2020/01/01"}) == "2020-01-01"
    assert read_date({"Date Read": "not a date", "Date Added": ""}) is None
    assert export_date("2999/01/01") is None               # in the future


def test_import_returns_dates_and_recency_changes_ranking(client):
    with open(FIXTURE, "rb") as f:
        res = client.post("/api/import", files={"file": ("export.csv", f, "text/csv")}).json()
    rated = res["rated"]
    assert sum(1 for r in rated if r["date"]) / len(rated) > 0.9
    dated = [{"id": r["id"], "rating": r["rating"], "date": r["date"]} for r in rated]
    undated = [{"id": r["id"], "rating": r["rating"]} for r in rated]
    top = lambda body: [b["id"] for b in client.post("/api/recommend", json={"ratings": body, "limit": 20}).json()["for_you"]]  # noqa: E731
    assert top(dated) != top(undated)                      # recent reads count more
    same_day = [{**r, "date": "2026-01-01"} for r in undated]
    assert top(same_day) == top(undated)                   # one shared date: no recency effect
    bad = [{**r, "date": "yesterday"} for r in undated]
    assert top(bad) == top(undated)                        # invalid dates are ignored, not rejected
