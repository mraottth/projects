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
    hg = _first(client, "hunger games")
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
