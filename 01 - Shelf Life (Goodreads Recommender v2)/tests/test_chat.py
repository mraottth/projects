"""Assistant tab: tool functions against real artifacts, and the stream loop with a fake Anthropic client (no API calls)."""

from types import SimpleNamespace as NS

import orjson
import pytest

from tests.test_api import _fantasy_reader, _first, client  # noqa: F401  (shared fixture)


def _shelf(client, **extra):  # noqa: F811
    from goodrec.api.chat import ChatShelf

    return ChatShelf(ratings=_fantasy_reader(client), **extra)


def _tools(client, **extra):  # noqa: F811
    from goodrec.api.chat import Toolbox

    return Toolbox(_shelf(client, **extra))


def test_recommendation_tools_match_the_app(client):  # noqa: F811
    tb = _tools(client)
    got = tb.run("get_recommendations", {"tags": ["Science"], "limit": 8})
    app = client.post("/api/recommend", json={"ratings": _fantasy_reader(client), "limit": 8,
                                               "filters": {"tags": ["Science"]}}).json()["for_you"]
    assert [b["work_id"] for b in got["books"]] == [b["id"] for b in app]
    assert [b["rank"] for b in got["books"]] == [b["rank"] for b in app]
    rated = {r["id"] for r in _fantasy_reader(client)}
    assert not rated & {b["work_id"] for b in got["books"]}
    pop = tb.run("readers_like_you", {"order": "popularity", "genres": ["Science Fiction"], "limit": 5})
    assert pop["books"] and all(b["genre"] == "Science Fiction" for b in pop["books"])
    rel = tb.run("readers_like_you", {"order": "rating", "relative": True, "limit": 5})
    app = client.post("/api/recommend", json={"ratings": _fantasy_reader(client), "limit": 5, "readers_sort": "rating",
                                               "readers_relative": True}).json()["similar_readers"]["books"]
    assert [b["work_id"] for b in rel["books"]] == [b["id"] for b in app]
    assert tb.run("get_recommendations", {"bogus": 1, "limit": 2})["books"]      # unknown args are ignored
    assert "error" in tb.run("nope", {})


def test_library_and_book_tools(client):  # noqa: F811
    hobbit = _first(client, "the hobbit")["id"]
    dune = _first(client, "dune")["id"]
    tb = _tools(client, reviews={hobbit: "Cozy and wise."}, to_read=[dune],
                outside=[{"title": "Project Hail Mary", "author": "Andy Weir", "rating": 5, "shelf": "read"}])
    rated = tb.run("my_library", {"shelf": "rated", "min_rating": 5})
    assert rated["total"] == 3 and all(b["your_rating"] == 5 for b in rated["books"])
    assert tb.run("my_library", {"shelf": "to_read"})["books"][0]["work_id"] == dune
    assert tb.run("my_library", {"shelf": "outside"})["books"][0]["title"] == "Project Hail Mary"
    assert tb.run("my_review", {"work_id": hobbit})["review"] == "Cozy and wise."
    d = tb.run("book_details", {"work_id": hobbit})
    assert d["you"] == "rated 4★" and d["your_review"] == "Cozy and wise." and len(d["similar_books"]) == 8
    d = tb.run("book_details", {"work_id": dune})
    assert d["you"] == "on to-read shelf" and "predicted_rating" in d
    s = tb.run("search_catalog", {"query": "a brief history of time"})
    assert s["books"] and "predicted_rating" in s["books"][0]
    s = tb.run("search_catalog", {"query": "the hobbit"})
    assert hobbit in {b["work_id"] for b in s["already_read"]} and hobbit not in {b["work_id"] for b in s["books"]}
    gt = tb.run("list_genres_and_tags", {})
    assert "History" in gt["genres"] and "Science" in gt["tags"]


def test_library_digest(client):  # noqa: F811
    from goodrec.api.chat import library_digest

    text = library_digest(_shelf(client, outside=[{"title": "Piranesi", "author": "Susanna Clarke", "rating": 5}]))
    assert "## Rated 5★ (3)" in text and "Piranesi — Susanna Clarke" in text and "harsher than" in text
    from goodrec.api.chat import ChatShelf

    assert "haven't rated" in library_digest(ChatShelf())


def test_import_keeps_reviews_and_outside_books(client):  # noqa: F811
    csv = ("Book Id,Title,Author,My Rating,Exclusive Shelf,My Review\n"
           "5907,The Hobbit,J.R.R. Tolkien,5,read,Loved it.<br/><br/>Bilbo &amp; co.\n"
           "999999999,Some 2021 Novel,New Author,4,read,\n"
           "999999998,Another New One,Someone,0,to-read,\n")
    res = client.post("/api/import", files={"file": ("export.csv", csv.encode(), "text/csv")}).json()
    hobbit = res["rated"][0]["id"]
    assert res["reviews"] == {str(hobbit): "Loved it.\n\nBilbo & co."}
    assert res["unmatched"][0] | {} == {"title": "Some 2021 Novel", "author": "New Author", "year": None,
                                        "rating": 4, "shelf": "read"}
    assert res["unmatched_to_read"][0]["title"] == "Another New One"


# ------------------------------------------------------------------ stream loop with a fake client

class FakeBlock(NS):
    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


class FakeStream:
    def __init__(self, blocks, stop):
        self.blocks, self.stop = blocks, stop

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for b in self.blocks:
            yield NS(type="content_block_start", content_block=b)
            if b.type == "text":
                yield NS(type="content_block_delta", delta=NS(type="text_delta", text=b.text))

    def get_final_message(self):
        return NS(content=self.blocks, stop_reason=self.stop, usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    """First turn calls get_recommendations; second turn answers with a book marker."""

    def __init__(self, answer_id, route="SIMPLE"):
        self.calls, self.routed = [], []
        self.answer_id, self.route = answer_id, route
        self.messages = NS(stream=self.stream, create=self.create)

    def create(self, **kw):  # the router's classification call
        self.routed.append(kw)
        if self.route is None:
            raise RuntimeError("router down")
        return NS(content=[NS(type="text", text=self.route)])

    def stream(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return FakeStream([FakeBlock(type="text", text="Let me look."),
                               FakeBlock(type="tool_use", id="t1", name="get_recommendations", input={"limit": 3})], "tool_use")
        return FakeStream([FakeBlock(type="text", text=f"Try [[book:{self.answer_id}]].")], "end_turn")


def _events(body: bytes):
    return [orjson.loads(line[6:]) for line in body.split(b"\n\n") if line.startswith(b"data: ")]


def test_chat_stream_runs_tools(client, monkeypatch):  # noqa: F811
    from goodrec.api import chat

    dune = _first(client, "dune")["id"]
    fake = FakeClient(dune)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(chat, "_client", lambda: fake)
    monkeypatch.setattr(chat, "limits", chat.Limits())
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "What next?"}],
                                       "shelf": {"ratings": _fantasy_reader(client)}})
    ev = _events(r.content)
    kinds = [e["type"] for e in ev]
    assert kinds[:2] == ["model", "text"] and "status" in kinds and kinds[-2:] == ["books", "done"]
    assert ev[0]["data"]["tier"] == "simple" and fake.calls[0]["model"].startswith("claude-haiku")
    assert "".join(e["data"] for e in ev if e["type"] == "text") == f"Let me look.\n\nTry [[book:{dune}]]."
    assert ev[-2]["data"][0]["id"] == dune
    # the tool result went back to the model, and the system prompt carries the library, cached
    second = fake.calls[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result" and "books" in second[-1]["content"][0]["content"]
    assert second[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "Rated 5★" in fake.calls[0]["system"][0]["text"]
    assert any(t.get("name") == "web_search" for t in fake.calls[0]["tools"])


def test_chat_guardrails(client, monkeypatch):  # noqa: F811
    from goodrec.api import chat

    msg = {"messages": [{"role": "user", "content": "hi"}]}
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.post("/api/chat", json=msg).status_code == 503
    assert client.get("/api/chat/status").json()["enabled"] is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(chat, "_client", lambda: FakeClient(0))
    monkeypatch.setattr(chat, "limits", chat.Limits())
    monkeypatch.setitem(chat.CFG, "per_visitor_daily", 1)
    assert client.post("/api/chat", json=msg).status_code == 200
    r = client.post("/api/chat", json=msg)
    assert r.status_code == 429 and "limit" in r.json()["detail"]
    long = {"messages": [{"role": "user", "content": "x"}] * (chat.CFG["max_turns"] + 1)}
    assert client.post("/api/chat", json=long).status_code == 429


@pytest.mark.parametrize("bad", [{"messages": []}, {"messages": [{"role": "assistant", "content": "hi"}]}])
def test_chat_rejects_bad_requests(client, monkeypatch, bad):  # noqa: F811
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert client.post("/api/chat", json=bad).status_code == 422


def test_model_routing(client):  # noqa: F811
    from goodrec.api.chat import CFG, ChatRequest, choose_tier

    def req(**kw):
        return ChatRequest(**{"messages": [{"role": "user", "content": "hi"}], **kw})

    fake = FakeClient(0, route="COMPLEX")
    assert choose_tier(fake, req()) == "complex" and len(fake.routed) == 1
    assert fake.routed[0]["model"] == CFG["router_model"]
    fake.route = "SIMPLE"
    assert choose_tier(fake, req()) == "simple"
    assert choose_tier(fake, req(tier="complex")) == "complex" and len(fake.routed) == 2      # buttons skip the router
    sticky = req(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y", "tier": "complex"},
                           {"role": "user", "content": "thanks"}])
    assert choose_tier(fake, sticky) == "complex" and len(fake.routed) == 2                    # stays on the strong model
    fake.route = None
    assert choose_tier(fake, req()) == "complex"                                                # router failure -> strong
