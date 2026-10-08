"""The "Assistant" tab: a Claude-powered reading assistant with tools over the recommender.

Claude runs a tool-use loop here on the server. The tools call the same route functions the UI uses
(recommend_route, insights, book_personal, Catalog.search), so ranks, predicted ratings, the YA default
and the prediction floor match what the user sees elsewhere. Claude adds judgment, conversation and
knowledge of books outside the 2017 dataset (plus Anthropic's server-side web search for new books).

Stateless like the rest of the API: the browser sends its shelf and the plain-text transcript each turn.
The response is a server-sent-event stream: status / text / books / done / error.
"""

import datetime as dt
import os
import re
import threading
from collections import defaultdict

import orjson
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from goodrec.api.schemas import FilterSpec, InsightsRequest, Rating, RecommendRequest
from goodrec.config import load_config
from goodrec.core.scoring import similar_books

router = APIRouter()


class _Api:
    """goodrec.api.main, imported lazily (main mounts this router, so a top-level import would be circular)."""

    def __getattr__(self, name):
        import goodrec.api.main as main

        return getattr(main, name)


api = _Api()
CFG = load_config()["chat"]
BOOK_REF = re.compile(r"\[\[book:(\d+)\]\]")


# ------------------------------------------------------------------ request

class OutsideBook(BaseModel):
    """A book from the user's Goodreads export that isn't in the catalog (usually published after 2017)."""
    title: str
    author: str = ""
    year: str | int | None = None
    rating: int | None = None
    shelf: str = ""
    review: str = ""


class ChatShelf(BaseModel):
    ratings: list[Rating] = []
    read: list[int] = []
    to_read: list[int] = []
    dismissed: list[int] = []
    reviews: dict[int, str] = {}           # work_id -> the user's review text
    outside: list[OutsideBook] = []


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=20_000)
    tier: str | None = None                # assistant turns: which tier answered (keeps conversations sticky)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    shelf: ChatShelf = ChatShelf()
    tier: str | None = Field(None, pattern="^(simple|complex)$")   # starter buttons pick the tier; else routed


# ------------------------------------------------------------------ limits

class Limits:
    """Per-visitor and global daily message caps (in memory, so per Cloud Run instance)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.day = None
        self.by_ip: dict[str, int] = defaultdict(int)
        self.total = 0

    def take(self, ip: str) -> str | None:
        with self.lock:
            today = dt.date.today()
            if today != self.day:
                self.day, self.by_ip, self.total = today, defaultdict(int), 0
            if self.total >= CFG["global_daily"]:
                return "The assistant has hit its daily limit for everyone. Please try again tomorrow."
            if self.by_ip[ip] >= CFG["per_visitor_daily"]:
                return f"You've reached today's limit of {CFG['per_visitor_daily']} messages. Please come back tomorrow."
            self.by_ip[ip] += 1
            self.total += 1
            return None


limits = Limits()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


# ------------------------------------------------------------------ tools

FILTER_PROPS = {
    "genres": {"type": "array", "items": {"type": "string"},
               "description": "Parent genres (exact names from list_genres_and_tags), e.g. ['History', 'Science & Nature']. A book matches if its genre is any of these."},
    "tags": {"type": "array", "items": {"type": "string"},
             "description": "Finer tags (exact names from list_genres_and_tags), e.g. ['Science']. A book matches if it has any of these."},
    "text": {"type": "string", "description": "Words that must appear in the title, author, description or tags, e.g. 'history of science'."},
    "year_min": {"type": "integer"}, "year_max": {"type": "integer"},
    "min_avg_rating": {"type": "number", "description": "Minimum Goodreads average rating, e.g. 3.9."},
    "include_ya": {"type": "boolean", "description": "Include young adult books (hidden by default)."},
    "sort": {"type": "string", "enum": ["match", "predicted"],
             "description": "'match' = the recommender's best-match ranking (default); 'predicted' = highest predicted rating first."},
    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Default 10."},
}

TOOLS = [
    {"name": "get_recommendations",
     "description": "The user's personalized 'For you' recommendations from the recommender (books they haven't read), "
                    "optionally filtered. Returns rank, predicted rating (their scale), the books they rated that led to "
                    "each pick, and how similar readers rated it. Use this first for any request for suggestions.",
     "input_schema": {"type": "object", "properties": FILTER_PROPS}},
    {"name": "readers_like_you",
     "description": "The 'From similar readers' list: books the ~300 readers with the most similar taste have read, "
                    "excluding books the user has read. 'order': 'popularity' (most read by them), 'rating' (their "
                    "average rating) or 'predicted' (the user's predicted rating). 'relative': popularity compared with "
                    "all readers (books distinctive to readers like them), or ratings compared with the Goodreads "
                    "average (books they or the user would like more than readers generally). Same filters as "
                    "get_recommendations. Needs at least 5 ratings.",
     "input_schema": {"type": "object", "properties": {
         "order": {"type": "string", "enum": ["popularity", "rating", "predicted"]},
         "relative": {"type": "boolean", "description": "Default false."},
         **{k: v for k, v in FILTER_PROPS.items() if k != "sort"}},
         "required": ["order"]}},
    {"name": "search_catalog",
     "description": "Find books in the catalog (Goodreads, published through 2017) by title/author/keywords. 'books' are "
                    "ones the user hasn't read (with their predicted rating; 'you' marks to-read); 'already_read' are "
                    "matches they've read or rated, which must not be recommended.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "maximum": 10}},
                      "required": ["query"]}},
    {"name": "book_details",
     "description": "Full details for one catalog book: description, tags, series, the user's predicted rating or actual "
                    "rating and review, how similar readers rated it, and similar books.",
     "input_schema": {"type": "object", "properties": {"work_id": {"type": "integer"}}, "required": ["work_id"]}},
    {"name": "my_library",
     "description": "Books on the user's own shelves. shelf='rated' (their ratings), 'read' (read, unrated), 'to_read' "
                    "(their want-to-read list, ranked by how well each fits their taste), or 'outside' (books from their "
                    "Goodreads export that aren't in the catalog, mostly published after 2017).",
     "input_schema": {"type": "object", "properties": {
         "shelf": {"type": "string", "enum": ["rated", "read", "to_read", "outside"]},
         "min_rating": {"type": "integer"}, "max_rating": {"type": "integer"},
         "genre": {"type": "string", "description": "Parent genre name."},
         "text": {"type": "string", "description": "Match in title or author."},
         "limit": {"type": "integer", "maximum": 100, "description": "Default 40."}}, "required": ["shelf"]}},
    {"name": "my_review",
     "description": "The user's own written review of a book they've read (catalog work_id).",
     "input_schema": {"type": "object", "properties": {"work_id": {"type": "integer"}}, "required": ["work_id"]}},
    {"name": "list_genres_and_tags",
     "description": "The exact genre and tag names the filters accept.",
     "input_schema": {"type": "object", "properties": {}}},
]


class Toolbox:
    """Tool implementations for one request (bound to the user's shelf)."""

    def __init__(self, shelf: ChatShelf):
        self.shelf = shelf
        self.base = dict(ratings=shelf.ratings, read=shelf.read, to_read=shelf.to_read, dismissed=shelf.dismissed)
        cat = api.state["cat"]
        self.rated = {r.id: r.rating for r in shelf.ratings}
        self.read, self.to_read = set(shelf.read), set(shelf.to_read)
        self.user = api._user(RecommendRequest(**self.base))
        self.cat = cat

    def run(self, name: str, args: dict) -> dict | list:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except HTTPException as e:
            return {"error": e.detail}
        except (TypeError, ValueError) as e:
            return {"error": f"bad arguments: {e}"}

    # -------------------------------------------------- helpers

    def _status(self, work_id: int) -> str | None:
        if work_id in self.rated:
            return f"rated {self.rated[work_id]}★"
        if work_id in self.read:
            return "read"
        if work_id in self.to_read:
            return "on to-read shelf"
        return None

    @staticmethod
    def _compact(b: dict, desc_chars: int = 220) -> dict:
        out = {"work_id": b["id"], "title": b["title"], "author": b["author"], "year": b["year"], "genre": b["genre"],
               "tags": b["tags"][:5], "goodreads_avg": b["avg_rating"], "ratings_count": b["ratings_count"]}
        if b.get("series"):
            out["series"] = f"{b['series']} #{b['series_pos']:g}" if b.get("series_pos") else b["series"]
        if desc_chars and b.get("description"):
            d = b["description"]
            out["description"] = d if len(d) <= desc_chars else d[:desc_chars].rsplit(" ", 1)[0] + "…"
        for k in ("rank", "predicted_rating", "readers_avg", "readers_n", "pct_read", "pct_read_overall"):
            if b.get(k) is not None:
                out[k] = b[k]
        if b.get("because"):
            out["because_you_liked"] = [x["title"] for x in b["because"]]
        if b.get("next_in_series"):
            out["next_in_series"] = True
        return out

    def _rec(self, filters: dict, limit: int, sort: str, **readers) -> dict:
        req = RecommendRequest(**self.base, filters=FilterSpec(**filters), limit=min(max(limit, 1), 20), sort=sort,
                               **readers)
        return api.recommend_route(req)

    @staticmethod
    def _split(args: dict) -> tuple[dict, int, str]:
        limit, sort = args.pop("limit", 10), args.pop("sort", "match")
        f = {k: v for k, v in args.items() if k in FilterSpec.model_fields and v not in (None, "", [])}
        return f, limit, sort

    # -------------------------------------------------- tools

    def t_get_recommendations(self, **args):
        f, limit, sort = self._split(args)
        res = self._rec(f, limit, sort)
        books = [self._compact(b) for b in res["for_you"]]
        return {"books": books, "matching_total": res["meta"]["total_candidates"],
                "note": None if books else "No recommendations match these filters; try loosening them."}

    def t_readers_like_you(self, order: str = "popularity", relative: bool = False, **args):
        f, limit, _ = self._split(args)
        res = self._rec(f, limit, "match", readers_sort=order, readers_relative=bool(relative))
        if not res["similar_readers"]:
            return {"error": "Needs at least 5 rated books in the catalog."}
        return {"n_similar_readers": res["similar_readers"]["n_neighbors"],
                "books": [self._compact(b) for b in res["similar_readers"]["books"]]}

    def t_search_catalog(self, query: str, limit: int = 8):
        hits = self.cat.search(query, min(limit, 10))
        idxs = [self.cat.idx_of[b["id"]] for b in hits]
        preds = {}
        if self.user.ratings and idxs:
            raw = api._raw(self.user)
            shown = api._shown(self.user, idxs, raw)
            preds = {b["id"]: round(float(p), 2) for b, p in zip(hits, shown)}
        out, seen = [], []
        for b in self.cat.books(idxs):
            if (s := self._status(b["id"])) and s != "on to-read shelf":
                seen.append({"work_id": b["id"], "title": b["title"], "author": b["author"], "you": s})
                continue
            c = self._compact(b, desc_chars=120)
            if s:
                c["you"] = s
            if b["id"] in preds:
                c["predicted_rating"] = preds[b["id"]]
            out.append(c)
        # Books they've read are listed separately so they aren't recommended back to them.
        return {"books": out, "already_read": seen}

    def t_book_details(self, work_id: int):
        idx = self.cat.idx_of.get(work_id)
        if idx is None:
            return {"error": "not in the catalog"}
        d = self.cat.detail(idx)
        out = self._compact(d, desc_chars=1500) | {"pages": d.get("num_pages")}
        if (s := self._status(work_id)):
            out["you"] = s
            if work_id in self.shelf.reviews:
                out["your_review"] = self.shelf.reviews[work_id]
        if self.shelf.ratings:
            pers = api.book_personal(work_id, InsightsRequest(ratings=self.shelf.ratings, read=self.shelf.read))
            if work_id not in self.rated:
                out["predicted_rating"] = pers["predicted_rating"]
            if pers["readers_avg"] is not None:
                out |= {"readers_like_you_avg": pers["readers_avg"], "readers_like_you_n": pers["readers_n"]}
        sims = self.cat.books(similar_books(api.state["art"], idx, 8))
        out["similar_books"] = [{"work_id": b["id"], "title": b["title"], "author": b["author"],
                                 **({"you": s} if (s := self._status(b["id"])) else {})} for b in sims]
        return out

    def t_my_library(self, shelf: str, min_rating: int | None = None, max_rating: int | None = None,
                     genre: str | None = None, text: str | None = None, limit: int = 40):
        limit = min(max(limit, 1), 100)
        t = (text or "").lower()
        if shelf == "outside":
            rows = [o.model_dump(exclude_defaults=True) for o in self.shelf.outside
                    if (min_rating is None or (o.rating or 0) >= min_rating)
                    and (max_rating is None or (o.rating or 9) <= max_rating)
                    and (not t or t in f"{o.title} {o.author}".lower())]
            return {"total": len(rows), "books": rows[:limit]}
        if shelf == "to_read":
            picks = api.recommend_route(RecommendRequest(**self.base, limit=200))["to_read_picks"]
            rows = [self._compact(b, desc_chars=0) for b in picks]
        else:
            ids = list(self.rated) if shelf == "rated" else list(self.read)
            rows = []
            for b in self.cat.books(api._to_idx(ids)):
                c = self._compact(b, desc_chars=0)
                if shelf == "rated":
                    c["your_rating"] = self.rated[b["id"]]
                    c["has_review"] = b["id"] in self.shelf.reviews
                rows.append(c)
            if shelf == "rated":
                rows.sort(key=lambda c: -c["your_rating"])
        rows = [c for c in rows
                if (min_rating is None or c.get("your_rating", 5) >= min_rating)
                and (max_rating is None or c.get("your_rating", 1) <= max_rating)
                and (not genre or (c["genre"] or "").lower() == genre.lower())
                and (not t or t in f"{c['title']} {c['author']}".lower())]
        return {"total": len(rows), "books": rows[:limit]}

    def t_my_review(self, work_id: int):
        text = self.shelf.reviews.get(work_id)
        return {"review": text} if text else {"review": None, "note": "No written review for this book."}

    def t_list_genres_and_tags(self):
        m = api.state["art"].meta
        return {"genres": list(m.genre_names), "tags": sorted(api.state["tag_idx"])}


# ------------------------------------------------------------------ system prompt

SYSTEM = """You are the reading assistant inside "Shelf Life", a book recommender built on Goodreads data. \
Think of yourself as a well-read friend and book-club partner: warm, candid, specific, never gushing. \
Today is {today}.

## What you know and how to use it
- Below is this reader's library: every book they rated (by stars), their to-read list, and books from their \
Goodreads export that aren't in our catalog. Use it: notice patterns, reference books they loved or disliked, and \
never recommend something they've already read.
- The recommender's catalog is ~105,000 Goodreads books published through 2017. Use the tools to get personalized \
picks from it: get_recommendations (their "For you" list, filterable), readers_like_you (what the ~300 most similar \
readers read and loved), my_library (their shelves, incl. to-read ranked by fit), book_details, search_catalog, \
my_review. Call list_genres_and_tags before filtering by a genre or tag name you're unsure of.
- For requests on a subject ("a nonfiction book about the history of science"), try filters (genres/tags/text) and, \
if the filtered list is thin, also search_catalog on the topic. Combine the recommender's picks with your own \
judgment; prefer books with a good predicted rating for this reader.
- For books published after 2017 you may use your own knowledge and web_search. Always say clearly that these come \
from outside the recommender (we have no predicted rating for them), and explain which of their books makes you \
think they'd like each one. Only recommend books you're confident exist; search the web to check when unsure.
- In book-club conversation (themes, characters, endings, what to read next), be a real conversation partner: share \
opinions, ask good questions, connect to the reader's other books. Warn before spoilers unless they've read it.

## Honesty
- Never recommend a book this reader has already read or rated (check their library and any 'you' / 'already_read' \
fields before writing). Decide your picks before you start writing, rather than correcting yourself mid-answer.
- Books on their to-read list are fair game, but say so ("already on your to-read list"); 'on to-read shelf' \
means unread, not rated.
- Only say a book is "in your recommendations" or quote a rank / predicted rating if a tool returned it. Predicted \
ratings are on this reader's own star scale.
- If the reader has few or no ratings, say personalization is limited and suggest importing their Goodreads library \
or rating books.

## Formatting
- The reader sees everything you write, including text between tool calls. Don't narrate your process \
("let me check", "both confirmed", "not confirmed via tool"); use tools silently and write only the answer.
- Write catalog books as [[book:WORK_ID]] followed by a short reason, e.g. "[[book:12345]] — for the same ...". The app \
shows each as a card with cover, title and author, so don't repeat the title and author next to the marker. Use this \
ONLY with a work_id a tool gave you.
- Write books outside the catalog as **Title** by Author (year).
- Keep answers focused: usually 3-6 recommendations, a sentence or two of reasoning each, in markdown. Ask a quick \
follow-up question when the request is vague.

{library}"""


def library_digest(shelf: ChatShelf) -> str:
    cat = api.state["cat"]
    rated = {r.id: r.rating for r in shelf.ratings}
    books = {b["id"]: b for b in cat.books(api._to_idx(list(rated) + shelf.to_read + shelf.read))}

    def line(b: dict) -> str:
        rev = " (reviewed)" if b["id"] in shelf.reviews else ""
        return f"- {b['title']} — {b['author']} ({b['year'] or '?'}) [id {b['id']}]{rev}"

    parts = ["# This reader's library"]
    if not rated and not shelf.outside:
        parts.append("They haven't rated any books yet.")
    if rated:
        parts.append(f"{len(rated)} rated books in the catalog.")
        ins = api.insights(InsightsRequest(ratings=shelf.ratings, read=shelf.read))
        if ins["harshness"]:
            h = ins["harshness"]
            parts.append(f"They rate {abs(h['bias']):.2f} stars {'below' if h['bias'] < 0 else 'above'} the Goodreads "
                         f"average on the same books (harsher than {h['harsher_than']:.0%} of readers).")
        top = ", ".join(f"{g['genre']} {g['books']}" + (f" (avg {g['your_avg']}★)" if g["your_avg"] else "")
                        for g in ins["genres"][:10])
        parts.append(f"Books by genre: {top}.")
        full = len(rated) <= CFG["library_max_lines"]
        for stars in (5, 4, 3, 2, 1):
            ids = [i for i, r in rated.items() if r == stars and i in books]
            if not ids:
                continue
            parts.append(f"\n## Rated {stars}★ ({len(ids)})")
            if full or stars != 3:
                parts.extend(line(books[i]) for i in ids)
            else:
                parts.append("(not listed; use my_library)")
    read = [books[i] for i in shelf.read if i in books]
    if read:
        parts.append(f"\n## Read, not rated ({len(read)})")
        parts.extend(line(b) for b in read[:300])
    tr = [books[i] for i in shelf.to_read if i in books]
    if tr:
        parts.append(f"\n## Want to read ({len(tr)})")
        parts.extend(line(b) for b in tr[:600])
    if shelf.outside:
        parts.append(f"\n## From their Goodreads export but not in our catalog ({len(shelf.outside)}; mostly post-2017)")
        for o in shelf.outside[:600]:
            what = f"{o.rating}★" if o.rating else ("want to read" if o.shelf == "to-read" else "read")
            parts.append(f"- {o.title} — {o.author} ({o.year or '?'}) {what}" + (" (reviewed)" if o.review else ""))
    return "\n".join(parts)


# ------------------------------------------------------------------ model routing

ROUTER = """You route messages for a book-recommendation chat assistant to a fast model or a strong model.

Answer COMPLEX if the latest message needs any of:
- books published after 2017, new or recent releases, or anything needing a web search
- analysis of the reader's taste, habits, patterns or blind spots
- in-depth discussion of a book (themes, characters, meaning, ending, comparisons), book-club style
- several constraints at once, a nuanced mood, or careful judgment between options

Answer SIMPLE for greetings, thanks, short clarifications, simple follow-ups, straightforward requests
("recommend a fantasy novel", "what should I read next", "something like X") and factual questions about a book.

Reply with exactly one word: SIMPLE or COMPLEX."""


def choose_tier(client, req: ChatRequest) -> str:
    if req.tier:
        return req.tier
    if any(m.role == "assistant" and m.tier == "complex" for m in req.messages):
        return "complex"                                   # sticky: don't drop a conversation back to the fast model
    prev = next((m.content for m in reversed(req.messages[:-1]) if m.role == "assistant"), "")
    prompt = (f"Previous assistant reply (truncated): {prev[:600]}\n\n" if prev else "") + f"Latest message: {req.messages[-1].content[:2000]}"
    try:
        r = client.messages.create(model=CFG["router_model"], max_tokens=5, system=ROUTER, timeout=3.0,
                                   messages=[{"role": "user", "content": prompt}])
        word = "".join(b.text for b in r.content if b.type == "text").strip().upper()
        return "simple" if word.startswith("SIMPLE") else "complex"
    except Exception:  # router slow or down: use the strong model
        return "complex"


# ------------------------------------------------------------------ route

def _sse(event: str, data) -> bytes:
    return b"data: " + orjson.dumps({"type": event, "data": data}) + b"\n\n"


STATUS = {"get_recommendations": "Checking your recommendations…", "readers_like_you": "Looking at readers like you…",
          "search_catalog": "Searching the catalog…", "book_details": "Reading up on a book…",
          "my_library": "Looking through your shelves…", "my_review": "Rereading your review…",
          "list_genres_and_tags": "Checking genres…", "web_search": "Searching the web…"}


def _client():
    import anthropic

    return anthropic.Anthropic()


@router.get("/api/chat/status")
def chat_status():
    return {"enabled": bool(os.environ.get("ANTHROPIC_API_KEY")), "models": {k: v["label"] for k, v in CFG["models"].items()},
            "per_visitor_daily": CFG["per_visitor_daily"], "max_turns": CFG["max_turns"]}


@router.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "The assistant isn't configured on this server (no ANTHROPIC_API_KEY).")
    if req.messages[-1].role != "user":
        raise HTTPException(422, "the last message must be from the user")
    if sum(m.role == "user" for m in req.messages) > CFG["max_turns"]:
        raise HTTPException(429, "This conversation has gotten long. Start a new chat to keep going.")
    if (msg := limits.take(_client_ip(request))):
        raise HTTPException(429, msg)
    tools = Toolbox(req.shelf)
    system = [{"type": "text", "text": SYSTEM.format(today=dt.date.today().isoformat(), library=library_digest(req.shelf)),
               "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    client = _client()
    tier = choose_tier(client, req)
    return StreamingResponse(run_chat(client, system, messages, tools, tier), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _with_cache_mark(messages: list[dict]) -> list[dict]:
    """Cache the conversation prefix: mark the last block of the last message (tool rounds reuse it)."""
    out = messages[:-1]
    last = dict(messages[-1])
    content = last["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [
        b if isinstance(b, dict) else b.model_dump(exclude_none=True) for b in content]
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return out + [{**last, "content": blocks}]


def run_chat(client, system, messages, tools: Toolbox, tier: str = "complex"):
    """Model <-> tool loop, streamed as SSE events."""
    m = CFG["models"][tier]
    yield _sse("model", {"tier": tier, "label": m["label"]})
    tool_defs = TOOLS + [{"type": CFG["web_search_tool"], "name": "web_search", "max_uses": CFG["web_search_max_uses"]}]
    text_out = []
    usage = defaultdict(int)
    try:
        for rnd in range(CFG["max_tool_rounds"] + 1):
            last = rnd == CFG["max_tool_rounds"]          # out of tool rounds: make the model answer now
            with client.messages.stream(model=m["model"], max_tokens=CFG["max_tokens"], system=system,
                                        tools=tool_defs, messages=_with_cache_mark(messages),
                                        **({"output_config": {"effort": m["effort"]}} if m.get("effort") else {}),
                                        **({"tool_choice": {"type": "none"}} if last else {})) as stream:
                for ev in stream:
                    if ev.type == "content_block_start" and ev.content_block.type in ("tool_use", "server_tool_use"):
                        yield _sse("status", STATUS.get(ev.content_block.name, "Working…"))
                    elif ev.type == "content_block_start" and ev.content_block.type == "thinking":
                        yield _sse("status", "Thinking…")
                    elif ev.type == "content_block_delta" and ev.delta.type == "text_delta":
                        text_out.append(ev.delta.text)
                        yield _sse("text", ev.delta.text)
                final = stream.get_final_message()
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                usage[k] += getattr(final.usage, k, 0) or 0
            messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in final.content]})
            if final.stop_reason == "pause_turn":          # server-side web search wants another pass
                continue
            if final.stop_reason != "tool_use":
                break
            results = []
            for b in final.content:
                if b.type == "tool_use":
                    out = tools.run(b.name, dict(b.input or {}))
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": orjson.dumps(out).decode()})
            messages.append({"role": "user", "content": results})
            if text_out and not "".join(text_out).endswith("\n"):
                text_out.append("\n\n")
                yield _sse("text", "\n\n")
        text = "".join(text_out)
        if not text.strip():   # e.g. the model spent its whole budget thinking (stop_reason max_tokens)
            yield _sse("error", f"The assistant didn't produce an answer (stopped: {final.stop_reason}). Please try again.")
            return
        ids = [int(i) for i in dict.fromkeys(BOOK_REF.findall(text))]
        books = tools.cat.books(api._to_idx(ids))
        yield _sse("books", [{k: b[k] for k in ("id", "title", "author", "year", "cover_url", "cover_url_small", "isbn", "genre")}
                             for b in books])
        yield _sse("done", {"usage": dict(usage)})
    except Exception as e:  # surface API errors (rate limits, overload) to the UI instead of a dropped stream
        yield _sse("error", f"{type(e).__name__}: {getattr(e, 'message', str(e))}")
