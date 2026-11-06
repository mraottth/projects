"""WP Job Manager boards are read through the plugin's listings endpoint (core/wp_job_manager.py):
every page of jobs, the source URL's own filters kept, and no Claude call. The card markup below
follows what civicworkstudio.com returned live (2026-11-06), with made-up jobs."""

import json

import pytest

from jobhunt.core import wp_job_manager
from jobhunt.core.page_fetch import FetchedPage
from jobhunt.ingest.adapters import custom_board

PAGE_HTML = """<html><body><ul class="job_listings"></ul>
<script id="wp-job-manager-ajax-filters-js-extra">
var job_manager_ajax_filters = {"ajax_url":"\\/jm-ajax\\/%%endpoint%%\\/","is_rtl":"0"};
</script></body></html>"""


def card(
    n: int,
    *,
    company: str | None = "Example Org",
    location: str = "Washington, DC",
    tags: str = "",
    posted: str = "2026-11-01",
) -> str:
    company_html = f"<strong>{company}</strong>" if company else ""
    return f"""<li class="post-{n} job_listing type-job_listing job-type-full-time">
  <a href="https://board.example/job/role-{n}/">
    <div class="position"><h3>Role {n}</h3><div class="company">{company_html}</div></div>
    <div class="location">{location}</div>
    <ul class="meta"><div class="cws-card-tags">{tags}</div>
      <li class="job-type full-time">Full-time</li>
      <li class="date"><time datetime="{posted}">Posted 5 days ago</time></li></ul>
  </a>
</li>"""


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


@pytest.fixture
def no_claude(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("Claude shouldn't be called for a WP Job Manager board")

    monkeypatch.setattr(custom_board, "get_client", fail)


def _serve(monkeypatch, pages: list, html: str = PAGE_HTML):
    """The board page, and the listings endpoint answering with `pages` in turn; returns the
    requests the endpoint got."""
    monkeypatch.setattr(
        custom_board, "fetch_page", lambda *a, **k: FetchedPage(text="Jobs", html=html, rendered=False)
    )
    requests = []

    def fake_post(url, *, data, **k):
        requests.append((url, data))
        return FakeResponse(pages[len(requests) - 1])

    monkeypatch.setattr(wp_job_manager.httpx, "post", fake_post)
    return requests


def _listing_page(cards: list[str], total: int, pages: int) -> dict:
    return {"found_jobs": bool(cards), "html": "".join(cards), "total_jobs": total, "max_num_pages": pages}


def test_every_page_is_read_with_the_urls_filters_and_no_claude_call(monkeypatch, no_claude):
    requests = _serve(
        monkeypatch,
        [
            _listing_page([card(1), card(2)], total=3, pages=2),
            _listing_page([card(3)], total=3, pages=2),
        ],
    )
    url = (
        "https://board.example/jobs/?cws_remote_only=1&search_keywords=data"
        "&filter_job_type%5B%5D=full-time&filter_job_type%5B%5D=part-time&filter_job_type%5B%5D="
    )
    postings, steps = custom_board.fetch_with_report({"config": {"url": url}, "company_name": None})

    assert [p["title"] for p in postings] == ["Role 1", "Role 2", "Role 3"]
    assert steps == ["read the board's listings (WP Job Manager): 3"]
    endpoint, first = requests[0]
    assert endpoint == "https://board.example/jm-ajax/get_listings/"
    assert first["form_data"] == url.split("?", 1)[1]  # the board's own filters, as its script sends them
    assert first["filter_job_type[]"] == ["full-time", "part-time"]  # and the standard ones directly
    assert first["search_keywords"] == ["data"]
    assert (first["page"], requests[1][1]["page"], first["per_page"]) == ("1", "2", "100")


def test_a_card_becomes_a_posting():
    tags = '<span class="cws-card-tag cws-tag-salary">$120k–$150k</span><span>Remote (US)</span>'
    [posting] = wp_job_manager.parse_cards(card(7, tags=tags), {"company_name": None})
    assert posting == {
        "external_id": None,  # URL-keyed, like jobs already stored from these boards
        "title": "Role 7",
        "company": "Example Org",
        "location": "Washington, DC",
        "remote_type": "remote",
        "salary_min": 120000.0,
        "salary_max": 150000.0,
        "salary_currency": "USD",
        "salary_period": "year",
        "description_text": None,
        "url": "https://board.example/job/role-7/",
        "posted_at": "2026-11-01",
        "raw_json": posting["raw_json"],
    }


def test_a_card_without_pay_remote_or_organization():
    [posting] = wp_job_manager.parse_cards(
        card(8, company=None, location="Austin, TX"), {"company_name": "Board Co"}
    )
    assert (posting["company"], posting["remote_type"], posting["salary_min"]) == ("Board Co", None, None)


def test_a_page_without_the_plugin_is_read_the_usual_way(monkeypatch):
    requests = _serve(monkeypatch, [], html="<html><ul><li>Analyst</li></ul></html>")
    monkeypatch.setattr(custom_board, "_extract_postings", lambda *a, **k: [{"title": "From Claude"}])
    postings, _ = custom_board.fetch_with_report({"config": {"url": "https://other.example/jobs"}})
    assert postings == [{"title": "From Claude"}] and requests == []


@pytest.mark.parametrize(
    "answer",
    [json.JSONDecodeError("not json", "", 0), {"found_jobs": True, "html": "<div>changed markup</div>"}],
    ids=["endpoint error", "unreadable cards"],
)
def test_falls_back_to_reading_the_page_when_the_endpoint_fails(monkeypatch, answer):
    _serve(monkeypatch, [answer])
    monkeypatch.setattr(custom_board, "_extract_postings", lambda *a, **k: [{"title": "From Claude"}])
    postings, steps = custom_board.fetch_with_report({"config": {"url": "https://board.example/jobs/"}})
    assert postings == [{"title": "From Claude"}]
    assert "couldn't" in steps[0]


def test_no_matches_is_an_answer_not_a_failure(monkeypatch, no_claude):
    _serve(monkeypatch, [_listing_page([], total=0, pages=0)])
    postings, steps = custom_board.fetch_with_report({"config": {"url": "https://board.example/jobs/?x=1"}})
    assert postings == [] and steps == ["read the board's listings (WP Job Manager): 0"]


def test_reading_stops_at_the_page_limit(monkeypatch, no_claude):
    pages = [_listing_page([card(i)], total=900, pages=9) for i in range(9)]
    requests = _serve(monkeypatch, pages)
    postings, steps = custom_board.fetch_with_report({"config": {"url": "https://board.example/jobs/"}})
    assert len(requests) == wp_job_manager.MAX_PAGES == len(postings)
    assert steps == [f"read the board's listings (WP Job Manager): {wp_job_manager.MAX_PAGES} of 900"]


def test_endpoint_follows_the_sites_own_setting():
    html = '{"ajax_url":"\\/blog\\/jm-ajax\\/%%endpoint%%\\/"}'
    assert wp_job_manager.listings_endpoint("https://x.example/blog/jobs/", html) == (
        "https://x.example/blog/jm-ajax/get_listings/"
    )
    assert wp_job_manager.listings_endpoint("https://x.example/jobs/", "<p>no plugin</p>") is None
