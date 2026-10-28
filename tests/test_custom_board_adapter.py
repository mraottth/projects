from jobhunt.core.page_fetch import FetchedPage
from jobhunt.ingest.adapters import custom_board
from jobhunt.ingest.adapters.custom_board import JobListing, JobListingsOutput
from tests.fakes import FakeAnthropicClient, FakeParseResponse


def _page(text, html="<html></html>", rendered=False):
    return FetchedPage(text=text, html=html, rendered=rendered)


def _listing(title):
    return JobListing(
        title=title,
        company=None,
        location=None,
        url=None,
        posted_at=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        salary_period=None,
    )


def test_fetch_extracts_multiple_postings(monkeypatch):
    monkeypatch.setattr(
        custom_board,
        "fetch_page",
        lambda *a, **k: _page("Senior Engineer at NYC Office [https://x.com/jobs/1] ..."),
    )

    listings = JobListingsOutput(
        postings=[
            JobListing(
                title="Senior Engineer",
                company="NYC Office of Technology",
                location="Brooklyn, NY",
                url="https://x.com/jobs/1",
                posted_at="2026-10-23",
                salary_min=100000,
                salary_max=180000,
                salary_currency="USD",
                salary_period="year",
            ),
            JobListing(
                title="UX Designer",
                company=None,
                location="Remote",
                url="https://x.com/jobs/2",
                posted_at=None,
                salary_min=None,
                salary_max=None,
                salary_currency=None,
                salary_period=None,
            ),
        ]
    )
    monkeypatch.setattr(
        custom_board, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(listings))
    )

    postings = custom_board.fetch({"config": {"url": "https://x.com/jobs"}, "company_name": "X Corp"})

    assert len(postings) == 2
    assert postings[0]["company"] == "NYC Office of Technology"
    assert postings[0]["description_text"] is None  # backfilled later, not here
    assert postings[1]["company"] == "X Corp"  # falls back to the source's configured company name


def test_fetch_caps_postings_per_fetch(monkeypatch):
    monkeypatch.setattr(custom_board, "fetch_page", lambda *a, **k: _page("many jobs..."))
    many = JobListingsOutput(
        postings=[
            JobListing(
                title=f"Job {i}",
                company="Acme",
                location=None,
                url=None,
                posted_at=None,
                salary_min=None,
                salary_max=None,
                salary_currency=None,
                salary_period=None,
            )
            for i in range(100)
        ]
    )
    monkeypatch.setattr(
        custom_board, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(many))
    )
    postings = custom_board.fetch({"config": {"url": "https://x.com/jobs"}})
    assert len(postings) == custom_board.MAX_POSTINGS_PER_FETCH


def test_embedded_greenhouse_board_is_read_through_its_api(monkeypatch):
    html = '<div id="grnhse_app"></div><script src="https://boards.greenhouse.io/embed/job_board/js?for=bluerose"></script>'
    monkeypatch.setattr(custom_board, "fetch_page", lambda *a, **k: _page("About us...", html))
    seen = {}

    def fake_greenhouse(source):
        seen.update(source)
        return [{"title": "Data Scientist"}]

    monkeypatch.setitem(custom_board.ATS_ADAPTERS, "ats_greenhouse", fake_greenhouse)

    def no_claude():
        raise AssertionError("an embedded ATS board shouldn't need a Claude call")

    monkeypatch.setattr(custom_board, "get_client", no_claude)

    postings = custom_board.fetch({"config": {"url": "https://x.org/jobs/"}, "company_name": "Blue Rose"})
    assert postings == [{"title": "Data Scientist"}]
    assert seen["config"] == {"board_token": "bluerose"}
    assert seen["company_name"] == "Blue Rose"


def test_empty_static_page_is_retried_in_the_browser(monkeypatch):
    calls = []

    def fake_fetch_page(url, *, force_render=False, **k):
        calls.append(force_render)
        if force_render:
            return _page("Research Analyst [https://x.org/jobs/1]", rendered=True)
        return _page("About us. Our mission. Contact.")

    monkeypatch.setattr(custom_board, "fetch_page", fake_fetch_page)
    replies = iter(
        [JobListingsOutput(postings=[]), JobListingsOutput(postings=[_listing("Research Analyst")])]
    )
    monkeypatch.setattr(
        custom_board, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(next(replies)))
    )

    postings = custom_board.fetch({"config": {"url": "https://x.org/jobs/"}})
    assert calls == [False, True]
    assert [p["title"] for p in postings] == ["Research Analyst"]


def test_page_already_rendered_is_not_retried(monkeypatch):
    calls = []

    def fake_fetch_page(url, *, force_render=False, **k):
        calls.append(force_render)
        return _page("No openings right now.", rendered=True)

    monkeypatch.setattr(custom_board, "fetch_page", fake_fetch_page)
    monkeypatch.setattr(
        custom_board,
        "get_client",
        lambda: FakeAnthropicClient(parse_result=FakeParseResponse(JobListingsOutput(postings=[]))),
    )
    assert custom_board.fetch({"config": {"url": "https://x.org/jobs/"}}) == []
    assert calls == [False]
