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


def test_extraction_runs_at_low_effort(monkeypatch):
    monkeypatch.setattr(custom_board, "fetch_page", lambda *a, **k: _page("Analyst [https://x.org/1]"))
    client = FakeAnthropicClient(
        parse_result=FakeParseResponse(JobListingsOutput(postings=[_listing("Analyst")]))
    )
    seen = {}
    original = client.messages.parse

    def recording_parse(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    client.messages.parse = recording_parse
    monkeypatch.setattr(custom_board, "get_client", lambda: client)
    custom_board.fetch({"config": {"url": "https://x.org/jobs/"}})
    assert seen["output_config"] == {"effort": "low"}


def test_few_postings_retry_in_the_browser_and_keep_the_bigger_result(monkeypatch):
    calls = []

    def fake_fetch_page(url, *, force_render=False, **k):
        calls.append(force_render)
        if force_render:
            return _page("A [https://x.org/1] B [https://x.org/2] C [https://x.org/3]", rendered=True)
        return _page("Featured: A [https://x.org/1]")

    monkeypatch.setattr(custom_board, "fetch_page", fake_fetch_page)
    replies = iter(
        [
            JobListingsOutput(postings=[_listing("A")]),
            JobListingsOutput(postings=[_listing("A"), _listing("B"), _listing("C")]),
        ]
    )
    monkeypatch.setattr(
        custom_board, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(next(replies)))
    )
    postings, steps = custom_board.fetch_with_report({"config": {"url": "https://x.org/jobs/"}})
    assert calls == [False, True]
    assert [p["title"] for p in postings] == ["A", "B", "C"]
    assert steps == ["read the page as plain HTML: 1", "read the page in the browser: 3"]


def test_a_posting_or_two_without_new_links_is_not_read_twice(monkeypatch):
    def fake_fetch_page(url, *, force_render=False, **k):
        if force_render:
            return _page("Featured: A [https://x.org/1] Updated today", rendered=True)
        return _page("Featured: A [https://x.org/1]")

    monkeypatch.setattr(custom_board, "fetch_page", fake_fetch_page)
    client = FakeAnthropicClient(parse_result=FakeParseResponse(JobListingsOutput(postings=[_listing("A")])))
    monkeypatch.setattr(custom_board, "get_client", lambda: client)
    postings, steps = custom_board.fetch_with_report({"config": {"url": "https://x.org/jobs/"}})
    assert [p["title"] for p in postings] == ["A"]
    assert steps == ["read the page as plain HTML: 1", "browser showed nothing new"]


def test_ats_requested_by_the_rendered_page_is_read_through_its_api(monkeypatch):
    def fake_fetch_page(url, *, force_render=False, **k):
        if force_render:
            html = "<div></div><!-- requests:\nhttps://boards-api.greenhouse.io/v1/boards/cfa/jobs\n-->"
            return _page("Openings", html=html, rendered=True)
        return _page("About us. Openings below.")

    monkeypatch.setattr(custom_board, "fetch_page", fake_fetch_page)
    monkeypatch.setitem(
        custom_board.ATS_ADAPTERS, "ats_greenhouse", lambda s: [{"title": s["config"]["board_token"]}]
    )
    monkeypatch.setattr(
        custom_board,
        "get_client",
        lambda: FakeAnthropicClient(parse_result=FakeParseResponse(JobListingsOutput(postings=[]))),
    )
    postings, steps = custom_board.fetch_with_report({"config": {"url": "https://x.org/jobs/"}})
    assert postings == [{"title": "cfa"}]
    assert steps[-1] == "read its Greenhouse board 'cfa': 1"


def test_linked_hosted_board_is_read_when_the_page_has_too_few(monkeypatch):
    opened = []

    def fake_fetch_page(url, *, force_render=False, **k):
        opened.append((url, force_render))
        if url == "https://apply.workable.com/bsd/":
            return _page("A [1] B [2] C [3] D [4]")
        link = '<a href="https://apply.workable.com/bsd/">See all openings</a>'
        return _page("See all openings [https://apply.workable.com/bsd/]", html=link, rendered=force_render)

    monkeypatch.setattr(custom_board, "fetch_page", fake_fetch_page)
    replies = iter(
        [
            JobListingsOutput(postings=[]),
            JobListingsOutput(postings=[_listing(t) for t in "ABCD"]),
        ]
    )
    monkeypatch.setattr(
        custom_board, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(next(replies)))
    )
    postings, steps = custom_board.fetch_with_report({"config": {"url": "https://bsd.example/careers/"}})
    assert [p["title"] for p in postings] == list("ABCD")
    assert opened[-1] == ("https://apply.workable.com/bsd/", False)
    assert steps[-1] == "read its job board https://apply.workable.com/bsd/: 4"


def test_an_embedded_board_is_remembered_and_read_directly_next_time(monkeypatch):
    # Harvey's careers page (2026-11-12): only a browser render showed its Ashby board, and the
    # daily run's renders didn't always get there, so the board's jobs were never updated.
    import json

    from jobhunt.db.connection import get_db

    with get_db() as conn:
        source_id = conn.execute(
            "INSERT INTO sources (kind, name, company_name, config_json) VALUES ('custom_board', 'Harvey', "
            "'Harvey AI', ?)",
            (json.dumps({"url": "https://harvey.example/careers"}),),
        ).lastrowid
    html = '<script src="https://jobs.ashbyhq.com/harvey/embed"></script>'
    monkeypatch.setattr(
        custom_board, "detect_embedded_ats", lambda *a, **k: ("ats_ashby", {"board_name": "harvey"})
    )
    monkeypatch.setattr(custom_board, "fetch_page", lambda *a, **k: _page("Careers", html))
    monkeypatch.setitem(custom_board.ATS_ADAPTERS, "ats_ashby", lambda source: [{"title": "Analyst"}])
    source = {
        "id": source_id,
        "config": {"url": "https://harvey.example/careers"},
        "company_name": "Harvey AI",
    }
    postings, _ = custom_board.fetch_with_report(source)
    assert postings == [{"title": "Analyst"}] and source["read_complete"]
    with get_db() as conn:
        saved = json.loads(
            conn.execute("SELECT config_json FROM sources WHERE id = ?", (source_id,)).fetchone()[0]
        )
    assert saved["embedded_ats"] == {"kind": "ats_ashby", "config": {"board_name": "harvey"}}

    # Next time: straight to the board, no page fetch, no render, no Claude.
    def no_page(*a, **k):
        raise AssertionError("a remembered board shouldn't need the page")

    monkeypatch.setattr(custom_board, "fetch_page", no_page)
    again = {"id": source_id, "config": saved, "company_name": "Harvey AI"}
    postings, steps = custom_board.fetch_with_report(again)
    assert postings == [{"title": "Analyst"}]
    assert steps == ["read its Ashby board 'harvey' (remembered): 1"]


def test_a_remembered_board_that_lists_nothing_falls_back_to_the_page(monkeypatch):
    monkeypatch.setitem(custom_board.ATS_ADAPTERS, "ats_ashby", lambda source: [])
    monkeypatch.setattr(custom_board, "detect_embedded_ats", lambda *a, **k: None)
    monkeypatch.setattr(
        custom_board, "wp_job_manager", type("W", (), {"fetch": staticmethod(lambda *a: None)})
    )
    monkeypatch.setattr(custom_board, "_extract_postings", lambda *a, **k: [{"title": "From the page"}] * 3)
    monkeypatch.setattr(custom_board, "fetch_page", lambda *a, **k: _page("Jobs: Analyst"))
    source = {
        "config": {
            "url": "https://x.org/jobs",
            "embedded_ats": {"kind": "ats_ashby", "config": {"board_name": "gone"}},
        }
    }
    postings, steps = custom_board.fetch_with_report(source)
    assert len(postings) == 3
    assert steps[0] == "its Ashby board 'gone' listed nothing; reading the page instead"
