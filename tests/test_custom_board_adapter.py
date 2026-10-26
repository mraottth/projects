from jobhunt.ingest.adapters import custom_board
from jobhunt.ingest.adapters.custom_board import JobListing, JobListingsOutput
from tests.fakes import FakeAnthropicClient, FakeParseResponse


def test_fetch_extracts_multiple_postings(monkeypatch):
    monkeypatch.setattr(
        custom_board,
        "fetch_page_text",
        lambda *a, **k: "Senior Engineer at NYC Office [https://x.com/jobs/1] ...",
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
    monkeypatch.setattr(custom_board, "fetch_page_text", lambda *a, **k: "many jobs...")
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
