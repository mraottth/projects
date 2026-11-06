import pytest

from jobhunt.core import page_fetch
from jobhunt.ingest import manual_import
from jobhunt.ingest.manual_import import ManualImportError, ParsedJob, parse_job_posting
from tests.fakes import FakeAnthropicClient, FakeParseResponse


def test_parse_job_posting_requires_input():
    with pytest.raises(ManualImportError):
        parse_job_posting()


def test_parse_job_posting_from_raw_text(monkeypatch):
    parsed = ParsedJob(
        title="Senior Backend Engineer",
        company="Acme Corp",
        location="Remote (US)",
        remote_type="remote",
        salary_min=150000,
        salary_max=190000,
        salary_currency="USD",
        salary_period="year",
        description_text="Build the thing.",
        posted_at="2026-10-20",
    )
    fake_client = FakeAnthropicClient(parse_result=FakeParseResponse(parsed))
    monkeypatch.setattr(manual_import, "get_client", lambda: fake_client)

    result = parse_job_posting(raw_text="Senior Backend Engineer at Acme Corp...")

    assert result["title"] == "Senior Backend Engineer"
    assert result["company"] == "Acme Corp"
    assert result["dedupe_key"]
    assert result["url"] is None


def test_fetch_url_text_rejects_thin_pages_even_after_browser_fallback(monkeypatch):
    class FakeResponse:
        text = "<html><body>Please log in</body></html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(page_fetch.httpx, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(
        page_fetch, "render_page_html", lambda *a, **k: "<html><body>still thin</body></html>"
    )

    with pytest.raises(ManualImportError, match="too little text"):
        manual_import.fetch_url_text("https://example.com/job/1")
