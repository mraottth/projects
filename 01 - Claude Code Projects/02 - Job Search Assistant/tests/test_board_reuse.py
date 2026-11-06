"""An unchanged board page reuses its earlier read instead of paying Claude again (asked 2026-11-04)."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core.page_fetch import FetchedPage
from jobhunt.db.connection import get_db
from jobhunt.ingest.adapters import custom_board
from jobhunt.ingest.adapters.custom_board import JobListing, JobListingsOutput
from tests.fakes import FakeParseResponse

PAGE = "Jobs\nData Analyst [https://x.org/1] Posted 3 days ago\nEngineer [https://x.org/2] Posted today\n"


def _listing(title):
    return JobListing(
        title=title,
        company=None,
        location=None,
        url=None,
        posted_at="2026-11-01",
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        salary_period=None,
    )


class CountingClient:
    def __init__(self, titles=("Data Analyst", "Engineer"), fail=False):
        self.calls = 0
        self.titles = titles
        self.fail = fail
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return FakeParseResponse(JobListingsOutput(postings=[_listing(t) for t in self.titles]))


def _source(url="https://x.org/jobs", company=None):
    return {"config": {"url": url}, "company_name": company}


def _read(monkeypatch, text, client, source=None):
    monkeypatch.setattr(custom_board, "get_client", lambda: client)
    monkeypatch.setattr(
        custom_board, "fetch_page", lambda *a, **k: FetchedPage(text=text, html="", rendered=False)
    )
    return custom_board.fetch_with_report(source or _source())


def test_an_unchanged_page_is_read_once(monkeypatch):
    client = CountingClient()
    first, _ = _read(monkeypatch, PAGE, client)
    again, steps = _read(monkeypatch, PAGE, client)
    assert client.calls == 1
    assert again == first and [p["title"] for p in again] == ["Data Analyst", "Engineer"]
    assert steps[0] == "read the page as plain HTML: 2 (unchanged, no Claude call)"


def test_a_changed_page_is_read_again(monkeypatch):
    client = CountingClient()
    _read(monkeypatch, PAGE, client)
    _read(monkeypatch, PAGE + "Designer [https://x.org/3] Posted today\n", client)
    assert client.calls == 2


def test_shifting_relative_dates_dont_count_as_a_change(monkeypatch):
    client = CountingClient()
    _read(monkeypatch, PAGE, client)
    shifted = PAGE.replace("3 days ago", "4 days ago").replace("Posted today", "Posted yesterday")
    postings, _ = _read(monkeypatch, shifted, client)
    assert client.calls == 1
    assert postings[0]["posted_at"] == "2026-11-01"  # the absolute date from the first read


def test_reused_postings_get_the_current_employer_name(monkeypatch):
    client = CountingClient()
    _read(monkeypatch, PAGE, client, _source())
    postings, _ = _read(monkeypatch, PAGE, client, _source(company="Acme"))
    assert client.calls == 1 and {p["company"] for p in postings} == {"Acme"}


def test_empty_or_failed_reads_are_not_kept(monkeypatch):
    empty = CountingClient(titles=())
    _read(monkeypatch, PAGE, empty)
    _read(monkeypatch, PAGE, empty)
    assert empty.calls == 2

    failing = CountingClient(fail=True)
    for _ in range(2):
        try:
            _read(monkeypatch, PAGE + "x", failing)
        except Exception:  # noqa: BLE001 - the LLM error surfaces as before
            pass
    assert failing.calls == 2
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM page_extractions").fetchone()[0] == 0


def test_the_same_text_on_another_url_is_read_separately(monkeypatch):
    client = CountingClient()
    _read(monkeypatch, PAGE, client, _source("https://x.org/jobs"))
    _read(monkeypatch, PAGE, client, _source("https://y.org/jobs"))
    assert client.calls == 2


def test_start_over_keeps_shared_board_reads(monkeypatch):
    _read(monkeypatch, PAGE, CountingClient())
    with TestClient(app) as http:
        assert http.post("/api/settings/delete-all-data", json={"confirm": "DELETE"}).status_code == 200
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM page_extractions").fetchone()[0] == 1
