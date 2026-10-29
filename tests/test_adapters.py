"""Adapter tests use fixture responses shaped exactly like the real APIs (verified live against
Stripe/Greenhouse, Ro/Lever, Linear/Ashby, and RemoteOK during development) -- no live network
calls in the test suite."""

from jobhunt.ingest.adapters import adzuna, ashby, greenhouse, lever, remoteok


class FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_greenhouse_adapter_normalizes_fields(monkeypatch):
    monkeypatch.setattr(
        greenhouse.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            {
                "jobs": [
                    {
                        "id": 8172508,
                        "title": "Abuse Investigator",
                        "company_name": "Stripe",
                        "location": {"name": "Dublin"},
                        "absolute_url": "https://stripe.com/jobs/search?gh_jid=8172508",
                        "first_published": "2026-09-03T13:32:53-04:00",
                        "content": "&lt;h2&gt;Who we are&lt;/h2&gt;&lt;p&gt;Stripe builds stuff.&lt;/p&gt;",
                    }
                ]
            }
        ),
    )
    postings = greenhouse.fetch({"config": {"board_token": "stripe"}, "company_name": "Stripe"})
    assert len(postings) == 1
    p = postings[0]
    assert p["title"] == "Abuse Investigator"
    assert p["location"] == "Dublin"
    assert p["posted_at"] == "2026-09-03T13:32:53-04:00"
    assert "Who we are" in p["description_text"]
    assert "<h2>" not in p["description_text"]


def test_lever_adapter_normalizes_fields(monkeypatch):
    monkeypatch.setattr(
        lever.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            [
                {
                    "id": "abc-123",
                    "text": "Analytics Engineer",
                    "categories": {"location": "New York, NY"},
                    "createdAt": 1758000000000,
                    "descriptionPlain": "Join the data team.",
                    "hostedUrl": "https://jobs.lever.co/ro/abc-123",
                    "workplaceType": "hybrid",
                }
            ]
        ),
    )
    postings = lever.fetch({"config": {"company_slug": "ro"}, "company_name": "Ro"})
    assert len(postings) == 1
    p = postings[0]
    assert p["title"] == "Analytics Engineer"
    assert p["remote_type"] == "hybrid"
    assert p["posted_at"].startswith("2025-09-16")


def test_lever_adapter_handles_unknown_company(monkeypatch):
    monkeypatch.setattr(
        lever.httpx, "get", lambda *a, **k: FakeResponse({"ok": False, "error": "Document not found"})
    )
    assert lever.fetch({"config": {"company_slug": "nonexistent"}}) == []


def test_ashby_adapter_normalizes_fields(monkeypatch):
    monkeypatch.setattr(
        ashby.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            {
                "jobs": [
                    {
                        "id": "d3bc1ced",
                        "title": "Senior Fullstack Engineer",
                        "location": "Europe",
                        "publishedAt": "2021-04-27T20:13:45.158+00:00",
                        "isRemote": True,
                        "workplaceType": "Remote",
                        "jobUrl": "https://jobs.ashbyhq.com/linear/d3bc1ced",
                        "descriptionPlain": "Build Linear.",
                    }
                ]
            }
        ),
    )
    postings = ashby.fetch({"config": {"board_name": "linear"}, "company_name": "Linear"})
    assert len(postings) == 1
    assert postings[0]["remote_type"] == "remote"


def test_remoteok_adapter_skips_legal_notice_and_filters_tags(monkeypatch):
    monkeypatch.setattr(
        remoteok.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            [
                {"legal": "terms..."},
                {
                    "id": "1",
                    "position": "Senior Software Engineer",
                    "company": "Acme",
                    "location": "",
                    "salary_min": 80000,
                    "salary_max": 100000,
                    "url": "https://remoteok.com/remote-jobs/1",
                    "date": "2026-10-25T12:34:21+00:00",
                    "description": "Build things.",
                    "tags": ["python", "backend"],
                },
                {
                    "id": "2",
                    "position": "Frontend Designer",
                    "company": "Beta",
                    "tags": ["design"],
                    "description": "Design things.",
                    "url": "https://remoteok.com/remote-jobs/2",
                    "date": "2026-10-25T12:00:00+00:00",
                },
            ]
        ),
    )
    all_postings = remoteok.fetch({"config": {}})
    assert len(all_postings) == 2

    filtered = remoteok.fetch({"config": {"tags": ["python"]}})
    assert len(filtered) == 1
    assert filtered[0]["title"] == "Senior Software Engineer"


def test_adzuna_adapter_requires_credentials(monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
    try:
        adzuna.fetch({"config": {}})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "ADZUNA_APP_ID" in str(exc)


def test_adzuna_adapter_normalizes_fields(monkeypatch):
    monkeypatch.setenv("ADZUNA_APP_ID", "x")
    monkeypatch.setenv("ADZUNA_APP_KEY", "y")
    monkeypatch.setattr(
        adzuna.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            {
                "results": [
                    {
                        "id": 123,
                        "title": "Backend Engineer",
                        "company": {"display_name": "Acme"},
                        "location": {"display_name": "Remote"},
                        "salary_min": 120000,
                        "salary_max": 150000,
                        "redirect_url": "https://adzuna.com/jobs/123",
                        "created": "2026-10-20T00:00:00Z",
                        "description": "desc",
                    }
                ]
            }
        ),
    )
    postings = adzuna.fetch({"config": {"what": "engineer", "where": "remote"}})
    assert len(postings) == 1
    assert postings[0]["company"] == "Acme"
    assert postings[0]["salary_min"] == 120000


def test_ats_pay_fields_become_salary(monkeypatch):
    monkeypatch.setattr(
        lever.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            [
                {
                    "id": "l1",
                    "text": "Analyst",
                    "categories": {},
                    "salaryRange": {
                        "min": 90000,
                        "max": 120000,
                        "currency": "USD",
                        "interval": "per-year-salary",
                    },
                }
            ]
        ),
    )
    p = lever.fetch({"config": {"company_slug": "acme"}})[0]
    assert (p["salary_min"], p["salary_max"], p["salary_currency"], p["salary_period"]) == (
        90000,
        120000,
        "USD",
        "year",
    )

    seen = {}

    def ashby_get(url, params=None, **k):
        seen.update(params or {})
        return FakeResponse(
            {
                "jobs": [
                    {
                        "id": "a1",
                        "title": "Data Lead",
                        "compensation": {
                            "summaryComponents": [
                                {"compensationType": "EquityPercentage", "minValue": 0.1, "maxValue": 0.2},
                                {
                                    "compensationType": "Salary",
                                    "interval": "1 YEAR",
                                    "currencyCode": "USD",
                                    "minValue": 150000,
                                    "maxValue": 180000,
                                },
                            ]
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(ashby.httpx, "get", ashby_get)
    p = ashby.fetch({"config": {"board_name": "acme"}})[0]
    assert seen["includeCompensation"] == "true"
    assert (p["salary_min"], p["salary_max"], p["salary_period"]) == (150000, 180000, "year")

    monkeypatch.setattr(
        greenhouse.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            {
                "jobs": [
                    {
                        "id": 1,
                        "title": "Analyst",
                        "location": {},
                        "pay_input_ranges": [
                            {"min_cents": 10000000, "max_cents": 13000000, "currency_type": "USD"}
                        ],
                    }
                ]
            }
        ),
    )
    p = greenhouse.fetch({"config": {"board_token": "acme"}})[0]
    assert (p["salary_min"], p["salary_max"], p["salary_period"]) == (100000, 130000, "year")


def test_ats_without_pay_fields_leaves_salary_empty(monkeypatch):
    monkeypatch.setattr(
        ashby.httpx, "get", lambda *a, **k: FakeResponse({"jobs": [{"id": "a", "title": "x"}]})
    )
    p = ashby.fetch({"config": {"board_name": "acme"}})[0]
    assert p["salary_min"] is None and p["salary_max"] is None


def test_lever_and_ashby_keep_description_formatting(monkeypatch):
    monkeypatch.setattr(
        lever.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            [
                {
                    "id": "l1",
                    "text": "Analyst",
                    "categories": {},
                    "description": "<p>Join the <b>data</b> team.</p>",
                    "descriptionPlain": "Join the data team.",
                    "lists": [{"text": "Requirements", "content": "<li>SQL</li><li>Python</li>"}],
                    "additional": "<p>We offer benefits.</p>",
                }
            ]
        ),
    )
    p = lever.fetch({"config": {"company_slug": "acme"}})[0]
    assert p["description_text"] == (
        "Join the **data** team.\n\n## Requirements\n\n- SQL\n- Python\n\nWe offer benefits."
    )

    monkeypatch.setattr(
        ashby.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            {"jobs": [{"id": "a", "title": "x", "descriptionHtml": "<h2>Role</h2><ul><li>One</li></ul>"}]}
        ),
    )
    assert ashby.fetch({"config": {"board_name": "acme"}})[0]["description_text"] == "## Role\n\n- One"
