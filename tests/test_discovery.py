import pytest

from jobhunt.core import discovery
from tests.fakes import FakeAnthropicClient, FakeCreateResponse


def test_suggest_companies_parses_json_and_logs_usage(monkeypatch):
    payload = (
        "I searched and found these.\n"
        '{"suggestions": ['
        '{"company_name": "Acme", "reason": "Hiring backend engineers now"},'
        '{"company_name": "Beta Corp", "reason": "Recently raised a round for this team"}'
        "]}"
    )
    monkeypatch.setattr(
        discovery, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse(payload))
    )

    suggestions = discovery.suggest_companies("Experienced backend engineer.", {})
    assert len(suggestions) == 2
    assert suggestions[0].company_name == "Acme"

    from jobhunt.db.connection import get_db

    with get_db() as conn:
        calls = conn.execute("SELECT * FROM llm_calls").fetchall()
    assert len(calls) == 1
    assert "discovery:suggest_companies" in calls[0]["note"]


def test_suggest_companies_raises_on_unparseable_response(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "get_client",
        lambda: FakeAnthropicClient(create_result=FakeCreateResponse("no json here at all")),
    )
    with pytest.raises(ValueError):
        discovery.suggest_companies(None, {})


def test_profile_text_includes_settings():
    text = discovery._profile_text(
        "profile text",
        {
            "prefilter_titles_include": ["Backend Engineer"],
            "prefilter_sectors": ["Climate Tech"],
            "prefilter_locations": ["Remote"],
            "prefilter_remote_only": True,
            "salary_floor": 150000,
        },
    )
    assert "Backend Engineer" in text
    assert "Climate Tech" in text
    assert "Remote" in text
    assert "150000" in text
