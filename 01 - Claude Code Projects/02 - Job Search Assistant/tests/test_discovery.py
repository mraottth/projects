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


def test_suggest_job_boards_parses_json_and_logs_usage(monkeypatch):
    payload = (
        "I searched and found these.\n"
        '{"suggestions": ['
        '{"name": "Climate Jobs List", "url": "https://climatejobslist.com", '
        '"reason": "Niche board for climate tech roles"},'
        '{"name": "Built In", "url": "https://builtin.com", "reason": "Strong regional tech coverage"}'
        "]}"
    )
    monkeypatch.setattr(
        discovery, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse(payload))
    )

    suggestions = discovery.suggest_job_boards("Experienced backend engineer.", {})
    assert len(suggestions) == 2
    assert suggestions[0].name == "Climate Jobs List"
    assert suggestions[0].url == "https://climatejobslist.com"

    from jobhunt.db.connection import get_db

    with get_db() as conn:
        calls = conn.execute("SELECT * FROM llm_calls").fetchall()
    assert len(calls) == 1
    assert "discovery:suggest_job_boards" in calls[0]["note"]


def test_suggest_companies_filters_out_excluded_names(monkeypatch):
    payload = (
        '{"suggestions": ['
        '{"company_name": "Acme", "reason": "Already tracked"},'
        '{"company_name": "Beta Corp", "reason": "New one"}'
        "]}"
    )
    monkeypatch.setattr(
        discovery, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse(payload))
    )

    suggestions = discovery.suggest_companies(None, {}, exclude_names=["acme"])
    assert [s.company_name for s in suggestions] == ["Beta Corp"]


def test_suggest_job_boards_filters_out_excluded_urls_by_hostname(monkeypatch):
    payload = (
        '{"suggestions": ['
        '{"name": "Climate Jobs List", "url": "https://www.climatejobslist.com/browse", '
        '"reason": "Already tracked"},'
        '{"name": "Built In", "url": "https://builtin.com", "reason": "New one"}'
        "]}"
    )
    monkeypatch.setattr(
        discovery, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse(payload))
    )

    suggestions = discovery.suggest_job_boards(None, {}, exclude_urls=["https://climatejobslist.com"])
    assert [s.name for s in suggestions] == ["Built In"]


def test_suggest_job_boards_raises_on_unparseable_response(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "get_client",
        lambda: FakeAnthropicClient(create_result=FakeCreateResponse("no json here at all")),
    )
    with pytest.raises(ValueError):
        discovery.suggest_job_boards(None, {})


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


class _ScriptedClient:
    """Returns the given responses in order and records each request's kwargs."""

    def __init__(self, *responses):
        self.calls = []
        self._responses = list(responses)
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _response(text, stop_reason="end_turn"):
    response = FakeCreateResponse(text)
    response.stop_reason = stop_reason
    return response


def test_discovery_resumes_after_pause_turn(monkeypatch):
    # A long web-search turn can pause before the JSON is written; the paused turn has to be sent
    # back so the server finishes it, otherwise there's nothing to parse.
    client = _ScriptedClient(
        _response("Searching niche boards...", stop_reason="pause_turn"),
        _response('{"suggestions": [{"name": "Tech Jobs for Good", "url": "https://techjobsforgood.com", '
                  '"reason": "Mission-driven roles"}]}'),
    )
    monkeypatch.setattr(discovery, "get_client", lambda: client)

    suggestions = discovery.suggest_job_boards(None, {})
    assert [s.name for s in suggestions] == ["Tech Jobs for Good"]
    assert len(client.calls) == 2
    resumed = client.calls[1]["messages"]
    assert resumed[-1]["role"] == "assistant"

    from jobhunt.db.connection import get_db

    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 2


def test_discovery_stops_resuming_after_the_continuation_cap(monkeypatch):
    paused = [_response("still going", stop_reason="pause_turn") for _ in range(10)]
    client = _ScriptedClient(*paused)
    monkeypatch.setattr(discovery, "get_client", lambda: client)

    with pytest.raises(ValueError):
        discovery.suggest_companies(None, {})
    assert len(client.calls) == discovery.MAX_CONTINUATIONS + 1


def test_discovery_requests_low_effort_and_a_small_search_budget(monkeypatch):
    client = _ScriptedClient(_response('{"suggestions": []}'))
    monkeypatch.setattr(discovery, "get_client", lambda: client)

    assert discovery.suggest_companies(None, {}) == []
    request = client.calls[0]
    assert request["output_config"] == {"effort": "low"}
    assert request["tools"][0]["max_uses"] == 3


def test_extract_json_ignores_stray_braces_before_the_suggestions():
    text = (
        'One posting said {"remote": true} in its listing.\n'
        '{"suggestions": [{"company_name": "Acme", "reason": "Hiring"}]}'
    )
    assert discovery._extract_json(text)["suggestions"][0]["company_name"] == "Acme"


def test_suggest_companies_returns_careers_urls(monkeypatch):
    client = _ScriptedClient(
        _response(
            '{"suggestions": ['
            '{"company_name": "Nava PBC", "reason": "Civic tech", '
            '"careers_url": "https://boards.greenhouse.io/nava"},'
            '{"company_name": "Small Org", "reason": "Mission fit", "careers_url": null}'
            "]}"
        )
    )
    monkeypatch.setattr(discovery, "get_client", lambda: client)

    suggestions = discovery.suggest_companies(None, {})
    assert suggestions[0].careers_url == "https://boards.greenhouse.io/nava"
    assert suggestions[1].careers_url is None


def test_guidance_and_like_reach_the_prompt(monkeypatch):
    client = _ScriptedClient(_response('{"suggestions": []}'))
    monkeypatch.setattr(discovery, "get_client", lambda: client)

    discovery.suggest_companies(None, {}, guidance="mission-driven tech startups", like="Nava PBC")
    prompt = client.calls[0]["messages"][0]["content"]
    assert "mission-driven tech startups" in prompt
    assert "Make this the main focus" in prompt
    assert "similar to: Nava PBC" in prompt


def test_unsteered_search_prompt_has_no_direction_sections(monkeypatch):
    client = _ScriptedClient(_response('{"suggestions": []}'))
    monkeypatch.setattr(discovery, "get_client", lambda: client)

    discovery.suggest_job_boards(None, {})
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Direction for this search" not in prompt
    assert "Find more like this" not in prompt
