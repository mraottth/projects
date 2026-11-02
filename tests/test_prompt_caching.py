from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import discovery, scoring
from jobhunt.core.llm import estimate_cost_usd, log_call
from jobhunt.core.scoring import JobScore, score_job
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeCreateResponse, FakeParseResponse, FakeUsage


class RecordingClient(FakeAnthropicClient):
    """Keeps every request's keyword arguments."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []
        parse, create = self.messages.parse, self.messages.create
        self.messages.parse = lambda **kw: self.requests.append(kw) or parse(**kw)
        self.messages.create = lambda **kw: self.requests.append(kw) or create(**kw)


def _score(monkeypatch, usage=None) -> dict:
    score = JobScore(
        interest_fit="High",
        interest_score=80,
        candidate_fit="Med",
        candidate_fit_score=50,
        summary_text="s",
        rationale_text="r",
    )
    response = FakeParseResponse(score)
    if usage:
        response.usage = usage
    client = RecordingClient(parse_result=response)
    monkeypatch.setattr(scoring, "get_client", lambda: client)
    job = {"id": None, "title": "Data Lead", "company": "Civic", "description_text": "Lead the data team."}
    score_job(
        job,
        candidate_profile="Matt: 8 years in analytics.",
        preference_profile=None,
        desired_titles=["Analytics Director"],
    )
    return client.requests[0]


def test_scoring_caches_the_candidate_context_and_sends_only_the_posting(monkeypatch):
    request = _score(monkeypatch)
    instructions, context = request["system"]
    assert instructions["text"] == scoring.SYSTEM_PROMPT and "cache_control" not in instructions
    assert context["cache_control"] == {"type": "ephemeral"}
    assert "Matt: 8 years in analytics." in context["text"] and "Analytics Director" in context["text"]
    assert "Data Lead" not in context["text"]  # the posting varies per call, so it stays out of the cache
    (message,) = request["messages"]
    assert (
        message["content"].startswith("## Job posting to score")
        and "Lead the data team." in message["content"]
    )


def test_scoring_context_is_identical_across_jobs(monkeypatch):
    first = _score(monkeypatch)["system"]
    second = _score(monkeypatch)["system"]
    assert first == second


def test_cache_reads_and_writes_are_logged_and_priced(monkeypatch):
    usage = FakeUsage(
        input_tokens=1000, output_tokens=100, cache_read_input_tokens=2000, cache_creation_input_tokens=0
    )
    _score(monkeypatch, usage)
    with get_db() as conn:
        row = conn.execute("SELECT * FROM llm_calls").fetchone()
    assert (row["cache_read_tokens"], row["cache_write_tokens"]) == (2000, 0)
    # Sonnet 5: $2/MTok input, $10/MTok output; a cache read is 0.1x input.
    assert abs(row["estimated_cost_usd"] - (1000 * 2 + 2000 * 0.2 + 100 * 10) / 1_000_000) < 1e-12


def test_cache_write_costs_more_than_plain_input():
    plain = estimate_cost_usd("claude-sonnet-5", 2000, 0)
    written = estimate_cost_usd("claude-sonnet-5", 0, 0, cache_write_tokens=2000)
    read = estimate_cost_usd("claude-sonnet-5", 0, 0, cache_read_tokens=2000)
    assert abs(written - plain * 1.25) < 1e-12 and abs(read - plain * 0.1) < 1e-12


def test_web_search_turns_use_automatic_caching():
    client = RecordingClient(create_result=FakeCreateResponse('{"suggestions": []}'))
    discovery.run_web_search("system", "find boards", "discovery: test", client=client)
    assert client.requests[0]["cache_control"] == {"type": "ephemeral"}


def test_usage_log_reports_cache_hit_rate_and_saving():
    log_call("scoring", "claude-sonnet-5", 1000, 100, cache_write_tokens=2000)
    for _ in range(3):
        log_call("scoring", "claude-sonnet-5", 1000, 100, cache_read_tokens=2000)
    with TestClient(app) as client:
        cache = client.get("/api/settings/usage-log").json()["cache"]
    assert abs(cache["hit_rate"] - 6000 / 12000) < 1e-9
    # 3 reads save 0.9x of 2000 tokens each; the one write costs an extra 0.25x of 2000.
    assert abs(cache["saved_usd"] - (3 * 2000 * 0.9 - 2000 * 0.25) * 2 / 1_000_000) < 1e-12
