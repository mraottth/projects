"""2026-11-03: every run failed with a bare "StopIteration". A preference-profile rebuild got a
Claude reply with no text block (its thinking used up max_tokens), and `next(...)` over the text
blocks raised StopIteration, which nothing caught, so the whole run died."""

from types import SimpleNamespace

import pytest

from jobhunt.core import preference, scoring, users
from jobhunt.core.llm import LLMError, response_text
from jobhunt.core.scoring import JobScore
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run
from tests.fakes import FakeAnthropicClient, FakeParseResponse, FakeUsage, give_profile


def _thinking_only_reply():
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="Let me weigh the feedback...")],
        usage=FakeUsage(input_tokens=500, output_tokens=1500),
        stop_reason="max_tokens",
    )


class _ThinkingOnlyClient:
    def __init__(self):
        self.messages = SimpleNamespace(create=lambda **kwargs: _thinking_only_reply())


def _score_client():
    score = JobScore(
        interest_fit="High",
        interest_score=80,
        candidate_fit="Med",
        candidate_fit_score=60,
        summary_text="s",
        rationale_text="r",
    )
    return FakeAnthropicClient(parse_result=FakeParseResponse(score))


def _source_for(*user_ids) -> int:
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        for user_id in user_ids:
            conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", (user_id, source))
    return source


def test_response_text():
    reply = SimpleNamespace(content=[SimpleNamespace(type="text", text="Hello")], stop_reason="end_turn")
    assert response_text(reply, "testing") == "Hello"
    with pytest.raises(LLMError, match=r"didn't return any text while testing \(stop reason: max_tokens\)"):
        response_text(_thinking_only_reply(), "testing")


def test_an_empty_preference_reply_no_longer_sinks_the_run(monkeypatch):
    give_profile(1)
    _source_for(1)
    with get_db() as conn:  # enough new feedback that a preference rebuild is due
        for i in range(6):
            job = conn.execute(
                "INSERT INTO jobs (user_id, dedupe_key, title, company) VALUES (1, ?, 'Old', 'Co')",
                (f"f{i}",),
            ).lastrowid
            conn.execute(
                "INSERT INTO feedback (user_id, job_id, interested) VALUES (1, ?, 'interested')", (job,)
            )
    monkeypatch.setattr(
        daily_run, "ADAPTERS", {"custom_board": lambda s: [{"title": "Analyst", "company": "Acme"}]}
    )
    monkeypatch.setattr(scoring, "get_client", _score_client)
    monkeypatch.setattr(preference, "get_client", _ThinkingOnlyClient)

    summary = daily_run.run_daily()

    assert summary["llm_scored_count"] == 1
    assert any("preference profile" in e and "didn't return any text" in e for e in summary["errors"])
    with get_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM runs WHERE user_id = 1").fetchone()[0] == 1
        )  # Last updated moves
        system = conn.execute("SELECT status FROM runs WHERE user_id IS NULL").fetchone()
    assert system["status"] == "ok"


def test_one_persons_failure_doesnt_stop_the_others(monkeypatch):
    b = users.create_user("b@example.com", "Bea", status="approved")["id"]
    give_profile(1)
    give_profile(b)
    _source_for(1, b)
    monkeypatch.setattr(
        daily_run, "ADAPTERS", {"custom_board": lambda s: [{"title": "Analyst", "company": "Acme"}]}
    )
    monkeypatch.setattr(scoring, "get_client", _score_client)
    real = daily_run._run_for_person

    def flaky(user_id, *args, **kwargs):
        if user_id == 1:
            raise StopIteration
        return real(user_id, *args, **kwargs)

    monkeypatch.setattr(daily_run, "_run_for_person", flaky)
    summary = daily_run.run_daily()

    assert summary["llm_scored_count"] == 1  # Bea, after the admin's part failed
    assert any(
        e.startswith("user 1: StopIteration (") and "test_run_resilience.py" in e for e in summary["errors"]
    )
    with get_db() as conn:
        rows = [r[0] for r in conn.execute("SELECT user_id FROM runs WHERE user_id IS NOT NULL")]
    assert rows == [b]  # no "Last updated" for a check that didn't happen


def test_the_daily_check_says_where_an_error_happened(monkeypatch):
    def boom(*a, **k):
        next(iter([]))

    monkeypatch.setattr(daily_run, "_poll_sources", boom)
    with pytest.raises(StopIteration):
        daily_run.run_daily()
    with get_db() as conn:
        error = conn.execute("SELECT error FROM runs WHERE user_id IS NULL").fetchone()["error"]
    assert error.startswith("StopIteration (test_run_resilience.py:") and "in boom)" in error
