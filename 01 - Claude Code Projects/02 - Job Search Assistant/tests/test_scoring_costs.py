"""Cheaper scoring (asked 2026-11-12): the title screen sets aside clear mismatches before anything
is scored (core/screening.py), and the 2pm run scores through the Message Batches API at half
price (core/scoring.py's score_jobs_batch), scoring live whatever a batch doesn't finish."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import llm, queue, scoring, screening
from jobhunt.core.screening import ScreenResult
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run
from tests.fakes import FakeAnthropicClient, FakeParseResponse, give_profile

SCORE = {
    "interest_fit": "High",
    "interest_score": 80,
    "candidate_fit": "Med",
    "candidate_fit_score": 60,
    "summary_text": "summary",
    "rationale_text": "rationale",
    "salary_min": None,
    "salary_max": None,
    "salary_currency": None,
    "salary_period": None,
}


@pytest.fixture(autouse=True)
def real_screen(monkeypatch):
    monkeypatch.setattr(daily_run, "screen_pending", screening.screen_pending)


def _jobs(*titles):
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (source,))
        ids = []
        for n, title in enumerate(titles):
            job_id = conn.execute(
                "INSERT INTO jobs (source_id, dedupe_key, title, company, description_text) "
                "VALUES (?, ?, ?, 'Acme', 'A role.')",
                (source, f"k{n}", title),
            ).lastrowid
            ids.append(job_id)
    for i, job_id in enumerate(ids):
        queue.enqueue(job_id, 100 - i, {})
    return source, ids


# ---- the title screen


def test_the_screen_sets_aside_clear_mismatches_and_they_are_never_scored(monkeypatch):
    source, (data, legal, maybe) = _jobs("Director of Data", "Senior Litigation Counsel", "Analytics Lead")
    seen = {}

    def parse(**kwargs):
        seen.update(kwargs)
        return FakeParseResponse(ScreenResult(set_aside=[2]))  # numbered in priority order

    client = FakeAnthropicClient()
    client.messages.parse = parse
    monkeypatch.setattr(screening, "get_client", lambda: client)
    outcome = screening.screen_pending([source])
    assert outcome == {"screened": 3, "set_aside": 1, "errors": []}
    assert seen["model"] == "claude-haiku-5-5"
    assert "2. Senior Litigation Counsel | Acme" in seen["messages"][0]["content"]
    assert queue.select_batch(10) == [data, maybe]  # the set-aside job never takes a slot

    # Already screened: not sent again.
    assert screening.screen_pending([source])["screened"] == 0

    # All Jobs marks it, so the page can hide it unless asked.
    with TestClient(app) as http:
        jobs = {j["title"]: j for j in http.get("/api/jobs/ranked").json()}
    assert jobs["Senior Litigation Counsel"]["screened_out"] is True
    assert jobs["Director of Data"]["screened_out"] is False


def test_a_screen_with_no_answer_leaves_jobs_unscreened(monkeypatch):
    source, ids = _jobs("Director of Data", "Analytics Lead")
    refused = FakeParseResponse(None)
    refused.stop_reason = "refusal"
    client = FakeAnthropicClient(parse_result=refused)
    monkeypatch.setattr(screening, "get_client", lambda: client)
    outcome = screening.screen_pending([source])
    assert outcome["screened"] == 0 and "no answer" in outcome["errors"][0]
    assert queue.select_batch(10) == ids  # scored as before


# ---- batch scoring


class FakeBatches:
    """messages.batches: ends after `polls` retrieves; `fail` custom ids come back errored."""

    def __init__(self, polls=1, fail=(), never_end=False):
        self.polls, self.fail, self.never_end = polls, set(fail), never_end
        self.requests, self.canceled = [], False

    def _batch(self, status):
        counts = SimpleNamespace(succeeded=0, errored=0, canceled=0, expired=0, processing=len(self.requests))
        return SimpleNamespace(id="batch_1", processing_status=status, request_counts=counts)

    def create(self, requests):
        self.requests = requests
        return self._batch("in_progress")

    def retrieve(self, batch_id):
        self.polls -= 1
        ended = (self.canceled or self.polls <= 0) and not (self.never_end and not self.canceled)
        return self._batch("ended" if ended else "in_progress")

    def cancel(self, batch_id):
        self.canceled = True
        return self._batch("canceling")

    def results(self, batch_id):
        for request in self.requests:
            cid = request["custom_id"]
            if cid in self.fail or self.canceled:
                yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="errored"))
                continue
            message = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=json.dumps(SCORE))],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=2000, output_tokens=350, cache_read_input_tokens=5000),
            )
            yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=message))


def _clients(monkeypatch, batches):
    client = FakeAnthropicClient(parse_result=FakeParseResponse(scoring.ScoredPosting(**SCORE)))
    client.messages.batches = batches
    monkeypatch.setattr(scoring, "get_client", lambda: client)
    monkeypatch.setattr(scoring, "BATCH_POLL_SECONDS", 0)
    return client


def _scored():
    with get_db() as conn:
        return conn.execute("SELECT note, estimated_cost_usd FROM llm_calls ORDER BY id").fetchall()


def test_the_2pm_run_scores_in_a_batch_at_half_price(monkeypatch):
    give_profile()
    _, ids = _jobs(*(f"Data role {n}" for n in range(6)))
    batches = FakeBatches(polls=2, fail={f"job-{ids[0]}"})
    _clients(monkeypatch, batches)
    monkeypatch.setattr(daily_run, "BATCH_TIME_BUDGET_SECONDS", 600)
    monkeypatch.setattr(daily_run, "ADAPTERS", {})

    screen = FakeAnthropicClient(parse_result=FakeParseResponse(ScreenResult(set_aside=[])))
    monkeypatch.setattr(screening, "get_client", lambda: screen)

    summary = daily_run.run_daily()
    assert summary["errors"] == []
    assert len(batches.requests) == 6
    params = batches.requests[0]["params"]
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["system"][1]["cache_control"] == {"type": "ephemeral"}  # cached like a live score
    rows = _scored()
    assert sum(1 for r in rows if r["note"] == "scoring (batch)") == 5
    assert sum(1 for r in rows if r["note"] is None) == 1  # the errored one, scored live
    batch_cost = next(r["estimated_cost_usd"] for r in rows if r["note"] == "scoring (batch)")
    full = llm.estimate_cost_usd(llm.DEFAULT_MODEL, 2000, 350, cache_read_tokens=5000)
    assert abs(batch_cost - full / 2) < 1e-9
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_scores").fetchone()[0] == 6


def test_a_batch_past_its_time_is_canceled_and_the_rest_scored_live(monkeypatch):
    give_profile()
    _, ids = _jobs(*(f"Data role {n}" for n in range(5)))
    batches = FakeBatches(never_end=True)
    _clients(monkeypatch, batches)
    scored, live, errors = daily_run._score_batch(ids, daily_run._score_args(), deadline=0)
    assert batches.canceled and scored == [] and sorted(live) == sorted(ids)


def test_someones_own_check_scores_live(monkeypatch):
    give_profile()
    _jobs(*(f"Data role {n}" for n in range(6)))
    batches = FakeBatches()
    _clients(monkeypatch, batches)
    monkeypatch.setattr(daily_run, "BATCH_TIME_BUDGET_SECONDS", 600)
    monkeypatch.setattr(daily_run, "ADAPTERS", {})
    screen = FakeAnthropicClient(parse_result=FakeParseResponse(ScreenResult(set_aside=[])))
    monkeypatch.setattr(screening, "get_client", lambda: screen)
    daily_run.run_daily(only_user=1)
    assert batches.requests == []  # waiting on the page: no batch
    assert all(r["note"] in (None, "title screen (6 jobs)") for r in _scored())


def test_batch_calls_are_logged_at_half_price():
    llm.log_call("scoring", llm.DEFAULT_MODEL, 1000, 100, note="scoring (batch)", batch=True)
    llm.log_call("scoring", llm.DEFAULT_MODEL, 1000, 100)
    halved, full = (r["estimated_cost_usd"] for r in _scored())
    assert abs(halved - full / 2) < 1e-12


def test_the_settings_card_reports_what_the_screen_did_and_put_back_works(monkeypatch):
    source, (data, legal, maybe) = _jobs("Director of Data", "Senior Litigation Counsel", "Analytics Lead")
    client = FakeAnthropicClient(parse_result=FakeParseResponse(ScreenResult(set_aside=[2])))
    monkeypatch.setattr(screening, "get_client", lambda: client)
    screening.screen_pending([source])
    with TestClient(app) as http:
        info = http.get("/api/settings/title-screen").json()
        assert (info["screened"], info["set_aside"], info["kept"], info["waiting"]) == (3, 1, 2, 0)
        assert info["calls_this_month"] == 1 and info["last_screened_at"]
        assert [j["title"] for j in info["recent"]] == ["Senior Litigation Counsel"]

        assert http.post(f"/api/settings/title-screen/{legal}/keep").json() == {"ok": True}
        assert http.post(f"/api/settings/title-screen/{legal}/keep").status_code == 404  # not set aside now
        assert http.get("/api/settings/title-screen").json()["set_aside"] == 0
    assert sorted(queue.select_batch(10)) == sorted([data, legal, maybe])  # scored like any other
