"""Rescoring My Jobs or All Jobs on request (asked 2026-11-04): a cost estimate first, a new score
for every job on the list, and once a day for anyone but the admin."""

import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import budget, rescore, scoring, users
from jobhunt.core.scoring import ScoredPosting
from jobhunt.db.connection import get_db
from tests.fakes import FakeUsage, give_profile

B_EMAIL = "b@example.com"
B = {"X-Dev-User": B_EMAIL}


class Client:
    def __init__(self, fail_titles=()):
        self.calls = 0
        self.fail_titles = fail_titles
        self._lock = threading.Lock()
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        with self._lock:
            self.calls += 1
        if any(t in kwargs["messages"][0]["content"] for t in self.fail_titles):
            raise RuntimeError("boom")
        score = ScoredPosting(
            interest_fit="High",
            interest_score=90,
            candidate_fit="High",
            candidate_fit_score=85,
            summary_text="new",
            rationale_text="new",
            salary_min=None,
            salary_max=None,
            salary_currency=None,
            salary_period=None,
        )
        return SimpleNamespace(parsed_output=score, usage=FakeUsage(input_tokens=2000, output_tokens=400))


def _source(*user_ids) -> int:
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        for uid in user_ids:
            conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", (uid, source))
    return source


def _job(title, source=None, user_id=None) -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobs (user_id, source_id, dedupe_key, title, company, description_text) "
            "VALUES (?, ?, ?, ?, 'Co', 'Do things.')",
            (user_id, source, f"k-{title}", title),
        ).lastrowid


def _track(job_id, user_id=1):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO applications (user_id, job_id, status) VALUES (?, ?, 'waiting')", (user_id, job_id)
        )


def _run(http, scope, headers=None):
    resp = http.post(f"/api/rescore/{scope}", headers=headers or {})
    if resp.status_code != 200:
        return resp.status_code, resp.json()["detail"]
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    return 200, events[-1]


def _latest(job_id, user_id=1):
    with get_db() as conn:
        return conn.execute(
            "SELECT summary_text FROM job_scores WHERE job_id = ? AND user_id = ? ORDER BY id DESC LIMIT 1",
            (job_id, user_id),
        ).fetchone()


def test_the_lists(monkeypatch):
    mine = _source(1)
    other = _source()
    a = _job("A", mine)
    b = _job("B", mine)
    _job("Join our Talent Community", mine)  # generic: never scored
    unfollowed = _job("C", other)
    pasted = _job("Pasted", user_id=1)
    _track(a)
    _track(pasted)
    _track(unfollowed)  # tracked from a source no longer followed: still on My Jobs
    assert sorted(rescore.job_ids("my_jobs")) == sorted([a, pasted, unfollowed])
    assert sorted(rescore.job_ids("all_jobs")) == [b]  # A is applied to: on My Jobs only (2026-11-12)


def test_rescoring_my_jobs_scores_each_job_again(monkeypatch):
    give_profile(1)
    client = Client(fail_titles=("Broken",))
    monkeypatch.setattr(scoring, "get_client", lambda: client)
    source = _source(1)
    a, broken, other = _job("A", source), _job("Broken", source), _job("Not tracked", source)
    _track(a)
    _track(broken)
    with get_db() as conn:  # a queued job scored here isn't paid for again by the daily run
        conn.execute(
            "INSERT INTO scoring_queue (user_id, job_id, priority, status) VALUES (1, ?, 1, 'pending')", (a,)
        )

    with TestClient(app) as http:
        status = http.get("/api/rescore/my_jobs").json()
        assert status["count"] == 2 and status["available_at"] is None
        assert status["estimate_usd"] == 0.02 and status["budget_remaining_usd"] is None  # admin

        code, done = _run(http, "my_jobs")
    assert code == 200 and (done["scored"], done["failed"], done["stopped"]) == (1, 1, None)
    assert client.calls == 2
    assert _latest(a)["summary_text"] == "new" and _latest(other) is None
    with get_db() as conn:
        assert (
            conn.execute("SELECT status FROM scoring_queue WHERE job_id = ?", (a,)).fetchone()[0] == "scored"
        )


def test_the_estimate_uses_your_own_scoring_cost():
    with get_db() as conn:
        for _ in range(5):
            conn.execute(
                "INSERT INTO llm_calls (user_id, call_type, model, estimated_cost_usd) "
                "VALUES (1, 'scoring', 'm', 0.02)"
            )
    source = _source(1)
    for i in range(10):
        _job(f"J{i}", source)
    assert rescore.status("all_jobs")["estimate_usd"] == 0.2


def test_once_a_day_except_for_the_admin(monkeypatch):
    b = users.create_user(B_EMAIL, "Bea", status="approved")["id"]
    give_profile(1)
    give_profile(b)
    monkeypatch.setattr(scoring, "get_client", Client)
    source = _source(1, b)
    _job("A", source)

    with TestClient(app) as http:
        assert _run(http, "all_jobs", B)[0] == 200
        code, detail = _run(http, "all_jobs", B)
        assert code == 429 and detail.startswith("You can rescore this list again in 24 hours")
        status = http.get("/api/rescore/all_jobs", headers=B).json()
        assert status["available_at"] is not None and 0 < status["budget_remaining_usd"] < 10
        # The other list has its own day.
        _track(_job("Mine", source, user_id=None), user_id=b)
        assert _run(http, "my_jobs", B)[0] == 200
        # The admin: no cooldown.
        assert _run(http, "all_jobs")[0] == 200
        assert _run(http, "all_jobs")[0] == 200


def test_the_budget_still_applies(monkeypatch):
    b = users.create_user(B_EMAIL, "Bea", status="approved")["id"]
    give_profile(b)
    client = Client()
    monkeypatch.setattr(scoring, "get_client", lambda: client)
    source = _source(b)
    for i in range(3):
        _job(f"J{i}", source)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO llm_calls (user_id, call_type, model, estimated_cost_usd) "
            "VALUES (?, 'scoring', 'm', 10)",
            (b,),
        )
    monkeypatch.setattr(scoring, "get_client", budget_checked_client(client))

    with TestClient(app) as http:
        code, done = _run(http, "all_jobs", B)
    assert code == 200 and done["scored"] == 0 and "budget" in done["stopped"]
    assert client.calls == 0


def budget_checked_client(client):
    """The real get_client's budget check in front of the fake client."""

    def get():
        budget.check(users.current_user_id())
        return client

    return get


def test_unknown_list():
    with TestClient(app) as http:
        assert http.get("/api/rescore/everything").status_code == 404
        assert http.post("/api/rescore/everything").status_code == 404
