"""The Admin page's scoring cost experiment (asked 2026-11-03): rescore judged jobs under the current
and the lean settings, compare, and touch nothing the job lists read."""

import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import scoring, scoring_experiment, users
from jobhunt.core.scoring import CURRENT, LEAN, SCORING_EFFORT, SHORT_RATIONALE, ScoredPosting
from jobhunt.db.connection import get_db
from tests.fakes import FakeUsage, give_profile

LONG = "Responsibilities. " * 1000  # about 18,000 characters


class RecordingClient:
    """Answers High/High at medium effort and Med/High at low effort, and records each request."""

    def __init__(self, fail_titles=()):
        self.calls = []
        self.fail_titles = fail_titles
        self._lock = threading.Lock()
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
        posting = kwargs["messages"][0]["content"]
        if any(t in posting for t in self.fail_titles):
            raise RuntimeError("boom")
        low = kwargs["output_config"]["effort"] == "low"
        score = ScoredPosting(
            interest_fit="Med" if low else "High",
            interest_score=55 if low else 80,
            candidate_fit="High",
            candidate_fit_score=75,
            summary_text="s",
            rationale_text="Your ==i:mission== fits.",
            salary_min=None,
            salary_max=None,
            salary_currency=None,
            salary_period=None,
        )
        return SimpleNamespace(
            parsed_output=score,
            usage=FakeUsage(input_tokens=2000, output_tokens=400 if low else 1200),
        )


def _job(title, user_id=1, description="Build dashboards.") -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company, description_text) "
            "VALUES (?, ?, ?, 'Co', ?)",
            (user_id, f"k-{title}-{user_id}", title, description),
        ).lastrowid


def _mark(job_id, interested, user_id=1):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, job_id, interested) VALUES (?, ?, ?)",
            (user_id, job_id, interested),
        )


def _track(job_id, status, user_id=1):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO applications (user_id, job_id, status) VALUES (?, ?, ?)", (user_id, job_id, status)
        )


def test_current_is_todays_request_and_lean_is_cheaper(monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(scoring, "get_client", lambda: client)
    job = {"id": None, "title": "Analyst", "company": "Co", "description_text": LONG}

    scoring.score_job(job, "profile", None)
    _, cost, output = scoring.score_job_with(job, "profile", None, settings=LEAN, note="x")

    current, lean = client.calls
    assert current["output_config"] == {"effort": SCORING_EFFORT}
    assert LONG in current["messages"][0]["content"]
    assert current["system"][0]["text"] == scoring.SYSTEM_PROMPT
    assert lean["output_config"] == {"effort": "low"}
    posting = lean["messages"][0]["content"]
    assert LONG not in posting and posting.endswith("...(truncated)") and len(posting) < 6300
    assert lean["system"][0]["text"].endswith(SHORT_RATIONALE)
    assert lean["system"][1] == current["system"][1]  # the cached context is the same
    assert output == 400 and cost is not None and cost > 0


def test_the_sample_is_my_judged_jobs_only():
    b = users.create_user("b@example.com", "Bea", status="approved")["id"]
    wanted = _job("Wanted")
    _mark(wanted, "interested")
    unwanted = _job("Unwanted")
    _mark(unwanted, "not_interested")
    tracked = _job("Tracked")
    _track(tracked, "interviewing")
    _job("Never judged")
    _mark(_job("Maybe"), "maybe")
    _mark(_job("Join our Talent Community"), "interested")  # a generic posting
    beas = _job("Bea's", user_id=b)
    _mark(beas, "interested", user_id=b)

    sample = {j["title"]: j for j in scoring_experiment.sample_jobs()}
    assert set(sample) == {"Wanted", "Unwanted", "Tracked"}
    assert sample["Tracked"]["judgment"] == "interested" and sample["Tracked"]["interviewed"]
    assert sample["Unwanted"]["judgment"] == "not_interested"


def test_the_sample_balances_wanted_and_unwanted():
    for i in range(15):
        _mark(_job(f"W{i}"), "interested")
    for i in range(4):
        _mark(_job(f"N{i}"), "not_interested")
    sample = scoring_experiment.sample_jobs(limit=10)
    assert sum(j["judgment"] == "not_interested" for j in sample) == 4
    assert len(sample) == 10


def test_a_run_compares_and_stores_nothing_on_the_job_lists(monkeypatch):
    give_profile(1)
    client = RecordingClient(fail_titles=("Broken",))
    monkeypatch.setattr(scoring, "get_client", lambda: client)
    for title in ("A", "B"):
        _mark(_job(title), "interested")
    _mark(_job("C"), "not_interested")
    _mark(_job("Broken"), "not_interested")

    result = scoring_experiment.run()

    assert len(client.calls) == 8  # 4 jobs x 2 arms
    arms = result["summary"]["arms"]
    assert arms["current"]["scored"] == 3 and arms["current"]["failed"] == 1
    assert arms["lean"]["cost_per_job_usd"] < arms["current"]["cost_per_job_usd"]
    assert arms["current"]["interested_rated_high_or_med"] == 1.0
    assert arms["current"]["not_interested_rated_low"] == 0.0
    agreement = result["summary"]["agreement"]
    assert agreement["interest_same"] == 0.0 and agreement["fit_same"] == 1.0
    assert agreement["interest_score_diff"] == 25.0
    broken = next(j for j in result["jobs"] if j["title"] == "Broken")
    assert broken["current"] is None and broken["lean"] is None

    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_scores").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM digest_entries").fetchone()[0] == 0
        notes = [r[0] for r in conn.execute("SELECT note FROM llm_calls WHERE call_type = 'scoring'")]
    assert sorted(set(notes)) == ["scoring experiment: current", "scoring experiment: lean"]
    assert len(notes) == 6


def test_the_admin_runs_it_from_the_admin_page(monkeypatch):
    give_profile(1)
    client = RecordingClient()
    monkeypatch.setattr(scoring, "get_client", lambda: client)
    _mark(_job("A"), "interested")
    users.create_user("b@example.com", "Bea", status="approved")

    with TestClient(app) as http:
        status = http.get("/api/admin/scoring-experiment").json()
        assert (status["sample_size"], status["result"]) == (1, None)
        assert status["estimate_usd"] > 0

        body = http.post("/api/admin/scoring-experiment").text
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        assert events[-1]["type"] == "done"
        assert events[-1]["result"]["summary"]["arms"]["lean"]["scored"] == 1

        assert http.get("/api/admin/scoring-experiment").json()["result"]["jobs"][0]["title"] == "A"
        # Only the admin.
        assert (
            http.get("/api/admin/scoring-experiment", headers={"X-Dev-User": "b@example.com"}).status_code
            == 403
        )
        assert (
            http.post("/api/admin/scoring-experiment", headers={"X-Dev-User": "b@example.com"}).status_code
            == 403
        )


def test_presets():
    assert (CURRENT.effort, CURRENT.max_posting_chars, CURRENT.short_rationale) == (
        SCORING_EFFORT,
        None,
        False,
    )
    assert (LEAN.effort, LEAN.max_posting_chars, LEAN.short_rationale) == ("low", 6000, True)
