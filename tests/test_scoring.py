from jobhunt.core import scoring
from jobhunt.core.scoring import JobScore, score_job, store_score
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeParseResponse


def _insert_job() -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO jobs (dedupe_key, title, company, location, description_text)
               VALUES ('k1', 'Engineer', 'Acme', 'NYC', 'Build things')""",
        )
        return cursor.lastrowid


def test_score_job_calls_llm_and_logs_usage(monkeypatch):
    fake_score = JobScore(
        interest_fit="High",
        interest_score=82,
        candidate_fit="Med",
        candidate_fit_score=55,
        summary_text="A backend role at a startup.",
        rationale_text="Matches your backend experience; interest inferred from past likes.",
    )
    fake_client = FakeAnthropicClient(parse_result=FakeParseResponse(fake_score, 123, 45))
    monkeypatch.setattr(scoring, "get_client", lambda: fake_client)

    job_id = _insert_job()
    with get_db() as conn:
        job = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    result = score_job(job, candidate_profile="Experienced backend engineer.", preference_profile=None)
    assert result.interest_fit == "High"

    with get_db() as conn:
        calls = conn.execute("SELECT * FROM llm_calls").fetchall()
    assert len(calls) == 1
    assert calls[0]["call_type"] == "scoring"
    assert calls[0]["input_tokens"] == 123
    assert calls[0]["job_id"] == job_id


def test_score_job_includes_desired_sectors_in_prompt(monkeypatch):
    fake_score = JobScore(
        interest_fit="High",
        interest_score=80,
        candidate_fit="High",
        candidate_fit_score=80,
        summary_text="summary",
        rationale_text="rationale",
    )
    fake_client = FakeAnthropicClient(parse_result=FakeParseResponse(fake_score))
    monkeypatch.setattr(scoring, "get_client", lambda: fake_client)
    captured = {}
    orig_parse = fake_client.messages.parse

    def spy_parse(**kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return orig_parse(**kwargs)

    fake_client.messages.parse = spy_parse

    job_id = _insert_job()
    with get_db() as conn:
        job = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    score_job(job, None, None, desired_sectors=["Climate Tech", "Healthcare"])
    assert "Climate Tech, Healthcare" in captured["content"]


def test_score_job_includes_desired_titles_and_salary_floor_in_prompt(monkeypatch):
    fake_score = JobScore(
        interest_fit="High",
        interest_score=80,
        candidate_fit="High",
        candidate_fit_score=80,
        summary_text="summary",
        rationale_text="rationale",
    )
    fake_client = FakeAnthropicClient(parse_result=FakeParseResponse(fake_score))
    monkeypatch.setattr(scoring, "get_client", lambda: fake_client)
    captured = {}
    orig_parse = fake_client.messages.parse

    def spy_parse(**kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return orig_parse(**kwargs)

    fake_client.messages.parse = spy_parse

    job_id = _insert_job()
    with get_db() as conn:
        job = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    score_job(job, None, None, desired_titles=["Analytics Director"], salary_floor=150000)
    assert "Analytics Director" in captured["content"]
    assert "150,000" in captured["content"]


def test_store_score_persists_row():
    job_id = _insert_job()
    score = JobScore(
        interest_fit="Low",
        interest_score=10,
        candidate_fit="High",
        candidate_fit_score=90,
        summary_text="summary",
        rationale_text="rationale",
    )
    score_id = store_score(job_id, score)
    with get_db() as conn:
        row = conn.execute("SELECT * FROM job_scores WHERE id = ?", (score_id,)).fetchone()
    assert row["interest_fit"] == "Low"
    assert row["candidate_fit_score"] == 90


def test_score_job_runs_at_medium_effort(monkeypatch):
    # Medium, not the default high: scoring runs up to daily_llm_cap times per run, so per-call
    # latency is most of how long a run takes.
    fake_score = JobScore(
        interest_fit="High",
        interest_score=80,
        candidate_fit="High",
        candidate_fit_score=80,
        summary_text="summary",
        rationale_text="rationale",
    )
    fake_client = FakeAnthropicClient(parse_result=FakeParseResponse(fake_score))
    monkeypatch.setattr(scoring, "get_client", lambda: fake_client)
    captured = {}
    orig_parse = fake_client.messages.parse

    def spy_parse(**kwargs):
        captured.update(kwargs)
        return orig_parse(**kwargs)

    fake_client.messages.parse = spy_parse

    job_id = _insert_job()
    with get_db() as conn:
        job = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    score_job(job, None, None)
    assert captured["output_config"] == {"effort": "medium"}
    assert captured["output_format"] is JobScore
