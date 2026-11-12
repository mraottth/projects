"""The Usage log in plain terms (core/usage.py, asked 2026-11-12): calls by category, back-to-back
calls of one kind gathered into one activity, for the last day, week or month."""

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import usage
from jobhunt.db.connection import get_db


def _call(call_type, minutes_ago, cost, note=None, job_id=None):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO llm_calls (user_id, call_type, job_id, model, input_tokens, output_tokens, "
            "estimated_cost_usd, created_at, note) VALUES (1, ?, ?, 'claude-sonnet-5', 100, 10, ?, "
            "datetime('now', ?), ?)",
            (call_type, job_id, cost, f"-{minutes_ago} minutes", note),
        )


def _job(title):
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company) VALUES (?, ?, 'Acme')", (title, title)
        ).lastrowid


def test_categories_cover_every_kind_of_call():
    assert usage.categorize("scoring", None) == "scoring"
    assert usage.categorize("scoring", "scoring experiment: lean") == "upkeep"
    assert usage.categorize("parsing", "custom_board:https://x.org/jobs") == "boards"
    assert usage.categorize("parsing", "discovery:suggest_companies (3 web search(es))") == "discovery"
    assert usage.categorize("parsing", "resolve_company:Acme (1 web search(es))") == "discovery"
    assert usage.categorize("parsing", "retag highlights") == "upkeep"
    assert usage.categorize("parsing", None) == "adding"
    assert usage.categorize("generation", "jobby chat (0 web search(es))") == "jobby"
    assert usage.categorize("generation", "resume (docx template, edit plan)") == "documents"
    assert usage.categorize("profile_rebuild", "application insights") == "profile"
    assert usage.categorize("career_profile", None) == "profile"


def test_a_run_of_scoring_is_one_activity_and_the_summary_adds_up():
    a, b, c = _job("Analyst"), _job("Director"), _job("Lead")
    for n, job in enumerate((a, b, c, a)):  # one job scored twice still counts once
        _call("scoring", 200 + n, 0.01, job_id=job)
    _call("generation", 100, 0.05, note="resume (docx template, edit plan)", job_id=b)
    _call("generation", 30, 0.02, note="jobby chat (0 web search(es))")
    _call("generation", 29, 0.001, note="jobby chat title")
    _call("generation", 20, 0.02, note="jobby chat (1 web search(es))")
    _call("scoring", 60 * 24 * 3, 0.01, job_id=a)  # three days ago: in the week, not the day

    day = usage.summary("day")
    assert [x["label"] for x in day["activities"]] == [
        "Jobby: 2 messages",
        "Wrote 1 document",
        "Scored 3 jobs",
    ]
    assert day["activities"][2]["cost"] == 0.04 and len(day["activities"][2]["calls"]) == 4
    assert day["activities"][1]["calls"][0]["detail"] == "Resume for Director — Acme"
    assert {c["label"]: (c["cost"], c["amount"]) for c in day["categories"]} == {
        "Tailored documents": (0.05, "1 document"),
        "Job scoring": (0.04, "3 jobs"),
        "Jobby": (0.041, "2 messages"),
    }
    assert day["total_cost"] == 0.131

    week = usage.summary("week")
    assert len(week["activities"]) == 4 and week["total_cost"] == 0.141
    assert usage.summary("nonsense")["period"] == "week"


def test_the_usage_log_route_includes_the_summary():
    _call("scoring", 5, 0.01, job_id=_job("Analyst"))
    with TestClient(app) as http:
        body = http.get("/api/settings/usage-log?period=month").json()
    assert body["summary"]["period"] == "month"
    assert body["summary"]["categories"][0]["label"] == "Job scoring"
