"""The minimum-salary dealbreaker (asked 2026-11-04): hide a job whose annualized pay, at the low
end of its posted range, is below the minimum; never one whose pay can't be told."""

import pytest
from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import dealbreakers, queue
from jobhunt.core.digest import add_to_digest, get_digest, today_str
from jobhunt.core.settings import update_settings
from jobhunt.db.connection import get_db

ON = {"salary_floor": 120000, "salary_dealbreaker": True}


def job(low=None, high=None, period="year", currency="USD"):
    return {"salary_min": low, "salary_max": high, "salary_period": period, "salary_currency": currency}


@pytest.mark.parametrize(
    "posting, expected",
    [
        (job(100000, 150000), 100000),  # the low end of the range
        (job(130000), 130000),
        (job(None, 90000), 90000),  # "up to 90k": the one figure given
        (job(50, 70, "hour"), 104000),  # x 2080
        (job(9000, None, "month"), 108000),
        (job(2000, None, "week"), 104000),
        (job(150000, None, None), 150000),  # no period, clearly yearly
        (job(45, 60, None), None),  # no period and surely not yearly: can't tell
        (job(), None),
        (job(100000, currency="GBP"), None),
        (job(100000, period="fortnight"), None),
    ],
)
def test_annual_low(posting, expected):
    assert dealbreakers.annual_low(posting) == expected


def test_what_it_hides():
    assert dealbreakers.below_floor(job(100000, 150000), ON)  # a range starting below the floor
    assert dealbreakers.below_floor(job(50, 55, "hour"), ON)
    assert not dealbreakers.below_floor(job(120000, 160000), ON)  # at the floor stays
    assert not dealbreakers.below_floor(job(65, 80, "hour"), ON)  # 135,200 a year
    assert not dealbreakers.below_floor(job(), ON)  # no pay listed
    assert not dealbreakers.below_floor(job(60000, currency="EUR"), ON)
    # Off, or no minimum set: nothing.
    assert not dealbreakers.below_floor(job(50000), {"salary_floor": 120000})
    assert not dealbreakers.below_floor(job(50000), {"salary_dealbreaker": True, "salary_floor": None})


def test_both_dealbreakers_together():
    settings = {**ON, "prefilter_locations": ["DC"], "location_dealbreaker": True}
    austin_low = {**job(90000), "location": "Austin, TX", "remote_type": None}
    assert dealbreakers.reasons(austin_low, settings) == ["location", "salary"]
    dc_good = {**job(150000), "location": "Washington, DC", "remote_type": None}
    assert dealbreakers.reasons(dc_good, settings) == []


def _jobs():
    update_settings(ON)
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (source,))
        ids = {}
        for title, low, high in [("Low", 90000, 140000), ("Good", 130000, 160000), ("Unlisted", None, None)]:
            ids[title] = conn.execute(
                "INSERT INTO jobs (source_id, dedupe_key, title, company, salary_min, salary_max, "
                "salary_currency, salary_period) VALUES (?, ?, ?, 'Co', ?, ?, 'USD', 'year')",
                (source, f"k-{title}", title, low, high),
            ).lastrowid
    return ids


def test_applied_like_the_location_dealbreaker():
    ids = _jobs()
    for job_id in ids.values():
        queue.enqueue(job_id, 50, {})
        add_to_digest(job_id)
    assert sorted(queue.select_batch(10)) == sorted([ids["Good"], ids["Unlisted"]])
    assert {j["title"] for j in get_digest(today_str())} == {"Good", "Unlisted"}
    with TestClient(app) as http:
        ranked = {j["title"]: j["dealbreakers"] for j in http.get("/api/jobs/ranked").json()}
        assert ranked == {"Low": ["salary"], "Good": [], "Unlisted": []}
        check = http.get("/api/settings/salary-check").json()
    assert check["enabled"] is True and check["floor"] == 120000
    assert check["preview"] == {
        "total": 3,
        "with_pay": 2,
        "hidden": 1,
        "examples": [{"title": "Low", "company": "Co", "annual_low": 90000}],
    }
    update_settings({"salary_dealbreaker": False})
    assert {j["title"] for j in get_digest(today_str())} == {"Low", "Good", "Unlisted"}
