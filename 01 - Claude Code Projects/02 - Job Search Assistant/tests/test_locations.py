"""The location dealbreaker (asked 2026-11-04). It must never hide a job it can't be sure about:
remote, unknown and multi-location jobs always stay, and "DC" covers every way of writing it plus
the metro area."""

import pytest
from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import locations, queue, rescore
from jobhunt.core.digest import add_to_digest, get_digest, today_str
from jobhunt.core.settings import update_settings
from jobhunt.db.connection import get_db

DC = ["DC"]


@pytest.mark.parametrize(
    "location",
    [
        "Washington, DC",
        "Washington, D.C.",
        "Washington D.C.",
        "Washington DC",
        "Washington, D. C.",
        "DC",
        "District of Columbia",
        "Washington, District of Columbia",
        "Washington, DC 20001",
        "US-DC-Washington",
        "Hybrid - Washington, DC",
        "Washington DC Metro Area",
        "DC Metro",
        # The metro, as asked
        "Arlington, VA",
        "Arlington",
        "Bethesda, MD",
        "Silver Spring, Maryland",
        "Tysons Corner, Virginia",
        "Reston, VA",
        "McLean, VA",
        "Alexandria, Virginia",
        "US-VA-Arlington",
        "Columbia, MD",
    ],
)
def test_dc_matches(location):
    assert locations.verdict(location, None, DC) == "match"


@pytest.mark.parametrize(
    "location",
    [
        None,
        "",
        "   ",
        "TBD",
        "Office",
        "HQ",
        "United States",
        "USA",
        "Washington",  # DC or the state: can't tell
        "Richmond",  # Virginia or California: one of them could be wanted
        "Hybrid",
    ],
)
def test_unknown_locations_stay(location):
    assert locations.verdict(location, None, DC) == "unknown"


@pytest.mark.parametrize(
    "location",
    [
        # Named alone, but only one place, or no place near DC whichever it is (asked 2026-11-12:
        # these were kept as unknown and leaked onto All Jobs).
        "London",  # England, Ontario or Kentucky: none is near DC
        "Paris",
        "Dublin",
        "Zürich",
        "São Paulo",
        "Palo Alto",
        "Sunnyvale",
        "Jersey City",
        "Hoboken",
        "Brooklyn",
        "Washington State",
        "Boulder",
    ],
)
def test_a_place_that_can_only_be_elsewhere_is_hidden(location):
    assert locations.verdict(location, None, ["Washington DC", "Remote"]) == "elsewhere"


def test_a_distinct_place_still_matches_its_own_area():
    assert locations.verdict("Palo Alto", None, ["California"]) == "match"
    assert locations.verdict("Sunnyvale", None, ["San Francisco"]) == "match"
    assert locations.verdict("London", None, ["United Kingdom"]) == "match"
    assert locations.verdict("Jersey City", None, ["New York"]) == "match"
    # A shared name decides nothing on its own.
    assert locations.verdict("Richmond", None, ["Virginia"]) == "unknown"


def test_a_bare_suburb_name_is_kept():
    # Springfield, VA is in the DC metro; with no state given, it's kept, never hidden.
    assert locations.verdict("Springfield", None, DC) == "match"
    assert locations.verdict("Springfield, IL", None, DC) == "elsewhere"


@pytest.mark.parametrize(
    "location",
    ["Remote", "Remote - US", "Remote (Seattle, WA)", "Anywhere", "Work from home", "Fully remote, USA"],
)
def test_remote_always_stays(location):
    assert locations.verdict(location, None, DC) == "remote"
    assert locations.verdict("Seattle, WA", "remote", DC) == "remote"


@pytest.mark.parametrize(
    "location",
    [
        # Vague or incomplete lists
        "Boston, MA +2 more",
        "Boston, MA + 2",
        "3 Locations",
        "Multiple locations",
        "Various",
        "Seattle, WA and more",
        "Austin, TX & others",
        "Nationwide",
        "Austin, TX; Seattle, WA; other locations",
        # A place in the list that can't be told
        "Seattle, WA; Headquarters",
        "Austin, TX; TBD",
        "Seattle, Bellevue",
    ],
)
def test_several_locations_stay_unless_all_clearly_elsewhere(location):
    assert locations.verdict(location, None, DC) == "multiple"


@pytest.mark.parametrize(
    "location",
    [
        "New York, NY; San Francisco, CA",
        "Austin, TX, Seattle, WA",
        "Austin or Dallas, TX",
        "Chicago, IL | Denver, CO",
        "San Francisco, CA | New York City, NY",  # as it appeared in a daily email (2026-11-10)
        "New York, NY / Boston, MA",
        "New York/San Francisco",
        "Paris, France, London, UK",
        "Kansas City, MO",
        "Virginia Beach, VA",
    ],
)
def test_a_definite_list_without_you_is_hidden(location):
    assert locations.verdict(location, None, DC) == "elsewhere"


@pytest.mark.parametrize(
    "location",
    [
        "New York, NY; Washington, DC",
        "Austin, TX, Arlington, VA",
        "Seattle, WA or Bethesda, MD",
        "Washington, DC, United States",
        "Arlington, VA 22201",
        "Locations: Washington, DC",
        "Boston, MA; DC +3 more",
    ],
)
def test_a_list_with_you_matches(location):
    assert locations.verdict(location, None, DC) == "match"


def test_how_lists_are_split():
    assert locations.split_places("Austin, TX, Seattle, WA") == ["Austin, TX", "Seattle, WA"]
    assert locations.split_places("Washington, DC, United States") == ["Washington, DC, United States"]
    assert locations.split_places("Paris, France, London, UK") == ["Paris, France", "London, UK"]
    assert locations.split_places("Portland, OR") == ["Portland, OR"]  # Oregon, not "or"
    assert locations.split_places("Austin or Dallas, TX") == ["Austin", "Dallas, TX"]
    assert locations.verdict("New York, NY or Remote", None, DC) == "remote"
    assert locations.verdict("Portland, OR", None, ["Portland"]) == "match"


@pytest.mark.parametrize(
    "location",
    [
        "Arlington, TX",
        "Seattle, Washington",
        "Seattle, WA",
        "San Francisco, CA",
        "San Francisco",
        "US-CA-San Francisco",
        "Portland, OR",
        "Baltimore, MD",
        "Richmond, VA",
        "New York City",
        "Austin, TX",
        "London, UK",
        "Paris, France",
        "Bangalore, India",
        "IN OFFICE - Chicago, IL",
        # How the work is done isn't a place (2026-11-10: "(Hybrid)" made one job unreadable, so kept).
        "New York City, NY (Hybrid)",
        "New York, NY (On-site)",
        "San Francisco, CA - In Office",
        "Portland, OR (hybrid)",
        "SF",
        "NYC / SF",
    ],
)
def test_clearly_elsewhere(location):
    assert locations.verdict(location, None, DC) == "elsewhere"


def test_entries_are_read_generously():
    def reads(entry, location):
        return locations.verdict(location, None, [entry])

    for entry in ["DC", "dc", "D.C.", "Washington, DC", "Washington DC", "District of Columbia", "DMV"]:
        assert reads(entry, "Arlington, VA") == "match", entry
        assert reads(entry, "Washington, D.C.") == "match", entry
    # "Washington" could be either, so both count.
    assert reads("Washington", "Washington, DC") == "match"
    assert reads("Washington", "Seattle, WA") == "match"
    # A city in a metro means its metro; a state means anywhere in it.
    assert reads("Arlington, VA", "Washington, DC") == "match"
    assert reads("Brooklyn", "New York, NY") == "match"
    assert reads("NYC", "Jersey City, NJ") == "match"
    assert reads("New York", "Buffalo, NY") == "match"
    assert reads("Virginia", "Arlington, VA") == "match"
    assert reads("VA", "Richmond, Virginia") == "match"
    assert reads("SF", "Palo Alto, CA") == "match"
    assert reads("Bay Area", "Oakland, CA") == "match"
    assert reads("US", "Austin, TX") == "match"
    assert reads("United States", "Toronto, Canada") == "elsewhere"
    assert reads("Portland", "Portland, ME") == "elsewhere"
    # An unknown city in a state: that city, in that state.
    assert reads("Ithaca, NY", "Ithaca, New York") == "match"
    assert reads("Ithaca, NY", "Albany, NY") == "elsewhere"
    # Not a place we know: matched as written, and Settings says so.
    assert reads("Narnia", "Narnia") == "match"
    assert locations.describe(["Narnia"])[0]["recognized"] is False
    assert locations.describe(["DC"])[0]["recognized"] is True
    # Several entries: any of them.
    assert locations.verdict("Boston, MA", None, ["DC", "Boston"]) == "match"


def test_off_unless_switched_on_and_given_locations():
    job = {"location": "Austin, TX", "remote_type": None}
    assert not locations.blocks(job, {"prefilter_locations": DC})
    assert not locations.blocks(job, {"prefilter_locations": [], "location_dealbreaker": True})
    assert locations.blocks(job, {"prefilter_locations": DC, "location_dealbreaker": True})


# --- where it applies ---


def _setup(dealbreaker=True):
    update_settings({"prefilter_locations": DC, "location_dealbreaker": dealbreaker})
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (source,))
        ids = {}
        for title, location in [
            ("Here", "Washington, DC"),
            ("There", "Austin, TX"),
            ("Unknown", None),
            ("Remote", "Remote"),
        ]:
            ids[title] = conn.execute(
                "INSERT INTO jobs (source_id, dedupe_key, title, company, location) "
                "VALUES (?, ?, ?, 'Co', ?)",
                (source, f"k-{title}", title, location),
            ).lastrowid
    return ids


def test_never_scored_and_comes_back_when_switched_off():
    ids = _setup()
    for job_id in ids.values():
        queue.enqueue(job_id, 50, {})
    picked = queue.select_batch(10)
    assert sorted(picked) == sorted([ids["Here"], ids["Unknown"], ids["Remote"]])
    with get_db() as conn:  # still waiting, not thrown away
        assert (
            conn.execute("SELECT status FROM scoring_queue WHERE job_id = ?", (ids["There"],)).fetchone()[0]
            == "pending"
        )
    update_settings({"location_dealbreaker": False})
    assert queue.select_batch(10) == [ids["There"]]


def test_cap_isnt_wasted_on_hidden_jobs():
    ids = _setup()
    queue.enqueue(ids["There"], 99, {})  # the best priority, but outside
    queue.enqueue(ids["Here"], 10, {})
    assert queue.select_batch(1) == [ids["Here"]]


def test_hidden_from_new_and_noteworthy_and_all_jobs_but_not_my_jobs():
    ids = _setup()
    for job_id in ids.values():
        add_to_digest(job_id)
    assert {j["title"] for j in get_digest(today_str())} == {"Here", "Unknown", "Remote"}
    with TestClient(app) as http:
        ranked = {j["title"]: j["dealbreakers"] for j in http.get("/api/jobs/ranked").json()}
        assert ranked == {"Here": [], "There": ["location"], "Unknown": [], "Remote": []}
        assert http.post(
            "/api/applications", json={"job_id": ids["There"], "status": "waiting"}
        ).status_code in (200, 201)
        assert [a["title"] for a in http.get("/api/applications").json()] == ["There"]
    assert ids["There"] not in rescore.job_ids("all_jobs")
    assert ids["There"] in rescore.job_ids("my_jobs")
    update_settings({"location_dealbreaker": False})
    # "There" is on My Jobs now (tracked above), which keeps it off New & Noteworthy either way.
    assert {j["title"] for j in get_digest(today_str())} == {"Here", "Unknown", "Remote"}
    with get_db() as conn:
        conn.execute(
            "DELETE FROM application_status_history WHERE application_id IN "
            "(SELECT id FROM applications WHERE job_id = ?)",
            (ids["There"],),
        )
        conn.execute("DELETE FROM applications WHERE job_id = ?", (ids["There"],))
    assert {j["title"] for j in get_digest(today_str())} == {"Here", "There", "Unknown", "Remote"}


def test_a_work_arrangement_never_hides_a_match():
    for location in [
        "Washington, DC (Hybrid)",
        "Hybrid - Arlington, VA",
        "New York, NY; Washington, DC (On-site)",
    ]:
        assert locations.verdict(location, None, DC) == "match", location
    assert locations.verdict("Hybrid", None, DC) == "unknown"


def test_the_daily_email_leaves_out_jobs_the_dealbreaker_hides():
    """The email reads New & Noteworthy, so a New York or San Francisco job never goes out
    (asked 2026-11-10, after an email listed some)."""
    from jobhunt.core import email_digest

    update_settings({"prefilter_locations": DC, "location_dealbreaker": True})
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (source,))
        for title, location in [("DC job", "Washington, DC"), ("NYC job", "New York City, NY (Hybrid)"),
                                ("SF job", "San Francisco, CA")]:  # fmt: skip
            job = conn.execute(
                "INSERT INTO jobs (source_id, dedupe_key, title, company, location) "
                "VALUES (?, ?, ?, 'Co', ?)",
                (source, title, title, location),
            ).lastrowid
            conn.execute(
                "INSERT INTO job_scores (user_id, job_id, model, interest_fit, interest_score, "
                "candidate_fit, candidate_fit_score, summary_text, rationale_text) "
                "VALUES (1, ?, 'x', 'High', 90, 'High', 90, 's', 'r')",
                (job,),
            )
            add_to_digest(job)
    assert [j["title"] for j in email_digest.qualifying_jobs()] == ["DC job"]


def test_location_check_previews_before_switching_on():
    _setup(dealbreaker=False)
    with TestClient(app) as http:
        check = http.get("/api/settings/location-check").json()
    assert check["enabled"] is False
    assert check["entries"][0]["entry"] == "DC" and check["entries"][0]["recognized"]
    assert "Washington, DC metro" in check["entries"][0]["matches"][0]
    assert check["preview"]["total"] == 4 and check["preview"]["hidden"] == 1
    assert check["preview"]["hidden_locations"] == [{"location": "Austin, TX", "count": 1}]
    # Jobs kept only because their location couldn't be placed are listed too (asked 2026-11-12).
    assert check["preview"]["unsure_locations"] == []
    with get_db() as conn:
        source = conn.execute("SELECT id FROM sources").fetchone()[0]
        for n, loc in enumerate(["Hybrid", "Hybrid", "Richmond"]):
            conn.execute(
                "INSERT INTO jobs (source_id, dedupe_key, title, company, location) "
                "VALUES (?, ?, 'T', 'Co', ?)",
                (source, f"u-{n}", loc),
            )
    with TestClient(app) as http:
        check = http.get("/api/settings/location-check").json()
    assert check["preview"]["unsure_locations"] == [
        {"location": "Hybrid", "count": 2},
        {"location": "Richmond", "count": 1},
    ]
