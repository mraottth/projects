"""Your LinkedIn network: Connections.csv, tiers, suggested sources and counts (D-036)."""

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import network, users
from jobhunt.core.network import Pick, Picks
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeParseResponse

CSV = """Notes:
"When exporting your connection data, you may notice that some of the email addresses are missing."

First Name,Last Name,URL,Email Address,Company,Position,Connected On
Ana,Lopez,https://www.linkedin.com/in/ana,,Civic Signal,Senior Technical Recruiter,12 Mar 2021
Ben,Ng,https://www.linkedin.com/in/ben,,Civic Signal Inc.,Office Manager,01 Jan 2020
Cy,Park,https://www.linkedin.com/in/cy,,Open Ballot Lab,Director of Analytics,05 May 2019
Di,Ross,https://www.linkedin.com/in/di,,Brightwater Health Data,Data Scientist,07 Jul 2022
Ed,Fox,https://www.linkedin.com/in/ed,,Pizza Palace,Shift Lead,08 Aug 2018
,,,,,,
"""


class Recorder(FakeAnthropicClient):
    """Picks the organizations named in `picks` ({name: why}) by their number in the list sent."""

    def __init__(self, picks: dict[str, str]):
        super().__init__()
        self.sent: list[str] = []

        def recording(**kwargs):
            listing = kwargs["messages"][0]["content"]
            self.sent.append(listing)
            numbered = {line.split(". ", 1)[1]: int(line.split(".")[0]) for line in listing.splitlines()[1:]}
            chosen = [
                Pick(number=numbered[n], why=w[1], rating=w[0])
                if isinstance(w, tuple)
                else Pick(number=numbered[n], why=w)
                for n, w in picks.items()
                if n in numbered
            ]
            return FakeParseResponse(Picks(picks=chosen))

        self.messages.parse = recording


def _source(name="Board", company=None, user_id=1) -> int:
    with get_db() as conn:
        sid = conn.execute(
            "INSERT INTO sources (kind, name, company_name, config_json) VALUES ('custom_board', ?, ?, '{}')",
            (name, company),
        ).lastrowid
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", (user_id, sid))
    return sid


def _job(source_id, title, company, interest=None, fit=None) -> int:
    with get_db() as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (source_id, dedupe_key, title, company) VALUES (?, ?, ?, ?)",
            (source_id, f"{title}-{company}", title, company),
        ).lastrowid
        if interest:
            conn.execute(
                "INSERT INTO job_scores (user_id, job_id, interest_fit, interest_score, candidate_fit, "
                "candidate_fit_score, summary_text, rationale_text, model) "
                "VALUES (1, ?, ?, 70, ?, 70, 's', 'r', 'm')",
                (job_id, interest, fit),
            )
    return job_id


def _upload(http, monkeypatch, picks=None, csv=CSV):
    client = Recorder(picks or {})
    monkeypatch.setattr(network, "get_client", lambda: client)
    resp = http.post(
        "/api/network/connections", files={"file": ("Connections.csv", csv.encode(), "text/csv")}
    )
    assert resp.status_code == 200, resp.text
    return resp.json(), client


def test_parsing_skips_linkedins_notes_and_blank_rows():
    rows = network.parse_csv(CSV.encode("utf-8-sig"))
    assert [r["first_name"] for r in rows] == ["Ana", "Ben", "Cy", "Di", "Ed"]
    assert rows[0]["connected_on"] == "2021-03-12"
    assert rows[0]["org_key"] == rows[1]["org_key"] == "civic signal"


def test_a_file_that_isnt_connections_is_refused():
    with TestClient(app) as http:
        resp = http.post("/api/network/connections", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert resp.status_code == 400 and "Connections.csv" in resp.json()["detail"]


def test_tiers_and_suggested_sources(monkeypatch):
    board = _source()
    _source("Open Ballot Lab", company="Open Ballot Lab")  # followed already: not suggested
    _job(board, "Analytics Lead", "Civic Signal", "High", "High")
    _job(board, "Director, Analytics", "Open Ballot Lab", "Low", "Low")
    with TestClient(app) as http:
        info, client = _upload(http, monkeypatch, picks={"Brightwater Health Data": "Health data nonprofit."})

    tiers = {level: sorted(p["name"] for p in rows) for level, rows in info["tiers"].items()}
    # Ana recruits where you have a strong match; Ben works there in another role; Cy is in the job's
    # line of work where the only opening isn't a match; Di is at an organization Claude picked.
    # Di's organization is only "of interest" to Claude (2026-11-13: that's Medium, not Low).
    assert tiers == {"High": ["Ana Lopez"], "Medium": ["Ben Ng", "Cy Park", "Di Ross"], "Low": []}
    assert (
        info["tiers"]["High"][0]["why"]
        == "Analytics Lead is a strong match for you; they're on the hiring side."
    )
    # Claude saw organization names only, and not ones already judged by your jobs.
    assert len(client.sent) == 1
    sent = client.sent[0]
    assert "Brightwater Health Data" in sent and "Pizza Palace" in sent
    # Every organization is rated now (2026-11-13), by name only: never people.
    assert "Civic Signal" in sent and "Ana" not in sent and "Lopez" not in sent
    names = [s["name"] for s in info["suggested_sources"]]
    assert names == ["Civic Signal", "Brightwater Health Data"]  # Open Ballot Lab is followed


def test_counts_on_cards_and_people_on_the_job_page(monkeypatch):
    board = _source()
    job = _job(board, "Analytics Lead", "Civic Signal, Inc.", "High", "Med")
    with TestClient(app) as http:
        _upload(http, monkeypatch)
        [ranked] = http.get("/api/jobs/ranked").json()
        assert ranked["connections"] == 2
        people = http.get(f"/api/network/job/{job}").json()
        assert [(p["name"], p["tier"]) for p in people] == [("Ana Lopez", "High"), ("Ben Ng", "Medium")]


def test_a_failed_claude_call_still_lists_the_free_tiers(monkeypatch):
    board = _source()
    _job(board, "Analytics Lead", "Civic Signal", "High", "High")
    monkeypatch.setattr(network, "get_client", lambda: (_ for _ in ()).throw(network.LLMError("down")))
    with TestClient(app) as http:
        info = http.post(
            "/api/network/connections", files={"file": ("Connections.csv", CSV.encode(), "text/csv")}
        ).json()
    assert [p["name"] for p in info["tiers"]["High"]] == ["Ana Lopez"]
    assert info["picks_error"] == "down"


def test_connections_are_private_and_can_be_removed(monkeypatch):
    bea = users.create_user("bea@example.com", "Bea", status="approved")["id"]
    with TestClient(app) as http:
        _upload(http, monkeypatch)
        theirs = http.get("/api/network", headers={"X-Dev-User": "bea@example.com"}).json()
        assert theirs["count"] == 0
        job = _job(_source(user_id=bea), "Analyst", "Civic Signal")
        assert http.get(f"/api/network/job/{job}", headers={"X-Dev-User": "bea@example.com"}).json() == []
        assert http.get("/api/network").json()["count"] == 5
        assert http.delete("/api/network/connections").json()["count"] == 0


# ---- the rules after the owner's review (2026-11-13)

HEAD = "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"


def _people(rows: str) -> str:
    return HEAD + rows


def _tier_of(info, name):
    for level, rows in info["tiers"].items():
        if any(p["name"] == name for p in rows):
            return level
    return None


def test_leaders_are_high_only_at_good_fit_organizations(monkeypatch):
    board = _source()
    _source("Blue Rose", company="Blue Rose")  # the source's short name
    _job(board, "Field Organizer", "US Digital Response")  # an opening, not a match for you
    csv = _people(
        "Amy,Ceo,,,Blue Rose Research,CEO,01 Jan 2020\n"
        "Bo,Chief,,,U.S. Digital Response,Chief Executive Officer,01 Jan 2020\n"
        "Cal,Lead,,,Somewhere Else,CEO,01 Jan 2020\n"
        "Dee,Dir,,,US Digital Response,Director of Finance,01 Jan 2020\n"
    )
    with TestClient(app) as http:
        info, _ = _upload(
            http, monkeypatch, picks={"Blue Rose Research": ("fit", "Election data research.")}, csv=csv
        )
    assert _tier_of(info, "Amy Ceo") == "High"  # a good fit, and "Blue Rose" source = "Blue Rose Research"
    # In your sources but not a good fit (2026-11-13: High is for good-fit organizations only).
    assert _tier_of(info, "Bo Chief") == "Medium"  # and "U.S." = "US"
    assert _tier_of(info, "Dee Dir") == "Medium"  # any role at an organization in your sources
    assert _tier_of(info, "Cal Lead") is None  # not in your sources and not rated
    assert "Somewhere Else" in {p["company"] for p in info["not_listed"]}


def test_a_leader_at_a_good_fit_organization_is_high_and_of_interest_is_medium(monkeypatch):
    csv = _people(
        "Ed,Ed,,,Common Ledger,Founder & CEO,01 Jan 2020\n"
        "Flo,Off,,,Common Ledger,Office Manager,01 Jan 2020\n"
        "Gil,Gil,,,Neighbor Org,Founder & CEO,01 Jan 2020\n"
    )
    picks = {
        "Common Ledger": ("fit", "Civic data nonprofit."),
        "Neighbor Org": ("interest", "Adjacent field."),
    }
    with TestClient(app) as http:
        info, _ = _upload(http, monkeypatch, picks=picks, csv=csv)
    assert _tier_of(info, "Flo Off") == "Medium"  # a good-fit organization, another role
    assert _tier_of(info, "Gil Gil") == "Medium"  # of interest only
    assert _tier_of(info, "Ed Ed") == "High"
    [org] = [o for o in info["suggested_sources"] if o["name"] == "Common Ledger"]
    assert (org["name"], org["tier"], org["leader"]) == ("Common Ledger", "High", True)


def test_every_good_job_counts_and_unrated_ones_in_your_line_of_work(monkeypatch):
    board = _source()
    _job(board, "Office Manager", "Civic Signal", "High", "High")
    _job(board, "Director of Data", "Civic Signal", "High", "High")  # the one Fay fits
    _job(board, "Senior Data Analyst", "Open Ballot Lab")  # unrated, your line of work
    with TestClient(app) as http:
        http.put("/api/settings", json={"prefilter_titles_include": ["Data Director"]})
        csv = _people(
            "Fay,Fit,,,Civic Signal,Head of Analytics,01 Jan 2020\n"
            "Gus,Hire,,,Open Ballot Lab,Talent Partner,01 Jan 2020\n"
            "Hal,Peep,,,Open Ballot Lab,People Analytics Manager,01 Jan 2020\n"
        )
        info, _ = _upload(http, monkeypatch, csv=csv)
    assert _tier_of(info, "Fay Fit") == "High"
    [fay] = [p for p in info["tiers"]["High"] if p["name"] == "Fay Fit"]
    assert fay["job_title"] == "Director of Data"  # the job she fits, not the first one there
    assert _tier_of(info, "Gus Hire") == "High"  # recruiter where an unrated data job waits
    # "People Analytics" is data work, not hiring: the same line of work as the data job.
    assert _tier_of(info, "Hal Peep") == "High"


def test_a_set_aside_job_doesnt_count_and_generic_words_dont_match(monkeypatch):
    board = _source()
    job = _job(board, "Data Program Manager", "Civic Signal")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO scoring_queue (user_id, job_id, priority, status, screen) "
            "VALUES (1, ?, 1, 'pending', 'drop')",
            (job,),
        )
    with TestClient(app) as http:
        http.put("/api/settings", json={"prefilter_titles_include": ["Data Director"]})
        csv = _people("Ivy,Prog,,,Civic Signal,Program Manager,01 Jan 2020\n")
        info, _ = _upload(http, monkeypatch, csv=csv)
    # Not a job you'd want (set aside), so not High; but Civic Signal is in your sources: Medium.
    assert _tier_of(info, "Ivy Prog") == "Medium"
    assert "line of work" not in [p for p in info["tiers"]["Medium"] if p["name"] == "Ivy Prog"][0]["why"]


def test_aligned_titles_are_medium_and_your_line_of_work_low(monkeypatch):
    with TestClient(app) as http:
        http.put("/api/settings", json={"prefilter_titles_include": ["Analytics Director"]})
        csv = _people(
            "Jo,Data,,,Tiny Nonprofit,Director of Data Science,01 Jan 2020\n"
            "Kit,Data,,,Tiny Nonprofit,Data Analyst,01 Jan 2020\n"
        )
        info, _ = _upload(http, monkeypatch, csv=csv)
    assert _tier_of(info, "Jo Data") == "Medium"  # a title close to what you're after, and as senior
    assert _tier_of(info, "Kit Data") == "Low"  # your line of work, a different level


def test_organization_names_link_loosely_but_not_generically():
    from jobhunt.core.network import _same_org, org_key

    def same(a, b):
        return _same_org(org_key(a), org_key(b))

    assert same("Blue Rose", "Blue Rose Research")
    assert same("Blue Rose Careers", "Blue Rose Research, Inc.")
    assert same("US Digital Response", "U.S. Digital Response")
    assert same("Civic Signal (acquired by X)", "Civic Signal")
    assert same("The Data Foundation", "Data Foundation")
    assert not same("Data", "Data Society")
    assert not same("Civic", "Civic Signal")


# ---- your current organization (asked 2026-11-13)


def _resume(text: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO resumes (user_id, kind, file_name, file_type, stored_path, raw_text, is_template) "
            "VALUES (1, 'resume', 'r.docx', 'docx', '/tmp/r.docx', ?, 1)",
            (text,),
        )


COLLEAGUES = _people(
    "A,One,,,DCCC,Data Director,01 Jan 2020\n"
    "B,Two,,,DCCC,Recruiter,01 Jan 2020\n"
    "C,Three,,,DCCC,CEO,01 Jan 2020\n"
    "D,Four,,,Blue Rose Research,CEO,01 Jan 2020\n"
    "E,Five,,,Blue Rose Research,Analyst,01 Jan 2020\n"
    "F,Six,,,Blue Rose Research,Analyst,01 Jan 2020\n"
)


def test_people_at_your_current_organization_are_left_out(monkeypatch):
    _source("Blue Rose", company="Blue Rose")
    _source("DCCC", company="DCCC")
    # The latest job comes first on the resume, so DCCC (not Blue Rose, listed later) is current.
    _resume(
        "Jordan Rivera\nDCCC — Director of Analytics, 2022–present\nBlue Rose Research — Analyst, 2019–2022"
    )
    with TestClient(app) as http:
        info, client = _upload(http, monkeypatch, csv=COLLEAGUES)
        assert (info["current_org"], info["current_org_guessed"], info["colleagues"]) == ("DCCC", True, 3)
        everyone = [p["company"] for rows in info["tiers"].values() for p in rows] + [
            p["company"] for p in info["not_listed"]
        ]
        assert "DCCC" not in everyone and "Blue Rose Research" in everyone
        assert all("DCCC" not in sent for sent in client.sent)  # never sent to Claude

        # Your own choice sticks, and "" leaves nobody out.
        http.put(
            "/api/settings", json={"network_current_org": "Blue Rose", "network_current_org_guessed": False}
        )
        info = http.get("/api/network").json()
        assert (info["current_org"], info["current_org_guessed"], info["colleagues"]) == (
            "Blue Rose",
            False,
            3,
        )
        http.put("/api/settings", json={"network_current_org": ""})
        assert http.get("/api/network").json()["colleagues"] == 0


def test_no_guess_without_a_resume_naming_them(monkeypatch):
    _resume("A resume that names none of these organizations.")
    with TestClient(app) as http:
        info, _ = _upload(http, monkeypatch, csv=COLLEAGUES)
    assert (info["current_org"], info["colleagues"]) == (None, 0)


def test_the_guess_reads_an_acronym_spelled_out_on_the_resume(monkeypatch):
    _resume("Democratic Congressional Campaign Committee, Director of Analytics\nBlue Rose Research, Analyst")
    with TestClient(app) as http:
        info, _ = _upload(http, monkeypatch, csv=COLLEAGUES)
    assert (info["current_org"], info["colleagues"]) == ("DCCC", 3)


def test_old_ratings_are_flagged_and_re_rating_replaces_them(monkeypatch):
    with TestClient(app) as http:
        _upload(http, monkeypatch, csv=_people("Ed,Ed,,,Common Ledger,Founder & CEO,01 Jan 2020\n"))
        # As saved before fit/interest: no rating, no version. They read as "of interest".
        http.put(
            "/api/settings",
            json={
                "network_orgs": {"common ledger": {"name": "Common Ledger", "why": "Civic."}},
                "network_orgs_version": None,
            },
        )
        info = http.get("/api/network").json()
        assert info["ratings_outdated"] and _tier_of(info, "Ed Ed") == "Medium"
        monkeypatch.setattr(
            network, "get_client", lambda: Recorder({"Common Ledger": ("fit", "Civic data.")})
        )
        info = http.post("/api/network/rate").json()
    assert not info["ratings_outdated"] and _tier_of(info, "Ed Ed") == "High"
