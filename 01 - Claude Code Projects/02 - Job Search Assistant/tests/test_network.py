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
    def __init__(self, result):
        super().__init__(parse_result=FakeParseResponse(result))
        self.sent: list[str] = []
        parse = self.messages.parse

        def recording(**kwargs):
            self.sent.append(kwargs["messages"][0]["content"])
            return parse(**kwargs)

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


def _upload(http, monkeypatch, picks=()):
    client = Recorder(Picks(picks=list(picks)))
    monkeypatch.setattr(network, "get_client", lambda: client)
    resp = http.post(
        "/api/network/connections", files={"file": ("Connections.csv", CSV.encode(), "text/csv")}
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
        info, client = _upload(http, monkeypatch, picks=[Pick(number=1, why="Health data nonprofit.")])

    tiers = {level: [p["name"] for p in rows] for level, rows in info["tiers"].items()}
    # Ana recruits where you have a strong match; Ben works there in another role; Cy is in the job's
    # line of work where the only opening isn't a match; Di is at an organization Claude picked.
    assert tiers == {"High": ["Ana Lopez"], "Medium": ["Ben Ng", "Cy Park"], "Low": ["Di Ross"]}
    assert (
        info["tiers"]["High"][0]["why"]
        == "Analytics Lead is a strong match for you; they're on the hiring side."
    )
    # Claude saw organization names only, and not ones already judged by your jobs.
    assert len(client.sent) == 1
    sent = client.sent[0]
    assert "Brightwater Health Data" in sent and "Pizza Palace" in sent
    assert "Civic Signal" not in sent and "Ana" not in sent and "Lopez" not in sent
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
