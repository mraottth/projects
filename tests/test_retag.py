import json

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import retag
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeParseResponse

OLD = "Your ==campaign analytics background== fits, and ==mission-driven civic tech== is what you want."


def _client(tags):
    return FakeAnthropicClient(parse_result=FakeParseResponse(retag.HighlightTags(tags=tags)))


def test_retag_splices_tags_in_order_without_rewording():
    new = retag.retag_rationale(OLD, "High", "Med", client=_client(["f", "i"]))
    assert new == (
        "Your ==f:campaign analytics background== fits, and ==i:mission-driven civic tech== is what you want."
    )


def test_a_mismatched_answer_changes_nothing():
    assert retag.retag_rationale(OLD, "High", "Med", client=_client(["f"])) is None
    assert retag.retag_rationale("No highlights here.", "High", "Med", client=_client([])) is None
    already = "A ==i:tagged== phrase."
    assert retag.retag_rationale(already, "High", "Med", client=_client(["i"])) is None


def _score(job_key, rationale):
    with get_db() as conn:
        job_id = (
            conn.execute(
                "INSERT OR IGNORE INTO jobs (dedupe_key, title, company) VALUES (?, 'Analyst', 'Acme')",
                (job_key,),
            ).lastrowid
            or conn.execute("SELECT id FROM jobs WHERE dedupe_key = ?", (job_key,)).fetchone()[0]
        )
        return conn.execute(
            "INSERT INTO job_scores (job_id, model, interest_fit, interest_score, candidate_fit, "
            "candidate_fit_score, summary_text, rationale_text) "
            "VALUES (?, 'x', 'High', 80, 'Med', 50, 's', ?)",
            (job_id, rationale),
        ).lastrowid


def test_streamed_pass_retags_only_each_jobs_latest_untagged_summary(monkeypatch):
    superseded = _score("a", OLD)
    latest = _score("a", OLD)  # same job, newer score: only this one is displayed
    tagged = _score("b", "Already ==i:tagged==.")
    monkeypatch.setattr(retag, "get_client", lambda: _client(["f", "i"]))

    with TestClient(app) as client:
        assert client.get("/api/settings/retag-highlights").json()["pending"] == 1
        body = client.post("/api/settings/retag-highlights").text
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert events[-1] == {"type": "done", "retagged": 1, "skipped": 0}

    with get_db() as conn:
        texts = {
            r["id"]: r["rationale_text"] for r in conn.execute("SELECT id, rationale_text FROM job_scores")
        }
    assert texts[latest].startswith("Your ==f:campaign")
    assert texts[superseded] == OLD
    assert texts[tagged] == "Already ==i:tagged==."
