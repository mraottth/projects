import pytest

from jobhunt.core import preference
from jobhunt.core.preference import rebuild_preference_profile
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeCreateResponse


def test_rebuild_requires_some_signal():
    with pytest.raises(ValueError, match="no feedback or application history"):
        rebuild_preference_profile()


def test_rebuild_from_feedback_and_applications(monkeypatch):
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company) VALUES ('k1', 'Engineer', 'Acme')"
        )
        job_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO feedback (job_id, interested, note) VALUES (?, 'interested', 'cool stack')",
            (job_id,),
        )
        cursor = conn.execute(
            "INSERT INTO applications (job_id, status) VALUES (?, 'interviewing')", (job_id,)
        )
        app_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO application_status_history (application_id, status) VALUES (?, 'interviewing')",
            (app_id,),
        )

    fake_client = FakeAnthropicClient(
        create_result=FakeCreateResponse("Prefers backend roles at small startups.", 200, 80)
    )
    monkeypatch.setattr(preference, "get_client", lambda: fake_client)

    text = rebuild_preference_profile()
    assert "startups" in text

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM preference_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
        calls = conn.execute("SELECT * FROM llm_calls WHERE call_type = 'profile_rebuild'").fetchall()

    assert row["based_on_feedback_count"] == 1
    assert row["based_on_application_events_count"] == 1
    assert len(calls) == 1
