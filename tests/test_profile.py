import pytest

from jobhunt.core import profile
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeCreateResponse


def _insert_resume(label, raw_text, source_modified_at, is_archived=0):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO resumes
               (kind, label, file_name, file_type, stored_path, raw_text,
                source_modified_at, is_archived)
               VALUES ('resume', ?, ?, 'docx', '/tmp/x.docx', ?, ?, ?)""",
            (label, f"{label}.docx", raw_text, source_modified_at, is_archived),
        )


def test_rebuild_requires_resumes():
    with pytest.raises(ValueError, match="no resumes/cover letters"):
        profile.rebuild_candidate_profile()


def test_rebuild_orders_by_effective_date_not_upload_order(monkeypatch):
    # Insert newest first to prove ordering comes from source_modified_at, not insertion order.
    _insert_resume("newest", "NEWEST CONTENT MARKER", "2026-01-01T00:00:00")
    _insert_resume("oldest", "OLDEST CONTENT MARKER", "2020-01-01T00:00:00")

    client = FakeAnthropicClient(create_result=FakeCreateResponse("profile text"))
    monkeypatch.setattr(profile, "get_client", lambda: client)
    captured = {}
    orig_create = client.messages.create

    def spy_create(**kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return orig_create(**kwargs)

    client.messages.create = spy_create

    profile.rebuild_candidate_profile()

    oldest_idx = captured["content"].index("OLDEST CONTENT MARKER")
    newest_idx = captured["content"].index("NEWEST CONTENT MARKER")
    assert oldest_idx < newest_idx


def test_rebuild_tags_archived_documents(monkeypatch):
    _insert_resume("old_one", "ARCHIVED CONTENT MARKER", "2020-01-01T00:00:00", is_archived=1)
    _insert_resume("current_one", "CURRENT CONTENT MARKER", "2026-01-01T00:00:00", is_archived=0)

    client = FakeAnthropicClient(create_result=FakeCreateResponse("profile text"))
    monkeypatch.setattr(profile, "get_client", lambda: client)
    captured = {}
    orig_create = client.messages.create

    def spy_create(**kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return orig_create(**kwargs)

    client.messages.create = spy_create

    profile.rebuild_candidate_profile()

    lines = captured["content"].split("\n\n---\n\n")
    archived_block = next(line for line in lines if "ARCHIVED CONTENT MARKER" in line)
    current_block = next(line for line in lines if "CURRENT CONTENT MARKER" in line)
    assert "archived" in archived_block.split("\n")[0]
    assert "archived" not in current_block.split("\n")[0]


def test_rebuild_stores_profile_and_logs_usage(monkeypatch):
    _insert_resume("r1", "Some resume content.", "2026-01-01T00:00:00")
    monkeypatch.setattr(
        profile, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse("built profile"))
    )

    text = profile.rebuild_candidate_profile()
    assert text == "built profile"
    assert profile.get_current_candidate_profile() == "built profile"

    with get_db() as conn:
        calls = conn.execute("SELECT * FROM llm_calls WHERE call_type = 'career_profile'").fetchall()
    assert len(calls) == 1
