import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import jobby
from jobhunt.core.settings import update_settings
from jobhunt.db.connection import get_db


class FakeStream:
    """Mimics anthropic's MessageStream: iterating yields raw events; get_final_message() returns
    the assembled message. `texts` are streamed as text deltas; `searches` server_tool_use blocks."""

    def __init__(self, texts, stop_reason="end_turn", searches=0):
        self._texts = texts
        self._stop_reason = stop_reason
        self._searches = searches

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for _ in range(self._searches):
            yield SimpleNamespace(
                type="content_block_start", content_block=SimpleNamespace(type="server_tool_use")
            )
        for text in self._texts:
            yield SimpleNamespace(
                type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
            )

    def get_final_message(self):
        content = [SimpleNamespace(type="server_tool_use") for _ in range(self._searches)]
        content.append(SimpleNamespace(type="text", text="".join(self._texts)))
        return SimpleNamespace(
            content=content,
            stop_reason=self._stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )


class FakeStreamingClient:
    def __init__(self, *streams):
        self.calls = []
        self._streams = list(streams)
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self._streams.pop(0)


def _insert_job(title="Analytics Director", company="Nava PBC", description="Lead the data team.") -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company, description_text) VALUES (1, ?, ?, ?, ?)",
            (f"k-{title}", title, company, description),
        ).lastrowid


def _insert_resume(raw_text, kind="resume", is_template=1, file_name="r.docx"):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO resumes (user_id, kind, file_name, file_type, stored_path, raw_text, is_template)
               VALUES (1, ?, ?, 'docx', '/tmp/x', ?, ?)""",
            (kind, file_name, raw_text, is_template),
        )


def _events(resp):
    return [json.loads(line[len("data: ") :]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_context_includes_job_profile_documents_and_tailored_docs():
    job_id = _insert_job(description="Own the analytics roadmap. SQL required.")
    _insert_resume("CURRENT RESUME TEXT")
    _insert_resume("OLD RESUME TEXT", is_template=0, file_name="old.docx")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO applications (user_id, job_id, status, notes) "
            "VALUES (1, ?, 'interviewing', 'Panel on Friday')",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO generated_documents (user_id, job_id, kind, content_text) "
            "VALUES (1, ?, 'resume', 'TAILORED TEXT')",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO candidate_profile (user_id, text, model) VALUES (1, 'Matt leads data teams.', 'x')"
        )
    update_settings({"looking_for_notes": "Mission-driven work"})

    context = jobby.build_context(job_id)
    for expected in [
        "Analytics Director",
        "Nava PBC",
        "Own the analytics roadmap",
        "Application status: interview",
        "Panel on Friday",
        "Matt leads data teams.",
        "Mission-driven work",
        "CURRENT RESUME TEXT",
        "TAILORED TEXT",
        "OLD RESUME TEXT",
    ]:
        assert expected in context, expected
    # Current documents come before older ones.
    assert context.index("CURRENT RESUME TEXT") < context.index("OLD RESUME TEXT")


def test_general_context_has_no_job():
    context = jobby.build_context(None)
    assert "None selected" in context


def test_older_documents_are_trimmed_to_a_budget():
    _insert_resume("x" * 50_000, is_template=0, file_name="huge.docx")
    context = jobby.build_context(None)
    assert len(context) < 10_000
    assert "[... trimmed ...]" in context


def test_send_message_streams_saves_and_logs(monkeypatch):
    job_id = _insert_job()
    client = FakeStreamingClient(FakeStream(["Great ", "question!"], searches=1))
    monkeypatch.setattr(jobby, "get_client", lambda: client)

    with TestClient(app) as http:
        resp = http.post(
            f"/api/jobby/thread/messages?job_id={job_id}", json={"content": "Run a mock interview"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _events(resp)
        assert [e["type"] for e in events] == ["status", "delta", "delta", "done"]
        assert events[-1]["assistant"]["content"] == "Great question!"

        thread = http.get(f"/api/jobby/thread?job_id={job_id}").json()
    assert thread["job"]["title"] == "Analytics Director"
    assert [(m["role"], m["content"]) for m in thread["messages"]] == [
        ("user", "Run a mock interview"),
        ("assistant", "Great question!"),
    ]

    request = client.calls[0]
    assert request["messages"] == [{"role": "user", "content": "Run a mock interview"}]
    # The context is cached so later turns re-read it cheaply.
    assert request["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "Analytics Director" in request["system"][1]["text"]
    assert request["tools"][0]["type"] == "web_search_20260209"

    with get_db() as conn:
        call = dict(conn.execute("SELECT * FROM llm_calls").fetchone())
    assert call["job_id"] == job_id
    assert call["note"].startswith("jobby chat")


def test_conversation_history_is_sent_on_later_turns(monkeypatch):
    client = FakeStreamingClient(FakeStream(["First reply."]), FakeStream(["Second reply."]))
    monkeypatch.setattr(jobby, "get_client", lambda: client)

    with TestClient(app) as http:
        http.post("/api/jobby/thread/messages", json={"content": "Hi Jobby"})
        http.post("/api/jobby/thread/messages", json={"content": "Tell me more"})

    assert client.calls[1]["messages"] == [
        {"role": "user", "content": "Hi Jobby"},
        {"role": "assistant", "content": "First reply."},
        {"role": "user", "content": "Tell me more"},
    ]


def test_pause_turn_is_resumed_and_text_joined(monkeypatch):
    client = FakeStreamingClient(
        FakeStream(["Looking into the company. "], stop_reason="pause_turn", searches=1),
        FakeStream(["They raised a Series B."]),
    )
    monkeypatch.setattr(jobby, "get_client", lambda: client)

    with TestClient(app) as http:
        events = _events(http.post("/api/jobby/thread/messages", json={"content": "Research them"}))
    assert events[-1]["assistant"]["content"] == "Looking into the company. They raised a Series B."
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "assistant"


def test_failed_reply_keeps_nothing_so_the_user_can_retry(monkeypatch):
    from jobhunt.core.llm import LLMError

    class Failing:
        messages = None

        def __init__(self):
            self.messages = self

        def stream(self, **kwargs):
            raise LLMError("Jobby's reply failed: overloaded")

    monkeypatch.setattr(jobby, "get_client", lambda: Failing())
    with TestClient(app) as http:
        events = _events(http.post("/api/jobby/thread/messages", json={"content": "Hello"}))
        assert events == [{"type": "error", "message": "Jobby's reply failed: overloaded"}]
        assert http.get("/api/jobby/thread").json()["messages"] == []


def test_threads_are_per_job_and_clearable(monkeypatch):
    job_a, job_b = _insert_job("A"), _insert_job("B")
    client = FakeStreamingClient(FakeStream(["for A"]), FakeStream(["general"]))
    monkeypatch.setattr(jobby, "get_client", lambda: client)

    with TestClient(app) as http:
        http.post(f"/api/jobby/thread/messages?job_id={job_a}", json={"content": "a"})
        http.post("/api/jobby/thread/messages", json={"content": "g"})
        assert len(http.get(f"/api/jobby/thread?job_id={job_a}").json()["messages"]) == 2
        assert http.get(f"/api/jobby/thread?job_id={job_b}").json()["messages"] == []
        assert len(http.get("/api/jobby/thread").json()["messages"]) == 2

        assert http.delete(f"/api/jobby/thread?job_id={job_a}").status_code == 200
        assert http.get(f"/api/jobby/thread?job_id={job_a}").json()["messages"] == []
        assert len(http.get("/api/jobby/thread").json()["messages"]) == 2  # general untouched

        assert http.get("/api/jobby/thread?job_id=9999").status_code == 404


def test_general_chat_sees_jobbys_last_read_on_the_applications():
    from jobhunt.core.settings import update_settings

    assert "What you told them about their applications" not in jobby.build_context(None)
    insights = {
        "text": "Civic data roles are landing.",
        "generated_at": "2026-11-02T15:00:00+00:00",
        "applications": 3,
        "applied": 3,
    }
    update_settings({"application_insights": insights})
    context = jobby.build_context(None)
    assert "What you told them about their applications (2026-11-02)" in context
    assert "Civic data roles are landing." in context


def test_cover_letters_are_only_a_short_writing_sample():
    """Cover letters skewed Jobby (asked 2026-11-10): only some applications needed one and only
    some are uploaded. The current one goes in short, labeled a writing sample, after the resumes;
    older ones are left out."""
    _insert_resume("CURRENT RESUME")
    _insert_resume("OLDER RESUME", is_template=0, file_name="old-resume.docx")
    _insert_resume("CURRENT LETTER " + "x" * 10_000, kind="cover_letter", file_name="letter.docx")
    _insert_resume("OLD LETTER ONE", kind="cover_letter", is_template=0, file_name="l1.docx")
    _insert_resume("OLD LETTER TWO", kind="cover_letter", is_template=0, file_name="l2.docx")
    context = jobby.build_context(None)
    assert "CURRENT LETTER" in context and "sample of how they write" in context
    assert "OLD LETTER" not in context and "2 older cover letters not shown" in context
    assert context.index("OLDER RESUME") < context.index("CURRENT LETTER")
    letter = context[context.index("CURRENT LETTER") :]
    assert len(letter.split("###")[0]) < jobby.COVER_LETTER_SAMPLE_CHARS + 100


def _scored_job(title, interest, fit, score, reason="Because ==i:the mission== and ==f:your SQL==."):
    job_id = _insert_job(title=title, company=f"{title} Co")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO job_scores (user_id, job_id, model, interest_fit, interest_score, candidate_fit, "
            "candidate_fit_score, summary_text, rationale_text) VALUES (1, ?, 'x', ?, ?, ?, ?, 's', ?)",
            (job_id, interest, score, fit, score, reason),
        )
    return job_id


def test_the_highest_scoring_jobs_and_preferences_are_in_context():
    """What Jobby should lean on (asked 2026-11-10): the jobs that score highest for them and why,
    and what their marks say they want."""
    best = _scored_job("Best Role", "High", "High", 95)
    _scored_job("Good Role", "Med", "High", 70)
    _scored_job("Low Fit Role", "High", "Low", 90)
    dismissed = _scored_job("Dismissed Role", "High", "High", 99)
    current = _scored_job("Current Role", "High", "High", 98)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, job_id, interested) VALUES (1, ?, 'not_interested')", (dismissed,)
        )
        conn.execute(
            "INSERT INTO applications (user_id, job_id, status) VALUES (1, ?, 'interviewing')", (best,)
        )
        conn.execute(
            "INSERT INTO preference_profile (user_id, text, updated_at) "
            "VALUES (1, 'Prefers civic tech.', datetime('now'))"
        )
    context = jobby.build_context(current)
    top = context[context.index("score highest for them") : context.index("## Their resumes")]
    assert top.index("Best Role") < top.index("Good Role")
    assert "on My Jobs: interviewing" in top
    assert "Because the mission and your SQL." in top  # highlight markup removed
    for absent in ["Low Fit Role", "Dismissed Role", "Current Role"]:
        assert absent not in top, absent
    assert "Prefers civic tech." in context
