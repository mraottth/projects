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
            "INSERT INTO jobs (dedupe_key, title, company, description_text) VALUES (?, ?, ?, ?)",
            (f"k-{title}", title, company, description),
        ).lastrowid


def _insert_resume(raw_text, kind="resume", is_template=1, file_name="r.docx"):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO resumes (kind, file_name, file_type, stored_path, raw_text, is_template)
               VALUES (?, ?, 'docx', '/tmp/x', ?, ?)""",
            (kind, file_name, raw_text, is_template),
        )


def _events(resp):
    return [json.loads(line[len("data: "):]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_context_includes_job_profile_documents_and_tailored_docs():
    job_id = _insert_job(description="Own the analytics roadmap. SQL required.")
    _insert_resume("CURRENT RESUME TEXT")
    _insert_resume("OLD RESUME TEXT", is_template=0, file_name="old.docx")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO applications (job_id, status, notes) "
            "VALUES (?, 'interview', 'Panel on Friday')",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO generated_documents (job_id, kind, content_text) "
            "VALUES (?, 'resume', 'TAILORED TEXT')",
            (job_id,),
        )
        conn.execute("INSERT INTO candidate_profile (text, model) VALUES ('Matt leads data teams.', 'x')")
    update_settings({"looking_for_notes": "Mission-driven work"})

    context = jobby.build_context(job_id)
    for expected in [
        "Analytics Director", "Nava PBC", "Own the analytics roadmap", "Application status: interview",
        "Panel on Friday", "Matt leads data teams.", "Mission-driven work", "CURRENT RESUME TEXT",
        "TAILORED TEXT", "OLD RESUME TEXT",
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
