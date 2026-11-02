import json
import threading

from fastapi.testclient import TestClient

from jobhunt.api import pipeline as pipeline_api
from jobhunt.api.main import app
from jobhunt.pipeline import daily_run

import pytest as _pytest

from tests.fakes import give_profile


@_pytest.fixture(autouse=True)
def _profile():
    give_profile()



def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_stream_reports_progress_then_the_summary(monkeypatch):
    monkeypatch.setattr(pipeline_api, "PROGRESS_INTERVAL_SECONDS", 0.01)
    seen_cap = {}

    def fake_run(cap=None, only_user=None, fresh_hours=None):
        seen_cap["cap"] = cap
        return {"sources_polled": 3, "jobs_fetched": 40, "llm_scored_count": 12, "errors": []}

    monkeypatch.setattr(daily_run, "run_daily", fake_run)
    with TestClient(app) as client:
        resp = client.post("/api/pipeline/run-daily/stream?cap=20")
    events = _events(resp.text)
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert events[0]["type"] == "started"
    assert events[-1] == {
        "type": "done",
        "summary": {"sources_polled": 3, "jobs_fetched": 40, "llm_scored_count": 12, "errors": []},
    }
    assert seen_cap["cap"] == 20
    # The finished run is also visible to a page that lost the stream.
    progress = daily_run.get_progress()
    assert progress["finished_run_id"] == events[0]["run_id"]


def test_stream_reports_a_failed_run(monkeypatch):
    monkeypatch.setattr(pipeline_api, "PROGRESS_INTERVAL_SECONDS", 0.01)

    def boom(cap=None, only_user=None, fresh_hours=None):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(daily_run, "run_daily", boom)
    with TestClient(app) as client:
        events = _events(client.post("/api/pipeline/run-daily/stream").text)
    assert events[-1] == {"type": "error", "message": "database is locked"}


def test_second_start_follows_the_run_in_progress(monkeypatch):
    release = threading.Event()
    calls = []

    def slow_run(cap=None, only_user=None, fresh_hours=None):
        calls.append(cap)
        release.wait(5)
        return {}

    monkeypatch.setattr(daily_run, "run_daily", slow_run)
    first = daily_run.start_run_daily()
    second = daily_run.start_run_daily()
    release.set()
    assert first == second
    for _ in range(200):
        if daily_run.get_progress()["finished_run_id"] == first:
            break
        threading.Event().wait(0.01)
    assert calls == [None]
    assert daily_run.start_run_daily() != first  # a new run once that one has finished
    release.set()
