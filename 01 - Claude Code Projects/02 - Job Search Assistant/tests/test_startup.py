"""A cold start serves its first request without waiting on startup chores: the salary backfill
runs in the background, and the Anthropic SDK isn't imported until a Claude call needs it."""

import subprocess
import sys
import threading

from fastapi.testclient import TestClient

from jobhunt.api import main


def test_first_request_does_not_wait_for_the_salary_backfill(monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_backfill():
        started.set()
        release.wait(timeout=10)
        finished.set()
        return 0

    monkeypatch.setattr(main, "backfill_salaries", slow_backfill)
    with TestClient(main.app) as client:
        assert started.wait(timeout=5)
        assert client.get("/api/me").status_code == 200  # served while the backfill is still going
        assert not finished.is_set()
        release.set()
    assert finished.is_set()  # shutdown waited for it


def test_a_failed_backfill_never_stops_the_app(monkeypatch):
    def broken():
        raise RuntimeError("bad row")

    monkeypatch.setattr(main, "backfill_salaries", broken)
    with TestClient(main.app) as client:
        assert client.get("/api/me").status_code == 200


def test_importing_the_app_does_not_load_the_anthropic_sdk():
    code = "import sys, jobhunt.api.main; print('anthropic' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_keep_warm_ping_answers_without_signing_in_or_touching_data():
    """Cloud Scheduler calls it every 10 minutes during the warm window (Makefile's keep-warm)."""
    with TestClient(main.app) as client:
        resp = client.get("/internal/ping", headers={"X-Dev-User": "nobody@example.com"})
    assert resp.status_code == 200 and resp.json() == {"ok": True}
