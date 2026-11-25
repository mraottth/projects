"""Cold start: the server answers while artifacts load in the background (no artifacts needed)."""

import threading
import time

from fastapi.testclient import TestClient

from goodrec.api import main


def test_api_answers_warming_until_loaded(monkeypatch):
    release = threading.Event()

    def slow_load():
        release.wait(10)
        main.state["art"] = type("Art", (), {"meta": type("Meta", (), {"n": 3})(), "user_factors": None})()

    monkeypatch.setattr(main, "_load_state", slow_load)
    with TestClient(main.app) as c:
        assert c.get("/api/health").json() == {"ok": False, "status": "warming"}
        r = c.get("/api/genres")
        assert r.status_code == 503 and r.json()["status"] == "warming" and r.headers["retry-after"] == "2"
        t = time.time()
        assert c.get("/api/health", params={"wait": 0.3}).json()["status"] == "warming"   # long-poll times out
        assert time.time() - t >= 0.25
        threading.Timer(0.2, release.set).start()
        assert c.get("/api/health", params={"wait": 5}).json() == {"ok": True, "status": "ready", "books": 3,
                                                                    "neighbor_users": 0}
        assert time.time() - t < 3                                                         # answered once loaded
        assert c.get("/api/nope").status_code != 503                                       # no longer held back
    main.state.clear()


def test_failed_load_is_reported_not_fatal(monkeypatch):
    def broken():
        raise FileNotFoundError("artifacts/manifest.json")

    monkeypatch.setattr(main, "_load_state", broken)
    with TestClient(main.app) as c:
        assert main.startup.ready.wait(5)
        assert c.get("/api/health").json() == {"ok": False, "status": "error"}
        r = c.get("/api/genres")
        assert r.status_code == 503 and r.json()["status"] == "error" and "retry-after" not in r.headers
