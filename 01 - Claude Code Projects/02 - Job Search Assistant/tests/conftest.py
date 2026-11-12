import pytest

from jobhunt.core import users
from jobhunt.db import connection


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "jobhunt.db"
    monkeypatch.setattr(connection, "DATA_DIR", tmp_path)
    monkeypatch.setattr(connection, "DB_PATH", db_path)
    # Uploaded/generated documents go under a temporary folder too, never data/resumes/.
    monkeypatch.setattr(users, "RESUME_STORAGE_DIR", tmp_path / "resumes")
    # Sign-in (api/auth.py): no IAP in tests; requests are the admin (user 1, created by init_db)
    # unless a test sends X-Dev-User. Direct calls into core/ run as the admin too.
    monkeypatch.delenv("IAP_AUDIENCE", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("JOBHUNT_ADMIN_EMAIL", raising=False)
    monkeypatch.setenv("JOBHUNT_DEV_EMAIL", connection.PLACEHOLDER_ADMIN_EMAIL)
    connection.init_db()
    # Adding a source kicks off a background fetch of it (daily_run.schedule_source_refresh); tests
    # that create sources mustn't reach real job boards. test_source_refresh.py calls it directly.
    from jobhunt.api import sources as sources_api

    monkeypatch.setattr(sources_api, "schedule_source_refresh", lambda source_id: None)
    # The title screen and batch scoring (asked 2026-11-12) call Claude in their own ways; tests that
    # exercise them (test_scoring_costs.py) switch them back on. Everything else scores live, unscreened.
    from jobhunt.pipeline import daily_run

    monkeypatch.setattr(
        daily_run, "screen_pending", lambda source_ids: {"screened": 0, "set_aside": 0, "errors": []}
    )
    monkeypatch.setattr(daily_run, "BATCH_TIME_BUDGET_SECONDS", 0)
    with users.acting_as(connection.ADMIN_ID):
        yield
