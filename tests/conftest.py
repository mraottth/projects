import pytest

from jobhunt.db import connection


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "jobhunt.db"
    monkeypatch.setattr(connection, "DATA_DIR", tmp_path)
    monkeypatch.setattr(connection, "DB_PATH", db_path)
    connection.init_db()
    # Adding a source kicks off a background fetch of it (daily_run.schedule_source_refresh); tests
    # that create sources mustn't reach real job boards. test_source_refresh.py calls it directly.
    from jobhunt.api import sources as sources_api

    monkeypatch.setattr(sources_api, "schedule_source_refresh", lambda source_id: None)
    yield
