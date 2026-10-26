import pytest

import jobhunt.db.connection as connection


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "jobhunt.db"
    monkeypatch.setattr(connection, "DATA_DIR", tmp_path)
    monkeypatch.setattr(connection, "DB_PATH", db_path)
    connection.init_db()
    yield
