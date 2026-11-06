import csv
import io
import sqlite3
from datetime import date, timedelta

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import backup
from jobhunt.core.reset import delete_all_data
from jobhunt.db import connection
from jobhunt.db.connection import get_db


def _track_job(title="Data Lead", company="Civic", status="waiting") -> int:
    with get_db() as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company) VALUES (1, ?, ?, ?)",
            (f"{title}-{company}", title, company),
        ).lastrowid
        conn.execute(
            "INSERT INTO applications (user_id, job_id, status, applied_at) VALUES (1, ?, ?, '2026-10-01')",
            (job_id, status),
        )
    return job_id


def _applications_in(path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [
            r[0] for r in conn.execute("SELECT j.title FROM applications a JOIN jobs j ON j.id = a.job_id")
        ]
    finally:
        conn.close()


def test_snapshot_copies_my_jobs():
    _track_job()
    path = backup.snapshot()
    assert path.name == f"jobhunt-{date.today().isoformat()}.db"
    assert _applications_in(path) == ["Data Lead"]


def test_a_second_daily_snapshot_replaces_the_first():
    _track_job()
    backup.snapshot()
    _track_job("Analyst", "Shop")
    path = backup.snapshot()
    assert sorted(_applications_in(path)) == ["Analyst", "Data Lead"]
    assert len(list(backup.backup_dir().glob("jobhunt-*.db"))) == 1


def test_old_daily_backups_are_pruned_and_event_backups_capped():
    folder = backup.backup_dir()
    folder.mkdir(parents=True)
    today = date(2026, 11, 2)
    for days_ago in (0, 29, 30, 31, 60):
        (folder / f"jobhunt-{(today - timedelta(days=days_ago)).isoformat()}.db").write_bytes(b"x")
    for i in range(12):
        (folder / f"jobhunt-2026-10-{i + 10:02d}-120000-before-reset.db").write_bytes(b"x")
    (folder / "notes.txt").write_text("not a backup")
    backup.prune(today)
    names = sorted(p.name for p in folder.iterdir())
    assert "jobhunt-2026-11-02.db" in names and "jobhunt-2026-10-03.db" in names  # 0 and 30 days old
    assert "jobhunt-2026-10-02.db" not in names and "jobhunt-2026-09-03.db" not in names
    events = [n for n in names if n.endswith("before-reset.db")]
    assert len(events) == backup.KEEP_EVENTS and events[0] == "jobhunt-2026-10-12-120000-before-reset.db"
    assert "notes.txt" in names


def test_start_over_keeps_a_backup_with_my_jobs_in_it():
    _track_job()
    result = delete_all_data()
    assert result["backup"].endswith("-before-reset.db")
    kept = backup.backup_dir() / result["backup"]
    assert _applications_in(kept) == ["Data Lead"]
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 0


def test_daily_run_takes_a_snapshot_first(monkeypatch):
    from jobhunt.pipeline import daily_run

    calls = []
    monkeypatch.setattr(daily_run.backup, "snapshot", lambda *a: calls.append(a))
    monkeypatch.setattr(
        daily_run.queue, "requeue_stranded", lambda: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    try:
        daily_run._run_daily("2026-11-02T00:00:00", cap=0)
    except RuntimeError:
        pass
    assert calls == [()]


def test_a_failed_snapshot_never_raises(monkeypatch):
    monkeypatch.setattr(backup, "copy_database", lambda target: (_ for _ in ()).throw(OSError("disk full")))
    assert backup.snapshot() is None


def test_no_database_means_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(connection, "DB_PATH", tmp_path / "missing.db")
    assert backup.snapshot() is None


def test_download_backup_is_a_working_database():
    _track_job()
    with TestClient(app) as client:
        resp = client.get("/api/settings/backup")
        listing = client.get("/api/settings/backups").json()
    assert resp.status_code == 200
    assert "jobhunt-backup-" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"SQLite format 3")
    assert listing == {"backups": []}  # downloading doesn't add to backups/


def test_my_jobs_csv_export():
    _track_job("Data Lead", "Civic", "rejected_no_interview")
    _track_job("Analyst", "Shop", "interviewing")
    with TestClient(app) as client:
        resp = client.get("/api/applications/export.csv")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    assert {(r["Title"], r["Organization"], r["Status"]) for r in rows} == {
        ("Data Lead", "Civic", "Rejected - no interview"),
        ("Analyst", "Shop", "Interviewing"),
    }


def test_seed_db_copies_the_old_database_once(tmp_path):
    from jobhunt.cli import seed_db

    _track_job()
    target = tmp_path / "local" / "jobhunt.db"
    assert seed_db.main([str(connection.DB_PATH), str(target)]) == 0
    assert _applications_in(target) == ["Data Lead"]
    assert seed_db.main([str(connection.DB_PATH), str(target)]) == 1  # never overwrites
    assert seed_db.main([str(tmp_path / "missing.db"), str(tmp_path / "x.db")]) == 1


def test_database_outside_the_data_folder(tmp_path, monkeypatch):
    # Production keeps the live database on local disk (Litestream, D-020) and backups/ on the bucket.
    monkeypatch.setattr(connection, "DB_PATH", tmp_path / "local" / "db" / "jobhunt.db")
    connection.init_db()
    _track_job()
    path = backup.snapshot()
    assert path.parent == connection.DATA_DIR / "backups"
    assert _applications_in(path) == ["Data Lead"]


def test_a_backup_is_built_locally_then_copied_in_one_pass(monkeypatch):
    # The Cloud Storage mount can't take SQLite writing a database (and its journal) in place, so
    # nothing but a finished file may be written into backups/ (2026-11-02 mount errors).
    _track_job()
    opened = []
    real_connect = backup.sqlite3.connect
    monkeypatch.setattr(
        backup.sqlite3,
        "connect",
        lambda path, *a, **k: opened.append(str(path)) or real_connect(path, *a, **k),
    )
    path = backup.snapshot()
    assert not any(p.startswith(str(backup.backup_dir())) for p in opened)
    assert _applications_in(path) == ["Data Lead"]
    assert not list(backup.backup_dir().glob("*.part"))
