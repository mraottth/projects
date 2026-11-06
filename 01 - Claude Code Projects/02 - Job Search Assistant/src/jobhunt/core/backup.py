"""Snapshots of the database, so My Jobs and everything else survive a bad write, a mistaken
"Start over", or a migration gone wrong.

The live database is one SQLite file on a Cloud Storage volume (see CLAUDE.md -> Deploy), and
before this nothing else ever copied it. `snapshot()` uses SQLite's online backup API, which gives
a consistent copy even while the app is writing, into `<data dir>/backups/`:
- `jobhunt-YYYY-MM-DD.db`, taken at the start of every daily run (a later run the same day
  replaces it), kept for `KEEP_DAYS` days;
- `jobhunt-YYYY-MM-DD-HHMMSS-<reason>.db` before "Start over" or a data migration, the newest
  `KEEP_EVENTS` kept.
`backups/` sits outside `RESUME_STORAGE_DIR`, so "Start over" (core/reset.py) never deletes it.
In production the live database is on the server's own disk, streamed to Cloud Storage by
Litestream (D-020), while `backups/` stays on the bucket mount: these snapshots are a second,
independent layer on top of Litestream's copy.
A failed snapshot is logged and never stops the run, reset or startup it guards."""

import logging
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from jobhunt.db import connection

log = logging.getLogger(__name__)

KEEP_DAYS = 30
KEEP_EVENTS = 10
_DAILY = re.compile(r"^jobhunt-(\d{4}-\d{2}-\d{2})\.db$")
_EVENT = re.compile(r"^jobhunt-\d{4}-\d{2}-\d{2}-\d{6}-[a-z-]+\.db$")


def backup_dir() -> Path:
    # Read at call time: tests point connection.DATA_DIR at a temporary folder.
    return connection.DATA_DIR / "backups"


def copy_sqlite(source_path: Path, target: Path) -> None:
    """A consistent copy of the SQLite database at `source_path` (SQLite's online backup API, safe
    while another connection is writing to it).

    The copy is built in a local temporary file and then written to `target` in one sequential
    pass. `target` is usually on the Cloud Storage mount (backups/), which can't take SQLite's
    random-access writes and journal file: writing the backup there directly failed live with
    "BufferedWriteHandler.OutOfOrderError" on the journal and the backup itself (2026-11-02 logs,
    the same mount errors that broke the database before D-020). A plain copy is fine there."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "copy.db"
        source = sqlite3.connect(source_path)
        dest = sqlite3.connect(local)
        try:
            source.backup(dest)
        finally:
            dest.close()
            source.close()
        partial = target.with_name(target.name + ".part")
        shutil.copyfile(local, partial)
        os.replace(partial, target)


def copy_database(target: Path) -> None:
    """A consistent copy of the live database at `target`."""
    copy_sqlite(connection.DB_PATH, target)


def snapshot(reason: str | None = None) -> Path | None:
    """Back the database up into backups/ and prune old copies. Returns the file written, or None
    when there's no database yet or the copy failed (logged, never raised)."""
    if not connection.DB_PATH.exists():
        return None
    now = datetime.now()
    name = f"jobhunt-{now:%Y-%m-%d}.db" if reason is None else f"jobhunt-{now:%Y-%m-%d-%H%M%S}-{reason}.db"
    target = backup_dir() / name
    try:
        copy_database(target)
        prune()
    except Exception:  # noqa: BLE001 - a backup must never take the app down with it
        log.exception("database snapshot to %s failed", target)
        return None
    return target


def prune(today: date | None = None) -> None:
    folder = backup_dir()
    if not folder.exists():
        return
    cutoff = (today or date.today()) - timedelta(days=KEEP_DAYS)
    events = []
    for path in folder.iterdir():
        daily = _DAILY.match(path.name)
        if daily and date.fromisoformat(daily.group(1)) < cutoff:
            path.unlink()
        elif _EVENT.match(path.name):
            events.append(path)
    for path in sorted(events, key=lambda p: p.name)[:-KEEP_EVENTS]:
        path.unlink()
