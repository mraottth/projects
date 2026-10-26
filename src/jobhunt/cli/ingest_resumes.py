"""One-time local bulk import:
uv run python -m jobhunt.cli.ingest_resumes <folder> [--kind resume|cover_letter]

Walks a folder of .pdf/.docx/.md/.txt files, copies each into JOBHUNT_RESUME_DIR, extracts text, and
inserts a `resumes` row. Intended for seeding the local dev DB with historical files before the first
deploy; production ingestion is via the ResumesPage upload endpoint (Cloud Run can't read a local folder).
"""

import argparse
import shutil
import uuid
from pathlib import Path

from jobhunt.config import RESUME_STORAGE_DIR
from jobhunt.core.resume_parse import extract_text
from jobhunt.db.connection import get_db, init_db

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}


def ingest_folder(folder: Path, kind: str) -> list[str]:
    init_db()
    RESUME_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    imported = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            raw_text = extract_text(path)
        except Exception as exc:  # noqa: BLE001 - keep importing the rest of the folder on a bad file
            print(f"skip {path.name}: {exc}")
            continue

        stored_name = f"{uuid.uuid4().hex}{path.suffix.lower()}"
        stored_path = RESUME_STORAGE_DIR / stored_name
        shutil.copy2(path, stored_path)

        with get_db() as conn:
            conn.execute(
                """INSERT INTO resumes (kind, label, file_name, file_type, stored_path, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (kind, path.stem, path.name, path.suffix.lstrip("."), str(stored_path), raw_text),
            )
        imported.append(path.name)
        print(f"imported {path.name} ({len(raw_text)} chars)")
    return imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--kind", choices=["resume", "cover_letter"], default="resume")
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"not a directory: {args.folder}")

    imported = ingest_folder(args.folder, args.kind)
    print(f"\nimported {len(imported)} file(s) from {args.folder}")


if __name__ == "__main__":
    main()
