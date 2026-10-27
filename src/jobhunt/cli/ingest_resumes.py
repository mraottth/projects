"""One-time (re-runnable) local bulk import: uv run python -m jobhunt.cli.ingest_resumes <folder>

Recursively walks a real resume folder -- messy in the way these actually are: paired .docx/.pdf
of the same document, Word lock files, an "Old Resumes"-style dumping-ground subfolder, duplicate
content saved under unrelated names ("... copy.pdf"), and a few filenames that are clearly meant
as reusable templates. Handles all of that:

- Skips Word lock files (`~$*`), `.DS_Store`, and anything already under the generated-output
  folder (so the app never re-ingests its own tailored output as if it were real history).
- Prefers .docx over .pdf when both exist for the same stem in the same directory (cleaner text,
  and it's the one write_resume_edits/write_cover_letter_docx can actually edit in place).
- Dedupes by content hash (normalized extracted text), not filename, so "... copy.pdf" and
  renamed duplicates across reruns don't get re-imported as distinct documents.
- Classifies kind by filename ("cover" -> cover_letter, else resume) and flags is_template when
  the filename contains "template", and is_archived when any folder in its path is a bare "old"
  (e.g. "Old Resumes") -- not a substring match, so "Cold Outreach" doesn't false-positive.
- Persists the root folder as the `resume_library_path` setting, so generation knows where to
  save new tailored documents without separate configuration.

Production ingestion is still also available via the Profile-page upload endpoint (Cloud Run can't
read a local folder), but recency/archival/template metadata only comes from this CLI path.
"""

import argparse
import hashlib
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from jobhunt.config import RESUME_STORAGE_DIR
from jobhunt.core import settings as settings_module
from jobhunt.core.resume_parse import extract_text
from jobhunt.db.connection import get_db, init_db

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}
GENERATED_FOLDER_NAME = "AI Generated"


def _normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def _is_archived(relative_dir_parts: tuple[str, ...]) -> bool:
    for part in relative_dir_parts:
        if "old" in {w.lower() for w in part.split()}:
            return True
    return False


def _discover_files(root: Path) -> list[Path]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$") or path.name == ".DS_Store":
            continue
        if GENERATED_FOLDER_NAME in path.relative_to(root).parts[:-1]:
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        candidates.append(path)
    return candidates


def _prefer_docx_over_pdf(files: list[Path]) -> list[Path]:
    by_stem: dict[tuple[Path, str], list[Path]] = {}
    for f in files:
        by_stem.setdefault((f.parent, f.stem.lower()), []).append(f)

    selected = []
    for group in by_stem.values():
        docxs = [f for f in group if f.suffix.lower() == ".docx"]
        selected.append(docxs[0] if docxs else group[0])
    return selected


def _existing_hashes() -> set[str]:
    with get_db() as conn:
        rows = conn.execute("SELECT content_hash FROM resumes WHERE content_hash IS NOT NULL").fetchall()
    return {row["content_hash"] for row in rows}


def ingest_folder(root: Path) -> list[str]:
    init_db()
    RESUME_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    settings_module.update_settings({"resume_library_path": str(root.resolve())})

    seen_hashes = _existing_hashes()
    files = _prefer_docx_over_pdf(_discover_files(root))

    imported = []
    for path in sorted(files):
        try:
            raw_text = extract_text(path)
        except Exception as exc:  # noqa: BLE001 - keep importing the rest of the folder on a bad file
            print(f"skip {path.relative_to(root)}: could not extract text ({exc})")
            continue
        if not raw_text.strip():
            print(f"skip {path.relative_to(root)}: no extractable text")
            continue

        content_hash = _normalized_hash(raw_text)
        if content_hash in seen_hashes:
            print(f"skip {path.relative_to(root)}: duplicate content of an already-ingested file")
            continue
        seen_hashes.add(content_hash)

        relative_dir_parts = path.relative_to(root).parent.parts
        is_archived = _is_archived(relative_dir_parts)
        is_template = "template" in path.stem.lower()
        kind = "cover_letter" if "cover" in path.stem.lower() else "resume"
        source_modified_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

        stored_name = f"{uuid.uuid4().hex}{path.suffix.lower()}"
        stored_path = RESUME_STORAGE_DIR / stored_name
        shutil.copy2(path, stored_path)

        with get_db() as conn:
            conn.execute(
                """INSERT INTO resumes
                   (kind, label, file_name, file_type, stored_path, raw_text,
                    source_path, source_modified_at, is_archived, is_template, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    kind,
                    path.stem,
                    path.name,
                    path.suffix.lstrip("."),
                    str(stored_path),
                    raw_text,
                    str(path),
                    source_modified_at,
                    int(is_archived),
                    int(is_template),
                    content_hash,
                ),
            )
        imported.append(str(path.relative_to(root)))

        tags = [t for t, flag in (("archived", is_archived), ("template", is_template)) if flag]
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"imported [{kind}]{tag_str} {path.relative_to(root)} ({len(raw_text)} chars)")

    return imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"not a directory: {args.folder}")

    imported = ingest_folder(args.folder)
    print(f"\nimported {len(imported)} file(s) from {args.folder}")
    if imported:
        print("Run the candidate-profile rebuild (Profile page) to pick up the new documents.")


if __name__ == "__main__":
    main()
