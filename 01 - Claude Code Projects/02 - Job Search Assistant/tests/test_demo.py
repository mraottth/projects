"""The public demo's recording (demo/record.py, asked 2026-11-11): it covers every GET the frontend
makes, its downloads are real files, and it holds only fictional data."""

import json
import re
import zipfile
from pathlib import Path

import pytest

from demo import record as demo_record

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "src"

# Paths the frontend only ever writes to (POST/PUT/PATCH/DELETE) or links to as a file. The demo
# refuses every write, so these need no recording.
NOT_RECORDED = {
    "/api/access/request",
    "/api/admin/scoring-experiment",
    "/api/admin/users",
    "/api/admin/users/{}",
    "/api/jobs/manual",
    "/api/jobs/{}/confirm",
    "/api/settings/title-screen/{}/keep",
    "/api/onboarding/sector-suggestions",
    "/api/settings/title-screen/run?recheck={}",
    "/api/jobs/{}/score",
    "/api/jobs/{}/feedback",
    "/api/applications/{}",
    "/api/settings/preference-profile/rebuild",
    "/api/resumes/upload",
    "/api/resumes/{}",
    "/api/resumes/profile/rebuild",
    "/api/resumes/profile/revise",
    "/api/onboarding/generate-cover-letter",
    "/api/sources/resolve",
    "/api/sources/{}",
    "/api/sources/{}/test",
    "/api/settings/test-email",
    "/api/settings/delete-account",
    "/api/settings/delete-all-data",
    "/api/settings/backup",
    "/api/applications/export.csv",
    "/api/generate/resume",
    "/api/generate/cover-letter",
    "/api/generate/documents/{}/download",
    "/api/resumes/{}/download",
    "/api/jobby/threads/{}",
    "/api/digest/refresh/stream",
    "/api/discovery/suggest/start",
    "/api/discovery/suggest-boards/start",
    "/api/pipeline/run-daily",
    "/api/pipeline/run-daily/stream",
    "/api/pipeline/run-daily/stream{}",
    "/api/jobby/thread/messages{}",
    "/api/rescore/{}",
    "/api/discovery/tasks/{}",  # polled only after a start, which the demo refuses
}


@pytest.fixture
def recording(tmp_path, monkeypatch):
    for key in demo_record.CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in demo_record.DEMO_ENV.items():
        monkeypatch.setenv(key, value)
    out = tmp_path / "demo-build"
    return out, demo_record.record(out)


def _frontend_paths() -> set[str]:
    """Every /api path written in the frontend, with ${...} parts as {}."""
    found = set()
    for file in FRONTEND.rglob("*.ts*"):
        for m in re.finditer(r"""["`](/api/[^"`]*)["`]""", file.read_text()):
            path = re.sub(r"\$\{[^}]*\}", "{}", m.group(1))
            found.add(path.split("${")[0] + "{}" if "${" in path else path)  # a nested `...` inside ${}
    return found


def test_every_get_the_frontend_makes_is_recorded(recording):
    _, data = recording
    recorded = list(data["responses"])
    missing = []
    for path in sorted(_frontend_paths() - NOT_RECORDED):
        pattern = re.compile("^" + re.escape(path).replace(r"\{\}", "[^/]*") + "$")
        if not any(pattern.match(p) for p in recorded):
            missing.append(path)
    assert not missing, f"add these to demo/record.py (or NOT_RECORDED if write-only): {missing}"


def test_recorded_answers_are_the_real_apps(recording):
    _, data = recording
    statuses = {p: r["status"] for p, r in data["responses"].items()}
    assert {p for p, s in statuses.items() if s != 200} == {"/api/settings/backups"}  # admin only
    me = data["responses"]["/api/me"]["body"]
    assert me["name"] == "Jordan Rivera" and not me["is_admin"]
    assert len(data["responses"]["/api/digest/today"]["body"]["entries"]) >= 6
    assert data["responses"]["/api/settings/title-screen"]["status"] == 200
    threads = data["responses"]["/api/jobby/threads"]["body"]
    assert len(threads) == 4
    for t in threads:
        assert data["responses"][f"/api/jobby/thread?thread_id={t['id']}"]["body"]["messages"]


def test_downloads_are_real_files(recording):
    out, data = recording
    assert len(data["downloads"]) == 8
    for file in data["downloads"].values():
        path = out / file
        if path.suffix == ".docx":
            assert zipfile.is_zipfile(path)
    # The tailored resume really is tracked changes made by the app's own writer.
    with zipfile.ZipFile(out / "downloads" / "document-1.docx") as z:
        assert "<w:ins " in z.read("word/document.xml").decode()


def test_only_fictional_people_and_places(recording):
    _, data = recording
    text = json.dumps({p: r for p, r in data["responses"].items() if p != "/api/changelog"})
    emails = set(re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text))
    assert emails and all(e.endswith("@example.com") for e in emails), emails
    urls = set(re.findall(r"https?://([\w.-]+)", text))
    assert all(host.endswith(".example") for host in urls), urls
    assert "gmail" not in text.lower()
