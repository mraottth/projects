"""Records the public demo (asked 2026-11-11): seed a throwaway database with the fictional persona,
run the real app on it, and save its answer to every GET the pages make.

    PYTHONPATH=src:. uv run python -m demo.record [OUT_DIR]   (default frontend/demo-build)

Writes OUT_DIR/demo-data.json ({recorded_at, responses: {path: {status, body}}, downloads}) and
OUT_DIR/downloads/*, saved under fixed names (resume-<id>.docx, document-<id>.docx, my-jobs.csv)
that api.ts's download links point at in a demo build. The demo build of the frontend (VITE_DEMO,
src/demo.ts) answers apiFetch() from it, so the demo needs no server and makes no Claude calls. The
database lives in a temp folder and is thrown away; data/ is never touched.
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every fixed GET the frontend makes (api.ts and the pages). test_demo.py checks api.ts against this
# plus the per-id paths below, so a new endpoint can't quietly break the demo.
FIXED = [
    "/api/me",
    "/api/onboarding/status",
    "/api/resumes",
    "/api/resumes/profile",
    "/api/jobs/ranked",
    "/api/jobs/ranked/descriptions",
    "/api/digest/today",
    "/api/digest/dates",
    "/api/digest/status",
    "/api/applications",
    "/api/applications/insights",
    "/api/sources",
    "/api/settings",
    "/api/settings/preference-profile",
    "/api/settings/location-check",
    "/api/settings/salary-check",
    "/api/settings/retag-highlights",
    "/api/settings/email-status",
    "/api/settings/backups",
    "/api/settings/usage-log?period=day",
    "/api/settings/usage-log?period=week",
    "/api/settings/usage-log?period=month",
    "/api/rescore/my_jobs",
    "/api/rescore/all_jobs",
    "/api/pipeline/progress",
    "/api/generate/documents",
    "/api/jobby/threads",
    "/api/jobby/thread",
    "/api/changelog",
]


# The Daily email card reads these to say email is set up; nothing is ever sent (no POST is made).
DEMO_ENV = {
    "JOBHUNT_DEV_EMAIL": "admin@localhost",
    "EMAIL_ADDRESS": "jordan.rivera@example.com",
    "GMAIL_APP_PASSWORD": "demo-not-a-password",
}
CLEARED_ENV = ("IAP_AUDIENCE", "K_SERVICE", "JOBHUNT_ADMIN_EMAIL", "ANTHROPIC_API_KEY", "APP_URL")


def configure_env() -> Path:
    """Points the app at a fresh temp folder (before jobhunt is imported) and returns it."""
    work = Path(tempfile.mkdtemp(prefix="jsa-demo-"))
    os.environ.update(
        {
            "JOBHUNT_DATA_DIR": str(work),
            "JOBHUNT_DB_PATH": str(work / "jobhunt.db"),
            "JOBHUNT_RESUME_DIR": str(work / "resumes"),
        }
    )
    for key in CLEARED_ENV:
        os.environ.pop(key, None)
    os.environ.update(DEMO_ENV)
    return work


def record(out_dir: Path) -> dict:
    """Seeds the database the environment points at and records the app's answers into out_dir."""
    from fastapi.testclient import TestClient

    from demo.seed import build
    from jobhunt.api import sources as sources_api
    from jobhunt.api.main import app

    sources_api.schedule_source_refresh = lambda *a, **k: None  # never reach a real board
    seeded = build()
    as_jordan = {"X-Dev-User": seeded["email"]}

    responses: dict[str, dict] = {}
    downloads: dict[str, str] = {}
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "downloads").mkdir(parents=True)

    with TestClient(app) as client:

        def get(path: str) -> None:
            resp = client.get(path, headers=as_jordan)
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            responses[path] = {"status": resp.status_code, "body": body}

        def download(path: str, name: str) -> None:
            resp = client.get(path, headers=as_jordan)
            assert resp.status_code == 200, (path, resp.status_code, resp.text[:200])
            (out_dir / "downloads" / name).write_bytes(resp.content)
            downloads[path] = f"downloads/{name}"

        for path in FIXED:
            get(path)
        job_ids = {r["id"] for r in responses["/api/jobs/ranked"]["body"]}
        job_ids |= {a["job_id"] for a in responses["/api/applications"]["body"]}
        for job_id in sorted(job_ids):
            get(f"/api/jobs/{job_id}")
            get(f"/api/generate/history?job_id={job_id}")
        for app_row in responses["/api/applications"]["body"]:
            get(f"/api/jobby/thread?job_id={app_row['job_id']}")
        for date in responses["/api/digest/dates"]["body"]:
            get(f"/api/digest/{date}")
        for chat in responses["/api/jobby/threads"]["body"]:
            get(f"/api/jobby/thread?thread_id={chat['id']}")

        for doc in responses["/api/resumes"]["body"]:
            download(f"/api/resumes/{doc['id']}/download", f"resume-{doc['id']}.docx")
        for doc in responses["/api/generate/documents"]["body"]:
            download(f"/api/generate/documents/{doc['id']}/download", f"document-{doc['id']}.docx")
        download("/api/applications/export.csv", "my-jobs.csv")

    data = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "responses": responses,
        "downloads": downloads,
    }
    (out_dir / "demo-data.json").write_text(json.dumps(data, separators=(",", ":")))
    return data


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "frontend" / "demo-build"
    work = configure_env()
    try:
        result = record(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    bad = {p: r["status"] for p, r in result["responses"].items() if r["status"] != 200}
    print(f"recorded {len(result['responses'])} responses, {len(result['downloads'])} downloads -> {out}")
    if bad:
        print("not 200:", bad)
