from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.db.connection import get_db


def _upload(client, filename="resume.txt", kind="resume", is_archived=False, label=None):
    data = {"kind": kind, "is_archived": "true" if is_archived else "false"}
    if label:
        data["label"] = label
    resp = client.post(
        "/api/resumes/upload",
        files={"file": (filename, b"Some resume text content.", "text/plain")},
        data=data,
    )
    return resp


def _resume_row(resume_id: int):
    with get_db() as conn:
        return dict(
            conn.execute(
                "SELECT id, kind, is_archived, is_template FROM resumes WHERE id = ?", (resume_id,)
            ).fetchone()
        )


def test_upload_becomes_current_by_default():
    with TestClient(app) as client:
        resp = _upload(client)
        assert resp.status_code == 200
        row = _resume_row(resp.json()["id"])
        assert row["is_template"] == 1
        assert row["is_archived"] == 0


def test_second_upload_demotes_the_first_to_old():
    with TestClient(app) as client:
        first = _upload(client, filename="r1.txt").json()["id"]
        second = _upload(client, filename="r2.txt").json()["id"]

        first_row = _resume_row(first)
        second_row = _resume_row(second)
        assert first_row["is_template"] == 0
        assert first_row["is_archived"] == 1
        assert second_row["is_template"] == 1
        assert second_row["is_archived"] == 0


def test_upload_as_archived_does_not_become_current():
    with TestClient(app) as client:
        current = _upload(client, filename="current.txt").json()["id"]
        old = _upload(client, filename="old.txt", is_archived=True).json()["id"]

        assert _resume_row(current)["is_template"] == 1
        old_row = _resume_row(old)
        assert old_row["is_template"] == 0
        assert old_row["is_archived"] == 1


def test_archiving_a_resume_via_patch_clears_its_current_flag():
    with TestClient(app) as client:
        resume_id = _upload(client).json()["id"]
        resp = client.patch(f"/api/resumes/{resume_id}", json={"is_archived": True})
        assert resp.status_code == 200

        row = _resume_row(resume_id)
        assert row["is_archived"] == 1
        assert row["is_template"] == 0


def test_promoting_an_old_resume_demotes_the_current_one():
    with TestClient(app) as client:
        current = _upload(client, filename="current.txt").json()["id"]
        old = _upload(client, filename="old.txt", is_archived=True).json()["id"]

        resp = client.patch(f"/api/resumes/{old}", json={"is_template": True})
        assert resp.status_code == 200

        assert _resume_row(old) == {"id": old, "kind": "resume", "is_archived": 0, "is_template": 1}
        current_row = _resume_row(current)
        assert current_row["is_template"] == 0
        assert current_row["is_archived"] == 1


def test_resume_and_cover_letter_current_flags_are_independent():
    with TestClient(app) as client:
        resume_id = _upload(client, kind="resume").json()["id"]
        cover_letter_id = _upload(client, kind="cover_letter").json()["id"]

        assert _resume_row(resume_id)["is_template"] == 1
        with get_db() as conn:
            cl_row = dict(
                conn.execute(
                    "SELECT is_template FROM resumes WHERE id = ?", (cover_letter_id,)
                ).fetchone()
            )
        assert cl_row["is_template"] == 1


def test_download_returns_the_uploaded_file_under_its_name(tmp_path, monkeypatch):
    from jobhunt.api import resumes

    monkeypatch.setattr(resumes, "RESUME_STORAGE_DIR", tmp_path)
    with TestClient(app) as client:
        resume_id = _upload(client, filename="My Resume.txt").json()["id"]
        resp = client.get(f"/api/resumes/{resume_id}/download")
        assert resp.status_code == 200
        assert resp.content == b"Some resume text content."
        assert "My%20Resume.txt" in resp.headers["content-disposition"]  # RFC 5987-encoded

        assert client.get("/api/resumes/9999/download").status_code == 404
