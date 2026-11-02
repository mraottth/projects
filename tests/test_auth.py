"""Sign-in (api/auth.py): only a verified IAP token's email is trusted in production."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from jobhunt.api import auth
from jobhunt.api.main import app
from jobhunt.db.connection import PLACEHOLDER_ADMIN_EMAIL

AUDIENCE = "/projects/123/locations/us-central1/services/jobhunt"
KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())


def _token(key=KEY, **claims) -> str:
    now = int(time.time())
    body = {
        "iss": auth.IAP_ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 600,
        "email": PLACEHOLDER_ADMIN_EMAIL,
        **claims,
    }
    return jwt.encode(body, key, algorithm="ES256")


@pytest.fixture
def iap(monkeypatch):
    monkeypatch.setenv("IAP_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(auth, "_signing_key", lambda token: KEY.public_key())


def _get(client, headers):
    return client.get("/api/settings", headers=headers)


def test_a_valid_token_signs_you_in(iap):
    with TestClient(app) as client:
        assert _get(client, {auth.IAP_HEADER: _token()}).status_code == 200
        prefixed = _token(email=f"accounts.google.com:{PLACEHOLDER_ADMIN_EMAIL}")
        assert _get(client, {auth.IAP_HEADER: prefixed}).status_code == 200


@pytest.mark.parametrize(
    "token, reason",
    [
        (None, "no IAP assertion"),
        ("not-a-jwt", "rejected"),
        (_token(key=OTHER_KEY), "rejected"),
        (_token(exp=int(time.time()) - 10), "rejected"),
        (_token(iss="https://evil.example"), "rejected"),
        (_token(aud="/projects/999/other"), "audience '/projects/999/other'"),
    ],
)
def test_anything_else_is_refused(iap, token, reason):
    with TestClient(app) as client:
        resp = _get(client, {auth.IAP_HEADER: token} if token else {})
    assert resp.status_code == 401
    assert reason in resp.json()["detail"]


def test_the_dev_header_is_ignored_once_iap_is_on(iap):
    with TestClient(app) as client:
        assert _get(client, {auth.DEV_HEADER: PLACEHOLDER_ADMIN_EMAIL}).status_code == 401


def test_dev_sign_in_is_refused_on_cloud_run(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "jobhunt")
    with TestClient(app) as client:
        assert _get(client, {}).status_code == 401


def test_no_sign_in_configured_refuses_everything(monkeypatch):
    monkeypatch.delenv("JOBHUNT_DEV_EMAIL")
    with TestClient(app) as client:
        resp = _get(client, {})
    assert resp.status_code == 401 and "isn't configured" in resp.json()["detail"]


def test_the_frontend_and_scheduler_routes_are_not_behind_it(monkeypatch):
    monkeypatch.delenv("JOBHUNT_DEV_EMAIL")
    with TestClient(app) as client:
        # /internal has its own secret; with none set locally, the check is skipped.
        assert client.post("/internal/run-daily", headers={"X-Scheduler-Secret": "x"}).status_code != 401
