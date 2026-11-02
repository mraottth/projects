"""Who's signed in, for every /api/* request (D-021).

In production, Identity-Aware Proxy signs people in with Google and adds a signed JWT to each
request (`x-goog-iap-jwt-assertion`). It's verified here: ES256 signature against Google's
published IAP keys, issuer, expiry and audience (`IAP_AUDIENCE`, set by `make deploy`). Only the
verified email is trusted; a plain header never is.

Locally (`make serve`) and in tests there's no IAP. `JOBHUNT_DEV_EMAIL` stands in as the signed-in
person, and `X-Dev-User` can switch to another email. Both are ignored whenever IAP_AUDIENCE is
set, and refused outright on Cloud Run (K_SERVICE), so a misconfigured deploy fails closed:
without either setting, every /api request is refused.

Then the email is matched to a user:
- approved: the request runs as them (core/users.py's current user);
- anyone else (unknown, pending, declined, blocked) only reaches /api/me and /api/access/*, which
  back the request-access and waiting pages.
"""

import json
import logging
import os
from functools import lru_cache

import jwt

from jobhunt.core.users import get_user_by_email, reset_current_user, set_current_user, touch_last_seen

logger = logging.getLogger(__name__)

IAP_HEADER = "x-goog-iap-jwt-assertion"
IAP_KEYS_URL = "https://www.gstatic.com/iap/verify/public_key-jwk"
IAP_ISSUER = "https://cloud.google.com/iap"
DEV_HEADER = "x-dev-user"

# Reachable by any signed-in Google account, approved or not.
OPEN_PATHS = ("/api/me", "/api/access/")


class AuthError(Exception):
    pass


@lru_cache(maxsize=1)
def _jwks_client() -> jwt.PyJWKClient:
    return jwt.PyJWKClient(IAP_KEYS_URL, cache_keys=True, lifespan=3600)


def _signing_key(token: str):
    return _jwks_client().get_signing_key_from_jwt(token).key


def verify_iap_token(token: str, audience: str) -> str:
    """The verified email from an IAP JWT, or AuthError."""
    try:
        claims = jwt.decode(
            token,
            _signing_key(token),
            algorithms=["ES256"],
            audience=audience,
            issuer=IAP_ISSUER,
            options={"require": ["exp", "iat", "aud", "iss", "email"]},
        )
    except jwt.InvalidAudienceError as exc:
        # Say which audience the token carried, so a wrong IAP_AUDIENCE is quick to fix (not secret).
        try:
            seen = jwt.decode(token, options={"verify_signature": False}).get("aud")
        except jwt.PyJWTError:
            seen = None
        raise AuthError(
            f"sign-in token is for a different app (audience {seen!r}, expected {audience!r})"
        ) from exc
    except jwt.PyJWTError as exc:
        raise AuthError(f"sign-in token rejected: {exc}") from exc
    email = (claims.get("email") or "").strip()
    # IAP's email claim can carry an "accounts.google.com:" prefix.
    email = email.removeprefix("accounts.google.com:")
    if not email:
        raise AuthError("sign-in token has no email")
    return email


def identify(headers: dict[str, str]) -> str:
    """The signed-in person's email, or AuthError."""
    audience = os.environ.get("IAP_AUDIENCE")
    if audience:
        token = headers.get(IAP_HEADER)
        if not token:
            raise AuthError("not signed in (no IAP assertion)")
        return verify_iap_token(token, audience)
    dev_email = os.environ.get("JOBHUNT_DEV_EMAIL")
    if dev_email and not os.environ.get("K_SERVICE"):
        return headers.get(DEV_HEADER) or dev_email
    raise AuthError(
        "sign-in isn't configured on this server (set IAP_AUDIENCE, or JOBHUNT_DEV_EMAIL locally)"
    )


async def _send_json(send, status: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


class AuthMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware), so the current user set here is seen by the route, its
    worker thread, and any response it streams."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith("/api/"):
            # The frontend's files (IAP still guards them) and /internal (its own secret).
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        try:
            email = identify(headers)
        except AuthError as exc:
            logger.warning("refused %s: %s", path, exc)
            await _send_json(send, 401, {"detail": str(exc), "code": "signed_out"})
            return

        scope.setdefault("state", {})["auth_email"] = email
        user = get_user_by_email(email)
        approved = user is not None and user["status"] == "approved"
        if not approved and not path.startswith(OPEN_PATHS):
            status = user["status"] if user else "none"
            await _send_json(send, 403, {"detail": "you don't have access yet", "code": status})
            return

        token = set_current_user(user["id"] if approved else None)
        try:
            if approved:
                touch_last_seen(user["id"])
            await self.app(scope, receive, send)
        finally:
            reset_current_user(token)
