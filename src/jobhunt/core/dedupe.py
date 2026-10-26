import hashlib
import re
from urllib.parse import urlparse, urlunparse


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_url(url: str) -> str:
    """Drop query string/fragment/trailing slash so the same listing re-fetched later still matches."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", "", ""))


def compute_dedupe_key(
    *,
    url: str | None,
    company: str,
    title: str,
    location: str | None,
    source_id: int | None = None,
    external_id: str | None = None,
) -> str:
    # Prefer (source, external_id): it's the most reliable key an ATS/aggregator gives us. URL
    # alone isn't safe for this -- Greenhouse (and others) put the actual job id in the query
    # string (e.g. "?gh_jid=123"), which normalize_url strips, so every posting on a board would
    # otherwise collapse onto the same key.
    if source_id is not None and external_id:
        basis = f"source:{source_id}:external:{external_id}"
    elif url:
        basis = f"url:{normalize_url(url)}"
    else:
        basis = f"job:{_normalize(company)}|{_normalize(title)}|{_normalize(location or '')}"
    return hashlib.sha256(basis.encode()).hexdigest()
