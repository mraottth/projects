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
    *, url: str | None, company: str, title: str, location: str | None
) -> str:
    if url:
        basis = f"url:{normalize_url(url)}"
    else:
        basis = f"job:{_normalize(company)}|{_normalize(title)}|{_normalize(location or '')}"
    return hashlib.sha256(basis.encode()).hexdigest()
