"""Lever's public postings API (api.lever.co) -- a public, embeddable per-company endpoint, not
scraping. Verified shape against a real board (Ro) during development: a bare list of postings,
each {"id", "text" (title), "categories": {"location", "commitment"}, "createdAt" (epoch ms),
"descriptionPlain", "hostedUrl", "workplaceType"}."""

from datetime import UTC, datetime

import httpx

BASE_URL = "https://api.lever.co/v0/postings/{company_slug}"

WORKPLACE_TYPE_MAP = {"remote": "remote", "hybrid": "hybrid", "onsite": "onsite"}


def fetch(source: dict) -> list[dict]:
    company_slug = source["config"]["company_slug"]
    resp = httpx.get(BASE_URL.format(company_slug=company_slug), params={"mode": "json"}, timeout=20.0)
    resp.raise_for_status()
    postings_raw = resp.json()
    if not isinstance(postings_raw, list):
        return []  # Lever returns {"ok": false, ...} for an unknown/misspelled company slug

    postings = []
    for p in postings_raw:
        categories = p.get("categories") or {}
        created_at_ms = p.get("createdAt")
        posted_at = (
            datetime.fromtimestamp(created_at_ms / 1000, tz=UTC).isoformat() if created_at_ms else None
        )
        postings.append(
            {
                "external_id": p.get("id"),
                "title": p.get("text"),
                "company": source.get("company_name") or company_slug,
                "location": categories.get("location"),
                "remote_type": WORKPLACE_TYPE_MAP.get((p.get("workplaceType") or "").lower()),
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "salary_period": None,
                "description_text": p.get("descriptionPlain"),
                "url": p.get("hostedUrl"),
                "posted_at": posted_at,
                "raw_json": p,
            }
        )
    return postings
