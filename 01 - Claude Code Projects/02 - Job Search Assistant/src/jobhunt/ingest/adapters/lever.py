"""Lever's public postings API (api.lever.co) -- a public, embeddable per-company endpoint, not
scraping. Verified shape against a real board (Ro) during development: a bare list of postings,
each {"id", "text" (title), "categories": {"location", "commitment"}, "createdAt" (epoch ms),
"descriptionPlain", "hostedUrl", "workplaceType"}."""

from datetime import UTC, datetime

import httpx

from jobhunt.core.html_extract import html_to_markdown

BASE_URL = "https://api.lever.co/v0/postings/{company_slug}"

WORKPLACE_TYPE_MAP = {"remote": "remote", "hybrid": "hybrid", "onsite": "onsite"}


def _salary(p: dict) -> dict:
    """Lever's optional `salaryRange` ({"min", "max", "currency", "interval": "per-year-salary" /
    "per-hour-wage" / ...}); empty when the posting doesn't publish one (the pipeline then falls
    back to reading the description, core/salary_extract.py). Shape per Lever's API docs, not yet
    confirmed against a live board here (the dev sandbox can't reach Lever)."""
    r = p.get("salaryRange") or {}
    interval = (r.get("interval") or "").lower()
    period = "hour" if "hour" in interval else ("year" if "year" in interval else None)
    if not (r.get("min") or r.get("max")) or period is None:
        return {}
    return {
        "salary_min": r.get("min"),
        "salary_max": r.get("max"),
        "salary_currency": r.get("currency"),
        "salary_period": period,
    }


def _description(p: dict) -> str | None:
    """The formatted description: `description` (HTML), then each of `lists` (a heading plus its
    <li> items, e.g. "Requirements"), then `additional` (HTML) -- Lever keeps those apart, and the
    plain `descriptionPlain` used before has no headings or bullets and omits the lists. Falls back
    to it when there's no HTML. Field shapes per Lever's API docs, not yet seen live from the dev
    sandbox; a missing field just falls through to the next."""
    parts = [html_to_markdown(p.get("description"))]
    for section in p.get("lists") or []:
        items = html_to_markdown(f"<ul>{section.get('content') or ''}</ul>")
        if items:
            parts.append(f"## {section['text']}\n\n{items}" if section.get("text") else items)
    parts.append(html_to_markdown(p.get("additional")))
    text = "\n\n".join(part for part in parts if part)
    return text or p.get("descriptionPlain")


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
                **_salary(p),
                "description_text": _description(p),
                "url": p.get("hostedUrl"),
                "posted_at": posted_at,
                "raw_json": p,
            }
        )
    return postings
