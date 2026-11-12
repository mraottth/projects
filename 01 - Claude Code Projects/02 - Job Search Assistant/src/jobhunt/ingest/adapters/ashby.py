"""Ashby's public job-board API (api.ashbyhq.com/posting-api) -- a public, embeddable per-company
endpoint, not scraping. Verified shape against a real board (Linear) during development:
{"jobs": [{"id", "title", "location", "publishedAt", "isRemote", "workplaceType", "jobUrl",
"descriptionPlain"}]}."""

import httpx

from jobhunt.core.html_extract import html_to_markdown

BASE_URL = "https://api.ashbyhq.com/posting-api/job-board/{board_name}"


def _salary(job: dict) -> dict:
    """The salary component of Ashby's optional `compensation` (requested with
    includeCompensation=true): {"summaryComponents": [{"compensationType": "Salary", "interval":
    "1 YEAR" / "1 HOUR", "currencyCode", "minValue", "maxValue"}]}. Empty when not published. Shape
    per Ashby's API docs, not yet confirmed against a live board here."""
    for c in (job.get("compensation") or {}).get("summaryComponents") or []:
        if (c.get("compensationType") or "").lower() != "salary":
            continue
        interval = (c.get("interval") or "").upper()
        period = "hour" if "HOUR" in interval else ("year" if "YEAR" in interval else None)
        if period and (c.get("minValue") or c.get("maxValue")):
            return {
                "salary_min": c.get("minValue"),
                "salary_max": c.get("maxValue"),
                "salary_currency": c.get("currencyCode"),
                "salary_period": period,
            }
    return {}


def fetch(source: dict) -> list[dict]:
    board_name = source["config"]["board_name"]
    resp = httpx.get(
        BASE_URL.format(board_name=board_name), params={"includeCompensation": "true"}, timeout=20.0
    )
    resp.raise_for_status()
    jobs = resp.json().get("jobs", [])

    postings = []
    for job in jobs:
        postings.append(_posting(job, source, board_name))
    return postings


def _remote_type(job: dict) -> str | None:
    """workplaceType when the board gives one; isRemote only when it doesn't. Harvey's board sends
    isRemote true with workplaceType "Hybrid" for a New York / San Francisco job (seen 2026-11-12), and
    trusting isRemote made it "remote", which the location dealbreaker always kept."""
    workplace_type = (job.get("workplaceType") or "").lower()
    if workplace_type in ("remote", "hybrid", "onsite"):
        return workplace_type
    if workplace_type in ("on-site", "in office", "inoffice"):
        return "onsite"
    return "remote" if job.get("isRemote") else None


def _location(job: dict) -> str | None:
    """The main location plus any secondaryLocations ("New York | San Francisco"), so the location
    dealbreaker sees every place a posting lists, not just the first."""
    places = [job.get("location")] + [
        (s or {}).get("location") for s in job.get("secondaryLocations") or [] if isinstance(s, dict)
    ]
    seen: list[str] = []
    for place in places:
        if place and place.strip() and place.strip() not in seen:
            seen.append(place.strip())
    return " | ".join(seen) or None


def _posting(job: dict, source: dict, board_name: str) -> dict:
    return {
        "external_id": job.get("id"),
        "title": job.get("title"),
        "company": source.get("company_name") or board_name,
        "location": _location(job),
        "remote_type": _remote_type(job),
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
        **_salary(job),
        # descriptionHtml keeps the headings and bullets descriptionPlain drops (per Ashby's
        # API docs; falls back to the plain text if it's absent).
        "description_text": html_to_markdown(job.get("descriptionHtml")) or job.get("descriptionPlain"),
        "url": job.get("jobUrl"),
        "posted_at": job.get("publishedAt"),
        "raw_json": job,
    }
