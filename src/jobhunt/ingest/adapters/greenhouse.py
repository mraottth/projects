"""Greenhouse's public job-board API (boards-api.greenhouse.io) -- an intentionally public,
embeddable endpoint per company, not scraping. Verified shape against a real board (Stripe) during
development: {"jobs": [{"id", "title", "company_name", "location": {"name"}, "absolute_url",
"first_published", "updated_at", "content"}]}."""

import html

import httpx

from jobhunt.core.html_extract import html_to_markdown

BASE_URL = "https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"


def _clean_description(content: str | None) -> str | None:
    # Greenhouse's `content` field is HTML with its tags entity-escaped (literal "&lt;h2&gt;" text),
    # so it must be unescaped before being parsed as HTML.
    if not content:
        return None
    return html_to_markdown(html.unescape(content))


def _salary(job: dict) -> dict:
    """Greenhouse's pay-transparency ranges (requested with pay_transparency=true):
    {"pay_input_ranges": [{"min_cents", "max_cents", "currency_type"}]}. Annual unless the range is
    clearly hourly money; empty when not published (the description is read instead). Shape per
    Greenhouse's API docs, not yet confirmed against a live board here."""
    for r in job.get("pay_input_ranges") or []:
        low, high = r.get("min_cents"), r.get("max_cents")
        if not (low or high):
            continue
        low = low / 100 if low else None
        high = high / 100 if high else None
        return {
            "salary_min": low,
            "salary_max": high,
            "salary_currency": r.get("currency_type"),
            "salary_period": "hour" if (high or low) < 1000 else "year",
        }
    return {}


def fetch(source: dict) -> list[dict]:
    board_token = source["config"]["board_token"]
    resp = httpx.get(
        BASE_URL.format(board_token=board_token),
        params={"content": "true", "pay_transparency": "true"},
        timeout=20.0,
    )
    resp.raise_for_status()
    jobs = resp.json().get("jobs", [])

    postings = []
    for job in jobs:
        postings.append(
            {
                "external_id": str(job["id"]),
                "title": job["title"],
                "company": job.get("company_name") or source.get("company_name") or board_token,
                "location": (job.get("location") or {}).get("name"),
                "remote_type": None,
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "salary_period": None,
                **_salary(job),
                "description_text": _clean_description(job.get("content")),
                "url": job.get("absolute_url"),
                "posted_at": job.get("first_published"),
                "raw_json": job,
            }
        )
    return postings
