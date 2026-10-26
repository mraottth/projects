"""Adzuna's public job-search API (api.adzuna.com) -- requires a free app_id/app_key
(ADZUNA_APP_ID/ADZUNA_APP_KEY). Confirmed the auth/request shape against the live endpoint during
development (AUTH_FAIL response without real credentials); the success response shape below
follows Adzuna's published API docs: {"results": [{"id", "title", "company": {"display_name"},
"location": {"display_name"}, "salary_min", "salary_max", "redirect_url", "created",
"description"}]}."""

import os

import httpx

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"


def fetch(source: dict) -> list[dict]:
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise RuntimeError("ADZUNA_APP_ID/ADZUNA_APP_KEY not configured")

    config = source.get("config", {})
    country = config.get("country", "us")
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": config.get("results_per_page", 50),
        "what": config.get("what", ""),
        "where": config.get("where", ""),
        "content-type": "application/json",
    }
    resp = httpx.get(BASE_URL.format(country=country), params=params, timeout=20.0)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    postings = []
    for r in results:
        postings.append(
            {
                "external_id": str(r.get("id")),
                "title": r.get("title"),
                "company": (r.get("company") or {}).get("display_name"),
                "location": (r.get("location") or {}).get("display_name"),
                "remote_type": None,
                "salary_min": r.get("salary_min"),
                "salary_max": r.get("salary_max"),
                "salary_currency": None,
                "salary_period": "year",
                "description_text": r.get("description"),
                "url": r.get("redirect_url"),
                "posted_at": r.get("created"),
                "raw_json": r,
            }
        )
    return postings
