"""Ashby's public job-board API (api.ashbyhq.com/posting-api) -- a public, embeddable per-company
endpoint, not scraping. Verified shape against a real board (Linear) during development:
{"jobs": [{"id", "title", "location", "publishedAt", "isRemote", "workplaceType", "jobUrl",
"descriptionPlain"}]}."""

import httpx

BASE_URL = "https://api.ashbyhq.com/posting-api/job-board/{board_name}"


def fetch(source: dict) -> list[dict]:
    board_name = source["config"]["board_name"]
    resp = httpx.get(BASE_URL.format(board_name=board_name), timeout=20.0)
    resp.raise_for_status()
    jobs = resp.json().get("jobs", [])

    postings = []
    for job in jobs:
        workplace_type = (job.get("workplaceType") or "").lower()
        remote_type = (
            "remote"
            if job.get("isRemote") or workplace_type == "remote"
            else ("hybrid" if workplace_type == "hybrid" else ("onsite" if workplace_type else None))
        )
        postings.append(
            {
                "external_id": job.get("id"),
                "title": job.get("title"),
                "company": source.get("company_name") or board_name,
                "location": job.get("location"),
                "remote_type": remote_type,
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "salary_period": None,
                "description_text": job.get("descriptionPlain"),
                "url": job.get("jobUrl"),
                "posted_at": job.get("publishedAt"),
                "raw_json": job,
            }
        )
    return postings
