"""RemoteOK's public API (remoteok.com/api) -- a single fixed feed (no query params), optionally
filtered client-side by `config.tags`. Verified shape during development: a JSON array whose first
element is a legal notice (must be skipped), then postings with {"id", "position", "company",
"location", "salary_min", "salary_max", "url", "date", "description", "tags"}."""

import html

import httpx

from jobhunt.core.html_extract import extract_readable_text

URL = "https://remoteok.com/api"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def fetch(source: dict) -> list[dict]:
    resp = httpx.get(URL, headers={"User-Agent": USER_AGENT}, timeout=20.0)
    resp.raise_for_status()
    rows = resp.json()
    if rows and "legal" in rows[0]:
        rows = rows[1:]

    tag_filter = {t.lower() for t in source.get("config", {}).get("tags", [])}

    postings = []
    for row in rows:
        if tag_filter and not (tag_filter & {t.lower() for t in row.get("tags", [])}):
            continue
        description = row.get("description")
        postings.append(
            {
                "external_id": row.get("id"),
                "title": row.get("position"),
                "company": row.get("company"),
                "location": row.get("location") or None,
                "remote_type": "remote",
                "salary_min": row.get("salary_min"),
                "salary_max": row.get("salary_max"),
                "salary_currency": "USD" if row.get("salary_min") else None,
                "salary_period": "year" if row.get("salary_min") else None,
                "description_text": extract_readable_text(html.unescape(description))
                if description
                else None,
                "url": row.get("url") or row.get("apply_url"),
                "posted_at": row.get("date"),
                "raw_json": row,
            }
        )
    return postings
