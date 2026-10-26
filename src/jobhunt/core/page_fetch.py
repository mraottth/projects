"""Shared 'fetch a page and get its readable text' helper: try a plain static GET first (cheap),
fall back to headless-browser rendering only when the static fetch yields too little text (a login
wall or a JS-rendered page -- confirmed necessary for at least one real site, techjobsforgov.com)."""

import httpx

from jobhunt.core.browser_fetch import BrowserFetchError, render_page_html
from jobhunt.core.html_extract import extract_readable_text

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

MIN_CHARS = 200


class PageFetchError(Exception):
    pass


def fetch_page_text(url: str, *, preserve_links: bool = False, max_chars: int = 20_000) -> str:
    try:
        resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=15.0)
        resp.raise_for_status()
        html = resp.text
    except httpx.HTTPError as exc:
        html = None
        static_error = str(exc)
    else:
        static_error = None

    text = (
        extract_readable_text(html, max_chars, base_url=url if preserve_links else None)
        if html is not None
        else ""
    )

    if len(text) >= MIN_CHARS:
        return text

    try:
        rendered_html = render_page_html(url)
    except BrowserFetchError as exc:
        reason = static_error or "page had too little text (login wall or JS-rendered page)"
        message = f"could not fetch {url}: {reason}; browser fallback also failed ({exc})"
        raise PageFetchError(message) from exc

    rendered_text = extract_readable_text(rendered_html, max_chars, base_url=url if preserve_links else None)
    if len(rendered_text) < MIN_CHARS:
        raise PageFetchError(
            f"fetched {url} but found too little text even after rendering JS "
            "-- the page may require login, or paste the content directly instead"
        )
    return rendered_text
