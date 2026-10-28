"""Shared 'fetch a page and get its readable text' helper: try a plain static GET first (cheap),
fall back to headless-browser rendering only when the static fetch yields too little text (a login
wall or a JS-rendered page -- confirmed necessary for at least one real site, techjobsforgov.com)."""

from dataclasses import dataclass

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


@dataclass
class FetchedPage:
    text: str
    html: str  # raw HTML, for callers that need what text extraction strips (see core/ats_detect.py)
    rendered: bool  # True if this came from the headless browser


def fetch_page_text(url: str, *, preserve_links: bool = False, max_chars: int = 20_000) -> str:
    return fetch_page(url, preserve_links=preserve_links, max_chars=max_chars).text


def fetch_page(
    url: str, *, preserve_links: bool = False, max_chars: int = 20_000, force_render: bool = False
) -> FetchedPage:
    """`force_render` skips the static GET: for a page whose static text was long enough to pass
    MIN_CHARS but held nothing useful (a careers page whose listings load via JS or an iframe)."""
    base_url = url if preserve_links else None
    static_error = None
    if not force_render:
        try:
            resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=15.0)
            resp.raise_for_status()
            html = resp.text
        except httpx.HTTPError as exc:
            static_error = str(exc)
        else:
            text = extract_readable_text(html, max_chars, base_url=base_url)
            if len(text) >= MIN_CHARS:
                return FetchedPage(text=text, html=html, rendered=False)

    try:
        rendered_html = render_page_html(url)
    except BrowserFetchError as exc:
        reason = static_error or "page had too little text (login wall or JS-rendered page)"
        message = f"could not fetch {url}: {reason}; browser fallback also failed ({exc})"
        raise PageFetchError(message) from exc

    rendered_text = extract_readable_text(rendered_html, max_chars, base_url=base_url)
    if len(rendered_text) < MIN_CHARS:
        raise PageFetchError(
            f"fetched {url} but found too little text even after rendering JS "
            "-- the page may require login, or paste the content directly instead"
        )
    return FetchedPage(text=rendered_text, html=rendered_html, rendered=True)
