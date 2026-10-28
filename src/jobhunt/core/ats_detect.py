"""Finds a Greenhouse/Lever/Ashby job board embedded in (or linked from) a careers page's HTML.

Many organizations' own /careers pages don't list jobs in their HTML at all: they embed an ATS
board (a script that injects an iframe, or an iframe directly), so the page's readable text is just
the site's nav, "about us" copy and footer -- plenty of text, zero postings. Reading that board
through its public API instead (ingest/adapters/greenhouse.py etc.) is free and exact, where asking
Claude to read the page found nothing (caught live: "Test fetch" on several careers pages came
back empty). Scans raw HTML, since html_extract strips the <script>/<iframe> tags that carry it."""

import re

# (kind, config key, pattern). "Embed" patterns only appear when a page hosts the board itself;
# plain links can also appear on a multi-employer board pointing at many companies' boards.
_EMBED_PATTERNS = [
    (
        "ats_greenhouse",
        "board_token",
        r"greenhouse\.io/embed/job_board(?:/js)?\?(?:[^\"'\s<>]*?[&;])?for=([A-Za-z0-9_-]+)",
    ),
    ("ats_greenhouse", "board_token", r"boards-api\.greenhouse\.io/v1/boards/([A-Za-z0-9_-]+)"),
    ("ats_lever", "company_slug", r"api\.lever\.co/v0/postings/([A-Za-z0-9_.-]+)"),
    ("ats_ashby", "board_name", r"jobs\.ashbyhq\.com/([A-Za-z0-9_.-]+)/embed"),
    ("ats_ashby", "board_name", r"api\.ashbyhq\.com/posting-api/job-board/([A-Za-z0-9_.-]+)"),
]
_LINK_PATTERNS = [
    ("ats_greenhouse", "board_token", r"(?:boards|job-boards)\.greenhouse\.io/([A-Za-z0-9_-]+)"),
    ("ats_lever", "company_slug", r"jobs\.lever\.co/([A-Za-z0-9_.-]+)"),
    ("ats_ashby", "board_name", r"jobs\.ashbyhq\.com/([A-Za-z0-9_.-]+)"),
]
# Path segments those URL shapes use that aren't a company's board name.
_NOT_A_BOARD = {"embed", "js", "v1", "api", "static", "assets", "favicon.ico"}


def _find(patterns, html: str) -> set[tuple[str, str, str]]:
    found = set()
    for kind, key, pattern in patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            slug = match.group(1).strip(".")
            if slug and slug.lower() not in _NOT_A_BOARD:
                found.add((kind, key, slug))
    return found


def detect_embedded_ats(html: str, *, single_organization: bool) -> tuple[str, dict] | None:
    """(kind, config) for the one ATS board this page hosts, or None. An embed counts on any page.
    A plain link only counts for a single-organization source: on a multi-employer board, a link
    to one company's Greenhouse board is just one posting among many. Either way, two or more
    distinct boards means it's a listing of other boards, so nothing is returned."""
    found = _find(_EMBED_PATTERNS, html)
    if not found and single_organization:
        found = _find(_LINK_PATTERNS, html)
    if len(found) != 1:
        return None
    kind, key, slug = found.pop()
    return kind, {key: slug}
