"""Spots "postings" that aren't a job: a general application, talent pool or resume drop ("Don't
see your role? Submit your resume", "Join our Talent Community"). Many careers pages and ATS
boards list one alongside real openings. They're kept out of scoring (no point paying to rate
one), New & Noteworthy and All Jobs, at the user's request. Matched on the title only: a real
job's description often mentions a talent community in passing."""

import re

_GENERIC_TITLE = re.compile(
    r"""
    \bgeneral\s+(application|applications|interest|consideration|submission|inquiry|pool)\b
    | \b(open|spontaneous|unsolicited|speculative|future)\s+applications?\b
    | \btalent\s+(pool|community|network|bank|pipeline|database)\b
    | \b(resume|résumé|cv)\s+(drop|bank|submission|database)\b
    | \bsubmit\s+(your|a)\s+(resume|résumé|cv|application)\b
    | \bjoin\s+our\s+talent\b
    | \bdon'?t\s+see\s+(a|the|your)\b
    | \bnot\s+seeing\s+(a|the|your)\b
    | \bfuture\s+(opportunities|openings|roles|positions|consideration)\b
    | \bexpressions?\s+of\s+interest\b
    | \binterest\s+form\b
    | \bget\s+on\s+our\s+radar\b
    | \bstay\s+in\s+touch\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_generic_posting(title: str | None) -> bool:
    return bool(title and _GENERIC_TITLE.search(title))
