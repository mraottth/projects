"""Pulls a pay range out of a job description's text ("$120,000 - $150,000", "$120K–$150K per
year", "$45-$60/hour"). The ATS adapters used to leave salary empty, so every job card said
"Salary: N/A" even when the posting listed pay (caught live). Rule-based and free, so it runs on
every posting; Claude never sees it. Conservative on purpose: a single amount only counts next to
a pay word, and figures outside a plausible pay range (a $5,000 signing bonus, $10M of funding)
are ignored."""

import re

_CURRENCY = {
    "$": "USD",
    "US$": "USD",
    "USD": "USD",
    "£": "GBP",
    "GBP": "GBP",
    "€": "EUR",
    "EUR": "EUR",
    "CA$": "CAD",
    "C$": "CAD",
    "CAD": "CAD",
}

_AMOUNT = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kK])?"
_SYMBOL = r"(US\$|CA\$|C\$|\$|£|€)"
_CODE = r"(USD|CAD|GBP|EUR)"
_DASH = r"\s*(?:-|–|—|to|and)\s*"  # "and": "between $160,000 and $210,000"

_RANGE = re.compile(
    rf"(?:{_SYMBOL}\s*|{_CODE}\s+){_AMOUNT}{_DASH}(?:{_SYMBOL}\s*|{_CODE}\s+)?{_AMOUNT}(?:\s*{_CODE})?",
)
# No currency sign, but labeled as pay: "Salary: 160,000 - 210,000 USD", "Pay: 160k-210k".
_BARE_RANGE = re.compile(
    rf"(?:salary|compensation|pay|wage)[^\d\n]{{0,25}}{_AMOUNT}{_DASH}{_AMOUNT}(?:\s*{_CODE})?", re.I
)
_SINGLE = re.compile(rf"(?:{_SYMBOL}\s*|{_CODE}\s+){_AMOUNT}")
_PAY_WORDS = re.compile(r"salary|compensation|pay\b|pay range|base|wage|hourly rate|annual|ote", re.I)
_HOURLY = re.compile(r"/\s*(?:hr|hour)\b|per\s+hour|an\s+hour|hourly", re.I)

MIN_ANNUAL, MAX_ANNUAL = 15_000, 1_500_000
MIN_HOURLY, MAX_HOURLY = 7, 500


def _value(number: str, k: str | None) -> float:
    value = float(number.replace(",", ""))
    return value * 1000 if k else value


def _plausible(low: float, high: float) -> str | None:
    """The pay period these figures fit, or None if they don't look like pay."""
    if MIN_ANNUAL <= low <= high <= MAX_ANNUAL:
        return "year"
    if MIN_HOURLY <= low <= high <= MAX_HOURLY:
        return "hour"
    return None


def _range_result(text: str, match: re.Match, n1, k1, n2, k2, currency_marks) -> dict | None:
    # "$120-150K": a k on the second figure also applies to a bare first figure under 1000.
    if k2 and not k1 and float(n1.replace(",", "")) < 1000:
        k1 = k2
    low, high = _value(n1, k1), _value(n2, k2)
    if low > high:
        return None
    period = _plausible(low, high)
    if period is None:
        return None
    context = text[max(0, match.start() - 80) : match.end() + 40]
    if period == "year" and _HOURLY.search(text[match.end() : match.end() + 20]):
        return None  # an hourly-looking label on a yearly-looking number: too ambiguous
    if period == "hour" and not _HOURLY.search(context):
        return None  # "$40 - $60" with no "per hour" is just as likely something else
    mark = next((m for m in currency_marks if m), "$")
    return {
        "salary_min": low,
        "salary_max": high,
        "salary_currency": _CURRENCY.get(mark, "USD"),
        "salary_period": period,
    }


def extract_salary(text: str | None) -> dict | None:
    """{"salary_min", "salary_max", "salary_currency", "salary_period"} or None."""
    if not text:
        return None
    for match in _RANGE.finditer(text):
        sym1, code1, n1, k1, sym2, code2, n2, k2, code3 = match.groups()
        if found := _range_result(text, match, n1, k1, n2, k2, (sym1, code1, sym2, code2, code3)):
            return found

    for match in _BARE_RANGE.finditer(text):
        n1, k1, n2, k2, code = match.groups()
        if found := _range_result(text, match, n1, k1, n2, k2, (code,)):
            return found

    for match in _SINGLE.finditer(text):
        sym, code, n, k = match.groups()
        value = _value(n, k)
        context = text[max(0, match.start() - 60) : match.end() + 30]
        if not _PAY_WORDS.search(context):
            continue
        period = _plausible(value, value)
        if period is None or (period == "hour" and not _HOURLY.search(context)):
            continue
        currency = _CURRENCY.get(sym or code, "USD")
        return {"salary_min": value, "salary_max": None, "salary_currency": currency, "salary_period": period}
    return None
