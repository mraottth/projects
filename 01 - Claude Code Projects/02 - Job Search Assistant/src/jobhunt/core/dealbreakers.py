"""Dealbreakers on "What are you looking for?": location (core/locations.py) and minimum salary
(asked 2026-11-04). A job either one rules out isn't scored, isn't on New & Noteworthy or in the
daily email, and is hidden on All Jobs unless asked for. My Jobs is never filtered.

Salary, like location, errs towards keeping a job: it's hidden only when the posted pay is known,
in US dollars, and its annualized low end (the range's minimum, or the one figure given) is below
the minimum. No pay listed, another currency, or a figure whose period can't be told all stay."""

from jobhunt.core import locations

# Hours, days, weeks and months in a working year, to compare any stated period with a yearly floor.
ANNUAL = {"year": 1, "hour": 2080, "day": 260, "week": 52, "month": 12}


def annual_low(job: dict) -> float | None:
    """The low end of the posted pay, per year, in USD; None when it can't be told."""
    low = job.get("salary_min") or job.get("salary_max")
    if not low:
        return None
    if (job.get("salary_currency") or "USD").upper() != "USD":
        return None
    period = job.get("salary_period")
    if period is None:
        # Most postings with no period are yearly, but a bare "45" is surely hourly; don't guess.
        return float(low) if low >= 1000 else None
    multiplier = ANNUAL.get(period)
    return float(low) * multiplier if multiplier else None


def salary_dealbreaker_on(settings: dict) -> bool:
    return bool(settings.get("salary_dealbreaker")) and bool(settings.get("salary_floor"))


def below_floor(job: dict, settings: dict) -> bool:
    if not salary_dealbreaker_on(settings):
        return False
    low = annual_low(job)
    return low is not None and low < float(settings["salary_floor"])


def reasons(job: dict, settings: dict) -> list[str]:
    """Which dealbreakers rule this job out: "location", "salary", both, or neither."""
    out = []
    if locations.blocks(job, settings):
        out.append("location")
    if below_floor(job, settings):
        out.append("salary")
    return out


def any_on(settings: dict) -> bool:
    return locations.dealbreaker_on(settings) or salary_dealbreaker_on(settings)


def blocks(job: dict, settings: dict) -> bool:
    return bool(reasons(job, settings))
