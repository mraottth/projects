"""The Usage log in plain terms (asked 2026-11-12: the raw call list was hard to make sense of).

Every Claude call in `llm_calls` gets a category from its call_type and note ("Job scoring",
"Jobby", "Tailored documents"...). For a period (the last day, week or month) this gives the cost
per category, with what it bought ("50 jobs", "8 messages"), and the calls themselves gathered
into activities: back-to-back calls of the same category (no more than GAP_MINUTES apart) are one
row ("Scored 23 jobs"), which opens into its calls.
"""

from dataclasses import dataclass
from datetime import datetime

from jobhunt.core.users import current_user_id
from jobhunt.db.connection import get_db

PERIODS = {"day": "-1 day", "week": "-7 days", "month": "-30 days"}
GAP_MINUTES = 15


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    unit: str  # what each activity counts: "job", "message", "page"...
    verb: str  # how an activity reads: "Scored {n}", "Read {n}"...


CATEGORIES = {
    c.key: c
    for c in [
        Category("scoring", "Job scoring", "job", "Scored {n}"),
        Category("boards", "Reading job boards", "page", "Read {n}"),
        Category("jobby", "Jobby", "message", "Jobby: {n}"),
        Category("documents", "Tailored documents", "document", "Wrote {n}"),
        Category("profile", "Profile & insights", "update", "Profile & insights: {n}"),
        Category("discovery", "Finding sources", "search", "Searched for sources: {n}"),
        Category("adding", "Adding jobs", "job", "Read {n} pasted-in"),
        Category("upkeep", "Upkeep", "call", "Upkeep: {n}"),
    ]
}
ORDER = list(CATEGORIES)


def categorize(call_type: str, note: str | None) -> str:
    note = (note or "").lower()
    if call_type == "scoring":
        return "upkeep" if note.startswith("scoring experiment") else "scoring"
    if call_type == "generation":
        return "jobby" if note.startswith("jobby") else "documents"
    if call_type in ("profile_rebuild", "career_profile"):
        return "profile"
    if call_type == "parsing":
        if note.startswith("custom_board"):
            return "boards"
        if note.startswith(("discovery", "resolve_company")):
            return "discovery"
        if note.startswith("retag"):
            return "upkeep"
        return "adding"  # manual job paste-in logs with no note
    return "upkeep"


def _units(category: str, calls: list[dict]) -> int:
    """What a set of calls amounts to: distinct jobs scored, board pages read, chat messages..."""
    if category == "scoring":
        return len({c["job_id"] for c in calls if c["job_id"]}) or len(calls)
    if category == "boards":
        return len({c["note"] for c in calls})
    if category == "jobby":
        return sum(1 for c in calls if not (c["note"] or "").startswith("jobby chat title"))
    return len(calls)


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")


def _detail(call: dict) -> str:
    """One call, in words, for an opened activity."""
    note = call["note"] or ""
    if call["job_title"]:
        where = f"{call['job_title']} — {call['job_company']}" if call["job_company"] else call["job_title"]
        if note.startswith(("resume", "cover_letter")):
            return f"{'Resume' if note.startswith('resume') else 'Cover letter'} for {where}"
        return where
    if note.startswith("custom_board:"):
        return note.removeprefix("custom_board:")
    if note.startswith("jobby chat title"):
        return "Naming the chat"
    if note.startswith("discovery:suggest_companies"):
        return "Suggesting organizations"
    if note.startswith("discovery:suggest_job_boards") or note.startswith("discovery:"):
        return "Suggesting job boards" if "board" in note else "Suggesting sources"
    if note.startswith("resolve_company:"):
        return f"Finding {note.removeprefix('resolve_company:').split(' (')[0]}'s job board"
    return note[:1].upper() + note[1:] if note else ""


def summary(period: str) -> dict:
    if period not in PERIODS:
        period = "week"
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT c.id, c.call_type, c.job_id, c.model, c.input_tokens, c.output_tokens,
                       c.estimated_cost_usd, c.created_at, c.note, c.cache_read_tokens,
                       j.title AS job_title, j.company AS job_company
                  FROM llm_calls c LEFT JOIN jobs j ON j.id = c.job_id
                 WHERE c.user_id = ? AND c.created_at >= datetime('now', '{PERIODS[period]}')
                 ORDER BY c.created_at DESC, c.id DESC""",
            (current_user_id(),),
        ).fetchall()
    calls = [dict(r) for r in rows]
    for c in calls:
        c["category"] = categorize(c["call_type"], c["note"])
        c["cost"] = c.pop("estimated_cost_usd") or 0.0
        c["detail"] = _detail(c)

    by_category: dict[str, list[dict]] = {}
    for c in calls:
        by_category.setdefault(c["category"], []).append(c)
    categories = [
        {
            "key": key,
            "label": CATEGORIES[key].label,
            "cost": round(sum(c["cost"] for c in by_category[key]), 4),
            "calls": len(by_category[key]),
            "amount": _plural(_units(key, by_category[key]), CATEGORIES[key].unit),
        }
        for key in ORDER
        if key in by_category
    ]
    categories.sort(key=lambda c: -c["cost"])

    activities: list[dict] = []
    for c in calls:  # newest first
        last = activities[-1] if activities else None
        if (
            last
            and last["category"] == c["category"]
            and _minutes_between(c["created_at"], last["started_at"]) <= GAP_MINUTES
        ):
            last["calls"].append(c)
            last["started_at"] = c["created_at"]
        else:
            activities.append(
                {
                    "category": c["category"],
                    "started_at": c["created_at"],
                    "ended_at": c["created_at"],
                    "calls": [c],
                }
            )
    for a in activities:
        cat = CATEGORIES[a["category"]]
        n = _units(a["category"], a["calls"])
        a["label"] = cat.verb.format(n=_plural(n, cat.unit))
        a["category_label"] = cat.label
        a["cost"] = round(sum(c["cost"] for c in a["calls"]), 4)
        a["calls"] = list(reversed(a["calls"]))  # oldest first, as it happened

    return {
        "period": period,
        "total_cost": round(sum(c["cost"] for c in calls), 4),
        "total_calls": len(calls),
        "categories": categories,
        "activities": activities,
    }


def _minutes_between(earlier: str, later: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S"
    return (datetime.strptime(later, fmt) - datetime.strptime(earlier, fmt)).total_seconds() / 60
