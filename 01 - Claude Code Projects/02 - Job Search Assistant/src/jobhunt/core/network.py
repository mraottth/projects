"""Who to reach out to, from your LinkedIn connections (asked 2026-11-13, D-036).

LinkedIn won't let a personal app read your connections (its sign-in shares only your own name,
email and photo; connections need a partner program), so the person uploads LinkedIn's own data
export, `Connections.csv`. It's kept per person (`connections`), replaced by each upload.

Each connection is put in a tier when the page asks, from the jobs as they are now:
- **High:** at an organization with a job you'd want (rated High/Med for both interest and fit, or
  on your My Jobs), in a role that matters for it: hiring-side, or the job's line of work.
- **Medium:** at such an organization in another role, or at one with other jobs in a relevant role.
- **Low:** at an organization with other jobs, or one Claude picked as somewhere you'd plausibly want
  to work (`pick_organizations`: organization names only, never people, one cheap call per upload).
Everyone else isn't listed. Organizations with listed connections that you don't follow a source
for are suggested as sources.
"""

import csv
import io
import re
from datetime import datetime

from pydantic import BaseModel

from jobhunt.core.job_merge import norm_org
from jobhunt.core.llm import LLMError, cache_tokens, get_client, llm_error_boundary, log_call
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.settings import get_settings, update_settings
from jobhunt.core.users import current_user_id
from jobhunt.db.connection import get_db

PICK_MODEL = "claude-haiku-5-5"
CHUNK = 200
MAX_ORGS = 2000
GOOD = {"High", "Med"}

HIRING = re.compile(r"\b(recruit\w*|talent|people|human resources|hr|hiring|sourcer)\b", re.I)
_NOT_FUNCTION = {
    "senior", "sr", "junior", "jr", "lead", "principal", "staff", "head", "director", "manager",
    "vp", "vice", "president", "chief", "of", "and", "the", "for", "a", "an", "in", "at", "to",
    "associate", "assistant", "deputy", "executive", "i", "ii", "iii", "iv", "team", "global",
}  # fmt: skip


class CsvError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------
# Upload


def parse_csv(data: bytes) -> list[dict]:
    """LinkedIn's Connections.csv: a few note lines, then "First Name,Last Name,URL,...". Raises
    CsvError when it isn't that file."""
    text = data.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.lower().startswith("first name,")), None)
    if start is None:
        raise CsvError(
            "That doesn't look like LinkedIn's Connections.csv (no \"First Name\" header). "
            "Upload the Connections file from LinkedIn's data export."
        )
    rows = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        first, last = (r.get("First Name") or "").strip(), (r.get("Last Name") or "").strip()
        if not first and not last:
            continue
        company = (r.get("Company") or "").strip() or None
        rows.append(
            {
                "first_name": first,
                "last_name": last,
                "url": (r.get("URL") or "").strip() or None,
                "company": company,
                "position": (r.get("Position") or "").strip() or None,
                "connected_on": _date(r.get("Connected On")),
                "org_key": norm_org(company) or None,
            }
        )
    return rows


def _date(value: str | None) -> str | None:
    value = (value or "").strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value or None


def replace_connections(rows: list[dict]) -> None:
    uid = current_user_id()
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM connections WHERE user_id = ?", (uid,))
        conn.executemany(
            "INSERT INTO connections (user_id, first_name, last_name, url, company, position, "
            "connected_on, org_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (uid, r["first_name"], r["last_name"], r["url"], r["company"], r["position"],
                 r["connected_on"], r["org_key"])
                for r in rows
            ],
        )  # fmt: skip
        conn.execute("COMMIT")
    update_settings({"network_uploaded_at": datetime.now().isoformat(timespec="seconds")})


# ---------------------------------------------------------------------------------------------
# Organizations Claude picks (names only)

PICK_PROMPT = """You help a job seeker decide which of their LinkedIn connections to reach out to. \
You'll get who they are and what they want, then a numbered list of organizations their \
connections work at. List the numbers of organizations where they'd plausibly want to work, given \
their background and goals, with a few words on why each. Leave out ones that are clearly a \
different field, and anything you don't recognize well enough to judge."""


class Pick(BaseModel):
    number: int
    why: str


class Picks(BaseModel):
    picks: list[Pick]


def pick_organizations(exclude: set[str] | None = None) -> dict:
    """Asks Claude which of the person's connections' other organizations suit them. Saves and
    returns {org_key: {"name", "why"}} (the `network_orgs` setting). Errors are raised as LLMError."""
    exclude = exclude or set()
    names = _org_names()
    orgs = [(k, n) for k, n in names.items() if k not in exclude][:MAX_ORGS]
    picked: dict[str, dict] = {}
    if orgs:
        context = _context()
        for start in range(0, len(orgs), CHUNK):
            chunk = orgs[start : start + CHUNK]
            for n, why in _pick_chunk(context, [name for _, name in chunk]).items():
                key, name = chunk[n - 1]
                picked[key] = {"name": name, "why": why}
    update_settings({"network_orgs": picked})
    return picked


def _org_names() -> dict[str, str]:
    """{org_key: the most common way the person's connections write it}."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT org_key, company, COUNT(*) AS n FROM connections
                WHERE user_id = ? AND org_key IS NOT NULL GROUP BY org_key, company ORDER BY n DESC""",
            (current_user_id(),),
        ).fetchall()
    names: dict[str, str] = {}
    for r in rows:
        names.setdefault(r["org_key"], r["company"])
    return names


def _context() -> str:
    s = get_settings()
    titles = ", ".join(s.get("prefilter_titles_include") or []) or "(not specified)"
    sectors = ", ".join(s.get("prefilter_sectors") or []) or "(not specified)"
    return (
        f"## Candidate profile\n{get_current_candidate_profile() or '(not yet built)'}\n\n"
        f"## Titles they're after\n{titles}\n\n"
        f"## Sectors\n{sectors}\n\n"
        f"## In their own words\n{s.get('looking_for_notes') or '(not specified)'}"
    )


def _pick_chunk(context: str, names: list[str]) -> dict[int, str]:
    listing = "\n".join(f"{n}. {name}" for n, name in enumerate(names, 1))
    client = get_client()
    with llm_error_boundary("reading your connections' organizations"):
        response = client.messages.parse(
            model=PICK_MODEL,
            max_tokens=4000,
            system=[
                {"type": "text", "text": PICK_PROMPT},
                {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": f"## Organizations\n{listing}"}],
            output_format=Picks,
            output_config={"effort": "low"},
        )
    log_call(
        call_type="parsing",
        model=PICK_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note=f"network organizations ({len(names)})",
        **cache_tokens(response.usage),
    )
    if response.parsed_output is None:
        raise LLMError(
            f"no answer about your connections' organizations (stop reason: {response.stop_reason})"
        )
    return {p.number: p.why.strip() for p in response.parsed_output.picks if 1 <= p.number <= len(names)}


# ---------------------------------------------------------------------------------------------
# Tiers


# Words for one line of work, so "Director of Analytics" and "Director of Data" count as the same.
_FAMILIES = {
    "data": "data", "analytics": "data", "analyst": "data", "analysis": "data", "insights": "data",
    "science": "data", "scientist": "data", "statistician": "data", "research": "data", "researcher": "data",
    "engineer": "engineering", "engineering": "engineering", "developer": "engineering",
    "software": "engineering",
    "product": "product", "design": "design", "designer": "design", "ux": "design",
    "marketing": "marketing", "growth": "marketing", "policy": "policy", "operations": "operations",
}  # fmt: skip


def _words(text: str | None) -> set[str]:
    return {
        _FAMILIES.get(w, w)
        for w in re.findall(r"[a-z]+", (text or "").lower())
        if w not in _NOT_FUNCTION and (len(w) > 2 or w in _FAMILIES)
    }


def _jobs_by_org(conn, uid: int) -> dict[str, list[dict]]:
    """The person's jobs (active, from followed sources, plus My Jobs), by normalized organization,
    each with its latest rating and whether it's on My Jobs."""
    rows = conn.execute(
        """SELECT j.id, j.title, j.company, js.interest_fit, js.candidate_fit,
                  a.status AS application_status
             FROM jobs j
             LEFT JOIN job_scores js ON js.id = (
                 SELECT id FROM job_scores WHERE user_id = :uid AND job_id = j.id ORDER BY id DESC LIMIT 1)
             LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = :uid
            WHERE a.id IS NOT NULL
               OR (j.status = 'active' AND j.source_id IN
                   (SELECT source_id FROM source_follows WHERE user_id = :uid))""",
        {"uid": uid},
    ).fetchall()
    by_org: dict[str, list[dict]] = {}
    for r in rows:
        job = dict(r)
        job["wanted"] = bool(job["application_status"]) or (
            job["interest_fit"] in GOOD and job["candidate_fit"] in GOOD
        )
        by_org.setdefault(norm_org(job["company"]), []).append(job)
    return by_org


def _best(jobs: list[dict]) -> dict:
    order = {"High": 0, "Med": 1, "Low": 2, None: 3}

    def key(j: dict) -> tuple:
        return (
            not j["application_status"],
            not j["wanted"],
            order[j["interest_fit"]],
            order[j["candidate_fit"]],
        )

    return sorted(jobs, key=key)[0]


def tier(person: dict, jobs: list[dict] | None, pick: dict | None) -> tuple[str, str, dict | None] | None:
    """(tier, why, the job it's about) for one connection, or None when not worth listing. The why
    is shown next to their position and organization, so it doesn't repeat them."""
    position = person.get("position") or ""
    if jobs:
        wanted = [j for j in jobs if j["wanted"]]
        job = _best(wanted or jobs)
        hiring = bool(HIRING.search(position))
        same_work = bool(_words(position) & _words(job["title"]))
        relevant = hiring or same_work
        if job["application_status"]:
            about = f"{job['title']} is on your My Jobs"
        elif wanted:
            about = f"{job['title']} is a strong match for you"
        else:
            about = f"Has an opening: {job['title']}"
        role = (
            "they're on the hiring side"
            if hiring
            else "they're in the same line of work"
            if same_work
            else None
        )
        why = f"{about}; {role}." if role else f"{about}."
        if wanted and relevant:
            return "High", why, job
        if wanted or relevant:
            return "Medium", why, job
        return "Low", why, job
    if pick:
        return "Low", f"{pick['why'].rstrip('.')}. No opening from your sources yet.", None
    return None


def _followed_org_keys(conn, uid: int) -> set[str]:
    rows = conn.execute(
        """SELECT s.name, s.company_name FROM sources s
             JOIN source_follows f ON f.source_id = s.id WHERE f.user_id = ?""",
        (uid,),
    ).fetchall()
    return {norm_org(v) for r in rows for v in (r["company_name"], r["name"]) if v}


def overview() -> dict:
    """Manage Sources' network section: the upload, the three tiers and sources to consider."""
    uid = current_user_id()
    settings = get_settings()
    picks = settings.get("network_orgs") or {}
    with get_db() as conn:
        people = [dict(r) for r in conn.execute("SELECT * FROM connections WHERE user_id = ?", (uid,))]
        by_org = _jobs_by_org(conn, uid) if people else {}
        followed = _followed_org_keys(conn, uid) if people else set()
    tiers: dict[str, list[dict]] = {"High": [], "Medium": [], "Low": []}
    orgs: dict[str, dict] = {}
    for p in people:
        verdict = tier(p, by_org.get(p["org_key"]), picks.get(p["org_key"]))
        if not verdict:
            continue
        level, why, job = verdict
        tiers[level].append(
            {
                "id": p["id"],
                "name": f"{p['first_name']} {p['last_name']}".strip(),
                "url": p["url"],
                "company": p["company"],
                "position": p["position"],
                "connected_on": p["connected_on"],
                "why": why,
                "job_id": job["id"] if job else None,
                "job_title": job["title"] if job else None,
            }
        )
        if p["org_key"] and p["org_key"] not in followed:
            o = orgs.setdefault(p["org_key"], {"name": p["company"], "count": 0, "best": level, "why": None})
            o["count"] += 1
            if ["High", "Medium", "Low"].index(level) < ["High", "Medium", "Low"].index(o["best"]):
                o["best"] = level
            if job and not o["why"]:
                o["why"] = (
                    f"Your sources list it there: {job['title']}"
                    + (" (a strong match for you)." if job["wanted"] else ".")
                    + " Following it directly catches every opening."
                )
            o["why"] = o["why"] or (picks.get(p["org_key"]) or {}).get("why")
    for rows in tiers.values():
        rows.sort(key=lambda r: (r["company"] or "", r["name"]))
    rank = {"High": 0, "Medium": 1, "Low": 2}
    suggestions = sorted(
        (
            {
                "name": o["name"],
                "connections": o["count"],
                "tier": o["best"],
                "why": o["why"],
            }
            for o in orgs.values()
        ),
        key=lambda s: (rank[s["tier"]], -s["connections"], s["name"]),
    )
    return {
        "uploaded_at": settings.get("network_uploaded_at"),
        "count": len(people),
        "picks_error": settings.get("network_orgs_error"),
        "tiers": tiers,
        "suggested_sources": suggestions,
    }


def counts_by_org() -> dict[str, int]:
    """{org_key: how many connections} for the job cards' chip."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT org_key, COUNT(*) AS n FROM connections WHERE user_id = ? AND org_key IS NOT NULL "
            "GROUP BY org_key",
            (current_user_id(),),
        ).fetchall()
    return {r["org_key"]: r["n"] for r in rows}


def add_counts(jobs: list[dict]) -> None:
    """Sets each job's `connections` count (0 when none), for the cards."""
    counts = counts_by_org()
    for job in jobs:
        job["connections"] = counts.get(norm_org(job.get("company")), 0)


def people_at(company: str | None) -> list[dict]:
    """The expanded job card's "People you know here", each with their tier for that organization."""
    key = norm_org(company)
    if not key:
        return []
    uid = current_user_id()
    picks = get_settings().get("network_orgs") or {}
    with get_db() as conn:
        people = [dict(r) for r in conn.execute(
            "SELECT * FROM connections WHERE user_id = ? AND org_key = ?", (uid, key))]  # fmt: skip
        jobs = _jobs_by_org(conn, uid).get(key) if people else None
    out = []
    for p in people:
        verdict = tier(p, jobs, picks.get(key))
        out.append(
            {
                "id": p["id"],
                "name": f"{p['first_name']} {p['last_name']}".strip(),
                "url": p["url"],
                "position": p["position"],
                "connected_on": p["connected_on"],
                "tier": verdict[0] if verdict else None,
            }
        )
    rank = {"High": 0, "Medium": 1, "Low": 2, None: 3}
    return sorted(out, key=lambda p: (rank[p["tier"]], p["name"]))
