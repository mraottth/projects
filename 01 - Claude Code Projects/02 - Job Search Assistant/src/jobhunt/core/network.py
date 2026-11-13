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

from typing import Literal

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

HIRING = re.compile(
    r"\b(recruit\w*|talent|human resources|hr|hiring|sourcer)\b"
    # "people" is hiring-side only as people ops ("VP of People", "People Partner"), not "People Analytics".
    r"|\bpeople\s+(operations|ops|partner|team|and culture|&\s*culture)\b"
    r"|\bof people\b(?!\s+(analytics|data|insights|science|research))",
    re.I,
)
# Senior leaders are good leads wherever they run an organization in your sources (asked 2026-11-13:
# Blue Rose's and US Digital Response's CEOs had ranked Medium and Low).
LEADER = re.compile(
    r"\b(ceo|cto|coo|cfo|cdo|cio|cpo|cmo|cso|chief|founder|co-?founder|president|executive director|"
    r"managing director|general manager|managing partner|founding partner|general partner|svp|evp|vp|"
    r"vice president|head of)\b",
    re.I,
)
DIRECTOR = re.compile(r"\bdirector\b", re.I)
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
                "org_key": org_key(company) or None,
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
# Organizations Claude rates (names only; asked 2026-11-13: High is for good-fit organizations)

RATINGS_VERSION = 2
PICK_PROMPT = """You help a job seeker decide which of their LinkedIn connections to reach out to. \
You'll get who they are and what they want, then a numbered list of organizations their \
connections work at. Rate the ones worth their attention:
- "fit": somewhere they'd clearly want to work, given their background, the roles they're after, \
their sectors and their own words;
- "interest": plausible or adjacent: a related field or mission, or a likely stepping stone.
Give a few words on why each. Leave out everything else, including organizations clearly in a \
different field and anything you don't recognize well enough to judge."""


class Pick(BaseModel):
    number: int
    rating: Literal["fit", "interest"] = "interest"
    why: str


class Picks(BaseModel):
    picks: list[Pick]


def pick_organizations(exclude: set[str] | None = None) -> dict:
    """Rates every one of the person's connections' organizations (except `exclude`, their own) with
    Claude, by name only. Saves and returns {org_key: {"name", "why", "rating"}} (the `network_orgs`
    setting, with `network_orgs_version`). Errors are raised as LLMError."""
    exclude = exclude or set()
    names = _org_names()
    orgs = [(k, n) for k, n in names.items() if k not in exclude][:MAX_ORGS]
    picked: dict[str, dict] = {}
    if orgs:
        context = _context()
        for start in range(0, len(orgs), CHUNK):
            chunk = orgs[start : start + CHUNK]
            for n, (rating, why) in _pick_chunk(context, [name for _, name in chunk]).items():
                key, name = chunk[n - 1]
                picked[key] = {"name": name, "why": why, "rating": rating}
    update_settings({"network_orgs": picked, "network_orgs_version": RATINGS_VERSION})
    return picked


def _org_names() -> dict[str, str]:
    """{org key: the most common way the person's connections write it}."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT company, COUNT(*) AS n FROM connections
                WHERE user_id = ? AND company IS NOT NULL GROUP BY company ORDER BY n DESC""",
            (current_user_id(),),
        ).fetchall()
    names: dict[str, str] = {}
    for r in rows:
        if key := org_key(r["company"]):
            names.setdefault(key, r["company"])
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
    return {
        p.number: (p.rating, p.why.strip())
        for p in response.parsed_output.picks
        if 1 <= p.number <= len(names)
    }


# ---------------------------------------------------------------------------------------------
# Tiers


# Words for one line of work, so "Director of Analytics" and "Director of Data" count as the same.
_FAMILIES = {
    "data": "data", "analytics": "data", "analyst": "data", "analysis": "data", "insights": "data",
    "science": "data", "scientist": "data", "statistician": "data", "research": "data", "researcher": "data",
    "engineer": "engineering", "engineering": "engineering", "developer": "engineering",
    "software": "engineering",
    "product": "product", "design": "design", "designer": "design", "ux": "design",
    "marketing": "marketing", "growth": "marketing", "policy": "policy",
}  # fmt: skip
# Words too broad to say two roles are the same line of work on their own (2026-11-13).
_GENERIC = {
    "program", "programs", "project", "projects", "operations", "services", "service", "public",
    "government", "health", "community", "strategy", "strategic", "business", "general", "office",
    "coordinator", "specialist", "consultant", "officer", "member", "fellow", "intern", "national",
    "state", "federal", "digital", "technology", "technical", "special", "new", "development",
    "partner", "partnerships", "relations", "affairs", "support", "management", "administrator",
}  # fmt: skip


def _words(text: str | None) -> set[str]:
    return {
        _FAMILIES.get(w, w)
        for w in re.findall(r"[a-z]+", (text or "").lower())
        if w not in _NOT_FUNCTION and w not in _GENERIC and (len(w) > 2 or w in _FAMILIES)
    }


# ---- organizations

_GENERIC_ORGS = {"data", "open", "united", "american", "national", "global", "digital", "civic", "public"}


# Words a source's name often adds to the organization's ("Blue Rose Careers", "Acme Jobs").
_SOURCE_WORDS = {"careers", "career", "jobs", "job", "openings", "board", "page", "hiring", "positions"}


def org_key(company: str | None) -> str:
    """An organization's name for matching: job_merge's normalization, without a parenthetical
    ("Civic Signal (acquired)"), a leading "the", or trailing source words ("Blue Rose Careers").
    `_same_org` then links a short name to the full one ("Blue Rose" and "Blue Rose Research")."""
    text = re.sub(r"\([^)]*\)", " ", company or "")
    # Runs of single letters are one abbreviation: "U.S." -> "us", like "US".
    joined = re.sub(r"\b(?:[a-z] )+[a-z]\b", lambda m: m.group(0).replace(" ", ""), norm_org(text))
    words = joined.removeprefix("the ").split()
    while len(words) > 1 and words[-1] in _SOURCE_WORDS:
        words.pop()
    return " ".join(words)


def _same_org(a: str, b: str) -> bool:
    """Exactly the same, or one is the other plus more words ("google" and "google deepmind"), when
    the shorter is a distinctive name: 5+ letters and not a generic word ("data" vs "data society")."""
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return long.startswith(short + " ") and len(short) >= 5 and short not in _GENERIC_ORGS


class _OrgIndex:
    """Looks organization keys up exactly, or by `_same_org` among keys sharing a first word."""

    def __init__(self, keys):
        self.keys = set(k for k in keys if k)
        self.by_first: dict[str, list[str]] = {}
        for k in self.keys:
            self.by_first.setdefault(k.split(" ")[0], []).append(k)

    def matches(self, key: str) -> list[str]:
        if not key:
            return []
        return [k for k in self.by_first.get(key.split(" ")[0], []) if _same_org(key, k)]


# ---- jobs


def _jobs_by_org(conn, uid: int) -> dict[str, list[dict]]:
    """The person's jobs (active, from followed sources, plus My Jobs), by organization key, each
    with its latest rating, its My Jobs status, and whether you'd want it (`wanted`): rated High/Med
    for interest and fit, on My Jobs, or not rated yet with a title in your line of work and not set
    aside by the relevance check."""
    desired = _desired_words()
    rows = conn.execute(
        """SELECT j.id, j.title, j.company, js.interest_fit, js.candidate_fit,
                  a.status AS application_status, q.screen
             FROM jobs j
             LEFT JOIN job_scores js ON js.id = (
                 SELECT id FROM job_scores WHERE user_id = :uid AND job_id = j.id ORDER BY id DESC LIMIT 1)
             LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = :uid
             LEFT JOIN scoring_queue q ON q.job_id = j.id AND q.user_id = :uid
            WHERE a.id IS NOT NULL
               OR (j.status = 'active' AND j.source_id IN
                   (SELECT source_id FROM source_follows WHERE user_id = :uid))""",
        {"uid": uid},
    ).fetchall()
    by_org: dict[str, list[dict]] = {}
    for r in rows:
        job = dict(r)
        rated = job["interest_fit"] is not None
        job["unrated"] = not rated
        job["wanted"] = bool(job["application_status"]) or (
            job["interest_fit"] in GOOD and job["candidate_fit"] in GOOD
            if rated
            else job["screen"] != "drop" and bool(_words(job["title"]) & desired)
        )
        by_org.setdefault(org_key(job["company"]), []).append(job)
    return by_org


def _desired_titles() -> list[str]:
    return [t for t in (get_settings().get("prefilter_titles_include") or []) if t]


def _desired_words(titles: list[str] | None = None) -> set[str]:
    return _words(" ".join(titles if titles is not None else _desired_titles()))


def _senior(title: str) -> bool:
    return bool(LEADER.search(title) or DIRECTOR.search(title))


def _aligned(position: str, titles: list[str]) -> bool:
    """A title particularly close to one you're after: the same line of work and the same
    seniority ("Director of Analytics" for "Analytics Director"; not "Data Analyst")."""
    words = _words(position)
    return any(words & _words(t) and _senior(position) == _senior(t) for t in titles)


def _followed_org_keys(conn, uid: int) -> set[str]:
    rows = conn.execute(
        """SELECT s.name, s.company_name FROM sources s
             JOIN source_follows f ON f.source_id = s.id WHERE f.user_id = ?""",
        (uid,),
    ).fetchall()
    return {org_key(v) for r in rows for v in (r["company_name"], r["name"]) if v}


def _rank(j: dict) -> tuple:
    order = {"High": 0, "Med": 1, "Low": 2, None: 3}
    return (not j["application_status"], not j["wanted"], order[j["interest_fit"]], order[j["candidate_fit"]])


def _about(job: dict) -> str:
    if job["application_status"]:
        return f"{job['title']} is on your My Jobs"
    if job["wanted"] and job["unrated"]:
        return f"{job['title']} is in your line of work (not rated yet)"
    if job["wanted"]:
        return f"{job['title']} is a strong match for you"
    return f"Has an opening: {job['title']}"


# ---- tiers


def tier(
    person: dict,
    jobs: list[dict] | None,
    pick: dict | None,
    followed: bool = False,
    titles: list[str] | None = None,
) -> tuple[str | None, str, dict | None]:
    """(tier or None for not listed, why, the job it's about). Revised 2026-11-13:
    - High: a good-fit organization (Claude rated it "fit", or it has a job you'd want) and a role
      that fits (a leader, hiring-side, or the line of work of that job or of your desired titles);
    - Medium: a good-fit organization in another role; any organization in your sources or rated
      "interest"; or a title particularly aligned with what you're after, anywhere;
    - Low: your general line of work, or hiring-side, anywhere else.
    The why sits next to their position and organization in the table, so it doesn't repeat them."""
    position = person.get("position") or ""
    titles = titles if titles is not None else _desired_titles()
    words = _words(position)
    hiring = bool(HIRING.search(position))
    leader = bool(LEADER.search(position) or DIRECTOR.search(position))
    my_line = bool(words & _desired_words(titles))
    aligned = _aligned(position, titles)
    jobs = jobs or []
    rating = (pick or {}).get("rating", "interest") if pick else None

    wanted = sorted((j for j in jobs if j["wanted"]), key=_rank)
    same = [j for j in wanted if words & _words(j["title"])]
    job = (same or wanted or sorted(jobs, key=_rank) or [None])[0]
    good_fit = rating == "fit" or bool(wanted)
    in_sources = followed or bool(jobs)

    if wanted:
        place = _about(job)
    elif rating == "fit":
        place = f"A good fit for you: {pick['why'].rstrip('.')}"
    elif rating == "interest":
        place = f"Of interest: {pick['why'].rstrip('.')}"
    elif jobs:
        place = f"In your sources. {_about(job)}"
    elif followed:
        place = "In your sources"
    else:
        place = ""
    if leader:
        role = "they're a leader there"
    elif hiring:
        role = "they're on the hiring side"
    elif same or my_line:
        role = "they're in your line of work"
    else:
        role = ""
    if aligned and not leader:
        role = "their title is close to what you're looking for"

    def why(*parts: str) -> str:
        text = "; ".join(p for p in parts if p)
        return (text[0].upper() + text[1:] + ".") if text else ""

    if good_fit and (leader or hiring or same or my_line):
        return "High", why(place, role), job
    if good_fit or in_sources or rating == "interest":
        return "Medium", why(place, role), job
    if aligned:
        return "Medium", why(role), job
    if my_line or hiring:
        return "Low", why(role, "no openings from your sources there yet"), job
    company = person.get("company")
    if jobs:
        return None, "A different line of work from the openings there.", None
    return (
        None,
        (f"No openings from your sources at {company}." if company else "No organization on their profile."),
        None,
    )


# ---- your current organization (asked 2026-11-13: "a ton of DCCC people" were listed)

MIN_COLLEAGUES = 3


def guess_current_org(conn, uid: int, people: list[dict]) -> str | None:
    """The organization you work at now, guessed for free: of those where you have at least
    MIN_COLLEAGUES connections, the one named earliest in your current resume (the latest job is
    usually listed first). None when nothing fits; the person can set it themselves."""
    row = conn.execute(
        """SELECT raw_text FROM resumes WHERE user_id = ? AND kind = 'resume'
            ORDER BY is_template DESC, is_archived ASC, id DESC LIMIT 1""",
        (uid,),
    ).fetchone()
    text = " " + org_key((row["raw_text"] if row else "") or "") + " "
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for p in people:
        if key := org_key(p["company"]):
            counts[key] = counts.get(key, 0) + 1
            names.setdefault(key, p["company"])
    found = [
        (pos, key)
        for key, n in counts.items()
        if n >= MIN_COLLEAGUES and (pos := _position_in(text, key)) >= 0
    ]
    return names[min(found)[1]] if found else None


_SMALL_WORDS = {"of", "the", "and", "for", "on", "in", "at", "to"}


def _position_in(text: str, key: str) -> int:
    """Where an organization is first named in the resume text (normalized, space-padded), or -1.
    A short one-word name also matches its spelled-out form by initials, since a resume often
    writes "Democratic Congressional Campaign Committee" where LinkedIn says "DCCC"."""
    pos = text.find(f" {key} ")
    if pos >= 0 or " " in key or not 2 <= len(key) <= 6:
        return pos
    words = [(m.start(), m.group()) for m in re.finditer(r"[a-z0-9]+", text)]
    for i in range(len(words)):
        initials, j = "", i
        while j < len(words) and len(initials) < len(key):
            if words[j][1] not in _SMALL_WORDS or not initials:
                initials += words[j][1][0]
            j += 1
            if not key.startswith(initials):
                break
        if initials == key and words[i][1] not in _SMALL_WORDS:
            return words[i][0]
    return -1


def current_org(conn, uid: int, people: list[dict], settings: dict) -> tuple[str | None, bool]:
    """(your current organization, whether it was guessed). Guessed once and saved, so the person's
    own change, or clearing it (""), sticks."""
    if settings.get("network_current_org") is not None:  # None: not decided yet (or connections removed)
        return settings["network_current_org"] or None, bool(settings.get("network_current_org_guessed"))
    if not people:
        return None, False
    guess = guess_current_org(conn, uid, people)
    update_settings({"network_current_org": guess or "", "network_current_org_guessed": bool(guess)})
    return guess, bool(guess)


def overview() -> dict:
    """Manage Sources' network section: the upload, the three tiers, who isn't listed and why, and
    organizations to consider adding."""
    uid = current_user_id()
    settings = get_settings()
    picks = settings.get("network_orgs") or {}
    desired = _desired_titles()
    with get_db() as conn:
        people = [dict(r) for r in conn.execute("SELECT * FROM connections WHERE user_id = ?", (uid,))]
        by_org = _jobs_by_org(conn, uid) if people else {}
        followed = _followed_org_keys(conn, uid) if people else set()
        mine, guessed = current_org(conn, uid, people, settings)
    jobs_index, followed_index, pick_index = _OrgIndex(by_org), _OrgIndex(followed), _OrgIndex(picks)
    tiers: dict[str, list[dict]] = {"High": [], "Medium": [], "Low": []}
    not_listed: list[dict] = []
    orgs: dict[str, dict] = {}
    mine_key, colleagues = org_key(mine), 0
    for p in people:
        key = org_key(p["company"])
        if mine_key and _same_org(key, mine_key):  # your colleagues aren't outreach leads
            colleagues += 1
            continue
        jobs = [j for k in jobs_index.matches(key) for j in by_org[k]]
        pick = next((picks[k] for k in pick_index.matches(key)), None)
        is_followed = bool(followed_index.matches(key))
        level, why, job = tier(p, jobs, pick, is_followed, desired)
        row = {
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
        if level is None:
            not_listed.append(row)
            continue
        tiers[level].append(row)
        if key and not is_followed:
            o = orgs.setdefault(
                key,
                {
                    "name": p["company"],
                    "count": 0,
                    "leader": False,
                    "wanted": False,
                    "why": None,
                    "tier": level,
                },
            )
            o["count"] += 1
            # An organization's tier is its best person's (asked 2026-11-13).
            if ["High", "Medium", "Low"].index(level) < ["High", "Medium", "Low"].index(o["tier"]):
                o["tier"] = level
            o["leader"] = o["leader"] or bool(LEADER.search(p["position"] or ""))
            if job:
                o["wanted"] = o["wanted"] or job["wanted"]
                o["why"] = o["why"] or (
                    f"Your sources list it there: {job['title']}"
                    + (" (a strong match for you)." if job["wanted"] and not job["unrated"] else ".")
                    + " Following it directly catches every opening."
                )
            o["why"] = o["why"] or (pick or {}).get("why")
    for rows in tiers.values():
        rows.sort(key=lambda r: (r["company"] or "", r["name"]))
    not_listed.sort(key=lambda r: r["name"])
    # In tiers like the people (an organization's is its best person's), most people you know first.
    order = {"High": 0, "Medium": 1, "Low": 2}
    suggestions = sorted(
        (
            {
                "name": o["name"],
                "connections": o["count"],
                "why": o["why"],
                "leader": o["leader"],
                "tier": o["tier"],
            }
            for o in orgs.values()
        ),
        key=lambda s: (order[s["tier"]], -s["connections"], not s["leader"], s["name"]),
    )
    return {
        "uploaded_at": settings.get("network_uploaded_at"),
        "count": len(people),
        "picks_error": settings.get("network_orgs_error"),
        "tiers": tiers,
        "not_listed": not_listed,
        "suggested_sources": suggestions,
        # Ratings saved before 2026-11-13 covered only organizations without jobs, with no fit/interest.
        "ratings_outdated": bool(people) and settings.get("network_orgs_version") != RATINGS_VERSION,
        "current_org": mine,
        "current_org_guessed": guessed,
        "colleagues": colleagues,
    }


def _connection_keys() -> dict[str, int]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT company, COUNT(*) AS n FROM connections "
            "WHERE user_id = ? AND company IS NOT NULL GROUP BY company",
            (current_user_id(),),
        ).fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        if key := org_key(r["company"]):
            counts[key] = counts.get(key, 0) + r["n"]
    return counts


def add_counts(jobs: list[dict]) -> None:
    """Sets each job's `connections` count (0 when none), for the cards."""
    counts = _connection_keys()
    index = _OrgIndex(counts)
    cache: dict[str, int] = {}
    for job in jobs:
        key = org_key(job.get("company"))
        if key not in cache:
            cache[key] = sum(counts[k] for k in index.matches(key))
        job["connections"] = cache[key]


def people_at(company: str | None) -> list[dict]:
    """The expanded job card's "People you know here", each with their tier for that organization."""
    key = org_key(company)
    if not key:
        return []
    uid = current_user_id()
    picks = get_settings().get("network_orgs") or {}
    desired = _desired_titles()
    with get_db() as conn:
        everyone = [dict(r) for r in conn.execute("SELECT * FROM connections WHERE user_id = ?", (uid,))]
        people = [p for p in everyone if _same_org(org_key(p["company"]), key)]
        by_org = _jobs_by_org(conn, uid) if people else {}
        followed = _OrgIndex(_followed_org_keys(conn, uid)) if people else _OrgIndex([])
    jobs_index, pick_index = _OrgIndex(by_org), _OrgIndex(picks)
    out = []
    for p in people:
        pkey = org_key(p["company"])
        jobs = [j for k in jobs_index.matches(pkey) for j in by_org[k]]
        pick = next((picks[k] for k in pick_index.matches(pkey)), None)
        level, _, _ = tier(p, jobs, pick, bool(followed.matches(pkey)), desired)
        out.append(
            {
                "id": p["id"],
                "name": f"{p['first_name']} {p['last_name']}".strip(),
                "url": p["url"],
                "position": p["position"],
                "connected_on": p["connected_on"],
                "tier": level,
            }
        )
    rank = {"High": 0, "Medium": 1, "Low": 2, None: 3}
    return sorted(out, key=lambda p: (rank[p["tier"]], p["name"]))
