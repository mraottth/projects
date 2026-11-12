"""One job, not two, when a pasted job is also found by a source (asked 2026-11-12, D-035).

A pasted job is a private row (`jobs.user_id`, dedupe key `u<id>:...`) on its owner's My Jobs. When
a source they follow finds the same posting, it's saved as a separate shared row (its key comes from
the source and the ATS job id), so All Jobs showed the job as untracked while My Jobs held the pasted
copy. Matching, most certain first:

1. The posting's identity from its URL (`posting_identity`): the ATS and its job id, so the
   several URLs one Greenhouse job can have agree, while two different `gh_jid`s don't. Merges
   automatically.
2. The same title and organization, normalized, when exactly one of the person's source jobs has
   them and the locations don't clearly differ.
3. The description only breaks a tie among several such jobs (word overlap >= DESCRIPTION_MATCH);
   it never matches anything on its own.

A merge keeps the source copy (shared, updated daily, closed when unlisted) and moves the person's
application, history, scores, feedback, documents, Jobby chat, queue and digest rows onto it.
"""

import logging
import re
from urllib.parse import parse_qsl, urlencode, urlparse

from jobhunt.db.connection import get_db

logger = logging.getLogger(__name__)

DESCRIPTION_MATCH = 0.8
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_TRACKING = re.compile(r"^(utm_.*|gh_src|source|src|ref|referrer|lever-source.*|lever-origin)$", re.I)


def posting_identity(url: str | None) -> str | None:
    """What a job URL points at: "greenhouse:123", "lever:<uuid>", "ashby:<uuid>"..., or for any
    other site its address with tracking parameters dropped (the query is kept: some sites put the
    job's id there)."""
    if not url or not url.strip():
        return None
    parsed = urlparse(url.strip() if "://" in url else f"https://{url.strip()}")
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    query = dict(parse_qsl(parsed.query))
    if gh := query.get("gh_jid"):
        return f"greenhouse:{gh}"
    if "greenhouse.io" in host and (m := re.search(r"/jobs/(\d+)", path)):
        return f"greenhouse:{m.group(1)}"
    if aj := query.get("ashby_jid"):
        return f"ashby:{aj.lower()}"
    if host.endswith("lever.co") and (m := re.search(rf"/({_UUID})", path, re.I)):
        return f"lever:{m.group(1).lower()}"
    if host.endswith("ashbyhq.com") and (m := re.search(rf"/({_UUID})", path, re.I)):
        return f"ashby:{m.group(1).lower()}"
    if host.endswith("myworkdayjobs.com") and (m := re.search(r"_([A-Za-z]*-?\d[\w-]*)$", path)):
        return f"workday:{host.split('.')[0]}:{m.group(1).lower()}"
    if host.endswith("smartrecruiters.com") and (m := re.search(r"/(\d{6,})", path)):
        return f"smartrecruiters:{m.group(1)}"
    kept = sorted((k, v) for k, v in query.items() if not _TRACKING.match(k))
    return f"url:{host}{path}" + (f"?{urlencode(kept)}" if kept else "")


_ORG_SUFFIXES = {"inc", "llc", "pbc", "ltd", "corp", "corporation", "co", "the", "company"}
_TITLE_WORDS = {"sr": "senior", "jr": "junior", "mgr": "manager", "dir": "director", "vp": "vice president"}


def _words(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def norm_org(company: str | None) -> str:
    return " ".join(w for w in _words(company) if w not in _ORG_SUFFIXES)


def norm_title(title: str | None) -> str:
    return " ".join(_TITLE_WORDS.get(w, w) for w in _words(title))


_PLACE_NOISE = {
    "remote",
    "hybrid",
    "onsite",
    "office",
    "united",
    "states",
    "usa",
    "us",
    "and",
    "or",
    "the",
    "in",
}


def _locations_compatible(a: str | None, b: str | None) -> bool:
    """False only when both name places and share none (so "Washington, DC" vs "Washington, DC
    (Hybrid)" agree, "New York, NY" vs "Austin, TX" don't). Remote or missing on either side agrees."""
    wa = {w for w in _words(a) if len(w) > 1 and w not in _PLACE_NOISE}
    wb = {w for w in _words(b) if len(w) > 1 and w not in _PLACE_NOISE}
    if not wa or not wb or "remote" in _words(a) or "remote" in _words(b):
        return True
    return bool(wa & wb)


def _overlap(a: str | None, b: str | None) -> float:
    wa, wb = ({w for w in _words(t) if len(w) > 2} for t in (a, b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def find_match(job: dict, candidates: list[dict]) -> dict | None:
    """The one candidate that is the same posting as `job`, or None when unsure."""
    identity = posting_identity(job.get("url"))
    if identity:
        same = [c for c in candidates if posting_identity(c.get("url")) == identity]
        if same:
            return sorted(same, key=lambda c: (c.get("status") != "active", c["id"]))[0]
    title, org = norm_title(job.get("title")), norm_org(job.get("company"))
    if not title or not org:
        return None
    named = [
        c
        for c in candidates
        if norm_title(c.get("title")) == title
        and norm_org(c.get("company")) == org
        and _locations_compatible(job.get("location"), c.get("location"))
    ]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        text = job.get("description_text")
        scored = sorted(((_overlap(text, c.get("description_text")), c) for c in named), key=lambda p: -p[0])
        best, runner_up = scored[0][0], scored[1][0]
        if best >= DESCRIPTION_MATCH and best > runner_up:
            return scored[0][1]
    return None


_COLUMNS = "id, source_id, user_id, title, company, location, url, description_text, status"


def source_jobs_for(conn, user_id: int) -> list[dict]:
    """Every job from a source the person follows: what a pasted job of theirs can merge into."""
    rows = conn.execute(
        f"""SELECT {_COLUMNS} FROM jobs
             WHERE source_id IN (SELECT source_id FROM source_follows WHERE user_id = ?)""",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def match_for_pasted(conn, user_id: int, job: dict) -> int | None:
    match = find_match(job, source_jobs_for(conn, user_id))
    return match["id"] if match else None


# Further along wins when both copies are on My Jobs.
_STATUS_RANK = {
    "saved": 0,
    "waiting": 1,
    "withdrawn": 1,
    "rejected_no_interview": 1,
    "interviewing": 2,
    "rejected_interviewed": 2,
    "offer": 3,
}


def merge_into(pasted_id: int, source_id: int, user_id: int) -> None:
    """Moves the person's rows from their pasted job onto the source job, then deletes the pasted
    job. One transaction: all of it or none."""
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _merge(conn, pasted_id, source_id, user_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    logger.info("merged pasted job %s into source job %s for user %s", pasted_id, source_id, user_id)


def _one(conn, sql: str, *params):
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def _merge(conn, pasted: int, source: int, uid: int) -> None:
    # The application: the one further along stays, with the other's notes, dates and history.
    pa = _one(conn, "SELECT * FROM applications WHERE user_id = ? AND job_id = ?", uid, pasted)
    sa = _one(conn, "SELECT * FROM applications WHERE user_id = ? AND job_id = ?", uid, source)
    if pa and sa:

        def rank(a: dict):
            return (
                -_STATUS_RANK.get(a["status"], 0),
                a["applied_at"] is None,
                a["applied_at"] or "",
                a["created_at"],
            )

        keep, drop = sorted([pa, sa], key=rank)
        notes = [n for n in (keep["notes"], drop["notes"]) if n and n.strip()]
        conn.execute(
            """UPDATE applications SET notes = ?, applied_at = COALESCE(applied_at, ?),
                      resume_doc_id = COALESCE(resume_doc_id, ?),
                      cover_letter_doc_id = COALESCE(cover_letter_doc_id, ?)
                WHERE id = ?""",
            ("\n\n".join(dict.fromkeys(notes)) or None, drop["applied_at"], drop["resume_doc_id"],
             drop["cover_letter_doc_id"], keep["id"]),
        )  # fmt: skip
        conn.execute("UPDATE application_status_history SET application_id = ? WHERE application_id = ?",
                     (keep["id"], drop["id"]))  # fmt: skip
        conn.execute("UPDATE generated_documents SET application_id = ? WHERE application_id = ?",
                     (keep["id"], drop["id"]))  # fmt: skip
        conn.execute("DELETE FROM applications WHERE id = ?", (drop["id"],))
    conn.execute("UPDATE applications SET job_id = ? WHERE user_id = ? AND job_id = ?", (source, uid, pasted))

    # The Jobby chat about it: the longer conversation stays.
    pt = _one(conn, "SELECT id FROM jobby_threads WHERE user_id = ? AND job_id = ?", uid, pasted)
    st = _one(conn, "SELECT id FROM jobby_threads WHERE user_id = ? AND job_id = ?", uid, source)
    if pt and st:

        def count(t: dict) -> int:
            sql = "SELECT COUNT(*) FROM jobby_messages WHERE thread_id = ?"
            return conn.execute(sql, (t["id"],)).fetchone()[0]

        drop = st if count(pt) > count(st) else pt
        conn.execute("DELETE FROM jobby_messages WHERE thread_id = ?", (drop["id"],))
        conn.execute("DELETE FROM jobby_threads WHERE id = ?", (drop["id"],))
    conn.execute(
        "UPDATE jobby_threads SET job_id = ? WHERE user_id = ? AND job_id = ?", (source, uid, pasted)
    )

    # Queue and digest: the source copy's own rows win.
    conn.execute(
        """DELETE FROM scoring_queue WHERE user_id = ? AND job_id = ?
             AND EXISTS (SELECT 1 FROM scoring_queue WHERE user_id = ? AND job_id = ?)""",
        (uid, pasted, uid, source),
    )
    conn.execute(
        """DELETE FROM digest_entries WHERE user_id = ? AND job_id = ?
             AND digest_date IN (SELECT digest_date FROM digest_entries WHERE user_id = ? AND job_id = ?)""",
        (uid, pasted, uid, source),
    )
    for table in (
        "scoring_queue",
        "digest_entries",
        "job_scores",
        "feedback",
        "generated_documents",
        "llm_calls",
    ):
        conn.execute(f"UPDATE {table} SET job_id = ? WHERE job_id = ?", (source, pasted))

    # What the pasted copy knew that the source copy doesn't.
    conn.execute(
        """UPDATE jobs SET (salary_min, salary_max, salary_currency, salary_period) =
               (SELECT salary_min, salary_max, salary_currency, salary_period FROM jobs WHERE id = :p)
            WHERE id = :s AND salary_min IS NULL AND salary_max IS NULL""",
        {"p": pasted, "s": source},
    )
    conn.execute(
        "UPDATE jobs SET posted_at = (SELECT posted_at FROM jobs WHERE id = :p) "
        "WHERE id = :s AND posted_at IS NULL",
        {"p": pasted, "s": source},
    )
    conn.execute("DELETE FROM jobs WHERE id = ? AND user_id = ? AND source_id IS NULL", (pasted, uid))


def merge_for_new_source_job(job_id: int) -> int:
    """After a source saves a new job: merges any follower's pasted copy of it. Returns how many."""
    with get_db() as conn:
        job = _one(conn, f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", job_id)
        if not job or job["source_id"] is None:
            return 0
        pasted = [
            dict(r)
            for r in conn.execute(
                f"""SELECT {_COLUMNS} FROM jobs WHERE source_id IS NULL AND user_id IN
                      (SELECT user_id FROM source_follows WHERE source_id = ?)""",
                (job["source_id"],),
            ).fetchall()
        ]
        identity, title, org = (
            posting_identity(job["url"]),
            norm_title(job["title"]),
            norm_org(job["company"]),
        )
        maybe = [
            p
            for p in pasted
            if (identity and posting_identity(p["url"]) == identity)
            or (norm_title(p["title"]) == title and norm_org(p["company"]) == org)
        ]
        pairs = [(p, match_for_pasted(conn, p["user_id"], p)) for p in maybe]
    merged = 0
    for p, match in pairs:
        if match == job_id:
            merge_into(p["id"], job_id, p["user_id"])
            merged += 1
    return merged


def merge_existing_duplicates() -> int:
    """Every pasted job that has a source copy, merged (at startup; free, no Claude)."""
    with get_db() as conn:
        pasted = [
            dict(r)
            for r in conn.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE source_id IS NULL AND user_id IS NOT NULL"
            ).fetchall()
        ]
        pairs = [(p, match_for_pasted(conn, p["user_id"], p)) for p in pasted]
    merged = 0
    for p, match in pairs:
        if match is not None:
            merge_into(p["id"], match, p["user_id"])
            merged += 1
    if merged:
        logger.info("merged %d pasted job(s) into their source copies", merged)
    return merged
