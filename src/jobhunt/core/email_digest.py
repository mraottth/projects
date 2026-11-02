"""The daily email: New & Noteworthy jobs worth a look, sent once the daily run finishes
(Cloud Scheduler runs it at 2:00pm ET). "Worth a look" means High interest with a High or Med fit,
per the user's choice; each job is emailed once (digest_entries.emailed_at), and a day with
nothing new sends nothing.

Sent through Gmail's SMTP with an app password, from the user's own address to themselves:
EMAIL_ADDRESS and GMAIL_APP_PASSWORD (a Secret Manager secret in production; see the Makefile's
email-secret), optional EMAIL_TO (defaults to EMAIL_ADDRESS) and APP_URL (adds "Open in the app"
links). With either required value unset, nothing is sent -- local dev and tests never email."""

import html
import logging
import os
import smtplib
from datetime import date, timedelta
from email.message import EmailMessage

from jobhunt.core.digest import get_digest
from jobhunt.db.connection import PLACEHOLDER_ADMIN_EMAIL, get_db
from jobhunt.core.users import current_user_id, get_user, optional_user_id

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_TIMEOUT = 20
GOOD_FITS = ("High", "Med")
# Today's digest plus yesterday's: a source added in the evening puts its jobs in that day's
# digest after the email already went out.
LOOKBACK_DAYS = 1

_CURRENCY_SYMBOLS = {"USD": "$", "CAD": "CA$", "GBP": "£", "EUR": "€"}
_PERIODS = {"year": "", "hour": "/hr"}  # yearly is the assumed default: no suffix


class EmailNotConfigured(Exception):
    pass


def _config() -> dict:
    sender = os.environ.get("EMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not password:
        raise EmailNotConfigured(
            "email isn't set up: store a Gmail app password with `make email-secret`, then "
            "redeploy with EMAIL_ADDRESS=you@gmail.com added to `make deploy`"
        )
    return {
        "sender": sender,
        "password": password,
        "to": _recipient(sender),
        "app_url": (os.environ.get("APP_URL") or "").rstrip("/"),
    }


def _recipient(sender: str) -> str:
    """Each person's email goes to their own sign-in address (D-021). The admin's can be
    redirected with EMAIL_TO, as before; with nobody signed in (or the admin's address not yet
    known) it's the sender's own inbox."""
    user_id = optional_user_id()
    user = get_user(user_id) if user_id is not None else None
    if user is None or user["is_admin"]:
        if os.environ.get("EMAIL_TO"):
            return os.environ["EMAIL_TO"]
    if user is None or user["email"] == PLACEHOLDER_ADMIN_EMAIL:
        return sender
    return user["email"]


def send_plain_email(to: str, subject: str, body: str) -> dict:
    """A one-off notice (access requests and decisions). Never raises: a failed or unconfigured
    send is returned, so it can't break what triggered it."""
    try:
        config = _config()
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = f"Job Search Assistant <{config['sender']}>"
        message["To"] = to
        message.set_content(body)
        _send(message, config)
    except EmailNotConfigured:
        return {"sent": False, "reason": "not configured"}
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("sending %r to %s failed", subject, to)
        return {"sent": False, "reason": f"error: {exc}"}
    return {"sent": True, "to": to}


def admin_address() -> str | None:
    """Where admin notices go: the admin's sign-in address, else the sender's inbox."""
    with get_db() as conn:
        row = conn.execute("SELECT email FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1").fetchone()
    if row and row["email"] != PLACEHOLDER_ADMIN_EMAIL:
        return row["email"]
    return os.environ.get("EMAIL_TO") or os.environ.get("EMAIL_ADDRESS")


def email_status() -> dict:
    try:
        return {"configured": True, "to": _config()["to"]}
    except EmailNotConfigured:
        return {"configured": False, "to": None}


def _salary_text(job: dict) -> str | None:
    """Mirrors the job card's salaryText(): "$120k-150k" (yearly), "$45-60/hr"."""
    low, high = job.get("salary_min"), job.get("salary_max")
    if not low and not high:
        return None

    def fmt(n: float) -> str:
        return f"{round(n / 1000)}k" if n >= 1000 else f"{n:g}"

    symbol = _CURRENCY_SYMBOLS.get(job.get("salary_currency") or "USD", f"{job.get('salary_currency')} ")
    amount = f"{fmt(low)}-{fmt(high)}" if low and high and low != high else fmt(low or high)
    period = job.get("salary_period")
    return f"{symbol}{amount}{_PERIODS.get(period, f'/{period}') if period else ''}"


def qualifying_jobs(include_emailed: bool = False) -> list[dict]:
    """Recent digest jobs with High interest and a High/Med fit, best first. get_digest() has
    already dropped dismissed, stale and general-application postings."""
    jobs, seen = [], set()
    for days_ago in range(LOOKBACK_DAYS + 1):
        for job in get_digest((date.today() - timedelta(days=days_ago)).isoformat()):
            if job["id"] in seen or job["interest_fit"] != "High" or job["candidate_fit"] not in GOOD_FITS:
                continue
            if job["emailed_at"] and not include_emailed:
                continue
            seen.add(job["id"])
            jobs.append(job)
    return jobs


def _subject(jobs: list[dict]) -> str:
    first = f"{jobs[0]['title']} at {jobs[0]['company']}"
    if len(jobs) == 1:
        return f"New job match: {first}"
    return f"{len(jobs)} new job matches: {first}, +{len(jobs) - 1} more"


def _job_line(job: dict) -> str:
    return " · ".join(p for p in (job.get("company"), job.get("location"), _salary_text(job)) if p)


def build_message(jobs: list[dict], sender: str, to: str, app_url: str, subject: str | None = None):
    message = EmailMessage()
    message["Subject"] = subject or _subject(jobs)
    message["From"] = f"Job Search Assistant <{sender}>"
    message["To"] = to

    text = ["New matches in New & Noteworthy:", ""]
    cards = []
    for job in jobs:
        app_link = f"{app_url}/jobs/{job['id']}" if app_url else None
        text += [job["title"], _job_line(job), job.get("summary_text") or ""]
        text += [link for link in (job.get("url"), app_link) if link] + [""]

        links = []
        if job.get("url"):
            links.append(f'<a href="{html.escape(job["url"])}">View posting</a>')
        if app_link:
            links.append(f'<a href="{html.escape(app_link)}">Open in the app</a>')
        cards.append(
            '<div style="padding:14px 0;border-top:1px solid #e6e4dd">'
            f'<div style="font-size:16px;font-weight:600;color:#1d1d1b">{html.escape(job["title"])}</div>'
            f'<div style="font-size:13px;color:#6b6a65;margin:2px 0 6px">{html.escape(_job_line(job))}</div>'
            f'<div style="font-size:14px;color:#1d1d1b">{html.escape(job.get("summary_text") or "")}</div>'
            f'<div style="font-size:13px;margin-top:6px">{" &nbsp;·&nbsp; ".join(links)}</div>'
            "</div>"
        )
    footer = f'<p style="font-size:12px;color:#6b6a65">{len(jobs)} High-interest match'
    footer += "es" if len(jobs) != 1 else ""
    footer += " from today's check."
    if app_url:
        footer += f' <a href="{html.escape(app_url)}/">See everything in New &amp; Noteworthy</a>.'
    footer += "</p>"

    message.set_content("\n".join(text))
    message.add_alternative(
        '<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:620px">'
        '<h2 style="font-size:18px;margin:0 0 6px">New job matches</h2>' + "".join(cards) + footer + "</div>",
        subtype="html",
    )
    return message


def _send(message: EmailMessage, config: dict) -> None:
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as smtp:
        smtp.login(config["sender"], config["password"])
        smtp.send_message(message)


def send_new_jobs_email() -> dict:
    """Called at the end of every daily run. Emails the qualifying jobs not yet emailed, then marks
    them emailed."""
    try:
        config = _config()
    except EmailNotConfigured:
        return {"sent": False, "reason": "not configured"}
    jobs = qualifying_jobs()
    if not jobs:
        return {"sent": False, "reason": "nothing new"}

    _send(build_message(jobs, config["sender"], config["to"], config["app_url"]), config)
    with get_db() as conn:
        conn.executemany(
            "UPDATE digest_entries SET emailed_at = datetime('now') "
            "WHERE user_id = ? AND job_id = ? AND digest_date = ?",
            [(current_user_id(), job["id"], job["digest_date"]) for job in jobs],
        )
    return {"sent": True, "count": len(jobs), "to": config["to"]}


TEST_EMAIL_LIMIT = 5


def send_test_email() -> dict:
    """Settings' "Send a test email": today's top matches, emailed or not, without marking them
    emailed -- so it never changes what the real daily email sends."""
    config = _config()
    jobs = qualifying_jobs(include_emailed=True)[:TEST_EMAIL_LIMIT]
    if jobs:
        message = build_message(
            jobs, config["sender"], config["to"], config["app_url"], subject=f"[Test] {_subject(jobs)}"
        )
    else:
        message = EmailMessage()
        message["Subject"] = "[Test] Job Search Assistant email is working"
        message["From"] = f"Job Search Assistant <{config['sender']}>"
        message["To"] = config["to"]
        message.set_content(
            "Email is set up. There are no High-interest matches in New & Noteworthy right now; "
            "new ones will arrive after the daily 2:00pm ET check."
        )
    _send(message, config)
    return {"sent": True, "count": len(jobs), "to": config["to"]}
