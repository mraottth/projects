"""Each person's monthly Claude budget (step 4 of D-021; `users.monthly_budget_usd`, default $10,
set on /admin).

- Spend is the estimated cost of their own calls this calendar month (`llm_calls.user_id`, priced
  by core/llm.py). Shared work, like the scheduled run reading a job board, is nobody's.
- `check()` runs at the one choke point every paid call passes through, `llm.get_client()`:
  someone at or over budget gets `BudgetExceeded`, an `LLMError`, so every route and the pipeline
  already turn it into a plain message. The daily run also skips their scoring up front.
- The admin is exempt: it's their key and their money. Their spend still shows.
- `after_call()` tells the admin, once per person per month, when someone passes 80% and when
  they reach 100% (`users.budget_alert_month` / `budget_block_month`)."""

import logging
import os
from datetime import date

from jobhunt.db.connection import get_db

log = logging.getLogger(__name__)

ALERT_AT = 0.8


def spend_this_month(user_id: int) -> float:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) AS spend FROM llm_calls "
            "WHERE user_id = ? AND created_at >= date('now', 'start of month')",
            (user_id,),
        ).fetchone()
    return float(row["spend"])


def _user(user_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def status(user_id: int) -> dict:
    user = _user(user_id) or {}
    budget = float(user.get("monthly_budget_usd") or 0)
    spend = spend_this_month(user_id)
    exempt = bool(user.get("is_admin"))
    return {
        "spend_this_month": round(spend, 4),
        "monthly_budget_usd": budget,
        "remaining": max(0.0, budget - spend),
        "over": not exempt and spend >= budget,
        "exempt": exempt,
    }


def _next_month_label() -> str:
    today = date.today()
    first = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
    return f"{first:%b} {first.day}"


def exceeded_message(budget: float) -> str:
    return (
        f"You've used your ${budget:.2f} Claude budget for this month. It resets on "
        f"{_next_month_label()}, or the admin can raise it."
    )


def check(user_id: int) -> None:
    """Raises BudgetExceeded when this person has used up this month's budget."""
    from jobhunt.core.llm import BudgetExceeded  # here: llm imports this module

    state = status(user_id)
    if state["over"]:
        raise BudgetExceeded(exceeded_message(state["monthly_budget_usd"]))


def after_call(user_id: int) -> None:
    """Emails the admin the first time this month someone passes 80%, and when they reach 100%.
    Never raises: a failed notice mustn't break the call that triggered it."""
    try:
        user = _user(user_id)
        if not user or user["is_admin"] or not user["monthly_budget_usd"]:
            return
        budget = float(user["monthly_budget_usd"])
        spend = spend_this_month(user_id)
        month = date.today().strftime("%Y-%m")
        if spend >= budget and user.get("budget_block_month") != month:
            _notify(user, spend, budget, "budget_block_month", month, reached=True)
        elif spend >= ALERT_AT * budget and user.get("budget_alert_month") != month:
            _notify(user, spend, budget, "budget_alert_month", month, reached=False)
    except Exception:  # noqa: BLE001
        log.exception("budget alert for user %s failed", user_id)


def _notify(user: dict, spend: float, budget: float, column: str, month: str, reached: bool) -> None:
    from jobhunt.core.email_digest import admin_address, send_plain_email

    with get_db() as conn:
        conn.execute(f"UPDATE users SET {column} = ? WHERE id = ?", (month, user["id"]))
        if reached:  # reaching 100% also covers the 80% notice
            conn.execute("UPDATE users SET budget_alert_month = ? WHERE id = ?", (month, user["id"]))
    who = user["name"] or user["email"]
    app_url = (os.environ.get("APP_URL") or "").rstrip("/")
    what = "has reached" if reached else "has used"
    after = (
        " Their paid features (scoring, Jobby, tailoring, discovery) are paused until next month or "
        "until you raise it."
        if reached
        else ""
    )
    to = admin_address()
    if to:
        send_plain_email(
            to,
            f"Budget: {who} {'reached' if reached else 'passed 80% of'} their monthly limit",
            f"{who} ({user['email']}) {what} ${spend:.2f} of their ${budget:.2f} Claude budget this month."
            f"{after}\n\nChange it here: {app_url}/admin\n",
        )
