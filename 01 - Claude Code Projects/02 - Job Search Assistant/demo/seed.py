"""Builds the public demo's database from demo/content.py (asked 2026-11-11).

Uses the real schema and the app's own code wherever no Claude call is involved: the digest
ranking, the resume edit writer, text extraction. Everything is fictional. Call `build(folder)`
with JOBHUNT_DATA_DIR / JOBHUNT_DB_PATH / JOBHUNT_RESUME_DIR already pointing into that folder
(demo/record.py does this) so nothing touches data/.
"""

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from docx import Document

from demo import content as C

DEMO_MODEL = "claude-sonnet-5"


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _write_docx(paragraphs: list[tuple[str, str]], path: Path) -> None:
    doc = Document()
    for style, text in paragraphs:
        if style == "job":
            company_title, dates = text.rsplit(" | ", 1)
            p = doc.add_paragraph(style="Normal")
            p.add_run(company_title).bold = True
            p.add_run(f" | {dates}")
        elif style == "bullet":
            doc.add_paragraph(text, style="List Bullet")
        else:
            doc.add_paragraph(text, style=style)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


def build(now: datetime | None = None) -> dict:
    """Seeds the database configured by the environment. Returns {"user_id", "email"}."""
    from jobhunt.core import digest, generate, llm
    from jobhunt.core.docx_writer import (
        CoverLetterContent,
        ResumeEditOp,
        extract_addressable_lines,
        write_cover_letter_docx,
        write_resume_edits,
    )
    from jobhunt.core.resume_parse import extract_text
    from jobhunt.core.settings import update_settings
    from jobhunt.core.users import acting_as, storage_dir
    from jobhunt.db.connection import get_db, init_db

    rng = random.Random(7)
    now = now or datetime.now(UTC).replace(microsecond=0)
    ago = lambda **kw: _ts(now - timedelta(**kw))  # noqa: E731
    day = lambda n: (now - timedelta(days=n)).date().isoformat()  # noqa: E731

    init_db()
    with get_db() as conn:
        uid = conn.execute(
            "INSERT INTO users (email, name, status, is_admin, last_seen_at, created_at, decided_at) "
            "VALUES (?, ?, 'approved', 0, ?, ?, ?)",
            (C.PERSON["email"], C.PERSON["name"], ago(minutes=5), ago(days=40), ago(days=40)),
        ).lastrowid

    with acting_as(uid):
        folder = storage_dir()

        # --- documents
        docs = {}
        for key, kind, paragraphs, is_template, archived, age in [
            ("resume", "resume", C.RESUME, 1, 0, 20),
            ("old_resume", "resume", C.OLD_RESUME, 0, 1, 900),
            ("cover", "cover_letter", C.COVER_LETTER, 1, 0, 25),
            ("old_cover", "cover_letter", C.OLD_COVER_LETTER, 0, 1, 1200),
        ]:
            name = {
                "resume": "Jordan Rivera - Resume 2026.docx",
                "old_resume": "Jordan Rivera - Resume (old).docx",
                "cover": "Jordan Rivera - Cover Letter.docx",
                "old_cover": "Cover letter - Fairway (old).docx",
            }[key]
            path = folder / f"{key}.docx"
            _write_docx(paragraphs, path)
            with get_db() as conn:
                docs[key] = conn.execute(
                    """INSERT INTO resumes (user_id, kind, file_name, file_type, stored_path, raw_text,
                       uploaded_at, source_modified_at, is_archived, is_template)
                       VALUES (?, ?, ?, 'docx', ?, ?, ?, ?, ?, ?)""",
                    (
                        uid,
                        kind,
                        name,
                        str(path),
                        extract_text(path),
                        ago(days=min(age, 40)),
                        ago(days=age),
                        archived,
                        is_template,
                    ),
                ).lastrowid

        with get_db() as conn:
            conn.execute(
                "INSERT INTO candidate_profile (user_id, text, generated_at, model, source_resume_ids_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (uid, C.CANDIDATE_PROFILE, ago(days=6), DEMO_MODEL, json.dumps(list(docs.values()))),
            )
            conn.execute(
                "INSERT INTO preference_profile (user_id, text, updated_at, based_on_feedback_count, "
                "based_on_application_events_count, model) VALUES (?, ?, ?, 14, 22, ?)",
                (uid, C.PREFERENCE_PROFILE, ago(days=3), DEMO_MODEL),
            )
        update_settings(C.SETTINGS)

        # --- sources
        sources = {}
        with get_db() as conn:
            for key, kind, name, company, config, hours, error in C.SOURCES:
                checked = ago(hours=hours)
                sources[key] = conn.execute(
                    """INSERT INTO sources (kind, name, company_name, config_json, last_polled_at,
                       last_success_at, last_error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        kind,
                        name,
                        company,
                        json.dumps(config),
                        checked,
                        ago(hours=hours if not error else hours + 24),
                        error,
                        ago(days=35),
                    ),
                ).lastrowid
                conn.execute(
                    "INSERT INTO source_follows (user_id, source_id, created_at) VALUES (?, ?, ?)",
                    (uid, sources[key], ago(days=35)),
                )
        source_company = {k: company or name for k, _, name, company, *_ in C.SOURCES}

        # --- jobs, scores and the queue
        jobs = {}
        areas = [
            "government services",
            "public health",
            "civic participation",
            "benefits access",
            "community programs",
            "climate resilience",
            "education",
        ]
        for i, (
            key,
            src,
            company,
            title,
            location,
            remote,
            salary,
            posted,
            interest,
            fit,
            summary,
            rationale,
        ) in enumerate(C.JOBS):
            company = company or source_company[src]
            pay = ""
            if salary:
                lo, hi, period = salary
                pay = (
                    f"The range for this role is ${lo:,}-${hi:,} per {period}. "
                    if period == "year"
                    else f"This role pays ${lo}-${hi} per hour. "
                )
            description = C.DESCRIPTION_TEMPLATE.format(
                company=company,
                area=areas[i % len(areas)],
                size=rng.choice([40, 60, 120, 250, 400]),
                title=title,
                verb=rng.choice(["lead", "shape", "strengthen"]),
                summary=summary or "You'll partner with program teams to measure what works.",
                extra=rng.choice(
                    [
                        "Build and mentor a small team of analysts and data engineers.",
                        "Own our data warehouse and the reporting built on it.",
                        "Design evaluations with external research partners.",
                    ]
                ),
                years=rng.choice([5, 7, 8, 10]),
                pay=pay,
            )
            slug = title.lower().replace(" ", "-").replace(",", "").replace("&", "and")
            first_seen = ago(days=(posted or 3), hours=-2)
            with get_db() as conn:
                job_id = conn.execute(
                    """INSERT INTO jobs (source_id, external_id, dedupe_key, title, company, location,
                       remote_type, salary_min, salary_max, salary_currency, salary_period,
                       description_text, url, posted_at, first_seen_at, last_seen_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sources[src],
                        f"demo-{key}",
                        f"demo:{key}",
                        title,
                        company,
                        location,
                        remote,
                        salary[0] if salary else None,
                        salary[1] if salary else None,
                        "USD" if salary else None,
                        salary[2] if salary else None,
                        description,
                        f"https://{company.lower().replace(' ', '')}.example/jobs/{slug}",
                        day(posted) if posted is not None else None,
                        first_seen,
                        ago(hours=3),
                    ),
                ).lastrowid
                jobs[key] = job_id
                scored = interest is not None
                conn.execute(
                    """INSERT INTO scoring_queue (user_id, job_id, prefilter_score, priority, status,
                       queued_at, scored_at, screen, screened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uid,
                        job_id,
                        rng.uniform(0, 5),
                        rng.uniform(0, 10),
                        "scored" if scored else "pending",
                        first_seen,
                        first_seen if scored else None,
                        "drop" if key in C.SET_ASIDE else "keep",
                        first_seen,
                    ),
                )
                if scored:
                    score = {"High": (82, 95), "Med": (55, 74), "Low": (15, 40)}
                    conn.execute(
                        """INSERT INTO job_scores (user_id, job_id, scored_at, model, interest_fit,
                           interest_score, candidate_fit, candidate_fit_score, summary_text,
                           rationale_text, prompt_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v3')""",
                        (
                            uid,
                            job_id,
                            first_seen,
                            DEMO_MODEL,
                            interest,
                            rng.randint(*score[interest]),
                            fit,
                            rng.randint(*score[fit]),
                            summary,
                            rationale,
                        ),
                    )

        # --- New & Noteworthy (the app's own ranking)
        for keys, n in ((C.DIGEST_YESTERDAY, 1), (C.DIGEST_TODAY, 0)):
            for key in keys:
                digest.add_to_digest(jobs[key], day(n))
        with get_db() as conn:
            conn.execute(
                "UPDATE digest_entries SET emailed_at = ? WHERE digest_date = ?", (ago(days=1), day(1))
            )

        # --- feedback and My Jobs
        with get_db() as conn:
            for key, mark in C.FEEDBACK:
                conn.execute(
                    "INSERT INTO feedback (user_id, job_id, interested, created_at) VALUES (?, ?, ?, ?)",
                    (uid, jobs[key], mark, ago(days=rng.randint(1, 10))),
                )
            apps = {}
            for key, status, applied, notes, history in C.APPLICATIONS:
                conn.execute(
                    "INSERT INTO feedback (user_id, job_id, interested, created_at) "
                    "VALUES (?, ?, 'interested', ?)",
                    (uid, jobs[key], ago(days=history[0][1])),
                )
                apps[key] = conn.execute(
                    """INSERT INTO applications (user_id, job_id, status, applied_at, notes, created_at,
                       updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uid,
                        jobs[key],
                        status,
                        ago(days=applied) if applied is not None else None,
                        notes,
                        ago(days=history[0][1]),
                        ago(days=history[-1][1]),
                    ),
                ).lastrowid
                for st, d in history:
                    conn.execute(
                        "INSERT INTO application_status_history (application_id, status, changed_at) "
                        "VALUES (?, ?, ?)",
                        (apps[key], st, ago(days=d, hours=-1)),
                    )

        # --- tailored documents: the real edit writer, with fixed plans instead of Claude
        out = folder / "generated"
        out.mkdir(parents=True, exist_ok=True)
        resume_path = folder / "resume.docx"
        for key, plan in C.TAILORED_RESUMES.items():
            ops = [ResumeEditOp(op=op, anchor_text=a, new_text=t) for op, a, t in plan]
            path = out / f"tailored-resume-{key}.docx"
            write_resume_edits(resume_path, ops, path)
            text = generate._render_lines_markdown(
                generate._apply_ops_to_lines(extract_addressable_lines(resume_path), ops)
            )
            _save_generated(
                get_db, uid, jobs[key], apps[key], "resume", docs["resume"], text, path, ago(days=3)
            )
        for key, paragraphs in C.TAILORED_COVER_LETTERS.items():
            path = out / f"tailored-cover-letter-{key}.docx"
            write_cover_letter_docx(folder / "cover.docx", CoverLetterContent(paragraphs=paragraphs), path)
            _save_generated(
                get_db,
                uid,
                jobs[key],
                apps[key],
                "cover_letter",
                docs["cover"],
                "\n\n".join(paragraphs),
                path,
                ago(days=3),
            )

        update_settings(
            {
                "application_insights": {
                    "text": C.APPLICATION_INSIGHTS,
                    "generated_at": (now - timedelta(hours=20)).isoformat(timespec="seconds"),
                    "applications": len(C.APPLICATIONS),
                    "applied": sum(1 for a in C.APPLICATIONS if a[2] is not None),
                },
                "last_daily_email": {
                    "at": (now - timedelta(hours=3)).isoformat(timespec="seconds"),
                    "sent": True,
                    "count": 4,
                    "to": C.PERSON["email"],
                    "reason": None,
                    "on_list_today": len(C.DIGEST_TODAY),
                    "best_interest_today": "High",
                },
                "last_refresh_at": (now - timedelta(hours=3)).isoformat(timespec="seconds"),
            }
        )

        # --- Jobby
        with get_db() as conn:
            for key, title, days, messages in C.JOBBY:
                start = now - timedelta(days=days, minutes=len(messages) * 3)
                thread = conn.execute(
                    "INSERT INTO jobby_threads (user_id, job_id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (uid, jobs[key] if key else None, title, _ts(start), _ts(start + timedelta(minutes=30))),
                ).lastrowid
                for n, (role, text) in enumerate(messages):
                    conn.execute(
                        "INSERT INTO jobby_messages (thread_id, role, content, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (thread, role, text, _ts(start + timedelta(minutes=3 * n))),
                    )

        # --- runs ("Last updated") and two weeks of usage
        with get_db() as conn:
            for n in range(14, -1, -1):
                started = now - timedelta(days=n, hours=3 if n == 0 else 0)
                conn.execute(
                    """INSERT INTO runs (user_id, run_date, started_at, finished_at, sources_polled,
                       jobs_fetched, jobs_deduped, prefiltered_count, llm_scored_count, cap_used,
                       carried_over_count, errors_json, status, started_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', 'ok', 'scheduled')""",
                    (
                        uid,
                        started.date().isoformat(),
                        _ts(started),
                        _ts(started + timedelta(minutes=6)),
                        len(C.SOURCES),
                        rng.randint(180, 260),
                        rng.randint(170, 240),
                        rng.randint(5, 14),
                        rng.randint(5, 12),
                        rng.randint(5, 12),
                        rng.randint(0, 4),
                    ),
                )
            calls = []
            job_ids = list(jobs.values())
            for n in range(14):
                for _ in range(rng.randint(5, 11)):
                    calls.append(
                        (
                            "scoring",
                            rng.choice(job_ids),
                            1900,
                            420,
                            5200 if _ else 0,
                            0 if _ else 5200,
                            "scoring",
                            n,
                        )
                    )
            calls += [
                ("generation", jobs["cs_dir"], 6200, 1500, 0, 0, "resume (docx template, edit plan)", 3),
                ("generation", jobs["cs_dir"], 4100, 900, 0, 0, "cover_letter (docx template)", 3),
                ("generation", jobs["hb_dir"], 6100, 1400, 0, 0, "resume (docx template, edit plan)", 3),
                ("generation", jobs["cs_dir"], 900, 650, 11800, 0, "jobby chat", 1),
                ("generation", jobs["hb_dir"], 1100, 900, 0, 12400, "jobby chat", 2),
                ("generation", None, 1500, 1100, 9800, 0, "jobby chat", 4),
                ("generation", None, 1200, 800, 0, 9600, "jobby chat", 9),
                ("profile_rebuild", None, 7600, 1700, 0, 0, "candidate profile", 6),
                ("profile_rebuild", None, 3900, 600, 0, 0, "preference profile", 3),
                ("profile_rebuild", None, 2800, 900, 0, 0, "application insights", 1),
                ("title screen", None, 3200, 60, 2600, 0, f"title screen ({len(C.JOBS)} jobs)", 0),
            ]
            for call_type, job_id, inp, outp, read, write, note, n in calls:
                model = "claude-haiku-5-5" if call_type == "title screen" else DEMO_MODEL
                call_type = "parsing" if call_type == "title screen" else call_type
                cost = llm.estimate_cost_usd(
                    model, inp, outp, cache_read_tokens=read, cache_write_tokens=write
                )
                conn.execute(
                    """INSERT INTO llm_calls (user_id, call_type, job_id, model, input_tokens, output_tokens,
                       estimated_cost_usd, created_at, note, cache_read_tokens, cache_write_tokens)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uid,
                        call_type,
                        job_id,
                        model,
                        inp,
                        outp,
                        cost,
                        ago(days=n, minutes=rng.randint(10, 600)),
                        note,
                        read,
                        write,
                    ),
                )
    return {"user_id": uid, "email": C.PERSON["email"], "jobs": jobs, "docs": docs}


def _save_generated(get_db, uid, job_id, app_id, kind, based_on, text, path, created) -> None:
    with get_db() as conn:
        doc_id = conn.execute(
            """INSERT INTO generated_documents (user_id, job_id, application_id, kind, based_on_resume_id,
               content_text, file_path, created_at, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (uid, job_id, app_id, kind, based_on, text, str(path), created, DEMO_MODEL),
        ).lastrowid
        column = "resume_doc_id" if kind == "resume" else "cover_letter_doc_id"
        conn.execute(f"UPDATE applications SET {column} = ? WHERE id = ?", (doc_id, app_id))
