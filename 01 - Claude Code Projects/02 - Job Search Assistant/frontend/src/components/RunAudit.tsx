import { type ReactNode, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  type RunAudit,
  type RunCalendar,
  type RunDayStatus,
} from "../api";
import { JobLink } from "./JobLink";
import { FitBadge, ScoreButton } from "./ScoredJobCard";

// The daily run audit (asked 2026-11-17, moved from Settings to under New & Noteworthy's list the
// same day): GET /api/digest/run-audit, core/run_audit.py.

// A local "YYYY-MM-DD" as a readable day ("Today", "Yesterday", "Mon, Nov 16").
function dayName(iso: string): string {
  const today = new Date();
  const local = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (iso === local(today)) return "Today";
  if (iso === local(yesterday)) return "Yesterday";
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

// Run times are stored as UTC, either ISO with an offset or SQLite's "YYYY-MM-DD HH:MM:SS".
function clock(value: string): string {
  const d = new Date(
    value.includes("T") ? value : `${value.replace(" ", "T")}Z`,
  );
  return d.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

const DAY_STATUS: Record<RunDayStatus, string> = {
  ran: "a check ran",
  read: "only the 10am read ran, nothing scored yet",
  failed: "a daily check failed",
  none: "no check ran",
};
const WEEKDAYS = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];

// "YYYY-MM" moved by `by` months.
function shiftMonth(month: string, by: number): string {
  const [y, m] = month.split("-").map(Number);
  const d = new Date(y, m - 1 + by, 1);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
}

/**
 * The audit's day picker (asked 2026-11-18, in place of a dropdown of the last 14 days): a month
 * of days colored by the person's checks, green when one ran, red when a daily check failed for
 * them, gray when none ran (GET /api/digest/run-calendar). Days before their account and days to
 * come can't be picked.
 */
function AuditCalendar({
  selected,
  onPick,
  reload,
  open,
}: {
  selected: string;
  onPick: (day: string) => void;
  reload: number;
  open: boolean;
}) {
  const [month, setMonth] = useState(selected.slice(0, 7));
  const [cal, setCal] = useState<RunCalendar | null>(null);

  // Opens on the chosen day's month, wherever it was left.
  useEffect(() => setMonth(selected.slice(0, 7)), [selected, open]);
  useEffect(() => {
    let live = true;
    api
      .getRunCalendar(month)
      .then((c) => live && setCal(c))
      .catch(() => live && setCal(null));
    return () => {
      live = false;
    };
  }, [month, reload]);

  const [y, m] = month.split("-").map(Number);
  const lead = new Date(y, m - 1, 1).getDay();
  const count = new Date(y, m, 0).getDate();
  const days = cal?.month === month ? cal.days : {};
  const title = new Date(y, m - 1, 1).toLocaleDateString(undefined, {
    month: "long",
    year: "numeric",
  });

  return (
    <div className="audit-cal">
      <div className="audit-cal-head">
        <button
          className="link"
          onClick={() => setMonth(shiftMonth(month, -1))}
          disabled={!cal || month <= cal.first.slice(0, 7)}
          aria-label="Previous month"
        >
          ‹
        </button>
        <strong>{title}</strong>
        <button
          className="link"
          onClick={() => setMonth(shiftMonth(month, 1))}
          disabled={!cal || month >= cal.today.slice(0, 7)}
          aria-label="Next month"
        >
          ›
        </button>
      </div>
      <div className="audit-cal-grid">
        {WEEKDAYS.map((w) => (
          <span key={w} className="audit-cal-weekday">
            {w}
          </span>
        ))}
        {Array.from({ length: lead }, (_, i) => (
          <span key={`lead-${i}`} />
        ))}
        {Array.from({ length: count }, (_, i) => {
          const iso = `${month}-${String(i + 1).padStart(2, "0")}`;
          const status = days[iso];
          return (
            <button
              key={iso}
              className={[
                "audit-cal-day",
                status ?? "off",
                iso === selected && "selected",
                iso === cal?.today && "today",
              ]
                .filter(Boolean)
                .join(" ")}
              disabled={!status}
              onClick={() => onPick(iso)}
              aria-pressed={iso === selected}
              title={
                status ? `${dayName(iso)}: ${DAY_STATUS[status]}` : undefined
              }
            >
              {i + 1}
            </button>
          );
        })}
      </div>
      <div className="audit-cal-legend small muted">
        <span>
          <i className="audit-cal-dot ran" /> Ran
        </span>
        <span>
          <i className="audit-cal-dot read" /> Read only
        </span>
        <span>
          <i className="audit-cal-dot none" /> Didn't run
        </span>
        <span>
          <i className="audit-cal-dot failed" /> Failed
        </span>
      </div>
    </div>
  );
}

/**
 * The chosen day, with its status color, as a button that opens the calendar (asked 2026-11-18:
 * hidden until wanted). It floats over the card, and closes on a pick, an outside click or Escape.
 */
function DayPicker({
  day,
  status,
  onPick,
  reload,
}: {
  day: string;
  status: RunDayStatus;
  onPick: (day: string) => void;
  reload: number;
}) {
  const ref = useRef<HTMLDetailsElement>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const close = (e: MouseEvent | KeyboardEvent) => {
      const el = ref.current;
      if (!el?.open) return;
      if (
        e instanceof KeyboardEvent
          ? e.key === "Escape"
          : !el.contains(e.target as Node)
      )
        el.open = false;
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, []);
  return (
    <details
      className="ms-dropdown audit-day-picker"
      ref={ref}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary title={`${dayName(day)}: ${DAY_STATUS[status]}`}>
        <span className="audit-day-label">
          <i className={`audit-cal-dot ${status}`} />
          {dayName(day)}
        </span>
      </summary>
      <div className="audit-day-popover">
        <AuditCalendar
          selected={day}
          open={open}
          reload={reload}
          onPick={(d) => {
            onPick(d);
            if (ref.current) ref.current.open = false;
          }}
        />
      </div>
    </details>
  );
}

/** A list that shows its first few rows, then "Show all N". */
function ShortList<T>({
  items,
  render,
  first = 10,
}: {
  items: T[];
  render: (item: T) => ReactNode;
  first?: number;
}) {
  const [all, setAll] = useState(false);
  const shown = all ? items : items.slice(0, first);
  return (
    <>
      <ul className="audit-list">{shown.map(render)}</ul>
      {items.length > first && (
        <button className="link small" onClick={() => setAll(!all)}>
          {all ? "Show fewer" : `Show all ${items.length}`}
        </button>
      )}
    </>
  );
}

/**
 * The daily run audit (asked 2026-11-17): for a day, the checks that ran, which organizations and
 * boards they covered, the new jobs per source, the jobs scored, and the jobs waiting now.
 */
export function RunAuditCard({ reload = 0 }: { reload?: number }) {
  const [day, setDay] = useState<string | undefined>(undefined);
  const [audit, setAudit] = useState<RunAudit | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Bumped when a waiting job is scored here, so it moves from Waiting to Scored.
  const [scoredHere, setScoredHere] = useState(0);

  useEffect(() => {
    api
      .getRunAudit(day)
      .then((a) => {
        setAudit(a);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [day, reload, scoredHere]);

  if (error)
    return (
      <div className="card">
        <h2>Daily run audit</h2>
        <p className="error-text small">{error}</p>
      </div>
    );
  if (!audit) return null;
  const newTotal = audit.new_jobs.reduce((n, s) => n + s.jobs, 0);
  const rescored = audit.scored.filter((s) => s.rescored).length;
  const { organizations: orgs, boards } = audit.sources;
  const q = audit.queue;

  return (
    <div className="card run-audit">
      <div className="usage-head">
        <h2>Daily run audit</h2>
        <DayPicker
          day={audit.date}
          // The calendar's rule (core/run_audit.calendar): failed, then ran (scored), then read only.
          status={
            audit.failures.length
              ? "failed"
              : audit.checks.some((c) => c.kind !== "read")
                ? "ran"
                : audit.checks.length
                  ? "read"
                  : "none"
          }
          onPick={setDay}
          reload={reload}
        />
      </div>

      <div className="audit-section">
        <h3>Checks</h3>
        {audit.failures.map((f) => (
          <p key={f.started_at} className="small error-text">
            Failed at {clock(f.started_at)}: {f.message}
          </p>
        ))}
        {audit.checks.length === 0 ? (
          <p className="small">No check ran this day.</p>
        ) : (
          // The 10am read stops after the relevance check; other checks score too (asked 2026-11-18).
          <ul className="audit-list">
            {audit.checks.map((c) => (
              <li key={c.started_at} className="small">
                {clock(c.started_at)}:{" "}
                {c.kind === "read" ? (
                  <span className="muted">
                    read sources and ran the relevance check, no scoring
                  </span>
                ) : (
                  `read sources and scored ${c.scored} ${c.scored === 1 ? "job" : "jobs"}`
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="audit-section">
        <h3>Sources checked</h3>
        {audit.checks.length === 0 ? (
          <p className="small muted">None.</p>
        ) : !audit.sources.recorded ? (
          <p className="small muted">
            Not recorded for checks before Nov 17, 2026.
          </p>
        ) : (
          <>
            <p className="small">
              Organizations: <strong>{orgs.checked}</strong> of {orgs.total} ·
              Job boards: <strong>{boards.checked}</strong> of {boards.total}
            </p>
            {audit.sources.failed.length > 0 && (
              // Closed by default (asked 2026-11-17): what failed, for when you want to look.
              <details className="audit-failures">
                <summary className="small">
                  {audit.sources.failed.length} couldn't be read
                </summary>
                <ul className="audit-list">
                  {audit.sources.failed.map((f) => (
                    <li key={f.name} className="small">
                      <span className="error-text">{f.name}:</span>{" "}
                      <span className="muted audit-error" title={f.error}>
                        {f.error}
                      </span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </>
        )}
      </div>

      <div className="audit-section">
        <h3>
          New jobs <span className="chip-count">{newTotal}</span>
        </h3>
        {audit.new_jobs.length === 0 ? (
          <p className="small muted">No new jobs this day.</p>
        ) : (
          <ShortList
            items={audit.new_jobs}
            render={(s) => (
              <li key={s.source} className="small audit-row">
                <Link to={`/all-jobs?source=${encodeURIComponent(s.source)}`}>
                  {s.source}
                </Link>
                <span className="muted">{s.jobs}</span>
              </li>
            )}
          />
        )}
      </div>

      <div className="audit-section">
        <h3>
          Scored <span className="chip-count">{audit.scored.length}</span>
          {rescored > 0 && (
            <span className="muted small"> ({rescored} rescored)</span>
          )}
        </h3>
        {audit.scored.length === 0 ? (
          <p className="small muted">Nothing was scored this day.</p>
        ) : (
          <ShortList
            items={audit.scored}
            render={(s) => (
              <li key={`${s.id}-${s.rescored}`} className="small audit-row">
                <span>
                  <JobLink jobId={s.id}>{s.title}</JobLink>{" "}
                  <span className="muted">· {s.company}</span>
                  {s.rescored && <span className="muted"> · rescored</span>}
                </span>
                <span className="audit-badges">
                  <FitBadge label="Interest" value={s.interest_fit} />
                  <FitBadge label="Fit" value={s.candidate_fit} />
                </span>
              </li>
            )}
          />
        )}
      </div>

      <div className="audit-section">
        <h3>
          Waiting to be scored{" "}
          <span className="chip-count">{q.jobs.length}</span>
        </h3>
        {(q.set_aside > 0 || q.hidden > 0) && (
          <p className="small muted">
            Not counted:{" "}
            {[
              q.set_aside > 0 &&
                `${q.set_aside} set aside by the relevance check`,
              q.hidden > 0 && `${q.hidden} hidden by your dealbreakers`,
            ]
              .filter(Boolean)
              .join(" · ")}
            .
          </p>
        )}
        {q.jobs.length === 0 ? (
          <p className="small muted">Nothing is waiting.</p>
        ) : (
          <>
            <p className="small muted">
              In the order the 2pm check scores them, up to your daily limit.
              Score one now to see it sooner (about a cent each).
            </p>
            <div className="audit-scroll">
              <ShortList
                items={q.jobs}
                render={(j) => (
                  <li key={j.id} className="small audit-row">
                    <span>
                      <JobLink jobId={j.id}>{j.title}</JobLink>{" "}
                      <span className="muted">· {j.company}</span>
                    </span>
                    <span className="audit-waiting-side">
                      <span className="muted">{j.source}</span>
                      <ScoreButton
                        jobId={j.id}
                        label="Score this job now"
                        onScored={() => setScoredHere((n) => n + 1)}
                      />
                    </span>
                  </li>
                )}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
