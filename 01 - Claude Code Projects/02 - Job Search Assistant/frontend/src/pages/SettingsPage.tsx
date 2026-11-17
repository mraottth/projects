import { useEffect, useState } from "react";
import {
  api,
  ApiError,
  type EmailStatus,
  Me,
  type RunAudit,
  Settings,
  type TitleScreenInsight,
  type UsagePeriod,
  type UsageSummary,
} from "../api";
import { Link } from "react-router-dom";
import { JobLink } from "../components/JobLink";
import { FitBadge } from "../components/ScoredJobCard";
import { Markdown } from "../components/Markdown";
import { RescoreButton } from "../components/RescoreButton";
import { Spinner } from "../components/Spinner";
import { useIsAdmin } from "../me";
import { DEMO } from "../demo";
import { PrivacyNote } from "./AccessPage";

function toList(s: string): string[] {
  return s
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

// The daily email (core/email_digest.py): status, plus a test send that doesn't affect the real one.
function DailyEmail() {
  const [status, setStatus] = useState<EmailStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    api.getEmailStatus().then(setStatus).catch(() => setStatus({ configured: false, to: null }));
  }, []);

  const test = async () => {
    setBusy(true);
    setResult(null);
    try {
      const r = await api.sendTestEmail();
      setResult({
        ok: true,
        text: r.count
          ? `Sent ${r.count} current match${r.count === 1 ? "" : "es"} to ${r.to}.`
          : `Sent to ${r.to} (no High or Med matches right now, so it's a short check-in).`,
      });
    } catch (e) {
      setResult({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>Daily email</h2>
      <p className="muted small">
        After the daily check at 2:00pm ET, new matches with High or Med interest and a High or Med fit are
        emailed to you, each one once. No new matches, no email.
      </p>
      {status === null ? (
        <Spinner label="Checking..." />
      ) : status.configured ? (
        <p className="small">
          Sending to <strong>{status.to}</strong>
        </p>
      ) : (
        <p className="small">
          <strong>Not set up on the server</strong>, so nothing is sent. Store a Gmail app password once with{" "}
          <code>make email-secret</code>; after that, every <code>make deploy</code> keeps email on.
        </p>
      )}
      {status?.last && <LastEmail last={status.last} />}
      <button className="ghost" onClick={test} disabled={busy || !status?.configured}>
        {busy ? <Spinner label="Sending..." /> : "Send a test email"}
      </button>
      {result && <p className={result.ok ? "muted small" : "error-text small"}>{result.text}</p>}
    </div>
  );
}

/** What the last check's email step did, so a missing email has a reason on screen. */
function LastEmail({ last }: { last: NonNullable<EmailStatus["last"]> }) {
  const when = new Date(last.at).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  let what: string;
  if (last.sent) what = `sent ${last.count} job${last.count === 1 ? "" : "s"} to ${last.to}.`;
  else if (last.reason === "not configured") what = "nothing sent, because email isn't set up on the server.";
  else if (last.reason === "nothing new")
    what =
      last.on_list_today === 0
        ? "nothing sent: New & Noteworthy had nothing new."
        : `nothing sent: none of the ${last.on_list_today} job${last.on_list_today === 1 ? "" : "s"} on ` +
          `New & Noteworthy had High or Med interest with a High or Med fit` +
          (last.best_interest_today ? ` (the best was ${last.best_interest_today} interest).` : ".");
  else what = `nothing sent (${last.reason ?? "unknown reason"}).`;
  return (
    <p className={`small ${last.sent ? "muted" : ""}`}>
      Last check ({when}): {what}
    </p>
  );
}

/**
 * The relevance check (core/screening.py; "title screen" in the code, renamed for people
 * 2026-11-12): what it has checked, set aside and kept, what's still waiting, what it cost, and its
 * latest set-asides, each with "Put back" for when it got one wrong. It can be run by hand: the
 * waiting jobs now, or every waiting job again after changing what you're looking for, with the
 * cost shown when it's over $0.10.
 */
function TitleScreenCard() {
  const [info, setInfo] = useState<TitleScreenInsight | null>(null);
  const [putBack, setPutBack] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState<"waiting" | "recheck" | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [result, setResult] = useState<string | null>(null);

  const load = () =>
    api
      .getTitleScreen()
      .then(setInfo)
      .catch(() => setInfo(null));
  useEffect(() => {
    load();
  }, []);

  if (!info) return null;
  const run = async (which: "waiting" | "recheck") => {
    setRunning(which);
    setProgress(null);
    setResult(null);
    setError(null);
    try {
      const r = await api.runRelevanceCheck(which === "recheck", (done, total) => setProgress({ done, total }));
      setResult(
        r.screened === 0
          ? "Nothing new to check."
          : `Checked ${r.screened.toLocaleString()} job${r.screened === 1 ? "" : "s"}; ${r.set_aside.toLocaleString()} set aside as low-relevance.`,
      );
      setPutBack(new Set());
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(null);
    }
  };
  const label = (verb: string, est: { jobs: number; usd: number }) =>
    `${verb} ${est.jobs.toLocaleString()} waiting job${est.jobs === 1 ? "" : "s"}` +
    (est.usd > info.show_estimate_over ? ` (about $${est.usd.toFixed(2)})` : "");
  const rate = info.screened ? Math.round(((info.set_aside) / info.screened) * 100) : 0;
  return (
    <div className="card">
      <h2>Relevance check</h2>
      <p className="muted small">
        Before jobs are scored, a quick, very cheap read of their titles sets aside the ones that are clearly
        not a fit, so your daily scoring goes to jobs worth a closer look. Anything it isn't sure about is
        kept. Low-relevance jobs aren't scored and are hidden on All Jobs unless you ask to see them. It
        runs with every check; you can also run it now.
      </p>
      {info.screened === 0 ? (
        <p className="small">
          Nothing checked yet. It runs before the next check scores jobs
          {info.waiting > 0 && ` (${info.waiting.toLocaleString()} waiting)`}.
        </p>
      ) : (
        <>
          <div className="screen-stats">
            <div>
              <strong>{info.screened.toLocaleString()}</strong>
              <span className="muted small">checked</span>
            </div>
            <div>
              <strong>{info.set_aside.toLocaleString()}</strong>
              <span className="muted small">set aside ({rate}%)</span>
            </div>
            <div>
              <strong>{info.kept.toLocaleString()}</strong>
              <span className="muted small">kept for scoring</span>
            </div>
            <div>
              <strong>{info.waiting.toLocaleString()}</strong>
              <span className="muted small">not checked yet</span>
            </div>
          </div>
          {info.skipped > 0 && (
            <p className="muted small">
              {info.skipped.toLocaleString()} more {info.skipped === 1 ? "is" : "are"} hidden by your location or pay
              dealbreakers, so {info.skipped === 1 ? "it's" : "they're"} never checked or scored.
            </p>
          )}
          <p className="muted small">
            {info.last_screened_at && `Last ran ${when(info.last_screened_at)}. `}
            This month: {info.calls_this_month} call{info.calls_this_month === 1 ? "" : "s"},{" "}
            {money(info.cost_this_month)}.
          </p>
        </>
      )}
      {!DEMO && (info.estimates.waiting.jobs > 0 || info.screened > 0) && (
        <div className="form-actions relevance-actions">
          {info.estimates.waiting.jobs > 0 && (
            <button className="primary small" onClick={() => run("waiting")} disabled={!!running}>
              {running === "waiting" ? (
                <Spinner label={progress?.total ? `Checked ${progress.done} of ${progress.total}...` : "Starting..."} />
              ) : (
                label("Check", info.estimates.waiting)
              )}
            </button>
          )}
          {info.estimates.recheck.jobs > 0 && info.screened > 0 && (
            <button
              className="ghost small"
              onClick={() => run("recheck")}
              disabled={!!running}
              title="After changing what you're looking for: checks every waiting job again"
            >
              {running === "recheck" ? (
                <Spinner label={progress?.total ? `Checked ${progress.done} of ${progress.total}...` : "Starting..."} />
              ) : (
                label("Re-check all", info.estimates.recheck)
              )}
            </button>
          )}
        </div>
      )}
      {result && <p className="muted small">{result}</p>}
      {info.recent.length > 0 && (
        <>
          <h3 className="screen-recent-head">Recently set aside</h3>
          <ul className="screen-recent">
            {info.recent.map((job) => (
              <li key={job.id}>
                <span className="screen-recent-title">
                  <JobLink jobId={job.id}>{job.title}</JobLink>
                  <span className="muted small">
                    {" "}
                    · {job.company}
                    {job.location && ` · ${job.location}`}
                  </span>
                </span>
                {putBack.has(job.id) ? (
                  <span className="muted small">Put back: it'll be scored</span>
                ) : (
                  <button
                    className="link small"
                    title="The screen got this one wrong: score it like any other job"
                    onClick={async () => {
                      setError(null);
                      try {
                        await api.putBackScreened(job.id);
                        setPutBack((s) => new Set(s).add(job.id));
                      } catch (e) {
                        setError(e instanceof Error ? e.message : String(e));
                      }
                    }}
                  >
                    Put back
                  </button>
                )}
              </li>
            ))}
          </ul>
          {error && <p className="error-text small">{error}</p>}
          {info.set_aside > 0 && (
            <Link className="link small" to="/all-jobs?set_aside=1">
              See all {info.set_aside.toLocaleString()} on All Jobs →
            </Link>
          )}
        </>
      )}
    </div>
  );
}

// Wipes everything and starts over at onboarding. Typing DELETE is the confirmation: it's the one
// irreversible action in the app, and the backend refuses without it too.
/** Type-DELETE confirmation for a destructive action (no window.confirm here; see CLAUDE.md). */
function TypedDelete({
  openLabel,
  confirmLabel,
  id,
  onConfirm,
}: {
  openLabel: string;
  confirmLabel: string;
  id: string;
  onConfirm: (typed: string) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await onConfirm(typed);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button className="ghost danger" onClick={() => setOpen(true)}>
        {openLabel}
      </button>
    );
  }
  return (
    <div className="fsection">
      <label className="flabel" htmlFor={id}>
        Type DELETE to confirm
      </label>
      <div className="form-actions" style={{ marginTop: 0 }}>
        <input id={id} value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" disabled={busy} />
        <button className="danger-solid" onClick={run} disabled={typed !== "DELETE" || busy}>
          {busy ? <Spinner label="Deleting..." /> : confirmLabel}
        </button>
        <button
          className="link small"
          onClick={() => {
            setOpen(false);
            setTyped("");
          }}
          disabled={busy}
        >
          Cancel
        </button>
      </div>
      {error && <p className="error-text small">{error}</p>}
    </div>
  );
}

function DeleteAllData() {
  const isAdmin = useIsAdmin();
  return (
    <div className="card danger-card">
      <h2>Start over</h2>
      <p className="muted small">
        Permanently deletes your resumes and cover letters, profile, followed sources, scores, My
        Jobs, Jobby conversations and settings, then takes you back to onboarding. Your Claude usage
        log is kept, and a backup of the database is saved first, so it can be restored if you
        change your mind. Nobody else's data is touched.
      </p>
      <TypedDelete
        id="confirm-delete"
        openLabel="Delete all my data…"
        confirmLabel="Delete everything"
        onConfirm={async (typed) => {
          await api.deleteAllData(typed);
          // A full reload, so the app re-checks onboarding status and drops every page's loaded data.
          window.location.assign("/onboarding");
        }}
      />
      {!isAdmin && (
        <>
          <h2 style={{ marginTop: "22px" }}>Delete my account</h2>
          <p className="muted small">
            Permanently deletes your account and everything in it, the same as Start over plus the account
            itself. You'd need to request access again to come back.
          </p>
          <TypedDelete
            id="confirm-delete-account"
            openLabel="Delete my account…"
            confirmLabel="Delete my account"
            onConfirm={async (typed) => {
              await api.deleteAccount(typed);
              // A full reload: with no account, the app shows the request-access page.
              window.location.assign("/");
            }}
          />
        </>
      )}
    </div>
  );
}

function PrefilterSettings() {
  const isAdmin = useIsAdmin();
  const [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.getSettings().then(setSettings);
  }, []);

  if (!settings) return <p className="muted">Loading...</p>;

  const save = async (patch: Partial<Settings>) => {
    setBusy(true);
    setSaved(false);
    try {
      const updated = await api.putSettings(patch);
      setSettings(updated);
      setSaved(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>Scoring tuning</h2>
      <p className="muted small">
        Cheap rule-based scoring runs before any Claude call, so the daily cap is spent on the most
        promising jobs first. Your desired titles/locations/salary live on the Profile page — these
        are extra tuning knobs on top of that.
      </p>

      <div className="fsection">
        <span className="flabel">Daily LLM cap (max jobs fully scored per day)</span>
        {isAdmin ? (
          <input
            type="number"
            defaultValue={settings.daily_llm_cap}
            onBlur={(e) => save({ daily_llm_cap: Number(e.target.value) })}
            style={{ width: "140px" }}
          />
        ) : (
          <p className="small" style={{ margin: 0 }}>
            {settings.daily_llm_cap} jobs a day, set by the admin.
          </p>
        )}
      </div>

      <div className="fsection">
        <span className="flabel">Required keywords (any match boosts score)</span>
        <input
          defaultValue={settings.prefilter_keywords_include.join(", ")}
          onBlur={(e) => save({ prefilter_keywords_include: toList(e.target.value) })}
          style={{ width: "100%" }}
        />
      </div>

      <div className="fsection" style={{ marginBottom: 0 }}>
        <span className="flabel">Excluded keywords (any match drops score to zero)</span>
        <input
          defaultValue={settings.prefilter_keywords_exclude.join(", ")}
          onBlur={(e) => save({ prefilter_keywords_exclude: toList(e.target.value) })}
          style={{ width: "100%" }}
        />
      </div>

      {busy && <p className="muted small">Saving...</p>}
      {saved && !busy && <p className="muted small">Saved.</p>}
    </div>
  );
}

/** Where the data lives and how it's backed up (core/backup.py), with a download of the whole
 * database to keep a copy off Google Cloud. */
function YourData() {
  const isAdmin = useIsAdmin();
  const [backups, setBackups] = useState<{ name: string; bytes: number }[] | null>(null);

  useEffect(() => {
    // The whole database holds everyone's data, so only the admin lists or downloads it.
    if (!isAdmin) return;
    api.listBackups().then((r) => setBackups(r.backups)).catch(() => setBackups([]));
  }, [isAdmin]);

  const latest = backups?.[0];
  return (
    <div className="card">
      <h2>Your data</h2>
      <p className="muted small">
        Everything (My Jobs, your documents, sources and settings) is stored in the app's private
        Google Cloud storage, visible in the app only to you. The app saves a copy of the database
        before every daily check, keeping 30 days, and before "Start over" or an upgrade that changes
        your data.
      </p>
      {isAdmin && backups && (
        <p className="small">
          {latest
            ? `${backups.length} backup${backups.length === 1 ? "" : "s"} kept; latest: ${latest.name}.`
            : "No backups yet: the first one is made at the next daily check."}
        </p>
      )}
      <div className="form-actions">
        {isAdmin && (
          <a className="ghost button-link" href={api.backupDownloadUrl} download>
            Download a backup
          </a>
        )}
        <a className="link small" href={api.applicationsCsvUrl} download>
          Export My Jobs as CSV
        </a>
      </div>
      <PrivacyNote />
    </div>
  );
}

const PERIODS: { key: UsagePeriod; label: string; phrase: string }[] = [
  { key: "day", label: "Day", phrase: "the last day" },
  { key: "week", label: "Week", phrase: "the last 7 days" },
  { key: "month", label: "Month", phrase: "the last 30 days" },
];

// Small amounts show to the tenth of a cent; anything over a dollar to the cent.
function money(usd: number): string {
  if (usd === 0) return "$0";
  if (usd < 0.01) return "<$0.01";
  return usd < 1 ? `$${usd.toFixed(3).replace(/0$/, "")}` : `$${usd.toFixed(2)}`;
}

/** "2026-11-12 14:16:28" (UTC) as local "Nov 12, 2:16 PM". */
function when(dbTime: string): string {
  const d = new Date(`${dbTime.replace(" ", "T")}Z`);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

/**
 * The Usage log (asked 2026-11-12: the raw call list was hard to read): what Claude cost by category
 * for the last day, week or month, then the calls gathered into activities ("Scored 23 jobs"), each
 * opening into its calls. core/usage.py does the grouping.
 */
function UsageLog() {
  const [period, setPeriod] = useState<UsagePeriod>("week");
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [cache, setCache] = useState<{ hit_rate: number; saved_usd: number } | null>(null);
  const [usage, setUsage] = useState<Me["usage"]>(null);
  const [open, setOpen] = useState<number | null>(null);
  // Clicking a category in the chart shows only its activities; clicking it again shows all
  // (asked 2026-11-17).
  const [only, setOnly] = useState<string | null>(null);
  const pick = (key: string) => {
    setOnly((k) => (k === key ? null : key));
    setOpen(null);
  };

  useEffect(() => {
    api.getUsageLog(period).then((r) => {
      setSummary(r.summary);
      setCache(r.cache);
      setOpen(null);
      setOnly((k) => (k && r.summary.categories.some((c) => c.key === k) ? k : null));
    });
  }, [period]);
  useEffect(() => {
    // Fresh numbers, rather than the ones App loaded when the page first opened.
    api.getMe().then((me) => setUsage(me.usage ?? null)).catch(() => {});
  }, []);

  const phrase = PERIODS.find((p) => p.key === period)!.phrase;
  const top = summary?.categories[0]?.cost ?? 0;
  const activities = summary ? summary.activities.filter((a) => !only || a.category === only) : [];
  const onlyLabel = summary?.categories.find((c) => c.key === only)?.label;
  return (
    <div className="card">
      <h2>Usage log</h2>
      {usage && (
        <p className={`small usage-line${usage.over ? " over" : ""}`}>
          This month: ${usage.spend_this_month.toFixed(2)}
          {usage.exempt ? " (no limit)" : ` of your $${usage.monthly_budget_usd.toFixed(2)} budget`} · Today:{" "}
          {usage.scored_today} of {usage.daily_score_cap} jobs scored
          {usage.over && " · Budget reached: scoring, Jobby, tailoring and discovery pause until next month."}
        </p>
      )}

      <div className="usage-head">
        <div className="seg" role="group" aria-label="Period">
          {PERIODS.map((p) => (
            <button key={p.key} className={period === p.key ? "on" : ""} onClick={() => setPeriod(p.key)}>
              {p.label}
            </button>
          ))}
        </div>
        {summary && (
          <span className="usage-total">
            <strong>{money(summary.total_cost)}</strong> <span className="muted small">in {phrase}</span>
          </span>
        )}
      </div>

      {summary && summary.categories.length > 0 && (
        <ul className={`usage-cats${only ? " filtered" : ""}`}>
          {summary.categories.map((c) => (
            <li
              key={c.key}
              className={only === c.key ? "on" : ""}
              role="button"
              tabIndex={0}
              aria-pressed={only === c.key}
              title={only === c.key ? "Show every category" : `Show only ${c.label}`}
              onClick={() => pick(c.key)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  pick(c.key);
                }
              }}
            >
              <span className="usage-cat-name">{c.label}</span>
              <span className="usage-cat-bar" aria-hidden="true">
                <span style={{ width: `${top ? Math.max(2, (c.cost / top) * 100) : 0}%` }} />
              </span>
              <span className="muted small usage-cat-amount">{c.amount}</span>
              <span className="usage-cat-cost">{money(c.cost)}</span>
            </li>
          ))}
        </ul>
      )}
      {cache && cache.hit_rate > 0 && (
        <p className="muted small">
          Prompt cache, last 7 days: {Math.round(cache.hit_rate * 100)}% of input read from cache, saving about $
          {cache.saved_usd.toFixed(2)}.
        </p>
      )}

      {onlyLabel && (
        <p className="small usage-only">
          Showing only <strong>{onlyLabel}</strong> ({activities.length}{" "}
          {activities.length === 1 ? "activity" : "activities"}) ·{" "}
          <button className="link small" onClick={() => setOnly(null)}>
            Show all
          </button>
        </p>
      )}
      {/* A fixed-height box that scrolls (asked 2026-11-06), with the header row kept in view. */}
      {summary && activities.length > 0 && (
        <div className="usage-log-scroll">
          <table className="usage-activities">
            <thead>
              <tr>
                <th>When</th>
                <th>What</th>
                <th>Kind</th>
                <th className="num">Cost</th>
              </tr>
            </thead>
            <tbody>
              {activities.map((a, i) => (
                <ActivityRows
                  key={`${a.started_at}-${i}`}
                  activity={a}
                  open={open === i}
                  onToggle={() => setOpen(open === i ? null : i)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
      {summary && summary.activities.length === 0 && <p className="empty-state">No Claude calls in {phrase}.</p>}
    </div>
  );
}

function ActivityRows({
  activity: a,
  open,
  onToggle,
}: {
  activity: UsageSummary["activities"][number];
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr className={`usage-activity${open ? " open" : ""}`} onClick={onToggle}>
        <td className="small">{when(a.ended_at)}</td>
        <td>
          <button className="link usage-toggle" aria-expanded={open}>
            {open ? "\u25BE" : "\u25B8"} {a.label}
          </button>
        </td>
        <td className="small muted">{a.category_label}</td>
        <td className="small num">{money(a.cost)}</td>
      </tr>
      {open &&
        a.calls.map((c) => (
          <tr key={c.id} className="usage-call">
            <td className="small muted">{when(c.created_at)}</td>
            <td className="small" colSpan={2}>
              {c.detail || "\u2014"}
              <span className="muted">
                {" "}
                · {c.input_tokens.toLocaleString()} in / {c.output_tokens.toLocaleString()} out
                {c.cache_read_tokens > 0 && `, ${c.cache_read_tokens.toLocaleString()} from cache`}
              </span>
            </td>
            <td className="small num">{money(c.cost)}</td>
          </tr>
        ))}
    </>
  );
}

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
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

// Run times are stored as UTC, either ISO with an offset or SQLite's "YYYY-MM-DD HH:MM:SS".
function clock(value: string): string {
  const d = new Date(value.includes("T") ? value : `${value.replace(" ", "T")}Z`);
  return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

/** A list that shows its first few rows, then "Show all N". */
function ShortList<T>({ items, render, first = 10 }: { items: T[]; render: (item: T) => React.ReactNode; first?: number }) {
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
function RunAuditCard() {
  const [day, setDay] = useState<string | undefined>(undefined);
  const [audit, setAudit] = useState<RunAudit | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getRunAudit(day)
      .then((a) => {
        setAudit(a);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [day]);

  if (error) return <div className="card"><h2>Daily run audit</h2><p className="error-text small">{error}</p></div>;
  if (!audit) return null;
  const days = audit.days.includes(audit.date) ? audit.days : [audit.date, ...audit.days];
  const newTotal = audit.new_jobs.reduce((n, s) => n + s.jobs, 0);
  const rescored = audit.scored.filter((s) => s.rescored).length;
  const { organizations: orgs, boards } = audit.sources;
  const q = audit.queue;

  return (
    <div className="card run-audit">
      <div className="usage-head">
        <h2>Daily run audit</h2>
        <span className="select-wrap">
          <select value={audit.date} onChange={(e) => setDay(e.target.value)} aria-label="Day">
            {days.map((d) => (
              <option key={d} value={d}>
                {dayName(d)}
              </option>
            ))}
          </select>
        </span>
      </div>
      <p className="muted small">
        What your checks did: the 2pm run, Refresh and "Check my sources now".
      </p>

      <div className="audit-section">
        <h3>Checks</h3>
        <p className="small">
          {audit.checks.length === 0
            ? "No check ran this day."
            : `${audit.checks.length} ${audit.checks.length === 1 ? "check" : "checks"}: ${audit.checks
                .map((c) => clock(c.started_at))
                .join(", ")}`}
        </p>
      </div>

      <div className="audit-section">
        <h3>Sources checked</h3>
        {audit.checks.length === 0 ? (
          <p className="small muted">None.</p>
        ) : !audit.sources.recorded ? (
          <p className="small muted">Not recorded for checks before Nov 17, 2026.</p>
        ) : (
          <>
            <p className="small">
              Organizations: <strong>{orgs.checked}</strong> of {orgs.total} · Job boards:{" "}
              <strong>{boards.checked}</strong> of {boards.total}
            </p>
            {audit.sources.failed.length > 0 && (
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
                <Link to={`/all-jobs?source=${encodeURIComponent(s.source)}`}>{s.source}</Link>
                <span className="muted">{s.jobs}</span>
              </li>
            )}
          />
        )}
      </div>

      <div className="audit-section">
        <h3>
          Scored <span className="chip-count">{audit.scored.length}</span>
          {rescored > 0 && <span className="muted small"> ({rescored} rescored)</span>}
        </h3>
        {audit.scored.length === 0 ? (
          <p className="small muted">Nothing was scored this day.</p>
        ) : (
          <ShortList
            items={audit.scored}
            render={(s) => (
              <li key={`${s.id}-${s.rescored}`} className="small audit-row">
                <span>
                  <JobLink jobId={s.id}>{s.title}</JobLink> <span className="muted">· {s.company}</span>
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
          Waiting to be scored now <span className="chip-count">{q.jobs.length}</span>
        </h3>
        {(q.set_aside > 0 || q.hidden > 0) && (
          <p className="small muted">
            Not counted:{" "}
            {[
              q.set_aside > 0 && `${q.set_aside} set aside by the relevance check`,
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
            <p className="small muted">In the order they'll be scored, up to your daily limit each day.</p>
            <div className="audit-scroll">
              <ShortList
                items={q.jobs}
                render={(j) => (
                  <li key={j.id} className="small audit-row">
                    <span>
                      <JobLink jobId={j.id}>{j.title}</JobLink> <span className="muted">· {j.company}</span>
                    </span>
                    <span className="muted">{j.source}</span>
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

function RescoreJobs() {
  const isAdmin = useIsAdmin();
  return (
    <div className="card">
      <h2>Rescore your jobs</h2>
      <p className="muted small">
        Each job is scored once, when it first shows up. After changing your profile or what you're looking
        for, rescore a list to see every job on it in that new light.
        {!isAdmin && " Each list can be rescored once a day."}
      </p>
      <RescoreButton scope="my_jobs" />
      <RescoreButton scope="all_jobs" />
    </div>
  );
}

export default function SettingsPage() {
  const [profile, setProfile] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = () =>
    api
      .getPreferenceProfile()
      .then((r) => setProfile(r.text))
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setProfile(null);
      });

  useEffect(() => {
    refresh();
  }, []);

  const rebuild = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.rebuildPreferenceProfile();
      setProfile(r.text);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1>Settings</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        Scoring tuning, your learned preferences, and what Claude usage is costing you.
      </p>

      <div className="card">
        <h2>Preference profile</h2>
        <p className="muted small">
          Learned from your interested/not-interested marks and application outcomes (application
          outcomes count for more). Rebuild after marking a batch of jobs or moving applications
          forward.
        </p>
        <button className="ghost" onClick={rebuild} disabled={busy}>
          {busy ? <Spinner label="Rebuilding..." /> : "Rebuild from feedback"}
        </button>
        {error && <p className="error-text small">{error}</p>}
        {profile ? (
          <div className="card-section">
            <Markdown text={profile} />
          </div>
        ) : (
          <p className="empty-state">No preference profile yet — mark some jobs or track an application.</p>
        )}
      </div>

      <RescoreJobs />

      <DailyEmail />
      <TitleScreenCard />
      <PrefilterSettings />
      <UsageLog />
      {!DEMO && <RunAuditCard />}
      <YourData />

      <DeleteAllData />
    </div>
  );
}
