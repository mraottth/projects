import { Fragment, useEffect, useState } from "react";
import {
  api,
  type AdminUser,
  type DailyRunStatus,
  type ExperimentScore,
  type ScoringExperimentResult,
} from "../api";
import { Markdown } from "../components/Markdown";
import { FitBadge, rationaleClass } from "../components/ScoredJobCard";
import { Spinner } from "../components/Spinner";

/**
 * Admin only (D-021): approve or decline access requests, and see each person's activity, today's
 * scoring against their daily cap, and this month's estimated Claude spend against their budget.
 */
export default function AdminPage() {
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [shared, setShared] = useState(0);
  const [dailyRun, setDailyRun] = useState<DailyRunStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  async function load() {
    try {
      const data = await api.listUsers();
      setUsers(data.users);
      setShared(data.shared_spend_this_month);
      setDailyRun(data.daily_run);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function update(user: AdminUser, patch: Parameters<typeof api.updateUser>[1]) {
    setBusyId(user.id);
    setError(null);
    try {
      await api.updateUser(user.id, patch);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyId(null);
    }
  }

  if (error && !users) return <p className="error">{error}</p>;
  if (!users) return <Spinner label="Loading…" />;

  const pending = users.filter((u) => u.status === "pending");
  const others = users.filter((u) => u.status !== "pending");
  const totalSpend = users.reduce((sum, u) => sum + u.spend_this_month, 0) + shared;

  return (
    <div className="admin-page">
      <h1>Admin</h1>
      {error && <p className="error">{error}</p>}

      {dailyRun && <DailyCheck status={dailyRun} />}

      <section className="card">
        <h2>Access requests</h2>
        {pending.length === 0 ? (
          <p className="muted">No one is waiting.</p>
        ) : (
          <ul className="admin-requests">
            {pending.map((u) => (
              <li key={u.id}>
                <div>
                  <strong>{u.name ?? u.email}</strong> <span className="muted">{u.email}</span>
                  <div className="muted small">Asked {when(u.created_at)}</div>
                  {u.note && <p className="admin-note">“{u.note}”</p>}
                </div>
                <div className="admin-actions">
                  <button className="primary" disabled={busyId === u.id} onClick={() => update(u, { status: "approved" })}>
                    Approve
                  </button>
                  <button className="ghost" disabled={busyId === u.id} onClick={() => update(u, { status: "declined" })}>
                    Decline
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card">
        <h2>People</h2>
        <p className="muted small">
          Claude spend this month: ${totalSpend.toFixed(2)} in all, including ${shared.toFixed(2)} for reading
          shared job boards. Spend is estimated from the usage log.
        </p>
        <table className="admin-table">
          <thead>
            <tr>
              <th>Person</th>
              <th>Status</th>
              <th>Last visit</th>
              <th>Scored today</th>
              <th>Spend this month</th>
              <th>Sources</th>
              <th>Jobs tracked</th>
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {others.map((u) => (
              <tr key={u.id}>
                <td>
                  <strong>{u.name ?? u.email}</strong>
                  {!!u.is_admin && <span className="pill admin-pill">admin</span>}
                  {u.name && <div className="muted small">{u.email}</div>}
                </td>
                <td>
                  {u.is_admin ? (
                    "approved"
                  ) : (
                    <select
                      value={u.status}
                      disabled={busyId === u.id}
                      onChange={(e) => update(u, { status: e.target.value as AdminUser["status"] })}
                    >
                      <option value="approved">approved</option>
                      <option value="blocked">blocked</option>
                      <option value="declined">declined</option>
                    </select>
                  )}
                </td>
                <td>{u.last_seen_at ? when(u.last_seen_at) : "never"}</td>
                <td>
                  {u.scored_today} of{" "}
                  <NumberCell
                    value={u.daily_score_cap}
                    min={0}
                    step={1}
                    onSave={(v) => update(u, { daily_score_cap: v })}
                  />
                </td>
                <td>
                  ${u.spend_this_month.toFixed(2)} of $
                  <NumberCell
                    value={u.monthly_budget_usd}
                    min={0}
                    step={1}
                    onSave={(v) => update(u, { monthly_budget_usd: v })}
                  />
                </td>
                <td>{u.sources_followed}</td>
                <td>{u.applications}</td>
                <td>{!u.is_admin && <DeleteUser user={u} onDeleted={load} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          Scoring only runs for people who opened the app in the last 7 days. Anyone who reaches their
          monthly budget has paid features paused until next month; you get an email at 80% and at 100%.
          Your own account has no limit.
        </p>
      </section>

      <ScoringExperiment />
    </div>
  );
}

function NumberCell({
  value,
  min,
  step,
  onSave,
}: {
  value: number;
  min: number;
  step: number;
  onSave: (v: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  return (
    <input
      className="admin-number"
      type="number"
      min={min}
      step={step}
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => {
        const v = Number(draft);
        if (draft !== "" && Number.isFinite(v) && v !== value) onSave(v);
        else setDraft(String(value));
      }}
    />
  );
}

function when(ts: string): string {
  const d = new Date(ts.includes("T") ? ts : ts.replace(" ", "T") + "Z");
  const days = Math.floor((Date.now() - d.getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return d.toLocaleDateString();
}

/** "today at 2:07 PM", "yesterday at …", or "Oct 30 at …": the daily check needs the time too. */
function at(ts: string): string {
  const d = new Date(ts);
  const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const day = new Date(d);
  day.setHours(0, 0, 0, 0);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diff = Math.round((today.getTime() - day.getTime()) / 86_400_000);
  if (diff === 0) return `today at ${time}`;
  if (diff === 1) return `yesterday at ${time}`;
  return `${d.toLocaleDateString([], { month: "short", day: "numeric" })} at ${time}`;
}

/** "Delete…" then type DELETE: removes the account and everything in it, and emails them. */
function DeleteUser({ user, onDeleted }: { user: AdminUser; onDeleted: () => void }) {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!open) {
    return (
      <button className="link small danger-link" onClick={() => setOpen(true)}>
        Delete…
      </button>
    );
  }
  return (
    <div className="admin-delete">
      <input
        aria-label={`Type DELETE to delete ${user.email}`}
        placeholder="Type DELETE"
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        disabled={busy}
      />
      <button
        className="danger-solid"
        disabled={typed !== "DELETE" || busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            await api.deleteUser(user.id, typed);
            onDeleted();
          } catch (e) {
            setError((e as Error).message);
            setBusy(false);
          }
        }}
      >
        {busy ? <Spinner label="Deleting…" /> : "Delete account"}
      </button>
      <button className="link small" onClick={() => setOpen(false)} disabled={busy}>
        Cancel
      </button>
      {error && <p className="error small">{error}</p>}
    </div>
  );
}

/** How the last 2pm ET check (or "Check every source now") went. A crash, or a server shut down
 * mid-run, shows here in red instead of only in Google Cloud's logs. */
function DailyCheck({ status }: { status: DailyRunStatus }) {
  const run = status.latest;
  if (!run) {
    return (
      <section className="card">
        <h2>Daily check</h2>
        <p className="muted small">No daily check has run yet. It runs every day at 2:00pm ET.</p>
      </section>
    );
  }
  const minutes = (from: string, to: string) =>
    Math.max(1, Math.round((new Date(to).getTime() - new Date(from).getTime()) / 60000));
  const who = run.started_by === "scheduled" ? "the 2pm check" : "your check";
  const bad = run.status === "failed" || run.status === "unfinished";
  let headline: string;
  if (run.status === "running") headline = `Running now: ${who} started ${at(run.started_at)}.`;
  else if (run.status === "ok")
    headline = `The last check finished ${at(run.finished_at!)}, in ${minutes(run.started_at, run.finished_at!)} minutes.`;
  else if (run.status === "failed")
    headline = `The last check failed ${at(run.finished_at ?? run.started_at)}.`;
  else headline = `The last check started ${at(run.started_at)} and never finished.`;

  return (
    <section className={`card daily-check${bad ? " bad" : ""}`}>
      <h2>Daily check</h2>
      <p style={{ margin: 0 }}>
        <span className={`pill daily-check-pill ${run.status}`}>
          {{ ok: "OK", running: "Running", failed: "Failed", unfinished: "Didn't finish" }[run.status]}
        </span>{" "}
        {headline}
      </p>
      {run.status === "ok" && (
        <p className="muted small">
          Read {run.sources_polled ?? 0} sources, found {run.jobs_fetched ?? 0} postings, and scored{" "}
          {run.llm_scored_count ?? 0} jobs for {run.people ?? 0} {run.people === 1 ? "person" : "people"}.
          {run.source_errors.length > 0 &&
            ` ${run.source_errors.length} problem${run.source_errors.length === 1 ? "" : "s"} along the way.`}
        </p>
      )}
      {run.error && <p className="error-text small">{run.error}</p>}
      {run.status === "unfinished" && (
        <p className="muted small">
          The server was probably shut down or ran out of memory partway through. Google Cloud's logs for
          that time will say more.
        </p>
      )}
      {bad && (
        <p className="muted small">
          {status.last_ok_finished_at
            ? `The last check that finished cleanly was ${at(status.last_ok_finished_at)}.`
            : "No check has finished cleanly yet."}
        </p>
      )}
      {run.source_errors.length > 0 && (
        <details className="small">
          <summary>Problems along the way (sources that couldn't be read, steps that failed for someone)</summary>
          <ul>
            {run.source_errors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}


const pct = (v: number | null) => (v === null ? "–" : `${Math.round(v * 100)}%`);
const cents = (v: number | null) => (v === null ? "–" : `${(v * 100).toFixed(1)}¢`);

/**
 * The scoring cost experiment (asked 2026-11-03): rescores jobs you've already judged under the
 * current settings and the cheaper "lean" ones, and compares cost and ratings. Nothing it scores
 * reaches your job lists.
 */
function ScoringExperiment() {
  const [info, setInfo] = useState<{ sample_size: number; estimate_usd: number } | null>(null);
  const [result, setResult] = useState<ScoringExperimentResult | null>(null);
  const [progress, setProgress] = useState<[number, number] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getScoringExperiment()
      .then((data) => {
        setInfo(data);
        setResult(data.result);
        if (data.running) run(); // follow a run already going
      })
      .catch((e) => setError((e as Error).message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function run() {
    setError(null);
    setProgress([0, 0]);
    try {
      setResult(await api.runScoringExperiment((done, total) => setProgress([done, total])));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setProgress(null);
    }
  }

  if (!info) return null;
  return (
    <section className="card">
      <h2>Scoring cost experiment</h2>
      <p className="muted small">
        Rescores up to 20 jobs you've already judged (marked Interested or Not interested, or tracked) twice: with
        today's settings, and with cheaper "lean" ones (less reasoning, a 2–3 sentence rationale, long postings
        cut at 6,000 characters). Then it compares what each cost and how each rated the jobs. Nothing here
        changes your job lists or live scoring.
      </p>
      {progress ? (
        <Spinner label={progress[1] ? `Scored ${progress[0]} of ${progress[1]}…` : "Starting…"} />
      ) : info.sample_size === 0 ? (
        !result && <p className="muted small">No judged jobs yet: mark a few jobs Interested or Not interested first.</p>
      ) : (
        <button className={result ? "ghost" : "primary"} onClick={run}>
          {result ? "Run again" : `Run on ${info.sample_size} jobs`} (about ${info.estimate_usd.toFixed(2)})
        </button>
      )}
      {error && <p className="error-text small">{error}</p>}
      {result && !progress && <ExperimentResult result={result} />}
    </section>
  );
}

function ExperimentResult({ result }: { result: ScoringExperimentResult }) {
  const { arms, agreement, judged } = result.summary;
  const [open, setOpen] = useState<number | null>(null);
  const saving =
    arms.current.cost_per_job_usd && arms.lean.cost_per_job_usd
      ? 1 - arms.lean.cost_per_job_usd / arms.current.cost_per_job_usd
      : null;
  return (
    <>
      <p className="small">
        Ran {at(result.ran_at)} on {result.jobs.length} jobs ({judged.interested} you wanted, {judged.not_interested}{" "}
        you didn't{judged.interviewed ? `, ${judged.interviewed} that reached an interview` : ""}).
        {saving !== null && (
          <>
            {" "}
            <strong>Lean costs {Math.round(saving * 100)}% less per job.</strong>
          </>
        )}
      </p>
      <table className="jobs-table experiment-table">
        <thead>
          <tr>
            <th></th>
            <th>Cost per job</th>
            <th>Output tokens</th>
            <th>Wanted → High/Med interest</th>
            <th>Not wanted → Low interest</th>
            {judged.interviewed > 0 && <th>Interviewed → High/Med fit</th>}
          </tr>
        </thead>
        <tbody>
          {(["current", "lean"] as const).map((name) => {
            const a = arms[name];
            return (
              <tr key={name}>
                <td>
                  <strong>{name === "current" ? "Current" : "Lean"}</strong>
                  {a.failed > 0 && <span className="muted small"> ({a.failed} failed)</span>}
                </td>
                <td>{cents(a.cost_per_job_usd)}</td>
                <td>{a.output_tokens_per_job ?? "–"}</td>
                <td>{pct(a.interested_rated_high_or_med)}</td>
                <td>{pct(a.not_interested_rated_low)}</td>
                {judged.interviewed > 0 && <td>{pct(a.interviewed_fit_high_or_med)}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        Lean gave the same interest rating as current on {pct(agreement.interest_same)} of jobs and the same fit
        rating on {pct(agreement.fit_same)}; scores differed by {agreement.interest_score_diff ?? "–"} (interest)
        and {agreement.fit_score_diff ?? "–"} (fit) points on average, out of 100.
      </p>
      <details className="small">
        <summary>Job by job (click a job to compare the two write-ups)</summary>
        <table className="jobs-table experiment-table">
          <thead>
            <tr>
              <th>Job</th>
              <th>You</th>
              <th>Current</th>
              <th>Lean</th>
            </tr>
          </thead>
          <tbody>
            {result.jobs.map((job, i) => {
              const differs =
                job.current &&
                job.lean &&
                (job.current.interest_fit !== job.lean.interest_fit ||
                  job.current.candidate_fit !== job.lean.candidate_fit);
              return (
                <Fragment key={job.job_id}>
                  <tr
                    className={differs ? "experiment-differs" : undefined}
                    onClick={() => setOpen(open === i ? null : i)}
                    style={{ cursor: "pointer" }}
                  >
                    <td>
                      {job.title}
                      <div className="muted small">{job.company}</div>
                    </td>
                    <td>
                      {job.judgment === "interested" ? "Wanted" : "Not wanted"}
                      {job.interviewed && <div className="muted small">interviewed</div>}
                    </td>
                    <td>
                      <Ratings score={job.current} />
                    </td>
                    <td>
                      <Ratings score={job.lean} />
                    </td>
                  </tr>
                  {open === i && (
                    <tr>
                      <td colSpan={4}>
                        <div className="experiment-rationales">
                          {(["current", "lean"] as const).map((name) => (
                            <div key={name}>
                              <strong>{name === "current" ? "Current" : "Lean"}</strong>
                              {job[name] ? (
                                <div
                                  className={`small ${rationaleClass(job[name]!.interest_fit, job[name]!.candidate_fit)}`}
                                >
                                  <Markdown text={job[name]!.rationale_text} />
                                </div>
                              ) : (
                                <p className="muted small">Failed.</p>
                              )}
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </details>
    </>
  );
}

function Ratings({ score }: { score: ExperimentScore | null }) {
  if (!score) return <span className="muted small">failed</span>;
  return (
    <div className="experiment-ratings">
      <FitBadge label="Interest" value={score.interest_fit} />
      <FitBadge label="Fit" value={score.candidate_fit} />
      <span className="muted small">{cents(score.cost_usd)}</span>
    </div>
  );
}
