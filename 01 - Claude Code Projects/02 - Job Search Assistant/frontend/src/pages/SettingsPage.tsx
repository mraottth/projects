import { useEffect, useState } from "react";
import { api, ApiError, type EmailStatus, LlmCall, Me, Settings } from "../api";
import { Markdown } from "../components/Markdown";
import { RescoreButton } from "../components/RescoreButton";
import { Spinner } from "../components/Spinner";
import { useIsAdmin } from "../me";
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

// One-off: older job summaries' highlights are untagged, so they all show amber. This has Claude
// label each as interest or fit (core/retag.py), after showing how many and roughly what it costs.
function ColorOlderHighlights() {
  const [status, setStatus] = useState<{ pending: number; estimated_cost_usd: number } | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(null);

  const load = () => api.getRetagStatus().then(setStatus).catch(() => undefined);
  useEffect(() => {
    load();
  }, []);

  const run = async () => {
    setBusy(true);
    setResult(null);
    setProgress(null);
    try {
      const r = await api.retagHighlights((done, total) => setProgress({ done, total }));
      setResult({
        ok: true,
        text:
          `Colored the highlights in ${r.retagged} summar${r.retagged === 1 ? "y" : "ies"}.` +
          (r.skipped ? ` ${r.skipped} couldn't be matched up and were left as they were.` : ""),
      });
      load();
    } catch (e) {
      setResult({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  if (!status || (status.pending === 0 && !result)) return null;
  return (
    <div className="card">
      <h2>Color older highlights</h2>
      <p className="muted small">
        Job summaries scored before highlights were colored still show every highlight in amber.
        This asks Claude which rating each highlight explains, interest or fit, and colors it to
        match, without changing any wording or ratings.
      </p>
      {status.pending > 0 && (
        <button className="ghost" onClick={run} disabled={busy}>
          {busy ? (
            <Spinner label={progress?.total ? `Colored ${progress.done} of ${progress.total}...` : "Starting..."} />
          ) : (
            `Color ${status.pending} summar${status.pending === 1 ? "y" : "ies"} (about $${Math.max(
              status.estimated_cost_usd,
              0.01,
            ).toFixed(2)})`
          )}
        </button>
      )}
      {result && <p className={result.ok ? "muted small" : "error-text small"}>{result.text}</p>}
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

function UsageLog() {
  const [calls, setCalls] = useState<LlmCall[]>([]);
  const [totals, setTotals] = useState<{ total_cost: number; today_cost: number; total_calls: number } | null>(
    null,
  );
  const [cache, setCache] = useState<{ hit_rate: number; saved_usd: number } | null>(null);

  const [usage, setUsage] = useState<Me["usage"]>(null);

  useEffect(() => {
    api.getUsageLog().then((r) => {
      setCalls(r.calls);
      setTotals(r.totals);
      setCache(r.cache);
    });
    // Fresh numbers, rather than the ones App loaded when the page first opened.
    api.getMe().then((me) => setUsage(me.usage ?? null)).catch(() => {});
  }, []);

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
      {totals && (
        <p className="muted small">
          ${totals.today_cost.toFixed(2)} today · ${totals.total_cost.toFixed(2)} total ·{" "}
          {totals.total_calls} calls
        </p>
      )}
      {cache && cache.hit_rate > 0 && (
        <p className="muted small">
          Prompt cache, last 7 days: {Math.round(cache.hit_rate * 100)}% of input read from cache, saving about $
          {cache.saved_usd.toFixed(2)}.
        </p>
      )}
      <table>
        <thead>
          <tr>
            <th>When</th>
            <th>Type</th>
            <th>Model</th>
            <th>Tokens</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {calls.map((c) => (
            <tr key={c.id}>
              <td className="small">{c.created_at}</td>
              <td className="small">{c.call_type}</td>
              <td className="small">{c.model}</td>
              <td className="small">
                {c.input_tokens}/{c.output_tokens}
                {c.cache_read_tokens > 0 && <span className="muted"> · {c.cache_read_tokens} cached</span>}
              </td>
              <td className="small">{c.estimated_cost_usd != null ? `$${c.estimated_cost_usd.toFixed(4)}` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {calls.length === 0 && <p className="empty-state">No Claude calls logged yet.</p>}
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
      <ColorOlderHighlights />
      <PrefilterSettings />
      <UsageLog />
      <YourData />

      <DeleteAllData />
    </div>
  );
}
