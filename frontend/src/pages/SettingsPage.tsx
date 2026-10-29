import { useEffect, useState } from "react";
import { api, ApiError, LlmCall, Settings } from "../api";
import { Markdown } from "../components/Markdown";
import { Spinner } from "../components/Spinner";

function toList(s: string): string[] {
  return s
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

// The daily email (core/email_digest.py): status, plus a test send that doesn't affect the real one.
function DailyEmail() {
  const [status, setStatus] = useState<{ configured: boolean; to: string | null } | null>(null);
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
          : `Sent to ${r.to} (no High-interest matches right now, so it's a short check-in).`,
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
        After the daily check at 2:00pm ET, new High-interest matches with a High or Med fit are
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
          Not set up yet: store a Gmail app password with <code>make email-secret</code>, then
          redeploy with <code>EMAIL_ADDRESS=you@gmail.com</code> added to <code>make deploy</code>.
        </p>
      )}
      <button className="ghost" onClick={test} disabled={busy || !status?.configured}>
        {busy ? <Spinner label="Sending..." /> : "Send a test email"}
      </button>
      {result && <p className={result.ok ? "muted small" : "error-text small"}>{result.text}</p>}
    </div>
  );
}

// Wipes everything and starts over at onboarding. Typing DELETE is the confirmation: it's the one
// irreversible action in the app, and the backend refuses without it too.
function DeleteAllData() {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.deleteAllData(typed);
      // A full reload, so the app re-checks onboarding status and drops every page's loaded data.
      window.location.assign("/onboarding");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <div className="card danger-card">
      <h2>Start over</h2>
      <p className="muted small">
        Permanently deletes your resumes and cover letters, profile, sources, jobs, My Jobs,
        Jobby conversations and settings, then takes you back to onboarding. Your Claude usage
        log is kept. This can't be undone.
      </p>
      {!open ? (
        <button className="ghost danger" onClick={() => setOpen(true)}>
          Delete all my data…
        </button>
      ) : (
        <div className="fsection">
          <label className="flabel" htmlFor="confirm-delete">
            Type DELETE to confirm
          </label>
          <div className="form-actions" style={{ marginTop: 0 }}>
            <input
              id="confirm-delete"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoComplete="off"
              disabled={busy}
            />
            <button className="danger-solid" onClick={run} disabled={typed !== "DELETE" || busy}>
              {busy ? <Spinner label="Deleting..." /> : "Delete everything"}
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
      )}
    </div>
  );
}

function PrefilterSettings() {
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
        <input
          type="number"
          defaultValue={settings.daily_llm_cap}
          onBlur={(e) => save({ daily_llm_cap: Number(e.target.value) })}
          style={{ width: "140px" }}
        />
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

function UsageLog() {
  const [calls, setCalls] = useState<LlmCall[]>([]);
  const [totals, setTotals] = useState<{ total_cost: number; today_cost: number; total_calls: number } | null>(
    null,
  );

  useEffect(() => {
    api.getUsageLog().then((r) => {
      setCalls(r.calls);
      setTotals(r.totals);
    });
  }, []);

  return (
    <div className="card">
      <h2>Usage log</h2>
      {totals && (
        <p className="muted small">
          ${totals.today_cost.toFixed(2)} today · ${totals.total_cost.toFixed(2)} total ·{" "}
          {totals.total_calls} calls
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

      <DailyEmail />
      <PrefilterSettings />
      <UsageLog />
      <DeleteAllData />
    </div>
  );
}
