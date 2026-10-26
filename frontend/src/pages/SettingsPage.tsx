import { useEffect, useState } from "react";
import { api, ApiError, LlmCall, Settings } from "../api";
import { Markdown } from "../components/Markdown";

function toList(s: string): string[] {
  return s
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
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
          {busy ? "Rebuilding..." : "Rebuild from feedback"}
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

      <PrefilterSettings />
      <UsageLog />
    </div>
  );
}
