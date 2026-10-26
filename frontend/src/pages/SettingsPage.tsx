import { useEffect, useState } from "react";
import { api, ApiError, LlmCall, Settings } from "../api";

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
      <h2>Pre-filter rules &amp; daily cap</h2>
      <p className="muted">
        Cheap rule-based scoring runs before any Claude call, so the daily cap is spent on the most
        promising jobs first. Leave a field blank to skip that rule.
      </p>

      <label>
        Daily LLM cap (max jobs fully scored per day)
        <input
          type="number"
          defaultValue={settings.daily_llm_cap}
          onBlur={(e) => save({ daily_llm_cap: Number(e.target.value) })}
          style={{ display: "block", width: "100%" }}
        />
      </label>

      <label>
        Required keywords (any match boosts score)
        <input
          defaultValue={settings.prefilter_keywords_include.join(", ")}
          onBlur={(e) => save({ prefilter_keywords_include: toList(e.target.value) })}
          style={{ display: "block", width: "100%" }}
        />
      </label>

      <label>
        Excluded keywords (any match drops score to zero)
        <input
          defaultValue={settings.prefilter_keywords_exclude.join(", ")}
          onBlur={(e) => save({ prefilter_keywords_exclude: toList(e.target.value) })}
          style={{ display: "block", width: "100%" }}
        />
      </label>

      <label>
        Preferred titles
        <input
          defaultValue={settings.prefilter_titles_include.join(", ")}
          onBlur={(e) => save({ prefilter_titles_include: toList(e.target.value) })}
          style={{ display: "block", width: "100%" }}
        />
      </label>

      <label>
        Preferred locations
        <input
          defaultValue={settings.prefilter_locations.join(", ")}
          onBlur={(e) => save({ prefilter_locations: toList(e.target.value) })}
          style={{ display: "block", width: "100%" }}
        />
      </label>

      <label>
        <input
          type="checkbox"
          defaultChecked={settings.prefilter_remote_only}
          onChange={(e) => save({ prefilter_remote_only: e.target.checked })}
        />{" "}
        Remote only
      </label>

      <label style={{ display: "block" }}>
        Minimum salary (max range below this is penalized)
        <input
          type="number"
          defaultValue={settings.salary_floor ?? ""}
          onBlur={(e) => save({ salary_floor: e.target.value ? Number(e.target.value) : null })}
          style={{ display: "block", width: "100%" }}
        />
      </label>

      {busy && <p className="muted">Saving...</p>}
      {saved && !busy && <p className="muted">Saved.</p>}
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
        <p className="muted">
          ${totals.today_cost.toFixed(2)} today · ${totals.total_cost.toFixed(2)} total ·{" "}
          {totals.total_calls} calls
        </p>
      )}
      <table style={{ width: "100%", fontSize: "0.9em" }}>
        <thead>
          <tr style={{ textAlign: "left" }}>
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
              <td>{c.created_at}</td>
              <td>{c.call_type}</td>
              <td>{c.model}</td>
              <td>
                {c.input_tokens}/{c.output_tokens}
              </td>
              <td>{c.estimated_cost_usd != null ? `$${c.estimated_cost_usd.toFixed(4)}` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {calls.length === 0 && <p className="muted">No Claude calls logged yet.</p>}
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

      <div className="card">
        <h2>Preference profile</h2>
        <p className="muted">
          Learned from your interested/not-interested marks and application outcomes (application
          outcomes count for more). Rebuild after marking a batch of jobs or moving applications
          forward.
        </p>
        <button onClick={rebuild} disabled={busy}>
          {busy ? "Rebuilding..." : "Rebuild from feedback"}
        </button>
        {error && <p style={{ color: "crimson" }}>{error}</p>}
        {profile ? (
          <p style={{ whiteSpace: "pre-wrap" }}>{profile}</p>
        ) : (
          <p className="muted">No preference profile yet — mark some jobs or track an application.</p>
        )}
      </div>

      <PrefilterSettings />
      <UsageLog />
    </div>
  );
}
