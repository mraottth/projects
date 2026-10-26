import { useEffect, useState } from "react";
import { api, ApiError } from "../api";

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

      <div className="card">
        <h2>Pre-filter rules, daily cap, usage log</h2>
        <p className="muted">Coming in Milestone 4.</p>
      </div>
    </div>
  );
}
