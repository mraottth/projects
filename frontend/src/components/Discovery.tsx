import { useEffect, useRef, useState } from "react";
import { api, CompanySuggestion, JobBoardSuggestion } from "../api";
import { Spinner } from "./Spinner";

// Shared by the Manage Sources page (run on demand via the button) and onboarding's "Sources to
// follow" step (autoRun: searching starts as soon as the step opens).

export function DiscoverCompaniesSection({ autoRun = false }: { autoRun?: boolean }) {
  const [suggestions, setSuggestions] = useState<CompanySuggestion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Record<string, string>>({});

  const runDiscovery = async () => {
    setBusy(true);
    setError(null);
    try {
      setSuggestions(await api.suggestCompanies());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // The ref keeps StrictMode's dev-only double mount from firing two (paid) Claude calls.
  const autoRan = useRef(false);
  useEffect(() => {
    if (autoRun && !autoRan.current) {
      autoRan.current = true;
      runDiscovery();
    }
  }, []);

  const approve = async (companyName: string, favorite: boolean) => {
    setStatus((s) => ({ ...s, [companyName]: "working" }));
    try {
      if (favorite) {
        const result = await api.addFavorite(companyName);
        setStatus((s) => ({
          ...s,
          [companyName]: result.resolution_status === "resolved" ? "added" : "not found",
        }));
      } else {
        await api.resolveSource(companyName);
        setStatus((s) => ({ ...s, [companyName]: "added" }));
      }
    } catch (e) {
      setStatus((s) => ({ ...s, [companyName]: e instanceof Error ? e.message : String(e) }));
    }
  };

  return (
    <div className="card">
      <h2>Discover organizations</h2>
      <p className="muted small">
        Uses your candidate profile and desired role (Profile page), plus a web search, to suggest
        organizations worth monitoring. Nothing is added until you approve it.
      </p>
      <button className="ghost" onClick={runDiscovery} disabled={busy}>
        {busy ? <Spinner label="Searching..." /> : "Suggest organizations"}
      </button>
      {error && <p className="error-text small">{error}</p>}

      {suggestions.map((s) => (
        <div key={s.company_name} className="card-section">
          <div className="card-title" style={{ fontSize: "0.95rem" }}>
            {s.company_name}
          </div>
          <p className="muted small">{s.reason}</p>
          {status[s.company_name] ? (
            <p className="muted small">{status[s.company_name]}</p>
          ) : (
            <div className="form-actions" style={{ marginTop: 0 }}>
              <button className="ghost small" onClick={() => approve(s.company_name, false)}>
                Monitor
              </button>
              <button className="ghost small" onClick={() => approve(s.company_name, true)}>
                Monitor &amp; favorite
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function DiscoverJobBoardsSection({ autoRun = false }: { autoRun?: boolean }) {
  const [suggestions, setSuggestions] = useState<JobBoardSuggestion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Record<string, string>>({});

  const runDiscovery = async () => {
    setBusy(true);
    setError(null);
    try {
      setSuggestions(await api.suggestJobBoards());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // The ref keeps StrictMode's dev-only double mount from firing two (paid) Claude calls.
  const autoRan = useRef(false);
  useEffect(() => {
    if (autoRun && !autoRan.current) {
      autoRan.current = true;
      runDiscovery();
    }
  }, []);

  const approve = async (board: JobBoardSuggestion, favorite: boolean) => {
    setStatus((s) => ({ ...s, [board.url]: "working" }));
    try {
      await api.createSource({
        kind: "custom_board",
        name: board.name,
        config: { url: board.url },
        is_favorite: favorite,
      });
      setStatus((s) => ({ ...s, [board.url]: "added" }));
    } catch (e) {
      setStatus((s) => ({ ...s, [board.url]: e instanceof Error ? e.message : String(e) }));
    }
  };

  return (
    <div className="card">
      <h2>Discover job boards</h2>
      <p className="muted small">
        Uses your candidate profile and desired role (Profile page), plus a web search, to suggest
        multi-employer boards worth monitoring — broader reach than tracking one organization at
        a time. Nothing is added until you approve it.
      </p>
      <button className="ghost" onClick={runDiscovery} disabled={busy}>
        {busy ? <Spinner label="Searching..." /> : "Suggest job boards"}
      </button>
      {error && <p className="error-text small">{error}</p>}

      {suggestions.map((s) => (
        <div key={s.url} className="card-section">
          <div className="card-title" style={{ fontSize: "0.95rem" }}>
            {s.name}
          </div>
          <p className="muted small">
            <code className="path">{s.url}</code>
          </p>
          <p className="muted small">{s.reason}</p>
          {status[s.url] ? (
            <p className="muted small">{status[s.url]}</p>
          ) : (
            <div className="form-actions" style={{ marginTop: 0 }}>
              <button className="ghost small" onClick={() => approve(s, false)}>
                Monitor
              </button>
              <button className="ghost small" onClick={() => approve(s, true)}>
                Monitor &amp; favorite
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
