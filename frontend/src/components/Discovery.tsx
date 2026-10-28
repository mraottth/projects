import { useEffect, useRef, useState } from "react";
import { api, CompanySuggestion, JobBoardSuggestion } from "../api";
import { detectSourceFromUrl } from "../sourceUrl";
import { Spinner } from "./Spinner";

// Shared by the Manage Sources page (run on demand via the button) and onboarding's "Sources to
// follow" step (autoRun: searching starts as soon as the step opens).

const SEARCHING_LABEL = "Searching the web — usually 1–2 minutes...";
const NO_RESULTS =
  "No new suggestions this time — try again, or add more detail under \"What are you looking for?\"";

function errorText(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

// Runs a discovery call, on mount too when autoRun is set. `results` stays null until a search
// finishes, so "searched and found nothing" can be told apart from "hasn't searched yet".
function useDiscovery<T>(fetch: () => Promise<T[]>, autoRun: boolean) {
  const [results, setResults] = useState<T[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setResults(await fetch());
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  // The ref keeps StrictMode's dev-only double mount from firing two (paid) Claude calls.
  const autoRan = useRef(false);
  useEffect(() => {
    if (autoRun && !autoRan.current) {
      autoRan.current = true;
      run();
    }
  }, []);

  return { results, busy, error, run };
}

// Per-suggestion "Monitor" state, keyed by name/URL: working, added, or an error message.
// A company suggestion that came with its careers URL is added directly, like a board; only one
// without falls back to resolving the name to a job board (slower, and it can come up empty).
function monitorCompany(s: CompanySuggestion) {
  const detected = s.careers_url ? detectSourceFromUrl(s.careers_url) : null;
  if (detected) {
    return api.createSource({
      kind: detected.kind,
      name: s.company_name,
      company_name: s.company_name,
      config: detected.config,
    });
  }
  return api.resolveSource(s.company_name).catch((e) => {
    throw new Error(
      `${errorText(e)} — you can add it by URL under Manage Sources → "Add a source".`,
    );
  });
}

function useApprovals() {
  const [status, setStatus] = useState<Record<string, string>>({});
  const approve = async (key: string, action: () => Promise<unknown>) => {
    setStatus((s) => ({ ...s, [key]: "Adding..." }));
    try {
      await action();
      setStatus((s) => ({ ...s, [key]: "✓ Monitoring" }));
    } catch (e) {
      setStatus((s) => ({ ...s, [key]: errorText(e) }));
    }
  };
  return { status, approve };
}

function SuggestionActions({ status, onMonitor }: { status?: string; onMonitor: () => void }) {
  if (status) return <p className="muted small">{status}</p>;
  return (
    <div className="form-actions" style={{ marginTop: 0 }}>
      <button className="ghost small" onClick={onMonitor}>
        Monitor
      </button>
    </div>
  );
}

export function DiscoverCompaniesSection({ autoRun = false }: { autoRun?: boolean }) {
  const { results, busy, error, run } = useDiscovery<CompanySuggestion>(api.suggestCompanies, autoRun);
  const { status, approve } = useApprovals();

  return (
    <div className="card">
      <h2>Discover organizations</h2>
      <p className="muted small">
        Uses your candidate profile and what you're looking for, plus a web search, to suggest
        organizations worth monitoring. Monitoring one checks its job board for new postings every
        day. Nothing is added until you click Monitor.
      </p>
      <button className="ghost" onClick={run} disabled={busy}>
        {busy ? <Spinner label={SEARCHING_LABEL} /> : "Suggest organizations"}
      </button>
      {error && <p className="error-text small">{error}</p>}
      {!busy && results?.length === 0 && <p className="empty-state small">{NO_RESULTS}</p>}

      {results?.map((s) => (
        <div key={s.company_name} className="card-section">
          <div className="card-title" style={{ fontSize: "0.95rem" }}>
            {s.company_name}
          </div>
          <p className="muted small">{s.reason}</p>
          <SuggestionActions
            status={status[s.company_name]}
            onMonitor={() => approve(s.company_name, () => monitorCompany(s))}
          />
        </div>
      ))}
    </div>
  );
}

export function DiscoverJobBoardsSection({ autoRun = false }: { autoRun?: boolean }) {
  const { results, busy, error, run } = useDiscovery<JobBoardSuggestion>(api.suggestJobBoards, autoRun);
  const { status, approve } = useApprovals();

  return (
    <div className="card">
      <h2>Discover job boards</h2>
      <p className="muted small">
        Suggests multi-employer boards worth monitoring — broader reach than tracking one
        organization at a time. Monitoring one checks it for new postings every day. Nothing is
        added until you click Monitor.
      </p>
      <button className="ghost" onClick={run} disabled={busy}>
        {busy ? <Spinner label={SEARCHING_LABEL} /> : "Suggest job boards"}
      </button>
      {error && <p className="error-text small">{error}</p>}
      {!busy && results?.length === 0 && <p className="empty-state small">{NO_RESULTS}</p>}

      {results?.map((s) => (
        <div key={s.url} className="card-section">
          <div className="card-title" style={{ fontSize: "0.95rem" }}>
            {s.name}
          </div>
          <p className="muted small">
            <code className="path">{s.url}</code>
          </p>
          <p className="muted small">{s.reason}</p>
          <SuggestionActions
            status={status[s.url]}
            onMonitor={() =>
              approve(s.url, () =>
                api.createSource({ kind: "custom_board", name: s.name, config: { url: s.url } }),
              )
            }
          />
        </div>
      ))}
    </div>
  );
}
