import { useEffect, useState } from "react";
import { api, Source, SourceKind } from "../api";

const KIND_LABELS: Record<SourceKind, string> = {
  ats_greenhouse: "Greenhouse (company board token)",
  ats_lever: "Lever (company slug)",
  ats_ashby: "Ashby (board name)",
  aggregator_adzuna: "Adzuna (keyword search)",
  aggregator_remoteok: "RemoteOK (tag filter)",
  custom_board: "Custom board (any URL)",
};

const CONFIG_FIELD: Record<SourceKind, { key: string; placeholder: string } | null> = {
  ats_greenhouse: { key: "board_token", placeholder: "e.g. stripe" },
  ats_lever: { key: "company_slug", placeholder: "e.g. ro" },
  ats_ashby: { key: "board_name", placeholder: "e.g. linear" },
  aggregator_adzuna: { key: "what", placeholder: "e.g. software engineer" },
  aggregator_remoteok: { key: "tags", placeholder: "e.g. python,backend (optional)" },
  custom_board: { key: "url", placeholder: "https://..." },
};

function AddSourceForm({ onAdded }: { onAdded: (source: Source) => void }) {
  const [kind, setKind] = useState<SourceKind>("ats_greenhouse");
  const [name, setName] = useState("");
  const [configValue, setConfigValue] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const field = CONFIG_FIELD[kind];

  const submit = async () => {
    if (!name.trim() || !field) return;
    setBusy(true);
    setError(null);
    try {
      const config =
        field.key === "tags"
          ? { tags: configValue.split(",").map((t) => t.trim()).filter(Boolean) }
          : { [field.key]: configValue.trim() };
      const source = await api.createSource({
        kind,
        name,
        company_name: companyName || null,
        config,
      });
      setName("");
      setConfigValue("");
      setCompanyName("");
      onAdded(source);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>Add a source</h2>
      <select value={kind} onChange={(e) => setKind(e.target.value as SourceKind)}>
        {Object.entries(KIND_LABELS).map(([k, label]) => (
          <option key={k} value={k}>
            {label}
          </option>
        ))}
      </select>
      <div style={{ marginTop: "0.4rem" }}>
        <input
          placeholder="Name (e.g. 'Stripe careers')"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ width: "100%", marginBottom: "0.4rem" }}
        />
        <input
          placeholder={field?.placeholder}
          value={configValue}
          onChange={(e) => setConfigValue(e.target.value)}
          style={{ width: "100%", marginBottom: "0.4rem" }}
        />
        <input
          placeholder="Company display name (optional)"
          value={companyName}
          onChange={(e) => setCompanyName(e.target.value)}
          style={{ width: "100%", marginBottom: "0.4rem" }}
        />
      </div>
      <button onClick={submit} disabled={busy}>
        Add source
      </button>
      {error && <p style={{ color: "crimson" }}>{error}</p>}
    </div>
  );
}

function SourceCard({ source, onChanged }: { source: Source; onChanged: () => void }) {
  const [testResult, setTestResult] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const test = async () => {
    setBusy(true);
    setTestResult(null);
    try {
      const r = await api.testSource(source.id);
      setTestResult(`found ${r.count} posting(s) — e.g. "${r.preview[0]?.title ?? "none"}"`);
    } catch (e) {
      setTestResult(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const toggle = async () => {
    await api.updateSource(source.id, { enabled: !source.enabled });
    onChanged();
  };

  const remove = async () => {
    await api.deleteSource(source.id);
    onChanged();
  };

  return (
    <div className="card">
      <h3>{source.name}</h3>
      <p className="muted">
        {KIND_LABELS[source.kind]} · {JSON.stringify(source.config)}
      </p>
      {source.last_error && <p style={{ color: "crimson" }}>Last error: {source.last_error}</p>}
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button onClick={test} disabled={busy}>
          Test fetch
        </button>
        <button onClick={toggle}>{source.enabled ? "Disable" : "Enable"}</button>
        <button onClick={remove}>Delete</button>
      </div>
      {testResult && <p className="muted">{testResult}</p>}
    </div>
  );
}

export default function SourcesPage() {
  const [sources, setSources] = useState<Source[]>([]);
  const [runResult, setRunResult] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const refresh = () => api.listSources().then(setSources);

  useEffect(() => {
    refresh();
  }, []);

  const runNow = async () => {
    setRunning(true);
    setRunResult(null);
    try {
      const summary = await api.runDailyNow();
      setRunResult(JSON.stringify(summary, null, 2));
      refresh();
    } catch (e) {
      setRunResult(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div>
      <h1>Sources</h1>
      <p className="muted">
        Company ATS boards (Greenhouse/Lever/Ashby), aggregator APIs (Adzuna/RemoteOK), and custom
        boards — any URL, including multi-employer boards. For LinkedIn/Indeed/Glassdoor, use "Add a
        job" on the Digest page instead (paste-in).
      </p>

      <AddSourceForm onAdded={() => refresh()} />

      <div className="card">
        <h2>Run the daily pipeline now</h2>
        <p className="muted">
          In production this runs automatically via Cloud Scheduler. Locally, trigger it manually
          to test.
        </p>
        <button onClick={runNow} disabled={running}>
          {running ? "Running..." : "Run now"}
        </button>
        {runResult && <pre style={{ whiteSpace: "pre-wrap" }}>{runResult}</pre>}
      </div>

      {sources.map((s) => (
        <SourceCard key={s.id} source={s} onChanged={refresh} />
      ))}
    </div>
  );
}
