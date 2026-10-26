import { useEffect, useState } from "react";
import { api, Favorite, Source, SourceKind } from "../api";

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
      <div className="fsection">
        <span className="flabel">Kind</span>
        <select value={kind} onChange={(e) => setKind(e.target.value as SourceKind)}>
          {Object.entries(KIND_LABELS).map(([k, label]) => (
            <option key={k} value={k}>
              {label}
            </option>
          ))}
        </select>
      </div>
      <div className="fsection">
        <span className="flabel">Name</span>
        <input
          placeholder="e.g. 'Stripe careers'"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ width: "100%" }}
        />
      </div>
      <div className="fsection">
        <span className="flabel">{field?.placeholder.startsWith("e.g.") ? field.key : "Value"}</span>
        <input
          placeholder={field?.placeholder}
          value={configValue}
          onChange={(e) => setConfigValue(e.target.value)}
          style={{ width: "100%" }}
        />
      </div>
      <div className="fsection" style={{ marginBottom: 0 }}>
        <span className="flabel">Company display name (optional)</span>
        <input
          placeholder="Company display name"
          value={companyName}
          onChange={(e) => setCompanyName(e.target.value)}
          style={{ width: "100%" }}
        />
      </div>
      <div className="form-actions">
        <button className="primary" onClick={submit} disabled={busy}>
          Add source
        </button>
      </div>
      {error && <p className="error-text small">{error}</p>}
    </div>
  );
}

function FavoritesSection() {
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => api.listFavorites().then(setFavorites);

  useEffect(() => {
    refresh();
  }, []);

  const add = async () => {
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.addFavorite(name.trim());
      setName("");
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const retry = async (id: number) => {
    setBusy(true);
    try {
      await api.retryFavorite(id);
      refresh();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>
        <span className="star">★</span> Favorite companies
      </h2>
      <p className="muted small">
        Any new posting from a favorite company always gets scored and flagged in the digest,
        regardless of the daily cap or your exclude-keyword rules.
      </p>
      <div className="form-row">
        <input
          placeholder="Company name, e.g. Anthropic"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ flex: "1 1 260px" }}
        />
        <button className="primary" onClick={add} disabled={busy}>
          Add favorite
        </button>
      </div>
      {error && <p className="error-text small">{error}</p>}

      <ul>
        {favorites.map((f) => (
          <li key={f.id}>
            {f.company_name} —{" "}
            {f.resolution_status === "resolved" && (
              <span className="badge badge-high">resolved via {f.source_kind}</span>
            )}
            {f.resolution_status === "pending" && <span className="badge badge-med">resolving...</span>}
            {f.resolution_status === "failed" && (
              <span>
                <span className="badge badge-low">couldn't find a board</span>{" "}
                <button className="link small" onClick={() => retry(f.id)} disabled={busy}>
                  retry
                </button>
              </span>
            )}
            {" — "}
            <button className="link small" onClick={() => api.removeFavorite(f.id).then(refresh)}>
              remove
            </button>
          </li>
        ))}
      </ul>
      {favorites.length === 0 && <p className="empty-state">No favorites yet.</p>}
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
      <div className="card-header">
        <div>
          <div className="card-title">
            {source.is_favorite && <span className="star">★ </span>}
            {source.name}
            {!source.enabled && <span className="pill" style={{ marginLeft: "8px" }}>disabled</span>}
          </div>
          <div className="card-meta">
            {KIND_LABELS[source.kind]} · <code className="path">{JSON.stringify(source.config)}</code>
          </div>
        </div>
      </div>
      {source.last_error && <p className="error-text small">Last error: {source.last_error}</p>}
      <div className="form-actions" style={{ marginTop: 0 }}>
        <button className="ghost small" onClick={test} disabled={busy}>
          Test fetch
        </button>
        <button className="ghost small" onClick={toggle}>
          {source.enabled ? "Disable" : "Enable"}
        </button>
        <button className="ghost small danger" onClick={remove}>
          Delete
        </button>
      </div>
      {testResult && <p className="muted small">{testResult}</p>}
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
      <p className="muted" style={{ marginBottom: "18px" }}>
        Company ATS boards (Greenhouse/Lever/Ashby), aggregator APIs (Adzuna/RemoteOK), and custom
        boards — any URL, including multi-employer boards. For LinkedIn/Indeed/Glassdoor, use "Add a
        job" on the Digest page instead (paste-in).
      </p>

      <FavoritesSection />
      <AddSourceForm onAdded={() => refresh()} />

      <div className="card">
        <h2>Run the daily pipeline now</h2>
        <p className="muted small">
          In production this runs automatically via Cloud Scheduler. Locally, trigger it manually
          to test.
        </p>
        <button className="ghost" onClick={runNow} disabled={running}>
          {running ? "Running..." : "Run now"}
        </button>
        {runResult && (
          <pre style={{ whiteSpace: "pre-wrap", marginTop: "10px" }}>
            <code>{runResult}</code>
          </pre>
        )}
      </div>

      {sources.map((s) => (
        <SourceCard key={s.id} source={s} onChanged={refresh} />
      ))}
      {sources.length === 0 && <p className="empty-state">No sources yet — add one above.</p>}
    </div>
  );
}
